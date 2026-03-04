//! Documentation browsing and search tools for the Thorium MCP server.
//!
//! These tools give AI agents progressive access to Thorium's documentation:
//! 1. [`get_docs_toc`](ThoriumMCP::get_docs_toc) -- orientation via the table of contents
//! 2. [`search_docs`](ThoriumMCP::search_docs) -- discovery via full-text search
//! 3. [`get_doc_page`](ThoriumMCP::get_doc_page) -- deep reading via individual page fetch
//!
//! # `content` vs `structured_content`
//!
//! Each tool returns both `content` and `structured_content` in its
//! [`CallToolResult`]. These serve different consumers:
//!
//! - **`content`** (required by the MCP spec) -- A human-readable text or JSON
//!   representation intended for display in chat UIs and for models that only
//!   inspect the `content` array.
//! - **`structured_content`** (optional MCP extension) -- A machine-readable
//!   JSON value with a stable schema, intended for programmatic consumers that
//!   need to parse fields reliably without scraping text.
//!
//! Both fields carry the same information; `structured_content` simply makes it
//! easier for tool-calling agents to extract specific values.

use std::path::Path;

use async_walkdir::WalkDir;
use futures::stream::StreamExt;
use rmcp::ErrorData;
use rmcp::handler::server::tool::Extension as RmcpExtension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router};
use schemars::JsonSchema;
use tracing::instrument;

use super::{McpConfig, ThoriumMCP};

/// Maximum number of search results to return from `search_docs`.
const MAX_SEARCH_RESULTS: usize = 10;

/// Maximum number of context snippets per search result.
const MAX_SNIPPETS_PER_PAGE: usize = 3;

/// Number of context lines to include above and below each snippet match.
const SNIPPET_CONTEXT_LINES: usize = 1;

/// A single entry from the documentation table of contents.
#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TocEntry {
    /// The display title of this documentation page.
    pub title: String,
    /// The relative path to this documentation page.
    pub path: String,
    /// The nesting depth of this entry (0 = top level).
    pub depth: usize,
}

/// The params needed to get a specific documentation page.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DocPage {
    /// The relative path to the documentation page (e.g. "concepts/files.md").
    pub path: String,
}

/// The params needed to search the documentation.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchDocs {
    /// The search query to find in the documentation.
    pub query: String,
    /// Maximum number of results to return (default: 10).
    #[serde(default)]
    pub max_results: Option<usize>,
    /// Maximum number of context snippets per result (default: 3).
    #[serde(default)]
    pub max_snippets: Option<usize>,
    /// Number of context lines above and below each match (default: 1).
    #[serde(default)]
    pub context_lines: Option<usize>,
}

/// A context snippet around a search match with its location in the source file.
#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Snippet {
    /// The text content of the snippet with surrounding context lines.
    pub text: String,
    /// The 1-based line number where the match occurs in the source file.
    pub line_offset: usize,
}

/// A single search result with context snippets.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// The relative path to the matching documentation page.
    pub path: String,
    /// The title of the matching page.
    pub title: String,
    /// The number of matches found in this page.
    pub match_count: usize,
    /// Context snippets around the matches.
    pub snippets: Vec<Snippet>,
}

/// Parse TOC entries from raw SUMMARY.md content.
///
/// Each line in SUMMARY.md is formatted as `"    - [Title](./path.md)"`
/// where leading whitespace indicates nesting depth (4 spaces per level).
/// Lines that don't contain a valid markdown link are skipped.
fn parse_toc_entries(content: &str) -> Vec<TocEntry> {
    content
        .lines()
        .filter_map(|line| {
            // count leading whitespace to determine nesting depth
            let leading_spaces = line.find(|c: char| !c.is_whitespace()).unwrap_or(0);
            let trimmed = &line[leading_spaces..];

            // split on '[' to reach the start of the link title
            let after_bracket = trimmed.split_once('[')?.1;
            // split on ']' to extract the title and the remainder
            let (title, after_title) = after_bracket.split_once(']')?;
            // split on '(' and ')' to extract the raw path from the link URL
            let raw_path = after_title.split_once('(')?.1.split_once(')')?.0;

            // strip the leading "./" that mdBook uses for relative paths
            let path = raw_path
                .strip_prefix("./")
                .unwrap_or(raw_path)
                .to_string();

            Some(TocEntry {
                title: title.to_string(),
                path,
                depth: leading_spaces / 4,
            })
        })
        .collect()
}

/// Read and parse the SUMMARY.md table of contents from the documentation root.
///
/// Combines file I/O with [`parse_toc_entries`] so callers don't need to know
/// about the `SUMMARY.md` convention.
///
/// # Errors
///
/// Returns `ErrorData` with `INTERNAL_ERROR` if `SUMMARY.md` cannot be read.
async fn parse_summary(docs_path: &Path) -> Result<Vec<TocEntry>, ErrorData> {
    let content = tokio::fs::read_to_string(docs_path.join("SUMMARY.md"))
        .await
        .map_err(|e| ErrorData {
            code: rmcp::model::ErrorCode::INTERNAL_ERROR,
            message: format!("Failed to read documentation index: {e}").into(),
            data: None,
        })?;
    Ok(parse_toc_entries(&content))
}

/// Validate a documentation path to prevent directory traversal.
///
/// Rejects paths containing `..` and verifies that the canonicalized target
/// remains within the documentation root.
///
/// # Errors
///
/// Returns `ErrorData` with `INVALID_PARAMS` if:
/// - The path contains `..` (directory traversal attempt)
/// - The resolved path falls outside the documentation directory
/// - The target file does not exist
///
/// Returns `ErrorData` with `INTERNAL_ERROR` if the base documentation
/// directory cannot be canonicalized.
fn validate_doc_path(docs_path: &Path, requested: &str) -> Result<std::path::PathBuf, ErrorData> {
    // reject paths with directory traversal components
    if requested.contains("..") {
        return Err(ErrorData {
            code: rmcp::model::ErrorCode::INVALID_PARAMS,
            message: "Invalid path: directory traversal is not allowed".into(),
            data: None,
        });
    }

    let full_path = docs_path.join(requested);

    // canonicalize both paths to resolve any symlinks and verify containment
    let canonical_base = docs_path.canonicalize().map_err(|e| ErrorData {
        code: rmcp::model::ErrorCode::INTERNAL_ERROR,
        message: format!("Failed to resolve docs directory: {e}").into(),
        data: None,
    })?;
    let canonical_target = full_path.canonicalize().map_err(|_| ErrorData {
        code: rmcp::model::ErrorCode::INVALID_PARAMS,
        message: format!("Documentation page not found: {requested}").into(),
        data: None,
    })?;

    if !canonical_target.starts_with(&canonical_base) {
        return Err(ErrorData {
            code: rmcp::model::ErrorCode::INVALID_PARAMS,
            message: "Invalid path: outside documentation directory".into(),
            data: None,
        });
    }

    Ok(canonical_target)
}

/// Collect all markdown file paths under a directory using async traversal.
///
/// Uses [`async_walkdir::WalkDir`] to avoid blocking the async runtime on
/// large directory trees.
///
/// # Errors
///
/// Returns `std::io::Error` if the directory cannot be read or any entry
/// within it cannot be inspected.
async fn collect_md_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    let mut paths = Vec::new();
    let mut walker = WalkDir::new(dir);
    while let Some(entry) = walker.next().await {
        let entry = entry.map_err(std::io::Error::from)?;
        let path = entry.path();
        if !path.is_dir() && path.extension().and_then(|e| e.to_str()) == Some("md") {
            paths.push(path);
        }
    }
    Ok(paths)
}

/// Extract the page title from markdown content (the first `# ` heading).
///
/// Falls back to the file stem (filename without extension) if a path is
/// provided and no `# ` heading is found. Returns `"Untitled"` only when
/// both the heading and the path are absent.
fn extract_title(content: &str, path: Option<&Path>) -> String {
    content
        .lines()
        .find_map(|line| line.trim().strip_prefix("# ").map(String::from))
        .unwrap_or_else(|| {
            path.and_then(|p| p.file_stem())
                .and_then(|s| s.to_str())
                .map(String::from)
                .unwrap_or_else(|| "Untitled".to_string())
        })
}

/// Extract context snippets around search matches.
///
/// Returns up to `max_snippets` text fragments, each containing the matching
/// line plus `context_lines` lines of surrounding context. Adjacent matches
/// are deduplicated to avoid overlapping snippets.
fn extract_snippets(
    content: &str,
    query_lower: &str,
    max_snippets: usize,
    context_lines: usize,
) -> Vec<Snippet> {
    let lines: Vec<&str> = content.lines().collect();
    let mut snippets = Vec::new();
    let mut last_snippet_line: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        if snippets.len() >= max_snippets {
            break;
        }
        if line.to_lowercase().contains(query_lower) {
            // skip if too close to the previous snippet to avoid overlap
            if let Some(last) = last_snippet_line {
                if i <= last + context_lines + 1 {
                    continue;
                }
            }
            // grab the matching line with context on each side
            let start = i.saturating_sub(context_lines);
            let end = (i + context_lines + 1).min(lines.len());
            let text: String = lines[start..end].join("\n");
            snippets.push(Snippet {
                text,
                line_offset: i + 1, // 1-based line number
            });
            last_snippet_line = Some(i);
        }
    }
    snippets
}

#[tool_router(router = docs_router, vis = "pub")]
impl ThoriumMCP {
    /// Get the table of contents for the Thorium documentation.
    ///
    /// # Errors
    ///
    /// Returns `ErrorData` with `INTERNAL_ERROR` if `SUMMARY.md` cannot be
    /// read from the configured docs path. Returns an authentication error
    /// if the MCP session token is invalid.
    #[tool(
        name = "get_docs_toc",
        description = "Get the table of contents for the Thorium documentation. Returns a structured list of all documentation pages with their titles, paths, and nesting depth. Call this first to understand what documentation is available."
    )]
    #[instrument(name = "ThoriumMCP::get_docs_toc", skip(self, parts), err(Debug))]
    pub async fn get_docs_toc(
        &self,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        McpConfig::grab_token(&parts)?;

        let entries = parse_summary(&self.conf.docs_path).await?;
        let serialized = serde_json::to_value(&entries).unwrap();

        Ok(CallToolResult {
            content: vec![Content::json(&entries)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        })
    }

    /// Get the content of a specific documentation page.
    ///
    /// # Errors
    ///
    /// Returns `ErrorData` with `INVALID_PARAMS` if the path is invalid or
    /// attempts directory traversal. Returns `INTERNAL_ERROR` if the file
    /// cannot be read. Returns an authentication error if the MCP session
    /// token is invalid.
    #[tool(
        name = "get_doc_page",
        description = "Get the content of a specific Thorium documentation page by its relative path (e.g. 'concepts/files.md'). Use get_docs_toc first to discover available page paths."
    )]
    #[instrument(name = "ThoriumMCP::get_doc_page", skip(self, parts), err(Debug))]
    pub async fn get_doc_page(
        &self,
        Parameters(DocPage { path }): Parameters<DocPage>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        McpConfig::grab_token(&parts)?;

        let full_path = validate_doc_path(&self.conf.docs_path, &path)?;
        let content =
            tokio::fs::read_to_string(&full_path)
                .await
                .map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode::INTERNAL_ERROR,
                    message: format!("Failed to read documentation page: {e}").into(),
                    data: None,
                })?;

        let serialized = serde_json::to_value(&serde_json::json!({
            "path": path,
            "content": content,
        }))
        .unwrap();

        Ok(CallToolResult {
            content: vec![Content::text(content)],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        })
    }

    /// Search across all Thorium documentation pages.
    ///
    /// # Errors
    ///
    /// Returns `ErrorData` with `INVALID_PARAMS` if the query is empty.
    /// Returns `INTERNAL_ERROR` if the documentation directory cannot be
    /// traversed. Returns an authentication error if the MCP session token
    /// is invalid. Individual file read failures are logged as warnings and
    /// skipped rather than aborting the entire search.
    #[tool(
        name = "search_docs",
        description = "Search across all Thorium documentation pages for a query string. Returns matching pages with titles, match counts, and context snippets. Use this to find documentation about a specific topic."
    )]
    #[instrument(name = "ThoriumMCP::search_docs", skip(self, parts), err(Debug))]
    pub async fn search_docs(
        &self,
        Parameters(SearchDocs {
            query,
            max_results,
            max_snippets,
            context_lines,
        }): Parameters<SearchDocs>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        McpConfig::grab_token(&parts)?;

        if query.trim().is_empty() {
            return Err(ErrorData {
                code: rmcp::model::ErrorCode::INVALID_PARAMS,
                message: "Search query must not be empty".into(),
                data: None,
            });
        }

        let max_results = max_results.unwrap_or(MAX_SEARCH_RESULTS);
        let max_snippets = max_snippets.unwrap_or(MAX_SNIPPETS_PER_PAGE);
        let context_lines = context_lines.unwrap_or(SNIPPET_CONTEXT_LINES);
        let query_lower = query.to_lowercase();

        let md_files =
            collect_md_files(&self.conf.docs_path)
                .await
                .map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode::INTERNAL_ERROR,
                    message: format!("Failed to read documentation directory: {e}").into(),
                    data: None,
                })?;

        let mut results: Vec<SearchResult> = Vec::new();
        for file_path in &md_files {
            let content = match tokio::fs::read_to_string(file_path).await {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        path = %file_path.display(),
                        "Failed to read documentation file: {e}"
                    );
                    continue;
                }
            };

            let match_count =
                content.to_lowercase().matches(&query_lower).count();
            if match_count == 0 {
                continue;
            }

            let rel_path = file_path
                .strip_prefix(&self.conf.docs_path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            results.push(SearchResult {
                title: extract_title(&content, Some(file_path)),
                snippets: extract_snippets(
                    &content,
                    &query_lower,
                    max_snippets,
                    context_lines,
                ),
                path: rel_path,
                match_count,
            });
        }

        results.sort_by(|a, b| b.match_count.cmp(&a.match_count));
        results.truncate(max_results);

        let serialized = serde_json::to_value(&results).unwrap();

        Ok(CallToolResult {
            content: vec![Content::json(&results)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use proptest::prelude::*;

    // ── parse_summary ──────────────────────────────────────────────

    #[test]
    fn parse_summary_basic() {
        let content = "\
# Summary

- [Intro](./intro.md)
- [Getting Started](./getting_started/getting_started.md)
    - [Registration](./getting_started/registration.md)
    - [Login](./getting_started/login.md)
        - [Advanced Login](./getting_started/advanced_login.md)
";
        let entries = parse_toc_entries(content);
        assert_eq!(
            entries,
            vec![
                TocEntry {
                    title: "Intro".into(),
                    path: "intro.md".into(),
                    depth: 0,
                },
                TocEntry {
                    title: "Getting Started".into(),
                    path: "getting_started/getting_started.md".into(),
                    depth: 0,
                },
                TocEntry {
                    title: "Registration".into(),
                    path: "getting_started/registration.md".into(),
                    depth: 1,
                },
                TocEntry {
                    title: "Login".into(),
                    path: "getting_started/login.md".into(),
                    depth: 1,
                },
                TocEntry {
                    title: "Advanced Login".into(),
                    path: "getting_started/advanced_login.md".into(),
                    depth: 2,
                },
            ]
        );
    }

    proptest! {
        #[test]
        fn parse_summary_roundtrip(
            title in "[A-Za-z]{1,30}",
            path in "[a-z]{1,10}/[a-z]{1,10}\\.md",
            depth in 0..5usize,
        ) {
            let indent = " ".repeat(depth * 4);
            let line = format!("{indent}- [{title}](./{path})");
            let entries = parse_toc_entries(&line);
            prop_assert_eq!(entries.len(), 1);
            prop_assert_eq!(&entries[0].title, &title);
            prop_assert_eq!(&entries[0].path, &path);
            prop_assert_eq!(entries[0].depth, depth);
        }

        #[test]
        fn parse_summary_skips_non_links(
            text in "[A-Za-z0-9 ]{1,50}",
        ) {
            let entries = parse_toc_entries(&text);
            prop_assert!(entries.is_empty());
        }
    }

    #[test]
    fn parse_summary_against_real_docs() {
        let summary_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("docs")
            .join("src")
            .join("SUMMARY.md");
        assert!(
            summary_path.exists(),
            "Real SUMMARY.md not found at {}",
            summary_path.display()
        );
        let content = fs::read_to_string(&summary_path).unwrap();
        let entries = parse_toc_entries(&content);
        assert!(
            entries.len() > 10,
            "Expected many TOC entries, got {}",
            entries.len()
        );
        assert_eq!(entries[0].title, "Intro");
        assert_eq!(entries[0].path, "intro.md");
        assert_eq!(entries[0].depth, 0);
    }

    // ── validate_doc_path ──────────────────────────────────────────

    proptest! {
        #[test]
        fn validate_doc_path_always_rejects_traversal(
            prefix in "[a-z]{0,5}",
            suffix in "[a-z]{1,10}",
        ) {
            let docs = tempfile::tempdir().unwrap();
            let path = format!("{prefix}/../{suffix}");
            let result = validate_doc_path(docs.path(), &path);
            prop_assert!(result.is_err());
            prop_assert!(
                result.unwrap_err().message.contains("directory traversal")
            );
        }
    }

    #[test]
    fn validate_doc_path_accepts_valid_path() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.md");
        fs::write(&file_path, "# Test").unwrap();

        let result = validate_doc_path(dir.path(), "test.md");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_path.canonicalize().unwrap());
    }

    #[test]
    fn validate_doc_path_rejects_nonexistent_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = validate_doc_path(dir.path(), "nonexistent.md").unwrap_err();
        assert!(err.message.contains("not found"));
    }

    // ── collect_md_files ───────────────────────────────────────────

    #[tokio::test]
    async fn collect_md_files_finds_markdown() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("dragons.md"), "# Here Be Dragons").unwrap();
        fs::write(dir.path().join("treasure_map.txt"), "X marks the spot").unwrap();
        fs::write(dir.path().join("loot.png"), "not markdown").unwrap();
        let sub = dir.path().join("dungeon");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("boss_fight.md"), "# The Final Boss").unwrap();

        let paths = collect_md_files(dir.path()).await.unwrap();
        assert_eq!(paths.len(), 2);
        let names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"dragons.md".to_string()));
        assert!(names.contains(&"boss_fight.md".to_string()));
    }

    #[tokio::test]
    async fn collect_md_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let paths = collect_md_files(dir.path()).await.unwrap();
        assert!(paths.is_empty());
    }

    #[tokio::test]
    async fn collect_md_files_against_real_docs() {
        let docs_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("docs")
            .join("src");
        assert!(
            docs_src.exists(),
            "Real docs not found at {}",
            docs_src.display()
        );
        let paths = collect_md_files(&docs_src).await.unwrap();
        assert!(
            paths.len() > 50,
            "Expected many md files, got {}",
            paths.len()
        );
        assert!(paths.iter().all(|p| p.extension().unwrap() == "md"));
    }

    // ── extract_title ──────────────────────────────────────────────

    proptest! {
        #[test]
        fn extract_title_finds_h1(
            title in "[A-Za-z]{1,30}",
        ) {
            let content = format!("some preamble\n## Not This\n# {title}\nBody text.");
            prop_assert_eq!(extract_title(&content, None), title);
        }

        #[test]
        fn extract_title_untitled_without_h1(
            body in "[a-z ]{1,100}",
        ) {
            // body only contains lowercase letters and spaces, never "# "
            prop_assert_eq!(extract_title(&body, None), "Untitled");
        }
    }

    #[test]
    fn extract_title_falls_back_to_filename() {
        let path = Path::new("concepts/getting_started.md");
        assert_eq!(
            extract_title("no heading here", Some(path)),
            "getting_started"
        );
    }

    // ── extract_snippets ───────────────────────────────────────────

    proptest! {
        #[test]
        fn extract_snippets_respects_max(
            max_snippets in 1..10usize,
        ) {
            // 20 well-spaced matches should never exceed max_snippets
            let content: String = (0..20)
                .map(|i| format!("match line {i}\npadding\npadding\npadding\n"))
                .collect();
            let snippets =
                extract_snippets(&content, "match", max_snippets, SNIPPET_CONTEXT_LINES);
            prop_assert!(snippets.len() <= max_snippets);
        }

        #[test]
        fn extract_snippets_no_false_positives(
            query in "[xyz]{3,8}",
            content in "[abc ]{10,200}",
        ) {
            // content with only a/b/c/spaces cannot match xyz-only queries
            let snippets = extract_snippets(
                &content,
                &query.to_lowercase(),
                5,
                SNIPPET_CONTEXT_LINES,
            );
            prop_assert!(snippets.is_empty());
        }

        #[test]
        fn extract_snippets_each_contains_query(
            query in "[a-z]{3,6}",
        ) {
            let content =
                format!("before\n{query} appears here\nafter\nmore\n{query} again\nend");
            let snippets =
                extract_snippets(&content, &query, 10, SNIPPET_CONTEXT_LINES);
            for snippet in &snippets {
                prop_assert!(
                    snippet.text.to_lowercase().contains(&query),
                    "Snippet {:?} does not contain query {:?}",
                    snippet.text,
                    query
                );
            }
        }

        #[test]
        fn extract_snippets_configurable_context(
            context_lines in 0..5usize,
        ) {
            let content = "a\nb\nc\nmatch\nd\ne\nf";
            let snippets = extract_snippets(content, "match", 1, context_lines);
            prop_assert_eq!(snippets.len(), 1);
            let line_count = snippets[0].text.lines().count();
            // snippet should contain at most: context above + match + context below
            let expected_max = context_lines + 1 + context_lines;
            prop_assert!(
                line_count <= expected_max,
                "Got {line_count} lines with context_lines={context_lines}, \
                 expected at most {expected_max}"
            );
            // "match" is on line 4 (1-based)
            prop_assert_eq!(snippets[0].line_offset, 4);
        }
    }

    proptest! {
        #[test]
        fn extract_snippets_deduplicates_adjacent(
            context_lines in 1..4usize,
        ) {
            // consecutive matches should be collapsed by deduplication
            let content = "foo line 1\nfoo line 2\nfoo line 3\n\n\n\n\n\nfoo distant\n";
            let snippets = extract_snippets(content, "foo", 10, context_lines);
            // 4 total matches but consecutive ones should be deduped
            prop_assert!(snippets.len() < 4);
            prop_assert!(!snippets.is_empty());
        }
    }
}
