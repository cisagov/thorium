//! The documentation related tools for the Thorium MCP server

use std::path::Path;

use rmcp::ErrorData;
use rmcp::handler::server::tool::Extension as RmcpExtension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router};
use schemars::JsonSchema;
use tracing::instrument;

use super::{McpConfig, ThoriumMCP};

/// A single entry from the documentation table of contents
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TocEntry {
    /// The display title of this documentation page
    pub title: String,
    /// The relative path to this documentation page
    pub path: String,
    /// The nesting depth of this entry (0 = top level)
    pub depth: usize,
}

/// The params needed to get a specific documentation page
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DocPage {
    /// The relative path to the documentation page (e.g. "concepts/files.md")
    pub path: String,
}

/// The params needed to search the documentation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchDocs {
    /// The search query to find in the documentation
    pub query: String,
}

/// A single search result with context snippets
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// The relative path to the matching documentation page
    pub path: String,
    /// The title of the matching page
    pub title: String,
    /// The number of matches found in this page
    pub match_count: usize,
    /// Context snippets around the matches
    pub snippets: Vec<String>,
}

/// An empty parameter set for tools that take no arguments
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Empty {}

/// Parse the SUMMARY.md file into a list of table of contents entries
///
/// # Arguments
///
/// * `content` - The raw SUMMARY.md content
fn parse_summary(content: &str) -> Vec<TocEntry> {
    let mut entries = Vec::new();
    for line in content.lines() {
        // skip lines that don't contain a markdown link
        let Some(bracket_start) = line.find('[') else {
            continue;
        };
        let Some(bracket_end) = line[bracket_start..].find(']') else {
            continue;
        };
        let bracket_end = bracket_start + bracket_end;
        let Some(paren_start) = line[bracket_end..].find('(') else {
            continue;
        };
        let paren_start = bracket_end + paren_start;
        let Some(paren_end) = line[paren_start..].find(')') else {
            continue;
        };
        let paren_end = paren_start + paren_end;

        let title = line[bracket_start + 1..bracket_end].to_string();
        let raw_path = line[paren_start + 1..paren_end].to_string();
        // strip the leading ./ if present
        let path = raw_path.strip_prefix("./").unwrap_or(&raw_path).to_string();

        // calculate depth from leading whitespace (4 spaces per level)
        let leading_spaces = line.len() - line.trim_start().len();
        let depth = leading_spaces / 4;

        entries.push(TocEntry {
            title,
            path,
            depth,
        });
    }
    entries
}

/// Validate a documentation path to prevent directory traversal
///
/// # Arguments
///
/// * `docs_path` - The base documentation directory
/// * `requested` - The requested relative path
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

/// Recursively collect all markdown file paths under a directory
///
/// # Arguments
///
/// * `dir` - The directory to walk
/// * `paths` - The output vector to collect paths into
fn collect_md_files(dir: &Path, paths: &mut Vec<std::path::PathBuf>) -> Result<(), std::io::Error> {
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_md_files(&path, paths)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            paths.push(path);
        }
    }
    Ok(())
}

/// Extract the page title from markdown content (the first # heading)
///
/// # Arguments
///
/// * `content` - The raw markdown content
fn extract_title(content: &str) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            return title.to_string();
        }
    }
    "Untitled".to_string()
}

/// Extract context snippets around search matches
///
/// # Arguments
///
/// * `content` - The raw markdown content
/// * `query_lower` - The lowercased search query
/// * `max_snippets` - Maximum number of snippets to return
fn extract_snippets(content: &str, query_lower: &str, max_snippets: usize) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut snippets = Vec::new();
    let mut last_snippet_line: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        if snippets.len() >= max_snippets {
            break;
        }
        if line.to_lowercase().contains(query_lower) {
            // skip if too close to the previous snippet
            if let Some(last) = last_snippet_line {
                if i <= last + 2 {
                    continue;
                }
            }
            // grab the matching line with one line of context on each side
            let start = i.saturating_sub(1);
            let end = (i + 2).min(lines.len());
            let snippet: String = lines[start..end].join("\n");
            snippets.push(snippet);
            last_snippet_line = Some(i);
        }
    }
    snippets
}

#[tool_router(router = docs_router, vis = "pub")]
impl ThoriumMCP {
    /// Get the table of contents for the Thorium documentation
    ///
    /// # Arguments
    ///
    /// * `parts` - The request parts required to validate authorization
    #[tool(
        name = "get_docs_toc",
        description = "Get the table of contents for the Thorium documentation. Returns a structured list of all documentation pages with their titles, paths, and nesting depth. Call this first to understand what documentation is available."
    )]
    #[instrument(name = "ThoriumMCP::get_docs_toc", skip(self, _params, parts), err(Debug))]
    pub async fn get_docs_toc(
        &self,
        Parameters(_params): Parameters<Empty>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // validate authorization
        McpConfig::grab_token(&parts)?;
        // read SUMMARY.md
        let summary_path = self.conf.docs_path.join("SUMMARY.md");
        let content =
            tokio::fs::read_to_string(&summary_path)
                .await
                .map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode::INTERNAL_ERROR,
                    message: format!("Failed to read documentation index: {e}").into(),
                    data: None,
                })?;
        // parse the summary into structured entries
        let entries = parse_summary(&content);
        // serialize our entries
        let serialized = serde_json::to_value(&entries).unwrap();
        // build our result
        let result = CallToolResult {
            content: vec![Content::json(&entries)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }

    /// Get the content of a specific documentation page
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters containing the page path
    /// * `parts` - The request parts required to validate authorization
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
        // validate authorization
        McpConfig::grab_token(&parts)?;
        // validate and resolve the path
        let full_path = validate_doc_path(&self.conf.docs_path, &path)?;
        // read the file
        let content =
            tokio::fs::read_to_string(&full_path)
                .await
                .map_err(|e| ErrorData {
                    code: rmcp::model::ErrorCode::INTERNAL_ERROR,
                    message: format!("Failed to read documentation page: {e}").into(),
                    data: None,
                })?;
        // build our result with the raw markdown as text content
        let serialized = serde_json::to_value(&serde_json::json!({
            "path": path,
            "content": content,
        }))
        .unwrap();
        let result = CallToolResult {
            content: vec![Content::text(content)],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }

    /// Search across all Thorium documentation pages
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters containing the search query
    /// * `parts` - The request parts required to validate authorization
    #[tool(
        name = "search_docs",
        description = "Search across all Thorium documentation pages for a query string. Returns matching pages with titles, match counts, and context snippets. Use this to find documentation about a specific topic."
    )]
    #[instrument(name = "ThoriumMCP::search_docs", skip(self, parts), err(Debug))]
    pub async fn search_docs(
        &self,
        Parameters(SearchDocs { query }): Parameters<SearchDocs>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // validate authorization
        McpConfig::grab_token(&parts)?;
        // validate query is not empty
        if query.trim().is_empty() {
            return Err(ErrorData {
                code: rmcp::model::ErrorCode::INVALID_PARAMS,
                message: "Search query must not be empty".into(),
                data: None,
            });
        }
        let query_lower = query.to_lowercase();
        // collect all markdown files
        let mut md_files = Vec::new();
        collect_md_files(&self.conf.docs_path, &mut md_files).map_err(
            |e| ErrorData {
                code: rmcp::model::ErrorCode::INTERNAL_ERROR,
                message: format!("Failed to read documentation directory: {e}").into(),
                data: None,
            },
        )?;
        // search each file
        let mut results: Vec<SearchResult> = Vec::new();
        for file_path in &md_files {
            let content = match tokio::fs::read_to_string(file_path).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let content_lower = content.to_lowercase();
            let match_count = content_lower.matches(&query_lower).count();
            if match_count == 0 {
                continue;
            }
            // get the relative path from the docs root
            let rel_path = file_path
                .strip_prefix(&self.conf.docs_path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();
            let title = extract_title(&content);
            let snippets = extract_snippets(&content, &query_lower, 3);
            results.push(SearchResult {
                path: rel_path,
                title,
                match_count,
                snippets,
            });
        }
        // sort by match count descending and limit to top 10
        results.sort_by(|a, b| b.match_count.cmp(&a.match_count));
        results.truncate(10);
        // serialize our results
        let serialized = serde_json::to_value(&results).unwrap();
        // build our result
        let result = CallToolResult {
            content: vec![Content::json(&results)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

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
        let entries = parse_summary(content);
        assert_eq!(entries.len(), 5);

        assert_eq!(entries[0].title, "Intro");
        assert_eq!(entries[0].path, "intro.md");
        assert_eq!(entries[0].depth, 0);

        assert_eq!(entries[1].title, "Getting Started");
        assert_eq!(entries[1].path, "getting_started/getting_started.md");
        assert_eq!(entries[1].depth, 0);

        assert_eq!(entries[2].title, "Registration");
        assert_eq!(entries[2].path, "getting_started/registration.md");
        assert_eq!(entries[2].depth, 1);

        assert_eq!(entries[3].title, "Login");
        assert_eq!(entries[3].depth, 1);

        assert_eq!(entries[4].title, "Advanced Login");
        assert_eq!(entries[4].depth, 2);
    }

    #[test]
    fn parse_summary_strips_dot_slash_prefix() {
        let content = "- [Page](./some/path.md)\n- [Other](other/path.md)\n";
        let entries = parse_summary(content);
        assert_eq!(entries[0].path, "some/path.md");
        assert_eq!(entries[1].path, "other/path.md");
    }

    #[test]
    fn parse_summary_skips_non_link_lines() {
        let content = "\
# Summary

Some random text here.

- [Actual Link](./page.md)

Another non-link line.
";
        let entries = parse_summary(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].title, "Actual Link");
    }

    #[test]
    fn parse_summary_empty_input() {
        let entries = parse_summary("");
        assert!(entries.is_empty());
    }

    #[test]
    fn parse_summary_against_real_docs() {
        let summary_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("docs")
            .join("src")
            .join("SUMMARY.md");
        if !summary_path.exists() {
            return; // skip if docs aren't available
        }
        let content = fs::read_to_string(&summary_path).unwrap();
        let entries = parse_summary(&content);
        // the real SUMMARY.md has many entries — just verify it parsed something
        assert!(entries.len() > 10, "Expected many TOC entries, got {}", entries.len());
        // first entry should be Intro
        assert_eq!(entries[0].title, "Intro");
        assert_eq!(entries[0].path, "intro.md");
        assert_eq!(entries[0].depth, 0);
    }

    // ── validate_doc_path ──────────────────────────────────────────

    #[test]
    fn validate_doc_path_rejects_traversal() {
        let docs = PathBuf::from("/tmp");
        let err = validate_doc_path(&docs, "../etc/passwd").unwrap_err();
        assert!(err.message.contains("directory traversal"));
    }

    #[test]
    fn validate_doc_path_rejects_hidden_traversal() {
        let docs = PathBuf::from("/tmp");
        let err = validate_doc_path(&docs, "foo/../../etc/passwd").unwrap_err();
        assert!(err.message.contains("directory traversal"));
    }

    #[test]
    fn validate_doc_path_accepts_valid_path() {
        // create a temp file to validate against
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

    #[test]
    fn collect_md_files_finds_markdown() {
        let dir = tempfile::tempdir().unwrap();
        // create some files
        fs::write(dir.path().join("page.md"), "# Page").unwrap();
        fs::write(dir.path().join("readme.txt"), "not markdown").unwrap();
        fs::write(dir.path().join("image.png"), "not markdown").unwrap();
        // create a subdirectory with another md file
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("nested.md"), "# Nested").unwrap();

        let mut paths = Vec::new();
        collect_md_files(dir.path(), &mut paths).unwrap();

        assert_eq!(paths.len(), 2);
        let names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"page.md".to_string()));
        assert!(names.contains(&"nested.md".to_string()));
    }

    #[test]
    fn collect_md_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let mut paths = Vec::new();
        collect_md_files(dir.path(), &mut paths).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn collect_md_files_against_real_docs() {
        let docs_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("docs")
            .join("src");
        if !docs_src.exists() {
            return; // skip if docs aren't available
        }
        let mut paths = Vec::new();
        collect_md_files(&docs_src, &mut paths).unwrap();
        // the real docs have 80+ markdown files
        assert!(paths.len() > 50, "Expected many md files, got {}", paths.len());
        // all should be .md files
        assert!(paths.iter().all(|p| p.extension().unwrap() == "md"));
    }

    // ── extract_title ──────────────────────────────────────────────

    #[test]
    fn extract_title_finds_h1() {
        let content = "Some preamble\n# My Title\n\nBody text here.";
        assert_eq!(extract_title(content), "My Title");
    }

    #[test]
    fn extract_title_returns_first_h1() {
        let content = "# First Title\n## Subtitle\n# Second Title";
        assert_eq!(extract_title(content), "First Title");
    }

    #[test]
    fn extract_title_ignores_h2_and_deeper() {
        let content = "## Not A Title\n### Also Not\nNo headings here.";
        assert_eq!(extract_title(content), "Untitled");
    }

    #[test]
    fn extract_title_handles_empty_content() {
        assert_eq!(extract_title(""), "Untitled");
    }

    #[test]
    fn extract_title_handles_leading_whitespace() {
        let content = "  # Indented Title\n";
        assert_eq!(extract_title(content), "Indented Title");
    }

    // ── extract_snippets ───────────────────────────────────────────

    #[test]
    fn extract_snippets_basic() {
        let content = "line 0\nline 1\nfoo bar baz\nline 3\nline 4\n";
        let snippets = extract_snippets(content, "foo", 3);
        assert_eq!(snippets.len(), 1);
        // should include the line before, the match line, and the line after
        assert!(snippets[0].contains("line 1"));
        assert!(snippets[0].contains("foo bar baz"));
        assert!(snippets[0].contains("line 3"));
    }

    #[test]
    fn extract_snippets_case_insensitive() {
        let content = "Hello World\nFOO BAR\nGoodbye";
        let snippets = extract_snippets(content, "foo", 3);
        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].contains("FOO BAR"));
    }

    #[test]
    fn extract_snippets_respects_max() {
        let content = "match1\n\n\n\nmatch2\n\n\n\nmatch3\n\n\n\nmatch4\n";
        let snippets = extract_snippets(content, "match", 2);
        assert_eq!(snippets.len(), 2);
    }

    #[test]
    fn extract_snippets_skips_adjacent_matches() {
        // matches on consecutive lines should be collapsed into one snippet
        let content = "foo line 1\nfoo line 2\nfoo line 3\n\n\n\nfoo line 7\n";
        let snippets = extract_snippets(content, "foo", 3);
        // first match at line 0 grabs context [0..2], so lines 1-2 are within +2
        // line 7 match (index 6) is far enough away
        assert_eq!(snippets.len(), 2);
    }

    #[test]
    fn extract_snippets_match_at_first_line() {
        let content = "match here\nline 1\nline 2\n";
        let snippets = extract_snippets(content, "match", 3);
        assert_eq!(snippets.len(), 1);
        // should not crash on saturating_sub(1) when match is at index 0
        assert!(snippets[0].contains("match here"));
        assert!(snippets[0].contains("line 1"));
    }

    #[test]
    fn extract_snippets_match_at_last_line() {
        let content = "line 0\nline 1\nmatch here";
        let snippets = extract_snippets(content, "match", 3);
        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].contains("line 1"));
        assert!(snippets[0].contains("match here"));
    }

    #[test]
    fn extract_snippets_no_matches() {
        let content = "nothing relevant here\nat all\n";
        let snippets = extract_snippets(content, "nonexistent", 3);
        assert!(snippets.is_empty());
    }
}
