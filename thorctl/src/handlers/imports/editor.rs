//! Editor integration for resolving merge conflicts
//!
//! Handles creating temporary files with conflict markers, opening the user's
//! editor, validating the resolved YAML, and presenting error recovery options.

use colored::Colorize;
use serde::de::DeserializeOwned;
use similar::{ChangeTag, TextDiff};
use thorium::{CtlConf, Error};
use uuid::Uuid;

/// Resolve the editor command to use: an explicit `--editor` override if one was
/// given, otherwise the configured `default_editor`.
///
/// # Arguments
///
/// * `editor_override` - An optional editor command from a `--editor` flag
/// * `conf` - The Thorctl config, whose `default_editor` is the fallback
pub(crate) fn resolve_editor<'a>(editor_override: Option<&'a str>, conf: &'a CtlConf) -> &'a str {
    editor_override.unwrap_or(&conf.default_editor)
}

// ─── Merge Conflict Generation ───────────────────────────────────────────────

/// Generate a string with git-style merge conflict markers showing the
/// differences between two text representations
///
/// Backs every editor merge view (YAML, JSON, TOML, and plain text); the marker
/// labels are supplied by the caller so the two sides read accurately for the
/// flow in use (import vs. on-disk export).
///
/// # Arguments
///
/// * `current` - The text representing the current/left side
/// * `incoming` - The text representing the incoming/right side
/// * `current_label` - The label for the current side's conflict marker
/// * `incoming_label` - The label for the incoming side's conflict marker
pub fn generate_conflict_view(
    current: &str,
    incoming: &str,
    current_label: &str,
    incoming_label: &str,
) -> String {
    let diff = TextDiff::from_lines(current, incoming);
    let mut output = String::new();
    // buffer for collecting consecutive changed lines
    let mut current_lines: Vec<&str> = Vec::new();
    let mut incoming_lines: Vec<&str> = Vec::new();

    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Equal => {
                // flush any buffered conflict before writing the equal line
                flush_conflict(
                    &mut output,
                    &mut current_lines,
                    &mut incoming_lines,
                    current_label,
                    incoming_label,
                );
                output.push_str(change.value());
            }
            ChangeTag::Delete => {
                current_lines.push(change.value());
            }
            ChangeTag::Insert => {
                incoming_lines.push(change.value());
            }
        }
    }
    // flush any remaining conflict at the end
    flush_conflict(
        &mut output,
        &mut current_lines,
        &mut incoming_lines,
        current_label,
        incoming_label,
    );
    output
}

/// Flush buffered conflict lines into the output string with git-style markers.
/// Drains `current_lines` and `incoming_lines` in place; does nothing if both
/// are empty.
///
/// # Arguments
///
/// * `output` - The output string to append conflict markers and lines to
/// * `current_lines` - Buffered lines from the current/left side
/// * `incoming_lines` - Buffered lines from the incoming/right side
/// * `current_label` - The label for the current side's conflict marker
/// * `incoming_label` - The label for the incoming side's conflict marker
fn flush_conflict(
    output: &mut String,
    current_lines: &mut Vec<&str>,
    incoming_lines: &mut Vec<&str>,
    current_label: &str,
    incoming_label: &str,
) {
    if current_lines.is_empty() && incoming_lines.is_empty() {
        return;
    }
    output.push_str("<<<<<<< ");
    output.push_str(current_label);
    output.push('\n');
    for line in current_lines.drain(..) {
        output.push_str(line);
        if !line.ends_with('\n') {
            output.push('\n');
        }
    }
    output.push_str("=======\n");
    for line in incoming_lines.drain(..) {
        output.push_str(line);
        if !line.ends_with('\n') {
            output.push('\n');
        }
    }
    output.push_str(">>>>>>> ");
    output.push_str(incoming_label);
    output.push('\n');
}

/// Check if the content contains any unresolved merge conflict markers.
/// Returns the 1-based line number of the first conflict marker found, if any.
///
/// # Arguments
///
/// * `content` - The file content to scan for conflict markers
fn find_conflict_markers(content: &str) -> Option<usize> {
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("<<<<<<<") || trimmed == "=======" || trimmed.starts_with(">>>>>>>")
        {
            return Some(line_num + 1);
        }
    }
    None
}

// ─── Editor Loop ─────────────────────────────────────────────────────────────

/// A parser error the editor loop can report with an optional source location
///
/// Abstracts over the formats the editor loop validates (YAML/JSON via
/// `serde_norway`, TOML via `toml`) so [`editor_loop_core`] can surface a
/// line/column when the parser provides one and otherwise fall back to the
/// error's own `Display`.
pub(crate) trait EditorParseError: std::fmt::Display {
    /// The 1-based `(line, column)` of the error, when the parser exposes one
    fn location(&self) -> Option<(usize, usize)>;
}

impl EditorParseError for serde_norway::Error {
    /// Pulls the line/column from the YAML error's location, if present
    fn location(&self) -> Option<(usize, usize)> {
        serde_norway::Error::location(self).map(|loc| (loc.line(), loc.column()))
    }
}

impl EditorParseError for toml::de::Error {
    /// TOML errors render a caret-annotated snippet in their `Display`, so the
    /// loop relies on that rather than a separate line/column
    fn location(&self) -> Option<(usize, usize)> {
        None
    }
}

/// Prompt the user to either retry editing or cancel after a validation error
fn prompt_error_action() -> Result<ErrorAction, Error> {
    let items = &[
        "Edit   - Reopen editor to fix the issue",
        "Cancel - Abandon changes for this resource",
    ];
    let selection = dialoguer::Select::new()
        .items(items)
        .default(0)
        .interact()
        .map_err(|err| Error::new(format!("Failed to read user input: {err}")))?;
    Ok(match selection {
        0 => ErrorAction::Edit,
        _ => ErrorAction::Cancel,
    })
}

/// Action the user wants to take after a validation error
enum ErrorAction {
    /// Reopen the editor to fix the issue
    Edit,
    /// Abandon changes for this resource
    Cancel,
}

/// An RAII guard that removes a temporary file when it is dropped
///
/// The editor loop has many exit points (editor launch failure, read failure,
/// parse error, user cancel via an error prompt, and success). Owning the temp
/// path in a guard guarantees the file is cleaned up on every path, including the
/// ones that propagate an error with `?`, so no stray edit files are left behind.
struct TempFile {
    /// The path to the temporary file to remove on drop
    path: std::path::PathBuf,
}

impl TempFile {
    /// Wraps a temp path in a guard that removes it on drop
    ///
    /// # Arguments
    ///
    /// * `path` - The temporary file path to own and clean up
    fn new(path: std::path::PathBuf) -> Self {
        TempFile { path }
    }
    /// Returns the path of the guarded temporary file
    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempFile {
    /// Removes the temporary file, ignoring errors since cleanup is best-effort
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The core editor loop: write `content` to a temp file, open the user's editor, and
/// on each save run `parse` to validate. On unresolved conflict markers or a parse
/// error it prints a helpful message (with line/column when the error carries a
/// location) and prompts the user to reopen the editor or cancel.
///
/// `parse` returns any [`EditorParseError`] on failure so the loop can surface its
/// location. [`editor_loop`] and [`editor_loop_validated`] are thin wrappers over this.
///
/// # Arguments
///
/// * `content` - The initial file content (e.g., YAML with merge conflict markers)
/// * `label` - A label used when naming the temp file (e.g., "image-group-name")
/// * `editor` - The editor command to open
/// * `ext` - The temp file extension (without a dot), so the editor highlights the
///   edited format (e.g., "yml", "json", "toml", "md")
/// * `parse` - Validates the saved text, returning the value to hand back
///
/// # Returns
///
/// The parsed value, or `None` if the user cancelled.
async fn editor_loop_core<R, E: EditorParseError>(
    content: &str,
    label: &str,
    editor: &str,
    ext: &str,
    parse: impl Fn(&str) -> Result<R, E>,
) -> Result<Option<R>, Error> {
    // create a temp directory
    let temp_dir = std::env::temp_dir().join("thorium");
    tokio::fs::create_dir_all(&temp_dir).await.map_err(|err| {
        Error::new(format!(
            "Failed to create temporary directory '{}': {}",
            temp_dir.to_string_lossy(),
            err
        ))
    })?;
    // own the temp path in a guard so every exit below (including `?` from the
    // error prompts) removes the file rather than leaking it
    let temp = TempFile::new(temp_dir.join(format!("merge-{}-{}.{ext}", label, Uuid::new_v4())));
    // write initial content
    write_temp_file(temp.path(), content).await?;
    loop {
        // open the editor
        let status = match tokio::process::Command::new(editor)
            .arg(temp.path())
            .status()
            .await
        {
            Ok(status) => status,
            Err(err) => {
                return Err(Error::new(format!(
                    "Unable to open editor '{editor}': {err}"
                )));
            }
        };
        if !status.success() {
            return Err(match status.code() {
                Some(code) => {
                    Error::new(format!("Editor '{editor}' exited with error code: {code}"))
                }
                None => Error::new(format!("Editor '{editor}' exited with error!")),
            });
        }
        // read back the file
        let resolved = match tokio::fs::read_to_string(temp.path()).await {
            Ok(content) => content,
            Err(err) => {
                return Err(Error::new(format!("Failed to read temporary file: {err}")));
            }
        };
        // check for unresolved conflict markers
        if let Some(line) = find_conflict_markers(&resolved) {
            eprintln!(
                "{} Unresolved merge conflict marker found at line {}. Please resolve all conflicts before saving.",
                "Error:".bright_red().bold(),
                line.to_string().bright_yellow(),
            );
            match prompt_error_action()? {
                ErrorAction::Edit => continue,
                ErrorAction::Cancel => {
                    return Ok(None);
                }
            }
        }
        // validate via the caller's parse function
        match parse(&resolved) {
            Ok(parsed) => {
                // valid — the guard cleans up the temp file as it drops
                return Ok(Some(parsed));
            }
            Err(err) => {
                // surface the location when the parser exposes one; otherwise the
                // error's own Display carries the detail (e.g. TOML's caret snippet)
                if let Some((line, column)) = err.location() {
                    eprintln!(
                        "{} Parse error at line {}, column {}: {}",
                        "Error:".bright_red().bold(),
                        line.to_string().bright_yellow(),
                        column.to_string().bright_yellow(),
                        err,
                    );
                } else {
                    eprintln!("{} Parse error: {}", "Error:".bright_red().bold(), err);
                }
                match prompt_error_action()? {
                    ErrorAction::Edit => continue,
                    ErrorAction::Cancel => {
                        return Ok(None);
                    }
                }
            }
        }
    }
}

/// Open a file in the user's editor with a validation loop, deserializing the result
/// to `T` on success (catches YAML syntax + schema errors). Returns `None` if the user
/// cancelled. See [`editor_loop_core`].
///
/// # Arguments
///
/// * `content` - The initial file content (e.g., YAML with merge conflict markers)
/// * `label` - A label used when naming the temp file
/// * `editor` - The editor command to open
pub async fn editor_loop<T>(content: &str, label: &str, editor: &str) -> Result<Option<T>, Error>
where
    T: DeserializeOwned,
{
    editor_loop_core(content, label, editor, "yml", |resolved| {
        serde_norway::from_str::<T>(resolved)
    })
    .await
}

/// Like [`editor_loop`], but validates the edited content as `T` (typed, with
/// line/column errors) while returning the full edited document as a
/// `serde_json::Value` — every field the user kept, not just `T`'s serialized fields.
/// Used by [`review_config_in_editor`] so scaffolded configs stay complete.
///
/// # Arguments
///
/// * `content` - The initial file content to edit
/// * `label` - A label used when naming the temp file
/// * `editor` - The editor command to open
pub(crate) async fn editor_loop_validated<T>(
    content: &str,
    label: &str,
    editor: &str,
) -> Result<Option<serde_json::Value>, Error>
where
    T: DeserializeOwned,
{
    editor_loop_core(content, label, editor, "yml", |resolved| {
        // validate against the typed config first (its error carries the line/column),
        // then re-parse the same text into a Value so every field is preserved
        serde_norway::from_str::<T>(resolved)?;
        serde_norway::from_str::<serde_json::Value>(resolved)
    })
    .await
}

/// Open an editor on a git-style conflict view between two text blobs, returning the
/// resolved text (or `None` if the user cancelled)
///
/// Generic over the save-time validator's error type so callers can validate the
/// merged result as YAML/JSON, TOML, or not at all (see [`EditorParseError`]). The
/// resolved text is returned verbatim — the caller decides where it lands.
///
/// # Arguments
///
/// * `current` - The current/left side of the merge (e.g., the on-disk file)
/// * `incoming` - The incoming/right side of the merge (e.g., the freshly exported file)
/// * `current_label` - The conflict-marker label for the current side
/// * `incoming_label` - The conflict-marker label for the incoming side
/// * `label` - A label used when naming the temp file
/// * `editor` - The editor command to open
/// * `ext` - The temp file extension (without a dot) for editor highlighting
/// * `validate` - Validates the saved text on each save
#[allow(clippy::too_many_arguments)]
pub(crate) async fn merge_in_editor<E: EditorParseError>(
    current: &str,
    incoming: &str,
    current_label: &str,
    incoming_label: &str,
    label: &str,
    editor: &str,
    ext: &str,
    validate: impl Fn(&str) -> Result<(), E>,
) -> Result<Option<String>, Error> {
    // build the conflict-marked document the user resolves in the editor
    let conflict = generate_conflict_view(current, incoming, current_label, incoming_label);
    // run the shared loop, handing back the resolved text once it validates
    editor_loop_core(&conflict, label, editor, ext, |resolved| {
        validate(resolved).map(|()| resolved.to_string())
    })
    .await
}

/// Write content to a temp file, creating or overwriting it
///
/// # Arguments
///
/// * `path` - The path of the temporary file to write
/// * `content` - The content to write to the file
async fn write_temp_file(path: &std::path::Path, content: &str) -> Result<(), Error> {
    tokio::fs::write(path, content).await.map_err(|err| {
        Error::new(format!(
            "Failed to write temporary file '{}': {}",
            path.to_string_lossy(),
            err
        ))
    })
}

/// Open a config in the user's editor for review, validating it against the typed
/// request `T`, and return the (possibly edited) config as curated, pretty JSON.
///
/// The config is presented as curated YAML for editing and validated as `T` (so bad
/// enum/type values are reported with line/column, with an Edit/Cancel retry). The
/// **full edited document** is written back in curated order — every field, not just
/// `T`'s serialized fields — so scaffolded configs stay complete. If the user cancels,
/// the unchanged default is returned (still in curated order). Shared by `toolbox
/// init`/`export` and `images`/`pipelines export`'s opt-in `--review` pass.
///
/// # Arguments
///
/// * `json_config` - The config to review, as a JSON string
/// * `label` - A label used to name the temporary edit file
/// * `editor` - The editor command to open (resolve via [`resolve_editor`])
/// * `order` - The curated top-level key order (see [`crate::utils::curated_yaml`])
pub(crate) async fn review_config_in_editor<T>(
    json_config: &str,
    label: &str,
    editor: &str,
    order: &[&str],
) -> Result<String, Error>
where
    T: DeserializeOwned,
{
    let value: serde_json::Value = serde_json::from_str(json_config)
        .map_err(|e| Error::new(format!("Failed to parse config for editor review: {e}")))?;
    let yaml = crate::utils::curated_yaml(&value, order)
        .map_err(|e| Error::new(format!("Failed to convert config to YAML: {e}")))?;
    match editor_loop_validated::<T>(&yaml, label, editor).await? {
        // write the edited document in curated order
        Some(resolved) => crate::utils::curated_json(&resolved, order),
        // cancelled: write the unchanged default, still in curated order
        None => crate::utils::curated_json(&value, order),
    }
}
