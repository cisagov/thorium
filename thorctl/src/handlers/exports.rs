//! Shared helpers for commands that export resources to disk
//!
//! Exports (`toolbox export`, `images export`, `pipelines export`) all write
//! config files that may already exist on disk from a previous run or a manual
//! edit. [`DiskConflictResolver`] gives them one consistent way to handle that:
//! a file whose on-disk content matches what we'd write is a silent no-op, and a
//! file that *differs* is either overwritten (`--overwrite`), resolved by prompting
//! the user (Merge / Overwrite / Skip / Overwrite-all / Skip-all / Quit), or —
//! when we can't prompt and `--overwrite` wasn't given — skipped with a warning so
//! nothing is silently clobbered.
//!
//! "Merge" opens the file's on-disk and freshly-exported versions in the user's
//! editor as a git-style conflict view (see [`crate::handlers::imports::editor`]),
//! validating the result on save against the file's format.
//!
//! The resolver must be driven from a single sequential task (it prompts and
//! remembers "all" choices), so concurrent exporters run it in a pre-flight pass
//! rather than from their worker pool.

use colored::Colorize;
use serde::de::DeserializeOwned;
use std::borrow::Cow;
use std::path::Path;
use thorium::Error;

use crate::handlers::imports::editor::{self, EditorParseError};
use crate::handlers::progress::Bar;

/// What happened when a file was offered to one of [`DiskConflictResolver`]'s
/// `write_text`/`write_yaml`/`write_toml` methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// The file was written (new, or an approved overwrite)
    Written,
    /// The file was left as-is (identical content, or a skipped conflict)
    Skipped,
    /// The user chose to stop the whole export
    Quit,
}

/// How a conflict on an existing, differing file resolves before prompting
enum Resolution {
    /// Overwrite without asking (`--overwrite` or a remembered "overwrite all")
    Overwrite,
    /// Skip without asking (a remembered "skip all", or non-interactive)
    Skip,
    /// Ask the user
    Prompt,
}

/// Resolves on-disk write conflicts for an export, remembering "all" choices for
/// the rest of the run
pub struct DiskConflictResolver {
    /// Overwrite differing files without asking
    overwrite: bool,
    /// Whether the session can prompt (a tty and not run non-interactively)
    can_prompt: bool,
    /// The user chose "overwrite all" earlier this run
    overwrite_all: bool,
    /// The user chose "skip all" earlier this run
    skip_all: bool,
    /// The editor command used to resolve a conflict via the "Merge" choice
    editor: String,
}

impl DiskConflictResolver {
    /// Create a resolver
    ///
    /// # Arguments
    ///
    /// * `overwrite` - Overwrite differing files without prompting
    /// * `can_prompt` - Whether the session can ask the user (tty, interactive)
    /// * `editor` - The editor command used when the user chooses "Merge"
    pub fn new(overwrite: bool, can_prompt: bool, editor: String) -> Self {
        Self {
            overwrite,
            can_prompt,
            overwrite_all: false,
            skip_all: false,
            editor,
        }
    }

    /// Write `content` to `path` with no save-time validation
    ///
    /// For files without a parseable schema (e.g. `description.md`); a "Merge"
    /// only checks for unresolved conflict markers.
    ///
    /// # Arguments
    ///
    /// * `path` - The file to write
    /// * `content` - The content we want on disk
    /// * `progress` - The progress bar (suspended while prompting)
    pub async fn write_text(
        &mut self,
        path: &Path,
        content: &str,
        progress: &Bar,
    ) -> Result<WriteOutcome, Error> {
        // markdown/plain text has no schema, so a merge only needs the marker check
        self.write_inner(path, content, progress, |_| {
            Ok::<(), serde_norway::Error>(())
        })
        .await
    }

    /// Write `content` to `path`, validating a "Merge" result as YAML/JSON that
    /// deserializes into `T`
    ///
    /// `serde_norway` parses JSON (a YAML subset), so this covers the on-disk
    /// `.json` resource and network-policy configs.
    ///
    /// # Arguments
    ///
    /// * `path` - The file to write
    /// * `content` - The content we want on disk
    /// * `progress` - The progress bar (suspended while prompting)
    pub async fn write_yaml<T: DeserializeOwned>(
        &mut self,
        path: &Path,
        content: &str,
        progress: &Bar,
    ) -> Result<WriteOutcome, Error> {
        // validate a merged config against its request type so a broken hand-merge
        // is caught on save (with line/column) rather than at import time
        self.write_inner(path, content, progress, |resolved| {
            serde_norway::from_str::<T>(resolved).map(|_| ())
        })
        .await
    }

    /// Write `content` to `path`, validating a "Merge" result as TOML that
    /// deserializes into `T`
    ///
    /// Covers `manifest.toml` and `config.toml`.
    ///
    /// # Arguments
    ///
    /// * `path` - The file to write
    /// * `content` - The content we want on disk
    /// * `progress` - The progress bar (suspended while prompting)
    pub async fn write_toml<T: DeserializeOwned>(
        &mut self,
        path: &Path,
        content: &str,
        progress: &Bar,
    ) -> Result<WriteOutcome, Error> {
        // validate a merged manifest/config so a broken hand-merge is caught on save
        self.write_inner(path, content, progress, |resolved| {
            toml::from_str::<T>(resolved).map(|_| ())
        })
        .await
    }

    /// Write `content` to `path`, resolving the case where `path` already exists
    /// with different content
    ///
    /// On a conflict the user can Merge (edit a conflict view, validated by
    /// `validate` on save), Overwrite, Skip, Overwrite/Skip all, or Quit.
    ///
    /// # Arguments
    ///
    /// * `path` - The file to write
    /// * `content` - The content we want on disk
    /// * `progress` - The progress bar (suspended while prompting)
    /// * `validate` - Validates a merged result on each save
    async fn write_inner<E: EditorParseError>(
        &mut self,
        path: &Path,
        content: &str,
        progress: &Bar,
        validate: impl Fn(&str) -> Result<(), E>,
    ) -> Result<WriteOutcome, Error> {
        let existing = tokio::fs::read_to_string(path).await.ok();
        // identical content already on disk: nothing to do
        if existing.as_deref() == Some(content) {
            return Ok(WriteOutcome::Skipped);
        }
        // the text we ultimately write — the new content, unless a merge replaces it
        let mut to_write = Cow::Borrowed(content);
        // an existing-but-different file needs a decision before we clobber it
        if let Some(existing) = &existing {
            match self.resolve(path, progress) {
                Resolution::Skip => return Ok(WriteOutcome::Skipped),
                Resolution::Overwrite => {}
                // `resolve` only returns Prompt when it didn't resolve itself
                Resolution::Prompt => match prompt_conflict(path, progress)? {
                    PromptChoice::Merge => {
                        // open a conflict view of on-disk vs new content in the editor,
                        // naming the temp file by the on-disk extension for highlighting
                        let ext = path
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .unwrap_or("txt");
                        let label = path
                            .file_stem()
                            .and_then(|stem| stem.to_str())
                            .unwrap_or("merge");
                        let merged = progress
                            .suspend_async(editor::merge_in_editor(
                                existing,
                                content,
                                "Existing (on disk)",
                                "New (from export)",
                                label,
                                &self.editor,
                                ext,
                                &validate,
                            ))
                            .await?;
                        // cancelled in the editor: keep the on-disk file
                        let Some(merged) = merged else {
                            progress.info_anonymous(format!("Skipping '{}'", path.display()));
                            return Ok(WriteOutcome::Skipped);
                        };
                        // write what the user resolved
                        to_write = Cow::Owned(merged);
                    }
                    PromptChoice::Overwrite => {}
                    PromptChoice::Skip => {
                        progress.info_anonymous(format!("Skipping '{}'", path.display()));
                        return Ok(WriteOutcome::Skipped);
                    }
                    PromptChoice::OverwriteAll => self.overwrite_all = true,
                    PromptChoice::SkipAll => {
                        self.skip_all = true;
                        progress.info_anonymous(format!("Skipping '{}'", path.display()));
                        return Ok(WriteOutcome::Skipped);
                    }
                    PromptChoice::Quit => return Ok(WriteOutcome::Quit),
                },
            }
        }
        // create the parent directory and write
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|err| {
                Error::new(format!(
                    "Failed to create directory '{}': {err}",
                    parent.display()
                ))
            })?;
        }
        tokio::fs::write(path, to_write.as_ref())
            .await
            .map_err(|err| Error::new(format!("Failed to write '{}': {err}", path.display())))?;
        Ok(WriteOutcome::Written)
    }

    /// Resolve a conflict from remembered state alone (no prompting, no IO),
    /// returning [`Resolution::Prompt`] when the caller still needs to ask
    fn resolve_kind(&self) -> Resolution {
        if self.overwrite || self.overwrite_all {
            Resolution::Overwrite
        } else if self.skip_all {
            Resolution::Skip
        } else if self.can_prompt {
            Resolution::Prompt
        } else {
            // non-interactive and not forced
            Resolution::Skip
        }
    }

    /// Resolve a conflict, warning when we auto-skip a differing file because we
    /// can't prompt and weren't told to overwrite
    fn resolve(&self, path: &Path, progress: &Bar) -> Resolution {
        let kind = self.resolve_kind();
        // the only un-chosen skip is the non-interactive case; surface it loudly
        if matches!(kind, Resolution::Skip) && !self.skip_all {
            progress.warning(format!(
                "Skipping '{}' — an existing on-disk copy differs; pass --overwrite to overwrite",
                path.display()
            ));
        }
        kind
    }
}

/// The user's per-file choice when an existing file differs
enum PromptChoice {
    /// Open an editor to merge the on-disk and new versions
    Merge,
    /// Overwrite this one file
    Overwrite,
    /// Keep this one file unchanged
    Skip,
    /// Overwrite this file and every later conflict
    OverwriteAll,
    /// Keep this file and every later conflict unchanged
    SkipAll,
    /// Stop the export
    Quit,
}

/// Map a 0-based menu selection to a [`PromptChoice`]
///
/// Merge is first (the default), followed by the overwrite/skip/all/quit options;
/// any out-of-range value falls back to Quit.
///
/// # Arguments
///
/// * `selection` - The 0-based index the user picked from the conflict menu
fn choice_from_selection(selection: usize) -> PromptChoice {
    match selection {
        0 => PromptChoice::Merge,
        1 => PromptChoice::Overwrite,
        2 => PromptChoice::Skip,
        3 => PromptChoice::OverwriteAll,
        4 => PromptChoice::SkipAll,
        _ => PromptChoice::Quit,
    }
}

/// Prompt the user about a single differing file
fn prompt_conflict(path: &Path, progress: &Bar) -> Result<PromptChoice, Error> {
    progress.suspend(|| {
        println!(
            "\n{} '{}' already exists on disk with different content.",
            "Conflict:".bright_yellow(),
            path.display().to_string().bright_blue(),
        );
        let items = &[
            "Merge         - open an editor to merge the on-disk and new versions",
            "Overwrite     - replace the file on disk",
            "Skip          - keep the file on disk unchanged",
            "Overwrite all - replace this and every later conflict",
            "Skip all      - keep this and every later conflict",
            "Quit          - stop the export",
        ];
        let selection = dialoguer::Select::new()
            .items(items)
            .default(0)
            .interact()
            .map_err(|err| Error::new(format!("Failed to read user input: {err}")))?;
        Ok(choice_from_selection(selection))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--overwrite` and "overwrite all" overwrite without prompting
    #[test]
    fn resolves_overwrite_without_prompt() {
        let forced = DiskConflictResolver::new(true, true, "vi".to_string());
        assert!(matches!(forced.resolve_kind(), Resolution::Overwrite));
        let mut all = DiskConflictResolver::new(false, true, "vi".to_string());
        all.overwrite_all = true;
        assert!(matches!(all.resolve_kind(), Resolution::Overwrite));
    }

    /// "skip all" skips without prompting
    #[test]
    fn resolves_skip_all_without_prompt() {
        let mut all = DiskConflictResolver::new(false, true, "vi".to_string());
        all.skip_all = true;
        assert!(matches!(all.resolve_kind(), Resolution::Skip));
    }

    /// a prompt-capable session defers to the user
    #[test]
    fn resolves_prompt_when_interactive() {
        let interactive = DiskConflictResolver::new(false, true, "vi".to_string());
        assert!(matches!(interactive.resolve_kind(), Resolution::Prompt));
    }

    /// a non-interactive, non-forced session skips (and would warn)
    #[test]
    fn resolves_skip_when_noninteractive() {
        let headless = DiskConflictResolver::new(false, false, "vi".to_string());
        assert!(matches!(headless.resolve_kind(), Resolution::Skip));
    }

    /// the conflict menu maps Merge first (the default) and Quit for any
    /// out-of-range selection
    #[test]
    fn selection_maps_merge_first() {
        assert!(matches!(choice_from_selection(0), PromptChoice::Merge));
        assert!(matches!(choice_from_selection(1), PromptChoice::Overwrite));
        assert!(matches!(choice_from_selection(2), PromptChoice::Skip));
        assert!(matches!(
            choice_from_selection(3),
            PromptChoice::OverwriteAll
        ));
        assert!(matches!(choice_from_selection(4), PromptChoice::SkipAll));
        assert!(matches!(choice_from_selection(5), PromptChoice::Quit));
        assert!(matches!(choice_from_selection(99), PromptChoice::Quit));
    }
}
