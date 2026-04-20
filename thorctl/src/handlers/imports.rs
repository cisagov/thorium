//! The shared conflict engine for importing resources into Thorium
//!
//! All import flows (`toolbox import`, `images import`, `pipelines import`)
//! funnel through this module so they share one model for handling resources
//! that already exist in the target instance:
//!
//! - **Interactive** (default): prompt per changed resource with an
//!   Edit/Skip/Apply/Quit merge flow backed by the user's editor
//! - **Overwrite** (`--overwrite`): apply every incoming change without prompting
//! - **Skip conflicts** (`--skip-conflicts`): never touch existing resources;
//!   warn with the specific fields that differ so the skip is auditable
//!
//! New resources are created in all three modes.

use colored::Colorize;
use futures::{StreamExt, TryStreamExt, stream};
use std::collections::HashSet;
use thorium::models::GroupRequest;
use thorium::{CtlConf, Error, Thorium};

use super::progress::Bar;

pub(crate) mod categorize;
pub(crate) mod create;
// the on-disk import driver depends on the image/pipeline import handlers, which
// are only compiled on the platforms that ship the docker-backed import/export
// commands
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) mod disk;
pub(crate) mod editor;
pub(crate) mod kind;
pub(crate) mod merge;
pub(crate) mod rollback;
pub(crate) mod summary;
pub(crate) mod update;

use kind::{ImageKind, ImportKind, PipelineKind};
use rollback::Journal;

/// Whether an existing image would actually be updated by its incoming request
///
/// Uses the same normalized comparison as the apply phase
/// ([`ImageKind::calculate_update`]) so the confirmation screen never labels a
/// difference that normalizes away (a trimmed description, a defaulted field) as a
/// conflict the apply phase would then treat as a no-op.
///
/// # Arguments
///
/// * `img` - The categorized image to test for an effective change
fn image_would_change(img: &categorize::CategorizedImage) -> bool {
    img.existing.as_ref().is_some_and(|existing| {
        ImageKind::calculate_update(existing.clone(), img.request.clone()).is_some()
    })
}

/// Whether an existing pipeline would actually be updated by its incoming request
///
/// The pipeline counterpart of [`image_would_change`], using
/// [`PipelineKind::calculate_update`] for the same normalized comparison.
///
/// # Arguments
///
/// * `pipe` - The categorized pipeline to test for an effective change
fn pipeline_would_change(pipe: &categorize::CategorizedPipeline) -> bool {
    pipe.existing.as_ref().is_some_and(|existing| {
        PipelineKind::calculate_update(existing.clone(), pipe.request.clone()).is_some()
    })
}

/// Whether an import's apply phase ran to completion or was stopped early
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportOutcome {
    /// Every resource was handled
    Completed,
    /// The user chose Quit in the interactive merge before the end
    Quit,
}

/// How an apply pass ended, plus the per-resource failures it collected
///
/// Per-resource create/update failures are collected and skipped instead of aborting
/// the import: the successful work is kept (never rolled back for these), and the
/// failures are reported together at the end so the run still makes as much progress as
/// possible. `outcome` still carries an interactive `Quit` early-exit.
pub struct ApplyOutcome {
    /// Whether the pass completed or the user quit an interactive merge
    pub outcome: ImportOutcome,
    /// Labels (e.g. `image 'group:name'`) of resources that failed to import
    pub failures: Vec<String>,
}

/// How an import should handle resources that already exist in Thorium
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictMode {
    /// Prompt per changed resource (Edit/Skip/Apply/Quit)
    Interactive,
    /// Apply all incoming changes without prompting
    Force,
    /// Leave existing resources untouched, warning with the changed fields
    SkipConflicts,
}

impl ConflictMode {
    /// Resolve the conflict mode from the standard pair of CLI flags
    ///
    /// The flags are mutually exclusive at the clap level, so both being set
    /// is unreachable; overwrite wins if it somehow happens.
    ///
    /// # Arguments
    ///
    /// * `overwrite` - The `--overwrite` flag (apply all changes without prompting)
    /// * `skip_conflicts` - The `--skip-conflicts` flag (leave existing untouched)
    pub fn from_flags(overwrite: bool, skip_conflicts: bool) -> Self {
        match (overwrite, skip_conflicts) {
            (true, _) => Self::Force,
            (false, true) => Self::SkipConflicts,
            (false, false) => Self::Interactive,
        }
    }
}

/// The categorized resources an import run will act on, used to show the user
/// an accurate summary before anything is applied
pub struct ImportPlan<'a> {
    /// Images that will be created fresh
    pub new_images: Vec<&'a categorize::CategorizedImage>,
    /// Images that already exist and will be updated, skipped, or prompted
    pub existing_images: Vec<&'a categorize::CategorizedImage>,
    /// Pipelines that will be created fresh
    pub new_pipelines: Vec<&'a categorize::CategorizedPipeline>,
    /// Pipelines that already exist and will be updated, skipped, or prompted
    pub existing_pipelines: Vec<&'a categorize::CategorizedPipeline>,
    /// Groups that will be created before resources are imported
    pub missing_groups: Vec<String>,
}

impl<'a> ImportPlan<'a> {
    /// Build a plan by partitioning categorized resources into new vs existing
    ///
    /// # Arguments
    ///
    /// * `images` - The categorized images this import will act on
    /// * `pipelines` - The categorized pipelines this import will act on
    /// * `missing_groups` - The groups that must be created first
    pub fn new(
        images: &'a [categorize::CategorizedImage],
        pipelines: &'a [categorize::CategorizedPipeline],
        missing_groups: Vec<String>,
    ) -> Self {
        // split images into ones with no live counterpart (new) and ones that exist
        let (new_images, existing_images) = images.iter().partition(|img| img.existing.is_none());
        // split pipelines the same way
        let (new_pipelines, existing_pipelines) =
            pipelines.iter().partition(|pipe| pipe.existing.is_none());
        Self {
            new_images,
            existing_images,
            new_pipelines,
            existing_pipelines,
            missing_groups,
        }
    }

    /// Whether any existing resource would actually be updated by its request
    ///
    /// New resources, resources that normalize to no change, and missing groups
    /// are not conflicts. Uses the same normalized comparison as the apply phase
    /// (see [`image_would_change`]/[`pipeline_would_change`]) to decide whether to
    /// prompt before importing: a clean import (only creates, or no effective
    /// changes) proceeds without confirmation.
    pub fn has_conflicts(&self) -> bool {
        self.existing_images.iter().any(|i| image_would_change(i))
            || self.existing_pipelines.iter().any(|p| pipeline_would_change(p))
    }
}

/// The label describing what will happen to existing resources in this mode
fn existing_label(kind: &str, mode: ConflictMode) -> String {
    match mode {
        ConflictMode::Force => format!("Existing {kind} (will be force-updated):"),
        ConflictMode::SkipConflicts => format!("Existing {kind} (changed will be skipped):"),
        ConflictMode::Interactive => format!("Existing {kind} (will prompt for action):"),
    }
}

/// Confirm an import with the user, showing what will be created and what
/// already exists
///
/// # Arguments
///
/// * `conf` - The Thorctl config (used to display the API URL)
/// * `plan` - The categorized resources this import will create or touch
/// * `username` - The current user's name, shown in the confirmation prompt
/// * `mode` - How existing resources will be handled, to label them accurately
pub fn confirm_import(
    conf: &CtlConf,
    plan: &ImportPlan,
    username: &str,
    mode: ConflictMode,
) -> Result<bool, Error> {
    if !plan.new_images.is_empty() {
        println!("{}", "New Images:".bright_green());
        for img in &plan.new_images {
            println!(
                "  {}:{} (group: {})",
                img.name, img.version, img.request.group
            );
        }
    }
    if !plan.existing_images.is_empty() {
        println!("{}", existing_label("Images", mode).bright_yellow());
        for img in &plan.existing_images {
            // label using the same normalized check the apply phase uses so a
            // difference that normalizes away doesn't show as "changed"
            let changed = image_would_change(img);
            let status = if changed {
                "changed".bright_yellow()
            } else {
                "unchanged".bright_blue()
            };
            println!(
                "  {}:{} (group: {}) [{}]",
                img.name, img.version, img.request.group, status
            );
        }
    }
    if !plan.new_pipelines.is_empty() {
        println!("{}", "New Pipelines:".bright_green());
        for pipe in &plan.new_pipelines {
            println!(
                "  {}:{} (group: {})",
                pipe.name, pipe.version, pipe.request.group
            );
        }
    }
    if !plan.existing_pipelines.is_empty() {
        println!("{}", existing_label("Pipelines", mode).bright_yellow());
        for pipe in &plan.existing_pipelines {
            // label using the same normalized check the apply phase uses so a
            // difference that normalizes away doesn't show as "changed"
            let changed = pipeline_would_change(pipe);
            let status = if changed {
                "changed".bright_yellow()
            } else {
                "unchanged".bright_blue()
            };
            println!(
                "  {}:{} (group: {}) [{}]",
                pipe.name, pipe.version, pipe.request.group, status
            );
        }
    }
    if !plan.missing_groups.is_empty() {
        println!("{}", "New Groups:".bright_green());
        for group in &plan.missing_groups {
            println!("  {group}");
        }
    }
    println!();
    let response = dialoguer::Confirm::new()
        .with_prompt(format!(
            "Import the above items to Thorium instance at '{}' as user '{}'?",
            conf.keys.api.bright_green(),
            username.bright_green()
        ))
        .interact()?;
    Ok(response)
}

/// Settle a finished (or stopped) import's journal, offering rollback
///
/// On a clean completion this is a no-op. When the apply phase stopped early —
/// the user quit the merge editor or a step errored — the changes applied so
/// far are either rolled back or reported, depending on whether we can prompt:
///
/// - interactive sessions are asked whether to roll back
/// - non-interactive sessions roll back automatically only with
///   `--rollback-on-failure`; otherwise the applied changes are listed so the
///   partial import is auditable
///
/// The original error (if any) is always propagated after the journal is
/// settled so exit codes still reflect the failure. A rollback that itself
/// fails never replaces that original error — it is surfaced as a warning and
/// the original result is still returned.
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to roll back changes
/// * `progress` - The progress bar
/// * `journal` - The journal of changes applied before the stop
/// * `result` - How the apply phase ended
/// * `can_prompt` - Whether the session can ask the user about rollback
/// * `rollback_on_failure` - Auto-rollback for sessions that can't prompt
pub async fn settle_journal(
    thorium: &Thorium,
    progress: &Bar,
    journal: Journal,
    result: Result<ImportOutcome, Error>,
    can_prompt: bool,
    rollback_on_failure: bool,
) -> Result<ImportOutcome, Error> {
    // a clean completion keeps everything; nothing to settle
    if matches!(result, Ok(ImportOutcome::Completed)) {
        return result;
    }
    // stopping before anything was applied needs no settling either
    if journal.is_empty() {
        return result;
    }
    if can_prompt {
        // ask the user whether the partial import should be undone
        let wants_rollback = progress.suspend(|| rollback::confirm_rollback(&journal))?;
        if wants_rollback {
            // a rollback failure must not mask the original error: surface it as a
            // warning and still fall through to return `result` below
            if let Err(err) = journal.rollback(thorium, progress).await {
                progress.warning(format!("Rollback failed: {err}"));
            }
        }
    } else if rollback_on_failure {
        progress.warning(format!(
            "Import stopped early; rolling back {} applied changes (--rollback-on-failure)",
            journal.len()
        ));
        // surface a rollback failure as a warning rather than replacing the
        // original error that triggered the rollback
        if let Err(err) = journal.rollback(thorium, progress).await {
            progress.warning(format!("Rollback failed: {err}"));
        }
    } else {
        // we can't prompt and weren't told to auto-rollback: leave the changes
        // but make the partial state visible
        progress.warning(format!(
            "Import stopped early with {} changes already applied (pass --rollback-on-failure to auto-undo):",
            journal.len()
        ));
        for line in journal.describe() {
            progress.warning(format!("  applied: {line}"));
        }
    }
    result
}

/// List the resource names with configs in an export directory's subdirectory
///
/// Scans `<export_dir>/<subdir>/*.json` and returns the file stems, sorted for
/// stable ordering. Used by the `--all` import flags.
///
/// # Arguments
///
/// * `export_dir` - The root of the on-disk export
/// * `subdir` - The resource subdirectory to scan ("images" or "pipelines")
pub async fn list_export_configs(
    export_dir: &std::path::Path,
    subdir: &str,
) -> Result<Vec<String>, Error> {
    let dir = export_dir.join(subdir);
    let mut entries = tokio::fs::read_dir(&dir)
        .await
        .map_err(|err| Error::new(format!("Failed to read '{}': {err}", dir.display())))?;
    let mut names = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|err| Error::new(format!("Failed to read '{}': {err}", dir.display())))?
    {
        let path = entry.path();
        // only json files are resource configs; tarballs etc live alongside them
        if path.extension().is_some_and(|ext| ext == "json")
            && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
        {
            names.push(stem.to_string());
        }
    }
    if names.is_empty() {
        return Err(Error::new(format!(
            "No resource configs found in '{}'",
            dir.display()
        )));
    }
    // sort so --all runs are deterministic
    names.sort_unstable();
    Ok(names)
}

/// Warn for each existing resource that differs from its incoming request
///
/// This is the `--skip-conflicts` path: nothing is applied, but every skipped
/// resource names the fields that differ so the operator can audit the skips.
/// Unchanged resources are silent.
///
/// # Arguments
///
/// * `existing` - The categorized resources that already exist in Thorium
/// * `progress` - The progress bar to log warnings through
pub fn warn_skipped<K: kind::ImportKind>(
    existing: &[&categorize::Categorized<K>],
    progress: &Bar,
) {
    for item in existing {
        // only changed resources warrant a warning
        let Some(current) = item.existing.as_ref() else {
            continue;
        };
        if let Some(update) = K::calculate_update(current.clone(), item.request.clone()) {
            progress.warning(format!(
                "Skipping {} '{}:{}' (differs: {}); re-run with --overwrite or resolve interactively",
                K::NOUN,
                K::group(&item.request),
                K::name(&item.request),
                summary::render_changed_fields(&update),
            ));
        }
    }
}

/// Apply the existing (already-in-Thorium) resources of one kind per the conflict mode
///
/// This is the single dispatch every import flow shares — `images import`,
/// `pipelines import`, and `toolbox import` — so force/skip/interactive behave
/// identically for both kinds. Resources that don't yet exist are handled
/// separately by [`create::import_new`].
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (used for the default editor in interactive merges)
/// * `existing` - The categorized resources that already exist in Thorium
/// * `mode` - How existing resources should be handled
/// * `editor` - The editor override for interactive merges
/// * `can_prompt` - Whether the session can drive an interactive merge
/// * `workers` - Max concurrent API actions in the apply phase (the global `--workers`)
/// * `progress` - The progress bar
/// * `journal` - The journal to record applied changes in
#[allow(clippy::too_many_arguments)]
pub async fn apply_existing<K: kind::ImportKind>(
    thorium: &Thorium,
    conf: &CtlConf,
    existing: Vec<&categorize::Categorized<K>>,
    mode: ConflictMode,
    editor: Option<&str>,
    can_prompt: bool,
    workers: usize,
    progress: &Bar,
    journal: &Journal,
) -> Result<ApplyOutcome, Error> {
    match mode {
        ConflictMode::Force => {
            // apply every incoming change without the editor, collecting any per-resource
            // failures so one bad update doesn't abort the rest
            let failures = create::force_update::<K>(thorium, existing, workers, progress, journal).await;
            Ok(ApplyOutcome {
                outcome: ImportOutcome::Completed,
                failures,
            })
        }
        ConflictMode::SkipConflicts => {
            // leave existing resources untouched, warning with what differs
            warn_skipped::<K>(&existing, progress);
            Ok(ApplyOutcome {
                outcome: ImportOutcome::Completed,
                failures: Vec::new(),
            })
        }
        ConflictMode::Interactive if !can_prompt => {
            // no terminal to drive the merge editor: skip differing resources with a
            // warning rather than erroring on a prompt (use --overwrite to apply changes)
            warn_skipped::<K>(&existing, progress);
            Ok(ApplyOutcome {
                outcome: ImportOutcome::Completed,
                failures: Vec::new(),
            })
        }
        ConflictMode::Interactive => {
            // the interactive merge drives per-resource choices with the user present, so
            // it keeps its own (halt-on-apply-error) behavior and collects no failures
            let outcome =
                merge::interactive_merge::<K>(thorium, existing, conf, editor, progress, journal)
                    .await?;
            Ok(ApplyOutcome {
                outcome,
                failures: Vec::new(),
            })
        }
    }
}

/// De-duplicate a list of resource names, preserving first-seen order and
/// warning about each dropped duplicate
///
/// Standalone imports take names straight from the CLI, where the same name can
/// appear twice (`images import a a`); creating it twice would conflict, so the
/// list is collapsed up front.
///
/// # Arguments
///
/// * `names` - The resource names provided on the command line
/// * `progress` - The progress bar to warn through
pub fn dedup_names(names: Vec<String>, progress: &Bar) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut unique = Vec::with_capacity(names.len());
    for name in names {
        if seen.insert(name.clone()) {
            unique.push(name);
        } else {
            progress.warning(format!("Ignoring duplicate '{}'", name.bright_yellow()));
        }
    }
    unique
}

/// Get the list of groups missing in Thorium that an import expects
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `wanted_groups` - The groups the import expects to exist
pub async fn get_missing_groups(
    thorium: &Thorium,
    mut wanted_groups: HashSet<String>,
) -> Result<Vec<String>, Error> {
    // get all existing groups already in Thorium
    let mut thorium_groups = HashSet::new();
    // use a very large limit to make sure we get all groups
    let mut cursor = thorium.groups.list().limit(crate::utils::LIST_ALL_LIMIT);
    loop {
        cursor
            .next()
            .await
            .map_err(|err| Error::new(format!("Error listing groups: {err}")))?;
        thorium_groups.extend(cursor.names.drain(..));
        if cursor.exhausted {
            break;
        }
    }
    // calculate which groups are missing
    Ok(wanted_groups
        .extract_if(|wanted| !thorium_groups.contains(wanted))
        .collect())
}

/// Create all of the given groups in Thorium and increment the progress bar
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `groups` - The groups to create
/// * `workers` - Max concurrent group creates (the global `--workers`)
/// * `progress` - The progress bar
/// * `journal` - The journal to record created groups in for rollback
pub async fn create_groups<T>(
    thorium: &Thorium,
    groups: Vec<T>,
    workers: usize,
    progress: &Bar,
    journal: &Journal,
) -> Result<(), Error>
where
    T: Into<String>,
{
    // create groups concurrently
    stream::iter(groups)
        .map(Ok::<_, Error>)
        .try_for_each_concurrent(workers.max(1), |missing_group| async {
            let group_request = GroupRequest::new(missing_group);
            thorium.groups.create(&group_request).await?;
            journal.created_group(&group_request.name);
            progress.inc(1);
            Ok(())
        })
        .await
        .map_err(|err| Error::new(format!("Error creating missing groups: {err}")))?;
    Ok(())
}
