//! Pipeline import support for thorctl
//!
//! Loads pipeline configs from an on-disk export directory and applies them
//! through the shared conflict engine. The orchestration (image pass first,
//! then pipelines, with one rollback journal spanning both) lives in the
//! parent pipelines handler.

use std::path::Path;
use thorium::models::PipelineRequest;
use thorium::{CtlConf, Error, Thorium};

use crate::handlers::imports::kind::PipelineKind;
use crate::handlers::imports::rollback::Journal;
use crate::handlers::imports::{self, ApplyOutcome, ConflictMode, categorize, create};
use crate::handlers::progress::Bar;

/// Load a pipeline request from the export directory and point it at our group
///
/// # Arguments
///
/// * `import_dir` - The export directory holding `pipelines/<name>.json`
/// * `group` - The group to import the pipeline into
/// * `name` - The name of the pipeline whose config we are loading
pub async fn load_request(
    import_dir: &Path,
    group: &str,
    name: &str,
) -> Result<PipelineRequest, Error> {
    categorize::load_request::<PipelineKind>(import_dir, group, name).await
}

/// Apply the categorized pipelines to Thorium according to the conflict mode
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (used for the default editor)
/// * `pipelines` - The categorized pipelines to apply
/// * `mode` - How to handle pipelines that already exist
/// * `editor` - The editor override for interactive merges
/// * `can_prompt` - Whether an interactive merge may prompt (interactive mode + TTY)
/// * `workers` - Max concurrent API actions in the apply phase (the global `--workers`)
/// * `progress` - The progress bar
/// * `journal` - The journal to record applied changes in
// The positional args mirror `apply_images`; the conflict dispatch itself is the
// shared `imports::apply_existing`.
#[allow(clippy::too_many_arguments)]
pub async fn apply_pipelines(
    thorium: &Thorium,
    conf: &CtlConf,
    pipelines: &[categorize::CategorizedPipeline],
    mode: ConflictMode,
    editor: Option<&str>,
    can_prompt: bool,
    workers: usize,
    progress: &Bar,
    journal: &Journal,
) -> Result<ApplyOutcome, Error> {
    let plan = imports::ImportPlan::new(&[], pipelines, Vec::new());
    // create the pipelines that don't exist yet, collecting per-pipeline failures
    let mut failures =
        create::import_new_pipelines(thorium, plan.new_pipelines, workers, progress, journal).await;
    // handle existing pipelines according to the conflict mode (shared dispatch)
    let existing = imports::apply_existing::<PipelineKind>(
        thorium,
        conf,
        plan.existing_pipelines,
        mode,
        editor,
        can_prompt,
        workers,
        progress,
        journal,
    )
    .await?;
    failures.extend(existing.failures);
    Ok(ApplyOutcome {
        outcome: existing.outcome,
        failures,
    })
}
