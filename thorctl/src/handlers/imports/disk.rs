//! The shared driver for on-disk imports (`images import`, `pipelines import`)
//!
//! Both commands share the same skeleton — categorize, confirm on conflict,
//! create the target group, apply images then pipelines under one rollback
//! journal, and settle. The only difference is whether there are pipelines to
//! apply, so `images import` simply passes an empty pipeline list. The toolbox
//! import has extra pre-apply steps (policies, bundled images, collisions) and
//! drives its own flow.

use std::collections::HashSet;
use thorium::{CtlConf, Error, Thorium};

use super::categorize::{CategorizedImage, CategorizedPipeline};
use super::rollback::Journal;
use super::{ApplyOutcome, ConflictMode, ImportOutcome, ImportPlan};
use crate::handlers::images::import::{ImageImportOpts, apply_images};
use crate::handlers::pipelines::import::apply_pipelines;
use crate::handlers::progress::{Bar, BarKind};

/// Run an on-disk import of already-categorized images and (optionally) pipelines
///
/// Images are applied before the pipelines that reference them, under one journal,
/// so a Quit or error partway through can offer to undo everything. Owns the
/// whole flow from the confirmation gate through settling the journal.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (used for the default editor and confirmation)
/// * `progress` - The progress bar
/// * `opts` - The image import options (carries mode/editor/group/registry/etc.)
/// * `images` - The categorized images to apply (docker work happens in `apply_images`)
/// * `pipelines` - The categorized pipelines to apply (empty for `images import`)
/// * `rollback_on_failure` - Auto-roll-back a partial import when we can't prompt
pub async fn run_disk_import(
    thorium: &Thorium,
    conf: &CtlConf,
    progress: &Bar,
    opts: &ImageImportOpts<'_>,
    mut images: Vec<CategorizedImage>,
    pipelines: Vec<CategorizedPipeline>,
    rollback_on_failure: bool,
) -> Result<(), Error> {
    // the target group is auto-created if it doesn't exist yet
    let missing_groups =
        super::get_missing_groups(thorium, HashSet::from([opts.group.to_string()])).await?;
    let plan = ImportPlan::new(&images, &pipelines, missing_groups);
    // we can prompt only in the interactive default with a TTY; this gates the
    // plan confirmation, the merge editor, and the rollback offer
    let can_prompt = opts.mode == ConflictMode::Interactive && opts.is_terminal;
    // confirm whenever we're about to run an interactive merge: interactive mode,
    // a TTY, and a real conflict. force/skip-conflicts and clean imports never prompt.
    if can_prompt && plan.has_conflicts() {
        let current_user = thorium
            .users
            .info()
            .await
            .map_err(|err| Error::new(format!("Error getting current user info: {err}")))?;
        let confirmed = progress
            .suspend(|| super::confirm_import(conf, &plan, &current_user.username, opts.mode))?;
        if !confirmed {
            return Ok(());
        }
    }
    let missing_groups = plan.missing_groups.clone();
    // the plan borrows the categorized images; drop it before applying mutates them
    drop(plan);
    // journal every applied change so a partial import can be rolled back
    let journal = Journal::new();
    let result: Result<ApplyOutcome, Error> = async {
        // create the target group if needed
        if !missing_groups.is_empty() {
            progress.refresh(
                "Creating groups",
                BarKind::Bound(missing_groups.len() as u64),
            );
            super::create_groups(thorium, missing_groups, opts.workers, progress, &journal)
                .await?;
        }
        // import the images first since the pipelines reference them; a Quit
        // in the image pass stops the pipeline pass too
        let images_applied =
            apply_images(thorium, conf, opts, &mut images, progress, &journal).await?;
        if images_applied.outcome == ImportOutcome::Quit {
            return Ok(images_applied);
        }
        // a no-op when `pipelines` is empty (the `images import` case)
        let pipelines_applied = apply_pipelines(
            thorium,
            conf,
            &pipelines,
            opts.mode,
            opts.editor,
            can_prompt,
            opts.workers,
            progress,
            &journal,
        )
        .await?;
        // carry forward both passes' collected failures
        Ok(ApplyOutcome {
            outcome: pipelines_applied.outcome,
            failures: images_applied
                .failures
                .into_iter()
                .chain(pipelines_applied.failures)
                .collect(),
        })
    }
    .await;
    // per-resource failures are kept (not rolled back) and reported below; settle the
    // journal only on the outcome/error so rollback still covers a Quit or fatal error
    let (settle_input, failures) = match result {
        Ok(applied) => (Ok(applied.outcome), applied.failures),
        Err(err) => (Err(err), Vec::new()),
    };
    // rollback can be offered only when we could prompt (interactive mode + TTY)
    super::settle_journal(
        thorium,
        progress,
        journal,
        settle_input,
        can_prompt,
        rollback_on_failure,
    )
    .await?;
    progress.refresh("Import complete!", BarKind::Timer);
    progress.finish();
    // surface every resource that failed to import and exit non-zero, after keeping the
    // resources that succeeded
    if !failures.is_empty() {
        return Err(Error::new(format!(
            "Import completed with {} failed resource(s): {}",
            failures.len(),
            failures.join(", ")
        )));
    }
    Ok(())
}
