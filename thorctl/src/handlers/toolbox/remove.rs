//! Remove a previously imported toolbox's resources from Thorium
//!
//! The manifest is the source of truth for what to remove: every pipeline and
//! image it names is deleted from the target instance if present. Pipelines
//! are deleted before the images they reference so the API never sees an
//! image deletion that would orphan a pipeline. Groups are never deleted.

use colored::Colorize;
use http::StatusCode;
use std::collections::{HashMap, HashSet};
use thorium::{CtlConf, Error, Thorium};

use super::shared;
use crate::args::toolbox::RemoveToolbox;
use crate::handlers::imports::categorize::{self, CategorizedImage, CategorizedPipeline};
use crate::handlers::progress::{Bar, BarKind};

/// A resource's `(group, name)` identity in Thorium
type Identity = (String, String);

/// Collapse categorized entries into the unique `(group, name)` identities that
/// exist and should be deleted, and flag the identities the manifest defines more
/// than once
///
/// Deleting the same identity twice would error on the second call, so existing
/// targets are de-duplicated (first-seen order). A duplicated identity also means
/// an import could have resolved the collision by renaming the extras to names
/// this manifest doesn't carry — so those copies can't be targeted for removal
/// and the caller should warn about them.
///
/// # Arguments
///
/// * `entries` - `(group, name, exists)` for every flattened manifest entry
fn dedup_targets<'a, I>(entries: I) -> (Vec<Identity>, HashSet<Identity>)
where
    I: IntoIterator<Item = (&'a str, &'a str, bool)>,
{
    // count every occurrence of each identity (existing or not) so any identity
    // the manifest names more than once can be flagged as a collision later
    let mut counts: HashMap<Identity, usize> = HashMap::new();
    // the ordered, de-duplicated identities we will actually try to delete
    let mut targets: Vec<Identity> = Vec::new();
    // identities already pushed to `targets`, so each is deleted at most once
    let mut seen: HashSet<Identity> = HashSet::new();
    for (group, name, exists) in entries {
        // an identity is the owned (group, name) pair; clone because it feeds
        // both the count map and the target/seen sets below
        let key = (group.to_string(), name.to_string());
        // tally this occurrence regardless of existence; the count drives the
        // duplicate warning even for identities that aren't delete targets
        *counts.entry(key.clone()).or_default() += 1;
        // only target identities that exist in the instance, and only the first
        // time each is seen, so a re-listed identity isn't deleted twice (the
        // second delete would 404/error) — first-seen order keeps output stable
        if exists && seen.insert(key.clone()) {
            targets.push(key);
        }
    }
    // an identity counted more than once was defined multiple times in the
    // manifest; a prior import may have renamed the extras to names we can't
    // derive here, so the caller warns those renamed copies may remain
    let duplicates = counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(key, _)| key)
        .collect();
    (targets, duplicates)
}

/// Confirm the removal with the user, listing what exists and what doesn't
///
/// # Arguments
///
/// * `conf` - The Thorctl config (used to display the API URL)
/// * `pipelines` - The categorized pipelines named by the manifest
/// * `images` - The categorized images named by the manifest
fn confirm_remove(
    conf: &CtlConf,
    pipelines: &[CategorizedPipeline],
    images: &[CategorizedImage],
) -> Result<bool, Error> {
    // a resource exists in the instance when categorization found a match; only
    // these are real delete targets, so they are what we list and confirm
    let found_pipelines: Vec<_> = pipelines
        .iter()
        .filter(|pipe| pipe.existing.is_some())
        .collect();
    let found_images: Vec<_> = images.iter().filter(|img| img.existing.is_some()).collect();
    // surface pipelines first because removal deletes them first; skip the header
    // entirely when none exist so the prompt isn't cluttered with empty sections
    if !found_pipelines.is_empty() {
        println!("{}", "Pipelines to delete:".bright_red());
        for pipe in &found_pipelines {
            println!("  {}:{}", pipe.request.group, pipe.request.name);
        }
    }
    // then images, which are deleted after the pipelines that may reference them
    if !found_images.is_empty() {
        println!("{}", "Images to delete:".bright_red());
        for img in &found_images {
            println!("  {}:{}", img.request.group, img.request.name);
        }
    }
    // collect everything the manifest names but the instance doesn't have, so the
    // user sees these are intentionally skipped (not silently dropped) — a missing
    // resource is a no-op, never a failure
    let missing: Vec<String> = pipelines
        .iter()
        .filter(|pipe| pipe.existing.is_none())
        .map(|pipe| format!("pipeline {}:{}", pipe.request.group, pipe.request.name))
        .chain(
            images
                .iter()
                .filter(|img| img.existing.is_none())
                .map(|img| format!("image {}:{}", img.request.group, img.request.name)),
        )
        .collect();
    // only show the skipped section when there is something skipped
    if !missing.is_empty() {
        println!("{}", "Not found (skipped):".bright_blue());
        for line in missing {
            println!("  {line}");
        }
    }
    // blank line separates the listing from the prompt for readability
    println!();
    // default to No so a stray Enter never deletes anything; the prompt names the
    // API url so the user can confirm they are pointed at the right instance
    let response = dialoguer::Confirm::new()
        .with_prompt(format!(
            "Delete the resources listed above from '{}'?",
            conf.keys.api.bright_green()
        ))
        .default(false)
        .interact()?;
    Ok(response)
}

/// Remove a toolbox's pipelines and images from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `cmd` - The toolbox remove command that was run
pub async fn remove(thorium: Thorium, conf: CtlConf, cmd: &RemoveToolbox) -> Result<(), Error> {
    // the manifest is the source of truth for what to remove; load it the same
    // way import does (path or URL) so removal targets exactly what was imported
    let location = &cmd.manifest;
    let (mut manifest, progress) = shared::get_manifest(location).await?;
    // resolve any URL-backed configs/policies so flattening sees concrete entries
    shared::resolve_manifest_configs(&mut manifest, &progress).await?;
    // apply the same group override an import would have used, so the (group, name)
    // identities we compute match the ones actually created in the instance
    if let Some(group_override) = &cmd.group_override {
        manifest = manifest.override_group(group_override);
    }
    // categorize against the instance so each entry carries whether it exists;
    // this is what lets us delete only what is present and skip the rest
    let images = categorize::categorize_images(
        &thorium,
        super::import::flatten_manifest_images(&manifest),
        &progress,
    )
    .await?;
    let pipelines = categorize::categorize_pipelines(
        &thorium,
        super::import::flatten_manifest_pipelines(&manifest),
        &progress,
    )
    .await?;
    // collapse duplicate (group, name) targets so we never delete the same
    // resource twice (the second call would error), and flag identities the
    // toolbox defines more than once
    let (pipeline_targets, pipeline_dups) = dedup_targets(
        pipelines.iter().map(|pipe| {
            (
                pipe.request.group.as_str(),
                pipe.request.name.as_str(),
                pipe.existing.is_some(),
            )
        }),
    );
    // same de-duplication for images, computed separately so image and pipeline
    // duplicate warnings can be labeled distinctly
    let (image_targets, image_dups) = dedup_targets(images.iter().map(|img| {
        (
            img.request.group.as_str(),
            img.request.name.as_str(),
            img.existing.is_some(),
        )
    }));
    // a duplicated identity means an import may have renamed the extras to names
    // we can't derive from this manifest, so those copies may still remain
    warn_duplicates("Pipeline", pipeline_dups, &progress);
    warn_duplicates("Image", image_dups, &progress);
    // nothing from this toolbox is present, so there is nothing to delete or
    // confirm; finish the progress bar and return success rather than prompting
    if pipeline_targets.is_empty() && image_targets.is_empty() {
        progress.finish();
        println!("Nothing to remove: no resources from this toolbox exist in the instance");
        return Ok(());
    }
    // deleting is irreversible, so confirm exactly what will be removed
    if !cmd.skip_confirm {
        // fail clearly (not with a raw dialoguer error) when we can't prompt
        crate::utils::require_confirm_terminal("--skip-confirm (-y)")?;
        let confirmed = progress.suspend(|| confirm_remove(&conf, &pipelines, &images))?;
        if !confirmed {
            return Ok(());
        }
    }
    // labels of resources whose deletion failed; collected so one failure doesn't abort
    // the rest — we make as much progress as possible and report everything at the end
    let mut failures: Vec<String> = Vec::new();
    // delete pipelines first so no image deletion can orphan one
    progress.refresh(
        "Deleting pipelines",
        BarKind::Bound(pipeline_targets.len() as u64),
    );
    for (group, name) in &pipeline_targets {
        // a missing resource is treated as already-removed, not a failure, so a
        // collision or a re-run never aborts the rest of the removal
        match thorium.pipelines.delete(group, name).await {
            Ok(_) => progress.info_anonymous(format!("Deleted pipeline '{group}:{name}'")),
            Err(err) if err.status() == Some(StatusCode::NOT_FOUND) => {
                progress.info_anonymous(format!("Pipeline '{group}:{name}' already removed"));
            }
            // log the failure and keep going so the remaining resources still get deleted
            Err(err) => {
                progress.warning(format!("Failed to delete pipeline '{group}:{name}': {err}"));
                failures.push(format!("pipeline {group}:{name}"));
            }
        }
        // advance the bar whether the delete succeeded, 404'd, or failed, since
        // every outcome is one fully-handled target
        progress.inc(1);
    }
    // images are deleted only after all pipelines, so by now nothing this run
    // tracked still references them; size the bar to the image target count
    progress.refresh(
        "Deleting images",
        BarKind::Bound(image_targets.len() as u64),
    );
    for (group, name) in &image_targets {
        // as with pipelines, a 404 means the image is already gone and counts as
        // success; only other errors are treated as failures
        match thorium.images.delete(group, name).await {
            Ok(_) => progress.info_anonymous(format!("Deleted image '{group}:{name}'")),
            Err(err) if err.status() == Some(StatusCode::NOT_FOUND) => {
                progress.info_anonymous(format!("Image '{group}:{name}' already removed"));
            }
            // log the failure and keep going; a pipeline that still references this image
            // (e.g. its own delete failed above) is the likely cause and is reported too
            Err(err) => {
                progress.warning(format!("Failed to delete image '{group}:{name}': {err}"));
                failures.push(format!("image {group}:{name}"));
            }
        }
        // advance the bar for every handled image target, regardless of outcome
        progress.inc(1);
    }
    // every target was attempted; if any failed, surface them all and exit non-zero so a
    // partial removal isn't silently reported as a success
    if !failures.is_empty() {
        progress.refresh("Removal finished with errors", BarKind::Timer);
        progress.finish();
        return Err(Error::new(format!(
            "Failed to delete {} resource(s): {}",
            failures.len(),
            failures.join(", ")
        )));
    }
    // every target deleted (or already absent) with no failures, so report success
    progress.refresh("Removal complete!", BarKind::Timer);
    progress.finish();
    Ok(())
}

/// Warn that duplicated `(group, name)` identities may have left renamed copies
/// behind that this removal can't target
///
/// # Arguments
///
/// * `kind` - "Image" or "Pipeline", used to start the message
/// * `duplicates` - The duplicated identities
/// * `progress` - The progress bar to warn through
fn warn_duplicates(kind: &str, duplicates: HashSet<Identity>, progress: &Bar) {
    // the set has no order; collect and sort so warnings are deterministic across
    // runs (stable output for logs and tests)
    let mut duplicates: Vec<Identity> = duplicates.into_iter().collect();
    duplicates.sort();
    // emit one warning per duplicated identity; the renamed copies a prior import
    // may have created can't be derived here, so we can only warn, not delete them
    for (group, name) in duplicates {
        progress.warning(format!(
            "{kind} '{}:{}' is defined more than once in the toolbox; if a prior import \
             renamed the duplicates, those renamed copies were not removed",
            group.bright_yellow(),
            name.bright_yellow(),
        ));
    }
}

/// Unit tests for the removal helpers that have no instance dependency
#[cfg(test)]
mod tests {
    use super::*;
    /// Existing identities are de-duplicated (kept once, in order) and repeated
    /// identities are flagged; non-existent entries are never delete targets
    #[test]
    fn dedups_existing_and_flags_duplicates() {
        // mix of a repeated existing identity, a unique existing identity, and a
        // non-existent one to exercise every branch of dedup_targets at once
        let entries = vec![
            ("static", "exiftool", true),
            // same identity again, so it must be counted as a duplicate
            ("static", "exiftool", true),
            ("static", "yara", true),
            // not present in the instance, so never a target or a duplicate
            ("static", "ghost", false),
        ];
        let (targets, duplicates) = dedup_targets(entries);
        // the repeated identity collapses to one target and order is preserved
        assert_eq!(
            targets,
            vec![
                ("static".to_string(), "exiftool".to_string()),
                ("static".to_string(), "yara".to_string()),
            ]
        );
        // the twice-listed identity is flagged; the once-listed one is not
        assert!(duplicates.contains(&("static".to_string(), "exiftool".to_string())));
        assert!(!duplicates.contains(&("static".to_string(), "yara".to_string())));
        // a non-existent identity is neither a delete target nor a duplicate
        assert!(!targets.contains(&("static".to_string(), "ghost".to_string())));
    }
}
