//! Main entry point for toolbox imports
//!
//! Orchestrates the import workflow: loading and validating the manifest, resolving
//! `(group, name)` collisions, categorizing resources against the instance, confirming
//! with the user, then applying — creating missing groups and network policies, pushing
//! any bundled container images, creating new resources, and merging existing ones — with
//! every applied change journaled so a partial import can be rolled back.

use colored::Colorize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thorium::models::{ImageRequest, PipelineRequest};
use thorium::{CtlConf, Error, Thorium};

use super::manifest::ToolboxManifest;
use super::{collisions, policies, shared};
use crate::args::toolbox::{ImportToolbox, ManifestLocation};
use crate::handlers::container;
use crate::handlers::imports::categorize;
use crate::handlers::imports::kind::{ImageKind, PipelineKind};
use crate::handlers::imports::rollback::Journal;
use crate::handlers::imports::{
    self, ApplyOutcome, ConflictMode, ImportOutcome, ImportPlan, create,
};
use crate::handlers::progress::{Bar, BarKind};

/// Flatten a toolbox manifest's images into (name, version, request) tuples
/// for categorization
///
/// # Arguments
///
/// * `manifest` - The toolbox manifest to flatten
pub(super) fn flatten_manifest_images(
    manifest: &ToolboxManifest,
) -> Vec<(String, String, ImageRequest)> {
    // expand each image into one tuple per version, skipping versions with no
    // embedded config (build-only entries carry tags but no request to import)
    manifest
        .images
        .iter()
        .flat_map(|(image_name, image_manifest)| {
            image_manifest
                .versions
                .iter()
                .filter_map(move |(version_name, version)| {
                    version
                        .config
                        .as_ref()
                        .map(|config| (image_name.clone(), version_name.clone(), config.clone()))
                })
        })
        .collect()
}

/// Flatten a toolbox manifest's pipelines into (name, version, request) tuples
/// for categorization
///
/// # Arguments
///
/// * `manifest` - The toolbox manifest to flatten
pub(super) fn flatten_manifest_pipelines(
    manifest: &ToolboxManifest,
) -> Vec<(String, String, PipelineRequest)> {
    // expand each pipeline into one tuple per version, skipping versions with no
    // embedded config (a version dropped by validation has no request to import)
    manifest
        .pipelines
        .iter()
        .flat_map(|(pipeline_name, pipeline_manifest)| {
            pipeline_manifest
                .versions
                .iter()
                .filter_map(move |(version_name, version)| {
                    version
                        .config
                        .as_ref()
                        .map(|config| (pipeline_name.clone(), version_name.clone(), config.clone()))
                })
        })
        .collect()
}

// ─── Bundled Image Handling ──────────────────────────────────────────────────

/// A container image bundled in the toolbox that must be pushed to the target registry
struct BundledPush {
    /// The toolbox image name, used for logging
    label: String,
    /// The path to the image's `.tar.gz` archive on disk
    tarball: PathBuf,
    /// The original image url to retag from (after the container load)
    source: String,
    /// The new registry url to tag and push to
    target: String,
}

/// Extract the tag from an image url, defaulting to "latest"
///
/// The tag is the substring after the last `:` that follows the last `/`, so registry
/// ports (e.g. `registry.local:5000/img`) are not mistaken for tags. A digest suffix
/// (`@sha256:...`) is not a usable push tag, so a digest-pinned reference falls back to
/// "latest" rather than treating the digest hex as the tag.
///
/// # Arguments
///
/// * `image_url` - The image url to extract a tag from
fn parse_tag(image_url: &str) -> &str {
    // isolate the final path segment so a registry port isn't read as a tag
    let last_segment = image_url.rsplit('/').next().unwrap_or(image_url);
    // drop any digest suffix; the repo portion before `@` is where a tag would live
    let without_digest = last_segment
        .split_once('@')
        .map_or(last_segment, |(repo, _digest)| repo);
    // the substring after the last `:` is the tag; a missing or empty tag (e.g. a
    // digest-only or bare reference) falls back to "latest" rather than guessing
    match without_digest.rsplit_once(':') {
        Some((_, tag)) if !tag.is_empty() => tag,
        _ => "latest",
    }
}

/// Determine the registry base path bundled images should be pushed under
///
/// Resolution order: the `--image-path-prefix` flag, then the prefix recorded in the
/// toolbox manifest, then an interactive prompt. Errors under `--skip-confirm` if unset.
///
/// # Arguments
///
/// * `cmd` - The import command (provides the `--image-path-prefix` flag)
/// * `manifest` - The toolbox manifest (provides a recorded fallback prefix)
/// * `can_prompt` - Whether the session can ask for the prefix (interactive + TTY)
/// * `progress` - The progress bar, suspended while prompting
fn resolve_image_path_prefix(
    cmd: &ImportToolbox,
    manifest: &ToolboxManifest,
    can_prompt: bool,
    progress: &Bar,
) -> Result<String, Error> {
    // an explicit flag wins
    if let Some(prefix) = cmd.image_path_prefix.as_deref()
        && !prefix.is_empty()
    {
        return Ok(prefix.to_string());
    }
    // otherwise fall back to the prefix recorded in the toolbox
    if let Some(prefix) = manifest.image_path_prefix.as_deref()
        && !prefix.is_empty()
    {
        return Ok(prefix.to_string());
    }
    // we can't ask for the prefix when running non-interactively (force,
    // skip-conflicts, or no TTY)
    if !can_prompt {
        return Err(Error::new(
            "This toolbox bundles container images; pass --image-path-prefix <registry-base> \
             to choose where they are pushed",
        ));
    }
    // prompt for a target registry base path
    progress.suspend(|| {
        dialoguer::Input::<String>::new()
            .with_prompt(
                "Target registry base path (e.g. registry.local/base) to push bundled images to",
            )
            .interact_text()
            .map_err(|e| Error::new(format!("Failed to read prefix input: {e}")))
    })
}

/// Prepare bundled images for import without performing any container work yet
///
/// Resolves the target registry prefix, computes each image's new url
/// (`<prefix>/<group>/<name>:<tag>`), rewrites the request to point there (so the
/// confirmation reflects the final location), and returns the work needed to push
/// each archive. The actual load/tag/push is deferred until after the user
/// confirms (see [`push_bundled_images`]).
///
/// # Arguments
///
/// * `cmd` - The import command
/// * `manifest` - The toolbox manifest
/// * `images` - The categorized images, whose `request.image` urls are rewritten in place
/// * `renames` - New manifest key to original key for collision-renamed images, used
///   to resolve a tarball saved on disk under the original name
/// * `can_prompt` - Whether the session can ask for a missing registry prefix
/// * `progress` - The progress bar
fn prepare_bundled_images(
    cmd: &ImportToolbox,
    manifest: &ToolboxManifest,
    images: &mut [categorize::CategorizedImage],
    renames: &HashMap<String, String>,
    can_prompt: bool,
    progress: &Bar,
) -> Result<Vec<BundledPush>, Error> {
    // bundled toolboxes carry tarballs on disk, so the manifest must be a local path
    let base_dir = match &cmd.manifest {
        ManifestLocation::Path(path) => path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        ManifestLocation::Url(_) => {
            return Err(Error::new(
                "This toolbox bundles container images and must be imported from a local path, not \
                 a URL; download the toolbox directory (including its images/ tarballs) and import \
                 the local toolbox.json",
            ));
        }
    };
    let prefix = resolve_image_path_prefix(cmd, manifest, can_prompt, progress)?;
    let prefix = prefix.trim_end_matches('/').to_string();
    let mut pushes = Vec::new();
    for img in images.iter_mut() {
        // an image with no container url has nothing to bundle/retag; warn so it isn't a
        // silent gap (the image is still created pointing at whatever url it carries)
        let source = match img.request.image.as_deref() {
            Some(url) if !url.is_empty() => url.to_string(),
            _ => {
                progress.warning(format!(
                    "Bundled toolbox image '{}' has no container url; nothing to push for it",
                    img.name
                ));
                continue;
            }
        };
        let tag = parse_tag(&source);
        let target = format!("{prefix}/{}/{}:{tag}", img.request.group, img.request.name);
        // a collision rename re-keys the image but leaves its tarball saved under the original
        // on-disk name, so the archive file is named for that original key. `img.name` is the
        // (possibly renamed) manifest key; `renames` maps it back to the original key.
        let source_name = renames
            .get(&img.name)
            .map_or(img.name.as_str(), String::as_str);
        // the tool's directory is recorded per-image in toolbox.json (built for all images), so the
        // tarball is found wherever export placed it (configured layout or a `=dir`). A collision
        // rename moves the whole version entry (its `dir` included) under the new key in
        // `manifest.images` (see `rename_image_member`), so the dir is looked up by the current key
        // `img.name`. The on-disk tarball file, however, is still named for the original key, which is
        // why `source_name` (above) is recovered from `renames`. An empty dir means an older toolbox
        // that predates the field — fall back to the historical `images/<name>` layout.
        let dir = manifest
            .images
            .get(&img.name)
            .and_then(|image_manifest| image_manifest.versions.get(&img.version))
            .map(|version| version.dir.as_str())
            .unwrap_or("");
        let tarball = if dir.is_empty() {
            base_dir
                .join("images")
                .join(source_name)
                .join(format!("{source_name}.tar.gz"))
        } else {
            base_dir.join(dir).join(format!("{source_name}.tar.gz"))
        };
        pushes.push(BundledPush {
            label: img.name.clone(),
            tarball,
            source,
            target: target.clone(),
        });
        // point the imported image at its new registry location
        img.request.image = Some(target);
    }
    Ok(pushes)
}

/// Load, retag, and push each bundled image to the target registry, best-effort
///
/// Each image is independent: a stat/load/tag/push failure is warned and collected, but the
/// remaining bundled images — and the resource creation that follows — still proceed. The Thorium
/// image is created regardless, so a push failure just means that image won't run until its
/// container is pushed to the target registry.
///
/// Returns the labels of bundled images whose container could not be pushed (for the run's
/// failure summary).
///
/// # Arguments
///
/// * `pushes` - The bundled images to push, as prepared by [`prepare_bundled_images`]
/// * `progress` - The progress bar to update as images are pushed
async fn push_bundled_images(pushes: &[BundledPush], progress: &Bar) -> Vec<String> {
    progress.refresh(
        "Pushing bundled images",
        BarKind::Bound(pushes.len() as u64),
    );
    let mut failures = Vec::new();
    for push in pushes {
        // attempt this image's full load -> retag -> push independently so one failure doesn't
        // abort the rest
        let outcome: Result<(), Error> = async {
            // make sure the archive exists before trying to load it; a stat error is surfaced
            // rather than masked as "missing"
            let exists = tokio::fs::try_exists(&push.tarball).await.map_err(|e| {
                Error::new(format!("failed to stat '{}': {e}", push.tarball.display()))
            })?;
            if !exists {
                return Err(Error::new(format!(
                    "bundled image archive not found at '{}' (was the toolbox exported with \
                     --with-images and copied whole?)",
                    push.tarball.display()
                )));
            }
            container::load(&push.tarball, progress).await?;
            // retag from the loaded archive's tag (`source`) to the target; surface the expected
            // source tag so a save/load naming mismatch is debuggable
            container::tag(&push.source, &push.target, progress)
                .await
                .map_err(|e| {
                    Error::new(format!(
                        "tagging '{}' -> '{}' failed (does the loaded archive contain '{}'?): {e}",
                        push.source, push.target, push.source
                    ))
                })?;
            container::push(&push.target, progress).await?;
            Ok(())
        }
        .await;
        if let Err(err) = outcome {
            progress.warning(format!(
                "Bundled image '{}' was not pushed: {err}; its Thorium image is still created but \
                 won't run until its container is pushed to '{}'",
                push.label, push.target
            ));
            failures.push(format!("{} (bundled image push)", push.label));
        }
        progress.inc(1);
    }
    failures
}

// ─── Apply Phase ─────────────────────────────────────────────────────────────

/// Apply a categorized toolbox import: create groups/policies, push bundled
/// images, create new resources, then resolve existing ones per the conflict mode
///
/// Returns how the apply phase ended without short-circuiting on a Quit, so the
/// caller can settle the journal (offer/apply rollback) afterwards. A Quit in the
/// image merge stops the pipeline pass too, so the rollback offer covers both.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (used for the default editor in interactive merges)
/// * `editor` - The editor override for interactive merges
/// * `mode` - How existing resources are handled
/// * `can_prompt` - Whether the session can drive an interactive merge
/// * `workers` - Max concurrent API actions in the apply phase (the global `--workers`)
/// * `plan` - The categorized resources and missing groups to apply
/// * `policy_plan` - The categorized network policies to create/warn about
/// * `bundled_pushes` - Bundled container images to push before creating resources
/// * `progress` - The progress bar
/// * `journal` - The journal recording applied changes for rollback
#[allow(clippy::too_many_arguments)]
async fn apply_resources(
    thorium: &Thorium,
    conf: &CtlConf,
    editor: Option<&str>,
    mode: ConflictMode,
    can_prompt: bool,
    workers: usize,
    plan: ImportPlan<'_>,
    policy_plan: &policies::PolicyPlan,
    bundled_pushes: &[BundledPush],
    progress: &Bar,
    journal: &Journal,
) -> Result<ApplyOutcome, Error> {
    // create any missing groups
    if !plan.missing_groups.is_empty() {
        progress.refresh(
            "Creating groups",
            BarKind::Bound(plan.missing_groups.len() as u64),
        );
        imports::create_groups(
            thorium,
            plan.missing_groups.clone(),
            workers,
            progress,
            journal,
        )
        .await?;
    }
    // create missing network policies before the images that reference them
    policies::create_policies(thorium, &policy_plan.new, progress, journal).await?;
    // apply planned policy updates before those images too (empty unless
    // --update-network-policy is set)
    policies::update_policies(thorium, &policy_plan.updates, progress, journal).await?;
    // existing-but-different policies left in place are surfaced (empty when updating)
    policies::warn_mismatched(policy_plan, progress);
    // push any bundled container images to the target registry before creating the resources
    // that reference them; best-effort, so a failed push warns and is collected rather than
    // aborting the rest of the import
    let mut failures: Vec<String> = if bundled_pushes.is_empty() {
        Vec::new()
    } else {
        push_bundled_images(bundled_pushes, progress).await
    };
    // import new resources, collecting per-resource failures so one bad image/pipeline
    // doesn't abort the rest (a pipeline whose image failed will fail too, and is
    // collected the same way)
    failures.extend(
        create::import_new_images(thorium, plan.new_images, workers, progress, journal).await,
    );
    failures.extend(
        create::import_new_pipelines(thorium, plan.new_pipelines, workers, progress, journal).await,
    );
    // handle existing resources via the shared dispatch; a Quit in the image pass
    // stops the pipeline pass too so the rollback offer covers everything. Note the asymmetry:
    // this pass is fail-fast — an interactive merge apply error propagates (via `?`) to settle
    // the journal and offer rollback — whereas the create passes above collect per-resource
    // failures and keep going.
    let images_applied = imports::apply_existing::<ImageKind>(
        thorium,
        conf,
        plan.existing_images,
        mode,
        editor,
        can_prompt,
        workers,
        progress,
        journal,
    )
    .await?;
    failures.extend(images_applied.failures);
    if images_applied.outcome == ImportOutcome::Quit {
        return Ok(ApplyOutcome {
            outcome: ImportOutcome::Quit,
            failures,
        });
    }
    let pipelines_applied = imports::apply_existing::<PipelineKind>(
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
    failures.extend(pipelines_applied.failures);
    Ok(ApplyOutcome {
        outcome: pipelines_applied.outcome,
        failures,
    })
}

// ─── Main Import Entry Point ─────────────────────────────────────────────────

/// Import a toolbox into Thorium by the given manifest file.
///
/// When images or pipelines already exist, the user is prompted interactively
/// to Edit (merge editor), Skip, Apply (accept incoming), or Quit for each
/// changed resource. `--overwrite` skips the editor and auto-applies all changes;
/// `--skip-conflicts` creates only new resources and leaves differing existing
/// ones untouched with a warning.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `cmd` - The toolbox import command that was run
/// * `workers` - Max concurrent API actions in the apply phase (the global `--workers`)
pub async fn import(
    thorium: Thorium,
    conf: CtlConf,
    cmd: &ImportToolbox,
    workers: usize,
) -> Result<(), Error> {
    // get the manifest from the location along with a progress bar
    let (mut manifest, progress) = shared::get_manifest(&cmd.manifest).await?;
    // resolve any URL-based configs before validation
    shared::resolve_manifest_configs(&mut manifest, &progress).await?;
    // 1) structural validation, BEFORE any override: drop intrinsically-invalid
    //    image/pipeline versions (unresolved configs, references to images absent
    //    from the manifest), warning about each
    shared::warn_dropped(&manifest.validate_structural(), &progress);
    // 2) snapshot each entry's pre-override group so collision resolution can
    //    disambiguate which pipeline wanted which image variant
    let source_groups = manifest.capture_source_groups();
    // 3) apply the group override
    if let Some(group_override) = &cmd.group_override {
        progress.info_anonymous(format!(
            "Forcing all images and pipelines into group '{}'",
            group_override.bright_yellow()
        ));
        manifest = manifest.override_group(group_override);
    }
    // 4) group-coherence validation, AFTER the override: drop pipelines whose order
    //    references images not present in their (final) group
    shared::warn_dropped(&manifest.validate_group_coherence(), &progress);
    // resolve how existing resources will be handled up front so the
    // confirmation screen can describe it accurately
    let mode = ConflictMode::from_flags(cmd.overwrite, cmd.skip_conflicts);
    // a session can prompt only in the interactive default (force/skip-conflicts are
    // explicit non-interactive opt-outs) and only with a real terminal. This gates
    // collision prompts, the merge editor, the plan confirmation, and rollback.
    let can_prompt =
        mode == ConflictMode::Interactive && std::io::IsTerminal::is_terminal(&std::io::stdin());
    // 5) detect & resolve (group, name) collisions the override (or a hand-authored
    //    manifest) introduced: rename + repoint interactively, else skip + warn,
    //    so the import never silently overwrites one resource with another
    let image_renames =
        collisions::resolve_collisions(&mut manifest, &source_groups, can_prompt, &progress)?;
    // 6) re-check coherence after any renames repointed pipelines
    shared::warn_dropped(&manifest.validate_group_coherence(), &progress);
    // get all the groups the (surviving) manifest expects to exist; computing this
    // after dropping/resolution avoids creating groups only needed by skipped resources
    let manifest_groups = if let Some(group_override) = &cmd.group_override {
        HashSet::from([group_override.to_string()])
    } else {
        manifest.groups()
    };
    // categorize images and pipelines by checking what already exists in Thorium
    let mut images =
        categorize::categorize_images(&thorium, flatten_manifest_images(&manifest), &progress)
            .await?;
    let pipelines = categorize::categorize_pipelines(
        &thorium,
        flatten_manifest_pipelines(&manifest),
        &progress,
    )
    .await?;
    // if this toolbox bundles container images, resolve where they'll be pushed and rewrite
    // each image url now so the confirmation reflects the final location; the actual
    // load/tag/push is deferred until after the user confirms
    let bundled_pushes = if manifest.bundled_images {
        prepare_bundled_images(
            cmd,
            &manifest,
            &mut images,
            &image_renames,
            can_prompt,
            &progress,
        )?
    } else {
        // --image-path-prefix only affects bundled toolboxes; warn so it isn't a silent no-op
        if cmd.image_path_prefix.is_some() {
            progress.warning(
                "--image-path-prefix has no effect: this toolbox does not bundle container images",
            );
        }
        Vec::new()
    };
    // collect the bundled network policies and check them against the target
    let policy_plan = policies::categorize_policies(
        &thorium,
        policies::collect_policies(&manifest, cmd.group_override.as_deref(), &progress),
        cmd.update_network_policy,
        &progress,
    )
    .await?;
    // check which groups are missing
    let missing_groups = imports::get_missing_groups(&thorium, manifest_groups.clone())
        .await
        .map_err(|err| Error::new(format!("Error retrieving missing groups: {err}")))?;
    // partition into new vs existing for the confirmation summary
    let plan = ImportPlan::new(&images, &pipelines, missing_groups);
    // confirm before an interactive merge OR before creating anything sensitive:
    // network policies are cluster security state and groups are access boundaries,
    // so neither should appear silently even in an otherwise clean import. This only
    // fires when `can_prompt` holds (interactive mode + TTY); force, skip-conflicts,
    // and non-TTY sessions never prompt.
    if can_prompt
        && (plan.has_conflicts()
            || !policy_plan.new.is_empty()
            || !policy_plan.updates.is_empty()
            || !policy_plan.mismatched.is_empty()
            || !plan.missing_groups.is_empty())
    {
        // the username is cosmetic (it's only shown in the prompt), so a lookup failure must not
        // abort the import — fall back to a placeholder and warn instead
        let username = match thorium.users.info().await {
            Ok(user) => user.username,
            Err(err) => {
                progress.warning(format!(
                    "Could not look up the current user ({err}); continuing"
                ));
                "<current user>".to_string()
            }
        };
        let confirmed = progress.suspend(|| {
            policies::print_plan(&policy_plan);
            imports::confirm_import(&conf, &plan, &username, mode)
        })?;
        if !confirmed {
            return Ok(());
        }
    }
    // journal every applied change so a partial import can be rolled back
    let journal = Journal::new();
    // run the apply phase, capturing how it ended without short-circuiting so
    // the journal can be settled (rollback offered/applied) afterwards
    let result = apply_resources(
        &thorium,
        &conf,
        cmd.editor.as_deref(),
        mode,
        can_prompt,
        workers,
        plan,
        &policy_plan,
        &bundled_pushes,
        &progress,
        &journal,
    )
    .await;
    // per-resource failures are kept (not rolled back) and reported below; settle the
    // journal only on the outcome/error so rollback still covers a Quit or fatal error
    let (settle_input, failures) = match result {
        Ok(applied) => (Ok(applied.outcome), applied.failures),
        Err(err) => (Err(err), Vec::new()),
    };
    // only an interactive session can be asked about rollback; without a terminal
    // (CI, pipes) or in a non-interactive mode, nobody can answer, so settle_journal
    // falls back to --rollback-on-failure. The same `can_prompt` that gated the
    // confirmation is reused here so the rollback offer matches the session.
    let outcome = imports::settle_journal(
        &thorium,
        &progress,
        journal,
        settle_input,
        can_prompt,
        cmd.rollback_on_failure,
    )
    .await?;
    // pick the terminal banner from both the outcome AND whether any resource failed, so the
    // final line never reads "Import complete!" right next to the failure list and non-zero
    // exit below (mirrors remove.rs's "Removal finished with errors")
    match outcome {
        ImportOutcome::Completed if failures.is_empty() => {
            progress.refresh("Import complete!", BarKind::Timer);
        }
        ImportOutcome::Completed => {
            progress.refresh("Import finished with errors", BarKind::Timer);
        }
        ImportOutcome::Quit => progress.refresh("Import stopped early", BarKind::Timer),
    }
    progress.finish();
    // surface every resource that failed to import and exit non-zero, after keeping the
    // resources that succeeded
    if !failures.is_empty() {
        return Err(Error::new(format!(
            "{} resource(s) failed to import: {}",
            failures.len(),
            failures.join(", ")
        )));
    }
    Ok(())
}

/// Unit tests for tag parsing, the one piece of bundled-image url handling that
/// is pure and testable without a registry or container runtime
#[cfg(test)]
mod tests {
    use super::*;
    /// A plain `name:tag` reference returns its tag
    #[test]
    fn parse_tag_reads_explicit_tag() {
        assert_eq!(parse_tag("registry.local/group/img:v1.2"), "v1.2");
    }
    /// A reference with no tag defaults to "latest"
    #[test]
    fn parse_tag_defaults_to_latest() {
        assert_eq!(parse_tag("registry.local/group/img"), "latest");
    }
    /// A registry port in the host is not mistaken for the tag
    #[test]
    fn parse_tag_ignores_registry_port() {
        assert_eq!(parse_tag("registry.local:5000/img"), "latest");
        assert_eq!(parse_tag("registry.local:5000/img:v2"), "v2");
    }
    /// A digest-pinned reference has no usable tag, so it falls back to "latest"
    /// rather than treating the digest hex as the tag
    #[test]
    fn parse_tag_digest_falls_back_to_latest() {
        assert_eq!(
            parse_tag("registry.local/img@sha256:abc123def456"),
            "latest"
        );
        assert_eq!(parse_tag("registry.local/img@sha256"), "latest");
    }
    /// A reference carrying both a tag and a digest keeps the real tag
    #[test]
    fn parse_tag_tag_and_digest_keeps_tag() {
        assert_eq!(parse_tag("registry.local/img:v1.2@sha256:abc123"), "v1.2");
    }
}
