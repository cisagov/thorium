//! Export Thorium images and pipelines into a toolbox directory structure

use colored::Colorize;
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, HashSet};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use thorium::models::{
    Image, ImageRequest, ImageVersion, NetworkPolicyRequest, Pipeline, PipelineRequest,
};
use thorium::{CtlConf, Error, Thorium};

use super::init::{generate_image_manifest, generate_pipeline_manifest, render_config_toml};
use super::manifest::{self, ToolboxManifest};
use super::{build, collisions, policies, shared};
use crate::args::Args;
use crate::args::toolbox::{BuildToolbox, ExportToolbox, ResourceSpec};
use crate::handlers::container;
use crate::handlers::exports::{DiskConflictResolver, WriteOutcome};
use crate::handlers::imports::editor::{resolve_editor, review_config_in_editor};
use crate::handlers::progress::{Bar, BarKind};
use crate::utils::images::list_all_images;
use crate::utils::pipelines::list_all_pipelines;

/// Render an image's version as a toolbox version label, defaulting to "latest"
///
/// # Arguments
///
/// * `version` - The image version to render, or `None` for the default label
fn version_label(version: &Option<ImageVersion>) -> String {
    match version {
        Some(ImageVersion::SemVer(v)) => v.to_string(),
        Some(ImageVersion::Custom(s)) => s.clone(),
        None => "latest".to_string(),
    }
}

// ─── Resource Resolution ─────────────────────────────────────────────────────

/// Resolves the images and pipelines an export run should write
///
/// Supports a full-group export (every image and pipeline in a group) and a targeted
/// export of named pipelines/images, deduplicating so a pipeline-referenced image and
/// a standalone `--images` selection never fetch the same image twice.
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to fetch resources
/// * `cmd` - The export args (group and/or named pipelines/images)
/// * `workers` - The number of concurrent fetches to run
async fn resolve_resources(
    thorium: &Thorium,
    cmd: &ExportToolbox,
    workers: usize,
) -> Result<(Vec<Image>, Vec<Pipeline>), Error> {
    // buffer_unordered with 0 never polls anything, so clamp to at least one worker
    let workers = workers.max(1);
    let mut images: Vec<Image> = Vec::new();
    let mut pipelines: Vec<Pipeline> = Vec::new();
    // tracks which image identities we've already queued so a pipeline-referenced
    // image and a standalone --image (or two pipelines) don't fetch it twice
    let mut seen_images: HashSet<(String, String)> = HashSet::new();
    // Full group export: list images and pipelines concurrently
    if let Some(group) = &cmd.group
        && cmd.pipelines.is_empty()
        && cmd.images.is_empty()
    {
        let (group_images, group_pipelines) = futures::try_join!(
            list_all_images(thorium, group),
            list_all_pipelines(thorium, group)
        )?;
        // seed the dedup set with every group image so a later --pipeline that references
        // one of them doesn't re-fetch it
        for img in group_images {
            seen_images.insert((img.group.clone(), img.name.clone()));
            images.push(img);
        }
        pipelines.extend(group_pipelines);
    }
    // Specific pipelines: fetch concurrently, then auto-resolve referenced images
    if !cmd.pipelines.is_empty() {
        // parse each "group/name" (falling back to --group) up front so a malformed spec
        // fails before any network calls are issued
        let specs = cmd
            .pipelines
            .iter()
            .map(|s| ResourceSpec::parse(s, cmd.group.as_deref()).map_err(Error::new))
            .collect::<Result<Vec<_>, _>>()?;
        // fetch the named pipelines bounded-parallel; each result keeps the error context
        // (group:name) so a failure points at the offending spec
        let fetched: Vec<Result<Pipeline, Error>> = stream::iter(specs)
            .map(|spec| async move {
                thorium
                    .pipelines
                    .get(&spec.group, &spec.name)
                    .await
                    .map_err(|e| {
                        Error::new(format!(
                            "Failed to get pipeline '{}:{}': {e}",
                            spec.group, spec.name
                        ))
                    })
            })
            .buffer_unordered(workers)
            .collect()
            .await;
        // walk every fetched pipeline's order and record each referenced image identity
        // exactly once, scoping the lookup to the pipeline's own group
        let mut referenced: Vec<(String, String)> = Vec::new();
        for pipeline in fetched {
            // propagate the first fetch error here rather than during the concurrent stream
            // so partial successes are still surfaced as a single failure
            let pipeline = pipeline?;
            for image_name in pipeline.order.iter().flatten() {
                let key = (pipeline.group.clone(), image_name.clone());
                // insert returns false when the identity was already queued, skipping the
                // duplicate fetch
                if seen_images.insert(key.clone()) {
                    referenced.push(key);
                }
            }
            pipelines.push(pipeline);
        }
        // pull in every image a pipeline depends on so the exported toolbox is self-contained
        let fetched_images: Vec<Result<Image, Error>> = stream::iter(referenced)
            .map(|(group, name)| async move {
                thorium.images.get(&group, &name).await.map_err(|e| {
                    Error::new(format!(
                        "Failed to get image '{group}:{name}' (referenced by a pipeline): {e}"
                    ))
                })
            })
            .buffer_unordered(workers)
            .collect()
            .await;
        // surface any referenced-image fetch failure as a hard error: a pipeline can't be
        // exported usefully without the images it runs
        for image in fetched_images {
            images.push(image?);
        }
    }
    // Specific standalone images: dedup against everything already queued, then fetch concurrently
    let standalone = cmd
        .images
        .iter()
        .map(|s| ResourceSpec::parse(s, cmd.group.as_deref()).map_err(Error::new))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        // keep only identities not already queued by a group export or a pipeline reference
        .filter_map(|spec| {
            let key = (spec.group, spec.name);
            seen_images.insert(key.clone()).then_some(key)
        })
        .collect::<Vec<_>>();
    let fetched_standalone: Vec<Result<Image, Error>> = stream::iter(standalone)
        .map(|(group, name)| async move {
            thorium
                .images
                .get(&group, &name)
                .await
                .map_err(|e| Error::new(format!("Failed to get image '{group}:{name}': {e}")))
        })
        .buffer_unordered(workers)
        .collect()
        .await;
    // a standalone image the user explicitly named that can't be fetched is fatal
    for image in fetched_standalone {
        images.push(image?);
    }
    // nothing was selected by any of the three paths: fail loudly rather than write an empty toolbox
    if images.is_empty() && pipelines.is_empty() {
        return Err(Error::new(
            "No resources to export. Specify --group, --pipelines, or --images.",
        ));
    }
    Ok((images, pipelines))
}

// ─── Toolbox-wide settings ───────────────────────────────────────────────────

/// The toolbox-wide settings written into the exported `config.toml`
///
/// Sourced either from an existing `config.toml` (`--config`) or the
/// `--name`/`--registry` flags.
struct ToolboxSettings {
    /// Human-readable toolbox name
    name: String,
    /// Primary container registry, unset when the toolbox declares no central one
    registry: Option<String>,
    /// Extra registries to additionally tag images for
    registries: Vec<String>,
    /// Default registry base path bundled images push under on import
    image_path_prefix: Option<String>,
    /// Directory (relative to the output root) image tool dirs are written under; `None` = `images`
    export_image_path: Option<String>,
    /// Directory (relative to the output root) pipeline tool dirs are written under; `None` = `pipelines`
    export_pipeline_path: Option<String>,
    /// Whether the toolbox bundles image tarballs (driven by `--with-images`)
    bundled_images: bool,
    /// The toolbox-wide default base-image config, preserved from a reused `--config`
    base_image: Option<build::BaseImage>,
}

/// Lexically normalize a path by folding `.` and `..` components without touching the filesystem
///
/// Used to compare a placement destination against the toolbox root when neither directory need
/// exist yet, so `..` can't be resolved by `canonicalize`. A `..` cancels a preceding normal
/// component; one at a relative root is kept (it still escapes), and one just past an absolute root
/// is dropped (it can't go above root). This is purely lexical — it does not follow symlinks.
///
/// # Arguments
///
/// * `path` - The path to normalize
fn lexical_normalize(path: &Path) -> PathBuf {
    // fold components onto a stack so a trailing `..` can pop the previous normal segment
    let mut stack: Vec<std::path::Component> = Vec::new();
    for comp in path.components() {
        match comp {
            // a bare `.` contributes nothing to the resolved path
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if let Some(std::path::Component::Normal(_)) = stack.last() {
                    // cancel the preceding real directory
                    stack.pop();
                } else if !matches!(
                    stack.last(),
                    Some(std::path::Component::RootDir | std::path::Component::Prefix(_))
                ) {
                    // keep a `..` that has no normal component to cancel (a relative-root escape);
                    // drop one sitting right on an absolute root, which can't go higher
                    stack.push(comp);
                }
            }
            // root, prefix, and normal components carry through verbatim
            other => stack.push(other),
        }
    }
    // reassemble the folded components into a path
    let mut out = PathBuf::new();
    for comp in stack {
        out.push(comp.as_os_str());
    }
    out
}

/// Resolve a per-resource `=dest` placement to a directory relative to the toolbox root
///
/// A relative `dest` is interpreted against the toolbox root; an absolute one is taken as given. Both
/// it and the root are made absolute (against the current directory) and lexically normalized, then
/// the dest is re-expressed relative to the root. The placement MUST land inside the toolbox — `build`
/// crawls the output tree, so files written outside it would never be discovered — so a dest that
/// resolves outside (or onto the root itself) is a hard error with an actionable message.
///
/// # Arguments
///
/// * `output` - The resolved toolbox output directory (the toolbox root)
/// * `dest` - The raw `=dest` string from the selection
fn resolve_dest_within(output: &Path, dest: &str) -> Result<String, Error> {
    // both paths are made absolute against the working directory so a relative output and a relative
    // dest are compared on equal footing
    let cwd = std::env::current_dir()
        .map_err(|e| Error::new(format!("Failed to read the current directory: {e}")))?;
    let make_abs = |path: &Path| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            cwd.join(path)
        }
    };
    let output_abs = lexical_normalize(&make_abs(output));
    let dest_path = Path::new(dest);
    // a relative dest is rooted at the toolbox; an absolute dest is taken literally
    let dest_abs = if dest_path.is_absolute() {
        lexical_normalize(dest_path)
    } else {
        lexical_normalize(&output_abs.join(dest_path))
    };
    // re-express the dest relative to the toolbox root; anything that won't strip (or strips to
    // empty, i.e. the root itself) is outside the toolbox and can't be a valid placement
    match dest_abs.strip_prefix(&output_abs) {
        Ok(rel) if !rel.as_os_str().is_empty() => Ok(rel.to_string_lossy().into_owned()),
        _ => Err(Error::new(format!(
            "destination '{dest}' resolves outside the toolbox root '{}'; a placement path must \
             point to a subdirectory inside the toolbox (build only includes files under it)",
            output.display()
        ))),
    }
}

/// Resolve the per-resource `=dest` placements from the `--images`/`--pipelines` selections
///
/// Returns a map of resource name to its destination directory **relative to the toolbox root** (the
/// stored/reconciliation form), with any absolute or `..`-bearing input resolved against the root by
/// [`resolve_dest_within`]. Keyed by name alone (the on-disk tool-directory leaf), not `(group, name)`:
/// in the rare case two selected resources share a name across groups and both carry `=dest`, the
/// last one parsed wins. A resource without an explicit `=dest` is absent and falls back to the
/// configured/default layout. Whole-group exports and auto-pulled dependency images aren't named here,
/// so they are never overridden. A dest that resolves outside the toolbox is a hard error.
///
/// # Arguments
///
/// * `cmd` - The export command
/// * `output` - The resolved toolbox output directory
fn resolve_dest_overrides(
    cmd: &ExportToolbox,
    output: &Path,
) -> Result<HashMap<String, String>, Error> {
    let mut overrides = HashMap::new();
    for spec in cmd.images.iter().chain(cmd.pipelines.iter()) {
        // resolve_resources already parsed and validated every spec, so a parse failure here is
        // unexpected; surface it rather than silently dropping the placement
        let parsed = ResourceSpec::parse(spec, cmd.group.as_deref()).map_err(Error::new)?;
        if let Some(dest) = parsed.dest {
            overrides.insert(parsed.name, resolve_dest_within(output, &dest)?);
        }
    }
    Ok(overrides)
}

/// Load the existing toolbox manifest at the output for append reconciliation
///
/// Prefers the committed `<output>/toolbox.json`; if that's missing or unparsable, falls back to
/// crawling the on-disk tool manifests (when the directory is a toolbox, i.e. has a `config.toml`)
/// so a deleted or stale `toolbox.json` doesn't make an append re-write resources already present.
/// Returns `None` for a fresh export (neither source available); every failure is best-effort
/// (`build`'s own duplicate-manifest check still guards).
///
/// # Arguments
///
/// * `output` - The resolved toolbox output directory
/// * `progress` - The progress bar, for the "found existing toolbox" notice
async fn load_existing_manifest(output: &Path, progress: &Bar) -> Option<ToolboxManifest> {
    let json_path = output.join("toolbox.json");
    // prefer the committed toolbox.json when it reads and parses
    if let Ok(bytes) = tokio::fs::read(&json_path).await
        && let Ok(existing) = serde_json::from_slice::<ToolboxManifest>(&bytes)
    {
        progress.info_anonymous(format!(
            "Found existing toolbox.json at '{}'; reconciling against it",
            json_path.display()
        ));
        return Some(existing);
    }
    // no usable toolbox.json: only crawl when the dir is actually a toolbox (has a config.toml),
    // otherwise there's nothing meaningful to reconcile against
    let config_path = output.join("config.toml");
    if !config_path.exists() {
        return None;
    }
    // crawl the on-disk tool manifests into the same shape as toolbox.json. build walks with
    // synchronous std::fs, so run it off the async runtime; any crawl/parse error means no index.
    // This is an index-only crawl: `use_image_path`/`tag_suffix` only affect derived tags (irrelevant
    // to the reconcile identity) and `output: None` because nothing is written — only `path` (the
    // crawl root) matters here.
    let build_cmd = BuildToolbox {
        config: config_path,
        use_image_path: false,
        output: None,
        path: Some(output.to_path_buf()),
        tag_suffix: None,
    };
    let value = tokio::task::spawn_blocking(move || build::build_in_memory(&build_cmd))
        .await
        .ok()?
        .ok()?;
    let existing = serde_json::from_value::<ToolboxManifest>(value).ok()?;
    progress.info_anonymous(format!(
        "No toolbox.json at '{}'; reconciling against the on-disk tool manifests",
        output.display()
    ));
    Some(existing)
}

/// Index an already-loaded toolbox manifest for append reconciliation
///
/// Returns `(images, pipelines)` maps keyed by `(group, name, version)`, each value the resource's
/// `(canonical-config JSON, on-disk dir)`. The canonical JSON drives the unchanged/differs comparison
/// and the recorded dir lets the write loops update a resource in place where it already lives. Empty
/// when `existing` is `None` (a fresh export with no toolbox to reconcile against).
///
/// # Arguments
///
/// * `existing` - The loaded existing toolbox manifest, or `None` for a fresh export
#[allow(clippy::type_complexity)]
fn index_existing(
    existing: Option<&ToolboxManifest>,
) -> (
    HashMap<(String, String, String), (String, String)>,
    HashMap<(String, String, String), (String, String)>,
) {
    let mut images = HashMap::new();
    let mut pipelines = HashMap::new();
    // no existing toolbox → empty indexes (fresh export)
    let Some(existing) = existing else {
        return (images, pipelines);
    };
    // index each embedded image config by identity, keeping its canonical JSON (for the unchanged
    // comparison) and recorded dir (to update it in place / catch a cross-directory duplicate)
    for image in existing.images.values() {
        for (version, entry) in &image.versions {
            if let Some(config) = &entry.config
                && let Ok(json) = crate::utils::canonical_json(config)
            {
                images.insert(
                    (config.group.clone(), config.name.clone(), version.clone()),
                    (json, entry.dir.clone()),
                );
            }
        }
    }
    // pipelines now record their dir too, so they reconcile the same way as images
    for pipeline in existing.pipelines.values() {
        for (version, entry) in &pipeline.versions {
            if let Some(config) = &entry.config
                && let Ok(json) = crate::utils::canonical_json(config)
            {
                pipelines.insert(
                    (config.group.clone(), config.name.clone(), version.clone()),
                    (json, entry.dir.clone()),
                );
            }
        }
    }
    (images, pipelines)
}

/// Collect the unique `(group, name)` identities of every image and pipeline in a toolbox manifest
///
/// Used by the no-selection "refresh all" export to know which tools to re-fetch from Thorium.
/// Deduplicates across version entries (a tool present at multiple versions is fetched once) and reads
/// the identity from each entry's embedded config.
///
/// # Arguments
///
/// * `existing` - The loaded existing toolbox manifest
#[allow(clippy::type_complexity)]
fn existing_resource_ids(
    existing: &ToolboxManifest,
) -> (Vec<(String, String)>, Vec<(String, String)>) {
    // a closure that flattens a name→versions map into the unique (group, name) of its configs,
    // preserving first-seen order so the fetch list is deterministic
    let collect = |entries: &mut dyn Iterator<Item = (String, String)>| {
        let mut seen = HashSet::new();
        let mut ids = Vec::new();
        for id in entries {
            if seen.insert(id.clone()) {
                ids.push(id);
            }
        }
        ids
    };
    let images = collect(
        &mut existing
            .images
            .values()
            .flat_map(|image| image.versions.values())
            .filter_map(|entry| entry.config.as_ref())
            .map(|config| (config.group.clone(), config.name.clone())),
    );
    let pipelines = collect(
        &mut existing
            .pipelines
            .values()
            .flat_map(|pipeline| pipeline.versions.values())
            .filter_map(|entry| entry.config.as_ref())
            .map(|config| (config.group.clone(), config.name.clone())),
    );
    (images, pipelines)
}

/// Re-fetch every image and pipeline a toolbox already contains, for a no-selection refresh export
///
/// Enumerates the toolbox's `(group, name)` tools (see [`existing_resource_ids`]) and fetches each from
/// Thorium bounded by `workers`. A tool that no longer exists in Thorium (or otherwise fails to fetch)
/// is **warned and skipped** rather than aborting the run, since `export` only refreshes the tools
/// already present and never prunes ones that vanished upstream.
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to fetch resources
/// * `existing` - The loaded existing toolbox manifest to refresh
/// * `workers` - The number of concurrent fetches to run
/// * `progress` - The progress bar, for the refresh notice and per-tool skip warnings
async fn refresh_existing_resources(
    thorium: &Thorium,
    existing: &ToolboxManifest,
    workers: usize,
    progress: &Bar,
) -> Result<(Vec<Image>, Vec<Pipeline>), Error> {
    // buffer_unordered with 0 never polls anything, so clamp to at least one worker
    let workers = workers.max(1);
    let (image_ids, pipeline_ids) = existing_resource_ids(existing);
    progress.info_anonymous(format!(
        "Refreshing {} image(s) and {} pipeline(s) already in the toolbox",
        image_ids.len(),
        pipeline_ids.len()
    ));
    // fetch the images bounded-parallel; a tool that no longer exists upstream is warned and dropped
    let images: Vec<Image> = stream::iter(image_ids)
        .map(|(group, name)| async move {
            thorium
                .images
                .get(&group, &name)
                .await
                .map_err(|e| (group, name, e))
        })
        .buffer_unordered(workers)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|result| match result {
            Ok(image) => Some(image),
            Err((group, name, e)) => {
                progress.warning(format!(
                    "Image '{group}:{name}' is in the toolbox but could not be fetched from Thorium \
                     ({e}); leaving its existing files unchanged"
                ));
                None
            }
        })
        .collect();
    // same lenient fetch for pipelines
    let pipelines: Vec<Pipeline> = stream::iter(pipeline_ids)
        .map(|(group, name)| async move {
            thorium
                .pipelines
                .get(&group, &name)
                .await
                .map_err(|e| (group, name, e))
        })
        .buffer_unordered(workers)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|result| match result {
            Ok(pipeline) => Some(pipeline),
            Err((group, name, e)) => {
                progress.warning(format!(
                    "Pipeline '{group}:{name}' is in the toolbox but could not be fetched from \
                     Thorium ({e}); leaving its existing files unchanged"
                ));
                None
            }
        })
        .collect();
    Ok((images, pipelines))
}

/// Build a `(group, name) → dir` lookup from a reconciliation index keyed by `(group, name, version)`
///
/// Lets the write loops find where a tool already lives by name (regardless of version), so a
/// re-export updates it in place instead of writing a duplicate at the default layout. When a tool
/// has versions recorded at different directories (an unusual, near-malformed toolbox) an arbitrary
/// non-empty dir wins (the index is a `HashMap`, so iteration order isn't stable) — `build`'s
/// duplicate check still guards a true conflict.
///
/// # Arguments
///
/// * `index` - The reconciliation index (`(group, name, version) → (json, dir)`)
fn dirs_by_name(
    index: &HashMap<(String, String, String), (String, String)>,
) -> HashMap<(String, String), String> {
    let mut dirs = HashMap::new();
    for ((group, name, _version), (_json, dir)) in index {
        // skip empty dirs (older toolboxes predating the field); keep one real dir per tool
        if !dir.is_empty() {
            dirs.entry((group.clone(), name.clone()))
                .or_insert_with(|| dir.clone());
        }
    }
    dirs
}

/// Build a `name → set of groups` lookup from a reconciliation index keyed by `(group, name, version)`
///
/// Lets a write loop notice when an exported tool's *name* already exists in the toolbox under a
/// *different* group — the tell-tale of a group rename (e.g. a tool that is `static2/<name>` in Thorium
/// but `static1/<name>` in the toolbox) where the user forgot `--group-override`. The group set is
/// sorted so the warning lists them deterministically.
///
/// # Arguments
///
/// * `index` - The reconciliation index (`(group, name, version) → (json, dir)`)
fn groups_by_name(
    index: &HashMap<(String, String, String), (String, String)>,
) -> HashMap<String, std::collections::BTreeSet<String>> {
    let mut by_name: HashMap<String, std::collections::BTreeSet<String>> = HashMap::new();
    for (group, name, _version) in index.keys() {
        by_name
            .entry(name.clone())
            .or_default()
            .insert(group.clone());
    }
    by_name
}

/// Build a `(name, version) → (group, dir)` lookup from a reconciliation index keyed by
/// `(group, name, version)`
///
/// This mirrors `build`'s identity for a tool — `(name, version)`, **group-independent** — so a write
/// loop can refuse to lay down a second `manifest.toml` for a `(name, version)` that already lives in
/// the toolbox (under any group, at any directory). Without this, exporting a tool from a Thorium group
/// that doesn't match the toolbox's group writes a fresh copy that `build` then rejects as a duplicate.
/// A valid toolbox holds each `(name, version)` once, so the last writer wins on the off chance of a
/// pre-existing on-disk duplicate.
///
/// # Arguments
///
/// * `index` - The reconciliation index (`(group, name, version) → (json, dir)`)
fn locs_by_name_version(
    index: &HashMap<(String, String, String), (String, String)>,
) -> HashMap<(String, String), (String, String)> {
    let mut locs = HashMap::new();
    for ((group, name, version), (_json, dir)) in index {
        locs.insert(
            (name.clone(), version.clone()),
            (group.clone(), dir.clone()),
        );
    }
    locs
}

/// The reconciliation outcome for one resource being exported into a (possibly existing) toolbox
#[derive(Debug, PartialEq, Eq)]
enum Placement {
    /// Write a brand-new resource at this directory (a full write, manifest generated)
    New(String),
    /// The resource is already current (byte-identical) at this directory — files are a resolver
    /// no-op and a redundant container re-bundle is skipped
    Unchanged(String),
    /// The resource exists and differs; update it in place at this directory (`--overwrite`)
    Update(String),
    /// The resource exists and differs but `--overwrite` is not set — skip it with a warning
    SkipDiffers,
    /// An explicit `=dest` points somewhere other than where the resource already lives — skip it
    /// (relocating would leave a duplicate); carries the existing directory for the message
    SkipMove(String),
}

/// Decide where to write a resource and how to reconcile it against the existing toolbox
///
/// Target-directory precedence: an explicit `=dest` → the directory the tool already occupies (so a
/// re-export updates it in place) → the configured/default layout. A tool is "already in the toolbox"
/// when it has a recorded directory (`existing_dir`); an exact byte-identical config makes it
/// `Unchanged`, a differing one is `Update` (with `--overwrite`) or `SkipDiffers` (without). An
/// explicit `=dest` that names a different directory than the tool's home is a `SkipMove` (a second
/// copy would fail `build`'s duplicate check).
///
/// # Arguments
///
/// * `explicit_dest` - The resolved per-resource `=dest`, if one was given
/// * `existing_dir` - The directory this tool already occupies in the toolbox, if any
/// * `existing_exact_json` - The canonical config of the exact `(group, name, version)` already in the
///   toolbox, if present
/// * `current_json` - The canonical config being exported (for the unchanged comparison)
/// * `default_rel` - The configured/default layout directory for this resource
/// * `overwrite` - Whether `--overwrite` is set
fn plan_placement(
    explicit_dest: Option<&str>,
    existing_dir: Option<&str>,
    existing_exact_json: Option<&str>,
    current_json: &str,
    default_rel: &str,
    overwrite: bool,
) -> Placement {
    // target precedence: explicit =dest > the dir the tool already occupies > configured/default
    let target_rel = explicit_dest
        .or(existing_dir)
        .unwrap_or(default_rel)
        .to_string();
    // an explicit =dest naming a different dir than where the tool lives would create a second copy
    // (build rejects duplicate manifests), so refuse the move
    if let (Some(dest), Some(dir)) = (explicit_dest, existing_dir)
        && dest != dir
    {
        return Placement::SkipMove(dir.to_string());
    }
    // no recorded directory → the tool isn't in the toolbox yet → brand new
    if existing_dir.is_none() {
        return Placement::New(target_rel);
    }
    // present and byte-identical → already current
    if existing_exact_json == Some(current_json) {
        return Placement::Unchanged(target_rel);
    }
    // present but differing (a config change, or a different version into the tool's dir)
    if overwrite {
        Placement::Update(target_rel)
    } else {
        Placement::SkipDiffers
    }
}

/// The decided action for one resource in a write loop, with all messages pre-rendered
///
/// This is the pure decision shared by the image and pipeline write loops: it resolves placement
/// ([`plan_placement`]), folds in build's `(name, version)` identity (the group-mismatch re-group /
/// skip), and renders any warnings — so the loop only has to emit messages and do I/O.
#[derive(Debug, PartialEq, Eq)]
enum WriteAction {
    /// Skip this resource, emitting this warning (a duplicate/move/differ that can't be written)
    Skip(String),
    /// Write this resource
    Write {
        /// The tool directory (relative to the toolbox root) to write into
        target_rel: String,
        /// Whether to (re)generate `manifest.toml` — `true` only for a genuinely new tool
        full_write: bool,
        /// The group this tool is being re-grouped from, when a group-mismatched `(name, version)` is
        /// updated in place under `--overwrite`; drives the "Re-grouping" notice
        regrouped_from: Option<String>,
        /// Whether the incoming config is byte-identical to the existing one (a no-op write)
        unchanged: bool,
        /// An optional informational warning to emit before writing (a softer cross-group signal)
        soft_warn: Option<String>,
    },
}

/// Decide how to write one resource against the existing toolbox, rendering messages but doing no I/O
///
/// Mirrors `build`'s identity rule: a tool is `(name, version)` toolbox-wide, independent of group. So
/// a placement of `New` whose `(name, version)` already lives in the toolbox under a different
/// group/dir would duplicate at build time — with `--overwrite` it is re-grouped in place, otherwise
/// skipped. Pure and deterministic so it can be unit-tested without a client or filesystem.
///
/// # Arguments
///
/// * `kind` - "image" or "pipeline", for the rendered messages
/// * `group` / `name` / `version` - The incoming resource's identity (group already overridden)
/// * `current_json` - The incoming config's canonical JSON (for the unchanged comparison)
/// * `explicit_dest` - The resolved per-resource `=dir`, if any
/// * `existing_dir_by_name` - The dir this `(group, name)` already occupies in the toolbox, if any
/// * `existing_exact_json` - The canonical config of the exact `(group, name, version)` in the toolbox
/// * `existing_loc` - Where `(name, version)` lives toolbox-wide as `(group, dir)`, if anywhere
/// * `other_groups` - The groups this `name` already appears under in the toolbox
/// * `default_rel` - The configured/default layout directory for this resource
/// * `overwrite` - Whether `--overwrite` is set
#[allow(clippy::too_many_arguments)]
fn decide_write(
    kind: &str,
    group: &str,
    name: &str,
    version: &str,
    current_json: &str,
    explicit_dest: Option<&str>,
    existing_dir_by_name: Option<&str>,
    existing_exact_json: Option<&str>,
    existing_loc: Option<&(String, String)>,
    other_groups: Option<&std::collections::BTreeSet<String>>,
    default_rel: &str,
    overwrite: bool,
) -> WriteAction {
    let placement = plan_placement(
        explicit_dest,
        existing_dir_by_name,
        existing_exact_json,
        current_json,
        default_rel,
        overwrite,
    );
    // build-identity collision: a New write whose (name, version) already lives elsewhere (a different
    // group/dir than we're writing under) would be a build-breaking duplicate. With --overwrite,
    // re-group it in place; without, skip rather than corrupt the toolbox.
    if let Placement::New(_) = &placement
        && let Some((existing_group, existing_dir)) = existing_loc
        && !existing_dir.is_empty()
    {
        if overwrite {
            return WriteAction::Write {
                target_rel: existing_dir.clone(),
                full_write: false,
                regrouped_from: Some(existing_group.clone()),
                unchanged: false,
                soft_warn: None,
            };
        }
        return WriteAction::Skip(format!(
            "{kind} '{name}:{version}' already exists in the toolbox at '{existing_dir}' under group \
             '{existing_group}'; not writing a duplicate under group '{group}' (it would fail build) \
             — re-run with --overwrite to re-group it in place"
        ));
    }
    match placement {
        // a genuinely new tool: full write, but flag the softer cross-group rename signal (same name
        // under a different group at a different version — allowed, but likely an unbridged rename)
        Placement::New(target_rel) => {
            let soft_warn = other_groups
                .filter(|groups| groups.iter().any(|other| other != group))
                .map(|groups| {
                    format!(
                        "{kind} '{name}' is being written under group '{group}', but the toolbox \
                         already has a {kind} named '{name}' under group(s) {}; this adds a separate \
                         copy — pass --group-override <toolbox-group> to reconcile against the existing \
                         one",
                        groups.iter().cloned().collect::<Vec<_>>().join(", ")
                    )
                });
            WriteAction::Write {
                target_rel,
                full_write: true,
                regrouped_from: None,
                unchanged: false,
                soft_warn,
            }
        }
        Placement::Unchanged(target_rel) => WriteAction::Write {
            target_rel,
            full_write: false,
            regrouped_from: None,
            unchanged: true,
            soft_warn: None,
        },
        Placement::Update(target_rel) => WriteAction::Write {
            target_rel,
            full_write: false,
            regrouped_from: None,
            unchanged: false,
            soft_warn: None,
        },
        Placement::SkipMove(existing) => WriteAction::Skip(format!(
            "{kind} '{name}:{version}' already exists in the toolbox at '{existing}'; not writing a \
             second copy at '{}' (it would fail build) — omit =dir to update it in place, or remove \
             the old copy first",
            explicit_dest.unwrap_or_default()
        )),
        Placement::SkipDiffers => WriteAction::Skip(format!(
            "{kind} '{name}:{version}' already exists in the toolbox and differs; not updated — pass \
             --overwrite to update it"
        )),
    }
}

/// Resolve the directory the exported toolbox is written to
///
/// An explicit `--output` always wins. Otherwise the output anchors on `--config` (the toolbox that
/// config lives in) so pointing at a toolbox's `config.toml` exports into it; with neither flag the
/// default is `./toolbox` for a brand-new toolbox. Mirrors `build`'s config-anchored path resolution.
///
/// # Arguments
///
/// * `cmd` - The export command
fn resolve_output(cmd: &ExportToolbox) -> PathBuf {
    // an explicit --output wins; else the --config directory; else the create-new default
    cmd.output.clone().unwrap_or_else(|| match &cmd.config {
        Some(config) => build::config_base_dir(config),
        None => PathBuf::from("./toolbox"),
    })
}

/// The `config.toml` at the export output root, if one already exists
///
/// Its presence makes the export an append into an existing toolbox: the file becomes the settings
/// source and is preserved unless `--overwrite-config`.
///
/// # Arguments
///
/// * `output` - The resolved toolbox output directory
fn existing_config_path(output: &Path) -> Option<PathBuf> {
    let path = output.join("config.toml");
    path.exists().then_some(path)
}

/// Build [`ToolboxSettings`] from a loaded `config.toml`, with bundling driven by the run
///
/// # Arguments
///
/// * `config` - The loaded toolbox config
/// * `with_images` - Whether this run bundles images (drives `bundled_images`, never the config's claim)
fn settings_from_config(config: build::ToolboxConfig, with_images: bool) -> ToolboxSettings {
    ToolboxSettings {
        name: config.name,
        registry: config.registry,
        registries: config.registries,
        image_path_prefix: config.image_path_prefix,
        export_image_path: config.export_image_path,
        export_pipeline_path: config.export_pipeline_path,
        // bundling reflects what this run actually exports, never the reused config's claim
        bundled_images: with_images,
        base_image: config.base_image,
    }
}

/// Resolve the toolbox-wide settings for an export
///
/// Priority: an explicit `--config` that exists (seed from another toolbox); else an existing
/// `config.toml` at the output root (append — its settings are reused and the file is preserved
/// unless `--overwrite-config`); else the `--name`/`--registry` flags (a new toolbox). A `--config`
/// that points at a missing file warns and falls through to the new-toolbox path rather than
/// hard-erroring, so a not-yet-created (or mistyped) target surfaces as a clear notice. Bundling is
/// always driven by the `--with-images` export action, not inherited from a config's `bundled_images`.
///
/// # Arguments
///
/// * `cmd` - The export command
/// * `existing_config` - The output's `config.toml` if it already exists (the append source)
/// * `progress` - The progress bar, for the "using existing config" notice and mismatch warnings
fn resolve_settings(
    cmd: &ExportToolbox,
    existing_config: Option<&Path>,
    progress: &Bar,
) -> Result<ToolboxSettings, Error> {
    // (1) an explicit --config seeds settings from another toolbox — when it exists. A --config that
    // names a missing file means the intended toolbox isn't there (yet), so warn and fall through to
    // creating a new toolbox from the flags instead of failing with a read error; a typo'd path
    // surfaces as this warning rather than a silent new toolbox
    if let Some(config_path) = &cmd.config {
        if config_path.exists() {
            return Ok(settings_from_config(
                build::load_config(config_path)?,
                cmd.with_images,
            ));
        }
        progress.warning(format!(
            "config.toml not found at '{}'; creating a new toolbox there instead of appending",
            config_path.display()
        ));
    }
    // (2) append: an existing config.toml at the output is the settings source
    if let Some(path) = existing_config {
        progress.info_anonymous(format!(
            "Using existing config.toml at '{}' for toolbox settings",
            path.display()
        ));
        let config = build::load_config(path)?;
        // warn when a run flag implies a setting the preserved config contradicts; the existing
        // config wins unless --overwrite-config, so the flag is otherwise silently ignored
        if !cmd.overwrite_config {
            if cmd.with_images && !config.bundled_images {
                progress.warning(
                    "--with-images is set but the existing config.toml has bundled_images = false; \
                     the existing setting is kept (pass --overwrite-config to update it)",
                );
            }
            if let Some(registry) = &cmd.registry
                && config.registry.as_deref() != Some(registry.as_str())
            {
                progress.warning(format!(
                    "--registry '{registry}' differs from the existing config.toml; the existing \
                     registry is kept (pass --overwrite-config to update it)"
                ));
            }
        }
        return Ok(settings_from_config(config, cmd.with_images));
    }
    // (3) new toolbox: derive from flags; registries/prefix/layout/base_image have no flag
    Ok(ToolboxSettings {
        name: cmd.name.clone(),
        registry: cmd.registry.clone(),
        registries: Vec::new(),
        image_path_prefix: None,
        export_image_path: None,
        export_pipeline_path: None,
        bundled_images: cmd.with_images,
        base_image: None,
    })
}

// ─── Manifest assembly ───────────────────────────────────────────────────────

/// Build an in-memory toolbox manifest from the resolved Thorium resources
///
/// Entries are keyed by `<group>/<name>` so that same-named resources from
/// different groups stay distinct until collision resolution runs (the manifest's
/// own maps are otherwise keyed by name). Each image carries the network policy
/// definitions it references so they can be written alongside it.
///
/// # Arguments
///
/// * `settings` - The resolved toolbox-wide settings (name/registry/bundling/etc.)
/// * `images` - The resolved Thorium images
/// * `pipelines` - The resolved Thorium pipelines
/// * `policies` - Fetched network policy definitions keyed by name
fn build_manifest(
    settings: &ToolboxSettings,
    images: &[Image],
    pipelines: &[Pipeline],
    policies: &HashMap<String, NetworkPolicyRequest>,
) -> ToolboxManifest {
    let mut image_entries: HashMap<String, manifest::ImageManifest> = HashMap::new();
    // map (group, name) -> exported version label so each pipeline's image map pins
    // the version we actually exported (rather than a name-keyed guess); built in the
    // same pass as the entries so the version label is only computed once per image
    let mut image_versions: HashMap<(String, String), String> = HashMap::new();
    for image in images {
        // the version label both keys this image's manifest entry and pins it in any pipeline
        // image map below, so compute it once here
        let version = version_label(&image.version);
        // the on-disk config is the image's Thorium request form
        let config = ImageRequest::from(image.clone());
        // bundle the definitions of the policies this image references, scoping each copy to the
        // group this toolbox exports the image in rather than the policy's full instance-wide
        // group set (so the exported toolbox doesn't reference groups it doesn't carry)
        let network_policies = config
            .network_policies
            .iter()
            .filter_map(|name| policies.get(name).cloned())
            .map(|mut policy| {
                policy.groups = vec![image.group.clone()];
                policy
            })
            .collect();
        // build_path is "./" because the manifest sits in the tool's own dir; config is embedded
        // inline (not config_from) and the bundled policies travel as definitions (not _from refs).
        // dir is empty here: this in-memory manifest only drives validation/collision/reconcile —
        // the authoritative per-image dir is computed by the auto-build that walks the written tree.
        let entry = manifest::ImageVersion {
            dir: String::new(),
            build_path: "./".to_string(),
            config_from: None,
            config: Some(config),
            network_policies_from: Vec::new(),
            network_policies,
        };
        // remember this image's exported version so pipelines can pin the exact version exported
        image_versions.insert((image.group.clone(), image.name.clone()), version.clone());
        // key the entry by <group>/<name> to keep same-named images from different groups
        // distinct until collision resolution collapses/renames them
        image_entries.insert(
            format!("{}/{}", image.group, image.name),
            manifest::ImageManifest {
                versions: HashMap::from([(version, entry)]),
            },
        );
    }
    let mut pipeline_entries: HashMap<String, manifest::PipelineManifest> = HashMap::new();
    for pipeline in pipelines {
        // flatten the (possibly staged) order into the unique image names this pipeline runs,
        // pinning each to the version we actually exported it under (falling back to "latest"
        // when the image wasn't part of this export, e.g. it lives outside the selection)
        let images_map = pipeline
            .order
            .iter()
            .flatten()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|name| {
                let version = image_versions
                    .get(&(pipeline.group.clone(), name.clone()))
                    .cloned()
                    .unwrap_or_else(|| "latest".to_string());
                (name, manifest::PipelineImage { version })
            })
            .collect();
        // pipelines carry no version axis here, so every entry is keyed "latest". dir is left empty:
        // this is the freshly-fetched in-memory entry, and its on-disk directory is resolved at write
        // time (an explicit `=dest`, the pipeline's existing dir, or the configured/default layout)
        let entry = manifest::PipelineVersion {
            dir: String::new(),
            description: pipeline.description.clone().unwrap_or_default(),
            images: images_map,
            config_from: None,
            config: Some(PipelineRequest::from(pipeline.clone())),
        };
        // key by <group>/<name> for the same group-distinctness reason as images above
        pipeline_entries.insert(
            format!("{}/{}", pipeline.group, pipeline.name),
            manifest::PipelineManifest {
                versions: HashMap::from([("latest".to_string(), entry)]),
            },
        );
    }
    ToolboxManifest {
        name: settings.name.clone(),
        registry: settings.registry.clone(),
        images: image_entries,
        pipelines: pipeline_entries,
        bundled_images: settings.bundled_images,
        image_path_prefix: settings.image_path_prefix.clone(),
    }
}

// ─── File Writing ────────────────────────────────────────────────────────────

/// How [`write_image_entry`] writes one image's files
#[derive(Clone, Copy)]
struct ImageWriteOptions {
    /// Mark the manifest `build = true` (a Dockerfile sits in the image's explicit `=dir`); otherwise
    /// the default reference-only manifest is written, pinned to the captured url via `exported_image_path`
    build: bool,
    /// (Re)generate `manifest.toml` — `true` for a full write (a new tool); `false` for an in-place
    /// update, which preserves the existing manifest's toolbox-authored build settings and only writes
    /// the manifest when it is missing (so the tool stays buildable)
    write_manifest: bool,
    /// Write the config's `image` url empty and omit `exported_image_path`, so a rebuild derives each
    /// image path from `config.toml` (a registry-agnostic release) instead of a pinned url
    strip_registry: bool,
    /// Open the config in an editor for review before writing
    review: bool,
}

/// Write a resolved image entry to the toolbox directory, resolving on-disk
/// conflicts; returns [`WriteOutcome::Quit`] if the user asked to stop
///
/// # Arguments
///
/// * `image_dir` - The tool directory to write this image's files into (resolved by the caller)
/// * `config` - The resolved image request (its `name` is the on-disk file stem)
/// * `version` - The toolbox version label to record
/// * `opts` - How to write this image (build flag, manifest (re)generation, registry stripping, review)
/// * `network_policies` - The policy definitions this image references
/// * `editor` - The editor command used when `opts.review` is set
/// * `resolver` - The on-disk conflict resolver
/// * `progress` - The progress bar
#[allow(clippy::too_many_arguments)]
async fn write_image_entry(
    image_dir: &Path,
    config: &ImageRequest,
    version: &str,
    opts: ImageWriteOptions,
    network_policies: &[NetworkPolicyRequest],
    editor: &str,
    resolver: &mut DiskConflictResolver,
    progress: &Bar,
) -> Result<WriteOutcome, Error> {
    // the config's own name is the json file stem; image_dir is the caller-resolved tool directory
    let name = &config.name;
    // --strip-registry publishes a registry-agnostic toolbox: clear the container url in the written
    // config (an empty `image`, the same form `init` scaffolds) so a rebuild derives the path from the
    // toolbox's own config.toml rather than a pinned url. The exported_image_path is dropped below too.
    let stripped = opts.strip_registry.then(|| {
        let mut cleared = config.clone();
        cleared.image = Some(String::new());
        cleared
    });
    let config = stripped.as_ref().unwrap_or(config);
    // curated (prioritized) field order so the written <name>.json matches what `init` scaffolds —
    // a consistent, edit-friendly layout across every spot that writes an image config. The order
    // is still deterministic (curated keys first, remaining keys sorted), so it stays diff-stable.
    let config_json =
        crate::utils::curated_json(config, crate::handlers::imports::merge::IMAGE_FIELD_ORDER)
            .map_err(|e| Error::new(format!("Failed to serialize image '{name}': {e}")))?;
    // let the user hand-edit the config first when --review is set; otherwise write it verbatim
    let final_json = if opts.review {
        // suspend the spinner while the editor owns the terminal
        progress
            .suspend_async(review_config_in_editor::<ImageRequest>(
                &config_json,
                &format!("export-image-{name}"),
                editor,
                crate::handlers::imports::merge::IMAGE_FIELD_ORDER,
            ))
            .await?
    } else {
        config_json
    };
    // short-circuit the whole export if the resolver prompt returns Quit at any write
    if resolver
        .write_yaml::<ImageRequest>(
            &image_dir.join(format!("{name}.json")),
            &final_json,
            progress,
        )
        .await?
        == WriteOutcome::Quit
    {
        return Ok(WriteOutcome::Quit);
    }
    // write the definition of every network policy this image references so an
    // import can create them in instances that lack them
    let mut policy_files: Vec<String> = Vec::new();
    for policy in network_policies {
        // name each policy file <policy>.policy.json so the manifest can list it under
        // network_policies_from and the importer recognizes it by suffix
        let file_name = format!("{}.policy.json", policy.name);
        // policy files are pretty-printed (not canonical) to stay human-readable on disk
        let policy_json = serde_json::to_string_pretty(policy).map_err(|e| {
            Error::new(format!(
                "Failed to serialize network policy '{}': {e}",
                policy.name
            ))
        })?;
        if resolver
            .write_yaml::<NetworkPolicyRequest>(&image_dir.join(&file_name), &policy_json, progress)
            .await?
            == WriteOutcome::Quit
        {
            return Ok(WriteOutcome::Quit);
        }
        policy_files.push(file_name);
    }
    // sort so regenerated manifests don't churn on set iteration order
    policy_files.sort_unstable();
    // (re)generate manifest.toml on a full write, or when an in-place update finds it missing (so the
    // tool stays buildable). An update with the manifest present skips this entirely, preserving the
    // toolbox-authored build/build_path/[base_image]/image_from that a regeneration would reset.
    let manifest_path = image_dir.join("manifest.toml");
    if opts.write_manifest || !manifest_path.exists() {
        // when build is set (a Dockerfile sits in this image's explicit dest dir), write a build = true
        // manifest with no exported_image_path so the rebuild builds from that context. --strip-registry
        // likewise omits exported_image_path so a rebuild derives the path from config.toml. otherwise
        // write the default reference-only (build = false) manifest and record the real registry url via
        // exported_image_path so a rebuild keeps the path the image actually lives at instead of
        // deriving one. image_name is set to the tool name (the second arg): it's irrelevant while the
        // image is pinned via exported_image_path, and only matters under build = true + --use-image-path.
        let manifest = generate_image_manifest(
            name,
            name,
            version,
            !opts.build,
            &policy_files,
            if opts.build || opts.strip_registry {
                None
            } else {
                config.image.as_deref()
            },
        );
        if resolver
            .write_toml::<build::ManifestToml>(&manifest_path, &manifest, progress)
            .await?
            == WriteOutcome::Quit
        {
            return Ok(WriteOutcome::Quit);
        }
    }
    // always write description.md so every tool carries a docs file; a None/empty description is
    // written as an empty file (never the literal "null"), which build treats as absent — leaving the
    // inline config value (also empty) untouched. The markdown is the toolbox's source of truth.
    if resolver
        .write_text(
            &image_dir.join("description.md"),
            config.description.as_deref().unwrap_or_default(),
            progress,
        )
        .await?
        == WriteOutcome::Quit
    {
        return Ok(WriteOutcome::Quit);
    }
    Ok(WriteOutcome::Written)
}

/// Write a resolved pipeline entry to the toolbox directory, resolving on-disk
/// conflicts; returns [`WriteOutcome::Quit`] if the user asked to stop
///
/// # Arguments
///
/// * `pipeline_dir` - The tool directory to write this pipeline's files into (resolved by the caller)
/// * `config` - The resolved pipeline request (its `name` is the on-disk file stem)
/// * `description` - The pipeline description to mirror to description.md
/// * `image_versions` - The (image name, version) pairs for the manifest's image map
/// * `review` - Open the config in an editor for review before writing
/// * `editor` - The editor command used when `review` is set
/// * `resolver` - The on-disk conflict resolver
/// * `progress` - The progress bar
#[allow(clippy::too_many_arguments)]
async fn write_pipeline_entry(
    pipeline_dir: &Path,
    config: &PipelineRequest,
    description: &str,
    image_versions: &[(String, String)],
    review: bool,
    editor: &str,
    resolver: &mut DiskConflictResolver,
    progress: &Bar,
) -> Result<WriteOutcome, Error> {
    // the config's own name is the json file stem; pipeline_dir is the caller-resolved tool directory
    let name = &config.name;
    // curated (prioritized) field order so the written <name>.json matches what `init` scaffolds —
    // a consistent, edit-friendly layout across every spot that writes a pipeline config. The order
    // is still deterministic (curated keys first, remaining keys sorted), so it stays diff-stable.
    // The config's `description` field is serialized verbatim (kept as `null` when unset, like the
    // Thorium struct) — the markdown source of truth is the description.md file written below.
    let config_json = crate::utils::curated_json(
        config,
        crate::handlers::imports::merge::PIPELINE_FIELD_ORDER,
    )
    .map_err(|e| Error::new(format!("Failed to serialize pipeline '{name}': {e}")))?;
    // let the user hand-edit the config first when --review is set; otherwise write it verbatim
    let final_json = if review {
        // suspend the spinner while the editor owns the terminal
        progress
            .suspend_async(review_config_in_editor::<PipelineRequest>(
                &config_json,
                &format!("export-pipeline-{name}"),
                editor,
                crate::handlers::imports::merge::PIPELINE_FIELD_ORDER,
            ))
            .await?
    } else {
        config_json
    };
    // short-circuit the whole export if the resolver prompt returns Quit at any write
    if resolver
        .write_yaml::<PipelineRequest>(
            &pipeline_dir.join(format!("{name}.json")),
            &final_json,
            progress,
        )
        .await?
        == WriteOutcome::Quit
    {
        return Ok(WriteOutcome::Quit);
    }
    // generate the per-tool manifest pinning each image to its exported version
    let manifest = generate_pipeline_manifest(name, image_versions);
    if resolver
        .write_toml::<build::ManifestToml>(&pipeline_dir.join("manifest.toml"), &manifest, progress)
        .await?
        == WriteOutcome::Quit
    {
        return Ok(WriteOutcome::Quit);
    }
    // always write description.md so every pipeline carries a docs file; a None/empty description is
    // an empty file (never the literal "null"), which build treats as absent — leaving the inline
    // config value untouched. The markdown is the toolbox's source of truth.
    if resolver
        .write_text(&pipeline_dir.join("description.md"), description, progress)
        .await?
        == WriteOutcome::Quit
    {
        return Ok(WriteOutcome::Quit);
    }
    Ok(WriteOutcome::Written)
}

// ─── Main Export ─────────────────────────────────────────────────────────────

/// Exports the selected images and pipelines into an on-disk toolbox directory
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to fetch resources
/// * `cmd` - The export args (target directory and resource selection)
/// * `args` - The top-level thorctl args (worker count)
/// * `conf` - The Thorctl config (editor resolution)
pub async fn export(
    thorium: Thorium,
    cmd: &ExportToolbox,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // when --group is combined with named --pipelines/--images it's only the default group for
    // those names, not a full-group export; say so to avoid the "why didn't it export the whole
    // group?" surprise
    if cmd.group.is_some() && (!cmd.pipelines.is_empty() || !cmd.images.is_empty()) {
        println!(
            "Note: --group is used only as the default group for the named --pipelines/--images; \
             omit them to export the whole group"
        );
    }
    // resolve the toolbox output directory once: an explicit --output, else the --config dir, else
    // ./toolbox. Announce a defaulted output (implicit-behavior rule) so it's clear where the
    // toolbox is written and why; an explicit --output is self-evident and gets no extra notice
    let output = resolve_output(cmd);
    if cmd.output.is_none() {
        if cmd.config.is_some() {
            println!(
                "No --output set; exporting into the toolbox at '{}' (from --config)",
                output.display().to_string().bright_cyan()
            );
        } else {
            println!(
                "No --output set; creating a new toolbox at '{}'",
                output.display().to_string().bright_cyan()
            );
        }
    }
    // resolve the editor up front so --review uses a consistent command across all configs
    let editor = resolve_editor(None, conf);
    let progress = Bar::new("toolbox export", "Exporting", BarKind::Timer);
    // load the existing toolbox once (committed toolbox.json or an on-disk crawl); it is reused both
    // to resolve a no-selection "refresh all" and to reconcile the writes below
    let existing_manifest = load_existing_manifest(&output, &progress).await;
    // decide what to export: an explicit selection wins; otherwise, with no selection, a refresh of
    // every tool already in the toolbox (gated by --overwrite since it rewrites them); otherwise error
    let has_selection = cmd.group.is_some() || !cmd.pipelines.is_empty() || !cmd.images.is_empty();
    let (images, pipelines) = if has_selection {
        resolve_resources(&thorium, cmd, args.workers).await?
    } else if let Some(existing) = &existing_manifest {
        if cmd.overwrite {
            refresh_existing_resources(&thorium, existing, args.workers, &progress).await?
        } else {
            return Err(Error::new(
                "No resources selected. Pass --overwrite to refresh every tool already in the \
                 toolbox, or specify --group/--pipelines/--images.",
            ));
        }
    } else {
        return Err(Error::new(
            "No resources to export. Specify --group, --pipelines, or --images.",
        ));
    };
    println!(
        "Exporting {} images and {} pipelines to '{}'{}",
        images.len().to_string().bright_green(),
        pipelines.len().to_string().bright_green(),
        output.display().to_string().bright_cyan(),
        if cmd.with_images {
            " (bundling container images)".bright_yellow().to_string()
        } else {
            String::new()
        },
    );
    // gather the unique (group, policy name) pairs referenced across all images, keeping
    // one referencing image per pair for error context
    let mut wanted: Vec<(String, String, String)> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for image in &images {
        for policy_name in &image.network_policies {
            // first image to reference a given (group, policy) wins as the error-context name
            if seen.insert((image.group.clone(), policy_name.clone())) {
                wanted.push((image.group.clone(), policy_name.clone(), image.name.clone()));
            }
        }
    }
    // fetch existing policies scoped to just the exported images' groups so a name that
    // is ambiguous across the whole instance still resolves uniquely within its group
    // (a global get-by-name 400s on cross-group ambiguity)
    let groups: Vec<String> = images
        .iter()
        .map(|image| image.group.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let index = policies::fetch_existing_in_groups(&thorium, &groups).await?;
    let mut policies: HashMap<String, NetworkPolicyRequest> = HashMap::new();
    // collect references that resolved to no definition; an export that omits a policy it
    // references produces a structurally-incomplete toolbox, so these drive a non-zero exit
    // (and an aggregated end-of-run summary) rather than being lost as mid-stream warnings
    let mut dangling_policies: Vec<String> = Vec::new();
    for (group, policy_name, image_name) in wanted {
        // look each referenced policy up by its (group, name) identity within the fetched index
        match index.get(&(group.clone(), policy_name.clone())) {
            // found: stash its request form keyed by name for build_manifest to attach
            Some(policy) => {
                policies.insert(policy_name, NetworkPolicyRequest::from(policy));
            }
            // a dangling reference in the source instance isn't fatal to the export, but the
            // toolbox will be missing that definition, so record it and warn
            None => {
                progress.warning(format!(
                    "Network policy '{policy_name}' (referenced by image '{image_name}' in group \
                     '{group}') was not found; the exported toolbox won't include its definition, so \
                     an import will rely on the target instance already having it",
                ));
                dangling_policies.push(format!("{policy_name} (group '{group}')"));
            }
        }
    }
    // detect an existing config.toml at the output root: it makes this an append into an existing
    // toolbox (its settings are the source, and it is preserved unless --overwrite-config)
    let existing_config = existing_config_path(&output);
    // resolve the toolbox-wide settings — explicit --config seed, else an existing config.toml at
    // the output (append), else the --name/--registry flags. Bundling is driven by --with-images.
    let settings = resolve_settings(cmd, existing_config.as_deref(), &progress)?;
    // prompts are only possible interactively (not --skip-conflicts) AND on a real terminal;
    // this gates both collision resolution and the on-disk conflict resolver below
    let can_prompt = !cmd.skip_conflicts && IsTerminal::is_terminal(&std::io::stdin());
    // --review opens an editor per config, which needs a real terminal; ignore it (with a
    // warning) when there's no TTY so a headless run doesn't hang waiting on an editor
    let review = if cmd.review && !IsTerminal::is_terminal(&std::io::stdin()) {
        progress.warning("--review needs a terminal; skipping the editor review for each config");
        false
    } else {
        cmd.review
    };
    // build an in-memory manifest and run it through the SAME validation and
    // collision-resolution flow as `toolbox import`, so duplicates/collisions are
    // resolved identically (de-dupe, rename + cascade, or skip) before anything
    // touches disk
    let mut manifest = build_manifest(&settings, &images, &pipelines, &policies);
    // snapshot each resource's original group BEFORE any override so collision resolution can
    // tell which members truly collided versus were collapsed into one group by --group-override
    let sources = manifest.capture_source_groups();
    if let Some(group) = &cmd.group_override {
        progress.info_anonymous(format!(
            "Overriding all image/pipeline export groups to '{}'",
            group.bright_yellow()
        ));
        manifest = manifest.override_group(group);
        // override_group only rewrites image/pipeline config groups; keep each bundled policy's
        // groups consistent with the overridden image groups on disk
        for image in manifest.images.values_mut() {
            for version in image.versions.values_mut() {
                for policy in &mut version.network_policies {
                    policy.groups = vec![group.clone()];
                }
            }
        }
    }
    // drop image versions with no config and pipelines that are structurally broken (warning each)
    shared::warn_dropped(&manifest.validate_structural(), &progress);
    // drop pipelines whose order references images not present in their group (warning each)
    shared::warn_dropped(&manifest.validate_group_coherence(), &progress);
    // collapse byte-identical duplicates and rename/skip true (group, name) collisions
    collisions::resolve_collisions(&mut manifest, &sources, can_prompt, &progress)?;
    // re-check coherence: collision renames/repointing can re-break a pipeline's group view
    shared::warn_dropped(&manifest.validate_group_coherence(), &progress);
    // map each explicitly-named resource (by tool name) to its optional `=destpath`, resolved to a
    // directory relative to the toolbox root, so the write loops can place a single resource at a
    // chosen directory instead of the configured layout. Keyed by name because that is the on-disk
    // tool directory leaf and names are unique after collision resolution; whole-group/auto-pulled
    // resources aren't named here so they use the configured/default layout. A dest that resolves
    // outside the toolbox is a hard error here, before anything is written.
    let dest_overrides = resolve_dest_overrides(cmd, &output)?;
    // index the existing toolbox (loaded once above) for append reconciliation: lets the write loops
    // skip unchanged resources and their bundle work, update a tool where it already lives, and refuse
    // to write a second copy at a different directory. Empty for a fresh export.
    let (existing_images, existing_pipelines) = index_existing(existing_manifest.as_ref());
    // index where each tool already lives by (group, name) so a re-export updates it in place (its
    // recorded directory) instead of writing a duplicate at the default layout
    let existing_image_dirs = dirs_by_name(&existing_images);
    let existing_pipeline_dirs = dirs_by_name(&existing_pipelines);
    // index the groups each tool name appears under, so writing a "new" tool whose name already exists
    // under a different group (a likely group rename) can warn instead of silently duplicating
    let existing_image_groups = groups_by_name(&existing_images);
    let existing_pipeline_groups = groups_by_name(&existing_pipelines);
    // index each tool's build identity (name, version) → (group, dir), independent of group, so a new
    // write that would duplicate an existing (name, version) — the exact thing `build` rejects — is
    // skipped with a --group-override hint rather than producing a broken toolbox
    let existing_image_locs = locs_by_name_version(&existing_images);
    let existing_pipeline_locs = locs_by_name_version(&existing_pipelines);
    // write the resolved manifest to disk, resolving on-disk conflicts
    let mut resolver = DiskConflictResolver::new(cmd.overwrite, can_prompt, editor.to_string());
    let mut stopped = false;
    // (image name, container url, tool dir) tarballs to bundle after the config pass; the
    // config writes stay sequential (the resolver prompts), but the heavy container
    // pull/save is run bounded-parallel below. The tool dir is carried so the tarball lands
    // beside the image's manifest (the configured layout), matching where import looks for it.
    let mut bundle_jobs: Vec<(String, Option<String>, PathBuf)> = Vec::new();
    'images: for image_manifest in manifest.images.values() {
        for (version, entry) in &image_manifest.versions {
            // an export always embeds a config; skip defensively so a configless entry
            // (shouldn't occur here) doesn't panic on unwrap
            let Some(config) = &entry.config else {
                continue;
            };
            // decide where and how to write this image against the existing toolbox (placement,
            // group-mismatch re-group/skip, and any cross-group warning) — see decide_write
            let key = (config.group.clone(), config.name.clone(), version.clone());
            let name_key = (config.group.clone(), config.name.clone());
            let current_json = crate::utils::canonical_json(config)?;
            let default_rel = format!(
                "{}/{}",
                settings.export_image_path.as_deref().unwrap_or("images"),
                config.name
            );
            let explicit_dest = dest_overrides.get(config.name.as_str()).map(String::as_str);
            let action = decide_write(
                "image",
                &config.group,
                &config.name,
                version,
                &current_json,
                explicit_dest,
                existing_image_dirs.get(&name_key).map(String::as_str),
                existing_images.get(&key).map(|(json, _dir)| json.as_str()),
                existing_image_locs.get(&(config.name.clone(), version.clone())),
                existing_image_groups.get(&config.name),
                &default_rel,
                cmd.overwrite,
            );
            // a skip outcome warns and moves on to the next resource (the rest still export)
            let (target_rel, full_write, regrouped_from, unchanged) = match action {
                WriteAction::Skip(msg) => {
                    progress.warning(msg);
                    continue;
                }
                WriteAction::Write {
                    target_rel,
                    full_write,
                    regrouped_from,
                    unchanged,
                    soft_warn,
                } => {
                    if let Some(warn) = soft_warn {
                        progress.warning(warn);
                    }
                    (target_rel, full_write, regrouped_from, unchanged)
                }
            };
            let image_dir = output.join(&target_rel);
            // when a NEW image is pointed at an explicit `=dir` that already holds a Dockerfile, the
            // user is folding this config into an existing build context, so mark its manifest
            // build = true. Only the explicit-dest fresh-write case is auto-detected.
            let build =
                full_write && explicit_dest.is_some() && image_dir.join("Dockerfile").exists();
            if build {
                progress.info_anonymous(format!(
                    "Found a Dockerfile in '{target_rel}'; marking image '{}' build = true",
                    config.name
                ));
            }
            // an unchanged image skips the redundant container re-bundle when its tarball is already
            // saved; a re-group/update refreshes the config in place (manifest preserved)
            let mut skip_bundle = false;
            if unchanged {
                skip_bundle =
                    !cmd.with_images || image_dir.join(format!("{}.tar.gz", config.name)).exists();
                progress.info_anonymous(format!(
                    "Unchanged: image '{}:{version}' already current in the toolbox",
                    config.name
                ));
            } else if let Some(from) = &regrouped_from {
                progress.info_anonymous(format!(
                    "Re-grouping image '{}:{version}' from '{from}' to '{}' in place at '{target_rel}'",
                    config.name, config.group
                ));
            } else if !full_write {
                progress.info_anonymous(format!(
                    "Updating image '{}:{version}' in place at '{target_rel}' (config only; manifest \
                     build settings preserved)",
                    config.name
                ));
            }
            let outcome = write_image_entry(
                &image_dir,
                config,
                version,
                ImageWriteOptions {
                    build,
                    write_manifest: full_write,
                    strip_registry: cmd.strip_registry,
                    review,
                },
                &entry.network_policies,
                editor,
                &mut resolver,
                &progress,
            )
            .await?;
            // a Quit at any prompt stops the whole export; flag it and bail out of both loops
            if outcome == WriteOutcome::Quit {
                stopped = true;
                break 'images;
            }
            // defer the heavy container pull/save to a bounded-parallel pass after all the
            // sequential (prompt-driven) config writes complete; skip it for an unchanged image
            // whose tarball is already bundled (skip_bundle)
            if cmd.with_images && !skip_bundle {
                bundle_jobs.push((config.name.clone(), config.image.clone(), image_dir));
            }
        }
    }
    // skip the pipeline pass entirely if the image pass was quit
    if !stopped {
        'pipelines: for pipeline_manifest in manifest.pipelines.values() {
            for (version, entry) in &pipeline_manifest.versions {
                // pipelines, like images, always carry a config in an export; skip defensively
                let Some(config) = &entry.config else {
                    continue;
                };
                // decide placement + reconciliation the same way images do (a pipeline manifest carries
                // no toolbox-authored build settings, so write_pipeline_entry always regenerates it —
                // the `full_write` flag only drives the "Updating in place" notice here)
                let key = (config.group.clone(), config.name.clone(), version.clone());
                let name_key = (config.group.clone(), config.name.clone());
                let current_json = crate::utils::canonical_json(config)?;
                let default_rel = format!(
                    "{}/{}",
                    settings
                        .export_pipeline_path
                        .as_deref()
                        .unwrap_or("pipelines"),
                    config.name
                );
                let explicit_dest = dest_overrides.get(config.name.as_str()).map(String::as_str);
                let action = decide_write(
                    "pipeline",
                    &config.group,
                    &config.name,
                    version,
                    &current_json,
                    explicit_dest,
                    existing_pipeline_dirs.get(&name_key).map(String::as_str),
                    existing_pipelines
                        .get(&key)
                        .map(|(json, _dir)| json.as_str()),
                    existing_pipeline_locs.get(&(config.name.clone(), version.clone())),
                    existing_pipeline_groups.get(&config.name),
                    &default_rel,
                    cmd.overwrite,
                );
                let (target_rel, full_write, regrouped_from, unchanged) = match action {
                    WriteAction::Skip(msg) => {
                        progress.warning(msg);
                        continue;
                    }
                    WriteAction::Write {
                        target_rel,
                        full_write,
                        regrouped_from,
                        unchanged,
                        soft_warn,
                    } => {
                        if let Some(warn) = soft_warn {
                            progress.warning(warn);
                        }
                        (target_rel, full_write, regrouped_from, unchanged)
                    }
                };
                if unchanged {
                    progress.info_anonymous(format!(
                        "Unchanged: pipeline '{}:{version}' already current in the toolbox",
                        config.name
                    ));
                } else if let Some(from) = &regrouped_from {
                    progress.info_anonymous(format!(
                        "Re-grouping pipeline '{}:{version}' from '{from}' to '{}' in place at \
                         '{target_rel}'",
                        config.name, config.group
                    ));
                } else if !full_write {
                    progress.info_anonymous(format!(
                        "Updating pipeline '{}:{version}' in place at '{target_rel}'",
                        config.name
                    ));
                }
                let pipeline_dir = output.join(&target_rel);
                // the resolved image map carries the (possibly renamed) names paired
                // with the versions we exported them under
                let mut image_versions: Vec<(String, String)> = entry
                    .images
                    .iter()
                    .map(|(name, image)| (name.clone(), image.version.clone()))
                    .collect();
                image_versions.sort();
                let outcome = write_pipeline_entry(
                    &pipeline_dir,
                    config,
                    &entry.description,
                    &image_versions,
                    review,
                    editor,
                    &mut resolver,
                    &progress,
                )
                .await?;
                // a Quit here stops the export before the bundling and config.toml passes
                if outcome == WriteOutcome::Quit {
                    stopped = true;
                    break 'pipelines;
                }
            }
        }
    }
    // a user quit leaves a partial repo with no config.toml/toolbox.json; report and exit cleanly
    if stopped {
        progress.refresh("Export stopped early", BarKind::Timer);
        progress.finish();
        return Ok(());
    }
    // bundle the queued container images in parallel (container pull/save here capture
    // their output rather than streaming it, so concurrency is safe). Bounded by --workers.
    // Bundling is best-effort: a failed pull/save warns and is collected so one bad image
    // doesn't abort the export (save cleans up its own partial archive on failure), but the
    // collected failures drive a non-zero exit at the end so a scripted export can't mistake
    // an incomplete bundle for a complete one.
    let mut bundle_failures: Vec<String> = Vec::new();
    if !bundle_jobs.is_empty() {
        // never spawn more workers than jobs, and never zero (buffer_unordered(0) stalls)
        let workers = std::cmp::min(args.workers, bundle_jobs.len()).max(1);
        progress.refresh("Bundling images", BarKind::Bound(bundle_jobs.len() as u64));
        // pull+save each container concurrently, pairing every result with its image name so a
        // failure can be reported even though the stream completes out of order
        let results: Vec<(String, Result<(), Error>)> = stream::iter(bundle_jobs)
            .map(|(name, url, dir)| {
                let progress = &progress;
                async move {
                    // save the tarball into the same tool directory the manifest was written to,
                    // so import (which reads the recorded per-image dir) finds it
                    let outcome = bundle_image(&dir, &name, url.as_deref(), progress).await;
                    progress.inc(1);
                    (name, outcome)
                }
            })
            .buffer_unordered(workers)
            .collect()
            .await;
        // warn (don't abort) on each failed bundle so the rest of the export still completes
        for (name, outcome) in results {
            if let Err(err) = outcome {
                progress.warning(format!(
                    "Could not bundle image '{name}': {err}; it will be missing from the toolbox, \
                     so importing '{name}' will fail to find its tarball"
                ));
                bundle_failures.push(name);
            }
        }
        // one summary warning so a multi-image failure is visible at a glance, not just per-image
        if !bundle_failures.is_empty() {
            progress.warning(format!(
                "{} image(s) could not be bundled ({}); the toolbox is marked bundled but is \
                 incomplete — re-export or push those images manually",
                bundle_failures.len(),
                bundle_failures.join(", ")
            ));
        }
    }
    // config.toml is the toolbox's sticky identity: an existing one is preserved (its settings were
    // the source above) unless --overwrite-config, so an append never clobbers the toolbox's
    // settings. A fresh export creates it (announced). The real image urls live in each image config
    // (see exported_image_path), so the registry here only matters for tools later marked buildable.
    let config_path = output.join("config.toml");
    if existing_config.is_some() && !cmd.overwrite_config {
        progress.info_anonymous(format!(
            "Keeping existing config.toml at '{}' (pass --overwrite-config to replace it)",
            config_path.display()
        ));
    } else {
        if existing_config.is_none() {
            progress.info_anonymous(format!(
                "No config.toml at '{}'; creating one",
                config_path.display()
            ));
        }
        let config_toml = render_config_toml(
            &settings.name,
            settings.registry.as_deref(),
            &settings.registries,
            settings.image_path_prefix.as_deref(),
            settings.export_image_path.as_deref(),
            settings.export_pipeline_path.as_deref(),
            settings.bundled_images,
            settings.base_image.as_ref(),
        );
        // written directly (not via the per-file resolver) because the sticky-config rule, not the
        // resolver's overwrite/skip behavior, governs config.toml
        tokio::fs::write(&config_path, config_toml)
            .await
            .map_err(|e| Error::new(format!("Failed to write '{}': {e}", config_path.display())))?;
    }
    // Auto-build toolbox.json, preserving the real image urls captured from Thorium.
    // build walks the tree with synchronous std::fs, so run it off the async runtime.
    let build_cmd = BuildToolbox {
        config: output.join("config.toml"),
        // leaf comes from the tool name, not image_name: an export pins urls via
        // exported_image_path, so the repo-path leaf is irrelevant here
        use_image_path: false,
        output: Some(output.join("toolbox.json")),
        path: Some(output.clone()),
        // an export records each image's real published url, so no tag suffix is applied
        tag_suffix: None,
    };
    // run the synchronous filesystem walk on a blocking thread; the outer ? unwraps the join
    // result and the inner ? the build result
    tokio::task::spawn_blocking(move || build::build(&build_cmd))
        .await
        .map_err(|err| Error::new(format!("Toolbox build task failed: {err}")))??;
    progress.finish();
    // a missing bundled tarball or an omitted (dangling) policy means the written toolbox is
    // not fully self-contained; report that plainly and exit non-zero so a scripted
    // export -> import handoff doesn't treat an incomplete toolbox as a success
    if bundle_failures.is_empty() && dangling_policies.is_empty() {
        println!(
            "\n{} Toolbox exported to '{}'. Import it with: thorctl toolbox import {}",
            "Done!".bright_green(),
            output.display(),
            output.join("toolbox.json").display()
        );
        return Ok(());
    }
    // build a human summary of what made the toolbox incomplete
    let mut problems: Vec<String> = Vec::new();
    if !bundle_failures.is_empty() {
        problems.push(format!(
            "{} image tarball(s) missing ({})",
            bundle_failures.len(),
            bundle_failures.join(", ")
        ));
    }
    if !dangling_policies.is_empty() {
        problems.push(format!(
            "{} referenced network polic(y/ies) not found and omitted ({})",
            dangling_policies.len(),
            dangling_policies.join(", ")
        ));
    }
    println!(
        "\n{} Toolbox written to '{}', but it is INCOMPLETE: {}. Resolve these and re-export \
         before importing.",
        "Warning:".bright_yellow(),
        output.display(),
        problems.join("; ")
    );
    Err(Error::new(format!(
        "export incomplete: {}",
        problems.join("; ")
    )))
}

/// Download and save an image's container image file into the toolbox bundle
///
/// Writes `<dir>/<name>.tar.gz` (the image's tool directory, beside its manifest). Images
/// without a container url are skipped.
///
/// # Arguments
///
/// * `dir` - The image's tool directory (where its manifest was written)
/// * `name` - The exported image name (the tarball's file stem)
/// * `url` - The image's container url, if any
/// * `progress` - The progress bar to route the skip warning through
async fn bundle_image(
    dir: &Path,
    name: &str,
    url: Option<&str>,
    progress: &Bar,
) -> Result<(), Error> {
    let Some(url) = url else {
        // route through the bar so the warning respects --quiet like every other one; the
        // toolbox is still marked bundled, so importing this image will fail to find its tarball
        progress.warning(format!(
            "Image '{}' has no container image url; skipping its tarball — importing it from this \
             bundled toolbox will fail to find '{}'",
            name.bright_cyan(),
            dir.join(format!("{name}.tar.gz")).display(),
        ));
        return Ok(());
    };
    // dedicated sub-bar so concurrent bundles each show their own pull/save progress
    let bar = Bar::new(name, "Bundling image", BarKind::Timer);
    // pull the container locally first so save has a local image to export
    container::pull(url, &bar).await?;
    // save the tarball into the image's tool directory (where its manifest was written); import
    // resolves this exact location from the per-image `dir` recorded in toolbox.json
    let tar = dir.join(format!("{name}.tar.gz"));
    container::save(url, &tar, &bar).await?;
    bar.finish_and_clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ExportToolbox, Placement, WriteAction, decide_write, dirs_by_name, existing_resource_ids,
        groups_by_name, locs_by_name_version, manifest, plan_placement, resolve_dest_within,
        resolve_output,
    };
    use std::collections::{BTreeSet, HashMap};
    use std::path::{Path, PathBuf};
    use thorium::models::{ImageRequest, PipelineRequest};

    /// Build an `ExportToolbox` with only the path-relevant fields set; the rest default to a
    /// no-op export so `resolve_output` can be exercised in isolation
    fn export_cmd(output: Option<&str>, config: Option<&str>) -> ExportToolbox {
        ExportToolbox {
            group: None,
            pipelines: Vec::new(),
            images: Vec::new(),
            group_override: None,
            output: output.map(PathBuf::from),
            config: config.map(PathBuf::from),
            name: "My Toolbox".to_string(),
            registry: None,
            skip_conflicts: false,
            review: false,
            overwrite: false,
            overwrite_config: false,
            with_images: false,
            strip_registry: false,
        }
    }

    /// An explicit `--output` always wins, regardless of `--config`
    #[test]
    fn resolve_output_prefers_explicit() {
        let cmd = export_cmd(Some("./dist"), Some("mytb/config.toml"));
        assert_eq!(resolve_output(&cmd), PathBuf::from("./dist"));
    }

    /// With no `--output`, the output anchors on the `--config` directory
    #[test]
    fn resolve_output_anchors_on_config() {
        // a config in a subdir → that subdir
        let cmd = export_cmd(None, Some("mytb/config.toml"));
        assert_eq!(resolve_output(&cmd), PathBuf::from("mytb"));
        // a bare config.toml has an empty parent → the current directory
        let bare = export_cmd(None, Some("config.toml"));
        assert_eq!(resolve_output(&bare), PathBuf::from("."));
    }

    /// With neither `--output` nor `--config`, the default is the create-new `./toolbox`
    #[test]
    fn resolve_output_defaults_to_toolbox() {
        let cmd = export_cmd(None, None);
        assert_eq!(resolve_output(&cmd), PathBuf::from("./toolbox"));
    }

    /// A relative `=dest` is interpreted relative to the toolbox root
    #[test]
    fn resolve_dest_within_relative_is_toolbox_rooted() {
        assert_eq!(
            resolve_dest_within(Path::new("/tb"), "tools/clamav").unwrap(),
            "tools/clamav"
        );
    }

    /// An absolute `=dest` inside the toolbox is re-expressed relative to the root
    #[test]
    fn resolve_dest_within_absolute_inside_is_relativized() {
        assert_eq!(
            resolve_dest_within(Path::new("/tb"), "/tb/pipelines/static/identify-files").unwrap(),
            "pipelines/static/identify-files"
        );
    }

    /// A `..`-bearing dest that normalizes back inside the toolbox is accepted (the explicit-path
    /// case: a path that re-descends into the toolbox)
    #[test]
    fn resolve_dest_within_dotdot_reentering_is_accepted() {
        assert_eq!(
            resolve_dest_within(Path::new("/a/b/tb"), "../tb/pipelines/x").unwrap(),
            "pipelines/x"
        );
    }

    /// A dest that lands outside the toolbox (absolute elsewhere, an escaping `..`, or the root
    /// itself) is rejected
    #[test]
    fn resolve_dest_within_outside_is_rejected() {
        assert!(resolve_dest_within(Path::new("/tb"), "/other/x").is_err());
        assert!(resolve_dest_within(Path::new("/tb"), "../escape").is_err());
        assert!(resolve_dest_within(Path::new("/tb"), "/tb").is_err());
    }

    /// `dirs_by_name` collapses the `(group, name, version)` index to `(group, name) → dir`, skipping
    /// empty (legacy) dirs and keeping the first real one
    #[test]
    fn dirs_by_name_collapses_versions_and_skips_empty() {
        let mut index: HashMap<(String, String, String), (String, String)> = HashMap::new();
        index.insert(
            ("g".into(), "a".into(), "1".into()),
            ("{}".into(), "images/a".into()),
        );
        // a legacy entry with no recorded dir is ignored
        index.insert(
            ("g".into(), "b".into(), "1".into()),
            ("{}".into(), String::new()),
        );
        let dirs = dirs_by_name(&index);
        assert_eq!(
            dirs.get(&("g".into(), "a".into())).map(String::as_str),
            Some("images/a")
        );
        assert!(!dirs.contains_key(&("g".into(), "b".into())));
    }

    /// `groups_by_name` maps each tool name to the set of groups it appears under, so a same-named
    /// tool spread across groups (a rename tell-tale) is detectable
    #[test]
    fn groups_by_name_collects_groups_per_name() {
        let mut index: HashMap<(String, String, String), (String, String)> = HashMap::new();
        index.insert(
            ("static1".into(), "clamav".into(), "latest".into()),
            ("{}".into(), "images/clamav".into()),
        );
        index.insert(
            ("static1".into(), "clamav".into(), "1.0".into()),
            ("{}".into(), "images/clamav".into()),
        );
        index.insert(
            ("other".into(), "exiftool".into(), "latest".into()),
            ("{}".into(), "images/exiftool".into()),
        );
        let by_name = groups_by_name(&index);
        // clamav appears under one group (collapsed across its two versions)
        assert_eq!(
            by_name
                .get("clamav")
                .map(|g| g.iter().cloned().collect::<Vec<_>>()),
            Some(vec!["static1".to_string()])
        );
        assert_eq!(
            by_name
                .get("exiftool")
                .map(|g| g.iter().cloned().collect::<Vec<_>>()),
            Some(vec!["other".to_string()])
        );
    }

    /// `locs_by_name_version` maps build's identity `(name, version)` → `(group, dir)`, so a
    /// group-mismatched write can find where the tool already lives regardless of group
    #[test]
    fn locs_by_name_version_maps_build_identity() {
        let mut index: HashMap<(String, String, String), (String, String)> = HashMap::new();
        index.insert(
            ("toolbox-grp".into(), "clamav".into(), "1.0".into()),
            ("{}".into(), "tools/clamav".into()),
        );
        let locs = locs_by_name_version(&index);
        // looked up by (name, version) with no group, it returns the group + dir it lives at
        assert_eq!(
            locs.get(&("clamav".to_string(), "1.0".to_string())),
            Some(&("toolbox-grp".to_string(), "tools/clamav".to_string()))
        );
        // a different version isn't a match (build identity is name+version)
        assert!(!locs.contains_key(&("clamav".to_string(), "2.0".to_string())));
    }

    /// A tool not already in the toolbox is a fresh write at the resolved target (an explicit `=dest`
    /// wins, else the default layout)
    #[test]
    fn plan_placement_new_resource() {
        assert_eq!(
            plan_placement(None, None, None, "{}", "images/a", false),
            Placement::New("images/a".into())
        );
        assert_eq!(
            plan_placement(Some("tools/a"), None, None, "{}", "images/a", false),
            Placement::New("tools/a".into())
        );
    }

    /// An existing tool with a byte-identical config is Unchanged at its own directory; without a
    /// `=dest` the existing dir is reused regardless of the default layout
    #[test]
    fn plan_placement_unchanged_reuses_existing_dir() {
        assert_eq!(
            plan_placement(None, Some("custom/a"), Some("{}"), "{}", "images/a", false),
            Placement::Unchanged("custom/a".into())
        );
    }

    /// An existing tool whose config differs is an Update with `--overwrite`, else SkipDiffers — both
    /// targeting the tool's existing directory
    #[test]
    fn plan_placement_differs_overwrite_vs_skip() {
        assert_eq!(
            plan_placement(
                None,
                Some("custom/a"),
                Some("{\"old\":1}"),
                "{\"new\":1}",
                "images/a",
                true
            ),
            Placement::Update("custom/a".into())
        );
        assert_eq!(
            plan_placement(
                None,
                Some("custom/a"),
                Some("{\"old\":1}"),
                "{\"new\":1}",
                "images/a",
                false
            ),
            Placement::SkipDiffers
        );
        // a different version of an existing tool (no exact match) is a difference too
        assert_eq!(
            plan_placement(None, Some("custom/a"), None, "{}", "images/a", false),
            Placement::SkipDiffers
        );
    }

    /// An explicit `=dest` pointing somewhere other than where the tool already lives is a SkipMove
    /// (relocating would leave a duplicate); the same dir falls through to the normal update path
    #[test]
    fn plan_placement_explicit_move_is_rejected() {
        assert_eq!(
            plan_placement(
                Some("other/a"),
                Some("custom/a"),
                Some("{}"),
                "{}",
                "images/a",
                true
            ),
            Placement::SkipMove("custom/a".into())
        );
        // =dest equal to the existing dir is fine — identical config → Unchanged
        assert_eq!(
            plan_placement(
                Some("custom/a"),
                Some("custom/a"),
                Some("{}"),
                "{}",
                "images/a",
                false
            ),
            Placement::Unchanged("custom/a".into())
        );
    }

    /// A genuinely new tool is a full write at the resolved dir; a same-name/different-group tool at a
    /// different version is still a full write but carries the soft cross-group warning
    #[test]
    fn decide_write_new_and_soft_warn() {
        assert_eq!(
            decide_write(
                "image", "g", "a", "1.0", "{}", None, None, None, None, None, "images/a", false,
            ),
            WriteAction::Write {
                target_rel: "images/a".into(),
                full_write: true,
                regrouped_from: None,
                unchanged: false,
                soft_warn: None,
            }
        );
        // exists under another group but at a different version → allowed, with a soft warning
        let other: BTreeSet<String> = ["toolbox-grp".to_string()].into_iter().collect();
        let WriteAction::Write {
            full_write,
            soft_warn,
            ..
        } = decide_write(
            "image",
            "static",
            "a",
            "2.0",
            "{}",
            None,
            None,
            None,
            None,
            Some(&other),
            "images/a",
            false,
        )
        else {
            panic!("expected a Write");
        };
        assert!(full_write);
        assert!(soft_warn.is_some());
    }

    /// A group-mismatched `(name, version)` already in the toolbox: re-grouped in place with
    /// `--overwrite`, skipped with a warning without it
    #[test]
    fn decide_write_regroups_or_skips_on_collision() {
        let loc = ("toolbox-grp".to_string(), "tools/a".to_string());
        // --overwrite → re-group in place at the existing dir (not a full write; manifest preserved)
        assert_eq!(
            decide_write(
                "image",
                "static",
                "a",
                "1.0",
                "{}",
                None,
                None,
                None,
                Some(&loc),
                None,
                "images/a",
                true,
            ),
            WriteAction::Write {
                target_rel: "tools/a".into(),
                full_write: false,
                regrouped_from: Some("toolbox-grp".into()),
                unchanged: false,
                soft_warn: None,
            }
        );
        // no --overwrite → skip (would be a build-breaking duplicate)
        assert!(matches!(
            decide_write(
                "image",
                "static",
                "a",
                "1.0",
                "{}",
                None,
                None,
                None,
                Some(&loc),
                None,
                "images/a",
                false,
            ),
            WriteAction::Skip(_)
        ));
    }

    /// An exact match drives Unchanged (identical) or Update (differs, with --overwrite), each at the
    /// tool's existing directory and never a full manifest rewrite
    #[test]
    fn decide_write_unchanged_and_update() {
        // byte-identical at the same (group,name) dir → Unchanged
        assert_eq!(
            decide_write(
                "image",
                "g",
                "a",
                "1.0",
                "{}",
                None,
                Some("images/a"),
                Some("{}"),
                None,
                None,
                "images/a",
                false,
            ),
            WriteAction::Write {
                target_rel: "images/a".into(),
                full_write: false,
                regrouped_from: None,
                unchanged: true,
                soft_warn: None,
            }
        );
        // differs + --overwrite → Update in place
        assert_eq!(
            decide_write(
                "image",
                "g",
                "a",
                "1.0",
                "{\"new\":1}",
                None,
                Some("images/a"),
                Some("{\"old\":1}"),
                None,
                None,
                "images/a",
                true,
            ),
            WriteAction::Write {
                target_rel: "images/a".into(),
                full_write: false,
                regrouped_from: None,
                unchanged: false,
                soft_warn: None,
            }
        );
    }

    /// Build a minimal toolbox manifest from `(group, name, versions)` image specs and
    /// `(group, name)` pipeline specs, for the enumeration test
    fn manifest_with(
        images: &[(&str, &str, &[&str])],
        pipelines: &[(&str, &str)],
    ) -> manifest::ToolboxManifest {
        let mut image_map = HashMap::new();
        for (group, name, versions) in images {
            let versions = versions
                .iter()
                .map(|v| {
                    (
                        (*v).to_string(),
                        manifest::ImageVersion {
                            dir: String::new(),
                            build_path: "./".to_string(),
                            config_from: None,
                            config: Some(ImageRequest::new(*group, *name)),
                            network_policies_from: Vec::new(),
                            network_policies: Vec::new(),
                        },
                    )
                })
                .collect();
            image_map.insert((*name).to_string(), manifest::ImageManifest { versions });
        }
        let mut pipeline_map = HashMap::new();
        for (group, name) in pipelines {
            let versions = HashMap::from([(
                "latest".to_string(),
                manifest::PipelineVersion {
                    dir: String::new(),
                    description: String::new(),
                    images: HashMap::new(),
                    config_from: None,
                    config: Some(PipelineRequest::new(*group, *name, serde_json::json!([]))),
                },
            )]);
            pipeline_map.insert((*name).to_string(), manifest::PipelineManifest { versions });
        }
        manifest::ToolboxManifest {
            name: "tb".to_string(),
            registry: None,
            pipelines: pipeline_map,
            images: image_map,
            bundled_images: false,
            image_path_prefix: None,
        }
    }

    /// `existing_resource_ids` lists every tool's `(group, name)` once, deduping across versions and
    /// covering both images and pipelines (the refresh-all enumeration)
    #[test]
    fn existing_resource_ids_dedups_and_covers_both() {
        let m = manifest_with(
            &[
                ("static", "clamav", &["1.0", "latest"]),
                ("static", "exiftool", &["latest"]),
            ],
            &[("static", "triage")],
        );
        let (mut images, pipelines) = existing_resource_ids(&m);
        // clamav has two versions but is enumerated once; exiftool once
        images.sort();
        assert_eq!(
            images,
            vec![
                ("static".to_string(), "clamav".to_string()),
                ("static".to_string(), "exiftool".to_string()),
            ]
        );
        assert_eq!(
            pipelines,
            vec![("static".to_string(), "triage".to_string())]
        );
    }
}
