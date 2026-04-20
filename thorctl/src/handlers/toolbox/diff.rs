//! Diff an on-disk toolbox against what a Thorium instance has imported
//!
//! Renders what an import of the toolbox would change in git-diff style:
//! resources present on both sides but differing show unified hunks, while a
//! resource present on only one side (only in the toolbox, or only in the
//! instance's groups) is noted on a single line so a fully-new or fully-extra
//! resource doesn't drown the real diff in full-body adds or deletes.
//! Comparison happens on the normalized request forms (the same shapes the
//! merge editor uses), so server-only fields like creators, bans, and
//! timestamps never show up as drift.
//!
//! The toolbox runs through the same pre-processing an import would (structural and
//! group-coherence validation, the group override, and non-interactive collision
//! resolution), so the preview reflects what an import would actually create/change
//! rather than the raw manifest.

use colored::Colorize;
use similar::TextDiff;
use std::collections::HashSet;
use thorium::models::{ImageRequest, PipelineRequest};
use thorium::{CtlConf, Error, Thorium};

use super::manifest::ToolboxManifest;
use super::{build, collisions, import, policies, shared};
use crate::args::toolbox::{BuildToolbox, DiffToolbox, ManifestLocation};
use crate::handlers::imports::categorize;
use crate::handlers::imports::merge::{MergeableImage, MergeablePipeline};
use crate::handlers::progress::Bar;
use crate::utils;
use crate::utils::images::list_all_images;
use crate::utils::pipelines::list_all_pipelines;

/// Running totals for the trailing summary line
#[derive(Default)]
struct DiffStats {
    /// Images/pipelines that exist on both sides but differ
    changed: usize,
    /// Images/pipelines/policies only present in the toolbox (an import would create them)
    only_toolbox: usize,
    /// Resources in the toolbox's groups that the toolbox doesn't name (informational only —
    /// an import never deletes them)
    only_instance: usize,
    /// Resources that match exactly
    unchanged: usize,
    /// Network policies that exist in the instance but differ from the toolbox's definition
    policy_changed: usize,
}

impl DiffStats {
    /// Whether an import of this toolbox would actually change anything — drives `--exit-code`
    ///
    /// Instance-only resources are deliberately excluded: an import only creates/updates, never
    /// deletes, so the instance merely having extra resources is not an import-actionable difference.
    fn any_actionable(&self) -> bool {
        self.changed > 0 || self.only_toolbox > 0 || self.policy_changed > 0
    }
}

/// Print one resource's diff with git-style file headers
///
/// # Arguments
///
/// * `old_label` - The old/left-side header (or "/dev/null" for new resources)
/// * `new_label` - The new/right-side header (or "/dev/null" for deletions)
/// * `old_text` - The old/left-side rendered text (image/pipeline/policy)
/// * `new_text` - The new/right-side rendered text
fn print_diff(old_label: &str, new_label: &str, old_text: &str, new_text: &str) {
    // line-granular diff so each changed YAML field reads as its own +/- pair
    let diff = TextDiff::from_lines(old_text, new_text);
    // render a git-style unified hunk with three lines of surrounding context
    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(old_label, new_label)
        .to_string();
    // colorize after the fact: the unified renderer emits plain text, so prefixes are the
    // only signal of each line's role
    for line in unified.lines() {
        // colorize like git: headers bold, adds green, removes red, hunks cyan
        if line.starts_with("---") || line.starts_with("+++") {
            println!("{}", line.bold());
        } else if line.starts_with('@') {
            println!("{}", line.cyan());
        } else if line.starts_with('+') {
            println!("{}", line.green());
        } else if line.starts_with('-') {
            println!("{}", line.red());
        } else {
            println!("{line}");
        }
    }
    println!();
}

/// Print a compact one-line note that a resource lives on only one side of the diff
///
/// Used for both toolbox-only and instance-only resources so a fully-new (or fully-extra)
/// resource is reported as a single line rather than a full-body add or delete.
///
/// # Arguments
///
/// * `root` - The side the resource is on: `"toolbox"` or the instance host (`host[:port]`)
/// * `group` - The resource's group
/// * `name` - The resource's name
/// * `kind` - The resource kind (`"image"` / `"pipeline"`)
/// * `absent_from` - The side it's missing from, for the trailing "not in …" clause
fn print_only_in(root: &str, group: &str, name: &str, kind: &str, absent_from: &str) {
    println!(
        "{} {root}/{group}/{name} ({kind}) — not in {absent_from}",
        "only in".yellow()
    );
}

/// Render a resource present on both sides: identical is silent, differing shows a hunk
///
/// # Arguments
///
/// * `stats` - The running diff totals to update
/// * `instance_label` - The instance-side header for the changed resource
/// * `toolbox_path` - The toolbox-side header
/// * `old_text` - The instance-side normalized YAML
/// * `new_text` - The toolbox-side normalized YAML
fn render_changed(
    stats: &mut DiffStats,
    instance_label: &str,
    toolbox_path: &str,
    old_text: &str,
    new_text: &str,
) {
    if old_text == new_text {
        // present on both sides and identical: nothing to show
        stats.unchanged += 1;
    } else {
        // present on both sides but changed: show the hunk
        stats.changed += 1;
        print_diff(instance_label, toolbox_path, old_text, new_text);
    }
}

/// Serialize an image request to the normalized YAML used for comparison
///
/// Uses a canonical (sorted-key) form so reordered map fields (e.g. `env`) don't
/// show up as drift.
///
/// # Arguments
///
/// * `request` - The image request to serialize
fn image_yaml(request: &ImageRequest) -> Result<String, Error> {
    utils::canonical_yaml(&MergeableImage::from(request.clone()))
        .map_err(|err| Error::new(format!("Failed to serialize image for diff: {err}")))
}

/// Serialize a pipeline request to the normalized YAML used for comparison
///
/// Uses a canonical (sorted-key) form so reordered map fields (e.g. `triggers`)
/// don't show up as drift.
///
/// # Arguments
///
/// * `request` - The pipeline request to serialize
fn pipeline_yaml(request: &PipelineRequest) -> Result<String, Error> {
    utils::canonical_yaml(&MergeablePipeline::from(request.clone()))
        .map_err(|err| Error::new(format!("Failed to serialize pipeline for diff: {err}")))
}

/// Whether a group-listing error just means the group isn't visible to us
///
/// A 404 (group absent) or 403 (not a member) means the group simply holds no
/// resources we can see, which is safely treated as empty. Any other status (auth,
/// transient 5xx) is a genuine failure the caller should surface instead of
/// silently under-reporting instance-only drift.
///
/// Note the asymmetry with `categorize`, which treats any non-404 status (including a 403) as an
/// error: a *named* toolbox resource in an inaccessible group is a real problem, whereas here an
/// unlistable group merely can't contribute instance-only entries.
///
/// # Arguments
///
/// * `err` - The error returned while listing a group's images or pipelines
fn is_invisible_group(err: &Error) -> bool {
    matches!(err.status().map(|status| status.as_u16()), Some(403 | 404))
}

/// Load the toolbox manifest from a file, URL, or repo directory
///
/// Directories are built in-memory with image urls preserved so the diff
/// reflects the configs exactly as they sit on disk.
///
/// # Arguments
///
/// * `location` - Where the toolbox lives
async fn load_manifest(location: &ManifestLocation) -> Result<(ToolboxManifest, Bar), Error> {
    // a directory is a toolbox repo checkout; build it in memory
    if let ManifestLocation::Path(path) = location
        && path.is_dir()
    {
        let config = path.join("config.toml");
        if !config.exists() {
            return Err(Error::new(format!(
                "'{}' has no config.toml; point at a toolbox repo root or a toolbox.json",
                path.display()
            )));
        }
        let build_cmd = BuildToolbox {
            config,
            use_image_path: false,
            // never written; build_in_memory ignores the output path
            output: Some(path.join("toolbox.json")),
            path: Some(path.clone()),
            // diff compares the on-disk toolbox as-is, so no tag suffix is applied
            tag_suffix: None,
        };
        // build walks the tree with synchronous std::fs, so run it off the async runtime
        let value = tokio::task::spawn_blocking(move || build::build_in_memory(&build_cmd))
            .await
            .map_err(|err| Error::new(format!("Toolbox build task failed: {err}")))??;
        let manifest: ToolboxManifest = serde_json::from_value(value)
            .map_err(|err| Error::new(format!("Failed to assemble toolbox manifest: {err}")))?;
        let progress = Bar::new(
            "",
            "Diffing toolbox",
            crate::handlers::progress::BarKind::Timer,
        );
        return Ok((manifest, progress));
    }
    shared::get_manifest(location).await
}

/// Reduce a Thorium API url to a `host[:port]` label for diff output
///
/// Strips the scheme and any path so the diff names the specific instance by its domain (and
/// explicit port, when one is given) rather than a generic "instance". Falls back to a manual
/// scheme/trailing-path strip if the url doesn't parse, and to the raw value if even that yields
/// nothing.
///
/// # Arguments
///
/// * `api_url` - The configured Thorium API url (`conf.keys.api`)
fn instance_host(api_url: &str) -> String {
    // prefer a real parse so userinfo/path/query are dropped and the port is explicit
    if let Ok(url) = url::Url::parse(api_url)
        && let Some(host) = url.host_str()
    {
        return match url.port() {
            Some(port) => format!("{host}:{port}"),
            None => host.to_string(),
        };
    }
    // fallback: drop a `scheme://` prefix and anything from the first path slash
    let without_scheme = api_url
        .split_once("://")
        .map_or(api_url, |(_scheme, rest)| rest);
    // drop any userinfo (user:pass@) before the host
    let after_userinfo = without_scheme
        .rsplit_once('@')
        .map_or(without_scheme, |(_userinfo, rest)| rest);
    // drop any path, query, or fragment after the host[:port]
    let host = after_userinfo
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_userinfo);
    if host.is_empty() {
        api_url.to_string()
    } else {
        host.to_string()
    }
}

/// Diff a toolbox against the instance and print the result
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (provides the instance url shown in the diff)
/// * `cmd` - The toolbox diff command that was run
pub async fn diff(thorium: Thorium, conf: &CtlConf, cmd: &DiffToolbox) -> Result<bool, Error> {
    // the instance's host[:port], used in place of a generic "instance" in the diff output
    let instance = instance_host(&conf.keys.api);
    let (mut manifest, progress) = load_manifest(&cmd.manifest).await?;
    shared::resolve_manifest_configs(&mut manifest, &progress).await?;
    // run the SAME pre-processing an import would, so the preview reflects what import actually
    // does: drop intrinsically-invalid versions, apply the group override, drop group-incoherent
    // pipelines, then resolve (group, name) collisions non-interactively (skip+warn, since diff
    // can't prompt), then re-check coherence after any renames.
    shared::warn_dropped(&manifest.validate_structural(), &progress);
    let sources = manifest.capture_source_groups();
    if let Some(group_override) = &cmd.group_override {
        manifest = manifest.override_group(group_override);
    }
    shared::warn_dropped(&manifest.validate_group_coherence(), &progress);
    // diff is read-only and can't prompt, so collisions take the non-interactive skip+warn path;
    // the returned image renames don't matter here (diff pushes no bundled images)
    collisions::resolve_collisions(&mut manifest, &sources, false, &progress)?;
    shared::warn_dropped(&manifest.validate_group_coherence(), &progress);
    // categorize everything against the live instance
    let images = categorize::categorize_images(
        &thorium,
        import::flatten_manifest_images(&manifest),
        &progress,
    )
    .await?;
    let pipelines = categorize::categorize_pipelines(
        &thorium,
        import::flatten_manifest_pipelines(&manifest),
        &progress,
    )
    .await?;
    let policy_plan = policies::categorize_policies(
        &thorium,
        policies::collect_policies(&manifest, cmd.group_override.as_deref(), &progress),
        // diff is read-only: always categorize as warn-only so it never plans a mutation
        false,
        &progress,
    )
    .await?;
    // the groups the toolbox targets, where we'll look for resources it doesn't name
    let groups = manifest.groups();
    // index the toolbox's image identities so instance-only images can be detected
    let toolbox_images: HashSet<(String, String)> = images
        .iter()
        .map(|img| (img.request.group.clone(), img.request.name.clone()))
        .collect();
    // index the toolbox's pipeline identities for the same instance-only detection
    let toolbox_pipelines: HashSet<(String, String)> = pipelines
        .iter()
        .map(|pipe| (pipe.request.group.clone(), pipe.request.name.clone()))
        .collect();
    // list every group's images and pipelines concurrently; a group we can't see
    // (absent or non-member) can't hold extra resources, so its listing is treated
    // as empty, but a genuine failure is surfaced rather than hiding drift
    let thorium_ref = &thorium;
    let progress_ref = &progress;
    let group_listings = futures::future::join_all(groups.iter().map(|group| async move {
        // list this group's images, warning on a real failure before treating it empty
        let images = match list_all_images(thorium_ref, group).await {
            Ok(images) => images,
            Err(err) if is_invisible_group(&err) => Vec::new(),
            Err(err) => {
                progress_ref.warning(format!(
                    "Could not list images in group '{group}': {err} — treating it as empty; any \
                     instance-only images there won't appear in this diff"
                ));
                Vec::new()
            }
        };
        // list this group's pipelines with the same visible-vs-genuine error handling
        let pipelines = match list_all_pipelines(thorium_ref, group).await {
            Ok(pipelines) => pipelines,
            Err(err) if is_invisible_group(&err) => Vec::new(),
            Err(err) => {
                progress_ref.warning(format!(
                    "Could not list pipelines in group '{group}': {err} — treating it as empty; any \
                     instance-only pipelines there won't appear in this diff"
                ));
                Vec::new()
            }
        };
        (images, pipelines)
    }))
    .await;
    let mut instance_only_images = Vec::new();
    let mut instance_only_pipelines = Vec::new();
    // fold every group's listing down to the resources the toolbox doesn't name; a listed
    // resource whose identity is absent from the toolbox index exists only on the instance
    for (group_images, group_pipelines) in group_listings {
        // keep only images the toolbox never names (matched on the (group, name) identity)
        for image in group_images {
            if !toolbox_images.contains(&(image.group.clone(), image.name.clone())) {
                instance_only_images.push(image);
            }
        }
        // keep only pipelines the toolbox never names, by the same identity comparison
        for pipeline in group_pipelines {
            if !toolbox_pipelines.contains(&(pipeline.group.clone(), pipeline.name.clone())) {
                instance_only_pipelines.push(pipeline);
            }
        }
    }
    // tear down the progress bar before any diff output so the rendered hunks aren't
    // interleaved with a live spinner on the same terminal
    progress.finish_and_clear();
    // accumulate the per-resource verdicts here so the trailing summary and the --exit-code
    // decision both read from one set of totals
    let mut stats = DiffStats::default();
    // images first: a toolbox image either has no instance counterpart (only-in note) or one
    // to diff against
    for img in &images {
        match &img.existing {
            // only in the toolbox: a compact note instead of a full-body new-file diff
            None => {
                stats.only_toolbox += 1;
                print_only_in(
                    "toolbox",
                    &img.request.group,
                    &img.request.name,
                    "image",
                    &instance,
                );
            }
            // present on both sides: normalize the live image through its request form
            // (so server-only fields don't show as drift) and diff
            Some(existing) => {
                let label = format!("{}/{}", img.request.group, img.request.name);
                let new_text = image_yaml(&img.request)?;
                let old_text = image_yaml(&ImageRequest::from(existing.clone()))?;
                render_changed(
                    &mut stats,
                    &format!("{instance}/{label} (image)"),
                    // use the config name on both sides so a differing manifest key doesn't read
                    // like a rename
                    &format!("toolbox/images/{}", img.request.name),
                    &old_text,
                    &new_text,
                );
            }
        }
    }
    // pipelines next, mirroring the image handling exactly
    for pipe in &pipelines {
        match &pipe.existing {
            // only in the toolbox: a compact note instead of a full-body new-file diff
            None => {
                stats.only_toolbox += 1;
                print_only_in(
                    "toolbox",
                    &pipe.request.group,
                    &pipe.request.name,
                    "pipeline",
                    &instance,
                );
            }
            Some(existing) => {
                let label = format!("{}/{}", pipe.request.group, pipe.request.name);
                let new_text = pipeline_yaml(&pipe.request)?;
                let old_text = pipeline_yaml(&PipelineRequest::from(existing.clone()))?;
                render_changed(
                    &mut stats,
                    &format!("{instance}/{label} (pipeline)"),
                    // use the config name on both sides (see the image header above)
                    &format!("toolbox/pipelines/{}", pipe.request.name),
                    &old_text,
                    &new_text,
                );
            }
        }
    }
    // policies: toolbox-only ones are noted on a single line, mismatches as summary lines
    // (the instance side is authoritative and never updated by imports)
    // a brand-new policy is an import-actionable create, so it counts toward only_toolbox
    for policy in &policy_plan.new {
        stats.only_toolbox += 1;
        println!(
            "{} toolbox/network-policies/{} (network policy) — not in {instance}",
            "only in".yellow(),
            policy.name
        );
    }
    for mismatch in &policy_plan.mismatched {
        // a mismatch counts once toward the policy total even when it prints two note lines
        // (rule/flag drift and a coverage gap are distinct reasons for the same policy)
        stats.policy_changed += 1;
        // attribute the difference to the instance's existing policy, splitting rule/flag
        // drift from a coverage gap so the suggested action matches each case
        let name = &mismatch.name;
        if !mismatch.drift.is_empty() {
            println!(
                "{} network policy '{name}' already exists on {instance} with a different \
                 definition (differs: [{}]); use --update-network-policy on import to overwrite it\n",
                "note:".cyan(),
                mismatch.drift.join(", ")
            );
        }
        if !mismatch.missing_groups.is_empty() {
            println!(
                "{} network policy '{name}' exists on {instance} but not in group(s) [{}]; \
                 use --update-network-policy on import to add those group(s)\n",
                "note:".cyan(),
                mismatch.missing_groups.join(", ")
            );
        }
    }
    // policy "unchanged" folds into the same total as images/pipelines
    stats.unchanged += policy_plan.unchanged;
    // resources only in the instance are listed compactly: a partial toolbox
    // (one pipeline out of a big group) would otherwise drown the real diff
    // in full-body deletions
    // instance-only images: a single note each (these never affect --exit-code; an import
    // never deletes them)
    for image in &instance_only_images {
        stats.only_instance += 1;
        print_only_in(
            &instance,
            &image.group,
            &image.name,
            "image",
            "this toolbox",
        );
    }
    // instance-only pipelines: the same compact treatment
    for pipeline in &instance_only_pipelines {
        stats.only_instance += 1;
        print_only_in(
            &instance,
            &pipeline.group,
            &pipeline.name,
            "pipeline",
            "this toolbox",
        );
    }
    // separate the instance-only block from the trailing summary with a blank line
    if stats.only_instance > 0 {
        println!();
    }
    // trailing summary like git's diffstat footer; network-policy differences are reported
    // separately (they share none of the image/pipeline counters except "unchanged")
    let policy_note = if stats.policy_changed > 0 {
        format!(
            ", {} network polic{} differ",
            stats.policy_changed,
            if stats.policy_changed == 1 {
                "y"
            } else {
                "ies"
            }
        )
    } else {
        String::new()
    };
    println!(
        "{} changed, {} only in toolbox, {} only in {instance}, {} unchanged{policy_note}",
        stats.changed, stats.only_toolbox, stats.only_instance, stats.unchanged
    );
    // git diff --exit-code semantics: exit non-zero only on import-actionable drift (instance-only
    // resources are excluded — see DiffStats::any_actionable). The actual process exit happens at
    // the dispatch boundary so this function returns normally and its resources drop first.
    Ok(cmd.exit_code && stats.any_actionable())
}

/// Unit tests for the host-derivation helper that names the instance in diff output
#[cfg(test)]
mod tests {
    use super::*;

    /// A full https url is reduced to its host with an explicit port preserved and the
    /// default port dropped
    #[test]
    fn instance_host_strips_scheme_keeps_explicit_port() {
        assert_eq!(
            instance_host("https://thorium.example.com:8443"),
            "thorium.example.com:8443"
        );
        assert_eq!(
            instance_host("https://thorium.example.com"),
            "thorium.example.com"
        );
    }

    /// A path/query after the host is dropped
    #[test]
    fn instance_host_drops_path() {
        assert_eq!(instance_host("http://10.0.0.5:9000/api/"), "10.0.0.5:9000");
    }

    /// A value with no scheme falls back to a manual strip and keeps host[:port]
    #[test]
    fn instance_host_handles_no_scheme() {
        assert_eq!(
            instance_host("thorium.example.com:8443"),
            "thorium.example.com:8443"
        );
        assert_eq!(
            instance_host("thorium.example.com/api"),
            "thorium.example.com"
        );
    }

    /// The no-parse fallback also drops userinfo and any query/fragment
    #[test]
    fn instance_host_fallback_strips_userinfo_and_query() {
        // no scheme + userinfo + query (won't strict-parse as a URL)
        assert_eq!(
            instance_host("user:pw@thorium.example.com:8443?x=1"),
            "thorium.example.com:8443"
        );
        assert_eq!(instance_host("svc@host/api#frag"), "host");
    }
}
