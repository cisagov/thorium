//! Build (and optionally push) the container images described by a toolbox
//!
//! This is the local stand-in for the CI pipeline that normally builds a
//! toolbox repo's images: walk `toolbox.json`, build every entry with build
//! enabled from its build context, alias the extra registry tags, and push.
//! With the images in a registry, `toolbox import` works as usual.

use colored::Colorize;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thorium::Error;

use crate::args::toolbox::BuildImagesToolbox;
use crate::handlers::container;
use crate::handlers::progress;

use super::build::{BaseImage, DEFAULT_BASE_IMAGE_ARG};

/// The slice of a toolbox.json image entry that drives a build
///
/// Deserialized standalone (not via [`super::manifest::ToolboxManifest`]) so
/// building doesn't require resolvable configs.
#[derive(Deserialize)]
struct BuildEntry {
    /// The build context to build from
    build_path: PathBuf,
    /// Whether this image should be built at all
    #[serde(default = "default_true")]
    build_image: bool,
    /// Every registry tag this image should carry
    #[serde(default)]
    image_tags: Vec<String>,
    /// This image's resolved base-image configuration (per-tool merged over global at build time)
    #[serde(default)]
    base_image: Option<BaseImage>,
}

/// The outcome of resolving an image's base-image substitution
enum BaseResolution {
    /// Apply this `(arg, image)` as a `--build-arg`
    Apply(String, String),
    /// A substitution was requested but the image opted out (`allow_override = false`)
    Withheld,
    /// No base-image substitution applies to this image
    NotApplicable,
}

/// Resolve an image's base-image substitution into a single decision
///
/// The entry's `base_image` is already the per-tool-over-global merge done at `toolbox build` time,
/// so this only layers the `--base-image` CLI escape hatch on top and applies the `allow_override`
/// gate. Returning one decision (rather than a bare `Option`) lets the caller drive both the
/// build-arg and the "withheld" notice from the same source, so they can't drift. `token`/`user`
/// are deliberately not consulted — `build-images` performs no base-registry login.
///
/// # Arguments
///
/// * `cli` - The `--base-image ARG=IMAGE` override, if given
/// * `entry` - This image's resolved `base_image` from `toolbox.json`, if any
fn resolve_base_build_arg(cli: Option<&(String, String)>, entry: Option<&BaseImage>) -> BaseResolution {
    // a substitution is "requested" if the operator passed --base-image or the entry carries its
    // own resolved image; this distinguishes a genuine opt-out (Withheld) from "nothing to do"
    let requested = cli.is_some() || entry.and_then(|base| base.image.as_ref()).is_some();
    // an image opts out of the image/image_arg substitution via allow_override; absent means allow,
    // matching the toolbox.json default so an unset entry behaves like an opted-in one
    let allow = entry.and_then(|base| base.allow_override).unwrap_or(true);
    // the opt-out gate is checked before the CLI flag on purpose: allow_override=false beats even
    // --base-image, so an image that pins its base is never silently re-based by an operator flag
    if !allow {
        return if requested {
            BaseResolution::Withheld
        } else {
            BaseResolution::NotApplicable
        };
    }
    // the CLI flag is the operator escape hatch and wins over the entry's resolved base, carrying
    // its own arg=image pair verbatim
    if let Some((arg, image)) = cli {
        return BaseResolution::Apply(arg.clone(), image.clone());
    }
    // with no CLI override, fall back to the entry's resolved image; the arg was defaulted to
    // DEFAULT_BASE_IMAGE_ARG at toolbox build time, but guard here too so a hand-edited entry with
    // an image and no arg still gets the default rather than being dropped
    if let Some(base) = entry
        && let Some(image) = base.image.clone()
    {
        let arg = base
            .image_arg
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_IMAGE_ARG.to_string());
        return BaseResolution::Apply(arg, image);
    }
    // no CLI flag and the entry has no image (e.g. token/user-only, or no base_image at all)
    BaseResolution::NotApplicable
}

/// Merge the generic `--build-arg` set with the resolved base-image override
///
/// The base override (already resolved by [`resolve_base_build_arg`]) is more specific than a
/// generic build arg, so it replaces any generic arg sharing its key.
///
/// # Arguments
///
/// * `generic` - The generic `--build-arg KEY=VALUE` pairs applied to every build
/// * `base_override` - The resolved `(arg, image)` override for this image, if any
fn resolve_build_args(
    generic: &[(String, String)],
    base_override: Option<&(String, String)>,
) -> Vec<(String, String)> {
    // start from a copy of the generic args so the caller's slice is left untouched
    let mut args = generic.to_vec();
    if let Some((key, value)) = base_override {
        // drop any generic arg sharing the override's key first, then append the override, so the
        // result never carries a duplicate key (which docker/podman would resolve unpredictably)
        args.retain(|(existing, _)| existing != key);
        args.push((key.clone(), value.clone()));
    }
    args
}

/// The default for manifest booleans that should be on unless explicitly disabled
fn default_true() -> bool {
    true
}

/// Append a tag suffix to the version of an image tag, e.g. `reg/x:1.0` + `-mybranch`
/// -> `reg/x:1.0-mybranch`
///
/// The suffix differentiates a feature-branch build from the mainline tag. It is only applied when
/// the reference has a `:version` after its final path segment (so a registry port like
/// `host:5000/img` isn't treated as a version, and a tag with no version is left untouched). A
/// digest-pinned reference (`…@sha256:…`) has no mutable version and is returned unchanged. An
/// absent or empty suffix is a no-op.
///
/// # Arguments
///
/// * `tag` - The image tag to suffix
/// * `suffix` - The suffix to append, if any
fn apply_tag_suffix(tag: &str, suffix: Option<&str>) -> String {
    // treat both an absent suffix and an explicit empty string as a no-op so a `--tag-suffix ""`
    // doesn't silently rewrite tags to themselves
    let Some(suffix) = suffix.filter(|suffix| !suffix.is_empty()) else {
        return tag.to_string();
    };
    // a digest-pinned reference (`…@sha256:…`) names an immutable content hash, not a mutable
    // version; suffixing it would produce a reference that points at nothing, so leave it as-is
    if tag.contains('@') {
        return tag.to_string();
    }
    // only the final path segment can hold a version, so scan for the version ':' starting after the
    // last '/'; this keeps a registry port like `host:5000/img` from being mistaken for a version
    let leaf_start = tag.rfind('/').map_or(0, |slash| slash + 1);
    // the version is always the tail of the reference, so when the leaf has a ':' the suffix can be
    // appended to the whole string and still land on the version; a leaf with no ':' has no version
    // to suffix and is returned untouched rather than mangling the bare name
    if tag[leaf_start..].contains(':') {
        format!("{tag}{suffix}")
    } else {
        tag.to_string()
    }
}

/// Only the parts of toolbox.json that builds care about
///
/// Deserialized as a subset on purpose: every other top-level toolbox.json field (name, registry,
/// pipelines, etc.) is intentionally ignored, so no `deny_unknown_fields` here.
#[derive(Deserialize)]
struct BuildManifest {
    /// The image entries mapped name -> version -> entry (each carries its resolved `base_image`)
    images: HashMap<String, HashMap<String, BuildEntry>>,
}

/// Build and optionally push every buildable image in a toolbox
///
/// # Arguments
///
/// * `cmd` - The build-images command that was run
pub async fn build_images(cmd: &BuildImagesToolbox) -> Result<(), Error> {
    // read the whole toolbox.json into memory; it is small and parsed in one pass
    let manifest_str = tokio::fs::read_to_string(&cmd.manifest)
        .await
        .map_err(|err| {
            Error::new(format!(
                "Failed to read toolbox manifest '{}': {err}",
                cmd.manifest.display()
            ))
        })?;
    // parse only the build-relevant subset; unknown top-level fields are ignored by design so a
    // full toolbox.json (name, registry, pipelines, …) deserializes without resolvable configs
    let manifest: BuildManifest = serde_json::from_str(&manifest_str).map_err(|err| {
        Error::new(format!(
            "Failed to parse toolbox manifest '{}': {err}",
            cmd.manifest.display()
        ))
    })?;
    // build_path values in toolbox.json are recorded relative to the manifest's own
    // directory, so resolve them against it rather than the cwd build-images runs from
    let manifest_dir = cmd.manifest.parent().unwrap_or_else(|| Path::new("."));
    // image names present in the manifest, captured before the map is consumed below so we can
    // flag requested --images that match nothing
    let available: HashSet<String> = manifest.images.keys().cloned().collect();
    // flatten the name -> version -> entry map into a flat (name, version, entry) target list,
    // narrowing by --images when given (an empty list means build everything)
    let mut targets: Vec<(String, String, BuildEntry)> = manifest
        .images
        .into_iter()
        // an explicit --image list narrows the build set to the named top-level image names
        .filter(|(name, _)| cmd.images.is_empty() || cmd.images.contains(name))
        .flat_map(|(name, versions)| {
            versions
                .into_iter()
                .map(move |(version, entry)| (name.clone(), version, entry))
        })
        .collect();
    // sort by (name, version) so the build order (and the printed log) is deterministic regardless
    // of the HashMap iteration order
    targets.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    // warn about any requested --images that matched nothing, so a typo isn't a silent no-op
    // when other names did match
    let unknown: Vec<&str> = cmd
        .images
        .iter()
        .filter(|name| !available.contains(*name))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        progress::warn(format!(
            "requested image(s) not found in the toolbox manifest: {}",
            unknown.join(", ")
        ));
    }
    // no targets is a hard error rather than a quiet success; the message distinguishes an empty
    // manifest from a --images filter that matched nothing so the user knows which to fix
    if targets.is_empty() {
        return Err(Error::new(if cmd.images.is_empty() {
            "No image entries found in the toolbox manifest".to_string()
        } else {
            format!(
                "No image entries in '{}' match the requested --images: {}",
                cmd.manifest.display(),
                cmd.images.join(", ")
            )
        }));
    }
    // warn once if the base image arg key also appears in the generic build args; the
    // base image value is more specific and wins for images that allow base overrides
    if let Some((base_key, _)) = &cmd.base_image
        && cmd.build_args.iter().any(|(key, _)| key == base_key)
    {
        progress::warn(format!(
            "build arg '{base_key}' is set by both --base-image and --build-arg; the --base-image \
             value takes precedence for images whose base_image.allow_override is true (the default)"
        ));
    }
    // token/user are pass-through for an external CI/CD pipeline; build-images does no
    // base-registry login, so note it once (rather than per image) if any target carries them
    if targets.iter().any(|(_, _, entry)| {
        entry
            .base_image
            .as_ref()
            .is_some_and(|base| base.token.is_some() || base.user.is_some())
    }) {
        progress::note(
            "[base_image] token/user are not used by build-images (it performs no base-registry \
             login); run docker/podman login yourself if a base image needs auth",
        );
    }
    // snapshot the build-behavior flags once; they apply uniformly to every image in the run
    let build_opts = container::BuildOptions {
        no_cache: cmd.no_cache,
        pull: cmd.pull,
    };
    // count of images successfully built, used to detect an all-skipped run at the end
    let mut built = 0usize;
    // labels of images whose build/push failed; collected so a single failure doesn't abort the
    // run (best-effort), then surfaced at the end. Stays empty under --exit-on-error since that
    // path returns immediately
    let mut failures: Vec<String> = Vec::new();
    for (name, version, entry) in targets {
        // human-readable identity for log lines and the failure list
        let label = format!("{name}:{version}");
        // entries can opt out of building (prebuilt images in a registry)
        if !entry.build_image {
            println!("{} {label} (build disabled)", "Skipping".bright_yellow());
            continue;
        }
        // a tag is required to anchor the build
        if entry.image_tags.is_empty() {
            println!("{} {label} (no image tags)", "Skipping".bright_yellow());
            continue;
        }
        // apply the tag suffix to every tag so a feature-branch run builds/pushes differentiated
        // tags (e.g. `:1.0-mybranch`) without colliding with the mainline tag in toolbox.json; this
        // is in-memory only and never rewrites the manifest on disk
        let tags: Vec<String> = entry
            .image_tags
            .iter()
            .map(|tag| apply_tag_suffix(tag, cmd.tag_suffix.as_deref()))
            .collect();
        // resolve the base-image substitution into one decision that drives both the notice and
        // the build-arg, so they can't disagree (CLI escape hatch over the entry's resolved
        // base_image, gated by allow_override)
        let base_override = match resolve_base_build_arg(cmd.base_image.as_ref(), entry.base_image.as_ref()) {
            BaseResolution::Apply(arg, image) => {
                // name where the override came from for "why this base?" debugging; the CLI flag
                // always wins in resolve_base_build_arg, so its presence alone identifies the source
                let source = if cmd.base_image.is_some() {
                    "--base-image"
                } else {
                    "toolbox.json"
                };
                println!(
                    "{} base image '{image}' (build-arg {arg}, from {source}) for {label}",
                    "Overriding".bright_cyan()
                );
                Some((arg, image))
            }
            BaseResolution::Withheld => {
                println!(
                    "{} base image override for {label} (allow_override = false)",
                    "Skipping".bright_yellow()
                );
                None
            }
            // either no substitution applies, or it was withheld above; either way pass no override
            BaseResolution::NotApplicable => None,
        };
        // merge this image's effective build args: the generic --build-arg set with the resolved
        // base override layered on top (the override replaces a generic arg sharing its key)
        let build_args = resolve_build_args(&cmd.build_args, base_override.as_ref());
        // resolve the build context against the manifest's directory since toolbox.json records
        // build_path relative to its own location; Path::join keeps an absolute build_path as-is
        let context = manifest_dir.join(&entry.build_path);
        println!(
            "{} {label} from '{}'",
            "Building".bright_green(),
            context.display()
        );
        // build/alias/push this image; build_one streams the runtime's output straight to the
        // terminal so long builds show live progress rather than buffering silently
        match build_one(&context, &tags, cmd.push, &build_args, build_opts).await {
            Ok(()) => {
                // report the image and how many tags it carried, noting the push when --push was set
                println!(
                    "{} {label} ({} tag{}{})",
                    "Built".bright_green(),
                    tags.len(),
                    if tags.len() == 1 { "" } else { "s" },
                    if cmd.push { ", pushed" } else { "" },
                );
                built += 1;
            }
            // --exit-on-error aborts the whole run at the first failure, leaving any later images
            // unbuilt; the failures list stays empty because we never reach the collecting arm
            Err(err) if cmd.exit_on_error => {
                return Err(Error::new(format!("Failed to build/push '{label}': {err}")));
            }
            // default best-effort behavior: log this failure and keep going so one broken image
            // doesn't block the rest; the run still exits non-zero via the failures list below
            Err(err) => {
                eprintln!("{} {label}: {err}", "Failed".bright_red());
                failures.push(label);
            }
        }
    }
    // a run where nothing built and nothing failed (everything skipped) shouldn't look like a
    // successful build
    if built == 0 && failures.is_empty() {
        println!(
            "{} no images were built (all entries were skipped)",
            "Done!".bright_yellow()
        );
    } else {
        println!(
            "\n{} {built} image{} built{}",
            "Done!".bright_green(),
            if built == 1 { "" } else { "s" },
            if cmd.push { " and pushed" } else { "" },
        );
    }
    // if any image failed under best-effort mode, list every failure and return an error so the
    // overall command exits non-zero rather than reporting a misleading success
    if !failures.is_empty() {
        eprintln!(
            "{} {} image{} failed: {}",
            "Errors:".bright_red(),
            failures.len(),
            if failures.len() == 1 { "" } else { "s" },
            failures.join(", "),
        );
        return Err(Error::new(format!("{} image(s) failed", failures.len())));
    }
    Ok(())
}

/// Build an image's primary tag, alias its extra tags, and optionally push them,
/// streaming the runtime's output to the terminal
///
/// # Arguments
///
/// * `context` - The build context, already resolved against the manifest directory
/// * `tags` - The image tags to build/alias/push (first is the primary build tag);
///   must be non-empty
/// * `push` - Whether to push every tag after building
/// * `build_args` - The `(key, value)` build args to pass to the build
/// * `opts` - The build-behavior flags (`--no-cache` / `--pull`)
///
/// # Errors
///
/// Returns the first failure across the build, the tag-aliasing, or the push; the caller can't tell
/// which phase failed from the `Result` alone (the error message names it).
async fn build_one(
    context: &Path,
    tags: &[String],
    push: bool,
    build_args: &[(String, String)],
    opts: container::BuildOptions,
) -> Result<(), Error> {
    // the first tag is the build target; the caller guarantees at least one tag by skipping
    // empty-tag entries, so indexing [0] can't panic here
    let primary_tag = &tags[0];
    // build the image exactly once under the primary tag; building per tag would rebuild identical
    // content, so the remaining tags are aliased onto this single build instead
    container::build_streamed(primary_tag, context, build_args, opts).await?;
    // alias every extra tag onto the primary build so all configured registries/tags point at the
    // same image without a rebuild
    for extra_tag in tags.iter().skip(1) {
        container::tag_streamed(primary_tag, extra_tag).await?;
    }
    // push happens only with --push, and only after every tag exists locally, so a partial alias
    // failure aborts before anything is published
    if push {
        for tag in tags {
            container::push_streamed(tag).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `(key, value)` pair as owned strings, for terse test fixtures
    fn arg(key: &str, value: &str) -> (String, String) {
        (key.to_string(), value.to_string())
    }

    /// A resolved `BaseImage` entry from an image, an optional arg, and an allow_override flag,
    /// for terse test fixtures
    fn entry(image: Option<&str>, image_arg: Option<&str>, allow_override: Option<bool>) -> BaseImage {
        BaseImage {
            image: image.map(str::to_string),
            image_arg: image_arg.map(str::to_string),
            token: None,
            user: None,
            allow_override,
        }
    }

    /// With no base image override, the generic build args pass through unchanged
    #[test]
    fn resolve_keeps_generic_without_override() {
        let generic = vec![arg("HTTP_PROXY", "p"), arg("VERSION", "1")];
        let resolved = resolve_build_args(&generic, None);
        assert_eq!(resolved, generic);
    }

    /// A resolved override is appended to the generic args
    #[test]
    fn resolve_appends_override() {
        let generic = vec![arg("VERSION", "1")];
        let base = arg("IMAGE", "ubuntu:22.04");
        let resolved = resolve_build_args(&generic, Some(&base));
        assert_eq!(resolved, vec![arg("VERSION", "1"), arg("IMAGE", "ubuntu:22.04")]);
    }

    /// The `(arg, image)` of an `Apply` outcome, or `None` for `Withheld`/`NotApplicable`
    fn applied(res: BaseResolution) -> Option<(String, String)> {
        match res {
            BaseResolution::Apply(arg, image) => Some((arg, image)),
            _ => None,
        }
    }

    /// An image that opts out (allow_override = false) withholds the override, even with a CLI flag
    #[test]
    fn base_build_arg_withheld_when_disallowed() {
        let cli = arg("IMAGE", "from-cli");
        let entry = entry(Some("from-tool"), Some("IMAGE"), Some(false));
        assert!(matches!(
            resolve_base_build_arg(Some(&cli), Some(&entry)),
            BaseResolution::Withheld
        ));
    }

    /// The CLI flag wins over the entry's resolved base image
    #[test]
    fn base_build_arg_cli_wins() {
        let cli = arg("IMAGE", "from-cli");
        let entry = entry(Some("from-tool"), Some("TOOL_ARG"), None);
        assert_eq!(
            applied(resolve_base_build_arg(Some(&cli), Some(&entry))),
            Some(arg("IMAGE", "from-cli"))
        );
    }

    /// The entry's resolved image is used with its arg when no CLI flag is given
    #[test]
    fn base_build_arg_uses_entry() {
        let entry = entry(Some("from-tool"), Some("TOOL_ARG"), None);
        assert_eq!(
            applied(resolve_base_build_arg(None, Some(&entry))),
            Some(arg("TOOL_ARG", "from-tool"))
        );
    }

    /// An entry image with no arg falls back to the built-in default
    #[test]
    fn base_build_arg_defaults_arg() {
        let entry = entry(Some("from-tool"), None, None);
        assert_eq!(
            applied(resolve_base_build_arg(None, Some(&entry))),
            Some((DEFAULT_BASE_IMAGE_ARG.to_string(), "from-tool".to_string()))
        );
    }

    /// No entry and no CLI is not applicable; a token/user-only entry is too (nothing to substitute)
    #[test]
    fn base_build_arg_not_applicable_when_unset() {
        assert!(matches!(
            resolve_base_build_arg(None, None),
            BaseResolution::NotApplicable
        ));
        let token_only = BaseImage {
            image: None,
            image_arg: None,
            token: Some("TOK".to_string()),
            user: Some("USR".to_string()),
            allow_override: None,
        };
        assert!(matches!(
            resolve_base_build_arg(None, Some(&token_only)),
            BaseResolution::NotApplicable
        ));
    }

    /// A non-empty suffix lands on the version; absent/empty leaves the tag unchanged
    #[test]
    fn apply_tag_suffix_appends_to_version() {
        assert_eq!(apply_tag_suffix("reg/x:1.0", Some("-mybranch")), "reg/x:1.0-mybranch");
        assert_eq!(apply_tag_suffix("reg/x:1.0", None), "reg/x:1.0");
        assert_eq!(apply_tag_suffix("reg/x:1.0", Some("")), "reg/x:1.0");
    }

    /// A digest-pinned reference and a tag with no version are left unchanged; a registry port is
    /// not mistaken for a version
    #[test]
    fn apply_tag_suffix_edges() {
        // digest reference has no mutable version
        assert_eq!(
            apply_tag_suffix("reg/x@sha256:abc123", Some("-br")),
            "reg/x@sha256:abc123"
        );
        // no version component -> untouched (don't mangle the name)
        assert_eq!(apply_tag_suffix("reg/x", Some("-br")), "reg/x");
        // a registry port is not a version, but the real version still gets the suffix
        assert_eq!(apply_tag_suffix("host:5000/x", Some("-br")), "host:5000/x");
        assert_eq!(apply_tag_suffix("host:5000/x:1.0", Some("-br")), "host:5000/x:1.0-br");
    }

    /// On a key collision the override replaces the generic arg rather than producing a
    /// duplicate key; a different key coexists
    #[test]
    fn resolve_override_replaces_generic_on_collision() {
        let generic = vec![arg("IMAGE", "from-build-arg"), arg("VERSION", "1")];
        let base = arg("IMAGE", "from-base-image");
        let resolved = resolve_build_args(&generic, Some(&base));
        // exactly one IMAGE entry, and it carries the override value
        let images: Vec<&(String, String)> =
            resolved.iter().filter(|(key, _)| key == "IMAGE").collect();
        assert_eq!(images, vec![&arg("IMAGE", "from-base-image")]);
        // a build arg with a different key is left in place
        assert!(resolved.contains(&arg("VERSION", "1")));
    }
}
