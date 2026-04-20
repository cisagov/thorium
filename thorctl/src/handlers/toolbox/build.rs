//! Builds a toolbox manifest from a directory of image and pipeline manifests

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use thorium::Error;
use thorium::models::{ImageRequest, ImageScaler, PipelineRequest};
use url::Url;
use walkdir::WalkDir;

use crate::args::toolbox::BuildToolbox;
use crate::handlers::progress;

// ─── TOML Input Models ─────────────────────────────────────────────────────

/// The `config.toml` at a toolbox repo's root — the toolbox-wide settings that
/// `toolbox build` copies into the generated `toolbox.json`
#[derive(Deserialize)]
pub(crate) struct ToolboxConfig {
    /// Human-readable toolbox name
    pub(super) name: String,
    /// Primary container registry images are tagged/pushed under
    /// (e.g. `ghcr.io/org/repo`); the base of every built image's tag
    ///
    /// Optional: when unset (and no `registries` are given), no tags are derived
    /// and each image's tag is taken from the `image` url in its own config. A K8s
    /// image with neither a registry nor an `image` url is unschedulable and fails
    /// the build.
    #[serde(default)]
    pub(super) registry: Option<String>,
    /// Extra registries to additionally tag each image for (multi-registry publish)
    #[serde(default)]
    pub(super) registries: Vec<String>,
    /// Whether the toolbox ships image tarballs alongside its configs for offline
    /// transfer (an import then loads + pushes them); sets `bundled_images` in toolbox.json
    #[serde(default)]
    pub(super) bundled_images: bool,
    /// Default registry base path bundled images are pushed under on import, used when
    /// `--image-path-prefix` isn't given (`<image_path_prefix>/<group>/<name>:<tag>`)
    #[serde(default)]
    pub(super) image_path_prefix: Option<String>,
    /// Directory (relative to the toolbox root) `export` writes image tool directories under;
    /// `None` means the default `images`. Only affects where `export` places files, not the
    /// recursive crawl `build` does.
    #[serde(default)]
    pub(super) export_image_path: Option<String>,
    /// Directory (relative to the toolbox root) `export` writes pipeline tool directories under;
    /// `None` means the default `pipelines`.
    #[serde(default)]
    pub(super) export_pipeline_path: Option<String>,
    /// The toolbox-wide default base-image configuration; per-tool `[base_image]` tables
    /// override it field by field at build time
    #[serde(default)]
    pub(super) base_image: Option<BaseImage>,
}

/// Base-image configuration recorded in `config.toml` (toolbox-wide default) and per-tool
/// `manifest.toml`, consolidated into one table
///
/// At build time the per-tool table is merged over the global one (field by field) and the result
/// is written onto each image entry in `toolbox.json`, so `build-images` (or any CI/CD consumer)
/// can pin a tool's base image and authenticate its pull. Overriding the base requires a
/// parameterized Dockerfile (`ARG <image_arg>` / `FROM ${<image_arg>}`).
///
/// `token`/`user` are opaque pass-through strings (intended to name CI/CD pipeline variables): they
/// are never resolved or manipulated by thorctl and are not used by the `build-images` subcommand —
/// they exist for an external CI/CD pipeline to log in to the base registry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BaseImage {
    /// The base image to build with (overrides the Dockerfile `FROM`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// The build-arg the Dockerfile reads (`FROM ${image_arg}`)
    ///
    /// Falls back to the global table's `image_arg`, then to [`DEFAULT_BASE_IMAGE_ARG`], when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_arg: Option<String>,
    /// The name of a CI/CD variable holding the base-registry token (pass-through; unused by
    /// `build-images`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// The name of a CI/CD variable holding the base-registry user (pass-through; unused by
    /// `build-images`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Whether the `image`/`image_arg` substitution applies to this image (default `true`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_override: Option<bool>,
}

/// The build-arg a base-image override uses when neither the per-tool nor the global `[base_image]`
/// names one
pub(crate) const DEFAULT_BASE_IMAGE_ARG: &str = "IMAGE";

/// Read and parse a toolbox `config.toml`
///
/// Shared by `toolbox build` (the walk root's config) and `toolbox export`
/// (`--config`, which reuses an existing toolbox's settings).
///
/// # Arguments
///
/// * `path` - The path to the `config.toml`
pub(super) fn load_config(path: &Path) -> Result<ToolboxConfig, Error> {
    // read the whole config; a missing/unreadable config.toml is fatal since build can't
    // proceed without the toolbox name and registry settings
    let config_str = std::fs::read_to_string(path).map_err(|e| {
        Error::new(format!(
            "Failed to read config file '{}': {e}",
            path.display()
        ))
    })?;
    // parse the TOML into the typed config; surface the path so a syntax error is locatable
    toml::from_str(&config_str).map_err(|e| {
        Error::new(format!(
            "Failed to parse config TOML '{}': {e}",
            path.display()
        ))
    })
}

/// A single image or pipeline `manifest.toml` — one per tool/pipeline directory
/// in a toolbox repo. `toolbox build` reads these (plus their JSON configs) to
/// assemble `toolbox.json`.
#[derive(Deserialize)]
pub(crate) struct ManifestToml {
    /// The resource's name — the top-level key for this entry in `toolbox.json`
    name: String,
    /// Whether this manifest describes an `image` or a `pipeline`
    #[serde(rename = "type")]
    manifest_type: ManifestType,
    /// The version label for this entry (defaults to `latest`)
    #[serde(default = "default_version")]
    version: String,
    /// (images) A repo-style container image path used as the tag leaf only when
    /// `toolbox build --use-image-path` is set (`<registry>/<image_name>:<version>`);
    /// otherwise tags use the tool `name`.
    image_name: Option<String>,
    /// (images) The real registry url an export captured for this image
    ///
    /// `toolbox export` records the url the image actually lives at here. When the
    /// image isn't built locally (`build = false`), the build uses this verbatim
    /// instead of a `registry + image_name`-derived guess, so an exported toolbox
    /// stays importable and deterministic. Ignored once `build = true` (the user
    /// added a build context and CI will push to the derived path).
    #[serde(default)]
    exported_image_path: Option<String>,
    /// The docker build context, relative to this manifest's directory (defaults to `./`)
    build_path: Option<String>,
    /// (images) Whether CI should build this image; `false` means it is already
    /// published to a registry and should only be referenced
    #[serde(default = "default_true")]
    build: bool,
    /// Markdown description; a `description.md` beside the manifest overrides this
    description: Option<String>,
    /// (pipelines) The images this pipeline runs, mapped image name -> version
    #[serde(default)]
    images: Option<HashMap<String, PipelineImageToml>>,
    /// Path (relative to the manifest) or URL to this resource's JSON config — the
    /// Thorium image/pipeline request that gets embedded into `toolbox.json`
    config_from: Option<String>,
    /// (images) Network policy definitions to bundle with the image; each entry is a
    /// local JSON file (relative to the manifest) or a URL
    #[serde(default)]
    network_policies_from: Option<Vec<String>>,
    /// (images) Reuse another toolbox image's container image instead of building or
    /// pinning one of this image's own — only the runtime config (args/env) differs
    ///
    /// Forces `build = false`; `build_path`, `[base_image]`, and
    /// `exported_image_path` have no effect. Resolved at build time to the referenced
    /// image's container url, which takes precedence over any `exported_image_path` or
    /// config url. An `image_from` that doesn't resolve to a known toolbox image (a
    /// missing target, a cycle, or a target with no container image) fails the build.
    #[serde(default)]
    image_from: Option<ImageFromToml>,
    /// (images) This image's base-image configuration
    ///
    /// Merged over the toolbox-wide `[base_image]` in `config.toml` (per-tool fields win) at
    /// build time. The `image`/`image_arg` substitution is gated by `allow_override` and overridden
    /// by the `--base-image` CLI flag; `token`/`user` are pass-through. Requires a parameterized
    /// Dockerfile for the image substitution.
    #[serde(default)]
    base_image: Option<BaseImage>,
}

/// An `image_from` reference: the toolbox image whose container image to reuse
#[derive(Deserialize)]
struct ImageFromToml {
    /// The name of the image whose container image to reuse
    name: String,
    /// The version of that image (defaults to "latest")
    #[serde(default = "default_version")]
    version: String,
}

/// Which kind of resource a `manifest.toml` describes
#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ManifestType {
    /// A container image
    Image,
    /// A pipeline composed of images
    Pipeline,
}

/// A pipeline's reference to one of its images in `manifest.toml`
#[derive(Deserialize)]
struct PipelineImageToml {
    /// The image version this pipeline expects
    version: String,
}

/// The default image/pipeline version when a manifest omits one
fn default_version() -> String {
    "latest".to_string()
}

/// The serde default for [`ManifestToml::build`]: an image is built unless its manifest
/// explicitly sets `build = false`
fn default_true() -> bool {
    true
}

// ─── JSON Output Models ────────────────────────────────────────────────────

/// The assembled `toolbox.json` — the importable manifest `toolbox build` writes
/// (name -> version -> entry, plus the toolbox-wide settings)
#[derive(Serialize)]
struct BuildOutput {
    /// Pipeline entries, mapped name -> version -> details
    pipelines: HashMap<String, HashMap<String, BuildPipelineVersion>>,
    /// Image entries, mapped name -> version -> details
    images: HashMap<String, HashMap<String, BuildImageVersion>>,
    /// The toolbox name (from config.toml)
    name: String,
    /// The primary registry (from config.toml), omitted when unset
    #[serde(skip_serializing_if = "Option::is_none")]
    registry: Option<String>,
    /// Any additional registries (from config.toml)
    registries: Vec<String>,
    /// Whether image tarballs are bundled alongside the configs
    bundled_images: bool,
    /// Default registry base path bundled images push under on import
    #[serde(skip_serializing_if = "Option::is_none")]
    image_path_prefix: Option<String>,
    /// The configured `export` image-layout dir (from config.toml), echoed for reference; omitted
    /// when unset. Affects only where `export` places files, not what import reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    export_image_path: Option<String>,
    /// The configured `export` pipeline layout dir (from config.toml), echoed for reference
    #[serde(skip_serializing_if = "Option::is_none")]
    export_pipeline_path: Option<String>,
    /// The raw toolbox-wide `[base_image]` (from config.toml), for reference; each image entry
    /// carries the resolved value
    #[serde(skip_serializing_if = "Option::is_none")]
    base_image: Option<BaseImage>,
}

/// One image version entry in `toolbox.json`
#[derive(Serialize)]
struct BuildImageVersion {
    /// The tool directory (where this image's `manifest.toml` and any bundled tarball live),
    /// relative to the directory holding this toolbox.json. `import` resolves a bundled image's
    /// tarball from here; it is distinct from `build_path` (the docker build context, which may
    /// point elsewhere) though the two coincide for the common `build_path = "./"`.
    dir: String,
    /// The docker build context, relative to the directory holding this toolbox.json
    build_path: String,
    /// Whether CI should build this image (vs. reference an already-published one)
    build_image: bool,
    /// Every registry tag this image should be built/pushed as
    image_tags: Vec<String>,
    /// URL the image config can be fetched from (when not embedded inline)
    #[serde(skip_serializing_if = "Option::is_none")]
    config_from: Option<String>,
    /// The embedded Thorium image config (request JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<serde_json::Value>,
    /// URLs to fetch network policy definitions from at import time
    #[serde(skip_serializing_if = "Vec::is_empty")]
    network_policies_from: Vec<String>,
    /// Network policy definitions bundled with this image
    #[serde(skip_serializing_if = "Vec::is_empty")]
    network_policies: Vec<serde_json::Value>,
    /// The resolved base-image configuration for this image (per-tool merged over global),
    /// consumed at build-images time
    #[serde(skip_serializing_if = "Option::is_none")]
    base_image: Option<BaseImage>,
}

/// One pipeline version entry in `toolbox.json`
#[derive(Serialize)]
struct BuildPipelineVersion {
    /// The tool directory (where this pipeline's `manifest.toml` lives), relative to the directory
    /// holding this `toolbox.json`. Mirrors an image's `dir` so `export` can find where a pipeline
    /// already lives and update it in place instead of writing a duplicate at the default layout.
    dir: String,
    /// The pipeline description (toolbox-facing; sourced from the manifest/description.md)
    description: String,
    /// The images this pipeline runs, mapped name -> version
    images: HashMap<String, PipelineImageOutput>,
    /// URL the pipeline config can be fetched from (when not embedded inline)
    #[serde(skip_serializing_if = "Option::is_none")]
    config_from: Option<String>,
    /// The embedded Thorium pipeline config (request JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<serde_json::Value>,
}

/// A pipeline's image reference in `toolbox.json`
#[derive(Serialize)]
struct PipelineImageOutput {
    /// The image version this pipeline expects
    version: String,
}

// ─── Build Logic ───────────────────────────────────────────────────────────

/// A resolved `config_from` reference: either the inline config value or a URL to
/// fetch it from at import time
struct LoadedConfig {
    /// The parsed config value when it lives on disk (or an empty object default)
    value: Option<serde_json::Value>,
    /// The URL to resolve the config from at import time, when `config_from` is a URL
    url: Option<String>,
}

/// Resolves a `config_from` reference into an inline value or an import-time URL
///
/// A `config_from` that parses as a URL is left as a URL for import to fetch; a
/// relative path is read and parsed from disk; absence yields an empty config object.
///
/// # Arguments
///
/// * `root` - The manifest's directory, that relative `config_from` paths resolve against
/// * `config_from` - The manifest's optional config reference (path or URL)
fn load_json_config(root: &Path, config_from: Option<&str>) -> Result<LoadedConfig, Error> {
    match config_from {
        // a config_from that parses as a URL is deferred: the body isn't read at build time,
        // only carried through for the importer to fetch (so build needs no network access)
        Some(config_from) if Url::parse(config_from).is_ok() => Ok(LoadedConfig {
            value: None,
            url: Some(config_from.to_string()),
        }),
        // otherwise it's a path relative to the manifest's directory: read and embed it now
        Some(config_from) => {
            // resolve the path against the manifest dir so configs are referenced relative to
            // their own tool, not the cwd the build ran from
            let config_path = root.join(config_from);
            // read the raw bytes; a referenced-but-unreadable config is an authoring bug, so fail
            let config_bytes = std::fs::read(&config_path).map_err(|e| {
                Error::new(format!(
                    "Failed to read config '{}': {e}",
                    config_path.display()
                ))
            })?;
            // parse as arbitrary JSON (not the typed request) so partial/stub configs still load;
            // typing happens later in canonicalize_config
            let value = serde_json::from_slice(&config_bytes).map_err(|e| {
                Error::new(format!(
                    "Failed to parse config '{}': {e}",
                    config_path.display()
                ))
            })?;
            Ok(LoadedConfig {
                value: Some(value),
                url: None,
            })
        }
        // no config_from at all yields an empty object, the marker build_image_version reads to
        // classify the tool as build-only (built but carrying no importable config)
        None => Ok(LoadedConfig {
            value: Some(serde_json::Value::Object(serde_json::Map::new())),
            url: None,
        }),
    }
}

/// Re-serialize an embedded config through its typed request so build's on-disk
/// form matches what `toolbox export` writes
///
/// Most importantly this runs resource fields (cpu/memory/ephemeral_storage)
/// through the request's canonical unit serializers — `export` emits `2Gi`/`1`
/// where a hand-written config might say `2048Mi`/`1000m` — but it normalizes
/// every other custom-serialized field the same way, so a built and an exported
/// toolbox describe identical resources identically.
///
/// Best-effort: a partial/stub config (or one whose body is resolved from a URL at
/// import time) that doesn't deserialize into `T` is left untouched, so loosely
/// specified toolboxes still build and still fail clearly at import if invalid.
///
/// # Arguments
///
/// * `config` - The finished embedded config, if any
fn canonicalize_config<T>(config: Option<serde_json::Value>) -> Option<serde_json::Value>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    config.map(|value| match serde_json::from_value::<T>(value.clone()) {
        // round-trip through the typed request to get its canonical serialization
        Ok(typed) => serde_json::to_value(&typed).unwrap_or(value),
        // not a complete `T` (a stub, or a config resolved later from a URL): keep as-is
        Err(_) => value,
    })
}

/// The non-empty `image` url already present in a loaded image config, if any
///
/// # Arguments
///
/// * `config` - The loaded image config to read the `image` url from
fn config_image_url(config: &serde_json::Value) -> Option<String> {
    // read `image` as a string and reject an empty one: an init scaffold leaves `image: ""`, which
    // is not a real url, so callers can treat empty the same as absent
    config
        .get("image")
        .and_then(serde_json::Value::as_str)
        .filter(|url| !url.is_empty())
        .map(str::to_string)
}

/// Set the `image` url on a loaded image config
///
/// A `None` config is a config resolved from a URL at import time (nothing local to
/// set); a non-object config can't take a url, which is surfaced as a warning rather
/// than silently dropped (mirrors `apply_description_md`).
///
/// # Arguments
///
/// * `config` - The loaded config to set the url on
/// * `name` - The image name, for the non-object warning
/// * `url` - The image url to set
fn set_config_image(config: &mut Option<serde_json::Value>, name: &str, url: &str) {
    match config {
        // the normal case: write the url into the config object, overwriting any existing `image`
        Some(serde_json::Value::Object(map)) => {
            map.insert(
                "image".to_string(),
                serde_json::Value::String(url.to_string()),
            );
        }
        // a config that parsed as something other than an object can't hold an `image` key; warn
        // instead of silently dropping the url so the misshapen config is noticed
        Some(_) => {
            progress::warn(format!(
                "{name}: config is not a JSON object; cannot set the image url"
            ));
        }
        // a URL-resolved config has nothing local to set; the importer fetches and validates it
        None => {}
    }
}

/// Whether a loaded image config needs a container image pulled from a registry
///
/// Only K8s images pull a container from a registry; `BareMetal`, `Windows`, `Kvm`,
/// and `External` images run without one, so they don't require a derived tag or an
/// `image` url. An absent `scaler` field defaults to K8s (matching the image model's
/// serde default). A config resolved from a URL at import time (`None`) can't be
/// introspected here, so it's treated as not requiring one — the API validates it at
/// import.
///
/// # Arguments
///
/// * `config` - The loaded image config to read the `scaler` from
fn config_requires_container_image(config: Option<&serde_json::Value>) -> bool {
    match config {
        Some(value) => match value.get("scaler").and_then(serde_json::Value::as_str) {
            // a named scaler requires a container image only when it is K8s
            Some(scaler) => matches!(ImageScaler::from_str(scaler), Ok(ImageScaler::K8s)),
            // an absent scaler defaults to K8s, which does require one
            None => true,
        },
        None => false,
    }
}

/// Derive an image's registry tags as `<registry>/[<prefix>/]<leaf>:<version>[<tag_suffix>]`
///
/// The leaf is the tool `name` by default, or the manifest `image_name` (a repo-style path)
/// when `use_image_path` is set. Empty registries are skipped (they can't anchor a real tag),
/// as is an empty leaf (nothing to tag — only reachable when `use_image_path` is set and the
/// manifest has no `image_name`).
///
/// # Arguments
///
/// * `image_name` - The image repository path from the manifest; the leaf when `use_image_path`
/// * `name` - The tool name; the leaf by default
/// * `version` - The image version, used as the tag
/// * `registries` - The registries to tag for
/// * `use_image_path` - Tag with the `image_name` path instead of the tool `name`
/// * `image_path_prefix` - Optional path inserted between the registry and the leaf
/// * `tag_suffix` - Optional suffix appended to the version (e.g. `-mybranch`)
fn derive_image_tags(
    image_name: &str,
    name: &str,
    version: &str,
    registries: &[String],
    use_image_path: bool,
    image_path_prefix: Option<&str>,
    tag_suffix: Option<&str>,
) -> Vec<String> {
    // pick the leaf to tag with: the image_name path when opted in, else the tool name
    let leaf = if use_image_path { image_name } else { name };
    // an empty leaf can't anchor a tag
    if leaf.is_empty() {
        return Vec::new();
    }
    // optionally insert a registry path prefix between the registry and the leaf
    let path = match image_path_prefix {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}/{leaf}"),
        _ => leaf.to_string(),
    };
    // append the tag suffix to the version so feature-branch builds get differentiated
    // tags (e.g. `1.0-mybranch`) instead of colliding with the mainline version
    let version = match tag_suffix {
        Some(suffix) if !suffix.is_empty() => format!("{version}{suffix}"),
        _ => version.to_string(),
    };
    let mut tags = Vec::new();
    for registry in registries {
        // an empty registry can't anchor a tag (it would render `/path:version`); skip it
        if registry.is_empty() {
            continue;
        }
        // one tag per registry; the first is later written into the config as the pull url and the
        // rest are mirror push targets
        let tag = format!("{registry}/{path}:{version}");
        // de-dup so a registry listed in both `registry` and `registries` produces a single tag
        if !tags.contains(&tag) {
            tags.push(tag);
        }
    }
    tags
}

/// Load an image's bundled network policies from its manifest entries
///
/// Local files are parsed inline; URL entries are passed through for
/// resolution at import time (mirroring `config_from`).
///
/// # Arguments
///
/// * `root` - The tool directory the manifest lives in
/// * `entries` - The `network_policies_from` entries from the manifest
fn load_network_policies(
    root: &Path,
    entries: Option<&Vec<String>>,
) -> Result<(Vec<String>, Vec<serde_json::Value>), Error> {
    let mut urls = Vec::new();
    let mut policies = Vec::new();
    // `entries` is doubly optional (no field, or an empty list); flatten handles both as no-op
    for entry in entries.into_iter().flatten() {
        // a URL entry is deferred for the importer to fetch, exactly like a URL config_from
        if Url::parse(entry).is_ok() {
            urls.push(entry.clone());
            continue;
        }
        // a local entry is read relative to the tool dir so policies are referenced per-tool
        let policy_path = root.join(entry);
        // a referenced-but-unreadable policy is an authoring bug, so fail rather than drop it
        let policy_bytes = std::fs::read(&policy_path).map_err(|e| {
            Error::new(format!(
                "Failed to read network policy '{}': {e}",
                policy_path.display()
            ))
        })?;
        // parse as arbitrary JSON and embed inline so the definition travels with the toolbox
        let policy = serde_json::from_slice(&policy_bytes).map_err(|e| {
            Error::new(format!(
                "Failed to parse network policy '{}': {e}",
                policy_path.display()
            ))
        })?;
        policies.push(policy);
    }
    Ok((urls, policies))
}

/// Inject a tool's `description.md` into its config's `description` field
///
/// The markdown file is the single source of truth for tool documentation:
/// when present beside the manifest it wins over any inline description, with
/// a warning when it overrides a differing non-empty value. Configs fetched
/// from URLs at import time can't be injected into, so that combination only
/// warns.
///
/// # Arguments
///
/// * `root` - The tool directory that may hold a description.md
/// * `name` - The tool name, used for warnings
/// * `config` - The loaded config to inject into, if any
fn apply_description_md(root: &Path, name: &str, config: &mut Option<serde_json::Value>) {
    let desc_path = root.join("description.md");
    let Ok(description) = std::fs::read_to_string(&desc_path) else {
        // no description.md is the common case; nothing to do
        return;
    };
    // trim trailing whitespace (editors leave a final newline) so the embedded description is
    // byte-stable across runs and matches what export writes back out
    let description = description.trim_end().to_string();
    // an empty (or whitespace-only) file is treated as absent, leaving any inline description intact
    if description.is_empty() {
        return;
    }
    match config {
        Some(serde_json::Value::Object(map)) => {
            // warn when stomping a differing inline description so the
            // conflict is visible instead of silently resolved
            if let Some(inline) = map.get("description").and_then(serde_json::Value::as_str)
                && !inline.is_empty()
                && inline != description
            {
                eprintln!(
                    "Warning: {name}: description.md overrides the differing inline description in its config"
                );
            }
            map.insert(
                "description".to_string(),
                serde_json::Value::String(description),
            );
        }
        Some(_) => {
            progress::warn(format!(
                "{name}: config is not a JSON object; skipping description.md"
            ));
        }
        // config_from URLs are resolved at import time, after build
        None => {
            progress::warn(format!(
                "{name}: description.md cannot be injected into a URL-based config"
            ));
        }
    }
}

/// Express a build context relative to `output_dir` (the directory holding the
/// generated `toolbox.json`)
///
/// `toolbox.json` is the source of truth for paths: anchoring each `build_path` to
/// its directory (rather than the cwd the build ran from) keeps the toolbox movable
/// and lets `toolbox build-images` resolve contexts against the manifest it reads.
///
/// # Arguments
///
/// * `context` - The docker build context directory
/// * `output_dir` - The directory the `toolbox.json` is written to
fn build_path_relative_to_output(context: &Path, output_dir: &Path) -> String {
    // absolutize both (cwd-prefixed, no existence/symlink requirement) so diff_paths
    // compares them lexically; fall back to the raw context if that ever fails
    let ctx = std::path::absolute(context).unwrap_or_else(|_| context.to_path_buf());
    let out = std::path::absolute(output_dir).unwrap_or_else(|_| output_dir.to_path_buf());
    let rel = pathdiff::diff_paths(&ctx, &out).unwrap_or(ctx);
    // use forward slashes so the committed toolbox.json is portable across platforms
    let rendered = rel.to_string_lossy().replace('\\', "/");
    // an empty diff means the context IS the output dir
    if rendered.is_empty() {
        ".".to_string()
    } else {
        rendered
    }
}

/// Merge a per-tool `[base_image]` over the toolbox-wide default, field by field
///
/// A per-tool field wins when set, otherwise the global field is used. When the merged result has
/// an `image` but no `image_arg`, `image_arg` defaults to [`DEFAULT_BASE_IMAGE_ARG`] so the entry is
/// self-contained. Returns `None` when neither side contributes any field.
///
/// # Arguments
///
/// * `global` - The toolbox-wide `[base_image]` (from config.toml)
/// * `per_tool` - The image's `[base_image]` (from its manifest.toml)
fn merge_base_image(global: Option<&BaseImage>, per_tool: Option<&BaseImage>) -> Option<BaseImage> {
    // nothing on either side -> no base-image config at all
    if global.is_none() && per_tool.is_none() {
        return None;
    }
    // each field takes the per-tool value when set, else falls back to the global value
    let image = per_tool
        .and_then(|base| base.image.clone())
        .or_else(|| global.and_then(|base| base.image.clone()));
    let mut image_arg = per_tool
        .and_then(|base| base.image_arg.clone())
        .or_else(|| global.and_then(|base| base.image_arg.clone()));
    // an image override with no explicit arg uses the standard build-arg name
    if image.is_some() && image_arg.is_none() {
        image_arg = Some(DEFAULT_BASE_IMAGE_ARG.to_string());
    }
    let token = per_tool
        .and_then(|base| base.token.clone())
        .or_else(|| global.and_then(|base| base.token.clone()));
    let user = per_tool
        .and_then(|base| base.user.clone())
        .or_else(|| global.and_then(|base| base.user.clone()));
    let allow_override = per_tool
        .and_then(|base| base.allow_override)
        .or_else(|| global.and_then(|base| base.allow_override));
    let merged = BaseImage {
        image,
        image_arg,
        token,
        user,
        allow_override,
    };
    // a merge that produced no meaningful field is treated as absent
    if merged.image.is_none()
        && merged.image_arg.is_none()
        && merged.token.is_none()
        && merged.user.is_none()
        && merged.allow_override.is_none()
    {
        None
    } else {
        Some(merged)
    }
}

/// Assembles one image's `toolbox.json` entry from its manifest and on-disk config
///
/// Resolves the build context path, embeds the (canonicalized) config and any network
/// policies, and derives the registry tags — either pinning the real path of an
/// already-published image or deriving `<registry>/[prefix/]<image_name>:<version>`.
///
/// # Arguments
///
/// * `manifest` - The image's parsed `manifest.toml`
/// * `root` - The manifest's directory
/// * `output_dir` - The directory the `toolbox.json` is written to
/// * `registries` - The registries to derive tags for
/// * `use_image_path` - Tag with the `image_name` path instead of the tool `name`
/// * `image_path_prefix` - Optional registry base path to prefix derived tags with
/// * `tag_suffix` - Optional suffix appended to each derived tag's version
/// * `global_base_image` - The toolbox-wide `[base_image]` the per-tool one merges over
#[allow(clippy::too_many_arguments)]
fn build_image_version(
    manifest: &ManifestToml,
    root: &Path,
    output_dir: &Path,
    registries: &[String],
    use_image_path: bool,
    image_path_prefix: Option<&str>,
    tag_suffix: Option<&str>,
    global_base_image: Option<&BaseImage>,
) -> Result<BuildImageVersion, Error> {
    // an absent image_name is the empty string so derive_image_tags can treat "no path" uniformly
    let image_name = manifest.image_name.as_deref().unwrap_or("");
    let name = &manifest.name;
    let version = &manifest.version;
    // merge this image's base-image config over the toolbox-wide default (per-tool wins); the result
    // is written onto the entry for build-images, except for image_from images (cleared below)
    let base_image = merge_base_image(global_base_image, manifest.base_image.as_ref());

    // an empty version would produce a broken registry tag like "registry/path:"
    // (see the tag assembly below), so reject it rather than emit junk
    if version.is_empty() {
        return Err(Error::new(format!(
            "Image '{name}' has an empty version; set a non-empty 'version' in its manifest.toml"
        )));
    }
    // surface base-image contradictions a tool declared within its OWN [base_image]. These check
    // the manifest's own fields (`own`), never the merged result, so an inherited toolbox-wide
    // default can't trigger a per-tool warning.
    if let Some(own) = manifest.base_image.as_ref() {
        if manifest.image_from.is_some() {
            // an image_from image is never built, so its [base_image] has no effect
            eprintln!(
                "Warning: {name}: [base_image] is ignored on an image_from image (it is never built)"
            );
        } else {
            // this manifest both sets an override image and disables overriding
            if own.image.is_some() && own.allow_override == Some(false) {
                eprintln!(
                    "Warning: {name}: [base_image].image is set but allow_override = false, so the base image substitution will be skipped"
                );
            }
            // this manifest names a build-arg but supplies no image, and none is inherited either
            if own.image.is_none()
                && own.image_arg.is_some()
                && base_image
                    .as_ref()
                    .and_then(|base| base.image.as_ref())
                    .is_none()
            {
                eprintln!(
                    "Warning: {name}: [base_image].image_arg is set without an image, so it has no effect"
                );
            }
        }
    }

    let build_path_str = manifest.build_path.as_deref().unwrap_or("./");
    // "./" / "." mean the manifest's own directory; anything else is joined onto it
    // with Path::join (portable, unlike a manual "/" concat)
    let context_dir = if build_path_str == "./" || build_path_str == "." {
        root.to_path_buf()
    } else {
        root.join(build_path_str)
    };
    // record the context relative to the toolbox.json's directory, not the cwd, so
    // the toolbox stays movable and build-images resolves it against the manifest
    let image_build_path = build_path_relative_to_output(&context_dir, output_dir);
    // the tool directory (where this manifest.toml and any bundled tarball live), relative to
    // the toolbox.json's directory; import resolves a bundled image's tarball from here. It is
    // independent of build_path (the build context) though they coincide for build_path = "./".
    let dir = build_path_relative_to_output(root, output_dir);

    // resolve config_from: an inline value (local path or absent->empty object) or a deferred URL
    let loaded = load_json_config(root, manifest.config_from.as_deref())?;
    let mut config = loaded.value;
    // bundle any network policy definitions the image references (local files inline, URLs deferred)
    let (network_policies_from, network_policies) =
        load_network_policies(root, manifest.network_policies_from.as_ref())?;

    // an image that reuses another's container image is never built and derives no tag
    // of its own: the container url is filled in after the walk (see resolve_image_from
    // in build_output) once every image's url is known. build_path/build/[base_image]/
    // exported_image_path have no effect here (a warning is emitted above if [base_image]
    // was set on an image_from image).
    if manifest.image_from.is_some() {
        // description.md still applies so the reused image keeps its own docs
        apply_description_md(root, name, &mut config);
        // normalize like every other config so build matches export
        let config = canonicalize_config::<ImageRequest>(config);
        return Ok(BuildImageVersion {
            dir,
            build_path: image_build_path,
            build_image: false,
            image_tags: Vec::new(),
            config_from: loaded.url,
            config,
            network_policies_from,
            network_policies,
            // an image_from image is never built, so it carries no base-image config (not even an
            // inherited toolbox-wide default)
            base_image: None,
        });
    }

    // an image whose manifest provides no Thorium config (no config_from, or an empty
    // `{}` config) is "build-only": it is built and kept in toolbox.json, but carries no
    // embedded config so `toolbox import` skips it. Don't synthesize a config just to
    // hold the derived tag — the tag lives only in image_tags for build-images, and the
    // config is left absent rather than emitted as `{"image": ...}`.
    let build_only = loaded.url.is_none()
        && config
            .as_ref()
            .is_none_or(|value| value.as_object().is_some_and(|map| map.is_empty()));
    if build_only {
        let image_tags = derive_image_tags(
            image_name,
            name,
            version,
            registries,
            use_image_path,
            image_path_prefix,
            tag_suffix,
        );
        // a build-enabled image that derives no tag builds nothing — import skips it (no
        // config) and build-images skips it (no tags) — so surface the silent no-op
        if manifest.build && image_tags.is_empty() {
            eprintln!(
                "Warning: {name}: build is enabled but no image tag could be derived; add a \
                 'registry' to config.toml (the tool name is the tag leaf; pass --use-image-path \
                 with an 'image_name' for a repo-style path). build-images will skip it."
            );
        }
        return Ok(BuildImageVersion {
            dir,
            build_path: image_build_path,
            build_image: manifest.build,
            image_tags,
            config_from: None,
            config: None,
            network_policies_from,
            network_policies,
            base_image,
        });
    }

    // description.md beside the manifest becomes the image's description
    apply_description_md(root, name, &mut config);

    // An image we don't build locally keeps the real registry path it already lives
    // at — the url an export recorded, or one already in its config — so an exported
    // toolbox round-trips instead of pointing at a `registry + image_name`-derived
    // guess. A buildable image (`build = true`) derives that path so CI/CD builds and
    // pushes to where toolbox.json says, even if the export recorded something else.
    let pinned = if manifest.build {
        None
    } else {
        manifest
            .exported_image_path
            .as_deref()
            .filter(|url| !url.is_empty())
            .map(str::to_string)
            .or_else(|| config.as_ref().and_then(config_image_url))
    };

    let image_tags = if let Some(url) = pinned {
        // use the real path verbatim for both the tag and the embedded config
        set_config_image(&mut config, name, &url);
        vec![url]
    } else {
        // derive <registry>/[prefix/]<image_name>:<version> for each registry
        let tags = derive_image_tags(
            image_name,
            name,
            version,
            registries,
            use_image_path,
            image_path_prefix,
            tag_suffix,
        );
        match tags.first() {
            Some(first) => {
                // warn if the derived tag silently replaces an image the user set in
                // the config (an init scaffold has an empty image, so this only fires
                // when a real tag is being overwritten)
                if let Some(existing) = config.as_ref().and_then(config_image_url)
                    && &existing != first
                {
                    eprintln!(
                        "Warning: {name}: build derived image tag '{first}' overrides the config's image '{existing}' (set build = false to keep the config's url)"
                    );
                }
                set_config_image(&mut config, name, first);
                tags
            }
            None => match config.as_ref().and_then(config_image_url) {
                // no registry tag could be derived (no registry, or no image_name), so
                // fall back to the image url already in the config — this lets a toolbox
                // without a central registry point each tool at its own. A buildable
                // image normally has its tag derived, so note the fallback per-image.
                Some(existing) => {
                    if manifest.build {
                        eprintln!(
                            "Warning: {name}: could not derive a registry tag (no registry, or --use-image-path with no image_name); falling back to the image '{existing}' set in its config"
                        );
                    }
                    vec![existing]
                }
                // nothing to tag the image with; build_output errors after the walk if
                // this image's scaler (K8s) requires a container image
                None => Vec::new(),
            },
        }
    };

    // normalize the finished config to its canonical form so build matches export
    // (resource units in particular); stub/URL configs pass through untouched
    let config = canonicalize_config::<ImageRequest>(config);

    Ok(BuildImageVersion {
        dir,
        build_path: image_build_path,
        build_image: manifest.build,
        image_tags,
        config_from: loaded.url,
        config,
        network_policies_from,
        network_policies,
        base_image,
    })
}

/// Assembles one pipeline's `toolbox.json` entry from its manifest and on-disk config
///
/// # Arguments
///
/// * `manifest` - The pipeline's parsed `manifest.toml`
/// * `root` - The manifest's directory
fn build_pipeline_version(
    manifest: &ManifestToml,
    root: &Path,
    output_dir: &Path,
) -> Result<BuildPipelineVersion, Error> {
    // the tool directory (where this manifest.toml lives), relative to the toolbox.json's directory —
    // recorded so export can find a pipeline's existing home and update it in place
    let dir = build_path_relative_to_output(root, output_dir);
    // a pipeline's manifest description seeds the entry; a sibling description.md overrides it below
    // (the manifest value lives on the entry, the override lands in the embedded config)
    let description = manifest.description.clone().unwrap_or_default();
    // project the manifest's image->version map onto the output shape; an absent images table
    // yields an empty map rather than failing, leaving structural validation to catch it on import
    let images: HashMap<String, PipelineImageOutput> = manifest
        .images
        .as_ref()
        .map(|imgs| {
            imgs.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        PipelineImageOutput {
                            version: v.version.clone(),
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    // resolve config_from the same way images do (inline value or deferred URL)
    let loaded = load_json_config(root, manifest.config_from.as_deref())?;
    let mut config = loaded.value;
    // description.md beside the manifest becomes the pipeline's description
    apply_description_md(root, &manifest.name, &mut config);
    // normalize the finished config to its canonical form so build matches export;
    // stub/URL configs pass through untouched
    let config = canonicalize_config::<PipelineRequest>(config);

    Ok(BuildPipelineVersion {
        dir,
        description,
        images,
        config_from: loaded.url,
        config,
    })
}

/// Build a toolbox manifest from a directory of image and pipeline manifests
///
/// A buildable image's registry url is derived from the registry/prefix and
/// `image_name`; an image that isn't built locally keeps its recorded
/// `exported_image_path` (or existing config url) so exported toolboxes round-trip.
///
/// # Arguments
///
/// * `cmd` - The build command arguments (config path, walk root, and output path)
pub fn build(cmd: &BuildToolbox) -> Result<(), Error> {
    // resolve the crawl root and output path (both default to the config's directory so a
    // toolbox is self-contained around its config.toml); announce them so the defaulted-from
    // -config behavior is visible rather than implicit
    let crawl = resolved_crawl(cmd);
    let output_path = resolved_output(cmd);
    println!(
        "Building toolbox from '{}' (writing '{}')",
        crawl.display(),
        output_path.display()
    );
    let output = build_output(cmd)?;
    // serialize through the canonical (sorted-key) form: `BuildOutput`'s image and
    // pipeline maps are `HashMap`s, so a direct `to_string_pretty` would emit keys in
    // random order and churn `toolbox.json` between otherwise-identical runs. Routing
    // through serde_json::Value (a sorted BTreeMap) makes the file byte-deterministic
    // so it can be committed and diffed.
    let json = crate::utils::canonical_json(&output)?;
    std::fs::write(&output_path, json)
        .map_err(|e| Error::new(format!("Failed to write '{}': {e}", output_path.display())))?;
    println!("Wrote toolbox manifest to '{}'", output_path.display());
    Ok(())
}

/// The directory a toolbox's `config.toml` lives in
///
/// Used as the default crawl root and the default `toolbox.json` location so a toolbox is
/// self-contained around its config. A bare `config.toml` has an empty parent, which is
/// treated as the current directory.
///
/// # Arguments
///
/// * `config` - The path to the toolbox's `config.toml`
pub(super) fn config_base_dir(config: &Path) -> PathBuf {
    config
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

/// Resolve the directory to crawl for manifests: an explicit `--path`, else the config's dir
///
/// # Arguments
///
/// * `cmd` - The build command arguments
fn resolved_crawl(cmd: &BuildToolbox) -> PathBuf {
    cmd.path
        .clone()
        .unwrap_or_else(|| config_base_dir(&cmd.config))
}

/// Resolve the `toolbox.json` output path: an explicit `--output`, else `toolbox.json` in the
/// config's directory
///
/// # Arguments
///
/// * `cmd` - The build command arguments
fn resolved_output(cmd: &BuildToolbox) -> PathBuf {
    cmd.output
        .clone()
        .unwrap_or_else(|| config_base_dir(&cmd.config).join("toolbox.json"))
}

/// Build a toolbox manifest in memory without writing it to disk
///
/// Used by `toolbox diff` to compare a toolbox repo checkout against a live
/// instance without requiring a regenerated `toolbox.json`.
///
/// # Arguments
///
/// * `cmd` - The build inputs (config path and walk root)
pub(super) fn build_in_memory(cmd: &BuildToolbox) -> Result<serde_json::Value, Error> {
    // run the same walk/assembly as `build`, just keeping the result in memory
    let output = build_output(cmd)?;
    // hand back a `Value` (not the canonical sorted-key string) since diff compares structures, not
    // bytes, and doesn't need the deterministic on-disk form
    serde_json::to_value(&output)
        .map_err(|e| Error::new(format!("Failed to serialize toolbox output: {e}")))
}

/// Resolve the concrete container url an `image_from` image should reuse
///
/// Follows the reuse chain (a target that is itself an `image_from`) until it reaches a
/// real image, then reads that image's container url — the url embedded in its config, or
/// failing that the first of its derived tags. Errors on a cycle, a target absent from the
/// toolbox, or a target that has no container image to reuse.
///
/// # Arguments
///
/// * `ref_name` - The name of the image declaring `image_from`
/// * `ref_version` - The version of the image declaring `image_from`
/// * `images` - Every built image entry, keyed by name then version
/// * `from_map` - Each `image_from` image's `(name, version)` mapped to its target's
fn resolve_image_from_url(
    ref_name: &str,
    ref_version: &str,
    images: &HashMap<String, HashMap<String, BuildImageVersion>>,
    from_map: &HashMap<(String, String), (String, String)>,
) -> Result<String, String> {
    // seed the visited set with the declaring image so a self- or mutual-reference is
    // caught as a cycle rather than looping forever
    let mut visited: HashSet<(String, String)> = HashSet::new();
    visited.insert((ref_name.to_string(), ref_version.to_string()));
    // start walking from this image's target
    let (mut name, mut version) = from_map
        .get(&(ref_name.to_string(), ref_version.to_string()))
        .cloned()
        .expect("caller only passes keys present in from_map");
    loop {
        // re-visiting a node means the chain loops back on itself
        if !visited.insert((name.clone(), version.clone())) {
            return Err(format!(
                "image_from chain cycles back to '{name}:{version}'"
            ));
        }
        // if this node also reuses another image, keep following the chain
        if let Some((next_name, next_version)) = from_map.get(&(name.clone(), version.clone())) {
            name.clone_from(next_name);
            version.clone_from(next_version);
            continue;
        }
        // base case: read the concrete url off the resolved image, preferring its embedded
        // config url and falling back to its first derived tag
        let entry = images
            .get(&name)
            .and_then(|versions| versions.get(&version))
            .ok_or_else(|| format!("image '{name}:{version}' not found in toolbox"))?;
        return entry
            .config
            .as_ref()
            .and_then(config_image_url)
            .or_else(|| entry.image_tags.first().cloned())
            .ok_or_else(|| format!("image '{name}:{version}' has no container image to reuse"));
    }
}

/// Walk the manifests and assemble the toolbox output structure
///
/// # Arguments
///
/// * `cmd` - The build command arguments
fn build_output(cmd: &BuildToolbox) -> Result<BuildOutput, Error> {
    let config = load_config(&cmd.config)?;

    // assemble the registries to derive image tags for: the primary registry plus any
    // extras, dropping blanks and dupes so an unset/empty registry contributes nothing
    // (an empty list means tags come from each image's own config instead)
    let mut registries: Vec<String> = Vec::new();
    for registry in config.registry.iter().chain(config.registries.iter()) {
        if !registry.is_empty() && !registries.contains(registry) {
            registries.push(registry.clone());
        }
    }

    // resolve the crawl root and output path (both default to the config's directory)
    let crawl = resolved_crawl(cmd);
    let output_path = resolved_output(cmd);
    // build_path values are recorded relative to the toolbox.json's directory; an
    // output with no parent (e.g. bare "toolbox.json") anchors to the cwd
    let output_dir = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let mut images: HashMap<String, HashMap<String, BuildImageVersion>> = HashMap::new();
    let mut pipelines: HashMap<String, HashMap<String, BuildPipelineVersion>> = HashMap::new();
    // each `image_from` image's `(name, version)` mapped to the `(name, version)` of the
    // image it reuses; resolved into a concrete container url after the walk, once every
    // image's url is known
    let mut image_from_map: HashMap<(String, String), (String, String)> = HashMap::new();
    // the source directory each `(name, version)` was first seen in, so a duplicate can name
    // both sides; a duplicate would otherwise silently overwrite in filesystem-walk order and
    // make toolbox.json non-deterministic
    let mut image_locations: HashMap<(String, String), String> = HashMap::new();
    let mut pipeline_locations: HashMap<(String, String), String> = HashMap::new();

    for entry in WalkDir::new(&crawl) {
        // a path we can't read (permissions, broken symlink) shouldn't vanish
        // silently from the output — surface it and keep walking the rest
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                progress::warn(format!(
                    "skipping unreadable path during toolbox walk: {err}"
                ));
                continue;
            }
        };
        let path = entry.path();
        // only manifest.toml files are tools; the directory-grouping convention is irrelevant — any
        // manifest.toml at any depth is picked up, so skip every other file
        if path.file_name() != Some(OsStr::new("manifest.toml")) {
            continue;
        }
        // a tool's directory (the manifest's parent) anchors all of its relative paths; skip a
        // manifest with no usable parent (e.g. one walked at the bare root) since there'd be nothing
        // to resolve its config/policies/build context against
        let root = match path.parent() {
            Some(r) if r != Path::new("") && r != Path::new(".") => r,
            _ => continue,
        };

        // unlike an unreadable directory entry above (skippable filesystem noise), a
        // manifest.toml that exists but won't read or parse is a real authoring bug — fail
        // the build loudly rather than silently dropping the tool
        let manifest_str = std::fs::read_to_string(path)
            .map_err(|e| Error::new(format!("Failed to read '{}': {e}", path.display())))?;
        let manifest: ManifestToml = toml::from_str(&manifest_str)
            .map_err(|e| Error::new(format!("Failed to parse '{}': {e}", path.display())))?;

        // name+version is the identity that must be unique within each kind (images, pipelines) and
        // the lookup key for image_from resolution and duplicate detection below
        let key = (manifest.name.clone(), manifest.version.clone());
        match manifest.manifest_type {
            ManifestType::Image => {
                // two image manifests with the same name+version would overwrite each other in
                // walk order; fail and name both directories so the author can disambiguate
                if let Some(prev) = image_locations.get(&key) {
                    return Err(Error::new(format!(
                        "duplicate image manifest '{}:{}' in '{prev}' and '{}'; each image's \
                         name+version must be unique across the toolbox",
                        key.0,
                        key.1,
                        root.display()
                    )));
                }
                // remember where this identity was first seen so a later duplicate can name both dirs
                image_locations.insert(key.clone(), root.display().to_string());
                // assemble the entry now; image_from urls are still placeholders, filled post-walk
                let version_entry = build_image_version(
                    &manifest,
                    root,
                    output_dir,
                    &registries,
                    cmd.use_image_path,
                    config.image_path_prefix.as_deref(),
                    cmd.tag_suffix.as_deref(),
                    config.base_image.as_ref(),
                )?;
                // record an image_from reference so it can be resolved to a concrete url
                // after the whole tree is walked (the target may appear later)
                if let Some(from) = &manifest.image_from {
                    image_from_map.insert(key.clone(), (from.name.clone(), from.version.clone()));
                }
                // file the entry under name -> version (nested so one name can have many versions)
                images
                    .entry(manifest.name.clone())
                    .or_default()
                    .insert(manifest.version.clone(), version_entry);
            }
            ManifestType::Pipeline => {
                // same determinism guard for pipelines
                if let Some(prev) = pipeline_locations.get(&key) {
                    return Err(Error::new(format!(
                        "duplicate pipeline manifest '{}:{}' in '{prev}' and '{}'; each pipeline's \
                         name+version must be unique across the toolbox",
                        key.0,
                        key.1,
                        root.display()
                    )));
                }
                pipeline_locations.insert(key.clone(), root.display().to_string());
                let version_entry = build_pipeline_version(&manifest, root, output_dir)?;
                // file the entry under name -> version, mirroring the images map
                pipelines
                    .entry(manifest.name.clone())
                    .or_default()
                    .insert(manifest.version.clone(), version_entry);
            }
        }
    }

    // resolve every image_from reference to a concrete container url now that the whole
    // tree has been walked and each non-reusing image carries its url. Collect all
    // failures so the build reports every bad reference in one pass.
    let mut from_errors: Vec<String> = Vec::new();
    // snapshot the referencing keys so the map can be borrowed immutably while we mutate
    // the images map below; sorted for deterministic processing/error order
    let mut refs: Vec<(String, String)> = image_from_map.keys().cloned().collect();
    refs.sort();
    for (name, version) in refs {
        match resolve_image_from_url(&name, &version, &images, &image_from_map) {
            Ok(url) => {
                // safe: the key came from a walked image, so its entry exists
                if let Some(entry) = images.get_mut(&name).and_then(|v| v.get_mut(&version)) {
                    // the shared url can only be embedded into a local (object) config; a
                    // config resolved from a URL at import time can't carry it, which
                    // defeats the purpose, so reject that combination
                    if entry.config.is_some() {
                        set_config_image(&mut entry.config, &name, &url);
                        entry.image_tags = vec![url];
                    } else {
                        from_errors.push(format!(
                            "{name}:{version} -> config is resolved from a URL; image_from needs a \
                             local config to embed the reused image url"
                        ));
                    }
                }
            }
            Err(reason) => from_errors.push(format!("{name}:{version} -> {reason}")),
        }
    }
    if !from_errors.is_empty() {
        from_errors.sort();
        return Err(Error::new(format!(
            "unresolvable image_from reference(s):\n  {}",
            from_errors.join("\n  ")
        )));
    }

    // fail the build if any K8s image couldn't be given a container tag, listing them all
    // so the user can fix every one in a single pass. Run after image_from resolution so
    // reused images are checked with their now-filled tag.
    let mut untagged_k8s: Vec<String> = Vec::new();
    for (name, versions) in &images {
        for (version, entry) in versions {
            if entry.image_tags.is_empty() && config_requires_container_image(entry.config.as_ref())
            {
                untagged_k8s.push(format!("{name}:{version}"));
            }
        }
    }
    if !untagged_k8s.is_empty() {
        untagged_k8s.sort();
        return Err(Error::new(format!(
            "no container image for K8s image(s) [{}]: set 'image' in each image's config, or add a \
             'registry' to config.toml to derive one (the tool name is used as the tag leaf; pass \
             --use-image-path with an 'image_name' in the manifest to use a repo-style path instead)",
            untagged_k8s.join(", ")
        )));
    }

    // assemble the final output; the top-level base_image is the raw global config for reference,
    // while each image entry already carries its own resolved (merged) value
    Ok(BuildOutput {
        pipelines,
        images,
        name: config.name,
        registry: config.registry,
        registries,
        bundled_images: config.bundled_images,
        image_path_prefix: config.image_path_prefix,
        export_image_path: config.export_image_path,
        export_pipeline_path: config.export_pipeline_path,
        base_image: config.base_image,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// build must emit resource fields in the same canonical unit form as export,
    /// so a built and an exported toolbox describe identical resources identically
    #[test]
    fn canonicalize_normalizes_resource_units() {
        let raw = serde_json::json!({
            "group": "g", "name": "n", "scaler": "K8s",
            "resources": {
                "cpu": "1000m", "memory": "2048Mi", "ephemeral_storage": "512Mi",
                "nvidia_gpu": 0, "amd_gpu": 0,
                "burstable": { "cpu": "1500m", "memory": "3Gi" }
            }
        });
        let out = canonicalize_config::<ImageRequest>(Some(raw)).expect("config present");
        // 1000m is an even core count -> bare number; 2048Mi -> 2Gi
        assert_eq!(out["resources"]["cpu"], serde_json::json!(1));
        assert_eq!(out["resources"]["memory"], serde_json::json!("2Gi"));
        // 512Mi is not a whole Gi, so it stays in Mi
        assert_eq!(
            out["resources"]["ephemeral_storage"],
            serde_json::json!("512Mi")
        );
        // 1500m is not an even core count, so it stays in millicpu
        assert_eq!(
            out["resources"]["burstable"]["cpu"],
            serde_json::json!("1500m")
        );
    }

    /// a stub/partial config that doesn't deserialize is left untouched so loosely
    /// specified toolboxes still build
    #[test]
    fn canonicalize_passes_through_unparseable_stub() {
        let stub = serde_json::json!({ "description": "todo" });
        let out = canonicalize_config::<ImageRequest>(Some(stub.clone())).expect("config present");
        assert_eq!(out, stub);
    }

    /// a missing config (e.g. resolved from a URL at import time) stays None
    #[test]
    fn canonicalize_keeps_none() {
        assert!(canonicalize_config::<ImageRequest>(None).is_none());
    }

    /// a context under the toolbox.json's directory drops that directory's prefix so
    /// the recorded build_path is anchored to toolbox.json (the source of truth)
    #[test]
    fn build_path_strips_output_dir_prefix() {
        let rel = build_path_relative_to_output(Path::new("tools/exiftool"), Path::new("tools"));
        assert_eq!(rel, "exiftool");
    }

    /// a context that IS the output directory records "."
    #[test]
    fn build_path_same_dir_is_dot() {
        let rel = build_path_relative_to_output(Path::new("tools"), Path::new("tools"));
        assert_eq!(rel, ".");
    }

    /// a context outside the output directory walks up with ".."
    #[test]
    fn build_path_outside_output_dir_walks_up() {
        let rel = build_path_relative_to_output(Path::new("other/foo"), Path::new("tools"));
        assert_eq!(rel, "../other/foo");
    }

    /// absolute paths are diffed against each other, not the cwd
    #[test]
    fn build_path_absolute_paths() {
        let rel = build_path_relative_to_output(Path::new("/a/b/c"), Path::new("/a/b"));
        assert_eq!(rel, "c");
    }

    /// by default a buildable image derives `<registry>/<name>:<version>` from the tool name
    /// (with the optional prefix) for each non-empty registry, de-duplicated
    #[test]
    fn derive_tags_default_uses_name() {
        let tags = derive_image_tags(
            "gnu.org/binutils/strings",
            "strings-16be",
            "latest",
            &["ghcr.io/o/r".to_string(), String::new()],
            false,
            None,
            None,
        );
        assert_eq!(tags, vec!["ghcr.io/o/r/strings-16be:latest"]);
    }

    /// `use_image_path` tags with the manifest `image_name` path as the leaf, with the prefix
    /// inserted before it
    #[test]
    fn derive_tags_use_image_path() {
        let tags = derive_image_tags(
            "gnu.org/binutils/strings",
            "strings-16be",
            "latest",
            &["ghcr.io/o/r".to_string()],
            true,
            Some("pre"),
            None,
        );
        assert_eq!(
            tags,
            vec!["ghcr.io/o/r/pre/gnu.org/binutils/strings:latest"]
        );
    }

    /// a tag suffix is appended to the version, so feature-branch builds don't collide
    #[test]
    fn derive_tags_appends_suffix_to_version() {
        let tags = derive_image_tags(
            "img-name",
            "tool",
            "1.0",
            &["reg".to_string()],
            false,
            None,
            Some("-mybranch"),
        );
        assert_eq!(tags, vec!["reg/tool:1.0-mybranch"]);
    }

    /// `use_image_path` with no `image_name` yields no tags; the default still derives from name
    #[test]
    fn derive_tags_empty_leaf() {
        // opted into the image_name path but the manifest has none -> nothing to tag
        assert!(
            derive_image_tags("", "tool", "1.0", &["reg".to_string()], true, None, None).is_empty()
        );
        // the default leaf is the tool name, so an absent image_name still derives a tag
        assert_eq!(
            derive_image_tags("", "tool", "1.0", &["reg".to_string()], false, None, None),
            vec!["reg/tool:1.0"]
        );
    }

    /// `config_image_url` returns the embedded url only when non-empty
    #[test]
    fn config_image_url_reads_non_empty() {
        assert_eq!(
            config_image_url(&serde_json::json!({"image": "reg/x:1"})),
            Some("reg/x:1".to_string())
        );
        assert_eq!(config_image_url(&serde_json::json!({"image": ""})), None);
        assert_eq!(config_image_url(&serde_json::json!({})), None);
    }

    /// Only K8s (and a config with no/absent scaler, which defaults to K8s) requires a
    /// container image; other scalers and a URL-resolved (None) config do not
    #[test]
    fn config_requires_container_image_only_k8s() {
        // an explicit K8s scaler requires a container image
        assert!(config_requires_container_image(Some(
            &serde_json::json!({"scaler": "K8s"})
        )));
        // an absent scaler defaults to K8s, so it requires one too
        assert!(config_requires_container_image(Some(
            &serde_json::json!({"name": "x"})
        )));
        // non-K8s scalers run without a container image
        for scaler in ["BareMetal", "Windows", "Kvm", "External"] {
            assert!(
                !config_requires_container_image(Some(&serde_json::json!({"scaler": scaler}))),
                "{scaler} should not require a container image"
            );
        }
        // a config resolved from a URL at import time can't be introspected here
        assert!(!config_requires_container_image(None));
    }

    /// `set_config_image` writes the url into an object config and no-ops on a None config
    #[test]
    fn set_config_image_sets_object() {
        let mut config = Some(serde_json::json!({"name": "x"}));
        set_config_image(&mut config, "x", "reg/x:1");
        assert_eq!(config.unwrap()["image"], serde_json::json!("reg/x:1"));
        let mut none: Option<serde_json::Value> = None;
        set_config_image(&mut none, "x", "reg/x:1");
        assert!(none.is_none());
    }

    /// Build a minimal `BuildImageVersion` carrying an optional config `image` url and
    /// a set of derived tags, for the `image_from` resolver tests
    ///
    /// # Arguments
    ///
    /// * `image` - The `image` url to embed in the config, if any
    /// * `tags` - The derived tags to populate `image_tags` with
    fn image_entry(image: Option<&str>, tags: &[&str]) -> BuildImageVersion {
        BuildImageVersion {
            dir: ".".to_string(),
            build_path: ".".to_string(),
            build_image: false,
            image_tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
            config_from: None,
            config: image.map(|url| serde_json::json!({ "image": url })),
            network_policies_from: Vec::new(),
            network_policies: Vec::new(),
            base_image: None,
        }
    }

    /// Assemble an images map from `(name, version, entry)` triples
    ///
    /// # Arguments
    ///
    /// * `entries` - The `(name, version, entry)` triples to nest into a name -> version map
    fn images_map(
        entries: Vec<(&str, &str, BuildImageVersion)>,
    ) -> HashMap<String, HashMap<String, BuildImageVersion>> {
        let mut map: HashMap<String, HashMap<String, BuildImageVersion>> = HashMap::new();
        for (name, version, entry) in entries {
            map.entry(name.to_string())
                .or_default()
                .insert(version.to_string(), entry);
        }
        map
    }

    /// Assemble an `image_from` map from `((ref_name, ref_ver), (target_name, target_ver))` pairs
    ///
    /// # Arguments
    ///
    /// * `refs` - The `((ref_name, ref_ver), (target_name, target_ver))` pairs to map
    fn from_map(
        refs: Vec<((&str, &str), (&str, &str))>,
    ) -> HashMap<(String, String), (String, String)> {
        refs.into_iter()
            .map(|((rn, rv), (tn, tv))| {
                (
                    (rn.to_string(), rv.to_string()),
                    (tn.to_string(), tv.to_string()),
                )
            })
            .collect()
    }

    /// An image_from image resolves to the target's embedded config image url
    #[test]
    fn image_from_resolves_config_url() {
        let images = images_map(vec![("a", "1", image_entry(Some("reg/a:1"), &[]))]);
        let from = from_map(vec![(("b", "latest"), ("a", "1"))]);
        assert_eq!(
            resolve_image_from_url("b", "latest", &images, &from),
            Ok("reg/a:1".to_string())
        );
    }

    /// A target with no config url falls back to its first derived tag
    #[test]
    fn image_from_falls_back_to_tag() {
        let images = images_map(vec![("a", "1", image_entry(None, &["reg/a:1"]))]);
        let from = from_map(vec![(("b", "latest"), ("a", "1"))]);
        assert_eq!(
            resolve_image_from_url("b", "latest", &images, &from),
            Ok("reg/a:1".to_string())
        );
    }

    /// A chain of reuse references follows through to the concrete image url
    #[test]
    fn image_from_resolves_transitive_chain() {
        // c -> b -> a, where only a has a concrete url
        let images = images_map(vec![
            ("a", "1", image_entry(Some("reg/a:1"), &[])),
            ("b", "1", image_entry(None, &[])),
            ("c", "1", image_entry(None, &[])),
        ]);
        let from = from_map(vec![(("c", "1"), ("b", "1")), (("b", "1"), ("a", "1"))]);
        assert_eq!(
            resolve_image_from_url("c", "1", &images, &from),
            Ok("reg/a:1".to_string())
        );
    }

    /// A reuse cycle is detected rather than looping forever
    #[test]
    fn image_from_detects_cycle() {
        let images = images_map(vec![
            ("a", "1", image_entry(None, &[])),
            ("b", "1", image_entry(None, &[])),
        ]);
        let from = from_map(vec![(("a", "1"), ("b", "1")), (("b", "1"), ("a", "1"))]);
        let err = resolve_image_from_url("a", "1", &images, &from).unwrap_err();
        assert!(err.contains("cycle"), "expected a cycle error, got: {err}");
    }

    /// A reference to an image the toolbox doesn't know about errors
    #[test]
    fn image_from_missing_target_errors() {
        let images = images_map(vec![]);
        let from = from_map(vec![(("b", "latest"), ("missing", "1"))]);
        let err = resolve_image_from_url("b", "latest", &images, &from).unwrap_err();
        assert!(
            err.contains("not found"),
            "expected a not-found error, got: {err}"
        );
    }

    /// A target that has no container image at all (no config url, no tags) errors
    #[test]
    fn image_from_target_without_url_errors() {
        let images = images_map(vec![("a", "1", image_entry(None, &[]))]);
        let from = from_map(vec![(("b", "latest"), ("a", "1"))]);
        let err = resolve_image_from_url("b", "latest", &images, &from).unwrap_err();
        assert!(
            err.contains("no container image"),
            "expected a no-container-image error, got: {err}"
        );
    }

    /// A `BaseImage` from optional fields, for terse merge fixtures
    ///
    /// # Arguments
    ///
    /// * `image` - The base image override, if any
    /// * `image_arg` - The build-arg name, if any
    /// * `token` - The CI/CD token-variable name, if any
    /// * `user` - The CI/CD user-variable name, if any
    /// * `allow_override` - Whether the substitution applies, if set
    fn base(
        image: Option<&str>,
        image_arg: Option<&str>,
        token: Option<&str>,
        user: Option<&str>,
        allow_override: Option<bool>,
    ) -> BaseImage {
        BaseImage {
            image: image.map(str::to_string),
            image_arg: image_arg.map(str::to_string),
            token: token.map(str::to_string),
            user: user.map(str::to_string),
            allow_override,
        }
    }

    /// Nothing on either side merges to no base-image config
    #[test]
    fn merge_base_image_none() {
        assert!(merge_base_image(None, None).is_none());
    }

    /// Per-tool fields win over the global default, field by field; unset per-tool fields
    /// inherit the global value
    #[test]
    fn merge_base_image_per_tool_over_global() {
        let global = base(
            Some("global:1"),
            Some("ARG_G"),
            Some("TOK_G"),
            Some("USR_G"),
            Some(true),
        );
        // per-tool overrides image only; the rest inherit the global
        let per_tool = base(Some("tool:1"), None, None, None, None);
        let merged = merge_base_image(Some(&global), Some(&per_tool)).expect("merged");
        assert_eq!(merged.image.as_deref(), Some("tool:1"));
        assert_eq!(merged.image_arg.as_deref(), Some("ARG_G"));
        assert_eq!(merged.token.as_deref(), Some("TOK_G"));
        assert_eq!(merged.user.as_deref(), Some("USR_G"));
        assert_eq!(merged.allow_override, Some(true));
    }

    /// An image with no explicit arg (anywhere) defaults image_arg to `IMAGE`
    #[test]
    fn merge_base_image_defaults_arg() {
        let per_tool = base(Some("tool:1"), None, None, None, None);
        let merged = merge_base_image(None, Some(&per_tool)).expect("merged");
        assert_eq!(merged.image_arg.as_deref(), Some(DEFAULT_BASE_IMAGE_ARG));
        assert_eq!(DEFAULT_BASE_IMAGE_ARG, "IMAGE");
    }

    /// token/user-only config merges through without an image (and no defaulted arg)
    #[test]
    fn merge_base_image_token_user_only() {
        let global = base(None, None, Some("TOK"), Some("USR"), None);
        let merged = merge_base_image(Some(&global), None).expect("merged");
        assert!(merged.image.is_none());
        assert!(merged.image_arg.is_none());
        assert_eq!(merged.token.as_deref(), Some("TOK"));
        assert_eq!(merged.user.as_deref(), Some("USR"));
    }
}
