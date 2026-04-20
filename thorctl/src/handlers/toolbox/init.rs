//! Scaffolds toolbox, image, and pipeline files with default configs
//!
//! Supports both interactive (default) and non-interactive (`-n`) modes.
//! Interactive mode builds the default config and opens it in the user's editor
//! to fill in (via `review_config_in_editor`). Non-interactive mode writes the
//! defaults with no editor.

use colored::Colorize;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use thorium::Error;
use thorium::models::{ImageRequest, PipelineRequest};
use walkdir::WalkDir;

use super::build::BaseImage;
use super::prompt::{self, ImageConfigAnswers, PipelineConfigAnswers};
use crate::args::Args;
use crate::args::toolbox::{Init, InitImage, InitPipeline, InitToolbox, PipelineSpec};
use crate::handlers::imports::editor;
use crate::handlers::imports::merge::{IMAGE_FIELD_ORDER, PIPELINE_FIELD_ORDER};

// ─── File Helpers ────────────────────────────────────────────────────────────

/// Writes a scaffolded file, skipping or overwriting an existing one based on `overwrite`
///
/// Returns `true` when the file was written and `false` when it already existed and
/// was skipped.
///
/// # Arguments
///
/// * `path` - The path to write to
/// * `contents` - The file contents to write
/// * `overwrite` - Overwrite an existing file instead of skipping it
pub(crate) async fn write_file(
    path: &Path,
    contents: &str,
    overwrite: bool,
) -> Result<bool, Error> {
    // stat once and surface a real IO error rather than treating it as "absent",
    // which would silently overwrite a file we couldn't read
    let exists = tokio::fs::try_exists(path)
        .await
        .map_err(|e| Error::new(format!("Failed to stat '{}': {e}", path.display())))?;
    // never clobber an existing file unless the caller opted in; report the skip so
    // the user knows a stale file was left untouched and how to force a replace
    if !overwrite && exists {
        println!(
            "{} {} (already exists; pass --overwrite to replace)",
            "Skipped".bright_yellow(),
            path.display()
        );
        return Ok(false);
    }
    // ensure the destination directory tree exists before writing into it
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            Error::new(format!(
                "Failed to create directory '{}': {e}",
                parent.display()
            ))
        })?;
    }
    // distinguish replacing a present file from creating a new one for the status line;
    // captured before the write because the file always exists afterward
    let overwritten = overwrite && exists;
    tokio::fs::write(path, contents)
        .await
        .map_err(|e| Error::new(format!("Failed to write '{}': {e}", path.display())))?;
    // tell the user which action happened (replaced vs newly created)
    if overwritten {
        println!("{} {}", "Overwrote".bright_yellow(), path.display());
    } else {
        println!("{} {}", "Created".bright_green(), path.display());
    }
    Ok(true)
}

/// Extracts the final directory component of a path as a string
///
/// # Arguments
///
/// * `path` - The path to take the directory name from
fn dir_name(path: &Path) -> Result<String, Error> {
    // take the trailing path component as the resource name, erroring on a path that
    // has no final component (e.g. `/` or one ending in `..`) or non-UTF-8 bytes
    path.file_name()
        .and_then(|n| n.to_str())
        .map(String::from)
        .ok_or_else(|| {
            Error::new(format!(
                "Cannot determine directory name for '{}'",
                path.display()
            ))
        })
}

/// Reject a resource name that isn't a valid identifier before it is interpolated
/// into a TOML manifest template
///
/// The interactive wizard validates through `prompt_name`; the `--non-interactive`
/// path takes names verbatim (directory names, `--image`, `--group`), so it must
/// run the same check or a crafted name could break out of the template.
///
/// # Arguments
///
/// * `kind` - The resource kind, for the error message ("group", "image", …)
/// * `name` - The name to validate
fn validate_resource_name(kind: &str, name: &str) -> Result<(), Error> {
    // groups allow a longer name than images/pipelines, matching the API's own per-kind caps
    let max = if kind == "group" {
        prompt::GROUP_NAME_MAX
    } else {
        prompt::RESOURCE_NAME_MAX
    };
    // reuse the wizard's name check so interactive and non-interactive paths enforce the same
    // rule, then prefix the error with the resource kind for context
    prompt::validate_name(name, max)
        .map_err(|err| Error::new(format!("Invalid {kind} name '{name}': {err}")))
}

/// Reject an export-layout path that isn't a safe relative subpath of the toolbox root
///
/// A configured layout dir (`export_image_path`/`export_pipeline_path`) or a per-resource
/// `=destpath` must stay inside the toolbox, so an absolute path or one escaping via `..` is
/// rejected before it is written into `config.toml` or used to place files.
///
/// # Arguments
///
/// * `kind` - The setting's name, for the error message
/// * `path` - The path to validate
pub(crate) fn validate_relative_subpath(kind: &str, path: &str) -> Result<(), Error> {
    let candidate = Path::new(path);
    // an absolute path would place files outside the toolbox root entirely
    if candidate.is_absolute() {
        return Err(Error::new(format!(
            "{kind} '{path}' must be a relative path inside the toolbox, not an absolute path"
        )));
    }
    // a `..` component would climb out of the toolbox root
    if candidate
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(Error::new(format!(
            "{kind} '{path}' must stay inside the toolbox (no '..' components)"
        )));
    }
    Ok(())
}

/// A minimal view of a `manifest.toml` for discovering a toolbox's existing images
///
/// Only the fields needed to identify image entries are deserialized; everything else in the
/// manifest is ignored. Used by [`collect_toolbox_images`] to validate `init pipeline -c`
/// references and to detect an `init image -c` duplicate identity.
#[derive(Deserialize)]
struct ManifestProbe {
    /// The resource name
    name: String,
    /// `"image"` or `"pipeline"`
    #[serde(rename = "type")]
    manifest_type: String,
    /// The version label; defaults to `latest` when the manifest omits it
    #[serde(default = "default_probe_version")]
    version: String,
}

/// The default version label for a manifest that omits one (matches `build`'s default)
fn default_probe_version() -> String {
    "latest".to_string()
}

/// Walk a toolbox (the directory of its `config.toml`) for image manifests, mapping each image
/// name to the versions found for it
///
/// Used by `init -c` as the toolbox's resolution source: it lets `init pipeline` confirm a
/// referenced image exists and pin its real version, and `init image` detect a duplicate
/// name+version. Unreadable/unparsable manifests are skipped (a best-effort discovery, not a build).
///
/// # Arguments
///
/// * `config` - The path to the toolbox's `config.toml`
fn collect_toolbox_images(config: &Path) -> HashMap<String, Vec<String>> {
    // the toolbox root is the config's directory (a bare `config.toml` means the cwd)
    let root = config
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(|| Path::new("."), |parent| parent);
    let mut images: HashMap<String, Vec<String>> = HashMap::new();
    // walk every manifest.toml under the toolbox root, recording image (name, version) pairs
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if entry.file_name() != "manifest.toml" {
            continue;
        }
        // skip a manifest that won't read or parse; discovery is best-effort
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(probe) = toml::from_str::<ManifestProbe>(&text) else {
            continue;
        };
        // only images are a resolution source for a pipeline's references
        if probe.manifest_type == "image" {
            images.entry(probe.name).or_default().push(probe.version);
        }
    }
    images
}

/// Pick the version to pin for a referenced image found in the toolbox: prefer `latest`, else the
/// first discovered version
///
/// # Arguments
///
/// * `versions` - The versions discovered for the image in the toolbox
fn pin_version(versions: &[String]) -> String {
    versions
        .iter()
        .find(|version| version.as_str() == "latest")
        .or_else(|| versions.first())
        .cloned()
        .unwrap_or_else(default_probe_version)
}

/// Escape a value for use inside a TOML basic (double-quoted) string
///
/// `config.toml`'s `name`/`registry` are free-form (spaces, slashes), so they
/// can't go through [`validate_resource_name`]; escaping instead keeps a stray
/// quote or newline from corrupting the generated TOML.
///
/// # Arguments
///
/// * `value` - The raw string to escape
pub(crate) fn toml_escape(value: &str) -> String {
    // backslash must be escaped first so the escapes introduced for the other
    // characters below aren't themselves doubled by a later pass
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Render a resource name as a single TOML table-key segment
///
/// A non-empty name composed only of TOML bare-key characters (ASCII letters, digits, `-`, `_`) is
/// emitted unquoted — which every API-validated image/pipeline name is, since those are bounded to
/// 1–25 lowercase alphanumeric or `-` characters. Anything else (e.g. a hand-edited manifest with a
/// dot, space, or other special character) falls back to a quoted, escaped basic-string key so the
/// generated TOML stays valid and unambiguous (an unquoted dot would be parsed as a table path).
///
/// # Arguments
///
/// * `name` - The resource name to render as a key segment
pub(crate) fn toml_key(name: &str) -> String {
    // a non-empty name of only bare-key characters can be written without quotes
    let bare = !name.is_empty()
        && name
            .chars()
            .all(|chr| chr.is_ascii_alphanumeric() || chr == '-' || chr == '_');
    if bare {
        name.to_string()
    } else {
        // fall back to a quoted basic-string key, escaping anything that would break the quoting
        format!("\"{}\"", toml_escape(name))
    }
}

/// Render a toolbox `config.toml` from its toolbox-wide settings
///
/// Shared by `toolbox init` and `toolbox export` so the two can't drift. An unset
/// `registry`, empty `registries`, an unset `image_path_prefix`, and unset
/// `export_image_path`/`export_pipeline_path` are emitted as commented-out placeholders to
/// document the available knobs; `bundled_images` is only written when true.
///
/// # Arguments
///
/// * `name` - The toolbox name
/// * `registry` - The primary container registry, or `None` to leave it unset
/// * `registries` - Extra registries to additionally tag for
/// * `image_path_prefix` - The default bundled-image registry base path, if any
/// * `export_image_path` - The dir `export` writes image tool dirs under, or `None` for `images`
/// * `export_pipeline_path` - The dir `export` writes pipeline tool dirs under, or `None` for `pipelines`
/// * `bundled_images` - Whether the toolbox bundles image tarballs
/// * `base_image` - The toolbox-wide default base-image configuration, if any
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_config_toml(
    name: &str,
    registry: Option<&str>,
    registries: &[String],
    image_path_prefix: Option<&str>,
    export_image_path: Option<&str>,
    export_pipeline_path: Option<&str>,
    bundled_images: bool,
    base_image: Option<&BaseImage>,
) -> String {
    // name is the one always-present required key, so it anchors the top of the file
    let mut out = format!("name = \"{}\"\n", toml_escape(name));
    // a set registry is written out; an unset one is a commented placeholder so the
    // generated config documents the knob without forcing a (possibly wrong) value
    match registry {
        Some(registry) => out.push_str(&format!("registry = \"{}\"\n", toml_escape(registry))),
        None => out.push_str("# registry = \"\"\n"),
    }
    // emit extra registries as a real array only when present; otherwise a commented
    // empty-array placeholder documents the knob
    if registries.is_empty() {
        out.push_str("# registries = []\n");
    } else {
        let quoted: Vec<String> = registries
            .iter()
            .map(|registry| format!("\"{}\"", toml_escape(registry)))
            .collect();
        out.push_str(&format!("registries = [{}]\n", quoted.join(", ")));
    }
    // only write bundled_images when true; the false default is left implicit rather
    // than spelled out as a placeholder
    if bundled_images {
        out.push_str("bundled_images = true\n");
    }
    // a set prefix is written out; an unset one is a commented placeholder
    match image_path_prefix {
        Some(prefix) => out.push_str(&format!(
            "image_path_prefix = \"{}\"\n",
            toml_escape(prefix)
        )),
        None => out.push_str("# image_path_prefix = \"\"\n"),
    }
    // export layout dirs: a set value is written out; an unset one is a commented placeholder
    // documenting the default (export writes under `images`/`pipelines` when unset)
    match export_image_path {
        Some(path) => out.push_str(&format!("export_image_path = \"{}\"\n", toml_escape(path))),
        None => out.push_str("# export_image_path = \"images\"\n"),
    }
    match export_pipeline_path {
        Some(path) => out.push_str(&format!(
            "export_pipeline_path = \"{}\"\n",
            toml_escape(path)
        )),
        None => out.push_str("# export_pipeline_path = \"pipelines\"\n"),
    }
    // the base-image config is a TOML table, so it must come after every scalar key; an unset one
    // is a commented placeholder documenting the knobs
    match base_image {
        Some(base) => {
            // open the table; the blank line keeps it visually separate from the scalars above
            out.push_str("\n[base_image]\n");
            // each base-image field is optional, so only emit the ones that are set
            if let Some(image) = &base.image {
                out.push_str(&format!("image = \"{}\"\n", toml_escape(image)));
            }
            if let Some(image_arg) = &base.image_arg {
                out.push_str(&format!("image_arg = \"{}\"\n", toml_escape(image_arg)));
            }
            if let Some(token) = &base.token {
                out.push_str(&format!("token = \"{}\"\n", toml_escape(token)));
            }
            if let Some(user) = &base.user {
                out.push_str(&format!("user = \"{}\"\n", toml_escape(user)));
            }
            // allow_override is a bool, not a string, so it is written without quoting/escaping
            if let Some(allow) = base.allow_override {
                out.push_str(&format!("allow_override = {allow}\n"));
            }
        }
        None => out.push_str(
            "\n# [base_image]\n\
             # image = \"\"\n\
             # image_arg = \"IMAGE\"\n\
             # token = \"\"\n\
             # user = \"\"\n\
             # allow_override = true\n",
        ),
    }
    out
}

// ─── Config Builders ─────────────────────────────────────────────────────────

/// Renders the default image config JSON, with every `ImageRequest` field present
///
/// Emits all fields (including ones carrying default values) so the scaffolded
/// `<name>.json` is a complete, editable starting point.
///
/// # Arguments
///
/// * `answers` - The wizard answers seeding the config's identity and key fields
fn build_image_config(answers: &ImageConfigAnswers) -> String {
    // build the full ImageRequest shape with every field spelled out (even defaulted
    // ones) so the scaffolded file is a complete, editable reference; the wizard answers
    // seed identity and the few interactively chosen fields
    serde_json::to_string_pretty(&serde_json::json!({
        "group": answers.group,
        "name": answers.name,
        "version": null,
        "scaler": answers.scaler,
        "image": answers.image_tag,
        "lifetime": null,
        "modifiers": null,
        "timeout": answers.timeout,
        "resources": {
            "cpu": answers.cpu,
            "memory": answers.memory,
            "ephemeral_storage": "0Mi",
            "nvidia_gpu": 0,
            "amd_gpu": 0
        },
        "spawn_limit": "Unlimited",
        "volumes": [],
        "env": {},
        "args": {
            "entrypoint": null,
            "command": null,
            "reaction": null,
            "repo": null,
            "commit": null,
            "output": "None"
        },
        "description": answers.description,
        "security_context": {
            "user": null,
            "group": null,
            "allow_privilege_escalation": false
        },
        "collect_logs": true,
        "generator": answers.generator,
        "dependencies": {
            "samples": {
                "location": "/tmp/thorium/samples",
                "kwarg": null,
                "strategy": "Paths"
            },
            "ephemeral": {
                "location": "/tmp/thorium/ephemeral",
                "kwarg": null,
                "strategy": "Paths",
                "names": []
            },
            "results": {
                "images": [],
                "location": "/tmp/thorium/prior-results",
                "kwarg": "None",
                "strategy": "Paths",
                "names": []
            },
            "repos": {
                "location": "/tmp/thorium/repos",
                "kwarg": null,
                "strategy": "Paths"
            },
            "tags": {
                "enabled": false,
                "location": "/tmp/thorium/prior-tags",
                "kwarg": null,
                "strategy": "Paths"
            },
            "children": {
                "enabled": false,
                "images": [],
                "location": "/tmp/thorium/prior-children",
                "kwarg": null,
                "strategy": "Paths"
            }
        },
        "display_type": answers.display_type,
        "output_collection": {
            "handler": "Files",
            "files": {
                "results": "/tmp/thorium/results",
                "result_files": "/tmp/thorium/result-files",
                "tags": "/tmp/thorium/tags",
                "names": []
            },
            "children": "/tmp/thorium/children",
            "auto_tag": {},
            "groups": []
        },
        "child_filters": {
            "mime": [],
            "file_name": [],
            "file_extension": [],
            "submit_non_matches": false
        },
        "clean_up": null,
        "kvm": null,
        "network_policies": []
    }))
    // the template is a fixed shape built from owned strings, so serialization cannot fail
    .expect("static JSON template must serialize")
}

/// Renders the default pipeline config JSON, with every `PipelineRequest` field present
///
/// # Arguments
///
/// * `answers` - The wizard answers seeding the config's identity and order/sla
fn build_pipeline_config(answers: &PipelineConfigAnswers) -> String {
    // emit the full PipelineRequest shape; triggers starts empty for the user to fill in
    serde_json::to_string_pretty(&serde_json::json!({
        "group": answers.group,
        "name": answers.name,
        "order": answers.order,
        "sla": answers.sla,
        "triggers": {},
        "description": answers.description
    }))
    // the template is a fixed shape built from owned values, so serialization cannot fail
    .expect("static JSON template must serialize")
}

// ─── Manifest Generators ────────────────────────────────────────────────────

/// Renders an image's `manifest.toml` from its identity and build settings
///
/// # Arguments
///
/// * `name` - The tool name
/// * `image_name_field` - The `image_name` manifest field (already TOML-escaped)
/// * `version` - The image version
/// * `no_build` - Whether to mark the image as not built by CI (`build = false`)
/// * `policy_files` - Bundled network policy definition files to reference
/// * `exported_image_path` - The real registry url an export captured, if any
pub(crate) fn generate_image_manifest(
    name: &str,
    image_name_field: &str,
    version: &str,
    no_build: bool,
    policy_files: &[String],
    exported_image_path: Option<&str>,
) -> String {
    // a Thorium version is a free-form Custom(String) on export, so escape it before it
    // goes into a TOML basic string; a stray quote/newline would otherwise corrupt the
    // generated manifest (name/image_name_field are already validated/escaped by callers)
    let version = toml_escape(version);
    // lay down the required scalar keys first; config_from points at the sibling JSON
    // named after the tool, and build_path defaults to the manifest's own directory
    let mut manifest = format!(
        "name = \"{name}\"\n\
         type = \"image\"\n\
         config_from = \"{name}.json\"\n\
         build_path = \"./\"\n\
         image_name = \"{image_name_field}\"\n\
         version = \"{version}\"\n"
    );
    // record the real registry url an export captured, so a build that doesn't
    // rebuild this image (build = false) keeps it instead of deriving a path
    if let Some(path) = exported_image_path.filter(|path| !path.is_empty()) {
        manifest.push_str(&format!(
            "exported_image_path = \"{}\"\n",
            toml_escape(path)
        ));
    }
    // reference any bundled network policy definition files
    if !policy_files.is_empty() {
        let quoted: Vec<String> = policy_files
            .iter()
            .map(|file| format!("\"{file}\""))
            .collect();
        manifest.push_str(&format!(
            "network_policies_from = [{}]\n",
            quoted.join(", ")
        ));
    }
    // write an explicit `build = false` when the image is reference-only; otherwise leave
    // the `true` default as a commented hint documenting how to flip it
    if no_build {
        manifest.push_str("build = false\n");
    } else {
        manifest.push_str("# build = true\n");
    }
    // a commented per-tool base-image config (a TOML table, so it trails the scalar keys); token
    // and user are CI/CD variable names, not used by `build-images`
    manifest.push_str(
        "\n\
        # [base_image]\n\
        # image = \"\"\n\
        # image_arg = \"IMAGE\"\n\
        # token = \"\"\n\
        # user = \"\"\n\
        # allow_override = true\n",
    );
    manifest
}

/// Renders a pipeline's `manifest.toml` from its name and referenced images
///
/// # Arguments
///
/// * `name` - The pipeline name
/// * `images` - The (image name, version) pairs the pipeline references
pub(crate) fn generate_pipeline_manifest(name: &str, images: &[(String, String)]) -> String {
    // lay down the required scalar keys; config_from points at the sibling JSON config
    let mut manifest = format!(
        "name = \"{name}\"\n\
         type = \"pipeline\"\n\
         version = \"latest\"\n\
         config_from = \"{name}.json\"\n"
    );
    // append an [images.<name>] table per referenced image so the manifest's image map mirrors the
    // images the pipeline's order runs. The name is rendered as a bare key when it is TOML-bare-safe
    // (every API-validated name is — 1–25 lowercase alphanumeric or '-'), only falling back to a
    // quoted/escaped key for a hand-edited name with a special character; the version is always a
    // basic string, escaped defensively against a stray quote/newline
    for (image_name, version) in images {
        manifest.push_str(&format!(
            "\n[images.{}]\nversion = \"{}\"\n",
            toml_key(image_name),
            toml_escape(version)
        ));
    }
    manifest
}

// ─── Shared Write Helpers ────────────────────────────────────────────────────

/// Writes an image's `manifest.toml`, `<name>.json`, and `description.md`
///
/// Builds the default config, optionally opens it in the editor for review, then
/// re-reads identity from the saved config so the manifest and filename stay
/// consistent with whatever the user kept.
///
/// # Arguments
///
/// * `path` - The image directory to write into
/// * `answers` - The wizard answers seeding the config
/// * `overwrite` - Overwrite existing files instead of skipping them
/// * `open_editor` - Open the config in the editor before writing (interactive mode)
/// * `editor` - The editor command to open the config with
async fn write_image_files(
    path: &Path,
    answers: &ImageConfigAnswers,
    overwrite: bool,
    open_editor: bool,
    editor: &str,
) -> Result<(), Error> {
    // build the default config, then (interactively) let the user fill it in via
    // the editor — the editor edits exactly what is written to <name>.json
    let config_json = build_image_config(answers);
    let final_json = if open_editor {
        editor::review_config_in_editor::<ImageRequest>(
            &config_json,
            &format!("init-image-{}", answers.name),
            editor,
            IMAGE_FIELD_ORDER,
        )
        .await?
    } else {
        // non-interactive: emit the default in curated key order with all fields present,
        // matching the layout the editor path would have produced
        let value: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| Error::new(format!("Invalid default image config: {e}")))?;
        crate::utils::curated_json(&value, IMAGE_FIELD_ORDER)?
    };
    // re-parse the final JSON to read identity back out; the editor path may have changed
    // name/group, so the manifest and filename must follow the saved file, not the answers
    let value: serde_json::Value = serde_json::from_str(&final_json).map_err(|e| {
        Error::new(format!(
            "image config for '{}' is not valid JSON: {e}",
            path.display()
        ))
    })?;
    // identity comes from the saved config, not the wizard answers, so a name/group the
    // user changed in the editor still drives the manifest and filename
    let name = json_str_field(&value, "name").ok_or_else(|| {
        Error::new(format!(
            "image config for '{}' is missing a 'name' field",
            path.display()
        ))
    })?;
    let group = json_str_field(&value, "group").ok_or_else(|| {
        Error::new(format!(
            "image config for '{}' is missing a 'group' field",
            path.display()
        ))
    })?;
    // reject names that aren't valid identifiers before interpolating them into the
    // TOML template, since an editor-supplied name has not been through prompt validation
    validate_resource_name("image", &name)?;
    validate_resource_name("group", &group)?;
    let manifest = generate_image_manifest(
        &name,
        // image_name may be a path (slashes), so escape it before it lands in the
        // `image_name = "…"` TOML template
        &toml_escape(&answers.image_name),
        "latest",
        answers.no_build,
        &[],
        // newly scaffolded images are built, not exported, so no pinned registry path
        None,
    );
    // write the manifest and the JSON config under the config's own name so the two stay
    // in lockstep with whatever identity the user saved
    write_file(&path.join("manifest.toml"), &manifest, overwrite).await?;
    write_file(&path.join(format!("{name}.json")), &final_json, overwrite).await?;
    // description.md is the source of truth toolbox build injects, so seed it from
    // the (possibly edited) config description; an empty description yields a bare stub
    let description = json_str_field(&value, "description").filter(|d| !d.is_empty());
    let description_md = description_stub(&name, description.as_deref());
    write_file(&path.join("description.md"), &description_md, overwrite).await?;
    Ok(())
}

/// Read a string field from a JSON config value, if present and a string
///
/// # Arguments
///
/// * `value` - The JSON config value to read from
/// * `field` - The name of the field to read
fn json_str_field(value: &serde_json::Value, field: &str) -> Option<String> {
    // None unless the key exists and holds a string; a missing key or non-string value
    // both collapse to None so callers can treat "absent" and "wrong type" alike
    value
        .get(field)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Collect the unique image names referenced across a pipeline config's `order`,
/// preserving first-seen order. Used to keep the manifest's image map in sync with
/// an order edited in the editor.
///
/// Both order forms are accepted: the flat form (`["a", "b"]`, a single implicit
/// stage) and the staged form (`[["a", "b"], ["c"]]`). A hand-edited pipeline config
/// can legitimately use the flat form, and silently treating it as no images would
/// leave the generated manifest's `[images.*]` map empty so the rebuilt pipeline
/// wouldn't declare the images it actually runs.
///
/// # Arguments
///
/// * `value` - The pipeline config value whose `order` is scanned for image names
fn unique_order_images(value: &serde_json::Value) -> Vec<String> {
    // `seen` dedupes while `images` preserves first-seen order, since a HashSet alone
    // would lose the ordering the manifest map should reflect
    let mut seen = std::collections::HashSet::new();
    let mut images = Vec::new();
    // record an image name the first time it appears; insert() is false on a repeat so
    // duplicates within or across stages are dropped while order is preserved
    let mut record = |name: &str| {
        if seen.insert(name.to_string()) {
            images.push(name.to_string());
        }
    };
    // a non-array order (or an absent one) simply yields no images rather than erroring
    if let Some(order) = value.get("order").and_then(|o| o.as_array()) {
        for entry in order {
            match entry {
                // a flat entry is itself an image name
                serde_json::Value::String(name) => record(name),
                // a staged entry is an array of image names; non-string members are skipped
                serde_json::Value::Array(stage) => {
                    for image in stage {
                        if let Some(name) = image.as_str() {
                            record(name);
                        }
                    }
                }
                // anything else is malformed; skip it rather than aborting
                _ => {}
            }
        }
    }
    images
}

/// Resolve the editor for an init subcommand: an explicit `--editor` override,
/// else the configured `default_editor`, else the built-in default. Init is
/// offline, so the config is loaded best-effort.
///
/// # Arguments
///
/// * `editor_override` - The explicit `--editor` value, if given
/// * `args` - The top-level thorctl args (used to locate the config)
fn resolve_editor(editor_override: Option<&str>, args: &Args) -> String {
    // an explicit --editor always wins
    if let Some(editor) = editor_override {
        return editor.to_string();
    }
    // fall back to the config's default_editor; init is offline so a missing/unreadable
    // config is non-fatal — swallow the error and drop through to the built-in default
    if let Ok(conf) = thorium::CtlConf::from_path(&args.config) {
        return conf.default_editor;
    }
    // last resort when there is no override and no usable config
    thorium::client::conf::default_default_editor()
}

/// Resolve the group for an init subcommand from `--group` or an interactive prompt
///
/// # Arguments
///
/// * `group` - The explicit `--group` value, if given
/// * `non_interactive` - Whether `--non-interactive` is set (errors instead of prompting)
fn resolve_group(group: &Option<String>, non_interactive: bool) -> Result<String, Error> {
    match group {
        // an explicit --group is taken verbatim (it is validated later before use)
        Some(group) => Ok(group.clone()),
        // non-interactive can't prompt, so a missing group is a hard error rather than
        // silently defaulting to some group the user didn't choose
        None if non_interactive => Err(Error::new(
            "--group is required in non-interactive mode".to_string(),
        )),
        // interactive: ask the user for the group
        None => prompt::prompt_group_name("Group name"),
    }
}

/// Build the starting contents for a scaffolded description.md
///
/// Seeds from the wizard's description answer when one was given so the
/// markdown file (the source of truth for `toolbox build`) starts in sync.
///
/// # Arguments
///
/// * `name` - The tool's name
/// * `description` - The description entered in the wizard, if any
fn description_stub(name: &str, description: Option<&str>) -> String {
    // treat a blank description as absent so an empty wizard answer doesn't add a stray
    // empty line under the heading
    match description.filter(|d| !d.is_empty()) {
        // an existing description goes under the Overview heading
        Some(description) => format!("# {name}\n\n# Overview\n\n{description}\n"),
        // no description: emit just the title and an empty Overview section for the user
        None => format!("# {name}\n\n# Overview\n"),
    }
}

/// Writes a pipeline's `manifest.toml`, `<name>.json`, and `description.md`
///
/// Builds the default config, optionally opens it in the editor for review, then
/// derives the manifest's image map from the (possibly edited) config order.
///
/// # Arguments
///
/// * `path` - The pipeline directory to write into
/// * `answers` - The wizard answers seeding the config
/// * `overwrite` - Overwrite existing files instead of skipping them
/// * `open_editor` - Open the config in the editor before writing (interactive mode)
/// * `editor` - The editor command to open the config with
/// * `toolbox` - When `-c` is set, the toolbox's `config.toml` path and its discovered images;
///   every referenced image must exist there (else an error) and is version-pinned from it.
///   `None` pins each referenced image to `latest` (no validation source).
async fn write_pipeline_files(
    path: &Path,
    answers: &PipelineConfigAnswers,
    overwrite: bool,
    open_editor: bool,
    editor: &str,
    toolbox: Option<(&Path, &HashMap<String, Vec<String>>)>,
) -> Result<(), Error> {
    // build the default config, then (interactively) let the user fill it in via the
    // editor — the editor edits exactly what is written to <name>.json
    let config_json = build_pipeline_config(answers);
    let final_json = if open_editor {
        editor::review_config_in_editor::<PipelineRequest>(
            &config_json,
            &format!("init-pipeline-{}", answers.name),
            editor,
            PIPELINE_FIELD_ORDER,
        )
        .await?
    } else {
        // non-interactive: emit the default in curated key order with all fields present
        let value: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| Error::new(format!("Invalid default pipeline config: {e}")))?;
        crate::utils::curated_json(&value, PIPELINE_FIELD_ORDER)?
    };
    // re-parse the saved JSON so identity and the (possibly edited) order drive the
    // manifest, not the original wizard answers
    let value: serde_json::Value = serde_json::from_str(&final_json).map_err(|e| {
        Error::new(format!(
            "pipeline config for '{}' is not valid JSON: {e}",
            path.display()
        ))
    })?;
    // identity comes from the saved config so an edited name/group still drives the
    // manifest and filename
    let name = json_str_field(&value, "name").ok_or_else(|| {
        Error::new(format!(
            "pipeline config for '{}' is missing a 'name' field",
            path.display()
        ))
    })?;
    let group = json_str_field(&value, "group").ok_or_else(|| {
        Error::new(format!(
            "pipeline config for '{}' is missing a 'group' field",
            path.display()
        ))
    })?;
    // reject names that aren't valid identifiers before interpolating into the TOML template
    validate_resource_name("pipeline", &name)?;
    validate_resource_name("group", &group)?;
    // derive the manifest's image map from the config's order so editing the order in the editor
    // keeps the manifest's referenced images in sync. With a toolbox (-c) each image must exist
    // there (hard error otherwise — init never creates images) and is pinned to the toolbox's
    // version; without one, each is pinned to "latest" since the order carries names only.
    let images: Vec<(String, String)> = unique_order_images(&value)
        .into_iter()
        .map(|image| match toolbox {
            Some((config, available)) => match available.get(&image) {
                Some(versions) => Ok((image, pin_version(versions))),
                None => Err(Error::new(format!(
                    "pipeline references image '{image}', which is not in the toolbox at '{}'; \
                     add it (e.g. with `thorctl toolbox init image`) before building — init does \
                     not create it",
                    config.display()
                ))),
            },
            None => Ok((image, "latest".to_string())),
        })
        .collect::<Result<_, Error>>()?;
    // image names pulled from the order are interpolated into TOML table headers, so they
    // too must be valid identifiers
    for (image, _) in &images {
        validate_resource_name("image", image)?;
    }
    // write the manifest and JSON config keyed on the saved name so they stay in lockstep
    let manifest = generate_pipeline_manifest(&name, &images);
    write_file(&path.join("manifest.toml"), &manifest, overwrite).await?;
    write_file(&path.join(format!("{name}.json")), &final_json, overwrite).await?;
    // seed description.md from the (possibly edited) config description, blank treated as absent
    let description = json_str_field(&value, "description").filter(|d| !d.is_empty());
    let description_md = description_stub(&name, description.as_deref());
    write_file(&path.join("description.md"), &description_md, overwrite).await?;
    Ok(())
}

// ─── Subcommand Dispatch ─────────────────────────────────────────────────────

/// Dispatches an `init` subcommand to its scaffolding handler
///
/// # Arguments
///
/// * `cmd` - The init subcommand (toolbox, image, or pipeline)
/// * `args` - The top-level thorctl args
pub async fn handle(cmd: &Init, args: &Args) -> Result<(), Error> {
    // route each init variant to its dedicated scaffolder
    match cmd {
        Init::Toolbox(cmd) => init_toolbox(cmd, args).await,
        Init::Image(cmd) => init_image(cmd, args).await,
        Init::Pipeline(cmd) => init_pipeline(cmd, args).await,
    }
}

/// Scaffolds a single image directory from `init image` args
///
/// # Arguments
///
/// * `cmd` - The `init image` args
/// * `args` - The top-level thorctl args
async fn init_image(cmd: &InitImage, args: &Args) -> Result<(), Error> {
    // the tool name defaults to the target directory's basename
    let dir = dir_name(&cmd.path)?;
    // the manifest image_name defaults to the build directory name, overridable
    // via --image-name
    let default_image_name = cmd.image_name.clone().unwrap_or_else(|| dir.clone());
    // resolve the group up front (prompt or --group) so it seeds the config answers
    let group = resolve_group(&cmd.group, cmd.non_interactive)?;
    // with -c, refuse to scaffold an image whose name+version already exists in the toolbox (unless
    // --overwrite), catching the duplicate identity now rather than as a build-time error. The
    // scaffold writes the `latest` version. `-c` is a resolution source only — it never moves files.
    if let Some(config) = &cmd.config {
        let available = collect_toolbox_images(config);
        if !cmd.overwrite
            && available
                .get(&dir)
                .is_some_and(|versions| versions.iter().any(|version| version == "latest"))
        {
            return Err(Error::new(format!(
                "image '{dir}:latest' already exists in the toolbox at '{}'; pass --overwrite to \
                 replace it",
                config.display()
            )));
        }
    }
    // seed the wizard answers from the resolved defaults; no_build flows into build=false
    let answers = ImageConfigAnswers::defaults(&dir, &group, cmd.no_build, &default_image_name);
    // pick the editor only matters in interactive mode but is resolved unconditionally
    let editor = resolve_editor(cmd.editor.as_deref(), args);
    // interactive mode (the negation of --non-interactive) opens the editor before writing
    write_image_files(
        &cmd.path,
        &answers,
        cmd.overwrite,
        !cmd.non_interactive,
        &editor,
    )
    .await?;
    // point the user at the next step now that the image directory exists
    println!(
        "\n{} Add this image to a toolbox's config.toml, then run {} to produce a toolbox.json",
        "Done!".bright_green(),
        "thorctl toolbox build".bright_cyan()
    );
    Ok(())
}

/// Scaffolds a single pipeline directory from `init pipeline` args
///
/// # Arguments
///
/// * `cmd` - The `init pipeline` args
/// * `args` - The top-level thorctl args
async fn init_pipeline(cmd: &InitPipeline, args: &Args) -> Result<(), Error> {
    // the pipeline name defaults to the target directory's basename
    let dir = dir_name(&cmd.path)?;
    // resolve the group (prompt or --group) before building the answers
    let group = resolve_group(&cmd.group, cmd.non_interactive)?;
    // non-interactive can't prompt for images and the default order is built from them,
    // so an empty --images would scaffold an empty pipeline — reject it instead
    if cmd.non_interactive && cmd.images.is_empty() {
        return Err(Error::new(
            "--images is required in non-interactive mode".to_string(),
        ));
    }
    // defaults put every --images entry into a single parallel stage as the order
    let mut answers = PipelineConfigAnswers::defaults(&dir, &group, &cmd.images);
    // an explicit --order replaces that default ordering
    if let Some(order_str) = &cmd.order {
        answers.order = serde_json::from_str(order_str)
            .map_err(|e| Error::new(format!("Invalid --order JSON: {e}")))?;
        // every image named in --order must also be declared in --images; otherwise the manifest
        // would carry a dangling, version-less image entry, so this is a hard error (below).
        // flatten collapses the staged order to the set of all named images
        let ordered: std::collections::HashSet<&str> =
            answers.order.iter().flatten().map(String::as_str).collect();
        let provided: std::collections::HashSet<&str> =
            cmd.images.iter().map(String::as_str).collect();
        // images that appear in the order but were never listed in --images
        let mut unlisted: Vec<&str> = ordered.difference(&provided).copied().collect();
        // every ordered image must be declared in --images, so the manifest never carries a
        // dangling, version-less image entry; a stray order entry is a hard error
        if !unlisted.is_empty() {
            unlisted.sort_unstable();
            return Err(Error::new(format!(
                "--order references image(s) not in --images: {}; add them to --images (a pipeline \
                 may only reference images it declares)",
                unlisted.join(", ")
            )));
        }
    }
    // with -c, the toolbox is the resolution source: referenced images must exist in it (else an
    // error) and are version-pinned from it. Walk it once up front and announce the source.
    let toolbox_images = match &cmd.config {
        Some(config) => {
            println!(
                "Resolving pipeline images against toolbox '{}'",
                config.display()
            );
            Some((config.as_path(), collect_toolbox_images(config)))
        }
        None => None,
    };
    let toolbox = toolbox_images
        .as_ref()
        .map(|(config, images)| (*config, images));
    // editor is only consulted in interactive mode but resolved unconditionally
    let editor = resolve_editor(cmd.editor.as_deref(), args);
    // interactive mode (the negation of --non-interactive) opens the editor before writing
    write_pipeline_files(
        &cmd.path,
        &answers,
        cmd.overwrite,
        !cmd.non_interactive,
        &editor,
        toolbox,
    )
    .await?;
    // point the user at the next step now that the pipeline directory exists
    println!(
        "\n{} Add this pipeline to a toolbox's config.toml, then run {} to produce a toolbox.json",
        "Done!".bright_green(),
        "thorctl toolbox build".bright_cyan()
    );
    Ok(())
}

/// Scaffolds a full toolbox directory (config.toml plus image/pipeline subdirs)
///
/// # Arguments
///
/// * `cmd` - The `init toolbox` args
/// * `args` - The top-level thorctl args
async fn init_toolbox(cmd: &InitToolbox, args: &Args) -> Result<(), Error> {
    // derive each image's tool name from its directory basename; collected eagerly so a
    // bad path fails before anything is written, and reused as the default pipeline binding
    let image_names: Vec<String> = cmd
        .images
        .iter()
        .map(|p| dir_name(p))
        .collect::<Result<_, _>>()?;
    // parse each --pipeline string into its path and optional colon-bound image list
    let pipeline_specs: Vec<PipelineSpec> = cmd
        .pipelines
        .iter()
        .map(|s| PipelineSpec::parse(s))
        .collect();
    // the export-layout dirs are written into config.toml and later used to place files, so
    // reject anything that would escape the toolbox root before writing the config
    if let Some(path) = &cmd.image_path {
        validate_relative_subpath("--image-path", path)?;
    }
    if let Some(path) = &cmd.pipeline_path {
        validate_relative_subpath("--pipeline-path", path)?;
    }
    let config_toml = if let Some(config_path) = &cmd.config {
        // seed the new toolbox from an existing config.toml (mutually exclusive with
        // --name/--registry); carries name, registry, registries, image_path_prefix,
        // export paths, and bundled_images forward verbatim
        let template = super::build::load_config(config_path)?;
        render_config_toml(
            &template.name,
            template.registry.as_deref(),
            &template.registries,
            template.image_path_prefix.as_deref(),
            template.export_image_path.as_deref(),
            template.export_pipeline_path.as_deref(),
            template.bundled_images,
            template.base_image.as_ref(),
        )
    } else {
        // no --config: take name/registry from flags non-interactively, else prompt for them
        let (tb_name, tb_registry) = if cmd.non_interactive {
            (cmd.name.clone(), cmd.registry.clone())
        } else {
            let tb = prompt::prompt_toolbox_config(&cmd.name, cmd.registry.as_deref())?;
            (tb.name, tb.registry)
        };
        // a from-scratch config has no extra registries, prefix, bundling, or base image; the
        // export-layout dirs come from --image-path/--pipeline-path (commented defaults when unset)
        render_config_toml(
            &tb_name,
            tb_registry.as_deref(),
            &[],
            None,
            cmd.image_path.as_deref(),
            cmd.pipeline_path.as_deref(),
            false,
            None,
        )
    };
    // write config.toml at the toolbox root before scaffolding the per-tool dirs. config.toml is
    // sticky: an existing one is preserved unless --overwrite-config, so re-running init in a
    // toolbox doesn't clobber its settings (per-tool files use --overwrite, handled below).
    let config_path = cmd.toolbox_dir.join("config.toml");
    if config_path.exists() && !cmd.overwrite_config {
        println!(
            "{} {} (already exists; pass --overwrite-config to replace)",
            "Skipped".bright_yellow(),
            config_path.display()
        );
        // warn that the settings flags are ignored while the existing config is kept
        if cmd.config.is_some()
            || cmd.registry.is_some()
            || cmd.image_path.is_some()
            || cmd.pipeline_path.is_some()
        {
            println!(
                "{} keeping the existing config.toml; --config/--registry/--image-path/\
                 --pipeline-path are ignored (pass --overwrite-config to apply them)",
                "Warning:".bright_yellow()
            );
        }
    } else {
        // create-on-fresh, or replace when --overwrite-config; force the write since the sticky
        // check above already governs whether we reach here
        write_file(&config_path, &config_toml, true).await?;
    }
    // one group is resolved once and shared by every scaffolded image and pipeline
    let group = resolve_group(&cmd.group, cmd.non_interactive)?;
    // editor and the interactive flag are computed once and threaded into each write
    let editor = resolve_editor(cmd.editor.as_deref(), args);
    let open_editor = !cmd.non_interactive;
    // scaffold each image dir, pairing the original path with its derived tool name
    for (image_path, image_name) in cmd.images.iter().zip(&image_names) {
        // the manifest image_name defaults to the image's basename (matching `init image`)
        let answers = ImageConfigAnswers::defaults(image_name, &group, false, image_name);
        write_image_files(image_path, &answers, cmd.overwrite, open_editor, &editor).await?;
    }
    for spec in &pipeline_specs {
        // the pipeline name is its directory basename
        let pipeline_name = dir_name(&spec.path)?;
        // colon-bound images from the spec take precedence; with no colon the pipeline
        // runs every image the toolbox is scaffolding
        let pipeline_images = spec.images.clone().unwrap_or_else(|| image_names.clone());
        // a pipeline that binds an image the toolbox isn't scaffolding (no matching
        // --images entry) builds a pipeline referencing an image that won't exist
        // here; warn rather than fail so the user can wire it up themselves. only the
        // explicit colon-bound case can name a stray image, so skip the check otherwise
        if spec.images.is_some() {
            // bound images with no matching scaffolded --images entry; sorted so the warning lists
            // them deterministically (matching init_pipeline's sorted unlisted output)
            let mut unlisted: Vec<&str> = pipeline_images
                .iter()
                .filter(|img| !image_names.contains(img))
                .map(String::as_str)
                .collect();
            unlisted.sort_unstable();
            if !unlisted.is_empty() {
                println!(
                    "{} pipeline '{}' binds image(s) not in --images: {}",
                    "Warning:".bright_yellow(),
                    pipeline_name,
                    unlisted.join(", ")
                );
            }
        }
        // seed the pipeline's order from its bound images and scaffold its dir. No toolbox
        // resolution source here: init toolbox scaffolds the images itself in the same run, so the
        // pipeline's references pin "latest" (the scaffolded images' version).
        let answers = PipelineConfigAnswers::defaults(&pipeline_name, &group, &pipeline_images);
        write_pipeline_files(
            &spec.path,
            &answers,
            cmd.overwrite,
            open_editor,
            &editor,
            None,
        )
        .await?;
    }
    // point the user at the next step now that the whole toolbox skeleton exists
    println!(
        "\n{} Run {} to produce a toolbox.json",
        "Done!".bright_green(),
        "thorctl toolbox build".bright_cyan()
    );
    Ok(())
}

/// Unit tests for the pure rendering helpers (config TOML, default configs, and the
/// description stub) that don't need filesystem or editor interaction
#[cfg(test)]
mod tests {
    use super::prompt::{ImageConfigAnswers, PipelineConfigAnswers};
    use super::{
        BaseImage, build_image_config, build_pipeline_config, description_stub,
        generate_image_manifest, generate_pipeline_manifest, render_config_toml,
        unique_order_images,
    };

    /// The scaffolded description.md is just the tool name + an Overview section (no
    /// placeholder prose); an existing description is placed under Overview
    #[test]
    fn description_stub_uses_overview_section() {
        // an absent description yields just the title and an empty Overview, no placeholder prose
        let empty = description_stub("clamav", None);
        assert_eq!(empty, "# clamav\n\n# Overview\n");
        assert!(!empty.contains("Describe what this tool"));
        // a present description is placed under the Overview heading
        let with_desc = description_stub("clamav", Some("scans files"));
        assert_eq!(with_desc, "# clamav\n\n# Overview\n\nscans files\n");
    }

    /// The scaffolded image default must deserialize into the real `ImageRequest`
    /// (the type `toolbox import` and the editor validation parse it as) and include
    /// the `version`/`lifetime`/`modifiers` fields so the saved config is complete
    #[test]
    fn image_template_deserializes_into_request() {
        // the scaffolded default must parse as the real ImageRequest the importer/editor use
        let answers = ImageConfigAnswers::defaults("clamav", "static", false, "clamav");
        let json = build_image_config(&answers);
        serde_json::from_str::<thorium::models::ImageRequest>(&json)
            .expect("default image config must deserialize into ImageRequest");
        // re-parse as untyped JSON to assert the explicitly-null fields are present, since
        // ImageRequest deserialization alone wouldn't catch a dropped key
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        for key in ["version", "lifetime", "modifiers"] {
            assert!(value.get(key).is_some(), "image template missing '{key}'");
        }
    }

    /// The scaffolded pipeline default must deserialize into the real `PipelineRequest`
    #[test]
    fn pipeline_template_deserializes_into_request() {
        // the scaffolded default must parse as the real PipelineRequest the importer/editor use
        let answers = PipelineConfigAnswers::defaults("triage", "static", &["clamav".to_string()]);
        let json = build_pipeline_config(&answers);
        serde_json::from_str::<thorium::models::PipelineRequest>(&json)
            .expect("default pipeline config must deserialize into PipelineRequest");
    }

    /// Version pinning for a referenced toolbox image prefers `latest`, else the first discovered
    #[test]
    fn pin_version_prefers_latest() {
        use super::pin_version;
        // latest wins even when listed after another version
        assert_eq!(
            pin_version(&["1.0".to_string(), "latest".to_string()]),
            "latest"
        );
        // with no latest, the first discovered version is pinned
        assert_eq!(pin_version(&["2.1".to_string(), "2.0".to_string()]), "2.1");
        // an empty set falls back to latest (defensive; callers pass non-empty)
        assert_eq!(pin_version(&[]), "latest");
    }

    /// The order is scanned for image names in both the staged and the flat form, so a
    /// hand-edited pipeline config using either shape keeps the generated manifest's
    /// image map populated (first-seen order, duplicates dropped)
    #[test]
    fn unique_order_images_accepts_flat_and_staged() {
        use serde_json::json;
        // the staged form: an array of stages
        let staged = json!({ "order": [["a", "b"], ["a", "c"]] });
        assert_eq!(unique_order_images(&staged), vec!["a", "b", "c"]);
        // the flat form: a single implicit stage of image names
        let flat = json!({ "order": ["a", "b", "a", "c"] });
        assert_eq!(unique_order_images(&flat), vec!["a", "b", "c"]);
        // an absent or non-array order yields no images rather than erroring
        assert!(unique_order_images(&json!({})).is_empty());
        assert!(unique_order_images(&json!({ "order": "nope" })).is_empty());
    }

    /// A free-form version carrying TOML metacharacters (a Custom version captured on
    /// export) must be escaped so the generated image manifest stays valid TOML and the
    /// version round-trips intact rather than breaking out into injected keys
    #[test]
    fn image_manifest_escapes_version() {
        // a version with an embedded quote and newline plus an injected key assignment
        let nasty = "1.0\"\nmalicious = \"pwned";
        let manifest = generate_image_manifest("clamav", "clamav", nasty, false, &[], None);
        // the whole manifest must still parse as TOML
        let parsed: toml::Value =
            toml::from_str(&manifest).expect("escaped image manifest must be valid TOML");
        // the version decodes back to exactly the original string
        assert_eq!(parsed["version"].as_str(), Some(nasty));
        // the injected assignment never became a real top-level key
        assert!(parsed.get("malicious").is_none());
    }

    /// A dotted image name must be emitted as a quoted key (not a nested table) and a
    /// A bare-key-safe name is emitted unquoted while a name with a special character (a dot, which
    /// would otherwise parse as a nested table) falls back to a quoted/escaped key; the version is
    /// always escaped — so the image map stays valid and faithful to the referenced (name, version)
    /// pairs
    #[test]
    fn pipeline_manifest_quotes_only_when_needed() {
        // a normal dashed name is bare-key-safe; a dotted name is not (it would nest as a bare key);
        // the dotted entry's version carries an embedded quote/newline and an injected assignment
        let images = vec![
            ("detect-it-easy".to_string(), "latest".to_string()),
            ("clam.av".to_string(), "1\"\nx = \"y".to_string()),
        ];
        let manifest = generate_pipeline_manifest("triage", &images);
        // the dashed name is written bare; only the dotted name is quoted
        assert!(manifest.contains("[images.detect-it-easy]"));
        assert!(manifest.contains("[images.\"clam.av\"]"));
        // the whole manifest must still parse as TOML
        let parsed: toml::Value =
            toml::from_str(&manifest).expect("escaped pipeline manifest must be valid TOML");
        let imgs = parsed["images"].as_table().expect("images must be a table");
        // the bare name resolves to a single key with its version
        assert_eq!(imgs["detect-it-easy"]["version"].as_str(), Some("latest"));
        // the dotted name is a single key, not a nested images.clam.av sub-table
        assert!(imgs.contains_key("clam.av"));
        assert_eq!(imgs["clam.av"]["version"].as_str(), Some("1\"\nx = \"y"));
        // the injected assignment never escaped into the images table
        assert!(imgs.get("x").is_none());
    }

    /// `toml_key` emits a bare key for bare-key-safe names and a quoted/escaped key otherwise
    #[test]
    fn toml_key_quotes_only_non_bare_names() {
        // ascii alphanumerics, '-', and '_' are valid TOML bare-key characters
        assert_eq!(super::toml_key("detect-it-easy"), "detect-it-easy");
        assert_eq!(super::toml_key("under_score"), "under_score");
        assert_eq!(super::toml_key("Mixed123"), "Mixed123");
        // a dot, a space, or an embedded quote forces a quoted, escaped key
        assert_eq!(super::toml_key("clam.av"), "\"clam.av\"");
        assert_eq!(super::toml_key("two words"), "\"two words\"");
        assert_eq!(super::toml_key("a\"b"), "\"a\\\"b\"");
        // an empty name is never a valid bare key
        assert_eq!(super::toml_key(""), "\"\"");
    }

    /// With no extras, registries and image_path_prefix are emitted as commented
    /// placeholders and bundled_images is omitted
    #[test]
    fn render_config_minimal() {
        // a name + registry with no extras: registries/prefix become commented placeholders
        let toml = render_config_toml(
            "My TB",
            Some("ghcr.io/o/r"),
            &[],
            None,
            None,
            None,
            false,
            None,
        );
        assert!(toml.contains("name = \"My TB\""));
        assert!(toml.contains("registry = \"ghcr.io/o/r\""));
        assert!(toml.contains("# registries = []"));
        assert!(toml.contains("# image_path_prefix = \"\""));
        // unset export-layout dirs are commented placeholders documenting the defaults
        assert!(toml.contains("# export_image_path = \"images\""));
        assert!(toml.contains("# export_pipeline_path = \"pipelines\""));
        assert!(!toml.contains("bundled_images"));
        // an unset base image is a commented placeholder
        assert!(toml.contains("# [base_image]"));
    }

    /// An unset registry is emitted as a commented placeholder, not `registry = ""`
    #[test]
    fn render_config_no_registry() {
        // an unset registry must be a commented placeholder, never an active empty value
        let toml = render_config_toml("My TB", None, &[], None, None, None, false, None);
        assert!(toml.contains("# registry = \"\""));
        // the active (uncommented) registry line must not be present
        assert!(!toml.contains("\nregistry = "));
    }

    /// Extra registries, an image_path_prefix, bundled_images, and a base image are
    /// written out
    #[test]
    fn render_config_full() {
        // every optional knob set: each must be written out as an active key, not a placeholder
        let base = BaseImage {
            image: Some("ubuntu:22.04".to_string()),
            image_arg: Some("IMAGE".to_string()),
            token: Some("BASE_TOKEN".to_string()),
            user: Some("BASE_USER".to_string()),
            allow_override: Some(true),
        };
        let toml = render_config_toml(
            "TB",
            Some("reg"),
            &["reg".to_string(), "reg2".to_string()],
            Some("prefix/path"),
            Some("tools/images"),
            Some("tools/pipelines"),
            true,
            Some(&base),
        );
        assert!(toml.contains("registries = [\"reg\", \"reg2\"]"));
        assert!(toml.contains("bundled_images = true"));
        assert!(toml.contains("image_path_prefix = \"prefix/path\""));
        assert!(!toml.contains("# image_path_prefix"));
        // the export-layout dirs are written out as active keys, not placeholders
        assert!(toml.contains("export_image_path = \"tools/images\""));
        assert!(toml.contains("export_pipeline_path = \"tools/pipelines\""));
        assert!(!toml.contains("# export_image_path"));
        // the base image table is written out, not a commented placeholder
        assert!(toml.contains("[base_image]"));
        assert!(toml.contains("image = \"ubuntu:22.04\""));
        assert!(toml.contains("image_arg = \"IMAGE\""));
        assert!(toml.contains("token = \"BASE_TOKEN\""));
        assert!(toml.contains("user = \"BASE_USER\""));
        assert!(toml.contains("allow_override = true"));
        assert!(!toml.contains("# [base_image]"));
    }

    /// Free-form values with quotes are escaped so the TOML stays valid
    #[test]
    fn render_config_escapes() {
        // a quote in the free-form name must be backslash-escaped so the TOML stays valid
        let toml = render_config_toml("a\"b", Some("r"), &[], None, None, None, false, None);
        assert!(toml.contains("name = \"a\\\"b\""));
    }
}
