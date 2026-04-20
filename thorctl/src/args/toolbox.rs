//! Arguments for toolbox-related Thorctl commands
//!
//! Short-flag policy (applies to every toolbox subcommand, and mirrors the wider `thorctl` CLI):
//! - Short flags are reserved for non-destructive selection/IO inputs and standard run toggles, used
//!   consistently across commands: `-g` group, `-i` images/import, `-p` pipelines, `-o` output,
//!   `-c` config, `-L` list, `-n` non-interactive, `-e` exit-on-error.
//! - Destructive, "force", and scope/reconciliation flags are intentionally **long-only** (no short),
//!   so a stray short can't trigger a less-reversible action: `--skip-confirm`, `--force`, `--forced`,
//!   `--overwrite`, `--overwrite-config`, `--skip-conflicts`, `--group-override`,
//!   `--update-network-policy`, `--exit-code`, `--strip-registry`, `--with-images`, `--review`,
//!   `--rollback-on-failure`.
//! - Known pre-existing cross-command exceptions (left as-is to avoid breaking users): `-g` vs `-G`
//!   are both used for groups elsewhere in `thorctl`, and `-c` means `--config` here but a boolean
//!   toggle in some other commands. Don't extend these; new flags should follow the rules above.

use clap::Parser;
use std::path::PathBuf;
use url::Url;

/// A command to interact with Thorium toolboxes
#[derive(Parser, Debug)]
pub enum Toolbox {
    /// Import a toolbox into Thorium
    ///
    /// A Thorium toolbox is an external collection of tools and pipelines pre-configured
    /// and ready to run in Thorium. If images or pipelines already exist in Thorium,
    /// an interactive editor will open to review and resolve differences. Use --overwrite
    /// to automatically apply all incoming changes without the editor.
    #[clap(version, author)]
    Import(ImportToolbox),
    /// Build a toolbox manifest from image and pipeline manifests
    ///
    /// Walks the current directory for image and pipeline manifest.toml files,
    /// reads their associated JSON configs, and produces a toolbox.json file
    /// suitable for import into Thorium.
    #[clap(version, author)]
    Build(BuildToolbox),
    /// Initialize toolbox, image, or pipeline scaffolding
    ///
    /// Generate default manifest.toml and JSON config files. Existing files
    /// are never overwritten.
    #[clap(version, author, subcommand)]
    Init(Init),
    /// Export Thorium images and pipelines into a toolbox directory structure
    ///
    /// Fetches image and pipeline configs from a running Thorium instance,
    /// creates a toolbox directory with manifest.toml and JSON config files,
    /// and produces a toolbox.json ready for import elsewhere.
    #[clap(version, author)]
    Export(ExportToolbox),
    /// Remove a previously imported toolbox from Thorium
    ///
    /// Deletes the pipelines and images named by a toolbox manifest from the
    /// target instance. Pipelines are deleted before the images they
    /// reference; resources that don't exist are reported and skipped.
    /// Groups are never deleted.
    #[clap(version, author)]
    Remove(RemoveToolbox),
    /// Diff an on-disk toolbox against what a Thorium instance has imported
    ///
    /// Shows what an import of this toolbox would change, rendered like git
    /// diff: resources only in the toolbox appear as new files, resources
    /// only in the instance's groups as deletions, and changed resources as
    /// unified hunks.
    #[clap(version, author)]
    Diff(DiffToolbox),
    /// Build (and optionally push) the container images in a toolbox
    ///
    /// Walks a toolbox.json for image entries with build enabled, builds each
    /// entry's docker context with its first tag, aliases the remaining tags,
    /// and pushes all tags with --push. For forks without CI: build, push,
    /// then import as usual.
    #[clap(version, author, name = "build-images")]
    BuildImages(BuildImagesToolbox),
}

/// Parse a `KEY=VALUE` build arg, splitting on the first `=`
///
/// Used by `--base-image` and `--build-arg`. The key must be non-empty; the value may
/// be empty and may itself contain `=` (only the first `=` is the separator), so image
/// references and values with `=` pass through intact.
///
/// # Arguments
///
/// * `raw` - The raw `KEY=VALUE` argument string
fn parse_build_arg(raw: &str) -> Result<(String, String), String> {
    // split on the first '=' so a value that itself contains '=' (or a registry ref with
    // ':' and '/') is preserved whole; a missing '=' has no key/value boundary and errors
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got '{raw}'"))?;
    // reject an empty key (a leading '='); a nameless build arg can't be applied to a build
    if key.is_empty() {
        return Err(format!("build arg key must not be empty in '{raw}'"));
    }
    // own both halves so the parsed pair outlives the borrowed input string
    Ok((key.to_string(), value.to_string()))
}

/// Build the container images described by a toolbox manifest
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct BuildImagesToolbox {
    /// The path to the toolbox.json holding the build entries (default: ./toolbox.json)
    #[clap(value_name = "TOOLBOX.JSON", default_value = "toolbox.json")]
    pub manifest: PathBuf,
    /// Only build these images, comma-separated. Format: `name[,name,...]`
    /// (default: every image with build enabled)
    #[clap(
        short = 'i',
        long = "images",
        value_name = "NAME",
        value_delimiter = ','
    )]
    pub images: Vec<String>,
    /// Push every built tag to its registry after building
    #[clap(long)]
    pub push: bool,
    /// Override an image's base image: `ARG=IMAGE`, passed to the build as
    /// `--build-arg ARG=IMAGE`
    ///
    /// Run-global escape hatch that overrides each image's resolved `[base_image]`. Only
    /// applied to images whose `[base_image].allow_override` is true (the default); images
    /// that opt out are built with their default base. `ARG` is the build arg the
    /// Dockerfile reads (e.g. `IMAGE` for `FROM ${IMAGE}`). On a key collision with a
    /// `--build-arg`, this value wins for images that allow the override.
    #[clap(long, value_parser = parse_build_arg, value_name = "ARG=IMAGE")]
    pub base_image: Option<(String, String)>,
    /// Extra build arg passed to every image build: `KEY=VALUE` (repeatable)
    ///
    /// Unlike `--base-image`, these are passed to every build regardless of each entry's
    /// `base_image.allow_override` gate.
    #[clap(long = "build-arg", value_parser = parse_build_arg, value_name = "KEY=VALUE")]
    pub build_args: Vec<(String, String)>,
    /// Append this suffix to every tag built and pushed, without touching toolbox.json
    ///
    /// Lets a CI/CD feature-branch run build/push differentiated images (e.g.
    /// `:1.0-mybranch`) from an unmodified toolbox.json so they don't collide with the
    /// mainline `:1.0`. Pass the separator you want (e.g. `-mybranch`).
    // allow_hyphen_values so a leading-dash suffix like `-mybranch` is taken as the
    // value rather than parsed as another flag
    #[clap(long, value_name = "SUFFIX", allow_hyphen_values = true)]
    pub tag_suffix: Option<String>,
    /// Stop at the first image whose build or push fails
    ///
    /// By default the build keeps going on failure — logging each one and exiting
    /// non-zero at the end if any image failed. This flag instead aborts the run
    /// on the first failure.
    #[clap(short = 'e', long = "exit-on-error")]
    pub exit_on_error: bool,
    /// Build without the layer cache (passes `--no-cache` to docker/podman)
    ///
    /// Applies to every image built in this run.
    #[clap(long)]
    pub no_cache: bool,
    /// Force a fresh pull of referenced images before building (passes `--pull` to
    /// docker/podman)
    ///
    /// Like docker/podman `--pull`, this refreshes every image the build references
    /// (including the `FROM` base) instead of using a locally cached one. Applies to
    /// every image built in this run.
    #[clap(long)]
    pub pull: bool,
}

/// Diff an on-disk toolbox against a running Thorium instance
#[derive(Parser, Debug)]
pub struct DiffToolbox {
    /// A toolbox.json (path or URL) or a toolbox repo directory. Format: `path | url | dir`
    ///
    /// Directories are built in-memory from their manifests, so the diff
    /// reflects the current on-disk configs without regenerating toolbox.json.
    #[clap(value_name = "PATH | URL | DIR")]
    pub manifest: ManifestLocation,
    /// Compare against this group instead of the groups recorded in the toolbox
    ///
    /// Use this when the toolbox was imported with --group-override.
    #[clap(long, value_name = "GROUP")]
    pub group_override: Option<String>,
    /// Exit with code 1 when any difference exists (git diff semantics)
    #[clap(long)]
    pub exit_code: bool,
}

/// Remove a toolbox's pipelines and images from Thorium
#[derive(Parser, Debug)]
pub struct RemoveToolbox {
    /// The toolbox manifest to use: a local file path or a URL. Format: `path | url`
    #[clap(value_name = "PATH | URL")]
    pub manifest: ManifestLocation,
    /// Remove from this group instead of the groups recorded in the manifest
    ///
    /// Use this when the toolbox was imported with --group-override.
    #[clap(long, value_name = "GROUP")]
    pub group_override: Option<String>,
    /// Skip the confirmation dialog
    #[clap(long)]
    pub skip_confirm: bool,
}

/// Subcommands for `thorctl toolbox init`
#[derive(Parser, Debug)]
pub enum Init {
    /// Initialize a full toolbox with config.toml, image, and pipeline files
    #[clap(version, author)]
    Toolbox(InitToolbox),
    /// Initialize a single image with a manifest.toml and JSON config
    #[clap(version, author)]
    Image(InitImage),
    /// Initialize a single pipeline with a manifest.toml and JSON config
    #[clap(version, author)]
    Pipeline(InitPipeline),
}

/// The location of the toolbox manifest, either by URL or by file path
#[derive(Debug, Clone)]
pub enum ManifestLocation {
    /// The manifest is at this URL
    Url(Url),
    /// The manifest is at this file path
    Path(PathBuf),
}

impl std::str::FromStr for ManifestLocation {
    type Err = String;

    /// Parse a string into a [`ManifestLocation`]
    ///
    /// A URL is preferred over a path so a remote `toolbox.json` can be fetched; only
    /// inputs that fail URL parsing are treated as local files.
    ///
    /// # Arguments
    ///
    /// * `s` - The raw manifest location string (a URL or a filesystem path)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // prefer a URL interpretation so remote manifests are fetched rather than
        // mistaken for a relative path
        if let Ok(url) = Url::parse(s) {
            return Ok(Self::Url(url));
        }
        // anything that isn't a valid URL is taken as a local filesystem path; this
        // parse is infallible so the location always resolves to one of the two variants
        Ok(Self::Path(PathBuf::from(s)))
    }
}

/// Download a toolbox manifest and import it into Thorium
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ImportToolbox {
    /// The toolbox manifest to use: a local file path or a URL. Format: `path | url`
    #[clap(value_name = "PATH | URL")]
    pub manifest: ManifestLocation,
    /// Force the tools and pipelines to be imported to a specific group
    ///
    /// The group will be created if it doesn't already exist
    #[clap(long, value_name = "GROUP")]
    pub group_override: Option<String>,
    /// Overwrite existing images/pipelines without opening the editor
    #[clap(long, conflicts_with = "skip_conflicts")]
    pub overwrite: bool,
    /// Skip existing images/pipelines that differ instead of updating them
    ///
    /// New resources are still created. Each skipped resource logs a warning
    /// listing the fields that differ, making this safe for non-interactive
    /// (CI/agent) imports that must never overwrite local changes.
    #[clap(long)]
    pub skip_conflicts: bool,
    /// Automatically roll back applied changes if the import stops early
    ///
    /// Only applies when the session can't prompt (--overwrite, --skip-conflicts, or
    /// no TTY); interactive sessions are asked instead. Registry pushes are not
    /// undone, only Thorium state.
    #[clap(long)]
    pub rollback_on_failure: bool,
    /// Override the default editor for reviewing merge conflicts
    #[clap(long, value_name = "EDITOR")]
    pub editor: Option<String>,
    /// Target registry base path for images bundled in the toolbox
    ///
    /// Only used when the toolbox bundles container images (exported with
    /// `--with-images`). Each bundled image is loaded, retagged, and pushed to
    /// `<image-path-prefix>/<group>/<name>:<tag>`, and its Thorium config is
    /// rewritten to point there. If omitted for a bundled toolbox, the prefix
    /// recorded in the manifest is used, otherwise you are prompted for one.
    #[clap(long, value_name = "REGISTRY/BASE")]
    pub image_path_prefix: Option<String>,
    /// Update existing Thorium network policies to match the toolbox
    ///
    /// Without this, an existing policy that differs is left untouched and only a
    /// warning is logged. With it, the toolbox's definition is authoritative:
    /// rule/flag differences overwrite the policy, and a groups-only difference
    /// adds the toolbox's groups to the existing policy. Group coverage is only
    /// ever added, never removed.
    ///
    /// This overwrites cluster network-security state. In non-interactive runs
    /// (--skip-conflicts, --overwrite, or no TTY) it applies WITHOUT a prompt.
    /// Note: changing a policy's default_policy does not retroactively remove it
    /// from images created during this same import, so that part is not unwound
    /// by a rollback.
    #[clap(long)]
    pub update_network_policy: bool,
}

/// Build a toolbox manifest from image and pipeline manifests
#[derive(Parser, Debug, Clone)]
pub struct BuildToolbox {
    /// Path to the toolbox TOML config file (default: config.toml in the current directory)
    #[clap(
        short = 'c',
        long = "config",
        value_name = "CONFIG.TOML",
        default_value = "config.toml"
    )]
    pub config: PathBuf,
    /// Tag every image with its manifest `image_name` (a repo-style path) as the leaf:
    /// `<registry>/[prefix/]<image_name>:<version>`
    ///
    /// By default images are tagged with the tool `name`
    /// (`<registry>/[prefix/]<name>:<version>`). Set this to use the path-style `image_name`
    /// from each manifest instead. No effect on images pinned to an explicit url.
    #[clap(long)]
    pub use_image_path: bool,
    /// Output path for the generated toolbox.json
    ///
    /// Defaults to a `toolbox.json` beside the --config file (the toolbox root). Overriding this
    /// only redirects the artifact; it does not change where manifests are crawled from.
    #[clap(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,
    /// Root directory to walk for image/pipeline manifests
    ///
    /// Defaults to the directory containing --config (the toolbox root). Overriding this only
    /// changes the crawl root; it does not change where toolbox.json is written.
    #[clap(long, value_name = "DIR")]
    pub path: Option<PathBuf>,
    /// Append this suffix to every derived image tag's version, baking it into the
    /// generated toolbox.json (tags and embedded image urls)
    ///
    /// Lets a CI/CD feature-branch build produce a toolbox.json that references
    /// differentiated images (e.g. `:1.0-mybranch`) instead of colliding with the
    /// mainline `:1.0`. Only affects derived `<registry>/...:<version>` tags; images
    /// pinned to an explicit url are left untouched. Pass the separator you want
    /// (e.g. `-mybranch`).
    // allow_hyphen_values so a leading-dash suffix like `-mybranch` is taken as the
    // value rather than parsed as another flag
    #[clap(long, value_name = "SUFFIX", allow_hyphen_values = true)]
    pub tag_suffix: Option<String>,
}

/// Initialize a full toolbox with config.toml, image, and pipeline files
#[derive(Parser, Debug)]
pub struct InitToolbox {
    /// Image build directories, comma-separated. Format: `path[,path,...]` (each gets a
    /// manifest.toml + JSON config)
    #[clap(
        short = 'i',
        long = "images",
        value_name = "PATH",
        required = true,
        value_delimiter = ','
    )]
    pub images: Vec<PathBuf>,
    /// Pipeline directories, optionally binding images. Format: `path[:image,...]`
    ///
    /// Use the optional `[:image,...]` colon suffix to bind specific images, e.g.
    /// `-p ./pipelines/capa:capa,yara`. Without it, all --images are bound. Repeat -p for
    /// multiple pipelines.
    #[clap(
        short = 'p',
        long = "pipeline",
        value_name = "PATH[:IMAGE,...]",
        verbatim_doc_comment
    )]
    pub pipelines: Vec<String>,
    /// Group name to use in generated configs (prompted interactively if omitted)
    #[clap(short = 'g', long = "group", value_name = "GROUP")]
    pub group: Option<String>,
    /// Path to the toolbox root directory where config.toml will be created
    /// (default: current directory)
    #[clap(long, value_name = "DIR", default_value = ".")]
    pub toolbox_dir: PathBuf,
    /// Seed the new config.toml from an existing one (name, registry, registries,
    /// image_path_prefix, export paths, bundled_images) instead of --name/--registry
    ///
    /// Mutually exclusive with --name, --registry, --image-path, and --pipeline-path.
    #[clap(short = 'c', long = "config", value_name = "CONFIG.TOML", conflicts_with_all = ["name", "registry", "image_path", "pipeline_path"], verbatim_doc_comment)]
    pub config: Option<PathBuf>,
    /// Toolbox name for config.toml (default: "My Toolbox")
    #[clap(long, value_name = "NAME", default_value = "My Toolbox")]
    pub name: String,
    /// Container registry for config.toml, e.g. ghcr.io/org/repo
    ///
    /// Optional: when omitted, the toolbox declares no central registry and each
    /// image's tag is taken from the `image` url in its own config.
    #[clap(long, value_name = "REGISTRY")]
    pub registry: Option<String>,
    /// Directory (relative to the toolbox root) `export` writes image tool dirs under (default: images)
    ///
    /// Sets `export_image_path` in config.toml. Must be a relative subpath (no absolute path, no
    /// `..`). Only affects where `export` places files — `build` still discovers manifests at any depth.
    #[clap(long, value_name = "DIR")]
    pub image_path: Option<String>,
    /// Directory (relative to the toolbox root) `export` writes pipeline tool dirs under (default: pipelines)
    ///
    /// Sets `export_pipeline_path` in config.toml. Must be a relative subpath (no absolute path, no `..`).
    #[clap(long, value_name = "DIR")]
    pub pipeline_path: Option<String>,
    /// The editor to use when filling in configs (defaults to your configured `default_editor`)
    #[clap(long, value_name = "EDITOR")]
    pub editor: Option<String>,
    /// Skip interactive prompts and use defaults for all fields
    #[clap(short = 'n', long)]
    pub non_interactive: bool,
    /// Overwrite existing per-tool files instead of skipping them
    ///
    /// Covers the scaffolded manifest.toml/JSON/description files. It does NOT touch an existing
    /// config.toml — use --overwrite-config for that.
    #[clap(long)]
    pub overwrite: bool,
    /// Overwrite an existing config.toml instead of preserving it
    ///
    /// By default an existing config.toml is kept (so re-running init in a toolbox doesn't clobber
    /// its settings). Distinct from --overwrite, which covers per-tool files.
    #[clap(long)]
    pub overwrite_config: bool,
}

/// A parsed pipeline spec from the --pipeline flag
#[derive(Debug)]
pub struct PipelineSpec {
    /// Path to the pipeline directory
    pub path: PathBuf,
    /// Specific image names for this pipeline, or None to include all
    pub images: Option<Vec<String>>,
}

impl PipelineSpec {
    /// Parse a pipeline argument string.
    ///
    /// `"./pipelines/capa:capa,yara"` → path `./pipelines/capa`, images `["capa", "yara"]`
    /// `"./pipelines/capa"` → path `./pipelines/capa`, images `None` (all images)
    ///
    /// # Arguments
    ///
    /// * `s` - The pipeline argument string to parse
    pub fn parse(s: &str) -> Self {
        // split on the LAST colon so a path that itself contains a colon keeps everything
        // up to the final one as the directory and only the trailing segment is the image list
        if let Some((path, images_str)) = s.rsplit_once(':') {
            // split the post-colon segment into individual image names, trimming
            // whitespace and dropping empties so `capa, yara,` yields just `["capa","yara"]`
            let images: Vec<String> = images_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            // a colon with no real image names (e.g. a trailing `:` or `: `) is treated as
            // "no binding": fall back to the FULL original string as the path so the colon
            // isn't silently stripped off a directory that legitimately contained one
            if images.is_empty() {
                Self {
                    path: PathBuf::from(s),
                    images: None,
                }
            } else {
                Self {
                    path: PathBuf::from(path),
                    images: Some(images),
                }
            }
        } else {
            // no colon at all means bind every image, so record the whole string as the path
            Self {
                path: PathBuf::from(s),
                images: None,
            }
        }
    }
}

/// Initialize a single image with a manifest.toml and JSON config
#[derive(Parser, Debug)]
pub struct InitImage {
    /// Path to the image build directory (the scaffolded files land here)
    #[clap(value_name = "PATH")]
    pub path: PathBuf,
    /// The manifest `image_name`: the registry tag path leaf used at build time, formatted as a
    /// repo-style path (`<registry>/<image_name>:<version>`); defaults to the build directory name
    #[clap(long, value_name = "IMAGE_NAME")]
    pub image_name: Option<String>,
    /// Group name to use in the generated image config (prompted interactively if omitted)
    #[clap(short = 'g', long = "group", value_name = "GROUP")]
    pub group: Option<String>,
    /// Validate against an existing toolbox's config.toml (resolution source, not placement)
    ///
    /// When set, scaffolding errors if an image of the same name+version already exists in that
    /// toolbox (pass --overwrite to replace). Does not move files — the positional path is the
    /// destination.
    #[clap(short = 'c', long = "config", value_name = "CONFIG.TOML")]
    pub config: Option<PathBuf>,
    /// Skip building this image in CI/CD (image already exists in registry)
    #[clap(long)]
    pub no_build: bool,
    /// The editor to use when filling in the config (defaults to your configured `default_editor`)
    #[clap(long, value_name = "EDITOR")]
    pub editor: Option<String>,
    /// Skip interactive prompts and use defaults for all fields
    #[clap(short = 'n', long)]
    pub non_interactive: bool,
    /// Overwrite existing files instead of skipping them
    #[clap(long)]
    pub overwrite: bool,
}

/// Initialize a single pipeline with a manifest.toml and JSON config
#[derive(Parser, Debug)]
pub struct InitPipeline {
    /// Path to the pipeline directory (the scaffolded files land here)
    #[clap(value_name = "PATH")]
    pub path: PathBuf,
    /// Image names this pipeline runs, comma-separated. Format: `image[,image,...]` (prompted
    /// interactively if omitted)
    #[clap(
        short = 'i',
        long = "images",
        value_name = "IMAGE",
        value_delimiter = ','
    )]
    pub images: Vec<String>,
    /// Group name to use in the generated pipeline config (prompted interactively if omitted)
    #[clap(short = 'g', long = "group", value_name = "GROUP")]
    pub group: Option<String>,
    /// Pipeline order as JSON: a list of parallel stages, e.g. `[["img1","img2"],["img3"]]`
    /// (defaults to all images in a single parallel stage)
    #[clap(long, value_name = "JSON")]
    pub order: Option<String>,
    /// Resolve the pipeline's images against an existing toolbox's config.toml (the "look here"
    /// source, not placement)
    ///
    /// When set, every referenced image must exist in that toolbox (else an error — `init pipeline`
    /// never creates images), and each is version-pinned from the toolbox instead of `latest`. Does
    /// not move files — the positional path is the destination.
    #[clap(short = 'c', long = "config", value_name = "CONFIG.TOML")]
    pub config: Option<PathBuf>,
    /// The editor to use when filling in the config (defaults to your configured `default_editor`)
    #[clap(long, value_name = "EDITOR")]
    pub editor: Option<String>,
    /// Skip interactive prompts and use defaults for all fields
    #[clap(short = 'n', long)]
    pub non_interactive: bool,
    /// Overwrite existing files instead of skipping them
    #[clap(long)]
    pub overwrite: bool,
}

/// Export Thorium images and pipelines into a toolbox directory
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExportToolbox {
    /// Export all images and pipelines from this Thorium group (the source group to read from)
    ///
    /// Exported tools are written under this same group unless `--group-override` sets a different
    /// destination group. When re-exporting into an existing toolbox whose tools live under a
    /// different group, `--overwrite` re-groups them in place to this group (no `--group-override`
    /// needed); without `--overwrite` a mismatched tool is skipped with a warning rather than
    /// duplicated.
    #[clap(short = 'g', long = "group", value_name = "GROUP")]
    pub group: Option<String>,
    /// Export specific pipelines. Format: `group/name[=dir]` (or `name[=dir]` with --group),
    /// comma-separated for multiple.
    ///
    /// Images referenced by exported pipelines are auto-included. The optional `[=dir]` suffix
    /// writes that pipeline's files to `dir`, e.g. `static/av=pipelines/av`; without it the
    /// configured/default layout is used. `dir` may be relative (interpreted against the toolbox
    /// root) or absolute, but must resolve to a directory inside the toolbox. `[=dir]` is placement
    /// only — it never changes which pipeline (or its images) is selected.
    #[clap(
        short = 'p',
        long = "pipelines",
        value_name = "GROUP/NAME[=DIR]",
        value_delimiter = ','
    )]
    pub pipelines: Vec<String>,
    /// Export specific standalone images. Format: `group/name[=dir]` (or `name[=dir]` with
    /// --group), comma-separated for multiple.
    ///
    /// The optional `[=dir]` suffix writes that image's files to `dir`, e.g.
    /// `static/clamav=tools/clamav` to fold a config into an existing build-context dir; naming an
    /// auto-pulled dependency image this way also redirects it. `dir` may be relative (interpreted
    /// against the toolbox root) or absolute, but must resolve to a directory inside the toolbox.
    /// `[=dir]` is placement only.
    #[clap(
        short = 'i',
        long = "images",
        value_name = "GROUP/NAME[=DIR]",
        value_delimiter = ','
    )]
    pub images: Vec<String>,
    /// Destination group: write every exported config under this group instead of its source group
    ///
    /// Use it to read from one Thorium group (`-g`/`--pipelines`/`--images`) but store the tools under
    /// a different group in the toolbox, or to align an export with an existing toolbox's group so the
    /// tools reconcile and update in place. Warns if name collisions would occur across source groups.
    #[clap(long, value_name = "GROUP")]
    pub group_override: Option<String>,
    /// Root directory for the exported toolbox
    ///
    /// Defaults to the `--config` directory when `--config` is given (so pointing at a toolbox's
    /// config.toml exports into that toolbox), otherwise `./toolbox` for a brand-new toolbox. An
    /// explicit value always wins — pass it to seed settings from one toolbox into a different
    /// output directory.
    #[clap(short = 'o', long = "output", value_name = "DIR")]
    pub output: Option<PathBuf>,
    /// Seed the toolbox-wide settings (name, registry, registries, image_path_prefix,
    /// bundled_images) from *another* toolbox's config.toml instead of --name/--registry
    ///
    /// This is for starting a new toolbox from an existing one's settings. Appending into a
    /// toolbox that already has a config.toml does NOT need this: an existing
    /// <output>/config.toml is auto-detected, reused, and preserved (settings-source priority is
    /// --config > existing <output>/config.toml > --name/--registry). Mutually exclusive with
    /// --name and --registry.
    ///
    /// Giving --config also anchors --output to the config's directory unless --output is set, so
    /// `export -c mytb/config.toml ...` exports into `mytb/`. To seed settings from one toolbox into
    /// a different directory, pass --output explicitly.
    #[clap(short = 'c', long = "config", value_name = "CONFIG.TOML", conflicts_with_all = ["name", "registry"])]
    pub config: Option<PathBuf>,
    /// Toolbox name for config.toml (default: "My Toolbox")
    #[clap(long, value_name = "NAME", default_value = "My Toolbox")]
    pub name: String,
    /// Container registry for config.toml, e.g. ghcr.io/org/repo
    ///
    /// Optional: when omitted, the exported toolbox declares no central registry and
    /// relies on each image's own `image` url (captured on export unless `--strip-registry` is used).
    #[clap(long, value_name = "REGISTRY")]
    pub registry: Option<String>,
    /// Skip on-disk conflicts: write new configs and leave differing existing ones
    /// untouched with a warning (use --overwrite to overwrite instead)
    #[clap(long)]
    pub skip_conflicts: bool,
    /// Open each config in an editor to review/tweak it before writing
    #[clap(long)]
    pub review: bool,
    /// Update a matched in-toolbox resource that differs (otherwise it is skipped with a warning)
    ///
    /// When a tool with the same group/name already exists and its config differs, this updates it in
    /// place. For an image, only the Thorium config (JSON), description, and policy files are
    /// rewritten — its manifest.toml build settings (build/build_path/[base_image]/image_from) are
    /// preserved; a pipeline's manifest is regenerated. It does NOT touch config.toml — use
    /// --overwrite-config for that.
    #[clap(long, conflicts_with = "skip_conflicts")]
    pub overwrite: bool,
    /// Overwrite an existing config.toml with this run's settings
    ///
    /// By default an existing config.toml is preserved (and its settings reused), so exporting into
    /// an existing toolbox never clobbers its settings. Pass this to replace it. Distinct from
    /// --overwrite, which covers per-tool files.
    #[clap(long)]
    pub overwrite_config: bool,
    /// Bundle each image's container image file into the toolbox for offline transfer
    ///
    /// Downloads (docker pull) and saves (docker save) each image into its tool directory as
    /// `<dir>/<name>.tar.gz` (the configured export layout, default `images/<name>`). The resulting
    /// toolbox can be moved to an offline environment and imported with `--image-path-prefix` to
    /// push the images into a local registry. Requires docker.
    #[clap(long)]
    pub with_images: bool,
    /// Write each image's container url as empty so the release carries no hard-coded registry path
    ///
    /// Clears the `image` url in every exported image config and omits the manifest's
    /// `exported_image_path`, so a rebuild derives each image's path from the toolbox's own
    /// `config.toml` registry/`image_path_prefix` instead of a pinned url. Use it to publish a
    /// registry-agnostic toolbox a consumer points at their own registry. Pipelines carry no url, so
    /// this only affects images. **Conflicts with `--with-images`** (a bundled import needs the url to
    /// tag and push the saved tarball).
    #[clap(long, conflicts_with = "with_images")]
    pub strip_registry: bool,
}

/// A parsed group/name resource reference, with an optional on-disk destination
#[derive(Debug, Clone)]
pub struct ResourceSpec {
    /// The group the resource belongs to
    pub group: String,
    /// The name of the resource
    pub name: String,
    /// The optional per-resource destination directory (relative to the toolbox root) parsed from
    /// a `group/name=dest` selection; `None` uses the toolbox's configured/default export layout.
    /// Placement only — it never affects which resource is selected.
    pub dest: Option<String>,
}

impl ResourceSpec {
    /// Parse `group/name`, `name`, or either with an optional `=dest` placement suffix
    ///
    /// `static/clamav=tools/clamav` selects `static/clamav` and writes its files into `tools/clamav`;
    /// the `=dest` is placement only. Only emptiness is checked here — the dest may be absolute or
    /// relative (and may contain `..`); `export` resolves it against the toolbox root and enforces the
    /// must-stay-inside-the-toolbox rule (see `resolve_dest_within`).
    ///
    /// # Arguments
    ///
    /// * `s` - The resource reference to parse
    /// * `default_group` - The group to fall back to when the reference has no group prefix
    pub fn parse(s: &str, default_group: Option<&str>) -> Result<Self, String> {
        // split off an optional `=dest` placement suffix on the FIRST `=`, so the left side is the
        // group/name reference and the right side is the destination directory. The dest may be
        // absolute or relative (a relative dest is later interpreted against the toolbox root); it is
        // only checked for emptiness here — `export` resolves it and verifies it lands inside the
        // toolbox once the output root is known
        let (reference, dest) = match s.split_once('=') {
            Some((reference, dest)) => {
                if dest.is_empty() {
                    return Err(format!("'{s}' has an empty destination path after '='"));
                }
                (reference, Some(dest.to_string()))
            }
            None => (s, None),
        };
        // split the reference on the FIRST slash so an explicit `group/name` always wins; everything
        // after the first slash is the name (resource names may themselves contain slashes)
        if let Some((group, name)) = reference.split_once('/') {
            Ok(Self {
                group: group.to_string(),
                name: name.to_string(),
                dest,
            })
        } else {
            // with no `group/` prefix the reference is bare, so it can only be resolved when
            // a default group was supplied (from `--group`); otherwise the group is ambiguous
            match default_group {
                Some(g) => Ok(Self {
                    group: g.to_string(),
                    name: reference.to_string(),
                    dest,
                }),
                // reject rather than guess a group so a bare name can't silently land in the
                // wrong place when the caller never set one
                None => Err(format!(
                    "'{reference}' must be in group/name format when --group is not set"
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple `KEY=VALUE` splits into its key and value
    #[test]
    fn parse_build_arg_splits_key_value() {
        // the `:` in the image tag lives in the value, so it must not affect the split
        assert_eq!(
            parse_build_arg("IMAGE=ubuntu:22.04"),
            Ok(("IMAGE".to_string(), "ubuntu:22.04".to_string()))
        );
    }

    /// Only the first `=` separates, so values containing `=` (and registry refs with
    /// `:` and `/`) survive intact
    #[test]
    fn parse_build_arg_splits_on_first_equals() {
        // a value that is itself a chain of `=` keeps every `=` after the first
        assert_eq!(
            parse_build_arg("OPTS=a=b=c"),
            Ok(("OPTS".to_string(), "a=b=c".to_string()))
        );
        // a full registry reference (with `/` and `:`) passes through untouched as the value
        assert_eq!(
            parse_build_arg("IMAGE=ghcr.io/org/base:1.0"),
            Ok(("IMAGE".to_string(), "ghcr.io/org/base:1.0".to_string()))
        );
    }

    /// An empty value is allowed; a missing `=` or empty key is rejected
    #[test]
    fn parse_build_arg_rejects_malformed() {
        // a trailing `=` yields an empty value, which is intentionally permitted
        assert_eq!(
            parse_build_arg("EMPTY="),
            Ok(("EMPTY".to_string(), String::new()))
        );
        // no `=` at all has no key/value boundary, so it must error
        assert!(parse_build_arg("no-equals").is_err());
        // an empty key (leading `=`) is rejected because a nameless build arg is unusable
        assert!(parse_build_arg("=value").is_err());
    }

    /// A `group/name` reference parses into its parts with no destination
    #[test]
    fn resource_spec_parses_group_name() {
        let spec = ResourceSpec::parse("static/clamav", None).expect("group/name parses");
        assert_eq!(spec.group, "static");
        assert_eq!(spec.name, "clamav");
        assert!(spec.dest.is_none());
        // a bare name resolves only with a default group
        let bare = ResourceSpec::parse("clamav", Some("static")).expect("bare name with --group");
        assert_eq!(bare.group, "static");
        assert_eq!(bare.name, "clamav");
        assert!(ResourceSpec::parse("clamav", None).is_err());
    }

    /// An `=dest` suffix sets the placement directory, split on the first `=`
    #[test]
    fn resource_spec_parses_destination() {
        // the reference and destination split on the first '='
        let spec = ResourceSpec::parse("static/clamav=tools/clamav", None).expect("dest parses");
        assert_eq!(spec.group, "static");
        assert_eq!(spec.name, "clamav");
        assert_eq!(spec.dest.as_deref(), Some("tools/clamav"));
        // a bare name with a destination still needs a default group for the reference
        let bare = ResourceSpec::parse("clamav=tools/clamav", Some("static")).expect("bare + dest");
        assert_eq!(bare.name, "clamav");
        assert_eq!(bare.dest.as_deref(), Some("tools/clamav"));
        // absolute and parent-escaping destinations parse here (export resolves them against the
        // toolbox root and rejects only those that land outside it); an empty dest is rejected
        assert_eq!(
            ResourceSpec::parse("static/clamav=/abs/path", None)
                .expect("absolute dest parses")
                .dest
                .as_deref(),
            Some("/abs/path")
        );
        assert_eq!(
            ResourceSpec::parse("static/clamav=../rel", None)
                .expect("parent-relative dest parses")
                .dest
                .as_deref(),
            Some("../rel")
        );
        assert!(ResourceSpec::parse("static/clamav=", None).is_err());
    }
}
