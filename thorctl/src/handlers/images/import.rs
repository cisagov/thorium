//! Image import support for thorctl
//!
//! Imports image configs from an on-disk export directory through the shared
//! conflict engine: new images are created, existing ones are merged
//! interactively, force-updated, or skipped with a field-level warning
//! depending on the conflict mode. Every applied change is journaled so a
//! partial import can be rolled back by the caller.

use std::path::Path;
use thorium::models::{ImageRequest, ImageScaler, ImageUpdate};
use thorium::{CtlConf, Error, Thorium};

use crate::args::images::ImportImages;
use crate::handlers::container;
use crate::handlers::imports::kind::ImageKind;
use crate::handlers::imports::rollback::Journal;
use crate::handlers::imports::{
    self, ApplyOutcome, ConflictMode, ImportOutcome, categorize, create,
};
use crate::handlers::progress::Bar;

/// The options that drive an image import pass
///
/// Borrowed from the `images import` command directly, or assembled by
/// `pipelines import` for its image pass.
pub struct ImageImportOpts<'a> {
    /// The directory holding `images/<name>.json` configs (and tarballs)
    pub import_dir: &'a Path,
    /// The group to import the images into
    pub group: &'a str,
    /// The registry to retag and push loaded container images to
    pub registry: Option<&'a str>,
    /// The registry to override image urls with in Thorium
    pub registry_override: Option<&'a str>,
    /// Skip container load/retag/push entirely
    pub skip_push: bool,
    /// Only update existing images' registry urls instead of full conflict handling
    pub migrate_registry: bool,
    /// How to handle images that already exist
    pub mode: ConflictMode,
    /// The editor override for interactive merges
    pub editor: Option<&'a str>,
    /// Whether stdin is a terminal (the prerequisite for any interactive prompt;
    /// the mode decides whether a prompt is actually wanted)
    pub is_terminal: bool,
    /// Max concurrent API actions in the apply phase (the global `--workers`)
    pub workers: usize,
}

impl<'a> ImageImportOpts<'a> {
    /// Build image import options from the `images import` command
    ///
    /// # Arguments
    ///
    /// * `cmd` - The import command to pull options from
    /// * `workers` - Max concurrent API actions in the apply phase (`--workers`)
    pub fn from_cmd(cmd: &'a ImportImages, workers: usize) -> Self {
        Self {
            import_dir: &cmd.import,
            group: &cmd.group,
            registry: cmd.registry.as_deref(),
            registry_override: cmd.registry_override.as_deref(),
            skip_push: cmd.skip_push,
            migrate_registry: cmd.migrate_registry,
            mode: ConflictMode::from_flags(cmd.overwrite, cmd.skip_conflicts),
            editor: cmd.editor.as_deref(),
            is_terminal: std::io::IsTerminal::is_terminal(&std::io::stdin()),
            workers,
        }
    }
}

/// Rewrite an image url onto a new registry, keeping its repository path
///
/// # Arguments
///
/// * `image` - The image request whose url may be rewritten
/// * `registry` - The registry to rewrite onto, if any
fn override_registry(image: &ImageRequest, registry: Option<&str>) -> Option<String> {
    match (registry, &image.image) {
        (Some(registry), Some(old_url)) => {
            // try to split our old image url into the path and the domain
            let url_path = match old_url.split_once('/') {
                Some((_, old_path)) => old_path,
                None => old_url,
            };
            // build our new url for the new registry
            Some(format!("{registry}/{url_path}"))
        }
        _ => None,
    }
}

/// Load an image request from the export directory and point it at our group
///
/// # Arguments
///
/// * `opts` - The image import options
/// * `name` - The name of the image whose config we are loading from disk
async fn load_request(opts: &ImageImportOpts<'_>, name: &str) -> Result<ImageRequest, Error> {
    categorize::load_request::<ImageKind>(opts.import_dir, opts.group, name).await
}

/// Perform the container side of an image import, rewriting the request's url
///
/// K8s-scaled images with a tarball in the export are loaded, retagged to the
/// target registry if one was given, and pushed. All other scalers are
/// config-only. The `--registry-override` rewrite applies regardless since it
/// only changes what url Thorium stores.
///
/// # Arguments
///
/// * `opts` - The image import options
/// * `name` - The name of the image being imported
/// * `image_req` - The request whose image url may be rewritten
/// * `bar` - The progress bar to update with container status
async fn prepare_container(
    opts: &ImageImportOpts<'_>,
    name: &str,
    image_req: &mut ImageRequest,
    bar: &Bar,
) -> Result<(), Error> {
    // container work only applies to K8s-scaled images when pushing is allowed
    if image_req.scaler == ImageScaler::K8s && !opts.skip_push {
        let tar_path = opts
            .import_dir
            .join("images")
            .join(format!("{name}.tar.gz"));
        // a config-only export has no tarball; that's fine, just skip container.
        // a real stat error (e.g. bad permissions) is surfaced, not hidden as "absent"
        let tar_exists = tokio::fs::try_exists(&tar_path)
            .await
            .map_err(|e| Error::new(format!("Failed to stat '{}': {e}", tar_path.display())))?;
        if tar_exists {
            container::load(&tar_path, bar).await?;
            // retag onto the target registry before pushing if one was given
            if let Some(retagged) = override_registry(image_req, opts.registry) {
                let old_url = image_req
                    .image
                    .as_deref()
                    .ok_or_else(|| Error::new("Cannot retag image without an image URL"))?;
                container::tag(old_url, &retagged, bar).await?;
                image_req.image = Some(retagged);
            }
            if let Some(image_url) = &image_req.image {
                container::push(image_url, bar).await?;
            }
        } else {
            bar.info(format!(
                "No image archive for '{name}' (config-only); skipping container load/push"
            ));
        }
    }
    // the stored url override applies to every scaler
    if let Some(overridden) = override_registry(image_req, opts.registry_override) {
        image_req.image = Some(overridden);
    }
    Ok(())
}

/// Load, categorize, and container-prepare the named images from an export directory
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `opts` - The image import options
/// * `names` - The names of the images to import
/// * `progress` - The progress bar
pub async fn categorize_from_disk(
    thorium: &Thorium,
    opts: &ImageImportOpts<'_>,
    names: &[String],
    progress: &Bar,
) -> Result<Vec<categorize::CategorizedImage>, Error> {
    // load every request up front so malformed configs fail before any changes
    let mut items = Vec::with_capacity(names.len());
    for name in names {
        let request = load_request(opts, name).await?;
        items.push((name.clone(), "latest".to_string(), request));
    }
    categorize::categorize_images(thorium, items, progress).await
}

/// Apply the categorized images to Thorium according to the conflict mode
///
/// The container work (load/retag/push) happens here, after the caller has
/// confirmed the import, so cancelling at the confirmation costs nothing.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (used for the default editor)
/// * `opts` - The image import options
/// * `images` - The categorized images to apply
/// * `progress` - The progress bar
/// * `journal` - The journal to record applied changes in
pub async fn apply_images(
    thorium: &Thorium,
    conf: &CtlConf,
    opts: &ImageImportOpts<'_>,
    images: &mut [categorize::CategorizedImage],
    progress: &Bar,
    journal: &Journal,
) -> Result<ApplyOutcome, Error> {
    // move container images and rewrite urls before touching Thorium state.
    // kept sequential on purpose: container load/push stream their own progress to the
    // terminal, and parallel container output would interleave into noise
    for img in images.iter_mut() {
        let name = img.name.clone();
        prepare_container(opts, &name, &mut img.request, progress).await?;
    }
    // --migrate-registry is a targeted operation: update existing images'
    // urls (or create missing ones) and skip normal conflict handling
    if opts.migrate_registry {
        let outcome = migrate_registries(thorium, images, progress, journal).await?;
        return Ok(ApplyOutcome {
            outcome,
            failures: Vec::new(),
        });
    }
    let plan = imports::ImportPlan::new(images, &[], Vec::new());
    // create the images that don't exist yet, collecting per-image failures
    let mut failures =
        create::import_new_images(thorium, plan.new_images, opts.workers, progress, journal).await;
    // handle existing images according to the conflict mode (shared dispatch)
    let existing = imports::apply_existing::<ImageKind>(
        thorium,
        conf,
        plan.existing_images,
        opts.mode,
        opts.editor,
        opts.is_terminal,
        opts.workers,
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

/// Update existing images' registry urls, creating any that are missing
///
/// `--migrate-registry` is intentionally narrow: on an image that already exists
/// it touches only the stored image url, leaving every other field as-is.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `images` - The categorized images to migrate
/// * `progress` - The progress bar
/// * `journal` - The journal to record applied changes in
async fn migrate_registries(
    thorium: &Thorium,
    images: &[categorize::CategorizedImage],
    progress: &Bar,
    journal: &Journal,
) -> Result<ImportOutcome, Error> {
    for img in images {
        match (&img.existing, &img.request.image) {
            // the image exists and has a url to migrate to
            (Some(existing), Some(image_url)) => {
                let update = ImageUpdate::default().image(image_url);
                thorium
                    .images
                    .update(&img.request.group, &img.request.name, &update)
                    .await
                    .map_err(|err| {
                        Error::new(format!("Error migrating image '{}': {err}", img.name))
                    })?;
                journal.updated_image(existing.clone());
            }
            // the image exists but the incoming config has no url to apply
            (Some(_), None) => progress.info(format!(
                "Image '{}' has no image url to migrate; skipping",
                img.name
            )),
            // the image doesn't exist yet, so create it like a normal import
            (None, _) => {
                thorium.images.create(&img.request).await.map_err(|err| {
                    Error::new(format!("Error importing image '{}': {err}", img.name))
                })?;
                journal.created_image(&img.request.group, &img.request.name);
            }
        }
        progress.inc(1);
    }
    Ok(ImportOutcome::Completed)
}
