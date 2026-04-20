use thorium::{Error, client::Thorium, models::Image};

use crate::args::Args;
use crate::args::{
    DescribeCommand,
    images::{DescribeImages, GetImages, Images},
};

use crate::utils;

mod bans;
mod edit;
mod notifications;

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "macos"))] {
        use std::io::IsTerminal;

        use futures::stream::{self, StreamExt};
        use thorium::CtlConf;
        use thorium::models::ImageRequest;

        use crate::args::images::{ExportImages, ImportImages};
        use crate::handlers::imports;
        use crate::handlers::exports::{DiskConflictResolver, WriteOutcome};
        use crate::handlers::progress::{Bar, BarKind};
        use super::Controller;

        mod export;
        pub(crate) mod import;

        use export::ImageExportWorker;
    }
}

struct GetImagesLine;

impl GetImagesLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<30} | {:<20} | {:<10} | {:<50}",
            "IMAGE NAME", "GROUP", "SCALER", "DESCRIPTION",
        );
        println!("{:-<31}+{:-<22}+{:-<12}+{:-<50}", "", "", "", "");
    }

    /// Print an image's info
    ///
    /// # Arguments
    ///
    /// * `image` - The image to print
    pub fn print_image(image: &Image) {
        // limit our description preview to at most 40 characters so the column stays aligned
        let description = utils::render::truncate_description(image.description.as_deref(), 40);
        // print our image info
        println!(
            "{:<30} | {:<20} | {:<10} | {}",
            image.name,
            image.group,
            image.scaler.as_str(),
            description
        );
    }
}

/// Get image info from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The image get command to execute
async fn get(thorium: Thorium, cmd: &GetImages) -> Result<(), Error> {
    GetImagesLine::header();
    // get the current user's groups if no groups were specified
    let groups = if cmd.groups.is_empty() {
        utils::groups::get_all_groups(&thorium).await?
    } else {
        cmd.groups.clone()
    };
    // get image cursors for all groups specified
    let image_cursors = groups.iter().map(|group| {
        let cursor = thorium
            .images
            .list(group)
            .page_size(cmd.page_size as u64)
            .details();
        // only apply a limit if the user didn't request no limit
        if cmd.no_limit {
            cursor
        } else {
            cursor.limit(cmd.limit as u64)
        }
    });
    // retrieve the images in each cursor until we've reached our limit
    // or all cursors are exhausted
    let mut images: Vec<Image> = Vec::new();
    for mut cursor in image_cursors {
        while !cursor.exhausted {
            cursor.next().await?;
            // remove images with a non-matching scaler
            if let Some(scaler) = &cmd.scaler {
                cursor.details.retain(|image| &image.scaler == scaler);
            }
            if cmd.alpha {
                // save images for sorting later if alphabetize flag is set
                images.append(&mut cursor.details);
            } else {
                // otherwise print immediately if no need to alphabetize
                cursor.details.iter().for_each(GetImagesLine::print_image);
            }
        }
    }
    // sort and print in alphabetical order if alpha flag was set
    if cmd.alpha {
        images.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        images.iter().for_each(GetImagesLine::print_image);
    }
    Ok(())
}

/// Describe a specific image in full
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The describe image command to execute
async fn describe(thorium: Thorium, cmd: &DescribeImages) -> Result<(), Error> {
    cmd.describe(&thorium).await
}

/// Delete images from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The delete images command to execute
async fn delete(thorium: Thorium, cmd: &crate::args::images::DeleteImages) -> Result<(), Error> {
    use colored::Colorize;
    // deleting is irreversible, so confirm exactly what will be removed
    if !cmd.skip_confirm {
        // fail clearly (not with a raw dialoguer error) when we can't prompt
        utils::require_confirm_terminal("--skip-confirm (-y)")?;
        println!("{}", "Images to delete:".bright_red());
        for image in &cmd.images {
            println!("  {}:{}", cmd.group, image);
        }
        let confirmed = dialoguer::Confirm::new()
            .with_prompt("Delete the images listed above?")
            .default(false)
            .interact()?;
        if !confirmed {
            return Ok(());
        }
    }
    for image in &cmd.images {
        match thorium.images.delete(&cmd.group, image).await {
            Ok(_) => println!("Deleted image '{}:{}'", cmd.group, image),
            // a missing image isn't fatal to the rest of the batch
            Err(err) if err.status() == Some(http::StatusCode::NOT_FOUND) => {
                eprintln!(
                    "{}: image '{}:{}' not found; skipping",
                    "Warning".bright_yellow(),
                    cmd.group,
                    image
                );
            }
            Err(err) => {
                return Err(Error::new(format!(
                    "Failed to delete image '{}:{}': {err}",
                    cmd.group, image
                )));
            }
        }
    }
    Ok(())
}

/// Import images into Thorium through the shared conflict engine
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The import images command to execute
/// * `conf` - The Thorctl config
/// * `workers` - The maximum number of concurrent workers to use
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub async fn import(
    thorium: &Thorium,
    cmd: &ImportImages,
    conf: &CtlConf,
    workers: usize,
) -> Result<(), Error> {
    let progress = Bar::new("", "Importing images", BarKind::Timer);
    let opts = import::ImageImportOpts::from_cmd(cmd, workers);
    // no explicit list imports every config in the export directory
    let names = if cmd.images.is_empty() {
        imports::list_export_configs(&cmd.import, "images").await?
    } else {
        imports::dedup_names(cmd.images.clone(), &progress)
    };
    // load the requests and check what already exists before changing anything
    let images = import::categorize_from_disk(thorium, &opts, &names, &progress).await?;
    // no pipelines for a standalone image import; the shared driver handles the rest
    imports::disk::run_disk_import(
        thorium,
        conf,
        &progress,
        &opts,
        images,
        Vec::new(),
        cmd.rollback_on_failure,
    )
    .await
}

/// Export images from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The export images command to execute
/// * `args` - The shared Thorctl args (for the worker count)
/// * `conf` - The Thorctl config
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub async fn export(
    thorium: &Thorium,
    cmd: &ExportImages,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // no explicit list exports every image in the group
    let names: Vec<String> = if cmd.images.is_empty() {
        crate::utils::images::list_all_images(thorium, &cmd.group)
            .await?
            .into_iter()
            .map(|image| image.name)
            .collect()
    } else {
        cmd.images.clone()
    };
    // pre-flight: write each config sequentially so on-disk conflicts can be
    // resolved interactively without the worker pool prompting concurrently
    let can_prompt = !cmd.skip_conflicts && std::io::stdin().is_terminal();
    let editor = crate::handlers::imports::editor::resolve_editor(None, conf).to_string();
    let mut resolver = DiskConflictResolver::new(cmd.overwrite, can_prompt, editor);
    let progress = Bar::new("images export", "Exporting configs", BarKind::Timer);
    let images_dir = cmd.output.join("images");
    // fetch the images concurrently (bounded by --workers); the writes below stay
    // sequential so the conflict resolver can prompt without racing
    let fetch_workers = std::cmp::min(args.workers, names.len()).max(1);
    let fetched: Vec<(String, Result<Image, Error>)> = stream::iter(names)
        .map(|name| async move {
            let result = thorium.images.get(&cmd.group, &name).await;
            (name, result)
        })
        .buffer_unordered(fetch_workers)
        .collect()
        .await;
    // images whose container tarball still needs exporting after their config lands,
    // paired with the container url so the worker pool doesn't have to re-fetch them
    let mut docker_jobs: Vec<(String, String)> = Vec::new();
    // names we couldn't fetch; collected so the export exits non-zero instead of
    // silently reporting success after skipping resources
    let mut failed: Vec<String> = Vec::new();
    for (name, result) in fetched {
        let image = match result {
            Ok(image) => image,
            Err(err) => {
                progress.error(format!("Failed to get image '{name}': {err}"));
                failed.push(name);
                continue;
            }
        };
        let request = ImageRequest::from(image.clone());
        // curated (prioritized) field order so an exported image config matches the layout `init`
        // and toolbox export produce — one consistent, edit-friendly format everywhere (still
        // deterministic: curated keys first, remaining keys sorted)
        let config_json = crate::utils::curated_json(
            &request,
            crate::handlers::imports::merge::IMAGE_FIELD_ORDER,
        )
        .map_err(|e| Error::new(format!("Failed to serialize image '{name}': {e}")))?;
        // optionally open the config in an editor for review before writing
        let config_json = if cmd.review {
            progress
                .suspend_async(crate::handlers::imports::editor::review_config_in_editor::<
                    thorium::models::ImageRequest,
                >(
                    &config_json,
                    &format!("export-image-{name}"),
                    &conf.default_editor,
                    crate::handlers::imports::merge::IMAGE_FIELD_ORDER,
                ))
                .await?
        } else {
            config_json
        };
        let outcome = resolver
            .write_yaml::<ImageRequest>(
                &images_dir.join(format!("{name}.json")),
                &config_json,
                &progress,
            )
            .await?;
        if outcome == WriteOutcome::Quit {
            progress.refresh("Export stopped early", BarKind::Timer);
            progress.finish();
            return Ok(());
        }
        // queue the container tarball export for images that have a url. When the
        // config write was skipped (identical on disk or a skipped conflict) avoid
        // redundantly re-pulling/saving a large tarball that's already present, but
        // still export one that's missing so a skipped config can't leave a resource
        // without its image.
        if !cmd.config_only
            && let Some(url) = &image.image
        {
            let tarball = images_dir.join(format!("{name}.tar.gz"));
            if outcome == WriteOutcome::Written || !tarball.exists() {
                docker_jobs.push((name.clone(), url.clone()));
            }
        }
    }
    progress.finish();
    // export container tarballs in parallel for the approved images
    if !docker_jobs.is_empty() {
        let workers = std::cmp::min(args.workers, docker_jobs.len());
        let mut controller = Controller::<ImageExportWorker>::spawn(
            "Exporting Images",
            thorium,
            workers,
            conf,
            args,
            cmd,
        )
        .await;
        for job in docker_jobs {
            if let Err(error) = controller.add_job(job).await {
                controller.error(&error.to_string());
            }
        }
        controller.finish().await?;
    }
    // surface skipped images as a non-zero exit so a partial export isn't read as success
    if !failed.is_empty() {
        return Err(Error::new(format!(
            "Failed to export {} image(s): {}",
            failed.len(),
            failed.join(", ")
        )));
    }
    Ok(())
}

/// Handle all images commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The reactions command to execute
pub async fn handle(args: &Args, cmd: &Images) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        super::update::ask_update(&thorium).await?;
    }
    // call the right reactions handler
    match cmd {
        Images::Get(cmd) => get(thorium, cmd).await,
        Images::Describe(cmd) => describe(thorium, cmd).await,
        Images::Notifications(cmd) => notifications::handle(thorium, cmd).await,
        Images::Bans(cmd) => bans::handle(thorium, cmd).await,
        Images::Edit(cmd) => edit::edit(thorium, &conf, cmd).await,
        Images::Delete(cmd) => delete(thorium, cmd).await,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        Images::Import(cmd) => {
            // resolve the container runtime (docker/podman) before any image work
            crate::handlers::container::init_runtime(
                args.container_runtime,
                conf.container_runtime,
            );
            import(&thorium, cmd, &conf, args.workers).await
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        Images::Export(cmd) => {
            // resolve the container runtime (docker/podman) before any image work
            crate::handlers::container::init_runtime(
                args.container_runtime,
                conf.container_runtime,
            );
            export(&thorium, cmd, args, &conf).await
        }
    }
}
