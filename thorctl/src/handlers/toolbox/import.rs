//! Handlers for importing a toolbox to Thorium

use colored::Colorize;
use futures::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use thorium::models::ScrubbedUser;
use thorium::{CtlConf, Error, Thorium};

use super::manifest::{ImageManifest, PipelineManifest, ToolboxManifest};
use crate::args::toolbox::ImportToolbox;
use crate::handlers::progress::{Bar, BarKind};
use crate::handlers::toolbox::shared;

/// Proceed on 409 CONFLICT errors, printing the given
/// message to stdout if we get a 409
macro_rules! proceed_on_conflict {
    ($fallible:expr, $progress:expr, $($arg:tt)*) => {{
        match $fallible {
            Ok(_) => Ok(()),
            Err(err) => match err.status() {
                Some(status) => {
                    if status == http::StatusCode::CONFLICT {
                        // print a log line to the progress bar
                        $progress.info_anonymous(format!($($arg)*));
                        Ok(())
                    } else {
                        Err(err)
                    }
                }
                None => Err(err),
            },
        }
    }};
}

/// Import the manifest's images
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `images` - The images to import
/// * `progress` - The progress bar
async fn import_images(
    thorium: &Thorium,
    images: &HashMap<String, ImageManifest>,
    progress: &Bar,
) -> Result<(), Error> {
    // make an iterator over each image to create
    let image_iter = images.iter().flat_map(|(image_name, manifest)| {
        manifest
            .versions
            .iter()
            .map(move |(version_name, image)| (image_name, version_name, image))
    });
    progress.refresh(
        "Importing images",
        BarKind::Bound(image_iter.clone().count() as u64),
    );
    // create the images concurrently
    stream::iter(image_iter)
        .map(Ok::<_, Error>)
        .try_for_each_concurrent(10, |(image_name, version_name, image)| async move {
            proceed_on_conflict!(
                thorium.images.create(&image.config).await,
                progress,
                "image '{}:{}' already exists in group '{}'! Skipping import...",
                image_name.bright_yellow(),
                version_name.bright_yellow(),
                image.config.group.bright_yellow()
            )
            .map_err(|err| {
                Error::new(format!(
                    "Error importing image '{image_name}:{version_name}': {err}"
                ))
            })?;
            progress.inc(1);
            Ok(())
        })
        .await?;
    Ok(())
}

/// Import the manifest's pipelines
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `pipelines` - The pipelines to import
/// * `progress` - The progress bar
async fn import_pipelines(
    thorium: &Thorium,
    pipelines: &HashMap<String, PipelineManifest>,
    progress: &Bar,
) -> Result<(), Error> {
    // make an iterator over each pipeline to create
    let pipeline_iter = pipelines.iter().flat_map(|(pipeline_name, manifest)| {
        manifest
            .versions
            .iter()
            .map(move |(version_name, pipeline)| (pipeline_name, version_name, pipeline))
    });
    progress.refresh(
        "Importing pipelines",
        BarKind::Bound(pipeline_iter.clone().count() as u64),
    );
    // create the pipelines concurrently
    stream::iter(pipeline_iter)
        .map(Ok::<_, Error>)
        .try_for_each_concurrent(10, |(pipeline_name, version_name, pipeline)| async move {
            proceed_on_conflict!(
                thorium.pipelines.create(&pipeline.config).await,
                progress,
                "Pipeline '{}:{}' already exists in group '{}'! Skipping import...",
                pipeline_name.bright_yellow(),
                version_name.bright_yellow(),
                pipeline.config.group.bright_yellow()
            )
            .map_err(|err| {
                Error::new(format!(
                    "Error importing pipeline '{pipeline_name}:{version_name}': {err}"
                ))
            })?;
            progress.inc(1);
            Ok(())
        })
        .await?;
    Ok(())
}

/// Confirm the manifest with the user
///
/// # Arguments
///
/// * `conf` - The Thorctl config
/// * `manifest` - The manifest to confirm
/// * `manifest_groups` - The groups the manifest expects to exist
/// * `current_user` - The user importing the manifest
fn confirm_manifest(
    conf: &CtlConf,
    manifest: &ToolboxManifest,
    manifest_groups: &HashSet<String>,
    current_user: &ScrubbedUser,
) -> Result<bool, Error> {
    // display the manifest's info
    println!("{}", "Images:".bright_yellow());
    for image_name in manifest.images.keys().sorted_unstable() {
        println!("  {image_name}");
    }
    println!("\n{}", "Pipelines:".bright_yellow());
    for pipeline_name in manifest.pipelines.keys().sorted_unstable() {
        println!("  {pipeline_name}");
    }
    println!("\n{}", "Groups:".bright_yellow());
    for group in manifest_groups.iter().sorted_unstable() {
        println!("  {group}");
    }
    println!();
    // confirm with the user that they want to import
    let response = dialoguer::Confirm::new()
        .with_prompt(format!(
            "Import the above items to Thorium instance at '{}' as user '{}'?",
            conf.keys.api.bright_green(),
            current_user.username.bright_green()
        ))
        .interact()?;
    Ok(response)
}

/// Import a toolbox into Thorium by the given manifest file
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `cmd` - The toolbox import command that was run
pub async fn import(thorium: Thorium, conf: CtlConf, cmd: &ImportToolbox) -> Result<(), Error> {
    // get the manifest from the location along with a progress bar
    let (mut manifest, progress) = shared::get_manifest(&cmd.manifest).await?;
    // validate the manifest
    if let Err(err) = manifest.validate() {
        return Err(Error::new(format!(
            "Invalid toolbox manifest: {}",
            err.msg().unwrap_or_else(|| "Unknown error".to_string())
        )));
    }
    // get all the groups the manifest expects to exist, overriding them if we're set to
    let manifest_groups = if let Some(group_override) = &cmd.group_override {
        progress.info_anonymous(format!(
            "Overriding all image/pipeline import groups to '{}'",
            group_override.bright_yellow()
        ));
        // replace all groups in the manifest with the override and get the modified manifest
        manifest = manifest.override_group(group_override);
        // validate the manifest again after overriding groups
        if let Err(err) = manifest.validate() {
            return Err(Error::new(format!(
                "Invalid toolbox manifest after group override '{}': {}",
                group_override.bright_yellow(),
                err.msg().unwrap_or_else(|| "Unknown error".to_string())
            )));
        }
        // return a set with just our group override since we replaced it
        HashSet::from([group_override.to_string()])
    } else {
        // get all of the groups the manifest refers to
        manifest.groups()
    };
    // confirm with the user that it's okay to import the manifest
    if !cmd.skip_confirm {
        // get info on the current user
        let current_user = thorium
            .users
            .info()
            .await
            .map_err(|err| Error::new(format!("Error getting current user info: {err}")))?;
        let confirmed = progress
            .suspend(|| confirm_manifest(&conf, &manifest, &manifest_groups, &current_user))?;
        if !confirmed {
            return Ok(());
        }
    }
    // create any groups the manifest expects that Thorium doesn't yet have
    shared::get_and_create_missing_groups(&thorium, manifest_groups, &progress).await?;
    // first import the manifest's images
    import_images(&thorium, &manifest.images, &progress)
        .await
        .map_err(|err| Error::new(format!("Error importing images: {err}")))?;
    // then import the manifest's pipelines
    import_pipelines(&thorium, &manifest.pipelines, &progress)
        .await
        .map_err(|err| Error::new(format!("Error importing pipelines: {err}")))?;
    // inform the user the import is complete
    progress.refresh("Import complete!", BarKind::Timer);
    progress.finish();
    Ok(())
}
