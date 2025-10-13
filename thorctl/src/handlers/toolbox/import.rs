use colored::Colorize;
use futures::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use reqwest::header::CONTENT_LENGTH;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use thorium::models::GroupRequest;
use thorium::{CtlConf, Error, Thorium};
use tokio::io::AsyncReadExt;
use url::Url;

use super::manifest::{ImageManifest, PipelineManifest, ToolboxManifest};
use crate::args::toolbox::{ImportToolbox, ManifestLocation};
use crate::handlers::progress::{Bar, BarKind};

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

/// Create any groups the manifest expects to exist that are missing in the
/// Thorium instance
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `manifest_groups` - The groups the manifest expects to exist
/// * `progress` - The progress bar
async fn create_missing_groups(
    thorium: &Thorium,
    mut manifest_groups: HashSet<String>,
    progress: &Bar,
) -> Result<(), Error> {
    // get all existing groups already in Thorium
    let mut thorium_groups = HashSet::new();
    // use a very large limit to make sure we get all groups
    let mut cursor = thorium.groups.list().limit(1_000_000);
    loop {
        cursor
            .next()
            .await
            .map_err(|err| Error::new(format!("Error listing groups: {err}")))?;
        thorium_groups.extend(cursor.names.drain(..));
        if cursor.exhausted {
            break;
        }
    }
    // calculate which groups are missing
    let missing_groups: Vec<String> = manifest_groups
        .extract_if(|manifest_group| !thorium_groups.contains(manifest_group))
        .collect();
    if !missing_groups.is_empty() {
        progress.refresh(
            "Importing groups",
            BarKind::Bound(missing_groups.len() as u64),
        );
        // create all the missing groups;
        // we only want to create the groups that are needed because group create
        // returns a 401 rather than a 409 if the group already exists
        stream::iter(missing_groups)
            .map(Ok::<_, Error>)
            .try_for_each_concurrent(10, |missing_group| async {
                let group_request = GroupRequest::new(missing_group);
                thorium.groups.create(&group_request).await?;
                progress.inc(1);
                Ok(())
            })
            .await
            .map_err(|err| Error::new(format!("Error creating missing groups: {err}")))?;
    }
    Ok(())
}

/// Confirm the manifest with the user
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `manifest` - The manifest to confirm
/// * `manifest_groups` - The groups the manifest expects to exist
async fn confirm_manifest(
    thorium: &Thorium,
    conf: &CtlConf,
    manifest: &ToolboxManifest,
    manifest_groups: &HashSet<String>,
) -> Result<bool, Error> {
    // get info on the current user
    let current_user = thorium
        .users
        .info()
        .await
        .map_err(|err| Error::new(format!("Error getting current user info: {err}")))?;
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

/// Read and parse the manifest file at the given URL
///
/// # Arguments
///
/// * `url` - The manifest URL
/// * `progress` - The progress bar
async fn get_manifest_from_url(url: &Url, progress: &Bar) -> Result<ToolboxManifest, Error> {
    // get the manifest file from the URL
    let resp = reqwest::get(url.clone())
        .await
        .map_err(|err| Error::new(format!("Error downloading toolbox manifest: {err}")))?;
    // check if the response was an error
    match resp.error_for_status() {
        Ok(resp) => {
            // try to get the content length for the progress bar
            if let Some(content_length) = resp.content_length().or(resp
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|content_length_header| {
                    println!("Got content length header");
                    content_length_header.to_str().ok()
                })
                .and_then(|content_length_str| {
                    println!("Got content length str: {content_length_str}");
                    content_length_str.parse::<u64>().ok()
                }))
            {
                progress.refresh("Downloading manifest...", BarKind::IO(content_length));
            } else {
                progress.refresh("Downloading manifest...", BarKind::UnboundIO);
            }
            // get the manifest file as bytes;
            // we use serde_json here instead of reqwest's JSON capabilities for
            // better error logging
            let mut manifest_bytes = Vec::new();
            let mut manifest_bytes_stream = resp.bytes_stream();
            while let Some(bytes) = manifest_bytes_stream.next().await {
                let bytes = bytes.map_err(|err| {
                    Error::new(format!(
                        "Error downloading toolbox manifest response body: {err}"
                    ))
                })?;
                progress.inc(bytes.len() as u64);
                manifest_bytes.extend_from_slice(&bytes);
            }
            // parse the manifest data
            serde_json::from_slice(&manifest_bytes)
                .map_err(|err| Error::new(format!("Malformed toolbox manifest: {err}")))
        }
        Err(err) => Err(Error::new(format!(
            "Error downloading toolbox manifest: {err}"
        ))),
    }
}

/// Read and parse the manifest file at the given path
///
/// # Arguments
///
/// * `path` - The manifest file path
/// * `progress` - The progress bar
async fn get_manifest_from_path(path: &Path, progress: &Bar) -> Result<ToolboxManifest, Error> {
    // open the manifest file at the path
    let mut manifest_file = tokio::fs::File::open(path).await.map_err(|err| {
        Error::new(format!(
            "Error opening manifest file '{}': {}",
            path.display(),
            err
        ))
    })?;
    // try to get the file's length
    match manifest_file
        .metadata()
        .await
        .ok()
        .map(|metadata| metadata.len())
    {
        Some(file_len) => progress.refresh("Reading manifest file...", BarKind::IO(file_len)),
        None => progress.refresh("Reading manifest file...", BarKind::UnboundIO),
    }
    // read the file
    let mut manifest_bytes = Vec::new();
    loop {
        let bytes_read = manifest_file
            .read_buf(&mut manifest_bytes)
            .await
            .map_err(|err| {
                Error::new(format!(
                    "Error reading manifest file '{}': {}",
                    path.display(),
                    err
                ))
            })?;
        if bytes_read == 0 {
            break;
        }
        progress.inc(bytes_read as u64);
    }
    // parse the manifest file
    serde_json::from_slice(&manifest_bytes)
        .map_err(|err| Error::new(format!("Malformed toolbox manifest file: {err}")))
}

/// Import a toolbox into Thorium by the given manifest file
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `cmd` - The toolbox import command that was run
pub async fn import(thorium: Thorium, conf: CtlConf, cmd: &ImportToolbox) -> Result<(), Error> {
    // get the toolbox manifest by URL or file path
    let (mut manifest, progress) = match &cmd.manifest {
        ManifestLocation::Url(manifest_url) => {
            // create the progress bar
            let progress = Bar::new("", "Downloading manifest...", BarKind::UnboundIO);
            let manifest = get_manifest_from_url(manifest_url, &progress).await?;
            (manifest, progress)
        }
        ManifestLocation::Path(manifest_path) => {
            // create the progress bar
            let progress = Bar::new("", "Reading manifest file...", BarKind::UnboundIO);
            let manifest = get_manifest_from_path(manifest_path, &progress).await?;
            (manifest, progress)
        }
    };
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
        let confirmed = confirm_manifest(&thorium, &conf, &manifest, &manifest_groups).await?;
        if !confirmed {
            return Ok(());
        }
    }
    // create any groups the manifest expects that Thorium doesn't yet have
    create_missing_groups(&thorium, manifest_groups, &progress).await?;
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
