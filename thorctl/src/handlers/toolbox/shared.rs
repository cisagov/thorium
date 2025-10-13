//! Functionality shared between toolbox handlers

use futures::{StreamExt, TryStreamExt, stream};
use http::header::CONTENT_LENGTH;
use std::collections::HashSet;
use std::path::Path;
use thorium::models::GroupRequest;
use thorium::{Error, Thorium};
use tokio::io::AsyncReadExt;
use url::Url;

use crate::args::toolbox::ManifestLocation;
use crate::handlers::progress::{Bar, BarKind};
use crate::handlers::toolbox::manifest::ToolboxManifest;

/// Get a [`ToolboxManifest`] from a [`ManifestLocation`]
///
/// # Arguments
///
/// * `location` - The location the manifest is found at
///
/// # Returns
///
/// Returns the [`ToolboxManifest`] along with a [`Bar`] used to track download/reading progress
pub async fn get_manifest(location: &ManifestLocation) -> Result<(ToolboxManifest, Bar), Error> {
    // get the toolbox manifest by URL or file path
    match location {
        ManifestLocation::Url(manifest_url) => {
            // create the progress bar
            let progress = Bar::new("", "Downloading manifest...", BarKind::UnboundIO);
            let manifest = get_manifest_from_url(manifest_url, &progress).await?;
            Ok((manifest, progress))
        }
        ManifestLocation::Path(manifest_path) => {
            // create the progress bar
            let progress = Bar::new("", "Reading manifest file...", BarKind::UnboundIO);
            let manifest = get_manifest_from_path(manifest_path, &progress).await?;
            Ok((manifest, progress))
        }
    }
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

/// Get the list of groups missing in Thorium that the manifest expects
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `manifest_groups` - The groups the manifest expects to exist
pub async fn get_missing_groups(
    thorium: &Thorium,
    mut manifest_groups: HashSet<String>,
) -> Result<Vec<String>, Error> {
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
    Ok(manifest_groups
        .extract_if(|manifest_group| !thorium_groups.contains(manifest_group))
        .collect())
}

/// Create all of the given groups in Thorium and increment the progress bar
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `groups` - The groups to create
/// * `progress` - The progress bar
pub async fn create_groups<T>(
    thorium: &Thorium,
    groups: Vec<T>,
    progress: &Bar,
) -> Result<(), Error>
where
    T: Into<String>,
{
    // create groups concurrently
    stream::iter(groups)
        .map(Ok::<_, Error>)
        .try_for_each_concurrent(10, |missing_group| async {
            let group_request = GroupRequest::new(missing_group);
            thorium.groups.create(&group_request).await?;
            progress.inc(1);
            Ok(())
        })
        .await
        .map_err(|err| Error::new(format!("Error creating missing groups: {err}")))?;
    Ok(())
}

/// See which groups in the manifest are missing from Thorium and create them
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `manifest_groups` - The groups the manifest expects to exist
/// * `progress` - The progress bar
pub async fn get_and_create_missing_groups(
    thorium: &Thorium,
    manifest_groups: HashSet<String>,
    progress: &Bar,
) -> Result<(), Error> {
    // calculate which groups are missing
    let missing_groups = get_missing_groups(thorium, manifest_groups).await?;
    // create the missing groups if we have any
    if !missing_groups.is_empty() {
        progress.refresh(
            "Importing groups",
            BarKind::Bound(missing_groups.len() as u64),
        );
        // create all the missing groups;
        // we only want to create the groups that are needed because group create
        // returns a 401 rather than a 409 if the group already exists
        create_groups(thorium, missing_groups, progress).await?;
    }
    Ok(())
}
