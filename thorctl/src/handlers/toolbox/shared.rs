//! Functionality shared between toolbox handlers

use colored::Colorize;
use futures::StreamExt;
use http::header::CONTENT_LENGTH;
use std::path::Path;
use thorium::Error;
use tokio::io::AsyncReadExt;
use url::Url;

use crate::args::toolbox::ManifestLocation;
use crate::handlers::progress::{Bar, BarKind};
use crate::handlers::toolbox::manifest::{DroppedItems, ToolboxManifest};

/// Warn through the progress bar about every image/pipeline version a validation
/// pass dropped, so skipped resources are visible before anything is applied
///
/// Shared by `toolbox import` (before the confirmation screen) and `toolbox
/// export` (before writing to disk).
///
/// # Arguments
///
/// * `dropped` - The report returned by a `validate_*` pass
/// * `progress` - The progress bar to log warnings through
pub fn warn_dropped(dropped: &DroppedItems, progress: &Bar) {
    // emit one warning per dropped image; `warning` (not `info_anonymous`) is used so
    // skipped resources still reach the user in quiet mode or when output isn't a tty
    // (warnings fall back to stderr), keeping silent drops from going unnoticed
    for (name, reason) in &dropped.images {
        progress.warning(format!(
            "Skipping invalid image '{}': {}",
            name.bright_yellow(),
            reason
        ));
    }
    // emit one warning per dropped pipeline; a pipeline can be dropped for several
    // independent reasons at once, so join them into a single readable line
    for (name, reasons) in &dropped.pipelines {
        progress.warning(format!(
            "Skipping invalid pipeline '{}': {}",
            name.bright_yellow(),
            reasons.join("; ")
        ));
    }
}

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
    // branch on whether the manifest lives at a URL or a local path; the two sources
    // need different fetch logic and a differently-worded progress message
    match location {
        ManifestLocation::Url(manifest_url) => {
            // start the bar unbounded since the content length isn't known until the
            // response headers arrive inside the URL fetch
            let progress = Bar::new("", "Downloading manifest...", BarKind::UnboundIO);
            let manifest = get_manifest_from_url(manifest_url, &progress).await?;
            Ok((manifest, progress))
        }
        ManifestLocation::Path(manifest_path) => {
            // start the bar unbounded; the file size isn't known until the path fetch
            // stats the file, at which point it switches the bar to a bounded mode
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
    // issue the GET; a transport-level failure (DNS, connection) surfaces here before
    // any status code is known
    let resp = reqwest::get(url.clone())
        .await
        .map_err(|err| Error::new(format!("Error downloading toolbox manifest: {err}")))?;
    // turn a non-2xx status (404, 500, ...) into an error so we don't try to parse an
    // error page body as a manifest
    match resp.error_for_status() {
        Ok(resp) => {
            // bound the bar when the size is known: prefer reqwest's parsed length,
            // falling back to parsing the raw Content-Length header ourselves, since
            // some responses (e.g. chunked) leave `content_length()` as None
            if let Some(content_length) = resp.content_length().or(resp
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|content_length_header| content_length_header.to_str().ok())
                .and_then(|content_length_str| content_length_str.parse::<u64>().ok()))
            {
                progress.refresh("Downloading manifest...", BarKind::IO(content_length));
            } else {
                progress.refresh("Downloading manifest...", BarKind::UnboundIO);
            }
            // stream the body chunk by chunk into a buffer, advancing the bar as bytes
            // arrive; the bytes are collected (rather than handed to reqwest's `.json()`)
            // so we can run serde_json ourselves and emit a more descriptive parse error
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
            // parse the fully buffered body into the manifest model
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
    // open the file up front so a missing/unreadable path fails before the bar advances
    let mut manifest_file = tokio::fs::File::open(path).await.map_err(|err| {
        Error::new(format!(
            "Error opening manifest file '{}': {}",
            path.display(),
            err
        ))
    })?;
    // bound the bar with the file size when stat succeeds; a stat failure is non-fatal
    // (`.ok()` swallows it) and simply leaves the bar unbounded rather than aborting
    match manifest_file
        .metadata()
        .await
        .ok()
        .map(|metadata| metadata.len())
    {
        Some(file_len) => progress.refresh("Reading manifest file...", BarKind::IO(file_len)),
        None => progress.refresh("Reading manifest file...", BarKind::UnboundIO),
    }
    // read the whole file into a buffer, advancing the bar per read; a `read_buf` of 0
    // bytes signals EOF and ends the loop
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
    // parse the fully buffered file into the manifest model
    serde_json::from_slice(&manifest_bytes)
        .map_err(|err| Error::new(format!("Malformed toolbox manifest: {err}")))
}

/// Fetch a JSON config from a URL and deserialize it
///
/// # Arguments
///
/// * `url` - The URL to fetch the JSON config from
async fn fetch_json_config<T: serde::de::DeserializeOwned>(url: &str) -> Result<T, Error> {
    // issue the GET; transport-level failures surface here before any status is known
    let resp = reqwest::get(url)
        .await
        .map_err(|e| Error::new(format!("Failed to fetch config from '{url}': {e}")))?;
    // reject a non-2xx status so an error page body is never parsed as a config
    let resp = resp
        .error_for_status()
        .map_err(|e| Error::new(format!("Failed to fetch config from '{url}': {e}")))?;
    // buffer the whole body; configs are small, so there's no streaming/progress here
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| Error::new(format!("Failed to read config response from '{url}': {e}")))?;
    // deserialize into the caller-chosen request type (ImageRequest / PipelineRequest /
    // NetworkPolicyRequest), keeping the source url in the error for diagnosability
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::new(format!("Failed to parse config from '{url}': {e}")))
}

/// Resolve any URL-based configs and network policies in the manifest by fetching them
///
/// Only versions that carry a `config_from` URL (and no inline `config`) have their
/// config fetched; likewise every `network_policies_from` URL is fetched and folded
/// into the version's `network_policies`. Versions with neither an inline config nor a
/// `config_from` are left with `config` as `None` — they are not resolved here and are
/// dropped later by structural validation.
///
/// # Arguments
///
/// * `manifest` - The manifest whose URL-based configs/policies are fetched in place
/// * `progress` - The progress bar tracking the fetches
pub async fn resolve_manifest_configs(
    manifest: &mut ToolboxManifest,
    progress: &Bar,
) -> Result<(), Error> {
    use thorium::models::{ImageRequest, NetworkPolicyRequest, PipelineRequest};
    // pre-count every remote fetch so the bar can be bounded; this counting pass must
    // mirror the fetching passes below exactly, or the bar total will drift from reality
    let mut url_count = 0u64;
    for image_manifest in manifest.images.values() {
        for version in image_manifest.versions.values() {
            // an image config is fetched only when sourced from a URL and not already
            // inline; an inline `config` takes precedence and skips the fetch
            if version.config_from.is_some() && version.config.is_none() {
                url_count += 1;
            }
            // every `network_policies_from` entry is a URL that will be fetched, so each
            // counts toward the total regardless of inline config
            url_count += version.network_policies_from.len() as u64;
        }
    }
    for pipeline_manifest in manifest.pipelines.values() {
        for version in pipeline_manifest.versions.values() {
            // pipelines have no network policies, so only the URL-sourced config counts
            if version.config_from.is_some() && version.config.is_none() {
                url_count += 1;
            }
        }
    }
    // short-circuit when there's nothing remote to fetch, leaving the bar in whatever
    // (unbounded) state the caller set so we don't flash an empty bounded bar
    if url_count == 0 {
        return Ok(());
    }
    // now that the exact total is known, switch the bar to bounded so it shows real progress
    progress.refresh("Fetching remote configs", BarKind::Bound(url_count));
    // resolve each image version's URL-sourced config and network policies in place
    for image_manifest in manifest.images.values_mut() {
        for version in image_manifest.versions.values_mut() {
            // fetch the config only when it's URL-sourced and not already inline, matching
            // the counting pass; embed the result so downstream validation sees a config
            if let Some(url) = &version.config_from
                && version.config.is_none()
            {
                let config: ImageRequest = fetch_json_config(url).await?;
                version.config = Some(config);
                progress.inc(1);
            }
            // drain each policy URL into the resolved `network_policies` list; draining
            // (rather than iterating) moves the urls out so they aren't re-fetched and the
            // resolved manifest no longer carries unresolved `network_policies_from` urls
            for url in version.network_policies_from.drain(..) {
                let policy: NetworkPolicyRequest = fetch_json_config(&url).await?;
                version.network_policies.push(policy);
                progress.inc(1);
            }
        }
    }
    // resolve each pipeline version's URL-sourced config in place (pipelines carry no policies)
    for pipeline_manifest in manifest.pipelines.values_mut() {
        for version in pipeline_manifest.versions.values_mut() {
            // fetch the config only when it's URL-sourced and not already inline
            if let Some(url) = &version.config_from
                && version.config.is_none()
            {
                let config: PipelineRequest = fetch_json_config(url).await?;
                version.config = Some(config);
                progress.inc(1);
            }
        }
    }
    Ok(())
}
