//! Collects and syncs reaction cache data

use async_walkdir::WalkDir;
use crossbeam::channel::Sender;
use futures::StreamExt;
use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use thorium::models::{
    GenericJob, Image, OnDiskFile, ReactionCache, ReactionCacheFileUpdate, ReactionCacheUpdate,
};
use thorium::{Error, Thorium};
use tracing::instrument;
use uuid::Uuid;

use crate::log;

/// The cache objects that were downloaded
#[derive(Debug, Clone, Default)]
pub struct DownloadedCache {
    /// The full cache that we downloaded
    pub cache: Option<ReactionCache>,
    /// The path to the generic cache map if one was downloaded
    pub generic: Option<PathBuf>,
    /// The files that were downloaded
    pub files: Vec<PathBuf>,
    /// When our cache files were downloaded
    pub downloaded_at: Option<SystemTime>,
}

/// Sync this jobs generic cache file with our reactions cache
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `job` - The job that was just executed
/// * `reaction` - The id of the reaction to sync our cache against
/// * `root` - The root path to our cache on disk
/// * `downloaded_at` - When this cache was initiall downloaded
/// * `logs` - A stream of logs to send to Thorium for this job
#[instrument(name = "cache:sync_generic_cache", skip_all, err(Debug))]
async fn sync_generic_cache(
    thorium: &Thorium,
    job: &GenericJob,
    reaction: Uuid,
    root: &Path,
    old: &DownloadedCache,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // build the full path to our generic cache data
    let generic_path = root.join("generic.json");
    // get when our generic cache was last modified
    let modified = match tokio::fs::metadata(&generic_path).await {
        // get when this file was modified
        Ok(meta) => meta.modified()?,
        // if this cache file just doesn't exist then just return
        Err(error) => {
            // check if this was a file not found error
            match error.kind() {
                // this cache file doesn't exist just skip it
                ErrorKind::NotFound => return Ok(()),
                // reraise this io error
                _ => return Err(Error::IO(error)),
            }
        }
    };
    // check if this was modified after our cache was fully downloaded
    if Some(modified) > old.downloaded_at {
        // log that we are updating our generic cache
        log!(logs, "Updating generic cache");
        // load this cache and send it to Thorium
        let data = tokio::fs::read(&generic_path).await?;
        // deserialize our cache data
        let mut generic: HashMap<String, String> = serde_json::from_slice(&data)?;
        // only filter new/changed keys and buld a remove key list if we had an old cache
        let remove_generic = match &old.cache {
            Some(old_cache) => {
                // get the generic keys to remove from our cache
                let remove_generic = old_cache
                    .generic
                    .keys()
                    .filter(|key| !generic.contains_key(*key))
                    .map(|key| key.to_owned().clone())
                    .collect::<Vec<String>>();
                // only add new or updated cache items to our cache
                generic.retain(|key, value| {
                    // we only need to sync new/changed keys
                    match old_cache.generic.get(key) {
                        Some(old_value) if value != old_value => true,
                        Some(_) => false,
                        None => true,
                    }
                });
                remove_generic
            }
            None => Vec::default(),
        };
        // place this generic cache back in our cache
        let update = ReactionCacheUpdate {
            generic,
            remove_generic,
        };
        // sync our updated cache to Thorium
        thorium
            .reactions
            .update_cache(&job.group, reaction, &update)
            .await?;
    }
    Ok(())
}

// Sync any cache files if they are new or have changed
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `job` - The job that was just executed
/// * `reaction` - The id of the reaction to sync our cache against
/// * `root` - The root path to our cache on disk
/// * `downloaded_at` - When this cache was initiall downloaded
/// * `logs` - A stream of logs to send to Thorium for this job
#[instrument(name = "cache:sync_cache_files", skip_all, err(Debug))]
async fn sync_cache_files(
    thorium: &Thorium,
    job: &GenericJob,
    reaction: Uuid,
    root: &Path,
    old: &DownloadedCache,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // build the path to our cache files
    let cache_file_path = root.join("files");
    // build a walker over all of our cache files
    // check each one to see if its new or has changed
    let mut walker = WalkDir::new(&cache_file_path);
    // get the next entry to check
    while let Some(entry) = walker.next().await {
        // check if we failed to read this entry
        let entry = entry.unwrap();
        // get this files metadata
        let meta = entry.metadata().await?;
        // only check if files have changed
        if meta.is_file() {
            // check if this file was modified after we downloaded our cache
            if Some(meta.modified()?) > old.downloaded_at {
                // get this entries path
                let entry_path = entry.path();
                // log that we are updating this cache file
                log!(
                    logs,
                    format!("Updating {} cache file", entry_path.display())
                );
                // build the update to sync this cache file update with
                let update = ReactionCacheFileUpdate::default()
                    .file(OnDiskFile::new(entry_path).trim_prefix(&cache_file_path));
                // this file has been modified so sync its new state to Thorium
                thorium
                    .reactions
                    .update_cache_files(&job.group, reaction, update)
                    .await?;
            }
        }
    }
    Ok(())
}

/// Scan and sync and changes or new files to this reactions cache
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image that this agent is executing jobs in
/// * `job` - The job that was just executed
/// * `root` - The root path to our cache on disk
/// * `downloaded_at` - When this cache was initiall downloaded
/// * `logs` - A stream of logs to send to Thorium for this job
#[instrument(name = "cache:sync", skip_all, err(Debug))]
pub async fn sync<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    root: P,
    old: &DownloadedCache,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // don't bother syncing our cache if its not enabled
    if image.dependencies.cache.enabled {
        // convert our root to a path
        let root = root.as_ref();
        // get our reaction id or our parents reaction id if we are using our parent cache
        let reaction = image.dependencies.cache.get_reaction_id(job);
        // sync our generic cache if it has changed
        sync_generic_cache(thorium, job, reaction, root, old, logs).await?;
        // sync any new cache files
        sync_cache_files(thorium, job, reaction, root, old, logs).await?;
    }
    Ok(())
}
