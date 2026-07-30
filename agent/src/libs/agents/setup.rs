//! Setup an environment for executing a Thorium job

use crossbeam::channel::Sender;
use futures::{StreamExt, stream};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use thorium::Error;
use thorium::Thorium;
use thorium::client::ResultsClient;
use thorium::models::{
    Association, AssociationKind, AssociationTarget, DependencyPassStrategy, Directionality,
    EntityKinds, EntityListOpts, EntityMetadata, FileDownloadOpts, FileNamingStrategy,
    FileSystemDependencySettings, GenericJob, Image, ReactionCache, RepoDownloadOpts,
    ResultGetParams, TreeNode, TreeOpts, TreeQuery, TreeRelationships,
};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::{Level, event, instrument};
use uuid::Uuid;

use crate::libs::DownloadedCache;
use crate::{log, purge};

/// Create any required parent dirs for this file
///
/// # Arguments
///
/// * `path` - The path to check the parent dirs for
/// * `created_dirs` - The set of directories we have already created
async fn create_parents(path: &Path, created_dirs: &mut HashSet<PathBuf>) -> Result<(), Error> {
    // get this result files parent dir
    if let Some(parent) = path.parent() {
        // if this parent dir isn't already in our created map then create it
        if !created_dirs.contains(parent) {
            // create all of our parent dirs
            tokio::fs::create_dir_all(&parent).await?;
            // add this to our created dirs set
            created_dirs.insert(parent.to_path_buf());
        }
    }
    Ok(())
}

/// Download a reactions generic cache
///
/// # Arguments
///
/// * `location` - The path to download cache data too
/// * `cache` - This reactions cache
/// * `downloading` - The cache data that has been downloaded
/// * `logs` - This jobs logs
#[instrument(name = "setup::write_generic_cache", skip_all, err(Debug))]
pub async fn write_generic_cache(
    location: &Path,
    cache: &ReactionCache,
    downloaded: &mut DownloadedCache,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // Only write if our cache contains data
    if !cache.generic.is_empty() {
        // build the path to store our generic cache info
        let cache_path = location.join("generic.json");
        // serialize our generic cache
        let serialized = serde_json::to_string(&cache.generic)?;
        // download and uncart this file to disk
        log!(logs, "Writing generic cache to {}", cache_path.display());
        // write our generic cache to disk
        tokio::fs::write(&cache_path, &serialized).await?;
        // set the path that we downloaded our cache too
        downloaded.generic = Some(cache_path);
    }
    Ok(())
}

/// Help download cache files
///
/// # Arguments
///
/// * `thorium` - A thorium client
/// * `job` - The current job we are executing
/// * `reaction` - The id of the reaction we are downloading cache files from
/// * `sub` - The relative path for the cache file we are downloading
/// * `write_path` - The path to write this downloaded file too
/// * `logs` - This jobs logs
#[instrument(
    name = "setup::download_cache_file_helper",
    skip(thorium, job, write_path),
    err(Debug)
)]
async fn download_cache_file_helper(
    thorium: &Thorium,
    job: &GenericJob,
    reaction: Uuid,
    sub: &str,
    write_path: PathBuf,
    mut logs: Sender<String>,
) -> Result<(), Error> {
    // build our options for downloading this file
    // make sure to always uncart these files
    let mut opts = FileDownloadOpts::default().uncart();
    // log that we are downloading this cache file
    log!(&mut logs, "Downloading cache file {sub}");
    // make any required sub folders if needed
    if let Some(parent) = write_path.parent() {
        // create our parent folders
        tokio::fs::create_dir_all(parent)
            .await
            // give our error some extra context on what failed
            .map_err(|error| {
                Error::new(format!(
                    "Failed to create parent dir {}: {error}",
                    parent.display()
                ))
            })?;
    }
    // create a future download this cache file and stream it to disk
    thorium
        .reactions
        .download_from_cache(&job.group, reaction, sub, write_path, &mut opts)
        .await
        // give our error some extra context on what failed
        .map_err(|error| Error::new(format!("Failed to download cache file '{sub}' : {error}")))?;
    Ok(())
}

/// Download our reactiosn cache file to disk
///
/// # Arguments
///
/// * `thorium` - A thorium client
/// * `location` - The path to download cache data too
/// * `cache` - This reactions cache
/// * `job` - The current job we are executing
/// * `reaction` - The id of the reaction we are downloading cache files from
/// * `downloading` - The cache data that has been downloaded
/// * `logs` - This jobs logs
#[instrument(name = "setup::download_cache_files", skip_all, err(Debug))]
pub async fn download_cache_files(
    thorium: &Thorium,
    location: &Path,
    cache: &ReactionCache,
    job: &GenericJob,
    reaction: Uuid,
    downloaded: &mut DownloadedCache,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // build a path to store our cache files at
    let file_root = location.join("files");
    // create a folder for our cache files
    tokio::fs::create_dir(&file_root).await?;
    // only download cache files if we have some
    if !cache.files.is_empty() {
        // build a set of futures for downloading our files
        let mut futs = Vec::with_capacity(cache.files.len());
        // build a future for downloading all of our cache files
        for sub in &cache.files {
            // get the path to write this file to on disk
            let write_path = file_root.join(sub);
            // add this path to our downloaded paths
            downloaded.files.push(write_path.clone());
            // clone our logs channel
            let local_logs = logs.clone();
            // download this cache file
            let fut =
                download_cache_file_helper(thorium, job, reaction, sub, write_path, local_logs);
            // add this to our download futures
            futs.push(fut);
        }
        // download our cache files in parallel
        stream::iter(futs)
            .buffer_unordered(3)
            .collect::<Vec<Result<(), Error>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<()>, Error>>()?;
    }
    Ok(())
}

/// Download our reactions cache from for Thorium
///
/// # Arguments
///
/// * `thorium` - A thorium client
/// * `image` - The image this worker is executing
/// * `job` - The current job we are executing
/// * `location` - The path to download cache data too
/// * `logs` - This jobs logs
#[instrument(name = "setup::download_cache", skip_all, err(Debug))]
pub async fn download_cache<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    location: P,
    logs: &mut Sender<String>,
) -> Result<DownloadedCache, Error> {
    // start with a default empty cache download object
    let mut downloaded = DownloadedCache::default();
    // don't bother downloading our cache if its not enabled
    if image.dependencies.cache.enabled {
        // get our reaction id or our parents reaction id if we are using our parent cache
        let reaction = image.dependencies.cache.get_reaction_id(job);
        // download our reaction cache
        let cache = thorium.reactions.get_cache(&job.group, reaction).await?;
        // convert our location to a path
        let location = location.as_ref();
        // create our cache folder
        tokio::fs::create_dir_all(&location).await?;
        // write our generic cache to disk
        write_generic_cache(location, &cache, &mut downloaded, logs).await?;
        // download our cache files
        download_cache_files(
            thorium,
            location,
            &cache,
            job,
            reaction,
            &mut downloaded,
            logs,
        )
        .await?;
        // add our full cache to our downloaded cache object
        downloaded.cache = Some(cache);
        // update our cache downloaded at time
        downloaded.downloaded_at = Some(SystemTime::now());
    }
    Ok(downloaded)
}

/// Build the path to write a downloaded sample too
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `base` - The base path to write any downloaded samples too
/// * `sha256` - The sha256 of the sample to download
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::build_sample_path", skip_all, err(Debug))]
pub async fn build_sample_path(
    thorium: &Thorium,
    image: &Image,
    base: &Path,
    sha256: &str,
    logs: &mut Sender<String>,
) -> Result<PathBuf, Error> {
    // get the to use for this file
    match image.dependencies.samples.naming {
        // just use the sha256 for our name
        FileNamingStrategy::Sha256 => {
            // log the sha256 we are downloading
            event!(Level::INFO, sha256 = sha256);
            // add a log to disk that are downloading this file
            log!(logs, "Downloading sample {}", sha256);
            Ok(base.join(sha256))
        }
        FileNamingStrategy::MostRecent => {
            // get this samples submission info
            let info = thorium.files.get(sha256).await?;
            // get the most recent submission with a file name
            let target = match info.submissions.iter().find_map(|sub| sub.name.as_ref()) {
                Some(name_str) => {
                    // cast this file name to a PathBuf
                    let name_path = PathBuf::from(name_str);
                    // if this is a absolute path then just use the file name otherwise append the full relative path
                    if name_path.has_root() {
                        // if this file name is just this root then add our sha256
                        match name_path.file_name() {
                            Some(name) => {
                                // add a log to disk that are downloading this file
                                log!(logs, "Downloading sample {}", name.display());
                                base.join(name)
                            }
                            None => {
                                // log that we only discovered a root
                                log!(
                                    logs,
                                    "{sha256} is just a root! Using the sha256 instead of the {name_str}"
                                );
                                // add our sha256 instead
                                base.join(sha256)
                            }
                        }
                    } else {
                        // add the relative path
                        base.join(name_path)
                    }
                }
                None => {
                    // log that we didn't find any submissions with file names
                    log!(
                        logs,
                        "{sha256} has no submissions with a file name! Using the sha256."
                    );
                    // add a log to disk that are downloading this file
                    log!(logs, "Downloading sample {sha256}");
                    // add our sha256 instead
                    base.join(sha256)
                }
            };
            // log that we are writting this sample to a different name then its sha256
            event!(
                Level::INFO,
                sha256 = sha256,
                target = target.to_string_lossy().to_string()
            );
            Ok(target)
        }
    }
}

/// Downloads any requested samples or ephemeral files from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading samples for
/// * `target` - The target folder to write these samples too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_samples", skip_all, err(Debug))]
pub async fn download_samples<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // build the path to save these samples too
    let target = target.as_ref().to_path_buf();
    // create a list to the paths to our downloaded samples
    let mut samples = Vec::with_capacity(job.samples.len());
    // build the options for downloading this file
    let mut opts = FileDownloadOpts::default().uncart();
    // crawl over any samples and try to download them
    for sha256 in &job.samples {
        // keep track of how many times we have tried to download this sample
        let mut attempts = 0;
        // build the path to download our files too
        let dl_target = build_sample_path(thorium, image, &target, sha256, logs).await?;
        // retry this sample until it works or we have tried 3 times
        loop {
            // download and uncart this file to disk
            let dl_attempt = thorium.files.download(sha256, &dl_target, &mut opts).await;
            // if this download ran into an IO or 500 error then try again
            match dl_attempt {
                // this download worked so continue
                Ok(_) => break,
                // An error occured check if we should retry or fail out this job
                Err(error) => {
                    // increment our attempt count
                    attempts += 1;
                    // if we have made three attempts then fail this job
                    if attempts >= 3 {
                        return Err(error);
                    }
                    // check what kind of error this was
                    match error {
                        Error::IO(error) => {
                            // log that this download failed
                            log!(logs, "Downloading {sha256} failed with {error:?}");
                        }
                        Error::Thorium { code, msg } => {
                            // log that this download failed
                            log!(logs, "Downloading {sha256} failed with {code}: {msg:?}");
                        }
                        // treat all other errors as fatal
                        error => return Err(error),
                    }
                }
            }
            // delete this incorrectly downloaded file
            purge!(dl_target);
        }
        // only pass in downloaded samples if its enabled
        if image.dependencies.samples.strategy != DependencyPassStrategy::Disabled {
            // add this downloaded sample to our list
            samples.push(dl_target);
        }
    }
    Ok(samples)
}

/// Downloads any requested ephemeral files from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading ephemeral files for
/// * `target` - The target folder to write these ephemeral files too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_ephemeral", skip_all, err(Debug))]
pub async fn download_ephemeral<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // build the path to save this repo too
    let mut target = target.as_ref().to_path_buf();
    // create a list to the paths to our downloaded ephemeral files
    let mut ephemerals = Vec::with_capacity(job.ephemeral.len() + job.parent_ephemeral.len());
    // crawl over any ephemeral files and download them
    for name in &job.ephemeral {
        // check if this image restricts what files to download or not
        if !image.dependencies.ephemeral.names.is_empty() {
            // this image restricts what ephemeral files it depends on so check if this file is
            // one of them
            if !image.dependencies.ephemeral.names.contains(name) {
                // this file is not one of the files this image depends on so skip it
                continue;
            }
        }
        // build the target path for this upload
        target.push(name);
        // log the sha256 we are downloading
        event!(Level::INFO, name = name);
        // download this ephemeral file
        log!(logs, "Downloading ephemeral file {}", name);
        let data = thorium
            .reactions
            .download_ephemeral(&job.group, &job.reaction, name)
            .await?;
        // create and write this ephemeral file to disk
        let mut fp = File::create(&target).await?;
        fp.write_all(&data).await?;
        // only pass in downloaded ephemeral files if its enabled
        if image.dependencies.ephemeral.strategy != DependencyPassStrategy::Disabled {
            // track the path to this file so we can delete it later
            ephemerals.push(target.clone());
        }
        // pop this samples hash
        target.pop();
    }
    Ok(ephemerals)
}

/// Downloads any requested ephemeral files for parent reaction from Thorium
///
/// # Arguments
///
/// * `ephemeral` - The paths to the ephemeral files we have already downloaded
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading parent ephemeral files for
/// * `target` - The target folder to write these parent ephemeral files too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_parent_ephemeral", skip_all, err(Debug))]
pub async fn download_parent_ephemeral<P: AsRef<Path>>(
    ephemerals: &mut Vec<PathBuf>,
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // crawl over any ephemeral files and download them
    for (name, parent) in &job.parent_ephemeral {
        // check if this image restricts what files to download or not
        if !image.dependencies.ephemeral.names.is_empty() {
            // this image restricts what ephemeral files it depends on so check if this file is
            // one of them
            if !image.dependencies.ephemeral.names.contains(name) {
                // this file is not one of the files this image depends on so skip it
                continue;
            }
        }
        // build the target path for this upload
        let mut target = target.as_ref().to_path_buf();
        target.push(name);
        // log that we are downloading this parent ephemeral file
        event!(Level::INFO, name = name);
        log!(
            logs,
            "Downloading ephemeral file {} from parent {}",
            name,
            parent
        );
        // download this ephemeral file
        let data = thorium
            .reactions
            .download_ephemeral(&job.group, parent, name)
            .await?;
        // create and write this ephemeral file to disk
        let mut fp = File::create(&target).await?;
        fp.write_all(&data).await?;
        // only pass in downloaded parent ephemeral files if its enabled
        if image.dependencies.ephemeral.strategy != DependencyPassStrategy::Disabled {
            // track the path to this file so we can delete it later
            ephemerals.push(target);
        }
    }
    Ok(())
}

/// Downloads any requested repos from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading repos for
/// * `target` - The target folder to write these repos too
/// * `commits` - The commit that each repo is checked out too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_repos", skip_all, err(Debug))]
pub async fn download_repos<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    commits: &mut HashMap<String, String>,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // build the path to save these repos too
    let target = target.as_ref().to_path_buf();
    // create a list to the paths to our downloaded repos
    let mut repos = Vec::with_capacity(job.repos.len());
    // crawl over any samples and try to download them
    for repo in &job.repos {
        // log that we are downloading this repo
        event!(Level::INFO, repo = repo.url);
        log!(logs, "Downloading repo {}", repo.url);
        // build our download options
        let mut opts = RepoDownloadOpts::default();
        // if we have a commitish then set that
        if let Some(commitish) = &repo.commitish {
            opts.commitish = Some(commitish.clone());
        }
        // set our commitish kind if it exists
        if let Some(kind) = repo.kind {
            opts.kinds.push(kind);
        }
        // download and unpack this repo to disk
        let untarred = thorium
            .repos
            .download_unpack(&repo.url, &opts, &target)
            .await?;
        // get this repos commit
        commits.insert(repo.url.clone(), untarred.commit()?);
        // only pass in downloaded parent ephemeral files if its enabled
        if image.dependencies.repos.strategy != DependencyPassStrategy::Disabled {
            repos.push(untarred.path);
        }
    }
    Ok(repos)
}

/// Downloads any requested tags from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading tags for
/// * `target` - The target folder to write these tags too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_tags", skip_all, err(Debug))]
pub async fn download_tags<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // build the path to save these tags too
    let mut target = target.as_ref().to_path_buf();
    // create a list to the paths to our downloaded tags
    let mut tags = Vec::with_capacity(job.samples.len());
    // crawl over any samples and try to download them
    for sha256 in &job.samples {
        // log the sha256 we are getting tags for
        event!(Level::INFO, sha256 = sha256);
        // build the target path for this download
        target.push(sha256);
        // add the json extension
        target.set_extension("json");
        // get this samples tags and write them to disk
        log!(logs, "Downloading tags from {}", sha256);
        // get this samples info
        let sample = thorium.files.get(sha256).await?;
        // get this samples tags without any group info
        let simple_tags = sample.simple_tags();
        // serialize this samples tags
        let serialized = serde_json::to_string(&simple_tags)?;
        // open the file to write our tags too
        let mut file = File::create(&target).await?;
        // write these tags
        file.write_all(serialized.as_bytes()).await?;
        // only pass in downloaded tags if its enabled
        if image.dependencies.tags.strategy != DependencyPassStrategy::Disabled {
            // add this downloaded tag to our list
            tags.push(target.clone());
        }
        // pop this samples hash
        target.pop();
    }
    // crawl over any repos and try to download them
    for repo in &job.repos {
        // log the sha256 we are gettting tags for
        event!(Level::INFO, repo = &repo.url);
        // convert this url to a path
        let path = PathBuf::from(repo.url.clone());
        // get this repos name
        let name = path.file_name().unwrap().to_str().unwrap();
        // build the target path for this download
        target.push(name);
        // add the json extension
        target.set_extension("json");
        // get this repos tags and write them to disk
        log!(logs, "Downloading tags from {}", repo.url);
        // get this repos info
        let repo = thorium.repos.get(&repo.url).await?;
        // get this repos tags without any group info
        let simple_tags = repo.simple_tags();
        // serialize this repos tags
        let serialized = serde_json::to_string(&simple_tags)?;
        // open the file to write our tags too
        let mut file = File::create(&target).await?;
        // write these tags
        file.write_all(serialized.as_bytes()).await?;
        // only pass in downloaded tags if its enabled
        if image.dependencies.tags.strategy != DependencyPassStrategy::Disabled {
            // add this downloaded tag to our list
            tags.push(target.clone());
        }
        // pop this repos hash
        target.pop();
    }
    Ok(tags)
}

/// A key pointing to an item to download results for
enum ResultKey<'a> {
    /// The key points to a sample with the given SHA256
    Sample { sha256: &'a str },
    /// The key points to a repo with the given URL
    Repo { url: &'a str },
}

/// Downloads any requested results from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading results for
/// * `target` - The target folder to write these results too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_results", skip_all, err(Debug))]
pub async fn download_results<P: Into<PathBuf>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // create a list to the paths to our downloaded results
    let mut downloaded = Vec::with_capacity(image.dependencies.results.images.len());
    // only download results if this tool depends on any
    if !image.dependencies.results.images.is_empty() {
        // build our get result params options to get hidden results too
        let params = ResultGetParams::default()
            .hidden()
            // pull from the specified tools
            .tools(image.dependencies.results.images.clone());
        // build a set of paths we've already created as we go
        let mut created_dirs = HashSet::new();
        // build the root path for all results
        let root = target.into();
        // download results/result-files for samples
        for sha256 in &job.samples {
            let downloaded_path = download_results_helper(
                ResultKey::Sample { sha256 },
                thorium,
                &params,
                &image.dependencies.results.names,
                &root,
                logs,
                &mut created_dirs,
            )
            .await?;
            if image.dependencies.results.strategy != DependencyPassStrategy::Disabled {
                downloaded.push(downloaded_path);
            }
        }
        // download results/result-files for repos
        for repo in &job.repos {
            let downloaded_path = download_results_helper(
                ResultKey::Repo { url: &repo.url },
                thorium,
                &params,
                &image.dependencies.results.names,
                &root,
                logs,
                &mut created_dirs,
            )
            .await?;
            if image.dependencies.results.strategy != DependencyPassStrategy::Disabled {
                downloaded.push(downloaded_path);
            }
        }
    }
    Ok(downloaded)
}

/// Download results for an item at the given key
///
/// # Returns
///
/// Returns the path to the results downloaded for this item
///
/// # Arguments
///
/// * `key` - The key to the item to download results for
/// * `thorium` - The Thorium Client
/// * `params` - The params to use when downloading results
/// * `root` - The root directory all results should be stored in
/// * `logs` - The channel to send logs to
/// * `created_dirs` - The set of directories we've already created while downloading results
async fn download_results_helper(
    key: ResultKey<'_>,
    thorium: &Thorium,
    params: &ResultGetParams,
    file_names: &[String],
    root: &Path,
    logs: &mut Sender<String>,
    created_dirs: &mut HashSet<PathBuf>,
) -> Result<PathBuf, Error> {
    // see if we're getting results for a sample or a repo
    let (key_str, results) = match key {
        ResultKey::Sample { sha256 } => {
            // get results for the sample
            log!(logs, "Downloading results for sample '{}'", sha256);
            (sha256, thorium.files.get_results(sha256, params).await?)
        }
        ResultKey::Repo { url } => {
            // get results for the repo
            log!(logs, "Downloading results for repo '{}'", url);
            (url, thorium.repos.get_results(url, params).await?)
        }
    };
    if !file_names.is_empty() {
        // log that we're going to filter result files
        log!(
            logs,
            "Only downloading result files matching these names: {:?}",
            file_names
        );
    }
    let mut nested = root.join(key_str);
    // crawl over each tools results
    for (tool, mut output) in results.results {
        // build the path for this result blob
        nested.push(&tool);
        if let Some(first_output) = output.first_mut() {
            // create the dir for these results
            tokio::fs::create_dir_all(&nested).await?;
            // serialize this result
            let serialized = serde_json::to_string(&first_output.result)?;
            // build our results path and open a handle to it
            nested.push("results");
            let mut file = File::create(&nested).await?;
            // write this result out
            file.write_all(serialized.as_bytes()).await?;
            // reset our path for result files
            nested.pop();
            nested.push("result-files");
            tokio::fs::create_dir_all(&nested).await?;
            // filter out any result files that aren't in our file name list if we have any
            if !file_names.is_empty() {
                let filtered = first_output
                    .files
                    .extract_if(.., |result_file| !file_names.contains(result_file))
                    .collect::<Vec<String>>();
                if !filtered.is_empty() {
                    // log if we've filtered out any files
                    log!(
                        logs,
                        "Result files from tool '{}' filtered out by image dependency settings: {:?}",
                        tool,
                        filtered
                    );
                }
            }
            for result_file in &first_output.files {
                event!(Level::INFO, key = key_str, result_file = result_file);
                log!(
                    logs,
                    "Downloading results file '{}' from tool '{}'",
                    result_file,
                    tool
                );
                // see if we're getting result files for a sample or a repo
                let attachment = match key {
                    ResultKey::Sample { sha256 } => {
                        thorium
                            .files
                            .download_result_file(sha256, &tool, &first_output.id, result_file)
                            .await?
                    }
                    ResultKey::Repo { url } => {
                        thorium
                            .repos
                            .download_result_file(url, &tool, &first_output.id, result_file)
                            .await?
                    }
                };
                // build the path to write this result file off to disk at
                let target_path = nested.join(result_file);
                // create any needed parent dirs for this result file
                create_parents(&target_path, created_dirs).await?;
                // create a file handle for this file
                let mut file = tokio::fs::File::create(&target_path).await?;
                // write our response body to disk
                file.write_all(&attachment.data[..]).await?;
            }
            // pop the result-files directory
            nested.pop();
        }
        // pop the tool directory
        nested.pop();
    }
    // return the path to the results for this item
    Ok(nested)
}

/// Downloads any requested children from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading children for
/// * `target` - The target folder to write these children too
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_children", skip_all, err(Debug))]
pub async fn download_children<P: Into<PathBuf>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // create a list to the paths to our downloaded children
    let mut downloaded = Vec::with_capacity(image.dependencies.children.images.len());
    // build the params for getting results
    let result_params = ResultGetParams::default()
        // if we have limited the images to get children for then add those
        .tools(&image.dependencies.children.images);
    // build the path to save these samples too
    let mut target = target.into();
    // build the options for downloading this file
    let mut opts = FileDownloadOpts::default().uncart();
    // download children for all samples we depended on
    for sha256 in &job.samples {
        // get the results for this sample
        let results = thorium.files.get_results(sha256, &result_params).await?;
        // add this samples sha256
        target.push(sha256);
        // step over our results and build a list of children
        for (tool, outputs) in results.results {
            // get the children from the last result
            if let Some(output) = outputs.first() {
                // check if this results has any children
                if !output.children.is_empty() {
                    // log that sha256 we are downloading results
                    log!(logs, "Downloading children from {} - {}", sha256, tool);
                    // add this tools name
                    target.push(tool);
                    // create all of our parent dirs
                    tokio::fs::create_dir_all(&target).await?;
                    // download these children
                    for child in output.children.keys() {
                        // add this childs sha256
                        target.push(child);
                        log!(logs, "Downloading child: {}", child);
                        // download this child
                        thorium.files.download(child, &target, &mut opts).await?;
                        // add this path to our downloaded children
                        downloaded.push(target.clone());
                        // remove our childs sha256
                        target.pop();
                    }
                    // remove our tool name
                    target.pop();
                }
            }
        }
        // remove our samples sha256
        target.pop();
    }
    // download children for all repos we depended on
    for repo_dep in &job.repos {
        // get our repo name
        if let Some(repo) = repo_dep.url.split('/').next_back() {
            // get the results for this repo
            let results = thorium.repos.get_results(repo, &result_params).await?;
            // add this repos name
            target.push(repo);
            // step over our results and build a list of children
            for (tool, outputs) in results.results {
                // get the children from the last result
                if let Some(output) = outputs.first() {
                    // check if this results has any children
                    if !output.children.is_empty() {
                        // log that repo and tool we are downloading results from
                        log!(logs, "Downloading children from {} - {}", repo, tool);
                        // add this tools name
                        target.push(tool);
                        // create all of our parent dirs
                        tokio::fs::create_dir_all(&target).await?;
                        // download these children
                        for child in output.children.keys() {
                            // add this childs sha256
                            target.push(child);
                            log!(logs, "Downloading child: {}", child);
                            // download this child
                            thorium.files.download(child, &target, &mut opts).await?;
                            // add this path to our downloaded children
                            downloaded.push(target.clone());
                            // remove our childs sha256
                            target.pop();
                        }
                        // remove our tool name
                        target.pop();
                    }
                }
            }
            // remove our repos name
            target.pop();
        }
    }
    Ok(downloaded)
}

/// Reconstruct any prior filesystems this image depends on from Thorium
///
/// Unlike [`download_children`], which downloads loose children flat, this rebuilds the directory
/// structure of filesystems dumped by earlier tools so later tools see the original layout. The
/// structure is walked via the trees API and each files original name is recovered from the
/// submission its `FileIn` association was linked with.
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are reconstructing filesystems for
/// * `target` - The target folder to reconstruct these filesystems under
/// * `logs` - The channel to use when sending logs to Thorium
#[instrument(name = "setup::download_filesystems", skip_all, err(Debug))]
pub async fn download_filesystems<P: Into<PathBuf>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // get our filesystem dependency settings
    let settings = &image.dependencies.filesystems;
    // build the root path to reconstruct filesystems under
    let root = target.into();
    // create a list of the paths to our reconstructed files
    let mut downloaded = Vec::new();
    // build the options for downloading files
    let mut opts = FileDownloadOpts::default().uncart();
    // track the dirs we have already created while reconstructing
    let mut created_dirs = HashSet::new();
    // reconstruct filesystems for each sample we depend on
    for sha256 in &job.samples {
        reconstruct_sample_filesystems(
            thorium,
            settings,
            sha256,
            &root,
            &mut opts,
            &mut created_dirs,
            &mut downloaded,
            logs,
        )
        .await?;
    }
    Ok(downloaded)
}

/// Find and reconstruct all filesystems that were dumped for a single sample
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `settings` - The filesystem dependency settings for our image
/// * `sha256` - The sha256 of the sample to reconstruct filesystems for
/// * `root` - The root path to reconstruct filesystems under
/// * `opts` - The options to use when downloading files
/// * `created_dirs` - The set of directories we've already created
/// * `downloaded` - The paths to our reconstructed files
/// * `logs` - The channel to use when sending logs to Thorium
#[expect(clippy::too_many_arguments)]
async fn reconstruct_sample_filesystems(
    thorium: &Thorium,
    settings: &FileSystemDependencySettings,
    sha256: &str,
    root: &Path,
    opts: &mut FileDownloadOpts,
    created_dirs: &mut HashSet<PathBuf>,
    downloaded: &mut Vec<PathBuf>,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // list all filesystem entities that were dumped for this sample
    let list_opts = EntityListOpts::default()
        .tag("Parent", sha256)
        .kinds([EntityKinds::FileSystem]);
    // get a cursor over these filesystem entities
    let mut cursor = thorium.entities.list_details(&list_opts).await?;
    // crawl over all of the filesystem entities for this sample
    loop {
        // reconstruct each filesystem entity on this page
        for entity in &cursor.data {
            // only handle filesystem entities
            if let EntityMetadata::FileSystem(filesystem) = &entity.metadata {
                // skip filesystems that weren't dumped by one of our restricted images if we have any
                if !settings.images.is_empty()
                    && !filesystem
                        .tools
                        .iter()
                        .any(|tool| settings.images.contains(tool))
                {
                    continue;
                }
                // log that we are reconstructing this filesystem
                log!(
                    logs,
                    "Reconstructing filesystem {} for {}",
                    entity.name,
                    sha256
                );
                // reconstruct this filesystem on disk
                reconstruct_filesystem(
                    thorium,
                    settings,
                    sha256,
                    entity.id,
                    &entity.name,
                    root,
                    opts,
                    created_dirs,
                    downloaded,
                    logs,
                )
                .await?;
            }
        }
        // stop crawling once our cursor is exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of filesystem entities
        cursor.refill().await?;
    }
    Ok(())
}

/// Reconstruct a single filesystem on disk by walking its tree
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `settings` - The filesystem dependency settings for our image
/// * `sha256` - The sha256 of the sample this filesystem was dumped for
/// * `fs_id` - The id of the root filesystem entity to reconstruct
/// * `fs_name` - The name of the filesystem being reconstructed
/// * `root` - The root path to reconstruct filesystems under
/// * `opts` - The options to use when downloading files
/// * `created_dirs` - The set of directories we've already created
/// * `downloaded` - The paths to our reconstructed files
/// * `logs` - The channel to use when sending logs to Thorium
#[expect(clippy::too_many_arguments)]
async fn reconstruct_filesystem(
    thorium: &Thorium,
    settings: &FileSystemDependencySettings,
    sha256: &str,
    fs_id: Uuid,
    fs_name: &str,
    root: &Path,
    opts: &mut FileDownloadOpts,
    created_dirs: &mut HashSet<PathBuf>,
    downloaded: &mut Vec<PathBuf>,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // build the tree options limiting how deep we grow and skipping parents
    let tree_opts = TreeOpts::default().gather_parents(false).limit(1024);
    // build a query starting from this filesystem entity bounded to just this filesystem so its
    // folders and files auto expand instead of being hinted
    let mut query = TreeQuery::default().entity(fs_id);
    query.bounds.filesystem = vec![fs_id];
    // materialize the whole filesystem tree in one shot
    let tree = thorium.trees.start(&tree_opts, &query).await?;
    // warn if the filesystem was too deep to fully materialize so we don't silently truncate
    if !tree.growable.is_empty() {
        log!(
            logs,
            "Filesystem {} was too deep to fully reconstruct - some files may be missing",
            fs_name
        );
    }
    // find the root node for this filesystem in our tree
    let fs_root_hash = tree.data_map.iter().find_map(|(hash, node)| match node {
        TreeNode::Entity(entity)
            if entity.id == fs_id && matches!(entity.metadata, EntityMetadata::FileSystem(_)) =>
        {
            Some(*hash)
        }
        _ => None,
    });
    // get our root node hash or give up on this filesystem if its missing from the tree
    let fs_root_hash = match fs_root_hash {
        Some(hash) => hash,
        None => {
            log!(
                logs,
                "Could not find the root of filesystem {} in its tree",
                fs_name
            );
            return Ok(());
        }
    };
    // build the base path all files in this filesystem will be reconstructed under
    let base = root.join(sha256).join(fs_name);
    // track the reconstructed path for each folder node as we walk down the tree
    let mut folder_paths: HashMap<u64, PathBuf> = HashMap::new();
    // the filesystem root maps to our base path
    folder_paths.insert(fs_root_hash, base);
    // walk the filesystem tree breadth first from its root
    let mut queue = VecDeque::new();
    queue.push_back(fs_root_hash);
    while let Some(node_hash) = queue.pop_front() {
        // get the reconstructed path for this folder node
        let parent_path = match folder_paths.get(&node_hash) {
            Some(path) => path.clone(),
            None => continue,
        };
        // get the branches leaving this node if it has any
        let branches = match tree.branches.get(&node_hash) {
            Some(branches) => branches,
            None => continue,
        };
        // step over every branch leaving this node
        for branch in branches {
            // only follow branches to our children (folders/files in this folder)
            if branch.direction != Directionality::To {
                continue;
            }
            // only follow filesystem association branches
            let association = match &branch.relationship {
                TreeRelationships::Association(association) => association,
                _ => continue,
            };
            // handle this branch based on the kind of thing it points too
            match association.kind {
                // this branch points to a child folder in this folder
                AssociationKind::FolderIn => {
                    // get the child folder node
                    if let Some(TreeNode::Entity(child)) = tree.data_map.get(&branch.node) {
                        // only descend into folders that belong to this filesystem
                        if let EntityMetadata::Folder(folder) = &child.metadata {
                            // skip folders from other filesystems
                            if folder.filesystem_id != fs_id {
                                continue;
                            }
                            // the special "/" root folder maps to our current path, others get appended
                            let child_path = if child.name == "/" {
                                parent_path.clone()
                            } else {
                                parent_path.join(&child.name)
                            };
                            // record this folders path and queue it for walking
                            folder_paths.insert(branch.node, child_path);
                            queue.push_back(branch.node);
                        }
                    }
                }
                // this branch points to a file in this folder
                AssociationKind::FileIn => {
                    // get the sha256 of the file to download
                    let file_sha256 = match &association.other {
                        AssociationTarget::File(file_sha256) => file_sha256,
                        // this file in association doesn't point at a file so skip it
                        _ => continue,
                    };
                    // resolve this files original name from the submission it was linked with,
                    // falling back to its sha256 for older filesystems that lack a submission link
                    let name = resolve_file_name(association, branch.node, &tree.data_map)
                        .unwrap_or_else(|| file_sha256.clone());
                    // build the path to reconstruct this file at
                    let file_path = parent_path.join(name);
                    // create any parent dirs for this file
                    create_parents(&file_path, created_dirs).await?;
                    // log that we are reconstructing this file
                    log!(
                        logs,
                        "Reconstructing {} at {}",
                        file_sha256,
                        file_path.display()
                    );
                    // download this file to its reconstructed path
                    thorium.files.download(file_sha256, &file_path, opts).await?;
                    // only pass in reconstructed files if its enabled
                    if settings.strategy != DependencyPassStrategy::Disabled {
                        downloaded.push(file_path);
                    }
                }
                // ignore any other association kinds
                _ => {}
            }
        }
    }
    Ok(())
}

/// Resolve a files original name from the submission its association was linked with
///
/// Falls back to any submission that has a name so reconstruction still works for older filesystems
/// that predate submission linked associations.
///
/// # Arguments
///
/// * `association` - The `FileIn` association linking this file to its folder
/// * `file_hash` - The tree node hash of the file
/// * `data_map` - The map of node hashes to their tree nodes
fn resolve_file_name(
    association: &Association,
    file_hash: u64,
    data_map: &HashMap<u64, TreeNode>,
) -> Option<String> {
    // get the sample node for this file
    let sample = match data_map.get(&file_hash) {
        Some(TreeNode::Sample(sample)) => sample,
        // this file isn't a sample node so we can't resolve its name
        _ => return None,
    };
    // get the submission this file was linked with if we know it
    let linked = association.submissions.as_ref().and_then(|subs| subs.other);
    // try to find the exact submission this file was placed under
    if let Some(submission_id) = linked {
        // look for the linked submission and use its name if it has one
        if let Some(name) = sample
            .submissions
            .iter()
            .find(|sub| sub.id == submission_id)
            .and_then(|sub| sub.name.clone())
        {
            return Some(name);
        }
    }
    // fall back to any submission that has a name
    sample.submissions.iter().find_map(|sub| sub.name.clone())
}
