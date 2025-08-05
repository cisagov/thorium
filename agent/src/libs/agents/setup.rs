//! Set up an environment for executing a Thorium job

use crossbeam::channel::Sender;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thorium::client::ResultsClient;
use thorium::models::{
    DependencyPassStrategy, FileDownloadOpts, GenericJob, Image, RepoDownloadOpts, ResultGetParams,
};
use thorium::Error;
use thorium::Thorium;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::{event, instrument, Level};

use crate::log;

/// Create any required parent dirs for this file
///
/// # Arguments
///
/// * `path` - The path to check the parent dirs for
/// * `created_dirs` - The set of directories we have already created
async fn create_parents(path: &PathBuf, created_dirs: &mut HashSet<PathBuf>) -> Result<(), Error> {
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

/// Downloads any requested samples or ephemeral files from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for Thorium
/// * `image` - The image our job is based on
/// * `job` - The job we are downloading samples for
/// * `target` - The target folder to write these samples too
/// * `logs` - The channel to use when sending logs to Thorium
/// * `span` - The span to log traces under
#[instrument(name = "setup::download_samples", skip_all, err(Debug))]
pub async fn download_samples<P: AsRef<Path>>(
    thorium: &Thorium,
    image: &Image,
    job: &GenericJob,
    target: P,
    logs: &mut Sender<String>,
) -> Result<Vec<PathBuf>, Error> {
    // build the path to save these samples too
    let mut target = target.as_ref().to_path_buf();
    // create a list to the paths to our downloaded samples
    let mut samples = Vec::with_capacity(job.samples.len());
    // build the options for downloading this file
    let mut opts = FileDownloadOpts::default().uncart();
    // crawl over any samples and try to download them
    for sha256 in job.samples.iter() {
        // log the sha256 we are downloading
        event!(Level::INFO, sha256 = sha256);
        // build the target path for this upload
        target.push(sha256);
        // download and uncart this file to disk
        log!(logs, "Downloading sample {}", sha256);
        thorium.files.download(sha256, &target, &mut opts).await?;
        // only pass in downloaded samples if its enabled
        if image.dependencies.samples.strategy != DependencyPassStrategy::Disabled {
            // add this downloaded sample to our list
            samples.push(target.clone())
        }
        // pop this samples hash
        target.pop();
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
    for name in job.ephemeral.iter() {
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
    for (name, parent) in job.parent_ephemeral.iter() {
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
    for repo in job.repos.iter() {
        // log that we are downloading this repo
        event!(Level::INFO, repo = repo.url);
        log!(logs, "Downloading repo {}", repo.url);
        // build our download options
        let mut opts = RepoDownloadOpts::default();
        // if we have a commitish then set that
        if let Some(commitish) = &repo.commitish {
            opts.commitish = Some(commitish.to_string());
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
    for sha256 in job.samples.iter() {
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
            tags.push(target.clone())
        }
        // pop this samples hash
        target.pop();
    }
    // crawl over any repos and try to download them
    for repo in job.repos.iter() {
        // log the sha256 we are getting tags for
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
            tags.push(target.clone())
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
/// * `created_dirs` - The set of directories we've already created
///                    while downloading results
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
                    for (child, _) in &output.children {
                        // add this child's sha256
                        target.push(child);
                        log!(logs, "Downloading child: {}", child);
                        // download this child
                        thorium.files.download(&child, &target, &mut opts).await?;
                        // add this path to our downloaded children
                        downloaded.push(target.clone());
                        // remove our child's sha256
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
        if let Some(repo) = repo_dep.url.split('/').last() {
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
                        for (child, _) in &output.children {
                            // add this child's sha256
                            target.push(child);
                            log!(logs, "Downloading child: {}", child);
                            // download this child
                            thorium.files.download(&child, &target, &mut opts).await?;
                            // add this path to our downloaded children
                            downloaded.push(target.clone());
                            // remove our child's sha256
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
