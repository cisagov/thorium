//! Handles collecting results for the agent and sending them back to the API

use crossbeam::channel::Sender;
use std::collections::HashMap;
use std::path::Path;
use thorium::client::ResultsClient;
use thorium::models::{
    GenericJob, Image, OnDiskFile, OutputDisplayType, OutputRequest, Repo, Sample,
};
use thorium::{Error, Thorium};
use tracing::instrument;
use uuid::Uuid;
use walkdir::WalkDir;

use super::helpers;
use crate::log;

/// Whether this result should be uploaded to the db or to s3
#[derive(Debug, Clone)]
pub enum ResultTarget {
    /// This result should go to the db
    Db(String),
    /// The result should go to  s3
    S3 { results: String, len: u64 },
}

impl ResultTarget {
    /// Build the correct result for an output request
    pub fn to_output_request(&self, logs: &mut Sender<String>) -> Result<String, Error> {
        // if  this is going to s3 then add it to our files and set a warning
        match self {
            ResultTarget::Db(result) => Ok(result.clone()),
            ResultTarget::S3 { len, .. } => {
                // log that our results file is over 1 MB
                log!(logs, "Warning: Results file exists but is {}B", len);
                // create an output with the warning that the result was too large to display
                let mut output = HashMap::with_capacity(1);
                output.insert(
                    "Warnings",
                    vec!["result stored as result file since it was bigger then 1 MB"],
                );
                // serialize our results
                let new_results = serde_json::to_string(&output)?;
                Ok(new_results)
            }
        }
    }

    pub fn get_results(&self) -> &String {
        match self {
            ResultTarget::Db(results) => results,
            ResultTarget::S3 { results, .. } => results,
        }
    }
}

/// A raw output request that later gets duplicated for each possible
/// input (samples, repos ...)
#[allow(clippy::module_name_repetitions)]
pub struct RawResults {
    /// Whether this results should be scanned for tags or not
    pub scan: bool,
    /// The serialized output for this result
    pub results: ResultTarget,
    /// Any files tied to this result
    pub files: Vec<OnDiskFile>,
    /// The display type of this result
    pub display_type: OutputDisplayType,
}

impl RawResults {
    /// Create a sample output request for these raw results
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the sample we are uploading results for
    /// * `image` - The image we are uploading results for
    pub fn to_sample_req(
        &self,
        sha256: &str,
        image: &Image,
        logs: &mut Sender<String>,
    ) -> Result<OutputRequest<Sample>, Error> {
        // convert our results
        let results = self.results.to_output_request(logs)?;
        // build our output request
        let req = OutputRequest::<Sample>::new(
            sha256.to_owned(),
            image.name.clone(),
            results,
            self.display_type,
        )
        .files(self.files.clone());
        Ok(req)
    }

    /// Create a repo output request for these raw results
    ///
    /// # Arguments
    ///
    /// * `repo` - The url of the repo we are uploading results for
    /// * `image` - The image we are uploading results for
    pub fn to_repo_req(
        &self,
        repo: &str,
        image: &Image,
        logs: &mut Sender<String>,
    ) -> Result<OutputRequest<Repo>, Error> {
        // convert our results
        let results = self.results.to_output_request(logs)?;
        // build our output request
        let req = OutputRequest::<Repo>::new(
            repo.to_owned(),
            image.name.clone(),
            results,
            self.display_type,
        )
        .files(self.files.clone());
        Ok(req)
    }
}

/// Checks the filesystem for results to send to Thorium
///
/// # Arguments
///
/// * `job` - The job we are collecting results from
/// * `image` - The image we are collecting results in
/// * `path` - The path to collect results at
/// * `logs` - The logs to send to the API
#[instrument(name = "results::collect_file", skip_all, fields(path = path.to_string_lossy().into_owned()), err(Debug))]
async fn collect_file(
    image: &Image,
    path: &Path,
    logs: &mut Sender<String>,
) -> Result<RawResults, Error> {
    // check to see if this path exists
    if path.exists() {
        // check the size of this file to determine if it should be a result file
        let metadata = path.metadata()?;
        // only try to ingest results if this is a file
        if metadata.is_file() {
            // check if our results file length is too large or empty
            let raw_result = match metadata.len() {
                // results is empty so don't bother uploading it
                len if len == 0 => {
                    // log that our results file is empty
                    log!(logs, "Warning: Results file exists but is empty");
                    // create an output with the warning that the result was empty
                    let mut output = HashMap::with_capacity(1);
                    output.insert("Warnings", vec!["Results file exists but is empty"]);
                    // serialize our results
                    let results = serde_json::to_string(&output)?;
                    // build our raw results
                    RawResults {
                        scan: false,
                        results: ResultTarget::Db(results),
                        files: Vec::default(),
                        display_type: OutputDisplayType::Json,
                    }
                }
                // results is too large to be stored in the DB
                len if len > 1_000_000 => {
                    // read in our results
                    let results = tokio::fs::read_to_string(path).await?;
                    // build our result file to store
                    let file = OnDiskFile::new(path)
                        .trim_prefix(path.parent().unwrap_or_else(|| Path::new("/")));
                    // build our raw results
                    RawResults {
                        scan: true,
                        results: ResultTarget::S3 { results, len },
                        files: vec![file],
                        display_type: OutputDisplayType::Json,
                    }
                }
                // the result is the correct size to be stored in the DB
                _ => {
                    // read in our results
                    let results = tokio::fs::read_to_string(path).await?;
                    // build our raw results
                    RawResults {
                        scan: image.display_type == OutputDisplayType::Json,
                        results: ResultTarget::Db(results),
                        files: Vec::default(),
                        display_type: image.display_type,
                    }
                }
            };
            Ok(raw_result)
        } else {
            // log that our results file is over 1 MB
            log!(logs, "Warning: Results file is not a file");
            // create an output with the warning that the result was not a file
            let mut output = HashMap::with_capacity(1);
            output.insert("Warnings", vec!["Results file is not a file"]);
            // serialize our results
            let results = serde_json::to_string(&output)?;
            // build our raw results
            let raw_result = RawResults {
                scan: false,
                results: ResultTarget::Db(results),
                files: Vec::default(),
                display_type: OutputDisplayType::Json,
            };
            Ok(raw_result)
        }
    } else {
        // log that our results file is over 1 MB
        log!(logs, "Warning: No results file found");
        // create an output with the warning that the result was not found
        let mut output = HashMap::with_capacity(1);
        output.insert("Warnings", vec!["No non file results found"]);
        // serialize our results
        let results = serde_json::to_string(&output)?;
        // build our raw results
        let raw_result = RawResults {
            scan: false,
            results: ResultTarget::Db(results),
            files: Vec::default(),
            display_type: OutputDisplayType::Json,
        };
        Ok(raw_result)
    }
}

/// Checks the filesystem for result files to send to Thorium
///
/// This looks for result files not results to store in s3.
///
/// # Arguments
///
/// * `path` - The path to collect result files from
/// * `outputs` - The output requests to add our result files too
/// * `logs` - The logs to send to the API
#[instrument(name = "results::collect_result_files", skip_all, fields(path = path.to_string_lossy().into_owned()), err(Debug))]
fn collect_result_files(
    path: &Path,
    mut raw: RawResults,
    logs: &mut Sender<String>,
) -> Result<RawResults, Error> {
    // check to see if this path exists
    if path.exists() {
        // check the size of this file to determine if it should be a result file
        let metadata = path.metadata()?;
        // only try to ingest results files if this path is a directory
        if metadata.is_dir() {
            // recursively walk through this directory skipping any hidden files
            let files = WalkDir::new(path)
                .follow_links(true)
                .into_iter()
                .filter_entry(|entry| !helpers::is_hidden(entry))
                .filter_map(Result::ok)
                .filter(helpers::is_file)
                .map(|entry| OnDiskFile::new(entry.into_path()).trim_prefix(path))
                .collect::<Vec<OnDiskFile>>();
            // log all results files that were found
            for path in &files {
                log!(logs, "Found result file {}", path.path.to_string_lossy());
            }
            // add this to our result file paths
            raw.files.extend(files);
        }
    }
    Ok(raw)
}

/// Collects any results from executing a job
///
/// # Arguments
///
/// * `job` - The job we are collecting results from
/// * `image` - The image to collect result and result files in
/// * `results` - The path to look for results at
/// * `results_files` - The path to look for result files at
/// * `logs` - The logs to send to the API
#[instrument(
    name = "results::collect",
    skip_all,
    fields(
        results = results.as_ref().to_string_lossy().into_owned(),
        result_files = result_files.as_ref().to_string_lossy().into_owned()
    ),
    err(Debug))]
pub async fn collect<P: AsRef<Path>>(
    image: &Image,
    results: P,
    result_files: P,
    logs: &mut Sender<String>,
) -> Result<RawResults, Error> {
    // call the correct output collector
    let outputs = collect_file(image, results.as_ref(), logs).await?;
    // we have results so collect any result files
    collect_result_files(result_files.as_ref(), outputs, logs)
}

///  Send any collected results to Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `outputs` - The results to submit to Thorium
#[instrument(name = "results::submit", skip_all, err(Debug))]
pub async fn submit(
    thorium: &Thorium,
    raw: &RawResults,
    job: &GenericJob,
    image: &Image,
    logs: &mut Sender<String>,
) -> Result<Vec<Uuid>, Error> {
    // track the results we create
    let mut ids = Vec::with_capacity(job.samples.len() + job.repos.len());
    // send our results for samples
    for sha256 in &job.samples {
        // build an output request for this samples
        let req = raw.to_sample_req(sha256, image, logs)?;
        // send this request to the API
        let id = thorium.files.create_result(req).await?;
        // add this new result ID to our list
        ids.push(id.id);
    }
    // send our results for repos
    for repo in &job.repos {
        // build an output request for this repos
        let req = raw.to_repo_req(&repo.url, image, logs)?;
        // send this request to the API
        let id = thorium.repos.create_result(req).await?;
        // add this new result ID to our list
        ids.push(id.id);
    }
    Ok(ids)
}
