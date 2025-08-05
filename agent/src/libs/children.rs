//! collects children files from jobs and submits them to Thorium

use crossbeam::channel::Sender;
use futures::stream::{self, StreamExt};
use itertools::Itertools;
use regex::Regex;
use reqwest::StatusCode;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use thorium::models::{
    CarvedOrigin, ChildFilters, GenericJob, Image, OriginRequest, PcapNetworkProtocol,
    RepoDependency, SampleRequest,
};
use thorium::{Error, Thorium};
use tracing::{event, instrument, Level};
use uuid::Uuid;
use walkdir::WalkDir;

use super::helpers;
use crate::log;

/// A cache of compiled child filter regular expressions mapped to their raw
/// String representation
// TODO: only one agent is running at a time, so we don't really need a Mutex here;
// unfortunately, we can't make this thread-local because the Agent is run on a tokio
// task and can be on any thread
static CHILD_FILTERS_CACHE: LazyLock<Mutex<HashMap<String, Regex>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Create all directories to look for children in
///
/// # Arguments
///
/// * `root` - The root path to look for children files at
/// * `span` - The span to log traces under
#[instrument(name = "children::setup", skip_all, err(Debug))]
pub async fn setup<P: AsRef<Path>>(root: P) -> Result<(), Error> {
    // make the root children dir
    tokio::fs::create_dir_all(&root).await?;
    // make all sub children dir
    tokio::fs::create_dir_all(&root.as_ref().join("source")).await?;
    tokio::fs::create_dir_all(&root.as_ref().join("unpacked")).await?;
    let carved = root.as_ref().join("carved");
    tokio::fs::create_dir_all(&carved.join("unknown")).await?;
    tokio::fs::create_dir_all(&carved.join("pcap")).await?;
    Ok(())
}

/// Recursively walk a directory and get all non hidden files
///
/// # Arguments
///
/// * `path` - The path to the directory to start looking for child files at
pub fn get_children(path: &PathBuf) -> Vec<PathBuf> {
    // only crawl this dir if it exists
    if path.exists() {
        // walk this dir and get all children
        WalkDir::new(&path)
            .follow_links(true)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| !helpers::is_hidden(entry))
            .filter(helpers::is_file)
            .map(|entry| entry.path().to_path_buf())
            .collect::<Vec<PathBuf>>()
    } else {
        // this path doesn't exist so just return an empty vec
        Vec::default()
    }
}

/// Determine if a file is a supporting build file or not
///
/// This is very naive and primarily uses the extension to determine it.
///
/// # Arguments
///
/// * `child` - The path to the child file to check
fn is_supporting(child: &Path) -> bool {
    if let Some(ext) = child.extension().and_then(|ext| ext.to_str()) {
        match ext {
            // this extension is .so, so it is a supporting build file
            "so" => true,
            // this extension doesn't match any of our supporting build file
            // extensions, so it is not a support build file
            _ => false,
        }
    } else {
        // this file has no extension or is not UTF-8, so it is not a supporting build file
        false
    }
}

/// Check if the haystack matches any of the given filters
///
/// Returns false if no filters are given
///
/// # Arguments
///
/// * `haystack` - The haystack we're trying to match to the set of filters
/// * `filters` - The set of filters we're matching on
/// * `filters_cache` - A cache of compiled regexes from our filters
#[instrument(
    name = "children::matches_any_filter",
    skip(filters, filters_cache),
    err(Debug)
)]
fn matches_any_filter(
    haystack: &str,
    filters: &HashSet<String>,
    filters_cache: &mut HashMap<String, Regex>,
) -> Result<bool, Error> {
    for raw_filter in filters {
        let compiled_filter = filters_cache
            // get a compiled regex from the cache or
            // compile a regex and add it to the cache
            .entry(raw_filter.clone())
            // error out if the regex is invalid
            .or_insert(Regex::new(raw_filter)?);
        if compiled_filter.is_match(haystack) {
            // we have a match, so short-circuit and return true
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if any of the child filters match the given child
///
/// # Arguments
///
/// * `child` - The child we're matching
/// * `child_filters` - The child filters we're matching on
/// * `filters_cache` - A cache of compiled regexes from our filters
/// * `logs` - The logs to send to the API
#[instrument(
    name = "children::child_matches_any",
    skip(filters, filters_cache, logs),
    err(Debug)
)]
fn child_matches_any(
    child: &Path,
    filters: &ChildFilters,
    filters_cache: &mut HashMap<String, Regex>,
    logs: &mut Sender<String>,
) -> Result<bool, Error> {
    let mime = if filters.mime.is_empty() {
        None
    } else {
        match infer::get_from_path(child) {
            Ok(maybe_mime) => {
                if let Some(mime) = maybe_mime {
                    if matches_any_filter(mime.mime_type(), &filters.mime, filters_cache)? {
                        log!(
                            logs,
                            format!(
                                "Child '{}' has MIME type '{}' that matches a filter!",
                                child.to_string_lossy(),
                                mime.mime_type()
                            )
                        );
                        return Ok(true);
                    }
                    Some(mime)
                } else {
                    None
                }
            }
            Err(err) => {
                // we got some MIME error, so log, but try matching on the other filters
                log!(
                    logs,
                    format!(
                        "Unable to get MIME type for child '{}': {}",
                        child.to_string_lossy(),
                        err
                    )
                );
                None
            }
        }
    };
    // get file name and convert to UTF-8; otherwise, we can't match
    if let Some(file_name) = child.file_name().and_then(|name| name.to_str()) {
        if matches_any_filter(file_name, &filters.file_name, filters_cache)? {
            log!(
                logs,
                format!(
                    "Child '{}' has file name that matches a filter!",
                    child.to_string_lossy(),
                )
            );
            return Ok(true);
        }
    }
    // get file extension and convert to UTF-8; otherwise, we can't match
    if let Some(file_extension) = child.extension().and_then(|ext| ext.to_str()) {
        if matches_any_filter(file_extension, &filters.file_extension, filters_cache)? {
            log!(
                logs,
                format!(
                    "Child '{}' has file extension that matches a filter!",
                    child.to_string_lossy(),
                )
            );
            return Ok(true);
        }
    }
    // none of the filters matched
    if let Some(mime) = mime {
        log!(
            logs,
            format!(
                "Child '{}' with MIME type '{}' did not match any filters!",
                mime.mime_type(),
                child.to_string_lossy(),
            )
        );
    } else {
        log!(
            logs,
            format!(
                "Child '{}' did not match any filters!",
                child.to_string_lossy(),
            )
        );
    }
    Ok(false)
}

/// Get the parents for all input samples and repos
async fn get_parent_groups(thorium: &Thorium, job: &GenericJob) -> Result<Vec<String>, Error> {
    // build a set of groups for this job
    let mut groups = HashSet::with_capacity(3);
    // get the info on all the samples we depend on
    for sha256 in &job.samples {
        // get info on this sample
        let sample = thorium.files.get(&sha256).await?;
        // add this samples groups to our hashset
        groups.extend(sample.groups().into_iter().map(|name| name.to_owned()));
    }
    // get the info on all the repos we depend on
    for repo_target in &job.repos {
        // get info on this repo
        let repo = thorium.repos.get(&repo_target.url).await?;
        // add this repos groups to our hashset
        groups.extend(repo.groups().into_iter().map(|name| name.to_owned()));
    }
    // convert our hash set to a vec
    let group_vec = groups.into_iter().collect();
    Ok(group_vec)
}

/// Children filtered by a set of filters configured in the tool
struct FilteredChildren {
    /// The list of children that matched at least one of the filters
    matches: Vec<PathBuf>,
    /// The list of children that didn't match any of the filters
    non_matches: Vec<PathBuf>,
}

impl FilteredChildren {
    /// Filter a list of children into two lists – ones that match, and ones that
    /// don't match – and save the lists to a new `FilteredChildren`
    ///
    /// # Arguments
    ///
    /// * `children` - The children to filter
    /// * `filters` - The filters to use
    /// * `filters_cache` - A cache of compiled regexes from our filters
    /// * `logs` - The logs to send to the API
    #[instrument(name = "FilteredChildren::filter", skip_all, err(Debug))]
    fn filter(
        children: &mut Vec<PathBuf>,
        filters: &ChildFilters,
        filters_cache: &mut HashMap<String, Regex>,
        logs: &mut Sender<String>,
    ) -> Result<Self, Error> {
        // save a list of children that matched/didn't match
        let mut matches = Vec::new();
        let mut non_matches = Vec::new();
        // clear out the children from the given Vec into a temporary vec to match on
        let mut temp = std::mem::take(children);
        for child in temp.drain(..) {
            // check if the child matched any of our filters
            if child_matches_any(&child, filters, filters_cache, logs)? {
                // at least one filter matched, so add it to the matched list
                matches.push(child);
            } else {
                // no filters matched, so add it to the non_matched list
                non_matches.push(child);
            }
        }
        Ok(Self {
            matches,
            non_matches,
        })
    }
}

/// Children carved from a sample
#[derive(Default)]
struct CarvedChildren {
    /// Child files carved from a packet capture
    pub pcap: Vec<PathBuf>,
    /// Child files carved from an unknown or unspecified file type
    pub unknown: Vec<PathBuf>,
}

impl CarvedChildren {
    /// Returns true if there are no carved children
    fn is_empty(&self) -> bool {
        self.unknown.is_empty() && self.pcap.is_empty()
    }
}

/// The different types of children files to submit to Thorium
pub struct Children {
    /// The root path where children are located, configured in the image
    root: PathBuf,
    /// The tags to set for any submitted files
    pub tags: HashMap<String, HashSet<String>>,
    /// Child files from building from source
    source: Vec<PathBuf>,
    /// Child files from unpacked samples
    unpacked: Vec<PathBuf>,
    /// Children carved from a sample
    carved: CarvedChildren,
}

/// Submit children files 10 at a time for a given sample/repo
macro_rules! submit {
    ($sample:expr, $children:expr, $origin:expr, $results:expr, $groups:expr, $depth:expr, $tags:expr, $thorium:expr, $logs:expr, $msg:literal) => {
        async {
            // submit any children 10 at a time
            stream::iter($children.clone())
                .map(|child| {
                    // if any children were found then generate our origin requests
                    let origin = $origin($sample, &child).result_ids($results.to_vec());
                    // build this origins sample request
                    let req = SampleRequest::new(child, $groups.to_vec()).origin(origin);
                    // set our trigger depth if we have one
                    let mut req = match $depth {
                        Some(trigger_depth) => req.trigger_depth(trigger_depth),
                        None => req,
                    };
                    // inject the tags for this child
                    req.tags.clone_from(&$tags);
                    // submit this sample to Thorium
                    $thorium.files.create(req)
                })
                .buffer_unordered(10)
                .collect::<Vec<Result<_, _>>>()
                .await
                .into_iter()
                .map(|res| match res {
                    // this child was submitted successfully so log its sha256 and id
                    Ok(sub) => {
                        log!($logs, "{} Submitted {} - {}", $msg, sub.sha256, sub.id);
                        Ok(())
                    }
                    // this child failed to submit so log an error
                    Err(err) => {
                        // check if this was an error of if this sample has just already been added
                        match err.status() {
                            Some(StatusCode::CONFLICT) => {
                                log!($logs, "{} Already Exists", $msg);
                                Ok(())
                            }
                            _ => {
                                log!($logs, "{} Error: {:#?}", $msg, err);
                                Err(err)
                            }
                        }
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        }
    };
}

impl Children {
    fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            root: path.as_ref().to_path_buf(),
            tags: HashMap::default(),
            source: Vec::default(),
            unpacked: Vec::default(),
            carved: CarvedChildren::default(),
        }
    }

    /// Collect children files
    ///
    /// # Arguments
    ///
    /// * `path` - The path to collect children files at
    /// * `logs` - The logs to send to the API
    pub async fn collect<P: AsRef<Path>>(
        path: P,
        logs: &mut Sender<String>,
    ) -> Result<Self, Error> {
        // build a Children object and collect any tags
        let mut children = Children::new(path).collect_tags(logs)?;
        // collect any children
        children.source();
        children.unpacked();
        children.carved();
        Ok(children)
    }

    /// Returns true if there are no children
    fn is_empty(&self) -> bool {
        self.unpacked.is_empty() && self.source.is_empty() && self.carved.is_empty()
    }

    /// Gather all tags to apply to any children
    ///
    /// # Arguments
    ///
    /// * `path` - The path to gather children tags at
    /// * `logs` - The logs to send to the API
    fn collect_tags(mut self, logs: &mut Sender<String>) -> Result<Self, Error> {
        // build the path to the tags file in our children folder
        let tags_path = self.root.join("tags");
        // read in out tags file
        if let Ok(tags_str) = std::fs::read_to_string(&tags_path) {
            // try to convert our tags_str to a HashMap
            self.tags = serde_json::from_str(&tags_str)?;
            // log any tags that we find
            for (key, value) in &self.tags {
                logs.send(format!(
                    "Found child tag {}={}",
                    key,
                    value.iter().join(", ")
                ))?;
            }
        }
        Ok(self)
    }

    /// Collect source children files and submit them to Thorium
    ///
    /// # Arguments
    ///
    /// * `path` - The root path to look for the source folder and its children files in
    fn source(&mut self) {
        // build the path to our source children
        let root = self.root.join("source");
        // recursively walk through this directory skipping any hidden files
        self.source = get_children(&root);
    }

    /// Collect unpacked children files and submit them to Thorium
    ///
    /// # Arguments
    ///
    /// * `path` - The root path to look for the unpacked folder and its children files in
    fn unpacked(&mut self) {
        // build the path to our source children
        let root = self.root.join("unpacked");
        // recursively walk through this directory skipping any hidden files
        self.unpacked = get_children(&root);
    }

    /// Collect carved children files
    ///
    /// # Arguments
    ///
    /// * `path` - The root path to look for the carved folder and its children files in
    fn carved(&mut self) {
        // build the path to our source children
        let root = self.root.join("carved");
        let unknown_root = root.join("unknown");
        let pcap_root = root.join("pcap");
        // recursively walk through this directory skipping any hidden files
        self.carved = CarvedChildren {
            unknown: get_children(&unknown_root),
            pcap: get_children(&pcap_root),
        };
    }

    /// Filter children based on the image's configured child filters
    ///
    /// # Arguments
    ///
    /// * `children` - The children to filter
    /// * `filters` - The child filters to apply
    /// * `logs` - The logs to send to the API
    #[instrument(name = "Children::filter", skip_all, err(Debug))]
    fn filter(&mut self, filters: &ChildFilters, logs: &mut Sender<String>) -> Result<(), Error> {
        // get a lock on our child filters cache;
        // TODO: only one agent is running at a time, so we don't expect others to need
        // the cache at the same time; the Agent *is* run on a tokio task though, so we
        // can't make the cache thread-local
        let mut filters_cache = CHILD_FILTERS_CACHE
            .lock()
            .map_err(|err| Error::new(format!("Error locking filter mutex: {err}")))?;
        // filter children from each of our lists
        let mut unpacked_filtered =
            FilteredChildren::filter(&mut self.unpacked, filters, &mut filters_cache, logs)?;
        let mut source_filtered =
            FilteredChildren::filter(&mut self.source, filters, &mut filters_cache, logs)?;
        let mut carved_pcap_filtered =
            FilteredChildren::filter(&mut self.carved.pcap, filters, &mut filters_cache, logs)?;
        let mut carved_unknown_filtered =
            FilteredChildren::filter(&mut self.carved.unknown, filters, &mut filters_cache, logs)?;
        if filters.submit_non_matches {
            // we're submitting ones that don't match, so re-add the non-matches to self to be submitted
            self.unpacked.append(&mut unpacked_filtered.non_matches);
            self.source.append(&mut source_filtered.non_matches);
            self.carved
                .pcap
                .append(&mut carved_pcap_filtered.non_matches);
            self.carved
                .unknown
                .append(&mut carved_unknown_filtered.non_matches);
        } else {
            // we're submitting ones that match, so re-add matches to self to be submitted
            self.unpacked.append(&mut unpacked_filtered.matches);
            self.source.append(&mut source_filtered.matches);
            self.carved.pcap.append(&mut carved_pcap_filtered.matches);
            self.carved
                .unknown
                .append(&mut carved_unknown_filtered.matches);
        }
        Ok(())
    }

    /// submit source children files to Thorium
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `job` - The job we are collecting results from
    /// * `results` - The result IDs from the job that uploaded these children
    /// * `depth` - The depth for this child source sample
    /// * `commits` - The commit that each repo is checked out too
    /// * `logs` - The logs to send to the API
    #[instrument(name = "Children::submit_source", skip_all, err(Debug))]
    pub async fn submit_source(
        &self,
        thorium: &Thorium,
        job: &GenericJob,
        results: &[Uuid],
        depth: Option<u8>,
        commits: &HashMap<String, String>,
        groups: &Vec<String>,
        logs: &mut Sender<String>,
    ) -> Result<(), Error> {
        // only attempt to ingest children if some exist
        if !self.source.is_empty() {
            // get the flags for this build job if any were set
            let flags = match job.args.kwargs.get("--flags") {
                Some(flags) => flags.clone(),
                None => job.args.to_vec(),
            };
            // get the build system or error
            let system = match job.args.kwargs.get("--builder") {
                Some(system_vec) => match system_vec.get(0) {
                    Some(system) => system,
                    // default to the stage name if --builder isn't set
                    None => &job.stage,
                },
                None => &job.stage,
            };
            // submit for each repo that was bound in
            for repo in &job.repos {
                // get the current commit for this repo
                let commit = match commits.get(&repo.url) {
                    Some(commit) => commit,
                    None => {
                        event!(Level::ERROR, repo = repo.url, msg = "Missing commit");
                        log!(
                            logs,
                            "Repo {} missing commit info - skipping children upload",
                            repo.url
                        );
                        continue;
                    }
                };
                submit!(
                    repo,
                    self.source,
                    |repo: &RepoDependency, child: &Path| {
                        // if this file has an extension that is not .so it is a supporting file
                        let supporting = is_supporting(child);
                        OriginRequest::source(
                            &repo.url,
                            repo.commitish.as_ref(),
                            commit,
                            flags.iter(),
                            system,
                            supporting,
                        )
                    },
                    results,
                    groups,
                    depth,
                    self.tags,
                    thorium,
                    logs,
                    "Source Child"
                )
                .await?;
            }
        }
        Ok(())
    }

    /// submit children files to Thorium
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `job` - The job we are collecting results from
    /// * `results` - The result IDs from the job that uploaded these children
    /// * `logs` - The logs to send to the API
    #[instrument(name = "Children::submit_unpacked", skip_all, err(Debug))]
    pub async fn submit_unpacked(
        &self,
        thorium: &Thorium,
        job: &GenericJob,
        results: &[Uuid],
        depth: Option<u8>,
        groups: &[String],
        logs: &mut Sender<String>,
    ) -> Result<(), Error> {
        // get the name of the tool that unpacked this sample
        let tool = &job.stage;
        // submit for each parent sample that was bound in
        for sample in &job.samples {
            submit!(
                sample,
                self.unpacked,
                |sample: &String, _child: &Path| {
                    OriginRequest::unpacked(sample.clone(), Some(tool.to_owned()))
                },
                results,
                groups,
                depth,
                self.tags,
                thorium,
                logs,
                "Unpacked Child"
            )
            .await?;
        }
        Ok(())
    }

    /// Try to get metadata for children carved from the PCAP
    ///
    /// The metadata is a map of file name to a JSON object of metadata
    ///
    /// Logs any errors and returns an empty map if the metadata file is
    /// missing, empty, or an error occurred
    ///
    /// # Arguments
    ///
    /// * `logs` - The logs to send to the API
    async fn get_pcap_metadata(&self, logs: &mut Sender<String>) -> HashMap<String, PcapMetadata> {
        let pcap_metadata_path = self.root.join("carved/pcap/thorium_pcap_metadata.json");
        match PcapMetadata::map_from_file(&pcap_metadata_path).await {
            Ok(maybe_metadata) => {
                match maybe_metadata {
                    Some(metadata) => {
                        let num_pcap = self.carved.pcap.len();
                        if num_pcap != 0 && metadata.is_empty() {
                            log!(logs, "{} Carved-PCAP children were found, but metadata file '{}' is empty!", num_pcap, pcap_metadata_path.to_string_lossy());
                        } else if num_pcap == 0 && !metadata.is_empty() {
                            log!(logs, "No Carved-PCAP children were found, but metadata file '{}' has data!", pcap_metadata_path.to_string_lossy());
                        }
                        metadata
                    }
                    None => {
                        let num_pcap = self.carved.pcap.len();
                        if num_pcap != 0 {
                            log!(logs, "{} Carved-PCAP children were found, but metadata file '{}' is missing!", num_pcap, pcap_metadata_path.to_string_lossy());
                        }
                        HashMap::default()
                    }
                }
            }
            Err(Error::IO(err)) => {
                log!(
                    logs,
                    "The metadata for Carved-PCAP children in '{}' could not be read: {}",
                    pcap_metadata_path.to_string_lossy(),
                    err
                );
                HashMap::default()
            }
            Err(Error::Serde(err)) => {
                log!(
                    logs,
                    "The metadata for Carved-PCAP children in '{}' is formatted incorrectly: {}",
                    pcap_metadata_path.to_string_lossy(),
                    err
                );
                HashMap::default()
            }
            Err(err) => {
                log!(logs, "An unknown error occurred reading metadata for Carved-PCAP children in '{}': {}", pcap_metadata_path.to_string_lossy(), err);
                HashMap::default()
            }
        }
    }

    /// submit carved children files to Thorium
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `job` - The job we are collecting results from
    /// * `results` - The result IDs from the job that uploaded these children
    /// * `groups` - The groups to submit to
    /// * `logs` - The logs to send to the API
    #[instrument(name = "Children::submit_carved", skip_all, err(Debug))]
    pub async fn submit_carved(
        &self,
        thorium: &Thorium,
        job: &GenericJob,
        results: &[Uuid],
        depth: Option<u8>,
        groups: &[String],
        logs: &mut Sender<String>,
    ) -> Result<(), Error> {
        // get the name of the tool that unpacked this sample
        let tool = &job.stage;
        // try to get metadata on the pcap files
        let pcap_metadata = self.get_pcap_metadata(logs).await;
        // submit for each parent sample that was bound in
        for sample in &job.samples {
            // submit samples carved from a PCAP
            submit!(
                sample,
                self.carved.pcap,
                |sample: &String, child: &Path| {
                    // see if this child has metadata
                    if let Some(metadata) = child
                        .file_name()
                        // try to get a str representation of the file name
                        .and_then(|file_name| file_name.to_str())
                        // see if we have metadata on this file
                        .and_then(|file_str| pcap_metadata.get(file_str))
                    {
                        OriginRequest::carved_pcap(
                            sample.clone(),
                            Some(tool.to_owned()),
                            metadata.src_ip,
                            metadata.dest_ip,
                            metadata.src_port,
                            metadata.dest_port,
                            metadata.proto.clone(),
                            metadata.url.clone(),
                        )
                    } else {
                        // this child has no metadata so just submit without it
                        OriginRequest::carved_pcap(
                            sample.clone(),
                            Some(tool.to_owned()),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    }
                },
                results,
                groups,
                depth,
                self.tags,
                thorium,
                logs,
                "Carved-PCAP Child"
            )
            .await?;
            // submit samples carved from an unknown source
            submit!(
                sample,
                self.carved.unknown,
                |sample: &String, _child: &Path| {
                    OriginRequest::carved_unknown(sample.clone(), Some(tool.to_owned()))
                },
                results,
                groups,
                depth,
                self.tags,
                thorium,
                logs,
                "Carved-Unknown Child"
            )
            .await?;
        }
        Ok(())
    }

    /// submit all children files to Thorium
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `job` - The job we are submitting children for
    /// * `results` - The result IDs to submit children for
    /// * `depth` - The current depth of triggers
    /// * `commits` - The commits for any repos passed in as inputs
    /// * `image` - The image for this job
    /// * `logs` - The logs to send to the API
    #[instrument(name = "Children::submit", skip(self, thorium, logs), err(Debug))]
    pub async fn submit(
        &mut self,
        thorium: &Thorium,
        job: &GenericJob,
        results: &[Uuid],
        depth: Option<u8>,
        commits: &HashMap<String, String>,
        image: &Image,
        logs: &mut Sender<String>,
    ) -> Result<(), Error> {
        // if we have any children, get our groups and submit them
        if !self.is_empty() {
            // get the groups we want to submit these unpacked samples too
            let mut groups = get_parent_groups(thorium, job).await?;
            // if our groups for this image are restricted then restrict to those groups
            if !image.output_collection.groups.is_empty() {
                groups.retain(|group| image.output_collection.groups.contains(group));
            }
            // filter children based on configured image child filters if we have any
            if !image.child_filters.is_empty() {
                self.filter(&image.child_filters, logs)?;
            }
            // submit children
            self.submit_source(thorium, job, results, depth, commits, &groups, logs)
                .await?;
            self.submit_unpacked(thorium, job, results, depth, &groups, logs)
                .await?;
            self.submit_carved(thorium, job, results, depth, &groups, logs)
                .await?;
        }
        Ok(())
    }
}

/// Metadata about a child carved from a PCAP, derived from [`CarvedOrigin::Pcap`]
///
/// This is a separate struct because we can't have an enum variant return from a function,
/// and we don't want to have to force tool developers to include the carved origin type
/// for every value in the metadata map (see `get_pcap_metadata()`)
#[derive(Deserialize, Debug)]
struct PcapMetadata {
    /// The source IP this sample was sent from
    #[serde(alias = "src")]
    src_ip: Option<IpAddr>,
    /// The destination IP this sample was going to
    #[serde(alias = "dest")]
    dest_ip: Option<IpAddr>,
    /// The source port this sample was sent from
    src_port: Option<u16>,
    /// The destination port this sample was going to
    dest_port: Option<u16>,
    /// The type of protocol this sample was transported in
    #[serde(alias = "protocol")]
    proto: Option<PcapNetworkProtocol>,
    /// The URL this file was retrieved from or sent to
    url: Option<String>,
}

impl PcapMetadata {
    /// Try to get metadata on PCAP-carved children from the given file
    ///
    /// The metadata is a map of the child's file name to a JSON object of metadata
    ///
    /// Returns `None` if no metadata file exists
    ///
    /// # Arguments
    ///
    /// * `path` - The path to check for metadata
    async fn map_from_file(path: &Path) -> Result<Option<HashMap<String, Self>>, Error> {
        // make sure the metadata path exists
        if tokio::fs::try_exists(&path).await? {
            // read the data to a string
            let metadata_raw = tokio::fs::read_to_string(&path).await?;
            // try deserializing
            let metadata = serde_json::from_str(&metadata_raw)?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }
}

impl From<PcapMetadata> for CarvedOrigin {
    /// Convert the `PcapMetadata` to a [`CarvedOrigin`]
    ///
    /// Implemented to guarantee everything in [`CarvedOrigin::Pcap`] is in
    /// `PcapMetadata`
    fn from(metadata: PcapMetadata) -> Self {
        CarvedOrigin::Pcap {
            src_ip: metadata.src_ip,
            dest_ip: metadata.dest_ip,
            src_port: metadata.src_port,
            dest_port: metadata.dest_port,
            proto: metadata.proto,
            url: metadata.url,
        }
    }
}
