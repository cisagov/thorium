//! Handles files commands
use download::FilesDownloadWorker;
use futures::stream::{self, StreamExt};
use futures::TryStreamExt;
use http::status::StatusCode;
use owo_colors::OwoColorize;
use regex::RegexSet;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use thorium::models::{
    FileDeleteOpts, ReactionRequest, Sample, SampleCheck, SampleListLine, SampleSubmissionResponse,
    SubmissionChunk,
};
use thorium::{CtlConf, Error, Thorium};
use uuid::Uuid;
use walkdir::DirEntry;

mod download;

use super::{update, Controller};
use crate::args::files::{DeleteFiles, DescribeFiles, DownloadFiles, Files, GetFiles, UploadFiles};
use crate::args::{Args, DescribeCommand, SearchParameterized};
use crate::utils;

/// A single line for a file upload log
struct UploadLine;

macro_rules! upload_print {
    ($status:expr, $path:expr, $sha256:expr, $id:expr, $msg:expr) => {
        println!(
            "{:<4} | {:<32} | {:<64} | {:<36} | {:<24}",
            $status,
            $path.to_string_lossy(),
            $sha256,
            $id,
            $msg
        )
    };
}

impl UploadLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{} | {:<32} | {:<64} | {:<36} | {:<24}",
            "CODE", "PATH", "SHA256", "SUBMISSION", "MESSAGE"
        );
        println!("{:-<5}+{:-<34}+{:-<66}+{:-<38}+{:-<26}", "", "", "", "", "");
    }

    /// Build and print a successful file upload log line
    ///
    /// # Arguments
    ///
    /// * `path` - The path this file was uploaded from
    /// * `resp` - The submission response from the API
    pub fn uploaded(path: &Path, resp: &SampleSubmissionResponse) {
        // print an uploaded line
        upload_print!("200".bright_green(), path, resp.sha256, resp.id, "-");
    }

    /// Build and print that this file was already uploaded
    ///
    /// # Arguments
    ///
    /// * `path` - The path this file was uploaded from
    /// * `sha256` - The sha256 for this file
    pub fn conflict(path: &Path, sha256: &str) {
        upload_print!("409".bright_blue(), path, sha256, "-", "Already Exists");
    }

    /// Build and print that we failed to upload a file
    ///
    /// # Arguments
    ///
    /// * `path` - The path this file was uploaded from
    /// * `sha256` - The sha256 for this file
    pub fn error(path: &Path, err: &thorium::Error) {
        // get the error message if one was set
        let msg = err.msg().unwrap_or_else(|| "-".to_owned());
        // show either the reqwest body error or the hyper error
        match err.status() {
            // we have a status so well return the code and body as a message
            Some(code) => upload_print!(code.as_str().bright_red(), path, "-", "-", msg),
            // no status code is present so just use '-' painted bright red
            None => upload_print!("-".bright_red(), path, "-", "-", msg),
        };
    }
}

/// Hashes a file and uploads it if it doesn't exist
///
/// # Arguments
///
///  * `thorium` - A Thorium client
///  * `cmd` - The upload files command to execute
///  * `entry` - The file entry we are uploading
async fn uploader(
    thorium: &Thorium,
    cmd: &UploadFiles,
    entry: &DirEntry,
    reaction_reqs: Vec<ReactionRequest>,
) -> Result<(), Error> {
    // get the path for this file
    let path = entry.path();
    // get the sha256 for this file
    let sha256 = utils::sha256(path).await?;
    // check if this file has already been uploaded
    let exists = thorium
        .files
        .exists(&SampleCheck::new(sha256.clone()))
        .await?;
    // if this id does not already exist then upload it
    if exists.id.is_none() {
        // Build the sample request for this file
        let sample_req = cmd.build_req(path);
        // upload this file
        let resp = thorium.files.create(sample_req).await;
        // determine if we should print an error message or not
        match resp {
            Ok(resp) => {
                UploadLine::uploaded(path, &resp);
                // create reactions for the new file concurrently
                stream::iter(
                    reaction_reqs
                        .into_iter()
                        .map(|req| req.sample(sha256.clone())),
                )
                .map(Ok)
                .try_for_each_concurrent(10, |req| async move {
                    thorium.reactions.create(&req).await.map(|_| ())
                })
                .await?;
            }
            Err(err) => {
                // if this file was already uploaded then don't print an error
                if err.status() == Some(StatusCode::CONFLICT) {
                    UploadLine::conflict(path, &sha256);
                } else {
                    UploadLine::error(path, &err);
                }
            }
        }
    } else {
        UploadLine::conflict(path, &sha256);
    }
    Ok(())
}

/// Build base reaction requests from the given pipelines
/// or error if they're invalid or don't exist
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The files upload command
pub async fn build_reaction_reqs(
    thorium: &Thorium,
    cmd: &UploadFiles,
) -> Result<Vec<ReactionRequest>, Error> {
    // create list of base reaction requests
    let reaction_reqs = match &cmd.pipelines {
            Some(pipelines) =>
                stream::iter(pipelines
                    .iter())
                    .map(|pipeline| async move {
                        // attempt to parse the pipeline name/group
                        let parse_err = || Error::new(format!(
                            "Error parsing pipeline '{pipeline}'; pipeline should be formatted '<PIPELINE>:<GROUP>'",
                        ));
                        let mut split = pipeline.split(':');
                        let name = split.next().ok_or_else(parse_err)?;
                        let group = split.next().ok_or_else(parse_err)?;
                        if split.next().is_some() {
                            return Err(parse_err());
                        }
                        thorium.pipelines.get(group, name).await.map_err(|err|
                            Error::new(format!("Invalid pipeline '{pipeline}': {err}")))?;
                        // create a base reaction request
                        Ok(ReactionRequest::new(group, name))
                    })
                    .buffer_unordered(25)
                    .collect::<Vec<Result<ReactionRequest, Error>>>().await,
            None => Vec::default(),
        };
    // recollect the possible Vec<Result> to a Result<Vec> or just convert the default Vec to a Result
    reaction_reqs.into_iter().collect()
}

/// Crawl a directory and upload files
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The files or directories to crawl and upload
async fn upload(thorium: &Thorium, cmd: &UploadFiles) -> Result<(), Error> {
    // get reaction requests
    let reaction_reqs = build_reaction_reqs(thorium, cmd).await?;
    // build the set of regexs to determine which files to include or skip
    let filter = RegexSet::new(&cmd.filter)?;
    let skip = RegexSet::new(&cmd.skip)?;
    // print the upload logs headers
    UploadLine::header();
    // crawl over each path and upload them if they are new
    for target in &cmd.targets {
        stream::iter(
            // retrieve all file entries at the given target path, filtered based on settings
            utils::fs::get_filtered_entries(
                target,
                &filter,
                &skip,
                cmd.include_hidden,
                cmd.filter_dirs,
            ),
        )
        .map(|entry| {
            let reaction_reqs = reaction_reqs.clone();
            async move {
                // upload the entry if it's new
                if let Err(err) = uploader(thorium, cmd, &entry, reaction_reqs).await {
                    UploadLine::error(entry.path(), &err);
                }
            }
        })
        .buffer_unordered(25)
        .collect::<Vec<()>>()
        .await;
    }
    Ok(())
}

/// Download all requested files from Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full download command/args
async fn download_targets(
    cmd: &DownloadFiles,
    added: &mut HashSet<String>,
    controller: &mut Controller<FilesDownloadWorker>,
) {
    // add any files from our cmd
    for sha256 in &cmd.sha256s {
        // check if this repo has already been added to be downloaded
        if added.insert(sha256.clone()) {
            // try to add this download job
            if let Err(error) = controller.add_job(sha256.clone()).await {
                // log this error
                controller.error(&error.to_string());
            }
        }
    }
}

/// Download files from Thorium based on search parameters
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full download command/args
async fn download_search(
    thorium: &Thorium,
    cmd: &DownloadFiles,
    added: &mut HashSet<String>,
    controller: &mut Controller<FilesDownloadWorker>,
) -> Result<(), Error> {
    // make sure the command is valid
    cmd.validate_search()?;
    // build a search object from our args
    let search = cmd.build_file_opts()?;
    // build a cursor object
    let mut cursor = thorium.files.list(&search).await?;
    // crawl over this cursor until its exhausted
    loop {
        // add each of these sha256s to be downloaded
        for line in cursor.data.drain(..) {
            // check if this repo has already been added to be downloaded
            if added.insert(line.sha256.clone()) {
                // try to add this download job
                if let Err(error) = controller.add_job(line.sha256).await {
                    // log this error
                    controller.error(&error.to_string());
                }
            }
        }
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of data
        cursor.refill().await?;
    }
    Ok(())
}

/// Download all requested files from Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full download command/args
async fn download(
    thorium: &Thorium,
    cmd: &DownloadFiles,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // create a new worker controller
    let mut controller = Controller::<FilesDownloadWorker>::spawn(
        "Downloading Files",
        thorium,
        args.workers,
        conf,
        args,
        cmd,
    )
    .await;
    // track and remove any duplicate repos as those can cause hangs
    let mut added = HashSet::with_capacity(100);
    // add the user specified targets to be downloaded
    download_targets(cmd, &mut added, &mut controller).await;
    // if no sha256s were provided then search for samples to download
    if cmd.sha256s.is_empty() {
        // add targets based on searching in Thorium
        if let Err(error) = download_search(thorium, cmd, &mut added, &mut controller).await {
            // log this error
            controller.error(&error.to_string());
        }
    }
    // wait for all our workers to complete
    controller.finish().await?;
    Ok(())
}

/// A single line for a file upload log
struct GetLine;

impl GetLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<64} | {:<36} | {:<28}",
            "SHA256", "SUBMISSION", "UPLOADED "
        );
        println!("{:-<65}+{:-<38}+{:-<28}", "", "", "");
    }

    /// Print the files get header when tags are used to query
    pub fn header_tags() {
        println!("{:<64} | {:<28}", "SHA256", "UPLOADED ");
        println!("{:-<65}+{:-<28}", "", "");
    }

    /// Build and print a successful file list line
    ///
    /// # Arguments
    ///
    ///* `line` - The sample list line to print
    pub fn list(line: &SampleListLine) {
        // if a submission was set the get it as a string or use "-"
        let submission = line.submission.map_or("-".to_string(), String::from);
        // print a list file line
        println!(
            "{:<64} | {:<36} | {:<28}",
            line.sha256, submission, line.uploaded
        );
    }

    /// Build and print a successful file list line from a query with tags
    ///
    /// # Arguments
    ///
    ///* `line` - The sample list line to print
    pub fn list_tags(line: &SampleListLine) {
        // print a list file line
        println!("{:<64} | {:<28}", line.sha256, line.uploaded);
    }
}

/// Gets information on files
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full get command/args
async fn get(thorium: &Thorium, cmd: &GetFiles) -> Result<(), Error> {
    // print the header for getting file info, omitting submission if tags are included
    if cmd.tags.is_empty() {
        GetLine::header();
    } else {
        GetLine::header_tags();
    }
    // build a search object from our args
    let opts = cmd.build_file_opts()?;
    // build a cursor object
    let mut cursor = thorium.files.list(&opts).await?;
    // track the submission IDs that we find
    let mut ids = HashSet::with_capacity(10000);
    // track the number of pages we have crawled
    let mut pages = 0;
    // crawl over this cursor until its exhausted
    loop {
        // crawl over each submission in this chunk and check if its a dup
        for sub in cursor.data.iter().flat_map(|sample| &sample.submission) {
            if !ids.insert(sub.clone()) {
                panic!("Duplicate on page {pages} - {sub}");
            }
        }
        // crawl the files listed and print info about them
        if cmd.tags.is_empty() {
            cursor.data.iter().for_each(GetLine::list);
        } else {
            cursor.data.iter().for_each(GetLine::list_tags);
        }
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of data
        cursor.refill().await?;
        pages += 1;
    }
    Ok(())
}

/// Describe files by displaying/saving their JSON-formatted details
///
/// * `thorium` - The Thorium client
/// * `cmd` - The [`DescribeFiles`] command to execute
async fn describe(thorium: &Thorium, cmd: &DescribeFiles) -> Result<(), Error> {
    cmd.describe(thorium).await
}

/// A single line for a file upload log
struct DeleteLine;

impl DeleteLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<64} | {:<36} | {:<18} | {:<24}",
            "SHA256", "SUBMISSION", "GROUP", "MESSAGE"
        );
        println!("{:-<65}+{:-<38}+{:-<20}+{:-<26}", "", "", "", "");
    }

    /// Print that we successfully deleted a submission
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sample that was deleted
    /// * `submission` - The submission that was deleted
    pub fn deleted(sha256: &str, submission: &Uuid, groups: &Vec<String>) {
        // print a delete file line
        if groups.is_empty() {
            println!("{:<64} | {:<28} | -{:<18} | -", sha256, submission, "");
        } else {
            for group in groups {
                println!("{sha256:<64} | {submission:<28} | {group:<18} | -");
            }
        }
    }

    /// Log an error from writing a samples details
    pub fn error(sha256: &str, submission: Option<&Uuid>, err: &Error) {
        // print a delete file line
        match submission {
            Some(sub) => println!("{:<64} | {:<28} | {:<18} | {}", sha256, sub, "-", err),
            None => println!("{:<64} | {:<28} | {:<18} | {}", sha256, "-", "-", err),
        }
    }
}

/// The sample SHA256 and list of submissions UUID's designating files for deletion
struct DeleteTarget {
    sha256: String,
    submissions: Vec<Uuid>,
}

/// Parse the sha256 and list of submissions from the raw target string
///
/// # Arguments
///
/// * `target` - The raw target string
fn parse_target(target: &str) -> DeleteTarget {
    // get the submission if its specified or get all of our submissions
    if target.contains(':') {
        // split this into a sha256 and a submission
        let mut split = target.split(':');
        // get our sha256
        let sha256 = split.by_ref().take(1).collect::<String>();
        // get our submissions
        let submissions = split
            .map(Uuid::parse_str)
            .filter_map(|res| {
                match res {
                    Ok(sub) => Some(sub),
                    Err(error) => {
                        // log this error
                        DeleteLine::error(&sha256, None, &Error::from(error));
                        None
                    }
                }
            })
            .collect::<Vec<Uuid>>();
        return DeleteTarget {
            sha256,
            submissions,
        };
    }
    DeleteTarget {
        sha256: target.to_owned(),
        submissions: Vec::new(),
    }
}

/// Delete a sample or its submissions based on a specific target given by the user
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `target` - The sha256 + optional submissions to delete
/// * `users` - The users to delete submissions for (may be empty)
/// * `groups` - The groups to delete from (may be empty)
async fn delete_specific(
    thorium: &Thorium,
    target: DeleteTarget,
    users: &HashSet<String>,
    groups: &Vec<String>,
) {
    // retrieve the file's details
    let details = match thorium.files.get(&target.sha256).await {
        Ok(details) => details,
        Err(error) => {
            // log error and cancel if the file could not be retrieved
            DeleteLine::error(&target.sha256, None, &error);
            return;
        }
    };
    // create a map of submission id's to submissions
    let submission_map: HashMap<Uuid, &SubmissionChunk> = details
        .submissions
        .iter()
        .map(|sub| (sub.id, sub))
        .collect();
    // select either all of the file's submissions or only the given submissions
    let submissions = if target.submissions.is_empty() {
        submission_map.keys().copied().collect()
    } else {
        target.submissions
    };
    let mut opts = FileDeleteOpts::default();
    // if groups were specified, add them to the our delete opts; otherwise,
    // delete submissions from all groups
    if !groups.is_empty() {
        opts = opts.groups(groups.clone());
    }
    // crawl over the submissions we are deleting
    for sub_id in &submissions {
        // attempt to retrieve the submission by its given id
        match submission_map.get(sub_id) {
            Some(sub) => {
                if !users.is_empty() && !users.contains(&sub.submitter) {
                    // error and continue if the submitter is not among the list of users to delete from (if any were given)
                    DeleteLine::error(
                        &target.sha256,
                        Some(sub_id),
                        &Error::new(
                            "The submission was not submitted by any of the given users!"
                                .to_string(),
                        ),
                    );
                    continue;
                }
            }
            None => {
                // error and skip if the submission is not found in the sample
                DeleteLine::error(
                    &target.sha256,
                    Some(sub_id),
                    &Error::new(
                        "The sample does not have a submission with the given id!".to_string(),
                    ),
                );
                continue;
            }
        }
        // delete this submission and log the result
        match thorium.files.delete(&target.sha256, sub_id, &opts).await {
            Ok(_) => DeleteLine::deleted(&target.sha256, sub_id, groups),
            Err(error) => DeleteLine::error(&target.sha256, Some(sub_id), &error),
        }
    }
}

/// Delete samples that we already have details for
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `details` - The sample details to delete
/// * `users` - The users to delete submissions for (may be empty)
/// * `groups` - The groups to delete submissions from (may be empty)
async fn delete_from_details(
    thorium: &Thorium,
    details: &Sample,
    users: &HashSet<String>,
    groups: &Vec<String>,
) {
    let submissions: Vec<&Uuid> = if users.is_empty() {
        // don't filter submissions (delete all submissions) if no users were given
        details.submissions.iter().map(|sub| &sub.id).collect()
    } else {
        // select only submissions submitted by the given list of users
        details
            .submissions
            .iter()
            .filter_map(|sub| users.contains(&sub.submitter).then_some(&sub.id))
            .collect()
    };
    let mut opts = FileDeleteOpts::default();
    // if groups were specified, add them to the our delete opts; otherwise,
    // delete all submissions
    if !groups.is_empty() {
        opts = opts.groups(groups.clone());
    }
    for sub in &submissions {
        match thorium.files.delete(&details.sha256, sub, &opts).await {
            Ok(_) => DeleteLine::deleted(&details.sha256, sub, groups),
            Err(error) => DeleteLine::error(&details.sha256, Some(sub), &error),
        }
    }
}

/// Deletes files from Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full get command/args
async fn delete(thorium: &Thorium, cmd: &DeleteFiles) -> Result<(), Error> {
    // make sure the command has targets or at least one search parameter
    cmd.validate_search()?;
    // make sure the force flag is set if any search parameters were given
    if cmd.has_parameters() && !cmd.force {
        return Err(Error::new(
            "--force/-f is required when deleting files without \
            explicitly supplying their SHA256's",
        ));
    }
    // print the header for getting file info
    DeleteLine::header();
    // create the set of users to delete from
    let users = if cmd.users.is_empty() {
        if cmd.all_users {
            // provide an empty set to signal to delete from all users
            HashSet::default()
        } else {
            // if no users were given, default to just the current user
            HashSet::from([thorium.users.info().await?.username])
        }
    } else {
        cmd.users.iter().cloned().collect()
    };
    if cmd.has_targets() {
        // delete explicitly specified targets if any were given
        stream::iter(&cmd.targets)
            .for_each_concurrent(10, |raw_target| {
                delete_specific(thorium, parse_target(raw_target), &users, &cmd.groups)
            })
            .await;
    }
    if cmd.has_parameters() {
        // search for samples to delete if any search parameters were given
        let opts = cmd.build_file_opts()?;
        // build a cursor object
        let mut cursor = thorium.files.list_details(&opts).await?;
        // crawl over this cursor until its exhausted
        loop {
            stream::iter(&cursor.data)
                .for_each_concurrent(10, |sample| {
                    delete_from_details(thorium, sample, &users, &cmd.groups)
                })
                .await;
            // check if this cursor has been exhausted
            if cursor.exhausted() {
                break;
            }
            // get the next page of data
            cursor.refill().await?;
        }
    }
    Ok(())
}

/// Handle all files commands or print files docs
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The files command to execute
pub async fn handle(args: &Args, cmd: &Files) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        update::ask_update(&thorium).await?;
    }
    // call the right files handler
    match cmd {
        Files::Upload(cmd) => upload(&thorium, cmd).await,
        Files::Download(cmd) => download(&thorium, cmd, args, &conf).await,
        Files::Get(cmd) => get(&thorium, cmd).await,
        Files::Describe(cmd) => describe(&thorium, cmd).await,
        Files::Delete(cmd) => delete(&thorium, cmd).await,
    }
}
