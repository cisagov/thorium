use futures::stream::{self, StreamExt};
use http::StatusCode;
use owo_colors::OwoColorize;
use std::collections::HashSet;
use std::fmt::Display;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use thorium::client::ResultsClient;
use thorium::models::{
    OnDiskFile, Output, OutputDisplayType, OutputRequest, ResultGetParams, Sample,
};
use thorium::{Error, Thorium};
use tokio::fs::{create_dir_all, File};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;
use walkdir::WalkDir;

use super::update;
use crate::args::results::{GetResults, Results, ResultsPostProcessing, UploadResults};
use crate::args::{Args, SearchParameterized};
use crate::utils;

/// prints out a single downloaded result line
macro_rules! get_print {
    ($code:expr, $sample:expr, $msg:expr) => {
        println!("{:<4} | {:<64} | {:<32} ", $code, $sample, $msg)
    };
}

/// A single line for a reaction creation log
struct GetLine;

impl GetLine {
    /// Print this log lines header
    pub fn header() {
        println!("CODE | {:<64} | {:<32} ", "SAMPLE/REPO", "MESSAGE");
        println!("{:-<5}+{:-<66}+{:-<34}", "", "", "");
    }

    /// Print a log line for a retrieved result
    pub fn downloaded<D: Display>(target: D) {
        // log this line
        get_print!(200.bright_green(), target, "-");
    }

    /// Print an error log line for a result that could not be retrieved
    pub fn error<D: Display>(target: D, err: &Error) {
        // get the error message for this line
        let msg = err.msg().unwrap_or_else(|| "-".to_owned());
        // get the error status
        let status = err.status();
        // get a default "-" if no status; otherwise, map to a str
        let status_str = status.as_ref().map_or("-", StatusCode::as_str);
        // log this line
        get_print!(status_str.bright_red(), target, msg);
    }
}

/// Find and create a unique file for this result_file
///
/// # Arguments
///
/// * `file_path` - The file path to make unique
async fn create_unique_file(file_path: &PathBuf) -> Result<File, Error> {
    // get our this paths prefix
    let prefix = file_path.file_prefix().unwrap_or_default().to_owned();
    // get our result files name
    let name = file_path.file_name().unwrap_or_default();
    // get the size of our prefix in bytes
    let prefix_size = prefix.as_encoded_bytes().len();
    // get our suffix
    let suffix = name.slice_encoded_bytes(prefix_size..);
    let parent = file_path
        .parent()
        .map(|parent| parent.to_path_buf())
        .unwrap_or_else(|| PathBuf::from_str("/").unwrap());
    // the integer to append to this duplicate file name
    let mut num = 1;
    // keep trying new file names until we find one that is unique
    loop {
        // clone our prefix
        let mut new_name = prefix.clone();
        // add our unique suffix
        new_name.push(format!(" ({num})"));
        new_name.push(suffix);
        // append a number to this file to make it unique
        let possible_path = parent.join(new_name);
        // check if this path doesn't already exist
        if !tokio::fs::try_exists(&possible_path).await? {
            // open the file to write this result file too
            break Ok(tokio::fs::File::create(&possible_path).await?);
        }
        // increment our number and try again
        num += 1;
    }
}

/// Get an individual results file from a result
///
/// # Arguments
///
/// * `client` - The client to use to get the results
/// * `key` - The key to use to retrieve the entity that the results are attached to
/// * `id` - The result's id
/// * `sub_path` - The result file's sub path
/// * `tool` - The tool the result file originates from
/// * `root` - The root path the result file should be written to
async fn get_results_file(
    client: &impl ResultsClient,
    key: &str,
    id: &Uuid,
    sub_path: &str,
    tool: &str,
    root: &Path,
) -> Result<(), Error> {
    // build the path to write this file too
    let file_path = root.join(sub_path);
    // skip any files with invalid parents
    if let Some(parent) = file_path.parent() {
        // download our result file
        let result_file = client
            .download_result_file(&key, tool, id, sub_path)
            .await?;
        // create the dir to save this too
        create_dir_all(&parent).await?;
        // check if this file already exists
        let mut file = if tokio::fs::try_exists(&file_path).await? {
            // find and create a unique file path for this result file
            create_unique_file(&file_path).await?
        } else {
            // open the file to write this result file too
            tokio::fs::File::create(&file_path).await?
        };
        // write our result file to disk
        file.write_all(&result_file.data).await?;
    }
    Ok(())
}

/// Download results for specific tools and optionally any files
///
/// # Arguments
///
/// * `client` - The client to use to get the results
/// * `cmd` - The command to use for downloading results
/// * `key` - The key to use to retrieve the entity that the results are attached to
/// * `params` - The params to use when getting results
async fn get_results(
    client: &impl ResultsClient,
    cmd: &GetResults,
    key: &str,
    params: &ResultGetParams,
) -> Result<(), Error> {
    // get this samples results
    let results = client.get_results(&key, params).await?;
    // only write results if we have some
    if !results.results.is_empty() {
        // build the root path for this samples results
        let mut root = PathBuf::from(&cmd.output);
        root.push(key);
        // create a dir for this samples results
        create_dir_all(&root).await?;
        // crawl and write each tools results and their files to disk
        for (tool, output) in results.results {
            // write just the first output off to disk (the most recent)
            if let Some(recent_output) = output.into_iter().next() {
                // append our tools name to our root path
                root.push(&tool);
                // create a folder to store this tools results
                create_dir_all(&root).await?;
                // add a directory for result files
                root.push("result-files");
                // download and write each of this tool's result files off to disk
                for sub_path in &recent_output.files {
                    if let Err(err) =
                        get_results_file(client, key, &recent_output.id, sub_path, &tool, &root)
                            .await
                    {
                        // log an error if we're unable to download a result file, but keep going
                        GetLine::error(key, &err);
                    }
                }
                // pop our result files dir
                root.pop();
                // process and save this result
                write_result(recent_output, &root, &cmd.post_processing, cmd.condensed).await?;
                // if we had tool results to download then pop our tool name now that we are done
                root.pop();
            }
        }
    }
    Ok(())
}

/// Process and write this result
///
/// # Arguments
///
/// * `output` - The output containing the result to write
/// * `root` - The root path to write this result to
/// * `post_processing` - The post processing settings to use
/// * `condensed` - Whether to save in a condensed format
async fn write_result(
    mut output: Output,
    root: &Path,
    post_processing: &ResultsPostProcessing,
    condensed: bool,
) -> Result<(), Error> {
    // get the result data to write to the file depending on our processing settings
    let result_data = match post_processing {
        ResultsPostProcessing::Strip => {
            // get the result data from the output
            if output.display_type == OutputDisplayType::Json || !output.result.is_string() {
                if condensed {
                    serde_json::to_string(&output.result)? + "\n"
                } else {
                    serde_json::to_string_pretty(&output.result)? + "\n"
                }
            } else {
                // otherwise deserialize to a raw String
                serde_json::from_value(output.result)?
            }
        }
        ResultsPostProcessing::Split => {
            // remove the actual data from the output and replace it with a note to see
            // the data file
            let data = output.result.take();
            output.result = serde_json::json!("See 'results'");
            let result_data = if output.display_type == OutputDisplayType::Json || !data.is_string()
            {
                if condensed {
                    serde_json::to_string(&data)? + "\n"
                } else {
                    serde_json::to_string_pretty(&data)? + "\n"
                }
            } else {
                // otherwise deserialize to a raw String
                serde_json::from_value(data)?
            };
            // serialize our metadata
            let serialized = if condensed {
                serde_json::to_string(&output)? + "\n"
            } else {
                serde_json::to_string_pretty(&output)? + "\n"
            };
            // open a file handle and write out our result's metadata
            let metadata_path = root.join("results_metadata.json");
            let mut file = tokio::fs::File::create(&metadata_path).await?;
            file.write_all(serialized.as_bytes()).await?;
            // return the output's data
            result_data
        }
        // just return the serialized output
        ResultsPostProcessing::Full => {
            if condensed {
                serde_json::to_string(&output)? + "\n"
            } else {
                serde_json::to_string_pretty(&output)? + "\n"
            }
        }
    };
    // push the result file name
    let results_path = root.join("results");
    // open a file handle and write out our data
    let mut file = tokio::fs::File::create(&results_path).await?;
    file.write_all(result_data.as_bytes()).await?;
    Ok(())
}

/// Gets results for multiple files based on search parameters
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full reaction creation command/args
/// * `params` - The params to use when getting results
async fn get_search_files(
    thorium: &Thorium,
    cmd: &GetResults,
    params: &ResultGetParams,
) -> Result<(), Error> {
    // download results if any search parameters were given
    let opts = cmd.build_file_opts()?;
    // build a cursor object
    let mut cursor = thorium.files.list(&opts).await?;
    // crawl over this cursor until its exhausted
    loop {
        // convert our names list to just a list of SHA256's
        let sha256s = cursor
            .data
            .drain(..)
            .map(|line| line.sha256)
            .collect::<HashSet<String>>();
        // download the target samples results
        stream::iter(sha256s.iter())
            .for_each_concurrent(None, |sha256| async move {
                match get_results(&thorium.files, cmd, sha256, params).await {
                    Ok(()) => GetLine::downloaded(sha256),
                    Err(err) => GetLine::error(sha256, &err),
                }
            })
            .await;
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            return Ok(());
        }
        // get the next page of data
        cursor.refill().await?;
    }
}

/// Gets results for multiple repos based on search parameters
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full reaction creation command/args
/// * `params` - The params to use when getting results
async fn get_search_repos(
    thorium: &Thorium,
    cmd: &GetResults,
    params: &ResultGetParams,
) -> Result<(), Error> {
    // download results if any search parameters were given
    let opts = cmd.build_repo_opts()?;
    // build a cursor object
    let mut cursor = thorium.repos.list(&opts).await?;
    // crawl over this cursor until its exhausted
    loop {
        // convert our names list to just a list of SHA256's
        let repos = cursor
            .data
            .drain(..)
            .map(|line| line.url)
            .collect::<HashSet<String>>();
        // download the target samples results
        stream::iter(repos.iter())
            .for_each_concurrent(None, |repo| async move {
                match get_results(&thorium.repos, cmd, repo, params).await {
                    Ok(()) => GetLine::downloaded(repo),
                    Err(err) => GetLine::error(repo, &err),
                }
            })
            .await;
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            return Ok(());
        }
        // get the next page of data
        cursor.refill().await?;
    }
}

/// Downloads results
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `args` - The args given to the `thorctl` command
/// * `cmd` - The full result get command/args
/// * `workers` - The number of workers to use
async fn get(thorium: &Thorium, cmd: &GetResults, workers: usize) -> Result<(), Error> {
    // ensure the search configuration is valid
    cmd.validate_search()?;
    // print the download results header
    GetLine::header();
    // build the params for downloading results
    let params = ResultGetParams::default()
        .tools(cmd.tools.clone())
        .groups(cmd.results_groups.clone());
    // create the root directory
    if let Err(err) = tokio::fs::create_dir_all(&cmd.output).await {
        return Err(Error::new(format!(
            "Unable to create output directory: {err}"
        )));
    };
    // download the target samples results if any were given
    stream::iter(&cmd.files)
        .for_each_concurrent(Some(workers), |sha256| {
            let params_ref = &params;
            async move {
                match get_results(&thorium.files, cmd, sha256, params_ref).await {
                    Ok(()) => GetLine::downloaded(sha256),
                    Err(err) => GetLine::error(sha256, &err),
                }
            }
        })
        .await;
    // download the target repos results if any were given
    stream::iter(&cmd.repos)
        .for_each_concurrent(Some(workers), |repo| {
            let params_ref = &params;
            async move {
                match get_results(&thorium.repos, cmd, repo, params_ref).await {
                    Ok(()) => GetLine::downloaded(repo),
                    Err(err) => GetLine::error(repo, &err),
                }
            }
        })
        .await;
    if let Some(file_list) = &cmd.file_list {
        // if a file list path was given, get results for all SHA256's in the file
        let sha256s: HashSet<String> = utils::fs::lines_set_from_file(file_list).await?;
        stream::iter(sha256s.into_iter())
            .for_each_concurrent(Some(workers), |sha256| {
                let params_ref = &params;
                async move {
                    match get_results(&thorium.files, cmd, &sha256, params_ref).await {
                        Ok(()) => GetLine::downloaded(&sha256),
                        Err(err) => GetLine::error(&sha256, &err),
                    }
                }
            })
            .await;
    }
    if let Some(repo_list) = &cmd.repo_list {
        // if a repo list path was given, get results for all repos in the file
        let repos: HashSet<String> = utils::fs::lines_set_from_file(repo_list).await?;
        stream::iter(repos.into_iter())
            .for_each_concurrent(Some(workers), |repo| {
                let params_ref = &params;
                async move {
                    match get_results(&thorium.repos, cmd, &repo, params_ref).await {
                        Ok(()) => GetLine::downloaded(&repo),
                        Err(err) => GetLine::error(&repo, &err),
                    }
                }
            })
            .await;
    }
    if cmd.has_parameters() || cmd.apply_to_all() {
        // download results based off of search parameters if parameters are set
        // or if we want to download from all
        if !cmd.repos_only {
            // get results for files if repos_only hasn't been set
            get_search_files(thorium, cmd, &params).await?;
        }
        if cmd.include_repos || cmd.repos_only {
            // get results for repos if they should be included in the search
            get_search_repos(thorium, cmd, &params).await?;
        }
    }
    Ok(())
}

/// Uploads a single result to Thorium
async fn upload_helper(
    thorium: &Thorium,
    cmd: &UploadResults,
    sha256: String,
    entry: &DirEntry,
) -> Result<(), Error> {
    // build a path to our results file
    let mut path = entry.path();
    path.push(cmd.results.as_ref().unwrap_or(&"results".to_owned()));
    // build a list of results_files and children
    let mut results_files = Vec::default();
    // start recursively walking through this directory ignoring any hidden files
    for entry in WalkDir::new(entry.path())
        .into_iter()
        .filter_map(Result::ok)
    {
        // if this is a file then determine what to do with it
        if entry.file_type().is_file() {
            // build the on disk file object for this result file
            let on_disk = OnDiskFile::new(entry.path()).trim_prefix(entry.path());
            results_files.push(on_disk);
        }
    }
    // try to read in this results file
    let results_string = std::fs::read_to_string(path)?;
    // create our result request
    let results = OutputRequest::<Sample>::new(sha256, &cmd.tool, results_string, cmd.display_type)
        .files(results_files);
    thorium.files.create_result(results).await?;
    Ok(())
}

/// Crawl target directories and upload their results to Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full result upload command/args
pub async fn upload(thorium: &Thorium, cmd: &UploadResults) -> Result<(), Error> {
    // crawl over each path and upload them if they are new
    for path in &cmd.targets {
        // walk through the top level directories
        for entry in std::fs::read_dir(path)?.filter_map(Result::ok) {
            // try to extract this file name as a string
            if let Ok(name) = entry.file_name().into_string() {
                // skip anything thats not the same length as a sha256
                if name.len() != 64 {
                    continue;
                }
                // get the filetype of this entry
                let filetype = entry.file_type()?;
                // if this is a file then upload it as a singular result otherwise crawl the directory
                if filetype.is_file() {
                    panic!("NOT DONE YET");
                } else {
                    // this file is either a directory or a symlink so crawl it
                    upload_helper(thorium, cmd, name, &entry).await?;
                }
            }
        }
    }
    Ok(())
}

/// Handle all results commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The results command to execute
pub async fn handle(args: &Args, cmd: &Results) -> Result<(), Error> {
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
    // call the right results handler
    match cmd {
        Results::Get(cmd) => get(&thorium, cmd, args.workers).await,
        Results::Upload(cmd) => upload(&thorium, cmd).await,
    }
}
