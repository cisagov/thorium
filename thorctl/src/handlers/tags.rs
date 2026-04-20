use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use futures::stream::{self, StreamExt};
use http::StatusCode;
use owo_colors::OwoColorize;
use thorium::{
    Error, Thorium,
    models::{Repo, Sample, TagDeleteRequest, TagRequest},
};

use crate::args::{Args, SearchParameterized};
use crate::args::{
    Mode,
    tags::{AddTags, DeleteTags, GetTags, Tags},
};
use crate::utils;

/// Get a list of tags for a file or repo
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The get tags command that was run
async fn get(thorium: Thorium, cmd: &GetTags) -> Result<(), Error> {
    // check if we were given a sha256 or a repo
    let mode = Mode::try_from(&cmd.sha256_or_repo)?;
    // get tags depending on the mode
    let tags = match mode {
        Mode::File => thorium.files.get(&cmd.sha256_or_repo).await?.tags,
        Mode::Repo => thorium.repos.get(&cmd.sha256_or_repo).await?.tags,
    };
    // sort the tags by key and then by value
    let mut tags_sorted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (key, values) in tags {
        tags_sorted
            .entry(key)
            .or_default()
            .extend(values.into_keys());
    }
    // serialize to an output file or stdout in condensed/pretty mode
    match (&cmd.output, cmd.condensed) {
        (None, true) => {
            serde_json::to_writer(std::io::stdout(), &tags_sorted)?;
            println!();
        }
        (None, false) => {
            serde_json::to_writer_pretty(std::io::stdout(), &tags_sorted)?;
            println!();
        }
        (Some(output_path), true) => {
            let mut output_file = std::fs::File::create(output_path)?;
            serde_json::to_writer(&mut output_file, &tags_sorted)?;
            writeln!(&mut output_file)?;
        }
        (Some(output_path), false) => {
            let mut output_file = std::fs::File::create(output_path)?;
            serde_json::to_writer_pretty(&mut output_file, &tags_sorted)?;
            writeln!(&mut output_file)?;
        }
    }
    Ok(())
}

/// Attempt to condense raw tags to a single tag request
macro_rules! raw_tags_to_req {
    ($cmd:expr, $tags:ident, $groups:ident, $build:ident) => {
        $cmd.$tags.iter().try_fold(
            $build::default().groups($cmd.$groups.clone()),
            |mut tag_req, raw_tag| {
                // split this combined tag by our delimiter
                let mut split = raw_tag.split($cmd.delimiter);
                let key = split.next();
                let values: Vec<&str> = split.collect();
                match key {
                    Some(key) => {
                        if values.is_empty() {
                            Err(Error::new("Invalid tag: Tags must have at least one value"))
                        } else {
                            tag_req.add_values_ref(key, values);
                            Ok(tag_req)
                        }
                    }
                    None => Err(Error::new("Invalid tag: Tags must have a key")),
                }
            },
        )
    };
}

/// Tag a group of objects concurrently
macro_rules! tag_concurrent {
    ($thorium:expr, $req:expr, $objects:expr, $tag_type:ident) => {
        stream::iter($objects.iter()).for_each_concurrent(None, |obj| async {
            match $thorium.$tag_type.tag(obj, &$req).await {
                Ok(_) => LogLine::log(obj),
                Err(err) => LogLine::error(&err),
            }
        })
    };
}

/// Delete tags from a group of objects concurrently
macro_rules! delete_tags_concurrent {
    ($thorium:expr, $req:expr, $objects:expr, $tag_type:ident) => {
        stream::iter($objects.iter()).for_each_concurrent(None, |obj| async {
            match $thorium.$tag_type.delete_tags(obj, &$req).await {
                Ok(_) => LogLine::log(obj),
                Err(err) => LogLine::error(&err),
            }
        })
    };
}

struct LogLine {}

impl LogLine {
    /// Print this log lines header
    pub fn header() {
        println!("{:<5} | {:<72} | {:<30}", "CODE", "SAMPLE/REPO", "ERROR",);
        println!("{:-<6}+{:-<74}+{:-<30}", "", "", "");
    }

    /// Print a log that we completed a tag operation
    ///
    /// # Arguments
    ///
    /// * `object` - The sample/repo that was edited
    pub fn log(object: &str) {
        println!("{:<5} | {:<72} | -", 200.bright_green(), object);
    }

    /// Print an error line
    ///
    /// # Arguments
    ///
    /// * `err` - The error to print
    pub fn error(err: &Error) {
        println!(
            "{:<5} | {:<72} | {}",
            err.status()
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                .as_u16()
                .bright_red(),
            "-",
            err.msg().unwrap_or("An unknown error occurred".to_string()),
        );
    }
}

/// Add tags to files/repos
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The add tags command to execute
async fn add(thorium: Thorium, cmd: &AddTags) -> Result<(), Error> {
    // make sure the command has a valid search configuration
    cmd.validate_search()?;
    // condense all tags to a single tag request for samples and repos
    let file_tag_req: TagRequest<Sample> = raw_tags_to_req!(cmd, add_tags, add_groups, TagRequest)?;
    let repo_tag_req: TagRequest<Repo> = raw_tags_to_req!(cmd, add_tags, add_groups, TagRequest)?;
    // print a header
    LogLine::header();
    if !cmd.files.is_empty() {
        // send tag requests for explicitly given files
        tag_concurrent!(thorium, file_tag_req, cmd.files, files).await;
    }
    if !cmd.repos.is_empty() {
        // send tag requests for explicitly given repos
        tag_concurrent!(thorium, repo_tag_req, cmd.repos, repos).await;
    }
    // if tags or groups were set, commence a search
    if cmd.has_parameters() {
        if !cmd.repos_only {
            // search for files if repos_only hasn't been set
            let opts = cmd.build_file_opts()?;
            // build a cursor object
            let mut cursor = thorium.files.list(&opts).await?;
            // crawl over this cursor until its exhausted
            loop {
                // convert our lines to just a list of SHA256's
                let sha256s = cursor
                    .data
                    .drain(..)
                    .map(|line| line.sha256)
                    .collect::<Vec<String>>();
                // tag all retrieved files
                tag_concurrent!(thorium, file_tag_req, sha256s, files).await;
                // check if this cursor has been exhausted
                if cursor.exhausted() {
                    break;
                }
                // get the next page of data
                cursor.refill().await?;
            }
        }
        if cmd.include_repos || cmd.repos_only {
            // search for repos if they should be included in the search
            let opts = cmd.build_repo_opts()?;
            let mut cursor = thorium.repos.list(&opts).await?;
            loop {
                let repo_urls = cursor
                    .data
                    .drain(..)
                    .map(|line| line.url)
                    .collect::<Vec<String>>();
                // tag all retrieved repos
                tag_concurrent!(thorium, repo_tag_req, repo_urls, repos).await;
                // check if this cursor has been exhausted
                if cursor.exhausted() {
                    break;
                }
                // get the next page of data
                cursor.refill().await?;
            }
        }
    }
    Ok(())
}

/// Delete tags from files/repos
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The delete tags command to execute
async fn delete(thorium: Thorium, cmd: &DeleteTags) -> Result<(), Error> {
    // make sure the command has a valid search configuration
    cmd.validate_search()?;
    // condense all tags to a single repo tag request
    let repo_tag_req: TagDeleteRequest<Repo> =
        raw_tags_to_req!(cmd, delete_tags, delete_groups, TagDeleteRequest)?;
    let file_tag_req: TagDeleteRequest<Sample> =
        raw_tags_to_req!(cmd, delete_tags, delete_groups, TagDeleteRequest)?;
    // print a header
    LogLine::header();
    if !cmd.files.is_empty() {
        // send tag requests for explicitly given files
        delete_tags_concurrent!(thorium, file_tag_req, cmd.files, files).await;
    }
    if !cmd.repos.is_empty() {
        // send tag requests for explicitly given repos
        delete_tags_concurrent!(thorium, repo_tag_req, cmd.repos, repos).await;
    }
    // if tags or groups were set, commence a search
    if cmd.has_parameters() {
        if !cmd.repos_only {
            // search for files if repos_only hasn't been set
            let opts = cmd.build_file_opts()?;
            // build a cursor object
            let mut cursor = thorium.files.list(&opts).await?;
            // crawl over this cursor until its exhausted
            loop {
                // convert our lines to just a list of SHA256's
                let sha256s = cursor
                    .data
                    .drain(..)
                    .map(|line| line.sha256)
                    .collect::<Vec<String>>();
                // tag all retrieved files
                delete_tags_concurrent!(thorium, file_tag_req, sha256s, files).await;
                // check if this cursor has been exhausted
                if cursor.exhausted() {
                    break;
                }
                // get the next page of data
                cursor.refill().await?;
            }
        }
        if cmd.include_repos || cmd.repos_only {
            // search for repos if they should be included in the search
            let opts = cmd.build_repo_opts()?;
            let mut cursor = thorium.repos.list(&opts).await?;
            loop {
                let repo_urls = cursor
                    .data
                    .drain(..)
                    .map(|line| line.url)
                    .collect::<Vec<String>>();
                // tag all retrieved repos
                delete_tags_concurrent!(thorium, repo_tag_req, repo_urls, repos).await;
                // check if this cursor has been exhausted
                if cursor.exhausted() {
                    break;
                }
                // get the next page of data
                cursor.refill().await?;
            }
        }
    }
    Ok(())
}

/// Handle all tags commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The tags command to execute
pub async fn handle(args: &Args, cmd: &Tags) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // call the right tags handler
    match cmd {
        Tags::Get(cmd) => get(thorium, cmd).await,
        Tags::Add(cmd) => add(thorium, cmd).await,
        Tags::Delete(cmd) => delete(thorium, cmd).await,
    }
}
