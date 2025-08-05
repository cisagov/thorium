use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use colored::Colorize;
use futures::stream::{self, StreamExt};
use futures::TryStreamExt;
use itertools::Itertools;
use owo_colors::OwoColorize;
use thorium::models::{Reaction, ReactionListParams, ReactionStatus};
use thorium::Thorium;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use super::progress::Bar;
use super::update;
use crate::args::reactions::{
    DescribeReactions, GetReactions, LogsReactions, ReactionTarget, Reactions,
};
use crate::args::{self, Args, DescribeCommand, SearchParameterized};
use crate::utils;
use crate::Error;

pub mod create;

use create::create;

/// prints out a single info line
macro_rules! info_print {
    ($code:expr, $pipeline:expr, $status:expr, $samples:expr, $creator:expr, $id:expr, $msg:expr) => {
        println!(
            "{:<4} | {:<25} | {:<9} | {:<64} | {:<15} | {:<28} | {:<32}",
            $code, $pipeline, $status, $samples, $creator, $id, $msg
        )
    };
}

/// A single line for a reaction info log
struct InfoLine;

impl InfoLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{} | {:<25} | {:<9} | {:<64} | {:<15} | {:<36} | {:<32}",
            "CODE", "PIPELINE", "STATUS", "SAMPLES", "CREATOR", "ID", "MESSAGE"
        );
        println!(
            "{:-<5}+{:-<27}+{:-<11}+{:-<66}+{:-<17}+{:-<38}+{:-<34}",
            "", "", "", "", "", "", ""
        );
    }

    /// Print an ok log line for basic info on a reaction
    ///
    /// # Arguments
    ///
    /// * `reaction` - The reaction to print basic info for
    pub fn info(reaction: &Reaction) {
        // print either the sample if its a single sample or the number of samples
        let samples = match reaction.samples.len() {
            1 => reaction.samples[0].clone(),
            len if len > 1 => format!("{len} samples"),
            _ => "-".to_owned(),
        };
        // get the right color of text for the status
        // formatting enums seems to break fmt's ability to right align text
        // so use manual str's instead
        match reaction.status {
            ReactionStatus::Created => info_print!(
                "200".bright_green(),
                reaction.pipeline,
                "Created".bright_purple(),
                samples,
                reaction.creator,
                reaction.id,
                "-"
            ),
            ReactionStatus::Started => info_print!(
                "200".bright_green(),
                reaction.pipeline,
                "Started".bright_blue(),
                samples,
                reaction.creator,
                reaction.id,
                "-"
            ),
            ReactionStatus::Completed => info_print!(
                "200".bright_green(),
                reaction.pipeline,
                "Completed".bright_green(),
                samples,
                reaction.creator,
                reaction.id,
                "-"
            ),
            ReactionStatus::Failed => info_print!(
                "200".bright_green(),
                reaction.pipeline,
                "Failed".bright_red(),
                samples,
                reaction.creator,
                reaction.id,
                "-"
            ),
        }
    }

    /// Print an error log line for a Thorium client error
    ///
    /// # Arguments
    ///
    /// * `id` - The reaction id we tried to get info about
    /// * `err` - The error to print
    pub fn error(id: &Uuid, err: &thorium::Error) {
        // get the error message if one was set
        let msg = err.msg().unwrap_or_else(|| "-".to_owned());
        // show either the reqwest body error or the hyper error
        match err.status() {
            // we have a status so well return the code and body as a message
            Some(code) => info_print!(code.as_str().bright_red(), "-", "-", "-", "-", id, msg),
            // no status code is present so just use '-' painted bright red
            None => info_print!("-".bright_red(), "", "-", "-", "-", id, msg),
        };
    }
}

/// crawls reactions for a specific pipeline
///
/// # Arguments
///
/// * `client` - The Thorium client to use
/// * `pipe` - The pipeline we are restricting our cursor too
/// * `cmd` - The full command for this operation
macro_rules! crawl_pipeline {
    ($client:expr, $pipe:expr, $cmd:expr) => {
        // build a cursor object
        $client
            .reactions
            .list(&$cmd.group, $pipe)
            .limit($cmd.limit)
            .details()
            .exec()
            .await
    };
}

/// crawls reactions for a specific tag
///
/// # Arguments
///
/// * `client` - The Thorium client to use
/// * `tag` - The tag we are restricting our cursor too
/// * `cmd` - The full command for this operation
macro_rules! crawl_tag {
    ($client:expr, $tag:expr, $cmd:expr) => {
        // build a cursor object
        $client
            .reactions
            .list_tag(&$cmd.group, $tag)
            .limit($cmd.limit)
            .details()
            .exec()
            .await
    };
}

/// crawls reactions for a specific pipeline and status
///
/// # Arguments
///
/// * `client` - The Thorium client to use
/// * `pipe` - The pipe we are restricting our cursor too
/// * `status` - The status to restrict our cursor too
/// * `cmd` - The full command for this operation
macro_rules! crawl_status {
    ($client:expr, $pipe:expr, $status:expr, $cmd:expr) => {{
        // try to cast our status string to a ReactionStatus
        let Ok(status) = $status.parse() else {
            panic!("status must be one of Created, Started, Completed, Failed")
        };
        // build a cursor object
        $client
            .reactions
            .list_status(&$cmd.group, $pipe, &status)
            .limit($cmd.limit)
            .details()
            .exec()
            .await
    }};
}

/// Print information on specific reactions
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full get command/args
async fn info_specific(thorium: &Thorium, cmd: &GetReactions) -> Result<(), Error> {
    // print our info line header
    InfoLine::header();
    // crawl over all reaction IDs and get info on them
    stream::iter(&cmd.targets)
        .map(|target| async move {
            match thorium.reactions.get(&cmd.group, target).await {
                Ok(info) => InfoLine::info(&info),
                Err(err) => InfoLine::error(target, &err),
            };
            Ok(())
        })
        .buffer_unordered(25)
        .collect::<Vec<Result<(), Error>>>()
        .await;
    Ok(())
}

/// Deletes a specific reaction by id
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full get command/args
async fn delete_specific(thorium: &Thorium, cmd: &GetReactions) -> Result<(), Error> {
    // print our info line header
    InfoLine::header();
    // crawl over all reaction IDs and get info on them
    stream::iter(&cmd.targets)
        .map(|target| async move {
            // get this reactions info
            match thorium.reactions.get(&cmd.group, target).await {
                // delete this reaction
                Ok(info) => match thorium.reactions.delete(&info.group, &info.id).await {
                    Ok(_) => InfoLine::info(&info),
                    Err(err) => InfoLine::error(target, &err),
                },
                Err(err) => InfoLine::error(target, &err),
            };
            Ok(())
        })
        .buffer_unordered(25)
        .collect::<Vec<Result<(), Error>>>()
        .await;
    Ok(())
}

/// Calls the right get info method
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full get command/args
async fn get(thorium: &Thorium, cmd: &GetReactions) -> Result<(), Error> {
    // determine the correct action to take based on the args specified
    let mut cursor = match (cmd.targets.is_empty(), &cmd.pipeline, &cmd.status, &cmd.tag) {
        // get info on specific reactions by id
        (false, None, None, None) => return info_specific(thorium, cmd).await,
        // get info on reactions for a specific pipeline
        (true, Some(pipe), None, None) => crawl_pipeline!(thorium, pipe, cmd)?,
        // get info on reactions for a specific pipeline
        (true, Some(pipe), Some(status), None) => crawl_status!(thorium, pipe, status, cmd)?,
        // get info on reactions for a specific tag
        (true, None, None, Some(tag)) => crawl_tag!(thorium, tag, cmd)?,
        _ => panic!("UNKNOWN ARG COMBO"),
    };
    // print our header for this return
    InfoLine::header();
    // crawl over this cursor until its exhausted
    loop {
        // print info for each reaction we pulled
        for reaction in &cursor.details {
            InfoLine::info(reaction);
        }
        // check if this cursor has been exhausted
        if cursor.exhausted {
            break;
        }
        // get the next page of data
        cursor.next().await?;
    }
    Ok(())
}

/// Describe reactions by displaying/saving all of their JSON-formatted details
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The describe reactions command to execute
async fn describe(thorium: &Thorium, cmd: &DescribeReactions) -> Result<(), Error> {
    cmd.describe(thorium).await
}

/// Map error to message and return
// TODO: move to utils if this proves reuseable
macro_rules! error_and_return {
    ($func:expr, $msg:expr) => {
        $func.map_err(|err| Error::new(format!("{}: {}", $msg, err)))
    };
}

/// Convert a single status update to a string
macro_rules! status_update_to_string {
    ($status:expr) => {
        if let Some(msg) = &$status.msg {
            format!("[{}] {}: {}", $status.timestamp, $status.action, msg)
        } else {
            format!("[{}] {}", $status.timestamp, $status.action)
        }
    };
}

/// Write a reaction's status logs to disk
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `reaction` - The reaction to write status logs for
/// * `reaction_subdir` - The reaction's subdirectory
async fn write_status_logs(
    thorium: &Thorium,
    reaction: &Reaction,
    reaction_subdir: &Path,
) -> Result<(), Error> {
    // get the reaction's status logs
    let status_logs = error_and_return!(
        thorium
            .reactions
            .status_logs(&reaction.group, &reaction.id)
            .await,
        format!(
            "Unable to retrieve reaction status logs for reaction '{}'",
            reaction.id
        )
    )?;
    // create a file for the status logs
    let status_file_path = reaction_subdir.join(PathBuf::from(reaction.id.to_string()));
    let mut status_file = error_and_return!(
        tokio::fs::File::create(&status_file_path).await,
        format!(
            "Unable to create file '{}'",
            status_file_path.to_string_lossy()
        )
    )?;
    // write the status logs
    for log in status_logs {
        // write the log to a temporary buf and then to the file; this is necessary because
        // tokio::fs::File does not implement std::io::Write needed for writeln!
        let mut buf: Vec<u8> = Vec::new();
        error_and_return!(
            writeln!(buf, "{}", status_update_to_string!(log)),
            format!(
                "Error writing status logs to '{}'",
                status_file_path.to_string_lossy()
            )
        )?;
        error_and_return!(
            status_file.write_all(&buf).await,
            format!(
                "Error writing status logs to '{}'",
                status_file_path.to_string_lossy()
            )
        )?;
    }
    Ok(())
}

/// Write logs for a particular reaction in its own subdirectory
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `reaction` - The reaction to write logs for
/// * `output` - The base output path to write logs to
/// * `params` - The params to set when retrieving logs
/// * `progress` - The progress bar
async fn write_reaction_logs(
    thorium: &Thorium,
    reaction: Reaction,
    output: &Path,
    params: &ReactionListParams,
    progress: &Bar,
) -> Result<(), Error> {
    // retrieve information about the reaction's pipeline
    let pipeline = error_and_return!(
        thorium
            .pipelines
            .get(&reaction.group, &reaction.pipeline)
            .await,
        format!("Unable to retrieve pipeline for reaction '{}'", reaction.id)
    )?;
    // create a subdirectory for the reaction underneath the pipeline
    let reaction_subdir = PathBuf::from(output)
        .join(&pipeline.name)
        .join(reaction.id.to_string());
    error_and_return!(
        tokio::fs::create_dir_all(&reaction_subdir).await,
        format!(
            "Unable to create directory '{}'",
            reaction_subdir.to_string_lossy()
        )
    )?;
    // write the reaction's status logs
    write_status_logs(thorium, &reaction, &reaction_subdir).await?;
    // write logs for every stage
    for stage in pipeline.order.iter().flatten().unique() {
        let logs = error_and_return!(
            thorium
                .reactions
                .logs(&reaction.group, &reaction.id, stage, params)
                .await,
            format!(
                "Unable to retrieve logs for stage '{}' in pipeline '{}' in reaction '{}'",
                stage, pipeline.name, reaction.id
            )
        )?;
        if logs.logs.is_empty() {
            // warn that a stage's logs were empty
            progress.info_anonymous(format!(
                "Stage '{}' in reaction '{}' has no logs",
                stage.bright_yellow(),
                reaction.id.bright_green()
            ));
            // skip this stage
            continue;
        }
        let file_path = reaction_subdir.join(stage);
        // only create the file if it's new
        let mut file = match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path)
            .await
        {
            Ok(file) => file,
            Err(err) => match err.kind() {
                // skip this stage if it already exists in case another task already created it
                std::io::ErrorKind::AlreadyExists => continue,
                _ => {
                    return Err(Error::new(format!(
                        "Unable to create file '{}': {err}",
                        file_path.to_string_lossy()
                    )))
                }
            },
        };
        for log in &logs.logs {
            // write the line to a temporary buf and then to the file; this is necessary because
            // tokio::fs::File does not implement std::io::Write needed for writeln!
            let mut buf: Vec<u8> = Vec::new();
            error_and_return!(
                writeln!(buf, "{log}"),
                format!("Error writing logs to '{}'", file_path.to_string_lossy())
            )?;
            error_and_return!(
                file.write_all(&buf).await,
                format!("Error writing logs to '{}'", file_path.to_string_lossy())
            )?;
        }
    }
    Ok(())
}

/// Write logs for a particular reaction to stdout
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `reaction` - The reaction to write logs for
/// * `params` - The params to set when retrieving logs
async fn write_reaction_logs_stdout(
    thorium: &Thorium,
    reaction: Reaction,
    params: &ReactionListParams,
) -> Result<(), Error> {
    // retrieve information about the reaction's pipeline
    let pipeline = error_and_return!(
        thorium
            .pipelines
            .get(&reaction.group, &reaction.pipeline)
            .await,
        format!("Unable to retrieve pipeline for reaction '{}'", reaction.id)
    )?;
    let stage_iter = pipeline.order.into_iter().flatten().unique();
    let stage_logs: HashMap<String, Vec<String>> = futures::stream::iter(stage_iter.clone())
        .map(|stage| async {
            let logs = thorium
                .reactions
                .logs(&reaction.group, &reaction.id, &stage, params)
                .await;
            match logs {
                Ok(logs) => Ok((stage, logs.logs)),
                Err(err) => Err(Error::new(format!(
                    "Unable to retrieve logs for stage '{}' in pipeline '{}' in reaction '{}': {}",
                    stage, pipeline.name, reaction.id, err
                ))),
            }
        })
        .buffered(100)
        .try_collect()
        .await?;
    // get the reaction's status logs
    let status_logs = error_and_return!(
        thorium
            .reactions
            .status_logs(&reaction.group, &reaction.id)
            .await,
        format!(
            "Unable to retrieve reaction status logs for reaction '{}'",
            reaction.id
        )
    )?;
    // print reaction header
    println!(
        "{}",
        format!("Reaction '{}' (Pipeline '{}')", reaction.id, pipeline.name).bright_green()
    );
    // print each status log to stdout
    for log in status_logs {
        println!("    {}", status_update_to_string!(&log));
    }
    // write the reaction's stage logs to stdout
    for stage in stage_iter {
        // print stage header
        println!("{}", format!("Stage '{stage}'").bright_yellow());
        for log in stage_logs.get(&stage).unwrap() {
            // print the log to stdout
            println!("{log}");
        }
        // print an extra newline
        println!();
    }
    Ok(())
}

/// Write logs for reactions given in positional arguments
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `reactions` - The reactions to write logs for
/// * `output` - The base output path to write logs to
/// * `params` - The params to use when retrieving Reaction logs
/// * `progress` - The progress bar to track progress
async fn logs_positionals(
    thorium: &Thorium,
    reactions: &[String],
    output: &Option<PathBuf>,
    params: &ReactionListParams,
    progress: &Option<Bar>,
) -> Result<(), Error> {
    // concurrently retrieve reactions and write logs for each reaction
    futures::stream::iter(reactions)
        .map(Ok)
        .try_for_each_concurrent(None, |reaction| async move {
            // parse a reaction target from the arg
            let reaction_target = ReactionTarget::try_from(reaction)?;
            // retrieve the reaction
            let reaction = reaction_target.get_reaction(thorium).await?;
            // write reaction logs to a file or to stdout
            match (output, progress) {
                (Some(output), Some(progress)) => {
                    write_reaction_logs(thorium, reaction, output, params, progress).await?;
                }
                (None, None) => write_reaction_logs_stdout(thorium, reaction, params).await?,
                // output and progress are always either both Some or both None
                _ => (),
            }
            // increment progress if we're tracking it
            progress.as_ref().inspect(|p| p.inc(1));
            Ok(())
        })
        .await
}

/// Write logs for reactions given in a reaction list file
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `list_file` - The reaction list file
/// * `output` - The base output path to write logs to
/// * `params` - The params to use when retrieving Reaction logs
/// * `progress` - The progress bar to track progress
async fn logs_list(
    thorium: &Thorium,
    list_file: &Path,
    output: &Option<PathBuf>,
    params: &ReactionListParams,
    progress: &Option<Bar>,
) -> Result<(), Error> {
    // open the reaction list file
    let file = error_and_return!(
        std::fs::File::open(list_file),
        format!(
            "Unable to open list file at '{}'",
            list_file.to_string_lossy(),
        )
    )?;
    let reader = std::io::BufReader::new(file);
    // concurrently retrieve reactions for each line and write logs for each reaction
    futures::stream::iter(reader.lines())
        .map(Ok)
        .try_for_each_concurrent(None, |line| async {
            let line = error_and_return!(
                line,
                format!("Error reading '{}'", list_file.to_string_lossy())
            )?;
            // parse a reaction target from the line
            let reaction_target = ReactionTarget::try_from(line)?;
            // retrieve the reaction
            let reaction = reaction_target.get_reaction(thorium).await?;
            // write reaction logs to a file or to stdout
            match (output, progress) {
                (Some(output), Some(progress)) => {
                    write_reaction_logs(thorium, reaction, output, params, progress).await?;
                }
                (None, None) => write_reaction_logs_stdout(thorium, reaction, params).await?,
                // output and progress are always either both Some or both None
                _ => (),
            }
            // increment progress if we're tracking it
            progress.as_ref().inspect(|p| p.inc(1));
            Ok(())
        })
        .await
}

/// Get and write logs from all reactions found by the given cursor
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cursor` - The cursor of reactions to retrieve logs for
/// * `output` - The output directory in which to store all of the logs
/// * `params` - The params to use when retrieving Reaction logs
/// * `progress` - The progress bar tracking how many logs have been written
async fn write_cursor_logs(
    thorium: &Thorium,
    mut cursor: thorium::client::Cursor<Reaction>,
    output: &Option<PathBuf>,
    params: &ReactionListParams,
    progress: &Option<Bar>,
) -> Result<(), Error> {
    loop {
        // concurrently retrieve and write logs all reactions in cursor
        futures::stream::iter(cursor.details.drain(..))
            .map(Result::<Reaction, Error>::Ok)
            .try_for_each_concurrent(None, |reaction| async move {
                // write reaction logs to a file or to stdout
                match (output, progress) {
                    (Some(output), Some(progress)) => {
                        write_reaction_logs(thorium, reaction, output, params, progress).await?;
                    }
                    (None, None) => write_reaction_logs_stdout(thorium, reaction, params).await?,
                    // output and progress are always either both Some or both None
                    _ => (),
                }
                // increment progress if we're tracking it
                progress.as_ref().inspect(|p| p.inc(1));
                Ok(())
            })
            .await?;
        // exit the loop if the cursor is exhausted
        if cursor.exhausted {
            break;
        }
        // get more reactions
        cursor.next().await?;
    }
    Ok(())
}

/// Write logs for reactions by searching with the given cursors
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The reaction logs command that was run
/// * `output` - The output directory to write logs to
/// * `params` - The params to use when retrieving Reaction logs
/// * `progress` - The progress bar to track progress
async fn logs_search(
    thorium: &Thorium,
    cmd: &LogsReactions,
    output: &Option<PathBuf>,
    params: &ReactionListParams,
    progress: &Option<Bar>,
) -> Result<(), Error> {
    // generate reaction cursors based on the given command
    let cursors = args::reactions::params_to_cursors(thorium, cmd, &cmd.pipelines).await?;
    // crawl reactions for each cursor and write logs for all stages
    futures::stream::iter(cursors)
        .map(Ok)
        .try_for_each_concurrent(None, |cursor| {
            write_cursor_logs(thorium, cursor, output, params, progress)
        })
        .await
}

/// Retrieve reaction logs
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The reaction logs command to execute
async fn logs(thorium: &Thorium, cmd: &LogsReactions) -> Result<(), Error> {
    cmd.validate_search()?;
    // create a progress bar if not printing to stdout
    let progress = cmd
        .output
        .is_some()
        .then_some(Bar::new_unbounded("Writing logs", ""));
    // create params for listing logs (specifically containing the max number of log lines)
    let params = ReactionListParams::default().limit(cmd.log_limit);
    // write logs for reactions in positional arguments
    logs_positionals(thorium, &cmd.reactions, &cmd.output, &params, &progress).await?;
    if let Some(list_file) = &cmd.reaction_list {
        // write logs for reactions in a list file if one was given
        logs_list(thorium, list_file, &cmd.output, &params, &progress).await?;
    }
    if cmd.has_parameters() {
        // write logs with a search if the command has parameters
        logs_search(thorium, cmd, &cmd.output, &params, &progress).await?;
    }
    // finish the progress bar if we have one
    progress.inspect(|p| p.finish_with_message("✅"));
    Ok(())
}

/// Calls the right delete method
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full get command/args
async fn delete(thorium: &Thorium, cmd: &GetReactions) -> Result<(), Error> {
    // determine the correct action to take based on the args specified
    let mut cursor = match (cmd.targets.is_empty(), &cmd.pipeline, &cmd.status, &cmd.tag) {
        // get info on specific reactions by id
        (false, None, None, None) => return delete_specific(thorium, cmd).await,
        // get info on reactions for a specific pipeline
        (true, Some(pipe), None, None) => crawl_pipeline!(thorium, pipe, cmd)?,
        // get info on reactions for a specific pipeline
        (true, Some(pipe), Some(status), None) => crawl_status!(thorium, pipe, status, cmd)?,
        // get info on reactions for a specific tag
        (true, None, None, Some(tag)) => crawl_tag!(thorium, tag, cmd)?,
        _ => panic!("UNKNOWN ARG COMBO"),
    };
    // print our header for this return
    InfoLine::header();
    // crawl over this cursor until its exhausted
    loop {
        // delete all reactions that we have pulled
        for reaction in &cursor.details {
            thorium
                .reactions
                .delete(&reaction.group, &reaction.id)
                .await?;
            // print our delete log
            InfoLine::info(reaction);
        }
        // check if this cursor has been exhausted
        if cursor.exhausted {
            break;
        }
        // get the next page of data
        cursor.next().await?;
    }
    Ok(())
}

/// Handle all reactions commands or print reactions docs
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The reactions command to execute
pub async fn handle(args: &Args, cmd: &Reactions) -> Result<(), Error> {
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
    // call the right reactions handler
    match cmd {
        Reactions::Get(cmd) => get(&thorium, cmd).await,
        Reactions::Describe(cmd) => describe(&thorium, cmd).await,
        Reactions::Logs(cmd) => logs(&thorium, cmd).await,
        Reactions::Delete(cmd) => delete(&thorium, cmd).await,
        Reactions::Create(cmd) => create(thorium, cmd).await,
    }
}
