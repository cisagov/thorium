use owo_colors::{OwoColorize, Rgb};
use rand::seq::SliceRandom;
use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use thorium::{
    client::{LogsCursor, ResultsClient},
    models::{Output, Pipeline, ReactionRequest, ReactionStatus, ResultGetParams},
    Thorium,
};
use tokio::time::{sleep, Duration};
use tokio::{io::AsyncWriteExt, sync::mpsc};
use uuid::Uuid;

use super::update;
use crate::args::repos::RepoTarget;
use crate::args::run::Run;
use crate::args::{Args, Mode};
use crate::utils;
use crate::Error;

/// The rate in seconds to poll the Thorium api for status updates
const REFRESH_RATE: Duration = Duration::from_secs(1);

/// Watch the status of a reaction and notify listeners when
/// the reaction is complete
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `group` - The group in which the reaction is being run
/// * `id` - The UUID of the running reaction
/// * `complete` - Signals that the reaction is complete
async fn watch_reaction_complete(
    thorium: Arc<Thorium>,
    group: String,
    id: Uuid,
    complete: Arc<AtomicBool>,
) -> Result<(), Error> {
    // Wait until the reaction is complete
    while !matches!(
        thorium.reactions.get(&group, &id).await?.status,
        ReactionStatus::Completed | ReactionStatus::Failed
    ) {
        sleep(REFRESH_RATE).await;
    }
    // Notify the listeners
    complete.store(true, Ordering::Relaxed);
    Ok(())
}

/// Async task that forwards logs for a stage to a channel
///
/// # Arguments
///
/// * `stage` - The stage of the reaction from which to forward logs
/// * `cursor` - The logs cursor
/// * `log_tx` - The channel through which to forward the logs
/// * `complete` - Whether the reaction is complete
async fn forward_logs(
    stage: String,
    mut cursor: LogsCursor,
    log_tx: mpsc::Sender<(String, String)>,
    complete: Arc<AtomicBool>,
) -> Result<(), Error> {
    // forward logs while the cursor is not exhausted and the reaction is incomplete
    loop {
        // move the cursor forward
        cursor.next().await?;
        // forward the log lines
        for line in &cursor.logs.logs {
            if log_tx
                .send((stage.clone(), line.to_string()))
                .await
                .is_err()
            {
                break;
            }
        }
        // if we are out of logs and the reaction is complete, exit the loop
        if cursor.exhausted && complete.load(Ordering::Relaxed) {
            break;
        }
        sleep(REFRESH_RATE).await;
    }
    Ok(())
}

/// Outputs anything on the channel to the console and
/// finishes once all senders have been closed.
///
/// Returns a map of stages/tools to colors for future logging.
///
/// # Arguments
///
/// * `log_rx` - The channel to poll for log information
async fn log(mut log_rx: mpsc::Receiver<(String, String)>) -> HashMap<String, Rgb> {
    // create a vector of shuffled colors to highlight different stages in a reaction
    let colors = {
        let mut colors = vec![
            Rgb(255, 0, 0),   // red
            Rgb(255, 128, 0), // orange
            Rgb(255, 255, 0), // yellow
            Rgb(128, 255, 0), // lime green
            Rgb(0, 255, 0),   // green
            Rgb(0, 255, 128), // seafoam green
            Rgb(0, 255, 255), // baby blue
            Rgb(0, 128, 255), // light blue
            Rgb(0, 0, 255),   // blue
            Rgb(128, 0, 255), // purple
            Rgb(255, 0, 255), // light purple
            Rgb(255, 0, 128), // pink
        ];
        colors.shuffle(&mut rand::rng());
        colors
    };
    let mut index = 0;
    let mut stage_colors = HashMap::new();
    // poll for logs from the receiver
    while let Some((stage, line)) = log_rx.recv().await {
        let color = stage_colors.entry(stage.clone()).or_insert({
            // pick a new color
            let new_color = colors[index];
            index += 1;
            index %= colors.len();
            new_color
        });
        // print the log
        let label = format!("{stage}:");
        println!("\r{} {}", label.color(*color).bold(), line);
    }
    stage_colors
}

/// Write all attachments connected to a result to disk
///
/// # Arguments
///
/// * `attachments` - The attachments to retrieve and write to disk
/// * `result_id` - The id of the result
/// * `tool` - The tool that created the result
/// * `sha256_or_repo` - The sha256 of the file or the repo URL from which the result was generated
/// * `out_path` - The output path to write the attachments to
/// * `client` - The Thorium client
/// * `run_mode` - The mode our run command is in
async fn write_all_attachments(
    attachments: &[String],
    result_id: &Uuid,
    tool: &str,
    sha256_or_repo: &str,
    out_path: &Path,
    client: &Thorium,
    run_mode: &Mode,
) -> Result<(), Error> {
    for attachment_name in attachments {
        // download this result file
        let attachment = match run_mode {
            Mode::File => {
                client
                    .files
                    .download_result_file(sha256_or_repo, tool, result_id, attachment_name)
                    .await?
            }
            Mode::Repo => {
                client
                    .repos
                    .download_result_file(sha256_or_repo, tool, result_id, attachment_name)
                    .await?
            }
        };
        let attachment_path = PathBuf::from(attachment_name);
        if let Some(attachment_parent) = attachment_path.parent() {
            tokio::fs::create_dir_all(out_path.join(attachment_parent)).await?;
        }
        // build the path to write this result file off to disk at
        let target_path = out_path.join(attachment_name);
        // create a file handle for this file
        let mut file = tokio::fs::File::create(&target_path).await?;
        // write our response body to disk
        file.write_all(&attachment.data[..]).await?;
    }
    Ok(())
}

/// Write a single result to disk
///
/// # Arguments
///
/// * `output` - The [`Output`] object containing the result
/// * `tool` - The tool that created the result
/// * `sha256_or_repo` - The sha256 of the file or the repo URL from which the result was generated
/// * `base_out_path` - The base output path to write the result to
/// * `client` - The Thorium client
/// * `run_mode` - The mode our run command is in
async fn write_result(
    output: Output,
    tool: String,
    sha256_or_repo: &str,
    base_out_path: &Path,
    client: &Thorium,
    run_mode: &Mode,
) -> Result<(), Error> {
    let mut out_path = base_out_path.join(&tool);
    // write the result attachments to disk if they exist
    if !output.files.is_empty() {
        // create a base directory to contain all of the attachments
        tokio::fs::create_dir_all(&out_path).await?;
        // retrieve and write all attachments to disk
        write_all_attachments(
            &output.files,
            &output.id,
            &tool,
            sha256_or_repo,
            &out_path,
            client,
            run_mode,
        )
        .await?;
        // push an additional filename to write the actual contents of the result to
        out_path.push(&tool);
    }
    // write the result contents to disk
    let file = std::fs::File::create(&out_path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &output.result)?;
    writer.flush()?;
    Ok(())
}

/// Write the results of a reaction to disk
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The run subcommand
/// * `pipeline` - The pipeline of the reaction
/// * `tool_colors` - A map of tools to colors created while logging
/// * `run_mode` - The mode our run command is in
async fn write_results(
    thorium: &Thorium,
    cmd: &Run,
    pipeline: Pipeline,
    tool_colors: HashMap<String, Rgb>,
    run_mode: &Mode,
) -> Result<(), Error> {
    // retrieve the file/repo's results
    let output_map = match run_mode {
        Mode::File => {
            thorium
                .files
                .get_results(
                    &cmd.sha256_or_repo,
                    &ResultGetParams::default()
                        .tools(pipeline.order.into_iter().flatten())
                        .hidden(),
                )
                .await?
        }
        Mode::Repo => {
            // parse the URL from the repo
            let repo_url = cmd
                .sha256_or_repo
                .split(':')
                .next()
                // this should never occur because we've already run a reaction with the repo
                .ok_or(Error::new("The repo URL is empty!"))?;
            thorium
                .repos
                .get_results(
                    // provide the URL with no commitish
                    repo_url,
                    &ResultGetParams::default()
                        .tools(pipeline.order.into_iter().flatten())
                        .hidden(),
                )
                .await?
        }
    };
    if !output_map.results.is_empty() {
        println!("Retrieving results...");
        // generate a base output path if one wasn't given
        let base_out_path = match run_mode {
            Mode::File => cmd.output.clone().unwrap_or(PathBuf::from(format!(
                "{}_{}",
                &cmd.sha256_or_repo, &cmd.pipeline
            ))),
            Mode::Repo => {
                let repo = cmd.sha256_or_repo.split('/').last().unwrap_or_default();
                cmd.output
                    .clone()
                    .unwrap_or(PathBuf::from(format!("{}_{}", repo, &cmd.pipeline)))
            }
        };
        tokio::fs::create_dir_all(&base_out_path).await?;
        // map the results to futures that write the results to disk concurrently
        futures::future::join_all(
            output_map
                .results
                .into_iter()
                // map the results lists to only the most recent result
                .filter_map(|(tool, outputs)| {
                    outputs.first().map(|output| (tool, output.to_owned()))
                })
                // map to future that will write the result
                .map(|(tool, output)| async {
                    let color = tool_colors.get(&tool).unwrap_or(&Rgb(0, 255, 0));
                    let tool_name = tool.clone();
                    match write_result(
                        output,
                        tool,
                        &cmd.sha256_or_repo,
                        &base_out_path,
                        thorium,
                        run_mode,
                    )
                    .await
                    {
                        Ok(()) => println!(
                            "Successfully retrieved results from tool {}",
                            &tool_name.color(*color).bold()
                        ),
                        Err(err) => println!(
                            "An error occurred retrieving results from tool {}: {}",
                            &tool_name.color(*color).bold(),
                            err.msg()
                                .unwrap_or("An unknown error has occurred".to_string())
                        ),
                    }
                }),
        )
        .await;
        println!(
            "Finished retrieving results! Results saved to \"{}\"",
            &base_out_path.to_string_lossy()
        );
    }
    Ok(())
}

/// Run the command on Thorium and stream back the logs to the console
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The run command to execute
async fn run(thorium: Arc<Thorium>, cmd: &Run) -> Result<(), Error> {
    // find the pipeline's group if none was given
    let group = if let Some(group) = &cmd.group {
        group.clone()
    } else {
        utils::pipelines::find_pipeline_group(&thorium, &cmd.pipeline).await?
    };
    // generate a request to create a reaction
    let mut req = ReactionRequest::new(group.clone(), cmd.pipeline.clone()).sla(cmd.sla);
    // get our run mode based on the command
    let run_mode = Mode::try_from(&cmd.sha256_or_repo)?;
    // supply a file or a repo dependency depending on our mode
    req = match &run_mode {
        Mode::File => {
            // supply the request a sample if we're in file mode
            println!(
                "Running {} on sample {}",
                cmd.pipeline.bright_green().bold(),
                cmd.sha256_or_repo.bright_yellow().bold()
            );
            req.sample(&cmd.sha256_or_repo)
        }
        Mode::Repo => {
            println!(
                "Running {} on repo {}",
                cmd.pipeline.bright_green().bold(),
                cmd.sha256_or_repo.bright_yellow().bold()
            );
            // otherwise supply a repo
            let repo_target = RepoTarget::try_from(&cmd.sha256_or_repo).map_err(|err| {
                Error::new(format!(
                    "The given target is neither a valid SHA256 nor a valid repo! {}",
                    err.msg().unwrap_or_default()
                ))
            })?;
            req.repo(repo_target.into())
        }
    };
    // simultaneously create the reaction and retrieve the underlying pipeline
    let (reaction, pipeline) = tokio::try_join!(
        thorium.reactions.create(&req),
        thorium.pipelines.get(&group, &cmd.pipeline)
    )?;
    println!("Created reaction: {}", reaction.id.bright_green().bold());
    // create the log and complete channels
    let (log_tx, log_rx) = mpsc::channel(50);
    let complete = Arc::new(AtomicBool::new(false));
    // create log forwarders for each stage in the pipeline
    for stage in pipeline.order.iter().flatten() {
        tokio::spawn(forward_logs(
            stage.to_owned(),
            thorium.reactions.logs_cursor(&group, &reaction.id, stage),
            log_tx.clone(),
            complete.clone(),
        ));
    }
    // drop the extra logs channel to ensure it doesn't keep their respective channels open
    drop(log_tx);
    // continuously check if the reaction is complete in a new thread
    tokio::spawn(watch_reaction_complete(
        thorium.clone(),
        group.clone(),
        reaction.id,
        complete.clone(),
    ));
    // log all information in the channel until the reaction is complete
    // save the tool colors to use when printing result logs
    let tool_colors = log(log_rx).await;
    println!("Reaction {} complete!", reaction.id.bright_green().bold());
    // write the results of the reaction to disk
    write_results(&thorium, cmd, pipeline, tool_colors, &run_mode).await?;
    Ok(())
}

/// Handle all run commands or print run info
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The run command to execute
pub async fn handle(args: &Args, cmd: &Run) -> Result<(), Error> {
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
    // call the run handler
    run(Arc::new(thorium), cmd).await
}
