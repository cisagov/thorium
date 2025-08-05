//! Handle cluster login/logout/status and other commands

use colored::Colorize;
use std::io::Write;
use thorium::models::{Node, NodeHealth, NodeListParams, Worker, WorkerStatus};
use thorium::{CtlConf, Error, Keys, Thorium};
use tokio::fs::create_dir_all;

use super::update;
use crate::args::clusters::{ClusterStatus, ClusterWorkers, Clusters, Login};
use crate::args::Args;
use crate::utils;

/// Get a users credentials
fn get_creds(cmd: &Login) -> Result<(String, String), Error> {
    // get this users username if we don't have one
    let username = match &cmd.user {
        Some(username) => username.to_owned(),
        None => {
            // we don't have a username so we need to get one
            print!("Username: ");
            // flush standard out so it prints our username prompt
            std::io::stdout().flush()?;
            // read in our username from stdout
            let mut username = String::new();
            std::io::stdin().read_line(&mut username)?;
            // remove the newline
            username.trim_end().to_string()
        }
    };
    // get this users password if we didn't get it already
    let password = match &cmd.password {
        Some(password) => password.to_owned(),
        None => rpassword::prompt_password("Password: ")?,
    };
    Ok((username, password))
}

/// Login to a Thorium cluster and update our creds
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The login command that was run
pub async fn login(args: &Args, cmd: &Login) -> Result<(), Error> {
    // get existing config unless we're clearing settings
    let config = if cmd.clear_settings {
        // refrain from getting a config if we want to clear settings
        None
    } else {
        // otherwise try to get existing settings
        CtlConf::from_path(&args.config).ok()
    };
    // get this users creds
    let (username, password) = get_creds(cmd)?;
    // build a thorium client
    let mut builder = Thorium::build(&cmd.url);
    // set the builder settings to be the config's if we have one
    if let Some(config) = &config {
        builder.settings = config.client.clone();
    }
    // override client settings with args
    if let Some(invalid_certs) = cmd.invalid_certs {
        builder.settings.invalid_certs = invalid_certs;
    }
    if let Some(invalid_hostnames) = cmd.invalid_hostnames {
        builder.settings.invalid_hostnames = invalid_hostnames;
    }
    if let Some(certificate_authorities) = &cmd.certificate_authorities {
        builder
            .settings
            .certificate_authorities
            .clone_from(certificate_authorities);
    }
    // warn the user if the config isn't set to skip the warning
    if !config
        .as_ref()
        .is_some_and(|config| config.skip_insecure_warning.unwrap_or_default())
    {
        utils::warn_insecure(
            &cmd.url,
            builder.settings.invalid_certs,
            builder.settings.invalid_hostnames,
            &builder.settings.certificate_authorities,
        )?;
    }
    // login to Thorium and get a valid client
    let thorium = builder.basic_auth(username, password).build().await?;
    // get info on our user
    let user = thorium.users.info().await?;
    // build a key to save to this users config
    let keys = Keys::new_token(&cmd.url, user.token);
    // either unwrap and update the keys of our existing conf or create a new one
    let config = match config {
        Some(mut config) => {
            config.keys = keys;
            config
        }
        None => CtlConf::new(keys),
    };
    if let Some(parent) = args.config.parent() {
        // make sure the path for our config file exists
        create_dir_all(parent).await?;
    }
    // open the file to write this config off to disk
    let config_file = std::fs::File::create(&args.config)?;
    // write this config file off to disk
    serde_yaml::to_writer(config_file, &config)?;
    println!("🦀🎉 Login Succeeded! 🎉🦀");
    // check if we need to update
    if !args.skip_update {
        update::ask_update(&thorium).await?;
    }
    Ok(())
}

macro_rules! status_print {
    ($cluster:expr, $node:expr, $status:expr, $workers:expr, $cpu:expr, $memory:expr, $storage:expr) => {
        println!(
            "{:<16} | {:<16} | {:<10} | {:<7} | {:<10} | {:<12} | {}",
            $cluster, $node, $status, $workers, $cpu, $memory, $storage
        )
    };
}

/// A single line for basic node status
struct StatusLine;

impl StatusLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<16} | {:<16} | {:<10} | {:<7} | {:<7} | {:<9} | {:<9} ",
            "CLUSTER", "NODE", "STATUS", "WORKERS", "CPU (mCPU)", "MEMORY (MiB)", "STORAGE (MiB)"
        );
        println!(
            "{:-<17}+{:-<18}+{:-<12}+{:-<9}+{:-<12}+{:-<14}+{:-<14}",
            "", "", "", "", "", "", ""
        );
    }

    /// Print this nodes status
    ///
    /// # Arguments
    ///
    /// * `node` - The node to print info on
    pub fn print(node: &Node) {
        // get our nodes health as a colored string
        let health = match node.health {
            NodeHealth::Healthy => "Healthy".bright_green(),
            NodeHealth::Unhealthy => "Unhealthy".bright_red(),
            NodeHealth::Disabled(_) => "Disabled".bright_yellow(),
            NodeHealth::Registered => "Registered".bright_purple(),
        };
        status_print!(
            &node.cluster,
            &node.name,
            &health,
            &node.workers.len(),
            node.resources.cpu,
            node.resources.memory,
            node.resources.ephemeral_storage
        );
    }
}

/// Get the status of Thorium cluster
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use for dumping cluster status
async fn status(thorium: &Thorium, cmd: &ClusterStatus) -> Result<(), Error> {
    // print the header for getting node info
    StatusLine::header();
    // build the params for getting the target clusters node info
    let params = NodeListParams::from(cmd);
    // build the cursor for listing our nodes
    let mut cursor = thorium.system.list_node_details(&params).await?;
    // loop until we have crawled all of our nodes
    loop {
        // crawl the files listed and print info about them
        cursor.data.iter().for_each(StatusLine::print);
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of data
        cursor.refill().await?;
    }
    Ok(())
}

macro_rules! worker_print {
    ($user:expr, $group:expr, $pipeline:expr, $stage:expr, $scaler:expr, $status:expr, $node:expr) => {
        println!(
            "{:<16} | {:<16} | {:<25} | {:<16} | {:<8} | {:<8} | {:<16}",
            $user, $group, $pipeline, $stage, $scaler, $status, $node
        )
    };
}

/// A single line for basic node status
struct WorkerLine;

impl WorkerLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<16} | {:<16} | {:<25} | {:<16} | {:<8} | {:<8} | {:<16}",
            "USER", "GROUP", "PIPELINE", "STAGE", "SCALER", "STATUS", "NODE"
        );
        println!(
            "{:-<17}+{:-<18}+{:-<27}+{:-<18}+{:-<10}+{:-<10}+{:-<18}",
            "", "", "", "", "", "", ""
        );
    }

    /// Print this workers status
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to print info on
    pub fn print(worker: &Worker) {
        // get our workers status as a colored string
        let status = match worker.status {
            WorkerStatus::Spawning => "Spawning".bright_blue(),
            WorkerStatus::Running => "Running".bright_green(),
            WorkerStatus::Shutdown => "Shutdown".bright_magenta(),
        };
        worker_print!(
            &worker.user,
            &worker.group,
            &worker.pipeline,
            &worker.stage,
            &worker.scaler.to_string(),
            status,
            &worker.node
        );
    }
}

/// Get the status of Thorium cluster
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use for dumping cluster status
async fn workers(thorium: &Thorium, cmd: &ClusterWorkers) -> Result<(), Error> {
    // print the header for getting worker info
    WorkerLine::header();
    // build the params for getting the target clusters node info
    let params = NodeListParams::from(cmd);
    // build the cursor for listing our nodes
    let mut cursor = thorium.system.list_node_details(&params).await?;
    // loop until we have crawled all of our nodes
    loop {
        // crawl the nodes listed and print info about their workers
        cursor
            .data
            .iter()
            .flat_map(|node| &node.workers)
            .for_each(|(_, worker)| WorkerLine::print(worker));
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of data
        cursor.refill().await?;
    }
    Ok(())
}

/// Handle all context commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The context command to execute
pub async fn handle(args: &Args, cmd: &Clusters) -> Result<(), Error> {
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
        Clusters::Status(cmd) => status(&thorium, cmd).await,
        Clusters::Workers(cmd) => workers(&thorium, cmd).await,
    }
}
