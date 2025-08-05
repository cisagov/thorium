//! The restore controller for Thorium
use async_walkdir::Filtering;
use async_walkdir::WalkDir;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use kanal::{AsyncReceiver, AsyncSender};
use rkyv::validation::validators::DefaultValidator;
use rkyv::Archive;
use scylla::client::session::Session;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use thorium::Conf;
use thorium::CtlConf;
use thorium::Thorium;
use tokio::task::JoinHandle;

use super::utils;
use crate::args::Args;
use crate::args::RestoreBackup;
use crate::backup::tables::{
    Comment, Commitish, CommitishList, Node, Output, OutputStream, RepoData, RepoList, S3Id,
    SamplesList, Tag,
};
use crate::backup::{
    Monitor, MonitorUpdate, Restore, RestoreWorker, S3Monitor, S3MonitorUpdate, S3Restore,
    S3RestoreWorker,
};
use crate::Error;

/// A singular table restore
struct TableRestore<R: Restore> {
    /// A Thorium config
    conf: Conf,
    /// The scylla client for this table
    scylla: Arc<Session>,
    /// The kanal channel workers should send kanal channel updates over
    updates_tx: AsyncSender<MonitorUpdate>,
    /// The kanal channel to receive archive map updates on
    updates_rx: AsyncReceiver<MonitorUpdate>,
    /// The kanal channel to send orders to workers on
    orders_tx: AsyncSender<PathBuf>,
    /// The kanal channel workers should get orders on
    orders_rx: AsyncReceiver<PathBuf>,
    /// A progress bar for this tables restore
    progress: MultiProgress,
    /// The number of workers to use when restoring data
    worker_count: usize,
    /// The currently active workers
    active: FuturesUnordered<JoinHandle<Result<RestoreWorker<R>, Error>>>,
}

impl<R: Restore> TableRestore<R> {
    /// Create a new table restore object
    ///
    /// # Arguments
    ///
    /// * `config` - A thorium config
    /// * `scylla` - The scylla client to use
    /// * `worker_count` - The number of workers to use
    pub fn new(conf: &Conf, scylla: &Arc<Session>, worker_count: usize) -> Self {
        // build our kanal channel for monitor updates
        let (updates_tx, updates_rx) = kanal::unbounded_async();
        // build our kanal channel for orders
        let (orders_tx, orders_rx) = kanal::unbounded_async();
        // build our table restore object
        TableRestore {
            conf: conf.clone(),
            scylla: scylla.clone(),
            updates_tx,
            updates_rx,
            orders_tx,
            orders_rx,
            progress: MultiProgress::default(),
            worker_count,
            active: FuturesUnordered::default(),
        }
    }

    /// Build our restore workers
    async fn build_workers(&mut self) -> Result<(), Error>
    where
        <R as Archive>::Archived:
            for<'a> bytecheck::CheckBytes<DefaultValidator<'a>> + std::fmt::Debug,
    {
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner} Restored Rows: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "📦🦀🌽     🚀",
            "📦 🦀🌽    🚀",
            "📦  🦀🌽   🚀",
            "📦   🦀🌽  🚀",
            "📦    🦀🌽 🚀",
            "📦     🦀🌽🚀",
            "📦       🦀🚀",
            "📦      🦀 🚀",
            "📦     🦀  🚀",
            "📦    🦀   🚀",
            "📦   🦀    🚀",
            "📦  🦀     🚀",
            "📦 🦀      🚀",
            "📦🦀       🚀",
        ]);
        for _ in 0..self.worker_count {
            // create a new progress bar for this worker
            let bar = ProgressBar::new_spinner();
            // set a steady tick rate
            bar.enable_steady_tick(std::time::Duration::from_millis(120));
            // set this bars style
            bar.set_style(bar_style.clone());
            // add this progress bar to our main bar
            let bar = self.progress.add(bar);
            // create a new worker
            let worker =
                RestoreWorker::<R>::new(&self.scylla, &self.conf, self.updates_tx.clone(), bar)
                    .await?;
            // clone our orders channel
            let orders_rx = self.orders_rx.clone();
            // spawn this worker
            let handle = tokio::spawn(async move { worker.restore(orders_rx).await });
            // add this task to our futures set
            self.active.push(handle);
        }
        Ok(())
    }

    /// Start our monitor
    fn start_monitor(&self) -> JoinHandle<()> {
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} {elapsed_precise} Total Restored Rows: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "🦀📋         ",
            " 🦀📋        ",
            "  🦀📋       ",
            "   🦀📋      ",
            "    🦀📋     ",
            "     🦀📋    ",
            "      🦀📋   ",
            "       🦀📋  ",
            "        🦀📋 ",
            "         🦀📋",
            "        🦀📋 ",
            "       🦀📋  ",
            "      🦀📋   ",
            "     🦀📋    ",
            "    🦀📋     ",
            "   🦀📋      ",
            "  🦀📋       ",
            " 🦀📋        ",
            "🦀📋         ",
        ]);
        // create a new progress bar for this worker
        let bar = ProgressBar::new_spinner();
        // set a steady tick rate
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        // set this bars style
        bar.set_style(bar_style.clone());
        // add this progress bar to our main bar
        let bar = self.progress.add(bar);
        // clone our update channel
        let update_rx = self.updates_rx.clone();
        // create our monitor
        let monitor = Monitor::new(bar, update_rx);
        // start our monitor
        tokio::spawn(async move { monitor.start().await })
    }

    /// Spawn our workers and have them start restoring data
    ///
    /// # Arguments
    ///
    /// * `path` - The path to our this tables backup directory
    async fn spawn(&mut self, mut path: PathBuf) -> Result<(), Error> {
        // build the path to this tables backup map
        path.push(R::name());
        path.push("maps");
        // crawl all of our maps for this table
        let mut walker = WalkDir::new(path).filter(|entry| async move {
            if !entry.file_name().to_string_lossy().ends_with(".thoriummap") {
                Filtering::Ignore
            } else {
                Filtering::Continue
            }
        });
        while let Some(entry) = walker.next().await {
            // log any errors
            match entry {
                Ok(entry) => self.orders_tx.send(entry.path()).await?,
                Err(error) => self.progress.println(&format!("Error: {error:#?}"))?,
            }
        }
        Ok(())
    }

    /// Wait for all of our workers to finish
    async fn wait_for_workers(&mut self) -> Result<(), Error> {
        // wait for our orders queue to become empty
        loop {
            // check if our order queue is empty
            if self.orders_tx.is_empty() {
                break;
            }
            // sleep for 1 second between checks
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        // shut down our orders channel
        self.orders_tx.close()?;
        // wait for all of our remaining workers to finish
        while let Some(worker) = self.active.next().await {
            // check if this worker ran into a problem
            let worker = worker??;
            // shutdown this workers progress bar
            worker.shutdown();
        }
        Ok(())
    }

    /// Start restoring this table
    async fn restore(&mut self, path: PathBuf) -> Result<(), Error>
    where
        <R as Archive>::Archived:
            for<'a> bytecheck::CheckBytes<DefaultValidator<'a>> + std::fmt::Debug,
    {
        // get the name of the table to scrub without any '_'
        let pretty_name = R::pretty_name();
        // get the path to this specific scrub
        let table_path = path.join(R::name());
        // check if the subdir for this table exists
        if !tokio::fs::try_exists(&table_path).await.map_err(|err| {
            Error::new(format!(
                "Failed to check if path '{}' exists: {}",
                table_path.to_string_lossy(),
                err
            ))
        })? {
            // the path is missing, so skip this restore
            self.progress.println(format!(
                "Path '{}' missing. Skipping {} restore...",
                table_path.to_string_lossy(),
                pretty_name
            ))?;
            return Ok(());
        }
        // log the table we are backing up
        self.progress.println(format!("Restoring {pretty_name}"))?;
        // run our restore prep function
        R::prep(&self.scylla, &self.conf.thorium.namespace).await?;
        // start our global monitor
        let handle = self.start_monitor();
        // build our workers
        self.build_workers().await?;
        // spawn our restore workers
        self.spawn(path).await?;
        // wait for all of our workers to finish
        self.wait_for_workers().await?;
        // tell our monitor to finish
        self.updates_tx.send(MonitorUpdate::Finished).await?;
        // wait for our tracker monitor finish
        handle.await?;
        Ok(())
    }
}

/// A singular s3 restore
struct S3RestoreController<R: S3Restore> {
    /// The Thorium config for the cluster we are restoring objects for
    conf: Conf,
    /// The kanal channel workers should send kanal channel updates over
    updates_tx: AsyncSender<S3MonitorUpdate>,
    /// The kanal channel to receive s3 monitor updates on
    updates_rx: AsyncReceiver<S3MonitorUpdate>,
    /// The kanal channel to send orders to workers on
    orders_tx: AsyncSender<PathBuf>,
    /// The kanal channel workers should get orders on
    orders_rx: AsyncReceiver<PathBuf>,
    /// A progress bar for this tables restore
    progress: MultiProgress,
    /// The number of workers to use when restoring data
    worker_count: usize,
    /// The currently active workers
    active: FuturesUnordered<JoinHandle<Result<S3RestoreWorker<R>, Error>>>,
}

impl<R: S3Restore> S3RestoreController<R> {
    /// Create a new table restore object
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config for the cluster we are restoring
    /// * `worker_count` - The number of workers to use
    pub fn new(conf: &Conf, worker_count: usize) -> Self {
        // build our kanal channels
        let (updates_tx, updates_rx) = kanal::unbounded_async();
        let (orders_tx, orders_rx) = kanal::unbounded_async();
        // build our table restore object
        S3RestoreController {
            conf: conf.clone(),
            updates_tx,
            updates_rx,
            orders_tx,
            orders_rx,
            progress: MultiProgress::default(),
            worker_count,
            active: FuturesUnordered::default(),
        }
    }

    /// Build our restore workers
    async fn build_workers(&mut self) -> Result<(), Error> {
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner} Restored S3 Objects: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "📦🦀🌽     🚀",
            "📦 🦀🌽    🚀",
            "📦  🦀🌽   🚀",
            "📦   🦀🌽  🚀",
            "📦    🦀🌽 🚀",
            "📦     🦀🌽🚀",
            "📦       🦀🚀",
            "📦      🦀 🚀",
            "📦     🦀  🚀",
            "📦    🦀   🚀",
            "📦   🦀    🚀",
            "📦  🦀     🚀",
            "📦 🦀      🚀",
            "📦🦀       🚀",
        ]);
        for _ in 0..self.worker_count {
            // create a new progress bar for this worker
            let bar = ProgressBar::new_spinner();
            // set a steady tick rate
            bar.enable_steady_tick(std::time::Duration::from_millis(120));
            // set this bars style
            bar.set_style(bar_style.clone());
            // add this progress bar to our main bar
            let bar = self.progress.add(bar);
            // create a new worker
            let worker = S3RestoreWorker::<R>::new(&self.conf, &self.updates_tx, bar);
            // clone our orders channel
            let orders_rx = self.orders_rx.clone();
            // spawn this worker
            let handle = tokio::spawn(async move { worker.restore(orders_rx).await });
            // add this task to our futures set
            self.active.push(handle);
        }
        Ok(())
    }

    /// Start our global progress tracker
    pub fn start_monitor(&mut self) -> JoinHandle<Result<(), Error>> {
        // get a handle to our update receive channel
        let update_rx = self.updates_rx.clone();
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} {elapsed_precise} Total Restored S3 Objects: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "🦀📋         ",
            " 🦀📋        ",
            "  🦀📋       ",
            "   🦀📋      ",
            "    🦀📋     ",
            "     🦀📋    ",
            "      🦀📋   ",
            "       🦀📋  ",
            "        🦀📋 ",
            "         🦀📋",
            "        🦀📋 ",
            "       🦀📋  ",
            "      🦀📋   ",
            "     🦀📋    ",
            "    🦀📋     ",
            "   🦀📋      ",
            "  🦀📋       ",
            " 🦀📋        ",
            "🦀📋         ",
        ]);
        // create a new progress bar for this worker
        let bar = ProgressBar::new_spinner();
        // set a steady tick rate
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        // set this bars style
        bar.set_style(bar_style.clone());
        // add this progress bar to our main bar
        let bar = self.progress.add(bar);
        // build our s3 monitor
        let monitor = S3Monitor::new(update_rx, bar);
        // start our s3 monitor
        tokio::spawn(async move { monitor.start().await })
    }

    /// Spawn our workers and have them start restoring data
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the root of the subset of data to restore
    async fn spawn(&mut self, mut path: PathBuf) -> Result<(), Error> {
        // change our path to the correct objects path
        path.push("objects");
        // build our walk dir stream
        let mut walker = WalkDir::new(path);
        // begin crawling the target dir and restoring its objects
        while let Some(entry) = walker.next().await {
            // log any entries that we fail to open
            match entry {
                // don't add any directories to our list
                Ok(entry) => {
                    if entry.metadata().await?.is_file() {
                        self.orders_tx.send(entry.path()).await?;
                    }
                }
                // log this error
                Err(error) => self
                    .progress
                    .println(format!("WalkDir Error: {error:#?}"))?,
            }
        }
        Ok(())
    }

    /// Wait for all of our workers to finish
    async fn wait_for_workers(&mut self) -> Result<(), Error> {
        // wait for our orders queue to become empty
        loop {
            // check if our order queue is empty
            if self.orders_tx.is_empty() {
                break;
            }
            // sleep for 1 second between checks
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        // shut down our orders channel
        self.orders_tx.close()?;
        // wait for all of our remaining workers to finish
        while let Some(worker) = self.active.next().await {
            // check if this worker ran into a problem
            worker??;
        }
        Ok(())
    }

    /// Start restoring this table
    ///
    /// # Arguments
    ///
    /// * `path` - The path to restore objects too
    async fn restore(&mut self, mut path: PathBuf) -> Result<(), Error> {
        // get our name without '_'
        let pretty_name = R::pretty_name();
        // nest our path by our table name
        path.push(R::name());
        // check if the subdir for this table exists
        if !tokio::fs::try_exists(&path).await.map_err(|err| {
            Error::new(format!(
                "Failed to check if path '{}' exists: {}",
                path.to_string_lossy(),
                err
            ))
        })? {
            // the path is missing, so skip this restore
            self.progress.println(format!(
                "Path '{}' missing. Skipping {} restore...",
                path.to_string_lossy(),
                pretty_name
            ))?;
            return Ok(());
        }
        // log the table we are backing up
        self.progress.println(format!("Restoring {pretty_name}"))?;
        // start our global tracker
        let handle = self.start_monitor();
        // build our workers
        self.build_workers().await?;
        // spawn our restore workers
        self.spawn(path).await?;
        // wait for all of our workers to finish
        self.wait_for_workers().await?;
        // tell our map updater to finish
        self.updates_tx.send(S3MonitorUpdate::Finished).await?;
        // wait for our global tracker to finish
        handle.await??;
        Ok(())
    }
}

/// The restore controller for Thorium
pub struct RestoreController {
    /// The Thorctl config to use to restore
    ctl_conf: CtlConf,
    /// The samples list table
    samples_list: TableRestore<SamplesList>,
    /// The s3 IDs table
    s3_ids: TableRestore<S3Id>,
    /// The comments table
    comments: TableRestore<Comment>,
    /// The results table
    results: TableRestore<Output>,
    /// The results table
    results_stream: TableRestore<OutputStream>,
    /// The tags table
    tags: TableRestore<Tag>,
    /// The repo_data table
    repo_data: TableRestore<RepoData>,
    /// The repos_list table
    repos_list: TableRestore<RepoList>,
    /// The commits table
    commitish: TableRestore<Commitish>,
    /// The commits_list table
    commitish_list: TableRestore<CommitishList>,
    /// The nodes table
    nodes: TableRestore<Node>,
    /// The samples/repos s3 objects
    s3_ids_objects: S3RestoreController<S3Id>,
    /// The comment attachment objects
    comment_attachments: S3RestoreController<Comment>,
    /// The result files objects
    result_files: S3RestoreController<Output>,
}

impl RestoreController {
    /// Create a new restore controller
    ///
    /// # Arguments
    ///
    /// * `config` - The Thorium config for the cluster to restore
    /// * `ctl_conf` - The Thorctl config to use to restore
    /// * `scylla` - A scylla client for this cluster
    /// * `workers` - The number of workers to use
    pub fn new(
        config: &Conf,
        ctl_conf: CtlConf,
        scylla: &Arc<Session>,
        workers: usize,
    ) -> Result<Self, Error> {
        // get confirmation from the user before continuing
        Self::confirm(config)?;
        // build our table restore objects
        let samples_list = TableRestore::new(config, scylla, workers);
        let s3_ids = TableRestore::new(config, scylla, workers);
        let comments = TableRestore::new(config, scylla, workers);
        let results = TableRestore::new(config, scylla, workers);
        let results_stream = TableRestore::new(config, scylla, workers);
        let tags = TableRestore::new(config, scylla, workers);
        let repo_data = TableRestore::new(config, scylla, workers);
        let repos_list = TableRestore::new(config, scylla, workers);
        let commits = TableRestore::new(config, scylla, workers);
        let commits_list = TableRestore::new(config, scylla, workers);
        let nodes = TableRestore::new(config, scylla, workers);
        // build our s3 restore objects
        let s3_ids_objects = S3RestoreController::new(config, workers);
        let comment_attachments = S3RestoreController::new(config, workers);
        let result_files = S3RestoreController::new(config, workers);
        // build our controller
        let controller = RestoreController {
            ctl_conf,
            samples_list,
            s3_ids,
            comments,
            results,
            results_stream,
            tags,
            repo_data,
            repos_list,
            commitish: commits,
            commitish_list: commits_list,
            nodes,
            s3_ids_objects,
            comment_attachments,
            result_files,
        };
        Ok(controller)
    }

    /// Confirm the user wants top restore data
    ///
    /// # Arguments
    ///
    /// * `config` - The config file we are using
    fn confirm(config: &Conf) -> Result<(), Error> {
        // log our target namespace
        println!("Namespace: {}", config.thorium.namespace);
        // log scylla nodes
        println!("Scylla:");
        // crawl and print our scylla ips
        for node in &config.scylla.nodes {
            println!("  - {node}");
        }
        // log the buckets we will be overwriting
        println!("Buckets:");
        println!("  Files: {}", config.thorium.files.bucket);
        println!("  Repos: {}", config.thorium.repos.bucket);
        println!("  Attachments: {}", config.thorium.attachments.bucket);
        println!("  Results: {}", config.thorium.results.bucket);
        // ask the user for permission
        let response = dialoguer::Confirm::new()
            .with_prompt("Do you want to restore data to the above databases:")
            .interact()?;
        // check their response
        if !response {
            // exit our current process
            println!("Exiting!");
            std::process::exit(0);
        }
        Ok(())
    }

    /// Restore this clusters redis data from disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this backup too
    async fn restore_redis(&self, mut path: PathBuf) -> Result<(), Error> {
        // build a Thorium client
        let client = Thorium::from_ctl_conf(self.ctl_conf.clone()).await?;
        // build the path to our backup file
        path.push("redis.json");
        // check if the redis backup exists
        if !tokio::fs::try_exists(&path).await.map_err(|err| {
            Error::new(format!(
                "Failed to check if path '{}' exists: {}",
                path.to_string_lossy(),
                err
            ))
        })? {
            println!(
                "Redis backup at '{}' not found. Skipping redis restore...",
                path.to_string_lossy()
            );
            return Ok(());
        }
        // load our backup from disk
        let backup_str = tokio::fs::read_to_string(&path).await?;
        // deserialize our backup
        let backup = serde_json::from_str(&backup_str)?;
        // restore this backup
        client.system.restore(&backup).await?;
        Ok(())
    }

    /// Restore a clusters data from disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the data to restore
    pub async fn restore(&mut self, path: &Path) -> Result<(), Error> {
        let path = path.to_path_buf();
        // restore our redis cluster
        self.restore_redis(path.clone()).await?;
        // restore our tables
        self.samples_list.restore(path.clone()).await?;
        self.s3_ids.restore(path.clone()).await?;
        self.comments.restore(path.clone()).await?;
        self.results.restore(path.clone()).await?;
        self.results_stream.restore(path.clone()).await?;
        self.tags.restore(path.clone()).await?;
        self.repo_data.restore(path.clone()).await?;
        self.repos_list.restore(path.clone()).await?;
        self.commitish.restore(path.clone()).await?;
        self.commitish_list.restore(path.clone()).await?;
        self.nodes.restore(path.clone()).await?;
        // restore our s3 objects
        self.s3_ids_objects.restore(path.clone()).await?;
        self.comment_attachments.restore(path.clone()).await?;
        self.result_files.restore(path).await?;
        // tell the user to restart all API pods to complete the restore
        println!("Restore Complete!");
        println!("Please restart all API pods to complete the backup.");
        Ok(())
    }
}

/// Handle a restore sub command
///
/// # Arguments
///
/// * `restore_args` - The args for the restore handler
/// * `args` - The Thoradm args
pub async fn handle(restore_args: &RestoreBackup, args: &Args) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    // load our Thorctl config
    let ctl_conf = CtlConf::from_path(&args.ctl_conf)?;
    // build a new scylla client
    let scylla = Arc::new(utils::get_scylla_client(&config).await?);
    // build the controller for this cluster
    let mut controller = RestoreController::new(&config, ctl_conf, &scylla, args.workers)?;
    // retore this cluster from disk
    controller.restore(&restore_args.backup).await?;
    Ok(())
}
