//! The backup controller for Thorium
use ahash::AHasher;
use async_walkdir::Filtering;
use async_walkdir::WalkDir;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use indicatif::MultiProgress;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use kanal::AsyncReceiver;
use kanal::AsyncSender;
use rkyv::validation::validators::DefaultValidator;
use rkyv::Archive;
use scylla::client::session::Session;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thorium::Conf;
use thorium::CtlConf;
use thorium::Thorium;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;

use super::utils;
use crate::args::Args;
use crate::args::BackupComponents;
use crate::args::NewBackup;
use crate::backup::tables::{
    Comment, Commitish, CommitishList, Node, Output, OutputStream, RepoData, RepoList, S3Id,
    SamplesList, Tag,
};
use crate::backup::{Backup, BackupWorker, Monitor, MonitorUpdate, S3Backup, S3BackupWorker};
use crate::Error;

/// Build the range of tokens to backup
fn build_token_ranges(chunk_count: u64) -> BTreeMap<u64, (i64, i64)> {
    // determine how large each of of our chunks should be
    let chunk_size = (u64::MAX / chunk_count) as i64;
    // build our hasher
    let mut hasher = AHasher::default();
    // crawl over our token ranges and build them
    let mut tokens = BTreeMap::default();
    let mut start = i64::MIN;
    for _ in 1..chunk_count {
        // calculate the end for this chunk
        let end = start + chunk_size;
        // hash our start and end
        hasher.write_i64(start);
        hasher.write_i64(end);
        // get this chunks hash
        let chunk_key = hasher.finish();
        // add this chunk to our token list
        tokens.insert(chunk_key, (start, end));
        // increment our start position
        start = if end < 0 { end - 1 } else { end + 1 };
    }
    // hash our start and the end of our token range
    hasher.write_i64(start);
    hasher.write_i64(i64::MAX);
    // get this chunks hash
    let chunk_key = hasher.finish();
    // add our final chunk
    tokens.insert(chunk_key, (start, i64::MAX));
    tokens
}

/// A singular table backup
pub struct TableBackup<B: Backup> {
    /// The namespace for this table
    namespace: String,
    /// The scylla client for this table
    scylla: Arc<Session>,
    /// The kanal channel workers should send kanal channel updates over
    updates_tx: AsyncSender<MonitorUpdate>,
    /// The kanal channel to receive archive map updates on
    updates_rx: AsyncReceiver<MonitorUpdate>,
    /// The kanal channel to send orders to workers on
    orders_tx: AsyncSender<(i64, i64)>,
    /// The kanal channel workers should get orders on
    orders_rx: AsyncReceiver<(i64, i64)>,
    /// A progress bar for this tables backup
    progress: MultiProgress,
    /// The number of workers to spawn
    worker_count: usize,
    /// The currently active workers
    active: FuturesUnordered<JoinHandle<Result<BackupWorker<B>, Error>>>,
}

impl<B: Backup> TableBackup<B> {
    /// Create a new table backup
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace of the table to back up
    /// * `scylla` - The Scylla client to use
    /// * `workers` - The number of workers to use
    pub fn new<T: Into<String>>(namespace: T, scylla: &Arc<Session>, workers: usize) -> Self {
        // build our kanal channel for monitor updates
        let (updates_tx, updates_rx) = kanal::unbounded_async();
        // build our kanal channel for orders
        let (orders_tx, orders_rx) = kanal::unbounded_async();
        // build our table backup object
        TableBackup {
            namespace: namespace.into(),
            scylla: scylla.clone(),
            updates_tx,
            updates_rx,
            orders_tx,
            orders_rx,
            progress: MultiProgress::new(),
            worker_count: workers,
            active: FuturesUnordered::default(),
        }
    }

    /// Build a single backup worker
    ///
    /// # Arguments
    ///
    /// * `path` - The path for this worker to store archives at
    async fn spawn_workers(&self, path: &Path) -> Result<(), Error> {
        let path = path.to_path_buf();
        // nest our archives in a data folder and put the maps in a map folder
        let data_path = [path.clone(), "data".into()].iter().collect();
        let map_path = [path.clone(), "maps".into()].iter().collect();
        // create our data dir
        tokio::fs::create_dir_all(&data_path).await?;
        tokio::fs::create_dir_all(&map_path).await?;
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} Backed Up Rows: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "🦀🌽     📦",
            " 🦀🌽    📦",
            "  🦀🌽   📦",
            "   🦀🌽  📦",
            "    🦀🌽 📦",
            "     🦀🌽📦",
            "       🦀📦",
            "      🦀 📦",
            "     🦀  📦",
            "    🦀   📦",
            "   🦀    📦",
            "  🦀     📦",
            " 🦀      📦",
            "🦀       📦",
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
            let worker = BackupWorker::<B>::new(
                &self.scylla,
                &self.namespace,
                self.updates_tx.clone(),
                &data_path,
                &map_path,
                bar,
            )
            .await?;
            // clone our orders channel
            let orders_rx = self.orders_rx.clone();
            // spawn this worker
            let handle = tokio::spawn(async move { worker.start(orders_rx).await });
            // add this task to our futures set
            self.active.push(handle);
        }
        Ok(())
    }

    /// Start our monitor
    pub fn start_monitor(&mut self) -> JoinHandle<()> {
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} {elapsed_precise} Total Backed Up Rows: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "🦀📋       ",
            " 🦀📋      ",
            "  🦀📋     ",
            "   🦀📋    ",
            "    🦀📋   ",
            "     🦀📋  ",
            "       🦀📋",
            "      🦀📋 ",
            "     🦀📋  ",
            "    🦀📋   ",
            "   🦀📋    ",
            "  🦀📋     ",
            " 🦀📋      ",
            "🦀📋       ",
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

    /// Start sending token segments to our workers
    ///
    /// # Arguments
    ///
    /// * `chunk_count` - The number of chunks to break our token ring into
    async fn start(&mut self, chunk_count: u64) -> Result<(), Error> {
        // build our token range
        let tokens = build_token_ranges(chunk_count);
        // add our tank range into our channel
        for (_, segment) in tokens {
            // add this segment to our channel
            self.orders_tx.send(segment).await?;
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
            match worker {
                Ok(Ok(worker)) => worker.shutdown(),
                Ok(Err(error)) => self.progress.println(format!("Worker Err: {error:#?}"))?,
                Err(error) => self.progress.println(format!("JOIN ERR: {error:#?}"))?,
            };
        }
        Ok(())
    }

    /// Start backing up this table
    ///
    /// # Arguments
    ///
    /// * `components` - The components set to backup
    /// * `path` - The path to store all of this Thorium clusters backups in
    /// * `chunk_count` - The number of chunks to break our token range into
    pub async fn backup(
        &mut self,
        components: &HashSet<BackupComponents>,
        mut path: PathBuf,
        chunk_count: u64,
    ) -> Result<(), Error> {
        // check if we're supposed to backup this component
        if components.contains(&BackupComponents::All)
            || components.contains(&B::backup_component())
        {
            // log the table we are backing up
            self.progress
                .println(format!("Backing up {}", B::pretty_name()))?;
            // nest our path by our table name
            path.push(B::name());
            // start our archive map updater
            let handle = self.start_monitor();
            // build our workers
            self.spawn_workers(&path).await?;
            // start backing up data
            self.start(chunk_count).await?;
            // wait for all of our workers to finish
            self.wait_for_workers().await?;
            // tell our monitor to finish and exit
            self.updates_tx.send(MonitorUpdate::Finished).await?;
            // wait for our map updater to finish
            handle.await?;
        } else {
            // we're not supposed to backup this component so skip it
            self.progress
                .println(format!("Skipping {} backup...", B::pretty_name()))?;
        }
        Ok(())
    }
}

impl<B: Backup> std::fmt::Debug for TableBackup<B> {
    /// Allow TableBackup to be debug printed
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableBackup")
            .field("namespace", &self.namespace)
            .field("table", &B::name())
            .finish()
    }
}

/// A singular table backup
pub struct S3BackupController<S: S3Backup> {
    /// The namespace for this table
    namespace: String,
    /// The Thorium config for the cluster we are backing up
    conf: Conf,
    /// The kanal channel workers should send kanal channel updates over
    updates_tx: AsyncSender<MonitorUpdate>,
    /// The kanal channel to receive archive map updates on
    updates_rx: AsyncReceiver<MonitorUpdate>,
    /// The kanal channel to send orders to workers on
    orders_tx: AsyncSender<PathBuf>,
    /// The kanal channel workers should get orders on
    orders_rx: AsyncReceiver<PathBuf>,
    /// A progress bar for this tables backup
    progress: MultiProgress,
    /// The number of workers to spawn
    worker_count: usize,
    /// The currently active workers
    active: FuturesUnordered<JoinHandle<Result<S3BackupWorker<S>, Error>>>,
}

impl<S: S3Backup> S3BackupController<S> {
    /// Create a new table backup
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace of the table to back up
    /// * `conf` - The Thorium config for the cluster we are backing up
    /// * `workers` - The number of workers to use
    pub fn new<T: Into<String>>(namespace: T, conf: &Conf, workers: usize) -> Self {
        // build our kanal channel for monitor updates
        let (updates_tx, updates_rx) = kanal::unbounded_async();
        // build our kanal channel for orders
        let (orders_tx, orders_rx) = kanal::unbounded_async();
        // build our table backup object
        S3BackupController {
            namespace: namespace.into(),
            conf: conf.clone(),
            updates_tx,
            updates_rx,
            orders_tx,
            orders_rx,
            progress: MultiProgress::new(),
            worker_count: workers,
            active: FuturesUnordered::default(),
        }
    }

    /// Spawn our s3 backup workers
    async fn spawn_workers(&self, path: &mut PathBuf) -> Result<(), Error>
    where
        <S as Archive>::Archived:
            for<'a> bytecheck::CheckBytes<DefaultValidator<'a>> + std::fmt::Debug,
    {
        // build the path to our object directory
        path.push("objects");
        // create the dir for object storage if doesn't yet exist
        tokio::fs::create_dir_all(&path).await?;
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} Backed Up S3 Objects: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "🦀🌽     📦",
            " 🦀🌽    📦",
            "  🦀🌽   📦",
            "   🦀🌽  📦",
            "    🦀🌽 📦",
            "     🦀🌽📦",
            "       🦀📦",
            "      🦀 📦",
            "     🦀  📦",
            "    🦀   📦",
            "   🦀    📦",
            "  🦀     📦",
            " 🦀      📦",
            "🦀       📦",
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
            let worker = S3BackupWorker::<S>::new(&self.conf, &self.updates_tx, bar, path);
            // clone our orders channel
            let orders_rx = self.orders_rx.clone();
            // spawn this worker
            let handle = tokio::spawn(async move { worker.backup(orders_rx).await });
            // add this task to our futures set
            self.active.push(handle);
        }
        Ok(())
    }

    /// Start our global progress tracker
    fn start_global_tracker(&self) -> JoinHandle<()> {
        // clone our update channel
        let update_rx = self.updates_rx.clone();
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} {elapsed_precise} Total Backed Up S3 Objects: {msg} {bytes} {binary_bytes_per_sec}",
        )
        .unwrap()
        .tick_strings(&[
            "🦀📋       ",
            " 🦀📋      ",
            "  🦀📋     ",
            "   🦀📋    ",
            "    🦀📋   ",
            "     🦀📋  ",
            "       🦀📋",
            "      🦀📋 ",
            "     🦀📋  ",
            "    🦀📋   ",
            "   🦀📋    ",
            "  🦀📋     ",
            " 🦀📋      ",
            "🦀📋       ",
        ]);
        // create a new progress bar for this worker
        let bar = ProgressBar::new_spinner();
        // set a steady tick rate
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        // set this bars style
        bar.set_style(bar_style.clone());
        // add this progress bar to our main bar
        let bar = self.progress.add(bar);
        // create our monitor
        let monitor = Monitor::new(bar, update_rx);
        // start our monitor
        tokio::spawn(async move { monitor.start().await })
    }

    /// Start backing up objects from s3
    ///
    /// # Arguments
    ///
    /// * `path` - The path for our workers to store archives at
    async fn start(&mut self, path: &mut PathBuf) -> Result<(), Error> {
        // pop our objects directory
        path.pop();
        // switch to our map directory
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

    // Start backing up this table
    ///
    /// # Arguments
    ///
    /// * `path` - The path to store all of this Thorium clusters backups in
    pub async fn backup(
        &mut self,
        components: &HashSet<BackupComponents>,
        mut path: PathBuf,
    ) -> Result<(), Error>
    where
        <S as Archive>::Archived:
            for<'a> bytecheck::CheckBytes<DefaultValidator<'a>> + std::fmt::Debug,
    {
        // get our name without '_'
        let pretty_name = S::name().replace('_', " ");
        // check if we're supposed to backup this component
        if components.contains(&BackupComponents::All)
            || components.contains(&S::backup_component())
        {
            // log the table we are backing up
            self.progress.println(format!("Backing up {pretty_name}"))?;
            // nest our path by our table name
            path.push(S::name());
            // start our archive map updater
            let handle = self.start_global_tracker();
            // build our workers
            self.spawn_workers(&mut path).await?;
            // start backing up data
            self.start(&mut path).await?;
            // wait for all of our workers to finish
            self.wait_for_workers().await?;
            // tell our map updater to finish
            self.updates_tx.send(MonitorUpdate::Finished).await?;
            // wait for our map updater to finish
            handle.await?;
        } else {
            // we're not supposed to backup this component so skip it
            self.progress
                .println(format!("Skipping {pretty_name} backup"))?;
        }
        Ok(())
    }
}

impl<S: S3Backup> std::fmt::Debug for S3BackupController<S> {
    /// Allow TableBackup to be debug printed
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Backup")
            .field("namespace", &self.namespace)
            .field("table", &S::name())
            .finish()
    }
}

/// The backup controller for a single Thorium cluster
#[derive(Debug)]
pub struct BackupController {
    /// The Thorctl config
    ctl_conf: CtlConf,
    /// The number of chunks to split our token range into
    chunks: u64,
    /// The components to backup
    components: HashSet<BackupComponents>,
    /// The samples list table
    samples_list: TableBackup<SamplesList>,
    /// The s3 IDs table
    s3_ids: TableBackup<S3Id>,
    /// The comments table
    comments: TableBackup<Comment>,
    /// The results table
    results: TableBackup<Output>,
    /// The results stream table
    results_stream: TableBackup<OutputStream>,
    /// The tags table
    tags: TableBackup<Tag>,
    /// The repo data table
    repo_data: TableBackup<RepoData>,
    /// The repo list table
    repos_list: TableBackup<RepoList>,
    /// The commitish table
    commitish: TableBackup<Commitish>,
    /// The commitish list table
    commitish_list: TableBackup<CommitishList>,
    /// The nodes tables
    nodes: TableBackup<Node>,
    /// The s3 IDs S3 backup
    s3_ids_objects: S3BackupController<S3Id>,
    /// The comments S3 backup
    comment_attachments: S3BackupController<Comment>,
    /// The result files S3 backup
    result_files: S3BackupController<Output>,
}

impl BackupController {
    /// Build a new backup controller for a single Thorium cluster
    ///
    /// # Arguments
    ///
    /// * `config` - The Thorium config for the cluster to backup
    /// * `ctl_conf` - The Thorctl config to use for the backup
    /// * `scylla` - The client to use with scylla
    /// * `workers` - The number of workers to spawn
    /// * `multiplier` - The multiplier to use with our worker count
    /// * `components` - The components of Thorium to backup
    pub fn new(
        config: &Conf,
        ctl_conf: CtlConf,
        scylla: &Arc<Session>,
        workers: usize,
        multiplier: u64,
        components: HashSet<BackupComponents>,
    ) -> Self {
        // get this clusters namespace
        let namespace = &config.thorium.namespace;
        // build our table backups
        let samples_list = TableBackup::new(namespace, scylla, workers);
        let s3_ids = TableBackup::new(namespace, scylla, workers);
        let comments = TableBackup::new(namespace, scylla, workers);
        let results = TableBackup::new(namespace, scylla, workers);
        let results_stream = TableBackup::new(namespace, scylla, workers);
        let tags = TableBackup::new(namespace, scylla, workers);
        let repo_data = TableBackup::new(namespace, scylla, workers);
        let repos_list = TableBackup::new(namespace, scylla, workers);
        let commits = TableBackup::new(namespace, scylla, workers);
        let commits_list = TableBackup::new(namespace, scylla, workers);
        let nodes = TableBackup::new(namespace, scylla, workers);
        // build our s3 backups
        let s3_ids_objects = S3BackupController::new(namespace, config, workers);
        let comment_attachments = S3BackupController::new(namespace, config, workers);
        let result_files = S3BackupController::new(namespace, config, workers);
        // build our cluster backup controller
        BackupController {
            ctl_conf,
            chunks: workers as u64 * multiplier,
            components,
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
        }
    }

    /// Backup this clusters redis data to disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this backup too
    async fn backup_redis(&self, mut path: PathBuf) -> Result<(), Error> {
        // check if we're supposed to backup Redis
        if self.components.contains(&BackupComponents::All)
            || self.components.contains(&BackupComponents::Redis)
        {
            // create our data dir
            tokio::fs::create_dir_all(&path).await?;
            // build a Thorium client
            let client = Thorium::from_ctl_conf(self.ctl_conf.clone()).await?;
            // get a backup of our redis data
            let backup = client.system.backup().await?;
            // serialize our backup to json
            // we use json instead of rkyv because datetimes seem to be causing problems
            let backup_str = serde_json::to_string(&backup)?;
            // build the path to write our backup off to disk
            path.push("redis.json");
            // get a handle to the file to write our redis backup too
            let mut file = File::create(&path).await?;
            // write our serialized backup to disk
            file.write_all(backup_str.as_bytes()).await?;
        } else {
            // skip redis backup
            println!("Skipping redis backup...");
        }
        Ok(())
    }

    /// Backup this cluster to disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write this backup too
    #[rustfmt::skip]
    pub async fn backup(&mut self, path: &Path) -> Result<(), Error> {
        let path = path.to_path_buf();
        // backup our redis data to disk
        self.backup_redis(path.clone()).await?;
        // backup our s3 data
        self.samples_list.backup(&self.components, path.clone(), self.chunks).await?;
        self.s3_ids.backup(&self.components, path.clone(), self.chunks).await?;
        self.comments.backup(&self.components, path.clone(), self.chunks).await?;
        self.results.backup(&self.components, path.clone(), self.chunks).await?;
        self.results_stream.backup(&self.components, path.clone(), self.chunks).await?;
        self.tags.backup(&self.components, path.clone(), self.chunks).await?;
        self.repo_data.backup(&self.components, path.clone(), self.chunks).await?;
        self.repos_list.backup(&self.components, path.clone(), self.chunks).await?;
        self.commitish.backup(&self.components, path.clone(), self.chunks).await?;
        self.commitish_list.backup(&self.components, path.clone(), self.chunks).await?;
        self.nodes.backup(&self.components, path.clone(), self.chunks).await?;
        // backup our s3 data
        self.s3_ids_objects.backup(&self.components, path.clone()).await?;
        self.comment_attachments.backup(&self.components, path.clone()).await?;
        self.result_files.backup(&self.components, path).await?;
        Ok(())
    }
}

/// Handle a backup take sub command
///
/// # Arguments
///
/// * `backup_args` - The args for the backup handler
/// * `args` - The Thoradm args
pub async fn handle(backup_args: &NewBackup, args: &Args) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    let ctl_conf = CtlConf::from_path(&args.ctl_conf)?;
    // build a new scylla client
    let scylla = Arc::new(utils::get_scylla_client(&config).await?);
    // get a deduped set of components to backup
    let components = backup_args.components.iter().copied().collect();
    // build the controller for this cluster
    let mut controller = BackupController::new(
        &config,
        ctl_conf,
        &scylla,
        args.workers,
        backup_args.multiplier,
        components,
    );
    // backup this cluster to disk
    controller.backup(&backup_args.output).await?;
    Ok(())
}
