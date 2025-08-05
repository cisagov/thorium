//! Scrub a backup for bitrot

use async_walkdir::Filtering;
use async_walkdir::WalkDir;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use kanal::{AsyncReceiver, AsyncSender};
use std::path::PathBuf;
use tokio::task::JoinHandle;

use crate::args::ScrubBackup;
use crate::backup::tables::{
    Comment, Commitish, CommitishList, Node, Output, OutputStream, RepoData, RepoList, S3Id,
    SamplesList, Tag,
};
use crate::backup::{Monitor, MonitorUpdate, Scrub, ScrubWorker};
use crate::Error;

/// A controller for a single table to scrub
pub struct TableScrub<S: Scrub> {
    /// The kanal channel workers should send kanal channel updates over
    updates_tx: AsyncSender<MonitorUpdate>,
    /// The kanal channel to receive archive map updates on
    updates_rx: AsyncReceiver<MonitorUpdate>,
    /// The kanal channel to send orders to workers on
    orders_tx: AsyncSender<PathBuf>,
    /// The kanal channel workers should get orders on
    orders_rx: AsyncReceiver<PathBuf>,
    /// A progress bar for this tables scrub
    progress: MultiProgress,
    /// The number of workers to when scrubbing data
    worker_count: usize,
    /// The currently active workers
    active: FuturesUnordered<JoinHandle<Result<ScrubWorker<S>, Error>>>,
}

impl<S: Scrub> TableScrub<S> {
    /// Create a new table scrub object
    ///
    /// # Arguments
    ///
    /// * `worker_count` - The number of workers to use when scrubbing data
    pub fn new(worker_count: usize) -> Self {
        // build our kanal channel for monitor updates
        let (updates_tx, updates_rx) = kanal::unbounded_async();
        // build our kanal channel for orders
        let (orders_tx, orders_rx) = kanal::unbounded_async();
        TableScrub {
            updates_tx,
            updates_rx,
            orders_tx,
            orders_rx,
            progress: MultiProgress::default(),
            worker_count,
            active: FuturesUnordered::default(),
        }
    }

    /// Build the workers for this table scrubber
    pub fn build_workers(&mut self) {
        // build the style for our progress bar
        let bar_style =
            ProgressStyle::with_template("{spinner} Scrubbed {msg} {bytes} {binary_bytes_per_sec}")
                .unwrap()
                .tick_strings(&[
                    "🦀🧼     ",
                    " 🦀🧼    ",
                    "  🦀🧼   ",
                    "   🦀🧼  ",
                    "    🦀🧼 ",
                    "     🦀🧼",
                    "     🧼🦀",
                    "    🧼🦀 ",
                    "   🧼🦀  ",
                    "  🧼🦀   ",
                    " 🧼🦀    ",
                    "🧼🦀     ",
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
            let worker = ScrubWorker::<S>::new(self.updates_tx.clone(), bar);
            // clone our orders channel
            let orders_rx = self.orders_rx.clone();
            // spawn this worker
            let handle = tokio::spawn(async move { worker.scrub(orders_rx).await });
            // add this task to our futures set
            self.active.push(handle);
        }
    }

    /// Start our global progress tracker
    async fn start_global_tracker(&self) -> JoinHandle<()> {
        // get a handle to our update receive channel
        let update_rx = self.updates_rx.clone();
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} {elapsed_precise} Total Scrubbed {bytes} {binary_bytes_per_sec}",
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
        // create our monitor
        let monitor = Monitor::new(bar, update_rx);
        // start our monitor
        tokio::spawn(async move { monitor.start().await })
    }

    /// Start scrubbing data
    ///
    /// # Arguments
    ///
    /// * `path` - The path to our this tables backup directory
    pub async fn spawn(&mut self, mut path: PathBuf) -> Result<(), Error> {
        // build the path to this tables backup map
        path.push(S::name());
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
                Err(error) => self.progress.println(&format!("Error: {:#?}", error))?,
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

    /// Scrub this tables backed up data
    pub async fn scrub(&mut self, path: PathBuf) -> Result<(), Error> {
        // get the name of the table to scrub without any '_'
        let pretty_name = S::pretty_name();
        // get the path to this specific scrub
        let table_path = path.join(S::name());
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
                "Path '{}' missing. Skipping {} scrub...",
                table_path.to_string_lossy(),
                pretty_name
            ))?;
            return Ok(());
        }
        // log the table we are backing up
        self.progress.println(format!("Scrubbing {pretty_name}"))?;
        // start our global tracker
        let handle = self.start_global_tracker().await;
        // build our workers
        self.build_workers();
        // spawn our scrub workers
        self.spawn(path).await?;
        // wait for all of our workers to finish
        self.wait_for_workers().await?;
        // tell our global tracker to finish
        self.updates_tx.send(MonitorUpdate::Finished).await?;
        // wait for our global tracker to finish
        handle.await?;
        Ok(())
    }
}

/// The scrub controller for a single Thorium backup
pub struct ScrubController {
    /// The samples list table
    samples_list: TableScrub<SamplesList>,
    /// The s3 IDs table
    s3_ids: TableScrub<S3Id>,
    /// The comments table
    comments: TableScrub<Comment>,
    /// The results table
    results: TableScrub<Output>,
    /// The results stream table
    results_stream: TableScrub<OutputStream>,
    /// The tags table
    tags: TableScrub<Tag>,
    /// The repo data table
    repo_data: TableScrub<RepoData>,
    /// The repo list table
    repos_list: TableScrub<RepoList>,
    /// The commit table
    commitish: TableScrub<Commitish>,
    /// The commit list table
    commitish_list: TableScrub<CommitishList>,
    /// The nodes tables
    nodes: TableScrub<Node>,
}

impl ScrubController {
    /// Create a new scrub controller
    pub fn new(worker_count: usize) -> Self {
        ScrubController {
            samples_list: TableScrub::new(worker_count),
            s3_ids: TableScrub::new(worker_count),
            comments: TableScrub::new(worker_count),
            results: TableScrub::new(worker_count),
            results_stream: TableScrub::new(worker_count),
            tags: TableScrub::new(worker_count),
            repo_data: TableScrub::new(worker_count),
            repos_list: TableScrub::new(worker_count),
            commitish: TableScrub::new(worker_count),
            commitish_list: TableScrub::new(worker_count),
            nodes: TableScrub::new(worker_count),
        }
    }

    /// Scrub this backup
    pub async fn scrub(&mut self, backup: PathBuf) -> Result<(), Error> {
        // scrub our tables
        self.samples_list.scrub(backup.clone()).await?;
        self.s3_ids.scrub(backup.clone()).await?;
        self.comments.scrub(backup.clone()).await?;
        self.results.scrub(backup.clone()).await?;
        self.results_stream.scrub(backup.clone()).await?;
        self.tags.scrub(backup.clone()).await?;
        self.repo_data.scrub(backup.clone()).await?;
        self.repos_list.scrub(backup.clone()).await?;
        self.commitish.scrub(backup.clone()).await?;
        self.commitish_list.scrub(backup.clone()).await?;
        self.nodes.scrub(backup.clone()).await?;
        Ok(())
    }
}

/// Handle the backup scrub command
pub async fn handle(scrub_args: &ScrubBackup, workers: usize) -> Result<(), Error> {
    // create a new table scrub controller
    let mut controller = ScrubController::new(workers);
    // start scrubbing data
    controller.scrub(scrub_args.backup.clone()).await?;
    Ok(())
}
