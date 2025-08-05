//! The controllers used for thoradm working with data in scylla

use ahash::AHasher;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use kanal::{AsyncReceiver, AsyncSender};
use scylla::client::session::Session;
use std::collections::BTreeMap;
use std::hash::Hasher;
use std::sync::Arc;
use tokio::task::JoinHandle;

use crate::shared::monitor::{Monitor, MonitorUpdate};
use crate::Error;

/// The trait required for scalably crawling data in scylla
pub trait ScyllaCrawlSupport: Sized + Send + 'static {
    /// The arguments to specify when creating workers
    type WorkerArgs: Clone;

    /// Set the progress bars style for workers
    fn bar_style() -> Result<ProgressStyle, Error>;

    /// Set the progress bar style for this controllers monitor
    fn monitor_bar_style() -> Result<ProgressStyle, Error>;

    /// Build a single worker for this controller
    async fn build_worker(
        scylla: &Arc<Session>,
        namespace: &str,
        updates: AsyncSender<MonitorUpdate>,
        args: &Self::WorkerArgs,
        dry_run: bool,
        bar: ProgressBar,
    ) -> Result<Self, Error>;

    /// Start crawling data in scylla
    fn start(
        self,
        rx: AsyncReceiver<(i64, i64)>,
    ) -> impl std::future::Future<Output = Result<Self, Error>> + Send;

    /// Shutdown this worker
    fn shutdown(self);
}

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

pub struct ScyllaCrawlController<S: ScyllaCrawlSupport + Send> {
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
    /// The arguments to specify when creating workers
    pub args: S::WorkerArgs,
    /// The number of workers to spawn
    worker_count: usize,
    /// The currently active workers
    active: FuturesUnordered<JoinHandle<Result<S, Error>>>,
}

impl<S: ScyllaCrawlSupport + Send> ScyllaCrawlController<S> {
    /// Create a new controller
    pub fn new<N: Into<String>>(
        namespace: N,
        scylla: &Arc<Session>,
        args: S::WorkerArgs,
        workers: usize,
    ) -> Result<Self, Error> {
        // build our kanal channel for monitor updates
        let (updates_tx, updates_rx) = kanal::unbounded_async();
        // build our kanal channel for orders
        let (orders_tx, orders_rx) = kanal::unbounded_async();
        // build our controller
        let controller = ScyllaCrawlController {
            namespace: namespace.into(),
            scylla: scylla.clone(),
            updates_tx,
            updates_rx,
            orders_tx,
            orders_rx,
            progress: MultiProgress::new(),
            args,
            worker_count: workers,
            active: FuturesUnordered::default(),
        };
        Ok(controller)
    }
    /// Start our monitor
    fn start_monitor(&self) -> Result<JoinHandle<()>, Error> {
        // get this progress bars style
        let bar_style = S::monitor_bar_style()?;
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
        let handle = tokio::spawn(async move { monitor.start().await });
        Ok(handle)
    }

    /// Build and spawn all of our workers
    ///
    /// # Arguments
    ///
    /// * `dry_run` - Whether this controller is doing a dry run or not
    async fn spawn_all(&mut self, dry_run: bool) -> Result<(), Error> {
        // get this progress bars style
        let bar_style = S::bar_style()?;
        // create the right number of workers
        for _ in 0..self.worker_count {
            // create a new progress bar for this worker
            let bar = ProgressBar::new_spinner();
            // set a steady tick rate
            bar.enable_steady_tick(std::time::Duration::from_millis(120));
            // set this bars style
            bar.set_style(bar_style.clone());
            // add this progress bar to our main bar
            let bar = self.progress.add(bar);
            // build our worker
            let worker = S::build_worker(
                &self.scylla,
                &self.namespace,
                self.updates_tx.clone(),
                &self.args,
                dry_run,
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

    /// Generate our token ranges to crawl
    ///
    /// # Arguments
    ///
    /// * `chunk_count` - The number of chunks to break our token ring into
    async fn generate_token_ranges(&mut self, chunk_count: u64) -> Result<(), Error> {
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

    /// Start crawling data in scylla
    ///
    /// # Arguments
    ///
    /// * `chunk_count` - The number of chunks to split the work into
    /// * `dry_run` - Whether this controller is doing a dry run or not
    pub async fn start(&mut self, chunk_count: u64, dry_run: bool) -> Result<(), Error> {
        // start this controllers monitor
        let handle = self.start_monitor()?;
        // spawn all of our workers
        self.spawn_all(dry_run).await?;
        // generate our token ranges
        self.generate_token_ranges(chunk_count).await?;
        // wait for all workers to complete
        self.wait_for_workers().await?;
        // tell our monitor to finish and exit
        self.updates_tx.send(MonitorUpdate::Finished).await?;
        // wait for our map updater to finish
        handle.await?;
        Ok(())
    }
}
