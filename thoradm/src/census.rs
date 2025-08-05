//! Census commands in thoradm

use bb8_redis::bb8::Pool;
use bb8_redis::RedisConnectionManager;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use kanal::{AsyncReceiver, AsyncSender};
use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use std::marker::PhantomData;
use std::sync::Arc;
use thorium::models::backends::TagSupport;
use thorium::models::{
    CensusSupport, Commitish, Repo, Sample, TagCensusCaseInsensitive, TagRequest,
};
use thorium::Conf;

use crate::args::{Args, CensusKinds, CensusSubCommands, NewCensus};
use crate::backup::Utils;
use crate::shared::monitor::MonitorUpdate;
use crate::shared::scylla::{ScyllaCrawlController, ScyllaCrawlSupport};
use crate::Error;

pub struct CensusWorker<T: CensusSupport> {
    /// The keyspace/namespace for this worker
    namespace: String,
    /// The scylla client to talk to Scylla with
    scylla: Arc<Session>,
    /// The redis client to talk to redis with
    redis: Pool<RedisConnectionManager>,
    /// The prepared statement to use
    prepared: PreparedStatement,
    /// The kanal channel workers should send backup updates over
    #[allow(dead_code)]
    updates: AsyncSender<MonitorUpdate>,
    /// The current number of partitions this worker has counted
    partitions_counted: u64,
    /// The progress bar to write error messages with
    progress: ProgressBar,
    /// Whether to write data to the db or not
    dry_run: bool,
    /// The table we are taking a census of
    phantom: PhantomData<T>,
}

impl<T: CensusSupport> CensusWorker<T> {
    pub async fn new(
        scylla: &Arc<Session>,
        redis: &Pool<RedisConnectionManager>,
        namespace: &str,
        updates: AsyncSender<MonitorUpdate>,
        dry_run: bool,
        progress: ProgressBar,
    ) -> Result<Self, Error> {
        // get our prepared statement
        let prepared = T::scan_prepared_statement(scylla, namespace).await?;
        // build this backup worker
        let worker = CensusWorker {
            namespace: namespace.to_owned(),
            scylla: scylla.clone(),
            redis: redis.clone(),
            prepared,
            updates,
            partitions_counted: 0,
            dry_run,
            progress,
            phantom: PhantomData,
        };
        Ok(worker)
    }

    /// Start backing up data and streaming it to archives
    ///
    /// # Arguments
    ///
    /// * `orders` - The channel to receive map paths on
    #[rustfmt::skip]
    pub async fn start(mut self, orders: AsyncReceiver<(i64, i64)>) -> Result<Self, Error> {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            let (start, end) = match orders.recv().await {
                Ok(path) => path,
                Err(kanal::ReceiveError::Closed) => break,
                Err(kanal::ReceiveError::SendClosed) => break,
            };
            // build and execute our paged query
            let rows_stream = self
                .scylla
                .execute_iter(self.prepared.clone(), &(start, end))
                .await?;
            // build a typed iter for these rows
            let mut typed_stream = match rows_stream.rows_stream::<T::Row>() {
                Ok(typed_stream) => typed_stream,
                Err(error) => {
                    // build our error message
                    let msg = format!("Failed to set type for row stream: with {:#?}", error);
                    // log that we failed to cast this row
                    self.progress.println(msg.clone());
                    // continue to the next row
                    continue;
                }
            };
            // build the redis command to insert info into redis
            let mut pipe = redis::pipe();
            // track how many partitions we counted before we handle this stream
            let old_counted = self.partitions_counted;
            // crawl over our typed stream
            while let Some(typed_row) = typed_stream.next().await {
                // error out if any of our typed rows fail
                let typed_row = match typed_row {
                    Ok(typed_row) => typed_row,
                    Err(error) => {
                        // build our error message
                        let msg = format!("Census failure: with {:#?}", error);
                        // log that we failed to cast this row
                        self.progress.println(msg.clone());
                        // continue to the next row
                        continue;
                    }
                };
                // increment our partition count
                self.partitions_counted += 1;
                // skip building a pipeline if this is a dry run
                if !self.dry_run {
                    // group this partitions count, bucket, and count grouping
                    let count = T::get_count(&typed_row);
                    let bucket = T::get_bucket(&typed_row);
                    let grouping = bucket / 10_000;
                    // build the key to this partitions counts/stream
                    let count_key = T::count_key_from_row(&self.namespace, &typed_row, grouping);
                    let stream_key = T::stream_key_from_row(&self.namespace, &typed_row);
                    // add data into redis
                    pipe.cmd("hset").arg(count_key).arg(bucket).arg(count)
                        .cmd("zadd").arg(stream_key).arg(bucket).arg(bucket);
                }
            }
                // skip saving things to redis if this is a dry run
            if !self.dry_run {
                // get a connection to our redis db
                let mut conn = self.redis.get().await.unwrap();
                // execute our redis pipeline
                let _: () = pipe.atomic().query_async(&mut *conn).await?;
            }
            // set our current row count progress message
            self.progress
                .set_message(self.partitions_counted.to_string());
            // calculate how many partitions we counted in this loop
            let items = (self.partitions_counted - old_counted) as usize;
            // build the update to send to our controller
            let update = MonitorUpdate::Update { items, bytes: 0 };
            // send an update to our controller
            self.updates.send(update).await?;
        }
        Ok(self)
    }
}

impl<T: CensusSupport + Utils> ScyllaCrawlSupport for CensusWorker<T> {
    /// The arguments to specify when creating workers
    type WorkerArgs = Pool<RedisConnectionManager>;

    /// Set the progress bars style for workers
    fn bar_style() -> Result<ProgressStyle, Error> {
        // build the style string for this
        let style_str = format!("{{spinner:.green}} {} Censused: {{msg}}", T::name());
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(&style_str)
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
        Ok(bar_style)
    }

    /// Set the progress bar style for this controllers monitor
    fn monitor_bar_style() -> Result<ProgressStyle, Error> {
        // build the style string for our monitor
        let style_str = format!(
            "{{spinner:.green}} {{elapsed_precise}} Total Censused {}: {{msg}}",
            T::name()
        );
        // build the style for our progress bar
        let bar_style = ProgressStyle::with_template(&style_str)
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
        Ok(bar_style)
    }

    /// Build a single worker for this controller
    async fn build_worker(
        scylla: &Arc<Session>,
        namespace: &str,
        updates: AsyncSender<MonitorUpdate>,
        args: &Self::WorkerArgs,
        dry_run: bool,
        bar: ProgressBar,
    ) -> Result<Self, Error> {
        // build our census worker
        CensusWorker::new(scylla, args, namespace, updates, dry_run, bar).await
    }

    /// Start crawling data in scylla
    async fn start(self, rx: AsyncReceiver<(i64, i64)>) -> Result<Self, Error> {
        self.start(rx).await
    }

    /// Shutdown this worker
    fn shutdown(self) {}
}

/// Take a new full census of tag data
///
/// # Arguments
///
/// * `args` - The command line args for thorctl
/// * `dry_run` - Whether this is a dry run or not
async fn new_tags(args: &Args, dry_run: bool) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    // build a new scylla client
    let scylla = Arc::new(crate::shared::scylla::get_client(&config).await?);
    // build a new redis client
    let redis = crate::shared::redis::get_client(&config).await?;
    // build a new tag census controller
    // we are using sample but the type doesn't matter for tag scans actually
    let mut controller = ScyllaCrawlController::<CensusWorker<TagRequest<Sample>>>::new(
        &config.thorium.namespace,
        &scylla,
        redis,
        args.workers,
    )?;
    // start taking a census of tag data
    controller.start(1000, dry_run).await?;
    Ok(())
}

impl<T: TagSupport> Utils for TagCensusCaseInsensitive<T> {
    fn name() -> &'static str {
        "tags_case_insensitive"
    }
}

/// Take a new full census of tag data
///
/// # Arguments
///
/// * `args` - The command line args for thorctl
/// * `dry_run` - Whether this is a dry run or not
async fn new_tags_case_insensitive(args: &Args, dry_run: bool) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    // build a new scylla client
    let scylla = Arc::new(crate::shared::scylla::get_client(&config).await?);
    // build a new redis client
    let redis = crate::shared::redis::get_client(&config).await?;
    // build a new tag census controller
    // we are using sample but the type doesn't matter for tag scans actually
    let mut controller =
        ScyllaCrawlController::<CensusWorker<TagCensusCaseInsensitive<Sample>>>::new(
            &config.thorium.namespace,
            &scylla,
            redis,
            args.workers,
        )?;
    // start taking a census of tag data
    controller.start(1000, dry_run).await?;
    Ok(())
}

impl Utils for Sample {
    fn name() -> &'static str {
        "samples_list"
    }
}

/// Take a new full census of sample data
///
/// # Arguments
///
/// * `args` - The command line args for thorctl
/// * `dry_run` - Whether this is a dry run or not
async fn new_samples(args: &Args, dry_run: bool) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    // build a new scylla client
    let scylla = Arc::new(crate::shared::scylla::get_client(&config).await?);
    // build a new redis client
    let redis = crate::shared::redis::get_client(&config).await?;
    // build a new sample census controller
    let mut controller = ScyllaCrawlController::<CensusWorker<Sample>>::new(
        &config.thorium.namespace,
        &scylla,
        redis,
        args.workers,
    )?;
    // start taking a census of tag data
    controller.start(1000, dry_run).await?;
    Ok(())
}

impl Utils for Repo {
    fn name() -> &'static str {
        "repos_list"
    }
}

/// Take a new full census of repo data
///
/// # Arguments
///
/// * `args` - The command line args for thorctl
/// * `dry_run` - Whether this is a dry run or not
async fn new_repos(args: &Args, dry_run: bool) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    // build a new scylla client
    let scylla = Arc::new(crate::shared::scylla::get_client(&config).await?);
    // build a new redis client
    let redis = crate::shared::redis::get_client(&config).await?;
    // build a new repo census controller
    let mut controller = ScyllaCrawlController::<CensusWorker<Repo>>::new(
        &config.thorium.namespace,
        &scylla,
        redis,
        args.workers,
    )?;
    // start taking a census of tag data
    controller.start(1000, dry_run).await?;
    Ok(())
}

impl Utils for Commitish {
    fn name() -> &'static str {
        "commitish_list"
    }
}

/// Take a new full census of commitishes data
///
/// # Arguments
///
/// * `args` - The command line args for thorctl
/// * `dry_run` - Whether this is a dry run or not
async fn new_commitishes(args: &Args, dry_run: bool) -> Result<(), Error> {
    // load our config
    let config = Conf::new(&args.cluster_conf)?;
    // build a new scylla client
    let scylla = Arc::new(crate::shared::scylla::get_client(&config).await?);
    // build a new redis client
    let redis = crate::shared::redis::get_client(&config).await?;
    // build a new commitishes census controller
    let mut controller = ScyllaCrawlController::<CensusWorker<Commitish>>::new(
        &config.thorium.namespace,
        &scylla,
        redis,
        args.workers,
    )?;
    // start taking a census of tag data
    controller.start(1000, dry_run).await?;
    Ok(())
}

impl CensusKinds {
    /// Execute a specific census kind
    ///
    /// # Arguments
    ///
    /// * `args` - The command line args for thorctl
    /// * `dry_run` - Whether this is a dry run or not
    async fn start(self, args: &Args, dry_run: bool) -> Result<(), Error> {
        // perform a specific kind of census
        match self {
            // All gets handled in the new function and shouldn't occur ever
            CensusKinds::All => panic!("All wasn't mapped to all census kinds?"),
            CensusKinds::Tags => new_tags(args, dry_run).await,
            CensusKinds::TagsCaseInsensitive => new_tags_case_insensitive(args, dry_run).await,
            CensusKinds::Files => new_samples(args, dry_run).await,
            CensusKinds::Repos => new_repos(args, dry_run).await,
            CensusKinds::Commitishes => new_commitishes(args, dry_run).await,
        }
    }
}

/// Execute any requested censuses
async fn new(cmd: &NewCensus, args: &Args) -> Result<(), Error> {
    // if we have all set then replace it with all of the census types
    let kinds = if cmd.census_kinds.contains(&CensusKinds::All) {
        // add all possible census kinds
        vec![
            CensusKinds::Tags,
            CensusKinds::TagsCaseInsensitive,
            CensusKinds::Files,
            CensusKinds::Repos,
            CensusKinds::Commitishes,
        ]
    } else {
        // clone our census kinds
        cmd.census_kinds.clone()
    };
    // iterate over the requested census kinds and perform them
    for kind in &kinds {
        // perform a census on the correct data
        kind.start(args, cmd.dry_run).await?;
    }
    Ok(())
}

/// Handle census commands
///
/// # Arguments
///
/// * `cmd` - The census commands
pub async fn handle(cmd: &CensusSubCommands, args: &Args) -> Result<(), Error> {
    match cmd {
        CensusSubCommands::New(cmd) => new(cmd, args).await?,
    }
    Ok(())
}
