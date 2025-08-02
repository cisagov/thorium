//! Stream data to our search store

use futures::stream::{FuturesUnordered, StreamExt};
use kanal::{AsyncReceiver, AsyncSender};
use redis::aio::MultiplexedConnection;
use scylla::client::session::Session;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use thorium::{client::SearchEventsClient, models::SearchEventPopOpts, Conf, Error, Thorium};
use tokio::task::JoinHandle;
use tracing::{event, instrument, Level};

use crate::events::{CompactEvent, EventCompactable};
use crate::init::{InitSession, InitSessionInfo, InitSessionKeys};
use crate::sources::DataSource;
use crate::stores::SearchStore;
use crate::worker::StreamWorker;
use crate::{args::Args, monitor::Monitor};
use crate::{
    index::IndexMapping,
    msg::{Job, JobStatus},
};

/// Stream data from Thorium to a full text search store
pub struct SearchStreamer<D: DataSource, S: SearchStore> {
    /// The command line args passed in
    args: Args,
    /// The thorium config for this cluster
    conf: Conf,
    /// A Thorium client
    thorium: Arc<Thorium>,
    /// A scylla client
    scylla: Arc<Session>,
    /// A multiplexed connection to Redis
    redis: MultiplexedConnection,
    /// The transmit side of the queue of time chunks to stream
    jobs_tx: AsyncSender<Job<D>>,
    /// The receive side of the queue of time chunks to stream
    jobs_rx: AsyncReceiver<Job<D>>,
    /// The channel to send progress updates on
    progress_tx: AsyncSender<JobStatus>,
    /// The channel to listen for progress updates on
    progress_rx: AsyncReceiver<JobStatus>,
    /// The source of our data
    data_source: PhantomData<D>,
    /// The search store to stream data too,
    search_store: PhantomData<S>,
}

impl<D: DataSource + 'static, S: SearchStore> SearchStreamer<D, S>
where
    D::IndexType: IndexMapping<S>,
{
    /// Create a new search streamer
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `scylla` - The scylla client
    /// * `redis` - A redis multiplexed connection
    /// * `args` - The command line args for our search streamer
    /// * `conf` - The Thorium config
    pub fn new(
        thorium: Arc<Thorium>,
        scylla: Arc<Session>,
        redis: MultiplexedConnection,
        args: &Args,
        conf: Conf,
    ) -> Self {
        // build our job queue channels
        let (jobs_tx, jobs_rx) = kanal::bounded_async(1000);
        // build our progress queue channels
        let (progress_tx, progress_rx) = kanal::bounded_async(10000);
        // build our search streamer
        SearchStreamer {
            args: args.clone(),
            conf,
            thorium,
            scylla,
            redis,
            jobs_tx,
            jobs_rx,
            progress_tx,
            progress_rx,
            data_source: PhantomData,
            search_store: PhantomData,
        }
    }

    /// Start streaming data to our search store
    #[instrument(name = "SearchStreamer::start", skip_all, err(Debug))]
    pub async fn start(mut self) -> Result<(), Error> {
        // keep a set of all of our futures
        let mut futures = FuturesUnordered::new();
        // create the search source instance
        let source = D::new(&self.scylla, &self.conf.thorium.namespace).await?;
        let store = S::new(&self.conf)?;
        // spawn our workers
        self.spawn_workers(&mut futures, source, &store);
        // try initiating the search store and starting/resuming an init session
        let maybe_init_session = self.try_init(store).await?;
        // get the tokens remaining
        let tokens_remaining = maybe_init_session
            .as_ref()
            .map(|s| s.tokens_remaining.clone())
            // if we have no init session, return an empty map so we don't process any
            // tokens in the init feeder
            .unwrap_or_default();
        // spawn our monitor
        self.spawn_monitor(&mut futures, maybe_init_session);
        // spawn the feeder for init jobs
        self.spawn_init_feeder(&mut futures, tokens_remaining);
        // wait for all of our futures to complete
        while let Some(result) = futures.next().await {
            // check if any of our futures failed
            result??;
            // the only task that will return not in error is the init task,
            // so if we got here, we need to spawn the regular feeder task
            self.spawn_feeder(&mut futures);
        }
        Ok(())
    }

    /// Spawn this streamers monitor
    ///
    /// # Arguments
    ///
    /// * `futures` - A set of futures to track
    fn spawn_monitor(
        &self,
        futures: &mut FuturesUnordered<JoinHandle<Result<(), Error>>>,
        init_session: Option<InitSession>,
    ) {
        // build our monitor
        let monitor = Monitor::<D>::new(
            D::event_client(&self.thorium).clone(),
            self.progress_rx.clone(),
            init_session,
        );
        // spawn our monitor
        futures.push(tokio::spawn(monitor.start()));
    }

    /// Spawn this streamers workers
    ///
    /// # Arguments
    ///
    /// * `futures` - A set of futures to track
    /// * `source` - The data source instance to give to the workers
    fn spawn_workers(
        &mut self,
        futures: &mut FuturesUnordered<JoinHandle<Result<(), Error>>>,
        source: D,
        store: &S,
    ) {
        // create and spawn this streamers workers
        for _ in 1..self.conf.thorium.search_streamer.workers.into() {
            // create a new worker
            let worker: StreamWorker<D, S> = StreamWorker::new(
                self.scylla.clone(),
                self.jobs_rx.clone(),
                self.progress_tx.clone(),
                source.clone(),
                store.clone(),
            );
            // spawn this worker
            futures.push(tokio::spawn(worker.start()));
        }
        // create a new worker
        let worker: StreamWorker<D, S> = StreamWorker::new(
            self.scylla.clone(),
            self.jobs_rx.clone(),
            self.progress_tx.clone(),
            source,
            store.clone(),
        );
        // spawn this worker
        futures.push(tokio::spawn(worker.start()));
    }

    /// Spawn our init feeder
    fn spawn_init_feeder(
        &mut self,
        futures: &mut FuturesUnordered<JoinHandle<Result<(), Error>>>,
        tokens_remaining: BTreeMap<i64, i64>,
    ) {
        // spawn our feeder
        let handle = tokio::spawn(Self::feed_init_queue(
            self.jobs_tx.clone(),
            tokens_remaining,
        ));
        // add this future to our futures set
        futures.push(handle);
    }

    /// Spawn our feeder
    fn spawn_feeder(&mut self, futures: &mut FuturesUnordered<JoinHandle<Result<(), Error>>>) {
        // spawn our feeder
        let handle = tokio::spawn(Self::feed_queue(self.thorium.clone(), self.jobs_tx.clone()));
        // add this future to our futures set
        futures.push(handle);
    }

    /// Attempt to initiate the search store and see if an init session is needed
    ///
    /// Returns an init session if one is needed
    ///
    /// # Arguments
    ///
    /// * `store` - The store we're trying to init
    #[instrument(name = "SearchStreamer::try_init", skip_all, fields(store = S::STORE_NAME, data = D::DATA_NAME), err(Debug))]
    async fn try_init(&mut self, store: S) -> Result<Option<InitSession>, Error> {
        // calculate the number of chunks to split the db into based on the config
        let chunk_count = self
            .conf
            .thorium
            .search_streamer
            .workers
            .saturating_mul(self.conf.thorium.search_streamer.init.chunks_per_worker)
            .get() as u64;
        // create our init keys
        let keys = InitSessionKeys::new(&self.conf, S::STORE_NAME, D::DATA_NAME);
        // check to see if we have an init session already in progress
        let maybe_init_info = InitSessionInfo::query(&keys.info, &mut self.redis).await?;
        // try to initiate the search store and see if we need to run an init session
        let need_init = store
            .init(&D::IndexType::all_indexes(), self.args.reindex)
            .await?;
        // we need to create an init session before listening for events if the search store
        // was just initiated (is brand new and empty) or we have an init session in Redis (we're
        // resuming an init job)
        match (need_init, maybe_init_info) {
            // the store is newly initiated but we have init session info from a previous initiating session;
            // the initiating session was interrupted and the store was deleted/renamed, so delete the init
            // session and start from scratch
            (true, Some(_old_init_info)) => {
                // log that we're replacing the session
                event!(
                    Level::WARN,
                    "Search store newly initiated but init session exists in db! Deleting session and restarting..."
                );
                // create a new session
                Ok(Some(
                    InitSession::create(chunk_count, keys, &mut self.redis).await?,
                ))
            }
            // the store is newly initiated and we have no init session so create one
            (true, None) => {
                // create a new session
                Ok(Some(
                    InitSession::create(chunk_count, keys, &mut self.redis).await?,
                ))
            }
            // the store is not newly initiated but we have init info, so we're resuming a session
            (false, Some(old_init_info)) => {
                // try to resume the session
                match InitSession::resume(old_init_info, chunk_count, keys.clone(), &mut self.redis)
                    .await?
                {
                    // we were able to resume the session, so return that
                    Some(resumed_session) => {
                        event!(
                            Level::INFO,
                            msg = "Resuming init session",
                            session_start = resumed_session.info.start.to_rfc3339()
                        );
                        Ok(Some(resumed_session))
                    }
                    // we failed to resume the session, so create a new one
                    None => {
                        event!(
                            Level::WARN,
                            msg = "Failed to resume init session! Creating a new one..."
                        );
                        Ok(Some(
                            InitSession::create(chunk_count, keys, &mut self.redis).await?,
                        ))
                    }
                }
            }
            // the store is not newly initiated and we have no init session, so no init needed
            (false, None) => Ok(None),
        }
    }

    /// Send init jobs for all token ranges to workers
    ///
    /// # Arguments
    ///
    /// * `jobs_tx` - The channel to send init jobs to workers
    /// * `tokens_remaining` - The token ranges remaining to process
    #[instrument(name = "SearchStreamer::feed_init_queue", skip_all, fields(store = S::STORE_NAME, data = D::DATA_NAME), err(Debug))]
    async fn feed_init_queue(
        jobs_tx: AsyncSender<Job<D>>,
        tokens_remaining: BTreeMap<i64, i64>,
    ) -> Result<(), Error> {
        // send init jobs for each token range
        for (start, end) in tokens_remaining {
            jobs_tx.send(Job::Init { start, end }).await?;
        }
        loop {
            // check to see if all jobs have been picked up
            if jobs_tx.is_empty() {
                // all init jobs have been picked up so we can exit
                break;
            }
            // sleep for a second between checks
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        Ok(())
    }

    /// Check for events in Thorium and feed jobs to workers forever
    ///
    /// # Arguments
    ///
    /// * `jobs_tx` - The channel to push jobs into
    #[instrument(name = "SearchStreamer::feed_queue", skip_all, fields(store = S::STORE_NAME, data = D::DATA_NAME), err(Debug))]
    async fn feed_queue(thorium: Arc<Thorium>, jobs_tx: AsyncSender<Job<D>>) -> Result<(), Error> {
        // reset all events before we start feeding;
        // makes sure that any events that were in-flight are put back in the queue
        D::event_client(&thorium)
            .reset_all()
            .await
            .map_err(|err| Error::new(format!("Failed to reset Thorium search events: {err}")))?;
        tokio::time::sleep(Duration::from_secs(5)).await;
        event!(Level::INFO, "Handling events");
        // pop events up to 300 at a time;
        // the more we pop, the better compaction performs, but the API request will take longer
        let opts = SearchEventPopOpts::default().limit(300);
        loop {
            // check if we have any events
            let events = D::event_client(&thorium).pop(&opts).await?;
            if events.is_empty() {
                // sleep for 10 seconds if we got no events
                tokio::time::sleep(Duration::from_secs(10)).await;
            } else {
                // compact "like" events together to avoid re-streaming data needlessly when
                // an item is modified multiple times in a short period
                let compacted_events = Self::compact_events(events);
                // send the events as messages to the workers
                for compacted_event in compacted_events {
                    jobs_tx.send(Job::Event { compacted_event }).await?;
                }
            }
        }
    }

    /// Compact "like" search events down to avoid streaming data multiple times needlessly
    ///
    /// Particularly useful when a single item/group combo is modified many times in a short
    /// period. Data may be edited hundreds of times but only streamed once
    ///
    /// # Arguments
    ///
    /// * `events` - The events to compact
    #[instrument(name = "SearchStreamer::compact_events", skip_all, fields(store = S::STORE_NAME, data = D::DATA_NAME))]
    fn compact_events(events: Vec<D::Event>) -> Vec<D::CompactEvent> {
        // get a count of our events before compaction
        let num_events = events.len();
        // create a map for compacting
        let mut event_map: HashMap<_, D::CompactEvent> = HashMap::new();
        for event in events {
            // see if we already have an event with this compacting key
            match event_map.entry(event.compact_by()) {
                // we already have an event like this, so append it to the existing one
                Entry::Occupied(mut occupied_entry) => {
                    occupied_entry.get_mut().append(event);
                }
                // this is the first of its kind, so convert it into a compactable event
                Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(event.into());
                }
            }
        }
        // log how much we've compacted events by
        event!(
            Level::INFO,
            events = num_events,
            compacted = event_map.len()
        );
        // return our compacted events
        event_map.into_values().collect()
    }
}
