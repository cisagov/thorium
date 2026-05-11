//! The different kinds of workers for events in Thorium

use chrono::prelude::*;
use futures::lock::Mutex;
use futures::stream::{self, StreamExt};
use kanal::{AsyncReceiver, AsyncSender};
use linearize::Linearize;
use std::collections::HashMap;
use std::ops::AddAssign;
use std::sync::Arc;
use std::time::Duration;
use thorium::models::{Event, EventPopOpts, EventType, ScrubbedUser};
use thorium::{Error, Thorium};
use tokio::task::JoinHandle;
use tracing::{Level, event, instrument};

pub mod hunts;
pub mod reaction_triggers;

pub use hunts::sigma::{SigmaRuleContext, SigmaRuleWorker};
pub use reaction_triggers::ReactionTriggerWorker;

use crate::libs::EventWorkerCache;
use crate::libs::cache::TriggerCache;
use crate::libs::stats::EventWorkerStatChannels;

/// The different kind of cache info that workers may subscribe too
#[derive(Clone, strum::EnumDiscriminants)]
#[strum_discriminants(name(WorkerCacheUpdateKinds))]
pub enum WorkerCacheUpdates {
    /// An updated reaction trigger cache
    ReactionTriggers(TriggerCache),
    /// An updated sigma rule context
    SigmaRulesContext(SigmaRuleContext),
    /// There has been an update to users in thorium
    Users(HashMap<String, ScrubbedUser>),
}

impl WorkerCacheUpdates {
    /// Build the correct update from a cache and an update kind
    pub fn new(cache: &EventWorkerCache, kind: WorkerCacheUpdateKinds) -> Self {
        // build the correct update based on our update kind
        match kind {
            WorkerCacheUpdateKinds::ReactionTriggers => {
                Self::ReactionTriggers(cache.triggers.clone())
            }
            WorkerCacheUpdateKinds::SigmaRulesContext => {
                Self::SigmaRulesContext(cache.sigma.clone())
            }
            WorkerCacheUpdateKinds::Users => Self::Users(cache.users.clone()),
        }
    }
}

/// The different messages that a controller may send a worker
#[derive(Clone)]
pub enum WorkerMsg {
    /// This worker should updates its cached info
    Update(WorkerCacheUpdates),
}

/// The core trait that the different kind of event works need to support
pub trait EventWorkerSupportCore {
    /// The stats for all workers of this kind
    type Stats: Default + AddAssign + Eq + PartialEq + Clone;

    /// Build this specific kind of worker
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of all info a worker may need to be created
    /// * `thorium` - A Thorium client
    async fn new(cache: &EventWorkerCache, thorium: &Arc<Thorium>) -> Self;

    /// The type of events this worker can handle
    fn event_kind() -> EventType;

    /// The types of cache updates this worker subscribes too
    fn cache_subscriptions() -> Vec<WorkerCacheUpdateKinds>;

    /// Apply an update to our local worker cache
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this workers cache
    fn apply_cache_update(&mut self, update: WorkerCacheUpdates);

    /// Report our latest stats in an event
    fn stats(stats: &Self::Stats, prior: &Self::Stats);
}

/// The trait that the different kind of event workers need to handle
pub trait EventWorkerSupport: EventWorkerSupportCore {
    /// The method to call to handle or process a single event
    async fn process(&self, event: Event, stats: &Arc<Mutex<Self::Stats>>) -> Result<(), Error>;
}

/// The differnt kind of workers for this controller
#[derive(Linearize, Clone, Copy, strum::Display, strum::EnumIter)]
pub enum EventWorkerKinds {
    /// A reaction trigger worker
    ReactionTriggers,
    /// A sigma scanner worker,
    SigmaScanner,
}

impl EventWorkerKinds {
    /// Check if this kind of worker subscribes to this kind of update
    ///
    /// # Arguments
    ///
    /// * `update_kind` - The kind of update in question
    pub fn is_subscribed(&self, update_kind: WorkerCacheUpdateKinds) -> bool {
        // get the kinds of udpates we subscribe too
        let subscribed = match self {
            Self::ReactionTriggers => ReactionTriggerWorker::cache_subscriptions(),
            Self::SigmaScanner => SigmaRuleWorker::cache_subscriptions(),
        };
        // check if this is something we are subscribed too
        subscribed.contains(&update_kind)
    }

    /// Spawn a single worker for this worker kind
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info for all event worker kinds
    /// * `stats_channels` - The channels to send stats updates to our controller over
    /// * `thorium` - A Thorium client
    pub async fn spawn(
        &self,
        cache: &EventWorkerCache,
        stats_channels: &EventWorkerStatChannels,
        thorium: &Arc<Thorium>,
    ) -> (AsyncSender<WorkerMsg>, JoinHandle<Result<(), Error>>) {
        // create a new challen for this worker
        let (msg_tx, msg_rx) = kanal::unbounded_async();
        // spawn the correct worker kind
        let handle = match self {
            EventWorkerKinds::ReactionTriggers => {
                // get the send side of our states channel
                let stats_tx = &stats_channels.reaction_trigger.0;
                // build a worker for reaction triggers rules
                let worker = EventWorkerMutable::<ReactionTriggerWorker>::new(
                    cache, thorium, &msg_rx, stats_tx,
                )
                .await;
                // spawn this sigma worker
                tokio::task::spawn(worker.start())
            }
            EventWorkerKinds::SigmaScanner => {
                // get the send side of our states channel
                let stats_tx = &stats_channels.sigma_rules.0;
                // build a worker for sigma rules
                let worker =
                    EventWorker::<SigmaRuleWorker>::new(cache, thorium, &msg_rx, stats_tx).await;
                // spawn this sigma worker
                tokio::task::spawn(worker.start())
            }
        };
        (msg_tx, handle)
    }
}

/// A worker for some kind of event
///
/// This worker does not require mutable access to its internal state while working.
/// This allows it to process events in bulk and out of order using buffer_unordered.
pub struct EventWorker<W: EventWorkerSupport> {
    /// A client for Thorium
    thorium: Arc<Thorium>,
    /// A channel for messages to this worker from our controller
    msg_rx: AsyncReceiver<WorkerMsg>,
    /// A channel to send stats updates over for unified reporting across workers
    stats_tx: AsyncSender<W::Stats>,
    /// The timestamp to retry failed events if there are any
    retry_ts: Option<DateTime<Utc>>,
    /// The internal worker we are wrapping
    internal: W,
    /// This workers local stats
    stats: Arc<Mutex<W::Stats>>,
}

impl<W: EventWorkerSupport> EventWorker<W> {
    /// Create a new worker
    ///
    /// # Arguments
    ///
    /// * `cache` - The shared cache between workers of all kinds
    /// * `thorium` - A Thorium client
    /// * `msg_rx` - A channel to listen for messages from our controller on
    /// * `stats_tx` - A channel to send stats updates over
    pub async fn new(
        cache: &EventWorkerCache,
        thorium: &Arc<Thorium>,
        msg_rx: &AsyncReceiver<WorkerMsg>,
        stats_tx: &AsyncSender<W::Stats>,
    ) -> Self {
        // build our new internal worker
        let internal = W::new(cache, thorium).await;
        // build our worker wrapper
        EventWorker {
            thorium: thorium.clone(),
            msg_rx: msg_rx.clone(),
            stats_tx: stats_tx.clone(),
            retry_ts: None,
            internal,
            stats: Arc::new(Mutex::new(W::Stats::default())),
        }
    }

    /// Handle any messages from our controller
    async fn handle_messages(&mut self) -> Result<(), Error> {
        // try to get a message from our controller
        // this doesn't wait on the waitlist so we may not get messages immediately but eventually we will
        if let Some(msg) = self.msg_rx.try_recv()? {
            // Handle the message from our controller
            match msg {
                WorkerMsg::Update(update) => self.internal.apply_cache_update(update),
            }
        }
        Ok(())
    }

    /// The hot loop for an event handler worker
    ///
    /// This is its own function to allow us to easily trace it.
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for getting events
    #[instrument(name = "EventWorker::process", skip_all, err(Debug))]
    async fn process(&mut self, opts: &EventPopOpts) -> Result<bool, Error> {
        // check if we have any errored events to retry
        if let Some(retry_ts) = self.retry_ts {
            // check if its time to retry errors
            if Utc::now() > retry_ts {
                // log that we are resetting events
                event!(Level::INFO, msg = "Resetting in flight events");
                // reset any still in flight events
                self.thorium.events.reset_all(W::event_kind()).await?;
                // reset our retry timestamp
                self.retry_ts = None;
            }
        }
        // Try to get some events to handle
        let events = self.thorium.events.pop(W::event_kind(), opts).await?;
        // evaluate these events if we got any
        if events.is_empty() {
            // we got no events so return false
            Ok(false)
        } else {
            // we have some events so handle them
            let process_results = stream::iter(events)
                .map(|event| self.internal.process(event, &self.stats))
                .buffer_unordered(10)
                .collect::<Vec<Result<(), Error>>>()
                .await;
            // if any of our events failed then set our retry timestamp
            for result in process_results {
                // if we ran into an error log it and set our retry timer if its not already set
                if let Err(error) = result {
                    // log this error
                    event!(Level::ERROR, error = error.to_string());
                    // set our retry time if not already set
                    if self.retry_ts.is_none() {
                        // get a timestamp for 3 minutes in the future
                        let future_ts = Utc::now() + chrono::Duration::minutes(3);
                        // set the timestamp for when to retry these errors
                        self.retry_ts = Some(future_ts);
                    }
                }
            }
            Ok(true)
        }
    }

    /// Start scanning and handling events
    pub async fn start(mut self) -> Result<(), Error> {
        // Get at most 1000 events at a time
        let opts = EventPopOpts::default().limit(1000);
        // resest any tasks from a previous worker
        self.thorium.events.reset_all(W::event_kind()).await?;
        // keep looping and handling results
        loop {
            // check if we have any messages from our controller
            self.handle_messages().await?;
            // try to get any events and evaluate them
            let got_events = self.process(&opts).await?;
            // if we got some events then handle them otherwise sleep for 3 seconds
            if !got_events {
                // sleep for 3 seconds to keep from spamming the API needlessly
                tokio::time::sleep(Duration::from_secs(3)).await;
                // restart our loop and check for new events
                continue;
            } else {
                // swap our current stats out
                let stats = std::mem::take(&mut self.stats);
                // get our inner stats value and forward it to our stats worker
                match Arc::into_inner(stats) {
                    Some(stats_arc) => self.stats_tx.send(stats_arc.into_inner()).await?,
                    None => return Err(Error::new("Multiple Stats Arc refs?")),
                }
            }
        }
    }
}

/// The trait that the different kind of event workers need to handle
pub trait EventWorkerMutableSupport: EventWorkerSupportCore {
    /// The method to call to handle or process a single event
    ///
    /// # Arguments
    ///
    /// * `events` - The events to process
    /// * `stats` - The stats to track/update
    async fn process(
        &mut self,
        events: Vec<Event>,
        stats: &mut <Self as EventWorkerSupportCore>::Stats,
    ) -> Result<(), Error>;
}

/// A worker for some kind of event
///
/// This worker does require mutable access to its internal state while processing events.
/// This means events are passed in bulk to the processing method allowing the internal
/// worker/state to process things in bulk if possible.
pub struct EventWorkerMutable<W: EventWorkerMutableSupport> {
    /// A client for Thorium
    thorium: Arc<Thorium>,
    /// A channel for messages to this worker from our controller
    msg_rx: AsyncReceiver<WorkerMsg>,
    /// A channel to send stats updates over for unified reporting across workers
    stats_tx: AsyncSender<W::Stats>,
    /// The timestamp to retry failed events if there are any
    retry_ts: Option<DateTime<Utc>>,
    /// The internal worker we are wrapping
    internal: W,
    /// This workers local stats
    stats: W::Stats,
}

impl<W: EventWorkerMutableSupport> EventWorkerMutable<W> {
    /// Create a new worker
    ///
    /// # Arguments
    ///
    /// * `cache` - The shared cache between workers of all kinds
    /// * `thorium` - A Thorium client
    /// * `msg_rx` - A channel to listen for messages from our controller on
    /// * `stats_tx` - A channel to send stats updates over
    pub async fn new(
        cache: &EventWorkerCache,
        thorium: &Arc<Thorium>,
        msg_rx: &AsyncReceiver<WorkerMsg>,
        stats_tx: &AsyncSender<W::Stats>,
    ) -> Self {
        // build our new internal worker
        let internal = W::new(cache, thorium).await;
        // build our worker wrapper
        EventWorkerMutable {
            thorium: thorium.clone(),
            msg_rx: msg_rx.clone(),
            stats_tx: stats_tx.clone(),
            retry_ts: None,
            internal,
            stats: W::Stats::default(),
        }
    }

    /// Handle any messages from our controller
    async fn handle_messages(&mut self) -> Result<(), Error> {
        // try to get a message from our controller
        // this doesn't wait on the waitlist so we may not get messages immediately but eventually we will
        if let Some(msg) = self.msg_rx.try_recv()? {
            // Handle the message from our controller
            match msg {
                WorkerMsg::Update(update) => self.internal.apply_cache_update(update),
            }
        }
        Ok(())
    }

    /// The hot loop for an event handler worker
    ///
    /// This is its own function to allow us to easily trace it.
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for getting events
    #[instrument(name = "EventWorker::process", skip_all, err(Debug))]
    async fn process(&mut self, opts: &EventPopOpts) -> Result<bool, Error> {
        // check if we have any errored events to retry
        if let Some(retry_ts) = self.retry_ts {
            // check if its time to retry errors
            if Utc::now() > retry_ts {
                // log that we are resetting events
                event!(Level::INFO, msg = "Resetting in flight events");
                // reset any still in flight events
                self.thorium.events.reset_all(W::event_kind()).await?;
                // reset our retry timestamp
                self.retry_ts = None;
            }
        }
        // Try to get some events to handle
        let events = self.thorium.events.pop(W::event_kind(), opts).await?;
        // evaluate these events if we got any
        if events.is_empty() {
            // we got no events so return false
            Ok(false)
        } else {
            // process all of the events on this page
            self.internal.process(events, &mut self.stats).await?;
            Ok(true)
        }
    }

    /// Start scanning and handling events
    pub async fn start(mut self) -> Result<(), Error> {
        // Get at most 1000 events at a time
        let opts = EventPopOpts::default().limit(1000);
        // resest any tasks from a previous worker
        self.thorium.events.reset_all(W::event_kind()).await?;
        // keep looping and handling results
        loop {
            // check if we have any messages from our controller
            self.handle_messages().await?;
            // try to get any events and evaluate them
            let got_events = self.process(&opts).await?;
            // if we got some events then handle them otherwise sleep for 3 seconds
            if !got_events {
                // sleep for 3 seconds to keep from spamming the API needlessly
                tokio::time::sleep(Duration::from_secs(3)).await;
                // restart our loop and check for new events
                continue;
            } else {
                // wipe our current stats
                let stats = std::mem::take(&mut self.stats);
                // we processed some events so send our latest stats
                self.stats_tx.send(stats).await?;
            }
        }
    }
}
