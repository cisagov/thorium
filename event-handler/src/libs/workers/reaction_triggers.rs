//! The workers for handling reaction trigger events in Thorium
use chrono::prelude::*;
use futures_locks::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use thorium::models::{
    Event, EventData, EventIds, EventType, ReactionRequest, RepoDependencyRequest, TagType,
};
use thorium::{Error, Thorium};
use tracing::{Level, event, instrument};
use uuid::Uuid;

use crate::libs::EventWorkerCache;
use crate::libs::cache::{DataCache, FilteredEvents, TriggerCache};
use crate::libs::workers::{
    EventWorkerMutableSupport, EventWorkerSupportCore, WorkerCacheUpdateKinds, WorkerCacheUpdates,
};

/// The stats for reaction triggers
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct ReactionTriggerStats {
    /// Track the total number of events handled
    seen: usize,
    /// Track the total number of reactions triggered by events
    triggered: usize,
    /// Track the total number of errors from creating reactions
    errors: usize,
}

impl std::ops::AddAssign for ReactionTriggerStats {
    fn add_assign(&mut self, rhs: Self) {
        self.seen = self.seen.saturating_add(rhs.seen);
        self.triggered = self.triggered.saturating_add(rhs.triggered);
        self.errors = self.errors.saturating_add(rhs.errors);
    }
}

/// A worker for handling reaction trigger events in Thorium
pub struct ReactionTriggerWorker {
    /// A shared Thorium client
    thorium: Arc<Thorium>,
    /// A shared trigger cache
    triggers: TriggerCache,
    /// The timestamp to retry failed events if there are any
    retry_ts: Arc<RwLock<Option<DateTime<Utc>>>>,
    /// A cache of events this worker is processing
    event_cache: HashMap<Uuid, Event>,
    /// A cache of data for the events this worker is processing
    data_cache: DataCache,
}

impl ReactionTriggerWorker {
    /// Perform a final evaluation with all data for any still potential events
    ///
    /// # Arguments
    ///
    /// * `filtered` - The events that were filtered in this loop
    #[instrument(name = "ReactionTriggerWorker::final_eval", skip_all, fields(clears = filtered.confirmed.len()))]
    fn final_eval<'a>(&self, filtered: &mut FilteredEvents<'a>) {
        // Iterate over all still potential events and try to confirm a triggers
        // conditions have been met
        for (event_id, triggers) in filtered.potentials.drain(..) {
            // keep a list of confirmed triggers if we find any
            let mut found = Vec::default();
            // get this events data
            let event = match self.event_cache.get(&event_id) {
                Some(event) => event,
                None => {
                    // log this event
                    event!(
                        Level::ERROR,
                        msg = "Missing event",
                        event = event_id.to_string()
                    );
                    // continue our loop
                    continue;
                }
            };
            // get this events user
            let user = match self.triggers.users.get(&event.user) {
                Some(user) => user,
                None => {
                    // log this event
                    event!(Level::ERROR, msg = "Missing user", user = &event.user);
                    // continue our loop
                    continue;
                }
            };
            // check each still potential trigger for this event
            for (group, pipeline, trigger) in triggers {
                // check if this triggers conditions were met with extra data in our cache
                if self.data_cache.check(user, event, trigger) {
                    // move this event to our found vec
                    found.push((group, pipeline, trigger));
                }
            }
            // add all of this events now confirmed triggers if we found any
            if !found.is_empty() {
                filtered.confirmed.push((event_id, found));
            }
            // add this to our cleared ids list
            filtered.clears.push(event_id);
        }
    }

    /// Create any reactions for confirmed triggers and clear the rest
    ///
    /// # Arguments
    ///
    /// * `filtered` - The events that were filtered in this loop
    /// * `stats` - The stats for this worker kind
    #[instrument(name = "ReactionTriggerWorker::create", skip_all, fields(clears = filtered.confirmed.len()), err(Debug))]
    async fn create<'a>(
        &self,
        filtered: &mut FilteredEvents<'a>,
        stats: &mut ReactionTriggerStats,
    ) -> Result<(), Error> {
        // build a list of reaction requests
        let mut reqs = HashMap::with_capacity(10);
        // build reaction requests for all of our confirmed reactions
        for (id, triggers) in filtered.confirmed.drain(..) {
            // get the event data for this event id
            let event = match self.event_cache.get(&id) {
                Some(event) => event,
                None => {
                    // log that we are missing an event and continue
                    event!(Level::ERROR, missing = true, id = id.to_string());
                    // add this missing event to our clear list
                    filtered.clears.push(id);
                    // continue checking events
                    continue;
                }
            };
            // get the new depth for this event
            let depth = event.depth + 1;
            // create reactions for each of the confirmed triggers from this event
            for (group, pipeline, _) in triggers {
                // build the base reaction request for this trigger
                let req = ReactionRequest::new(group, pipeline).trigger_depth(depth);
                // add our dependency info
                let req = match &event.data {
                    EventData::NewSample { sample, .. } => req.sample(sample),
                    EventData::NewTags { tag_type, item, .. } => {
                        // add either a sample dependency or repo dependency basd on tag type
                        match tag_type {
                            TagType::Files => req.sample(item),
                            TagType::Repos => req.repo(RepoDependencyRequest::new(item)),
                            // should be impossible to get an unsupported tag type because we error
                            // when getting the event's data, but handle the error here anyway
                            _ => {
                                return Err(Error::new(format!(
                                    "Events are not supported for tag type '{tag_type}'"
                                )));
                            }
                        }
                    }
                    EventData::SigmaScannableResults { .. } => continue,
                };
                // get an entry to this users reaction requests
                let entry: &mut Vec<ReactionRequest> = reqs.entry(event.user.clone()).or_default();
                // add this users reaction request
                entry.push(req);
            }
            // also add this to our clear list
            filtered.clears.push(id);
        }
        // create our reactions by user
        let creates = self.thorium.reactions.create_bulk_by_user(&reqs).await?;
        // log the reactions we created
        for (username, resp) in &creates {
            // log the reactions we created
            event!(Level::INFO, username, created = resp.created.len());
            // if any errors occured then log those
            for error in resp.errors.values() {
                event!(Level::ERROR, username, error);
            }
            // increment our stats
            stats.triggered = stats.triggered.saturating_add(resp.created.len());
            stats.errors = stats.errors.saturating_add(resp.errors.len());
        }
        Ok(())
    }

    /// Clear out any old events that didn't trigger anything
    ///
    /// # Arguments
    ///
    /// * `filtered` - The events that were filtered in this loop
    #[instrument(name = "ReactionTriggerWorker::clear", skip_all, fields(clears = filtered.clears.len()), err(Debug))]
    async fn clear<'a>(&self, filtered: &mut FilteredEvents<'a>) -> Result<(), Error> {
        // build the list of event ids to clear
        let mut event_ids = EventIds::from(filtered.clears.drain(..));
        // add anything still in the potential list
        event_ids
            .ids
            .extend(filtered.potentials.drain(..).map(|(id, _)| id));
        // clear all requested events
        self.thorium
            .events
            .clear(EventType::ReactionTrigger, &event_ids)
            .await?;
        Ok(())
    }

    /// Evaluate a page of events and spawn any triggers whose conditions have been met
    ///
    /// # Arguments
    ///
    /// * `events` - The events to evaluate
    /// * `stats` - The stats for this worker kind
    #[instrument(name = "ReactionTriggerWorker::evaluate", skip_all, fields(events = events.len()), err(Debug))]
    async fn evaluate(
        &mut self,
        events: Vec<Event>,
        stats: &mut ReactionTriggerStats,
    ) -> Result<(), Error> {
        // increment our total number of events seen
        stats.seen = stats.seen.saturating_add(events.len());
        // create a struct for cleared events
        let mut filtered = FilteredEvents::with_capacity(50, 50, 1000);
        // crawl over and check if any events trigger anything
        self.triggers
            .filter(&mut self.event_cache, events, &mut filtered);
        // gather any required data in our data cache
        self.data_cache
            .gather(&self.thorium, &filtered, &self.event_cache, &self.retry_ts)
            .await?;
        // perform a final evaluation of all events with the new cached data
        self.final_eval(&mut filtered);
        // create the reactions for this page of events
        self.create(&mut filtered, stats).await?;
        // clear any events that did not trigger anything
        self.clear(&mut filtered).await?;
        // clear our data and event cache
        self.data_cache.clear();
        self.event_cache.clear();
        Ok(())
    }
}

impl EventWorkerSupportCore for ReactionTriggerWorker {
    /// The stats for all workers of this kind
    type Stats = ReactionTriggerStats;

    /// Build this specific kind of worker
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of all info a worker may need to be created
    /// * `thorium` - A Thorium client
    async fn new(cache: &EventWorkerCache, thorium: &Arc<Thorium>) -> Self {
        // build this worker
        ReactionTriggerWorker {
            thorium: thorium.clone(),
            triggers: cache.triggers.clone(),
            retry_ts: Arc::new(RwLock::new(None)),
            event_cache: HashMap::with_capacity(1000),
            data_cache: DataCache::default(),
        }
    }

    /// The type of events this worker can handle
    #[inline]
    fn event_kind() -> EventType {
        EventType::ReactionTrigger
    }

    /// The types of cache updates this worker subscribes too
    #[inline]
    fn cache_subscriptions() -> Vec<WorkerCacheUpdateKinds> {
        vec![
            WorkerCacheUpdateKinds::ReactionTriggers,
            WorkerCacheUpdateKinds::Users,
        ]
    }

    /// Apply an update to our local worker cache
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this workers cache
    fn apply_cache_update(&mut self, update: WorkerCacheUpdates) {
        // Apply an updates
        match update {
            WorkerCacheUpdates::ReactionTriggers(triggers) => self.triggers = triggers,
            WorkerCacheUpdates::Users(users) => self.triggers.users = users,
            WorkerCacheUpdates::SigmaRulesContext(_) => (),
        }
    }

    /// Report our latest stats in an event
    ///
    /// This will only report new/changed stats.
    ///
    /// # Arguments
    ///
    /// * `stats` - The new stats to log
    /// * `prior` - The previously reported stats
    #[instrument(name = "EventWorkerSupportCore::stats", skip_all)]
    fn stats(stats: &Self::Stats, _: &Self::Stats) {
        // our trigger stats are very simplistic so there is no need to check against
        // any prior stats here as the controller prevents us from reemiting the
        // same stats
        event!(
            Level::INFO,
            worker = "ReactionTriggerWorker",
            seen = stats.seen,
            triggered = stats.triggered,
            errors = stats.errors
        );
    }
}

impl EventWorkerMutableSupport for ReactionTriggerWorker {
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
    ) -> Result<(), Error> {
        self.evaluate(events, stats).await
    }
}
