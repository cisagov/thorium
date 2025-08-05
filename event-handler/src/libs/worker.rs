//! The workers for handling events in Thorium
use chrono::prelude::*;
use futures_locks::RwLock;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;
use thorium::models::{
    Event, EventData, EventIds, EventPopOpts, EventType, ReactionRequest, RepoDependencyRequest,
    TagType,
};
use thorium::{Error, Thorium};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::cache::{DataCache, FilteredEvents, TriggerCache};

/// A worker for handling events in Thorium
pub struct EventWorker {
    /// A shared Thorium client
    thorium: Arc<Thorium>,
    /// A shared trigger cache
    triggers: Arc<RwLock<TriggerCache>>,
    /// The kind of events to trigger
    kind: EventType,
    /// Track the total number of events handled
    total_seen: usize,
    /// Track the total number of reactions triggered by events
    total_triggered: usize,
    /// Track the total number of errors from creating reactions
    total_errors: usize,
    /// The timestamp to retry failed events if there are any
    retry_ts: Option<DateTime<Utc>>,
}

impl EventWorker {
    /// Create a new worker
    ///
    /// # Arguments
    ///
    /// * `thorium` - A thorium client
    /// * `triggers` - A cache of triggers to act on
    /// * `kind` - The kind of event worker to create
    pub fn new(
        thorium: &Arc<Thorium>,
        triggers: &Arc<RwLock<TriggerCache>>,
        kind: EventType,
    ) -> Self {
        EventWorker {
            thorium: thorium.clone(),
            triggers: triggers.clone(),
            kind,
            total_seen: 0,
            total_triggered: 0,
            total_errors: 0,
            retry_ts: None,
        }
    }

    /// Filter events that are guaranteed to not be able to
    ///
    /// # Arguments
    ///
    /// * `event_cache` - A cache of events
    /// * `cache` - A cache of trigger info
    /// * `events` - The events to filter
    #[instrument(name = "EventWorker::filter", skip_all)]
    pub fn filter<'a>(
        &self,
        event_cache: &mut HashMap<Uuid, Event>,
        cache: &'a TriggerCache,
        events: Vec<Event>,
    ) -> FilteredEvents<'a> {
        // create a struct for cleared events
        let mut filtered = FilteredEvents::with_capacity(50, 50, 1000);
        // crawl over and check if any events trigger anything
        cache.filter(event_cache, events, &mut filtered);
        filtered
    }

    /// Perform a final evaluation with all data for any still potential events
    ///
    /// # Arguments
    ///
    /// * `trigger_cache` - A cache of trigger info
    /// * `event_cache` - A cache of events
    /// * `data_cache` - A cache of data in Thorium
    /// * `filtered` - The events that were filtered in this loop
    #[instrument(name = "EventWorker::final_eval", skip_all, fields(clears = filtered.confirmed.len()))]
    fn final_eval<'a>(
        &self,
        trigger_cache: &TriggerCache,
        event_cache: &mut HashMap<Uuid, Event>,
        data_cache: &DataCache,
        filtered: &mut FilteredEvents<'a>,
    ) {
        // Iterate over all still potential events and try to confirm a triggers
        // conditions have been met
        for (event_id, triggers) in filtered.potentials.drain(..) {
            // keep a list of confirmed triggers if we find any
            let mut found = Vec::default();
            // get this events data
            let event = match event_cache.get(&event_id) {
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
            let user = match trigger_cache.users.get(&event.user) {
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
                if data_cache.check(user, &event, &trigger) {
                    // move this event to our found vec
                    found.push((group, pipeline, trigger));
                }
            }
            // add all of this events now confirmed triggers if we found any
            if !found.is_empty() {
                filtered.confirmed.push((event_id, found));
            }
            // add this to our cleared IDs list
            filtered.clears.push(event_id);
        }
    }

    /// Create any reactions for confirmed triggers and clear the rest
    ///
    /// # Arguments
    ///
    /// * `event_cache` - A cache of events
    /// * `filtered` - The events that were filtered in this loop
    #[instrument(name = "EventWorker::create", skip_all, fields(clears = filtered.confirmed.len()), err(Debug))]
    async fn create<'a>(
        &mut self,
        event_cache: &HashMap<Uuid, Event>,
        filtered: &mut FilteredEvents<'a>,
    ) -> Result<(), Error> {
        // build a list of reaction requests
        //let mut reqs = Vec::with_capacity(filtered.confirmed.len());
        let mut reqs = HashMap::with_capacity(10);
        // build reaction requests for all of our confirmed reactions
        for (id, triggers) in filtered.confirmed.drain(..) {
            // get the event data for this event id
            let event = match event_cache.get(&id) {
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
                        // add either a sample dependency or repo dependency based on tag type
                        match tag_type {
                            TagType::Files => req.sample(item),
                            TagType::Repos => req.repo(RepoDependencyRequest::new(item)),
                        }
                    }
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
            // if any errors occurred then log those
            for (_, error) in &resp.errors {
                event!(Level::ERROR, username, error);
            }
            // increment our stats
            self.total_triggered = self.total_triggered.saturating_add(resp.created.len());
            self.total_errors = self.total_errors.saturating_add(resp.errors.len());
        }
        Ok(())
    }

    /// Clear out any old events that didn't trigger anything
    ///
    /// # Arguments
    ///
    /// * `filtered` - The events that were filtered in this loop
    #[instrument(name = "EventWorker::clear", skip_all, fields(clears = filtered.clears.len()), err(Debug))]
    async fn clear<'a>(&self, filtered: FilteredEvents<'a>) -> Result<(), Error> {
        // build the list of event IDs to clear
        let mut event_ids = EventIds::from(filtered.clears);
        // add anything still in the potential list
        event_ids
            .ids
            .extend(filtered.potentials.into_iter().map(|(id, _)| id));
        // clear all requested events
        self.thorium.events.clear(self.kind, &event_ids).await?;
        Ok(())
    }

    /// Evaluate a page of events and spawn any triggers whose conditions have been met
    ///
    /// # Arguments
    ///
    /// * `event_cache` - A cache of events
    /// * `data_cache` - A cache of data in Thorium
    /// * `events` - The events to evaluate
    #[instrument(name = "EventWorker::evaluate", skip_all, fields(events = events.len()), err(Debug))]
    async fn evaluate(
        &mut self,
        event_cache: &mut HashMap<Uuid, Event>,
        data_cache: &mut DataCache,
        events: Vec<Event>,
    ) -> Result<(), Error> {
        // increment our total number of events seen
        self.total_seen = self.total_seen.saturating_add(events.len());
        // get lock to our trigger cache
        let lock = self.triggers.read().await;
        // get our trigger cache
        let cache = lock.deref();
        // split our events into potentials and events to clear
        let mut filtered = self.filter(event_cache, cache, events);
        // gather any required data in our data cache
        data_cache
            .gather(&self.thorium, &filtered, &event_cache, &mut self.retry_ts)
            .await?;
        // perform a final evaluation of all events with the new cached data
        self.final_eval(cache, event_cache, data_cache, &mut filtered);
        // create the reactions for this page of events
        self.create(&event_cache, &mut filtered).await?;
        // clear any events that did not trigger anything
        self.clear(filtered).await?;
        // drop the lock on our the trigger cache
        drop(lock);
        // clear our data and event cache
        data_cache.clear();
        event_cache.clear();
        // log the current worker stats
        event!(
            Level::INFO,
            seen = self.total_seen,
            triggered = self.total_triggered,
            errors = self.total_errors
        );
        Ok(())
    }

    /// The hot loop for an event handler worker
    ///
    /// This is its own function to allow us to easily trace it.
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for getting events
    /// * `event_cache` - A cache of events
    /// * `data_cache` - A cache of data in Thorium
    #[instrument(name = "EventWorker::hot_loop", skip_all, err(Debug))]
    async fn hot_loop(
        &mut self,
        opts: &EventPopOpts,
        event_cache: &mut HashMap<Uuid, Event>,
        data_cache: &mut DataCache,
    ) -> Result<bool, Error> {
        // check if we have any errored events to retry
        if let Some(retry_ts) = self.retry_ts {
            // check if its time to retry errors
            if Utc::now() > retry_ts {
                // log that we are resetting events
                event!(Level::INFO, msg = "Resetting in flight events");
                // reset any still in flight events
                self.thorium.events.reset_all(self.kind).await?;
                // reset our retry timestamp
                self.retry_ts = None;
            }
        }
        // Try to get some events to handle
        let events = self.thorium.events.pop(self.kind, opts).await?;
        // evaluate these events if we got any
        if events.is_empty() {
            // we got no events so return false
            Ok(false)
        } else {
            // we have some events so check if they are already in our cache
            self.evaluate(event_cache, data_cache, events).await?;
            Ok(true)
        }
    }

    /// Start scanning and handling events
    pub async fn start(mut self) -> Result<(), Error> {
        // Get at most 1000 events at a time
        let opts = EventPopOpts::default().limit(1000);
        // create a data cache object
        let mut data_cache = DataCache::default();
        let mut event_cache = HashMap::with_capacity(1000);
        // reset any tasks from a previous worker
        self.thorium.events.reset_all(self.kind).await?;
        // keep looping and handling results
        loop {
            // try to get any events and evaluate them
            let got_events = self
                .hot_loop(&opts, &mut event_cache, &mut data_cache)
                .await?;
            // if we got some events then handle them otherwise sleep for 3 seconds
            if got_events {
                // sleep for 3 seconds to keep from spamming the API needlessly
                tokio::time::sleep(Duration::from_secs(3)).await;
                // restart our loop and check for new events
                continue;
            }
        }
    }
}
