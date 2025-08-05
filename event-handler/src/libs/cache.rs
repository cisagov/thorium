//! A cache of pipeline trigger info

use chrono::{prelude::*, Duration};
use futures::stream::{self, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use thorium::models::{
    Event, EventData, EventTrigger, Repo, Sample, ScrubbedUser, TagType, TriggerPotential,
};
use thorium::{Error, Thorium};
use tracing::{event, instrument, Level};
use uuid::Uuid;

pub type EventsVec<'a> = Vec<(Uuid, Vec<(&'a String, &'a String, &'a EventTrigger)>)>;

#[derive(Debug, Default)]
pub struct FilteredEvents<'a> {
    /// Events that are confirmed to meet at least one triggers conditions
    pub confirmed: EventsVec<'a>,
    /// Events that have the potential to meet some triggers conditions
    pub potentials: EventsVec<'a>,
    /// Events that have been confirmed to not meet any trigger conditions
    pub clears: Vec<Uuid>,
}

impl<'a> FilteredEvents<'a> {
    /// Create a new filtered events with an initial capacity
    ///
    /// # Arguments
    ///
    /// * `confirmed_capacity` - The capacity to set for the confirmed vec
    /// * `potential_capacity` - The capacity to set for the potential vec
    /// * `clear_capacity` - The capacity to set for cleared events
    pub fn with_capacity(
        confirmed_capacity: usize,
        potential_capacity: usize,
        clear_capacity: usize,
    ) -> Self {
        FilteredEvents {
            confirmed: Vec::with_capacity(confirmed_capacity),
            potentials: Vec::with_capacity(potential_capacity),
            clears: Vec::with_capacity(clear_capacity),
        }
    }
}

/// The different triggers currently cached
pub struct TriggerCache {
    /// The users we know about
    pub users: HashMap<String, ScrubbedUser>,
    /// The triggers for our pipelines by group/pipeline
    pub triggers: HashMap<String, HashMap<String, HashMap<String, EventTrigger>>>,
    /// The max depth to check for new triggers at
    max_depth: u8,
}

impl TriggerCache {
    /// Rebuild the trigger portion of our cache
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for Thorium
    /// * `span` - The span to log traces under
    async fn get_triggers(
        thorium: &Arc<Thorium>,
    ) -> Result<HashMap<String, HashMap<String, HashMap<String, EventTrigger>>>, Error> {
        // assume we will have at least 10 groups
        let mut triggers = HashMap::with_capacity(10);
        // get a cursor for all groups we can see
        let mut groups_cursor = thorium.groups.list().page(100).exec().await?;
        // crawl over the groups in this cursor
        loop {
            // crawl over the groups on this page
            for group in groups_cursor.names.drain(..) {
                // get an entry to this groups pipeline map
                let group_entry: &mut HashMap<String, HashMap<String, EventTrigger>> =
                    triggers.entry(group.clone()).or_default();
                // build a cursor for the pipelines in this group
                let pipelines_cursor = thorium.pipelines.list(&group).details().exec().await?;
                // crawl over the pipelines in this group
                for pipeline in pipelines_cursor.details.into_iter() {
                    // skip any pipeline with no triggers
                    if !pipeline.triggers.is_empty() {
                        // get an entry to this pipelines triggers
                        let pipeline_entry = group_entry.entry(pipeline.name).or_default();
                        // clear any old triggers
                        pipeline_entry.clear();
                        // add the triggers to our map
                        pipeline_entry.extend(pipeline.triggers);
                    }
                }
            }
            // check if this cursor has been exhausted
            if groups_cursor.exhausted {
                break;
            }
            // get the next page of data
            groups_cursor.next().await?;
        }
        Ok(triggers)
    }

    /// Build a new trigger cache
    async fn get_users(thorium: &Arc<Thorium>) -> Result<HashMap<String, ScrubbedUser>, Error> {
        // get all users in Thorium
        let list = thorium.users.list_details().await?;
        // create a user map preallocated for the number of users we found
        let mut users = HashMap::with_capacity(list.len());
        // convert this user list into a map of users by name
        for user in list {
            // add this user to our map
            users.insert(user.username.clone(), user);
        }
        // get all pipelines in Thorium so we can
        Ok(users)
    }

    /// Build a new trigger cache
    pub async fn new(thorium: &Arc<Thorium>, max_depth: u8) -> Result<Self, Error> {
        // get all users in Thorium
        let users = Self::get_users(thorium).await?;
        // get all pipeline triggers in Thorium
        let triggers = Self::get_triggers(thorium).await?;
        // build a new trigger cache
        let cache = TriggerCache {
            users,
            triggers,
            max_depth,
        };
        Ok(cache)
    }

    /// check if this event could trigger any triggers
    fn check_event_helper<'a>(
        &'a self,
        user: &'a ScrubbedUser,
        event: &Event,
        filtered: &mut FilteredEvents<'a>,
    ) {
        // build a list of potential/confirmed triggers for this event
        let mut confirmed = Vec::default();
        let mut potential = Vec::default();
        // crawl over this users groups
        for group in &user.groups {
            // get this groups pipelines if any exist
            if let Some(pipeline_map) = self.triggers.get(group) {
                // check if any of these pipelines could potentially be triggered
                for (pipeline, triggers) in pipeline_map {
                    // check all of this pipelines triggers
                    for (_, trigger) in triggers {
                        // check if this triggers conditions could be potentially met
                        // This will filter out all true negative but triggers but
                        // could have false positives
                        match event.could_trigger(trigger) {
                            TriggerPotential::Confirmed => {
                                confirmed.push((group, pipeline, trigger))
                            }
                            TriggerPotential::Potentially => {
                                potential.push((group, pipeline, trigger))
                            }
                            TriggerPotential::CanNot => filtered.clears.push(event.id),
                        }
                    }
                }
            }
        }
        // check if we found any confirmed triggers or not
        if !confirmed.is_empty() {
            filtered.confirmed.push((event.id, confirmed))
        }
        // check if we found any confirmed triggers or not
        if !potential.is_empty() {
            filtered.potentials.push((event.id, potential))
        }
    }

    /// Filters a single event that does not meet at least some conditions for a trigger
    fn check_event<'a>(&'a self, event: &Event, filtered: &mut FilteredEvents<'a>) {
        // skip any events that are at their max depth
        if event.depth >= self.max_depth {
            // add this event to the clear list
            filtered.clears.push(event.id);
        }
        // get this users info
        match self.users.get(&event.user) {
            // get all the potential triggers for this event
            Some(user) => self.check_event_helper(user, event, filtered),
            // this user isn't in our cache for some reason so ignore this event for now
            None => {
                // log that we don't have this users info
                event!(Level::WARN, missing_user = &event.user);
                // add this event to the clear list
                filtered.clears.push(event.id);
            }
        };
    }

    /// Filter any events that will not hit any triggers
    pub fn filter<'a>(
        &'a self,
        event_cache: &mut HashMap<Uuid, Event>,
        events: Vec<Event>,
        filtered: &mut FilteredEvents<'a>,
    ) {
        // split out events up into potential events and events cannot meet any triggers
        for event in events {
            // check this event
            self.check_event(&event, filtered);
            // add this event to our event cache
            event_cache.insert(event.id, event);
        }
    }
}

/// The different futures for getting data for the data cache
enum DataCacheFuture {
    /// The info on a single file
    Files(Sample),
    /// The info on a single repo
    Repos(Repo),
}

impl DataCacheFuture {
    /// Get info about this events data
    pub async fn get(thorium: &Thorium, event: &Event) -> Result<Option<Self>, Error> {
        // get info on this events data
        match &event.data {
            EventData::NewTags { tag_type, item, .. } => {
                // get all info on this item
                let wrapped = match tag_type {
                    // get info on this file
                    TagType::Files => Self::Files(thorium.files.get(item).await?),
                    TagType::Repos => Self::Repos(thorium.repos.get(item).await?),
                };
                Ok(Some(wrapped))
            }
            _ => Ok(None),
        }
    }
}

/// The per event compaction loop cache of data
#[derive(Debug, Clone, Default)]
pub struct DataCache {
    /// The samples we have info about
    samples: HashMap<String, Sample>,
    /// the repos to get info on
    repos: HashMap<String, Repo>,
}

impl DataCache {
    /// Gather data on filtered events
    #[instrument(name = "EventWorker::gather", skip_all, fields(potentials = filtered.potentials.len()), err(Debug))]
    pub async fn gather<'a>(
        &mut self,
        thorium: &Thorium,
        filtered: &FilteredEvents<'a>,
        event_cache: &HashMap<Uuid, Event>,
        retry_ts: &mut Option<DateTime<Utc>>,
    ) -> Result<(), Error> {
        // A set of futures for our data requests
        let mut futures = Vec::default();
        // Spawn tasks to gather any data required for these events
        for (event_id, _) in &filtered.potentials {
            // get this events data
            match event_cache.get(event_id) {
                // get info on this item
                Some(event) => futures.push(DataCacheFuture::get(thorium, event)),
                None => event!(
                    Level::ERROR,
                    msg = "missing event",
                    event = event_id.to_string()
                ),
            }
        }
        // execute these futures 30 at a time
        let mut future_stream = stream::iter(futures).buffer_unordered(30);
        // get items from our future stream
        while let Some(get) = future_stream.next().await {
            // check if this request failed
            let get = match get {
                Ok(get) => get,
                Err(error) => {
                    // log this error
                    event!(
                        Level::ERROR,
                        msg = "DataCacheFuture::get failure",
                        error = error.to_string()
                    );
                    // set our retry timestamp for 3 minutes in the future if its not already set
                    if retry_ts.is_none() {
                        // get a timestamp for 3 minutes in the future
                        let future_ts = Utc::now() + Duration::minutes(3);
                        // set the timestamp for when to retry these errors
                        *retry_ts = Some(future_ts);
                    }
                    // continue to the next future
                    continue;
                }
            };
            // check if we got any data
            match get {
                Some(DataCacheFuture::Files(file)) => {
                    self.samples.insert(file.sha256.clone(), file);
                }
                Some(DataCacheFuture::Repos(repo)) => {
                    self.repos.insert(repo.url.clone(), repo);
                }
                None => (),
            }
        }
        Ok(())
    }

    /// Check an event and a trigger against the data in this cache
    #[instrument(name = "EventWorker::check", skip_all)]
    pub fn check(&self, user: &ScrubbedUser, event: &Event, trigger: &EventTrigger) -> bool {
        // handle each event type correctly
        match (&event.data, trigger) {
            // new sample is always true if this is a new sample trigger
            (EventData::NewSample { .. }, _) => trigger == &EventTrigger::NewSample,
            (
                EventData::NewTags { tag_type, item, .. },
                EventTrigger::Tag {
                    tag_types,
                    required,
                    not,
                },
            ) => {
                // skip any tag types that don't match
                if !tag_types.contains(tag_type) {
                    // this trigger is not valid for this events tag type
                    return false;
                }
                // get the right data from our cache and compare against it
                match tag_type {
                    TagType::Files => {
                        // try to get this file from our cache
                        let file = match self.samples.get(item) {
                            Some(file) => file,
                            None => {
                                // log that we are missing data
                                event!(
                                    Level::ERROR,
                                    missing_data = true,
                                    tag_type = "Files",
                                    sha256 = item
                                );
                                // return false since we are missing this data
                                return false;
                            }
                        };
                        // check against all the tags for this file
                        Event::check_all_tag_trigger(&user.groups, &file.tags, required, not)
                    }
                    TagType::Repos => {
                        // try to get this repo from our cache
                        let repo = match self.repos.get(item) {
                            Some(repo) => repo,
                            None => {
                                // log that we are missing data
                                event!(
                                    Level::ERROR,
                                    missing_data = true,
                                    tag_type = "Repos",
                                    url = item
                                );
                                // return false since we are missing this data
                                return false;
                            }
                        };
                        // check against all the tags for this repo
                        Event::check_all_tag_trigger(&user.groups, &repo.tags, required, not)
                    }
                }
            }
            (EventData::NewTags { .. }, EventTrigger::NewSample) => false,
        }
    }

    /// Empty this data cache
    pub fn clear(&mut self) {
        // empty our caches
        self.samples.clear();
        self.repos.clear();
    }
}
