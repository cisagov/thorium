//! A cache of pipeline trigger info

use chrono::{Duration, prelude::*};
use futures::stream::{self, StreamExt};
use futures_locks::RwLock;
use reqwest::StatusCode;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thorium::models::{
    Event, EventData, EventTrigger, Repo, Sample, ScrubbedUser, TagType, TriggerPotential, UserRole,
};
use thorium::{Error, Thorium};
use tracing::{Level, event, instrument};
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
#[derive(Clone)]
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
    async fn get_triggers(
        thorium: &Thorium,
    ) -> Result<HashMap<String, HashMap<String, HashMap<String, EventTrigger>>>, Error> {
        // assume we will have at least 10 groups
        let mut triggers = HashMap::with_capacity(10);
        // get a cursor for all groups we can see
        let mut groups_cursor = thorium.groups.list().page_size(100).exec().await?;
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
    ///
    /// # Arguments
    ///
    /// * `thorium` - a Thorium client
    async fn get_users(thorium: &Thorium) -> Result<HashMap<String, ScrubbedUser>, Error> {
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
    ///
    /// # Arguments
    ///
    /// * `thorium` - A thorium client
    /// * `max_depth` - The max depth of an event to handle
    pub async fn new(thorium: &Thorium, max_depth: u8) -> Result<Self, Error> {
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

    /// Decide how the max-depth guard in [`Self::check_event`] should act on an event
    ///
    /// It returns a `(clear, halt)` pair: `clear` is whether the event should be added to the
    /// clear list, and `halt` is whether trigger evaluation should stop for this event.
    ///
    /// An event at or beyond `max_depth` is both cleared and halted so it can never confirm
    /// a trigger and spawn a reaction at depth+1. This is the only loop breaker in the trigger
    /// pipeline, so cyclic trigger configurations rely on it to terminate.
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth of the event being checked
    /// * `max_depth` - The max depth an event is allowed to reach
    fn max_depth_guard(depth: u8, max_depth: u8) -> (bool, bool) {
        // mark the event to be cleared if it is at or beyond its max depth
        let clear = depth >= max_depth;
        // a depth-capped event must not be evaluated any further or it could still fire
        let halt = clear;
        // hand back both decisions
        (clear, halt)
    }

    /// Decide whether an event that matched no triggers should be cleared
    ///
    /// It returns whether the event should be added to the clear list once all of a user's
    /// triggers have been checked. An event that matched no triggers at all (both lists are
    /// empty) lands in neither `confirmed` nor `potentials`, so it must be cleared here or it
    /// would leak in the in-flight queue forever. Events that did match a trigger are cleared
    /// later once their confirmed/potential triggers are handled.
    ///
    /// # Arguments
    ///
    /// * `confirmed_empty` - Whether no triggers were confirmed for this event
    /// * `potential_empty` - Whether no triggers were potentially met for this event
    fn should_clear_no_matches(confirmed_empty: bool, potential_empty: bool) -> bool {
        // only clear events that matched nothing since matched events are cleared later
        confirmed_empty && potential_empty
    }

    /// check if this event could trigger any triggers
    fn check_event_helper<'a>(
        &'a self,
        user: &'a ScrubbedUser,
        event: &Event,
        filtered: &mut FilteredEvents<'a>,
    ) {
        // build a list of potential/confirmed triggers for this eventi
        let mut confirmed = Vec::default();
        let mut potential = Vec::default();
        // crawl over this users groups
        for group in &user.groups {
            // get this groups pipelines if any exist
            if let Some(pipeline_map) = self.triggers.get(group) {
                // check if any of these pipelines could potentially be triggered
                for (pipeline, triggers) in pipeline_map {
                    // check all of this pipelines triggers
                    for trigger in triggers.values() {
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
                            // this trigger can never fire for this event, but other
                            // triggers still might so don't clear this event yet
                            TriggerPotential::CanNot => {}
                        }
                    }
                }
            }
        }
        // capture whether we matched anything before we move the lists into filtered
        let confirmed_empty = confirmed.is_empty();
        let potential_empty = potential.is_empty();
        // check if we found any confirmed triggers or not
        if !confirmed.is_empty() {
            filtered.confirmed.push((event.id, confirmed))
        }
        // check if we found any confirmed triggers or not
        if !potential.is_empty() {
            filtered.potentials.push((event.id, potential))
        }
        // clear any event that matched no triggers so it does not leak
        if Self::should_clear_no_matches(confirmed_empty, potential_empty) {
            // add this event to the clear list so it does not leak in the in-flight queue
            filtered.clears.push(event.id);
        }
    }

    /// Filters a single event that does not meet at least some conditions for a trigger
    ///
    /// Events from disabled users are skipped and cleared since acting on
    /// their behalf would only fail.
    fn check_event<'a>(&'a self, event: &Event, filtered: &mut FilteredEvents<'a>) {
        // check if this event is at its max depth
        let (clear, halt) = Self::max_depth_guard(event.depth, self.max_depth);
        // skip any events that are at their max depth
        if clear {
            // add this event to the clear list
            filtered.clears.push(event.id);
        }
        // stop evaluating this event if the guard says to halt
        if halt {
            // this depth-capped event should fire nothing, so return early
            return;
        }
        // get this users info
        match self.users.get(&event.user) {
            // skip events from disabled users since masquerading as them would only fail
            Some(user) if user.role == UserRole::Disabled => {
                // log that we are skipping this disabled users event
                event!(Level::INFO, disabled_user = &event.user, skipped = true);
                // add this event to the clear list so it gets popped from the queue
                filtered.clears.push(event.id);
            }
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
        // split out events up into potential events and events can not meet any triggers
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
    /// A file that Thorium reported does not exist
    MissingFile(String),
    /// A repo that Thorium reported does not exist
    MissingRepo(String),
}

impl DataCacheFuture {
    /// Check if an error means the data we requested does not exist
    ///
    /// Retrying a request for data that does not exist will never succeed.
    ///
    /// # Arguments
    ///
    /// * `error` - The error to check
    fn is_not_found(error: &Error) -> bool {
        error.status() == Some(StatusCode::NOT_FOUND)
    }

    /// Get info about this events data
    pub async fn get(thorium: &Thorium, event: &Event) -> Result<Option<Self>, Error> {
        // get info on this events data
        match &event.data {
            EventData::NewTags { tag_type, item, .. } => {
                // get all info on this item
                match tag_type {
                    // get info on this file
                    TagType::Files => match thorium.files.get(item).await {
                        Ok(file) => Ok(Some(Self::Files(file))),
                        // this file doesn't exist so remember that instead of retrying
                        Err(error) if Self::is_not_found(&error) => {
                            Ok(Some(Self::MissingFile(item.clone())))
                        }
                        Err(error) => Err(error),
                    },
                    // get info on this repo
                    TagType::Repos => match thorium.repos.get(item).await {
                        Ok(repo) => Ok(Some(Self::Repos(repo))),
                        // this repo doesn't exist so remember that instead of retrying
                        Err(error) if Self::is_not_found(&error) => {
                            Ok(Some(Self::MissingRepo(item.clone())))
                        }
                        Err(error) => Err(error),
                    },
                    // we can never get data for unsupported tag types so retrying won't help
                    _ => {
                        // log that this tag type is not supported
                        event!(
                            Level::WARN,
                            msg = "Events are not supported for tag type",
                            tag_type = tag_type.as_str()
                        );
                        Ok(None)
                    }
                }
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
    /// The samples that Thorium reported do not exist
    missing_samples: HashSet<String>,
    /// The repos that Thorium reported do not exist
    missing_repos: HashSet<String>,
}

impl DataCache {
    /// Gather data on filtered events
    #[instrument(name = "EventWorker::gather", skip_all, fields(potentials = filtered.potentials.len()), err(Debug))]
    pub async fn gather<'a>(
        &mut self,
        thorium: &Thorium,
        filtered: &FilteredEvents<'a>,
        event_cache: &HashMap<Uuid, Event>,
        retry_ts: &Arc<RwLock<Option<DateTime<Utc>>>>,
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
                    // so any events left in flight because of this error get retried
                    if retry_ts.read().await.is_none() {
                        // get a timestamp for 3 minutes in the future
                        let future_ts = Utc::now() + Duration::minutes(3);
                        // set the timestamp for when to retry these errors
                        *retry_ts.write().await = Some(future_ts);
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
                Some(DataCacheFuture::MissingFile(sha256)) => {
                    self.missing_samples.insert(sha256);
                }
                Some(DataCacheFuture::MissingRepo(url)) => {
                    self.missing_repos.insert(url);
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
                    // this is an unsupported tag type that we can never get data for, so it
                    // can never meet this trigger
                    _ => false,
                }
            }
            (EventData::NewTags { .. }, EventTrigger::NewSample) => false,
            (EventData::SigmaScannableResults { .. }, _) => false,
        }
    }

    /// Check whether this cache has all the data it will ever get for an event
    ///
    /// Only `NewTags` events need extra data; every other event kind never misses data here.
    /// Data that Thorium reported does not exist and tag types we can never get data for
    /// count as present, since retrying will never find them. So this only returns false
    /// when an event's data is missing because of an error that a retry may fix.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to check for data
    pub fn has_data(&self, event: &Event) -> bool {
        match &event.data {
            // tag events need their sample or repo to be in our cache or known to not exist
            EventData::NewTags { tag_type, item, .. } => match tag_type {
                TagType::Files => {
                    self.samples.contains_key(item) || self.missing_samples.contains(item)
                }
                TagType::Repos => {
                    self.repos.contains_key(item) || self.missing_repos.contains(item)
                }
                // unsupported tag types never have data to gather so retrying won't help
                _ => true,
            },
            // other events need no extra data
            _ => true,
        }
    }

    /// Empty this data cache
    pub fn clear(&mut self) {
        // empty our caches
        self.samples.clear();
        self.repos.clear();
        self.missing_samples.clear();
        self.missing_repos.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thorium::models::UserSettings;

    /// Build a scrubbed user for testing trigger cache filtering
    ///
    /// # Arguments
    ///
    /// * `username` - The username for this test user
    /// * `role` - The role for this test user
    fn test_user(username: &str, role: UserRole) -> ScrubbedUser {
        ScrubbedUser {
            username: username.to_owned(),
            role,
            email: "fake@fake.gov".to_owned(),
            groups: Vec::default(),
            actual_groups: Vec::default(),
            token: "token".to_owned(),
            token_expiration: Utc::now(),
            unix: None,
            settings: UserSettings::default(),
            local: true,
            verified: true,
            has_image: false,
        }
    }

    /// Build a new sample event for a user
    ///
    /// # Arguments
    ///
    /// * `user` - The username this event should come from
    fn test_event(user: &str) -> Event {
        Event {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            parent: None,
            user: user.to_owned(),
            data: EventData::NewSample {
                groups: Vec::default(),
                sample: "sample".to_owned(),
            },
            depth: 0,
        }
    }

    #[test]
    fn check_event_skips_disabled_users() {
        // build a cache with one enabled and one disabled user and no triggers
        let mut users = HashMap::default();
        users.insert("enabled".to_owned(), test_user("enabled", UserRole::User));
        users.insert(
            "disabled".to_owned(),
            test_user("disabled", UserRole::Disabled),
        );
        let cache = TriggerCache {
            users,
            triggers: HashMap::default(),
            max_depth: 5,
        };
        // check an event from our disabled user
        let disabled_event = test_event("disabled");
        let mut filtered = FilteredEvents::default();
        cache.check_event(&disabled_event, &mut filtered);
        // the disabled users event should be cleared and trigger nothing
        assert_eq!(filtered.clears, vec![disabled_event.id]);
        assert!(filtered.confirmed.is_empty());
        assert!(filtered.potentials.is_empty());
        // check an event from our enabled user
        let enabled_event = test_event("enabled");
        let mut filtered = FilteredEvents::default();
        cache.check_event(&enabled_event, &mut filtered);
        // the enabled users event is evaluated and cleared since it matches no triggers
        assert_eq!(filtered.clears, vec![enabled_event.id]);
    }

    /// The max-depth guard is the only loop-breaker in the whole trigger pipeline, so for an
    /// event at or beyond `max_depth`, `max_depth_guard` reports both `clear` and `halt`. This
    /// makes [`TriggerCache::check_event`] return before ever reaching trigger evaluation, so the
    /// event cannot confirm a trigger, spawn a reaction at depth+1, and let cyclic trigger
    /// configurations recurse forever.
    #[test]
    fn max_depth_guard_halts_evaluation_at_max_depth() {
        // an event exactly at the max depth must be cleared AND halt further evaluation
        let (clear, halt) = TriggerCache::max_depth_guard(5, 5);
        // the depth-capped event is added to the clear list
        assert!(
            clear,
            "an event at max depth should be added to the clear list"
        );
        // and evaluation must stop so the event cannot confirm a trigger and spawn a reaction
        assert!(
            halt,
            "an event at max depth must halt evaluation so it cannot fire a trigger"
        );
        // an event beyond the max depth must likewise halt
        let (clear, halt) = TriggerCache::max_depth_guard(6, 5);
        // it is cleared as well
        assert!(clear, "an event beyond max depth should be cleared");
        // and it too must halt evaluation
        assert!(halt, "an event beyond max depth must halt evaluation");
    }

    /// An event below its max depth is still evaluated normally, so the guard neither clears it
    /// nor halts it.
    #[test]
    fn max_depth_guard_allows_shallow_events() {
        // an event well below the max depth should be neither cleared nor halted
        let (clear, halt) = TriggerCache::max_depth_guard(0, 5);
        // it is not added to the clear list
        assert!(
            !clear,
            "a shallow event should not be cleared by the depth guard"
        );
        // and evaluation continues so it can match triggers
        assert!(!halt, "a shallow event should not halt evaluation");
    }

    /// When an event matched no triggers (both confirmed and potential are empty),
    /// `should_clear_no_matches` reports it should be cleared. Such an event lands in neither
    /// confirmed nor potentials, so clearing it here is what lets
    /// [`ReactionTriggerWorker::clear`] remove it instead of leaking it in the Redis in-flight
    /// map/queue forever.
    #[test]
    fn events_matching_no_triggers_are_cleared() {
        // an event with no confirmed and no potential triggers matched nothing at all
        let should_clear = TriggerCache::should_clear_no_matches(true, true);
        // it must be cleared so it does not accumulate in the in-flight queue forever
        assert!(
            should_clear,
            "an event that matched no triggers must be cleared to avoid a Redis leak"
        );
    }

    /// An event that did match a trigger (confirmed or potential non-empty) is not cleared by
    /// this path, since it is handled by the confirmed/potential lists instead.
    #[test]
    fn events_matching_some_trigger_are_not_cleared_here() {
        // an event with a confirmed trigger is handled elsewhere, so this path should not clear it
        assert!(
            !TriggerCache::should_clear_no_matches(false, true),
            "an event with a confirmed trigger should not be cleared by the no-match path"
        );
        // an event with a potential trigger is likewise handled elsewhere
        assert!(
            !TriggerCache::should_clear_no_matches(true, false),
            "an event with a potential trigger should not be cleared by the no-match path"
        );
    }

    /// Driving [`TriggerCache::check_event`] with an event from an enabled user whose groups have
    /// no triggers at all adds that event to the clear list, so an event that matched no
    /// triggers never leaks in the Redis in-flight queue.
    #[test]
    fn check_event_clears_event_matching_no_triggers() {
        // build a cache with one enabled user and no triggers
        let mut users = HashMap::default();
        users.insert("enabled".to_owned(), test_user("enabled", UserRole::User));
        let cache = TriggerCache {
            users,
            triggers: HashMap::default(),
            max_depth: 5,
        };
        // check an event from our enabled user
        let event = test_event("enabled");
        let mut filtered = FilteredEvents::default();
        cache.check_event(&event, &mut filtered);
        // the event matched nothing so it must be cleared
        assert_eq!(
            filtered.clears,
            vec![event.id],
            "an event that matched no triggers must be cleared to avoid a Redis leak"
        );
    }

    /// Build a trigger cache whose single enabled user is in a group with a `NewSample` trigger
    fn new_sample_trigger_cache() -> TriggerCache {
        // build an enabled user in the corn group
        let mut user = test_user("enabled", UserRole::User);
        user.groups = vec!["corn".to_owned()];
        let mut users = HashMap::default();
        users.insert("enabled".to_owned(), user);
        // give the corn group a pipeline triggered by new samples
        let mut pipeline_triggers = HashMap::default();
        pipeline_triggers.insert("on-new-sample".to_owned(), EventTrigger::NewSample);
        let mut pipelines = HashMap::default();
        pipelines.insert("harvest".to_owned(), pipeline_triggers);
        let mut triggers = HashMap::default();
        triggers.insert("corn".to_owned(), pipelines);
        TriggerCache {
            users,
            triggers,
            max_depth: 5,
        }
    }

    /// An event at its max trigger depth confirms no trigger (not even a matching `NewSample`
    /// trigger) when driven through [`TriggerCache::check_event`], so no reaction is spawned at
    /// depth+1 and cyclic trigger configurations cannot recurse forever.
    #[test]
    fn check_event_at_max_depth_confirms_nothing() {
        let cache = new_sample_trigger_cache();
        // an event exactly at the max depth
        let mut event = test_event("enabled");
        event.depth = 5;
        let mut filtered = FilteredEvents::default();
        cache.check_event(&event, &mut filtered);
        // the depth-capped event must fire nothing
        assert!(
            filtered.confirmed.is_empty(),
            "an event at max depth must not confirm any trigger"
        );
    }

    /// The same event below its max depth confirms the `NewSample` trigger when driven through
    /// [`TriggerCache::check_event`].
    #[test]
    fn check_event_below_max_depth_confirms_trigger() {
        let cache = new_sample_trigger_cache();
        // a shallow event
        let event = test_event("enabled");
        let mut filtered = FilteredEvents::default();
        cache.check_event(&event, &mut filtered);
        // the trigger is confirmed for this event
        assert_eq!(filtered.confirmed.len(), 1);
    }
}
