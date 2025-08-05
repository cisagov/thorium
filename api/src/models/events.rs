//! The events in Thorium for triggers and other things to act on

use chrono::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

use super::{InvalidEnum, TagType};

/// The different types of events
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum EventType {
    /// A trigger that may cause a reaction to be spawned
    ReactionTrigger,
}

impl fmt::Display for EventType {
    /// Cleanly print an event type
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EventType::ReactionTrigger => write!(f, "ReactionTrigger"),
        }
    }
}

impl EventType {
    /// Cast our event type to a str
    pub fn as_str(&self) -> &str {
        match self {
            EventType::ReactionTrigger => "ReactionTrigger",
        }
    }
}

impl From<&EventData> for EventType {
    /// Get the event type for some event data
    fn from(data: &EventData) -> Self {
        match data {
            &EventData::NewSample { .. } => EventType::ReactionTrigger,
            &EventData::NewTags { .. } => EventType::ReactionTrigger,
        }
    }
}

impl FromStr for EventType {
    type Err = InvalidEnum;

    /// Convert this str to an [`EventType`]
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "ReactionTrigger" => Ok(EventType::ReactionTrigger),
            _ => Err(InvalidEnum(format!("Unknown EventType: {raw}"))),
        }
    }
}

/// The data for an event
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreJson))]
pub enum EventData {
    /// A new sample was uploaded
    NewSample { groups: Vec<String>, sample: String },
    /// Some new tags were added
    NewTags {
        /// The type of tags that were created
        tag_type: TagType,
        /// The item that tags were added too
        item: String,
        /// The groups these tags were added too
        groups: Vec<String>,
        /// The new tags that were added
        tags: HashMap<String, HashSet<String>>,
    },
}

/// An request for a new event in Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EventRequest {
    /// The parent for this event
    pub parent: Uuid,
    /// this events data
    pub data: EventData,
}

/// Whether a trigger is, could be, or cannot be triggered
pub enum TriggerPotential {
    /// The triggers conditions are met by an event
    Confirmed,
    /// This trigger conditions could potentially be met by an event
    Potentially,
    /// This triggers conditions cannot be met by an event
    CanNot,
}

/// An event in Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Event {
    /// The id for this event
    pub id: Uuid,
    /// The timestamp for this event
    pub timestamp: DateTime<Utc>,
    /// The parent for this event
    pub parent: Option<Uuid>,
    /// The user that caused this trigger
    pub user: String,
    /// this events data
    pub data: EventData,
    /// This events current trigger depth,
    pub depth: u8,
}

impl Event {
    /// Check if all conditions of this tag trigger have been met or not
    ///
    /// Unlike check_tag_trigger this does not return a potentially but will instead yes or no
    pub fn check_all_tag_trigger(
        visible: &Vec<String>,
        tags: &HashMap<String, HashMap<String, HashSet<String>>>,
        required: &HashMap<String, Vec<String>>,
        not: &HashMap<String, Vec<String>>,
    ) -> bool {
        // check if any of our not values were set
        for (not_key, not_values) in not {
            // check if this not key was set
            if let Some(values) = tags.get(not_key) {
                // check if any of our not values were set
                if not_values
                    .iter()
                    .filter_map(|not_value| values.get(not_value))
                    .any(|groups| groups.iter().any(|group| visible.contains(group)))
                {
                    // this trigger will not trigger as it contains a not value
                    return false;
                }
            }
        }
        // check any of our required values were set
        for (req_key, req_values) in required {
            // check if this required key was set
            match tags.get(req_key) {
                // A required tag key exists
                Some(values) => {
                    // check if any of our required values were set
                    if !req_values
                        .iter()
                        .filter_map(|req| values.get(req))
                        .all(|groups| groups.iter().any(|group| visible.contains(group)))
                    {
                        // this trigger will not trigger as its missing a required tag value
                        return false;
                    }
                }
                // this trigger will not trigger as its missing a required tag key
                None => return false,
            }
        }
        // This triggers conditions have been fully met
        true
    }

    /// Check if a new tag event trigger occurred
    fn check_tag_trigger(
        new_type: &TagType,
        new_tags: &HashMap<String, HashSet<String>>,
        trigger_types: &Vec<TagType>,
        required: &HashMap<String, Vec<String>>,
        not: &HashMap<String, Vec<String>>,
    ) -> TriggerPotential {
        // make sure our tag types match
        if trigger_types.contains(new_type) {
            // check if any of our not values were set
            for (not_key, not_values) in not {
                // check if this not key was set
                if let Some(values) = new_tags.get(not_key) {
                    // check if any of our not values were set
                    if not_values
                        .iter()
                        .any(|not_value| values.contains(not_value))
                    {
                        // this trigger will not trigger
                        return TriggerPotential::CanNot;
                    }
                }
            }
            // check any of our required values were set
            for (req_key, req_values) in required {
                // check if this required key was set
                if let Some(values) = new_tags.get(req_key) {
                    // check if any of our required values were set
                    if req_values.iter().any(|req| values.contains(req)) {
                        // this trigger may potentially trigger
                        return TriggerPotential::Potentially;
                    }
                }
            }
        }
        // default to this trigger will not trigger
        TriggerPotential::CanNot
    }
    /// Check if this event could potentially trigger a trigger
    pub fn could_trigger(&self, trigger: &EventTrigger) -> TriggerPotential {
        match (&self.data, trigger) {
            (EventData::NewSample { .. }, EventTrigger::NewSample) => TriggerPotential::Confirmed,
            (
                EventData::NewTags { tag_type, tags, .. },
                EventTrigger::Tag {
                    tag_types,
                    required,
                    not,
                },
            ) => Self::check_tag_trigger(tag_type, tags, tag_types, required, not),
            (EventData::NewSample { .. }, EventTrigger::Tag { .. }) => TriggerPotential::CanNot,
            (EventData::NewTags { .. }, EventTrigger::NewSample) => TriggerPotential::CanNot,
        }
    }
}

/// A list of event IDs
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EventIds {
    /// A list of event IDs
    pub ids: Vec<Uuid>,
}

impl From<Vec<Uuid>> for EventIds {
    fn from(ids: Vec<Uuid>) -> Self {
        EventIds { ids }
    }
}

/// Default the event pop limit to 50
fn default_event_pop_limit() -> usize {
    50
}

/// The params for popping events
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EventPopOpts {
    /// The max number of events to pop and return
    #[serde(default = "default_event_pop_limit")]
    pub limit: usize,
}

impl Default for EventPopOpts {
    /// create a default EventPopOpts
    fn default() -> Self {
        EventPopOpts {
            limit: default_event_pop_limit(),
        }
    }
}

impl EventPopOpts {
    /// Set the maximum number of events to pop
    ///
    /// # Arguments
    ///
    /// * `limit` - The limit to set
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// The different kind of event triggers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum EventTrigger {
    /// A trigger based on an event
    Tag {
        /// The types of tags we can trigger on
        tag_types: Vec<TagType>,
        /// The tags to require to be set
        required: HashMap<String, Vec<String>>,
        /// The tags to not run on if set
        not: HashMap<String, Vec<String>>,
    },
    /// A trigger based on a new sample
    NewSample,
}

/// The current event marks
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EventMarks {
    /// The timestamp for oldest event to act on
    pub watermark: DateTime<Utc>,
    /// The timestamp for the currently newest event to act on
    pub tidemark: Option<DateTime<Utc>>,
}

/// The current status of our event cache
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EventCacheStatus {
    /// Whether we need to refresh our triggers cache
    pub triggers: bool,
}

/// The query params for getting the event cache status
#[derive(Deserialize, Serialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EventCacheStatusOpts {
    /// Whether to reset any cache statuses
    #[serde(default)]
    pub reset: bool,
}

impl EventCacheStatusOpts {
    /// Set the rest flag to true
    #[must_use]
    pub fn reset(mut self) -> Self {
        // set our reset flag to true
        self.reset = true;
        self
    }
}
