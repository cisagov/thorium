//! Events for results

use std::collections::HashSet;
use thorium::models::{OutputKind, ResultSearchEvent, SearchEventType};
use uuid::Uuid;

use super::{CompactEvent, EventCompactable};

/// A compacted result event comprising one or more "like" events
/// (same item/output kind)
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CompactResultEvent {
    /// The id's of the events compacted
    pub ids: Vec<Uuid>,
    /// The type of search event this is
    pub event_type: SearchEventType,
    /// The type of item this event pertains to
    pub result_kind: OutputKind,
    /// The item whose results were edited
    pub item: String,
    /// The groups whose results were edited
    pub groups: HashSet<String>,
}

impl CompactEvent<ResultSearchEvent> for CompactResultEvent {
    fn append(&mut self, event: ResultSearchEvent) {
        // add this event's id to our list
        self.ids.push(event.id);
        // set the event type to whichever this event is
        //
        // since we're popping events in order from the API, we can be assured that
        // if an item is modified many times and then immediately deleted, the delete
        // event will come last; we can naively just set the event type to the latest
        // appended event
        self.event_type = event.event_type;
        // add the groups to groups superset
        self.groups.extend(event.groups);
    }

    fn get_type(&self) -> SearchEventType {
        self.event_type
    }

    fn get_ids(&self) -> Vec<Uuid> {
        self.ids.clone()
    }
}

impl EventCompactable<CompactResultEvent> for ResultSearchEvent {
    // compact by the item and the item's type
    type CompactBy = (String, OutputKind);

    fn compact_by(&self) -> Self::CompactBy {
        (self.item.clone(), self.result_kind)
    }
}

impl From<ResultSearchEvent> for CompactResultEvent {
    fn from(event: ResultSearchEvent) -> Self {
        Self {
            ids: vec![event.id],
            event_type: event.event_type,
            result_kind: event.result_kind,
            item: event.item,
            groups: event.groups.into_iter().collect(),
        }
    }
}
