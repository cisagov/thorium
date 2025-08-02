//! Events for tags

use std::collections::HashSet;
use thorium::models::{SearchEventType, TagSearchEvent, TagType};
use uuid::Uuid;

use super::{CompactEvent, EventCompactable};

/// A compacted result event comprising one or more "like" events
/// (same item/tag type)
pub struct CompactTagEvent {
    /// The id's of the events compacted
    pub ids: Vec<Uuid>,
    /// The type of search event this is
    pub event_type: SearchEventType,
    /// The type of item this event pertains to
    pub tag_type: TagType,
    /// The item whose tags were edited
    pub item: String,
    /// The groups whose tags were edited
    pub groups: HashSet<String>,
}

impl CompactEvent<TagSearchEvent> for CompactTagEvent {
    fn append(&mut self, event: TagSearchEvent) {
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

impl EventCompactable<CompactTagEvent> for TagSearchEvent {
    // compact by the item and the item's type
    type CompactBy = (String, TagType);

    fn compact_by(&self) -> Self::CompactBy {
        (self.item.clone(), self.tag_type)
    }
}

impl From<TagSearchEvent> for CompactTagEvent {
    fn from(event: TagSearchEvent) -> Self {
        CompactTagEvent {
            ids: vec![event.id],
            event_type: event.event_type,
            tag_type: event.tag_type,
            item: event.item,
            groups: event.groups.into_iter().collect(),
        }
    }
}
