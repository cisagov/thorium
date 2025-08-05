//! Logic for events

use std::hash::Hash;
use thorium::models::{SearchEvent, SearchEventType};
use uuid::Uuid;

mod results;
mod tags;

pub use results::CompactResultEvent;
pub use tags::CompactTagEvent;

/// An event that can be compacted
pub trait EventCompactable<C: From<Self>>: SearchEvent {
    /// The type to use to match on other "like" events
    type CompactBy: Hash + Eq + Clone;

    /// Returns the thing we're compacting on from `self`
    fn compact_by(&self) -> Self::CompactBy;
}

/// An event that is comprised of other "like" events
///
/// We compact events to avoid re-streaming data multiple times when
/// an item is modified many times at once
pub trait CompactEvent<E: SearchEvent>: From<E> {
    /// Append the given event to `self`
    ///
    /// # Arguments
    ///
    /// * `event` - The event to append
    fn append(&mut self, event: E);

    /// Returns the type of this compacted search event
    fn get_type(&self) -> SearchEventType;

    /// Returns all of the comprised events IDs
    fn get_ids(&self) -> Vec<Uuid>;
}
