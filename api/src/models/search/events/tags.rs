//! Models for tag search events

use uuid::Uuid;

use super::{SearchEvent, SearchEventType};
use crate::models::TagType;

/// An event relating to tags in the search store
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TagSearchEvent {
    /// The event's unique ID
    pub id: Uuid,
    /// The number of times we've attempted to stream this event
    pub attempts: u8,
    /// The type of search event this is
    pub event_type: SearchEventType,
    /// The type of item this event pertains to
    pub tag_type: TagType,
    /// The item whose tags were edited
    pub item: String,
    /// The groups whose tags were edited
    pub groups: Vec<String>,
}

impl SearchEvent for TagSearchEvent {
    /// Return a url component for this search event
    fn url() -> &'static str {
        "tags"
    }
}
