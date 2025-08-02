//! Models for result search events

use uuid::Uuid;

use super::{SearchEvent, SearchEventType};
use crate::models::OutputKind;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ResultSearchEvent {
    /// The event's unique ID
    pub id: Uuid,
    /// The number of times we've attempted to stream this event
    pub attempts: u8,
    /// The type of search event this is
    pub event_type: SearchEventType,
    /// The type of item this event pertains to
    pub result_kind: OutputKind,
    /// The item whose results were edited
    pub item: String,
    /// The groups whose results were edited
    pub groups: Vec<String>,
}

impl SearchEvent for ResultSearchEvent {
    fn url() -> &'static str {
        "results"
    }
}
