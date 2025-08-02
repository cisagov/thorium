//! Backend logic for result search events

use uuid::Uuid;

use super::SearchEventBackend;
use crate::models::backends::OutputSupport;
use crate::models::{ResultSearchEvent, SearchEventType};

impl SearchEventBackend for ResultSearchEvent {
    fn key() -> &'static str {
        "results"
    }

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn attempted(&mut self) -> u8 {
        self.attempts += 1;
        self.attempts
    }
}

impl ResultSearchEvent {
    /// Create a new result search event signalling results were modified
    ///
    /// # Arguments
    ///
    /// * `item` - The item whose results were modified
    /// * `groups` - The groups where results were modified
    #[must_use]
    pub fn modified<O: OutputSupport>(item: String, groups: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            attempts: 0,
            event_type: SearchEventType::Modified,
            result_kind: O::output_kind(),
            item,
            groups,
        }
    }

    /// Create a new result search event signalling the item was deleted
    ///
    /// # Arguments
    ///
    /// * `item` - The item that was deleted
    /// * `groups` - The groups where the item was deleted
    #[must_use]
    pub fn deleted<O: OutputSupport>(item: String, groups: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            attempts: 0,
            event_type: SearchEventType::Deleted,
            result_kind: O::output_kind(),
            item,
            groups,
        }
    }
}
