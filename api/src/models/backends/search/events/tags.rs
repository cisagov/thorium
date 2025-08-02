//! Backend logic for tag search events

use uuid::Uuid;

use super::SearchEventBackend;
use crate::models::backends::TagSupport;
use crate::models::{SearchEventType, TagSearchEvent};

impl SearchEventBackend for TagSearchEvent {
    fn key() -> &'static str {
        "tags"
    }

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn attempted(&mut self) -> u8 {
        self.attempts += 1;
        self.attempts
    }
}

impl TagSearchEvent {
    /// Create a new tag search event signalling tags were modified
    ///
    /// # Arguments
    ///
    /// * `item` - The item whose tags were modified
    /// * `groups` - The groups whose tags were modified
    #[must_use]
    pub fn modified<T: TagSupport>(item: String, groups: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            attempts: 0,
            event_type: SearchEventType::Modified,
            tag_type: T::tag_kind(),
            item,
            groups,
        }
    }

    /// Create a new tag search event signalling the item was deleted
    ///
    /// # Arguments
    ///
    /// * `item` - The item that was deleted
    /// * `groups` - The groups where the item was deleted
    #[must_use]
    pub fn deleted<T: TagSupport>(item: String, groups: Vec<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            attempts: 0,
            event_type: SearchEventType::Deleted,
            tag_type: T::tag_kind(),
            item,
            groups,
        }
    }
}
