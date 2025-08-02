//! Models for search events

use uuid::Uuid;

use serde::{Deserialize, Serialize};

mod results;
mod tags;

pub use results::ResultSearchEvent;
pub use tags::TagSearchEvent;

/// A search event
pub trait SearchEvent: Sized + Serialize + for<'de> Deserialize<'de> {
    /// Return a url component for this search event
    fn url() -> &'static str;
}

/// The type of search events that can occur
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum SearchEventType {
    /// The item was modified
    Modified,
    /// The item was deleted
    Deleted,
}

/// Default the event pop limit to 300
fn default_search_event_pop_limit() -> usize {
    300
}

/// The params for popping events
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SearchEventPopOpts {
    /// The max number of events to pop and return
    #[serde(default = "default_search_event_pop_limit")]
    pub limit: usize,
}

impl Default for SearchEventPopOpts {
    fn default() -> Self {
        Self {
            limit: default_search_event_pop_limit(),
        }
    }
}

impl SearchEventPopOpts {
    /// Set the maximum number of search events to pop
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

/// The status of a batch of search events
#[derive(Debug, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SearchEventStatus {
    /// The events that succeeded
    pub successes: Vec<Uuid>,
    /// The events that failed
    pub failures: Vec<Uuid>,
}

impl SearchEventStatus {
    /// Returns true if the status report is empty (no successes *or* failures)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.successes.is_empty() && self.failures.is_empty()
    }
}
