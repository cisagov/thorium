//! Database keys for search events

use std::marker::PhantomData;

use crate::models::SearchEventBackend;
use crate::utils::Shared;

/// The keys related to search events in Redis
#[derive(Debug)]
pub struct SearchEventKeys<S: SearchEventBackend> {
    phantom: PhantomData<S>,
}

impl<S: SearchEventBackend> SearchEventKeys<S> {
    /// Derives the key for the search event queue for the given elastic index
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub fn queue(shared: &Shared) -> String {
        format!(
            "{ns}:search-streamer:{event_type}:queue",
            ns = shared.config.thorium.namespace,
            event_type = S::key()
        )
    }

    /// Derives the key for the map of all search events in flight
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub fn in_flight_map(shared: &Shared) -> String {
        format!(
            "{ns}:search-streamer:{event_type}:in_flight_map",
            ns = shared.config.thorium.namespace,
            event_type = S::key()
        )
    }

    /// Derives the key for the queue of all search events in flight
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub fn in_flight_queue(shared: &Shared) -> String {
        format!(
            "{ns}:search-streamer:{event_type}:in_flight_queue",
            ns = shared.config.thorium.namespace,
            event_type = S::key()
        )
    }
}
