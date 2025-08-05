//! The backend functions for events
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use chrono::prelude::*;
use tracing::instrument;
use uuid::Uuid;

use super::db::{self};
use crate::models::backends::TagSupport;
use crate::models::{EventCacheStatus, EventPopOpts};
use crate::{
    is_admin,
    models::{Event, EventData, EventRow, EventType, TagRequest},
};
use crate::{
    models::User,
    utils::{ApiError, Shared},
};

impl TryFrom<EventRow> for Event {
    type Error = ApiError;
    /// Cast an event row to an event
    ///
    /// # Arguments
    ///
    /// * `event_row` - The event row to cast
    fn try_from(row: EventRow) -> Result<Event, Self::Error> {
        // convert our event row to an event
        let event = Event {
            id: row.id,
            timestamp: row.timestamp,
            parent: row.parent,
            user: row.user,
            data: row.data,
            depth: u8::try_from(row.trigger_depth)?,
        };
        Ok(event)
    }
}

impl Event {
    /// Create a new tag event
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating new tags
    /// * `key` - The key to save this tag too in the db
    /// * `req` - The tag request we are making events for
    #[must_use]
    pub fn new_tag<T: TagSupport>(user: &User, key: String, req: TagRequest<T>) -> Self {
        // generate a random event id
        let id = Uuid::new_v4();
        // get the current timestamp
        let timestamp = Utc::now();
        // build our event data
        let data = EventData::NewTags {
            tag_type: T::tag_kind(),
            item: key,
            groups: req.groups,
            tags: req.tags,
        };
        // build our tag event
        Event {
            id,
            timestamp,
            parent: None,
            user: user.username.clone(),
            data,
            depth: req.trigger_depth,
        }
    }

    /// Create a new sample event
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating new tags
    /// * `groups` - The groups this new sample was added too
    /// * `sample` - The sha256 of the new sample
    /// * `depth` - This new samples trigger's depth
    #[must_use]
    pub fn new_sample(user: &User, groups: Vec<String>, sample: String, depth: u8) -> Self {
        // generate a random event id
        let id = Uuid::new_v4();
        // get the current timestamp
        let timestamp = Utc::now();
        // build our event data
        let data = EventData::NewSample { groups, sample };
        // build our tag event
        Event {
            id,
            timestamp,
            parent: None,
            user: user.username.clone(),
            data,
            depth,
        }
    }

    /// Pop some events from a specific queue
    #[instrument(name = "Event::pop", skip(user, shared), err(Debug))]
    pub async fn pop(
        user: &User,
        kind: EventType,
        count: usize,
        shared: &Shared,
    ) -> Result<Vec<Event>, ApiError> {
        // only admins can pop events
        is_admin!(user);
        // try to pop some events from redis
        db::events::pop(kind, count, shared).await
    }

    /// Clear some events from a specific queue
    #[instrument(name = "Event::clear", skip(user, ids, shared), fields(count = ids.len()), err(Debug))]
    pub async fn clear(
        user: &User,
        kind: EventType,
        ids: &Vec<Uuid>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // only admins can clear events
        is_admin!(user);
        // try to clear some events
        db::events::clear(kind, ids, shared).await
    }

    /// Reset any in flight events
    #[instrument(name = "Event::reset_all", skip(user, shared), err(Debug))]
    pub async fn reset_all(user: &User, kind: EventType, shared: &Shared) -> Result<(), ApiError> {
        // only admins can reset all events
        is_admin!(user);
        // try to reset all in flight events
        db::events::reset_all(kind, shared).await
    }

    /// Get our current event cache status
    ///
    /// Only admins can get the event cache status.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting the event cache status
    /// * `clear` - Whether to clear the event cache statuses or not
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Event::get_cache_status", skip(user, shared), err(Debug))]
    pub async fn get_cache_status(
        user: &User,
        clear: bool,
        shared: &Shared,
    ) -> Result<EventCacheStatus, ApiError> {
        // only admins can get the event cache status
        is_admin!(user);
        // get our event cache status
        db::events::get_cache_status(clear, shared).await
    }
}

impl<S> FromRequestParts<S> for EventPopOpts
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}
