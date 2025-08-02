//! Backend-related logic for search events

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use tracing::instrument;
use uuid::Uuid;

use crate::is_admin;
use crate::models::backends::db;
use crate::models::SearchEventStatus;
use crate::models::User;
use crate::models::{SearchEvent, SearchEventPopOpts};
use crate::utils::{ApiError, Shared};

pub mod results;
pub mod tags;

/// The maximum number of times we'll attempt to stream an event
pub const MAX_ATTEMPTS: u8 = 10;

/// The number of seconds to backoff an event per-attempt
const BACKOFF_SECS: [i32; MAX_ATTEMPTS as usize] = [
    180,   // 3 minutes
    360,   // 6 minutes
    720,   // 12 minutes
    1440,  // 24 minutes
    2880,  // 48 minutes
    5760,  // 1 hours 36 minutes
    11520, // 3 hours 12 minutes
    23040, // 6 hours 24 minutes
    46080, // 12 hours 48 minutes
    92160, // 25 hours 36 minutes
];

/// The percentage of +/- random jitter to add to the backoff
const BACKOFF_JITTER: f64 = 0.05;

/// Defines shared backend logic for search events
#[async_trait::async_trait]
pub(crate) trait SearchEventBackend: SearchEvent {
    /// Return a key component used for deriving redis keys
    fn key() -> &'static str;

    /// Get the event's id
    fn get_id(&self) -> Uuid;

    /// Increments the number of times we've attempted to stream this event by 1
    /// and returns the new number of attempts
    fn attempted(&mut self) -> u8;

    /// Increments the number of times we've attempted to stream the event by 1
    /// and returns the time the event should be backed off based on the times it's
    /// been retried so far
    ///
    /// Backoff is `3 minutes * 2^(attempts - 1)`. Minimum is 3 minutes; maximum is
    /// about 1 day with 10 attempts.
    ///
    /// # Arguments
    ///
    /// * `now` - The current(ish) time in UTC
    // ignore casting warnings because we're only growing exponentially for a small
    // number of attempts, so casting is safe
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn backoff(&mut self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        // increment the attempts count by 1
        let attempts = self.attempted();
        // check the number of times we've attempted to stream this event
        if attempts > MAX_ATTEMPTS {
            // we've tried more than our max attempts already, so return no backoff
            // and abandon this event
            None
        } else {
            // base backoff duration is 3 minutes
            let backoff = Duration::seconds(i64::from(BACKOFF_SECS[usize::from(attempts - 1)]));
            // add jitter - random value between -5% and +5% of the exponential backoff
            let mut rng = rand::rng();
            let jitter_range = (backoff.num_seconds() as f64 * BACKOFF_JITTER) as i64;
            let jitter = Duration::seconds(rng.random_range(-jitter_range..jitter_range));
            // add the backoff to now with jitter
            Some(now + backoff + jitter)
        }
    }

    /// Pop some search events from a specific queue
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is popping events
    /// * `count` - The maximum number of search events to pop
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SearchEvent::pop", skip(user, shared), err(Debug))]
    async fn pop(user: &User, count: usize, shared: &Shared) -> Result<Vec<Self>, ApiError> {
        // only admins can pop search events
        is_admin!(user);
        // try to pop some search events from redis
        db::search::events::pop::<Self>(count, shared).await
    }

    /// Clear search events from the in-flight queue and
    /// re-add failures to the main queue
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is popping events
    /// * `status` - The status of the events
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SearchEvent::status", skip_all, err(Debug))]
    async fn status(
        user: &User,
        status: SearchEventStatus,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // only admins can clear search events
        is_admin!(user);
        // handle search event status
        db::search::events::status::<Self>(status, shared).await
    }

    /// Reset all search events in our in flight event queue/map
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is popping events
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SearchEvent::reset_all", skip_all, err(Debug))]
    async fn reset_all(user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can reset all search events
        is_admin!(user);
        // reset all in-flight search events
        db::search::events::reset_all::<Self>(shared).await
    }
}

impl<S> FromRequestParts<S> for SearchEventPopOpts
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

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::models::{
        backends::search::events::{BACKOFF_JITTER, BACKOFF_SECS, MAX_ATTEMPTS},
        ResultSearchEvent, Sample,
    };

    use super::SearchEventBackend;

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn backoff() {
        let mut event = ResultSearchEvent::modified::<Sample>("sha256".to_string(), vec![]);
        let now = Utc::now();
        // check correct backoff value for each possible attempt
        for i in 1..=MAX_ATTEMPTS {
            let backoff = event.backoff(now);
            assert_eq!(event.attempts, i);
            assert!(backoff.is_some_and(|backoff| {
                let since = backoff.signed_duration_since(now).num_seconds();
                // calculate the min/max of the range when accounting for +/- jitter
                let factor = i64::from(2u32.pow(u32::from(i - 1)));
                let min =
                    ((i64::from(BACKOFF_SECS[0]) * factor) as f64 * (1.0 - BACKOFF_JITTER)) as i64;
                let max =
                    ((i64::from(BACKOFF_SECS[0]) * factor) as f64 * (1.0 + BACKOFF_JITTER)) as i64;
                // make sure we got a value within our expected range
                since >= min && since <= max
            }));
        }
        // check that we now get no backoff value because we've passed maximum attempts
        let backoff = event.backoff(now);
        assert_eq!(event.attempts, MAX_ATTEMPTS + 1);
        assert!(backoff.is_none());
        let backoff = event.backoff(now);
        assert_eq!(event.attempts, MAX_ATTEMPTS + 2);
        assert!(backoff.is_none());
    }
}
