//! Logic shared between multiple types of events

use chrono::{DateTime, Utc};
use tracing::{event, instrument, Level};

use crate::utils::{ApiError, Shared};
use crate::{conn, deserialize};

/// Classify popped events into mature (deserialized) events and immature resets
///
/// Events are popped oldest first and keep that order, since compacting events for the same
/// item relies on later events superseding earlier ones.
///
/// # Arguments
///
/// * `now` - A timestamp from before we pulled events
/// * `serialized` - The serialized events to check for maturity
fn classify_immature<D: for<'de> serde::de::Deserialize<'de>>(
    now: DateTime<Utc>,
    serialized: Vec<(String, f64)>,
) -> Result<(Vec<D>, Vec<(String, f64)>, Vec<(String, f64)>), ApiError> {
    // convert our datetime to a timestamp 3 seconds in the past
    let now_ts = (now.timestamp() - 3) as f64;
    // keep track of the events to reset and our deserialized events
    let mut resets = Vec::default();
    let mut events = Vec::with_capacity(serialized.len());
    let mut filtered_serial = Vec::with_capacity(serialized.len());
    // find first mature event that we retrieved
    for (serial, timestamp) in serialized {
        // check if this timestamp is mature or not yet
        if timestamp < now_ts {
            // try to deserialize this mature event
            let event: D = deserialize!(&serial);
            // add our deserialized event
            events.push(event);
            // add our still serialized but filtered info
            filtered_serial.push((serial, timestamp));
        } else {
            // this event is not yet mature so add it to the reste list
            resets.push((serial, timestamp));
        }
    }
    Ok((events, filtered_serial, resets))
}

/// Filter and reset any events that are younger then 3 seconds
///
/// We are filtering events younger then 3 seconds to ensure the DB has a chance
/// to reach consistency.
///
/// # Arguments
///
/// * `key` - The sorted set we are filtering immature events from
/// * `now` - A timestamp from before we pulled events
/// * `serialized` - The serialized events to check for maturity
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::events::shared::filter_immature", skip_all, fields(pre_filter = serialized.len()), err(Debug))]
pub async fn filter_immature<D: for<'de> serde::de::Deserialize<'de>>(
    key: &String,
    now: DateTime<Utc>,
    serialized: Vec<(String, f64)>,
    shared: &Shared,
) -> Result<(Vec<D>, Vec<(String, f64)>), ApiError> {
    // classify our popped events into mature events and immature resets
    let (events, filtered_serial, resets) = classify_immature(now, serialized)?;
    if !resets.is_empty() {
        // log the number of immature events that we found
        event!(Level::INFO, immature = resets.len());
        // try to reset these immature events
        reset_pops(key, resets, shared).await?;
    }
    Ok((events, filtered_serial))
}

/// Try to reset popped events in the case of an error or other failure
///
/// If this reset fails then these events will simply be lost.
///
/// # Arguments
///
/// * `key` - The sorted set to restore these popped events too
/// * `pops` - The popped events to restore
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::events::shared::reset_pops", skip_all, fields(pops = pops.len()), err(Debug))]
pub async fn reset_pops(
    key: &String,
    pops: Vec<(String, f64)>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build a pipe to reset any immature events
    let mut pipe = redis::pipe();
    // add all of our immature event reset commands to this pipeline
    for (serial, timestamp) in pops {
        pipe.cmd("zadd").arg(key).arg(timestamp).arg(serial);
    }
    // execute this redis pipeline
    let _: () = pipe.query_async(conn!(shared)).await?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// `filter_immature` returns mature events in the oldest-first order `ZPOPMIN` produced.
    /// Downstream compaction (`CompactEvent::append`) assumes events arrive oldest-first ("the
    /// delete event will come last"), so preserving that order keeps compaction from keeping the
    /// OLDEST event's type, losing deletes and leaving ghost documents in Elasticsearch.
    #[test]
    fn mature_events_preserve_oldest_first_order() {
        // pin "now" far enough ahead that every event below is mature (timestamp < now - 3)
        let now = DateTime::from_timestamp(1_000_000, 0).unwrap();
        // ZPOPMIN returns members ascending by score, so the popped list is oldest-first: the
        // payloads (100, 200, 300) double as their scores so the intended order is unambiguous
        let serialized = vec![
            ("100".to_owned(), 100.0),
            ("200".to_owned(), 200.0),
            ("300".to_owned(), 300.0),
        ];
        // classify the popped events
        let (events, _filtered, resets): (Vec<u64>, _, _) =
            classify_immature(now, serialized).unwrap();
        // all three events are mature so none should be reset
        assert!(resets.is_empty(), "no mature event should be reset");
        // the mature events must come back oldest-first, matching ZPOPMIN's ascending order
        assert_eq!(
            events,
            vec![100, 200, 300],
            "mature events must preserve oldest-first order, not be reversed to newest-first"
        );
    }
}
