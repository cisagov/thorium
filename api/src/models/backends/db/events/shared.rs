//! Logic shared between multiple types of events

use chrono::{DateTime, Utc};
use tracing::{event, instrument, Level};

use crate::utils::{ApiError, Shared};
use crate::{conn, deserialize};

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
    // convert our datetime to a timestamp 3 seconds in the past
    let now_ts = (now.timestamp() - 3) as f64;
    // keep track of the events to reset and our deserialized events
    let mut resets = Vec::default();
    let mut events = Vec::with_capacity(serialized.len());
    let mut filtered_serial = Vec::with_capacity(serialized.len());
    // find first mature event that we retrieved
    for (serial, timestamp) in serialized.into_iter().rev() {
        // check if this timestamp is mature or not yet
        if timestamp < now_ts {
            // try to deserialize this mature event
            let event: D = deserialize!(&serial);
            // add our deserialized event
            events.push(event);
            // add our still serialized but filtered info
            filtered_serial.push((serial, timestamp));
        } else {
            // this event is not yet mature so add it to the reset list
            resets.push((serial, timestamp));
        }
    }
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
