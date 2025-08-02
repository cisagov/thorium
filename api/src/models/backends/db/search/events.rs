//! Logic for interacting with search events in the event queue
//!
//! The search event queues are currently stored in Redis

use chrono::Utc;
use tracing::{event, instrument, Level};

use crate::models::backends::db::events;
use crate::models::backends::db::keys::SearchEventKeys;
use crate::models::{backends, SearchEventBackend, SearchEventStatus};
use crate::utils::{ApiError, Shared};
use crate::{conn, exec_query, query, serialize};

/// Save a search event to the queue
///
/// # Arguments
///
/// * `search_event` - The search event to save
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::search::events::create", skip_all, err(Debug))]
pub(crate) async fn create<S: SearchEventBackend>(
    search_event: S,
    shared: &Shared,
) -> Result<(), ApiError> {
    // derive our key from the index
    let key = SearchEventKeys::<S>::queue(shared);
    // serialize the event
    let serialized = serialize!(&search_event);
    // get a timestamp for the queue
    let now = Utc::now().timestamp();
    // add the event to the queue
    exec_query!(redis::cmd("zadd").arg(key).arg(now).arg(serialized), shared).await?;
    Ok(())
}

/// Get some number of search events to evaluate
///
/// # Arguments
///
/// * `index` - The index of the events to pop
/// * `count` - The number of events to pop at most
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::search::events::pop", skip(shared), err(Debug))]
pub(crate) async fn pop<S: SearchEventBackend>(
    count: usize,
    shared: &Shared,
) -> Result<Vec<S>, ApiError> {
    // build the key to the right event queue
    let key = SearchEventKeys::<S>::queue(shared);
    // get our current timestamp
    let now = Utc::now();
    // try to pop some events from the right event queue
    let serialized: Vec<(String, f64)> =
        query!(redis::cmd("zpopmin").arg(&key).arg(count), shared).await?;
    // filter out and reset any events that are not yet mature
    let (events, filtered) =
        events::shared::filter_immature::<S>(&key, now, serialized, shared).await?;
    // build the key to the in flight map and the in flight queue
    let map_key = SearchEventKeys::<S>::in_flight_map(shared);
    let queue_key = SearchEventKeys::<S>::in_flight_queue(shared);
    // build a redis pipeline to add these events to the in flight event queue
    let mut pipe = redis::pipe();
    // add all of our events to the in flight event queue
    for ((serial, timestamp), event) in filtered.iter().zip(events.iter()) {
        let id = event.get_id().to_string();
        // save the event to the in-flight map and queue
        pipe.cmd("hset")
            .arg(&map_key)
            .arg(&id)
            .arg(serial)
            .cmd("zadd")
            .arg(&queue_key)
            .arg(timestamp)
            .arg(&id);
    }
    // execute this pipeline
    let _: () = pipe.query_async(conn!(shared)).await?;
    Ok(events)
}

/// Clear search events from the in-flight queue and re-add
/// failed events back to the main queue with a delay
///
/// # Arguments
///
/// * `status` - The status of a batch of search events
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::search::events::clear", skip_all, fields(successes = status.successes.len(), failures = status.failures.len()), err(Debug))]
#[allow(clippy::cast_precision_loss)]
pub(crate) async fn status<S: SearchEventBackend>(
    status: SearchEventStatus,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build the key to the in flight map and the in flight queue
    let map_key = SearchEventKeys::<S>::in_flight_map(shared);
    let queue_key = SearchEventKeys::<S>::in_flight_queue(shared);
    // build a redis pipeline
    let mut pipe = redis::pipe();
    // remove successful events from our our in flight data map and queue if we have any
    if !status.successes.is_empty() {
        let successes = status
            .successes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        pipe.cmd("hdel").arg(&map_key).arg(&successes);
        pipe.cmd("zrem").arg(&queue_key).arg(&successes);
        // execute the pipe
        exec_query!(pipe, shared).await?;
    }
    // handle any failures
    if !status.failures.is_empty() {
        // remove and retrieve failed events from our in flight data map and delete from the queue
        let failures = status
            .failures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let failed_events: Vec<Option<String>> =
            query!(redis::cmd("hmget").arg(&map_key).arg(&failures), shared).await?;
        // set the time these errors can be processed to three minutes from now;
        // ignore cast warning because Unix time in secs will fit in a 52-bit mantissa
        let now = Utc::now();
        let resets = failed_events
            .into_iter()
            .zip(failures.iter())
            .filter_map(|(maybe_raw_event, failure_id)| match maybe_raw_event {
                Some(raw_event) => match serde_json::from_str::<S>(&raw_event) {
                    Ok(mut event) => {
                        // calculate the backoff for this event based on the number of attempts
                        match event.backoff(now) {
                            Some(backoff) => {
                                // reserialize the event with its exponential backoff
                                Some((
                                    serde_json::to_string(&event).unwrap(),
                                    backoff.timestamp() as f64,
                                ))
                            }
                            None => {
                                // the event has been attempted too many times,
                                // so abandon the event
                                event!(
                                    Level::ERROR,
                                    msg = format!(
                                        "Abandoning {} search event '{}'",
                                        S::key(),
                                        event.get_id()
                                    ),
                                    reason = format!(
                                        "Attempted more than {} times",
                                        backends::search::events::MAX_ATTEMPTS
                                    )
                                );
                                None
                            }
                        }
                    }
                    Err(err) => {
                        event!(
                            Level::ERROR,
                            error = format!("Invalid {} search event: {}", S::key(), err)
                        );
                        None
                    }
                },
                // we got no event with this id, so log an error
                None => {
                    event!(
                        Level::ERROR,
                        error = format!(
                            "Failed event '{}' is missing from the {} in-flight search event map",
                            failure_id,
                            S::key()
                        )
                    );
                    None
                }
            })
            .collect::<Vec<_>>();
        // reset the errors if we have any after our check
        if !resets.is_empty() {
            let queue_key = SearchEventKeys::<S>::queue(shared);
            events::shared::reset_pops(&queue_key, resets, shared).await?;
        }
        // finally, delete the failed events from the map and queue
        let mut pipe = redis::pipe();
        pipe.cmd("zrem")
            .arg(&queue_key)
            .arg(&failures)
            .cmd("hdel")
            .arg(&map_key)
            .arg(&failures);
        exec_query!(pipe, shared).await?;
    }
    Ok(())
}

/// Reset all search events in our in flight event queue/map
///
/// # Arguments
///
/// * `index` - The index of the events to reset
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::search::events::reset_all", skip(shared), err(Debug))]
pub(crate) async fn reset_all<S: SearchEventBackend>(shared: &Shared) -> Result<(), ApiError> {
    // build the key to the in-flight search event map/queue
    let map_key = SearchEventKeys::<S>::in_flight_map(shared);
    let queue_key = SearchEventKeys::<S>::in_flight_queue(shared);
    // get the key to the right event queue to reset our events into
    let main_queue_key = SearchEventKeys::<S>::queue(shared);
    // track the number of events we are resetting
    let mut total_reset = 0;
    // pop from the in-flight search event queue 1000 at a time
    loop {
        // try to pop some events from the right event queue
        let popped: Vec<(String, f64)> =
            query!(redis::cmd("zpopmin").arg(&queue_key).arg(1000), shared).await?;
        // if we popped no data then we have no more events to reset
        if popped.is_empty() {
            break;
        }
        let popped_ids = popped.iter().map(|(id, _)| id).collect::<Vec<_>>();
        // get the events' data from the map
        let maybe_serialized: Vec<Option<String>> =
            query!(redis::cmd("hmget").arg(&map_key).arg(&popped_ids), shared).await?;
        let serialized = maybe_serialized
            .into_iter()
            .zip(popped_ids.iter())
            .filter_map(|(maybe_serialized, id)| {
                match maybe_serialized {
                    Some(serialized) => Some(serialized),
                    // this event is missing from the map, so log an error
                    None => {
                        event!(
                            Level::ERROR,
                            error = format!(
                                "Event '{}' is missing from the {} in-flight search event map",
                                id,
                                S::key()
                            )
                        );
                        None
                    }
                }
            })
            .collect::<Vec<_>>();
        // build a redis pipeline to reset these events
        let mut pipe = redis::pipe();
        //  crawl over the events we retrieved and reset them
        for ((event_id, timestamp), data) in popped.iter().zip(&serialized) {
            // reset this event back into our event queue
            pipe.cmd("zadd")
                .arg(&main_queue_key)
                .arg(timestamp)
                .arg(data);
            // remove this events data from our in flight data map
            pipe.cmd("hdel").arg(&map_key).arg(event_id);
        }
        // execute this pipeline
        let _: () = pipe.query_async(conn!(shared)).await?;
        // increment the total number of events that we have reset
        total_reset += popped.len();
        // log the number of events we have reset so far
        event!(Level::INFO, total_reset);
    }
    Ok(())
}
