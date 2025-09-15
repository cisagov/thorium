use bb8_redis::redis::cmd;
use chrono::prelude::*;
use std::cmp::min;
use std::num::NonZeroU64;
use tracing::instrument;

use super::keys::streams::StreamKeys;
use crate::models::{StreamDepth, StreamObj};
use crate::utils::{ApiError, Shared};
use crate::{conn, internal_err, not_found, query};

/// Reads objects from a stream between a start and end timestamp
///
/// # Arguments
///
/// * `group` - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to read from
/// * `start_ts` - The youngest point in the stream to use
/// * `end_ts` - The oldest point in the stream to use
/// * `skip` - The number of entries in the stream to skip
/// * `limit` - The max number of objects to retrieve
/// * `shared` - The shared objects in Thorium
#[rustfmt::skip]
#[instrument(name = "db::streams::read", skip(shared), err(Debug))]
pub async fn read(
    group: &str,
    namespace: &str,
    stream: &str,
    start_ts: i64,
    end_ts: i64,
    skip: u64,
    limit: u64,
    shared: &Shared,
) -> Result<Vec<StreamObj>, ApiError> {
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // read objects with their scores from the stream
    let raw: Vec<(String, i64)> = query!(
        cmd("zrangebyscore").arg(key).arg(start_ts).arg(end_ts)
            .arg("withscores").arg("LIMIT").arg(skip).arg(limit),
        shared).await?;
    let objs = raw.into_iter()
        .map(|raw| StreamObj::new(raw.1, raw.0))
        .collect();
    Ok(objs)
}

/// Reads objects from a stream between a start and end timestamp without scores for a prebuilt key
///
/// # Arguments
///
/// * `key` - The key in redis for this stream
/// * `start` - The youngest point in the stream to use
/// * `end` - The oldest point in the stream to use
/// * `limit` - The max number of objects to retrieve
/// * `shared` - The shared objects in Thorium
#[rustfmt::skip]
#[instrument(name = "db::streams::read_no_scores_by_key", skip(shared), err(Debug))]
pub async fn read_no_scores_by_key(
    key: &str,
    start: i64,
    end: i64,
    skip: u64,
    limit: u64,
    shared: &Shared,
) -> Result<Vec<String>, ApiError> {
    // read objects with their scores from the stream
    let raw: Vec<String> = query!(
        cmd("zrangebyscore").arg(key).arg(start).arg(end)
            .arg("LIMIT").arg(skip).arg(limit),
        shared).await?;
    Ok(raw)
}

/// Reads objects from a stream between a start and end timestamp without scores
///
/// # Arguments
///
/// * `group` - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to read from
/// * `start` - The youngest point in the stream to use
/// * `end` - The oldest point in the stream to use
/// * `limit` - The max number of objects to retrieve
/// * `shared` - The shared objects in Thorium
#[rustfmt::skip]
#[instrument(name = "db::streams::read_no_scores", skip(shared), err(Debug))]
pub async fn read_no_scores(
    group: &str,
    namespace: &str,
    stream: &str,
    start: i64,
    end: i64,
    limit: u64,
    shared: &Shared,
) -> Result<Vec<String>, ApiError> {
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // read objects with their scores from the stream
    let raw: Vec<String> = query!(
        cmd("zrangebyscore").arg(key).arg(start).arg(end)
            .arg("LIMIT").arg(0).arg(limit),
        shared).await?;
    Ok(raw)
}

/// deletes a object from a stream
///
/// # Arguments
///
/// * `group` - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to delete an object from
/// * `start` - The stream to remove an object from
/// * `obj` - The object to remove
/// * `shared` - The shared objects in Thorium
pub async fn delete(
    group: &str,
    namespace: &str,
    stream: &str,
    obj: &str,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // try to delete object from stream
    let deleted: bool = query!(cmd("zrem").arg(key).arg(obj), shared).await?;
    // error if object wasn't deleted
    if !deleted {
        return not_found!("Failed to remove object from stream".to_owned());
    }
    Ok(())
}

/// Count the number of objects in a stream between two points in time
///
/// # Arguments
///
/// * 'group - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to get the depth of
/// * `start` - The youngest point in the stream to use
/// * `end` - The oldest point in the stream to use
/// * `shared` - The shared objects in Thorium
#[instrument(name = "db::streams::depth", skip(shared), err(Debug))]
pub async fn depth(
    group: &str,
    namespace: &str,
    stream: &str,
    start: i64,
    end: i64,
    shared: &Shared,
) -> Result<StreamDepth, ApiError> {
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // count objects in a stream
    let depth = query!(cmd("zcount").arg(key).arg(start).arg(end), shared).await?;
    Ok(StreamDepth::new(depth, start, end))
}

/// Count the number of objects in a stream between two points in time
///
/// # Arguments
///
/// * 'group - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to get the depth of
/// * `start` - The youngest point in the stream to use
/// * `end` - The oldest point in the stream to use
/// * `split` - The length of each depth probe in seconds
/// * `shared` - The shared objects in Thorium
#[instrument(name = "db::streams::depth_range", skip(shared), err(Debug))]
pub async fn depth_range(
    group: &str,
    namespace: &str,
    stream: &str,
    start: i64,
    end: i64,
    split: NonZeroU64,
    shared: &Shared,
) -> Result<Vec<StreamDepth>, ApiError> {
    // convert our non zero split to a usize and an i64
    let split_usize = split.get() as usize;
    let split_i64 = split.get() as i64;
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // loop and build tuple of start and end for each depth probe
    // start with start - 1 so we get valid inclusive ranges
    let start = start - 1;
    let depth_pairs: Vec<(i64, i64)> = (start..end)
        .step_by(split_usize)
        .map(|start| (start + 1, min(start + split_i64, end)))
        .collect();
    // build zcount for each part of the stream
    let counts: Vec<i64> = depth_pairs
        .iter()
        .fold(redis::pipe().atomic(), |pipe, pair| {
            pipe.cmd("zcount").arg(&key).arg(pair.0).arg(pair.1)
        })
        .query_async(conn!(shared))
        .await?;
    // cast to vector of stream depths
    let depths = counts
        .into_iter()
        .enumerate()
        .map(|depth| StreamDepth::new(depth.1, depth_pairs[depth.0].0, depth_pairs[depth.0].1))
        .collect();
    //let depth = conn.zcount(key, start as isize, end as isize)?;
    Ok(depths)
}

/// Get the earliest objects timestamp in a stream
///
/// # Arguments
///
/// * 'group - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to get the earliest timestamp from
/// * `shared` - The shared objects in Thorium
pub async fn earliest(
    group: &str,
    namespace: &str,
    stream: &str,
    shared: &Shared,
) -> Result<DateTime<Utc>, ApiError> {
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // get earliest point in stream
    let raw: (String, i64) = query!(
        cmd("zrange").arg(key).arg(0).arg(0).arg("withscores"),
        shared
    )
    .await?;
    // convert to a timestamp
    match DateTime::from_timestamp(raw.1, 0) {
        Some(datetime) => Ok(datetime),
        None => {
            return internal_err!(format!(
                "Earliest point in stream does not have a valid timestamp - {}",
                raw.1
            ));
        }
    }
}

/// Get the latest objects timestamp in a stream
///
/// # Arguments
///
/// * 'group - The group this stream is in
/// * `namespace` - The namespace of this stream within this group
/// * `stream` - The name of the stream to get the latest timestamp from
/// * `shared` - The shared objects in Thorium
pub async fn latest(
    group: &str,
    namespace: &str,
    stream: &str,
    shared: &Shared,
) -> Result<DateTime<Utc>, ApiError> {
    // build stream keys
    let key = StreamKeys::stream(group, namespace, stream, shared);
    // get earliest point in stream
    let raw: (String, i64) = query!(
        cmd("zrange").arg(key).arg(-1).arg(-1).arg("withscores"),
        shared
    )
    .await?;
    // convert to a timestamp
    match DateTime::from_timestamp(raw.1, 0) {
        Some(datetime) => Ok(datetime),
        None => {
            return internal_err!(format!(
                "Earliest point in stream does not have a valid timestamp - {}",
                raw.1
            ));
        }
    }
}
