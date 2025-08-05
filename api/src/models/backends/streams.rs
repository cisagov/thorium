//! Wrappers for interacting with streams within Thorium with different backends
//! Currently only Redis is supported

use async_recursion::async_recursion;
use chrono::prelude::*;
use tracing::{instrument, span, Level, Span};

use super::db;
use crate::models::{Deadline, Group, RawJob, Stream, StreamDepth, StreamObj, User};
use crate::utils::{ApiError, Shared};
use crate::{at_least, bad};

impl From<Deadline> for StreamObj {
    /// Cast a deadline to a StreamObj
    ///
    /// # Arguments
    ///
    /// * `deadline` - The deadline to convert
    fn from(deadline: Deadline) -> Self {
        // cast deadline data to stream data without timestamp
        // were using a format macro so we get a consistent order
        let data = format!("{{\"group\":\"{}\",\"pipeline\":\"{}\",\"stage\":\"{}\",\"creator\":\"{}\",\"job_id\":\"{}\",\"reaction\":\"{}\"}}",
            deadline.group,
            deadline.pipeline,
            deadline.stage,
            deadline.creator,
            deadline.job_id,
            deadline.reaction);

        // cast to stream object
        StreamObj {
            timestamp: deadline.as_seconds(),
            data,
        }
    }
}

impl From<RawJob> for StreamObj {
    /// Casts a RawJob to a StreamObj
    ///
    /// # Arguments
    ///
    /// * `job` - The job to cast
    fn from(job: RawJob) -> Self {
        // cast to deadline
        let deadline = Deadline::from(job);
        StreamObj::from(deadline)
    }
}

impl From<&RawJob> for StreamObj {
    /// Casts a reference to a RawJob to a StreamObj
    ///
    /// # Arguments
    ///
    /// * `job` - The reference to a job to cast
    fn from(job: &RawJob) -> Self {
        // cast to deadline
        let deadline = Deadline::from(job);
        StreamObj::from(deadline)
    }
}

impl StreamDepth {
    /// Creates a new stream depth object
    ///
    /// # Arguments
    ///
    /// * `depth` - The number of objects between two points in a stream
    /// * `start` - The earliest date that was used when counting objects
    /// * `end` - The latest date that was used when counting objects
    pub(super) fn new(depth: i64, start: i64, end: i64) -> Self {
        // convert start to DateTime
        let start = DateTime::from_timestamp(start, 0).unwrap();
        // convert end to DateTime
        let end = DateTime::from_timestamp(end, 0).unwrap();
        StreamDepth { start, end, depth }
    }
}

impl Stream {
    /// Read from a stream
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is reading from the stream
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `start` - The earliest point in the stream to use
    /// * `end` - The oldest point in the stream to use
    /// * `skip` - The number of entries in the stream to skip
    /// * `limit` - The most objects to return at once
    /// * `shared` - Shared objects in Thorium
    #[allow(clippy::too_many_arguments)]
    #[instrument(name = "Stream::read", skip(user, shared), err(Debug))]
    pub async fn read(
        user: &User,
        group: &str,
        namespace: &str,
        stream: &str,
        start: i64,
        end: i64,
        skip: u64,
        limit: u64,
        shared: &Shared,
    ) -> Result<Vec<StreamObj>, ApiError> {
        // authorize user is apart of this group
        Group::authorize(user, group, shared).await?;
        db::streams::read(group, namespace, stream, start, end, skip, limit, shared).await
    }

    /// Deletes an object from a stream
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting an object from the stream
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `obj` - The object to remove from the stream
    /// * `shared` - Shared objects in Thorium
    /// * `id` - The request ID
    #[instrument(name = "Stream::delete", skip(user, group, shared), err(Debug))]
    pub async fn delete(
        user: &User,
        group: &Group,
        namespace: &str,
        stream: &str,
        obj: &str,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // authorize user is apart of this group
        group.editable(user)?;
        db::streams::delete(&group.name, namespace, stream, obj, shared).await
    }

    /// Deletes an object from a system stream
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `obj` - The object to remove from the stream
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Stream::system_delete", skip(shared), err(Debug))]
    pub async fn system_delete(
        namespace: &str,
        stream: &str,
        obj: &str,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // use correct backend to delete object from stream
        db::streams::delete("system", namespace, stream, obj, shared).await
    }

    /// Gets the depth of a stream in a defined range
    ///
    /// # Arguments
    ///
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `start` - The earliest point in the stream to use
    /// * `end` - The latest point in the stream to use
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Stream::depth", skip(shared), err(Debug))]
    pub async fn depth(
        group: &Group,
        namespace: &str,
        stream: &str,
        start: i64,
        end: i64,
        shared: &Shared,
    ) -> Result<StreamDepth, ApiError> {
        // get depth of stream from Backend
        db::streams::depth(&group.name, namespace, stream, start, end, shared).await
    }

    /// Gets the depth of a stream in a defined range
    ///
    /// # Arguments
    ///
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `start` - The earliest point in the stream to use
    /// * `end` - The oldest point in the stream to use
    /// * `split` - The length of each depth range in seconds
    /// * `shared` - Shared objects in Thorium
    #[allow(clippy::too_many_arguments)]
    #[instrument(name = "Stream::depth_range", skip(shared), err(Debug))]
    pub async fn depth_range(
        group: &Group,
        namespace: &str,
        stream: &str,
        start: i64,
        end: i64,
        split: i64,
        shared: &Shared,
    ) -> Result<Vec<StreamDepth>, ApiError> {
        // throw an error if we try to retrieve more than 10000 depths
        if at_least!((end - start) / 10, 1) / split > 10_000 {
            return bad!("cannot retrieve more than 10,000 depths at once".to_owned());
        }

        // get depths from correct backend
        db::streams::depth_range(&group.name, namespace, stream, start, end, split, shared).await
    }

    /// Gets the earliest point in the stream
    ///
    /// # Arguments
    ///
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `shared` - Shared objects in Thorium
    pub async fn earliest(
        group: &Group,
        namespace: &str,
        stream: &str,
        shared: &Shared,
    ) -> Result<DateTime<Utc>, ApiError> {
        // get depths from correct backend
        db::streams::earliest(&group.name, namespace, stream, shared).await
    }

    /// Gets the latest point in the stream
    ///
    /// # Arguments
    ///
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `shared` - Shared objects in Thorium
    pub async fn latest(
        group: &Group,
        namespace: &str,
        stream: &str,
        shared: &Shared,
    ) -> Result<DateTime<Utc>, ApiError> {
        // get depths from correct backend
        db::streams::latest(&group.name, namespace, stream, shared).await
    }

    /// Recursively maps a stream into pages
    ///
    /// # Arguments
    ///
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream
    /// * `start` - The earliest point in the stream to use
    /// * `end` - The oldest point in the stream to use
    /// * `level` - The current level of recursion
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[allow(clippy::too_many_arguments)]
    #[async_recursion]
    async fn mapper(
        group: &Group,
        namespace: &str,
        stream: &str,
        start: i64,
        end: i64,
        level: u64,
        shared: &Shared,
        span: &Span,
    ) -> Result<Vec<StreamDepth>, ApiError> {
        // assume we can fit the entire range into one map
        // default to breaking it into 10 splits
        let split = at_least!((end - start) / 10, 1);
        let mut maps =
            Self::depth_range(group, namespace, stream, start, end, split, shared).await?;

        // check if any of the depths have over 10k jobs in them
        // if they do recursively remap them
        let mut remapped = Vec::default();
        for map in maps.drain(..) {
            // if this map has a depth of zero drop it
            if map.depth == 0 {
                continue;
            } else if map.depth > 10_000 && level < 10 {
                // remap this in smaller chunks
                remapped.append(
                    &mut Self::mapper(
                        group,
                        namespace,
                        stream,
                        map.start.timestamp(),
                        map.end.timestamp(),
                        level + 1,
                        shared,
                        span,
                    )
                    .await?,
                );
            } else {
                // valid map push it
                remapped.push(map);
            }
        }

        Ok(remapped)
    }

    /// Maps a stream into pages using mapper
    ///
    /// This will map from the date 2020-07-01 to 9999-07-01. So if Thorium
    /// still exists in the year 10000 we'll have a Y2K like problem ¯\_(ツ)_/¯.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is mapping this stream
    /// * `group` - The Group the stream is in
    /// * `namespace` - The namespace for this stream within this group
    /// * `stream` - The name of the stream to map
    /// * `min` - The minimum number of jobs needed to return if any jobs exist
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    pub async fn map(
        user: &User,
        group: &str,
        namespace: &str,
        stream: &str,
        min: u64,
        shared: &Shared,
        span: &Span,
    ) -> Result<Vec<StreamDepth>, ApiError> {
        // Start our recursive stream map
        let span = span!(
            parent: span,
            Level::INFO,
            "Stream Map",
            namespace = namespace,
            stream = stream,
            min = min
        );
        // authorize user is apart of this group
        let group = Group::get(user, group, shared).await?;
        // arbitrary past value
        let past = chrono::DateTime::parse_from_rfc3339("2020-07-01T00:00:00-00:00")?
            .with_timezone(&Utc)
            .timestamp();
        // arbitrary distant future value
        let distant_future = chrono::DateTime::parse_from_rfc3339("9999-07-01T00:00:00-00:00")?
            .with_timezone(&Utc)
            .timestamp();
        // see how many jobs actually exist
        let total = Self::depth(&group, namespace, stream, past, distant_future, shared).await?;
        // short circuit if no jobs
        if total.depth == 0 {
            return Ok(Vec::default());
        }

        // choose the right mapper based on job count
        // just map all the jobs if less than 10k
        if total.depth < 10_000 {
            let earliest = Self::earliest(&group, namespace, stream, shared).await?;
            let map = Self::mapper(
                &group,
                namespace,
                stream,
                earliest.timestamp(),
                distant_future,
                0,
                shared,
                &span,
            )
            .await?;
            return Ok(map);
        }

        // map 1 day at a time until we hit minimum bound or last job
        let mut maps = Vec::default();
        let count: i64 = 0;
        // start at earliest job and short circuit if we hit latest
        let earliest = Self::earliest(&group, namespace, stream, shared).await?;
        let latest = Self::latest(&group, namespace, stream, shared).await?;
        for days in 1i64..36500 {
            // get end timestamp for current day
            let end = earliest + chrono::Duration::days(days);
            // map next day
            let mut new = Self::mapper(
                &group,
                namespace,
                stream,
                earliest.timestamp(),
                end.timestamp(),
                0,
                shared,
                &span,
            )
            .await?;
            // short circuit if latest is before end
            if latest < end {
                maps.append(&mut new);
                break;
            }

            // count latest and short circuit if we hit our minimum bound
            new.iter().fold(count, |count, submap| count + submap.depth);
            if count > min as i64 {
                maps.append(&mut new);
                break;
            }
        }
        Ok(maps)
    }
}
