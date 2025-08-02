use chrono::{DateTime, Utc};
use redis::{aio::MultiplexedConnection, AsyncCommands, RedisError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thorium::{Conf, Error};
use tracing::{event, instrument, Level};

/// A session that intiates a search store, pulling data from the data
/// base and streaming it efficiently while tracking progress
#[derive(Debug)]
pub struct InitSession {
    /// A multiplexed connection to Redis
    pub redis: MultiplexedConnection,
    /// Information about this init session stored in Redis
    pub info: InitSessionInfo,
    /// Keys to this init session in Redis
    pub keys: InitSessionKeys,
    /// The token ranges remaining to be streamed
    pub tokens_remaining: BTreeMap<i64, i64>,
}

impl InitSession {
    /// Create a new init session and store its info in Redis
    ///
    /// Clears
    ///
    /// # Arguments
    ///
    /// * `chunk_count` - The number of chunks we're making in Scylla
    /// * `keys` - Keys to this init session in Redis
    /// * `redis` - A multiplexed connection to Redis
    #[instrument(name = "InitSession::create", skip_all, err(Debug))]
    pub async fn create(
        chunk_count: u64,
        keys: InitSessionKeys,
        redis: &mut MultiplexedConnection,
    ) -> Result<Self, Error> {
        let chunk_size = calculate_chunk_size(chunk_count);
        // create new info for this session
        let info = InitSessionInfo::new(chunk_count, chunk_size);
        // serialize the session info
        let raw_info = serde_json::to_string(&info).unwrap();
        let _: () = redis::pipe()
            // set the session's info
            .set(&keys.info, &raw_info)
            // delete any log in case one exists
            .del(&keys.log)
            .query_async(redis)
            .await
            .map_err(|err| Error::new(format!("Error create init session in Redis: {err}")))?;
        // calculate the tokens remaining for this session
        let tokens_remaining = build_token_ranges(chunk_count);
        event!(Level::INFO, "Created new init session");
        Ok(InitSession {
            redis: redis.clone(),
            info,
            keys,
            tokens_remaining,
        })
    }

    /// Attempt to resume an init session from the given info
    ///
    /// Returns `None` if the session can't be resumed
    ///
    /// # Arguments
    ///
    /// * `resume_info` - Info on the session to resume
    /// * `chunk_count` - The number of chunks we're making in Scylla
    /// * `keys` - Keys to this init session in Redis
    /// * `redis` - A multiplexed connection to Redis
    #[instrument(name = "InitSession::resume", skip_all, err(Debug))]
    pub async fn resume(
        resume_info: InitSessionInfo,
        chunk_count: u64,
        keys: InitSessionKeys,
        redis: &mut MultiplexedConnection,
    ) -> Result<Option<Self>, Error> {
        let chunk_size = calculate_chunk_size(chunk_count);
        // calculate the tokens remaining for this session
        let mut tokens_remaining = build_token_ranges(chunk_count);
        if resume_info.resumable(chunk_count, chunk_size) {
            // read the log
            // if we get an error here, a db error occurred so we need to propagate that error
            let token_ranges: Vec<String> = redis.smembers(&keys.log).await.map_err(|err| {
                Error::new(format!("Error querying Redis for init session log: {err}"))
            })?;
            for raw_token_range in token_ranges {
                let (start, _end) = match parse_token_range(&raw_token_range) {
                    Ok(range) => range,
                    Err(err) => {
                        // we found a bad token range, so log it and return None
                        // since we can't resume this session
                        event!(
                            Level::ERROR,
                            msg = "Failed to parse token range",
                            token_range = raw_token_range,
                            error = err.to_string()
                        );
                        return Ok(None);
                    }
                };
                // remove this token range from our remaining map
                tokens_remaining.remove(&start);
            }
            // return a session with the tokens completed already removed
            Ok(Some(InitSession {
                redis: redis.clone(),
                info: resume_info,
                keys,
                tokens_remaining,
            }))
        } else {
            // return no session because it can't be resumed
            Ok(None)
        }
    }

    /// Attempt to log a completed token range in Redis
    ///
    /// # Arguments
    ///
    /// * `start` - The start token completed
    /// * `end` - The end token completed
    #[instrument(name = "InitSession::log", skip(self), err(Debug))]
    pub async fn log(&mut self, start: i64, end: i64) -> Result<(), RedisError> {
        // serialize the token range
        let token_range = serialize_token_range(start, end);
        // add it to Redis
        self.redis.sadd(&self.keys.log, token_range).await
    }

    /// Finish this init session, deleting its data from Redis
    #[instrument(name = "InitSession::finish", skip_all, err(Debug))]
    pub async fn finish(&mut self) -> Result<(), RedisError> {
        // delete the info and log keys
        self.redis.del(&[&self.keys.info, &self.keys.log]).await
    }

    /// Returns a string duration for how long this session took to complete
    ///
    /// Format: `T<HOURS>:<MINUTES>:<SECONDS>.<NANOSECONDS>`
    ///
    /// Example: `5:42:10.004`
    pub fn duration(&self) -> String {
        let duration = Utc::now() - self.info.start;
        let secs = duration.num_seconds() % 60;
        let mins = (duration.num_seconds() / 60) % 60;
        format!(
            "T{}:{:0>2}:{:0>2}.{:0>3}",
            duration.num_hours(),
            mins,
            secs,
            duration.subsec_nanos()
        )
    }
}

/// Information on an init session encoded in the first line
/// of an init session file
///
/// This info is crucial to verify that an init session can be
/// resumed successfully based on current settings
#[derive(Debug, Serialize, Deserialize)]
pub struct InitSessionInfo {
    /// The number of chunks in this init session
    pub chunk_count: u64,
    /// The chunk size used in this init session
    pub chunk_size: i64,
    /// The time this init session was started in UTC
    pub start: DateTime<Utc>,
}

impl InitSessionInfo {
    /// Create info on a new init session
    ///
    /// Runs a system call to get the current time
    ///
    /// # Arguments
    ///
    /// * `chunk_count` - The number of chunks in this init session
    /// * `chunk_size` - The chunk size to use in this init session
    pub fn new(chunk_count: u64, chunk_size: i64) -> Self {
        Self {
            chunk_count,
            chunk_size,
            start: Utc::now(),
        }
    }

    /// Verify that this session can be resumed based on the given chunk count and size
    ///
    /// # Arguments
    ///
    /// * `chunk_count` - The configured chunk count
    /// * `chunk_size` - The chunk size calculated from the count
    #[instrument(name = "InitSessionInfo::resumable", skip_all)]
    fn resumable(&self, chunk_count: u64, chunk_size: i64) -> bool {
        if self.chunk_count != chunk_count {
            event!(
                Level::WARN,
                "Incompatible init session: chunk count has changed ({} -> {})",
                self.chunk_count,
                chunk_count
            );
            return false;
        } else if self.chunk_size != chunk_size {
            event!(
                Level::WARN,
                "Incompatible init session: chunk size has changed ({} -> {})",
                self.chunk_size,
                chunk_size
            );
            return false;
        }
        true
    }

    /// Attempt to query for the init session info at the given key from Redis
    ///
    /// # Arguments
    ///
    /// * `key` - The key to the init session info
    /// * `redis` - The Redis connection to use
    #[instrument(name = "InitSessionInfo::query", skip(redis), err(Debug))]
    pub async fn query(
        key: &str,
        redis: &mut MultiplexedConnection,
    ) -> Result<Option<Self>, Error> {
        // attempt to get the info from redis
        let raw_info: Option<String> = redis.get(key).await.map_err(|err| {
            Error::new(format!(
                "Error retrieving init session info from Redis: {err}"
            ))
        })?;
        // try to deserialize the info
        let info = raw_info
            .map(|raw| serde_json::from_str::<Self>(&raw))
            .transpose()
            .map_err(|err| Error::new(format!("Malformed init session info: {err}")))?;
        Ok(info)
    }
}

/// Keys for an [`InitSession`] in Redis
#[derive(Debug, Clone)]
pub struct InitSessionKeys {
    /// The key to this session's info in Redis
    pub info: String,
    /// The key to this session's log in Redis
    pub log: String,
}

impl InitSessionKeys {
    /// Create new keys for an init session stored in Redis for this
    /// search store/data source combination
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium cluster configuration
    /// * `store` - The name of the search store
    /// * `data` - The name of the data source
    pub fn new(conf: &Conf, store: &str, data: &str) -> Self {
        Self {
            info: format!(
                "{ns}:search-streamer:init:info:{store}-{data}",
                ns = conf.thorium.namespace,
                store = store,
                data = data
            ),
            log: format!(
                "{ns}:search-streamer:init:log:{store}-{data}",
                ns = conf.thorium.namespace,
                store = store,
                data = data
            ),
        }
    }
}

/// Calculate the size of each chunk based on the number of chunks
///
///  # Arguments
///
/// * `chunk_count` - The number of chunks
fn calculate_chunk_size(chunk_count: u64) -> i64 {
    // will panic if chunk count is 0 or 1
    (u64::MAX / chunk_count).try_into().unwrap()
}

/// Build the range of tokens to stream
///
/// # Arguments
///
/// * `chunk_count` - The number of chunks
fn build_token_ranges(chunk_count: u64) -> BTreeMap<i64, i64> {
    // determine how large each of of our chunks should be
    let chunk_size = calculate_chunk_size(chunk_count);
    // crawl over our token ranges and build them
    let mut tokens = BTreeMap::new();
    let mut start = i64::MIN;
    for _ in 1..chunk_count {
        // calculate the end for this chunk
        let end = start + chunk_size;
        // add this chunk to our token list
        tokens.insert(start, end);
        // increment our start position (+1 because our prepared statements are
        // inclusive: <= and >=)
        start = end + 1;
    }
    // add our final chunk
    tokens.insert(start, i64::MAX);
    tokens
}

/// The delimiter to use for token ranges when serializing/deserializing
/// to/from Redis
const TOKEN_RANGE_DELIMITER: char = ':';

/// Try to parse a token range from the given str from Redis
///
/// # Arguments
///
/// * `raw` - The raw str to parse a token range from
#[instrument(name = "init::parse_token_range", err(Debug))]
fn parse_token_range(raw: &str) -> Result<(i64, i64), Error> {
    let mut split = raw.split(TOKEN_RANGE_DELIMITER);
    let start = split
        .next()
        .ok_or(Error::new("empty token range"))?
        .parse::<i64>()
        .map_err(|err| Error::new(format!("invalid start token: {err}")))?;
    let end = split
        .next()
        .ok_or(Error::new("missing end token".to_string()))?
        .parse::<i64>()
        .map_err(|err| Error::new(format!("invalid end token: {err}")))?;
    if split.next().is_some() {
        return Err(Error::new("malformed token range'".to_string()));
    }
    Ok((start, end))
}

/// Serialize a token range to a string to store in Redis
///
/// # Arguments
///
/// * `start` - The start of the range
/// * `end` - The end of the range
fn serialize_token_range(start: i64, end: i64) -> String {
    format!("{start}{TOKEN_RANGE_DELIMITER}{end}")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn chunk_count() {
        let chunk_count = 1238;
        let token_ranges = build_token_ranges(chunk_count);
        assert_eq!(token_ranges.len() as u64, chunk_count);
    }

    #[test]
    fn token_ranges() {
        let chunk_count = 10452;
        let mut token_ranges = build_token_ranges(chunk_count);
        let (start, end) = token_ranges.pop_first().unwrap();
        assert_eq!(start, i64::MIN);
        let (final_start, final_end) = token_ranges.pop_last().unwrap();
        let mut p_end = end;
        for (start, end) in &token_ranges {
            assert_eq!(*start, p_end + 1);
            p_end = *end;
        }
        assert_eq!(final_start, p_end + 1);
        assert_eq!(final_end, i64::MAX);
    }
}
