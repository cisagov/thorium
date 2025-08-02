//! The features for working with census data in redis

use crate::conn;
use crate::models::CensusKeys;
use crate::utils::{ApiError, Shared};

/// Increment the cached count for these census keys
///
/// * `keys` - The census keys to update
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn incr_cache(
    keys: Vec<CensusKeys>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build a redis pipeline and update our cache info
    let mut pipe = redis::pipe();
    // add each key to this pipeline
    for key in keys {
        pipe.cmd("hincrby").arg(key.count).arg(key.bucket).arg(1)
            .cmd("zadd").arg(key.stream).arg(key.bucket).arg(key.bucket);
    }
    // increment our cache info
    pipe.exec_async(conn!(shared)).await?;
    Ok(())
}

/// Decrement the cached count for these census keys
///
/// * `keys` - The census keys to update
/// * `bucket` - The buckets to use
/// * `grouping` - This census keys groupings
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn decr_cache(
    keys: Vec<CensusKeys>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build a redis pipeline and update our cache info
    let mut pipe = redis::pipe();
    // add each key to this pipeline
    for key in keys {
        pipe.cmd("hincrby").arg(key.count).arg(key.bucket).arg(-1);
    }
    // increment our cache info
    pipe.exec_async(conn!(shared)).await?;
    Ok(())
}
