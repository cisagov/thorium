use super::keys::logs;
use crate::models::StatusUpdate;
use crate::serialize;
use crate::utils::{ApiError, Shared};

/// Builds a [`redis::Pipeline`] with commands to push [`StatusUpdate`]s to Redis
///
/// # Arguments
///
/// * `pipe` - The Redis [`redis::Pipeline`] to build on top of
/// * `job` - The job object to add to redis
/// * `shared` - Shared Thorium objects
pub fn build<'a>(
    pipe: &'a mut redis::Pipeline,
    casts: &[StatusUpdate],
    shared: &Shared,
) -> Result<&'a mut redis::Pipeline, ApiError> {
    // inject commands to push status logs updates to their respective lists
    for update in casts {
        pipe.cmd("rpush")
            .arg(logs::queue_name(update, shared))
            .arg(serialize!(&update));
    }
    Ok(pipe)
}
