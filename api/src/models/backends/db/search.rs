//! Handles searches, including creating/retrieving cursors in the db and sending requests to Elastic

use tracing::{Level, event, instrument};

use super::ElasticCursor;
use crate::models::{ApiCursor, ElasticDoc, ElasticSearchParams, User};
use crate::utils::{ApiError, Shared};

pub mod events;

/// Search for results matching a query in elastic
///
/// # Arguments
///
/// * `user` - The user that is searching results
/// * `params` - The query params for searching results
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::results::search", skip(user, shared), err(Debug))]
pub async fn search(
    user: &User,
    params: ElasticSearchParams,
    shared: &Shared,
) -> Result<ApiCursor<ElasticDoc>, ApiError> {
    // if are searching no groups just shortcircuit and return nothing
    if params.groups.is_empty() {
        // log that this user has no groups
        event!(Level::WARN, msg = "User has no groups");
        // return an empty cursor
        return Ok(ApiCursor {
            cursor: None,
            data: Vec::default(),
        });
    }
    // get our cursor or build a new one scoped to the creating user
    let mut cursor = ElasticCursor::from_params(params, &user.username, shared).await?;
    //  get the next page of data
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    let data = std::mem::take(&mut cursor.data);
    // determine if this cursor has been exhausted or not
    if data.len() < cursor.limit as usize {
        // this cursor is exhausted so omit the cursor ID
        Ok(ApiCursor { cursor: None, data })
    } else {
        // this cursor is exhausted so omit the cursor ID
        Ok(ApiCursor {
            cursor: Some(cursor.id),
            data,
        })
    }
}
