//! Handles searches, including creating/retrieving cursors in the db and sending requests to Elastic

use tracing::instrument;

use super::ElasticCursor;
use crate::models::{ApiCursor, ElasticDoc, ElasticSearchParams};
use crate::utils::{ApiError, Shared};

pub mod events;

/// Search for results matching a query in elastic
///
/// # Arguments
///
/// * `params` - The query params for searching results
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::results::search", skip(shared), err(Debug))]
pub async fn search(
    params: ElasticSearchParams,
    shared: &Shared,
) -> Result<ApiCursor<ElasticDoc>, ApiError> {
    // get our cursor or build a new one
    let mut cursor = ElasticCursor::from_params(params, shared).await?;
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
