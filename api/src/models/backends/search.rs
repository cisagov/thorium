//! Handles search requests in the backend

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use chrono::{DateTime, TimeZone, Utc};

use super::db;
use crate::models::{ApiCursor, ElasticDoc, ElasticIndex, ElasticSearchParams, User};
use crate::utils::{ApiError, Shared};

pub mod events;

/// Search results in elastic and return a list of sha256s
///
/// # Arguments
///
/// * `user` - The user that is listing samples
/// * `params` - The query params for searching results
/// * `shared` - Shared objects in Thorium
pub async fn search(
    user: &User,
    mut params: ElasticSearchParams,
    shared: &Shared,
) -> Result<ApiCursor<ElasticDoc>, ApiError> {
    // authorize the groups to list files from
    user.authorize_groups(&mut params.groups, shared).await?;
    // search for results documents in elastic
    db::search::search(params, shared).await
}

impl ElasticSearchParams {
    /// Get the earliest date for documents in all of the given indexes
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    ///
    /// # Panics
    ///
    /// Panics if `self` has no indexes and [`ElasticIndex`] has no variants (there are no possible indexes)
    pub fn end(&self, shared: &Shared) -> Result<DateTime<Utc>, ApiError> {
        match self.end {
            Some(end) => Ok(end),
            None => {
                // get the earliest a document can be found in all of the given indexes
                let earliest = self
                    .indexes
                    .iter()
                    .map(|index| index.earliest(shared))
                    .min()
                    .unwrap_or_else(|| {
                        // if we have no indexes, just get the earliest for all indexes
                        ElasticIndex::earliest_all(shared)
                    });
                match Utc.timestamp_opt(earliest, 0) {
                    chrono::LocalResult::Single(default_end) => Ok(default_end),
                    _ => crate::internal_err!(format!(
                        "default earliest timestamp is invalid or ambiguous - {earliest}",
                    )),
                }
            }
        }
    }
}

impl<S> FromRequestParts<S> for ElasticSearchParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}
