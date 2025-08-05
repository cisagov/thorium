//! Handle export operations in the backend
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use chrono::prelude::*;
use futures_util::Future;
use scylla::transport::errors::QueryError;
use scylla::QueryResult;
use std::collections::HashSet;
use tracing::instrument;
use uuid::Uuid;

use crate::is_admin;
use crate::models::exports::ExportErrorRequest;
use crate::models::{
    ApiCursor, Export, ExportError, ExportErrorRow, ExportListParams, ExportOps, ExportRequest,
    ExportUpdate, User,
};
use crate::utils::{ApiError, Shared};

use super::db::{exports, CursorCore, ScyllaCursorSupport};

impl Export {
    /// Make sure this export doesn't already exist and if it doesn't create it
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating an export operation
    /// * `req` - The export request to base this export off of
    /// * `op` - The type of export operation that is being created
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Export::create", skip(user, shared), err(Debug))]
    pub async fn create(
        user: &User,
        req: ExportRequest,
        op: ExportOps,
        shared: &Shared,
    ) -> Result<Export, ApiError> {
        // only admins can create exports
        is_admin!(user);
        // Create this export operation
        exports::create(&user.username, req, op, shared).await
    }

    /// Gets info about a specific export operation by name
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting an export operation
    /// * `name` - The name of the export to get
    /// * `op` - The type of export operation that is being retrieved
    /// * `shared` - Shared Thorium objects
    /// * `req_id` - This requests ID
    #[instrument(name = "Export::get", skip(user, shared), err(Debug))]
    pub async fn get(
        user: &User,
        name: &str,
        op: ExportOps,
        shared: &Shared,
    ) -> Result<Export, ApiError> {
        // only admins can get exports
        is_admin!(user);
        // try to get this export operation
        exports::get(name, op, shared).await
    }

    /// Updates an export operation by name
    ///
    /// # Arguments
    ///
    /// * `update` - The update that is being applied
    /// * `op` - The type of export operation that is being updated
    /// * `update` - The update to apply to this export
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Export::update", skip(self, shared), err(Debug))]
    pub async fn update(
        &self,
        op: ExportOps,
        update: &ExportUpdate,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // try to update this export operation
        exports::update(&self.name, op, update, shared).await
    }

    /// Saves an error that occurred during this export operation
    ///
    /// # Arguments
    ///
    /// * `op` - The type of export operation to add an error for
    /// * `error` - The error to save
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Export::create_error", skip(self, shared), err(Debug))]
    pub async fn create_error(
        &self,
        op: ExportOps,
        error: &ExportErrorRequest,
        shared: &Shared,
    ) -> Result<Uuid, ApiError> {
        // Save this error to scylla
        exports::create_error(self, op, error, shared).await
    }

    /// List the errors for an export operation
    ///
    /// # Arguments
    ///
    /// * `op` - The type of export operation to list errors from
    /// * `params` - The query params to use when listing errors
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "Export::list_error", skip(self, shared), err(Debug))]
    pub async fn list_error(
        &self,
        op: ExportOps,
        params: ExportListParams,
        shared: &Shared,
    ) -> Result<ApiCursor<ExportError>, ApiError> {
        // get a chunk of export errors list
        let scylla_cursor = exports::list_errors(self.name.clone(), op, params, shared).await?;
        // convert our scylla cursor to a user facing cursor
        Ok(ApiCursor::from(scylla_cursor))
    }

    /// Deletes a error from an export operation
    ///
    /// This will not return an error if the error to delete does not exist.
    ///
    /// # Arguments
    ///
    /// * `op` - The type of export operation to delete an error from
    /// * `error_id` - The id of the error to delete
    /// * `shared` - Shared Thorium objects
    pub async fn delete_error(
        &self,
        op: ExportOps,
        error_id: &Uuid,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // delete this export error from scylla
        exports::delete_error(self, op, error_id, shared).await
    }
}

impl From<ExportErrorRow> for ExportError {
    /// Convert an export error row from scylla into an export error
    ///
    /// # Arguments
    ///
    /// * `row` - The row to convert
    fn from(row: ExportErrorRow) -> Self {
        ExportError {
            id: row.id,
            start: row.start,
            end: row.end,
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            code: row.code.map(|code| code as u16),
            msg: row.msg,
        }
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl CursorCore for ExportError {
    /// The params to build this cursor from
    type Params = ExportListParams;

    /// The extra info to filter with
    type ExtraFilters = (String, ExportOps);

    /// The type of data to group our rows by
    type GroupBy = ExportOps;

    /// The data structure to store tie info in
    type Ties = Vec<Uuid>;

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &mut Self::Params) -> Option<Uuid> {
        params.cursor.take()
    }

    // Get our start and end timestamps
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_start_end(
        params: &Self::Params,
        shared: &Shared,
    ) -> Result<(DateTime<Utc>, DateTime<Utc>), ApiError> {
        // get our end timestamp
        let end = params.end(shared)?;
        Ok((params.start, end))
    }

    /// Get any group restrictions from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_group_by(params: &mut Self::Params) -> Vec<Self::GroupBy> {
        std::mem::take(&mut params.kinds)
    }

    /// Get our extra filters from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_extra_filters(_params: &mut Self::Params) -> Self::ExtraFilters {
        unimplemented!("USE FROM PARAMS EXTRA INSTEAD!")
    }

    /// Get our the max number of rows to return
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_limit(params: &Self::Params) -> usize {
        params.limit
    }

    /// Get the partition size for this cursor
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    fn partition_size(shared: &Shared) -> u16 {
        // get our partition size
        shared.config.thorium.results.partition_size
    }

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tie(&self, ties: &mut Self::Ties) {
        // add our tie
        ties.push(self.id);
    }

    /// Determines if a new item is a duplicate or not
    ///
    /// # Arguments
    ///
    /// * `set` - The current set of deduped data
    fn dedupe_item(&self, dedupe_set: &mut HashSet<String>) -> bool {
        // if this is already in our dedupe set then skip it
        dedupe_set.insert(self.id.to_string())
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl ScyllaCursorSupport for ExportError {
    /// The intermediate list row to use
    type IntermediateRow = ExportErrorRow;

    /// The unique key for this cursors row
    type UniqueType<'a> = Uuid;

    /// Get the timestamp from this items intermediate row
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a timestamp for
    fn get_intermediate_timestamp(intermediate: &Self::IntermediateRow) -> DateTime<Utc> {
        intermediate.start
    }

    /// Get the timestamp for this item
    ///
    /// # Arguments
    ///
    /// * `item` - The item to get a timestamp for
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.start
    }

    /// Get the unique key for this intermediate row if it exists
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a unique key for
    fn get_intermediate_unique_key<'a>(
        intermediate: &'a Self::IntermediateRow,
    ) -> Self::UniqueType<'a> {
        intermediate.id
    }

    /// Get the unique key for this row if it exists
    fn get_unique_key<'a>(&'a self) -> Self::UniqueType<'a> {
        self.id
    }

    /// Add a group to a specific returned line
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add to this line
    fn add_group_to_line(&mut self, _group: String) {
        unimplemented!("THIS TYPE DOESN'T DO THIS");
    }

    /// Add a group to a specific returned line
    fn add_intermediate_to_line(&mut self, _intermediate: Self::IntermediateRow) {
        unimplemented!("THIS TYPE DOESN'T DO THIS");
    }

    /// builds the query string for getting data from ties in the last query
    ///
    /// # Arguments
    ///
    /// * `group` - The group that this query is for
    /// * `_filters` - Any filters to apply to this query
    /// * `year` - The year to get data for
    /// * `bucket` - The bucket to get data for
    /// * `uploaded` - The timestamp to get the remaining tied values for
    /// * `breaker` - The value to use as a tie breaker
    /// * `limit` - The max number of rows to return
    /// * `shared` - Shared Thorium objects
    fn ties_query(
        ties: &mut Self::Ties,
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: i32,
        uploaded: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<Vec<impl Future<Output = Result<QueryResult, QueryError>>>, ApiError> {
        // allocate space for 300 futures
        let mut futures = Vec::with_capacity(ties.len());
        // if any ties were found then get the rest of them and add them to data
        for id in ties.drain(..) {
            // execute our query
            let future = shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.exports.list_ties_error,
                (year, bucket, &extra.0, extra.1, uploaded, id, limit),
            );
            // add this future to our set
            futures.push(future);
        }
        Ok(futures)
    }

    /// builds the query string for getting the next page of values
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `_filters` - Any filters to apply to this query
    /// * `year` - The year to get data for
    /// * `bucket` - The bucket to get data for
    /// * `start` - The earliest timestamp to get data from
    /// * `end` - The oldest timestamp to get data from
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    #[allow(clippy::too_many_arguments)]
    async fn pull(
        _group: &Self::GroupBy,
        extra: &Self::ExtraFilters,
        year: i32,
        buckets: Vec<i32>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<QueryResult, QueryError> {
        // execute our query
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.exports.list_pull_error,
                (year, buckets, &extra.0, extra.1, start, end, limit),
            )
            .await
    }
}

#[axum::async_trait]
impl<S> FromRequestParts<S> for ExportListParams
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
