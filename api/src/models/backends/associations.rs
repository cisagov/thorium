//! Backend api support for associations

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use chrono::prelude::*;
use futures_util::Future;
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;
use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use uuid::Uuid;

use super::db;
use crate::models::backends::db::{CursorCore, ScyllaCursor, ScyllaCursorSupport};
use crate::models::{
    ApiCursor, Association, AssociationListParams, AssociationListRow, AssociationRequest,
    AssociationTarget, AssociationTargetColumn, Directionality, Entity, ListableAssociation, Repo,
    Sample, TreeNode, User,
};
use crate::utils::{ApiError, Shared};

impl AssociationTarget {
    /// Make sure this target exists and get its groups
    pub async fn get_groups(&self, user: &User, shared: &Shared) -> Result<Vec<String>, ApiError> {
        match self {
            AssociationTarget::File(sha256) => {
                // get this sample and its groups
                let sample = Sample::get(user, sha256, shared).await?;
                // get this samples groups
                let groups = sample
                    .groups()
                    .into_iter()
                    .map(|group| group.to_owned())
                    .collect::<Vec<String>>();
                Ok(groups)
            }
            AssociationTarget::Repo(url) => {
                // get this repo
                let repo = Repo::get(user, url, shared).await?;
                // get this repos groups
                let groups = repo
                    .groups()
                    .into_iter()
                    .map(|group| group.to_owned())
                    .collect::<Vec<String>>();
                Ok(groups)
            }
            AssociationTarget::Entity { id, .. } => {
                // get a specific entity
                let entity = Entity::get(user, *id, shared).await?;
                // get this entities groups
                Ok(entity.groups)
            }
        }
    }
}

impl AssociationRequest {
    /// Apply this association request to the desired entities/objects
    pub async fn apply(self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        // if we don't have any groups set in this request then get source objects groups
        let (groups, groups_set) = if self.groups.is_empty() {
            // we don't have any groups explicitly set so get our source objects groups
            let groups = self.source.get_groups(user, shared).await?;
            // we didn't have any groups explicitly set
            (groups, false)
        } else {
            (self.groups, true)
        };
        // build a list of targets and the groups to apply their associations too
        let mut target_list = Vec::with_capacity(self.targets.len());
        // keep track of the total number of rows we will need to insert
        let mut size_hint = 0;
        // validate that all target groups are valid and editable
        for target in self.targets {
            // if we didn't have any groups set then get our target objects groups and use their intersection
            let target_groups = if groups_set {
                // we had explicit groups set so just use those
                groups.clone()
            } else {
                // we didn't have any groups set so use their intersection
                // get the groups for this associations target
                let mut target_groups = target.get_groups(user, shared).await?;
                // get the intersection of our groups
                target_groups.retain(|group| groups.contains(group));
                target_groups
            };
            // we will need one insert for each group
            size_hint += target_groups.len();
            // add our target and its validated editable groups
            target_list.push((target, target_groups));
        }
        // get the direction for this association
        let direction = if self.is_bidirectional {
            Directionality::Bidirectional
        } else {
            Directionality::To
        };
        // save this association
        db::associations::create(
            user,
            size_hint,
            self.kind,
            self.source,
            &target_list,
            direction,
            shared,
        )
        .await?;
        Ok(())
    }
}

impl AssociationTarget {
    /// Convert this association target to a column and any extra data
    pub fn to_column(self) -> Result<(AssociationTargetColumn, Option<String>), ApiError> {
        match self {
            AssociationTarget::Entity { id, name } => {
                Ok((AssociationTargetColumn::Entity(id), Some(name)))
            }
            AssociationTarget::File(sha256) => Ok((AssociationTargetColumn::File(sha256), None)),
            AssociationTarget::Repo(url) => Ok((AssociationTargetColumn::Repo(url), None)),
        }
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl CursorCore for ListableAssociation {
    /// The params to build this cursor from
    type Params = AssociationListParams;

    /// The extra info to filter with
    type ExtraFilters = String;

    /// The type of data to group our rows by
    type GroupBy = String;

    /// The data structure to store tie info in
    type Ties = HashMap<String, String>;

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &Self::Params) -> Option<Uuid> {
        params.cursor.clone()
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
        // get our end timestmap
        let end = params.end(shared)?;
        Ok((params.start, end))
    }

    /// Get any group restrictions from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_group_by(params: &mut Self::Params) -> Vec<Self::GroupBy> {
        std::mem::take(&mut params.groups)
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
        shared.config.thorium.associations.partition_size
    }

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tie(&self, ties: &mut Self::Ties) {
        // if its not already in the tie map then add each of its groups to our map
        for group in &self.groups {
            // if this group doesn't already have a tie entry then add it
            ties.entry(group.clone())
                .or_insert_with(|| self.other.clone());
        }
    }

    /// Determines if a new item is a duplicate or not
    ///
    /// # Arguments
    ///
    /// * `set` - The current set of deduped data
    fn dedupe_item(&self, dedupe_set: &mut HashSet<String>) -> bool {
        // if this is already in our dedupe set then skip it
        if dedupe_set.contains(&self.other) {
            // we already have this commit so skip it
            false
        } else {
            // add this new commit to our dedupe set
            dedupe_set.insert(self.other.clone());
            // keep this new commit
            true
        }
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl ScyllaCursorSupport for ListableAssociation {
    /// The intermediate list row to use
    type IntermediateRow = AssociationListRow;

    /// The unique key for this cursors row
    type UniqueType<'a> = (&'a String, Directionality);

    /// Get the timestamp from this items intermediate row
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a timestamp for
    fn get_intermediate_timestamp(intermediate: &Self::IntermediateRow) -> DateTime<Utc> {
        intermediate.created
    }

    /// Get the timestamp for this item
    ///
    /// # Arguments
    ///
    /// * `item` - The item to get a timestamp for
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.created
    }

    /// Get the unique key for this intermediate row if it exists
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a unique key for
    fn get_intermediate_unique_key<'a>(
        intermediate: &'a Self::IntermediateRow,
    ) -> Self::UniqueType<'a> {
        (&intermediate.other, intermediate.direction)
    }

    /// Get the unique key for this row if it exists
    fn get_unique_key<'a>(&'a self) -> Self::UniqueType<'a> {
        (&self.other, self.direction)
    }

    /// Add a group to a specific returned line
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add to this line
    fn add_group_to_line(&mut self, group: String) {
        // add this group
        self.add_group(group);
    }

    /// Add a group to a specific returned line
    fn add_intermediate_to_line(&mut self, intermediate: Self::IntermediateRow) {
        // add this intermediate rows group
        self.add_group(intermediate.group);
    }

    /// Build all of the keys needs to retrieve census data
    ///
    /// # Arguments
    ///
    /// * `group_by` - The values to group our the rows by
    /// * `extra` - The extra values required to list this data
    /// * `year` - The year we are looking for census data for
    /// * `keys` - The vec to add our census stream keys too
    /// * `shared` - Shared Thorium objects
    fn census_keys<'a>(
        group_by: &'a Vec<Self::GroupBy>,
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: u32,
        keys: &mut Vec<(&'a Self::GroupBy, String, i32)>,
        shared: &Shared,
    ) {
        // build the keys for each census stream we are going to crawl
        for group in group_by {
            // build the key for this associations census stream
            let key = super::db::keys::associations::census_stream(group, year, extra, shared);
            // add this key to our keys
            keys.push((group, key, bucket as i32));
        }
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
    ) -> Result<Vec<impl Future<Output = Result<QueryResult, ExecutionError>>>, ApiError> {
        // allocate space for 300 futures
        let mut futures = Vec::with_capacity(ties.len());
        // if any ties were found then get the rest of them and add them to data
        for (group, target) in ties.drain() {
            // execute our query
            let future = shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.associations.list_ties,
                (group, year, bucket, extra, uploaded, target, limit),
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
        group: &Self::GroupBy,
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: Vec<i32>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError> {
        // execute our query
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.associations.list_pull,
                (group, year, bucket, extra, start, end, limit),
            )
            .await
    }
}

impl Association {
    /// List all associations to and from a specific source
    pub async fn list(
        user: &User,
        mut params: AssociationListParams,
        source: &AssociationTargetColumn,
        shared: &Shared,
    ) -> Result<ApiCursor<Association>, ApiError> {
        // make sure these groups are visible to our user
        user.authorize_groups(&mut params.groups, shared).await?;
        // list associations for our some entity/object
        let cursor = db::associations::list(params, source, shared).await?;
        // convet our cursor to an api cursor
        let api_cursor = ApiCursor::try_from(cursor)?;
        Ok(api_cursor)
    }

    /// Generate a tree node hash for this item
    pub fn tree_hash(&self) -> u64 {
        // build a hasher
        let mut hasher = gxhash::GxHasher::with_seed(1234);
        // hash this specific items info
        match &self.other {
            AssociationTarget::Entity { id, .. } => hasher.write_u128(id.as_u128()),
            AssociationTarget::File(sha256) => hasher.write(sha256.as_bytes()),
            AssociationTarget::Repo(url) => hasher.write(url.as_bytes()),
        }
        // finalize our hasher
        let hash = hasher.finish();
        hash
    }

    /// get the data for this association
    pub async fn get_tree_node(&self, user: &User, shared: &Shared) -> Result<TreeNode, ApiError> {
        match &self.other {
            AssociationTarget::File(sha256) => {
                // get this samples data
                let sample = Sample::get(user, sha256, shared).await?;
                Ok(TreeNode::Sample(sample))
            }
            AssociationTarget::Repo(url) => {
                // get this repos data
                let repo = Repo::get(user, url, shared).await?;
                Ok(TreeNode::Repo(repo))
            }
            AssociationTarget::Entity { id, .. } => {
                // get this entities data
                let entity = Entity::get(user, *id, shared).await?;
                Ok(TreeNode::Entity(entity))
            }
        }
    }
}

impl TryFrom<ScyllaCursor<ListableAssociation>> for ApiCursor<Association> {
    /// The error to return on failures
    type Error = ApiError;

    /// Convert a scylla cursor to an api cursor for associations
    fn try_from(cursor: ScyllaCursor<ListableAssociation>) -> Result<Self, Self::Error> {
        // get our cursor id if we aren't exhausted yet
        let id = if cursor.exhausted() {
            None
        } else {
            Some(cursor.id)
        };
        // build alist of assocations
        let data = cursor
            .data
            .into_iter()
            .map(|listable| Association::try_from(listable))
            .collect::<Result<Vec<Association>, ApiError>>()?;
        // build our api cursor with our converted data
        Ok(ApiCursor { cursor: id, data })
    }
}

impl<S> FromRequestParts<S> for AssociationListParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new()
                .max_depth(5)
                .deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}
