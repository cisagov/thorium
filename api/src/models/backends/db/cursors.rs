//! Cursors for data in Thorium
use bb8_redis::redis::cmd;
use chrono::prelude::*;
use elasticsearch::SearchParts;
use futures::stream::{self, StreamExt};
use futures_util::Future;
use scylla::client::pager::QueryPager;
use scylla::deserialize::row::DeserializeRow;
use scylla::errors::{ExecutionError, PagerExecutionError};
use scylla::response::query_result::QueryResult;
use scylla::serialize::value::SerializeValue;
use scylla::statement::prepared::PreparedStatement;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::{Ord, Ordering};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::elastic::{self, ElasticResponse};
use super::keys::{cursors, tags};
use crate::models::{ApiCursor, CensusKeys, CensusSupport, ElasticDoc, TagListRow};
use crate::models::{ElasticSearchParams, TagType};
use crate::utils::{helpers, ApiError, Shared};
use crate::{
    bad, conn, deserialize, internal_err, internal_err_unwrapped, log_scylla_err, not_found, query,
    serialize,
};

/// The different kinds of cursors
pub enum CursorKind {
    /// A group based cursor in Scylla
    Scylla,
    /// A very basic non bucketed cursor in Scylla
    SimpleScylla,
    /// A group-based cursor in Scylla with no concept of time bucketing
    GroupedScylla,
    /// A cursor based on data in Elastic
    Elastic,
    /// A Tree of data in Thorium
    Tree,
}

impl CursorKind {
    /// Get our cursor kind as a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            CursorKind::Scylla => "Scylla",
            CursorKind::SimpleScylla => "SimpleScylla",
            CursorKind::GroupedScylla => "GroupedScylla",
            CursorKind::Elastic => "Elastic",
            CursorKind::Tree => "Tree",
        }
    }
}

/// The data to retain throughout this cursors life
#[derive(Serialize, Deserialize, Debug)]
pub struct ScyllaCursorRetain<D: CursorCore> {
    /// The start timestamp for this cursor
    pub start: DateTime<Utc>,
    /// The end timestamp for this cursor
    pub end: DateTime<Utc>,
    /// The tag based filters to use with this cursor
    pub extra_filter: D::ExtraFilters,
    /// The values to group the rows from this cursor by
    pub group_by: Vec<D::GroupBy>,
    /// Any ties in past iterations of this cursor
    pub ties: D::Ties,
    /// Tags data to retain if this is a tags cursor
    pub tags_retain: Option<TagsRetain>,
}

/// Tag data to retain throughout a cursor's life if it's
/// a tags cursor
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TagsRetain {
    /// Whether this cursor should crawl the tags DB or not
    pub tags: HashMap<String, Vec<String>>,
    /// The type of tags we're crawling
    pub tag_type: TagType,
    /// The total number of tags we are searching
    pub tags_required: usize,
    /// Whether we're searching on case-insensitive tags
    pub case_insensitive: bool,
    /// Any ties in past iterations in a tag based cursor
    pub ties: HashMap<String, String>,
}

/// The core logic all cursors must implement
pub trait CursorCore: Debug + Serialize + for<'a> Deserialize<'a> {
    /// The params to build this cursor form
    type Params;

    /// The extra info to filter with
    type ExtraFilters: Debug + Serialize + for<'a> Deserialize<'a>;

    /// The type of data to group our rows by
    type GroupBy: std::fmt::Display
        + Debug
        + Clone
        + Hash
        + Eq
        + PartialEq
        + Serialize
        + SerializeValue
        + for<'a> Deserialize<'a>;

    /// The data structure to store tie info in
    type Ties: Debug + Default + Serialize + for<'a> Deserialize<'a>;

    /// The number of buckets to crawl at once for non tag queries
    ///
    /// This is 99 by default
    ///
    /// # Arguments
    ///
    /// * `extra_filters` - The extra filters for this query
    fn bucket_limit(_extra_filters: &Self::ExtraFilters) -> u32 {
        99
    }

    /// Get the partition size for this cursor
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    fn partition_size(shared: &Shared) -> u16;

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &mut Self::Params) -> Option<Uuid>;

    // Get our start and end timestamps
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_start_end(
        params: &Self::Params,
        shared: &Shared,
    ) -> Result<(DateTime<Utc>, DateTime<Utc>), ApiError>;

    /// Get any values to group rows in this cursor by from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_group_by(params: &mut Self::Params) -> Vec<Self::GroupBy>;

    /// Get our extra filters from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_extra_filters(params: &mut Self::Params) -> Self::ExtraFilters;

    /// Get our tag filters from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_tag_filters(
        _params: &mut Self::Params,
    ) -> Option<(TagType, HashMap<String, Vec<String>>)> {
        None
    }

    /// Get whether the matching on tags should be case-insensitive
    ///
    /// # Arguments
    ///
    /// `params` - The params to use to build this cursor
    fn get_tags_case_insensitive(_params: &Self::Params) -> bool {
        // return false by default
        false
    }

    /// Get our the max number of rows to return
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_limit(params: &Self::Params) -> usize;

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tie(&self, ties: &mut Self::Ties);

    /// Determines if a new item is a duplicate or not
    ///
    /// # Arguments
    ///
    /// * `set` - The current set of deduped data
    fn dedupe_item(&self, dedupe_set: &mut HashSet<String>) -> bool;

    /// Get the tag clustering key for a row without the timestamp
    fn get_tag_clustering_key(&self) -> &String {
        unimplemented!("This type does not support tags");
    }
}

/// Abstracts cursor logic across different tables and data types
#[async_trait::async_trait]
pub trait ScyllaCursorSupport: CursorCore {
    /// The intermediate list row to use
    type IntermediateRow: Into<Self> + for<'frame, 'metadata> DeserializeRow<'frame, 'metadata>;

    /// The unique key for this cursors row
    type UniqueType<'a>: Debug + PartialEq;

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tag_tie(&self, _ties: &mut HashMap<String, String>) {
        unimplemented!("This type does not support tags");
    }

    /// Get the timestamp from this items intermediate row
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a timestamp for
    fn get_intermediate_timestamp(intermediate: &Self::IntermediateRow) -> DateTime<Utc>;

    /// Get the timestamp for this item
    ///
    /// # Arguments
    ///
    /// * `item` - The item to get a timestamp for
    fn get_timestamp(&self) -> DateTime<Utc>;

    /// Get the unique key for this intermediate row
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a unique key for
    fn get_intermediate_unique_key<'a>(
        intermediate: &'a Self::IntermediateRow,
    ) -> Self::UniqueType<'a>;

    /// Get the unique key for this row
    fn get_unique_key<'a>(&'a self) -> Self::UniqueType<'a>;

    /// Add a group to a specific returned line
    fn add_group_to_line(&mut self, _group: String) {
        unimplemented!("This type does not support tags");
    }

    /// Add a group to a specific returned line
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a unique key for
    fn add_intermediate_to_line(&mut self, intermediate: Self::IntermediateRow);

    /// Convert a tag list row into our list line
    ///
    /// # Arguments
    ///
    /// * `row` - The tag row to convert
    fn from_tag_row(_row: TagListRow) -> Self {
        unimplemented!("Doesn't support tags");
    }

    /// Build all of the keys needs to retrieve census data
    fn census_keys<'a>(
        groups: &'a Vec<Self::GroupBy>,
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: u32,
        keys: &mut Vec<(&'a Self::GroupBy, String, i32)>,
        shared: &Shared,
    );

    /// builds the query string for getting data from ties in the last query
    ///
    /// # Arguments
    ///
    /// * `group` - The group that this query is for
    /// * `filters` - Any filters to apply to this query
    /// * `year` - The year to get data for
    /// * `bucket` - The bucket to get data for
    /// * `uploaded` - The timestamp to get the remaining tied values for
    /// * `breaker` - The value to use as a tie breaker
    /// * `limit` - The max number of rows to return
    /// * `shared` - Shared Thorium objects
    #[allow(clippy::too_many_arguments)]
    fn ties_query(
        ties: &mut Self::Ties,
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: i32,
        uploaded: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<Vec<impl Future<Output = Result<QueryResult, ExecutionError>>>, ApiError>;

    /// builds the query string for getting the next page of values
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our query too
    /// * `filters` - Any filters to apply to this query
    /// * `year` - The year to get data for
    /// * `buckets` - The buckets to get data for
    /// * `start` - The earliest timestamp to get data from
    /// * `end` - The oldest timestamp to get data from
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    #[allow(clippy::too_many_arguments)]
    async fn pull(
        group: &Self::GroupBy,
        extra: &Self::ExtraFilters,
        year: i32,
        buckets: Vec<i32>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError>;

    /// Convert [`QueryResult`]s into a sorted Vec of self
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to cast and sort
    /// * `sorted` - The map to store sorted data in
    /// * `mapped` - The currently available number of rows we have sorted
    /// * `tags` - Whether this a tags cursor not not
    /// * `span` - The span to log traces under
    async fn sort(
        queries: Vec<QueryResult>,
        sorted: &mut BTreeMap<DateTime<Utc>, VecDeque<Self>>,
        mapped: &mut usize,
    ) -> Result<(), ApiError>
    where
        Self: Sized,
    {
        // crawl over each stream and cast their rows to partition counts
        for query in queries {
            // enable casting to types for this query
            let query_rows = query.into_rows_result()?;
            // set the type to cast this stream too
            let mut typed_stream = query_rows.rows::<Self::IntermediateRow>()?;
            // cast our rows to typed values
            while let Some(row) = typed_stream.next() {
                // raise any errors from casting
                let cast = row?;
                // get the timestamp and unique key for our intermediate row
                let timestamp = Self::get_intermediate_timestamp(&cast);
                let inter_unique = Self::get_intermediate_unique_key(&cast);
                // get an entry to the list for this timestamp
                let entry = sorted.entry(timestamp).or_default();
                // check if this entry has already been added to this vec
                let pos = entry
                    .iter()
                    .position(|item| inter_unique == item.get_unique_key());
                // drop our reference to our intermediate unique value
                drop(inter_unique);
                // if we found this item already exists then add our intermediate row
                match pos {
                    Some(pos) => {
                        // get the right list line and add our intermediate row
                        entry.get_mut(pos).unwrap().add_intermediate_to_line(cast);
                    }
                    None => {
                        // cast our row and add it
                        entry.push_back(cast.into());
                        // increment our mapped count
                        *mapped += 1
                    }
                }
            }
        }
        Ok(())
    }

    /// Sort any possibly ambiguous rows by their cluster key
    fn sort_by_cluster_key(ambiguous: &mut VecDeque<Self>) {
        // create a btreemap to sort this data
        let mut sorted = BTreeMap::default();
        // keep a list of our ambiguous rows
        let mut to_sort = HashMap::with_capacity(ambiguous.len());
        // go through and get our clustering keys and insert them into our map
        for (index, row) in ambiguous.drain(..).enumerate() {
            // get this rows clustering key
            let ckey = row.get_tag_clustering_key().clone();
            // insert this into our map
            sorted.insert(ckey, index);
            // add this row to our sort list
            to_sort.insert(index, row);
        }
        // rebuild this ambiguous row list in order
        for (_, index) in sorted.iter() {
            // get this row and insert it
            if let Some(row) = to_sort.remove(&index) {
                // add this row in the correct order
                ambiguous.push_back(row);
            }
        }
    }

    /// Convert [`QueryResult`]s into a sorted Vec of self for tag based queries
    ///
    /// # Arguments
    ///
    /// * `mapping` - The mapping of items and their tag data to sort
    /// * `tags` - Whether this a tags cursor not not
    /// * `sorted` - The map to store sorted data in
    /// * `mapped_count` - The currently available number of rows we have sorted
    fn sort_tags(
        mapping: &mut HashMap<String, TagMapping>,
        tags: &HashMap<String, Vec<String>>,
        sorted: &mut BTreeMap<DateTime<Utc>, VecDeque<Self>>,
        mapped_count: &mut usize,
    ) -> Result<(), ApiError>
    where
        Self: Sized,
    {
        // count the number of key/value pairs we are trying to match
        let required = tags.values().map(|values| values.len()).sum::<usize>();
        // go through this list in order
        for (_, mapped) in mapping.drain() {
            // skip any mappings where we don't have the required number of tags
            if mapped.tags.len() == required {
                // turn our this items casts into a stream
                let mut row_stream = mapped.rows.into_iter();
                // build the sample list line from the first row
                let mut line = Self::from_tag_row(row_stream.next().unwrap());
                // add all the rest of our row to this line
                for row in row_stream {
                    line.add_group_to_line(row.group);
                }
                // get this lines timestamp
                let timestamp = line.get_timestamp();
                // get an entry to the list for this timestamp
                let entry = sorted.entry(timestamp).or_default();
                // add our new sample list line
                entry.push_back(line);
                // increment our mapped count
                *mapped_count += 1;
            }
        }
        Ok(())
    }
}

/// A row containing a count of objects within a partition
pub struct CursorCountRow {
    /// The group this row is for
    pub group: Option<String>,
    /// The year this count is from
    pub year: Option<i32>,
    /// The bucket this count is from
    pub bucket: Option<i32>,
    /// The number of objects in this count
    pub count: i64,
}

/// A count of objects from a group for a specific year/bucket combo
#[derive(Debug)]
pub struct CursorCount {
    /// The year this count is from
    pub year: i32,
    /// The bucket this count is from
    pub bucket: i32,
    /// The number of objects in this count
    pub count: i64,
}

fn update_first_last(
    first: &mut i32,
    last: &mut i32,
    oldest_first: &mut i32,
    new_first: i32,
    new_last: i32,
) {
    // check if this bucket range moves the first bucket with data down
    if new_first < *first {
        *first = new_first;
    }
    // check if this bucket range brings the overlapping buckets with data up
    if new_last > *last {
        *last = new_last;
    }
    // check if this increments the largest bucket found so far
    if new_first < *oldest_first {
        *oldest_first = new_first;
    }
}

/// A tie query for data based on tags
async fn ties_tags_query_helper<'a, D: CursorCore>(
    ties_prepared: &PreparedStatement,
    kind: TagType,
    group: &str,
    year: i32,
    bucket: i32,
    key: &'a str,
    value: &'a str,
    start: DateTime<Utc>,
    breaker: &str,
    shared: &Shared,
) -> Result<(&'a str, &'a str, QueryPager), PagerExecutionError> {
    // build a paged cursor for this data
    let query = shared
        .scylla
        .session
        .execute_iter(
            ties_prepared.clone(),
            (kind, group, year, bucket, key, value, start, breaker),
        )
        .await?;
    Ok((key, value, query))
}

/// A query for data based on tags
#[instrument(
    name = "ScyllaCursor::tags_query_helper",
    skip(kind, group, buckets, start, end, shared),
    fields(buckets_len = buckets.len()),
    err(Debug)
)]
async fn tags_query_helper<'a, D: CursorCore>(
    query_prepared: &PreparedStatement,
    kind: TagType,
    group: &D::GroupBy,
    year: i32,
    buckets: Vec<i32>,
    key: &'a str,
    value: &'a str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    shared: &Shared,
) -> Result<(&'a str, &'a str, QueryPager), PagerExecutionError> {
    // build the query to pull a page of matching tag data
    let query = shared
        .scylla
        .session
        .execute_iter(
            query_prepared.clone(),
            (kind, group, year, buckets.clone(), key, value, start, end),
        )
        .await?;
    // check if there is more
    Ok((key, value, query))
}

#[derive(Default, Debug)]
pub struct TagMapping {
    /// They key/values this data has
    tags: HashSet<(String, String)>,
    /// The rows for this tag
    rows: Vec<TagListRow>,
}

impl TagMapping {
    /// Add a row to this mapping
    ///
    /// # Arguments
    ///
    /// * `key` - The key to add
    /// * `value` - The value to add
    /// * `row` - The row to add
    pub fn add(&mut self, key: &str, value: &str, row: TagListRow) {
        // add our new key/value
        self.tags.insert((key.to_string(), value.to_string()));
        // add this row
        self.rows.push(row);
    }
}

async fn parse_tag_queries<'a>(
    queries: Vec<(&'a str, &'a str, QueryPager)>,
    mapping: &mut HashMap<String, TagMapping>,
) -> Result<(), ApiError> {
    // crawl over our queries and deserialize them
    for (key, value, query) in queries {
        // set the type to cast this stream too
        let mut typed_stream = query.rows_stream::<TagListRow>()?;
        // cast our rows to typed values
        while let Some(row) = typed_stream.next().await {
            // check if we failed to deserialize this row
            let cast = row?;
            // get an entry to this rows info in our tag mapping map
            let entry: &mut TagMapping = mapping.entry(cast.item.clone()).or_default();
            // add this tag row to our mapping
            entry.add(key, value, cast);
        }
    }
    Ok(())
}

/// A cursor for a listing group permissioned data within scylla
#[derive(Debug)]
pub struct ScyllaCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: ScyllaCursorSupport,
    D: Debug,
{
    /// The ID for this cursor
    pub id: Uuid,
    /// The cursor settings/data to retain across cursor iterations
    pub retain: ScyllaCursorRetain<D>,
    /// The year that we last got data from
    pub year: i32,
    /// The bucket that we last got data from
    pub bucket: u32,
    /// The final year we are going to get data from
    pub end_year: i32,
    /// The final bucket we will get data from
    pub end_bucket: u32,
    /// The partition size to use
    partition_size: u16,
    /// The max number of items to return at once
    pub limit: usize,
    /// Whether to dedupe this data or not
    pub dedupe: bool,
    /// The set of values to use when deduping
    dedupe_set: HashSet<String>,
    /// The data this cursor has retrieved
    pub data: Vec<D>,
    /// The internal sorted data to return to the user
    sorted: BTreeMap<DateTime<Utc>, VecDeque<D>>,
    /// The currently available number of rows we have sorted
    pub mapped: usize,
    /// whether this cursor has been exhausted or not
    pub buckets_exhausted: bool,
}

/// lowercase our tags if needed
fn maybe_lower_tags<D: ScyllaCursorSupport>(
    params: &D::Params,
    tags: HashMap<String, Vec<String>>,
) -> (bool, HashMap<String, Vec<String>>) {
    // check if this is a case insensitive filter
    if D::get_tags_case_insensitive(params) {
        // this is case insensitive so shift all lowercasable chars to their lowercase variants
        // pre allocate a map for our lowercased tags
        let mut lowered: HashMap<String, Vec<String>> = HashMap::with_capacity(tags.len());
        // step over each tag key and its values and lowercase them if possible
        for (key, values) in &tags {
            // lowercase our key
            let lower_key = key.to_lowercase();
            // get an entry to this newly lowercased tag key
            let entry = lowered.entry(lower_key).or_default();
            // build an iter of lowercased values
            let lower_values = values.iter().map(|value| value.to_lowercase());
            // extend this lowercased key with its lowercased values
            entry.extend(lower_values);
        }
        // return our lowercased tags
        (true, lowered)
    } else {
        // this is not case insensitive so just return our current tags
        (false, tags)
    }
}

impl<D> ScyllaCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug + std::marker::Send,
    D: Serialize,
    D: ScyllaCursorSupport,
    D: Debug,
{
    /// Get the retain info for tags if this is a tags cursor
    ///
    /// # Arguments
    ///
    /// * `params` - The params to build this cursor from
    fn get_tags_retain(params: &mut D::Params) -> Option<TagsRetain> {
        // get our tag filters
        if let Some((tag_type, tags)) = D::get_tag_filters(params) {
            // lowercase our tags if necesary
            let (case_insensitive, tags) = maybe_lower_tags::<D>(params, tags);
            // calculate the total number of tags required for this query
            let tags_required = tags.values().map(Vec::len).sum();
            Some(TagsRetain {
                tags,
                tag_type,
                tags_required,
                case_insensitive,
                ties: HashMap::new(),
            })
        } else {
            None
        }
    }

    /// Create a new cursor object from just params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to build this cursor from
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "ScyllaCursor::from_params",
        skip(params, dedupe, shared),
        err(Debug)
    )]
    pub async fn from_params(
        mut params: D::Params,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&mut params) {
            ScyllaCursor::get(id, params, dedupe, shared).await
        } else {
            // this is a new cursor
            // get the start/end timestamps from our cursor
            let (start, end) = D::get_start_end(&params, shared)?;
            // make sure that start is not before end
            if start < end {
                return bad!("You cannot start before you end!".to_owned());
            }
            // get the years this cursor should start and end in
            let year = start.year();
            let end_year = end.year();
            // get data for tags from our params
            let tags_retain = Self::get_tags_retain(&mut params);
            let chunk = if tags_retain.is_some() {
                // this is a tag based cursor so use our tag partition size
                shared.config.thorium.tags.partition_size
            } else {
                // this is not a tag based query so use our types partitions size
                D::partition_size(shared)
            };
            // get our buckets
            let bucket = u32::try_from(helpers::partition(start, year, chunk))?;
            let end_bucket = u32::try_from(helpers::partition(end, end_year, chunk))?;
            // get our extra filters
            let extra_filter = D::get_extra_filters(&mut params);
            // get our group restrictions
            let groups = D::get_group_by(&mut params);
            // build the inital data to retain across cursor iterations
            let retain = ScyllaCursorRetain {
                start,
                end,
                extra_filter,
                group_by: groups,
                ties: D::Ties::default(),
                tags_retain,
            };
            // build our cursor
            let cursor = ScyllaCursor {
                id: Uuid::new_v4(),
                retain,
                year,
                bucket,
                end_year,
                end_bucket,
                partition_size: chunk,
                limit: D::get_limit(&params),
                dedupe,
                dedupe_set: HashSet::default(),
                data: Vec::default(),
                sorted: BTreeMap::default(),
                mapped: 0,
                buckets_exhausted: false,
            };
            Ok(cursor)
        }
    }

    /// Create a new cursor object from params with a set extra fields
    ///
    /// # Arguments
    ///
    /// * `params` - The params to build this cursor from
    /// * `extra_filter` - The extra filters to use in this cursor
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "ScyllaCursor::from_params_extra",
        skip(params, extra_filter, shared),
        err(Debug)
    )]
    pub async fn from_params_extra(
        mut params: D::Params,
        extra_filter: D::ExtraFilters,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&mut params) {
            ScyllaCursor::get(id, params, dedupe, shared).await
        } else {
            // this is a new cursor
            // get the start/end timestamps from our cursor
            let (start, end) = D::get_start_end(&params, shared)?;
            // make sure that start is not before end
            if start < end {
                return bad!("You cannot start before you end!".to_owned());
            }
            // get the years this cursor should start and end in
            let year = start.year();
            let end_year = end.year();
            // get data for tags from our params
            let tags_retain = Self::get_tags_retain(&mut params);
            let chunk = if tags_retain.is_some() {
                // this is a tag based cursor so use our tag partition size
                shared.config.thorium.tags.partition_size
            } else {
                // this is not a tag based query so use our types partitions size
                D::partition_size(shared)
            };
            // get our buckets
            let bucket = u32::try_from(helpers::partition(start, year, chunk))?;
            let end_bucket = u32::try_from(helpers::partition(end, end_year, chunk))?;
            // get our group restrictions
            let groups = D::get_group_by(&mut params);
            // build the inital data to retain across cursor iterations
            let retain = ScyllaCursorRetain {
                start,
                end,
                extra_filter,
                group_by: groups,
                ties: D::Ties::default(),
                tags_retain,
            };
            // build our cursor
            let cursor = ScyllaCursor {
                id: Uuid::new_v4(),
                retain,
                year,
                bucket,
                end_year,
                end_bucket,
                partition_size: chunk,
                limit: D::get_limit(&params),
                dedupe,
                dedupe_set: HashSet::default(),
                data: Vec::default(),
                sorted: BTreeMap::default(),
                mapped: 0,
                buckets_exhausted: false,
            };
            Ok(cursor)
        }
    }

    /// Gets a cursors data from Redis
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The uuid of the cursor to retrieve if one is known
    /// * `params` - The params to build this cursor from
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared thorium objects
    #[instrument(name = "ScyllaCursor::get", skip(params, shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        params: D::Params,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ScyllaCursor<D>, ApiError> {
        // build the key to our cursor data in redis
        let key = cursors::data(CursorKind::Scylla, &cursor_id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // try to deserialize this cursors retained data
                let retain: ScyllaCursorRetain<D> = deserialize!(&data);
                // tag based cursors have a different chunk size
                let chunk = if retain.tags_retain.is_some() {
                    shared.config.thorium.tags.partition_size
                } else {
                    // this is not a tag based query so use our types partitions size
                    D::partition_size(shared)
                };
                // determine our current year and our end year
                let year = retain.start.year();
                let end_year = retain.end.year();
                // get our starting/ending bucket
                let bucket = u32::try_from(helpers::partition(retain.start, year, chunk))?;
                let end_bucket = u32::try_from(helpers::partition(retain.end, end_year, chunk))?;
                // rebuild our cursor
                let cursor = ScyllaCursor {
                    id: cursor_id,
                    retain,
                    year,
                    bucket,
                    end_year,
                    end_bucket,
                    partition_size: chunk,
                    limit: D::get_limit(&params),
                    dedupe,
                    dedupe_set: HashSet::default(),
                    data: Vec::default(),
                    sorted: BTreeMap::default(),
                    mapped: 0,
                    buckets_exhausted: false,
                };
                Ok(cursor)
            }
            None => not_found!(format!("Cursor {} doesn't exist", cursor_id)),
        }
    }

    /// Check if we have any ties queries
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of rows to return to the user plus 1
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::query_ties", skip(self, shared), err(Debug))]
    async fn query_ties(
        &mut self,
        limit: i32,
        shared: &Shared,
    ) -> Result<Vec<QueryResult>, ApiError> {
        // build our tie query futures
        let futures = D::ties_query(
            &mut self.retain.ties,
            &self.retain.extra_filter,
            self.year,
            self.bucket as i32,
            self.retain.start,
            limit,
            shared,
        )?;
        // wait for all of our futures to complete 50 at a time
        let queries = stream::iter(futures)
            .buffer_unordered(50)
            .collect::<Vec<Result<QueryResult, ExecutionError>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<QueryResult>, ExecutionError>>()?;
        Ok(queries)
    }

    /// Find the buckets that have been confirmed to contain data
    #[rustfmt::skip]
    async fn find_buckets<'a>(
        &self,
        stream_keys: &mut Vec<(&'a D::GroupBy, String, i32)>,
        found: &mut HashMap<&'a D::GroupBy, Vec<i32>>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // set our end bucket to be f64::MAX or our end bucket if we are the end year
        let end = if self.year == self.end_year {
            self.end_bucket
        } else {
            0
        };
        // have an allocated vec to swap with our stream keys vec
        let mut swap: Vec<(&D::GroupBy, String, i32)> = Vec::with_capacity(self.retain.group_by.len());
        // build a redis pipeline to get all of the valid buckets for these tags
        let mut pipe = redis::pipe();
        // do this for each stream key
        for (_, stream_key, oldest_first) in stream_keys.iter() {
            // get the next 100 items
            pipe.cmd("zrange").arg(stream_key).arg(oldest_first).arg(end)
                    .arg("byscore").arg("limit").arg(0).arg(100).arg("rev");
        }
        // execute our queries
        let bucket_strings: Vec<Vec<String>> = pipe.query_async(conn!(shared)).await?;
        // swap our stream keys
        std::mem::swap(stream_keys, &mut swap);
        // convert all of our buckets to signed ints
        for (buckets_string, (group, stream_key, _)) in bucket_strings.into_iter().zip(swap.into_iter()) {
            // convert these buckets to an i32
            let buckets = buckets_string
                .iter()
                .map(|val| val.parse::<i32>())
                .collect::<Result<Vec<i32>, _>>()?;
            // get the final item in this list
            if let Some(oldest_first) = buckets.last() {
                // increment our oldest first by 1 so we don't loop over buckets
                let incr_oldest_first = *oldest_first - 1;
                // add our new oldest first
                stream_keys.push((group, stream_key, incr_oldest_first));
                // add these buckets to our found buckets
                found.insert(group, buckets);
            }
        }
        Ok(())
    }

    /// Crawl partitions and pull data from them
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of rows to return to the user plus 1
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::query", skip(self, shared), err(Debug))]
    async fn query(&mut self, limit: i32, shared: &Shared) -> Result<(), ApiError> {
        // allocate space for the futures we are about to spawn
        let mut futures = Vec::with_capacity(self.retain.group_by.len());
        // build all of the keys we are getting valid buckets for
        let mut stream_keys = Vec::with_capacity(self.retain.group_by.len());
        // keep a map of the buckets we have found by group
        let mut found = HashMap::with_capacity(self.retain.group_by.len());
        // loop until we have enough data to return
        loop {
            // if stream keys are empty then recreate them
            if stream_keys.is_empty() {
                // build the keys for each census stream we are going to crawl
                D::census_keys(
                    &self.retain.group_by,
                    &self.retain.extra_filter,
                    self.year,
                    self.bucket,
                    &mut stream_keys,
                    shared,
                );
            }
            // get the next buckets that contain data
            self.find_buckets(&mut stream_keys, &mut found, shared)
                .await?;
            // if we found no buckets then check if we have exhausted this cursor
            if found.is_empty() {
                // if we are in the final year then this cursor is exhausted
                if self.year <= self.end_year {
                    // set this cursors buckets  to be exhausted
                    self.buckets_exhausted = true;
                    // this cursor is exhausted so return
                    return Ok(());
                }
                // there is no more data in this year but we have more years to check
                // decrement our year
                self.year = self.year.saturating_sub(1);
                // get a duration for the next year
                let year = NaiveDate::from_ymd_opt(self.year, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 1)
                    .unwrap();
                let next_year = NaiveDate::from_ymd_opt(self.year - 1, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 1)
                    .unwrap();
                let duration = year - next_year;
                // reset our bucket to the last possible partition of the year
                self.bucket = duration.num_seconds() as u32 / self.partition_size as u32;
                // go to the next loop iteration
                continue;
            }
            // query against these buckets
            for (group, buckets) in found.drain() {
                // build the future for this group and its set of buckets
                let future = D::pull(
                    group,
                    &self.retain.extra_filter,
                    self.year,
                    buckets,
                    self.retain.start,
                    self.retain.end,
                    limit,
                    shared,
                );
                // add the futures to our set
                futures.push(future);
            }
            // wait for all of our futures to complete 50 at a time
            let queries = stream::iter(futures.drain(..))
                .buffer_unordered(50)
                .collect::<Vec<Result<QueryResult, ExecutionError>>>()
                .await
                .into_iter()
                .collect::<Result<Vec<QueryResult>, ExecutionError>>()?;
            // cast and sort the rows we just retrieved
            D::sort(queries, &mut self.sorted, &mut self.mapped).await?;
            // if we found enough data for this page of our cursor then break out
            if self.mapped >= self.limit || self.exhausted_time() {
                break;
            }
        }
        Ok(())
    }

    /// Filter down ordered lists of buckets to their intersecting ranges and the oldest bucket to skip too
    ///
    /// # Arguments
    ///
    /// * `tags` - The tags to find bucket intersections for
    /// * `tags_required` - The number of tags required
    /// * `pre_filter` - The unfiltered buckets for our tags
    /// * `oldest_first` - The oldest first bucket across all our bucket ranges
    /// * `possible` - Whether an intersection is still possible
    #[instrument(name = "ScyllaCursor::filter_bucket_intersection", skip_all)]
    fn filter_bucket_intersection(
        &self,
        tags: &HashMap<String, Vec<String>>,
        tags_required: usize,
        pre_filter: Vec<Vec<i32>>,
        oldest_first: &mut i32,
        possible: &mut bool,
    ) -> Vec<i32> {
        // get the first and last bucket that we know contains data
        let mut first = 0;
        let mut last = i32::MAX;
        // build a map of our buckets by tag key/value pair
        let mut map: HashMap<(&String, &String), BTreeSet<i32>> =
            HashMap::with_capacity(tags_required);
        // convert our bucket list into a stream
        let mut bucket_stream = pre_filter.into_iter();
        // step over our bucket data in the same order we retrieved it
        // get each tag key we are going to be querying for
        for (key, values) in tags {
            // check for census info for each group
            for _ in &self.retain.group_by {
                // get all the buckets that contain data for each value
                for value in values {
                    // get this values buckets
                    let buckets = bucket_stream.next().unwrap();
                    // skip any empty bucket lists
                    if !buckets.is_empty() {
                        // get our first and last items in this bucket list
                        let local_first = *buckets.first().unwrap();
                        let local_last = *buckets.last().unwrap();
                        // update our first_last values
                        update_first_last(
                            &mut first,
                            &mut last,
                            oldest_first,
                            local_first,
                            local_last,
                        );
                        // get an entry to this key/value bucket set
                        let entry: &mut BTreeSet<i32> = map.entry((key, value)).or_default();
                        // add the buckets for this group
                        entry.extend(buckets);
                    }
                }
            }
        }
        // check if we didn't find data for every tag key/value pair
        if map.len() < tags_required {
            // set that it is no longer possible to have an intersection
            *possible = false;
            // just return an empty vec since its no longer possible to have an intersection
            return vec![];
        }
        // build a hashset of buckets in any order
        let mut in_range: HashMap<i32, i32> = HashMap::with_capacity(1000);
        // add all of our valid buckets
        for (_, buckets) in &map {
            // check each potential bucket in this bucket range
            for bucket in buckets {
                // check if this bucket is in our overlapping range
                if *bucket > first && *bucket < last {
                    // add this potential overlapping bucket or increment its count
                    let entry: &mut i32 = in_range.entry(*bucket).or_default();
                    // increment its count
                    *entry += 1;
                }
            }
        }
        // filter down to buckets in all of our bucket ranges
        let intersection = in_range
            .iter()
            .filter(|(_, count)| **count as usize == tags_required)
            .map(|(bucket, _)| *bucket)
            .collect::<Vec<i32>>();
        intersection
    }

    /// Finds the next chunk of buckets that contain data for some tags
    ///
    /// # Arguments
    ///
    /// * `tags_retain` - Tag data retained for tags cursors
    /// * `shared` - Shared Thorium objects
    #[rustfmt::skip]
    #[instrument(name = "ScyllaCursor::tags_find_buckets", skip_all, err(Debug))]
    async fn tags_find_buckets(
        &self,
        tags_retain: &TagsRetain,
        shared: &Shared,
    ) -> Result<Vec<i32>, ApiError> {
        // set our end bucket to be f64::MAX or our end bucket if we are the end year
        let end = if self.year == self.end_year {
            self.end_bucket
        } else {
            0
        };
        // track the oldest first bucket we have seen in this loop so far
        let mut oldest_first = self.bucket as i32;
        // track whether an intersection is possible this year
        let mut possible = true;
        // build all of the keys we are getting valid buckets for
        let mut stream_keys = Vec::with_capacity(10);
        // decide on which census stream function to call based on if we're case insensitive
        let census_stream_fn = if tags_retain.case_insensitive {
            tags::census_stream_case_insensitive
        } else {
            tags::census_stream
        };
        // get each tag key we are going to be querying for
        for (key, values) in &tags_retain.tags {
            // check for census info for each group
            for group in &self.retain.group_by {
                // get all the buckets that contain data for each value
                for value in values {
                    // build the key for this tags bucket stream
                    let stream_key = census_stream_fn(tags_retain.tag_type, group, key, value, self.year, shared);
                    // add this stream key to our list
                    stream_keys.push(stream_key);
                }
            }
        }
        // loop until we have exhausted this years buckets
        loop {
            // build a redis pipeline to get all of the valid buckets for these tags
            let mut pipe = redis::pipe();
            // do this for each stream key
            for stream_key in &stream_keys {
                // get the next 100 items
                pipe.cmd("zrange").arg(stream_key).arg(oldest_first).arg(end)
                        .arg("byscore").arg("limit").arg(0).arg(100).arg("rev");
            }
            // execute our queries
            let bucket_strings: Vec<Vec<String>> = pipe.query_async(conn!(shared)).await?;
            // preallocate a list for all of our different tag key buckets
            let mut pre_intersection = Vec::with_capacity(tags_retain.tags.len());
            // convert all of our buckets to signed ints
            for buckets_string in bucket_strings {
                // convert these buckets to an i32
                let buckets = buckets_string
                    .iter()
                    .map(|val| val.parse::<i32>())
                    .collect::<Result<Vec<i32>, _>>()?;
                // add this tag key/val combo
                pre_intersection.push(buckets);
            }
            // get the intersection of all buckets that all tags are in
            let intersection =
                self.filter_bucket_intersection(&tags_retain.tags, tags_retain.tags_required, pre_intersection, &mut oldest_first, &mut possible);
            // if we have intersecting buckets or an intersection is not possible for this year then return
            if !intersection.is_empty() || !possible {
                return Ok(intersection);
            }
        }
    }

    /// Check if this cursor has been exhausted
    pub fn exhausted(&mut self) -> bool {
        // check if we have any more mapped values and our buckets are exhausted
        self.buckets_exhausted && self.mapped == 0
    }

    /// Check if this cursor mapped the full bounds of its data
    fn exhausted_time(&self) -> bool {
        // check if our year/bucket are at or past our end year/bucket
        self.year <= self.end_year && self.bucket <= self.end_bucket
    }

    /// Consume our sorted data in order to fill our user facing data buffer
    fn consume_sorted(&mut self) -> bool {
        // keep looping until we have enough data to return or no more sorted data
        loop {
            // keep consuming until our user facing data buffer is full
            'outer: while self.data.len() < self.limit {
                // keep popping data from the end of our btree
                match self.sorted.pop_last() {
                    // we have a list of data to return
                    Some((timestamp, mut item)) => {
                        // if we have multiple rows with the same timestamp then disambiguate them
                        // only tags have to do this
                        if self.retain.tags_retain.is_some() && item.len() > 1 {
                            // sort our ambiguous rows by their cluster key
                            D::sort_by_cluster_key(&mut item);
                        }
                        // consume only enough data to fill our data
                        while self.data.len() < self.limit {
                            match item.pop_front() {
                                Some(item) => {
                                    // consume this mapped value
                                    self.mapped -= 1;
                                    // only dedupe if we are configured too
                                    if self.dedupe {
                                        // dedupe our data and then see if we need to get more
                                        if !item.dedupe_item(&mut self.dedupe_set) {
                                            // this is a duplicate item so skip it
                                            continue;
                                        }
                                    }
                                    // add this data to our user facing vec
                                    self.data.push(item);
                                }
                                None => continue 'outer,
                            }
                        }
                        // if we have another item then add it to our tie map
                        if let Some(tied_row) = item.front() {
                            // if this a tags cursor add this to our tag ties
                            if let Some(tags_retain) = &mut self.retain.tags_retain {
                                // add this tie to our tag tie map
                                tied_row.add_tag_tie(&mut tags_retain.ties);
                            } else {
                                // add this to our regular query tie map
                                tied_row.add_tie(&mut self.retain.ties);
                            }
                        }
                        // this item still has more data so readd it
                        self.sorted.insert(timestamp, item);
                        // we have all the data that we need
                        break;
                    }
                    // we have no more data so break
                    None => {
                        // update our start timestamp if we found some data
                        if let Some(last) = self.data.last() {
                            // update our start timestamp
                            self.retain.start = last.get_timestamp();
                        }
                        return false;
                    }
                }
            }
            // check if we have enough data to try and return yet
            if self.data.len() >= self.limit {
                // update our start timestamp if we found some data
                if let Some(last) = self.data.last() {
                    // update our start timestamp
                    self.retain.start = last.get_timestamp();
                }
                return true;
            }
        }
    }

    /// Gets the next page of queries for this cursor for data for general queries
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::next_general", skip_all, err(Debug))]
    async fn next_general(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // Get the limit + 1 of data for each group so we can check if we end on any ties
        let limit = (self.limit + 1) as i32;
        // get data from any previously tied queries
        let tied_queries = self.query_ties(limit, shared).await?;
        // if we had any queries based on ties then consume them
        if !tied_queries.is_empty() {
            // cast and sort the rows we just retrieved
            D::sort(tied_queries, &mut self.sorted, &mut self.mapped).await?;
            // consume our sorted data and check if we have enough data to return
            if self.consume_sorted() {
                // we have enough data so return
                return Ok(());
            }
        }
        // determine which partitions have data
        self.query(limit, shared).await?;
        // consume enough of our sorted data
        self.consume_sorted();
        Ok(())
    }

    #[instrument(name = "ScyllaCursor::tag_ties", skip_all, err(Debug))]
    async fn tag_ties(
        &self,
        tags_retain: &mut TagsRetain,
        mapping: &mut HashMap<String, TagMapping>,
        shared: &Shared,
        prepared: &PreparedStatement,
    ) -> Result<(), ApiError> {
        // calculate the number of futures we will be spawning
        let capacity = tags_retain.ties.len() * tags_retain.tags_required;
        // build a list of futures for our queries
        let mut futures = Vec::with_capacity(capacity);
        // query for each of our ties
        for (group, breaker) in &tags_retain.ties {
            // query for each tag key and all of its values
            for (key, values) in &tags_retain.tags {
                // query for each value for this tag key
                for value in values {
                    // execute the query to get this group/tag/key combos rows
                    let query = ties_tags_query_helper::<D>(
                        prepared,
                        tags_retain.tag_type,
                        group,
                        self.year,
                        self.bucket as i32,
                        key,
                        value,
                        self.retain.start,
                        breaker,
                        shared,
                    );
                    // add this future out our futures list
                    futures.push(query);
                }
            }
        }
        // execute our futures 50 at a time
        let queries = stream::iter(futures)
            .buffer_unordered(50)
            .collect::<Vec<Result<(&str, &str, QueryPager), PagerExecutionError>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<(&str, &str, QueryPager)>, PagerExecutionError>>()?;
        // build our tag mapping object
        parse_tag_queries(queries, mapping).await?;
        // clear our tag ties
        tags_retain.ties.clear();
        Ok(())
    }

    #[instrument(name = "ScyllaCursor::tag_query", skip_all, err(Debug))]
    async fn tag_query(
        &mut self,
        tags_retain: &TagsRetain,
        mapping: &mut HashMap<String, TagMapping>,
        shared: &Shared,
        prepared: &PreparedStatement,
    ) -> Result<(), ApiError> {
        // calculate the size of the futures were about to spawn
        let capacity = tags_retain.tags_required * self.retain.group_by.len();
        // have a vec of futures
        let mut futures = Vec::with_capacity(capacity);
        // loop until we have found enough data to return
        loop {
            // get the next 100 buckets that contain data
            let buckets = self.tags_find_buckets(tags_retain, shared).await?;
            // query for each tag key/value
            for (key, values) in &tags_retain.tags {
                // query for each value for this tag key
                for value in values {
                    // build this query for each group
                    for group in &self.retain.group_by {
                        // chunk our buckets into groups of 100
                        for bucket_chunk in buckets.chunks(100) {
                            // execute the query to get this group/tag/key combos rows
                            let query = tags_query_helper::<D>(
                                prepared,
                                tags_retain.tag_type,
                                group,
                                self.year,
                                bucket_chunk.to_vec(),
                                key,
                                value,
                                self.retain.start,
                                self.retain.end,
                                shared,
                            );
                            // add this future out our futures list
                            futures.push(query);
                        }
                    }
                }
            }
            // execute our futures 50 at a time
            let queries = stream::iter(futures.drain(..))
                .buffer_unordered(50)
                .collect::<Vec<Result<(&str, &str, QueryPager), PagerExecutionError>>>()
                .await
                .into_iter()
                .collect::<Result<Vec<(&str, &str, QueryPager)>, PagerExecutionError>>()?;
            // build our tag mapping object
            parse_tag_queries(queries, mapping).await?;
            // only retain mappings that matched on all of the required tags
            mapping.retain(|_, mapped| mapped.tags.len() == tags_retain.tags_required);
            // if we have enough data to return then return
            if mapping.len() >= self.limit {
                break;
            }
            // update our bucket counter correctly
            // this can't be moved to a function due to borrow check sadness
            match (self.year.cmp(&self.end_year), buckets.len() >= 100) {
                // we have more buckets this year too query so just update our bucket
                (Ordering::Greater, true) | (Ordering::Equal, true) => {
                    // TODO does this overlap?
                    self.bucket = (buckets.last().unwrap() - 1) as u32;
                }
                // we have more years to query so decrement our year and update our bucket
                (Ordering::Greater, false) => {
                    // decrement our year
                    self.year = self.year.saturating_sub(1);
                    // get a duration for the next year
                    let year = NaiveDate::from_ymd_opt(self.year, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 1)
                        .unwrap();
                    let next_year = NaiveDate::from_ymd_opt(self.year - 1, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 1)
                        .unwrap();
                    let duration = year - next_year;
                    // reset our bucket to the last possible partition of the year
                    self.bucket = duration.num_seconds() as u32 / self.partition_size as u32;
                }
                // we have reached our end bucket and end year
                (Ordering::Equal, false) | (Ordering::Less, false) => {
                    match buckets.last() {
                        Some(bucket) => self.bucket = *bucket as u32,
                        None => self.bucket = self.end_bucket,
                    }
                    // set this cursor to be exhausted
                    self.buckets_exhausted = true;
                    break;
                }
                // we have gone past our end year bounds somehow
                (Ordering::Less, true) => {
                    event!(
                        Level::ERROR,
                        msg = "Gone past end year bound?",
                        year = self.year,
                        end_year = self.end_year,
                        bucket = self.bucket,
                        end_bucket = self.bucket
                    );
                    // return an error to the user
                    return internal_err!("Cursor Failure".to_owned());
                }
            }
        }
        Ok(())
    }

    /// Gets the next page of queries for this cursor for data for general queries
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::next_tags", skip(self, shared), err(Debug))]
    async fn next_tags(
        &mut self,
        mut tags_retain: TagsRetain,
        shared: &Shared,
    ) -> Result<TagsRetain, ApiError> {
        // keep a mapping of the tags data we retrieved
        let mut mapping = HashMap::with_capacity(tags_retain.tags_required * self.limit);
        // get the right prepared statements depending on if this cursor is case insensitive
        let (ties_prepared, query_prepared) = if tags_retain.case_insensitive {
            (
                &shared.scylla.prep.tags.list_ties_case_insensitive,
                &shared.scylla.prep.tags.list_pull_case_insensitive,
            )
        } else {
            (
                &shared.scylla.prep.tags.list_ties,
                &shared.scylla.prep.tags.list_pull,
            )
        };
        // check if we have any current ties
        if !tags_retain.ties.is_empty() {
            // get any tie data
            self.tag_ties(&mut tags_retain, &mut mapping, shared, ties_prepared)
                .await?;
        }
        // get tag data using normal queries
        self.tag_query(&tags_retain, &mut mapping, shared, query_prepared)
            .await?;
        // add our mapped data to our sorted items
        D::sort_tags(
            &mut mapping,
            &tags_retain.tags,
            &mut self.sorted,
            &mut self.mapped,
        )?;
        // consume our sorted data and return if needed
        self.consume_sorted();
        // return the modified retained tags data
        Ok(tags_retain)
    }

    /// Gets the next page of data for this cursor
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::next", skip_all, err(Debug))]
    pub async fn next(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // crawl over the tags table if we have tag filters
        match self.retain.tags_retain.take() {
            // we have tags to filter on
            Some(tags_retain) => {
                let tags_retain = self.next_tags(tags_retain, shared).await?;
                self.retain.tags_retain = Some(tags_retain);
                Ok(())
            }
            // we don't have any tags to filter on
            None => self.next_general(shared).await,
        }
    }

    /// Saves this cursor to Redis
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // serialize our retained data
        let data = serialize!(&self.retain);
        // build the key to save this cursor data too
        let key = cursors::data(CursorKind::Scylla, &self.id, shared);
        // save this cursors data to redis
        let _: () = query!(
            cmd("set").arg(key).arg(data).arg("EX").arg(2_628_000),
            shared
        )
        .await?;
        Ok(())
    }
}

impl<D> From<ScyllaCursor<D>> for ApiCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug + std::marker::Send,
    D: Serialize,
    D: ScyllaCursorSupport,
    D: Debug,
{
    /// convert this scylla cursor to a user facing cursor
    fn from(mut scylla_cursor: ScyllaCursor<D>) -> Self {
        // if our cursor is exhausted then don't include a cursor id
        let id = if scylla_cursor.exhausted() {
            None
        } else {
            Some(scylla_cursor.id)
        };
        // build our cursor object
        ApiCursor {
            cursor: id,
            data: scylla_cursor.data,
        }
    }
}

/// The data retained for the lifetime of this scylla simple cursor
#[derive(Serialize, Deserialize, Debug)]
pub struct SimpleScyllaCursorRetain {
    /// The partitions we are crawling for data
    partitions: Vec<String>,
    /// Our current position in our partitions vec
    index: usize,
    /// The clustering key to start listing values at next time
    tie: Option<String>,
}

/// Abstracts simple cursor logic across different tables and data types
#[async_trait::async_trait]
pub trait SimpleCursorExt {
    /// Query scylla for the next page of data for this simple cursor
    ///
    /// # Arguments
    ///
    /// * `partition` - The partition to query data for
    /// * `tie` - The cluster key to use when breaking ties
    /// * `limit` - The max amount of data to retrieve at once
    /// * `shared` - Shared Thorium objects
    async fn query(
        partition: &str,
        tie: &Option<String>,
        limit: usize,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError>;

    /// Gets the cluster key to start the next page of data after
    fn get_tie(&self) -> Option<String>;
}

/// A simple non group permissioned cursor
#[derive(Debug)]
pub struct SimpleScyllaCursor<D>
where
    for<'de> D: Deserialize<'de>,
    D: Serialize,
    D: SimpleCursorExt,
{
    /// The ID for this cursor
    pub id: Uuid,
    /// The data that is retained throughout this cursors life
    pub retain: SimpleScyllaCursorRetain,
    /// The max number of items to return at once
    pub limit: usize,
    /// Whether this cursor is exhausted or not
    pub exhausted: bool,
    /// Whether this cursor has data in redis or not
    pub in_redis: bool,
    /// The data to return
    pub data: Vec<D>,
}

impl<D> SimpleScyllaCursor<D>
where
    for<'de> D: Deserialize<'de>,
    D: Serialize,
    D: SimpleCursorExt,
    D: for<'a, 'b> DeserializeRow<'a, 'b>,
{
    /// Create a new simple cursor object
    ///
    /// This will default to a limit of 50 if its not passed in
    ///
    /// # Arguments
    ///
    /// * `partitions` - The partitions to list data from
    #[must_use]
    pub fn new(partitions: Vec<String>, limit: usize) -> Self {
        // create an initial simple scylla cursor
        let retain = SimpleScyllaCursorRetain {
            partitions,
            index: 0,
            tie: None,
        };
        // build our simple cursor
        SimpleScyllaCursor {
            id: Uuid::new_v4(),
            retain,
            limit,
            exhausted: false,
            in_redis: false,
            data: Vec::with_capacity(limit),
        }
    }

    /// Set the max number of items this cursor should return at once
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of items that should be returned at once
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        // update the limit for this cursor
        self.limit = limit;
        self
    }

    /// Gets a simple cursors info from Redis
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The id of the cursor to retrieve
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SimpleScyllaCursor::get", skip(shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        limit: usize,
        shared: &Shared,
    ) -> Result<SimpleScyllaCursor<D>, ApiError> {
        // build the key to our cursor data in redis
        let key = cursors::data(CursorKind::SimpleScylla, &cursor_id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // deserialize our retained data
                let retain = deserialize!(&data);
                let cursor = SimpleScyllaCursor {
                    id: cursor_id,
                    retain,
                    limit,
                    exhausted: false,
                    in_redis: true,
                    data: Vec::with_capacity(limit),
                };
                Ok(cursor)
            }
            // we didn't find any cursor data
            None => not_found!(format!("Cursor {} doesn't exist", cursor_id)),
        }
    }

    /// Get the next page of this cursors data
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SimpleScyllaCursor::next", skip_all, err(Debug))]
    pub async fn next(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // if we have no partitions then exhaust this cursor and return empty
        if self.retain.partitions.is_empty() {
            self.exhausted = true;
            return Ok(());
        }
        // loop until we find enough data or exhaust our partitions
        loop {
            // get the number of items to try to get this loop
            let limit = self.limit - self.data.len();
            // query scylla for the data for this simple cursor
            let query = D::query(
                &self.retain.partitions[self.retain.index],
                &self.retain.tie,
                limit,
                shared,
            )
            .await?;
            // enable casting to types for this query
            let query_rows = query.into_rows_result()?;
            // get the number of rows in this typed stream
            let cnt = query_rows.rows_num();
            // set the type to cast this stream too
            let typed_iter = query_rows.rows::<D>()?;
            // check if we found any rows
            if cnt > 0 {
                // cast our rows into the correct objects and log any errors
                let found = typed_iter.filter_map(|res| log_scylla_err!(res));
                // add this data to the data to return
                self.data.extend(found);
                // if cnt is less than our limit then go to the next partition
                if cnt < limit {
                    // check if we are at the last index
                    if self.retain.index == self.retain.partitions.len() - 1 {
                        // we are at the last index so just return what we have
                        self.exhausted = true;
                        break;
                    }
                    // we have more data to get so continue on
                    self.retain.index += 1;
                    // reset our tie value
                    self.retain.tie = None;
                }
                // if we have all the data we need then return
                if self.data.len() == self.limit {
                    // set our tie value if we have any data
                    if let Some(last) = self.data.last() {
                        self.retain.tie = last.get_tie();
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    /// Saves this cursor to Redis
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SimpleScyllaCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // build the key to save this cursor data too
        let key = cursors::data(CursorKind::Scylla, &self.id, shared);
        // either save or delete this cursor based on whether its exhausted or not
        if self.exhausted {
            // only delete if this cursor was actually written to redis
            if self.in_redis {
                // delete this cursor from redis
                let _: () = query!(cmd("del").arg(key), shared).await?;
            }
        } else {
            // serialize our partitions
            let data = serialize!(&self.retain);
            // build the key to save this cursor data too
            let key = cursors::data(CursorKind::SimpleScylla, &self.id, shared);
            // save this cursors data to redis
            let _: () = query!(
                cmd("set").arg(key).arg(data).arg("EX").arg(2_628_000),
                shared
            )
            .await?;
        }
        Ok(())
    }
}

impl<D> From<SimpleScyllaCursor<D>> for ApiCursor<D>
where
    for<'de> D: Deserialize<'de>,
    D: Serialize,
    D: SimpleCursorExt,
{
    /// convert this simple scylla cursor to a user facing cursor
    fn from(scylla_cursor: SimpleScyllaCursor<D>) -> Self {
        // if our cursor is exhausted then don't include a cursor id
        let id = if scylla_cursor.exhausted {
            None
        } else {
            Some(scylla_cursor.id)
        };
        // build our cursor object
        ApiCursor {
            cursor: id,
            data: scylla_cursor.data,
        }
    }
}

/// The core features required for implementing a Scylla cursor crawling grouped rows
///
/// The fundamental difference compared to [`ScyllaCursorSupport`] is this cursor has
/// no concept of time or bucketing by time
pub trait GroupedScyllaCursorSupport: Sized {
    /// The params to build this cursor form
    type Params;

    /// Any extra info to filter with
    type ExtraFilters: Clone + Debug + Serialize + for<'a> Deserialize<'a>;

    /// The intermediary component type casted from a Scylla row that is used to build `Self`
    type RowType: Into<Self> + for<'frame, 'metadata> DeserializeRow<'frame, 'metadata>;

    /// The type of data our rows are grouped by (AKA the partition key)
    type GroupBy: Hash + Eq + Debug + Serialize + for<'a> Deserialize<'a>;

    /// The type `Self` is sorted by in Scylla (AKA the clustering key);
    /// the resulting list of `Self` will be returned ordered by this type
    type SortBy: Ord + Debug + Serialize + for<'a> Deserialize<'a>;

    /// The type used to break ties when the same instance of `SortBy` is found in multiple
    /// groups, but they are unique entities (i.e. the two entities have the same name, but
    /// none of the same groups)
    type SortTieBreaker: Eq;

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &mut Self::Params) -> Option<Uuid>;

    /// Get the groups to query from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_groups(params: &mut Self::Params) -> HashSet<Self::GroupBy>;

    /// Get our extra filters from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_extra_filters(params: &mut Self::Params) -> Self::ExtraFilters;

    /// Get our the max number of rows to return
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_limit(params: &Self::Params) -> Result<i32, ApiError>;

    /// Get the `GroupBy` value from a casted row
    ///
    /// # Arguments
    ///
    /// * `row` - The row to get the `GroupBy` value from
    fn row_get_group_by(row: &Self::RowType) -> Self::GroupBy;

    /// Get the `SortBy` value from a casted row
    ///
    /// # Arguments
    ///
    /// * `row` - The row to get the `SortBy` value from
    fn row_get_sort_by(row: &Self::RowType) -> Self::SortBy;

    /// Get the tie-breaking value from the casted row
    ///
    /// # Arguments
    ///
    /// * `row` - The row to get the `SortBy` value from
    fn row_get_tie_breaker(row: &Self::RowType) -> &Self::SortTieBreaker;

    /// Return the value we're sorting by from `self`
    fn get_sort_by(&self) -> Self::SortBy;

    /// Get the tie-breaking value from `self`
    fn get_tie_breaker(&self) -> &Self::SortTieBreaker;

    /// Convert `self` to a tie to re-retrieve later
    fn to_tie(self) -> (Self::SortBy, Vec<Self::GroupBy>);

    /// Add a row to `self`, probably by just adding the row's group
    ///
    /// # Arguments
    ///
    /// * `row` - The component row to add
    fn add_row(&mut self, row: Self::RowType);

    /// Builds the query string for getting data from ties in the last query
    ///
    /// Ties occur when two groups have the same `SortBy` value and we weren't
    /// able to return them all last iteration. These would be skipped if we
    /// proceeded to query from our last `SortBy` value, so we need to get them
    /// explicitly
    ///
    /// # Arguments
    ///
    /// * `ties` - The ties to get data for
    /// * `extra` - Any extra filters to apply to this query
    /// * `limit` - The max number of rows to return
    /// * `shared` - Shared Thorium objects
    fn ties_query(
        ties: &[(Self::SortBy, Vec<Self::GroupBy>)],
        extra: &Self::ExtraFilters,
        limit: i32,
        shared: &Shared,
    ) -> Vec<impl Future<Output = Result<QueryResult, ExecutionError>>>;

    /// Builds the query for getting the first page of values
    ///
    /// The Scylla query must have a `PER PARTITION LIMIT` equal to [`Self::limit`] as
    /// defined by [`Self::Params`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `extra` - Any extra filters to apply to this query
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    fn pull(
        group: &Self::GroupBy,
        extra: &Self::ExtraFilters,
        limit: i32,
        shared: &Shared,
    ) -> impl Future<Output = Result<QueryResult, ExecutionError>>;

    /// Builds the query for getting the next page of values
    ///
    /// The Scylla query must have a `PER PARTITION LIMIT` equal to [`Self::limit`] as
    /// defined by [`Self::Params`], as well as pull from everything greater than
    /// `current_sort_by` to work properly
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `extra` - Any extra filters to apply to this query
    /// * `current_sort_by` - The current sort value we left off at
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    fn pull_more(
        group: &Self::GroupBy,
        extra: &Self::ExtraFilters,
        current_sort_by: &Self::SortBy,
        limit: i32,
        shared: &Shared,
    ) -> impl Future<Output = Result<QueryResult, ExecutionError>>;
}

/// The data to retain throughout this cursors life
#[derive(Serialize, Deserialize, Debug)]
pub struct GroupedScyllaCursorRetain<D: GroupedScyllaCursorSupport> {
    /// The tag based filters to use with this cursor
    pub extra_filter: D::ExtraFilters,
    /// The values to group the rows from this cursor by
    pub group_by: HashSet<D::GroupBy>,
    /// Any ties in past iterations of this cursor that would be skipped
    /// if we continued from our current clustering key
    pub ties: Vec<(D::SortBy, Vec<D::GroupBy>)>,
    /// The current sort value we left off at last query
    pub current_sort_by: Option<D::SortBy>,
}

/// A cursor for a listing group permissioned data within scylla
#[derive(Debug)]
pub struct GroupedScyllaCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: GroupedScyllaCursorSupport,
    D: Debug,
{
    /// The ID for this cursor
    pub id: Uuid,
    /// The cursor settings/data to retain across cursor iterations
    pub retain: GroupedScyllaCursorRetain<D>,
    /// The max number of items to return at once
    pub limit: i32,
    /// Whether this cursor has data in redis or not
    pub in_redis: bool,
    /// The data this cursor has retrieved
    pub data: Vec<D>,
}

impl<D> GroupedScyllaCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: GroupedScyllaCursorSupport,
    D: Debug,
{
    /// Create a new cursor object from just params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to build this cursor from
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "GroupedScyllaCursor::from_params",
        skip(params, dedupe, shared),
        err(Debug)
    )]
    pub async fn from_params(
        mut params: D::Params,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<GroupedScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&mut params) {
            GroupedScyllaCursor::get(id, params, dedupe, shared).await
        } else {
            // get our extra filters
            let extra_filter = D::get_extra_filters(&mut params);
            // get our group restrictions
            let groups = D::get_groups(&mut params);
            // build the inital data to retain across cursor iterations
            let retain = GroupedScyllaCursorRetain {
                extra_filter,
                group_by: groups,
                ties: Vec::default(),
                current_sort_by: None,
            };
            // build our cursor
            let cursor = GroupedScyllaCursor {
                id: Uuid::new_v4(),
                retain,
                limit: D::get_limit(&params)?,
                in_redis: false,
                data: Vec::default(),
            };
            Ok(cursor)
        }
    }

    /// Create a new cursor object from params with a set extra fields
    ///
    /// # Arguments
    ///
    /// * `params` - The params to build this cursor from
    /// * `extra_filter` - The extra filters to use in this cursor
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "GroupedScyllaCursor::from_params_extra",
        skip(params, extra_filter, shared),
        err(Debug)
    )]
    pub async fn from_params_extra(
        mut params: D::Params,
        extra_filter: D::ExtraFilters,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<GroupedScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&mut params) {
            GroupedScyllaCursor::get(id, params, dedupe, shared).await
        } else {
            // get our group restrictions
            let groups = D::get_groups(&mut params);
            // build the inital data to retain across cursor iterations
            let retain = GroupedScyllaCursorRetain {
                extra_filter,
                group_by: groups,
                ties: Vec::default(),
                current_sort_by: None,
            };
            // build our cursor
            let cursor = GroupedScyllaCursor {
                id: Uuid::new_v4(),
                retain,
                limit: D::get_limit(&params)?,
                in_redis: false,
                data: Vec::default(),
            };
            Ok(cursor)
        }
    }

    /// Gets a cursors data from scylla
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The uuid of the cursor to retrieve if one is known
    /// * `params` - The params to build this cursor from
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared thorium objects
    #[instrument(name = "GroupedScyllaCursor::get", skip(params, shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        params: D::Params,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<GroupedScyllaCursor<D>, ApiError> {
        // build the key to our cursor data in redis
        let key = cursors::data(CursorKind::GroupedScylla, &cursor_id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // deserialize our retained data
                let retain: GroupedScyllaCursorRetain<D> = deserialize!(&data);
                // build and return our cursor
                let cursor = GroupedScyllaCursor {
                    id: cursor_id,
                    retain,
                    limit: D::get_limit(&params)?,
                    in_redis: true,
                    data: Vec::default(),
                };
                Ok(cursor)
            }
            // we didn't find any cursor data
            None => not_found!(format!(
                "cursor '{}' was not found; maybe it expired?",
                cursor_id
            )),
        }
    }

    /// Check if we have any ties queries
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "GroupedScyllaCursor::query_ties",
        skip(self, shared),
        err(Debug)
    )]
    async fn query_ties(&self, shared: &Shared) -> Result<Vec<QueryResult>, ApiError> {
        // build our tie query futures
        let futures = D::ties_query(
            &self.retain.ties,
            &self.retain.extra_filter,
            // get an extra row to check for more ties
            self.limit + 1,
            shared,
        );
        // wait for all of our futures to complete 50 at a time
        let queries = stream::iter(futures)
            .buffer_unordered(50)
            .collect::<Vec<Result<QueryResult, ExecutionError>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<QueryResult>, ExecutionError>>()?;
        Ok(queries)
    }

    /// Remove any groups from our list that didn't return any data
    ///
    /// # Arguments
    ///
    /// * `rows` - The casted rows we retrieved last query
    fn prune_exhausted_groups(&mut self, rows: &[D::RowType]) {
        // create a map of group to the number of rows it returned
        let mut non_empty_groups: HashSet<D::GroupBy> = HashSet::new();
        for row in rows {
            non_empty_groups.insert(D::row_get_group_by(row));
        }
        // retain only groups that are not empty
        self.retain
            .group_by
            .retain(|group| non_empty_groups.contains(group));
    }

    /// Build a sorted map of the sort value to a list of unique entities with that
    /// sort value, where each entity is composed of one or more rows
    ///
    /// Entities' uniqueness is determined using their [`GroupedScyllaCursorSupport::SortTieBreaker`]
    ///
    /// # Arguments
    ///
    /// * rows - The rows to build the entities from
    fn build_sorted_map(rows: Vec<D::RowType>) -> BTreeMap<D::SortBy, VecDeque<D>> {
        let mut map: BTreeMap<D::SortBy, VecDeque<D>> = BTreeMap::new();
        for row in rows {
            // get the list of entities by their sort name
            let entity_list = map.entry(D::row_get_sort_by(&row)).or_default();
            // see if we've seen this entity before using our tie breakers
            if let Some(entity) = entity_list
                .iter_mut()
                .find(|n| n.get_tie_breaker() == D::row_get_tie_breaker(&row))
            {
                // if we've already seen this policy, add the row to it
                entity.add_row(row);
            } else {
                // otherwise, cast the row to the entity and add it to our list
                entity_list.push_back(row.into());
            }
        }
        map
    }

    /// Cast the query results to their row type
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to cast
    #[instrument(name = "GroupedScyllaCursor::cast_rows", skip_all, err(Debug))]
    fn cast_rows(queries: Vec<QueryResult>) -> Result<Vec<D::RowType>, ApiError> {
        let mut casts = Vec::new();
        // cast each row into the RowType
        for query in queries {
            // convert this into a rows query
            let query_rows = query.into_rows_result()?;
            // get each row returned by this query
            for row in query_rows.rows::<D::RowType>()? {
                // raise any errors from casting
                casts.push(row?);
            }
        }
        Ok(casts)
    }

    /// Crawl partitions and pull more data from them starting from the first item
    /// after `current_sort_by`
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "GroupedScyllaCursor::query_more",
        skip(self, shared),
        err(Debug)
    )]
    async fn query_more(
        &self,
        current_sort_by: &D::SortBy,
        shared: &Shared,
    ) -> Result<Vec<QueryResult>, ApiError> {
        // create our futures
        let mut futures = Vec::new();
        for group in &self.retain.group_by {
            futures.push(D::pull_more(
                group,
                &self.retain.extra_filter,
                current_sort_by,
                // get an extra row to check for exhausted rows and ties
                self.limit + 1,
                shared,
            ));
        }
        // query for each group 50 at a time
        let query_results = stream::iter(futures)
            .buffer_unordered(50)
            .collect::<Vec<Result<QueryResult, ExecutionError>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<QueryResult>, ExecutionError>>()?;
        // return our results
        Ok(query_results)
    }

    /// Crawl partitions and pull data from them
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "GroupedScyllaCursor::query", skip(self, shared), err(Debug))]
    async fn query(&self, shared: &Shared) -> Result<Vec<QueryResult>, ApiError> {
        // create our futures
        let mut futures = Vec::new();
        for group in &self.retain.group_by {
            futures.push(D::pull(
                group,
                &self.retain.extra_filter,
                // get an extra row to check for exhausted rows and ties
                self.limit + 1,
                shared,
            ));
        }
        // query for each group 50 at a time
        let query_results = stream::iter(futures)
            .buffer_unordered(50)
            .collect::<Vec<Result<QueryResult, ExecutionError>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<QueryResult>, ExecutionError>>()?;
        // return our results
        Ok(query_results)
    }

    /// Consume our sorted data map in order to fill our user facing data buffer
    ///
    /// # Arguments
    ///
    /// * `sorted_map` - The data sorted by the `SortBy` value mapped to a list of
    ///   unique entities containing that `SortBy` value
    ///
    /// # Panics
    ///
    /// Panics if the 32-bit signed integer `limit` is larger than the underlying
    /// system's addressing size, which should never happen for 32 and 64-bit systems
    fn consume_sorted(&mut self, mut sorted_map: BTreeMap<D::SortBy, VecDeque<D>>) {
        let limit: usize = self.limit.try_into().unwrap();
        // keep consuming until our user facing data buffer is full or we run out of data
        // TODO: use let chains when we upgrade to Rust 2024
        while self.data.len() < limit {
            if let Some((_sort_by, mut list)) = sorted_map.pop_first() {
                // keep consuming from each list until our user facing data buffer is full or we run out of data
                while self.data.len() < limit {
                    if let Some(item) = list.pop_front() {
                        // add this data to our user facing vec
                        self.data.push(item);
                    } else {
                        break;
                    }
                }
                if !list.is_empty() {
                    // if we still have data left, we must have had ties;
                    // save those ties to our cursor to get next iteration
                    self.retain.ties = list.into_iter().map(D::to_tie).collect();
                }
            } else {
                break;
            }
        }
    }

    /// Gets the next page of queries for this cursor for data for general queries
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    ///
    /// # Panics
    ///
    /// Panics if the 32-bit signed integer `limit` is larger than the underlying
    /// system's addressing size, which should never happen for 32 and 64-bit systems
    #[instrument(name = "GroupedScyllaCursor::next", skip_all, err(Debug))]
    pub async fn next(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // get data from any previously tied queries
        let tied_query_results = self.query_ties(shared).await?;
        // if we had any queries based on ties then consume them
        if !tied_query_results.is_empty() {
            // cast the rows we just retrieved
            let rows = Self::cast_rows(tied_query_results)?;
            let sorted_map = Self::build_sorted_map(rows);
            // consume the sorted data and add it to our user-facing data buffer
            self.consume_sorted(sorted_map);
            // see if our ties returned all we needed and return early if so
            if self.data.len() >= self.limit as usize {
                return Ok(());
            }
        }
        // retrieve data
        let query_results = if let Some(current_sort_by) = &self.retain.current_sort_by {
            // if this is a subsequent search, continue from where we left off
            self.query_more(current_sort_by, shared).await?
        } else {
            // start from the beginning if this is a new cursor
            self.query(shared).await?
        };
        // cast the rows we just retrieved to the component row type
        let rows = Self::cast_rows(query_results)?;
        // prune any groups that returned no rows
        self.prune_exhausted_groups(&rows);
        // build a sorted map of values
        let sorted_map = Self::build_sorted_map(rows);
        // consume the sorted data and add it to our user-facing data buffer
        self.consume_sorted(sorted_map);
        // save the current sort by we left off at
        self.retain.current_sort_by = self.data.last().map(D::get_sort_by);
        Ok(())
    }

    /// Saves this cursor to Scylla
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "GroupedScyllaCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // build the key to save this cursor data too
        let key = cursors::data(CursorKind::GroupedScylla, &self.id, shared);
        // either save or delete this cursor based on whether its exhausted or not
        if self.exhausted() {
            // only delete if this cursor was actually written to redis
            if self.in_redis {
                // delete this cursor from redis
                let _: () = query!(cmd("del").arg(key), shared).await?;
            }
        } else {
            // serialize our partitions
            let data = serialize!(&self.retain);
            // build the key to save this cursor data too
            let key = cursors::data(CursorKind::GroupedScylla, &self.id, shared);
            // save this cursors data to redis
            let _: () = query!(
                cmd("set").arg(key).arg(data).arg("EX").arg(2_628_000),
                shared
            )
            .await?;
        }
        Ok(())
    }

    /// Check whether this cursor is exhausted
    fn exhausted(&self) -> bool {
        // if none of our groups return data anymore, the cursor is exhausted
        self.retain.group_by.is_empty()
    }
}

impl<D> From<GroupedScyllaCursor<D>> for ApiCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: GroupedScyllaCursorSupport,
    D: Debug,
{
    /// convert this scylla cursor to a user facing cursor
    fn from(scylla_cursor: GroupedScyllaCursor<D>) -> Self {
        // if our cursor is exhausted then don't include a cursor id
        let id = if scylla_cursor.exhausted() {
            None
        } else {
            Some(scylla_cursor.id)
        };
        // build our cursor object
        ApiCursor {
            cursor: id,
            data: scylla_cursor.data,
        }
    }
}

/// A cursor that determines if some data exists or not
pub struct ExistsCursor {
    /// The timestamp to start getting more data for this cursor at
    pub start: DateTime<Utc>,
    /// The year we are last got data from
    pub year: i32,
    /// The bucket chunk we last got data from
    pub bucket: u32,
    /// The final date that our cursor should get data from
    pub end: DateTime<Utc>,
    /// The final year we are going to get data from
    pub end_year: i32,
    /// The final bucket we will get data from
    pub end_bucket: u32,
    /// The size of the partitions in seconds
    pub partition_size: u16,
}

impl ExistsCursor {
    /// Create a new [`ExistsCursor`]
    ///
    /// # Arguments
    ///
    /// * `start` - The timestamp to start checking for data at
    /// * `end` - The timestamp to stop checking for data at
    /// * `partition_size` - The partition size to use
    pub fn new(
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        partition_size: u16,
    ) -> Result<Self, ApiError> {
        // get the years this cursor should start and end in
        let year = start.year();
        let end_year = end.year();
        // build our exists cursor
        let cursor = ExistsCursor {
            start,
            year,
            bucket: u32::try_from(helpers::partition(start, year, partition_size))?,
            end,
            end_year,
            end_bucket: u32::try_from(helpers::partition(start, year, partition_size))?,
            partition_size,
        };
        Ok(cursor)
    }

    /// Check if any rows exist across all partitions
    ///
    /// # Arguments
    ///
    /// * `prepared` - The statement to execute
    /// * `key` - The key to use when checking for the existence of rows
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(
        name = "ExistsCursor::exists",
        skip(self, prepared, shared),
        err(Debug)
    )]
    pub async fn exists(
        mut self,
        prepared: &PreparedStatement,
        key: &str,
        shared: &Shared,
    ) -> Result<bool, ApiError> {
        // create a weeks worth of futures polling
        let mut futures = Vec::with_capacity(100);
        loop {
            // check if we are in the final year to list and stop at the correct bucket
            let end = if self.year == self.end_year {
                std::cmp::max(self.bucket.saturating_sub(99), self.end_bucket)
            } else {
                self.bucket.saturating_sub(99)
            };
            let buckets = (end..=self.bucket)
                .map(|bucket| bucket as i32)
                .collect::<Vec<i32>>();
            // build the query to see if data exists in these partitions
            let query = shared
                .scylla
                .session
                .execute_unpaged(prepared, (self.year, buckets, key));
            // add this future out our futures list
            futures.push(query);
            // if we have more than 100 futures to crawl then send them all at once
            if futures.len() >= 100 {
                // build our stream of futures
                let mut stream = stream::iter(futures.drain(..)).buffer_unordered(50);
                // check all partitions for data
                while let Some(query) = stream.next().await {
                    // unwrap our query
                    let query = query?;
                    // convert our query into a rows query
                    let query_rows = query.into_rows_result()?;
                    // check if any rows were returned
                    if query_rows.rows_num() > 0 {
                        // log that we found data
                        event!(Level::INFO, exists = true);
                        return Ok(true);
                    }
                }
            }
            // update our bucket counter correctly
            match (
                self.year.cmp(&self.end_year),
                end == 0,
                end <= self.end_bucket,
            ) {
                // we have more buckets this year to query so just update our bucket
                (Ordering::Greater, false, _) | (Ordering::Equal, _, false) => {
                    self.bucket = end - 1;
                }
                // we have more years to query to decrement our year and update our bucket
                (Ordering::Greater, true, _) => {
                    // decrement our year
                    self.year = self.year.saturating_sub(1);
                    // get a duration for the next year
                    let year = NaiveDate::from_ymd_opt(self.year, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 1)
                        .unwrap();
                    let next_year = NaiveDate::from_ymd_opt(self.year - 1, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 1)
                        .unwrap();
                    let duration = year - next_year;
                    // reset our bucket to the last possible partition of the year
                    self.bucket = duration.num_seconds() as u32 / self.partition_size as u32;
                }
                // were done querying so update our bucket and break
                (_, _, _) => {
                    self.bucket = end.saturating_sub(1);
                    break;
                }
            }
        }
        // if we hae any remaining futures then execute them
        if !futures.is_empty() {
            // build our stream of futures
            let mut stream = stream::iter(futures.drain(..)).buffer_unordered(50);
            // check all partitions for data
            while let Some(query) = stream.next().await {
                // unwrap our query
                let query = query?;
                // convert our query into a rows query
                let query_rows = query.into_rows_result()?;
                // check if any rows were returned
                if query_rows.rows_num() > 0 {
                    // log that we found data
                    event!(Level::INFO, exists = true);
                    return Ok(true);
                }
            }
        }
        // log that we did not find data
        event!(Level::INFO, exists = false);
        Ok(false)
    }
}

/// The data to retain throughout this cursors life
#[derive(Serialize, Deserialize, Debug)]
pub struct ElasticCursorRetain {
    /// The timestamp to start listing from for this cursor
    pub start: DateTime<Utc>,
    /// The timestamp to stop listing at for this cursor
    pub end: DateTime<Utc>,
    /// The serialized groups this cursor is searching in
    pub groups: Vec<String>,
    /// The query to send
    pub query: String,
    /// The point in time info for this cursor
    pub pit: String,
    /// The search after values for this cursor
    pub search_after: Vec<i64>,
}

/// A cursor for data in elastic
#[derive(Debug)]
pub struct ElasticCursor {
    /// The id for this cursor
    pub id: Uuid,
    /// The info to retain throughout this cursors lifetime
    pub retain: ElasticCursorRetain,
    /// The max number of items to return at once
    pub limit: i64,
    /// The data this cursor has retrieved
    pub data: Vec<ElasticDoc>,
}

impl ElasticCursor {
    /// Create or get an elastic cursor based on search params
    ///
    /// # Arguments
    ///
    /// * `params` - The elastic search params to use
    /// * `shared` - Shared Thorium objects
    pub async fn from_params(
        mut params: ElasticSearchParams,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // if a cursor was specified then get an existing cursor
        if let Some(id) = params.cursor.take() {
            // get an existing cursor
            ElasticCursor::get(id, params, shared).await
        } else {
            // we don't have an existing cursor so make a new one
            ElasticCursor::new(params, shared).await
        }
    }

    /// Create a new elastic cursor
    ///
    /// # Arguments
    ///
    /// * `params` - The elastic search params to use
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ElasticCursor::new", skip(shared), err(Debug))]
    async fn new(params: ElasticSearchParams, shared: &Shared) -> Result<ElasticCursor, ApiError> {
        // get the names of our indexes
        let indexes = params
            .indexes
            .iter()
            .map(|index| index.full_name(&shared.config.elastic))
            .collect::<Vec<&str>>();
        // get a new point in time id for this search
        let pit = elastic::PointInTime::new(&indexes, shared).await?.id;
        // build an initial cursor retained data struct
        let retain = ElasticCursorRetain {
            start: params.start,
            end: params.end(shared)?,
            groups: params.groups,
            query: params.query,
            pit,
            search_after: Vec::default(),
        };
        // build a new elastic cursor
        let cursor = ElasticCursor {
            id: Uuid::new_v4(),
            retain,
            limit: i64::from(params.limit),
            data: Vec::default(),
        };
        Ok(cursor)
    }

    /// Gets a cursors data for elastic
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the cursor to get
    /// * `params` - The elastic search params to use
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ElasticCursor::get", skip(shared), err(Debug))]
    pub async fn get(
        id: Uuid,
        params: ElasticSearchParams,
        shared: &Shared,
    ) -> Result<ElasticCursor, ApiError> {
        // build the key to our cursor data in redis
        let key = cursors::data(CursorKind::Elastic, &id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // try to deserialize our cursor data
                let retain = deserialize!(&data);
                // build our cursor
                let cursor = ElasticCursor {
                    id,
                    retain,
                    limit: params.limit.into(),
                    data: Vec::default(),
                };
                Ok(cursor)
            }
            None => not_found!(format!("cursor {} was not found, perhaps it expired?", id)),
        }
    }

    //// Get the first page of data from elastic
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ElasticCursor::next", skip_all, fields(query = self.retain.query), err(Debug))]
    pub async fn next(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // build the group filters
        let group_filters = self
            .retain
            .groups
            .iter()
            .map(|group| {
                serde_json::json!({
                    "match": {
                        "group": {
                            "query": group,
                            "minimum_should_match": "100%",
                            "fuzziness": "0",
                            "fuzzy_transpositions": false,
                            "auto_generate_synonyms_phrase_query": false
                        }
                    }
                })
            })
            .collect::<serde_json::Value>();
        // build the correct body depending on if we have any previous sort info or not
        let mut body = serde_json::json!({
            "pit": { "id": self.retain.pit, "keep_alive": "1d" },
            "query": {
                "bool": {
                    "must": {
                        "query_string": {
                            "query": &self.retain.query,
                        }
                    },
                    "filter": [
                        {
                            "bool": {
                                "should": group_filters,
                            }
                        },
                        {
                            "range": {
                                "streamed": {
                                    "gte": self.retain.end,
                                    "lt": self.retain.start,
                                }
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "pre_tags": vec!["@kibana-highlighted-field@"],
                "post_tags": vec!["@/kibana-highlighted-field@"],
                "fields": { "*": {}},
                // set the max analyzed offset to elastic's configured maximum to
                // avoid errors when fields are larger; highlights are truncated instead
                "max_analyzed_offset": shared.config.elastic.max_analyzed_offset,
            }
        });
        // add search after info if it's in our query
        if !self.retain.search_after.is_empty() {
            // we know body is an object, so we're okay to unwrap here
            body.as_object_mut()
                .ok_or(internal_err_unwrapped!(
                    "Elasticsearch query body is not a valid JSON object!".to_string()
                ))?
                .insert("search_after".to_string(), json!(self.retain.search_after));
        }
        // get the next page of docs from elastic
        let resp = shared
            .elastic
            // we don't need to specify an index when using point in time;
            // the point in time contains which index(es) we're searching on
            .search(SearchParts::None)
            .stored_fields(&["*"])
            .size(self.limit)
            .sort(&["streamed:desc", "_shard_doc:desc"])
            .body(body)
            .send()
            .await?;
        // deserialize the response if no error occurred
        if resp.status_code().is_success() {
            // get the response
            let cast = resp.json::<ElasticResponse>().await?;
            // pull out just the hits
            self.data = cast.hits.hits;
        } else {
            // deserialize our error
            let error = resp.json::<elasticsearch::http::response::Error>().await?;
            // return our error
            return Err(ApiError::from(error));
        }
        // update our cursors retained info based on the last item returned
        if let Some(last) = self.data.last() {
            // set the new search after value
            self.retain.search_after.clone_from(&last.sort);
        }
        Ok(())
    }

    /// Saves an elastic cursor to Scylla
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ElasticCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // serialize our retained info
        let data = serialize!(&self.retain);
        // build the key to save this cursor data too
        let key = cursors::data(CursorKind::Elastic, &self.id, shared);
        // save this cursors data to redis
        let _: () = query!(
            cmd("set").arg(key).arg(data).arg("EX").arg(2_628_000),
            shared
        )
        .await?;
        Ok(())
    }
}
