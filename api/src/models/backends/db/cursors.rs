//! Cursors for data in Thorium
use bb8_redis::redis::cmd;
use chrono::prelude::*;
use elasticsearch::SearchParts;
use futures::stream::{self, StreamExt, TryStreamExt};
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
use tracing::{Level, event, instrument};
use uuid::Uuid;

use super::elastic::{self, ElasticResponse};
use super::keys::{cursors, tags};
use crate::models::{ApiCursor, ElasticDoc, TagCounts, TagKeyCounts, TagListRow, TagMap, User};
use crate::models::{ElasticSearchParams, TagType};
use crate::utils::{ApiError, Shared, helpers};
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
    /// A tags counting cursor
    TagsCount,
    /// A cursor based on data in Elastic
    Elastic,
    /// A Tree of data in Thorium
    Tree,
    /// A cursor based on data in redis
    Redis,
}

impl CursorKind {
    /// Get our cursor kind as a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            CursorKind::Scylla => "Scylla",
            CursorKind::SimpleScylla => "SimpleScylla",
            CursorKind::GroupedScylla => "GroupedScylla",
            CursorKind::TagsCount => "TagsCount",
            CursorKind::Elastic => "Elastic",
            CursorKind::Tree => "Tree",
            CursorKind::Redis => "Redis",
        }
    }
}

/// Get the last partition bucket of a year
///
/// This is the bucket of the final second of the year, so it accounts for the length of this
/// year (leap years included) rather than the year before it.
///
/// # Arguments
///
/// * `year` - The year to get the last bucket for
/// * `partition_size` - The size of each partition in seconds
fn last_year_bucket(year: i32, partition_size: u16) -> u32 {
    // get the final second of this year, which always exists for the years we crawl
    let last_second = NaiveDate::from_ymd_opt(year, 12, 31)
        .and_then(|date| date.and_hms_opt(23, 59, 59))
        .map(|time| time.and_utc())
        .unwrap_or_default();
    // get the bucket this last second is in
    u32::try_from(helpers::partition(last_second, year, partition_size)).unwrap_or_default()
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
    /// The values already returned by this cursor when deduping, retained so that duplicates
    /// on later pages are also deduped
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub dedupe_set: HashSet<String>,
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
    fn get_id(params: &Self::Params) -> Option<Uuid>;

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
            let typed_stream = query_rows.rows::<Self::IntermediateRow>()?;
            // cast our rows to typed values
            for row in typed_stream {
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
                // drop our reference to our interemdiate unique value
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
                        *mapped += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Sort any possibly ambigous rows by their cluster key
    fn sort_by_cluster_key(ambigous: &mut VecDeque<Self>) {
        // create a btreemap to sort this data
        let mut sorted = BTreeMap::default();
        // keep a list of our ambigous rows
        let mut to_sort = HashMap::with_capacity(ambigous.len());
        // go through and get our clustering keys and insert them into our map
        for (index, row) in ambigous.drain(..).enumerate() {
            // get this rows clustering key
            let ckey = row.get_tag_clustering_key().clone();
            // insert this into our map
            sorted.insert(ckey, index);
            // add this row to our sort list
            to_sort.insert(index, row);
        }
        // rebuild this ambiguous row list in order
        for index in sorted.values() {
            // get this row and insert it
            if let Some(row) = to_sort.remove(index) {
                // add this row in the correct order
                ambigous.push_back(row);
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

/// A tie query for data based on tags
#[expect(clippy::too_many_arguments)]
async fn ties_tags_query_helper<'a>(
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
    skip_all,
    fields(kind, key, value, buckets_len = buckets.len()),
    err(Debug)
)]
#[expect(clippy::too_many_arguments)]
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
            (kind, group, year, buckets, key, value, start, end),
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
    for (key, value, query) in queries.into_iter() {
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

/// A cursor for a listing group permisisoned data within scylla
#[derive(Debug)]
pub struct ScyllaCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: ScyllaCursorSupport,
    D: Debug,
{
    /// The Id for this cursor
    pub id: Uuid,
    /// The username of the user this cursor is scoped to
    pub username: String,
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
    /// The data this cursor has retrieved
    pub data: Vec<D>,
    /// The internal sorted data to return to the user
    pub sorted: BTreeMap<DateTime<Utc>, VecDeque<D>>,
    /// The currently available number of rows we have sorted
    pub mapped: usize,
    /// whether this cursor has been exhausted or not
    pub buckets_exhausted: bool,
    /// Whether this is a new cursor or it already existed and is being continued
    pub new: bool,
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
    /// * `username` - The username of the user this cursor is scoped to
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "ScyllaCursor::from_params",
        skip(params, dedupe, shared),
        err(Debug)
    )]
    pub async fn from_params(
        mut params: D::Params,
        username: &str,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&params) {
            ScyllaCursor::get(id, username, params, dedupe, shared).await
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
            // get the retain info for tags if this a tag based cursor
            let tags_retain = Self::get_tags_retain(&mut params);
            // get our tag filters
            let tags = D::get_tag_filters(&mut params);
            // get our partition size
            // tag based cursors have a different chunk size
            let chunk = if let Some((tag_type, _)) = &tags {
                // this is a tag based cursor so use our tag partition size
                shared.config.thorium.tags.map_type(tag_type).partition_size
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
                dedupe_set: HashSet::default(),
            };
            // build our cursor
            let cursor = ScyllaCursor {
                id: Uuid::new_v4(),
                username: username.to_owned(),
                retain,
                year,
                bucket,
                end_year,
                end_bucket,
                partition_size: chunk,
                limit: D::get_limit(&params),
                dedupe,
                data: Vec::default(),
                sorted: BTreeMap::default(),
                mapped: 0,
                buckets_exhausted: false,
                new: true,
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
    /// * `username` - The username of the user this cursor is scoped to
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
        username: &str,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&params) {
            ScyllaCursor::get(id, username, params, dedupe, shared).await
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
            // get the retain info for tags if this a tag based cursor
            let tags_retain = Self::get_tags_retain(&mut params);
            // get our tag filters
            let tags = D::get_tag_filters(&mut params);
            // get our partition size
            // tag based cursors have a different chunk size
            let chunk = if let Some((tags, _)) = &tags {
                // this is a tag based cursor so use our tag partition size
                shared.config.thorium.tags.map_type(tags).partition_size
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
                dedupe_set: HashSet::default(),
            };
            // build our cursor
            let cursor = ScyllaCursor {
                id: Uuid::new_v4(),
                username: username.to_owned(),
                retain,
                year,
                bucket,
                end_year,
                end_bucket,
                partition_size: chunk,
                limit: D::get_limit(&params),
                dedupe,
                data: Vec::default(),
                sorted: BTreeMap::default(),
                mapped: 0,
                buckets_exhausted: false,
                new: true,
            };
            Ok(cursor)
        }
    }

    /// Rebuild a cursor from the data it retained in redis
    ///
    /// The set of items already returned for deduping is part of the retained data, so an item
    /// whose duplicates fall on different pages is only returned once.
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The uuid of the cursor being rebuilt
    /// * `username` - The username of the user this cursor is scoped to
    /// * `retain` - The data this cursor retained in redis
    /// * `chunk` - The partition size to use
    /// * `limit` - The max number of items to return at once
    /// * `dedupe` - Whether to dedupe items or not
    fn rebuild(
        cursor_id: Uuid,
        username: &str,
        retain: ScyllaCursorRetain<D>,
        chunk: u16,
        limit: usize,
        dedupe: bool,
    ) -> Result<Self, ApiError> {
        // determin our current year and our end year
        let year = retain.start.year();
        let end_year = retain.end.year();
        // get our starting/ending bucket
        let bucket = u32::try_from(helpers::partition(retain.start, year, chunk))?;
        let end_bucket = u32::try_from(helpers::partition(retain.end, end_year, chunk))?;
        // rebuild our cursor scoped to the creating user
        let cursor = ScyllaCursor {
            id: cursor_id,
            username: username.to_owned(),
            retain,
            year,
            bucket,
            end_year,
            end_bucket,
            partition_size: chunk,
            limit,
            dedupe,
            data: Vec::default(),
            sorted: BTreeMap::default(),
            mapped: 0,
            buckets_exhausted: false,
            new: false,
        };
        Ok(cursor)
    }

    /// Gets a cursors data from Redis
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The uuid of the cursor to retrieve if one is known
    /// * `username` - The username of the user this cursor is scoped to
    /// * `params` - The params to build this cursor from
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared thorium objects
    #[instrument(name = "ScyllaCursor::get", skip(params, shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        username: &str,
        params: D::Params,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ScyllaCursor<D>, ApiError> {
        // build the key to our cursor data in redis scoped to the creating user
        let key = cursors::data(CursorKind::Scylla, username, &cursor_id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // try to deseruialize this cursors retained data
                let retain: ScyllaCursorRetain<D> = deserialize!(&data);
                // tag based cursors have a different chunk size
                let chunk = if let Some(tag_retain) = &retain.tags_retain {
                    // get the type of tag this cursor is for
                    let tag_type = &tag_retain.tag_type;
                    // get the right chunk size based on the type of tags this is
                    shared.config.thorium.tags.map_type(tag_type).partition_size
                } else {
                    // this is not a tag based query so use our types partitions size
                    D::partition_size(shared)
                };

                // rebuild our cursor scoped to the creating user
                Self::rebuild(
                    cursor_id,
                    username,
                    retain,
                    chunk,
                    D::get_limit(&params),
                    dedupe,
                )
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
    ///
    /// # Arguments
    ///
    /// * `bucket_limit` - The most buckets to return for each group
    /// * `stream_keys` - the stream keys to look for data in
    /// * `found` - The buckets that we found that contain data
    /// * `shared` - Shared Thorium objects
    #[rustfmt::skip]
    async fn find_buckets<'a>(
        &self,
        bucket_limit: u32,
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
                    .arg("byscore").arg("limit").arg(0).arg(bucket_limit).arg("rev");
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
                // decrement our oldest first by 1 so we don't loop over buckets
                let decr_oldest_first = *oldest_first - 1;
                // add our new oldest first
                stream_keys.push((group, stream_key, decr_oldest_first));
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
    async fn query(
        &mut self,
        limit: i32,
        bucket_limit: u32,
        shared: &Shared,
    ) -> Result<(), ApiError> {
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
            self.find_buckets(bucket_limit, &mut stream_keys, &mut found, shared)
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
                // reset our bucket to the last possible partition of the year
                self.bucket = last_year_bucket(self.year, self.partition_size);
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
    /// * `next_bucket` - The next bucket to start listing from next time
    /// * `possible` - Whether an intersection is still possible
    #[instrument(name = "ScyllaCursor::filter_bucket_intersection", skip_all)]
    fn filter_bucket_intersection(
        &self,
        group_by: &Vec<D::GroupBy>,
        tags: &HashMap<String, Vec<String>>,
        tags_required: usize,
        pre_filter: Vec<Vec<i32>>,
        next_bucket: &mut i32,
        possible: &mut bool,
    ) -> Vec<i32> {
        // get the first and last bucket that we know contains data
        let mut first = 0;
        let mut last = 0;
        // build a map of our buckets by tag key/value pair
        let mut map: HashMap<(&String, &String), BTreeSet<i32>> =
            HashMap::with_capacity(tags_required);
        // convert our bucket list into a stream
        let mut bucket_stream = pre_filter.into_iter();
        // step over our bucket data in the same order we retrieved it
        // get each tag key we are going to be querying for
        for (key, values) in tags {
            // get all the buckets that contain data for each value
            for value in values {
                // check for census info for each group
                for _ in group_by {
                    // get this values buckets
                    let buckets = bucket_stream.next().unwrap();
                    // skip any empty bucket lists
                    if !buckets.is_empty() {
                        // get our first and last items in this bucket list
                        let local_first = *buckets.first().unwrap();
                        let local_last = *buckets.last().unwrap();
                        // check if this bucket range moves the first bucket with data up
                        if local_first > first {
                            first = local_first;
                        }
                        // check if this bucket range brings the overlaping buckets with data up
                        if local_last > last {
                            last = local_last;
                        }
                        // get an entry to this key/value bucket set
                        let entry: &mut BTreeSet<i32> = map.entry((key, value)).or_default();
                        // add the buckets for this group
                        entry.extend(buckets);
                    }
                }
            }
        }
        // set our oldest first to be our last
        *next_bucket = last.saturating_sub(1);
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
        for buckets in map.values() {
            // check each potential bucket in this bucket range
            for bucket in buckets {
                // check if this bucket is in our overlapping range
                if *bucket <= first && *bucket >= last {
                    // add this potential overlapping bucket or increment its count
                    let entry: &mut i32 = in_range.entry(*bucket).or_default();
                    // increment its count
                    *entry += 1;
                }
            }
        }
        // filter down to buckets in all of our bucket ranges
        in_range
            .iter()
            .filter(|(_, count)| **count as usize == tags_required)
            .map(|(bucket, _)| *bucket)
            .collect::<Vec<i32>>()
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
        group_by: &Vec<D::GroupBy>,
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
        let mut next_bucket = self.bucket as i32;
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
            // get all the buckets that contain data for each value
            for value in values {
                // check for census info for each group
                for group in group_by {
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
                pipe.cmd("zrange").arg(stream_key).arg(next_bucket).arg(end)
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
            let mut intersection =
                self.filter_bucket_intersection(group_by, &tags_retain.tags, tags_retain.tags_required, pre_intersection, &mut next_bucket, &mut possible);
            // if we have intersecting buckets or an intersection is not possible for this year then return
            if !intersection.is_empty() || !possible {
                // sort our intersection
                intersection.sort_unstable();
                return Ok(intersection);
            }
        }
    }

    /// Check if this cursor has been exhausted
    pub fn exhausted(&self) -> bool {
        // check if we have any more mapped values and our buckets are exhausted
        self.buckets_exhausted && self.mapped == 0
    }

    /// Check if this cursor mapped the full bounds of its data
    fn exhausted_time(&self) -> bool {
        // check if our year/bucket are at or past our end year/bucket
        self.year <= self.end_year && self.bucket <= self.end_bucket
    }

    /// Consume our sorted data in order to fill our user facing data buffer
    fn consume_sorted(&mut self, mut tag_ties: Option<&mut HashMap<String, String>>) -> bool {
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
                        if tag_ties.is_some() && item.len() > 1 {
                            // sort our ambigous rows by their cluster key
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
                                        if !item.dedupe_item(&mut self.retain.dedupe_set) {
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
                        // add every leftover row to our tie map so that rows in groups other
                        // than the front row's aren't skipped by our next query
                        if let Some(tag_ties) = &mut tag_ties {
                            // tag ties overwrite each other so add them back to front so the
                            // front most row in each group wins
                            for tied_row in item.iter().rev() {
                                // add this tie to our tag tie map
                                tied_row.add_tag_tie(tag_ties);
                            }
                        } else {
                            // regular ties keep the first tie for each group
                            for tied_row in &item {
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
        // get the number of buckets to check at once
        let bucket_limit = D::bucket_limit(&self.retain.extra_filter);
        // get data from any previously tied queries
        let tied_queries = self.query_ties(limit, shared).await?;
        // if we had any queries based on ties then consume them
        if !tied_queries.is_empty() {
            // cast and sort the rows we just retrieved
            D::sort(tied_queries, &mut self.sorted, &mut self.mapped).await?;
            // consume our sorted data and check if we have enough data to return
            if self.consume_sorted(None) {
                // we have enough data so return
                return Ok(());
            }
        }
        // determine which partitions have data
        self.query(limit, bucket_limit, shared).await?;
        // consume enough of our sorted data
        self.consume_sorted(None);
        Ok(())
    }

    #[instrument(name = "ScyllaCursor::tag_ties", skip_all, err(Debug))]
    async fn tag_ties(
        &self,
        tags_retain: &mut TagsRetain,
        mapping: &mut HashMap<String, TagMapping>,
        prepared: &PreparedStatement,
        shared: &Shared,
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
                    let query = ties_tags_query_helper(
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
        group_by: &Vec<D::GroupBy>,
        tags_retain: &mut TagsRetain,
        mapping: &mut HashMap<String, TagMapping>,
        prepared: &PreparedStatement,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // calculate the size of the futures were about to spawn
        let capacity = tags_retain.tags_required * self.retain.group_by.len();
        // have a vec of futures
        let mut futures = Vec::with_capacity(capacity);
        // loop until we have found enough data to return
        loop {
            // get the next 100 buckets that contain data
            let buckets = self
                .tags_find_buckets(group_by, tags_retain, shared)
                .await?;
            // query for each tag key/value
            for (key, values) in &tags_retain.tags {
                // query for each value for this tag key
                for value in values {
                    // build this query for each group
                    for group in group_by {
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
            // add our mapped data to our sorted items
            D::sort_tags(
                mapping,
                &tags_retain.tags,
                &mut self.sorted,
                &mut self.mapped,
            )?;
            // consume our sorted data and return if needed
            self.consume_sorted(Some(&mut tags_retain.ties));
            // if we have enough data to return then return
            if self.mapped >= self.limit {
                break;
            }
            // update our bucket counter correctly
            // this can't be moved to a function due to borrow check sadness
            match (self.year.cmp(&self.end_year), buckets.is_empty()) {
                // we have more buckets this year too query so just update our bucket
                (Ordering::Greater | Ordering::Equal, false) => {
                    self.bucket = (buckets.first().unwrap() - 1) as u32;
                }
                // we have more years to query so decrement our year and update our bucket
                (Ordering::Greater, true) => {
                    // decrement our year
                    self.year = self.year.saturating_sub(1);
                    // reset our bucket to the last possible partition of the year
                    self.bucket = last_year_bucket(self.year, self.partition_size);
                }
                // we have reached our end bucket and end year
                (Ordering::Equal | Ordering::Less, true) => {
                    match buckets.last() {
                        Some(bucket) => self.bucket = *bucket as u32,
                        None => self.bucket = self.end_bucket,
                    }
                    // set this cursor to be exhausted
                    self.buckets_exhausted = true;
                    break;
                }
                // we have gone past our end year bounds somehow
                (Ordering::Less, false) => {
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
        // get the group by  this cusrsor will use
        // put it back when were done with the refs to avoid cloning
        let group_by = std::mem::take(&mut self.retain.group_by);
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
            self.tag_ties(&mut tags_retain, &mut mapping, ties_prepared, shared)
                .await?;
        }
        // add our mapped data to our sorted items
        D::sort_tags(
            &mut mapping,
            &tags_retain.tags,
            &mut self.sorted,
            &mut self.mapped,
        )?;
        // consume our sorted data
        self.consume_sorted(Some(&mut tags_retain.ties));
        // only query for more data if we don't have enough to return
        if self.mapped < self.limit {
            // get tag data using normal queries
            self.tag_query(
                &group_by,
                &mut tags_retain,
                &mut mapping,
                query_prepared,
                shared,
            )
            .await?;
        }
        // put our group by back
        self.retain.group_by = group_by;
        Ok(tags_retain)
    }

    /// Gets the next page of data for this cursor
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaCursor::next", skip_all, err(Debug))]
    pub async fn next(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // if we have more already sorted rows then return those first
        if self.mapped > 0 {
            // consume our remaining sorted values
            self.consume_sorted(None);
            // if we have enough data then return
            if self.data.len() >= self.limit {
                return Ok(());
            }
        }
        // crawl over the tags table if we have tag filters
        match self.retain.tags_retain.take() {
            // we have tags to filter on
            Some(tags_retain) => {
                // get the next page of data based on tags
                let tags_retain = self.next_tags(tags_retain, shared).await?;
                // restore our retained tags data
                self.retain.tags_retain = Some(tags_retain);
                Ok(())
            }
            // we don't any any tags to filter on
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
        // build the key to save this cursor data too scoped to the creating user
        let key = cursors::data(CursorKind::Scylla, &self.username, &self.id, shared);
        // get how long to hold onto this cursor
        let ttl = scylla_cursor_save_expire(self.new);
        // save this cursors data to redis
        let _: () = query!(cmd("set").arg(key).arg(data).arg("EX").arg(ttl), shared).await?;
        Ok(())
    }
}

/// Get how many seconds [`ScyllaCursor::save`] holds a cursor in redis for
///
/// A new cursor is only held for 8 hours so abandoned new cursors don't linger in redis, and a
/// cursor that is being paged is held for 7 days.
///
/// # Arguments
///
/// * `new` - Whether this is a newly created cursor
fn scylla_cursor_save_expire(new: bool) -> u64 {
    // if this is a new cursor only hold onto it for 8 hours
    // if its an existing cursor that is being paged then hold onto it for 7 days
    if new { 28_800 } else { 604_800 }
}

impl<D> From<ScyllaCursor<D>> for ApiCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug + std::marker::Send,
    D: Serialize,
    D: ScyllaCursorSupport,
    D: Debug,
{
    /// convert this scylla cursor to a user facing cursor
    fn from(scylla_cursor: ScyllaCursor<D>) -> Self {
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

/// The [`CursorKind`] the exhausted-cursor delete path in [`SimpleScyllaCursor::save`] uses
///
/// This must match [`simple_scylla_save_kind`], since the kind is part of the key and deleting
/// under any other kind never removes the cursor from redis.
fn simple_scylla_delete_kind() -> CursorKind {
    CursorKind::SimpleScylla
}

/// The [`CursorKind`] the write path in [`SimpleScyllaCursor::save`] uses
///
/// Split out of the save key construction so it can be unit tested against the delete path's
/// kind. The delete path must use this same kind, or the exhausted-cursor `DEL` targets a key
/// that never exists (the kind is part of the key) and never removes the cursor from redis.
fn simple_scylla_save_kind() -> CursorKind {
    CursorKind::SimpleScylla
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
    /// The username of the user this cursor is scoped to
    pub username: String,
    /// The dta that is retained thoughout this cursors life
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
    /// * `username` - The username of the user this cursor is scoped to
    /// * `limit` - The max number of items to return at once
    #[must_use]
    pub fn new(partitions: Vec<String>, username: &str, limit: usize) -> Self {
        // create an initial simple scylla cursor
        let retain = SimpleScyllaCursorRetain {
            partitions,
            index: 0,
            tie: None,
        };
        // build our simple cursor scoped to the creating user
        SimpleScyllaCursor {
            id: Uuid::new_v4(),
            username: username.to_owned(),
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
    /// * `username` - The username of the user this cursor is scoped to
    /// * `limit` - The max number of items to return at once
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SimpleScyllaCursor::get", skip(shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        username: &str,
        limit: usize,
        shared: &Shared,
    ) -> Result<SimpleScyllaCursor<D>, ApiError> {
        // build the key to our cursor data in redis scoped to the creating user
        let key = cursors::data(CursorKind::SimpleScylla, username, &cursor_id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // deserialize our retained data
                let retain = deserialize!(&data);
                let cursor = SimpleScyllaCursor {
                    id: cursor_id,
                    username: username.to_owned(),
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
            // if we found any rows then cast them and add them to our data
            if cnt > 0 {
                // cast our rows into the correct objects and log any errors
                let found = typed_iter.filter_map(|res| log_scylla_err!(res));
                // add this data to the data to return
                self.data.extend(found);
            }
            // getting fewer rows than our limit means this partition is exhausted
            if cnt < limit {
                // check if we are at the last index
                if self.retain.index == self.retain.partitions.len() - 1 {
                    // we are at the last index so just return what we have
                    self.exhausted = true;
                    break;
                }
                // we have more partitions to get data from so continue on
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
        Ok(())
    }

    /// Saves this cursor to Redis
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SimpleScyllaCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // build the key to save this cursor data too scoped to the creating user
        let key = cursors::data(
            simple_scylla_delete_kind(),
            &self.username,
            &self.id,
            shared,
        );
        // either save or delete this cursor based on whether its exhausted or not
        if self.exhausted {
            // only delete if this cursor was actually written to redis
            if self.in_redis {
                // delete this cursor from redis
                let _: () = query!(cmd("del").arg(key), shared).await?;
            }
        } else {
            // serialize our paritions
            let data = serialize!(&self.retain);
            // build the key to save this cursor data too scoped to the creating user
            let key = cursors::data(simple_scylla_save_kind(), &self.username, &self.id, shared);
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

/// A cursor for a listing group permisisoned data within scylla
#[derive(Debug)]
pub struct GroupedScyllaCursor<D>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: GroupedScyllaCursorSupport,
    D: Debug,
{
    /// The Id for this cursor
    pub id: Uuid,
    /// The username of the user this cursor is scoped to
    pub username: String,
    /// The cursor settings/data to retain across cursor iterations
    pub retain: GroupedScyllaCursorRetain<D>,
    /// The max number of items to return at once
    pub limit: i32,
    /// Whether this cursor has data in redis or not
    pub in_redis: bool,
    /// The data this cursor has retrieved
    pub data: Vec<D>,
    /// Whether this is a new cursor or it already existed and is being continued
    pub new: bool,
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
    /// * `username` - The username of the user this cursor is scoped to
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "GroupedScyllaCursor::from_params",
        skip(params, dedupe, shared),
        err(Debug)
    )]
    pub async fn from_params(
        mut params: D::Params,
        username: &str,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<GroupedScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&mut params) {
            GroupedScyllaCursor::get(id, username, params, dedupe, shared).await
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
                username: username.to_owned(),
                retain,
                limit: D::get_limit(&params)?,
                in_redis: false,
                data: Vec::default(),
                new: true,
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
    /// * `username` - The username of the user this cursor is scoped to
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
        username: &str,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<GroupedScyllaCursor<D>, ApiError> {
        // if we have a cursor id then try to get our cursor from the DB
        if let Some(id) = D::get_id(&mut params) {
            GroupedScyllaCursor::get(id, username, params, dedupe, shared).await
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
                username: username.to_owned(),
                retain,
                limit: D::get_limit(&params)?,
                in_redis: false,
                data: Vec::default(),
                new: true,
            };
            Ok(cursor)
        }
    }

    /// Gets a cursors data from scylla
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The uuid of the cursor to retrieve if one is known
    /// * `username` - The username of the user this cursor is scoped to
    /// * `params` - The params to build this cursor from
    /// * `dedupe` - Whether to dedupe items or not
    /// * `shared` - Shared thorium objects
    #[instrument(name = "GroupedScyllaCursor::get", skip(params, shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        username: &str,
        params: D::Params,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<GroupedScyllaCursor<D>, ApiError> {
        // build the key to our cursor data in redis scoped to the creating user
        let key = cursors::data(CursorKind::GroupedScylla, username, &cursor_id, shared);
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
                    username: username.to_owned(),
                    retain,
                    limit: D::get_limit(&params)?,
                    in_redis: true,
                    data: Vec::default(),
                    new: false,
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

    /// Consume the rows returned by our tie queries into our user facing data buffer
    ///
    /// Once our ties have been queried they are cleared so later pages don't query and return the
    /// same tied entities again; consuming the tied rows may record a new leftover tie. Returns
    /// whether the tied rows filled our data buffer.
    ///
    /// # Arguments
    ///
    /// * `rows` - The rows returned by our tie queries
    fn consume_tied_rows(&mut self, rows: Vec<D::RowType>) -> bool {
        let sorted_map = Self::build_sorted_map(rows);
        // our ties have been queried so clear them before any new leftover tie is recorded
        self.retain.ties.clear();
        // consume the sorted data and add it to our user-facing data buffer
        self.consume_sorted(sorted_map);
        // see if our ties returned all we needed
        self.data.len() >= self.limit as usize
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
            // consume our tied rows and return early if they returned all we needed
            if self.consume_tied_rows(rows) {
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
        // build the key to save this cursor data too scoped to the creating user
        let key = cursors::data(CursorKind::GroupedScylla, &self.username, &self.id, shared);
        // either save or delete this cursor based on whether its exhausted or not
        if self.exhausted() {
            // only delete if this cursor was actually written to redis
            if self.in_redis {
                // delete this cursor from redis
                let _: () = query!(cmd("del").arg(key), shared).await?;
            }
        } else {
            // serialize our paritions
            let data = serialize!(&self.retain);
            // build the key to save this cursor data too scoped to the creating user
            let key = cursors::data(CursorKind::GroupedScylla, &self.username, &self.id, shared);
            // if this is a new cursor only hold onto it for 8 hours
            // if its an existing cursor that is being paged then hold onto it for 7 days
            let ttl = if self.new { 28_800 } else { 604_800 };
            // save this cursors data to redis
            let _: () = query!(cmd("set").arg(key).arg(data).arg("EX").arg(ttl), shared).await?;
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
            // the walk stops at the end timestamp's bucket in the end year
            end_bucket: u32::try_from(helpers::partition(end, end_year, partition_size))?,
            partition_size,
        };
        Ok(cursor)
    }

    /// Get the next window of partitions [`ExistsCursor::exists`] checks and advance past it
    ///
    /// Split out of the partition walk in `exists` so it can be unit tested (the walk never
    /// depends on query results, so advancing before the window is queried is identical).
    /// Returns the year and buckets to query and whether this is the final window.
    /// The walk stops at `end_bucket` in `end_year`.
    fn next_window(&mut self) -> (i32, Vec<i32>, bool) {
        // get the year this window is in
        let year = self.year;
        // check if we are in the final year to list and stop at the correct bucket
        let end = if self.year == self.end_year {
            std::cmp::max(self.bucket.saturating_sub(99), self.end_bucket)
        } else {
            self.bucket.saturating_sub(99)
        };
        let buckets = (end..=self.bucket)
            .map(|bucket| bucket as i32)
            .collect::<Vec<i32>>();
        // update our bucket counter correctly
        let done = match (
            self.year.cmp(&self.end_year),
            end == 0,
            end <= self.end_bucket,
        ) {
            // we have more buckets this year to query so just update our bucket
            (Ordering::Greater, false, _) | (Ordering::Equal, _, false) => {
                self.bucket = end - 1;
                false
            }
            // we have more years to query to decrement our year and update our bucket
            (Ordering::Greater, true, _) => {
                // decrement our year
                self.year = self.year.saturating_sub(1);
                // reset our bucket to the last possible partition of the year
                self.bucket = last_year_bucket(self.year, self.partition_size);
                false
            }
            // were done querying so update our bucket and break
            (_, _, _) => {
                self.bucket = end.saturating_sub(1);
                true
            }
        };
        (year, buckets, done)
    }

    /// Check if any rows exist across all partitions
    ///
    /// # Arguments
    ///
    /// * `prepared` - The statment to execute
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
            // get the next window of partitions to check and advance our cursor
            let (year, buckets, done) = self.next_window();
            // build the query to see if data exists in these partitions
            let query = shared
                .scylla
                .session
                .execute_unpaged(prepared, (year, buckets, key));
            // add this future out our futures list
            futures.push(query);
            // if we have have more then 100 futures to crawl then send them all at once
            if futures.len() >= 100 {
                // build our stream of futures
                let mut stream = stream::iter(futures.drain(..)).buffer_unordered(50);
                // check all partitons for data
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
            // stop once we have queried our final window
            if done {
                break;
            }
        }
        // if we hae any remaining futures then execute them
        if !futures.is_empty() {
            // build our stream of futures
            let mut stream = stream::iter(futures.drain(..)).buffer_unordered(50);
            // check all partitons for data
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

pub trait TagCountCursorSupport: ScyllaCursorSupport + Send {
    /// The type for our data with tags
    type Details;

    /// Get or create a new backing cursor for the data whose tags we are counting
    ///
    /// # Arguments
    ///
    /// * `params` - The params used to get our backing cursor
    /// * `extra` - Any extra params needed to get our backing cursor
    /// * `username` - The username of the user this cursor is scoped to
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn backing_cursor(
        params: <Self as CursorCore>::Params,
        _extra: <Self as CursorCore>::ExtraFilters,
        username: &str,
        shared: &Shared,
    ) -> Result<ScyllaCursor<Self>, ApiError> {
        // get our backing cursor scoped to the creating user
        ScyllaCursor::from_params(params, username, true, shared).await
    }

    /// Get the details for our backing cursor
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is calling this cursor
    /// * `backing` - The backing cursor to get details from
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn get_details(
        user: &User,
        backing: &mut ScyllaCursor<Self>,
        shared: &Shared,
    ) -> Result<Vec<Self::Details>, ApiError>;

    /// Extract the tags from a single item returned by this cursor
    ///
    /// # Arguments
    ///
    /// * `item` - The item to extract tags from
    fn tags(item: Self::Details) -> TagMap;
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TagCountCursorRetain {
    /// The total number of items that were counted
    pub total: u64,
    /// The specific counts for each Tag
    pub tags: HashMap<String, TagKeyCounts>,
}

impl TagCountCursorRetain {
    /// Create a new ``TagCounts`` object with some default capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - The capacity to set
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        TagCountCursorRetain {
            total: 0,
            tags: HashMap::with_capacity(capacity),
        }
    }

    /// Add some tags to our count
    ///
    /// # Arguments
    ///
    /// * `tags` - The tags to add to our count
    pub fn add_tags(&mut self, tags: TagMap) {
        // increment our total count
        self.total += 1;
        // step over these tags keys
        for (key, value_map) in tags {
            // get an entry to this keys count
            let key_entry = self.tags.entry(key).or_default();
            // count each of our values
            for (value, _) in value_map {
                // get an entry to this counts map
                let value_entry = key_entry.values.entry(value).or_default();
                // increment this key/value pairs count
                *value_entry += 1;
                // increment the total count for this key
                key_entry.total += 1;
            }
        }
    }
}

pub struct ScyllaTagCountCursor<D: TagCountCursorSupport + Send> {
    /// The id for our tag count cursor
    pub id: Uuid,
    /// The username of the user this cursor is scoped to
    pub username: String,
    /// The info to retain across tag counts
    pub retain: TagCountCursorRetain,
    /// The cursor backing our counts
    backing: ScyllaCursor<D>,
    /// Whether this is a new cursor or it already existed and is being continued
    pub new: bool,
}

impl<D: TagCountCursorSupport + Send> ScyllaTagCountCursor<D> {
    /// Get or create a new tag count cursor
    ///
    /// # Arguments
    ///
    /// * `params` - The params for counting tags
    /// * `extra` - Any extra params needed to count tags
    /// * `username` - The username of the user this cursor is scoped to
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaTagCountCursor::from_params", skip_all, err(Debug))]
    pub async fn from_params(
        params: <D as CursorCore>::Params,
        extra: <D as CursorCore>::ExtraFilters,
        username: &str,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // get our cursor id if we have one
        let (id, retain, new) = match <D as CursorCore>::get_id(&params) {
            Some(id) => {
                // build the key to our cursor data in redis scoped to the creating user
                let key = cursors::data(CursorKind::TagsCount, username, &id, shared);
                // get our cursor from redis
                let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
                // if we didn't get any data then return a 404
                match data {
                    // deserialize our cursor data
                    Some(data) => (id, deserialize!(&data), false),
                    // this cursor data is missing so return an error
                    None => return not_found!(format!("Cursor {id} not found!")),
                }
            }
            None => (
                Uuid::new_v4(),
                TagCountCursorRetain::with_capacity(1000),
                true,
            ),
        };
        // get our backing cursor
        let mut backing = D::backing_cursor(params, extra, username, shared).await?;
        // override our backing cursors id to be the same as our tags count cursor
        backing.id = id;
        // scope our backing cursor to the same user as our tags count cursor
        backing.username = username.to_owned();
        // build our internal tag count cursor
        let internal = ScyllaTagCountCursor {
            id,
            username: username.to_owned(),
            retain,
            backing,
            new,
        };
        Ok(internal)
    }

    /// Count the next page of data
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is counting tags and objects
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaTagCountCursor::next", skip_all, err(Debug))]
    pub async fn next(&mut self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        // get the next page of data for this cursor
        self.backing.next(shared).await?;
        // get the details for the current page of data
        let details = D::get_details(user, &mut self.backing, shared).await?;
        // count the tags for each item
        for item in details {
            // get the tags for this item
            let tags = D::tags(item);
            // add the tags to our count
            self.retain.add_tags(tags);
        }
        Ok(())
    }

    /// Save this tag count cursor to redis
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ScyllaTagCountCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // save our backing cursor
        self.backing.save(shared).await?;
        // serialize our cursor info
        let serialized = serialize!(&self.retain);
        // build the key to save this cursor data too scoped to the creating user
        let key = cursors::data(CursorKind::TagsCount, &self.username, &self.id, shared);
        // if this is a new cursor only hold onto it for 8 hours
        // if its an existing cursor that is being paged then hold onto it for 7 days
        let ttl = if self.new { 28_800 } else { 604_800 };
        // save this cursors data to redis
        let _: () = query!(
            cmd("set").arg(key).arg(serialized).arg("EX").arg(ttl),
            shared
        )
        .await?;
        Ok(())
    }

    /// Check if our count cursor has exhausted its backing cursor
    pub fn exhausted(&self) -> bool {
        self.backing.exhausted()
    }
}

impl<D: TagCountCursorSupport> From<ScyllaTagCountCursor<D>> for TagCounts {
    /// Convert an ``ScyllaTagCountCursor`` to a user facing ``TagCounts``
    ///
    /// # Arguments
    ///
    /// * `cursor` - The internal tag count cursor to convert
    fn from(cursor: ScyllaTagCountCursor<D>) -> Self {
        // if our cursor is not exhausted then return a cursor id
        let cursor_id = if cursor.exhausted() {
            None
        } else {
            Some(cursor.id)
        };
        // build our user facing tag count cursor
        TagCounts {
            cursor: cursor_id,
            total: cursor.retain.total,
            tags: cursor.retain.tags,
        }
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
    /// The username of the user this cursor is scoped to
    pub username: String,
    /// The info to retain throughout this cursors lifetime
    pub retain: ElasticCursorRetain,
    /// The max number of items to return at once
    pub limit: i64,
    /// The data this cursor has retrieved
    pub data: Vec<ElasticDoc>,
    /// Whether this is a new cursor or it already existed and is being continued
    pub new: bool,
}

/// Build the tenant-isolation group filter for an elastic search
///
/// The group filter is the ONLY tenant boundary on the search path, so it must never collapse to a
/// query that matches every document. In Elasticsearch a `bool` query with zero `should` clauses is
/// rewritten to `match_all`, so a caller that reaches this helper with an empty group list would
/// otherwise search every tenant's documents. This helper is therefore safe by
/// construction: it rejects an empty group list rather than emitting an all-matching filter.
///
/// # Arguments
///
/// * `groups` - The groups this search is scoped to
fn build_group_filter(groups: &[String]) -> Result<serde_json::Value, ApiError> {
    // an empty group list must never produce an empty `should` array, which Elasticsearch
    // rewrites to `match_all` and would leak every tenant's documents. Reject it here so the
    // tenant-isolation filter can never degrade into an all-matching query.
    if groups.is_empty() {
        // refuse to build a filter that would match all documents
        return internal_err!(
            "Refusing to build an elastic group filter with no groups".to_owned()
        );
    }
    // build one exact-match clause per group so only documents in these groups can match
    let group_filters = groups
        .iter()
        .map(|group| {
            // match this group exactly with no fuzziness or synonym expansion
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
    Ok(group_filters)
}

/// Build the `query_string` clause for a user supplied elastic search query
///
/// The raw user query is embedded as the `must` clause of the search. Elasticsearch's `query_string`
/// defaults leave leading wildcards, regex syntax, and unbounded boolean clause chaining enabled
/// across all fields, so a single crafted query can force per-shard term-dictionary enumeration and
/// exhaust the shared cluster's search threadpools. This helper centralizes the clause so
/// the cost-bounding options can be enforced in one place.
///
/// # Arguments
///
/// * `query` - The raw user supplied query string
fn build_query_string_clause(query: &str) -> serde_json::Value {
    // disable leading wildcards so a single query cannot force per-shard term-dictionary
    // enumeration, and bound the automaton expansion of regex/fuzzy terms so a crafted query cannot
    // exhaust the shared cluster's search threadpools
    serde_json::json!({
        "query_string": {
            "query": query,
            // reject leading `*`/`?` wildcards that scan the entire term dictionary
            "allow_leading_wildcard": false,
            // cap the states a single regex/fuzzy term can expand into
            "max_determinized_states": 10_000,
        }
    })
}

impl ElasticCursor {
    /// Create or get an elastic cursor based on search params
    ///
    /// # Arguments
    ///
    /// * `params` - The elastic search params to use
    /// * `username` - The username of the user this cursor is scoped to
    /// * `shared` - Shared Thorium objects
    pub async fn from_params(
        mut params: ElasticSearchParams,
        username: &str,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // if a cursor was specified then get an existing cursor
        if let Some(id) = params.cursor.take() {
            // get an existing cursor
            ElasticCursor::get(id, username, params, shared).await
        } else {
            // we don't have an existing cursor so make a new one
            ElasticCursor::new(params, username, shared).await
        }
    }

    /// Create a new elastic cursor
    ///
    /// # Arguments
    ///
    /// * `params` - The elastic search params to use
    /// * `username` - The username of the user this cursor is scoped to
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ElasticCursor::new", skip(shared), err(Debug))]
    async fn new(
        params: ElasticSearchParams,
        username: &str,
        shared: &Shared,
    ) -> Result<ElasticCursor, ApiError> {
        // get the names of our indexes
        let indexes = params
            .indexes
            .iter()
            .map(|index| index.full_name(&shared.config.elastic))
            .collect::<Vec<&str>>();
        // get a new point in time id for this search
        let pit = elastic::PointInTime::new(&indexes, shared).await?.id;
        // build an intial cursor retained data struct
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
            username: username.to_owned(),
            retain,
            limit: i64::from(params.limit),
            data: Vec::default(),
            new: true,
        };
        Ok(cursor)
    }

    /// Gets a cursors data for elastic
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the cursor to get
    /// * `username` - The username of the user this cursor is scoped to
    /// * `params` - The elastic search params to use
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ElasticCursor::get", skip(shared), err(Debug))]
    pub async fn get(
        id: Uuid,
        username: &str,
        params: ElasticSearchParams,
        shared: &Shared,
    ) -> Result<ElasticCursor, ApiError> {
        // build the key to our cursor data in redis scoped to the creating user
        let key = cursors::data(CursorKind::Elastic, username, &id, shared);
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
                    username: username.to_owned(),
                    retain,
                    limit: params.limit.into(),
                    data: Vec::default(),
                    new: false,
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
        // if we have no groups then short circuit and do nothing
        if self.retain.groups.is_empty() {
            // log that this user has no groups
            event!(Level::WARN, msg = "User has no groups");
            return Ok(());
        }
        // build the group filters, rejecting an empty group list that would match all documents
        let group_filters = build_group_filter(&self.retain.groups)?;
        // build the user-query clause with whatever cost bounding is enforced
        let query_clause = build_query_string_clause(&self.retain.query);
        // build the correct body depending on if we have any previous sort info or not
        let mut body = serde_json::json!({
            "pit": { "id": self.retain.pit, "keep_alive": "1d" },
            "query": {
                "bool": {
                    "must": query_clause,
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
        // deserialize the response if no error occured
        if resp.status_code().is_success() {
            // get the respose
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
        // build the key to save this cursor data too scoped to the creating user
        let key = cursors::data(CursorKind::Elastic, &self.username, &self.id, shared);
        // if this is a new cursor only hold onto it for 8 hours
        // if its an existing cursor that is being paged then hold onto it for 7 days
        let ttl = if self.new { 28_800 } else { 604_800 };
        // save this cursors data to redis
        let _: () = query!(cmd("set").arg(key).arg(data).arg("EX").arg(ttl), shared).await?;
        Ok(())
    }
}

/// The core logic all cursors must implement
pub trait RedisCursorSupport: Debug + Serialize + for<'a> Deserialize<'a> {
    /// The params to use when listing data in redis
    type Params: Debug;

    /// The type of data to return when listing without details
    type Names: Debug + Serialize + for<'a> Deserialize<'a>;

    /// The type of data we are sorting on
    type Sort: Debug + Ord + PartialOrd + Serialize + for<'a> Deserialize<'a>;

    /// Get our cursor id from params if one was provided
    ///
    /// # Arguments
    ///
    /// * `params` - The params to get a cursor id from
    fn get_id(params: &Self::Params) -> Option<Uuid>;

    /// Get the amount of data to return at most for this page
    ///
    /// # Arguments
    ///
    /// * `params` - The params ot get limits from
    fn get_limit(params: &Self::Params) -> usize;

    /// Get the groups to return data from
    ///
    /// # Arguments
    ///
    /// * `params` - The params ot get groups from
    fn get_groups(params: &Self::Params) -> &Vec<String>;

    /// Build the keys to a specific groups name set
    ///
    /// # Arguments
    ///
    /// * `params` - The params to build keys from
    /// * `shared` - Shared Thorium objects
    fn name_key(params: &Self::Params, group: &str, shared: &Shared) -> String;

    /// Get the name of items in a single group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to get names in
    #[allow(async_fn_in_trait)]
    async fn get_names(
        params: &Self::Params,
        group_cursor: &RedisGroupCursor,
        shared: &Shared,
    ) -> Result<(u64, Vec<Self::Names>), ApiError>;

    /// Add new data to our cursor
    ///
    /// # Arguments
    ///
    /// * `names` - The names to add to this groups sorted list
    /// * `sorted` - The sorted list to add new names too
    fn add_names(names: Vec<Self::Names>, sorted: &mut BTreeMap<Self::Sort, Self::Names>);
}

/// A Single name row to possibly return to the user
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RedisCursorName<D: RedisCursorSupport> {
    /// The name to return
    pub name: D::Names,
    /// What group this name came from if it was the last item from this groups page
    pub group: Option<String>,
}

/// A single cursor for a group in redis
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RedisGroupCursor {
    /// The redis key to use for this cursor
    pub key: String,
    /// The value to use to continue this cursor in redis
    pub cursor: u64,
}

/// Get the initial page of data for a single group
///
/// # Arguments
///
/// * `params` - The parameters for listing data
/// * `group` - The group to list data from
/// * `shared` - Shared Thorium objects
async fn initial_names_helper<D: RedisCursorSupport>(
    params: &D::Params,
    group: String,
    shared: &Shared,
) -> Result<Option<(String, RedisGroupCursor, BTreeMap<D::Sort, D::Names>)>, ApiError> {
    // build the key to the data to list
    let key = D::name_key(params, &group, shared);
    // build a cursor with this key
    // default to 0 as the initial value for the cursor
    let mut group_cursor = RedisGroupCursor { key, cursor: 0 };
    // get this groups names
    let (new_cursor, names) = D::get_names(params, &group_cursor, shared).await?;
    // this group only has no data if its scan completed without returning names
    // an empty batch with a non zero cursor is valid and must keep this group alive
    if names.is_empty() && new_cursor == 0 {
        // this group contains no data
        Ok(None)
    } else {
        // this group contains data so update its cursor
        group_cursor.cursor = new_cursor;
        // instance a btree to sort our names into
        let mut sorted = BTreeMap::default();
        // sort our names into our new btree
        D::add_names(names, &mut sorted);
        // return the info for this group
        Ok(Some((group, group_cursor, sorted)))
    }
}

/// The data to retain throughout this redis cursors life
#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(bound = "")]
pub struct RedisCursorRetain<D: RedisCursorSupport> {
    /// The groups whose redis scans are unfinished and the info to continue those scans
    pub groups: HashMap<String, RedisGroupCursor>,
    /// A sorted set of names to return in future iterations sorted by group
    ///
    /// Every group that may still yield data has an entry here even if its
    /// buffered names are currently empty
    pub remaining: BTreeMap<String, BTreeMap<D::Sort, D::Names>>,
}

impl<D: RedisCursorSupport> RedisCursorRetain<D> {
    /// Get the first page of data
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters used for this cursors retained data
    /// * `shared` - Shared Thorium objects
    pub async fn new(params: &D::Params, shared: &Shared) -> Result<Self, ApiError> {
        // get the names of the groups we need to get data for
        let groups = D::get_groups(params).clone();
        // create a new empty retain to populate data into
        let mut retain = RedisCursorRetain {
            groups: HashMap::with_capacity(groups.len()),
            remaining: BTreeMap::default(),
        };
        // get the next page of data across all groups
        let cursor_iter = stream::iter(groups)
            .map(|group| initial_names_helper::<D>(params, group, shared))
            .buffer_unordered(10)
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .flatten();
        // step over and add any groups with data
        for (group, group_cursor, names) in cursor_iter {
            // only track this groups scan cursor if its scan has not yet completed
            // a cursor of 0 means this groups scan finished in a single pass
            if group_cursor.cursor != 0 {
                // add this group to our cursor
                retain.groups.insert(group.clone(), group_cursor);
            }
            // add this groups names to our cursor
            retain.remaining.insert(group, names);
        }
        Ok(retain)
    }
}

#[derive(Debug)]
pub struct RedisCursor<D: RedisCursorSupport>
where
    for<'de> D: Deserialize<'de> + Debug,
    D: Serialize,
    D: RedisCursorSupport,
    D: Debug,
{
    /// The Id for this cursor
    pub id: Uuid,
    /// The username of the user this cursor is scoped to
    pub username: String,
    /// The cursor settings/data to retain across cursor iterations
    pub retain: RedisCursorRetain<D>,
    /// The params to use with this cursor
    params: D::Params,
    /// The max amount of data to return at once
    pub limit: usize,
    /// Whether this cursor has been saved to redis before
    in_redis: bool,
    /// Whether this is a new cursor or it already existed and is being continued
    pub new: bool,
}

impl<D: RedisCursorSupport> RedisCursor<D> {
    /// Get or create a new redis cursor
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to use with this cursor
    /// * `username` - The username of the user this cursor is scoped to
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "RedisCursor::new", skip_all, err(Debug))]
    pub async fn new(params: D::Params, username: &str, shared: &Shared) -> Result<Self, ApiError> {
        // if a cursor id was passed then load that existing cursor from redis
        match D::get_id(&params) {
            Some(id) => RedisCursor::get(id, username, params, shared).await,
            None => {
                // get the limit on the amount of data to return
                let limit = D::get_limit(&params);
                // this is a new cursor so seed the first page of every group
                let retain = RedisCursorRetain::<D>::new(&params, shared).await?;
                // build our new cursor with a new id
                let cursor = RedisCursor {
                    id: Uuid::new_v4(),
                    username: username.to_owned(),
                    retain,
                    params,
                    limit,
                    in_redis: false,
                    new: true,
                };
                Ok(cursor)
            }
        }
    }

    /// Gets an existing redis cursors data from redis
    ///
    /// # Arguments
    ///
    /// * `cursor_id` - The id of the cursor to load
    /// * `username` - The username of the user this cursor is scoped to
    /// * `params` - The parameters to use with this cursor
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "RedisCursor::get", skip(params, shared), err(Debug))]
    pub async fn get(
        cursor_id: Uuid,
        username: &str,
        params: D::Params,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // build the key to our cursor data in redis scoped to the creating user
        let key = cursors::data(CursorKind::Redis, username, &cursor_id, shared);
        // get our cursor from redis
        let data: Option<String> = query!(cmd("get").arg(key), shared).await?;
        // check if we got any cursor data
        match data {
            Some(data) => {
                // try to deserialize this cursors retained data
                let retain: RedisCursorRetain<D> = deserialize!(&data);
                // rebuild our cursor
                let cursor = RedisCursor {
                    id: cursor_id,
                    username: username.to_owned(),
                    retain,
                    limit: D::get_limit(&params),
                    params,
                    in_redis: true,
                    new: false,
                };
                Ok(cursor)
            }
            None => not_found!(format!("Cursor {cursor_id} doesn't exist")),
        }
    }

    /// Check if this cursor has been exhausted
    pub fn exhausted(&self) -> bool {
        // check if we still have any remaining data to return
        self.retain.remaining.is_empty()
    }

    /// Consume sorted data and add it to an existing vec
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub async fn consume(&mut self, shared: &Shared) -> Result<ApiCursor<D::Names>, ApiError> {
        // get the correct size to preallocate for our api cursor data
        let size = std::cmp::min(self.limit, self.retain.remaining.len());
        // preallocate the vec of data to return
        let mut data = Vec::with_capacity(size);
        // keep consuming data until we have the requested amount
        while data.len() < self.limit {
            // get a mutable entry to the first group with data
            match self.retain.remaining.first_entry() {
                // we have a group that may have data to return still
                Some(mut first) => {
                    // get the next entry to return to the user
                    // we have to do this in a sub scope in order to avoid multiple
                    // mutable references at once
                    let maybe_next = {
                        // get a mutable entry to this entry
                        let entry = first.get_mut();
                        // try to get the first item from this group
                        entry.pop_first()
                    };
                    // try to get the first item from this group
                    match maybe_next {
                        Some((_, name)) => {
                            // add this name to our return data
                            data.push(name);
                        }
                        None => {
                            // this groups buffered names are drained so check if its
                            // redis scan still has more data to pull
                            // we have to refill before moving to the next group to ensure
                            // we don't return out of order data on the next page
                            match self.retain.groups.get_mut(first.key()) {
                                Some(group_cursor) => {
                                    // this groups scan is unfinished so get its next batch of names
                                    let (new_cursor, names) =
                                        D::get_names(&self.params, group_cursor, shared).await?;
                                    // update this groups cursor
                                    group_cursor.cursor = new_cursor;
                                    // get a mutable entry to this entry
                                    let entry = first.get_mut();
                                    // add these names to our remaining data
                                    D::add_names(names, entry);
                                    // if this scan has now completed then stop tracking its cursor
                                    if new_cursor == 0 {
                                        // get the group for this entry
                                        let group = first.key().clone();
                                        // remove this groups scan cursor since its scan is complete
                                        self.retain.groups.remove(&group);
                                        // if the final batch added no names then this group is
                                        // fully exhausted and can be removed
                                        if first.get().is_empty() {
                                            first.remove_entry();
                                        }
                                    }
                                    // if the batch was empty but the scan is unfinished we simply
                                    // loop and refill this group again on the next iteration
                                }
                                None => {
                                    // this groups scan already completed and its names are
                                    // drained so remove it from our remaining data
                                    first.remove_entry();
                                }
                            }
                        }
                    }
                }
                // we have no more data to return
                None => break,
            }
        }
        // drop any groups that are fully drained and fully scanned so our
        // exhausted check is accurate on exact page boundaries
        let groups = &self.retain.groups;
        self.retain
            .remaining
            .retain(|group, names| !names.is_empty() || groups.contains_key(group));
        // if our cursor is exhausted then don't include a cursor id
        let id = if self.exhausted() {
            None
        } else {
            Some(self.id)
        };
        // build our cursor object
        Ok(ApiCursor { cursor: id, data })
    }

    /// Saves this cursor to Redis
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "RedisCursor::save", skip_all, err(Debug))]
    pub async fn save(&self, shared: &Shared) -> Result<(), ApiError> {
        // build the key to save this cursor data too scoped to the creating user
        let key = cursors::data(CursorKind::Redis, &self.username, &self.id, shared);
        // either save or delete this cursor based on whether its exhausted or not
        if self.exhausted() {
            // only delete if this cursor was actually written to redis
            if self.in_redis {
                // delete this cursor from redis
                let _: () = query!(cmd("del").arg(key), shared).await?;
            }
        } else {
            // serialize our retained data
            let data = serialize!(&self.retain);
            // if this is a new cursor only hold onto it for 8 hours
            // if its an existing cursor that is being paged then hold onto it for 7 days
            let ttl = if self.new { 28_800 } else { 604_800 };
            // save this cursors data to redis
            let _: () = query!(cmd("set").arg(key).arg(data).arg("EX").arg(ttl), shared).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExistsCursor, GroupedScyllaCursor, GroupedScyllaCursorRetain, ScyllaCursor,
        ScyllaCursorRetain, build_group_filter, build_query_string_clause,
        scylla_cursor_save_expire, simple_scylla_delete_kind, simple_scylla_save_kind,
    };
    use crate::models::{NetworkPolicyListLine, NetworkPolicyListRow, SampleListLine};
    use crate::utils::helpers;
    use chrono::prelude::*;
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    /// `SimpleScyllaCursor::save` builds the exhausted-cursor `DEL` key with the same
    /// `CursorKind` as the save key. The kind is part of the redis key, so this is what lets the
    /// delete actually remove the exhausted simple cursor it wrote.
    #[test]
    fn simple_scylla_delete_kind_matches_save_kind() {
        // the delete path must key on the same cursor kind the save path wrote, or the DEL is a
        // no-op against a nonexistent key and the exhausted cursor lingers forever
        assert_eq!(
            simple_scylla_delete_kind().as_str(),
            simple_scylla_save_kind().as_str(),
            "the exhausted-cursor delete key must use the same CursorKind as the save key"
        );
    }

    /// the group filter must reject an empty group list rather than emit an empty `should`
    /// array that Elasticsearch rewrites to `match_all` (a full cross-tenant search read).
    #[test]
    fn group_filter_rejects_empty_groups() {
        // building a filter with no groups must fail rather than match every tenant's documents
        assert!(
            build_group_filter(&[]).is_err(),
            "empty group list produced a filter (match_all cross-tenant leak)"
        );
    }

    /// A normal, non-empty group list must still build one exact-match clause per group so tenant
    /// isolation works for the common case.
    #[test]
    fn group_filter_builds_a_clause_per_group() {
        // build a filter scoped to two groups
        let filter = build_group_filter(&["team-a".to_owned(), "team-b".to_owned()])
            .expect("non-empty group list should build a filter");
        // the filter must be an array with exactly one clause per group
        let clauses = filter
            .as_array()
            .expect("group filter should be a JSON array of match clauses");
        assert_eq!(
            clauses.len(),
            2,
            "expected one match clause per group in the tenant-isolation filter"
        );
    }

    /// the user `query_string` clause must be cost-bounded so a single crafted query cannot
    /// enumerate the entire term dictionary of the shared indexes and exhaust the search
    /// threadpools. At minimum leading wildcards must be disabled.
    #[test]
    fn query_string_clause_disables_leading_wildcards() {
        // build the clause for an arbitrary user query
        let clause = build_query_string_clause("password OR flag");
        // dig into the query_string options the clause carries
        let options = clause
            .get("query_string")
            .expect("clause should carry a query_string object");
        // leading wildcards must be explicitly disabled to bound multi-term expansion cost
        assert_eq!(
            options.get("allow_leading_wildcard"),
            Some(&serde_json::Value::Bool(false)),
            "query_string clause did not disable leading wildcards (DoS)"
        );
    }

    /// Build a UTC timestamp at midnight on a day
    ///
    /// # Arguments
    ///
    /// * `year` - The year of this timestamp
    /// * `month` - The month of this timestamp
    /// * `day` - The day of this timestamp
    fn day(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).unwrap()
    }

    /// Walk every window an exists cursor queries and collect the buckets checked in each year
    ///
    /// # Arguments
    ///
    /// * `cursor` - The exists cursor to walk
    fn walk(mut cursor: ExistsCursor) -> HashMap<i32, HashSet<i32>> {
        let mut checked: HashMap<i32, HashSet<i32>> = HashMap::new();
        // cap the walk so a broken cursor can't loop forever
        for _ in 0..100_000 {
            let (year, buckets, done) = cursor.next_window();
            checked.entry(year).or_default().extend(buckets);
            if done {
                return checked;
            }
        }
        panic!("the exists cursor walk never finished");
    }

    /// `ExistsCursor::new` computes `end_bucket` as the bucket of the `end` timestamp in
    /// `end_year`. This lets the walk reach the final year's early buckets, so `exists()` checks
    /// them and `prune_data` doesn't delete repo data blobs still referenced by commits there.
    #[test]
    fn exists_cursor_end_bucket_uses_end_timestamp() {
        // check from Sept 24 2026 back to Feb 1 2020 in hour sized partitions
        let start = day(2026, 9, 24);
        let end = day(2020, 2, 1);
        let cursor = ExistsCursor::new(start, end, 3_600).unwrap();
        // the walk must stop at the end timestamp's bucket
        assert_eq!(
            cursor.end_bucket,
            helpers::partition(end, 2020, 3_600) as u32,
            "the end bucket must be computed from the end timestamp and end year"
        );
    }

    /// Walking an exists cursor from 2026 back to Feb 1 2020 checks the buckets of early 2020
    /// (here March 1 2020), so data referenced only there isn't treated as unreferenced and
    /// deleted.
    #[test]
    fn exists_walk_covers_end_year_early_buckets() {
        let start = day(2026, 9, 24);
        let end = day(2020, 2, 1);
        let checked = walk(ExistsCursor::new(start, end, 3_600).unwrap());
        // a commit on March 1 2020 is inside the requested range
        let march = helpers::partition(day(2020, 3, 1), 2020, 3_600);
        assert!(
            checked
                .get(&2020)
                .is_some_and(|buckets| buckets.contains(&march)),
            "the exists walk must check every bucket of its final year back to the end timestamp"
        );
    }

    /// Within a single year, walking from Sept 24 back to Jan 15 checks March 1 rather than only
    /// the start's window.
    #[test]
    fn exists_walk_covers_same_year_range() {
        let start = day(2026, 9, 24);
        let end = day(2026, 1, 15);
        let checked = walk(ExistsCursor::new(start, end, 3_600).unwrap());
        let march = helpers::partition(day(2026, 3, 1), 2026, 3_600);
        assert!(
            checked
                .get(&2026)
                .is_some_and(|buckets| buckets.contains(&march)),
            "the exists walk must check every bucket between its start and end"
        );
    }

    /// When a cursor walk rolls back into an earlier year it resets its bucket to the last
    /// partition of that year, sized by that year's own length. Rolling back into a leap year
    /// (2024, after the 365 day 2023) therefore starts at bucket 8783 with hourly partitions, so
    /// every partition inside the requested range is checked, including the final hours of
    /// Dec 31 2024.
    #[test]
    fn exists_walk_covers_leap_year_december_31() {
        let start = day(2026, 9, 24);
        let end = day(2020, 2, 1);
        let checked = walk(ExistsCursor::new(start, end, 3_600).unwrap());
        // the last hour of 2024 (a leap year) is inside the requested range
        let last_hour = helpers::partition(
            Utc.with_ymd_and_hms(2024, 12, 31, 23, 30, 0).unwrap(),
            2024,
            3_600,
        );
        assert!(
            checked
                .get(&2024)
                .is_some_and(|buckets| buckets.contains(&last_hour)),
            "the exists walk must check the final hours of a leap year"
        );
    }

    /// Rolling back into a regular year after a leap year checks that year's final hour.
    #[test]
    fn exists_walk_covers_regular_year_december_31() {
        let start = day(2026, 9, 24);
        let end = day(2020, 2, 1);
        let checked = walk(ExistsCursor::new(start, end, 3_600).unwrap());
        // the last hour of 2025 (a regular year after the 2024 leap year) is inside the range
        let last_hour = helpers::partition(
            Utc.with_ymd_and_hms(2025, 12, 31, 23, 30, 0).unwrap(),
            2025,
            3_600,
        );
        assert!(
            checked
                .get(&2025)
                .is_some_and(|buckets| buckets.contains(&last_hour))
        );
    }

    /// Buckets in the years between the start and end years are checked.
    #[test]
    fn exists_walk_covers_middle_years() {
        let start = day(2026, 9, 24);
        let end = day(2020, 2, 1);
        let checked = walk(ExistsCursor::new(start, end, 3_600).unwrap());
        assert!(
            checked
                .get(&2023)
                .is_some_and(|buckets| buckets.contains(&100))
        );
    }

    /// Build a listed sample in some groups
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of this sample
    /// * `groups` - The groups this submission is in
    /// * `uploaded` - When this submission was uploaded
    fn sample(sha256: &str, groups: &[&str], uploaded: DateTime<Utc>) -> SampleListLine {
        SampleListLine {
            groups: groups.iter().map(|group| (*group).to_owned()).collect(),
            sha256: sha256.to_owned(),
            submission: Some(Uuid::new_v4()),
            uploaded,
        }
    }

    /// Build a sample listing cursor over 2020-2026 with some retained data
    ///
    /// # Arguments
    ///
    /// * `retain` - The data this cursor retained
    /// * `limit` - The max number of items to return at once
    /// * `dedupe` - Whether to dedupe items or not
    fn sample_cursor(
        retain: ScyllaCursorRetain<SampleListLine>,
        limit: usize,
        dedupe: bool,
    ) -> ScyllaCursor<SampleListLine> {
        ScyllaCursor::rebuild(Uuid::new_v4(), "user", retain, 3_600, limit, dedupe).unwrap()
    }

    /// Build fresh retained data for a sample listing cursor over 2020-2026
    fn sample_retain() -> ScyllaCursorRetain<SampleListLine> {
        ScyllaCursorRetain {
            start: day(2026, 9, 24),
            end: day(2020, 1, 2),
            extra_filter: (),
            group_by: vec!["g1".to_owned(), "g2".to_owned()],
            ties: HashMap::new(),
            tags_retain: None,
            dedupe_set: HashSet::new(),
        }
    }

    /// Add rows sharing one timestamp to a cursor's sorted buffer
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor to add rows to
    /// * `rows` - The rows to add, all sharing one timestamp
    fn add_sorted(cursor: &mut ScyllaCursor<SampleListLine>, rows: Vec<SampleListLine>) {
        let timestamp = rows[0].uploaded;
        cursor.mapped += rows.len();
        cursor.sorted.entry(timestamp).or_default().extend(rows);
    }

    /// When a page ends inside a run of rows sharing one timestamp, `consume_sorted` records a
    /// tie for the groups of every leftover row, not just the front one. The next page only pulls
    /// rows strictly older than that timestamp, so without a tie the leftover rows of group g1
    /// (which trail g2's in the run) would never be returned.
    #[test]
    fn consume_sorted_records_ties_for_all_leftover_groups() {
        let mut cursor = sample_cursor(sample_retain(), 4, false);
        // at one timestamp g2 has five rows followed by g1's two
        let at = day(2026, 9, 1);
        let mut rows: Vec<SampleListLine> = (0..5)
            .map(|i| sample(&format!("i{i}"), &["g2"], at))
            .collect();
        rows.extend((0..2).map(|i| sample(&format!("j{i}"), &["g1"], at)));
        add_sorted(&mut cursor, rows);
        // fill a page of 4
        cursor.consume_sorted(None);
        // g1's leftover rows at this timestamp must get a tie so the next page returns them
        assert!(
            cursor.retain.ties.contains_key("g1"),
            "every group with leftover rows at the page boundary needs a tie, got {:?}",
            cursor.retain.ties
        );
    }

    /// The front leftover row's group gets a tie and the page is filled in order.
    #[test]
    fn consume_sorted_records_tie_for_front_leftover_group() {
        let mut cursor = sample_cursor(sample_retain(), 4, false);
        let at = day(2026, 9, 1);
        let rows: Vec<SampleListLine> = (0..5)
            .map(|i| sample(&format!("i{i}"), &["g2"], at))
            .collect();
        let fifth = rows[4].submission.unwrap().to_string();
        add_sorted(&mut cursor, rows);
        cursor.consume_sorted(None);
        assert_eq!(cursor.data.len(), 4);
        assert_eq!(cursor.retain.ties.get("g2"), Some(&fifth));
    }

    /// The set of samples already returned by a deduped listing is retained between pages, so a
    /// sample whose submissions fall on different pages (here X) is only returned on the first.
    #[test]
    fn dedupe_persists_across_pages() {
        // page 1 returns sample X's newest submission
        let mut page_one = sample_cursor(sample_retain(), 1, true);
        add_sorted(&mut page_one, vec![sample("X", &["g1"], day(2026, 9, 2))]);
        page_one.consume_sorted(None);
        assert_eq!(page_one.data[0].sha256, "X");
        // persist the retained data between requests just like saving to redis does
        let saved = serde_json::to_string(&page_one.retain).unwrap();
        let retain: ScyllaCursorRetain<SampleListLine> = serde_json::from_str(&saved).unwrap();
        // page 2 sees X's older submission followed by sample Z
        let mut page_two = sample_cursor(retain, 1, true);
        add_sorted(&mut page_two, vec![sample("X", &["g1"], day(2026, 8, 2))]);
        add_sorted(&mut page_two, vec![sample("Z", &["g1"], day(2026, 7, 2))]);
        page_two.consume_sorted(None);
        // X was already returned so page 2 must skip it
        let returned: Vec<&str> = page_two
            .data
            .iter()
            .map(|line| line.sha256.as_str())
            .collect();
        assert_eq!(
            returned,
            vec!["Z"],
            "a deduped listing must not repeat X on a later page"
        );
    }

    /// Duplicate submissions of a sample within one page are deduped.
    #[test]
    fn dedupe_within_one_page() {
        let mut cursor = sample_cursor(sample_retain(), 2, true);
        add_sorted(&mut cursor, vec![sample("X", &["g1"], day(2026, 9, 2))]);
        add_sorted(&mut cursor, vec![sample("X", &["g1"], day(2026, 8, 2))]);
        add_sorted(&mut cursor, vec![sample("Z", &["g1"], day(2026, 7, 2))]);
        cursor.consume_sorted(None);
        let returned: Vec<&str> = cursor
            .data
            .iter()
            .map(|line| line.sha256.as_str())
            .collect();
        assert_eq!(returned, vec!["X", "Z"]);
    }

    /// `ScyllaCursor::save` holds a new cursor in redis for 8 hours (28800 seconds), so abandoned
    /// new cursors don't linger for a week.
    #[test]
    fn new_scylla_cursor_expires_in_8h() {
        assert_eq!(
            scylla_cursor_save_expire(true),
            28_800,
            "a new scylla cursor must only be held in redis for 8 hours"
        );
    }

    /// A cursor that is being paged is held for 7 days.
    #[test]
    fn paged_scylla_cursor_expires_in_7d() {
        assert_eq!(scylla_cursor_save_expire(false), 604_800);
    }

    /// Build a listed network policy row in a group
    ///
    /// # Arguments
    ///
    /// * `group` - The group this row is in
    /// * `name` - The name of this network policy
    /// * `id` - The id of this network policy
    fn policy_row(group: &str, name: &str, id: Uuid) -> NetworkPolicyListRow {
        NetworkPolicyListRow {
            group: group.to_owned(),
            name: name.to_owned(),
            id,
        }
    }

    /// Build a network policy listing cursor over groups g1 and g2
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of items to return at once
    /// * `ties` - The ties retained from a previous page
    fn policy_cursor(
        limit: i32,
        ties: Vec<(String, Vec<String>)>,
    ) -> GroupedScyllaCursor<NetworkPolicyListLine> {
        GroupedScyllaCursor {
            id: Uuid::new_v4(),
            username: "user".to_owned(),
            retain: GroupedScyllaCursorRetain {
                extra_filter: (),
                group_by: ["g1".to_owned(), "g2".to_owned()].into_iter().collect(),
                ties,
                current_sort_by: Some("b".to_owned()),
            },
            limit,
            in_redis: true,
            data: Vec::default(),
            new: false,
        }
    }

    /// `GroupedScyllaCursor::next` queries `retain.ties` at the start of every page and clears
    /// them once the tied entities are consumed, even when no new leftover exists. This keeps
    /// later pages from returning the same tied entity (here "c") again.
    #[test]
    fn grouped_cursor_clears_consumed_ties() {
        // page 1 left entity c behind as a tie
        let mut cursor = policy_cursor(2, vec![("c".to_owned(), vec!["g1".to_owned()])]);
        // page 2's tie query returns c, which fits on this page with room to spare
        let filled = cursor.consume_tied_rows(vec![policy_row("g1", "c", Uuid::new_v4())]);
        assert!(!filled);
        assert_eq!(cursor.data.len(), 1);
        // the consumed tie must not be queried again on the next page
        assert!(
            cursor.retain.ties.iter().all(|(sort_by, _)| sort_by != "c"),
            "a consumed tie must be cleared, got {:?}",
            cursor.retain.ties
        );
    }

    /// When tied rows overflow the page, the leftover is recorded as a tie.
    #[test]
    fn grouped_cursor_records_leftover_tie() {
        let mut cursor = policy_cursor(1, Vec::default());
        let filled = cursor.consume_tied_rows(vec![
            policy_row("g1", "c", Uuid::new_v4()),
            policy_row("g2", "c", Uuid::new_v4()),
        ]);
        assert!(filled);
        assert_eq!(
            cursor.retain.ties,
            vec![("c".to_owned(), vec!["g2".to_owned()])]
        );
    }
}
