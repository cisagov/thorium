//! Structures related to searching in Elastic Search

use chrono::prelude::*;
use uuid::Uuid;

#[cfg(feature = "api")]
use strum::{EnumIter, IntoEnumIterator};

/// The different elastic indexes
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "api", derive(EnumIter))]
pub enum ElasticIndex {
    /// The results for jobs on samples
    SampleResults,
    /// The results for jobs on repos
    RepoResults,
    /// The tags on samples
    SampleTags,
    /// The tags on repos
    RepoTags,
}

impl std::fmt::Display for ElasticIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ElasticIndex::SampleResults => write!(f, "SampleResults"),
            ElasticIndex::RepoResults => write!(f, "RepoResults"),
            ElasticIndex::SampleTags => write!(f, "SampleTags"),
            ElasticIndex::RepoTags => write!(f, "RepoTags"),
        }
    }
}

impl ElasticIndex {
    /// Get the full index name for a specific index
    ///
    /// # Arguments
    ///
    /// * `elastic_conf` - The elastic config
    #[must_use]
    pub fn full_name<'a>(&self, elastic_conf: &'a crate::conf::Elastic) -> &'a str {
        match self {
            ElasticIndex::SampleResults => &elastic_conf.results.samples,
            ElasticIndex::RepoResults => &elastic_conf.results.repos,
            ElasticIndex::SampleTags => &elastic_conf.tags.samples,
            ElasticIndex::RepoTags => &elastic_conf.tags.repos,
        }
    }

    /// Get the earliest date for documents in the given index
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    pub fn earliest(&self, shared: &crate::utils::Shared) -> i64 {
        match self {
            ElasticIndex::SampleResults | ElasticIndex::RepoResults => {
                shared.config.thorium.results.earliest
            }
            ElasticIndex::SampleTags | ElasticIndex::RepoTags => {
                shared.config.thorium.tags.earliest
            }
        }
    }

    /// Get the earliest date for documents in all possible indexes
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    ///
    /// # Panics
    ///
    /// Panics if `ElasticIndex` has no variants (there are no possible indexes)
    #[cfg(feature = "api")]
    pub fn earliest_all(shared: &crate::utils::Shared) -> i64 {
        Self::iter()
            .map(|index| index.earliest(shared))
            .min()
            // safe to unwrap because we'll always have at least one index
            .expect("No elastic indexes")
    }
}

/// Returns all elastic indexes as a default
fn default_search_indexes() -> Vec<ElasticIndex> {
    // only search on samples, as the Web UI is the only user of
    // search and only samples are currently supported there
    vec![ElasticIndex::SampleResults, ElasticIndex::SampleTags]
}

/// Default the Result list limit to 50
fn default_search_limit() -> u32 {
    50
}

/// The query params for searching results
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ElasticSearchParams {
    /// The indexes to search
    #[serde(default = "default_search_indexes")]
    pub indexes: Vec<ElasticIndex>,
    /// The query to use when searching
    #[serde(default)]
    pub query: String,
    /// The groups to search data from
    #[serde(default)]
    pub groups: Vec<String>,
    /// When to start searching data at
    #[serde(default = "Utc::now")]
    pub start: DateTime<Utc>,
    /// When to stop searching data at
    pub end: Option<DateTime<Utc>>,
    /// The cursor id to use if one exists
    pub cursor: Option<Uuid>,
    /// The max number of items to return in this response
    #[serde(default = "default_search_limit")]
    pub limit: u32,
}

/// The query params for searching results
#[derive(Serialize, Deserialize, Debug)]
pub struct ElasticSearchOpts {
    /// The indexes to search
    pub indexes: Vec<ElasticIndex>,
    /// The query to use when searching
    pub query: String,
    /// The groups to search data from
    pub groups: Vec<String>,
    /// When to start searching data at
    pub start: Option<DateTime<Utc>>,
    /// When to stop searching data at
    pub end: Option<DateTime<Utc>>,
    /// The cursor id to use if one exists
    pub cursor: Option<Uuid>,
    /// The max number of objects to retrieve on a single page
    pub page_size: usize,
    /// The max number of items to return with this cursor
    pub limit: Option<usize>,
}

impl ElasticSearchOpts {
    /// Create a new search query
    ///
    /// # Arguments
    ///
    /// * `query` - The search query to use
    pub fn new<T: Into<String>>(query: T) -> Self {
        // build a base search options struct
        ElasticSearchOpts {
            indexes: default_search_indexes(),
            query: query.into(),
            groups: Vec::default(),
            start: None,
            end: None,
            cursor: None,
            page_size: 50,
            limit: None,
        }
    }

    /// Set the indexes to search on
    ///
    /// # Arguments
    ///
    /// * `indexes` - The indexes to search on
    #[must_use]
    pub fn indexes(mut self, indexes: Vec<ElasticIndex>) -> Self {
        self.indexes = indexes;
        self
    }
}

// A specific document in elastic
#[derive(Deserialize, Serialize, Default, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ElasticDoc {
    /// The id for this document
    #[serde(alias = "_id")]
    pub id: String,
    /// The index this doc cames from
    #[serde(alias = "_index")]
    pub index: String,
    /// The score for this doc
    #[serde(alias = "_score")]
    pub score: Option<f64>,
    /// The actual document in elastic
    #[serde(alias = "_source")]
    pub source: Option<serde_json::Value>,
    /// The actual document in elastic
    pub highlight: Option<serde_json::Value>,
    /// The sort values for this document
    pub sort: Vec<i64>,
}
