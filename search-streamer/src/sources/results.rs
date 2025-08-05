//! Add support for streaming results to a search store

use std::collections::{HashMap, HashSet};

use chrono::prelude::*;
use futures::StreamExt;
use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use serde_json::{Value, json};
use thorium::client::ResultSearchEvents;
use thorium::models::{ElasticIndex, OutputKind, ResultSearchEvent};
use thorium::{Error, Thorium};
use tracing::{Level, event, instrument};
use uuid::Uuid;

use super::DataSource;
use crate::events::CompactResultEvent;
use crate::index::{IndexMapping, IndexTyped};
use crate::stores::{Elastic, StoreIdentifiable, StoreLookup};

mod scylla_utils;

use scylla_utils::{
    ResultEnumerateRow, ResultEventInfoRow, ResultInitDataRow, ResultRow, ResultsPrepared,
};

#[derive(Clone)]
pub struct Results {
    prepared: ResultsPrepared,
}

#[async_trait::async_trait]
impl DataSource for Results {
    const DATA_NAME: &'static str = "Results";

    // pull and send fewer results concurrently because results can
    // be *very* large; helps avoid timeouts and OOM errors
    const INIT_CONCURRENT: usize = 5;

    type DataBundle = ResultBundle;

    type IndexType = OutputKind;

    type InitRow = ResultEnumerateRow;

    type InitInfo = ResultInfo;

    type Event = ResultSearchEvent;

    type CompactEvent = CompactResultEvent;

    type EventClient = ResultSearchEvents;

    async fn new(scylla: &Session, ns: &str) -> Result<Self, Error> {
        let prepared = ResultsPrepared::prepare(scylla, ns).await?;
        Ok(Self { prepared })
    }

    fn event_client(thorium: &Thorium) -> &Self::EventClient {
        &thorium.search.events.results
    }

    fn enumerate_prepared(&self) -> &PreparedStatement {
        &self.prepared.enumerate
    }

    #[instrument(
        name = "DataSource<SampleResults>::to_value",
        skip(data, now),
        err(Debug)
    )]
    fn to_values(
        data: &[ResultBundle],
        data_type: &OutputKind,
        now: DateTime<Utc>,
    ) -> Result<Vec<Value>, Error> {
        let item_label = match data_type {
            OutputKind::Files => "sha256",
            OutputKind::Repos => "url",
        };
        let values =
            data.iter()
                .try_fold(Vec::new(), |mut values, bundle| -> Result<_, Error> {
                    // get a store id from the bundle
                    values.push(json!({"index": {"_id": bundle.as_store_id().to_string() }}));
                    // build the final document to add and add it to our vec of docs
                    values.push(json!({
                        item_label: &bundle.item,
                        "streamed": &now,
                        "group": bundle.group,
                        "results": bundle.results,
                        "files": bundle.files,
                        "children": bundle.children
                    }));
                    Ok(values)
                })?;
        Ok(values)
    }

    #[instrument(name = "DataSource<Results>::bundle_init", skip_all, err(Debug))]
    async fn bundle_init(
        &self,
        info: Vec<ResultInfo>,
        scylla: &Session,
    ) -> Result<Vec<(OutputKind, Vec<ResultBundle>)>, Error> {
        // only pull if we actually have info
        if info.is_empty() {
            Ok(Vec::new())
        } else {
            // get IDs from the results data
            let keys = info.iter().map(|i| &i.key).collect::<Vec<_>>();
            // pull results info
            let results_data = self.pull_results_data(&keys, scylla).await?;
            // pull the actual results for files/repos
            let all_ids = results_data.all_ids();
            let results = self.pull_results(&all_ids, scylla).await?;
            // bundle the results together
            let file_bundles = bundle_results(results_data.files, &results);
            let repo_bundles = bundle_results(results_data.repos, &results);
            Ok(vec![
                (OutputKind::Files, file_bundles),
                (OutputKind::Repos, repo_bundles),
            ])
        }
    }

    #[instrument(name = "DataSource<Results>::bundle_event", skip_all, err(Debug))]
    async fn bundle_event(
        &self,
        compacted_event: CompactResultEvent,
        scylla: &Session,
    ) -> Result<Vec<ResultBundle>, Error> {
        // pull results data for the event
        let results_data = self
            .pull_results_data_event(compacted_event, scylla)
            .await?;
        // get IDs from the info
        let ids = results_data
            .values()
            .flat_map(|i| i.keys())
            .collect::<Vec<_>>();
        // pull the actual results
        let results = self.pull_results(&ids, scylla).await?;
        Ok(bundle_results(results_data, &results))
    }
}

/// A map of result id's to (result, files, children)
type ResultsMap = HashMap<Uuid, (String, Vec<String>, HashMap<String, Uuid>)>;

/// A map of a key to a map of its results to their groups
type ResultsDataMap = HashMap<String, HashMap<Uuid, HashSet<String>>>;

/// A bundle of data to stream to elastic
#[derive(Debug)]
pub struct ResultBundle {
    /// The sample/repo these results are for
    pub item: String,
    /// The group the item is in
    pub group: String,
    /// The results
    pub results: Vec<String>,
    /// Any files tied to the results
    pub files: Vec<String>,
    /// Any children found by this tool
    pub children: Vec<String>,
}

/// Contains results info for both files and repos
#[derive(Default)]
struct ResultsData {
    /// Info for files
    pub files: ResultsDataMap,
    /// Info for repos
    pub repos: ResultsDataMap,
}

impl ResultsData {
    /// Get a list of all result ID's we'll need when we're pulling the results
    fn all_ids(&self) -> Vec<&Uuid> {
        self.files
            .iter()
            .flat_map(|(_, info)| info.keys())
            .chain(self.repos.iter().flat_map(|(_, info)| info.keys()))
            .collect()
    }
}

/// Bundle together results from the data map and results map
///
/// # Arguments
///
/// * `results_data` - The data on the results
/// * `results_map` - A map of the results themselves
#[instrument(name = "sources::results::bundle_results", skip_all)]
fn bundle_results(results_data: ResultsDataMap, results_map: &ResultsMap) -> Vec<ResultBundle> {
    let mut bundles = Vec::new();
    for (item, data) in results_data {
        let (mut results, files, children) = data.keys().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut results, mut files, mut children), id| {
                if let Some((r, f, c)) = results_map.get(id) {
                    results.push(r.clone());
                    files.extend(f.iter().cloned());
                    children.extend(c.keys().cloned());
                } else {
                    // we're missing this result, so log an error but continue on
                    event!(Level::ERROR, "Missing result with id '{id}'");
                }
                (results, files, children)
            },
        );
        for (_, groups) in data {
            for group in groups {
                bundles.push(ResultBundle {
                    item: item.clone(),
                    group,
                    // TODO: don't copy results; use references instead
                    results: results.clone(),
                    files: files.clone(),
                    children: children.clone(),
                });
            }
        }
    }
    bundles
}

impl Results {
    /// Pull results for the given result IDs
    ///
    /// # Argument
    ///
    /// * `ids` - The IDs of results to pull
    /// * `scylla` - The scylla client
    #[instrument(name = "sources::results::Results::pull_results", skip_all, err(Debug))]
    async fn pull_results(&self, ids: &[&Uuid], scylla: &Session) -> Result<ResultsMap, Error> {
        let mut result_map = ResultsMap::with_capacity(ids.len());
        // chunk into groups of 100
        for ids_chunk in ids.chunks(100) {
            let resp = scylla
                .execute_iter(self.prepared.results.clone(), (ids_chunk,))
                .await
                .map_err(|err| Error::new(format!("Error pulling results: {err}")))?;
            // cast to rows
            let mut typed_stream = resp.rows_stream::<ResultRow>().unwrap();
            while let Some(row) = typed_stream.next().await {
                // check the row is valid
                let row = row.unwrap();
                result_map.insert(row.id, (row.result, row.files, row.children));
            }
        }
        Ok(result_map)
    }

    /// Pull result info for the given keys
    ///
    /// # Argument
    ///
    /// * `keys` - The keys of items to pull result info for
    /// * `scylla` - The scylla client
    #[instrument(
        name = "sources::results::Results::pull_results_data",
        skip_all,
        err(Debug)
    )]
    async fn pull_results_data(
        &self,
        keys: &[&String],
        scylla: &Session,
    ) -> Result<ResultsData, Error> {
        let mut results_data = ResultsData::default();
        // chunk into groups of 100
        for keys_chunk in keys.chunks(100) {
            // get info on results for the given keys
            let resp = scylla
                .execute_iter(self.prepared.init_data.clone(), (keys_chunk,))
                .await
                .map_err(|err| Error::new(format!("Error pulling results info: {err}")))?;
            let mut typed_stream = resp.rows_stream::<ResultInitDataRow>().unwrap();
            // organize the info by key
            while let Some(row) = typed_stream.next().await {
                // make sure the row is valid
                let row = row.unwrap();
                // get the right map based on the row's kind
                let map = match row.kind {
                    OutputKind::Files => &mut results_data.files,
                    OutputKind::Repos => &mut results_data.repos,
                };
                // save info from the row to the map
                let info = map.entry(row.key).or_default();
                info.entry(row.id).or_default().insert(row.group);
            }
        }
        Ok(results_data)
    }

    /// Pull result data for the given event
    ///
    /// # Argument
    ///
    /// * `event` - The event to pull data for
    /// * `scylla` - The scylla client
    #[instrument(
        name = "sources::results::Results::pull_results_data_event",
        skip_all,
        err(Debug)
    )]
    async fn pull_results_data_event(
        &self,
        compacted_event: CompactResultEvent,
        scylla: &Session,
    ) -> Result<ResultsDataMap, Error> {
        let mut results_info = ResultsDataMap::new();
        // get a contiguous Vec of groups from our compacted event
        let groups = compacted_event.groups.into_iter().collect::<Vec<_>>();
        // chunk groups by 100
        for groups_chunk in groups.chunks(100) {
            // get info on results for the given keys
            let resp = scylla
                .execute_iter(
                    self.prepared.event_data.clone(),
                    (
                        &compacted_event.item,
                        compacted_event.result_kind,
                        groups_chunk,
                    ),
                )
                .await
                .map_err(|err| Error::new(format!("Error pulling info on event: {err}")))?;
            let mut typed_stream = resp.rows_stream::<ResultEventInfoRow>().unwrap();
            // organize the info by key
            while let Some(row) = typed_stream.next().await {
                // make sure the row is valid
                let row = row.unwrap();
                // save info from the row to the map
                let info = results_info
                    .entry(compacted_event.item.clone())
                    .or_default();
                info.entry(row.id).or_default().insert(row.group);
            }
        }
        Ok(results_info)
    }
}

/// The info required to pull a result bundle for a single item
pub struct ResultInfo {
    pub key: String,
}

impl From<ResultEnumerateRow> for ResultInfo {
    fn from(row: ResultEnumerateRow) -> Self {
        Self { key: row.key }
    }
}

impl From<ResultSearchEvent> for ResultInfo {
    fn from(event: ResultSearchEvent) -> Self {
        Self { key: event.item }
    }
}

/// A unique id to a result document in the search store
pub struct ResultStoreId<'a> {
    /// The item the document refers to
    item: &'a String,
    /// The group the item is in
    group: &'a String,
}

impl std::fmt::Display for ResultStoreId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.item, self.group)
    }
}

impl<'a> StoreIdentifiable<'a> for ResultBundle {
    type Id = ResultStoreId<'a>;

    fn as_store_id(&'a self) -> Self::Id {
        ResultStoreId {
            item: &self.item,
            group: &self.group,
        }
    }
}

impl IndexMapping<Elastic> for OutputKind {
    fn all_indexes() -> Vec<ElasticIndex> {
        vec![ElasticIndex::SampleResults, ElasticIndex::RepoResults]
    }

    fn map_index(&self) -> ElasticIndex {
        match self {
            OutputKind::Files => ElasticIndex::SampleResults,
            OutputKind::Repos => ElasticIndex::RepoResults,
        }
    }
}

impl IndexTyped for CompactResultEvent {
    type IndexType = OutputKind;

    fn index_type(&self) -> Self::IndexType {
        self.result_kind
    }
}

impl<'a> StoreLookup<'a> for CompactResultEvent {
    type Id = ResultStoreId<'a>;

    fn store_ids(&'a self) -> Vec<Self::Id> {
        // return all the combos of item+group for each group
        self.groups
            .iter()
            .map(|group| ResultStoreId {
                item: &self.item,
                group,
            })
            .collect()
    }
}
