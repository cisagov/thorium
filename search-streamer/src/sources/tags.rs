//! Add support for streaming tags to a search store

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use futures::StreamExt;
use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use serde_json::{Value, json};
use thorium::client::TagSearchEvents;
use thorium::models::{ElasticIndex, TagRow, TagSearchEvent, TagType};
use thorium::{Error, Thorium};
use tracing::instrument;

use super::DataSource;
use crate::events::CompactTagEvent;
use crate::index::{IndexMapping, IndexTyped};
use crate::stores::{Elastic, StoreIdentifiable, StoreLookup};

mod scylla_utils;

use scylla_utils::{TagEnumerateRow, TagEventRow, TagsPrepared};

#[derive(Clone)]
pub struct Tags {
    prepared: TagsPrepared,
}

#[async_trait::async_trait]
impl DataSource for Tags {
    const DATA_NAME: &'static str = "Tags";

    type DataBundle = TagBundle;

    type IndexType = TagType;

    type InitRow = TagEnumerateRow;

    type InitInfo = TagInfo;

    type Event = TagSearchEvent;

    type CompactEvent = CompactTagEvent;

    type EventClient = TagSearchEvents;

    async fn new(scylla: &Session, ns: &str) -> Result<Self, Error> {
        // create prepared statements
        let prepared = TagsPrepared::prepare(scylla, ns).await?;
        Ok(Self { prepared })
    }

    fn event_client(thorium: &Thorium) -> &Self::EventClient {
        &thorium.search.events.tags
    }

    fn enumerate_prepared(&self) -> &PreparedStatement {
        &self.prepared.enumerate
    }

    #[instrument(
        name = "DataSource<SampleTags>::to_value",
        skip(bundles, now),
        err(Debug)
    )]
    fn to_values(
        bundles: &[TagBundle],
        data_type: &TagType,
        now: DateTime<Utc>,
    ) -> Result<Vec<Value>, Error> {
        let item_label = match data_type {
            TagType::Files => "sha256",
            TagType::Repos => "url",
        };
        Ok(bundles.iter().fold(Vec::new(), |mut values, bundle| {
            let tag_pairs: Vec<String> =
                bundle
                    .tags
                    .iter()
                    .fold(Vec::new(), |mut tags, (key, values)| {
                        tags.extend(values.iter().map(|value| format!("{key}={value}")));
                        tags
                    });
            // convert the bundle into a store id, then
            values.push(json!({"index": {"_id": bundle.as_store_id().to_string() }}));
            values.push(json!({
                item_label: &bundle.item,
                "streamed": &now,
                "group": &bundle.group,
                "tags": tag_pairs,
            }));
            values
        }))
    }

    async fn bundle_init(
        &self,
        info: Vec<TagInfo>,
        scylla: &Session,
    ) -> Result<Vec<(TagType, Vec<TagBundle>)>, Error> {
        let (file_items, repo_items): (Vec<_>, Vec<_>) =
            info.into_iter().partition(|i| i.kind == TagType::Files);
        let file_items = file_items.into_iter().map(|i| i.item).collect::<Vec<_>>();
        let repo_items = repo_items.into_iter().map(|i| i.item).collect::<Vec<_>>();
        let (file_bundles, repo_bundles) = tokio::try_join!(
            self.pull(&TagType::Files, &file_items, scylla),
            self.pull(&TagType::Repos, &repo_items, scylla),
        )?;
        Ok(vec![
            (TagType::Files, file_bundles),
            (TagType::Repos, repo_bundles),
        ])
    }

    async fn bundle_event(
        &self,
        compacted_event: CompactTagEvent,
        scylla: &Session,
    ) -> Result<Vec<TagBundle>, Error> {
        self.pull_event(compacted_event, scylla).await
    }
}

/// An item+group combo mapped to its tags which themselves are a map of keys to sets of values
type TagMap = HashMap<(String, String), HashMap<String, HashSet<String>>>;

impl Tags {
    /// Pull data for the given type/items and bundle them together for streaming
    ///
    /// # Arguments
    ///
    /// * `tag_type` - The type of item we're pulling for
    /// * `items` - The items to pull
    /// * `scylla` - The scylla client
    #[instrument(name = "sources::tags::Tags::pull", skip_all, err(Debug))]
    async fn pull(
        &self,
        tag_type: &TagType,
        items: &[String],
        scylla: &Session,
    ) -> Result<Vec<TagBundle>, Error> {
        // create a map of item/group to tags
        let mut tag_map = TagMap::new();
        // chunk into groups of 100
        for items_chunk in items.chunks(100) {
            // get init rows
            let resp = scylla
                .execute_iter(self.prepared.init.clone(), (tag_type, items_chunk))
                .await
                .map_err(|err| Error::new(format!("Error pulling tag init info: {err}")))?;
            // cast to tag rows
            let mut typed_stream = resp.rows_stream::<TagRow>().unwrap();
            while let Some(row) = typed_stream.next().await {
                // make sure the row was successful
                let row = row.unwrap();
                // add the row to the map
                let tags = tag_map
                    .entry((row.item.clone(), row.group.clone()))
                    .or_default();
                tags.entry(row.key).or_default().insert(row.value);
            }
        }
        Ok(tag_map
            .into_iter()
            .map(|((item, group), tags)| TagBundle { item, group, tags })
            .collect())
    }

    /// Pull data for the given event and bundle it together for streaming
    ///
    /// # Arguments
    ///
    /// * `compacted_event` - The compacted event (possibly comprising of multiple events) to pull data for
    /// * `scylla` - The scylla client
    #[instrument(name = "sources::tags::Tags::pull", skip_all, err(Debug))]
    async fn pull_event(
        &self,
        compacted_event: CompactTagEvent,
        scylla: &Session,
    ) -> Result<Vec<TagBundle>, Error> {
        // create a map of item/group to tags
        let mut tag_map = TagMap::with_capacity(compacted_event.groups.len());
        // convert our groups to a contiguous Vec
        let groups = compacted_event.groups.into_iter().collect::<Vec<_>>();
        // chunk groups by 100
        for groups_chunk in groups.chunks(100) {
            let resp = scylla
                .execute_iter(
                    self.prepared.event.clone(),
                    (
                        &compacted_event.tag_type,
                        &compacted_event.item,
                        groups_chunk,
                    ),
                )
                .await
                .map_err(|err| Error::new(format!("Error pulling tag event info: {err}")))?;
            // cast to row
            let mut typed_stream = resp.rows_stream::<TagEventRow>().unwrap();
            while let Some(row) = typed_stream.next().await {
                // make sure the row was successful
                let row = row.unwrap();
                // add row to our map
                let tags = tag_map
                    .entry((compacted_event.item.clone(), row.group.clone()))
                    .or_default();
                tags.entry(row.key).or_default().insert(row.value);
            }
        }
        Ok(tag_map
            .into_iter()
            .map(|((item, group), tags)| TagBundle { item, group, tags })
            .collect())
    }
}

/// The info required to pull a tag bundle for a single item
pub struct TagInfo {
    /// The kind of tag that was pulled
    pub kind: TagType,
    /// That tag item
    pub item: String,
}

impl From<TagEnumerateRow> for TagInfo {
    fn from(row: TagEnumerateRow) -> Self {
        Self {
            kind: row.kind,
            item: row.item,
        }
    }
}

impl From<TagSearchEvent> for TagInfo {
    fn from(event: TagSearchEvent) -> Self {
        Self {
            kind: event.tag_type,
            item: event.item,
        }
    }
}

/// A bundle of tags for a given item
#[derive(Debug)]
pub struct TagBundle {
    /// The item these tags pertain to
    pub item: String,
    /// The group these tags are in
    pub group: String,
    /// The tags themselves
    pub tags: HashMap<String, HashSet<String>>,
}

/// A unique id to a tag document in the search store
pub struct TagStoreId<'a> {
    /// The item the document refers to
    item: &'a String,
    /// The group the item is in
    group: &'a String,
}

impl std::fmt::Display for TagStoreId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.item, self.group)
    }
}

impl<'a> StoreIdentifiable<'a> for TagBundle {
    type Id = TagStoreId<'a>;

    fn as_store_id(&'a self) -> Self::Id {
        TagStoreId {
            item: &self.item,
            group: &self.group,
        }
    }
}

impl IndexMapping<Elastic> for TagType {
    fn all_indexes() -> Vec<ElasticIndex> {
        vec![ElasticIndex::SampleTags, ElasticIndex::RepoTags]
    }

    fn map_index(&self) -> ElasticIndex {
        match self {
            TagType::Files => ElasticIndex::SampleTags,
            TagType::Repos => ElasticIndex::RepoTags,
        }
    }
}

impl IndexTyped for CompactTagEvent {
    type IndexType = TagType;

    fn index_type(&self) -> Self::IndexType {
        self.tag_type
    }
}

impl<'a> StoreLookup<'a> for CompactTagEvent {
    type Id = TagStoreId<'a>;

    fn store_ids(&'a self) -> Vec<Self::Id> {
        // return all the combos of item+group for each group
        self.groups
            .iter()
            .map(|group| TagStoreId {
                item: &self.item,
                group,
            })
            .collect()
    }
}
