//! The different sources of data to stream into a search store

use chrono::prelude::*;
use scylla::client::session::Session;
use scylla::deserialize::row::DeserializeRow;
use scylla::statement::prepared::PreparedStatement;
use serde_json::Value;
use thorium::client::SearchEventsClient;
use thorium::models::SearchEvent;
use thorium::{Error, Thorium};

mod results;
mod tags;

pub use results::Results;
pub use tags::Tags;

use crate::events::{CompactEvent, EventCompactable};
use crate::index::IndexTyped;
use crate::stores::{StoreIdentifiable, StoreLookup};

/// A source of data search streamer pulls from and streams into the search store
#[async_trait::async_trait]
pub trait DataSource: Send + Clone {
    /// The name of this data source
    const DATA_NAME: &'static str;

    /// The number of items to pull and send to the search store concurrently for
    /// this data type during the init phase
    const INIT_CONCURRENT: usize = 50;

    /// A single bundle of data that is streamed to the search store
    type DataBundle: for<'a> StoreIdentifiable<'a> + Send;

    /// The type defining which index to send data to
    type IndexType: Copy + Send;

    /// The Scylla row that is pulled to enumerate which data
    /// needs to be streamed
    type InitRow: 'static + for<'a, 'b> DeserializeRow<'a, 'b> + Send;

    /// Derived from [`Self::InitRow`] and necessary to pull a single bundle to
    /// stream during the init process
    type InitInfo: From<Self::InitRow> + Send;

    /// The events this data source is linked to to signal to the search streamer
    /// when and what to stream
    type Event: SearchEvent + EventCompactable<Self::CompactEvent> + Send;

    /// The compacted group of "like" [`Self::Event`] meant to avoid needlessly re-streaming
    /// data when items are modified multiple times in a short period
    type CompactEvent: CompactEvent<Self::Event>
        // must have a type that can be mapped to an index
        + IndexTyped<IndexType = Self::IndexType>
        // must be able to produce unique id's in the search store that are the same as
        // the main data bundle
        + for<'a> StoreLookup<'a, Id = <Self::DataBundle as StoreIdentifiable<'a>>::Id>
        + Send;

    /// The client that polls for events of type [`Self::Event`]
    type EventClient: SearchEventsClient<SearchEvent = Self::Event> + Clone + Send;

    /// Returns an instance of this data source
    ///
    /// # Arguments
    ///
    /// * `scylla` - The Scylla client
    /// * `ns` - The namespace the data is stored in
    async fn new(scylla: &Session, ns: &str) -> Result<Self, Error>;

    /// Retrieve a reference to the correct event sub-client from the Thorium client
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium main client
    fn event_client(thorium: &Thorium) -> &Self::EventClient;

    /// Return the enumerate prepared statement, using the Scylla `token()` function
    /// to get all streaming data spread across our partitions
    ///
    /// This prepared statement *must* pull items of type [`Self::InitRow`]
    fn enumerate_prepared(&self) -> &PreparedStatement;

    /// Cast this data to one or more serialized Json values
    ///
    /// # Arguments
    ///
    /// * `bundles` - The data bundles to serialize
    /// * `index_type` - The data's type to map to a specific index
    /// * `now` - The time this data was serialized/streamed
    fn to_values(
        bundles: &[Self::DataBundle],
        index_type: &Self::IndexType,
        now: DateTime<Utc>,
    ) -> Result<Vec<Value>, Error>;

    /// From a list of init info on items, pull bundles for each item and return
    /// a list of a list of bundles together with types able to map to which index
    /// they're supposed to be streamed to
    ///
    /// # Arguments
    ///
    /// * `info` - A list of info on items to pull bundles for
    /// * `scylla` - The Scylla client
    async fn bundle_init(
        &self,
        info: Vec<Self::InitInfo>,
        scylla: &Session,
    ) -> Result<Vec<(Self::IndexType, Vec<Self::DataBundle>)>, Error>;

    /// From an event, pull corresponding data bundles
    ///
    /// # Arguments
    ///
    /// * `compacted_event` - The compacted event (possibly comprising multiple events)
    ///                       triggering a data pull
    /// * `scylla` - The Scylla client
    async fn bundle_event(
        &self,
        compacted_event: Self::CompactEvent,
        scylla: &Session,
    ) -> Result<Vec<Self::DataBundle>, Error>;
}
