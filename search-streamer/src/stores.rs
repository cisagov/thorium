//! The different search stores to stream data too

use serde_json::Value;
use thorium::{Conf, Error};

mod elastic;

pub use elastic::Elastic;

#[async_trait::async_trait]
pub trait SearchStore: Clone + Sync + Send + 'static + Sized {
    /// The name of this search store
    const STORE_NAME: &'static str;

    /// The index type to use in the search store
    type Index: Send;

    /// Create a new search store client
    ///
    /// # Arguments
    ///
    /// * `conf` - A Thorium config
    /// * `index` - The index to send docs too
    fn new(conf: &Conf) -> Result<Self, Error>;

    /// Initiate the search store in case it hasn't been already
    ///
    /// # Arguments
    ///
    /// * `indexes` - The indexes to initiate
    /// * `reindex` - Whether we should force a reindex, whether or not
    ///               indexes already exist
    ///
    /// # Returns
    ///
    /// Returns true if the store did not already exist and was initiated
    /// in this function
    ///
    /// # Caveat
    ///
    /// Because a search store may reference multiple indexes, `init` will return
    /// if *any* of the indexes it's responsible for are unintialized, and all of
    /// the indexes with then be initialized by the `search-streamer`. No data will
    /// be lost if one of the indexes already existed, but data may be overridden
    /// to the current state of data in Scylla.
    async fn init(&self, indexes: &[Self::Index], reindex: bool) -> Result<bool, Error>;

    /// Create documents in our search store to be indexed
    ///
    /// # Arguments
    ///
    /// * `index` - The index to create the documents in
    /// * `values` - The JSON values to send
    async fn create(&self, index: Self::Index, values: Vec<Value>) -> Result<(), Error>;

    /// Delete documents from the search store
    ///
    /// # Arguments
    ///
    /// * `index` - The index to delete the document from
    /// * `store_ids` - The IDs of the documents in the store to delete
    async fn delete(&self, index: Self::Index, store_ids: &[String]) -> Result<(), Error>;
}

/// Describes a type that can produce a unique id to itself in the search store
///
/// Unlike [`StoreLookup`], the implementor itself is stored in the
/// search store along with its data
pub trait StoreIdentifiable<'a> {
    /// The id type containing the components of the id string
    type Id: 'a + std::fmt::Display;

    /// Get a unique id from `self` to refer to oneself in the search store
    fn as_store_id(&'a self) -> Self::Id;
}

/// Describes a type that can produce multiple unique id's referencing
/// documents in a search store
///
/// Unlike [`StoreIdentifiable`], the implementor is not itself stored in
/// the search store but instead references one or more documents in the store
pub trait StoreLookup<'a> {
    /// The id type containing the components of the id string
    type Id: 'a + std::fmt::Display;

    /// Get a list of one or more IDs that `self` is referring to
    fn store_ids(&'a self) -> Vec<Self::Id>;
}
