//! Contains logic for search indexes

use crate::stores::SearchStore;

/// Describes a type that can map to an index of the search store
pub trait IndexMapping<S: SearchStore> {
    /// Return a list of all possible indexes the implementor may refer to
    fn all_indexes() -> Vec<S::Index>;

    /// Map a specific instance of the implementor to its index
    fn map_index(&self) -> S::Index;
}

/// Describes a type that contains a type that can map to an index
pub trait IndexTyped {
    /// The index type that can be returned
    ///
    /// Should implement `Copy`, as the type should be a simple enum and
    /// could be able to return a near zero-cost copy of itself
    type IndexType: Copy;

    /// Get the index type from the implementor
    fn index_type(&self) -> Self::IndexType;
}
