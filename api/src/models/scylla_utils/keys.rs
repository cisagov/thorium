//! The keys trait for identifying things in scylla

pub mod images;
pub mod pipelines;

pub use images::ImageKey;
pub use pipelines::PipelineKey;

use serde::{Deserialize, Serialize};

/// The keys trait for identifying things in scylla
pub trait KeySupport {
    // require scylla bounds only if scylla-utils is enabled
    cfg_if::cfg_if! {
        // do not enable the utoipa requirements if the api feature is not enabled
        if #[cfg(all(feature = "scylla-utils", not(feature = "api")))] {
            /// The unique key used to identify the thing in scylla
            type Key: Clone
                + Serialize
                + for<'d> Deserialize<'d>
                + PartialEq
                + std::fmt::Debug
                + scylla::serialize::value::SerializeValue
                + for<'frame, 'metadata> scylla::deserialize::value::DeserializeValue<'frame, 'metadata>;
        } else if #[cfg(all(feature = "api", feature = "scylla-utils"))] {
            /// The unique key used to identify the thing in scylla
            type Key: Clone
                + Serialize
                + for<'d> Deserialize<'d>
                + PartialEq
                + std::fmt::Debug
                + utoipa::ToSchema
                + scylla::serialize::value::SerializeValue
                + for<'frame, 'metadata> scylla::deserialize::value::DeserializeValue<'frame, 'metadata>;
        } else {
            /// The unique key used to identify the thing in scylla
            type Key: Clone
                + Serialize
                + for<'d> Deserialize<'d>
                + PartialEq
                + std::fmt::Debug;
        }
    }

    /// The extra info stored in some requests that gets added to our key
    type ExtraKey: Clone + Serialize + for<'d> Deserialize<'d>;

    /// Build the key to use as part of the partition key when storing this data in scylla
    ///
    /// # Arguments
    ///
    /// * `key` - The root part of this key
    /// * `extra` - Any extra info required to build this key
    fn build_key(key: Self::Key, extra: &Self::ExtraKey) -> String;

    /// Build a URL component composed of the key to access the resource
    /// from the API
    ///
    /// # Arguments
    ///
    /// * `key` - The root part of this key
    /// * `extra` - Any extra info required to build this key
    fn key_url(key: &Self::Key, extra: Option<&Self::ExtraKey>) -> String;
}
