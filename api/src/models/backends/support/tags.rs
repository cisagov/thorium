//! Support for interacting with tags
use chrono::prelude::*;
use std::collections::HashMap;

use crate::models::{KeySupport, TagType};

// dependencies required for api
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use crate::models::{TagDeleteRequest, TagRequest, User};
        use crate::utils::{ApiError, Shared};
    }
}

/// Describes an entity that can be referred to by tag(s) and can
/// save tags to and delete tags from the database
pub trait TagSupport: KeySupport + Sized {
    /// Get the tag kind to write to the DB
    fn tag_kind() -> TagType;

    /// Get the earliest each group has seen this object
    fn earliest(&self) -> HashMap<&String, DateTime<Utc>>;

    /// Add some tags to an item
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating tags
    /// * `req` - The tag request to apply
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn tag(
        &self,
        user: &User,
        req: TagRequest<Self>,
        shared: &Shared,
    ) -> Result<(), ApiError>;

    /// Delete some tags from this item
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting tags
    /// * `req` - The tags to delete
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn delete_tags(
        &self,
        user: &User,
        req: TagDeleteRequest<Self>,
        shared: &Shared,
    ) -> Result<(), ApiError>;

    /// Gets tags for a specific item
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our returned tags too
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn get_tags(&mut self, groups: &Vec<String>, shared: &Shared) -> Result<(), ApiError>;
}
