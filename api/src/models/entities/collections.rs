//! Contains models for collections

use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};
use strum::{AsRefStr, EnumString};

#[cfg(feature = "client")]
use crate::multipart_set;

/// A dynamic collection of items in Thorium defined by various parameters
/// each item in the collection must
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CollectionEntity {
    /// The kind of collection this is
    pub collection_kind: CollectionKind,
    /// A set of tags of which each item in a collection must have *at least one*
    #[serde(default)]
    pub collection_tags: HashMap<String, HashSet<String>>,
    /// Whether tag matching for this collection should be case-insensitive
    #[serde(default)]
    pub tags_case_insensitive: bool,
    /// Whether the collection should include items in all of the user's groups rather than
    /// filtering on the collection's groups
    #[serde(default)]
    pub ignore_groups: bool,
    /// When to start listing data for the collection
    ///
    /// If not set, the collection starts at the current
    /// point in time whenever the collection is viewed
    pub start: Option<DateTime<Utc>>,
    /// When to stop listing data for the collection
    ///
    /// If not set, the collection has no end point in time
    pub end: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, EnumString, strum::Display, AsRefStr)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum CollectionKind {
    /// A collection of files
    Files,
    /// A collection of repos
    Repos,
}

/// A request to create a collection entity
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CollectionEntityRequest {
    /// The kind of collection to create
    pub collection_kind: CollectionKind,
    /// The tags defining this collection
    #[serde(default)]
    pub collection_tags: HashMap<String, HashSet<String>>,
    /// Whether tag matching for this collection should be case-insensitive
    #[serde(default)]
    pub tags_case_insensitive: Option<bool>,
    /// Whether the collection should include items in all of the user's groups rather than
    /// filtering on the collection's groups
    #[serde(default)]
    pub ignore_groups: Option<bool>,
    /// When to start listing data for the collection
    ///
    /// If not set, the collection starts at the current
    /// point in time whenever the collection is viewed
    #[serde(default)]
    pub start: Option<DateTime<Utc>>,
    /// When to stop listing data for the collection
    ///
    /// If not set, the collection has no end point in time
    #[serde(default)]
    pub end: Option<DateTime<Utc>>,
}

impl CollectionEntityRequest {
    /// Add this collection entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        mut form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // set the entity kind
        form = form.text("kind", super::EntityKinds::Collection.as_str());
        // add our collection metadata
        form = form.text(
            "metadata[collection_kind]",
            self.collection_kind.to_string(),
        );
        if let Some(start) = self.start {
            form = form.text("metadata[collection_start]", start.to_rfc3339());
        }
        if let Some(end) = self.end {
            form = form.text("metadata[collection_end]", end.to_rfc3339());
        }
        if let Some(tags_case_insensitive) = self.tags_case_insensitive {
            form = form.text(
                "metadata[collection_tags_case_insensitive]",
                tags_case_insensitive.to_string(),
            );
        }
        if let Some(ignore_groups) = self.ignore_groups {
            form = form.text(
                "metadata[collection_ignore_groups]",
                ignore_groups.to_string(),
            );
        }
        // add collection tags to this form
        for (key, mut values) in self.collection_tags {
            // build the tag key for this and_tag
            let tag_key = format!("metadata[collection_tags][{key}][]");
            // add this tags list of values to our form
            form = multipart_set!(form, &tag_key, values);
        }
        Ok(form)
    }
}
