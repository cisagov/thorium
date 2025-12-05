//! Structures related to Thorium "Entities"

use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use gxhash::GxHasher;
use std::collections::{BTreeSet, HashSet};
use std::hash::Hasher;
use strum::{AsRefStr, EnumDiscriminants, EnumIter, EnumString, IntoEnumIterator};
use uuid::Uuid;

use super::Association;
use crate::models::{
    Country, DeviceEntityRequest, TagMap, TreeBranch, TreeSupport, VendorEntity,
    VendorEntityRequest,
};

pub mod countries;
pub mod devices;
pub mod shared;
pub mod vendors;

use devices::DeviceEntity;
use shared::CriticalSector;

#[cfg(feature = "scylla-utils")]
use std::str::FromStr;

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use super::{TagRequest, User, TagDeleteRequest, Group, GroupAllowAction, UnhashedTreeBranch};
        use crate::utils::{ApiError, Shared};
        use chrono::TimeZone;
        use crate::models::Tree;
        use itertools::Itertools;
    }
}

// api/client imports
cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        use crate::models::scylla_utils::keys::KeySupport;
        use super::backends::TagSupport;
        use super::TagType;
        use std::collections::HashMap;
        use crate::{multipart_list, multipart_text, multipart_set};
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {

        /// The form for entity metadata
        #[derive(Debug, Default)]
        pub struct EntityMetadataForm {
            pub urls: Vec<String>,
            pub vendors: Vec<Uuid>,
            pub critical_system: Option<bool>,
            pub sensitive_location: Option<bool>,
            pub critical_sectors: BTreeSet<CriticalSector>,
            pub countries: BTreeSet<Country>,
        }

        /// A request to create a new entity
        #[derive(Debug, Default)]
        pub struct EntityForm {
            /// The entity's name
            pub name: Option<String>,
            /// The kind of entity this is
            pub kind: Option<EntityKinds>,
            /// The metadata for this specific entity kind
            pub metadata: EntityMetadataForm,
            /// The groups this entity should be in
            pub groups: Vec<String>,
            /// The tags for this entity
            pub tags: HashMap<String, HashSet<String>>,
            /// A description of this entity
            pub description: Option<String>,
            /// This entities image
            pub image: Option<String>,
        }

        /// Fields from the multipart form for updating an entity
        #[derive(Debug, Default)]
        pub struct EntityUpdateForm {
            pub name: Option<String>,
            pub metadata: EntityMetadataUpdateForm,
            pub add_groups: Vec<String>,
            pub remove_groups: Vec<String>,
            pub clear_image: Option<bool>,
            /// A description of this entity
            pub description: Option<String>,
            pub clear_description: Option<bool>
        }

        /// The form for updating entity metadata
        #[derive(Debug, Default)]
        pub struct EntityMetadataUpdateForm {
            pub add_urls: Vec<String>,
            pub remove_urls: Vec<String>,
            pub add_vendors: Vec<Uuid>,
            pub remove_vendors: Vec<Uuid>,
            pub critical_system: Option<bool>,
            pub clear_critical_system: Option<bool>,
            pub sensitive_location: Option<bool>,
            pub clear_sensitive_location: Option<bool>,
            pub add_critical_sectors: Vec<CriticalSector>,
            pub remove_critical_sectors: Vec<CriticalSector>,
            pub add_countries: Vec<Country>,
            pub remove_countries: Vec<Country>,

        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
// TODO schema example
pub struct Entity {
    /// The entity's unique ID
    pub id: Uuid,
    /// The name of this entity
    pub name: String,
    /// The kind of entity this is
    pub kind: EntityKinds,
    /// The metadata for our specific entity
    pub metadata: EntityMetadata,
    /// A description of this entity
    pub description: Option<String>,
    /// The user that submitted this entity
    pub submitter: String,
    /// The groups this entity was submitted too
    pub groups: Vec<String>,
    /// The tags for this entity
    pub tags: TagMap,
    /// This entities image
    pub image: Option<String>,
    /// The time this entity was created
    pub created: DateTime<Utc>,
}

impl TreeSupport for Entity {
    /// The data used to generate this types tree hash
    type HashType<'a> = &'a Uuid;

    /// Hash this child object
    fn tree_hash(&self) -> u64 {
        Self::tree_hash_direct(&self.id)
    }

    /// Hash this child object
    ///
    /// # Arguments
    ///
    /// * `input` - The data needed to generate this nodes tree hash
    /// * `hasher` - The hasher to write data to
    fn tree_hash_direct_with_hasher(input: Self::HashType<'_>, hasher: &mut GxHasher) {
        // hash this samples sha
        hasher.write_u128(input.as_u128());
    }

    /// Gather any initial nodes for a tree
    #[cfg(feature = "api")]
    async fn gather_initial(
        user: &User,
        query: &crate::models::TreeQuery,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<super::TreeNode>, crate::utils::ApiError> {
        // get info on all of the initial entities
        let entities =
            crate::models::backends::db::entities::get_many(&query.groups, &query.entities, shared)
                .await?;
        // build a list of initial data
        let mut initial = Vec::with_capacity(query.entities.len());
        // step over the entities we retrieved and add them to our tree
        for entity in entities {
            // wrap this entity in a tree node
            let node_data = super::TreeNode::Entity(entity);
            // add this tree node to our initial list
            initial.push(node_data);
        }
        Ok(initial)
    }

    /// Gather any children for this child node
    #[cfg(feature = "api")]
    async fn gather_children(
        &self,
        user: &User,
        tree: &Tree,
        ring: &crate::models::backends::trees::TreeRing,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // get a cursor for this entities associations
        if let Some(mut cursor) = self.list_associations(shared).await? {
            // get this entities tree hash
            let source_hash = self.tree_hash();
            // crawl through this entities associations and add them to the tree
            loop {
                // convert all of the listable associations to full associations
                let associations = cursor
                    .data
                    .drain(..)
                    .map(Association::try_from)
                    .collect::<Result<Vec<Association>, ApiError>>()?;
                // make sure we don't get the same target node multiple times
                let filtered_targets = associations
                    .iter()
                    .filter(|assoc| !ring.contains(tree, assoc.tree_hash()))
                    .unique_by(|assoc| assoc.tree_hash())
                    .map(Clone::clone)
                    .collect::<Vec<Association>>();
                // get this pages tree nodes in parallel
                let mut node_stream = stream::iter(filtered_targets)
                    .map(|assoc| async move { assoc.get_tree_node(user, shared).await })
                    .buffer_unordered(10);
                // add our tree nodes as we get them
                while let Some(node_result) = node_stream.next().await {
                    // if we failed to get this node then raise an error
                    let node = node_result?;
                    // add this node
                    ring.add_node(node);
                }
                // get an entry to our parent nodes relationships
                let entry = ring.relationships.entry(source_hash).or_default();
                // build the branches for these associations
                for association in associations {
                    // get the tree hash for what this association points too
                    let target_hash = association.tree_hash();
                    // get this associations direction
                    let direction = association.direction;
                    // build the relationship for this branch
                    let relationship = crate::models::TreeRelationships::Association(association);
                    // wrap our relationship in a branch
                    let branch = UnhashedTreeBranch::new(target_hash, relationship, direction);
                    // insert our relationship
                    entry.insert(branch);
                }
                // if our cursor is exhausted then stop crawling
                if cursor.exhausted() {
                    break;
                }
                // get the next page of data
                cursor.next(shared).await?;
            }
        }
        // entities only use associations for children
        Ok(())
    }

    /// Build an association target column for an object
    #[cfg(feature = "api")]
    fn build_association_target_column(&self) -> Option<super::AssociationTargetColumn> {
        // build a target for this entity
        let target = super::AssociationTargetColumn::Entity(self.id);
        Some(target)
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl KeySupport for Entity {
    // the entity's ID
    // doesn't work for uuid with Utoipa even with the uuid feature flag since this is a generic
    // https://github.com/juhaku/utoipa/issues/1346
    type Key = String;

    type ExtraKey = ();

    fn build_key(key: Self::Key, _extra: &Self::ExtraKey) -> String {
        key
    }

    fn key_url(key: &Self::Key, _extra: Option<&Self::ExtraKey>) -> String {
        key.to_owned()
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl TagSupport for Entity {
    /// Get the tag kind to write to the DB
    fn tag_kind() -> TagType {
        TagType::Entities
    }

    fn earliest(&self) -> HashMap<&String, DateTime<Utc>> {
        // instance a map for the earliest time each group has seen this entity
        let mut earliest = HashMap::with_capacity(self.groups.len());
        // for entities all groups wil always have the same timestamp for now
        for group in &self.groups {
            earliest.insert(group, self.created);
        }
        earliest
    }

    /// Add some tags to an entity
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating tags
    /// * `req` - The tag request to apply
    /// * `shared` - Shared Thorium objects
    #[tracing::instrument(
        name = "TagSupport<Entity>::tag",
        skip_all,
        fields(name = self.name, id = self.id.to_string()),
        err(Debug))
    ]
    #[cfg(feature = "api")]
    async fn tag(
        &self,
        user: &User,
        mut req: TagRequest<Self>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // make sure we have edit permissions in all groups and that
        // all groups allow for entities
        self.validate_check_allow_groups(
            user,
            &mut req.groups,
            Group::editable,
            "edit",
            Some(GroupAllowAction::Entities),
            shared,
        )
        .await?;
        // get the earliest for each group (just the time the entity was created)
        let earliest = self.earliest();
        let key = Self::build_key(self.id.to_string(), &());
        // save the tags to scylla
        super::backends::db::tags::create(user, key, req, &earliest, shared).await
    }

    /// Delete some tags from this entity
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting tags
    /// * `req` - The tags to delete
    /// * `shared` - Shared Thorium objects
    #[tracing::instrument(
        name = "TagSupport<Entity>::delete_tags",
        skip_all,
        fields(name = self.name, id = self.id.to_string()),
        err(Debug))
    ]
    #[cfg(feature = "api")]
    async fn delete_tags(
        &self,
        user: &User,
        mut req: TagDeleteRequest<Self>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // make sure we have edit permissions in all groups;
        // no need to check for the group action as deleting
        // is always allowed
        self.validate_check_allow_groups(
            user,
            &mut req.groups,
            Group::editable,
            "edit",
            None,
            shared,
        )
        .await?;
        // build our key
        let key = Self::build_key(self.id.to_string(), &());
        // delete the requested tags if they exist
        super::backends::db::tags::delete(&key, &req, shared).await
    }

    /// Gets tags for a specific entity
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our returned tags to
    /// * `shared` - Shared Thorium objects
    #[tracing::instrument(
        name = "TagSupport<Entity>::get_tags",
        skip_all,
        fields(name = self.name, id = self.id.to_string()),
        err(Debug))
    ]
    #[cfg(feature = "api")]
    async fn get_tags(&mut self, groups: &[String], shared: &Shared) -> Result<(), ApiError> {
        // build our key
        let key = Self::build_key(self.id.to_string(), &());
        // get the requested tags
        super::backends::db::tags::get(TagType::Entities, groups, &key, &mut self.tags, shared)
            .await
    }
}

/// The specific kind an entity is, including any data unique to its kind
#[derive(Debug, Clone, Serialize, Deserialize, EnumDiscriminants)]
// generate type just containing the entity kind's name with no data
#[strum_discriminants(name(EntityKinds))]
#[strum_discriminants(derive(
    Default,
    Serialize,
    Deserialize,
    AsRefStr,
    EnumString,
    EnumIter,
    strum::Display
))]
#[cfg_attr(
    feature = "scylla-utils",
    strum_discriminants(derive(thorium_derive::ScyllaStoreJson))
)]
#[cfg_attr(feature = "api", strum_discriminants(derive(utoipa::ToSchema)))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum EntityMetadata {
    /// A device entity
    Device(DeviceEntity),
    /// A vendor entity
    Vendor(VendorEntity),
    /// An entity that can't be described by any of the other variants
    #[strum_discriminants(default)]
    Other,
}

/// The specific kind an entity is, including any data unique to its kind
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum EntityMetadataRequest {
    /// A device entity
    Device(DeviceEntityRequest),
    /// A vendor entity
    Vendor(VendorEntityRequest),
    /// An entity that can't be described by any of the other variants
    Other,
}

impl EntityMetadataRequest {
    /// Add this entity metadata to a form
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // add our metadata
        match self {
            EntityMetadataRequest::Device(device) => device.add_to_form(form),
            EntityMetadataRequest::Vendor(vendor) => vendor.add_to_form(form),
            // just set our kind to other
            EntityMetadataRequest::Other => Ok(form.text("kind", EntityKinds::Other.as_str())),
        }
    }
}

impl EntityKinds {
    /// Gets a str representation of the entity kind name
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

/// A request to create an entity
pub struct EntityRequest {
    /// The entity's name
    pub name: String,
    /// The metadata for a specific kind of entity
    pub metadata: EntityMetadataRequest,
    /// The groups this entity should be in
    pub groups: Vec<String>,
    /// The tags for this entity
    pub tags: HashMap<String, HashSet<String>>,
    /// A description of this entity
    pub description: Option<String>,
}

impl EntityRequest {
    /// Cast this entity request into a form
    #[cfg(feature = "client")]
    pub fn to_form(mut self) -> Result<reqwest::multipart::Form, crate::Error> {
        // build the form we are going to send
        // disable percent encoding, as the API natively supports UTF-8
        let form = reqwest::multipart::Form::new().percent_encode_noop();
        // add the name of this entity
        let form = form.text("name", self.name);
        // add our entity metadata
        let form = self.metadata.add_to_form(form)?;
        // add our groups
        let mut form = multipart_list!(form, "groups[]", self.groups);
        // add any tags to this form
        for (key, mut values) in self.tags {
            // build the tag key to for this tag
            let tag_key = format!("tags[{key}]");
            // add this tags list of values to our form
            form = multipart_set!(form, &tag_key, values);
        }
        // add our description to this requet
        let form = multipart_text!(form, "description", self.description);
        Ok(form)
    }
}

/// The response from an entity creation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EntityResponse {
    /// The ID of the created entity
    pub id: Uuid,
}

impl EntityResponse {
    /// Create a new entity response
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the created entity
    #[must_use]
    pub fn new(id: Uuid) -> Self {
        Self { id }
    }
}

/// Set default for the entity list limit
fn default_list_limit() -> usize {
    50
}

/// Set the default for listing entities by kind
fn default_entity_kinds() -> Vec<EntityKinds> {
    // list all entities by default
    EntityKinds::iter().collect()
}

/// The params for listing entities
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EntityListParams {
    /// The groups to list data from
    #[serde(default)]
    pub groups: Vec<String>,
    /// When to start listing data at
    #[serde(default = "Utc::now")]
    pub start: DateTime<Utc>,
    /// When to stop listing data at
    pub end: Option<DateTime<Utc>>,
    /// The tags to filter on
    #[serde(default)]
    pub tags: HashMap<String, Vec<String>>,
    /// The cursor id to use if one exists
    pub cursor: Option<Uuid>,
    /// The max number of items to return in this response
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default = "default_entity_kinds")]
    pub kinds: Vec<EntityKinds>,
}

impl Default for EntityListParams {
    /// Create default entity list params
    fn default() -> Self {
        Self {
            groups: Vec::default(),
            start: Utc::now(),
            end: None,
            tags: HashMap::default(),
            cursor: None,
            limit: default_list_limit(),
            kinds: default_entity_kinds(),
        }
    }
}

impl EntityListParams {
    /// Get the end timestamp or get a sane default
    #[cfg(feature = "api")]
    pub fn end(&self, shared: &crate::utils::Shared) -> Result<DateTime<Utc>, ApiError> {
        match self.end {
            Some(end) => Ok(end),
            None => match Utc.timestamp_opt(shared.config.thorium.entities.earliest, 0) {
                chrono::LocalResult::Single(default_end) => Ok(default_end),
                _ => crate::internal_err!(format!(
                    "default earliest repos timestamp is invalid or ambigous - {}",
                    shared.config.thorium.entities.earliest
                )),
            },
        }
    }
}

// A single entity line missing supplementary data like name and kind
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EntityListLine {
    /// The group this entity is apart of (used only for cursor generation)
    #[serde(skip_serializing, skip_deserializing)]
    pub groups: HashSet<String>,
    /// The entity's unique ID
    pub id: Uuid,
    /// The entity's name
    pub name: String,
    /// The kind of entity this is (without the kind's data)
    pub kind: EntityKinds,
    /// The time this entity was created
    pub created: DateTime<Utc>,
}

/// An update to apply to an entity
#[derive(Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EntityUpdate {
    pub name: Option<String>,
    pub add_groups: Vec<String>,
    pub remove_groups: Vec<String>,
    pub description: Option<String>,
    pub clear_description: bool,
}

impl EntityUpdate {
    /// Convert this update to a multipart form
    #[cfg(feature = "client")]
    pub fn to_form(mut self) -> Result<reqwest::multipart::Form, crate::Error> {
        // build a form object to add our form data too
        let form = reqwest::multipart::Form::new()
            // disable percent encoding, as the API natively supports UTF-8
            .percent_encode_noop()
            // always set our clear description field
            .text("clear_description", self.clear_description.to_string());
        // set our name form field
        let form = multipart_text!(form, "name", self.name);
        // add the groups to add/remove to this form
        let form = multipart_list!(form, "add_groups", self.add_groups);
        let form = multipart_list!(form, "remove_groups", self.remove_groups);
        // set our description form field
        let form = multipart_text!(form, "description", self.description);
        Ok(form)
    }
}
