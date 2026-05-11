//! Structures related to Thorium "Entities"

use chrono::{DateTime, Utc};
use gxhash::GxHasher;
use linearize::Linearize;
use schemars::JsonSchema;
use std::hash::Hasher;
use std::net::IpAddr;
use std::{collections::HashSet, hash::Hash};
use strum::{AsRefStr, EnumDiscriminants, EnumString, IntoEnumIterator};
use uuid::Uuid;

use crate::models::{
    CollectionEntity, CollectionEntityRequest, CollectionKind, Country, DeviceEntityRequest,
    TagMap, TreeSupport, VendorEntity, VendorEntityRequest,
};

pub mod collections;
pub mod countries;
pub mod devices;
pub mod filesystem;
pub mod flags;
pub mod network_activity;
pub mod processes;
pub mod rules;
pub mod shared;
pub mod vendors;

use devices::DeviceEntity;
use filesystem::{FileSystemEntity, FileSystemFolderEntity};
use flags::Confidence;
use flags::Flag;
use network_activity::NetworkConnection;
use processes::{WindowsProcessEntity, WindowsProcessTreeEntity};
use rules::{SigmaRule, SigmaRuleAppliesTo};

// api/client imports
cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        use std::collections::HashMap;

        use super::TagType;
        use super::backends::TagSupport;
        use crate::models::scylla_utils::keys::KeySupport;
        use crate::{multipart_list, multipart_text, multipart_set};
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use std::collections::BTreeSet;
        use futures::stream::{self, StreamExt};
        use chrono::TimeZone;

        use network_activity::{NetConState, TransportLayerProtocol};
        use rules::SigmaActionToTake;
        use shared::CriticalSector;
        use super::{TagRequest, User, TagDeleteRequest, Group, GroupAllowAction, UnhashedTreeBranch};
        use crate::utils::{ApiError, Shared};
        use crate::models::Tree;

        /// The form for entity metadata
        #[derive(Debug, Default)]
        pub struct EntityMetadataForm {
            pub urls: Vec<String>,
            pub vendors: Vec<Uuid>,
            pub critical_system: Option<bool>,
            pub sensitive_location: Option<bool>,
            pub critical_sectors: BTreeSet<CriticalSector>,
            pub countries: BTreeSet<Country>,
            pub collection_kind: Option<CollectionKind>,
            pub collection_tags: HashMap<String, HashSet<String>>,
            pub collection_tags_case_insensitive: Option<bool>,
            pub collection_ignore_groups: Option<bool>,
            pub collection_start: Option<DateTime<Utc>>,
            pub collection_end: Option<DateTime<Utc>>,
            pub sha256: Option<String>,
            pub filesystem_id: Option<Uuid>,
            pub names_sha256: Option<String>,
            pub data_sha256: Option<String>,
            pub all_sha256: Option<String>,
            pub tools: Vec<String>,
            pub pid: Option<u64>,
            pub parent_pid: Option<u64>,
            pub name: Option<String>,
            pub image_path: Option<String>,
            pub command: Option<String>,
            pub offset: Option<u64>,
            pub threads: Option<u32>,
            pub handles: Option<u32>,
            pub is_wow64: Option<bool>,
            pub session_id: Option<u32>,
            pub create_time: Option<DateTime<Utc>>,
            pub exit_time: Option<DateTime<Utc>>,
            pub protocol: Option<TransportLayerProtocol>,
            pub source: Option<IpAddr>,
            pub source_port: Option<u16>,
            pub destination: Option<IpAddr>,
            pub destination_port: Option<u16>,
            pub state: Option<NetConState>,
            pub process: Option<String>,
            /// A sigma rule in yaml format
            pub sigma_rule: Option<String>,
            /// What this sigma rule applies too
            pub sigma_applies_to: Vec<SigmaRuleAppliesTo>,
            /// The action to take when a sigma rule hits
            pub sigma_actions: Vec<SigmaActionToTake>,
            /// The score that a rule applies
            pub score: Option<i64>,
            /// How suspicious this flag is where higher numbers are more suspicious
            pub suspicion: Option<i64>,
            /// How confident/reliable this flag is
            pub confidence: Option<Confidence>,
            /// The interesting, odd, or suspicious characteristic
            pub content: Option<String>,
            /// The reason for this Flag
            pub reasoning: Option<String>,
        }

        impl EntityMetadataForm {
            /// Ensure the data in the entity metadata form is valid
            ///
            /// # Errors
            ///
            /// - `collection_start` is older than `collection_end`
            pub fn validate(&self) -> Result<(), ApiError> {
                // ensure start is newer than end
                if let (Some(start), Some(end)) = (
                    self.collection_start.as_ref(),
                    self.collection_end.as_ref(),
                ) && start < end {
                    return crate::bad!(format!(
                        "Start must be more recent than end: Start '{start}' < End '{end}'"
                    ));
                }
                Ok(())
            }
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

        impl EntityForm {
            /// Ensure the data in the entity metadata form is valid
            pub fn validate(&self) -> Result<(), ApiError> {
                self.metadata.validate()?;
                Ok(())
            }
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
            pub add_collection_tags: HashMap<String, HashSet<String>>,
            pub delete_collection_tags: HashMap<String, HashSet<String>>,
            pub collection_tags_case_insensitive: Option<bool>,
            pub collection_ignore_groups: Option<bool>,
            pub collection_start: Option<DateTime<Utc>>,
            pub collection_end: Option<DateTime<Utc>>,
            pub clear_collection_start: Option<bool>,
            pub clear_collection_end: Option<bool>,
            pub add_tools: Vec<String>,
            pub remove_tools: Vec<String>,
            pub name: Option<String>,
            pub image_path: Option<String>,
            pub command: Option<String>,
            pub offset: Option<u64>,
            pub threads: Option<u32>,
            pub handles: Option<u32>,
            pub is_wow64: Option<bool>,
            pub session_id: Option<u32>,
            pub create_time: Option<DateTime<Utc>>,
            pub exit_time: Option<DateTime<Utc>>,
            pub protocol: Option<TransportLayerProtocol>,
            pub source: Option<IpAddr>,
            pub source_port: Option<u16>,
            pub destination: Option<IpAddr>,
            pub destination_port: Option<u16>,
            pub state: Option<NetConState>,
            pub pid: Option<u64>,
            pub process: Option<String>,
            /// A sigma rule in yaml format
            pub sigma_rule: Option<String>,
            /// The new things this sigma rule should apply too
            pub add_sigma_applies_to: Vec<SigmaRuleAppliesTo>,
            /// The things things sigma rule should no longer apply too
            pub remove_sigma_applies_to: Vec<SigmaRuleAppliesTo>,
            /// The new actions to take when a sigma rule hits
            pub add_sigma_actions: Vec<SigmaActionToTake>,
            /// The actions to remove by their index in this vec
            pub remove_sigma_actions: BTreeSet<usize>,
            /// The score that a rule applies
            pub score: Option<i64>,
            /// How suspicious this flag is where higher numbers are more suspicious
            pub suspicion: Option<i64>,
            /// How confident/reliable this flag is
            pub confidence: Option<Confidence>,
            /// The interesting, odd, or suspicious characteristic
            pub content: Option<String>,
            /// The reason for this Flag
            pub reasoning: Option<String>,
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

impl Entity {
    /// Hash the identifying info for this entity
    pub fn hash_identifying(&self) -> u64 {
        // Create a hasher with a static seed
        let mut hasher = GxHasher::with_seed(1234);
        // hash any required identifying info thats not metadata
        hasher.write(self.name.as_bytes());
        self.kind.hash(&mut hasher);
        // hash our identifying metadata
        self.metadata.hash_identifying(&mut hasher);
        // get this entities identifying hash
        hasher.finish()
    }
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
    #[tracing::instrument(name = "TreeSupport::<Entities>::gather_initial", skip_all, err(Debug))]
    async fn gather_initial(
        _user: &User,
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
    #[tracing::instrument(
        name = "TreeSupport::<Entities>::gather_children",
        skip_all,
        err(Debug)
    )]
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
                // build a full list of all associations
                let mut associations = Vec::with_capacity(cursor.data.len());
                // track which associations have been newly added
                let mut new_set = HashSet::with_capacity(cursor.data.len());
                // preallocate a list of associations that are filtered to just new ones
                let mut new_list = Vec::with_capacity(cursor.data.len());
                // convert all of the listable associations to full associations
                let assoc_iter = cursor.data.drain(..).map(super::Association::try_from);
                // iterate over our associations and filter them
                for cast in assoc_iter {
                    // check if the cast for this association failed
                    let assoc = cast?;
                    // get the hash for this association
                    let tree_hash = assoc.tree_hash();
                    // check if our ring already contains this association
                    if !ring.contains(tree, tree_hash).await {
                        // only add associations that aren't already in our new set
                        if !new_set.contains(&tree_hash) {
                            // add this associations treee hash to our new set
                            new_set.insert(tree_hash);
                            // add this association to our new list
                            new_list.push(assoc.clone());
                        }
                    }
                    // add this association
                    associations.push(assoc);
                }
                // get this pages tree nodes in parallel
                let mut node_stream = stream::iter(new_list)
                    .map(|assoc| async move { assoc.get_tree_node(user, shared).await })
                    .buffer_unordered(10);
                // add our tree nodes as we get them
                while let Some(node_result) = node_stream.next().await {
                    // if we failed to get this node then raise an error
                    let node = node_result?;
                    // add this node
                    ring.add_node(node).await;
                }
                // get an entry to our parent nodes relationships
                let entry = ring
                    .relationships
                    .entry_async(source_hash)
                    .await
                    .or_default();
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
                    // get the hash for this branch (not the tree hash the hash of the full object)
                    let full_hash = branch.full_hash();
                    // insert our relationship
                    entry.upsert_async(full_hash, branch).await;
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
        key.clone()
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
    strum::EnumIter,
    strum::Display,
    Hash,
    Linearize,
    JsonSchema,
))]
#[cfg_attr(feature = "python", strum_discriminants(pyo3::pyclass(from_py_object)))]
#[cfg_attr(
    feature = "scylla-utils",
    strum_discriminants(derive(thorium_derive::ScyllaStoreJson))
)]
#[cfg_attr(
    feature = "rkyv-support",
    strum_discriminants(derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))
)]
#[cfg_attr(
    feature = "rkyv-support",
    strum_discriminants(archive_attr(derive(Debug, bytecheck::CheckBytes)))
)]
#[cfg_attr(feature = "api", strum_discriminants(derive(utoipa::ToSchema)))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum EntityMetadata {
    /// A device entity
    Device(DeviceEntity),
    /// A vendor entity
    Vendor(VendorEntity),
    /// A collection entity
    ///
    /// Collections are dynamic lists of items in Thorium (e.g. samples, repos, etc.)
    /// based on search parameters like tags
    Collection(CollectionEntity),
    /// A filesystem entity
    FileSystem(FileSystemEntity),
    /// A folder within a filesystem entity
    Folder(FileSystemFolderEntity),
    /// A Windows process tree entity
    WindowsProcessTree(WindowsProcessTreeEntity),
    /// A Windows process
    WindowsProcess(WindowsProcessEntity),
    /// A Network connection
    NetworkConnection(NetworkConnection),
    /// A sigma rule to apply to data
    SigmaRule(SigmaRule),
    /// A flag on some suspicious data
    Flag(Flag),
    /// An entity that can't be described by any of the other variants
    #[strum_discriminants(default)]
    Other,
}

impl EntityMetadata {
    /// Hash the identifying info for this entity
    ///
    /// # Arguments
    ///
    /// * `hasher` - The hasher to use to hash any identifying data
    pub fn hash_identifying(&self, hasher: &mut GxHasher) {
        // get this entities identifying info
        let identifying = IdentifyingEntityInfo::from(self);
        // hash our identifying info
        identifying.hash(hasher);
    }
}

/// The specific kind an entity is, including any data unique to its kind
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum EntityMetadataRequest {
    /// A device entity
    Device(DeviceEntityRequest),
    /// A vendor entity
    Vendor(VendorEntityRequest),
    /// A request to create a collection entity
    Collection(CollectionEntityRequest),
    /// A filesystem entity
    FileSystem(FileSystemEntity),
    /// A filesystem folder entity
    Folder(FileSystemFolderEntity),
    /// A windows process tree that has no data
    WindowsProcessTree,
    /// A windows process
    WindowsProcess(WindowsProcessEntity),
    /// A network connection
    NetworkConnection(NetworkConnection),
    /// A sigma rule to apply to data
    SigmaRule(SigmaRule),
    /// A flag on some suspicious data
    Flag(Flag),
    /// An entity that can't be described by any of the other variants
    Other,
}

impl EntityMetadataRequest {
    /// Add this entity metadata to a form
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // add our metadata
        match self {
            EntityMetadataRequest::Device(device) => device.add_to_form(form),
            EntityMetadataRequest::Vendor(vendor) => vendor.add_to_form(form),
            EntityMetadataRequest::Collection(collection) => collection.add_to_form(form),
            EntityMetadataRequest::FileSystem(fs) => fs.add_to_form(form),
            EntityMetadataRequest::Folder(folder) => folder.add_to_form(form),
            // windows process tree entities don't really have any metadata
            EntityMetadataRequest::WindowsProcessTree => {
                Ok(form.text("kind", EntityKinds::WindowsProcessTree.as_str()))
            }
            EntityMetadataRequest::WindowsProcess(process) => process.add_to_form(form),
            EntityMetadataRequest::NetworkConnection(conn) => conn.add_to_form(form),
            EntityMetadataRequest::SigmaRule(rule) => rule.add_to_form(form),
            EntityMetadataRequest::Flag(flag) => flag.add_to_form(form),
            // just set our kind to other
            EntityMetadataRequest::Other => Ok(form.text("kind", EntityKinds::Other.as_str())),
        }
    }

    /// Get our entity kind
    pub fn kind(&self) -> EntityKinds {
        match self {
            EntityMetadataRequest::Device(_) => EntityKinds::Device,
            EntityMetadataRequest::Vendor(_) => EntityKinds::Vendor,
            EntityMetadataRequest::Collection(_) => EntityKinds::Collection,
            EntityMetadataRequest::FileSystem(_) => EntityKinds::FileSystem,
            EntityMetadataRequest::Folder(_) => EntityKinds::Folder,
            EntityMetadataRequest::WindowsProcessTree => EntityKinds::WindowsProcessTree,
            EntityMetadataRequest::WindowsProcess(_) => EntityKinds::WindowsProcess,
            EntityMetadataRequest::NetworkConnection(_) => EntityKinds::NetworkConnection,
            EntityMetadataRequest::SigmaRule(_) => EntityKinds::SigmaRule,
            EntityMetadataRequest::Flag(_) => EntityKinds::Flag,
            EntityMetadataRequest::Other => EntityKinds::Other,
        }
    }

    /// Deserialize a list of metadata requests from disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to read
    /// * `kind` - The kind of metadata request to read
    #[cfg(feature = "client")]
    pub async fn load_all<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>, crate::Error> {
        // read our target file from disk
        let data = tokio::fs::read(path.as_ref()).await?;
        // try to parse this to a list of entities
        let parsed = serde_json::from_slice(&data)?;
        Ok(parsed)
    }

    /// Convert this request into something that can be scanned by a sigma rule
    ///
    /// Will return None if an entity is not scannable by sigma rules.
    #[cfg(feature = "client")]
    pub fn to_sigma_scannable(&self) -> Result<Option<String>, crate::Error> {
        // unwrap the outer enum so we don't needlessly nest things
        match self {
            Self::Device(device) => Ok(Some(serde_json::to_string(device)?)),
            Self::Vendor(vendor) => Ok(Some(serde_json::to_string(vendor)?)),
            Self::Collection(collection) => Ok(Some(serde_json::to_string(collection)?)),
            Self::FileSystem(filesystem) => Ok(Some(serde_json::to_string(filesystem)?)),
            Self::Folder(folder) => Ok(Some(serde_json::to_string(folder)?)),
            Self::WindowsProcess(proc) => Ok(Some(serde_json::to_string(proc)?)),
            Self::NetworkConnection(conn) => Ok(Some(serde_json::to_string(conn)?)),
            Self::Flag(flag) => Ok(Some(serde_json::to_string(flag)?)),
            Self::WindowsProcessTree | Self::SigmaRule(_) | Self::Other => Ok(None),
        }
    }

    /// Hash the identifying info for this entity
    ///
    /// The identifying info is a subset entity data and metadata that combined
    /// allows users to effectively dedupe entities. This is not all info as that
    /// would mean that an entity with the same info (but different entity ids) in
    ///  group A would be considered different then an entity in group B. This is
    /// primarily used to prevent endless recreating the same entities when rescanning
    /// entities with rules. This is not the same as the tree hash.
    ///
    /// # Arguments
    ///
    /// * `hasher` - The hasher to use to hash any identifying data
    pub fn hash_identifying(&self, hasher: &mut GxHasher) {
        // get this entities identifying info
        let identifying = IdentifyingEntityInfo::from(self);
        // hash our identifying info
        identifying.hash(hasher);
    }
}

/// The different info that identifies approximately identical entities
///
/// This is intended to be used to prevent duplicate entities from being created when
/// rerunning tools.
/// The specific kind an entity is, including any data unique to its kind
#[derive(Debug, Clone, Hash)]
pub enum IdentifyingEntityInfo<'a> {
    /// A collection entity
    ///
    /// Collections are dynamic lists of items in Thorium (e.g. samples, repos, etc.)
    /// based on search parameters like tags
    Collection { kind: CollectionKind },
    /// A filesystem entity
    FileSystem { sha256: &'a String },
    /// A folder within a filesystem entity
    Folder { all_sha256: &'a String },
    /// A Windows process
    WindowsProcess { pid: u64 },
    /// A Network connection
    NetworkConnection {
        source: &'a IpAddr,
        destination: &'a IpAddr,
        destination_port: u16,
    },
    /// A sigma rule to apply to data
    SigmaRule { rule: &'a String },
    /// A flag on some suspicious data
    Flag {
        suspicion: i64,
        confidence: Confidence,
        reasoning: &'a String,
    },
    /// An entity that cannot be accurately/usefully identified to prevent duplicates
    Unidentifiable,
}

impl<'a> From<&'a EntityMetadata> for IdentifyingEntityInfo<'a> {
    fn from(meta: &'a EntityMetadata) -> Self {
        match meta {
            EntityMetadata::Collection(collection) => Self::Collection {
                kind: collection.collection_kind,
            },
            EntityMetadata::FileSystem(fs) => Self::FileSystem { sha256: &fs.sha256 },
            EntityMetadata::Folder(folder) => Self::Folder {
                all_sha256: &folder.all_sha256,
            },
            EntityMetadata::WindowsProcess(proc) => Self::WindowsProcess { pid: proc.pid },
            EntityMetadata::NetworkConnection(conn) => Self::NetworkConnection {
                source: &conn.source,
                destination: &conn.destination,
                destination_port: conn.destination_port,
            },
            EntityMetadata::SigmaRule(rule) => Self::SigmaRule { rule: &rule.rule },
            EntityMetadata::Flag(flag) => Self::Flag {
                suspicion: flag.suspicion,
                confidence: flag.confidence,
                reasoning: &flag.reasoning,
            },
            // These entities have no useful identifying info
            EntityMetadata::Device(_)
            | EntityMetadata::Vendor(_)
            | EntityMetadata::WindowsProcessTree(_)
            | EntityMetadata::Other => Self::Unidentifiable,
        }
    }
}

impl<'a> From<&'a EntityMetadataRequest> for IdentifyingEntityInfo<'a> {
    fn from(meta: &'a EntityMetadataRequest) -> Self {
        match meta {
            EntityMetadataRequest::Collection(collection) => Self::Collection {
                kind: collection.collection_kind,
            },
            EntityMetadataRequest::FileSystem(fs) => Self::FileSystem { sha256: &fs.sha256 },
            EntityMetadataRequest::Folder(folder) => Self::Folder {
                all_sha256: &folder.all_sha256,
            },
            EntityMetadataRequest::WindowsProcess(proc) => Self::WindowsProcess { pid: proc.pid },
            EntityMetadataRequest::NetworkConnection(conn) => Self::NetworkConnection {
                source: &conn.source,
                destination: &conn.destination,
                destination_port: conn.destination_port,
            },
            EntityMetadataRequest::SigmaRule(rule) => Self::SigmaRule { rule: &rule.rule },
            EntityMetadataRequest::Flag(flag) => Self::Flag {
                suspicion: flag.suspicion,
                confidence: flag.confidence,
                reasoning: &flag.reasoning,
            },
            // These entities have no useful identifying info
            EntityMetadataRequest::Device(_)
            | EntityMetadataRequest::Vendor(_)
            | EntityMetadataRequest::WindowsProcessTree
            | EntityMetadataRequest::Other => Self::Unidentifiable,
        }
    }
}

impl EntityKinds {
    /// Gets a str representation of the entity kind name
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }

    /// Convert this entity kind to a [`SigmaRuleAppliesTo`] if possible
    ///
    /// Not all entity kinds can be converted and any that can't will return [`None`]
    pub fn to_sigma_applies_to(&self) -> Option<SigmaRuleAppliesTo> {
        match self {
            Self::WindowsProcess => Some(SigmaRuleAppliesTo::WindowsProcesses),
            Self::NetworkConnection => Some(SigmaRuleAppliesTo::NetworkConnections),
            // all other entity kinds cannot be scanned with sigma rules
            Self::Device
            | Self::Vendor
            | Self::Collection
            | Self::FileSystem
            | Self::Folder
            | Self::WindowsProcessTree
            | Self::SigmaRule
            | Self::Flag
            | Self::Other => None,
        }
    }

    /// Get any root entity kinds if any exist
    ///
    /// Root kinds or root nodes are for entities with a tree like structure that require
    /// a root. For instance a filesystem starts from a root FileSystem entity.
    pub fn root_kinds(&self) -> &[EntityKinds] {
        match self {
            Self::WindowsProcess => &[Self::WindowsProcessTree],
            Self::Folder => &[Self::FileSystem],
            // all other entity kinds do not have required root kinds
            Self::Device
            | Self::Vendor
            | Self::Collection
            | Self::FileSystem
            | Self::WindowsProcessTree
            | Self::NetworkConnection
            | Self::SigmaRule
            | Self::Flag
            | Self::Other => &[],
        }
    }
}

impl From<SigmaRuleAppliesTo> for EntityKinds {
    fn from(applies_to: SigmaRuleAppliesTo) -> Self {
        // map what a sigma rule applies to to an entity kind
        match applies_to {
            SigmaRuleAppliesTo::WindowsProcesses => EntityKinds::WindowsProcess,
            SigmaRuleAppliesTo::NetworkConnections => EntityKinds::NetworkConnection,
        }
    }
}

impl From<&SigmaRuleAppliesTo> for EntityKinds {
    fn from(applies_to: &SigmaRuleAppliesTo) -> Self {
        EntityKinds::from(*applies_to)
    }
}

/// A request to create an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
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
    /// Create a new entity request
    pub fn new<I>(name: impl Into<String>, metadata: EntityMetadataRequest, groups: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        // convert our groups to a list of strings
        let groups = groups.into_iter().map(Into::into).collect();
        EntityRequest {
            name: name.into(),
            metadata,
            groups,
            tags: HashMap::default(),
            description: None,
        }
    }

    /// Add a tag to this entity request
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the tag to add
    /// * `value` - The value for the tag to add
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        // get an entry into this tags key list
        let entry = self.tags.entry(key.into()).or_default();
        // add our value
        entry.insert(value.into());
        self
    }

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
            let tag_key = format!("tags[{key}][]");
            // add this tags list of values to our form
            form = multipart_set!(form, &tag_key, values);
        }
        // add our description to this requet
        let form = multipart_text!(form, "description", self.description);
        Ok(form)
    }

    /// Get our entity kind
    pub fn kind(&self) -> EntityKinds {
        // get our entity kind based on our metadata
        self.metadata.kind()
    }

    /// Get the parent info for this entity request if it has a parent
    pub fn parent_info(&self) -> Option<EntityParentInfo> {
        // get the identifying info for any parent processes if it exists
        match &self.metadata {
            EntityMetadataRequest::WindowsProcess(proc) => {
                // if we have a parent pid set then use that
                proc.parent_pid.map(EntityParentInfo::WindowsParentProcess)
            }
            _ => None,
        }
    }

    /// Get this entities root node kind
    pub fn root_kind(&self) -> Option<EntityKinds> {
        // only certain kinds of entities use a root node
        match &self.metadata {
            EntityMetadataRequest::WindowsProcess(_) => Some(EntityKinds::WindowsProcessTree),
            EntityMetadataRequest::Folder(_) => Some(EntityKinds::FileSystem),
            // all other entity types do not have a root node
            EntityMetadataRequest::Device(_)
            | EntityMetadataRequest::Vendor(_)
            | EntityMetadataRequest::Collection(_)
            | EntityMetadataRequest::FileSystem(_)
            | EntityMetadataRequest::WindowsProcessTree
            | EntityMetadataRequest::NetworkConnection(_)
            | EntityMetadataRequest::SigmaRule(_)
            | EntityMetadataRequest::Flag(_)
            | EntityMetadataRequest::Other => None,
        }
    }

    /// Hash the identifying info for this entity
    ///
    /// The identifying info is a subset entity data and metadata that combined
    /// allows users to effectively dedupe entities. This is not all info as that
    /// would mean that an entity with the same info (but different entity ids) in
    ///  group A would be considered different then an entity in group B. This is
    /// primarily used to prevent endless recreating the same entities when rescanning
    /// entities with rules.
    pub fn hash_identifying(&self) -> u64 {
        // Create a hasher with a static seed
        let mut hasher = GxHasher::with_seed(1234);
        // hash any required identifying info thats not metadata
        hasher.write(self.name.as_bytes());
        self.metadata.kind().hash(&mut hasher);
        // hash our identifying metadata
        self.metadata.hash_identifying(&mut hasher);
        // get this entities identifying hash
        hasher.finish()
    }
}

/// The response from an entity creation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EntityResponse {
    /// The ID of the created entity
    pub id: Uuid,
    /// The name of this entity
    pub name: String,
}

impl EntityResponse {
    /// Create a new entity response
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the created entity
    #[must_use]
    pub fn new(id: Uuid, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
        }
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

/// The options that you can set when listing entities in Thorium
#[derive(Debug, Clone)]
pub struct EntityListOpts {
    /// The cursor to use to continue this search
    pub cursor: Option<Uuid>,
    /// The latest date to start listing entities from
    pub start: Option<DateTime<Utc>>,
    /// The oldest date to stop listing entities from
    pub end: Option<DateTime<Utc>>,
    /// The max number of objects to retrieve on a single page
    pub page_size: usize,
    /// The total number of objects to return with this cursor
    pub limit: Option<usize>,
    /// The groups limit our search to
    pub groups: Vec<String>,
    /// The tags to filter on
    pub tags: HashMap<String, Vec<String>>,
    /// Whether matching on tags should be case-insensitive
    pub tags_case_insensitive: bool,
    /// The different kinds of entities to list
    pub kinds: Vec<EntityKinds>,
}

impl Default for EntityListOpts {
    /// Build a default search
    fn default() -> Self {
        EntityListOpts {
            start: None,
            cursor: None,
            end: None,
            page_size: 50,
            limit: None,
            groups: Vec::default(),
            tags: HashMap::default(),
            tags_case_insensitive: false,
            kinds: Vec::default(),
        }
    }
}

impl EntityListOpts {
    /// Restrict the entity search to start at a specific date
    ///
    /// # Arguments
    ///
    /// * `start` - The date to start listing entities from
    #[must_use]
    pub fn start(mut self, start: DateTime<Utc>) -> Self {
        // set the date to start listing entities at
        self.start = Some(start);
        self
    }

    /// Set the cursor to use when continuing this search
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor id to use for this search
    #[must_use]
    pub fn cursor(mut self, cursor: Uuid) -> Self {
        // set cursor for this search
        self.cursor = Some(cursor);
        self
    }

    /// Restrict the entity search to stop at a specific date
    ///
    /// # Arguments
    ///
    /// * `end` - The date to stop listing entites at
    #[must_use]
    pub fn end(mut self, end: DateTime<Utc>) -> Self {
        // set the date to end listing entities at
        self.end = Some(end);
        self
    }

    /// The max number of entities to retrieve in a single page
    ///
    /// # Arguments
    ///
    /// * `page_size` - The max number of documents to return in a single request
    #[must_use]
    pub fn page_size(mut self, page_size: usize) -> Self {
        // set the date to end listing entities at
        self.page_size = page_size;
        self
    }

    /// Limit how many entities this search can return at once
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of objects to return over the lifetime of this cursor
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        // set the date to end listing entities at
        self.limit = Some(limit);
        self
    }

    /// Limit what groups we search in
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our search to
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // add the groups to restrict returned data too
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// List entities that match a specific tag
    ///
    /// # Arguments
    ///
    /// * `key` - The tag key to match against
    /// * `value` - The tag value to match against
    #[must_use]
    pub fn tag<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        // get an entry into this tags value list
        let entry = self.tags.entry(key.into()).or_default();
        // add this tags value
        entry.push(value.into());
        self
    }

    /// List entities that match a specific tag by ref
    ///
    /// # Arguments
    ///
    /// * `key` - The tag key to match against
    /// * `value` - The tag value to match against
    pub fn tag_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        // get an entry into this tags value list
        let entry = self.tags.entry(key.into()).or_default();
        // add this tags value
        entry.push(value.into());
    }

    /// Set for matching on tags to be case-insensitive
    #[must_use]
    pub fn tags_case_insensitive(mut self) -> Self {
        self.tags_case_insensitive = true;
        self
    }

    /// Limit what kinds of entities are returned
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of entities to restrict our list to
    #[must_use]
    pub fn kind(mut self, kind: EntityKinds) -> Self {
        // add this kind to our option struct
        self.kinds.push(kind);
        self
    }

    /// Limit what kinds of entities are returned
    ///
    /// # Arguments
    ///
    /// * `kinds` - The kinds of entities to restrict our list to
    #[must_use]
    pub fn kinds(mut self, kinds: impl IntoIterator<Item = EntityKinds>) -> Self {
        // add these entity kinds to our options
        self.kinds.extend(kinds);
        self
    }
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
    /// The different kinds of entities to list
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
    /// The new name to set
    pub name: Option<String>,
    /// The groups to add to this entity
    pub add_groups: Vec<String>,
    /// The groups to remove form this entity
    pub remove_groups: Vec<String>,
    /// The decsription to set for this entity
    pub description: Option<String>,
    /// Clear this entities description
    pub clear_description: bool,
    /// Add a tool to this entity
    pub add_tools: Vec<String>,
    /// Remoev a tool from this entity
    pub remove_tools: Vec<String>,
}

impl EntityUpdate {
    /// Change the name for this entity
    ///
    /// # Arguments
    ///
    /// * `name` - The name to change too
    pub fn name(mut self, name: impl Into<String>) -> Self {
        // update our name
        self.name = Some(name.into());
        self
    }

    /// Add a group to this entity
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add
    pub fn group(mut self, group: impl Into<String>) -> Self {
        // add this group
        self.add_groups.push(group.into());
        self
    }

    /// Remove a group from this entity
    ///
    /// # Arguments
    ///
    /// * `group` - The group to remove
    pub fn remove_group(mut self, group: impl Into<String>) -> Self {
        // add this group to our remove list
        self.remove_groups.push(group.into());
        self
    }

    /// Change the description for this entity
    ///
    /// # Arguments
    ///
    /// * `description` - The description to change too
    pub fn description(mut self, description: impl Into<String>) -> Self {
        // update our description
        self.description = Some(description.into());
        self
    }

    /// Clear this entities description
    pub fn clear_description(mut self) -> Self {
        self.clear_description = true;
        self
    }

    /// Add a tool to this entity
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to add
    pub fn tool(mut self, tool: impl Into<String>) -> Self {
        // add this tool
        self.add_tools.push(tool.into());
        self
    }

    /// Remove a tool from this entity
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to remove
    pub fn remove_tool(mut self, tool: impl Into<String>) -> Self {
        // add this tool to our remove list
        self.remove_tools.push(tool.into());
        self
    }

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

/// The differnt kinds of parent info for entities
pub enum EntityParentInfo {
    /// The pid for a parent process in Windows
    WindowsParentProcess(u64),
}

impl EntityParentInfo {
    /// Get the type of sigma applies to  this parent info points to
    pub fn sigma_applies_to(&self) -> SigmaRuleAppliesTo {
        match self {
            Self::WindowsParentProcess(_) => SigmaRuleAppliesTo::WindowsProcesses,
        }
    }

    /// Get the kind of entity this parent info is for
    pub fn entity_kind(&self) -> EntityKinds {
        match self {
            Self::WindowsParentProcess(_) => EntityKinds::WindowsProcess,
        }
    }

    /// Check if an entity is the parent we are looking for
    ///
    /// # Arguments
    ///
    /// * `child` - The potential child entity request to check against
    pub fn is_parent(&self, child: &EntityRequest) -> bool {
        match (self, &child.metadata) {
            (Self::WindowsParentProcess(ppid), EntityMetadataRequest::WindowsProcess(proc)) => {
                proc.pid == *ppid
            }
            _ => false,
        }
    }
}
