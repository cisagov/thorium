//! A Tree of data in Thorium

use gxhash::GxHasher;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Deserializer;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::hash::Hasher;
use std::str::FromStr;
use uuid::Uuid;

use crate::models::Association;
use crate::models::Entity;
use crate::models::InvalidEnum;
use crate::models::Repo;

use super::{Origin, Sample};

/// Help serde default the tree depth to 5
const fn default_tree_depth() -> usize {
    5
}

/// Help serde default the gather parents bool to true
const fn default_gather_parents() -> bool {
    true
}

/// Help serde default the gather related bool to true
const fn default_gather_related() -> bool {
    true
}

/// Help serde default the gather tag children bool to true
const fn default_gather_tag_children() -> bool {
    true
}

/// The parameters for building a tree in Thorium
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeParams {
    /// The depth to build this tree out too
    #[serde(default = "default_tree_depth")]
    pub limit: usize,
    /// Skip gathering parents
    #[serde(default = "default_gather_parents")]
    pub gather_parents: bool,
    /// Gather any related objects/entities
    #[serde(default = "default_gather_parents")]
    pub gather_related: bool,
    /// Gather children from tag nodes
    #[serde(default = "default_gather_tag_children")]
    pub gather_tag_children: bool,
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams {
            limit: default_tree_depth(),
            gather_parents: default_gather_parents(),
            gather_related: default_gather_related(),
            gather_tag_children: default_gather_tag_children(),
        }
    }
}

/// The parameters for building a tree in Thorium
#[derive(Serialize, Deserialize, Debug, JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeOpts {
    /// The depth to build this tree out too
    pub limit: usize,
    /// Skip gathering parents
    pub gather_parents: Option<bool>,
    /// Gather any related objects/entities
    pub gather_related: Option<bool>,
    /// Gather children from tag nodes
    pub gather_tag_children: Option<bool>,
}

impl Default for TreeOpts {
    fn default() -> Self {
        TreeOpts {
            limit: 50,
            gather_parents: None,
            gather_related: None,
            gather_tag_children: None,
        }
    }
}

impl TreeOpts {
    /// Set a new depth limit for this new tree
    ///
    /// # Arguments
    ///
    /// * `limit`
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TreeOpts;
    ///
    /// let opts = TreeOpts::default().limit(100);
    /// ```
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Enable gathering parents when building this tree
    ///
    /// # Arguments
    ///
    /// * `limit`
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TreeOpts;
    ///
    /// let opts = TreeOpts::default().gather_parents(true);
    /// ```
    #[must_use]
    pub fn gather_parents(mut self, enabled: bool) -> Self {
        self.gather_parents = Some(enabled);
        self
    }

    /// Enable gathering related when building this tree
    ///
    /// # Arguments
    ///
    /// * `limit`
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TreeOpts;
    ///
    /// let opts = TreeOpts::default().gather_related(true);
    /// ```
    #[must_use]
    pub fn gather_related(mut self, enabled: bool) -> Self {
        self.gather_related = Some(enabled);
        self
    }

    /// Enable gathering tag children when building this tree
    ///
    /// # Arguments
    ///
    /// * `limit`
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TreeOpts;
    ///
    /// let opts = TreeOpts::default().gather_tag_children(true);
    /// ```
    #[must_use]
    pub fn gather_tag_children(mut self, enabled: bool) -> Self {
        self.gather_tag_children = Some(enabled);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeTags {
    /// The tags to start building a tree with in aggregate
    pub tags: BTreeMap<String, BTreeSet<String>>,
}

impl TreeSupport for TreeTags {
    /// The data used to generate this types tree hash
    type HashType<'a> = &'a BTreeMap<String, BTreeSet<String>>;

    /// Hash this child object
    fn tree_hash(&self) -> u64 {
        Self::tree_hash_direct(&self.tags)
    }

    /// Hash this child object
    ///
    /// # Arguments
    ///
    /// * `input` - The data needed to generate this nodes tree hash
    /// * `hasher` - The hasher to write data to
    fn tree_hash_direct_with_hasher(input: Self::HashType<'_>, hasher: &mut GxHasher) {
        // hash all of our keys
        for (key, values) in input {
            // hash our key
            hasher.write(key.as_bytes());
            // iterate over all of the values
            for value in values {
                // hash our values for this key
                hasher.write(value.as_bytes());
            }
        }
    }

    /// Gather any initial nodes for a tree
    #[cfg(feature = "api")]
    async fn gather_initial(
        _user: &super::User,
        query: &TreeQuery,
        _shared: &crate::utils::Shared,
    ) -> Result<Vec<TreeNode>, crate::utils::ApiError> {
        // build a list of initial data
        let mut initial = Vec::with_capacity(query.tags.len());
        // build our initial tag nodes
        for tag in &query.tags {
            // conver these tags to a tree tag object
            let tree_tags = TreeTags { tags: tag.clone() };
            // build our tree tags node
            let node_data = TreeNode::Tag(tree_tags);
            // add this to our initial set
            initial.push(node_data);
        }
        Ok(initial)
    }
    /// Gather any children for this child node
    #[cfg(feature = "api")]
    async fn gather_children(
        &self,
        _user: &super::User,
        tree: &Tree,
        ring: &crate::models::backends::trees::TreeRing,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // get our tag nodes hash
        let tag_hash = self.tree_hash();
        // build the opts to get everything in the same groups this tag node is in
        let mut opts = crate::models::FileListOpts::default().groups(tree.groups.clone());
        // add this tag nodes filters to this listing opt
        // step over the keys and their values in this tag node
        for (key, values) in &self.tags {
            // get an entry to this tag keys values
            let entry = opts.tags.entry(key.clone()).or_default();
            // add this keys values
            entry.extend(values.iter().map(std::borrow::ToOwned::to_owned));
        }
        // convert our file list opts to params
        let params = crate::models::FileListParams::from(opts);
        // directly list samples in with this parent
        let mut cursor = crate::models::backends::db::files::list(params, true, shared).await?;
        // crawl this cursor and add its nodes to our tree
        loop {
            // we don't need to get any data we already have again
            let mut sha256s = cursor
                .data
                .drain(..)
                .map(|line| line.sha256)
                .collect::<Vec<String>>();
            // filter out any nodes that we already have in our tree
            sha256s.retain(|sha256| !ring.contains(tree, Sample::tree_hash_direct(sha256)));
            // only list details if there are node details to list
            if !sha256s.is_empty() {
                // get the details on these samples
                let details =
                    crate::models::backends::db::files::list_details(&tree.groups, sha256s, shared)
                        .await?;
                // wrap these samples in a tree node
                for sample in details {
                    // wrap this sample in a node data object
                    let node = TreeNode::Sample(sample);
                    // get our sample nodes hash
                    let node_hash = node.hash();
                    // add this node to our tree ring
                    ring.add_node(node);
                    // build the branch to and from our target node
                    let to = UnhashedTreeBranch::new(
                        tag_hash,
                        TreeRelationships::Tags,
                        Directionality::To,
                    );
                    let from = UnhashedTreeBranch::new(
                        node_hash,
                        TreeRelationships::Tags,
                        Directionality::From,
                    );
                    // add these relationships to our ring
                    ring.add_branch(node_hash, to);
                    ring.add_branch(tag_hash, from);
                }
            }
            // if our cursor is exhausted then stop crawling
            if cursor.exhausted() {
                break;
            }
            // we have more data in this cursor so get the next page
            cursor.next(shared).await?;
        }

        Ok(())
    }

    /// Build an association target column for an object
    #[cfg(feature = "api")]
    fn build_association_target_column(&self) -> Option<super::AssociationTargetColumn> {
        None
    }
}

/// The settings to use when finding related data in our tree
#[derive(Debug, Clone, Serialize, Deserialize, Default, JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeRelatedQuery {
    /// The tags to use when finding related data
    #[serde(default)]
    pub tags: Vec<BTreeMap<String, BTreeSet<String>>>,
}

impl TreeRelatedQuery {
    /// Check if we have any related queries set
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Clear all values in this related query
    pub(crate) fn clear(&mut self) {
        // clear our tags
        self.tags.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeQuery {
    /// The groups to limit our tree too
    #[serde(default)]
    pub groups: Vec<String>,
    /// The sha256s of the initial samples to build this tree from
    #[serde(default)]
    pub samples: Vec<String>,
    /// The different repo urls to build this tree from
    #[serde(default)]
    pub repos: Vec<String>,
    /// The entities to build this tree from
    #[serde(default)]
    pub entities: Vec<Uuid>,
    /// The different tag filters to build this tree from
    #[serde(default)]
    pub tags: Vec<BTreeMap<String, BTreeSet<String>>>,
    /// The settings for finding related data in our tree
    #[serde(default)]
    pub related: TreeRelatedQuery,
}

impl TreeQuery {
    /// Set the samples to use with this tree
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of a sample to build a tree from
    #[must_use]
    pub fn sample<T: Into<String>>(mut self, sha256: T) -> Self {
        // convert this sha256 into a string and add it
        self.samples.push(sha256.into());
        self
    }
}

pub trait TreeSupport:
    std::fmt::Debug + Clone + serde::Serialize + for<'de> serde::Deserialize<'de>
{
    /// The data used to generate this types tree hash
    type HashType<'a>: std::fmt::Debug;

    /// The seed for hashing this tree object
    ///
    /// You should basically never change this
    #[must_use]
    fn tree_seed() -> i64 {
        1234
    }

    /// Hash this child object
    fn tree_hash(&self) -> u64;

    /// Hash a child object directly
    ///
    /// # Arguments
    ///
    /// * `input` - The data needed to generate this nodes tree hash
    fn tree_hash_direct(input: Self::HashType<'_>) -> u64 {
        // get our hash seed
        let seed = Self::tree_seed();
        // instance a default gx hasher
        let mut hasher = GxHasher::with_seed(seed);
        // hash our tree node
        Self::tree_hash_direct_with_hasher(input, &mut hasher);
        // get our hash
        hasher.finish()
    }

    /// Hash a child object directly
    ///
    /// # Arguments
    ///
    /// * `input` - The data needed to generate this nodes tree hash
    /// * `hasher` - The hasher to write data to
    fn tree_hash_direct_with_hasher(input: Self::HashType<'_>, hasher: &mut GxHasher);

    /// Gather any initial nodes for a tree
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn gather_initial(
        user: &super::User,
        query: &TreeQuery,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<TreeNode>, crate::utils::ApiError>;

    /// Gather any parents for this child node
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is building this tree
    /// * `tree` - The tree to build
    /// * `ring` - The current growth ring for this tree
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn gather_parents(
        &self,
        _user: &super::User,
        _tree: &Tree,
        _ring: &crate::models::backends::trees::TreeRing,
        _shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        Ok(())
    }

    /// Gather any children for this child node
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn gather_children(
        &self,
        user: &super::User,
        tree: &Tree,
        ring: &crate::models::backends::trees::TreeRing,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError>;

    /// Build an association target column for an object
    #[cfg(feature = "api")]
    fn build_association_target_column(&self) -> Option<super::AssociationTargetColumn>;
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum U64OrString {
    U64(Vec<u64>),
    StringU64(Vec<String>),
}

pub fn u64_or_string<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    match U64OrString::deserialize(deserializer)? {
        U64OrString::U64(values) => Ok(values),
        U64OrString::StringU64(values) => values
            .into_iter()
            .map(|val| match val.parse::<u64>() {
                Ok(cast) => Ok(cast),
                Err(_) => Err(serde::de::Error::custom("Failed to parse u64 for growable")),
            })
            .collect::<Result<Vec<u64>, D::Error>>(),
    }
}

#[serde_with::serde_as]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeGrowQuery {
    /// The nodes to grow
    #[serde(deserialize_with = "u64_or_string")]
    pub growable: Vec<u64>,
}

impl TreeGrowQuery {
    /// Add a node to grow this tree from
    ///
    /// # Arguments
    ///
    /// * `node_hash` - The hash for the node to grow
    pub fn add_growable(mut self, node_hash: u64) -> Self {
        self.growable.push(node_hash);
        self
    }

    /// Add a node to grow this tree from
    ///
    /// # Arguments
    ///
    /// * `node_hash` - The hash for the node to grow
    pub fn add_growable_ref(&mut self, node_hash: u64) {
        self.growable.push(node_hash);
    }
}

/// The different leaves in a tree
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum TreeNode {
    /// A sample in Thorium
    Sample(Sample),
    /// A repo in Thorium
    Repo(Repo),
    /// A single specific tag in Thorium
    Tag(TreeTags),
    /// An entity in Thorium
    Entity(Entity),
}

impl TreeNode {
    /// Get the hash of each node
    #[must_use]
    pub fn hash(&self) -> u64 {
        // hash this node with a static seed to ensure consistent hashing
        match self {
            Self::Sample(sample) => sample.tree_hash(),
            Self::Repo(repo) => repo.tree_hash(),
            Self::Tag(tags) => tags.tree_hash(),
            Self::Entity(entity) => entity.tree_hash(),
        }
    }

    /// Get the hash of each node
    ///
    /// # Arguments
    ///
    /// * `hasher` - The hasher to use
    pub fn hash_with_hasher(&self, hasher: &mut GxHasher) {
        // hash this node with a static seed to ensure consistent hashing
        match self {
            Self::Sample(sample) => Sample::tree_hash_direct_with_hasher(&sample.sha256, hasher),
            Self::Repo(repo) => Repo::tree_hash_direct_with_hasher(&repo.url, hasher),
            Self::Tag(tags) => TreeTags::tree_hash_direct_with_hasher(&tags.tags, hasher),
            Self::Entity(entity) => Entity::tree_hash_direct_with_hasher(&entity.id, hasher),
        }
    }

    /// Check if this node has this tag filter
    ///
    /// # Arguments
    ///
    /// * `tag_filter` - The tag filters to check for
    #[must_use]
    pub fn has_tag_filters(&self, tag_filter: &BTreeMap<String, BTreeSet<String>>) -> bool {
        // if we have no tag filters then return false
        if tag_filter.is_empty() {
            return false;
        }
        // get the tags for this object if we have tags
        let tags = match self {
            Self::Sample(sample) => &sample.tags,
            Self::Repo(repo) => &repo.tags,
            Self::Tag(_) => return false,
            Self::Entity(entity) => &entity.tags,
        };
        // check if we have all the tags for this tag filte
        for (filter_key, filter_values) in tag_filter {
            // make sure we have the key for this filter
            match tags.get(filter_key) {
                Some(values) => {
                    // make sure we have all of the filter values
                    if !filter_values.iter().all(|fval| values.contains_key(fval)) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Get a nodes parent to check origin info against if possible
    #[must_use]
    pub fn get_origin_parent(&self) -> Option<&String> {
        match self {
            Self::Sample(sample) => Some(&sample.sha256),
            Self::Repo(repo) => Some(&repo.url),
            Self::Tag(_) | Self::Entity(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum TreeRelationships {
    /// This is an initial node
    Initial,
    /// This node is related due to an origin
    Origin(Origin),
    /// This node is related by tags
    Tags,
    /// This node is related by an association
    Association(Association),
}

/// The direction this branch is going
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash, Copy)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
pub enum Directionality {
    /// This association/relationship/branch is from the parent to a child
    To,
    /// This association/relationship/branch is from the child to the parent
    From,
    /// This association/relationship/branch is bidirectional
    Bidirectional,
}

impl Directionality {
    /// Get the opposite direction of ourselves
    ///
    /// Bidirectional will just return biderectional
    #[must_use]
    pub fn opposite(self) -> Self {
        match self {
            Self::To => Self::From,
            Self::From => Self::To,
            Self::Bidirectional => Self::Bidirectional,
        }
    }

    /// Cast this notification level to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::To => "To",
            Self::From => "From",
            Self::Bidirectional => "Bidirectional",
        }
    }
}

impl FromStr for Directionality {
    type Err = InvalidEnum;

    /// Conver this str to a ``Directionality``
    ///
    /// # Arguments
    ///
    /// * `raw` - The str to convert from
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "To" => Ok(Directionality::To),
            "From" => Ok(Directionality::From),
            "Directionality" => Ok(Directionality::Bidirectional),
            _ => Err(InvalidEnum(format!("Unknown Directionality: {raw}"))),
        }
    }
}

/// A branch between nodes in a relationship tree
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct UnhashedTreeBranch {
    /// The relationship for this branch
    pub relationship: TreeRelationships,
    // The node this is a branch too
    pub node: u64,
    /// The direction for this branch
    pub direction: Directionality,
}

impl UnhashedTreeBranch {
    pub(super) fn new(
        node: u64,
        relationship: TreeRelationships,
        direction: Directionality,
    ) -> Self {
        // hash this relationship
        UnhashedTreeBranch {
            relationship,
            node,
            direction,
        }
    }
}

/// A branch between nodes in a relationship tree
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeBranch {
    /// The relationship for this branch
    pub relationship: TreeRelationships,
    /// A hash for this relationship
    pub relationship_hash: u64,
    // The node this is a branch too
    pub node: u64,
    /// The direction for this branch
    pub direction: Directionality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Tree {
    /// This trees id
    pub id: Uuid,
    /// The groups this tree will search
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<String>,
    /// The initial nodes for this tree
    pub initial: Vec<u64>,
    /// The nodes that can be grown more on this tree
    pub growable: Vec<u64>,
    /// The info on each node in this tree
    pub data_map: HashMap<u64, TreeNode>,
    /// The data in the leaves of this tree
    pub branches: HashMap<u64, HashSet<TreeBranch>>,
    /// The settings to use to relate our in tree nodes with other data
    #[serde(default, skip_serializing_if = "TreeRelatedQuery::is_empty")]
    pub related: TreeRelatedQuery,
    /// The nodes that have already been sent
    #[serde(skip_serializing_if = "HashSet::is_empty")]
    pub sent: HashSet<u64>,
    ///// The associations that were discovered in this grow loop
    //pub associations: HashMap<AssociationTargetColumn, Vec<ListableAssociation>,
}

impl Default for Tree {
    /// Create a default tree
    fn default() -> Self {
        Tree {
            id: Uuid::new_v4(),
            groups: Vec::default(),
            initial: Vec::with_capacity(1),
            growable: Vec::with_capacity(10),
            data_map: HashMap::with_capacity(10),
            branches: HashMap::with_capacity(10),
            related: TreeRelatedQuery::default(),
            sent: HashSet::with_capacity(10),
        }
    }
}

impl Tree {
    /// Add a node to this tree
    ///
    /// # Arguments
    ///
    /// * `data` - The initial tree node to add
    pub fn add_initial(&mut self, node: TreeNode) {
        // hash our node
        let hash = node.hash();
        // add this initial node
        self.initial.push(hash);
        // add this node to our list of growable nodes
        self.growable.push(hash);
        // add this node to our data map
        self.data_map.insert(hash, node);
        // add this node to our sent nodes
        self.sent.insert(hash);
    }
}
