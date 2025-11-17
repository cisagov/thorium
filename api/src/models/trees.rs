//! A Tree of data in Thorium

use serde::Deserialize;
use serde::Deserializer;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hasher;
use uuid::Uuid;

use crate::models::Association;
use crate::models::Entity;
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
#[derive(Deserialize, Debug)]
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

/// Gather all files with some tag for a tree
#[cfg(feature = "api")]
pub(crate) async fn gather_samples_with_tags(
    user: &super::User,
    tags: &BTreeMap<String, BTreeSet<String>>,
    children: &mut Vec<(TreeNode, Vec<TreeRelationships>)>,
    shared: &crate::utils::Shared,
) -> Result<(), crate::utils::ApiError> {
    // build the params for listing files with these tags
    let mut opts = super::FileListOpts::default();
    // build our listing opts with all of our tag keys
    for (key, values) in tags {
        // get the values for this tag key
        for value in values {
            // add this tag key/value to our list opts
            opts.tag_ref(key, value);
        }
    }
    // list all samples with these tags
    let list = Sample::list(user, opts, true, shared).await?;
    // get the details on these samples
    let details = list.details(user, shared).await?;
    // All nodes we find will be related by tags
    let relationships = vec![TreeRelationships::Tags];
    // reserve space for any new children
    children.reserve(details.data.len());
    // for each sample in this details list build and add a node
    for sample in details.data {
        // build a tree node for samples
        let node = TreeNode::Sample(sample);
        // add this to our list of children nodes
        children.push((node, relationships.clone()));
    }
    Ok(())
}

/// Gather all repos with some tag for a tree
#[cfg(feature = "api")]
pub(crate) async fn gather_repos_with_tags(
    user: &super::User,
    tags: &BTreeMap<String, BTreeSet<String>>,
    children: &mut Vec<(TreeNode, Vec<TreeRelationships>)>,
    shared: &crate::utils::Shared,
) -> Result<(), crate::utils::ApiError> {
    // build the params for listing files with these tags
    let mut opts = super::RepoListOpts::default();
    // build our listing opts with all of our tag keys
    for (key, values) in tags {
        // get the values for this tag key
        for value in values {
            // add this tag key/value to our list opts
            opts.tag_ref(key, value);
        }
    }
    // list all repos with these tags
    let list = Repo::list(user, opts, shared).await?;
    // get the details on these repos
    let details = list.details(user, shared).await?;
    // All nodes we find will be related by tags
    let relationships = vec![TreeRelationships::Tags];
    // reserve space for any new children
    children.reserve(details.data.len());
    // for each sample in this details list build and add a node
    for repo in details.data {
        // build a tree node for repos
        let node = TreeNode::Repo(repo);
        // add this to our list of children nodes
        children.push((node, relationships.clone()));
    }
    Ok(())
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
    /// * `seed` - The seed to set the hasher to use
    fn tree_hash_direct<'a>(input: Self::HashType<'a>) -> u64 {
        // get our hash seed
        let seed = Self::tree_seed();
        // build a hasher
        let mut hasher = gxhash::GxHasher::with_seed(seed);
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
        // finalize our hasher
        hasher.finish()
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
        // build the opts to get everything tagged with this parent hash
        let mut opts = crate::models::FileListOpts::default().groups(tree.groups.clone());
        // add this tag nodes filters to this listing opt
        // step over the keys and their values in this tag node
        for (key, values) in &self.tags {
            // get an entry to this tag keys values
            let entry = opts.tags.entry(key.clone()).or_default();
            // add this keys values
            entry.extend(values.iter().map(|value| value.to_owned()));
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
            sha256s.retain(|sha256| !ring.contains(tree, Sample::tree_hash_direct(&sha256)));
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
                    // add this node to our tree ring
                    ring.add_node(node);
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeRelatedQuery {
    /// The tags to use when finding related data
    #[serde(default)]
    pub tags: Vec<BTreeMap<String, BTreeSet<String>>>,
}

impl TreeRelatedQuery {
    /// Check if we have any related queries set
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Clear all values in this related query
    pub(crate) fn clear(&mut self) {
        // clear our tags
        self.tags.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    fn tree_hash_direct<'a>(input: Self::HashType<'a>) -> u64;

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

    /// Find relationships by checking origin info for a node
    ///
    /// # Arguments
    ///
    /// * `parents` - The parents to check against
    #[cfg(feature = "api")]
    fn check_origins(
        &self,
        parents: &HashMap<&String, u64>,
        relationships: &dashmap::DashMap<u64, dashmap::DashSet<TreeBranch>>,
    ) {
        // most things have no origins
    }

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
    pub fn hash(&self) -> u64 {
        // hash this node with a static seed to ensure consistent hashing
        match self {
            Self::Sample(sample) => sample.tree_hash(),
            Self::Repo(repo) => repo.tree_hash(),
            Self::Tag(tags) => tags.tree_hash(),
            Self::Entity(entity) => entity.tree_hash(),
        }
    }

    /// Check if this node has this tag filter
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
        return true;
    }

    /// Get a nodes parent to check origin info against if possible
    pub fn get_origin_parent(&self) -> Option<&String> {
        match self {
            Self::Sample(sample) => Some(&sample.sha256),
            Self::Repo(repo) => Some(&repo.url),
            Self::Tag(_) => None,
            Self::Entity(_) => None,
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

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeBranch {
    /// Whether this branch results in a loop
    pub is_loop: bool,
    /// The relationship for this branch
    pub relationship: TreeRelationships,
    // The node this is a branch too
    pub node: u64,
}

impl TreeBranch {
    pub(super) fn new(node: u64, relationship: TreeRelationships) -> Self {
        TreeBranch {
            is_loop: false,
            relationship,
            node,
        }
    }
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

    /// Add a child node
    ///
    /// # Arguments
    ///
    /// * `node` - The node that we are adding
    /// * `parents` - The parents of the node we are adding
    pub fn add_node(
        &mut self,
        node: TreeNode,
        relationships: Vec<TreeRelationships>,
        parent: u64,
    ) -> Option<u64> {
        // hash our node
        let hash = node.hash();
        // add this node to our data map if it doesn't already exist
        let existing = self.data_map.insert(hash, node).is_some();
        // build and add the branch to this child from our parent
        for relationship in relationships {
            // build our branch
            let branch = TreeBranch {
                is_loop: false,
                relationship,
                node: hash,
            };
            // get an entry to this parents children
            let entry = self.branches.entry(parent).or_default();
            // add this child
            entry.insert(branch);
        }
        if existing { None } else { Some(hash) }
    }

    /// Add a child node by hash
    ///
    /// # Arguments
    ///
    /// * `hash` - The hash for the node we are adding relationships for
    /// * `parents` - The parents of the node we are adding
    pub fn add_node_by_hash(&mut self, hash: u64, relationship: TreeRelationships, parent: u64) {
        // build our branch
        let branch = TreeBranch {
            is_loop: false,
            relationship,
            node: hash,
        };
        // get an entry to this parents children
        let entry = self.branches.entry(parent).or_default();
        // add this child
        entry.insert(branch);
    }
}
