//! A Tree of data in Thorium

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hasher;
use uuid::Uuid;

use super::{Origin, Sample};

/// Help serde default the tree depth to 5
fn default_tree_depth() -> usize {
    5
}

/// The parameters for building a tree in Thorium
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TreeParams {
    /// The depth to build this tree out too
    #[serde(default = "default_tree_depth")]
    pub limit: usize,
}

impl Default for TreeParams {
    fn default() -> Self {
        TreeParams {
            limit: default_tree_depth(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeTags {
    /// The tags to start building a tree with in aggregate
    pub tags: BTreeMap<String, BTreeSet<String>>,
}

impl TreeSupport for TreeTags {
    /// Hash this child object
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed to set the hasher to use
    fn tree_hash(&self, seed: i64) -> u64 {
        // build a hasher
        let mut hasher = gxhash::GxHasher::with_seed(seed);
        // hash all of our keys
        for (key, values) in &self.tags {
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
    ) -> Result<Vec<TreeNodeData>, crate::utils::ApiError> {
        // build a list of initial data
        let mut initial = Vec::with_capacity(query.tags.len());
        // build our initial tag nodes
        for tag in &query.tags {
            // conver these tags to a tree tag object
            let tree_tags = TreeTags { tags: tag.clone() };
            // build our tree tags node
            let node_data = TreeNodeData::Tag(tree_tags);
            // add this to our initial set
            initial.push(node_data);
        }
        Ok(initial)
    }

    /// Gather any children for this child node
    #[cfg(feature = "api")]
    async fn gather_children(
        &self,
        user: &super::User,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<TreeNode>, crate::utils::ApiError> {
        // build the params for listing files with these tags
        let mut opts = super::FileListOpts::default();
        // build our listing opts with all of our tag keys
        for (key, values) in &self.tags {
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
        // build a list of related children
        let mut children = Vec::with_capacity(details.data.len());
        // for each sample in this details list build and add a node
        for sample in details.data {
            // wrap this sample in a node data object
            let data = TreeNodeData::Sample(sample);
            // build the tree node for this child sample
            let node = TreeNode::new(relationships.clone(), data);
            // add this to our list of children nodes
            children.push(node);
        }
        Ok(children)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeQuery {
    /// The sha256s of the initial samples to build this tree from
    #[serde(default)]
    pub samples: Vec<String>,
    /// The different tag filters to build this tree from
    #[serde(default)]
    pub tags: Vec<BTreeMap<String, BTreeSet<String>>>,
}

pub trait TreeSupport:
    std::fmt::Debug + Clone + serde::Serialize + for<'de> serde::Deserialize<'de>
{
    /// Hash this child object
    fn tree_hash(&self, seed: i64) -> u64;

    /// Gather any initial nodes for a tree
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn gather_initial(
        user: &super::User,
        query: &TreeQuery,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<TreeNodeData>, crate::utils::ApiError>;

    /// Gather any children for this child node
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn gather_children(
        &self,
        user: &super::User,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<TreeNode>, crate::utils::ApiError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeGrowQuery {
    /// The nodes to grow
    pub growable: Vec<u64>,
}

/// The different leaves in a tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNodeData {
    /// A sample in Thorium
    Sample(Sample),
    /// A single specific tag in Thorium
    Tag(TreeTags),
}

impl TreeNodeData {
    /// Get the hash of each node
    pub fn hash(&self) -> u64 {
        // hash this node
        match self {
            Self::Sample(sample) => sample.tree_hash(1234),
            Self::Tag(tags) => tags.tree_hash(1234),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeRelationships {
    /// This is an initial node
    Initial,
    /// This node is related due to an origin
    Origin(Origin),
    /// This node is related by tags
    Tags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// This nodes relationship with its parent
    pub relationship: Vec<TreeRelationships>,
    /// The data for this node
    pub data: TreeNodeData,
}

impl TreeNode {
    /// Create a new TreeNode
    pub fn new(relationship: Vec<TreeRelationships>, data: TreeNodeData) -> Self {
        TreeNode { relationship, data }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    /// This trees id
    pub id: Uuid,
    /// The initial nodes for this tree
    pub initial: Vec<u64>,
    /// The nodes that can be grown more on this tree
    pub growable: Vec<u64>,
    /// The info on each node in this tree
    pub data_map: HashMap<u64, TreeNode>,
    /// The data in the leaves of this tree
    pub branches: HashMap<u64, HashSet<u64>>,
    /// The nodes that have already been sent
    #[serde(skip_serializing_if = "HashSet::is_empty")]
    pub sent: HashSet<u64>,
}

impl Default for Tree {
    /// Create a default tree
    fn default() -> Self {
        Tree {
            id: Uuid::new_v4(),
            initial: Vec::with_capacity(1),
            growable: Vec::with_capacity(10),
            data_map: HashMap::with_capacity(10),
            branches: HashMap::with_capacity(10),
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
    pub fn add_initial(&mut self, data: TreeNodeData) {
        // wrap our data in a tree node
        let node = TreeNode::new(vec![TreeRelationships::Initial], data);
        // hash our node
        let hash = node.data.hash();
        // add this initial node
        self.initial.push(hash);
        // add this node to our list of growable nodes
        self.growable.push(hash);
        // add this node to our data map
        self.data_map.insert(hash, node);
    }

    /// Add a child node
    ///
    /// # Arguments
    ///
    /// * `node` - The node that we are adding
    /// * `parents` - The parents of the node we are adding
    pub fn add_node(&mut self, node: TreeNode, parents: Vec<u64>) -> Option<u64> {
        // hash our node
        let hash = node.data.hash();
        // add this node to our data map if it doesn't already exist
        let existing = self.data_map.insert(hash, node).is_some();
        // link this child to its parent
        for parent in parents {
            // get an entry to this parents children
            let entry = self.branches.entry(parent).or_default();
            // add this child
            entry.insert(hash);
        }
        if existing {
            None
        } else {
            Some(hash)
        }
    }
}
