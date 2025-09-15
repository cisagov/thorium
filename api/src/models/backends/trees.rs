//! Build out trees based on data in Thorium's database

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use dashmap::{DashMap, DashSet};
use uuid::Uuid;

use super::db;
use crate::bad;
use crate::models::trees::TreeTags;
use crate::models::{
    Association, AssociationListParams, AssociationTargetColumn, Entity, FileListOpts,
    FileListParams, Repo, Sample, Tree, TreeBranch, TreeNode, TreeParams, TreeQuery,
    TreeRelationships, TreeSupport, User,
};
use crate::utils::{ApiError, Shared};

impl TreeQuery {
    /// Make sure our query is not empty and error if it is
    pub fn check_empty(&self) -> Result<(), ApiError> {
        if self.samples.is_empty()
            && self.repos.is_empty()
            && self.entities.is_empty()
            && self.tags.is_empty()
        {
            bad!("Initial starting data must be set!".to_owned())
        } else {
            Ok(())
        }
    }
}

impl TreeNode {
    /// Gather all of the parents for this node
    pub async fn gather_parents(
        &self,
        user: &User,
        tree: &Tree,
        ring: &TreeRing,
        shared: &Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // gather parents for this new data
        match &self {
            TreeNode::Sample(sample) => sample.gather_parents(user, tree, ring, shared).await,
            // Only files actually have parents so the rest of these are basically noops
            TreeNode::Repo(repo) => repo.gather_parents(user, tree, ring, shared).await,
            TreeNode::Tag(tags) => tags.gather_parents(user, tree, ring, shared).await,
            TreeNode::Entity(entity) => entity.gather_parents(user, tree, ring, shared).await,
        }
    }

    /// Gather all of the children for this node
    pub async fn gather_children(
        &self,
        user: &User,
        params: &TreeParams,
        tree: &Tree,
        ring: &TreeRing,
        shared: &Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // gather children for this new data
        match &self {
            TreeNode::Sample(sample) => sample.gather_children(user, tree, ring, shared).await,
            TreeNode::Repo(repo) => repo.gather_children(user, tree, ring, shared).await,
            TreeNode::Tag(tags) => {
                // only gather children from tag nodes if we want too
                if params.gather_tag_children {
                    tags.gather_children(user, tree, ring, shared).await
                } else {
                    Ok(())
                }
            }
            // entities only use associations to gather children
            // so this is basically a no op
            TreeNode::Entity(entity) => entity.gather_children(user, tree, ring, shared).await,
        }
    }

    /// Get the tags for this node
    pub fn get_tags(&self) -> Option<&HashMap<String, HashMap<String, HashSet<String>>>> {
        // get the tags for each object if they have tags
        match self {
            Self::Sample(sample) => Some(&sample.tags),
            Self::Repo(repo) => Some(&repo.tags),
            Self::Tag(_) => None,
            Self::Entity(entity) => Some(&entity.tags),
        }
    }

    /// Gather any related nodes based on related query settings
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree to gather related nodes for
    /// * `ring` - The current tree growth ring
    pub async fn gather_related(
        &self,
        tree: &Tree,
        ring: &TreeRing,
    ) -> Result<(), crate::utils::ApiError> {
        // if we have any tag query params set then get our tags
        if !tree.related.tags.is_empty() {
            // get this nodes tags
            if let Some(tags) = self.get_tags() {
                // check if any of our related tag filters matches this sample
                'filter: for filter in &tree.related.tags {
                    // build up the tag filter node to add to our tree
                    let mut tag_node = TreeTags {
                        tags: BTreeMap::default(),
                    };
                    // step over the tags in this filter and make sure this sample has all of them
                    for (key, values) in filter {
                        // get this tags filters values
                        if let Some(found_values) = tags.get(key) {
                            // if we have no values then just check for the presence of this tag
                            if values.is_empty() {
                                // get an entry to this keys values
                                let entry = tag_node.tags.entry(key.to_owned()).or_default();
                                // we have no values so just add all of the matching tags we do have to this filter
                                entry.extend(found_values.keys().map(|value| value.to_owned()));
                            } else {
                                // we have specific values to look for so make sure all of those match
                                if !values.iter().all(|value| found_values.contains_key(value)) {
                                    // we do not have all of the required values for this filter
                                    // continue on to the next possible filter
                                    continue 'filter;
                                }
                                // add all of the matching values to our new tag node
                                tag_node.tags.insert(key.to_owned(), values.clone());
                            }
                        } else {
                            // we didn't find this key so continue on to the next filter
                            continue 'filter;
                        }
                    }
                    // we met all of the conditions for this tag filter so add it to our tree if its new
                    // get the hash for this tag node
                    let tag_hash = tag_node.tree_hash();
                    // check if this node is already in our tree
                    if !ring.contains(tree, tag_hash) {
                        // wrap this tag node in a tree node
                        let node = TreeNode::Tag(tag_node);
                        // this node doesn't already exist so add it
                        ring.add_node(node);
                    }
                    // get this nodes hash
                    let node_hash = self.hash();
                    // get an entry to this nodes relationships
                    let entry = ring.relationships.entry(node_hash).or_default();
                    // create this tags relationship
                    let relationship = TreeRelationships::Tags;
                    // wrap our relationship in a branch
                    let branch = TreeBranch::new(tag_hash, relationship);
                    // add this tag relationship
                    entry.insert(branch);
                }
            }
        }
        Ok(())
    }

    /// Check this nodes origins
    ///
    /// # Arguments
    ///
    /// * `parents` - The parents to check against
    /// * `relationships` - The relationships to compare against
    pub fn check_origins(
        &self,
        parents: &HashMap<&String, u64>,
        relationships: &DashMap<u64, DashSet<TreeBranch>>,
    ) {
        match self {
            TreeNode::Sample(sample) => sample.check_origins(parents, relationships),
            // repo nodes do not have origins
            TreeNode::Repo(_) => (),
            // tag nodes do not have origins
            TreeNode::Tag(_) => (),
            // entities do not have origins
            TreeNode::Entity(_) => (),
        }
    }

    /// gather all of the associations for this node
    pub fn build_association_target(&self) -> Option<AssociationTargetColumn> {
        match &self {
            TreeNode::Sample(sample) => sample.build_association_target_column(),
            TreeNode::Repo(repo) => repo.build_association_target_column(),
            // This currently just always returns none as tags do not support associations
            TreeNode::Tag(tag) => tag.build_association_target_column(),
            TreeNode::Entity(entity) => entity.build_association_target_column(),
        }
    }

    /// Grow a node by crawling its associations
    async fn gather_associations(
        &self,
        tree: &Tree,
        ring: &TreeRing,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // build the association target for this node
        if let Some(target) = self.build_association_target() {
            // get our current nodes hash
            let source_hash = self.hash();
            // use default params for listing associations
            let mut params = AssociationListParams::default();
            // make sure we limit our search to groups our tree can use
            params.groups = tree.groups.clone();
            // list associations for this node
            let mut cursor = db::associations::list(params, &target, shared).await?;
            // step over our associations until our cursor is exhausted
            loop {
                // add these associations to our map of associations
                for association in cursor.data.drain(..) {
                    // convert this association
                    let converted = Association::try_from(association)?;
                    // get an entry to this nodes associations
                    let entry = ring.associations.entry(source_hash).or_default();
                    // add this association
                    entry.insert(converted);
                }
                // if our cursor is exhausted then break
                if cursor.exhausted() {
                    break;
                }
                // get the next page of data
                cursor.next(shared).await?;
            }
        }
        Ok(())
    }
}

/// Gather any children for this child node
async fn gather_tag_filter_children(
    user: &User,
    tags: &BTreeMap<String, BTreeSet<String>>,
    shared: &Shared,
) -> Result<Vec<(TreeNode, Vec<TreeRelationships>)>, crate::utils::ApiError> {
    // start with empty children
    let mut children = Vec::default();
    // get any children for for files
    crate::models::trees::gather_samples_with_tags(user, tags, &mut children, shared).await?;
    // get any children for repos
    crate::models::trees::gather_repos_with_tags(user, tags, &mut children, shared).await?;
    Ok(children)
}

/// The data to add to our tree for a single grow round
#[derive(Default, Debug)]
pub struct TreeRing {
    /// The nodes that are newly added across growth events
    pub added: DashSet<u64>,
    /// The newly added nodes during this grow round
    pub nodes: DashMap<u64, TreeNode>,
    /// The new associations populate
    pub associations: DashMap<u64, DashSet<Association>>,
    /// The newly added relationships during this grow round
    pub relationships: DashMap<u64, DashSet<TreeBranch>>,
}

impl TreeRing {
    /// Check if either our ring or tree contains an id
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree to check
    /// * `hash` - The node has to look for
    pub fn contains(&self, tree: &Tree, hash: u64) -> bool {
        self.nodes.contains_key(&hash) || tree.data_map.contains_key(&hash)
    }
    /// Add a new node
    ///
    /// # Arguments
    ///
    /// * `node` - The node to add
    pub fn add_node(&self, node: TreeNode) {
        // hash our newly added node
        let hash = node.hash();
        // insert our new node
        self.nodes.insert(hash, node);
        // add our new node to our added set
        self.added.insert(hash);
    }

    /// Gather any file nodes with a specific parent value
    ///
    /// # Arguments
    ///
    /// * `tree` - The current tree we are getting children from a specific prent
    /// * `key` - The key to use when finding children by tags
    /// * `parent` - The parent to look for children for
    /// * `shared` - Shared Thorium objects
    pub async fn gather_files_from_parent(
        &self,
        tree: &Tree,
        key: &str,
        parent: &str,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // build the opts to get everything tagged with this parent hash
        let opts = FileListOpts::default()
            .tag(key, parent)
            .groups(tree.groups.clone());
        // convert our file list opts to params
        let params = FileListParams::from(opts);
        // directly list samples in with this parent
        let mut cursor = db::files::list(params, true, shared).await?;
        // crawl this cursor and add its nodes to our tree
        loop {
            // we don't need to get any data we already have again
            let mut sha256s = cursor
                .data
                .drain(..)
                .map(|line| line.sha256)
                .collect::<Vec<String>>();
            // filter out any nodes that we already have in our tree
            sha256s.retain(|sha256| !self.contains(tree, Sample::tree_hash_direct(&sha256)));
            // get the details on these samples
            let details = db::files::list_details(&tree.groups, sha256s, shared).await?;
            // wrap these samples in a tree node
            for sample in details {
                // wrap this sample in a node data object
                let node = TreeNode::Sample(sample);
                // add this node to our tree ring
                self.add_node(node);
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
}

impl Tree {
    /// Build or get an existing tree from params
    ///
    /// # Arguments
    ///
    /// * `user` - The user who is building a tree
    /// * `query` - The query to use to start a tree
    /// * `shared` - Shared Thorium objects
    pub async fn from_query(
        user: &User,
        mut query: TreeQuery,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // make sure we have some initial starting data for this query
        query.check_empty()?;
        // start with a default tree
        let mut tree = Tree::default();
        // authorize this user can find data in all of the specified groups
        // if no groups are set then use all groups this user can see
        user.authorize_groups(&mut query.groups, shared).await?;
        // get our initial data
        let samples = Sample::gather_initial(user, &query, shared).await?;
        let repos = Repo::gather_initial(user, &query, shared).await?;
        let tags = TreeTags::gather_initial(user, &query, shared).await?;
        let entities = Entity::gather_initial(user, &query, shared).await?;
        // set the groups to restrict our tree too
        tree.groups = query.groups;
        // add our initial samples
        for sample in samples {
            // add our initial node
            tree.add_initial(sample);
        }
        // add our initial repos
        for repo in repos {
            tree.add_initial(repo);
        }
        // add our initial entities
        for entity in entities {
            tree.add_initial(entity);
        }

        // add our initial tags
        for tag in tags {
            // add our initial node
            tree.add_initial(tag);
        }
        // add our tags for building relationships between nodes
        tree.related = query.related;
        Ok(tree)
    }

    /// Filter out any nodes that have no children and are not growable
    pub fn filter_childless(&mut self) {
        // build a list of all nodes that are childless and not growable
        let childless = self
            .data_map
            .iter()
            .filter(|(hash, _)| !self.branches.contains_key(hash))
            .filter(|(hash, _)| !self.growable.contains(hash))
            .map(|(hash, _)| *hash)
            .collect::<Vec<u64>>();
        // step over all branches and remove any childless non growable nodes
        for (_, branches) in self.branches.iter_mut() {
            // remove our childless nodes
            branches.retain(|branch| !childless.contains(&branch.node));
        }
        // drop any empty branches
        self.branches.retain(|_, branch| !branch.is_empty());
        // drop node data for our childless nodes
        self.data_map.retain(|node, _| !childless.contains(node));
    }

    /// Gather any of the children or associations for a node
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is growing this tree from a specific node
    /// * `hash` - The hash for the node to grow
    /// * `ring` - The current growth ring for this tree
    /// * `params` - The params for growing this tree
    /// * `shared` - Shared Thorium objects
    async fn grow_from_node(
        &self,
        user: &User,
        hash: u64,
        ring: &TreeRing,
        params: &TreeParams,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // if we already have this nodes info then get it
        match self.data_map.get(&hash) {
            // This initial node exists so we can grow from it
            Some(node) => {
                // check if we want to gather this nodes parents
                if params.gather_parents {
                    // gather this nodes parents
                    node.gather_parents(user, self, ring, shared).await?;
                }
                // gather this nodes children
                node.gather_children(user, params, self, ring, shared)
                    .await?;
                // check if we want to gather this nodes related nodes
                if params.gather_related {
                    // gather any related nodes based on our related queries
                    node.gather_related(self, ring).await?;
                }
                // gather any relationships or children from this nodes associations
                node.gather_associations(self, ring, shared).await?;
                Ok(())
            }
            // We are missing a node that was requested to be grown so return an error
            None => bad!(format!("{} is not a valid growable node", hash)),
        }
    }

    /// Get all of our  associations nodes and add them to our relationship map
    ///
    /// # Arguments
    ///
    /// * `user` - The user who is building out this tree
    /// * `ring` - The current tree ring to grow
    /// * `shared` - Shared Thorium objects
    pub async fn get_association_nodes(
        &mut self,
        user: &User,
        ring: &mut TreeRing,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // take our associations from our ring
        let ring_assoc = std::mem::take(&mut ring.associations);
        // step over all associations and get their data
        for (source_hash, associations) in ring_assoc {
            // step over the associations for this source node
            for association in associations {
                // get this associations tree hash
                let other_hash = association.tree_hash();
                // check if we already have this associations other node
                if !ring.contains(self, other_hash) {
                    // get this associations other node
                    let node = association.get_tree_node(user, shared).await?;
                    // add this node to our ring
                    ring.add_node(node);
                }
                // get the source and target hash
                let (source, target) = if association.to_source {
                    (source_hash, other_hash)
                } else {
                    (other_hash, source_hash)
                };
                // build the relationship for this node
                let relationship = TreeRelationships::Association(association);
                // wrap this relationship in a branch
                let branch = TreeBranch::new(target, relationship);
                // get an entry to the source nodes branches
                let entry = ring.relationships.entry(source).or_default();
                // add this association to our relationships map in this tree ring
                entry.insert(branch);
            }
        }
        Ok(())
    }

    /// Merge a tree ring into our tree
    ///
    /// # Arguments
    ///
    /// * `ring` - The tree ring to merge
    pub fn merge_ring(&mut self, ring: &mut TreeRing) -> Result<(), ApiError> {
        // get a list of newly added ids
        let new_hashes = ring
            .nodes
            .iter()
            .map(|item| *item.key())
            .collect::<Vec<u64>>();
        // get our current ring nodes
        let nodes = std::mem::take(&mut ring.nodes);
        // merge our new nodes into our tree
        self.data_map.extend(nodes.into_iter());
        // instance our parents and their node hashes
        let mut parents = HashMap::with_capacity(self.data_map.len());
        // build a set of parents to check origin info against
        for (node_hash, node) in &self.data_map {
            // get this nodes parent to look for in origins
            if let Some(parent) = node.get_origin_parent() {
                // insert this node into our parent map
                parents.insert(parent, *node_hash);
            }
        }
        // clear our growable nodes
        self.growable.clear();
        // crawl over our new nodes and build relationships back to any existing parents
        for hash in new_hashes {
            // add this new hash to our growable set
            self.growable.push(hash);
            // if we already have this nodes info then get it
            match self.data_map.get(&hash) {
                // this node exists so check its origins for relationships
                Some(node) => node.check_origins(&parents, &ring.relationships),
                // We are missing a node that was requested to be grown so return an error
                None => return bad!(format!("{} is not a valid node", hash)),
            }
        }
        Ok(())
    }

    /// Add our ring relationships to our tree
    fn add_relationships(&mut self, relationships: DashMap<u64, DashSet<TreeBranch>>) {
        // clear any existing branches
        self.branches.clear();
        // step over our current relationships and add them
        for (hash, branches) in relationships {
            // get an entry to this nodes branches
            let entry = self.branches.entry(hash).or_default();
            // ignore any branches that loop back to themselves
            for branch in branches {
                // only add branchs that do not loop back to themselves
                if branch.node != hash {
                    // add our branches
                    entry.insert(branch);
                }
            }
        }
    }

    /// Build a tree based on data in Thorium's database
    pub async fn grow(
        &mut self,
        user: &User,
        params: &TreeParams,
        shared: &Shared,
    ) -> Result<DashSet<u64>, ApiError> {
        // track how many times this tree has grown
        let mut rings = 0;
        // have a tree ring for each growth
        let mut ring = TreeRing::default();
        // keep growing this tree until we reach the specified depth
        while rings < params.limit {
            // if we have no more growable nodes then end early
            if self.growable.is_empty() {
                break;
            }
            // start crawling this tree's initial nodes
            for hash in &self.growable {
                // try to grow the tree from this node
                self.grow_from_node(user, *hash, &ring, params, shared)
                    .await?;
            }
            // get any data missing from any associations
            self.get_association_nodes(user, &mut ring, shared).await?;
            // merge our current ring but not its relationships into our tree
            self.merge_ring(&mut ring)?;
            // increment our rings counter
            rings += 1;
        }
        // replace our relationships in our tree
        self.add_relationships(ring.relationships);
        // return our newly added nodes
        Ok(ring.added)
    }

    /// Trim a new to only new nodes that have not already been sent
    ///
    /// # Arguments
    ///
    /// * `grown` - The nodes that we grew on this tree
    /// * `added` - The nodes that were newly added to this tree
    pub fn trim(&mut self, grown: Vec<u64>, added: DashSet<u64>) {
        // drop any info from nodes that we have already sent
        self.data_map.retain(|key, _| added.contains(key));
        self.branches
            .retain(|key, _| added.contains(key) || grown.contains(key));
    }

    /// Save this trees info to the db
    pub async fn save(&mut self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        db::trees::save(&user, self, shared).await
    }

    /// Load an existing tree
    pub async fn load(user: &User, id: &Uuid, shared: &Shared) -> Result<Self, ApiError> {
        // Load this tree from the db
        db::trees::load(&user, id, shared).await
    }

    /// Clear any info that we don't send to users
    pub fn clear_non_user_facing(&mut self) {
        // clear all non user facing info
        self.related.clear();
        self.sent.clear();
        self.groups.clear();
    }
}

impl<S> FromRequestParts<S> for TreeParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}
