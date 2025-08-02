//! Build out trees based on data in Thorium's database

use std::collections::HashSet;

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use uuid::Uuid;

use super::db;
use crate::bad;
use crate::models::trees::TreeTags;
use crate::models::{
    Sample, Tree, TreeNode, TreeNodeData, TreeParams, TreeQuery, TreeSupport, User,
};
use crate::utils::{ApiError, Shared};

impl TreeQuery {
    /// Make sure our query is not empty and error if it is
    pub fn check_empty(&self) -> Result<(), ApiError> {
        if self.samples.is_empty() && self.tags.is_empty() {
            bad!("Initial starting data must be set!".to_owned())
        } else {
            Ok(())
        }
    }
}

impl TreeNode {
    /// Gather all of the children for this node
    pub async fn gather_children(
        &self,
        user: &User,
        tree: &Tree,
        shared: &Shared,
    ) -> Result<Vec<TreeNode>, ApiError> {
        // gather children for this new data
        match &self.data {
            TreeNodeData::Sample(sample) => sample.gather_children(user, shared).await,
            TreeNodeData::Tag(tags) => tags.gather_children(user, shared).await,
        }
    }
}

impl Tree {
    /// Build or get an existing tree from params
    pub async fn from_query(
        user: &User,
        query: TreeQuery,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // make sure we have some initial starting data for this query
        query.check_empty()?;
        // start with a default tree
        let mut tree = Tree::default();
        // get our initial data
        // TODO this in parallel?
        let samples = Sample::gather_initial(user, &query, shared).await?;
        let tags = TreeTags::gather_initial(user, &query, shared).await?;
        // add our initial samples
        for sample in samples {
            tree.add_initial(sample);
        }
        // add our initial tags
        for tag in tags {
            tree.add_initial(tag);
        }
        Ok(tree)
    }

    /// Build a tree based on data in Thorium's database
    pub async fn grow(
        &mut self,
        user: &User,
        params: &TreeParams,
        shared: &Shared,
    ) -> Result<HashSet<u64>, ApiError> {
        // keep a list of children to add
        let mut children = Vec::with_capacity(10);
        // track how many times this tree has grown
        let mut rings = 0;
        // keep a set of the newly added nodes
        let mut added = HashSet::with_capacity(self.growable.len() * 3);
        // keep growing this tree until we reach the specified depth
        while rings < params.limit {
            // start crawling this tree's initial nodes
            for hash in &self.growable {
                // get this nodes info
                if let Some(initial) = self.data_map.get(hash) {
                    // get this intial nodes children
                    let node_children = initial.gather_children(user, &self, shared).await?;
                    // extend our children list with our node children
                    children.push((*hash, node_children));
                }
            }
            // reset our growable nodes
            self.growable.drain(..);
            // add all of our children to our tree
            for (parent_hash, node_children) in children.drain(..) {
                // add all of the children for this specific node
                for child in node_children {
                    // add our child node
                    if let Some(child_hash) = self.add_node(child, vec![parent_hash]) {
                        // add this child node to our set of growable nodes
                        self.growable.push(child_hash);
                        // add this to our newly added nodes
                        added.insert(child_hash);
                        // add this node to our sent nodes
                        self.sent.insert(child_hash);
                    }
                }
            }
            // if we have no more growable nodes then end early
            if self.growable.is_empty() {
                return Ok(added);
            }
            // increment our rings counter
            rings += 1;
        }
        Ok(added)
    }

    /// Trim a new to only new nodes that have not already been sent
    pub fn trim(&mut self, added: HashSet<u64>) {
        // drop any info from nodes that we have already sent
        self.data_map.retain(|key, _| added.contains(key));
        self.branches.retain(|key, _| added.contains(key));
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
