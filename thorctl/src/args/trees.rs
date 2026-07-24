//! Arguments for tree-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use std::collections::{BTreeMap, BTreeSet};

use clap::Parser;
use thorium::models::{TreeOpts, TreeQuery};
use thorium::Error;
use uuid::Uuid;

/// The commands to send to the trees task handler
#[derive(Parser, Debug)]
pub enum Trees {
    /// Delete the descendant entities under a set of initial tree nodes
    #[clap(version, author)]
    Delete(DeleteTree),
}

/// A command to delete the descendant entities under some initial tree nodes
///
/// The tree is built from the given initial nodes and traversed downward only.
/// An entity is deleted only if every one of its parents is also being deleted
/// (or is an initial node); any entity still reachable from outside the deletion
/// scope is preserved, as are all of its descendants.
#[derive(Parser, Debug, Clone)]
pub struct DeleteTree {
    /// The entity ids to use as initial nodes to traverse down from
    #[clap(long)]
    pub entities: Vec<Uuid>,
    /// The sample sha256s to use as initial nodes to traverse down from
    #[clap(long)]
    pub samples: Vec<String>,
    /// The repo urls to use as initial nodes to traverse down from
    #[clap(long)]
    pub repos: Vec<String>,
    /// The tag filters to use as initial nodes, formatted as `KEY=VALUE`
    ///
    /// A tag with multiple values can be specified as `KEY=VALUE1=VALUE2`. All
    /// tag filters are combined into a single filter that must all match.
    #[clap(long)]
    pub tags: Vec<String>,
    /// The groups to restrict the tree to
    #[clap(short, long)]
    pub groups: Vec<String>,
    /// The max depth (number of growth rings) to build the tree to
    ///
    /// By default the entire tree is materialized so that every parent of every
    /// node is discovered; capping the depth risks missing an external parent
    /// and deleting an entity that should be preserved.
    #[clap(long)]
    pub depth: Option<usize>,
    /// Preview the entities that would be deleted without actually deleting them
    #[clap(long)]
    pub dry_run: bool,
    /// Skip the confirmation prompt before deleting
    #[clap(long, visible_alias = "yes")]
    pub force: bool,
    /// Log why each entity in the tree is or isn't being deleted
    #[clap(long)]
    pub debug: bool,
}

impl DeleteTree {
    /// Parse our `KEY=VALUE` tag args into a single tag filter map
    ///
    /// # Errors
    ///
    /// Returns an error if a tag arg has no `=` delimiter or an empty key
    fn parse_tags(&self) -> Result<BTreeMap<String, BTreeSet<String>>, Error> {
        // build a map to collect our tag filters into
        let mut filters: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        // parse each raw tag arg into a key and its values
        for raw in &self.tags {
            // split this tag on its delimiter
            let mut split = raw.split('=');
            // the first segment is the tag key
            let key = split.next().filter(|key| !key.is_empty()).ok_or_else(|| {
                Error::new(format!(
                    "Invalid tag '{raw}': tags must be formatted as 'KEY=VALUE'"
                ))
            })?;
            // the remaining segments are this key's values
            let values = split.map(str::to_owned).collect::<BTreeSet<String>>();
            // make sure we got at least one value for this key
            if values.is_empty() {
                return Err(Error::new(format!(
                    "Invalid tag '{raw}': tags must be formatted as 'KEY=VALUE'"
                )));
            }
            // add this key's values to our filter map
            filters.entry(key.to_owned()).or_default().extend(values);
        }
        Ok(filters)
    }

    /// Build a [`TreeQuery`] from our initial node args
    ///
    /// # Errors
    ///
    /// Returns an error if no initial nodes were specified or a tag is malformed
    pub fn to_query(&self) -> Result<TreeQuery, Error> {
        // make sure the user gave us at least one initial node to build a tree from
        if self.entities.is_empty()
            && self.samples.is_empty()
            && self.repos.is_empty()
            && self.tags.is_empty()
        {
            return Err(Error::new(
                "At least one initial node (--entities/--samples/--repos/--tags) must be specified"
                    .to_owned(),
            ));
        }
        // parse our tag filters into a single filter map
        let tag_filters = self.parse_tags()?;
        // only add a tag filter to our query if we actually have one
        let tags = if tag_filters.is_empty() {
            Vec::new()
        } else {
            vec![tag_filters]
        };
        // build our tree query from our initial node args
        Ok(TreeQuery {
            groups: self.groups.clone(),
            samples: self.samples.clone(),
            repos: self.repos.clone(),
            entities: self.entities.clone(),
            tags,
            ..Default::default()
        })
    }

    /// Build the [`TreeOpts`] to use when building the tree
    ///
    /// We always gather parents so that every external parent is discovered and
    /// we skip gathering related nodes since they are not descendants.
    #[must_use]
    pub fn to_opts(&self) -> TreeOpts {
        // grow the entire tree by default so no external parents are missed
        TreeOpts::default()
            .limit(self.depth.unwrap_or(usize::MAX))
            .gather_parents(true)
            .gather_related(false)
    }
}
