use chrono::prelude::*;
use chrono::{Duration, Utc};
use hashbrown::HashMap;
use std::collections::BTreeMap;
use thorium::models::{Image, Pools, Requisition, Resources, SpawnedUpdate, Worker};
use thorium::same;

use crate::libs::helpers;

/// A currently active and spawned resource
#[derive(Debug, Clone)]
pub struct Spawned {
    /// The req that was tied to this pod
    pub req: Requisition,
    /// The cluster this req was spawned on
    pub cluster: String,
    /// The node this req was spawned on
    pub node: String,
    /// The unique name for this resource
    pub name: String,
    /// The resources in use by this spawned resource
    pub resources: Resources,
    /// What pool this worker was spawned with
    pub pool: Pools,
    /// Whether this resource needs to be spawned
    pub spawn: bool,
    /// Whether this resource has been told to scale down yet or not
    pub scaled_down: bool,
    /// When this resource can be scaled down to prevent flapping
    pub down_scalable: DateTime<Utc>,
}

impl Spawned {
    /// Create a new spawned object
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster this worker was spawned on
    /// * `node` - The node this worker was spawned on
    /// * `req` - The requisition that led to this worker
    /// * `image` - The image for this worker
    /// * `pool` - The pool this worker was spawned in
    pub fn new<T: Into<String>>(
        cluster: T,
        node: T,
        req: Requisition,
        image: &Image,
        pool: Pools,
    ) -> Self {
        // generate a random name
        let append = helpers::gen_string(8);
        let name = format!("{}-{}-{}", &req.pipeline, &req.stage, append);
        // allow for 3x this images standard execution time + 25%
        let single_run_budget = 3.0 * image.runtime + (3.0 * image.runtime * 0.25);
        // calculate when this worker can be safely scaled down
        let down_scalable = Utc::now() + Duration::seconds(single_run_budget.ceil() as i64);
        // create our spawned object
        Spawned {
            req,
            cluster: cluster.into(),
            node: node.into(),
            name,
            resources: image.resources.clone(),
            pool,
            spawn: true,
            scaled_down: false,
            down_scalable,
        }
    }
}

impl PartialEq<Worker> for Spawned {
    /// Check if a worker and a Spawned worker are the same
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to compare against
    fn eq(&self, other: &Worker) -> bool {
        // make sure this is for the correct cluster
        same!(self.name, other.name);
        same!(self.cluster, other.cluster);
        same!(self.node, other.node);
        true
    }
}

impl PartialEq<Spawned> for Spawned {
    /// Check if a worker and a Spawned worker are the same
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to compare against
    fn eq(&self, other: &Spawned) -> bool {
        // make sure this is for the correct cluster
        same!(self.name, other.name);
        same!(self.cluster, other.cluster);
        same!(self.node, other.node);
        true
    }
}

impl From<&Spawned> for SpawnedUpdate {
    /// Cast our spawned worker to a spawned update object
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to build a spawned update for
    fn from(spawned: &Spawned) -> Self {
        SpawnedUpdate {
            req: spawned.req.clone(),
            node: spawned.node.clone(),
            name: spawned.name.clone(),
            pool: spawned.pool,
            resources: spawned.resources.clone(),
            scaled_down: spawned.scaled_down,
        }
    }
}

/// A Map of requisitions along with the number of pods possible to spawn
#[derive(Default, Debug, Clone)]
pub struct ReqMap {
    /// The resources to spawn
    pub spawns: HashMap<String, BTreeMap<DateTime<Utc>, Vec<Spawned>>>,
    /// The spawned resources to scale down
    pub scale_down: HashMap<String, Vec<Spawned>>,
}

impl ReqMap {
    /// Adds a resource to be scaled down
    ///
    /// # Arguments
    ///
    /// * `spawn` - The spawn to set to be scaled down
    pub fn scale_down(&mut self, spawn: Spawned) {
        // get an entry to the correct clusters scale down list
        let entry = self
            .scale_down
            .entry(spawn.cluster.clone())
            .or_insert_with(|| Vec::with_capacity(1));
        // add this spawn to this clusters scale down list
        entry.push(spawn);
    }
}
