//! The dry run scheduler for Thorium

use chrono::prelude::*;
use std::collections::{BTreeMap, HashSet};
use hashbrown::HashMap;
use thorium::models::{NodeHealth, SystemSettings};
use thorium::{Conf, Error, Thorium};
use tracing::{instrument};

use super::{
    Allocatable, AllocatableUpdate, NodeAllocatableUpdate, NodeResources, Scheduler, Spawned,
    WorkerDeletion,
};
use crate::libs::scaler::ErrorOutKinds;
use crate::libs::{BanSets, Cache, Tasks};

/// A pretend node in a dry run cluster
#[derive(Debug, Clone)]
pub struct DryRunNode {
    /// The health of this node
    health: NodeHealth,
    /// The resources this node has
    pub resources: NodeResources,
    /// The workers in different states on this node
    pub workers: Vec<Spawned>,
}

impl DryRunNode {
    /// Create a dry run node
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this node
    fn new(name: String) -> Self {
        // create our default nodes resources
        let mut resources = NodeResources::new(name);
        // give our pretend node 32 cores and 64 GiB of ram
        resources.available.cpu = 32000;
        resources.available.memory = 65536;
        // assume 128 GiB of ephemeral storage
        resources.available.ephemeral_storage = 131_072;
        // assume 100 worker slots
        resources.available.worker_slots = 100;
        // set our total resources
        resources.total = resources.available.clone();
        // we can spawn at most 2 items per loop
        resources.spawn_slots = 2;
        // build a pretend dry run node
        DryRunNode {
            health: NodeHealth::Healthy,
            resources,
            workers: Vec::with_capacity(10),
        }
    }
}

/// A dry run scheduler for testing scheduling
#[derive(Debug, Clone)]
pub struct DryRun {
    /// The different fake nodes in our dry run cluster
    pub nodes: HashMap<String, DryRunNode>,
}

impl Default for DryRun {
    /// Create a default 3 node pretend cluster
    fn default() -> Self {
        // build a map to store our nodes
        let mut nodes = HashMap::with_capacity(3);
        // add these nodes to our cluster
        for i in 0..3 {
            // build a name for this node
            let name = format!("Node-{i}");
            // create and insert our node
            nodes.insert(name.clone(), DryRunNode::new(name));
        }
        // build our dry run cluster
        DryRun { nodes }
    }
}

impl DryRun {
    /// Create some DryRun cluster schedulers
    pub fn new(schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>, conf: &Conf) {
        // create a default cluster for each k8s cluster
        for (name, _) in conf.thorium.scaler.k8s.clusters.iter() {
            schedulers.insert(name.clone(), Box::new(DryRun::default()));
        }
    }
}

/// The methods required to be used as a Thorium scheduler
#[async_trait::async_trait]
impl Scheduler for DryRun {
    /// Determine when a task should be executed again
    ///
    /// # Arguments
    ///
    /// * `task` - The task we want to run again
    fn task_delay(&self, task: &Tasks) -> i64 {
        // get how long to wait before executing this task again
        match task {
            Tasks::ZombieJobs => 30,
            Tasks::LdapSync => 600,
            Tasks::CacheReload => 600,
            Tasks::Resources => 120,
            Tasks::UpdateRuntimes => 300,
            Tasks::Cleanup => 25,
            Tasks::DecreaseFairShare => 600,
        }
    }

    /// Schedulers need to be able to determine how many resources they have
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `settings` - The current Thorium system settings
    /// * `span` - The span to log traces under
    #[instrument(name = "Scheduler<K8s>::resources_available", skip_all, err(Debug))]
    async fn resources_available(
        &mut self,
        _thorium: &Thorium,
        _settings: &SystemSettings,
    ) -> Result<AllocatableUpdate, Error> {
        // start with a default cluster resource update
        let mut update = AllocatableUpdate::default();
        // get all of our nodes resources and workers
        for (name, node) in self.nodes.iter() {
            // handle this nodes health correctly
            match node.health {
                NodeHealth::Unhealthy => {
                    update.removes.insert(name.clone());
                }
                NodeHealth::Disabled(_) => {
                    update.removes.insert(name.clone());
                }
                NodeHealth::Registered => {
                    // build a node update
                    let node_update = NodeAllocatableUpdate::new(
                        node.resources.available.clone(),
                        node.resources.total.clone(),
                    );
                    // add this update
                    update.nodes.insert(name.clone(), node_update);
                }
                NodeHealth::Healthy => {
                    // build a node update
                    let mut node_update = NodeAllocatableUpdate::new(
                        node.resources.available.clone(),
                        node.resources.total.clone(),
                    );
                    // add our active workers
                    node_update
                        .active
                        .extend(node.workers.iter().map(|spawn| spawn.name.clone()));
                    // add this update
                    update.nodes.insert(name.clone(), node_update);
                }
            }
        }
        Ok(update)
    }

    /// Schedulers need to be able to prepare their environment for new users and groups
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the cluster we are setting up
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    async fn setup(
        &mut self,
        _name: &str,
        _cache: &Cache,
        _bans: &mut BanSets,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Schedulers need to be able to sync their environment to the contents of
    /// a new cache
    ///
    /// * `name` - The name of the cluster we are setting up
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    /// * `span` - The span to log traces under
    async fn sync_to_new_cache(
        &mut self,
        _name: &str,
        _cache: &Cache,
        _bans: &mut BanSets,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Schedulers need to be able to scale resources up based on requisitions
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium
    /// * `req` - The requisition to scale
    /// * `scale` - What to scale this requisition to
    async fn spawn(
        &mut self,
        _cache: &Cache,
        spawns: &BTreeMap<DateTime<Utc>, Vec<Spawned>>,
    ) -> HashMap<String, Error> {
        // track our errors
        let mut errors = HashMap::default();
        // crawl over the spawns and spawn them
        for (_, spawns) in spawns.iter() {
            // crawl over the spawns in this spawn group
            for spawn in spawns {
                // get the node for this spawn
                let node = match self.nodes.get_mut(&spawn.node) {
                    Some(node) => node,
                    None => {
                        // build the error to set
                        let error = Error::new(format!("Failed to find node {}", spawn.node));
                        // set our error
                        errors.insert(spawn.name.clone(), error);
                        // skip to the next spawn
                        continue;
                    }
                };
                // make sure we have enough resources for this spawn
                if node.resources.available.enough(&spawn.resources) {
                    // consume these resources
                    node.resources.available.consume(&spawn.resources, 1);
                    // add this spawned worker to our active set
                    node.workers.push(spawn.clone());
                } else {
                    // build the error to set
                    let error =
                        Error::new(format!("Node {} has insufficent resources", spawn.node));
                    // set our error
                    errors.insert(spawn.name.clone(), error);
                    // skip to the next spawn
                    continue;
                }
            }
        }
        errors
    }

    /// Schedulers need to be able to scale resources down based on requisitions
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium
    /// * `scaledowns` - The workers to scale down
    /// * `span` - The span to log traces under
    async fn delete(
        &mut self,
        _thorium: &Thorium,
        _cache: &Cache,
        scaledowns: Vec<Spawned>,
    ) -> Vec<WorkerDeletion> {
        // track our worker deletions
        let mut deletes = Vec::with_capacity(scaledowns.len());
        // crawl over our scale downs and remove them from nodes
        for scale_down in scaledowns {
            // try to get this workers node
            let node = match self.nodes.get_mut(&scale_down.node) {
                Some(node) => node,
                None => {
                    // build the error to set
                    let error = Error::new(format!("Failed to find node {}", scale_down.node));
                    // build the worker delete object
                    let delete = WorkerDeletion::Error {
                        delete: scale_down,
                        error,
                    };
                    // add this failed delete
                    deletes.push(delete);
                    // skip to the next spawn
                    continue;
                }
            };
            // get this nodes worker
            node.workers.retain(|spawn| spawn.name != scale_down.name);
            // release this workers resources
            node.resources.available += scale_down.resources.clone();
            // build our delete object
            let delete = WorkerDeletion::Deleted(scale_down);
            // add this delete to our list
            deletes.push(delete);
        }
        deletes
    }

    /// Clears out any failed or terminal resources in specified groups
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `allocatable` - The currently allocatable resources by this scaler
    /// * `groups` - The groups to clear failing resources from
    /// * `failed` - A set of failed workers to add too
    /// * `terminal` - A set of terminal workers to add too
    /// * `error_out` - The pods whose workers we should fail out instead of just resetting
    async fn clear_terminal(
        &mut self,
        _thorium: &Thorium,
        _allocatable: &Allocatable,
        _groups: &HashSet<String>,
        _failed: &mut HashSet<String>,
        _terminal: &mut HashSet<String>,
        _error_out: &mut HashSet<ErrorOutKinds>,
    ) -> Result<(), Error> {
        Ok(())
    }
}
