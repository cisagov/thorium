//! Directly schedules jobs onto nodes with reactors

use chrono::prelude::*;
use std::collections::{BTreeMap, HashSet};
use hashbrown::HashMap;
use thorium::models::{
    ImageScaler, Node, NodeListParams, SystemSettings, WorkerStatus, WorkerUpdate,
};
use thorium::{Conf, Error, Thorium};
use tracing::{event, instrument, Level};

use super::{
    Allocatable, AllocatableUpdate, NodeAllocatableUpdate, Scheduler, Spawned, WorkerDeletion,
};
use crate::libs::scaler::ErrorOutKinds;
use crate::libs::{BanSets, Cache, Tasks};

#[derive(Debug)]
pub struct Direct {
    /// The name of this cluster
    pub cluster: String,
    /// The scaler we are directly scheduling workers for
    pub scaler: ImageScaler,
    /// The workers that have been scaled down
    pub scaled_down: Vec<Spawned>,
    /// The workers that we failed to patch and need to retry later
    pub retry: Vec<Spawned>,
}

impl Direct {
    /// Create a new windows direct scheduler
    ///
    /// # Arguments
    ///
    /// * `schedulers` - The map of schedulers to add this scheduler too
    /// * `conf` - The config to use when building our windows schedulers
    pub fn build_windows(schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>, conf: &Conf) {
        // crawl our windows clusters
        for cluster in conf.thorium.scaler.windows.clusters.iter() {
            // build this scheduler
            let direct = Direct {
                cluster: cluster.to_owned(),
                scaler: ImageScaler::Windows,
                scaled_down: Vec::default(),
                retry: Vec::default(),
            };
            schedulers.insert(cluster.to_owned(), Box::new(direct));
        }
    }

    /// Create a new bare metal direct scheduler
    ///
    /// # Arguments
    ///
    /// * `schedulers` - The map of schedulers to add this scheduler too
    /// * `conf` - The config to use when building our bare metal schedulers
    pub fn build_bare_metal(
        schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>,
        conf: &Conf,
    ) {
        // crawl our kvm clusters
        for (cluster, _) in conf.thorium.scaler.bare_metal.clusters.iter() {
            // build this scheduler
            let direct = Direct {
                cluster: cluster.to_owned(),
                scaler: ImageScaler::BareMetal,
                scaled_down: Vec::default(),
                retry: Vec::default(),
            };
            schedulers.insert(cluster.to_owned(), Box::new(direct));
        }
    }

    /// Create a new kvm direct scheduler
    ///
    /// # Arguments
    ///
    /// * `schedulers` - The map of schedulers to add this scheduler too
    /// * `conf` - The config to use when building our kvm schedulers
    pub fn build_kvm(schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>, conf: &Conf) {
        // crawl our kvm clusters
        for (cluster, _) in conf.thorium.scaler.kvm.clusters.iter() {
            // build this scheduler
            let direct = Direct {
                cluster: cluster.to_owned(),
                scaler: ImageScaler::Kvm,
                scaled_down: Vec::default(),
                retry: Vec::default(),
            };
            schedulers.insert(cluster.to_owned(), Box::new(direct));
        }
    }

    /// Check if any of our pending scale downs have completed
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    #[instrument(name = "Direct::check_deletes", skip_all)]
    pub async fn check_deletes(&mut self, thorium: &Thorium) -> Vec<WorkerDeletion> {
        // track the workers we have deleted this loop
        let mut deleted = Vec::with_capacity(self.scaled_down.len());
        // get the nodes we have pending deletes for
        let pending_nodes = self.scaled_down.iter().map(|spawn| spawn.node.clone());
        // build the node list params for this cluster
        let params = NodeListParams::default()
            .cluster(&self.cluster)
            .scaler(self.scaler)
            .nodes(pending_nodes);
        // build a list of the workers we checking
        let mut temp = Vec::with_capacity(self.scaled_down.len());
        // get the nodes for this cluster
        match thorium.system.list_node_details(&params).await {
            Ok(mut cursor) => {
                // crawl this cursor until it has been exhausted
                loop {
                    // crawl over all nodes on this page
                    for node in cursor.data.drain(..) {
                        // cast our list of workers into a hashset
                        let alive = node
                            .workers
                            .into_iter()
                            .map(|(name, _)| name)
                            .collect::<HashSet<String>>();
                        // determine which workers have died
                        for worker in self.scaled_down.drain(..) {
                            // if this worker is still listed then its not dead yet
                            if alive.contains(&worker.name) {
                                temp.push(worker);
                            } else {
                                // this worker has been deleted
                                deleted.push(WorkerDeletion::Deleted(worker));
                            }
                        }
                        // swap our temp list and our main list
                        std::mem::swap(&mut self.scaled_down, &mut temp);
                    }
                    // check if this cursor is exhausted
                    if cursor.exhausted() || self.scaled_down.is_empty() {
                        // this cursor is exhausted so break out of our loop
                        break;
                    }
                    // this cursor has more data so get the next page
                    if let Err(error) = cursor.refill().await {
                        // log this error and break out
                        event!(Level::ERROR, error = true, error_msg = error.to_string());
                        break;
                    }
                }
            }
            Err(error) => {
                // cast our error to a string
                let error_msg = error.to_string();
                // log this error
                event!(Level::ERROR, error = true, error_msg = error_msg);
            }
        }
        deleted
    }
}

/// Determine the amount of free resources on a node based on its workers
fn update_node(node: Node, update: &mut AllocatableUpdate) {
    // build or node update
    let mut node_update = NodeAllocatableUpdate::new(node.resources, node.resources);
    // get a mutable ref to our resources for this node
    let resources = &mut node_update.available;
    // crawl over the workers on this node
    for (_, worker) in node.workers {
        // consume the resources for this worker
        resources.cpu = resources.cpu.saturating_sub(worker.resources.cpu);
        resources.memory = resources.memory.saturating_sub(worker.resources.memory);
        resources.ephemeral_storage = resources
            .ephemeral_storage
            .saturating_sub(worker.resources.ephemeral_storage);
        resources.nvidia_gpu = resources
            .nvidia_gpu
            .saturating_sub(worker.resources.nvidia_gpu);
        resources.amd_gpu = resources.amd_gpu.saturating_sub(worker.resources.amd_gpu);
        // add this worker to our active list
        node_update.active.insert(worker.name.clone());
    }
    // add this node to our update
    update.nodes.insert(node.name, node_update);
}

/// The methods required to be used as a Thorium scheduler
#[async_trait::async_trait]
impl Scheduler for Direct {
    /// Determine when a task should be executed again
    ///
    /// # Arguments
    ///
    /// * `task` - The task we want to run again
    fn task_delay(&self, task: &Tasks) -> i64 {
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
    #[instrument(name = "Scheduler<Direct>::resources_available", skip_all, fields(cluster = &self.cluster), err(Debug))]
    async fn resources_available(
        &mut self,
        thorium: &Thorium,
        _settings: &SystemSettings,
    ) -> Result<AllocatableUpdate, Error> {
        // build the node list params for this cluster
        let params = NodeListParams::default()
            .cluster(&self.cluster)
            .scaler(self.scaler);
        // get the nodes for this cluster
        let mut cursor = thorium.system.list_node_details(&params).await?;
        // build the updated resources object
        let mut update = AllocatableUpdate::default();
        // crawl this cursor until it has been exhausted
        loop {
            // crawl over all nodes on this page
            for node in cursor.data.drain(..) {
                // add this nodes resources to our cluster
                update_node(node, &mut update);
            }
            // check if this cursor is exhausted
            if cursor.exhausted() {
                // this cursor is exhausted so break out of our loop
                break;
            }
            // this cursor has more data so get the next page
            cursor.refill().await?;
        }
        Ok(update)
    }

    /// Schedulers need to be able to prepare their environment for new users and groups
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    /// * `span` - The span to log traces under
    #[instrument(name = "Scheduler<Direct>::setup", skip(_cache, _bans), err(Debug))]
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
        _spawns: &BTreeMap<DateTime<Utc>, Vec<Spawned>>,
    ) -> HashMap<String, Error> {
        // track any errors we encounter when registering workers
        HashMap::default()
    }

    /// Schedulers need to be able to scale resources down based on requisitions
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium
    /// * `scaledowns` - The workers to scale down
    #[instrument(name = "Scheduler<Direct>::delete", skip_all)]
    async fn delete(
        &mut self,
        thorium: &Thorium,
        _cache: &Cache,
        scaledowns: Vec<Spawned>,
    ) -> Vec<WorkerDeletion> {
        // check if any of our pending deletes have completed
        let deleted = self.check_deletes(thorium).await;
        // build the update to apply to all of these workers
        let update = WorkerUpdate::new(WorkerStatus::Shutdown);
        // crawl over our scale downs and patch their status to be shutdown
        for worker in scaledowns {
            // try to patch this workers status
            match thorium.system.update_worker(&worker.name, &update).await {
                Ok(_) => self.scaled_down.push(worker),
                Err(error) => {
                    // log that an error occurred
                    event!(Level::ERROR, error = true, error_msg = error.to_string());
                    // add this delete to be retried later
                    self.retry.push(worker);
                }
            }
        }
        deleted
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
    #[instrument(name = "Scheduler<Direct>::clear_terminal", skip_all)]
    async fn clear_terminal(
        &mut self,
        thorium: &Thorium,
        allocatable: &Allocatable,
        _groups: &HashSet<String>,
        _failed: &mut HashSet<String>,
        terminal: &mut HashSet<String>,
        _error_out: &mut HashSet<ErrorOutKinds>,
    ) -> Result<(), Error> {
        // get a list of the currently spawned workers
        let spawned = allocatable.spawn_names();
        println!("clusters -> {:#?}", allocatable.clusters);
        println!("spawned -> {:#?}", spawned);
        // build our node list params
        let params = NodeListParams::default().scaler(self.scaler).limit(500);
        // build a cursor over our current nodes
        let mut cursor = thorium.system.list_node_details(&params).await?;
        // keep crawling nodes until we have scanned them all
        while !cursor.data.is_empty() {
            println!("workers -> {:#?}", cursor.data);
            // condense our list of workers to a flat iter
            let workers = cursor.data.drain(..).flat_map(|node| node.workers);
            // prune any jobs that we know still exist
            let workers = workers
                .filter(|(name, _)| !spawned.contains(&name))
                .map(|(name, _)| name);
            // extend our terminal worker list
            terminal.extend(workers);
            // if our cursor is exhausted then break
            if cursor.exhausted() {
                break;
            }
            // get the next page of data
            cursor.refill().await?;
        }
        Ok(())
    }
}
