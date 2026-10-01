//! Directly schedules jobs onto nodes with reactors

use chrono::prelude::*;
use hashbrown::HashMap;
use std::collections::{BTreeMap, HashSet};
use thorium::conf::BurstableNodeResources;
use thorium::models::{
    ImageScaler, Node, NodeListParams, SystemSettings, WorkerStatus, WorkerUpdate,
};
use thorium::{Conf, Error, Thorium};
use tracing::{Level, event, instrument};

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
    /// Get the workers `Direct::delete` patches to Shutdown this pass
    ///
    /// This is the worker list `Direct::delete` iterates extracted so it can be unit tested.
    /// Workers whose Shutdown patch failed on an earlier pass are drained from `self.retry` and
    /// patched again along with this pass's new scale downs.
    ///
    /// # Arguments
    ///
    /// * `scaledowns` - The new workers to scale down this pass
    fn shutdown_targets(&mut self, scaledowns: Vec<Spawned>) -> Vec<Spawned> {
        // start with any workers whose Shutdown patch failed on an earlier pass
        let mut targets = std::mem::take(&mut self.retry);
        // add our new scale downs
        targets.extend(scaledowns);
        targets
    }

    /// Track the result of patching a worker to Shutdown in `Direct::delete`
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker we tried to patch to Shutdown
    /// * `result` - The result of the Shutdown patch
    fn record_shutdown_result(&mut self, worker: Spawned, result: Result<(), Error>) {
        match result {
            Ok(()) => self.scaled_down.push(worker),
            Err(error) => {
                // log that an error occured
                event!(Level::ERROR, error = true, error_msg = error.to_string());
                // add this delete to be retried later
                self.retry.push(worker);
            }
        }
    }

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
                    // classify the pending scale downs against the nodes on this page
                    Self::classify_page_deletes(
                        cursor.data.drain(..).collect(),
                        &mut self.scaled_down,
                        &mut temp,
                        &mut deleted,
                    );
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

    /// Classify a page of nodes worth of pending scale downs into survivors and deletions
    ///
    /// This is the per-node classification loop in [`Direct::check_deletes`] (it never needs
    /// `self`) extracted so its logic can be unit tested without a live Thorium client. Each
    /// pending scale down is only checked against its own node, so workers on other nodes stay
    /// pending until their own node is checked.
    ///
    /// # Arguments
    ///
    /// * `nodes` - The nodes on this page of the cursor
    /// * `scaled_down` - The pending scale downs to classify, drained and refilled with survivors
    /// * `temp` - Scratch space for survivors, swapped with `scaled_down` per node
    /// * `deleted` - The list of workers reported deleted, appended to in place
    fn classify_page_deletes(
        nodes: Vec<Node>,
        scaled_down: &mut Vec<Spawned>,
        temp: &mut Vec<Spawned>,
        deleted: &mut Vec<WorkerDeletion>,
    ) {
        // crawl over all nodes on this page
        for node in nodes {
            // cast our list of workers into a hashset
            let alive = node
                .workers
                .into_iter()
                .map(|(name, _)| name)
                .collect::<HashSet<String>>();
            // determine which workers on this node have died
            for worker in scaled_down.drain(..) {
                // workers on other nodes or still listed on this node are not dead yet
                if worker.node != node.name || alive.contains(&worker.name) {
                    temp.push(worker);
                } else {
                    // this worker has been deleted
                    deleted.push(WorkerDeletion::Deleted(worker));
                }
            }
            // swap our temp list and our main list
            std::mem::swap(scaled_down, temp);
        }
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

    /// Whether we have scale downs to check on or Shutdown patches to retry
    fn has_pending_deletes(&self) -> bool {
        !self.scaled_down.is_empty() || !self.retry.is_empty()
    }

    /// Schedulers need to be able to determine how many resources they have
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `settings` - The current Thorium system settings
    /// * `span` - The span to log traces under
    /// * `config` - The burstable resources config to use for this cluster
    #[instrument(name = "Scheduler<Direct>::resources_available", skip_all, fields(cluster = &self.cluster), err(Debug))]
    async fn resources_available(
        &mut self,
        thorium: &Thorium,
        _settings: &SystemSettings,
        _config: &BurstableNodeResources,
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
        // crawl over our scale downs and any retries and patch their status to be shutdown
        for worker in self.shutdown_targets(scaledowns) {
            // try to patch this workers status
            let result = thorium
                .system
                .update_worker(&worker.name, &update)
                .await
                .map(|_| ());
            // track whether this worker was scaled down or needs to be retried
            self.record_shutdown_result(worker, result);
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
        // build our node list params
        let params = NodeListParams::default().scaler(self.scaler).limit(500);
        // build a cursor over our current nodes
        let mut cursor = thorium.system.list_node_details(&params).await?;
        // keep crawling nodes until we have scanned them all
        while !cursor.data.is_empty() {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use thorium::models::{NodeHealth, Pools, Requisition, Resources, Worker, WorkerStatus};

    /// Build a minimal worker record living on the given node
    ///
    /// # Arguments
    ///
    /// * `name` - The unique name of the worker
    /// * `node` - The node the worker is running on
    fn worker(name: &str, node: &str) -> Worker {
        // build a worker whose only fields that matter to classification are its name and node
        Worker {
            cluster: "cluster".to_owned(),
            node: node.to_owned(),
            scaler: ImageScaler::BareMetal,
            name: name.to_owned(),
            user: "user".to_owned(),
            group: "group".to_owned(),
            pipeline: "pipeline".to_owned(),
            stage: "stage".to_owned(),
            status: WorkerStatus::Running,
            spawned: Utc::now(),
            heart_beat: None,
            resources: Resources::default(),
            pool: Pools::Deadline,
            active: None,
        }
    }

    /// Build a node hosting the given workers
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the node
    /// * `workers` - The workers currently alive on this node
    fn node(name: &str, workers: Vec<Worker>) -> Node {
        // index the workers by name the way the node details endpoint returns them
        let workers = workers
            .into_iter()
            .map(|worker| (worker.name.clone(), worker))
            .collect();
        // build a healthy node carrying those workers
        Node {
            cluster: "cluster".to_owned(),
            name: name.to_owned(),
            health: NodeHealth::Healthy,
            resources: Resources::default(),
            workers,
            heart_beat: None,
        }
    }

    /// Build a pending scale down for a worker still running on the given node
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the worker being scaled down
    /// * `node` - The node the worker is running on
    fn scaled_down(name: &str, node: &str) -> Spawned {
        // build a spawned record for the worker with only the fields classification reads set
        Spawned {
            req: Requisition::new("user", "group", "pipeline", "stage"),
            cluster: "cluster".to_owned(),
            node: node.to_owned(),
            name: name.to_owned(),
            resources: Resources::default(),
            pool: Pools::Deadline,
            spawn: false,
            scaled_down: true,
            down_scalable: Utc::now(),
        }
    }

    /// A worker still alive on its OWN node is NOT reported deleted, regardless of which other
    /// nodes appear in the page: with W1 alive on node1 and W2 alive on node2, both stay pending
    /// after both nodes are processed. This keeps pending deletes spread across several nodes
    /// from being reported Deleted, and their resources freed, while the reactor still runs them.
    #[test]
    fn workers_alive_on_their_own_node_are_not_deleted() {
        // two workers are pending scale down, one on each of two different nodes
        let mut pending = vec![scaled_down("W1", "node1"), scaled_down("W2", "node2")];
        // both workers are still alive on their respective nodes in this page of node details
        let nodes = vec![
            node("node1", vec![worker("W1", "node1")]),
            node("node2", vec![worker("W2", "node2")]),
        ];
        // scratch space and the resulting deletions
        let mut temp = Vec::new();
        let mut deleted = Vec::new();
        // classify this page of nodes worth of pending scale downs
        Direct::classify_page_deletes(nodes, &mut pending, &mut temp, &mut deleted);
        // neither worker actually died, so nothing should be reported deleted
        assert!(
            deleted.is_empty(),
            "workers still alive on their own node must not be reported deleted"
        );
        // and both should remain pending until their reactors actually stop them
        assert_eq!(
            pending.len(),
            2,
            "both still-alive workers must remain pending scale downs"
        );
    }

    /// A worker that is genuinely gone from its node is reported deleted and is no longer
    /// pending.
    #[test]
    fn workers_absent_from_their_node_are_deleted() {
        // one worker is pending scale down on node1
        let mut pending = vec![scaled_down("W1", "node1")];
        // node1 reports no workers alive, so W1 has genuinely been shut down
        let nodes = vec![node("node1", vec![])];
        // scratch space and the resulting deletions
        let mut temp = Vec::new();
        let mut deleted = Vec::new();
        // classify this page of nodes worth of pending scale downs
        Direct::classify_page_deletes(nodes, &mut pending, &mut temp, &mut deleted);
        // W1 is gone from its node so it must be reported deleted
        assert_eq!(
            deleted.len(),
            1,
            "a worker absent from its own node must be reported deleted"
        );
        // and it must no longer be pending
        assert!(
            pending.is_empty(),
            "a deleted worker must not remain pending"
        );
    }

    /// Build an empty direct scheduler with no pending scale downs or retries
    fn direct() -> Direct {
        Direct {
            cluster: "cluster".to_owned(),
            scaler: ImageScaler::BareMetal,
            scaled_down: Vec::default(),
            retry: Vec::default(),
        }
    }

    /// A worker whose Shutdown patch failed is among the next pass's shutdown targets, so the
    /// scale down is retried rather than lost, which would leave the worker (whose allocator
    /// state already says it was scaled down) permanently un-preemptible.
    #[test]
    fn failed_shutdown_is_retried_next_pass() {
        let mut direct = direct();
        // the Shutdown patch for this worker failed
        direct.record_shutdown_result(scaled_down("w1", "node-a"), Err(Error::new("timeout")));
        // the next delete pass has no new scale downs
        let targets = direct.shutdown_targets(Vec::default());
        // the failed worker must be patched again
        assert!(
            targets.iter().any(|worker| worker.name == "w1"),
            "a worker whose Shutdown patch failed must be retried on the next pass"
        );
    }

    /// A worker whose Shutdown patch succeeded is tracked as scaled down and is not patched
    /// again.
    #[test]
    fn successful_shutdown_is_not_retried() {
        let mut direct = direct();
        // the Shutdown patch for this worker succeeded
        direct.record_shutdown_result(scaled_down("w1", "node-a"), Ok(()));
        // it is waiting to be deleted
        assert_eq!(direct.scaled_down.len(), 1);
        // and it is not patched again
        assert!(direct.shutdown_targets(Vec::default()).is_empty());
    }
}
