//! All resources that are allocatable by this scaler
//!
//! This can be resources across multiple clusters

use chrono::prelude::*;
use hashbrown::HashMap;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use thorium::conf::{FairShareWeights, IsRestricted, SpawnSlots, WorkerRestrictions};
use thorium::models::{
    Deadline, Image, ImageScaler, NodeListParams, Pools, Requisition, Resources, SpawnLimits,
    SpawnMap, SystemSettings, Worker, WorkerDeleteMap,
};
use thorium::{Conf, Error, Thorium};
use tracing::{Level, Span, event, instrument};

mod pool;

use crate::from_now;
use crate::libs::schedulers::ReqMap;
use crate::libs::{BanSets, Cache, Spawned};
pub use pool::{Pool, PoolFrees};

/// An update for a specific node
#[derive(Debug, Clone)]
pub struct NodeAllocatableUpdate {
    /// The current resources available on this node
    pub available: Resources,
    /// The total resources available on this node
    pub total: Resources,
    /// The workers that are active on this node
    pub active: HashSet<String>,
}

impl NodeAllocatableUpdate {
    /// Create a new node update
    ///
    /// # Arguments
    ///
    /// * `available` - The availabe resources to schedule on this node
    /// * `total` - The total resources on this node
    pub fn new(available: Resources, total: Resources) -> Self {
        NodeAllocatableUpdate {
            available,
            total,
            active: HashSet::default(),
        }
    }
}

/// An update to the allocatable resources for a single cluster
#[derive(Debug, Clone, Default)]
pub struct AllocatableUpdate {
    /// The node resources to update
    pub nodes: HashMap<String, NodeAllocatableUpdate>,
    /// The nodes to remove from this cluster
    pub removes: HashSet<String>,
}

/// The resources we can allocate across all clusters this scaler can see
#[derive(Debug)]
pub struct Allocatable {
    /// The Thorium config in use by this scaler
    conf: Conf,
    /// The type of scheduler in use
    scaler_type: ImageScaler,
    /// The size of our deadline window
    deadline_window: u64,
    /// The ban lists to follow for problematic groups/images
    pub bans: BanSets,
    /// The fair share resource pool
    pub fairshare_pool: Pool,
    /// The deadlines resource pool
    pub deadlines_pool: Pool,
    /// The restrictions for what nodes images can spawn on
    pub restrictions: WorkerRestrictions,
    /// A map of resources for all clusters we can schedule on
    pub clusters: BTreeMap<u64, HashMap<String, ClusterResources>>,
    /// whether any clusters are currently low on resources
    pub low_resources: bool,
    /// A map of what fair share resources have been used by each user
    pub fair_share: BTreeMap<u64, HashSet<String>>,
    /// The fair share weight settings to use
    weights: FairShareWeights,
    /// The current number of each type of unscoped requisition across all pools
    pub counts: HashMap<Requisition, i64>,
    /// The number of each image type that has been spawned
    pub image_counts: HashMap<String, HashMap<String, u64>>,
    /// The count of workers spawned under fairshare
    fair_share_counts: HashMap<Requisition, u64>,
    /// The max number of pods we can spawn per loop
    spawn_limit: usize,
    /// The containing our currently pending changes to worker allocations
    pub changes: ReqMap,
}

impl Allocatable {
    /// Create a new allocatable object
    ///
    /// # Arguments
    ///
    /// * `scaler_type` - The current scheduler in use
    /// * `conf` - The Thorium config
    /// * `cache` - A cache of info from Thorium
    pub fn new(
        scaler_type: ImageScaler,
        conf: &Conf,
        settings: &SystemSettings,
        cache: &Cache,
    ) -> Self {
        // start all users at 0 for fair share
        let mut starter_set = HashSet::default();
        // crawl over all users and add them
        for (username, _) in &cache.users {
            // add this user to our starter set
            starter_set.insert(username.clone());
        }
        // add our fair share starter set at 0
        let mut fair_share = BTreeMap::default();
        fair_share.insert(0, starter_set);
        // build a new allocatable object
        Allocatable {
            conf: conf.clone(),
            scaler_type,
            deadline_window: conf.thorium.scaler.deadline_window,
            bans: BanSets::new(scaler_type),
            fairshare_pool: Pool::setup_fairshare(settings),
            deadlines_pool: Pool::default(),
            restrictions: conf.thorium.scaler.restrictions(scaler_type),
            clusters: BTreeMap::default(),
            low_resources: false,
            fair_share,
            weights: conf.thorium.scaler.fair_share_weights(scaler_type),
            counts: HashMap::default(),
            image_counts: HashMap::default(),
            fair_share_counts: HashMap::default(),
            spawn_limit: 0,
            changes: ReqMap::default(),
        }
    }

    /// Remove and return the cpu group and cluster info for a specfic cluster
    ///
    /// # Arguments
    ///
    /// * `name` - the name of the cluster to get
    pub fn remove_cluster(&mut self, name: &str) -> Option<(u64, String, ClusterResources)> {
        // crawl through the cpu groups and look for this node
        for (cpus, cluster_map) in &mut self.clusters {
            // if the target cluster is in this cpu_group then return it
            if let Some((cluster_name, cluster)) = cluster_map.remove_entry(name) {
                return Some((*cpus, cluster_name, cluster));
            }
        }
        None
    }

    /// Update the resources we have to schedule
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the cluster to update
    /// * `update` - The update to apply
    #[instrument(name = "Allocatable::update", skip(self, update))]
    pub fn update(&mut self, name: &str, update: AllocatableUpdate) {
        // get this clusters resources
        let (_, cluster_name, mut cluster) = match self.remove_cluster(name) {
            Some((cpus, cluster_name, cluster)) => (cpus, cluster_name, cluster),
            None => (
                0,
                name.to_owned(),
                ClusterResources::new(self.scaler_type, name.to_owned(), &self.conf),
            ),
        };
        // apply an update to this cluster
        let freed = cluster.update(update, &mut self.image_counts, &mut self.counts);
        // get an entry to this clusters new cpu group
        let cpu_entry = self.clusters.entry(cluster.resources.cpu).or_default();
        // place this updated cluster in its new cpu group
        cpu_entry.insert(cluster_name, cluster);
        // add any freed resources back to their respective pools
        self.fairshare_pool.release(freed.fairshare);
        self.deadlines_pool.release(freed.deadline);
    }

    /// Set the deadline pool to contain whatever resources are in no other pools
    pub fn resize_deadline_pool(&mut self) {
        // get the total size of all clusters
        let total = self
            .clusters
            .values()
            .flatten()
            .fold(Resources::default(), |acc, (_, cluster)| {
                acc + cluster.resources
            });
        // set the deadline pool size excluding the fairshare reserve
        self.deadlines_pool.resources =
            Self::deadline_pool_resources(total, self.fairshare_pool.resources);
    }

    /// Compute the resources the deadline pool should hold given the total cluster capacity
    ///
    /// This is the pool-sizing arithmetic in [`Allocatable::resize_deadline_pool`] (it never
    /// needs `self`) extracted so its logic can be unit tested in isolation. The deadline pool
    /// gets the total cluster capacity minus the fairshare pool's remaining reserve so both pools
    /// never claim the same capacity. The subtraction saturates at zero.
    ///
    /// # Arguments
    ///
    /// * `total` - The summed available resources across every cluster
    /// * `fairshare_reserve` - The fairshare pool's remaining reserve that should be held back
    fn deadline_pool_resources(total: Resources, fairshare_reserve: Resources) -> Resources {
        // start with the full cluster total
        let mut resources = total;
        // hold back whatever the fairshare pool has reserved
        resources -= fairshare_reserve;
        resources
    }

    /// Log all clusters current resources
    #[instrument(name = "Allocatable::log_resources", skip_all)]
    pub fn log_resources(&self) {
        // log our deadline pools resources
        event!(
            Level::INFO,
            pool = "deadlines",
            cpu = self.deadlines_pool.resources.cpu,
            memory = self.deadlines_pool.resources.memory,
            ephemeral_storage = self.deadlines_pool.resources.ephemeral_storage,
            nvidia_gpu = self.deadlines_pool.resources.nvidia_gpu,
            amd_gpu = self.deadlines_pool.resources.amd_gpu,
            burstable_cpu = self.deadlines_pool.resources.burstable.cpu,
            burstable_mem = self.deadlines_pool.resources.burstable.memory,
        );
        // log our fairshare pools resources
        event!(
            Level::INFO,
            pool = "fairshare",
            cpu = self.fairshare_pool.resources.cpu,
            memory = self.fairshare_pool.resources.memory,
            ephemeral_storage = self.fairshare_pool.resources.ephemeral_storage,
            nvidia_gpu = self.fairshare_pool.resources.nvidia_gpu,
            amd_gpu = self.fairshare_pool.resources.amd_gpu,
            burstable_cpu = self.fairshare_pool.resources.burstable.cpu,
            burstable_mem = self.fairshare_pool.resources.burstable.memory,
        );
        // crawl each cpu group
        for cpu_group in self.clusters.values() {
            // crawl the clusters in this cpu group
            for (cluster, cluster_info) in cpu_group {
                // log this clusters spawn slots
                event!(
                    Level::INFO,
                    cluster,
                    deadline_spawn_slots = self.deadlines_pool.spawn_slots.get(cluster),
                    fairshare_spawn_slots = self.fairshare_pool.spawn_slots.get(cluster),
                );
                // crawl the node cpu groups in this cluster
                for node_group in cluster_info.nodes.values() {
                    // crawl the nodes in this cpu group
                    for (node, node_info) in node_group {
                        // get this nodes resources
                        let resources = &node_info.available;
                        // log this nodes resources
                        event!(
                            Level::INFO,
                            cluster,
                            node,
                            cpu = resources.cpu,
                            memory = resources.memory,
                            ephemeral_storage = resources.ephemeral_storage,
                            nvidia_gpu = resources.nvidia_gpu,
                            amd_gpu = resources.amd_gpu,
                            burstable_cpu = resources.burstable.cpu,
                            burstable_mem = resources.burstable.memory,
                            spawn_slots_deadlines = node_info.spawn_slots.deadlines,
                            spawn_slots_fairshare = node_info.spawn_slots.fairshare,
                        );
                    }
                }
                // log our clusters resources
                event!(
                    Level::INFO,
                    cluster,
                    cpu = cluster_info.resources.cpu,
                    memory = cluster_info.resources.memory,
                    ephemeral_storage = cluster_info.resources.ephemeral_storage,
                    nvidia_gpu = cluster_info.resources.nvidia_gpu,
                    amd_gpu = cluster_info.resources.amd_gpu,
                    burstable_cpu = cluster_info.resources.burstable.cpu,
                    burstable_mem = cluster_info.resources.burstable.memory,
                );
            }
        }
    }

    /// Determine if any of our clusters are fully utilized and should try to scale down lower priority workers
    #[instrument(name = "Allocatable::mark_high_load", skip_all)]
    pub fn mark_high_load(&mut self) {
        // reset our global low resources flag
        self.low_resources = false;
        // step over all clusters in all cpu groups
        for (cluster, cluster_info) in self.clusters.values_mut().flatten() {
            // check if this cluster has less then 5% of its resoruces remaining {
            if !cluster_info.has_remaining(0.05) {
                // log this cluster is low on cpu/memory
                event!(Level::INFO, cluster, low_resources = true);
                // mark this cluster is low on resources
                cluster_info.low_resources = true;
                self.low_resources = true;
            } else {
                cluster_info.low_resources = false;
            }
        }
    }

    /// Increment this image spanwed count if this image has a spawn limit
    fn increment_image_count(&mut self, image: &Image) {
        // we only need to inrement this images count if it has a spawn limit set
        if let SpawnLimits::Basic(_) = image.spawn_limit {
            // get this groups image counts
            let image_map = match self.image_counts.get_mut(&image.group) {
                Some(image_entry) => image_entry,
                None => {
                    // insert a new group image count map
                    self.image_counts
                        .insert(image.group.clone(), HashMap::with_capacity(1));
                    // get our groups count map
                    self.image_counts.get_mut(&image.group).unwrap()
                }
            };
            // get our images count
            let count = match image_map.get_mut(&image.name) {
                Some(count) => count,
                None => {
                    // add our image to our group map
                    image_map.insert(image.name.clone(), 0);
                    // get a mutable ref to our count
                    image_map.get_mut(&image.name).unwrap()
                }
            };
            // increment our count
            *count += 1;
        }
    }

    /// Check if the target pool can support this image being spawned once
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to check
    /// * `image` - The image to check
    fn enough(&mut self, image: &Image, pool: Pools) -> bool {
        // check the correct pool
        let enough_resources = match pool {
            Pools::FairShare => self.fairshare_pool.enough(image),
            Pools::Deadline => self.deadlines_pool.enough(image),
        };
        // check if there are any spawn limits on this image
        let under_limit = match image.spawn_limit {
            SpawnLimits::Basic(limit) => {
                // get this groups image counts
                let image_map = match self.image_counts.get_mut(&image.group) {
                    Some(image_entry) => image_entry,
                    None => {
                        // insert a new group image count map
                        self.image_counts
                            .insert(image.group.clone(), HashMap::with_capacity(1));
                        // get our groups count map
                        self.image_counts.get_mut(&image.group).unwrap()
                    }
                };
                // get our images count
                let count = match image_map.get(&image.name) {
                    Some(count) => count,
                    None => {
                        // add our image to our group map
                        image_map.insert(image.name.clone(), 0);
                        // get a mutable ref to our count
                        image_map.get_mut(&image.name).unwrap()
                    }
                };
                // check if we are above our limit or not
                if *count < limit {
                    // we can spawn this
                    true
                } else {
                    false
                }
            }
            SpawnLimits::Unlimited => true,
        };
        // make sure we have enough resources and are under the limit
        enough_resources && under_limit
    }

    /// Consume the resources needed to spawn a single worker for a specific image
    ///
    /// # Arguments
    ///
    /// * `pool` - The pool to consume resources from
    /// * `image` - The image we are allocating resources for
    fn consume(&mut self, image: &Image, pool: Pools) {
        // the correct pool
        match pool {
            Pools::FairShare => self.fairshare_pool.consume(image),
            Pools::Deadline => self.deadlines_pool.consume(image),
        }
    }

    /// Find the least provisioned cluster that can support this image
    ///
    /// The response for this function is `(cpu_group, cluster_name, node)`.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to allocate resources for
    /// * `pool` - The pool we are trying to allocate resources in
    fn allocate_cluster_helper(
        &mut self,
        image: &Image,
        pool: Pools,
    ) -> Option<(u64, String, NodeResources)> {
        // crawl over all nodes until we find one that we can fit on
        for (cpus, cluster_map) in self.clusters.iter_mut().rev() {
            // iterate over the clusters that have the same number of cores
            for (cluster_name, cluster) in cluster_map.iter_mut() {
                // check if this image has any restrictions
                let nodes = match self.restrictions.check(cluster_name, image) {
                    IsRestricted::No => None,
                    IsRestricted::Yes(nodes) => Some(nodes),
                    IsRestricted::WrongCluster => {
                        continue;
                    }
                };
                // try to consume the resources for this image on a node
                if let Some(node) = cluster.allocate_node(image, nodes, pool) {
                    // clone our values
                    let cluster_name = cluster_name.to_owned();
                    // return the info we found
                    return Some((*cpus, cluster_name, node));
                }
            }
        }
        None
    }

    /// Allocate resources on the least provisioned cluster/node
    ///
    /// # Arguments
    ///
    /// * `image` - The image to allocate resources for
    /// * `pool` - The pool we are trying to allocate resources in
    fn allocate_cluster(&mut self, image: &Image, pool: Pools) -> Option<(String, String)> {
        // locate the cluster and node that we are allocating resources on
        if let Some((cpus, cluster_name, node)) = self.allocate_cluster_helper(image, pool) {
            // get our cluster from the target cpu group
            match self
                .clusters
                .get_mut(&cpus)
                .map(|clusters| clusters.remove(&cluster_name))
            {
                Some(Some(mut cluster)) => {
                    // get an entry to this nodes new cpu group
                    let cpu_group = cluster.nodes.entry(node.available.cpu).or_default();
                    // get our node name
                    let node_name = node.name.clone();
                    // add our node into its new cpu group
                    cpu_group.insert(node.name.clone(), node);
                    // consume the resources for this cluster
                    cluster.resources.consume(&image.resources, 1);
                    // get an entry to this clusters new cpu group
                    let cpu_group = self.clusters.entry(cluster.resources.cpu).or_default();
                    // add this cluster to its new cpu group
                    cpu_group.insert(cluster.name.clone(), cluster);
                    // return the cluster and node we spawned this on
                    return Some((cluster_name, node_name));
                }
                _ => {
                    panic!("AHHH");
                }
            }
        }
        None
    }

    /// try to allocate resources for a requisition
    ///
    /// # Arguments
    ///
    /// * `image` - The image to allocate resources for
    /// * `pool` - The pool we are trying to allocate resources in
    fn try_allocate(&mut self, image: &Image, pool: Pools) -> Option<(String, String)> {
        // check if we have enough resources in the target pool
        if self.enough(image, pool) {
            // try to allocate this image on a node
            return match self.allocate_cluster(image, pool) {
                Some((cluster, node)) => {
                    // consume the resources from the correct pool
                    self.consume(image, pool);
                    // incement the count for this image if it has a spawn limit
                    self.increment_image_count(image);
                    // return our cluster and node
                    Some((cluster, node))
                }
                None => None,
            };
        }
        // we could not spawn this image
        None
    }

    /// Calculate the increase in fairshare for a specific resource spec
    ///
    /// # Arguments
    ///
    /// * `rank` - A users current fair share rank
    /// * `count` - The number of pods to use for this increase
    /// * `resources` - The resource spec we are calculating an increase for
    pub fn calc_fair_share(&self, rank: u64, count: i64, resources: &Resources) -> u64 {
        // only calculate an increase if count is greater then 0
        if count > 0 {
            // calculate the increase in cost for the cpu
            let mut incr = resources.cpu * self.weights.cpu;
            // add the cost for the memory
            incr += resources.memory * self.weights.memory;
            // add this increase to our old rank
            rank + (incr * count as u64)
        } else {
            // count is zero so just use our our old rank
            rank
        }
    }

    /// Crawl over some users and try to spawn one job for each of them
    fn try_fair_share_spawn(
        &mut self,
        cache: &Cache,
        spawn_slots: &mut usize,
        map: &mut SpawnMap<'_>,
        rank: u64,
        users: HashSet<String>,
        no_spawns: &mut HashSet<String>,
    ) {
        // get our current span
        let span = Span::current();
        // crawl over the users to try and spawn fair share jobs for
        'user: for user in users {
            // if we are out of spawn slots then keep this user at their current rank
            if *spawn_slots == 0 {
                no_spawns.insert(user);
                continue;
            }
            // get the images this user is trying to spawn
            if let Some(job_stats) = map.get_mut(&user) {
                // crawl this users job stats from lowest current utilization to highest
                for reqs in job_stats.values_mut() {
                    // crawl over the images in this rank group
                    for (req, created) in reqs.iter_mut() {
                        // skip any images where created is 0
                        if *created == 0 {
                            continue;
                        }
                        // get this jobs image info
                        let Some(image) = cache.get_image(&req.group, &req.stage, &span) else {
                            continue;
                        };
                        // calculate a deadline based on our runtime
                        #[allow(clippy::cast_possible_truncation)]
                        let deadline = from_now!(image.runtime as i64);
                        // try to spawn this requisition
                        if let Some((cluster, node)) = self.try_allocate(image, Pools::FairShare) {
                            // build our newly spawned worker
                            let spawned =
                                Spawned::new(&cluster, &node, req.clone(), image, Pools::FairShare);
                            // get an entry to this clusters map in our change map
                            let cluster_entry = self.changes.spawns.entry(cluster).or_default();
                            // get an entry to the deadline group for this spawn
                            let spawns_entry = cluster_entry.entry(deadline).or_default();
                            // add this new worker allocation to our change map
                            spawns_entry.push(spawned);
                            // get this users new fair share rank
                            let new_rank = self.calc_fair_share(rank, 1, &image.resources);
                            // we spawned this image under fair share so increment our users fair share rank
                            let rank_group = self.fair_share.entry(new_rank).or_default();
                            // add this user to their new group
                            rank_group.insert(user);
                            // get an entry to this reqs pending count
                            let entry = self.fair_share_counts.entry(req.clone()).or_insert(0);
                            // increment our spawn count
                            *entry += 1;
                            // decrement our created count since we are spawning an image for this
                            *created -= 1;
                            // consume one spawn slot
                            consume_spawn_slot(spawn_slots);
                            // continue onto the next user
                            continue 'user;
                        }
                    }
                }
            }
            // this user did not spawn anything so add put them in our no spawn set
            no_spawns.insert(user);
        }
    }

    /// Allocate resources for fair share spawned workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium to use when scheduling
    /// * `spawn_slots` - The remaining spawn slots to fill
    #[instrument(name = "Allocatable::fairshare_allocation", skip_all, err(Debug))]
    async fn fairshare_allocation(
        &mut self,
        thorium: &Thorium,
        cache: &Cache,
        spawn_slots: &mut usize,
    ) -> Result<(), Error> {
        // get the system stats for for Thorium so we can assign resources to users
        let stats = thorium.system.stats().await?;
        // Build a map of each users outstanding job reqs
        let mut map = stats.users_jobs();
        // track the users that do not spawn any jobs
        let mut no_spawns = BTreeMap::default();
        // iterate over our users based on how many resources they have consumed
        // stop once we have used all of our spawn slots for this loop
        while *spawn_slots > 0 {
            // get the next lowest rank group or stop if we have crawled all users
            let Some((rank, users)) = self.fair_share.pop_first() else {
                break;
            };
            // get an entry to the current rank group in our no spawn map
            let entry = no_spawns.entry(rank).or_default();
            // try to spawn a job for each of the users in this rank group
            self.try_fair_share_spawn(cache, spawn_slots, &mut map, rank, users, entry);
        }
        // add the users who did not spawn anything back into our fair share tree
        for (rank, users) in no_spawns {
            // get an entry to this rank group
            let rank_entry = self.fair_share.entry(rank).or_default();
            // add our users to this rank group
            rank_entry.extend(users);
        }
        Ok(())
    }

    /// Gets a list of immediate deadlines to try to meet
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    #[instrument(name = "Allocatable::get_deadlines", skip_all, err(Debug))]
    async fn get_deadlines(
        &self,
        thorium: &Thorium,
        cache: &Cache,
    ) -> Result<Vec<Deadline>, thorium::Error> {
        // build arbitrary dates for deadline reading
        let start = DateTime::from_timestamp(0, 0).unwrap();
        let end = Utc::now() + chrono::Duration::weeks(1000);
        // read the upcoming deadlines from the deadline stream
        let window = self.deadline_window;
        let mut deadlines = thorium
            .jobs
            .deadlines(self.scaler_type, &start, &end, window)
            .await?;
        // log how many deadlines we retrieved
        let prefilter = deadlines.len();
        event!(Level::INFO, deadlines = prefilter);
        // remove any deadlines we can't spawn due to missing cache info or bans
        deadlines.retain(|dl| self.bans.filter_deadlines(cache, dl));
        // if we filtered any deadlines then log it
        if prefilter > deadlines.len() {
            // determine how many deadlines were filtered
            let filtered = prefilter - deadlines.len();
            event!(Level::INFO, filtered = filtered);
        }
        Ok(deadlines)
    }

    /// Reset all of our nodes spawn limits
    fn reset_spawns(&mut self) {
        // track the number of nodes that we have
        let mut node_cnt = 0;
        // crawl over all of our clusters
        for cluster_group in self.clusters.values_mut() {
            // crawl over all of our clusters in this cluster cpu group
            for (name, cluster) in cluster_group {
                // start with no spawn slots for this cluster
                let mut cluster_spots = SpawnSlots::empty();
                // crawl over over all node cpu groups
                for node_group in cluster.nodes.values_mut() {
                    // add this size of this group to our node cnt
                    node_cnt += node_group.len();
                    // step over all of the nodes in this node cpu group
                    for node in node_group.values_mut() {
                        // reset this nodes spawn slots
                        node.spawn_slots = cluster.spawn_slots_reset;
                        // add this nodes spawn slots to our cluster spawn slots
                        cluster_spots.add(node.spawn_slots);
                    }
                }
                // update this clusters spawn slot count for all pools
                self.deadlines_pool
                    .set_spawn_slot(name, cluster_spots.deadlines);
                // if we don't have fairshare enabled then just add our fairshare slots to the deadline pool
                if self.fairshare_pool.resources.some() {
                    // fairshare has resources to schedule
                    self.fairshare_pool
                        .set_spawn_slot(name, cluster_spots.fairshare);
                } else {
                    // fairshare has no resources to schedule
                    self.deadlines_pool
                        .set_spawn_slot(name, cluster_spots.fairshare);
                }
            }
        }
        // multiply our total number of nodes by 2 to get our ideal limit
        self.spawn_limit = node_cnt * 2;
    }

    /// Allocate resources for workers spawned in the deadline pool
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium to use when scheduling
    /// * `spawn_slots` - The remaining spawn slots to fill
    #[instrument(name = "Allocatable::deadline_allocation", skip_all, err(Debug))]
    async fn deadline_allocation(
        &mut self,
        thorium: &Thorium,
        cache: &Cache,
        spawn_slots: &mut usize,
    ) -> Result<(), Error> {
        // get our current span
        let span = Span::current();
        // get the deadlines to try to meet
        let deadlines = self.get_deadlines(thorium, cache).await?;
        // crawl over these deadlines and try to meet them
        for deadline in deadlines {
            // get this deadlines timestamp
            let timestamp = deadline.deadline;
            // build a requisition for this deadline
            let req = Requisition::from(deadline);
            // check if we spawned this image in the past
            if let Some(count) = self.counts.get_mut(&req) {
                // if we spawned this in the past then we should skip this spawn
                // otherwise we will repeatedly spwn workers for the same deadlines
                match count.cmp(&&mut 0) {
                    // we spawened a worker for this deadline in the past
                    Ordering::Greater => {
                        // decrement our count
                        *count -= 1;
                        continue;
                    }
                    // we have no longer spawned a worker to meet this deadline
                    _ => {
                        self.counts.remove(&req);
                    }
                }
            }
            // check if we are already trying to meet this request
            if let Some(current) = self.fair_share_counts.get_mut(&req) {
                // decrement our current count
                *current = current.saturating_sub(1);
                // if current is 0 then remove it
                if *current == 0 {
                    // remove this req from our fair share counts since we have consumed them all
                    self.fair_share_counts.remove(&req);
                }
                continue;
            }
            // get this jobs image info
            let Some(image) = cache.get_image(&req.group, &req.stage, &span) else {
                continue;
            };
            // try to allocate resources for this deadline
            if let Some((cluster, node)) = self.try_allocate(image, Pools::Deadline) {
                // build our newly spawned worker
                let spawned = Spawned::new(&cluster, &node, req.clone(), image, Pools::Deadline);
                // get an entry to this clusters map in our change map
                let cluster_entry = self.changes.spawns.entry(cluster).or_default();
                // get an entry to the deadline group for this spawn
                let spawns_entry = cluster_entry.entry(timestamp).or_default();
                // add this new worker allocation to our change map
                spawns_entry.push(spawned);
            } else {
                // we would like to spawn this image but can't so check if we are low on resources
                // and out of spawn slots
                if self.low_resources && *spawn_slots > 0 {
                    // try to find something to scale down to meet this deadline
                    if self.scale_down_to_meet(timestamp, &req, image) {
                        // consume a spawn slot
                        *spawn_slots -= 1;
                    }
                }
            }
            // if we have exhausted our spawn slots then exit early
            if *spawn_slots == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Scale down any existing workers to meet higher priority deadlines
    fn scale_down_to_meet(
        &mut self,
        mut deadline: DateTime<Utc>,
        req: &Requisition,
        image: &Image,
    ) -> bool {
        // increment our deadline by 1 minutes so we only get things with marginally lower priority
        deadline += chrono::Duration::minutes(1);
        // get our current timestamp
        let now = Utc::now();
        // track the workers we will scale down to meet this deadline
        let mut scale_downs = Vec::default();
        // crawl all cluster cpu groups to determine if we can scale anything down to meet this requisition
        for (_, cluster_group) in self.clusters.iter_mut().rev() {
            // crawl the clusters in this cpu group that are low on resources
            for cluster in cluster_group
                .values_mut()
                .filter(|cluster| cluster.low_resources)
            {
                // crawl the node cpu groups in this cluster
                for node_group in cluster.nodes.values_mut().rev() {
                    // crawl the nodes in this node cpu group
                    for node in node_group.values_mut() {
                        // track the resources we can free on this node
                        let mut freeable = Resources::default();
                        // clear our scale down set
                        scale_downs.clear();
                        // crawl the worker deadline groups on this node
                        for (_, spawned_group) in node.spawned.range_mut(deadline..).rev() {
                            // crawl the workers in this worker deadline group
                            for spawned in spawned_group {
                                // skip any workers that are not running or that are the same worker type
                                // or that haven't had time to complete at least one job
                                if is_scale_down_victim(spawned, req, now) {
                                    // add this workers resources to our freeable set
                                    freeable += spawned.resources;
                                    // add this worker to our scale down set
                                    scale_downs.push(spawned);
                                    // check if we have enough resources to spawn this image now
                                    if freeable.enough(&image.resources) {
                                        // set all of the statuses on these workers to be scale down
                                        for spawned in scale_downs.drain(..) {
                                            // set our scale down flag to true
                                            spawned.scaled_down = true;
                                            // add this worker to our scale down orders
                                            self.changes.scale_down(spawned.to_owned());
                                        }
                                        // we were able to find things to scale down
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // we could not find anything to scale down
        false
    }

    /// Allocate resources for all pools
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium to use when scheduling
    #[instrument(name = "Allocatable::deadline_allocation", skip_all, err(Debug))]
    pub async fn allocate(&mut self, thorium: &Thorium, cache: &Cache) -> Result<(), Error> {
        // log our clusters current resources
        self.log_resources();
        // check if any of our cluster is under high load
        self.mark_high_load();
        // empty our req map
        self.changes.spawns.clear();
        self.changes.scale_down.clear();
        // reset our nodes spawn slot count
        self.reset_spawns();
        // track the number of spawn slots we have consumed this loop
        let mut spawn_slots = self.spawn_limit;
        // clone our old counts so we can restore it
        let old_counts = self.counts.clone();
        // increase our fair share ranks
        self.increase_fair_share_ranks(cache);
        // try to allocate resources based on fair share
        self.fairshare_allocation(thorium, cache, &mut spawn_slots)
            .await?;
        // try to allocate resources based on deadline scheduling if we still have remaining spawn slots
        if spawn_slots > 0 {
            self.deadline_allocation(thorium, cache, &mut spawn_slots)
                .await?;
        }
        // remove any empty cluster cpu groups
        self.clusters.retain(|_, clusters| !clusters.is_empty());
        // restore our old counts
        self.counts = old_counts;
        Ok(())
    }

    /// Commit our finalized changes to their respective nodes
    #[instrument(name = "Allocatable::commit", skip_all)]
    pub fn commit(
        &mut self,
        changes: HashMap<DateTime<Utc>, Vec<Spawned>>,
    ) -> HashSet<Requisition> {
        // track our changes on a per req basis
        let mut reqs = HashSet::with_capacity(self.counts.len());
        // build a map to store what changes happended on what cluster/node
        let mut sorted: HashMap<String, HashMap<String, Vec<(DateTime<Utc>, Spawned)>>> =
            HashMap::with_capacity(self.clusters.len());
        // crawl our changes and start sorting
        for (deadline, spawns) in changes {
            // crawl the spawns in this deadline group
            for spawn in spawns {
                // add our req to our req map
                reqs.insert(spawn.req.clone());
                // get an entry to this spawns cluster map
                let cluster_entry = sorted.entry(spawn.cluster.clone()).or_default();
                // get an entry to this spawns node map
                let node_entry = cluster_entry.entry(spawn.node.clone()).or_default();
                // insert our spawn
                node_entry.push((deadline, spawn));
            }
        }
        // crawl over all clusters
        for (cluster_name, cluster) in self.clusters.values_mut().flatten() {
            // get this clusters committed changed
            if let Some(mut cluster_changes) = sorted.remove(cluster_name) {
                // crawl over the nodes in this cluster and add their spawns
                for (node_name, node) in cluster.nodes.values_mut().flatten() {
                    // get this nodes changes
                    if let Some(node_changes) = cluster_changes.remove(node_name) {
                        // add all of this nodes spawns
                        for (deadline, spawn) in node_changes {
                            // get an entry to the deadline group for this spawn
                            let dl_entry = node.spawned.entry(deadline).or_default();
                            // add our spawn
                            dl_entry.push(spawn);
                        }
                    }
                }
            }
        }
        reqs
    }

    /// Build a hashset of the names of all currently spawned resources
    pub fn spawn_names(&self) -> HashSet<&String> {
        // assume 100 objects
        let mut names = HashSet::with_capacity(100);
        // crawl all cluster cpu groups
        for cluster_group in self.clusters.values() {
            // crawl all clusters in this cpu group
            for cluster in cluster_group.values() {
                // crawl all node cpu groups
                for node_group in cluster.nodes.values() {
                    // crawl all nodes in each node group
                    for node in node_group.values() {
                        // build an iterator over our spawn names
                        let spawn_iter = node.spawned.values().flatten().map(|spawn| &spawn.name);
                        // extend our worker name set
                        names.extend(spawn_iter);
                    }
                }
            }
        }
        // return our set of spawned worker names
        names
    }

    // get the existing fair share ranks by username
    fn fair_share_by_user(&mut self) -> HashMap<String, u64> {
        // swap our map fair share map with an empty one
        let mut temp = BTreeMap::default();
        std::mem::swap(&mut self.fair_share, &mut temp);
        // calculate the size of our map
        let size = self.fair_share.values().map(HashSet::len).sum();
        // init our map to the correct size
        let mut map = HashMap::with_capacity(size);
        for (rank, users) in temp {
            // add each user into our map at this rank
            for user in users {
                map.insert(user, rank);
            }
        }
        map
    }

    /// Increase our fairshare ranks for all resources consumed by each user
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use when scheduling
    #[instrument(name = "Allocatable::increase_fair_share_ranks", skip_all)]
    pub fn increase_fair_share_ranks(&mut self, cache: &Cache) {
        // get our current span
        let span = Span::current();
        // get all fair share ranks by user
        let mut by_user = self.fair_share_by_user();
        // crawl over all spawned requisitions
        for (req, count) in &self.counts {
            // get this requisitions images info
            if let Some(image) = cache.get_image(&req.group, &req.stage, &span) {
                // get this users fair share rank and increment it
                match by_user.get_mut(&req.user) {
                    Some(rank) => *rank = self.calc_fair_share(*rank, *count, &image.resources),
                    None => {
                        // calculate a rank for a user not in the fairshare ranks yet
                        let rank = self.calc_fair_share(0, *count, &image.resources);
                        // insert our new rank
                        by_user.insert(req.user.clone(), rank);
                    }
                }
            }
        }
        // reinsert our fair share ranks
        for (user, rank) in by_user {
            // get an entry into this fair shar ranks set
            let entry = self.fair_share.entry(rank).or_default();
            // add our user to the correct rank entry
            entry.insert(user);
        }
    }

    /// Lower all users fair share rankings based on the current resources in the cluster
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    #[instrument(name = "Allocatable::decrease_fair_share_ranks", skip_all)]
    pub fn decrease_fair_share_ranks(&mut self, conf: &Conf) {
        // calculate the total amount of resources in each cluster
        let mut total =
            self.clusters
                .values()
                .flatten()
                .fold(Resources::default(), |mut acc, (_, cluster)| {
                    // sum our cpu and memory since thats all we need
                    acc.cpu += cluster.resources.cpu;
                    acc.memory += cluster.resources.memory;
                    acc
                });
        // get the divisor to apply to our resources
        let divisor = conf.thorium.scaler.fair_share_divisor(self.scaler_type);
        // apply this divisor to our totals
        total.cpu = total.cpu.saturating_div(divisor);
        total.memory = total.memory.saturating_div(divisor);
        // calculate the decrease we are going to apply to our fair share rank
        let decr = self.calc_fair_share(0, 1, &total);
        // sort all of our fair share ranks by user
        let mut sorted = self.fair_share_by_user();
        // apply this decrease to each user
        sorted
            .iter_mut()
            .for_each(|(_, rank)| *rank = rank.saturating_sub(decr));
        // add our sorted users bank into our fair share ranking map
        for (user, rank) in sorted {
            // get an entry into this fair shar ranks set
            let entry = self.fair_share.entry(rank).or_default();
            // add our user to the correct rank entry
            entry.insert(user);
        }
    }

    /// Delete any workers that failed and return a list of live workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium API
    /// * `failed` - The workers that have failed
    /// * `config` - The thorium config
    async fn prune_failed_workers(
        &mut self,
        thorium: &Thorium,
        failed: &HashSet<String>,
        config: &Conf,
    ) -> Result<HashSet<String>, Error> {
        // scan our nodes 50 at at ime
        let params = NodeListParams::default()
            .scaler(self.scaler_type)
            .clusters_from_config(config)
            .limit(50);
        // get info on the current Thorium nodes
        let mut cursor = thorium.system.list_node_details(&params).await?;
        // assume we have at least 300 workers
        let mut alive = HashSet::with_capacity(300);
        // track the workers that we need to delete from Thorium
        let mut deletes = WorkerDeleteMap::with_capacity(failed.len());
        // condense this to a map of workers
        while !cursor.data.is_empty() {
            // crawl over the nodes on this page
            for node in cursor.data.drain(..) {
                // sort the workers on this node into failed and alive workers
                prune_failed_node_workers(failed, node.workers, &mut alive, &mut deletes);
            }
            // if our cursor is exhausted then break
            if cursor.exhausted() {
                break;
            }
            // get the next page of data
            cursor.refill().await?;
        }
        // delete our failed workers from thorium
        thorium
            .system
            .delete_workers(self.scaler_type, &deletes)
            .await?;
        Ok(alive)
    }

    /// Free resources for all deleted workers by name
    ///
    /// This will also pull a list of current workers in Thorium and free any
    /// resources tied to no longer existing workers.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium API
    /// * `failed` - The workers that have failed
    /// * `deleted` - The workers that were deleted
    /// * `config` - The thorium config
    #[instrument(name = "Allocatable::free_deleted", skip_all)]
    pub async fn free_deleted(
        &mut self,
        thorium: &Thorium,
        failed: &HashSet<String>,
        deleted: &HashSet<String>,
        config: &Conf,
    ) -> Result<(), Error> {
        // get all currently alive workers
        let alive = self.prune_failed_workers(thorium, failed, config).await?;
        // track the resources we have freed
        let mut freed = PoolFrees::default();
        // swap our old cluster tree out for a new one
        let old_clusters = std::mem::take(&mut self.clusters);
        // crawl over our cluster cpu groups
        for cluster_group in old_clusters.into_values() {
            // crawl over the clsuters in this cpu group
            for (name, mut cluster) in cluster_group {
                // free any deleted workers in this cluster
                cluster.free(
                    &alive,
                    deleted,
                    &mut freed,
                    &mut self.image_counts,
                    &mut self.counts,
                );
                // get an entry to this clusters new cpu grop
                let cpu_entry = self.clusters.entry(cluster.resources.cpu).or_default();
                // add our cluster to its new cpu group
                cpu_entry.insert(name, cluster);
            }
        }
        // add any freed resources back to their respective pools
        self.fairshare_pool.release(freed.fairshare);
        self.deadlines_pool.release(freed.deadline);
        Ok(())
    }
}

/// Sort the workers Thorium lists on a node into failed workers to delete and alive workers
///
/// This is the per-node loop in [`Allocatable::prune_failed_workers`] extracted so it can be
/// unit tested. A failed worker's image count is NOT decremented here: the failed worker is left
/// out of `alive`, so `NodeResources::free` frees it and decrements its image count exactly once.
///
/// # Arguments
///
/// * `failed` - The workers that have failed
/// * `workers` - The workers Thorium lists on this node
/// * `alive` - The set of alive workers to add to
/// * `deletes` - The failed workers to delete from Thorium
fn prune_failed_node_workers(
    failed: &HashSet<String>,
    workers: impl IntoIterator<Item = (String, Worker)>,
    alive: &mut HashSet<String>,
    deletes: &mut WorkerDeleteMap,
) {
    // crawl over the workers on this node
    for (name, _) in workers {
        // if this worker has failed then add it to our delete map
        if failed.contains(&name) {
            // add this failed worker to our delete map
            // its image count is released when NodeResources::free frees it
            deletes.add_mut(name);
        } else {
            // this worker is alive so add its name to our alive set
            alive.insert(name);
        }
    }
}

/// Track a worker that the scheduler failed to spawn
///
/// This is the spawn error branch in `Scaler::schedule` extracted so it can be unit tested.
/// When a worker fails to spawn it is deleted from Thorium, and the fairshare pool resources and
/// spawn limit image count that `try_allocate` consumed for it are released here. The failed
/// worker never reaches `node.spawned` or Thorium's worker listing, so no other free path would
/// ever release them. Deadline pool resources are not released here since the deadline pool is
/// resized from observed cluster resources on every resources update.
///
/// # Arguments
///
/// * `fairshare_pool` - The fairshare pool the worker may have consumed resources from
/// * `image_counts` - The current counts for spawned images
/// * `spawn` - The worker that failed to spawn
/// * `deleted` - The workers that were deleted this loop
/// * `failed` - The workers to delete from Thorium
pub fn record_failed_spawn(
    fairshare_pool: &mut Pool,
    image_counts: &mut HashMap<String, HashMap<String, u64>>,
    spawn: Spawned,
    deleted: &mut HashSet<String>,
    failed: &mut WorkerDeleteMap,
) {
    // give any consumed fairshare resources back to the fairshare pool
    if spawn.pool == Pools::FairShare {
        fairshare_pool.release(spawn.resources);
    }
    // get this groups image map
    if let Some(image_map) = image_counts.get_mut(&spawn.req.group) {
        // get this images current spawn count
        if let Some(count) = image_map.get_mut(&spawn.req.stage) {
            // release the spawn limit slot this worker consumed
            *count = count.saturating_sub(1);
        }
    }
    // add this failed worker to our delete map
    deleted.insert(spawn.name.clone());
    // add this failed worker to our list of workers to remove
    failed.add_mut(spawn.name);
}

/// Consume a single spawn slot from the remaining per-loop spawn budget
///
/// This is the decrement that `try_fair_share_spawn` applies once per fairshare spawn; it
/// is extracted so the underflow behavior can be unit tested in isolation. The budget never
/// drops below zero.
///
/// # Arguments
///
/// * `spawn_slots` - The remaining spawn slots to decrement
fn consume_spawn_slot(spawn_slots: &mut usize) {
    // decrement the remaining spawn budget by one without underflowing
    *spawn_slots = spawn_slots.saturating_sub(1);
}

/// Decide whether an already spawned worker is a valid victim to scale down in order to
/// make room for a higher priority requisition
///
/// This is the pure decision that [`Allocatable::scale_down_to_meet`] applies to each
/// candidate worker; it is extracted so it can be unit tested without a live cluster.
///
/// # Arguments
///
/// * `spawned` - The already spawned worker being considered as a scale down victim
/// * `req` - The higher priority requisition we are trying to make room for
/// * `now` - The current timestamp used to check the worker has run long enough to scale down
fn is_scale_down_victim(spawned: &Spawned, req: &Requisition, now: DateTime<Utc>) -> bool {
    // only consider workers that have not already been scaled down, have had time to
    // complete at least one job, are in the deadline pool, and are not the same requisition
    // a worker is only the same requisition if every field matches
    !spawned.scaled_down
        && spawned.down_scalable < now
        && spawned.pool == Pools::Deadline
        && spawned.req != *req
}

/// All resources for a single cluster
#[derive(Debug, Clone)]
pub struct ClusterResources {
    /// The name of this cluster
    pub name: String,
    /// The amount of currently available resources for this cluster
    pub resources: Resources,
    /// The resources in this cluster in total
    pub total: Resources,
    /// Whether this cluster is low on resources or not
    pub low_resources: bool,
    /// A map of resources for one node
    pub nodes: BTreeMap<u64, HashMap<String, NodeResources>>,
    /// The spawnable slots per node to set
    pub spawn_slots_reset: SpawnSlots,
}

impl ClusterResources {
    /// Create a new resources object for a cluster
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this cluster
    pub fn new(scaler_type: ImageScaler, name: String, conf: &Conf) -> Self {
        // get our clusters original name if we have an alias
        let original = conf
            .thorium
            .scaler
            .get_original_name(scaler_type, &name)
            .unwrap_or(&name);
        // get this clusters max number of spawnable workers per node
        let spawn_slots = conf.thorium.scaler.spawn_slots(scaler_type, original);
        // build our cluster resources object
        ClusterResources {
            name,
            resources: Resources::default(),
            total: Resources::default(),
            low_resources: false,
            nodes: BTreeMap::default(),
            spawn_slots_reset: spawn_slots,
        }
    }

    /// Free any resources after applying a cluster update
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this cluster
    /// * `active` - The currently active workers on this cluster
    /// * `image_counts` - The current counts for spawned images
    /// * `req_counts` - The current counts of all requisitions
    fn free_after_update(
        &mut self,
        mut update: AllocatableUpdate,
        active: &HashSet<String>,
        image_counts: &mut HashMap<String, HashMap<String, u64>>,
        req_counts: &mut HashMap<Requisition, i64>,
    ) -> PoolFrees {
        // clear our  total resource count
        self.total = Resources::default();
        // track the resources we have freed
        let mut freed = PoolFrees::default();
        // track the current resources available for this cluster
        let mut new_available = Resources::default();
        // swap our node tree for a new one
        let old_nodes = std::mem::take(&mut self.nodes);
        // crawl the cpu groups in this cluster
        for mut node_group in old_nodes.into_values() {
            // crawl over the nodes in this cpu group any resources we can
            for (node_name, mut node) in node_group.drain() {
                // try to get this node from our update
                match update.nodes.remove(&node_name) {
                    Some(node_update) => {
                        // try to free any resources we can on this node
                        node.free(
                            active,
                            &HashSet::default(),
                            &mut freed,
                            image_counts,
                            req_counts,
                        );
                        // add this nodes updated resources to our new total
                        new_available += node_update.available;
                        // apply these changes to our node
                        node.available = node_update.available;
                        node.total = node_update.total;
                        // add this nodes total resources to our clusters total
                        self.total += node.total;
                        // get an entry to this nodes new cpu group
                        let cpu_entry = self.nodes.entry(node.available.cpu).or_default();
                        // add this node to its new cpu group
                        cpu_entry.insert(node_name, node);
                    }
                    // this node was not in the update
                    None => {
                        // check if this node was removed
                        if update.removes.contains(&node_name) {
                            // try to free any resources we can on this node
                            node.free(
                                active,
                                &HashSet::default(),
                                &mut freed,
                                image_counts,
                                req_counts,
                            );
                            // this node just gets dropped since it needs to be removed
                        } else {
                            // add this nodes updated resources to our new total
                            new_available += node.available;
                            // add this nodes total resources to our clusters total
                            self.total += node.total;
                            // just retain this nodes current info since it was not in the update
                            // get an entry to this nodes current cpu group
                            let cpu_entry = self.nodes.entry(node.available.cpu).or_default();
                            // add this node to its new cpu group
                            cpu_entry.insert(node_name, node);
                        }
                    }
                }
            }
        }
        // replace our total resource counts
        self.resources = new_available;
        // return the total freed resources in this cluster
        freed
    }

    /// Apply an update to this clusters allocatable resources
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this cluster
    /// * `active` - The currently active workers on this cluster
    /// * `image_counts` - The current counts for spawned images
    /// * `req_counts` - The current counts for workers by requisition
    pub fn update(
        &mut self,
        mut update: AllocatableUpdate,
        image_counts: &mut HashMap<String, HashMap<String, u64>>,
        req_counts: &mut HashMap<Requisition, i64>,
    ) -> PoolFrees {
        // get the names of all the active workers in this update
        let active: HashSet<String> = update
            .nodes
            .values_mut()
            .flat_map(|node| std::mem::take(&mut node.active))
            .collect();
        // assume we have at least 10 nodes
        let mut temp_nodes = HashMap::with_capacity(10);
        // take our current nodes so we can resort them later
        let nodes = std::mem::take(&mut self.nodes);
        // drain all of our current nodes so we can resort them later
        for node_group in nodes.into_values() {
            // add these nodes
            temp_nodes.extend(node_group);
        }
        // insert or update all of the nodes in our update
        // don't drain our update since free_after_update needs it to free inactive workers
        for (name, node_update) in &update.nodes {
            // get this nodes entry
            let entry = temp_nodes
                .entry(name.clone())
                .or_insert_with(|| NodeResources::new(name.clone(), self.spawn_slots_reset));
            // apply our update to this node
            entry.available = node_update.available;
            entry.total = node_update.total;
        }
        // sort all of our nodes back in
        for (name, node) in temp_nodes {
            // get an entry to this nodes cpu group
            let cpu_entry = self.nodes.entry(node.available.cpu).or_default();
            // add our node back in
            cpu_entry.insert(name, node);
        }
        // crawl all of our currently active workers and remove any that no longer exist
        self.free_after_update(update, &active, image_counts, req_counts)
    }

    /// Check if we have enough resources to spawn one of these images on some node
    ///
    /// # Arguments
    ///
    /// * `image` - The image to base these pods on
    /// * `nodes` - The nodes that this image is restricted to if any
    /// * `pool` - The pool we are trying to allocate resources in
    fn allocate_node(
        &mut self,
        image: &Image,
        nodes: Option<&HashSet<String>>,
        pool: Pools,
    ) -> Option<NodeResources> {
        // start crawling through the nodes by total cpu
        for node_map in self.nodes.values_mut().rev() {
            // if this image has node restrictions then follow them
            if let Some(restrictions) = nodes {
                // get the first node that has enough resources for us
                if let Some(name) = node_map
                    .iter_mut()
                    // filter out any nodes that do not meet our restrictions
                    .filter(|(name, _)| restrictions.contains(*name))
                    .find(|(_, node)| node.spawnable(image, pool))
                    .map(|(name, _)| name.to_owned())
                {
                    // get this node from our map
                    if let Some(mut node) = node_map.remove(&name) {
                        // consume the resources for this image
                        node.available.consume(&image.resources, 1);
                        // consume a spawn slot
                        node.spawn_slots.consume(pool);
                        // we spawned something so return true
                        return Some(node);
                    }
                }
            } else {
                // this image has no node restrictions
                // get the first node that has enough resources for us
                if let Some(name) = node_map
                    .iter()
                    .find(|(_, node)| node.spawnable(image, pool))
                    .map(|(name, _)| name.to_owned())
                {
                    // get this node from our map
                    if let Some(mut node) = node_map.remove(&name) {
                        // consume the resources for this image
                        node.available.consume(&image.resources, 1);
                        // consume a spawn slot
                        node.spawn_slots.consume(pool);
                        // we spawned something so return true
                        return Some(node);
                    }
                }
            }
        }
        None
    }

    /// Free any resources tied to workers that no longer exist
    ///
    /// # Arguments
    ///
    /// * `alive` - The set of all currently alive pods
    /// * `deleted` - The names of the workers to delete
    /// * `frees` - The resources we have freed
    /// * `image_counts` - The current counts for spawned images
    /// * `req_counts` - The current counts for workers by requisition
    fn free(
        &mut self,
        alive: &HashSet<String>,
        deleted: &HashSet<String>,
        freed: &mut PoolFrees,
        image_counts: &mut HashMap<String, HashMap<String, u64>>,
        req_counts: &mut HashMap<Requisition, i64>,
    ) {
        // track the current resources for this cluster
        let mut new_total = Resources::default();
        // swap our old node tree with a new one
        let old_nodes = std::mem::take(&mut self.nodes);
        // crawl the cpu groups in this cluster
        for mut node_group in old_nodes.into_values() {
            // crawl over the nodes in this cpu group any resources we can
            for (node_name, mut node) in node_group.drain() {
                // try to free any resources we can on this node
                node.free(alive, deleted, freed, image_counts, req_counts);
                // add this nodes updated resources to our new total
                new_total += node.available;
                // just retain this nodes current info since it was not in the update
                // get an entry to this nodes current cpu group
                let cpu_entry = self.nodes.entry(node.available.cpu).or_default();
                // add this node to its new cpu group
                cpu_entry.insert(node_name, node);
            }
        }
        // replace our total resource counts
        self.resources = new_total;
    }

    /// Determine if this cluster has a more then some % of resources remaining
    ///
    /// This currently only evaluates cpu/memory.
    ///
    /// # Arguments
    ///
    /// * `remaining` - The percentage to check for
    #[instrument(name = "ClusterResources::has_remaining", skip(self), fields(cluster = self.name))]
    pub fn has_remaining(&self, remaining: f64) -> bool {
        // if total resources for either then log it and return 0
        if self.total.cpu == 0 || self.total.memory == 0 {
            event!(
                Level::INFO,
                no_resources = true,
                cpu = self.total.cpu,
                memory = self.total.memory
            );
            // return true since we cannot free no resources on this cluster as it has none
            return true;
        }
        // get the % of remaining cpu and memory
        let cpu_remaining = self.resources.cpu as f64 / self.total.cpu as f64;
        let memory_remaining = self.resources.memory as f64 / self.total.memory as f64;
        // log the % of resources we have remaining
        event!(Level::INFO, cpu_remaining, memory_remaining);
        // check if are above our remaining value or not
        cpu_remaining > remaining && memory_remaining > remaining
    }
}

/// Resoruces for one node or k8s cluster
#[derive(Debug, Clone)]
pub struct NodeResources {
    /// The name of this node
    pub name: String,
    /// The available resources for this node
    pub available: Resources,
    /// the total resources this node has
    pub total: Resources,
    /// The workers that are spawned on this node
    pub spawned: BTreeMap<DateTime<Utc>, Vec<Spawned>>,
    /// The number of spawn slots for this node
    pub spawn_slots: SpawnSlots,
}

impl NodeResources {
    /// Create a new `NodeResources`
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this node
    pub fn new(name: String, spawn_slots: SpawnSlots) -> Self {
        NodeResources {
            name,
            available: Resources::default(),
            total: Resources::default(),
            spawned: BTreeMap::default(),
            spawn_slots,
        }
    }

    /// Check if an image is able to be spawned on a specific node
    ///
    /// # Arguments
    ///
    /// * `image` - The image we want to spawn
    pub fn spawnable(&self, image: &Image, pool: Pools) -> bool {
        // check if we have enough spawn slots for this pod
        if !self.spawn_slots.enough(pool) {
            return false;
        }
        // make sure this node has enough resources for this image
        self.available.enough(&image.resources)
    }

    /// Free any resources tied to workers that no longer exist
    ///
    /// # Arguments
    ///
    /// * `alive` - The set of all currently alive pods
    /// * `deleted` - The set of deleted workers
    /// * `freed` - The resources that we have freed in each pool
    /// * `image_counts` - The current counts for spawned images
    /// * `req_counts` - The current counts for workers by requisition
    fn free(
        &mut self,
        alive: &HashSet<String>,
        deleted: &HashSet<String>,
        freed: &mut PoolFrees,
        image_counts: &mut HashMap<String, HashMap<String, u64>>,
        req_counts: &mut HashMap<Requisition, i64>,
    ) {
        // crawl our worker deadline groups
        for workers in self.spawned.values_mut() {
            // remove any workers that need to be freed
            workers.retain(|worker| {
                // check if this worker is still in our active set
                if deleted.contains(&worker.name) || !alive.contains(&worker.name) {
                    // free these resources for our node
                    self.available += worker.resources;
                    // free resources from the right pool
                    freed.add(worker.pool, worker.resources);
                    // get this groups image map
                    if let Some(image_map) = image_counts.get_mut(&worker.req.group) {
                        // get this image current spawn count
                        if let Some(count) = image_map.get_mut(&worker.req.stage) {
                            // decrement this images count by 1
                            *count = count.saturating_sub(1);
                        }
                    }
                    // get the count for this requisition type
                    if let Some(req_count) = req_counts.get_mut(&worker.req) {
                        // decrement our count by 1
                        *req_count -= 1;
                    }
                    false
                } else {
                    true
                }
            });
        }
        // drop any empty spawned groups
        self.spawned.retain(|_, workers| !workers.is_empty());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use chrono::Duration;

    /// Build a spawned deadline worker eligible to be scaled down, tied to the given req
    ///
    /// The worker is placed in the deadline pool, marked not-yet-scaled-down, and given a
    /// past `down_scalable` time so it always satisfies the "has run long enough" checks.
    ///
    /// # Arguments
    ///
    /// * `req` - The requisition this worker was spawned for
    /// * `resources` - The resources this worker is currently holding
    fn scalable_worker(req: Requisition, resources: Resources) -> Spawned {
        // build a spawned worker that is otherwise a valid scale down candidate
        Spawned {
            req,
            cluster: "cluster".to_owned(),
            node: "node".to_owned(),
            name: "worker".to_owned(),
            resources,
            pool: Pools::Deadline,
            spawn: true,
            scaled_down: false,
            // set the down scalable time in the past so it is eligible to scale down
            down_scalable: Utc::now() - Duration::minutes(5),
        }
    }

    /// The victim filter in `scale_down_to_meet` excludes a worker only when it is the *same*
    /// requisition (matches on user AND group AND pipeline AND stage), so a worker that shares
    /// just one field with the incoming req (here the group) is still selected as a victim.
    #[test]
    fn victim_sharing_one_field_is_still_selected() {
        // the incoming requisition we are trying to make room for
        let req = Requisition::new("alice", "corn", "pipe-a", "stage-a");
        // a spawned worker that shares only the group but differs on user/pipeline/stage
        let other = Requisition::new("bob", "corn", "pipe-b", "stage-b");
        let worker = scalable_worker(other, Resources::new(500, 256, 0, 1));
        // this worker is a different requisition, so it must be a valid scale down victim
        assert!(
            is_scale_down_victim(&worker, &req, Utc::now()),
            "a worker sharing only one field with the incoming req is a different requisition \
             and must be selectable as a scale down victim"
        );
    }

    /// A worker for the exact same requisition is never selected as a victim of itself, so the
    /// scaler never scales down the very workers it is trying to spawn.
    #[test]
    fn identical_requisition_is_never_a_victim() {
        // the incoming requisition we are trying to make room for
        let req = Requisition::new("alice", "corn", "pipe-a", "stage-a");
        // a spawned worker for the exact same requisition
        let worker = scalable_worker(req.clone(), Resources::new(500, 256, 0, 1));
        // an identical requisition must be excluded from scale down candidates
        assert!(
            !is_scale_down_victim(&worker, &req, Utc::now()),
            "a worker for the same requisition must never be scaled down to make room for itself"
        );
    }

    /// Consuming a spawn slot when the per-loop budget is already zero leaves it at zero. The
    /// per-node fairshare capacity can exceed the global `spawn_limit`, and an underflow would
    /// either panic (overflow-checked builds) or wrap to `usize::MAX` and defeat the per-loop
    /// spawn cap.
    #[test]
    fn consuming_slot_at_zero_does_not_underflow() {
        // the per-loop spawn budget has already been fully exhausted
        let mut spawn_slots = 0usize;
        // consuming another slot must not wrap the budget around to a huge value
        consume_spawn_slot(&mut spawn_slots);
        assert_eq!(
            spawn_slots, 0,
            "consuming a spawn slot at zero must stay at zero, not underflow to usize::MAX"
        );
    }

    /// The deadline pool holds only "whatever resources are in no other pools": the total cluster
    /// capacity minus the fairshare pool's reserve (32000m - 8000m = 24000m cpu). This keeps both
    /// pools from claiming the same physical capacity and voiding the fairshare reservation.
    #[test]
    fn deadline_pool_excludes_fairshare_reserve() {
        // the summed available capacity across every cluster
        let total = Resources::new(32000, 65536, 0, 40);
        // the fairshare pool holds an 8-core / 16 GiB reserve that must be held back
        let fairshare_reserve = Resources::new(8000, 16384, 0, 10);
        // size the deadline pool given the total capacity and the fairshare reserve
        let deadline = Allocatable::deadline_pool_resources(total, fairshare_reserve);
        // the deadline pool must not claim the cpu the fairshare reserve guarantees
        assert_eq!(
            deadline.cpu, 24000,
            "the deadline pool must exclude the fairshare cpu reserve (32000 - 8000)"
        );
        // nor may it claim the reserved memory
        assert_eq!(
            deadline.memory, 49152,
            "the deadline pool must exclude the fairshare memory reserve (65536 - 16384)"
        );
    }

    /// With no fairshare reserve configured the deadline pool holds the full cluster capacity, so
    /// the deadline pool is not shrunk when nothing is reserved.
    #[test]
    fn deadline_pool_is_full_total_without_fairshare_reserve() {
        // the summed available capacity across every cluster
        let total = Resources::new(32000, 65536, 0, 40);
        // no fairshare reserve is held back
        let deadline = Allocatable::deadline_pool_resources(total, Resources::default());
        // the deadline pool gets everything
        assert_eq!(deadline, total);
    }

    /// Build a spawned worker in group `corn` for image `husker`
    ///
    /// # Arguments
    ///
    /// * `name` - The unique name of this worker
    /// * `pool` - The pool this worker was spawned in
    /// * `resources` - The resources this worker holds
    fn corn_worker(name: &str, pool: Pools, resources: Resources) -> Spawned {
        Spawned {
            name: name.to_owned(),
            pool,
            ..scalable_worker(
                Requisition::new("user", "corn", "pipeline", "husker"),
                resources,
            )
        }
    }

    /// A failed spawn gives back the fairshare pool resources and spawn limit image count that
    /// `try_allocate` consumed for it. Nothing else releases them, so this keeps failed spawns
    /// from permanently shrinking the fairshare pool and occupying spawn limit slots until the
    /// scaler restarts.
    #[test]
    fn failed_spawn_releases_pool_and_image_count() {
        // a fairshare pool that has already given 2 cores / 4 GiB to the spawn that will fail
        let resources = Resources::new(2000, 4096, 0, 0);
        let mut fairshare_pool = Pool::default();
        fairshare_pool.resources = Resources::new(6000, 12288, 0, 0);
        // the spawn limited image was counted when it was allocated
        let mut image_counts: HashMap<String, HashMap<String, u64>> = HashMap::default();
        image_counts
            .entry("corn".to_owned())
            .or_default()
            .insert("husker".to_owned(), 1);
        // the scheduler fails to create this fairshare worker
        let spawn = corn_worker("w1", Pools::FairShare, resources);
        let mut deleted = HashSet::default();
        let mut failed = WorkerDeleteMap::default();
        record_failed_spawn(
            &mut fairshare_pool,
            &mut image_counts,
            spawn,
            &mut deleted,
            &mut failed,
        );
        // its consumed resources must be returned to the fairshare pool
        assert_eq!(
            fairshare_pool.resources,
            Resources::new(8000, 16384, 0, 0),
            "a failed spawn must release its fairshare pool resources"
        );
        // and its spawn limit slot must be released
        assert_eq!(
            image_counts["corn"]["husker"], 0,
            "a failed spawn must release its spawn limit image count"
        );
    }

    /// A failed spawn is deleted from Thorium.
    #[test]
    fn failed_spawn_is_deleted() {
        let mut fairshare_pool = Pool::default();
        let mut image_counts: HashMap<String, HashMap<String, u64>> = HashMap::default();
        let spawn = corn_worker("w1", Pools::FairShare, Resources::default());
        let mut deleted = HashSet::default();
        let mut failed = WorkerDeleteMap::default();
        record_failed_spawn(
            &mut fairshare_pool,
            &mut image_counts,
            spawn,
            &mut deleted,
            &mut failed,
        );
        // the failed worker is tracked as deleted and removed from Thorium
        assert!(deleted.contains("w1"));
        assert_eq!(failed.workers, vec!["w1".to_owned()]);
    }

    /// Build a worker registered in Thorium in group `corn` for image `husker`
    ///
    /// # Arguments
    ///
    /// * `name` - The unique name of this worker
    fn registered_corn_worker(name: &str) -> Worker {
        Worker {
            cluster: "cluster".to_owned(),
            node: "node".to_owned(),
            scaler: ImageScaler::K8s,
            name: name.to_owned(),
            user: "user".to_owned(),
            group: "corn".to_owned(),
            pipeline: "pipeline".to_owned(),
            stage: "husker".to_owned(),
            status: thorium::models::WorkerStatus::Running,
            spawned: Utc::now(),
            heart_beat: None,
            resources: Resources::default(),
            pool: Pools::Deadline,
            active: None,
        }
    }

    /// Run the two free paths `free_deleted` applies to a node running two `husker` workers
    ///
    /// # Arguments
    ///
    /// * `failed` - The workers that have failed
    fn free_two_husker_workers(failed: &HashSet<String>) -> u64 {
        // two husker workers are running so the spawn limit count is 2
        let mut image_counts: HashMap<String, HashMap<String, u64>> = HashMap::default();
        image_counts
            .entry("corn".to_owned())
            .or_default()
            .insert("husker".to_owned(), 2);
        // Thorium lists both workers on this node
        let listed = vec![
            ("w1".to_owned(), registered_corn_worker("w1")),
            ("w2".to_owned(), registered_corn_worker("w2")),
        ];
        // prune any failed workers from the listing
        let mut alive = HashSet::default();
        let mut deletes = WorkerDeleteMap::default();
        prune_failed_node_workers(failed, listed, &mut alive, &mut deletes);
        // the scaler's node also tracks both spawned workers
        let mut node = NodeResources::new("node".to_owned(), SpawnSlots::default());
        node.spawned.insert(
            Utc::now(),
            vec![
                corn_worker("w1", Pools::Deadline, Resources::default()),
                corn_worker("w2", Pools::Deadline, Resources::default()),
            ],
        );
        // free any workers that are no longer alive
        node.free(
            &alive,
            &HashSet::default(),
            &mut PoolFrees::default(),
            &mut image_counts,
            &mut HashMap::default(),
        );
        image_counts["corn"]["husker"]
    }

    /// Across both free paths in `free_deleted` (`prune_failed_workers` then
    /// `NodeResources::free`) a failed worker decrements its image count exactly once, so one
    /// failed worker out of two drops the count from 2 to 1. This keeps spawn limited images from
    /// running more concurrent workers than their limit allows.
    #[test]
    fn failed_worker_decrements_image_count_once() {
        // one of the two running workers failed
        let failed: HashSet<String> = ["w1".to_owned()].into_iter().collect();
        // one worker is still running so the count must be 1
        assert_eq!(
            free_two_husker_workers(&failed),
            1,
            "a single failed worker must only decrement its image count once"
        );
    }

    /// When no worker failed the image count is unchanged.
    #[test]
    fn alive_workers_keep_their_image_count() {
        assert_eq!(free_two_husker_workers(&HashSet::default()), 2);
    }

    /// The result of applying a resources update to a cluster running one `husker` worker
    struct UpdatedCluster {
        /// The cluster after the update was applied
        cluster: ClusterResources,
        /// The resources the update freed
        freed: PoolFrees,
        /// The image counts after the update was applied
        image_counts: HashMap<String, HashMap<String, u64>>,
        /// The requisition counts after the update was applied
        req_counts: HashMap<Requisition, i64>,
    }

    /// Apply a resources update to a cluster whose only node is running fairshare worker `w1`
    ///
    /// # Arguments
    ///
    /// * `active` - The workers the update reports as active on the node
    fn update_cluster_running_w1(active: &[&str]) -> UpdatedCluster {
        // w1 holds 2 cores and 4 GiB from the fairshare pool
        let worker = corn_worker("w1", Pools::FairShare, Resources::new(2000, 4096, 0, 0));
        // w1 is counted against its spawn limit and its requisition count
        let mut image_counts: HashMap<String, HashMap<String, u64>> = HashMap::default();
        image_counts
            .entry("corn".to_owned())
            .or_default()
            .insert("husker".to_owned(), 1);
        let mut req_counts: HashMap<Requisition, i64> = HashMap::default();
        req_counts.insert(worker.req.clone(), 1);
        // the scaler tracks w1 as spawned on its node
        let mut node = NodeResources::new("node".to_owned(), SpawnSlots::default());
        node.spawned.insert(Utc::now(), vec![worker]);
        // build a cluster containing just this node
        let mut cluster = ClusterResources {
            name: "cluster".to_owned(),
            resources: Resources::default(),
            total: Resources::default(),
            low_resources: false,
            nodes: BTreeMap::default(),
            spawn_slots_reset: SpawnSlots::default(),
        };
        cluster
            .nodes
            .entry(node.available.cpu)
            .or_default()
            .insert(node.name.clone(), node);
        // the resources update lists this node with the given active workers
        let node_resources = Resources::new(8000, 16384, 0, 10);
        let mut node_update = NodeAllocatableUpdate::new(node_resources, node_resources);
        node_update
            .active
            .extend(active.iter().map(|name| (*name).to_owned()));
        let mut update = AllocatableUpdate::default();
        update.nodes.insert("node".to_owned(), node_update);
        // apply the update to our cluster
        let freed = cluster.update(update, &mut image_counts, &mut req_counts);
        UpdatedCluster {
            cluster,
            freed,
            image_counts,
            req_counts,
        }
    }

    /// Check if a worker is still tracked as spawned anywhere in a cluster
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster to check
    /// * `name` - The name of the worker to look for
    fn is_spawned(cluster: &ClusterResources, name: &str) -> bool {
        cluster
            .nodes
            .values()
            .flat_map(HashMap::values)
            .flat_map(|node| node.spawned.values().flatten())
            .any(|spawned| spawned.name == name)
    }

    /// A resources update frees a worker that is missing from its node's active set: it is no
    /// longer tracked as spawned, its fairshare resources are returned, and it no longer counts
    /// against its spawn limit or requisition count.
    #[test]
    fn update_frees_inactive_workers() {
        // the update lists the node but w1 is no longer active on it
        let updated = update_cluster_running_w1(&[]);
        // w1 must no longer be tracked as spawned
        assert!(
            !is_spawned(&updated.cluster, "w1"),
            "a worker missing from its node's active set must be freed on a resources update"
        );
        // its fairshare resources must be returned
        assert_eq!(
            updated.freed.fairshare,
            Resources::new(2000, 4096, 0, 0),
            "an inactive worker's resources must be returned to its pool"
        );
        // and it must no longer count against its spawn limit or requisition count
        assert_eq!(updated.image_counts["corn"]["husker"], 0);
        assert_eq!(updated.req_counts.values().sum::<i64>(), 0);
    }

    /// A worker that is still active on its node is kept and still counted.
    #[test]
    fn update_frees_inactive_workers_keeps_active_workers() {
        // the update lists the node and w1 is still active on it
        let updated = update_cluster_running_w1(&["w1"]);
        // w1 is still spawned and nothing was freed
        assert!(is_spawned(&updated.cluster, "w1"));
        assert_eq!(updated.freed.fairshare, Resources::default());
        assert_eq!(updated.freed.deadline, Resources::default());
        // and it still counts against its spawn limit and requisition count
        assert_eq!(updated.image_counts["corn"]["husker"], 1);
        assert_eq!(updated.req_counts.values().sum::<i64>(), 1);
    }
}
