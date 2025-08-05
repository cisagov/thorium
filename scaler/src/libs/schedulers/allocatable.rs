//! All resources that are allocatable by this scaler
//!
//! This can be resources across multiple clusters

use chrono::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use hashbrown::HashMap;
use thorium::conf::{FairShareWeights, IsRestricted, WorkerRestrictions};
use thorium::models::{
    Deadline, Image, ImageScaler, NodeListParams, Pools, Requisition, Resources, SpawnLimits,
    SpawnMap, SystemSettings, WorkerDeleteMap,
};
use thorium::{Conf, Error, Thorium};
use tracing::{event, instrument, Level, Span};

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
    /// * `available` - The available resources to schedule on this node
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
pub struct Allocatable {
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

    /// Remove and return the cpu group and cluster info for a specific cluster
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
            None => (0, name.to_owned(), ClusterResources::new(name.to_owned())),
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
        self.deadlines_pool.resources = total;
    }

    /// Log all clusters current resources
    #[instrument(name = "Allocatable::log_resources", skip_all)]
    pub fn log_resources(&self) {
        // crawl each cpu group
        for cpu_group in self.clusters.values() {
            // crawl the clusters in this cpu group
            for (cluster, cluster_info) in cpu_group {
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
                            spawn_slots = node_info.spawn_slots
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
            // check if this cluster has less than 5% of its resources remaining {
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
                let count = match image_map.get_mut(&image.name) {
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
                    // increment our count
                    *count += 1;
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
    fn allocate_cluster_helper(&mut self, image: &Image) -> Option<(u64, String, NodeResources)> {
        // crawl over all nodes until we find one that we can fit on
        for (cpus, cluster_map) in self.clusters.iter_mut().rev() {
            // iterate over the clusters that have the same number of cores
            for (cluster_name, cluster) in cluster_map.iter_mut() {
                // check if this image has any restrictions
                let nodes = match self.restrictions.check(cluster_name, image) {
                    IsRestricted::No => None,
                    IsRestricted::Yes(nodes) => Some(nodes),
                    IsRestricted::WrongCluster => continue,
                };
                // try to consume the resources for this image on a node
                if let Some(node) = cluster.allocate_node(image, nodes) {
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
    fn allocate_cluster(&mut self, image: &Image) -> Option<(String, String)> {
        // locate the cluster and node that we are allocating resources on
        if let Some((cpus, cluster_name, node)) = self.allocate_cluster_helper(image) {
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
            return match self.allocate_cluster(image) {
                Some((cluster, node)) => {
                    // consume the resources from the correct pool
                    self.consume(image, pool);
                    // return our cluster and node
                    Some((cluster, node))
                }
                None => None,
            };
        }
        // we could not spawn this image
        None
    }

    /// Calculate the increase in fair share for a specific resource spec
    ///
    /// # Arguments
    ///
    /// * `rank` - A users current fair share rank
    /// * `count` - The number of pods to use for this increase
    /// * `resources` - The resource spec we are calculating an increase for
    pub fn calc_fair_share(&self, rank: u64, count: i64, resources: &Resources) -> u64 {
        // only calculate an increase if count is greater than 0
        if count > 0 {
            // calculate the increase in cost for the cpu
            let mut incr = resources.cpu * self.weights.cpu;
            // add the cost for the memory
            incr += resources.memory * self.weights.memory;
            // add this increase to our old rank
            rank + (incr * count as u64)
        } else {
            // count is zero so just use our old rank
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
                            *spawn_slots -= 1;
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
        // get the system stats for Thorium so we can assign resources to users
        let stats = thorium.system.stats().await?;
        // Build a map of each users outstanding job reqs
        let mut map = stats.users_jobs();
        // track the users that do not spawn any jobs
        let mut no_spawns = BTreeMap::default();
        // iterate over our users based on how many resources they have consumed
        while let Some((rank, users)) = self.fair_share.pop_first() {
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
            for clusters in cluster_group.values_mut() {
                // crawl over all node cpu groups
                for node_group in clusters.nodes.values_mut() {
                    // add this size of this group to our node cnt
                    node_cnt += node_group.len();
                    // reset this nodes spawn limit
                    node_group
                        .values_mut()
                        .for_each(|node| node.spawn_slots = 2);
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
                // otherwise we will repeatedly spawn workers for the same deadlines
                match count.cmp(&&mut 0) {
                    // we spawned a worker for this deadline in the past
                    Ordering::Greater => {
                        // decrement our count
                        *count -= 1;
                        // skip this deadline
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
                                if !spawned.scaled_down
                                    && spawned.down_scalable < now
                                    && spawned.pool == Pools::Deadline
                                    && spawned.req.user != req.user
                                    && spawned.req.group != req.group
                                    && spawned.req.pipeline != req.pipeline
                                    && spawned.req.stage != req.stage
                                {
                                    // add this workers resources to our freeable set
                                    freeable += image.resources;
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
        // build a map to store what changes happened on what cluster/node
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

    /// Increase our fair share ranks for all resources consumed by each user
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
                        // calculate a rank for a user not in the fair share ranks yet
                        let rank = self.calc_fair_share(0, *count, &image.resources);
                        // insert our new rank
                        by_user.insert(req.user.clone(), rank);
                    }
                }
            }
        }
        // reinsert our fair share ranks
        for (user, rank) in by_user {
            // get an entry into this fair share ranks set
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
            // get an entry into this fair share ranks set
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
    async fn prune_failed_workers(
        &mut self,
        thorium: &Thorium,
        failed: &HashSet<String>,
    ) -> Result<HashSet<String>, Error> {
        // scan our nodes 50 at a time
        let params = NodeListParams::default().scaler(self.scaler_type).limit(50);
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
                // crawl over the workers on this node
                for (name, worker) in node.workers {
                    // if this worker has failed then add it to our delete map
                    if failed.contains(&name) {
                        // add this failed worker to our delete map
                        deletes.add_mut(name);
                        // get this groups image map
                        if let Some(image_map) = self.image_counts.get_mut(&worker.group) {
                            // get this image current spawn count
                            if let Some(count) = image_map.get_mut(&worker.stage) {
                                // decrement this images count by 1
                                *count = count.saturating_sub(1);
                            }
                        }
                    } else {
                        // this worker is alive so add its name to our alive set
                        alive.insert(name);
                    }
                }
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
    #[instrument(name = "Allocatable::free_deleted", skip_all)]
    pub async fn free_deleted(
        &mut self,
        thorium: &Thorium,
        failed: &HashSet<String>,
        deleted: &HashSet<String>,
    ) -> Result<(), Error> {
        // get all currently alive workers
        let alive = self.prune_failed_workers(thorium, failed).await?;
        // track the resources we have freed
        let mut freed = PoolFrees::default();
        // swap our old cluster tree out for a new one
        let old_clusters = std::mem::take(&mut self.clusters);
        // crawl over our cluster cpu groups
        for cluster_group in old_clusters.into_values() {
            // crawl over the clusters in this cpu group
            for (name, mut cluster) in cluster_group {
                // free any deleted workers in this cluster
                cluster.free(
                    &alive,
                    deleted,
                    &mut freed,
                    &mut self.image_counts,
                    &mut self.counts,
                );
                // get an entry to this clusters new cpu group
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
}

impl ClusterResources {
    /// Create a new resources object for a cluster
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this cluster
    pub fn new(name: String) -> Self {
        ClusterResources {
            name,
            resources: Resources::default(),
            total: Resources::default(),
            low_resources: false,
            nodes: BTreeMap::default(),
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
        for (name, node_update) in update.nodes.drain() {
            // get this nodes entry
            let entry = temp_nodes
                .entry(name.clone())
                .or_insert_with(|| NodeResources::new(name));
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
    fn allocate_node(
        &mut self,
        image: &Image,
        nodes: Option<&HashSet<String>>,
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
                    .find(|(_, node)| node.spawnable(image))
                    .map(|(name, _)| name.to_owned())
                {
                    // get this node from our map
                    if let Some(mut node) = node_map.remove(&name) {
                        // consume the resources for this image
                        node.available.consume(&image.resources, 1);
                        // consume a spawn slot
                        node.spawn_slots -= 1;
                        // we spawned something so return true
                        return Some(node);
                    }
                }
            } else {
                // this image has no node restrictions
                // get the first node that has enough resources for us
                if let Some(name) = node_map
                    .iter()
                    .find(|(_, node)| node.spawnable(image))
                    .map(|(name, _)| name.to_owned())
                {
                    // get this node from our map
                    if let Some(mut node) = node_map.remove(&name) {
                        // consume the resources for this image
                        node.available.consume(&image.resources, 1);
                        // consume a spawn slot
                        node.spawn_slots -= 1;
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

    /// Determine if this cluster has a more than some % of resources remaining
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

/// Resources for one node or k8s cluster
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
    pub spawn_slots: u64,
}

impl NodeResources {
    /// Create a new `NodeResources`
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this node
    pub fn new(name: String) -> Self {
        NodeResources {
            name,
            available: Resources::default(),
            total: Resources::default(),
            spawned: BTreeMap::default(),
            spawn_slots: 2,
        }
    }

    /// Check if an image is able to be spawned on a specific node
    ///
    /// # Arguments
    ///
    /// * `image` - The image we want to spawn
    pub fn spawnable(&self, image: &Image) -> bool {
        // check if we have enough spawn slots for this pod
        if self.spawn_slots == 0 {
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
