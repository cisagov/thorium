//! Schedules resources in Thorium based on the deadline stream

use chrono::prelude::*;
use futures::stream::{self, StreamExt};
use futures::{poll, task::Poll};
use hashbrown::HashMap;
use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thorium::models::system::WorkerRegistrationList;
use thorium::models::{Deadline, ImageScaler, StageLogsAdd, WorkerDeleteMap, WorkerRegistration};
use thorium::{Conf, Error, Thorium};
use tokio::task::JoinHandle;
use tracing::{Level, event, instrument, span};

use super::Cache;
use super::schedulers::{self, Allocatable, ReqMap, Scheduler, WorkerDeletion};
use super::tasks::{self, TaskResult, Tasks, ZombieChecker};
use crate::args::Args;
use crate::from_now;
use crate::libs::Spawned;

/// returns false if this is false
macro_rules! filter {
    ($test:expr) => {
        if !$test {
            return false;
        }
    };
}

/// unwraps a cache group map and returns false if the target group doesn't exist
macro_rules! get_group {
    ($map:expr, $group:expr) => {
        if let Some(inner) = $map.get($group) {
            inner
        } else {
            return false;
        }
    };
}

/// spawn an async tokio task
macro_rules! spawn {
    ($vec:expr, $future:expr) => {
        $vec.push(tokio::spawn(async move { $future.await }))
    };
}

macro_rules! add_task {
    ($scaler:expr, $task:expr) => {{
        // get the datetime to start this task at
        let mut start = crate::from_now!($task.delay(&$scaler.conf) as i64);
        // increase by 1 until we have found an open slot to start this job
        loop {
            // determine if a task already exists for this date
            if $scaler.tasks.get(&start).is_none() {
                break;
            }
            // increment start by 1 and try again
            start += chrono::Duration::seconds(1);
        }
        $scaler.tasks.insert(start, $task)
    }};
}

/// The different reasons a worker may have failed out
///
/// Hash/Eq/PartialEq is implemented on solely the worker field for this. Meaning that when
/// placed in a hashed collection two different kinds with the same worker will
/// conflict/overwrite each other. This is done because having two different error out
/// reasons for the same worker is nonsensical and could result in trying to reset the
/// same worker twice.
#[derive(Debug, Clone, Eq)]
pub enum ErrorOutKinds {
    /// This worker exceeded its memory allocations/limits
    Oom(String),
    /// This worker cannot be created due to a missing prereq
    #[allow(dead_code)]
    StuckCreating(String),
    /// This worker cannot be created as its image cannot be pulled
    ImagePullBackoff(String),
}

impl ErrorOutKinds {
    /// create a report that a worker has OOM'd
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker that has been oomed
    pub fn oom<W: Into<String>>(worker: W) -> Self {
        ErrorOutKinds::Oom(worker.into())
    }

    /// create a report that a worker is stuck creating
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker that is stuck in the creating state
    #[allow(dead_code)]
    pub fn stuck_creating<W: Into<String>>(worker: W) -> Self {
        ErrorOutKinds::StuckCreating(worker.into())
    }

    /// create a report that a worker's image cannot be pulled
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker whose image cannot be pulled
    pub fn image_pull_backoff<W: Into<String>>(worker: W) -> Self {
        ErrorOutKinds::ImagePullBackoff(worker.into())
    }

    /// Get the name of the worker regardless of reason
    pub fn worker(&self) -> &String {
        match self {
            Self::Oom(worker) => worker,
            Self::StuckCreating(worker) => worker,
            Self::ImagePullBackoff(worker) => worker,
        }
    }

    /// Get our reason as a string
    pub fn reason_as_str(&self) -> &'static str {
        match self {
            Self::Oom(_) => "OOM",
            Self::StuckCreating(_) => "StuckCreating",
            Self::ImagePullBackoff(_) => "ImagePullBackoff",
        }
    }
}

// Only implement Hash on worker to prevent multiple
// error out reasons for the same worker being stored at once.
impl Hash for ErrorOutKinds {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // get our worker
        let worker = self.worker();
        // hash our worker
        state.write(worker.as_bytes());
    }
}

// Only implement Partial EQ on worker to prevent multiple
// error out reasons for the same worker being stored at once.
impl PartialEq for ErrorOutKinds {
    fn eq(&self, other: &Self) -> bool {
        // check if we have the same worker
        self.worker() == other.worker()
    }
}

/// Sync our ban sets against a newly pulled cache
///
/// This is the cache arm of `Scaler::check_tasks` extracted so it can be unit tested. The stale
/// bans from the old cache are cleared first and then every scheduler repopulates the ban sets
/// from the new cache, so the bans derived from the new cache are kept.
///
/// # Arguments
///
/// * `bans` - The ban sets to sync
/// * `sync` - Syncs every scheduler against the new cache, adding any bans it finds
async fn resync_bans<F>(bans: &mut BanSets, sync: F) -> Result<(), Error>
where
    F: AsyncFnOnce(&mut BanSets) -> Result<(), Error>,
{
    // clear the stale bans from our old cache before syncing
    bans.clear();
    // sync our schedulers against the new cache, adding any bans they find
    sync(bans).await?;
    Ok(())
}

/// Build the batches of worker registrations `Scaler::register_workers` submits
///
/// This is the batching in `Scaler::register_workers` extracted so it can be unit tested. A batch
/// is submitted once it reaches 25 workers and a fresh batch is started afterwards, so each worker
/// is registered exactly once.
///
/// # Arguments
///
/// * `reqs` - The requisitions whose spawns should be registered
fn registration_batches(reqs: &ReqMap) -> Vec<WorkerRegistrationList> {
    // track the batches to submit in order
    let mut batches = Vec::default();
    // build a list to track the new workers we are spawning
    let mut registrations = WorkerRegistrationList::default();
    // get the worker registration info for each cluster
    for (cluster, spawn_map) in &reqs.spawns {
        // crawl over the groups for this cluster
        for spawn_group in spawn_map.values() {
            // build a worker registration object for each spawned worker
            for spawn in spawn_group {
                // build this workers registration object
                let worker = WorkerRegistration::new(
                    cluster,
                    &spawn.node,
                    &spawn.name,
                    &spawn.req.user,
                    &spawn.req.group,
                    &spawn.req.pipeline,
                    &spawn.req.stage,
                    spawn.resources,
                    spawn.pool,
                );
                // add this new spawn to our list of workers to register
                registrations.add_mut(worker);
            }
            // if we have at least 25 workers then submit them to Thorium
            if registrations.workers.len() >= 25 {
                // submit this batch and start a new one so no worker is sent twice
                batches.push(std::mem::take(&mut registrations));
            }
        }
    }
    // skip registering new workers if we have none
    if !registrations.workers.is_empty() {
        batches.push(registrations);
    }
    batches
}

/// Get the clusters `Scaler::schedule` asks to delete workers and the scale downs for each
///
/// This is the cluster selection in `Scaler::schedule` extracted so it can be unit tested. Every
/// cluster with new scale down orders is returned with them, and every cluster whose scheduler
/// still has pending deletes from an earlier loop is returned with no new scale downs, so it is
/// still asked to validate them instead of leaving those workers in the being deleted state
/// until that cluster happens to get another scale down.
///
/// # Arguments
///
/// * `scale_down` - The new scale down orders by cluster
/// * `pending` - The clusters whose schedulers have pending deletes to validate
fn delete_targets(
    mut scale_down: HashMap<String, Vec<Spawned>>,
    pending: &HashSet<String>,
) -> HashMap<String, Vec<Spawned>> {
    // crawl over the clusters with pending deletes
    for cluster in pending {
        // make sure quiet clusters are still called to validate their pending deletes
        if !scale_down.contains_key(cluster) {
            scale_down.insert(cluster.clone(), Vec::default());
        }
    }
    scale_down
}

/// Banned groups and users due to setup errors
///
/// This will expire and reset every 10 minutes.
#[derive(Debug)]
pub struct BanSets {
    /// The type of scaler these bans are for
    scaler_type: ImageScaler,
    /// when this banlist should be refreshed
    pub expire: DateTime<Utc>,
    /// The groups to ban spawning resources for
    pub groups: HashSet<String>,
    /// The users to ban spawning resources for
    pub users: HashSet<String>,
    /// The requisitions to ban from spanwing
    pub reqs: HashMap<String, HashSet<String>>,
}

impl BanSets {
    /// Create a new ban set
    ///
    /// # Arguments
    ///
    /// * `scaler_type` - The type of scaler currently in use
    pub fn new(scaler_type: ImageScaler) -> Self {
        Self {
            scaler_type,
            expire: from_now!(600),
            groups: HashSet::default(),
            users: HashSet::default(),
            reqs: HashMap::default(),
        }
    }

    /// Filter deadlines that have been banned
    ///
    /// This does not do any checking of if we have resources to spawn them yet.
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium we use for scaling
    /// * `deadlines` - The deadlines to filter
    pub fn filter_deadlines(&self, cache: &Cache, deadline: &Deadline) -> bool {
        // if we don't know about this deadlines group then filter it
        filter!(cache.groups.contains(&deadline.group));
        // if we don't know about this deadlines users then filter it
        filter!(cache.users.contains_key(&deadline.creator));
        // only filter on docker image info if we are scheduling on k8s
        if self.scaler_type == ImageScaler::K8s {
            // if we don't know about this deadlines docker image then filter it
            filter!(get_group!(cache.docker, &deadline.group).contains_key(&deadline.stage));
        }
        // filter any banned groups
        filter!(!self.groups.contains(&deadline.group));
        // filter any banned users
        filter!(!self.users.contains(&deadline.creator));
        // if this group has any banned reqs then filter those out
        if let Some(banned_reqs) = self.reqs.get(&deadline.group) {
            // filter any banned requisition types
            filter!(!banned_reqs.contains(&deadline.stage));
        }
        true
    }

    /// check if this ban rule set is expired and clear it if so
    pub fn is_expired(&mut self) {
        // only clear our bans once their expiration has passed
        if self.expire <= Utc::now() {
            self.clear();
        }
    }

    /// Clear all banned items
    pub fn clear(&mut self) {
        self.expire = from_now!(600);
        self.groups.clear();
        self.users.clear();
        self.reqs.clear();
    }
}

/// Scales pods in k8s up and down based on the deadline stream
pub struct Scaler {
    /// The type of scheduler in use
    scaler_type: ImageScaler,
    /// The Thorium config
    conf: Conf,
    /// A client for Thorium
    thorium: Arc<Thorium>,
    /// A cache of info from Thorium we use for scaling
    pub cache: Arc<Cache>,
    /// The zombie checker
    pub zombies: ZombieChecker,
    /// The scheduler to use to scheduler resources
    schedulers: HashMap<String, Box<dyn Scheduler + Send>>,
    /// The clusters that need to be setup before we can schedule on them again
    unready: HashSet<String>,
    /// The resources currently available in this cluster
    pub allocatable: Allocatable,
    /// A queue of tasks to complete sorted by the time to start executing them
    tasks: BTreeMap<DateTime<Utc>, Tasks>,
    /// The currently active background tasks that have been spawned
    active: Vec<JoinHandle<Result<TaskResult, Error>>>,
}

impl Scaler {
    /// Creates a new Scaler
    ///
    /// # Arguments
    ///
    /// * `args` - The command line args passed to the scaler
    pub async fn new(args: Args) -> Result<Self, Error> {
        // try to load a config file
        let conf = Conf::new(&args.config)?;
        // build a Thorium client
        let thorium = Arc::new(Thorium::from_key_file(&args.auth).await?);
        // build our scaler
        Self::build(
            conf,
            args.auth,
            thorium,
            args.scaler,
            args.dry_run,
            &args.context_name,
        )
        .await
    }

    /// Creates a new Scaler
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config to use
    /// * `auth` - The path to the keys to use with Thorium
    /// * `thorium` - The Thorium client to use
    /// * `scaler` - The scaler type to set
    /// * `dry_run` - Whether we should use the dry run scheduler
    pub async fn build(
        conf: Conf,
        auth: String,
        thorium: Arc<Thorium>,
        scaler_type: ImageScaler,
        dry_run: bool,
        context_name: &String,
    ) -> Result<Self, Error> {
        // start our scaler creation span
        let span = span!(Level::INFO, "Scaler Initialization");
        // build the scheduler instances we are going to be scheduling jobs too
        let mut schedulers = HashMap::default();
        schedulers::new(&mut schedulers, &scaler_type, dry_run, context_name, &conf).await?;
        // build and load our cache
        let cache =
            Arc::new(Cache::new(thorium.clone(), conf.clone(), auth, scaler_type, &span).await?);
        // build a zombie checker
        let zombies = ZombieChecker::new(scaler_type, &thorium);
        // get our cluster settings
        let settings = thorium.system.get_settings().await?;
        // start with an empty allocatable object
        let allocatable = Allocatable::new(scaler_type, &conf, &settings, &cache);
        // build our task queue
        let tasks = tasks::Tasks::setup_queue(&conf);
        // all clusters need to be setup before we can schedule on them
        let unready = schedulers.keys().cloned().collect();
        // instance the scaler
        let scaler = Scaler {
            scaler_type,
            conf,
            thorium,
            cache,
            zombies,
            schedulers,
            unready,
            allocatable,
            tasks,
            active: Vec::default(),
        };
        Ok(scaler)
    }

    /// Setup any unready clusters and update the resources for all clusters this scaler can see
    ///
    /// Clusters that fail to setup or report their resources are removed from scheduling
    /// until they recover. This only errors if no clusters are available to schedule on.
    #[instrument(name = "Scaler::update_resources", skip_all, err(Debug))]
    async fn update_resources(&mut self) -> Result<(), Error> {
        // track how many clusters we can schedule on
        let mut ready = 0;
        // crawl over each cluster and get its available resources
        for (name, scheduler) in &mut self.schedulers {
            // setup this cluster if it hasn't been setup or has failed since
            if self.unready.contains(name) {
                if let Err(err) = scheduler
                    .setup(name, &self.cache, &mut self.allocatable.bans)
                    .await
                {
                    // log that we failed to setup this cluster
                    event!(
                        Level::ERROR,
                        msg = "Failed to setup cluster",
                        cluster = name,
                        error = err.msg()
                    );
                    // make sure we don't schedule anything on this cluster
                    self.allocatable.mark_unreachable(name);
                    continue;
                }
                // this cluster is setup and can be scheduled on again
                self.unready.remove(name);
                event!(Level::INFO, msg = "Cluster is setup", cluster = name);
            }
            // get the burstable resources config for this cluster
            let burstable_conf = self.conf.thorium.scaler.burstable(self.scaler_type, name);
            // get the available resources for this cluster
            match scheduler
                .resources_available(&self.thorium, &self.cache.settings, &burstable_conf)
                .await
            {
                Ok(update) => {
                    // update this clusters resources
                    self.allocatable.update(name, update);
                    ready += 1;
                }
                Err(err) => {
                    // log that we failed to get this clusters resources
                    event!(
                        Level::ERROR,
                        msg = "Failed to get cluster resources",
                        cluster = name,
                        error = err.msg()
                    );
                    // stop scheduling on this cluster and set it up again once it recovers
                    self.allocatable.mark_unreachable(name);
                    self.unready.insert(name.clone());
                }
            }
        }
        // make sure we have at least one cluster to schedule on
        if ready == 0 {
            return Err(Error::new("No clusters are available to schedule on"));
        }
        // resize our deadline pool
        self.allocatable.resize_deadline_pool();
        Ok(())
    }

    /// check if any of our spawned tasks have returned and handle the response
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    #[instrument(name = "Scaler::check_tasks", skip_all, err(Debug))]
    async fn check_tasks(&mut self) -> Result<(), Error> {
        // track any futures that haven't completed yet
        let mut uncompleted = Vec::default();
        // crawl over our active tasks and handle any that have completed
        for mut handle in self.active.drain(..) {
            // check if this future has completed
            if let Poll::Ready(join_result) = poll!(&mut handle) {
                // get our compelted task
                let completed = join_result??;
                // log that a task was completed
                event!(Level::INFO, task = completed.as_str());
                // check if a join error occured
                match completed {
                    // An updated cache has been pulled
                    TaskResult::Cache(cache) => {
                        // get our schedulers so they can sync against the new cache
                        let schedulers = &mut self.schedulers;
                        let unready = &self.unready;
                        // track the clusters that failed to sync
                        let mut failed = Vec::default();
                        // sync our ban sets against the new cache
                        resync_bans(&mut self.allocatable.bans, async |bans: &mut BanSets| {
                            // crawl over each cluster and sync any cache updates
                            for (name, scheduler) in schedulers.iter_mut() {
                                // unready clusters will be fully setup with this cache once they recover
                                if unready.contains(name) {
                                    continue;
                                }
                                if let Err(err) =
                                    scheduler.sync_to_new_cache(name, &cache, bans).await
                                {
                                    // log that we failed to sync this cluster
                                    event!(
                                        Level::ERROR,
                                        msg = "Failed to sync cluster to new cache",
                                        cluster = name,
                                        error = err.msg()
                                    );
                                    failed.push(name.clone());
                                }
                            }
                            Ok(())
                        })
                        .await?;
                        // stop scheduling on any failed clusters and set them up again once they recover
                        for name in failed {
                            self.allocatable.mark_unreachable(&name);
                            self.unready.insert(name);
                        }
                        // save the cache in the scaler for next update
                        self.cache = Arc::new(cache);
                        // add a new task for refreshing our cache
                        add_task!(self, Tasks::CacheReload)
                    }
                    // reschedule any other completed task
                    completed => add_task!(self, completed.next_task()),
                };
            } else {
                // this task hasn't completed so keep tracking it
                uncompleted.push(handle);
            }
        }
        // reinsert our uncompleted tasks
        self.active.append(&mut uncompleted);
        Ok(())
    }

    /// check if any tasks need to be executed and spawn them
    #[instrument(name = "Scaler::spawn_tasks", skip_all, err(Debug))]
    async fn spawn_tasks(&mut self) -> Result<(), Error> {
        // get the current timestamp to compare against
        let now = Utc::now();
        // get updated info on Thorium
        let info = self.thorium.system.get_info(Some(self.scaler_type)).await?;
        // check if we should reload our cache early
        if info.expired_cache(self.scaler_type) {
            // set our cache refresh to happen now
            self.tasks.retain(|_, task| *task != Tasks::CacheReload);
            self.tasks.insert(Utc::now(), Tasks::CacheReload);
        }
        // get any tasks we want to spawn and build a list of completed blocking tasks to rerun again
        let mut completed = Vec::default();
        // get the tasks that are ready to run
        let ready = self
            .tasks
            .extract_if(.., |time, _| time < &now)
            .collect::<Vec<_>>();
        for (_, task) in ready {
            // log that we are spawning a task
            event!(Level::INFO, task = task.as_str());
            // spawn or execute this task
            match task {
                // zombie checks require a mutable reference to the scheduler so we cannot run that
                // as a background task
                Tasks::ZombieJobs => {
                    // check for any zombie jobs and reset them
                    self.zombies.check(&self.allocatable).await?;
                    completed.push(Tasks::ZombieJobs);
                }
                // tell Thorium to sync ldap groups
                Tasks::LdapSync => {
                    // clone client so we can send it to another thread
                    let client = self.thorium.clone();
                    spawn!(self.active, tasks::sync_ldap(client));
                }
                // build a new cache object
                Tasks::CacheReload => {
                    // clone data so we can send it to another thread
                    let stale_cache = self.cache.clone();
                    let client = self.thorium.clone();
                    let scaler = self.scaler_type;
                    spawn!(self.active, Cache::refresh(stale_cache, client, scaler));
                }
                // get the total amount of resources available in the cluster
                Tasks::Resources => {
                    // update the resources for all clusters
                    self.update_resources().await?;
                    completed.push(Tasks::Resources);
                }
                // tell Thorium to update image runtimes
                Tasks::UpdateRuntimes => {
                    // clone client so we can send it to another thread
                    let client = self.thorium.clone();
                    spawn!(self.active, tasks::update_runtimes(client));
                }
                // tell Thorium to cleanup any expired data
                Tasks::Cleanup => {
                    // clone client so we can send it to another thread
                    let client = self.thorium.clone();
                    spawn!(self.active, tasks::cleanup(client));
                }
                // tell Thorium to decrease our users fair share ranks
                Tasks::DecreaseFairShare => {
                    // decrease our users fair share ranks
                    self.allocatable.decrease_fair_share_ranks(&self.conf);
                }
            };
        }
        // add any blocking completed tasks back to our task list
        for task in completed {
            add_task!(self, task);
        }
        Ok(())
    }

    /// Set all the workers we wish to spawn as being spawned
    ///
    /// # Arguments
    ///
    /// * `reqs` - The reqs we spawned
    #[instrument(name = "Scaler::register_workers", skip_all, err(Debug))]
    async fn register_workers(&self, reqs: &ReqMap) -> Result<(), Error> {
        // submit each batch of new workers to Thorium in order
        for registrations in registration_batches(reqs) {
            // try to register our new workers
            self.thorium
                .system
                .register_workers(self.scaler_type, &registrations)
                .await?;
        }
        Ok(())
    }

    /// Schedules resources based on our requisition map
    ///
    /// # Arguments
    ///
    /// * `reqs` - The requisition map to schedule off of
    /// * `span` - The span to log traces too
    #[allow(clippy::too_many_lines)]
    #[instrument(name = "Scaler::schedule", skip_all, err(Debug))]
    async fn schedule(&mut self) -> Result<HashSet<String>, Error> {
        // clone our current resource counts so we can track changes
        let past = self.allocatable.counts.clone();
        // track our succesful scale changes and the the correctly deleted ones
        let mut changes: HashMap<DateTime<Utc>, Vec<Spawned>> =
            HashMap::with_capacity(self.allocatable.changes.spawns.len());
        let mut deleted = HashSet::with_capacity(self.allocatable.changes.scale_down.len());
        // get the clusters whose schedulers still have pending deletes to validate
        let pending = self
            .schedulers
            .iter()
            .filter(|(_, scheduler)| scheduler.has_pending_deletes())
            .map(|(name, _)| name.clone())
            .collect::<HashSet<String>>();
        // get the clusters to scale down workers on or validate pending deletes for
        let scale_down = std::mem::take(&mut self.allocatable.changes.scale_down);
        let targets = delete_targets(scale_down, &pending);
        // start our delete workers span if we have workers to delete or pending deletes to validate
        if !targets.is_empty() {
            // start our delete workers span
            let del_span = span!(Level::INFO, "Deleting Workers");
            // crawl over the clusters we have scale down orders or pending deletes for
            for (cluster_name, deletes) in targets {
                // get a mutable reference to the target cluster
                if let Some(cluster) = self.schedulers.get_mut(&cluster_name) {
                    // delete the requested workers
                    let results = cluster.delete(&self.thorium, &self.cache, deletes).await;
                    // check the status of this clusters deletes
                    for result in results {
                        // check the status of this deletion
                        match result {
                            // this worker was successfully deleted
                            WorkerDeletion::Deleted(delete) => {
                                deleted.insert(delete.name);
                            }
                            // We ran into an error when deleting this worker
                            WorkerDeletion::Error { delete, error } => {
                                // log that we failed to delete a worker
                                event!(
                                    parent: &del_span,
                                    Level::ERROR,
                                    msg = "Failed to delete worker",
                                    cluster = cluster_name,
                                    req = delete.req.to_string(),
                                    name = delete.name,
                                    error = error.msg()
                                );
                            }
                        }
                    }
                }
            }
            // close our delete worker span
            drop(del_span);
        }
        // track the workers we fail to spawn
        let mut failed = WorkerDeleteMap::default();
        // start our spawn workers span if we have workers to spawn
        if !self.allocatable.changes.spawns.is_empty() {
            let spawn_span = span!(Level::INFO, "Spawning Workers");
            // Register our new workers before spawning them to avoid race conditions
            self.register_workers(&self.allocatable.changes).await?;
            // crawl over the clusters we have spawns for
            for (cluster_name, spawn_map) in self.allocatable.changes.spawns.drain() {
                // get a mutable reference to the target cluster
                if let Some(cluster) = self.schedulers.get_mut(&cluster_name) {
                    // spawn the new resources for this cluster
                    let errors = cluster.spawn(&self.cache, &spawn_map).await;
                    // check if any of our spawns failed
                    for (deadline, spawns) in spawn_map {
                        // crawl over the spawns in this spawn group
                        for spawn in spawns {
                            // if this resources name is in our error map then log the failed spawn
                            match errors.get(&spawn.name) {
                                // there was an error when spawning this resource
                                Some(error) => {
                                    // log that we failed to spawn a worker
                                    event!(
                                        parent: &spawn_span,
                                        Level::ERROR,
                                        msg = "Failed to span worker",
                                        cluster = cluster_name,
                                        req = spawn.req.to_string(),
                                        name = spawn.name,
                                        error = error.msg()
                                    );
                                    // track this failed worker
                                    schedulers::record_failed_spawn(
                                        &mut self.allocatable.fairshare_pool,
                                        &mut self.allocatable.image_counts,
                                        spawn,
                                        &mut deleted,
                                        &mut failed,
                                    );
                                }
                                None => {
                                    // get an entry to this resources current count
                                    let entry = self
                                        .allocatable
                                        .counts
                                        .entry(spawn.req.clone())
                                        .or_default();
                                    // increase this resources scale by 1
                                    *entry += 1;
                                    // log a that new worker was spawned
                                    event!(
                                        parent: &spawn_span,
                                        Level::INFO,
                                        msg = "Spawned worker",
                                        cluster = cluster_name,
                                        req = spawn.req.to_string(),
                                        name = spawn.name,
                                    );
                                    // get an entry to this spawns deadline group
                                    let dl_entry = changes.entry(deadline).or_default();
                                    // add this spawned worker to this deadline group
                                    dl_entry.push(spawn);
                                }
                            }
                        }
                    }
                }
            }
            // close our delete worker span
            drop(spawn_span);
        }
        // commit our finalized changes
        let req_changes = self.allocatable.commit(changes);
        // start our nested log scalers span
        let log_span = span!(Level::INFO, "Log Scale Changes");
        // log any successful changes to our resources
        for req in req_changes {
            // get the current and prior count for this resource
            let new = self.allocatable.counts.get(&req).unwrap_or(&0);
            let prior = past.get(&req).unwrap_or(&0);
            // calculate the change
            let change = new - prior;
            // log the change in scale
            event!(
                parent: &log_span,
                Level::INFO,
                msg = "Scaling workers",
                req = req.to_string(),
                new = new,
                change = change,
            );
        }
        // close our log scale changes span
        drop(log_span);
        // remove any workers we failed to spawn
        self.thorium
            .system
            .delete_workers(self.scaler_type, &failed)
            .await?;
        Ok(deleted)
    }

    /// Error out any workers jobs that have permenantly failed
    #[instrument(name = "Scaler::error_out", skip_all)]
    async fn error_out(&mut self, error_out: HashSet<ErrorOutKinds>, failed: &mut HashSet<String>) {
        // build a stream of our failed workers info
        let mut worker_stream = stream::iter(error_out)
            .map(|reason| async {
                // get this workers info
                let worker = self.thorium.system.get_worker(reason.worker()).await;
                // return our worker and kind tuple
                (reason, worker)
            })
            .buffer_unordered(10);
        // build an OOM ERROR LOG
        let add = StageLogsAdd::default();
        // fail out any workers we find
        while let Some((reason, worker_res)) = worker_stream.next().await {
            // log an error if we failed to get a worker
            match worker_res {
                Ok(worker) => {
                    // get this workers job if it has one
                    if let Some(active) = &worker.active {
                        // try to fail out this job
                        if let Err(error) = self.thorium.jobs.error(&active.job, &add).await {
                            // we failed to error out this job
                            event!(
                                Level::ERROR,
                                worker = worker.name,
                                error = error.msg(),
                                active = format!("{active:?}")
                            );
                        }
                        // log that we are erroring out this job
                        event!(
                            Level::INFO,
                            worker = worker.name,
                            job = active.job.to_string(),
                            reason = reason.reason_as_str(),
                        );
                    } else {
                        // log that we are erroring out this job
                        event!(
                            Level::INFO,
                            worker = worker.name,
                            active = "None",
                            reason = reason.reason_as_str(),
                        );
                    }
                    // add this worker to our failed set
                    failed.insert(worker.name);
                }
                Err(error) => {
                    // we faled to get this jobs worker
                    event!(Level::ERROR, error = error.msg());
                }
            }
        }
    }

    /// Clear any failed jobs across all clusters
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    #[instrument(name = "Scaler::clear_terminal", skip_all)]
    async fn clear_terminal(&mut self, mut terminal: HashSet<String>) {
        // build a list of all of our failed and terminal workers
        let mut failed = HashSet::default();
        let mut error_out = HashSet::default();
        // crawl over each cluster and get it setup for executing jobs
        for (name, scheduler) in &mut self.schedulers {
            // skip any clusters that are currently unreachable
            if self.unready.contains(name) {
                continue;
            }
            // if this fails just continue since failed resources may just be clutter
            if let Err(err) = scheduler
                .clear_terminal(
                    &self.thorium,
                    &self.allocatable,
                    &self.cache.groups,
                    &mut failed,
                    &mut terminal,
                    &mut error_out,
                )
                .await
            {
                event!(Level::ERROR, error = err.msg());
            }
        }
        // error out any errored out workers
        self.error_out(error_out, &mut failed).await;
        // free any terminal resources
        if let Err(error) = self
            .allocatable
            .free_deleted(&self.thorium, &failed, &terminal, &self.conf)
            .await
        {
            event!(Level::ERROR, error = error.msg());
        }
    }

    /// Initialize this scaler to begin scheduling jobs
    ///
    /// This does not need to be called before start and is used for step purposes.
    #[instrument(name = "Scaler::init", skip_all, err(Debug))]
    pub async fn init(&mut self) -> Result<(), Error> {
        // setup all clusters and get an initial count of their resources
        self.update_resources().await
    }

    /// Perform a single scale loop
    ///
    /// In most cases you want to just call `start`.
    #[instrument(name = "Scaler::single_scale_loop", skip_all, err(Debug))]
    pub async fn single_scale_loop(&mut self) -> Result<(), Error> {
        // check if any new tasks need to be spawned
        self.spawn_tasks().await?;
        // check if any tasks completed
        self.check_tasks().await?;
        // check to see if our ban rule sets have expired
        self.allocatable.bans.is_expired();
        // try to allocate resources to workers
        self.allocatable
            .allocate(&self.thorium, &self.cache)
            .await?;
        // schedule resources based on our requisition map and ban any failures
        let deleted = self.schedule().await?;
        // sleep for 1s to make sure worker registration has time to happen
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        // clear out any terminal resources
        self.clear_terminal(deleted).await;
        Ok(())
    }

    /// Start scaling pods up and down based on the deadline stream
    pub async fn start(&mut self) -> Result<(), Error> {
        // setup this scaler
        self.init().await?;
        // loop forever scaling pods up and down based on the deadline stream
        loop {
            // perform a single scale loop
            self.single_scale_loop().await?;
            // sleep for the configured dwell between scale attempts
            let dwell =
                std::time::Duration::from_secs(self.conf.thorium.scaler.dwell(self.scaler_type));
            tokio::time::sleep(dwell).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BanSets::is_expired` clears a ban set only once its `expire` timestamp has passed: an
    /// expired ban set (expire in the past) is cleared, while a still-valid one (expire in the
    /// future) is retained until it expires.
    #[test]
    fn is_expired_clears_only_after_expiry() {
        // an expired ban set: expire timestamp is in the past
        let mut expired = BanSets::new(ImageScaler::K8s);
        expired.expire = Utc::now() - chrono::Duration::seconds(60);
        // populate it with a banned group so we can observe whether it gets cleared
        expired.groups.insert("corn".to_string());
        // an expired ban set must be cleared
        expired.is_expired();
        assert!(
            expired.groups.is_empty(),
            "an expired ban set was not cleared"
        );

        // a still-valid ban set: expire timestamp is in the future
        let mut valid = BanSets::new(ImageScaler::K8s);
        valid.expire = Utc::now() + chrono::Duration::seconds(600);
        // populate it with a banned group
        valid.groups.insert("corn".to_string());
        // a still-valid ban set must be retained (not cleared early)
        valid.is_expired();
        assert!(
            valid.groups.contains("corn"),
            "a still-valid ban set was cleared before it expired"
        );
    }

    /// When a new cache is pulled the stale bans are dropped before every scheduler repopulates
    /// the ban sets from it, so a ban added while syncing the new cache is kept.
    #[tokio::test]
    async fn resync_keeps_bans_from_new_cache() {
        // a ban set holding a ban derived from the previous cache
        let mut bans = BanSets::new(ImageScaler::K8s);
        bans.groups.insert("stale".to_owned());
        // the sync against the new cache bans a group
        resync_bans(&mut bans, async |bans: &mut BanSets| {
            bans.groups.insert("new".to_owned());
            Ok(())
        })
        .await
        .unwrap();
        // the ban found in the new cache must be kept
        assert!(
            bans.groups.contains("new"),
            "bans derived from the new cache must not be cleared"
        );
    }

    /// Build a spawned deadline worker with a unique name
    ///
    /// # Arguments
    ///
    /// * `name` - The unique name of this worker
    fn spawned(name: String) -> Spawned {
        Spawned {
            req: thorium::models::Requisition::new("user", "group", "pipeline", "stage"),
            cluster: "cluster".to_owned(),
            node: "node".to_owned(),
            name,
            resources: thorium::models::Resources::default(),
            pool: thorium::models::Pools::Deadline,
            spawn: true,
            scaled_down: false,
            down_scalable: Utc::now(),
        }
    }

    /// Build a requisition map with groups of spawned workers in a single cluster
    ///
    /// # Arguments
    ///
    /// * `groups` - The number of workers in each deadline group
    fn spawn_map(groups: &[usize]) -> ReqMap {
        let mut reqs = ReqMap::default();
        let mut deadlines = BTreeMap::default();
        for (group, count) in groups.iter().enumerate() {
            // give each group its own deadline
            let deadline = Utc::now() + chrono::Duration::seconds(group as i64);
            let workers = (0..*count)
                .map(|i| spawned(format!("worker-{group}-{i}")))
                .collect();
            deadlines.insert(deadline, workers);
        }
        reqs.spawns.insert("cluster".to_owned(), deadlines);
        reqs
    }

    /// Each new worker is registered exactly once: 60 workers in three groups of 20 are split
    /// across batches without any worker being re-sent in a later batch, which keeps
    /// registration requests linear in the number of spawns.
    #[test]
    fn each_worker_registered_once() {
        // three deadline groups of 20 workers each
        let reqs = spawn_map(&[20, 20, 20]);
        // build the batches that would be submitted
        let batches = registration_batches(&reqs);
        // gather every registered worker name across all batches
        let names: Vec<&String> = batches
            .iter()
            .flat_map(|batch| batch.workers.iter().map(|worker| &worker.name))
            .collect();
        let unique: HashSet<&String> = names.iter().copied().collect();
        // every worker is registered and none of them more than once
        assert_eq!(unique.len(), 60);
        assert_eq!(
            names.len(),
            60,
            "each spawned worker must be registered exactly once"
        );
    }

    /// A small spawn wave is registered in a single batch.
    #[test]
    fn small_spawn_wave_is_registered_in_one_batch() {
        // a single group of 10 workers
        let reqs = spawn_map(&[10]);
        // build the batches that would be submitted
        let batches = registration_batches(&reqs);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].workers.len(), 10);
    }

    /// A cluster with pending deletes from an earlier loop but no new scale downs is still
    /// selected for `Scheduler::delete` (with no new scale downs), while clusters with new scale
    /// downs still get them. `Scheduler::delete` is the only place a scheduler validates its
    /// pending deletes, so this keeps those workers from staying registered in Thorium and in the
    /// allocator in the being deleted state until that cluster happens to get another scale down.
    #[test]
    fn pending_deletes_on_quiet_cluster_are_validated() {
        // cluster a has a new scale down this loop
        let mut scale_down = HashMap::default();
        scale_down.insert("a".to_owned(), vec![spawned("w1".to_owned())]);
        // cluster b has no new scale downs but its scheduler still has pending deletes
        let pending: HashSet<String> = ["b".to_owned()].into_iter().collect();
        // get the clusters we will call delete on
        let targets = delete_targets(scale_down, &pending);
        // the quiet cluster must still be called so it validates its pending deletes
        assert!(
            targets.contains_key("b"),
            "a cluster with pending deletes and no new scale downs must still be validated"
        );
        // it has no new scale downs to hand over
        assert!(targets["b"].is_empty());
        // and the cluster with new scale downs still gets them
        assert_eq!(targets["a"].len(), 1);
    }

    /// A cluster with neither new scale downs nor pending deletes is not called, and a cluster
    /// with only new scale downs keeps them.
    #[test]
    fn pending_deletes_on_quiet_cluster_idle_cluster_is_not_called() {
        // cluster a has a new scale down this loop
        let mut scale_down = HashMap::default();
        scale_down.insert("a".to_owned(), vec![spawned("w1".to_owned())]);
        // no cluster has pending deletes and cluster c is completely idle
        let targets = delete_targets(scale_down, &HashSet::default());
        // only the cluster with new scale downs is called
        assert_eq!(targets.len(), 1);
        assert!(!targets.contains_key("c"));
        assert_eq!(targets["a"][0].name, "w1");
    }
}
