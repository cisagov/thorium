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
use tracing::{event, instrument, span, Level};

use super::schedulers::{self, Allocatable, ReqMap, Scheduler, WorkerDeletion};
use super::tasks::{self, TaskResult, Tasks, ZombieChecker};
use super::Cache;
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

/// update the resources for all clusters this scaler can see
macro_rules! update_resources {
    ($thorium:expr, $schedulers:expr, $resources:expr, $cache:expr) => {
        // crawl over each cluster and get its available resources
        for (name, scheduler) in $schedulers.iter_mut() {
            // get the available resources for this cluster
            let update = scheduler
                .resources_available(&$thorium, &$cache.settings)
                .await?;
            // update this clusters resources
            $resources.update(name, update);
        }
        // resize our deadline pool
        $resources.resize_deadline_pool();
    };
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
    OOM(String),
    /// This worker cannot be created due to a missing prereq
    #[allow(dead_code)]
    StuckCreating(String),
}

impl ErrorOutKinds {
    /// create a report that a worker has OOM'd
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker that has been oomed
    pub fn oom<W: Into<String>>(worker: W) -> Self {
        ErrorOutKinds::OOM(worker.into())
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

    /// Get the name of the worker regardless of reason
    pub fn worker(&self) -> &String {
        match self {
            Self::OOM(worker) => worker,
            Self::StuckCreating(worker) => worker,
        }
    }

    /// Get our reason as a string
    pub fn reason_as_str(&self) -> &'static str {
        match self {
            Self::OOM(_) => "OOM",
            Self::StuckCreating(_) => "StuckCreating",
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
    /// The requisitions to ban from spawning
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
        if self.expire > Utc::now() {
            self.clear();
        }
    }

    /// Clear all banned items
    pub fn clear(&mut self) {
        self.expire = from_now!(600);
        self.groups.clear();
        self.users.clear();
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
        // instance the scaler
        let scaler = Scaler {
            scaler_type,
            conf,
            thorium,
            cache,
            zombies,
            schedulers,
            allocatable,
            tasks,
            active: Vec::default(),
        };
        Ok(scaler)
    }

    /// Setup all clusters to begin executing jobs
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    #[instrument(name = "Scaler::setup", skip_all, err(Debug))]
    async fn setup(&mut self) -> Result<(), Error> {
        // crawl over each cluster and get it setup for executing jobs
        for (name, scheduler) in &mut self.schedulers {
            // set up the scheduler before we schedule jobs
            scheduler
                .setup(name, &self.cache, &mut self.allocatable.bans)
                .await?;
        }
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
                // get our completed task
                let completed = join_result??;
                // log that a task was completed
                event!(Level::INFO, task = completed.as_str());
                // check if a join error occurred
                match completed {
                    // All zombie jobs were completed
                    TaskResult::ZombieJobs => add_task!(self, Tasks::ZombieJobs),
                    // Thorium has synced against LDAP
                    TaskResult::LdapSync => add_task!(self, Tasks::LdapSync),
                    // An updated cache has been pulled
                    TaskResult::Cache(cache) => {
                        // crawl over each cluster and sync any cache updates
                        for (name, scheduler) in &mut self.schedulers {
                            scheduler
                                .sync_to_new_cache(name, &cache, &mut self.allocatable.bans)
                                .await?;
                        }
                        // clear all items in our ban sets
                        self.allocatable.bans.clear();
                        // save the cache in the scaler for next update
                        self.cache = Arc::new(cache);
                        // add a new task for refreshing our cache
                        add_task!(self, Tasks::CacheReload)
                    }
                    // Thorium has updated image runtimes
                    TaskResult::UpdateRuntimes => add_task!(self, Tasks::UpdateRuntimes),
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
        for (_, task) in self.tasks.extract_if(.., |time, _| time < &now) {
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
                    update_resources!(&self.thorium, self.schedulers, self.allocatable, self.cache);
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
                    // try to register our new workers
                    self.thorium
                        .system
                        .register_workers(self.scaler_type, &registrations)
                        .await?;
                }
            }
        }
        // skip registering new workers if we have none
        if !registrations.workers.is_empty() {
            // set all the resources we want to spawn as being created
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
        // track our succesful scale changes and the correctly deleted ones
        let mut changes: HashMap<DateTime<Utc>, Vec<Spawned>> =
            HashMap::with_capacity(self.allocatable.changes.spawns.len());
        let mut deleted = HashSet::with_capacity(self.allocatable.changes.scale_down.len());
        // start our delete workers span if we have workers to delete
        if !self.allocatable.changes.scale_down.is_empty() {
            // start our delete workers span
            let del_span = span!(Level::INFO, "Deleting Workers");
            // crawl over the clusters we have scale down orders for
            for (cluster_name, deletes) in self.allocatable.changes.scale_down.drain() {
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
                                    // add this failed worker to our delete map
                                    deleted.insert(spawn.name.clone());
                                    // add this failed worker to our list of workers to remove
                                    failed.add_mut(spawn.name);
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

    /// Error out any workers jobs that have permanently failed
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
                            event!(Level::ERROR, worker = worker.name, error = error.msg());
                        }
                        // log that we are erroring out this job
                        event!(
                            Level::INFO,
                            worker = worker.name,
                            job = active.job.to_string(),
                            reason = reason.reason_as_str(),
                        );
                    }
                    // add this worker to our failed set
                    failed.insert(worker.name);
                }
                Err(error) => {
                    // we failed to get this jobs worker
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
        for scheduler in self.schedulers.values_mut() {
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
            .free_deleted(&self.thorium, &failed, &terminal)
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
        // setup all clusters before we schedule jobs
        self.setup().await?;
        // get an initial count of resources in the cluster
        update_resources!(self.thorium, self.schedulers, self.allocatable, self.cache);
        Ok(())
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
