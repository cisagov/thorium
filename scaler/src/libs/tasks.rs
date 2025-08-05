//! Monitors Thorium for zombie jobs and performs other actions cron like actions in the background
//!
//! The monitor tries to execute background actions when they are scheduled but because everything
//! happens on a single thread. It is likely that actions my not always complete on schedule.

use chrono::prelude::*;
use std::collections::{BTreeMap, HashSet};
use hashbrown::HashMap;
use std::sync::Arc;
use thorium::models::{
    ImageScaler, JobResets, NodeListParams, RunningJob, SystemComponents,
    WorkerDeleteMap,
};
use thorium::{Conf, Error, Thorium};
use tracing::{event, instrument, span, Level};
use uuid::Uuid;

use super::cache::Cache;
use super::schedulers::Allocatable;
use crate::from_now;

/// Actions to complete at specific times
#[derive(Debug, PartialEq)]
pub enum Tasks {
    /// Find and remove any zombie jobs
    ZombieJobs,
    /// Sync our group membership info against LDAP
    LdapSync,
    /// Reset our local cache of Thorium info
    CacheReload,
    /// Check the health of nodes and determine our total amount of resources
    Resources,
    /// Calculate the average runtimes of all images and update it
    UpdateRuntimes,
    /// Cleanup any expired info in Thorium
    Cleanup,
    /// Decrease any users fair share ranks
    DecreaseFairShare,
}

impl Tasks {
    /// Set up a tasks queue with for all tasks
    pub fn setup_queue(conf: &Conf) -> BTreeMap<DateTime<Utc>, Tasks> {
        // create an empty map
        let mut queue = BTreeMap::default();
        // insert our tasks in a spread out way to minimize collisions
        queue.insert(from_now!(23), Self::Cleanup);
        queue.insert(from_now!(30), Self::ZombieJobs);
        // only add the ldap sync if ldap is enabled
        if conf.thorium.auth.ldap.is_some() {
            queue.insert(from_now!(47), Self::LdapSync);
        }
        queue.insert(from_now!(55), Self::CacheReload);
        queue.insert(from_now!(57), Self::UpdateRuntimes);
        queue.insert(from_now!(63), Self::Resources);
        queue.insert(from_now!(600), Self::DecreaseFairShare);
        queue
    }

    /// Get the amount of time to wait before executing this task from our config
    pub fn delay(&self, conf: &Conf) -> u32 {
        match self {
            Tasks::ZombieJobs => conf.thorium.scaler.tasks.zombies,
            Tasks::LdapSync => conf.thorium.scaler.tasks.ldap_sync,
            Tasks::CacheReload => conf.thorium.scaler.tasks.cache_reload,
            Tasks::Resources => conf.thorium.scaler.tasks.resources,
            Tasks::UpdateRuntimes => conf.thorium.scaler.tasks.image_runtimes,
            Tasks::Cleanup => conf.thorium.scaler.tasks.cleanup,
            Tasks::DecreaseFairShare => conf.thorium.scaler.tasks.decrease_fair_share,
        }
    }

    /// Get our task as a str
    pub fn as_str(&self) -> &str {
        // add this task name to our trace
        match self {
            Tasks::ZombieJobs => "ZombieJobs",
            Tasks::LdapSync => "LdapSync",
            Tasks::CacheReload => "CacheReload",
            Tasks::Resources => "Resources",
            Tasks::UpdateRuntimes => "UpdateRuntimes",
            Tasks::Cleanup => "Cleanup",
            Tasks::DecreaseFairShare => "DecreaseFairShare",
        }
    }
}

/// The possible results from a task
pub enum TaskResult {
    /// Find and remove any zombie jobs
    ZombieJobs,
    /// Sync our group membership info against LDAP
    LdapSync,
    /// Reset our local cache of Thorium info
    Cache(Cache),
    /// Calculate the average runtimes of all images and update it
    UpdateRuntimes,
}

impl TaskResult {
    /// Get our task as a str
    pub fn as_str(&self) -> &str {
        // add this task name to our trace
        match self {
            TaskResult::ZombieJobs => "ZombieJobs",
            TaskResult::LdapSync => "LdapSync",
            TaskResult::Cache(_) => "CacheReload",
            TaskResult::UpdateRuntimes => "UpdateRuntimes",
        }
    }
}

/// Tracks and reset any zombie jobs whose workers have died
///
/// A job is not determined to be a zombie unless it has been detected in 2
/// consecutive zombie checks.
pub struct ZombieChecker {
    /// The scaler whose jobs were monitoring
    scaler: ImageScaler,
    /// A Thorium client
    thorium: Arc<Thorium>,
    /// The jobs that may be zombies
    maybe_jobs: HashMap<Uuid, bool>,
    /// The jobs that have been confirmed to be a zombie
    confirmed_jobs: Vec<Uuid>,
    /// The workers that may be zombies
    maybe_workers: HashMap<String, bool>,
    /// Whether we should suppress maybe zombie events
    suppress_maybes: bool,
    /// Whether we should suppress confirmed zombie events
    suppress_confirmed: bool,
}

impl ZombieChecker {
    /// Create a new zombie checker
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler we are monitoring
    /// * `thorium` - A client for the Thorium api
    pub fn new(scaler: ImageScaler, thorium: &Arc<Thorium>) -> Self {
        // assume we will have at most 50 zombie jobs
        ZombieChecker {
            scaler,
            thorium: thorium.clone(),
            maybe_jobs: HashMap::with_capacity(50),
            confirmed_jobs: Vec::with_capacity(50),
            maybe_workers: HashMap::with_capacity(50),
            suppress_maybes: false,
            suppress_confirmed: false,
        }
    }

    /// Get the currently running jobs
    async fn get_jobs(&self) -> Result<Vec<RunningJob>, Error> {
        // build arbitrary dates for reading the running jobs queue
        let start = DateTime::from_timestamp(0, 0).unwrap();
        let end = Utc::now() + chrono::Duration::weeks(1000);
        // get the running jobs for this page
        self.thorium
            .jobs
            .running(self.scaler, &start, &end, 100_000)
            .await
    }

    /// Scan the currently running jobs to find any zombie jobs
    ///
    /// # Arguments
    ///
    /// * `spawned` - The currently spawned workers in Thorium
    #[instrument(name = "ZombieChecker::scan_jobs", skip_all, err(Debug))]
    async fn scan_jobs(&mut self, spawned: &HashSet<&String>) -> Result<(), Error> {
        // get the currently running jobs
        let running = self.get_jobs().await?;
        // set all of our zombie job values to false to denote existing entries
        self.maybe_jobs.values_mut().for_each(|new| *new = false);
        // track the jobs we want to reset
        for job in running {
            // check if this worker is still active
            if !spawned.contains(&job.worker) {
                // add this worker to our maybe zombie set
                if self.maybe_jobs.insert(job.job_id.clone(), true).is_some() {
                    // this worker was already in our maybe set
                    // remove it from our maybe set and add it to our confirmed one
                    self.maybe_jobs.remove(&job.job_id);
                    self.confirmed_jobs.push(job.job_id);
                }
            } else {
                // this job has a spawned worker so make sure its not in our maybe set
                self.maybe_jobs.remove(&job.job_id);
            }
        }
        // remove any old zombie jobs
        self.maybe_jobs.retain(|_, new| *new);
        Ok(())
    }

    /// Check for any zombie jobs and reset any confirmed ones
    ///
    /// # Arguments
    ///
    /// * `spawned` - The currently spawned workers in Thorium
    #[instrument(name = "ZombieChecker::check_jobs", skip_all, err(Debug))]
    async fn check_jobs(&mut self, spawned: &HashSet<&String>) -> Result<(), Error> {
        // scan the currently active jobs for zombies
        self.scan_jobs(spawned).await?;
        // get the number of zombie jobs we are trying to reset
        let zombie_jobs = self.confirmed_jobs.len();
        // log the number of zombies jobs that we found
        event!(Level::INFO, zombie_jobs);
        // get the capacity to set for our reset request
        let capacity = std::cmp::min(zombie_jobs, 50);
        // build this list of jobs to reset
        let mut req = JobResets::with_capacity(self.scaler, "Worker not found", capacity)
            // set our component to be the scaler
            .as_component(SystemComponents::Scaler(self.scaler));
        // track the number of zombies that have been reset so far
        let mut progress = 0;
        // reset these jobs 50 at a time
        for chunk in self.confirmed_jobs[..].chunks(50) {
            // add these jobs to our reset request
            req.jobs.extend_from_slice(chunk);
            // reset jobs
            self.thorium.jobs.bulk_reset(&req).await?;
            // increment our progress counter
            progress += req.jobs.len();
            // remove all the jobs we reset
            req.jobs.clear();
            // log the number of zombies that we have reset
            event!(Level::INFO, progress);
        }
        // remove all confirmed zombies
        self.confirmed_jobs.truncate(50);
        self.confirmed_jobs.clear();
        Ok(())
    }

    /// Scan for any workers that are zombies
    #[instrument(name = "ZombieChecker::scan_workers", skip_all, err(Debug))]
    async fn scan_workers(&mut self, spawned: &HashSet<&String>) -> Result<WorkerDeleteMap, Error> {
        // build our params info
        // scan our nodes 500 at a time
        let params = NodeListParams::default().scaler(self.scaler).limit(500);
        // get info on the current Thorium nodes
        let mut cursor = self.thorium.system.list_node_details(&params).await?;
        // build our map of workers to delete
        let mut confirmed = WorkerDeleteMap::default();
        // set all of our zombie worker values to false to denote existing entries
        self.maybe_workers.values_mut().for_each(|new| *new = false);
        // keep crawling nodes until we have scanned them all
        while !cursor.data.is_empty() {
            // check each nodes workers
            for node in cursor.data.drain(..) {
                // check the workers on this node
                for (name, worker) in node.workers {
                    // check if this worker is a zombie worker
                    if !spawned.contains(&name) {
                        // add this zombie node to our maybe set
                        if self
                            .maybe_workers
                            .insert(worker.name.clone(), true)
                            .is_some()
                        {
                            // log that we have found a zombie worker
                            event!(
                                Level::WARN,
                                zombie = "Confirmed",
                                node = worker.node,
                                worker = name
                            );
                            // this worker was already in our maybe set
                            // remove it from our maybe set and add it to our confirmed one
                            self.maybe_workers.remove(&name);
                            // add this confirmed zombie worker to our delete map
                            confirmed.add_mut(name);
                        } else {
                            // log that we might have found a zombie worker
                            event!(
                                Level::WARN,
                                zombie = "Maybe",
                                node = worker.node,
                                worker = worker.name
                            );
                        }
                    } else {
                        // make sure this valid worker is not in our maybe set
                        self.maybe_workers.remove(&worker.name);
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
        // remove any old zombie workers
        self.maybe_workers.retain(|_, new| *new);
        Ok(confirmed)
    }

    /// Check for any zombie workers and delete any confirmed ones
    ///
    /// # Arguments
    ///
    /// * `allocatable` - The allocatable resources for this scheduler
    #[instrument(name = "ZombieChecker::check_workers", skip_all, err(Debug))]
    async fn check_workers(&mut self, spawned: &HashSet<&String>) -> Result<(), Error> {
        // scan for zombie workers
        let confirmed = self.scan_workers(spawned).await?;
        // if we have workers to delete then delete them
        if !confirmed.workers.is_empty() {
            // log the number of zombies workers that we found
            event!(Level::INFO, zombie_workers = confirmed.workers.len());
            // delete these workers in Thorium
            self.thorium
                .system
                .delete_workers(self.scaler, &confirmed)
                .await?;
        }
        Ok(())
    }

    /// Check for any zombie jobs/workers and reset/delete confirmed ones
    ///
    /// # Arguments
    ///
    /// * `allocatable` - The allocatable resources for this scheduler
    #[instrument(name = "ZombieChecker::check", skip_all, err(Debug))]
    pub async fn check(&mut self, allocatable: &Allocatable) -> Result<TaskResult, Error> {
        // get the names of all currently spawned resources
        let spawned = allocatable.spawn_names();
        // check for zombie jobs
        self.check_jobs(&spawned).await?;
        self.check_workers(&spawned).await?;
        Ok(TaskResult::ZombieJobs)
    }
}

impl std::fmt::Debug for ZombieChecker {
    /// Allow zombie checker to be printed in a debug format
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZombieChecker")
            .field("scaler", &self.scaler)
            .field("maybe_jobs", &self.maybe_jobs)
            .field("confirmed_jobs", &self.confirmed_jobs)
            .field("maybe_workers", &self.maybe_jobs)
            .finish_non_exhaustive()
    }
}

/// Try to sync ldap groups
///
/// # Arguments
///
/// * `thorium` - A reference counted Thorium client
pub async fn sync_ldap(thorium: Arc<Thorium>) -> Result<TaskResult, Error> {
    // start our ldap sync span
    let _ = span!(Level::INFO, "LDAP Sync");
    // try to sync our groups in ldap
    thorium.groups.sync_ldap().await?;
    Ok(TaskResult::LdapSync)
}

/// Try to update the image runtimes
///
/// # Arguments
///
/// * `thorium` - A reference counted Thorium client
pub async fn update_runtimes(thorium: Arc<Thorium>) -> Result<TaskResult, Error> {
    // start our update runtimes span
    let _ = span!(Level::INFO, "Updating Runtimes");
    // try to sync our groups in ldap
    thorium.images.update_runtimes().await?;
    Ok(TaskResult::UpdateRuntimes)
}

/// Clean up any expired data in Thorium
///
/// # Arguments
///
/// * `thorium` - A reference counted Thorium client
pub async fn cleanup(thorium: Arc<Thorium>) -> Result<TaskResult, Error> {
    // start our cleanup span
    let _ = span!(Level::INFO, "Cleanup");
    // try to clean up any expired data in Thorium
    thorium.system.cleanup().await?;
    Ok(TaskResult::UpdateRuntimes)
}
