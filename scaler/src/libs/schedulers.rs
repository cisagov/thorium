//! Abstracts the different schedulers for Thorium
//!
//! Currently we only support Kubernetes but that will likely change in the future.
use chrono::prelude::*;
use hashbrown::HashMap;
use std::collections::{BTreeMap, HashSet};
use thorium::models::{ImageScaler, SystemSettings};
use thorium::{Conf, Error, Thorium};

//pub mod baremetal;
mod allocatable;
pub mod direct;
pub mod dry_run;
pub mod k8s;
pub mod requisitions;

pub use allocatable::{Allocatable, AllocatableUpdate, NodeAllocatableUpdate, NodeResources};
pub use direct::Direct;
pub use k8s::K8s;
pub use requisitions::{ReqMap, Spawned};

use crate::libs::{BanSets, Cache, Tasks};

use self::dry_run::DryRun;

use super::scaler::ErrorOutKinds;

/// The outcome of deleting a specific worker
pub enum WorkerDeletion {
    /// This worker was successfully deleted
    Deleted(Spawned),
    /// This worker delete ran into an error
    Error { delete: Spawned, error: Error },
}

/// The methods required to be used as a Thorium scheduler
#[async_trait::async_trait]
pub trait Scheduler {
    /// Determine when a task should be executed again
    ///
    /// # Arguments
    ///
    /// * `task` - The task we want to run again
    #[allow(dead_code)]
    fn task_delay(&self, task: &Tasks) -> i64;

    /// Schedulers need to be able to determine how many resources they have
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `settings` - The current Thorium system settings
    /// * `span` - The span to log traces under
    async fn resources_available(
        &mut self,
        thorium: &Thorium,
        settings: &SystemSettings,
    ) -> Result<AllocatableUpdate, Error>;

    /// Schedulers need to be able to set up their environment before scheduling jobs
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the cluster we are setting up
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    async fn setup(&mut self, name: &str, cache: &Cache, bans: &mut BanSets) -> Result<(), Error>;

    /// Schedulers need to be able to sync their environment to the contents of
    /// a new cache
    ///
    /// * `name` - The name of the cluster we are setting up
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    /// * `span` - The span to log traces under
    async fn sync_to_new_cache(
        &mut self,
        name: &str,
        cache: &Cache,
        bans: &mut BanSets,
    ) -> Result<(), Error>;

    /// Schedulers need to be able to scale resources up based on requisitions
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium
    /// * `req` - The requisition to scale
    /// * `scale` - What to scale this requisition to
    async fn spawn(
        &mut self,
        cache: &Cache,
        spawns: &BTreeMap<DateTime<Utc>, Vec<Spawned>>,
    ) -> HashMap<String, Error>;

    /// Schedulers need to be able to scale resources down based on requisitions
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium
    /// * `scaledowns` - The workers to scale down
    async fn delete(
        &mut self,
        thorium: &Thorium,
        cache: &Cache,
        scaledowns: Vec<Spawned>,
    ) -> Vec<WorkerDeletion>;

    /// Clears out any failed or terminal resources in specified groups and return their names.
    ///
    /// If this scheduler uses a reactor it is likely to be a no op.
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
        thorium: &Thorium,
        allocatable: &Allocatable,
        groups: &HashSet<String>,
        failed: &mut HashSet<String>,
        terminal: &mut HashSet<String>,
        error_out: &mut HashSet<ErrorOutKinds>,
    ) -> Result<(), Error>;
}

/// Creates a new instance of the right schedulers
///
/// # Arguments
///
/// * `schedulers` - The map to add schedulers too
/// * `scaler` - The type of scaler to build
/// * `dry_run` - Whether this is a dry run or not
/// * `context_name` - The name of the context to use for k8s service accounts
/// * `conf` - The Thorium config
pub async fn new(
    schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>,
    scaler: &ImageScaler,
    dry_run: bool,
    context_name: &String,
    conf: &Conf,
) -> Result<(), Error> {
    // instance the correct scheduler
    match (dry_run, scaler) {
        // if dry run is true then use the dry run scheduler
        (true, _) => DryRun::new(schedulers, conf),
        // otherwise use the correct scheduler
        (false, ImageScaler::K8s) => K8s::new(context_name, schedulers, conf).await?,
        (false, ImageScaler::BareMetal) => Direct::build_bare_metal(schedulers, conf),
        (false, ImageScaler::Windows) => Direct::build_windows(schedulers, conf),
        (false, ImageScaler::Kvm) => Direct::build_kvm(schedulers, conf),
        (false, ImageScaler::External) => panic!("External scaler not supported"),
    };
    Ok(())
}
