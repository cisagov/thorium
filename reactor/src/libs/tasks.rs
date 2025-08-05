//! The tasks node reactor needs to periodically execute/handle

use chrono::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;
use sysinfo::{Disks, System};
use thorium::models::{NodeHealth, NodeUpdate, Resources};
use thorium::{Error, Thorium};
use tracing::{event, span, Level, Span};

/// gets a timestamp N seconds from now
#[doc(hidden)]
#[macro_export]
macro_rules! from_now {
    ($seconds:expr) => {
        Utc::now() + chrono::Duration::seconds($seconds)
    };
}

/// The tasks a node reactor needs to periodically execute/handle
#[derive(Debug, PartialEq)]
pub enum Tasks {
    /// Update the amount of resources on this node
    Resources,
}

impl Tasks {
    /// Set up a tasks queue with for all tasks
    pub fn setup_queue() -> BTreeMap<DateTime<Utc>, Tasks> {
        // create an empty map
        let mut queue = BTreeMap::default();
        // insert our tasks in a spread out way to minimize collisions
        queue.insert(from_now!(20), Self::Resources);
        queue
    }

    /// Get the amount of time to wait before executing this task from our config
    pub fn delay(&self) -> u32 {
        match self {
            Tasks::Resources => 20,
        }
    }

    /// Get our task as a str
    pub fn as_str(&self) -> &str {
        // add this task name to our trace
        match self {
            Tasks::Resources => "Resources",
        }
    }
}

/// Gets the currently available resources on this node
///
/// This will reserve 1.5 cores, 2 GiB of ram, and 8 GiB of storage for the host
///
/// # Arguments
///
/// * `span` - The span to log traces under
fn get_resources(system: &mut System, span: &Span) -> Result<Resources, Error> {
    // start our get resources span
    let span = span!(parent: span, Level::INFO, "Get Resources");
    // refresh our system info
    system.refresh_all();
    // get the total ram and cpu info
    let cpu = system.cpus().len() as u64;
    // convert our memory into kibibytes
    let memory = (system.total_memory() * 1000) / 1048576;
    let mut ephemeral_storage = 0;
    // get our disk info
    let disks = Disks::new_with_refreshed_list();
    // We display all disks' information:
    for disk in disks.list() {
        // get this disks mount point
        let mount = disk.mount_point();
        // only count disks that are mounted to /
        if mount == Path::new("/") || mount == Path::new("/tmp") {
            // set our  available space
            ephemeral_storage = (disk.available_space() * 1024) / 1048576;
            // stop looking for mounts if this is /tmp
            if mount == Path::new("/tmp") {
                break;
            }
        }
    }
    // build our resource
    let mut resources = Resources {
        cpu: cpu * 1000,
        memory,
        ephemeral_storage,
        worker_slots: 100,
        nvidia_gpu: 0,
        amd_gpu: 0,
    };
    // reserve some resources for the host
    let reserve = Resources {
        cpu: 1500,
        memory: 2048,
        ephemeral_storage: 8192,
        worker_slots: 0,
        nvidia_gpu: 0,
        amd_gpu: 0,
    };
    resources -= reserve;
    // log the resources that we have discovered
    event!(
        parent: &span,
        Level::INFO,
        cpu = resources.cpu,
        memory,
        resources.memory,
        storage = resources.ephemeral_storage,
        nvidia_gpu = resources.nvidia_gpu,
        amd_gpu = resources.amd_gpu
    );
    Ok(resources)
}

/// Get this nodes resources and update Thorium
pub async fn update_resources(
    cluster: &str,
    node: &str,
    thorium: &Thorium,
    system: &mut System,
    span: &Span,
) -> Result<(), Error> {
    // start our get and update resources span
    let span = span!(parent: span, Level::INFO, "Get/Update Resources");
    // get the resources this node has
    let resources = match get_resources(system, &span) {
        Ok(resources) => resources,
        Err(error) => {
            // log that we failed to get this nodes resources
            event!(parent: span, Level::ERROR, error = true, msg = error.msg());
            return Err(error);
        }
    };
    // build the update to apply to this node
    let update = NodeUpdate::new(NodeHealth::Healthy, resources).heart_beat();
    // update this nodes info in Thorium
    match thorium.system.update_node(cluster, node, &update).await {
        Ok(_) => Ok(()),
        Err(error) => {
            // log that we failed to update this  nodes resources
            event!(parent: span, Level::ERROR, error = true, msg = error.msg());
            Err(error)
        }
    }
}
