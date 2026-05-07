//! The tasks node reactor needs to periodically execute/handle

use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;
use sysinfo::{Disks, System};
use thorium::Thorium;
use thorium::models::{BurstableResources, NodeHealth, NodeUpdate, Resources};
use tracing::{Level, event, instrument};

use crate::Error;
use crate::args::Args;

/// The number of bytes in a MiB
const BYTES_IN_MIB: u64 = 1 << 20;

/// gets a timestamp N seconds from now
#[doc(hidden)]
#[macro_export]
macro_rules! from_now {
    ($seconds:expr) => {
        Utc::now() + Duration::from_secs($seconds)
    };
}

/// The tasks a node reactor needs to periodically execute/handle
#[derive(Debug, PartialEq, strum::AsRefStr)]
pub enum Tasks {
    /// Check for reactor/agent updates
    Update,
    /// Update the amount of resources on this node
    Resources,
}

impl Tasks {
    /// Setup a tasks queue with for all tasks
    pub fn setup_queue(args: &Args) -> BTreeMap<DateTime<Utc>, Tasks> {
        // create an empty map
        let mut queue = BTreeMap::default();
        // insert our tasks in a spread out way to minimize collisions
        if !args.skip_update {
            // only add the update task if we're not set to skip it
            queue.insert(from_now!(Tasks::Update.delay()), Self::Update);
        }
        queue.insert(from_now!(Tasks::Resources.delay()), Self::Resources);
        queue
    }

    /// Get the amount of time to wait before executing this task from our config
    pub fn delay(&self) -> u64 {
        match self {
            Tasks::Update => 31,
            Tasks::Resources => 18,
        }
    }

    /// Get our task as a str
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

/// Gets the currently available resources on this node
///
/// This will reserve 1.5 cores, 2 GiB of ram, and 8 GiB of storage for the host
fn get_resources(system: &mut System) -> Resources {
    // refresh our system info
    system.refresh_all();
    // get the total ram and cpu info
    let cpu = system.cpus().len() as u64;
    // convert our memory into MiB
    let memory = system.total_memory() / BYTES_IN_MIB;
    let mut ephemeral_storage = 0;
    // get our disk info
    let disks = Disks::new_with_refreshed_list();
    // We display all disks' information:
    for disk in disks.list() {
        // get this disks mount point
        let mount = disk.mount_point();
        // only count disks that are mounted to /
        if mount == Path::new("/") || mount == Path::new("/tmp") {
            // set our available space
            ephemeral_storage = disk.available_space() / BYTES_IN_MIB;
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
        burstable: BurstableResources::default(),
    };
    // reserve some resources for the host
    let reserve = Resources {
        cpu: 1500,
        memory: 2048,
        ephemeral_storage: 8192,
        worker_slots: 0,
        nvidia_gpu: 0,
        amd_gpu: 0,
        burstable: BurstableResources::default(),
    };
    resources -= reserve;
    resources
}

/// Get this nodes resources and update Thorium
#[instrument(name = "tasks::update_resources", skip_all, err(Debug))]
pub async fn update_resources(
    cluster: &str,
    node: &str,
    thorium: &Thorium,
    system: &mut System,
) -> Result<(), Error> {
    // get the resources this node has
    let resources = get_resources(system);
    // log the resources that we have discovered
    event!(
        Level::INFO,
        cpu = resources.cpu,
        memory = resources.memory,
        storage = resources.ephemeral_storage,
        nvidia_gpu = resources.nvidia_gpu,
        amd_gpu = resources.amd_gpu
    );
    // build the update to apply to this node
    let update = NodeUpdate::new(NodeHealth::Healthy, resources).heart_beat();
    // update this nodes info in Thorium
    thorium.system.update_node(cluster, node, &update).await?;
    Ok(())
}
