//! The watchers the operator uses to monitor its K8s clusters

use kube::Client;
use std::sync::Arc;
use tokio::task::JoinSet;

use crate::args::OperateCluster;
use crate::k8s::controller::SharedInfo;

mod mcp;
mod nodes;
mod thorium_cr;

/// How long in seconds to wait after encountering an error during reconcillation
const RECONCILE_ERROR_REQUEUE_SECS: u64 = 60u64;

/// Start the watchers for this cluster
///
/// # Arguments
///
/// * `name` - The name of the k8s cluster we are starting watchers for
/// * `client` - The client for the k8s cluster we are starting watchers for
/// * `args` - The command line args passed to the operator
/// * `shared` - Data shared across watchers
/// * `watchers` - A set to spawn our watcher tasks into
pub fn start(
    name: String,
    client: &Client,
    args: &OperateCluster,
    shared: &Arc<SharedInfo>,
    watchers: &mut JoinSet<()>,
) {
    // create a cr watcher
    watchers.spawn(thorium_cr::start(
        client.clone(),
        args.clone(),
        shared.clone(),
    ));
    // create a node watcher
    watchers.spawn(nodes::start(name, client.clone(), shared.clone()));
    // create an mcp pod watcher
    watchers.spawn(mcp::start(client.clone(), shared.clone()));
}
