use futures::StreamExt;
use k8s_openapi::api::core::v1::Node;
use kube::runtime::Controller;
use kube::runtime::controller::Action;
use kube::runtime::watcher::Config;
use kube::{Api, Client};
use std::sync::Arc;
use std::time::Duration;
use thorium::Error;

use crate::k8s::controller::SharedInfo;
use crate::k8s::crds::ThoriumCluster;

#[derive(Clone)]
struct NodeWatchContext {
    /// kube API client
    client: Client,
    /// The name of the k8s cluster this node is from
    k8s_name: String,
    // The shared thorium operator info
    shared: Arc<SharedInfo>,
}

/// Handle errors in the reconcile process
fn node_error_policy(_cluster: Arc<Node>, error: &Error, _state: Arc<NodeWatchContext>) -> Action {
    println!("Controller error:\n\t{}", error);
    println!(
        "Requeuing node provision reconciliation in {} seconds",
        super::RECONCILE_ERROR_REQUEUE_SECS
    );
    Action::requeue(Duration::from_secs(super::RECONCILE_ERROR_REQUEUE_SECS))
}

/// Reconcile changes to ThoriumCluster
///
/// Arguments
///
/// * `cluster` - Thorium cluster being changed
/// * `state` - Controller context including client instance and optional URL
async fn reconcile_nodes(node: Arc<Node>, ctx: Arc<NodeWatchContext>) -> Result<Action, Error> {
    // if we don't have any configs then just requeue this node in 30 seconds
    if ctx.shared.info.is_empty() {
        // don't scan this node for another 30 seconds
        return Ok(Action::requeue(Duration::from_secs(30)));
    }
    // get this nodes name or throw an error if it doesn't have one
    let name = node.metadata.name.as_ref().unwrap();
    // get this nodes labels so we can check if its already been enabled/disabled for Thorium
    if let Some(labels) = &node.metadata.labels {
        // get the info for this nodes k8s cluster
        let info = ctx.shared.get_for_node(&ctx.k8s_name, name)?;
        // get the current version of the api
        let api_version = info.thorium.updates.get_version().await?;
        // check if this node already has a Thorium enabled label
        if let Some(is_enabled) = labels.get("thorium") {
            // if this node is disabld then ignore it
            match is_enabled.as_str() {
                // this node is enabled check its version to see if we need to reprovision it
                "enabled" => {
                    // get the version deployed to this node
                    let node_version = labels.get("thorium_version");
                    // check the version deployed to this node
                    if let Some(node_version) = node_version {
                        // check if this node is already up to date
                        if api_version.compare_thorium(node_version)? {
                            // this node is already up to date so ignore it for 15 minutes
                            return Ok(Action::requeue(Duration::from_mins(15)));
                        }
                        println!(
                            "node {name} is running {node_version} but needs to be updated to {}",
                            api_version.thorium
                        );
                    }
                }
                // this node is disabled so ignore it for 15 minutes
                "disabled" => return Ok(Action::requeue(Duration::from_mins(15))),
                // this node has an unknown thorium enabled label
                unknown => {
                    return Err(Error::new(format!(
                        "Node {:?} has unknown 'thorium' label '{unknown}'. Expected values are 'enabled' or 'disabled'",
                        node.metadata.name
                    )));
                }
            }
        }
        // cleanup this provision pod if it exists
        crate::k8s::nodes::cleanup_provision_pod_specific(&info.meta, name).await?;
        // provision this node
        crate::k8s::nodes::deploy_provision_pod(&info.meta, name).await?;
        // label this node with its new version
        crate::k8s::nodes::label_node(&info.meta, name, &api_version.thorium.to_string()).await?;
        // don't scan this node for 15 minutes
        return Ok(Action::requeue(Duration::from_mins(15)));
    }
    println!("{:?} doesn't have any labels?", node.metadata.name);
    Ok(Action::requeue(Duration::from_secs(60)))
}

/// Watch our nodes for any changes
pub async fn start(k8s_name: String, client: Client, shared: Arc<SharedInfo>) {
    // build a node api
    let node_api: Api<Node> = Api::<Node>::all(client.clone());
    let clusters_api: Api<ThoriumCluster> = Api::<ThoriumCluster>::all(client.clone());
    // setup some state for our watcher
    let ctx = NodeWatchContext {
        client: client.clone(),
        k8s_name,
        shared: shared.clone(),
    };
    // create a controller to watch for changes in our nodes
    Controller::new(node_api, Config::default().any_semantic())
        .shutdown_on_signal()
        .owns(clusters_api, Config::default())
        .run(reconcile_nodes, node_error_policy, Arc::new(ctx))
        .filter_map(|x| async move { std::result::Result::ok(x) })
        .for_each(|_| futures::future::ready(()))
        .await;
}
