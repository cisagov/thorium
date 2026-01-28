//! Watches the api pods and applies any required labels

use futures::StreamExt;
use k8s_openapi::api::core::v1::{Pod, PodStatus};
use kube::api::{ListParams, Patch, PatchParams};
use kube::runtime::Controller;
use kube::runtime::controller::Action;
use kube::runtime::reflector::Lookup;
use kube::runtime::watcher::Config;
use kube::{Api, Client};
use std::sync::Arc;
use std::time::Duration;
use thorium::Error;

use crate::k8s::controller::SharedInfo;
use crate::k8s::crds::ThoriumCluster;

/// The context for our api pod watcher
#[derive(Clone)]
struct ApiWatchContext {
    /// kube API client
    client: Client,
    /// The currently designated mcp api pod
    mcp_pod: Option<String>,
    // The shared thorium operator info
    shared: Arc<SharedInfo>,
}

/// Handle errors in the reconcile process
fn api_pod_error_policy(_cluster: Arc<Pod>, error: &Error, _state: Arc<ApiWatchContext>) -> Action {
    println!("Controller error:\n\t{error}");
    println!(
        "Requeuing api pod reconciliation in {} seconds",
        super::RECONCILE_ERROR_REQUEUE_SECS
    );
    Action::requeue(Duration::from_secs(super::RECONCILE_ERROR_REQUEUE_SECS))
}

fn pod_is_alive(pod: &Pod) -> bool {
    // check if this pod is terminating or failed
    // get this pods phase
    match pod
        .status
        .as_ref()
        .and_then(|status| status.phase.as_deref())
    {
        // no action is needed as this pod is still running or starting up
        Some("Pending" | "Running") => true,
        // the current designated mcp api pod is dead or missing so assign a new one
        Some("Succeeded" | "Failed" | "Unknown" | _) | None => false,
    }
}

/// Reconcile changes to ThoriumCluster
///
/// Arguments
///
/// * `cluster` - Thorium cluster being changed
/// * `state` - Controller context including client instance and optional URL
async fn reconcile_api_pods(pod: Arc<Pod>, mut ctx: Arc<ApiWatchContext>) -> Result<Action, Error> {
    // if we don't have any configs then just requeue this node in 30 seconds
    if ctx.shared.info.is_empty() {
        // don't scan this pod for another 30 seconds
        return Ok(Action::requeue(Duration::from_secs(30)));
    }
    // skip any pods with out names
    let Some(name) = &pod.metadata.name else {
        // ignore this pod for one minute
        return Ok(Action::requeue(Duration::from_secs(60)));
    };
    // get this pods current namespace
    let namespace = match pod.namespace() {
        Some(ns) => ns.to_string(),
        // all pods we scan must be in a namespace
        None => {
            println!("Pod {name} is not in a namespace?");
            // ignore this pod for one minute
            return Ok(Action::requeue(Duration::from_secs(60)));
        }
    };
    // get our designated mcp pod info if this our designated mcp pod already
    let designated_pod = if ctx.mcp_pod.as_ref() == Some(name) {
        // we already have our designated mcp pods info so just use that
        Some(pod)
    } else {
        // this is not our currently designated mcp pod so check if that pod is alive or dead
        // build a Thorium api pod client
        let pod_api: Api<Pod> = Api::<Pod>::namespaced(ctx.client.clone(), "thorium");
        // get our designated mcp pods info
        match pod_api.get_opt(name).await {
            // return our designated mcp pods info
            Ok(Some(designated_pod)) => Some(Arc::new(designated_pod)),
            // our designated mcp pod is missing
            Ok(None) => None,
            Err(error) => {
                // we failed to get our pods info so build an error
                return Err(Error::new(format!(
                    "Error getting info on designated MCP pod {name}: {error:?}",
                )));
            }
        }
    };
    // if we found a designated pod then check if its still alive
    // otherwise assume its dead and rescan
    if let Some(designated_pod) = designated_pod {
        // check if our designated pod is still alive
        if pod_is_alive(&designated_pod) {
            // our designated mcp pod is still alive so don't scan it for 15 minutes
            return Ok(Action::requeue(Duration::from_mins(15)));
        }
    }
    // our designated mcp pod is dead so label a new one
    // build a Thorium api pod client
    let pod_api: Api<Pod> = Api::<Pod>::namespaced(ctx.client.clone(), &namespace);
    // scan for a pod to label
    if let Some(new_mcp_pod) = scan(&pod_api).await {
        // update our designated mcp pod record
        Arc::make_mut(&mut ctx).mcp_pod = Some(new_mcp_pod);
    }
    // no more action is needed for 60 seconds
    Ok(Action::requeue(Duration::from_secs(60)))
}

/// Label this api pod to be able to serve mcp queries
///
///  Arguments
///
/// * `pod_api` - K8s pod api client
/// * `pod` - The name of the pod to add the mcp label to
pub async fn add_label(pod_api: &Api<Pod>, pod: &str) {
    println!("labeling {pod} with mcp=enabled");
    // build a label json template
    let label = serde_json::json!({
        "metadata": {
            "labels": {
                "mcp": "enabled",
            }
        }
    });
    // patch the pod with the new label
    match pod_api
        .patch(pod, &PatchParams::default(), &Patch::Merge(&label))
        .await
    {
        Ok(_) => println!("pod {pod} labeled successfully with mcp=enabled"),
        Err(error) => println!("Failed to label pod {pod}: {error}"),
    }
}

/// Remove Thorium worker pod labels
///
///  Arguments
///
/// * `pod_api` - K8s pod api client
/// * `pod` - The name of the pod to remove the mcp label from
pub async fn remove_label(pod_api: &Api<Pod>, pod: &str) {
    let params = PatchParams::default();
    // build a label json template
    let label = serde_json::json!({
        "metadata": {
            "labels": {
                "mcp": null,
            }
        }
    });
    // patch the pod with the new label
    match pod_api.patch(pod, &params, &Patch::Merge(&label)).await {
        Ok(_) => {
            println!("Patched pod to remove label mcp=enabled from {pod}");
        }
        Err(kube::Error::Api(error)) => {
            // pod does not exist to remove label, continue on
            if error.code == 404 {
                println!("pod {pod} not found to remove label, skipping pod update");
            } else {
                println!("Failed to remove label {label} from pod {pod}: {error}");
            }
        }
        Err(error) => {
            println!(
                "Failed to remove label {} from pod {}: {}",
                &label, pod, error
            );
        }
    }
}

/// Scan our api pods for any existing mcp designated pods
async fn scan(pod_api: &Api<Pod>) -> Option<String> {
    // list all api pods
    let pods = pod_api
        .list(&ListParams::default().labels("app=api"))
        .await
        .expect("Failed to list api pods");
    // track the currently labelled mcp pods
    let mut is_mcp = Vec::with_capacity(1);
    // track the pods that were not labeled as mcp pods
    let mut not_mcp = Vec::with_capacity(10);
    // look for any existing mcp pods
    for pod in pods {
        // get this pods name and status
        let Some(name) = pod.metadata.name else {
            // log that we are skipping an api pod without a name
            println!("Skipping API pod without a name!");
            // skip to the next container
            continue;
        };
        // get this pods phase
        match pod
            .status
            .as_ref()
            .and_then(|status| status.phase.as_deref())
        {
            Some("Pending" | "Running") => (),
            Some(phase) => {
                // log that we are skipping an api pod with the wrong phase
                println!("Skipping API pod with phase: {phase}");
                // skip to the next container
                continue;
            }
            None => {
                // log that we are skipping an api pod without a phase
                println!("Skipping API pod without a phase!");
                // skip to the next container
                continue;
            }
        }
        // get this pods current labels
        if let Some(labels) = &pod.metadata.labels {
            // check if this pods is labelled as an mcp pod
            if let Some("enabled") = labels.get("mcp").map(String::as_str) {
                // add this to the list of mcp pods that we found
                is_mcp.push(name);
            } else {
                not_mcp.push(name);
            }
        }
    }
    // handle if we have one or moer mcp pods
    match is_mcp.len() {
        // we don't have any mcp pods so just pick one
        0 => {
            // just get the last found not mcp pod
            match not_mcp.pop() {
                Some(name) => {
                    // label this pod as being mcp enabled
                    add_label(pod_api, &name).await;
                    // return the newly designated mcp pod
                    Some(name)
                }
                None => {
                    // log that no pod could be marked as our mcp pod
                    println!("No viable API found to label for MCP support");
                    // we couldn't label a pod as being mcp enabled
                    None
                }
            }
        }
        // we found a single mcp pod
        1 => is_mcp.pop(),
        // we found multiple mcp pods
        _ => {
            // pop the one pod to keep
            let keep = is_mcp.pop();
            // remove the mcp enabled label for the rest of the pods
            for pod in is_mcp {
                remove_label(pod_api, &pod).await;
            }
            // return the name of the still designated mcp pod
            keep
        }
    }
}

/// Watch our nodes for any changes
pub async fn start(client: Client, shared: Arc<SharedInfo>) {
    // build a Thorium api pod client
    let pod_api: Api<Pod> = Api::<Pod>::all(client.clone());
    // Scan and make sure we have an existing mcp pod if possible
    let mcp_pod = scan(&pod_api).await;
    // setup some state for our watcher
    let ctx = ApiWatchContext {
        client: client.clone(),
        mcp_pod,
        shared: shared.clone(),
    };
    // set our config to only list api pods
    let config = Config::default().labels("app=api").any_semantic();
    // create a controller to watch for changes in our nodes
    Controller::new(pod_api, config)
        .shutdown_on_signal()
        .run(reconcile_api_pods, api_pod_error_policy, Arc::new(ctx))
        .filter_map(|x| async move { std::result::Result::ok(x) })
        .for_each(|_| futures::future::ready(()))
        .await;
}
