use k8s_openapi::api::core::v1::Pod;
use kube::api::{DeleteParams, Patch, PatchParams, PostParams};
use thorium::Error;

use super::clusters::ClusterMeta;

const TRACING_MOUNT_PATH: &str = "/tmp/tracing.yml";

/// Label Thorium worker nodes in a kubernetes cluster
///
/// Before the Thorium scaler can schedule reactions to run on k8s nodes, those nodes
/// must be labeled with thorium=enabled via the nodes k8s API. This method will label
/// each node listed under the scaler/k8s section of the CRD config.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn label_all_nodes(meta: &ClusterMeta) -> Result<(), Error> {
    let params = PatchParams::default();
    // label each node
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    for (_name, k8s_cluster) in clusters.iter() {
        for node in k8s_cluster.nodes.iter() {
            // build a label json template
            let label = serde_json::json!({
                "metadata": {
                    "labels": {
                        "thorium": "enabled"
                    }
                }
            });
            // patch the node with the new label
            match meta
                .node_api
                .patch(node, &params, &Patch::Merge(&label))
                .await
            {
                Ok(_) => {
                    println!("Node {} labeled successfully with thorium=enabled", node);
                }
                Err(error) => {
                    return Err(Error::new(format!(
                        "Failed to label node {}: {}",
                        node, error
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Remove Thorium worker node labels
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn delete_node_labels(meta: &ClusterMeta) -> Result<(), Error> {
    let params = PatchParams::default();
    // label each node
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    for (_name, k8s_cluster) in clusters.iter() {
        for node in k8s_cluster.nodes.iter() {
            // build a label json template
            let label = serde_json::json!({
                "metadata": {
                    "labels": {
                        "thorium": null
                    }
                }
            });
            // patch the node with the new label
            match meta
                .node_api
                .patch(node, &params, &Patch::Merge(&label))
                .await
            {
                Ok(_) => {
                    println!("Patched node to remove label thorium=enabled from {}", node);
                }
                Err(kube::Error::Api(error)) => {
                    // Node does not exist to remove label, continue on
                    if error.code == 404 {
                        println!(
                            "Node {} not found to remove label, skipping node update",
                            node
                        );
                        return Ok(());
                    }
                    return Err(Error::new(format!(
                        "Failed to remove label {} from node {}: {}",
                        &label, node, error
                    )));
                }
                Err(error) => {
                    return Err(Error::new(format!(
                        "Failed to remove label {} from node {}: {}",
                        &label, node, error
                    )))
                }
            }
        }
    }
    Ok(())
}

/// Cleanup node provision pods
///
/// This deletes node provision pods as part of a ThoriumCluster cleanup. Each node that that
/// is used by Thorium to schedule reactions will have a pod run on it to configure host
/// paths such as /opt/thorium. This method cleans up any pods that have run to provision host
/// paths.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn cleanup_provision_pods(meta: &ClusterMeta) -> Result<(), Error> {
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    for (_name, k8s_cluster) in clusters.iter() {
        for node in k8s_cluster.nodes.iter() {
            // build provisioner pod name
            let name = format!("node-provisioner-{}", node);
            let params = DeleteParams::default();
            // attempt deletion of the pod
            match meta.pod_api.delete(&name, &params).await {
                Ok(_) => println!("Cleaning up {} pod", &name),
                Err(kube::Error::Api(error)) => {
                    // don't panic if pods don't exist, thats the desired state
                    if error.code == 404 {
                        println!("Ignoring {} cleanup, pod does not exist", &name);
                        continue;
                    }
                    return Err(Error::new(format!(
                        "Failed to create {} pod: {}",
                        &name, error
                    )));
                }
                Err(error) => {
                    return Err(Error::new(format!(
                        "Failed to create {} pod: {}",
                        &name, error
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Deploy a single node provision pod
///
/// Node provision pods configure the /opt/thorium directory so that the Thorium agent
/// can run jobs on the system. The directory includes a tracing.yml config to enable
/// agent logging and the agent binary itself. The provision command itself is part of
/// the thorium admin binary thoradm.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `params` - K8s API post params
/// * `node` - Name of node to deploy pod
pub async fn deploy_provision_pod(
    meta: &ClusterMeta,
    params: &PostParams,
    node: &str,
) -> Result<(), Error> {
    let pod_name = format!("node-provisioner-{}", node);
    // set default resources for node provision pods
    let resources = serde_json::json!({"cpu": "250m", "memory": "250Mi"});
    let pod_template = serde_json::json!({
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "namespace": meta.namespace,
            "name": pod_name.clone(),
        },
        "spec": {
            "containers": [
                {
                    "name": "node-provisioner",
                    "image": meta.cluster.get_image(),
                    "imagePullPolicy": meta.cluster.spec.image_pull_policy.clone(),
                    "command": ["/app/thoradm"],
                    "args": ["provision", "node", "--keys", "/keys/keys.yml", "--k8s"],
                    "resources": {
                        "limits": resources.clone(),
                        "requests": resources.clone()
                    },
                    "volumeMounts": [
                        {
                            "name": "opt-mount",
                            "mountPath": "/opt"
                        },
                        {
                            "name": "keys",
                            "mountPath": "/keys/keys.yml",
                            "subPath": "keys.yml"
                        },
                        {
                            "name": "tracing",
                            "mountPath": TRACING_MOUNT_PATH.to_string(),
                            "subPath": "tracing.yml"
                        }
                    ]
                }
            ],
            "nodeName": node,
            "restartPolicy": "OnFailure",
            "volumes": [
                {
                    "name": "opt-mount",
                    "hostPath": {
                        "path": "/opt",
                        "type": "Directory"
                    }
                },
                {
                    "name": "keys",
                    "secret": {
                        "secretName": "keys"
                    }
                },
                {
                    "name": "tracing",
                    "configMap": {
                        "name": "tracing-conf"
                    }
                }
            ],
            "imagePullSecrets": [
                {
                    "name": "registry-token"
                }
            ]
        }
    });
    let pod: Pod = serde_json::from_value(pod_template)?;
    match meta.pod_api.create(&params, &pod).await {
        Ok(_) => println!("Node provision pod created: {}", pod_name),
        Err(error) => {
            // don't fail whole operator if node pods fail to create
            return Err(Error::new(format!(
                "Failed to create node provision pod: {}",
                error
            )));
        }
    }
    Ok(())
}

/// Deploy node provision pods to each thorium enabled k8s server
///
/// This function will spawn a node provision pod on each server designated to be used as
/// a compute node within the Thorium. This provision pod will configure the /opt/thorium
/// directory so that the thorium-agent can run jobs on that system.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn deploy_provision_pods(meta: &ClusterMeta) -> Result<(), Error> {
    // cleanup existing provision pods
    cleanup_provision_pods(meta).await?;
    // build k8s api client and params
    let params = PostParams::default();
    // apply a provision pod to each k8s node for each k8s cluster
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    for (_name, k8s_cluster) in clusters.iter() {
        for node in k8s_cluster.nodes.iter() {
            deploy_provision_pod(meta, &params, node).await?;
        }
    }
    Ok(())
}
