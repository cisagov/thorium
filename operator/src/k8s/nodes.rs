use k8s_openapi::api::core::v1::Pod;
use kube::api::{DeleteParams, ListParams, Patch, PatchParams, PostParams};
use thorium::{Error, Thorium, models::Version};

use super::clusters::ClusterMeta;

const TRACING_MOUNT_PATH: &str = "/tmp/tracing.yml";

pub async fn label_node(meta: &ClusterMeta, node: &str, version: &str) -> Result<(), Error> {
    println!("labeling {node} with thorium_version={version}");
    // build a label json template
    let label = serde_json::json!({
        "metadata": {
            "labels": {
                "thorium": "enabled",
                "thorium_version": &version,
            }
        }
    });
    // patch the node with the new label
    match meta
        .node_api
        .patch(node, &PatchParams::default(), &Patch::Merge(&label))
        .await
    {
        Ok(_) => {
            println!(
                "Node {node} labeled successfully with thorium=enabled and thorium_version={version}"
            );
            Ok(())
        }
        Err(error) => Err(Error::new(format!("Failed to label node {node}: {error}"))),
    }
}

/// Label Thorium worker nodes in a kubernetes cluster
///
/// Before the Thorium scaler can schedule reactions to run on k8s nodes, those nodes
/// must be labeled with thorium=enabled via the nodes k8s API. This method will label
/// each node listed under the scaler/k8s section of the CRD config.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn label_all_nodes(meta: &ClusterMeta, thorium: &Thorium) -> Result<(), Error> {
    // get the version of Thorium that is being deployed for this node
    let version = thorium.updates.get_version().await?.thorium.to_string();
    // label each node
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    for k8s_cluster in clusters.values() {
        for node in &k8s_cluster.nodes {
            // label this node
            label_node(meta, node, &version).await?;
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
                        "thorium": null,
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
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Cleanup a specific nodes provision pod if it exists
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata for a specific cluster
/// * `node` - The name of the node to cleanup a provision pod for
pub async fn cleanup_provision_pod_specific(meta: &ClusterMeta, node: &str) -> Result<(), Error> {
    // build provisioner pod name
    let name = format!("node-provisioner-{node}");
    let params = DeleteParams::default();
    // attempt deletion of the pod
    match meta.pod_api.delete(&name, &params).await {
        Ok(_) => {
            println!("Cleaning up {} pod", &name);
            Ok(())
        }
        Err(kube::Error::Api(error)) => {
            // don't panic if pods don't exist, thats the desired state
            if error.code == 404 {
                Ok(())
            } else {
                Err(Error::new(format!(
                    "Failed to create {} pod: {}",
                    &name, error
                )))
            }
        }
        Err(error) => Err(Error::new(format!(
            "Failed to create {} pod: {}",
            &name, error
        ))),
    }
}

/// Cleanup node provision pods
///
/// This deletes node provision pods as part of a ``ThoriumCluster`` cleanup. Each node that that
/// is used by Thorium to schedule reactions will have have a pod run on it to configure host
/// paths such as /opt/thorium. This method cleans up any pods that have run to provision host
/// paths.
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn cleanup_provision_pods(meta: &ClusterMeta) -> Result<(), Error> {
    // get a reference to our k8s cluster configs
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    // iterate over and clean up the provision nodes in all clusters
    for k8s_cluster in clusters.values() {
        for node in &k8s_cluster.nodes {
            // clean up this nodes provision pod if it exists
            cleanup_provision_pod_specific(meta, node).await?;
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
/// * `node` - Name of node to deploy pod
pub async fn deploy_provision_pod(meta: &ClusterMeta, node: &str) -> Result<(), Error> {
    let pod_name = format!("node-provisioner-{node}");
    // set default resources for node provision pods
    let resources = serde_json::json!({"cpu": "250m", "memory": "250Mi"});
    let require_registry_auth = if meta.cluster.spec.registry_auth.is_none() {
        false
    } else {
        true
    };
    // only include imagePullSecret if required
    let image_pull_secrets = if require_registry_auth {
        serde_json::json!([
            {
                "name": "registry-token"
            }
        ])
    } else {
        serde_json::json!([])
    };
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
            "imagePullSecrets": image_pull_secrets
        }
    });
    let pod: Pod = serde_json::from_value(pod_template)?;
    // create a provision pod for this node
    match meta.pod_api.create(&PostParams::default(), &pod).await {
        Ok(_) => println!("Node provision pod created: {pod_name}"),
        Err(error) => {
            // don't fail whole operator if node pods fail to create
            return Err(Error::new(format!(
                "Failed to create node provision pod: {error}",
            )));
        }
    }
    Ok(())
}

/// Deploy node provision pods to each thorium enabled k8s server
///
/// This function will spawn a node provision pod on each server designated to be used as
/// a compute node within the Thorium. This provision pod will configure the /opt/thorium
/// directory so that the thorium-agent can run jobs on that system. If no nodes are listed
/// then all nodes visible to the operator will be provisioned.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn deploy_provision_pods(meta: &ClusterMeta) -> Result<(), Error> {
    // cleanup existing provision pods
    cleanup_provision_pods(meta).await?;
    // apply a provision pod to each k8s node for each k8s cluster if any were set
    let clusters = meta.cluster.spec.config.thorium.scaler.k8s.clusters.clone();
    for (name, k8s_cluster) in &clusters {
        // skip any clusters with an api url as the operator doesn't yet support multiple k8s clusters
        if k8s_cluster.api_url.is_some() {
            println!("Skipping provisioning pods on {name} as it has a custom API url!");
            // skip to the next cluster
            continue;
        }
        // get an interator over the node names in our conf or from the k8s api
        let nodes = if k8s_cluster.nodes.is_empty() {
            // list all nodes in this cluster
            let nodes = meta.node_api.list(&ListParams::default()).await?;
            // build a list of nodes to deploy too
            nodes
                .items
                .into_iter()
                .filter_map(|node| node.metadata.name)
                .collect::<Vec<String>>()
        } else {
            // use the names of nodes in our config
            k8s_cluster.nodes.clone()
        };
        // deploy provision pods to nodes in this cluster
        for node in &nodes {
            // provision this node
            deploy_provision_pod(meta, node).await?;
        }
    }
    Ok(())
}
