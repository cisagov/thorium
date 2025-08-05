use k8s_openapi::api::core::v1::Node;
use kube::api::{Api, ListParams, ObjectList, Patch, PatchParams};
use serde_json::json;
use std::collections::HashSet;
use thorium::models::Resources;
use thorium::{Conf, Error};
use tracing::{event, instrument, Level};

use super::Pods;
use crate::libs::helpers;
use crate::libs::schedulers::NodeAllocatableUpdate;

// extract and convert a resource then subtract it
macro_rules! subtract {
    ($orig:expr, $func:expr, $map:expr, $name:expr) => {
        $orig = $orig.saturating_sub($func($map.get($name))?);
    };
}

/// Get the resources for a specific node
fn get_resources(node: &Node) -> Result<Resources, Error> {
    // throw an error if this node doesn't have a status object
    if let Some(status) = &node.status {
        // extract this nodes allocatable resources
        if let Some(alloc) = &status.allocatable {
            // validate and convert this resource counts to a standard format
            let mut cpu = helpers::cpu(alloc.get("cpu"))?;
            let mut memory = helpers::storage(alloc.get("memory"))?;
            let mut ephemeral_storage = helpers::storage(alloc.get("ephemeral-storage"))?;
            // take 2 cores or 2 Gibibytes from each of our resources
            cpu = cpu.saturating_sub(2000);
            memory = memory.saturating_sub(2048);
            ephemeral_storage = ephemeral_storage.saturating_sub(2048);
            // build our resources objects
            return Ok(Resources {
                cpu,
                memory,
                ephemeral_storage,
                worker_slots: 100,
                nvidia_gpu: 0,
                amd_gpu: 0,
            });
        }
    }
    // we could not get this nodes resources
    Err(Error::new(format!(
        "Failed to get resources for node {:#?}",
        node
    )))
}

/// Wrapper for node api routes in k8s
pub struct Nodes {
    /// API client for node commands in k8s
    api: Api<Node>,
    /// Wrapper for pod commands in k8s
    pods: Pods,
}

impl Nodes {
    /// Build new wrapper for k8s functions regarding nodes
    ///
    /// # Arguments
    ///
    /// * `client` - Kubernetes client
    /// * `conf` - Thorium Config
    /// * `cluster_name` - The name of this cluster
    /// * `context_name` - The name of this context
    pub fn new<T: Into<String>>(
        client: &kube::Client,
        conf: &Conf,
        cluster_name: T,
        context_name: &str,
    ) -> Self {
        // get node api
        let api: Api<Node> = Api::all(client.clone());
        // build pods wrapper
        let pods = Pods::new(client, conf, cluster_name, context_name);
        Nodes { api, pods }
    }

    /// List all nodes in this cluster
    ///
    /// # Arguments
    ///
    /// * `labels` - The labels to restrict to
    /// * `fields` - The field selectors to use
    pub async fn list(
        &self,
        labels: &[&str],
        fields: &[&str],
    ) -> Result<ObjectList<Node>, kube::Error> {
        // build list params
        let params = ListParams::default();
        // insert any label filters into list params
        let params = labels
            .iter()
            .fold(params, |params, label| params.labels(label));
        // insert any fields selectors into the list params
        let params = fields
            .iter()
            .fold(params, |params, field| params.fields(field));
        // get list of all nodes
        self.api.list(&params).await
    }

    /// Calculate the resources a single node has available
    ///
    /// # Arguments
    ///
    /// * `nodes` - The node to check for available resources
    #[instrument(name = "k8s::Nodes::resources_available", skip_all, err(Debug))]
    pub async fn resources_available(
        &self,
        node: Node,
    ) -> Result<Option<NodeAllocatableUpdate>, Error> {
        // get this nodes name
        let name = match node.metadata.name.clone() {
            Some(name) => name,
            None => return Err(Error::new("node does not have a name")),
        };
        // check the nodes taints and see if this node can be scheduled on
        if let Some(spec) = &node.spec {
            if let Some(taints) = &spec.taints {
                // return 0 resources because this node cannot be schedule on
                if taints.iter().any(|taint| taint.effect == "NoSchedule") {
                    event!(Level::WARN, node = &name, taint = "NoSchedule");
                    return Ok(None);
                }
            }
        }
        // get the total available resources for this node
        let total = get_resources(&node)?;
        event!(
            Level::INFO,
            node = &name,
            total_cpu = total.cpu,
            total_memory = total.memory,
            total_storage = total.ephemeral_storage,
            total_nvidia_gpu = total.nvidia_gpu,
            total_amd_gpu = total.amd_gpu,
            total_worker_slots = total.worker_slots
        );
        // clone our total resources to calculate whats actually allocatable
        let mut available = total.clone();
        // get list of all pods on this node
        let pods = self.pods.list_all(Some(name.clone())).await?;
        // build a list of currently active workers
        let mut active = HashSet::with_capacity(pods.items.len());
        // crawl over the pods on this node
        for pod in pods {
            // if this pod is Thorium-owned then add it to our assigned count
            if Pods::thorium_owned(&pod) {
                // add it to our active pod list
                if let Some(name) = pod.metadata.name.clone() {
                    active.insert(name);
                }
            }
            // skip any pods without a spec
            if let Some(spec) = pod.spec {
                // decrease this nodes pod slot number by 1 to account for this pod
                available.worker_slots = available.worker_slots.saturating_sub(1);
                // crawl over the resource requests for containers in this pod
                for requests in spec
                    .containers
                    .into_iter()
                    .filter_map(|cont| cont.resources)
                    .filter_map(|res| res.requests)
                {
                    // subtract this containers resources from our resource count
                    subtract!(available.cpu, helpers::cpu, requests, "cpu");
                    subtract!(available.memory, helpers::storage, requests, "memory");
                    subtract!(
                        available.ephemeral_storage,
                        helpers::storage,
                        requests,
                        "ephemeral-storage"
                    );
                    // log the resources of this node after subtracting this existing pod
                    event!(
                        Level::INFO,
                        node = &name,
                        existing = &pod.metadata.name,
                        cpu = total.cpu,
                        memory = total.memory,
                        storage = total.ephemeral_storage,
                        nvidia_gpu = total.nvidia_gpu,
                        amd_gpu = total.amd_gpu,
                        worker_slots = total.worker_slots
                    );
                }
            }
        }
        // build our node update
        let node_update = NodeAllocatableUpdate {
            available,
            total,
            active,
        };
        Ok(Some(node_update))
    }

    /// Label a node
    ///
    /// # Arguments
    ///
    /// * `node` - The name of the node to label
    /// * `label` - The label to create/overwrite
    /// * `value` - The value of the label to set/overwrite
    #[allow(dead_code)]
    pub async fn label<'a>(
        &self,
        node: &'a str,
        label: &str,
        value: &str,
    ) -> Result<&'a str, Error> {
        // build label patch
        let patch = json!({
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {
                "labels": {
                    label: value
                }
            }
        });
        // cast serde value to a Patch
        let patch = Patch::Apply(&patch);
        // build patch params
        let params = PatchParams {
            field_manager: Some("Thorium".to_owned()),
            ..Default::default()
        };
        // patch node labels
        let patched = self.api.patch(node, &params, &patch).await?;
        // make sure out patch was succesful
        if let Some(labels) = patched.metadata.labels {
            if labels.get(label) != Some(&value.to_owned()) {
                let msg = format!("Failed to label node {} with {}:{}", node, label, value);
                return Err(Error::new(msg));
            }
        }
        Ok(node)
    }
}
