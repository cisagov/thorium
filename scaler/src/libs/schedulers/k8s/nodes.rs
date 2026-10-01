use k8s_openapi::api::core::v1::Node;
use kube::api::{Api, ListParams, ObjectList, Patch, PatchParams};
use serde_json::json;
use std::collections::HashSet;
use thorium::conf::BurstableNodeResources;
use thorium::models::{BurstableResources, Resources};
use thorium::{Conf, Error};
use tracing::{Level, event, instrument};

use super::Pods;
use crate::libs::helpers;
use crate::libs::schedulers::NodeAllocatableUpdate;

// perform a saturating subtraction on a resource
macro_rules! subtract {
    ($orig:expr, $consumed:expr) => {
        $orig = $orig.saturating_sub($consumed);
    };
}

/// Get the resources for a specific node
fn get_resources(node: &Node, config: &BurstableNodeResources) -> Result<Resources, Error> {
    // throw an error if this node doesn't have a status object
    if let Some(status) = &node.status {
        // extract this nodes allocatable resources
        if let Some(alloc) = &status.allocatable {
            // validate and convert this resource counts to a standard format
            let mut cpu = helpers::cpu(alloc.get("cpu"))?;
            let mut memory = helpers::storage(alloc.get("memory"))?;
            let mut ephemeral_storage = helpers::storage(alloc.get("ephemeral-storage"))?;
            let nvidia_gpu = helpers::u64_quantity(alloc.get("nvidia.com/gpu"))?;
            let amd_gpu = helpers::u64_quantity(alloc.get("amd.com/gpu"))?;
            // get the total number of pods that are allocatable
            let worker_slots = helpers::u64_quantity(alloc.get("pods"))?;
            // take 2 cores or 2 Gibibytes from each of our resources
            cpu = cpu.saturating_sub(2000);
            memory = memory.saturating_sub(2048);
            ephemeral_storage = ephemeral_storage.saturating_sub(2048);
            // calculate how many burstable resources this node has
            let burstable = BurstableResources::new(cpu, memory, config);
            // build our resources objects
            return Ok(Resources {
                cpu,
                memory,
                ephemeral_storage,
                worker_slots,
                nvidia_gpu,
                amd_gpu,
                burstable,
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
    /// Wrapper for pod comamnds in k8s
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
        config: &BurstableNodeResources,
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
        let total = get_resources(&node, config)?;
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
            // subtract this pods resource requests from what is still available on the node
            Self::subtract_pod_resources(&mut available, &total, &name, &pod)?;
        }
        // build our node update
        let node_update = NodeAllocatableUpdate {
            available,
            total,
            active,
        };
        Ok(Some(node_update))
    }

    /// Subtract a single pods container resource requests from a nodes available resources
    ///
    /// This is the per-pod accounting loop in [`Nodes::resources_available`] (it never needs
    /// `self`) extracted so its logic can be unit tested without a live kube client. Terminal
    /// (Succeeded/Failed) pods hold no resources on the node so they are skipped.
    ///
    /// # Arguments
    ///
    /// * `available` - The resources still available on the node, decremented in place
    /// * `total` - The nodes total resources, used only for logging
    /// * `name` - The nodes name, used only for logging
    /// * `pod` - The pod whose container requests to subtract
    fn subtract_pod_resources(
        available: &mut Resources,
        total: &Resources,
        name: &str,
        pod: &k8s_openapi::api::core::v1::Pod,
    ) -> Result<(), Error> {
        // get this pods phase if it has one
        let phase = pod
            .status
            .as_ref()
            .and_then(|status| status.phase.as_deref());
        // skip terminal pods since the kubelet no longer reserves their resources
        if matches!(phase, Some("Succeeded" | "Failed")) {
            return Ok(());
        }
        // skip any pods without a spec
        if let Some(spec) = &pod.spec {
            // decrease this nodes pod slot number by 1 to account for this pod
            available.worker_slots = available.worker_slots.saturating_sub(1);
            // crawl over the resource requests for containers in this pod
            for (requests, limits) in spec
                .containers
                .iter()
                .filter_map(|cont| cont.resources.as_ref())
                .map(|res| (res.requests.as_ref(), res.limits.as_ref()))
            {
                // get our requested resources and consume any burstable ones
                match (requests, limits) {
                    // we have both requests and limits calculate
                    (Some(requests), Some(limits)) => {
                        // get our requests
                        let cpu_req = helpers::cpu(requests.get("cpu"))?;
                        let memory_req = helpers::storage(requests.get("memory"))?;
                        let ephemeral_storage_req =
                            helpers::storage(requests.get("ephemeral-storage"))?;
                        let nvidia_gpu_req = helpers::u64_quantity(requests.get("nvidia.com/gpu"))?;
                        let amd_gpu_req = helpers::u64_quantity(requests.get("amd.com/gpu"))?;
                        // calculate our limits
                        let cpu_lim = helpers::cpu(limits.get("cpu"))?;
                        let memory_lim = helpers::storage(limits.get("memory"))?;
                        // diff our requests against our limits and consume burstable resources
                        let cpu_burst = cpu_lim.saturating_sub(cpu_req);
                        let memory_burst = memory_lim.saturating_sub(memory_req);
                        // consume burstable resources
                        subtract!(available.burstable.cpu, cpu_burst);
                        subtract!(available.burstable.memory, memory_burst);
                        // consume our requested resources
                        subtract!(available.cpu, cpu_req);
                        subtract!(available.memory, memory_req);
                        subtract!(available.ephemeral_storage, ephemeral_storage_req);
                        subtract!(available.nvidia_gpu, nvidia_gpu_req);
                        subtract!(available.amd_gpu, amd_gpu_req);
                    }
                    // We only have requests so no burstabl resources were used
                    (Some(requests), None) => {
                        // get our requests
                        let cpu_req = helpers::cpu(requests.get("cpu"))?;
                        let memory_req = helpers::storage(requests.get("memory"))?;
                        let ephemeral_storage_req =
                            helpers::storage(requests.get("ephemeral-storage"))?;
                        let nvidia_gpu_req = helpers::u64_quantity(requests.get("nvidia.com/gpu"))?;
                        let amd_gpu_req = helpers::u64_quantity(requests.get("amd.com/gpu"))?;
                        // consume our requested resources
                        subtract!(available.cpu, cpu_req);
                        subtract!(available.memory, memory_req);
                        subtract!(available.ephemeral_storage, ephemeral_storage_req);
                        subtract!(available.nvidia_gpu, nvidia_gpu_req);
                        subtract!(available.amd_gpu, amd_gpu_req);
                    }
                    // no requests found somehow just ignore it
                    _ => continue,
                }
                // log the resources of this node after subtracting this existing pod
                event!(
                    Level::INFO,
                    node = name,
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
        Ok(())
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
                let msg = format!("Failed to label node {node} with {label}:{value}");
                return Err(Error::new(msg));
            }
        }
        Ok(node)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{Container, Pod, PodSpec, PodStatus, ResourceRequirements};
    use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
    use std::collections::BTreeMap;

    /// Build a pod in the given phase whose single container requests the given cpu/memory
    ///
    /// # Arguments
    ///
    /// * `name` - The name to give the pod
    /// * `phase` - The `status.phase` to set (e.g. "Running", "Succeeded")
    /// * `cpu` - The cpu request in millicpu (e.g. "500m")
    /// * `memory` - The memory request in mebibytes (e.g. "512Mi")
    fn pod_with_requests(name: &str, phase: &str, cpu: &str, memory: &str) -> Pod {
        // build the container resource requests map
        let mut requests = BTreeMap::new();
        requests.insert("cpu".to_owned(), Quantity(cpu.to_owned()));
        requests.insert("memory".to_owned(), Quantity(memory.to_owned()));
        // build a single container carrying those requests
        let container = Container {
            name: "tool".to_owned(),
            resources: Some(ResourceRequirements {
                requests: Some(requests),
                limits: None,
                claims: None,
            }),
            ..Default::default()
        };
        // build the pod carrying that container in the requested phase
        let mut pod = Pod::default();
        pod.metadata.name = Some(name.to_owned());
        pod.spec = Some(PodSpec {
            containers: vec![container],
            ..Default::default()
        });
        pod.status = Some(PodStatus {
            phase: Some(phase.to_owned()),
            ..Default::default()
        });
        pod
    }

    /// A terminal (Succeeded/Failed) pods requests are NOT subtracted from the nodes available
    /// resources, so terminal pods that still linger on a node do not understate its availability
    /// on every Resources tick.
    #[test]
    fn terminal_pod_requests_are_not_subtracted() {
        // a node that starts with 8 cores and 16 GiB available
        let total = Resources::new(8000, 16384, 0, 10);
        let mut available = total.clone();
        // a completed (Succeeded) pod whose pod object still lingers on the node
        let pod = pod_with_requests("done-worker", "Succeeded", "500m", "512Mi");
        // account for the terminal pod against the nodes available resources
        Nodes::subtract_pod_resources(&mut available, &total, "node-a", &pod).unwrap();
        // a terminal pod holds no resources, so cpu availability must be unchanged
        assert_eq!(
            available.cpu, total.cpu,
            "a Succeeded pods cpu request must not be subtracted from node availability"
        );
        // likewise its memory must not be subtracted
        assert_eq!(
            available.memory, total.memory,
            "a Succeeded pods memory request must not be subtracted from node availability"
        );
    }

    /// A Running pods requests are subtracted from the nodes available resources, so skipping
    /// terminal pods does not also skip live ones.
    #[test]
    fn running_pod_requests_are_subtracted() {
        // a node that starts with 8 cores and 16 GiB available
        let total = Resources::new(8000, 16384, 0, 10);
        let mut available = total.clone();
        // a live (Running) pod consuming half a core and 512 MiB
        let pod = pod_with_requests("live-worker", "Running", "500m", "512Mi");
        // account for the running pod against the nodes available resources
        Nodes::subtract_pod_resources(&mut available, &total, "node-a", &pod).unwrap();
        // a running pod holds its requested cpu, so availability drops by 500 millicpu
        assert_eq!(
            available.cpu,
            total.cpu - 500,
            "a Running pods cpu request must be subtracted from node availability"
        );
        // and its memory availability drops by the requested 512 MiB
        assert_eq!(
            available.memory,
            total.memory - 512,
            "a Running pods memory request must be subtracted from node availability"
        );
    }
}
