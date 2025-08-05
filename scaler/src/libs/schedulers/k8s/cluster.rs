use thorium::models::{NodeHealth, NodeUpdate, Resources};
use thorium::{Conf, Error, Thorium};
use tracing::{event, instrument, Level};

use super::Nodes;
use crate::libs::schedulers::AllocatableUpdate;

/// Wrapper for kubernetes cluster api routes
pub struct Cluster {
    /// The name of this cluster
    cluster_name: String,
    /// K8s nodes wrapper
    nodes: Nodes,
}

impl Cluster {
    /// Init k8s cluster wrapper
    ///
    /// # Arguments
    ///
    /// * `client` - Kubernetes client
    /// * `conf` - Thorium Config
    /// * `cluster_name` - The name of this k8s cluster
    /// * `context_name` - The name of this context
    pub fn new<T: Into<String>>(
        client: &kube::Client,
        conf: &Conf,
        cluster_name: T,
        context_name: &str,
    ) -> Self {
        // cast our cluster name
        let cluster_name = cluster_name.into();
        // build k8s wrappers
        let nodes = Nodes::new(client, conf, cluster_name.clone(), context_name);
        Cluster {
            cluster_name,
            nodes,
        }
    }

    /// Gets all available resources within the cluster
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `span` - The span to log traces under
    #[instrument(name = "k8s::Cluster::resources_available", skip_all)]
    pub async fn resources_available(&self, thorium: &Thorium) -> Result<AllocatableUpdate, Error> {
        // create an empty cluster update
        let mut update = AllocatableUpdate::default();
        // build label/field filters for listing nodes
        // only list Thorium-enabled nodes
        let labels = vec!["thorium==enabled"];
        // only list nodes we can schedule pods on
        let fields = vec!["spec.unschedulable==false"];
        // get list of nodes in cluster
        for node in self.nodes.list(&labels, &fields).await? {
            // get this name of this node or skip it
            if let Some(name) = node.metadata.name.clone() {
                // get the usable resources for this node
                match self.nodes.resources_available(node).await? {
                    Some(node_alloc_update) => {
                        // build our node update to send to the API
                        let node_update = NodeUpdate::new(
                            NodeHealth::Healthy,
                            node_alloc_update.available.clone(),
                        );
                        // update this node in Thorium
                        thorium
                            .system
                            .update_node(&self.cluster_name, &name, &node_update)
                            .await?;
                        // log the amount of resources we can schedule on this node
                        event!(
                            Level::INFO,
                            node = name,
                            cpu = node_alloc_update.available.cpu,
                            memory = node_alloc_update.available.memory,
                            storage = node_alloc_update.available.ephemeral_storage,
                            nvidia_gpu = node_alloc_update.available.nvidia_gpu,
                            amd_gpu = node_alloc_update.available.amd_gpu,
                            worker_slots = node_alloc_update.available.worker_slots,
                        );
                        // add this nodes updated resources count
                        update.nodes.insert(name, node_alloc_update);
                    }
                    None => {
                        // log that a node is no longer reachable
                        event!(Level::WARN, msg = "Unreachable node", node = name);
                        // build our node update
                        let node_update =
                            NodeUpdate::new(NodeHealth::Unhealthy, Resources::default());
                        // update this node in Thorium
                        thorium
                            .system
                            .update_node(&self.cluster_name, &name, &node_update)
                            .await?;
                        update.removes.insert(name);
                    }
                }
            }
        }
        Ok(update)
    }
}
