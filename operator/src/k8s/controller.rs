use crate::k8s::crds::ThoriumCluster;
use kube::api::{Api, ListParams};
use kube::client::Client;
use kube::config::{KubeConfigOptions, Kubeconfig};
use std::sync::Arc;
use thorium::{Error, Thorium};

use super::watchers;
use crate::args::OperateCluster;
use crate::k8s::clusters::ClusterMeta;
use crate::k8s::crds;

/// Info and client about our deployed Thorium cluster
#[derive(Clone)]
pub struct ThoriumInfo {
    /// A client for this Thorium clusters api
    pub thorium: Arc<Thorium>,
    /// The config for this Thorium cluster
    pub meta: Arc<ClusterMeta>,
}

/// The controller the Thorium k8s operator
#[derive(Default)]
pub struct SharedInfo {
    /// The info for different clusters in thorium
    pub info: papaya::HashMap<String, ThoriumInfo>,
}

impl SharedInfo {
    /// Get the Thorium info for a specific node
    pub fn get_for_node(&self, k8s_cluster: &str, node: &String) -> Result<ThoriumInfo, Error> {
        // iterate over our clusters
        for (_, info) in &self.info.pin() {
            // get a ref to our k8s clusters
            let k8s_config = &info.meta.cluster.spec.config.thorium.scaler.k8s;
            // get the k8s cluster this node comes from
            if let Some(cluster) = k8s_config.clusters.get(k8s_cluster) {
                // return the cluster that we found if it contains our node or is empty
                if cluster.nodes.is_empty() || cluster.nodes.contains(node) {
                    return Ok(info.to_owned());
                }
            }
        }
        Err(Error::new(format!("No config for {k8s_cluster}:{node}")))
    }
}

async fn get_k8s_clients() -> Result<Vec<(String, Client)>, Error> {
    // try to load the kubeconfig from the environment
    let Some(kube_conf) = Kubeconfig::from_env()? else {
        return Err(Error::new("Failed to load k8s config"));
    };
    // build a list of clients and their context names
    let mut clients = Vec::with_capacity(1);
    // iterate over all contexts in this kube config and build a client for each of them
    // ignoring any of the ignored clusters
    for context in &kube_conf.contexts {
        // build the options for getting a specific clusters config
        let opts = KubeConfigOptions {
            context: Some(context.name.clone()),
            ..Default::default()
        };
        // get this clusters config
        let cluster_conf = kube::Config::from_custom_kubeconfig(kube_conf.clone(), &opts).await?;
        // create a client based on this config
        let client = kube::Client::try_from(cluster_conf)?;
        // add this client and context name to our list of clients
        clients.push((context.name.clone(), client));
    }
    Ok(clients)
}

/// Initialize the controller and shared state (given the crd is installed)
///
/// Arguments
///
/// * `args` - Arguments passed to the thorium-operator operate sub command
pub async fn run(args: OperateCluster) {
    // TODO: explicitly set ring as default crypto provider, otherwise we get panics
    // when creating the kube client; possibly fixed in newer versions of the kube crate
    // so we might be able to remove this after upgrading kube
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to set 'ring' as default crypto provider");
    }
    // get clients for all Thorium clusters
    let clients = get_k8s_clients()
        .await
        .expect("Failed to get kubernetes clients");
    // initialize our shared info across controllers/watchers
    let shared = Arc::new(SharedInfo::default());
    // instance a set to spawn our watchers into
    let mut watchers = tokio::task::JoinSet::new();
    // spawn watchers for all clients
    for (name, client) in clients {
        // the crd always has to exist before we can read the resource from k8s
        // create the ThoriumCluster CRD in k8s
        crds::create_or_update(&client)
            .await
            .expect("failed to create ThoriumCluster CRD");
        // list ThoriumCluster resources
        let clusters_api: Api<ThoriumCluster> = Api::<ThoriumCluster>::all(client.clone());
        if let Err(error) = clusters_api.list(&ListParams::default().limit(1)).await {
            println!("Failed to list ThoriumCluster API: {error}");
            std::process::exit(1);
        }
        // start the watchers for this cluster
        watchers::start(name, &client, &args, &shared, &mut watchers);
    }
    // wait for either of our watchers to finish
    if let Some(result) = watchers.join_next().await {
        panic!("A watcher died/finished!: {result:#?}");
    }
}
