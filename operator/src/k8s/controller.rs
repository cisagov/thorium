use crate::k8s::crds::ThoriumCluster;
use futures::StreamExt;
use kube::{
    api::{Api, ListParams},
    client::Client,
    runtime::{
        controller::{Action, Controller},
        finalizer::{finalizer, Event as Finalizer},
        watcher::Config,
    },
};
use std::sync::Arc;
use thorium::Error;
use tokio::time::Duration;

use crate::args::OperateCluster;
use crate::k8s::clusters::ClusterMeta;
use crate::k8s::crds;
use crate::k8s::operate;

const RECONCILE_ERROR_REQUEUE_SECS: u64 = 60u64;

/// Controller state including kubeapi client and url
#[derive(Clone)]
pub struct State {
    /// kube API client
    client: Client,
    /// ingress route for Thorium API
    url: Option<String>,
}

/// Methods operating on controller state
impl State {
    /// Wrap state in Arc
    pub fn to_context(&self) -> Arc<State> {
        Arc::new(self.clone())
    }
}

/// Handle errors in the reconcile process
pub fn error_policy(_cluster: Arc<ThoriumCluster>, error: &Error, _state: Arc<State>) -> Action {
    println!("Controller error:\n\t{}", error);
    println!(
        "Requeuing ThoriumCluster reconciliation in {} seconds",
        RECONCILE_ERROR_REQUEUE_SECS
    );
    Action::requeue(Duration::from_secs(RECONCILE_ERROR_REQUEUE_SECS))
}

/// Reconcile changes to ThoriumCluster
///
/// Arguments
///
/// * `cluster` - Thorium cluster being changed
/// * `state` - Controller context including client instance and optional URL
pub async fn reconcile(cluster: Arc<ThoriumCluster>, state: Arc<State>) -> Result<Action, Error> {
    // build cluster metadata
    let meta = ClusterMeta::new(&cluster, &state.client).await?;
    let clusters_api: Api<ThoriumCluster> = Api::namespaced(meta.client.clone(), &meta.namespace);
    println!(
        "Reconciling ThoriumCluster changes for {} in namespace {}",
        meta.name, meta.namespace
    );
    finalizer(&clusters_api, crds::CRD_NAME, cluster, |event| async {
        match event {
            Finalizer::Apply(_cluster) => operate::apply(&meta, state.url.clone()).await,
            Finalizer::Cleanup(_cluster) => operate::cleanup(&meta).await,
        }
    })
    .await
    .map_err(|e| Error::new(format!("Finalizer error: {}", e)))
}

/// Initialize the controller and shared state (given the crd is installed)
///
/// Arguments
///
/// * `args` - Arguments passed to the thorium-operator operate sub command
pub async fn run(args: &OperateCluster) {
    // TODO: explicitly set ring as default crypto provider, otherwise we get panics
    // when creating the kube client; possibly fixed in newer versions of the kube crate
    // so we might be able to remove this after upgrading kube
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to set 'ring' as default crypto provider");
    }
    let client = Client::try_default()
        .await
        .expect("failed to create kube Client");
    // the crd always has to exist before we can read the resource from k8s
    // create the ThoriumCluster CRD in k8s
    crds::create_or_update(&client)
        .await
        .expect("failed to create ThoriumCluster CRD");
    // list ThoriumCluster resources
    let clusters_api: Api<ThoriumCluster> = Api::<ThoriumCluster>::all(client.clone());
    if let Err(e) = clusters_api.list(&ListParams::default().limit(1)).await {
        println!("Failed to list ThoriumCluster API: {}", e);
        std::process::exit(1);
    }
    let state = State {
        client: client.clone(),
        url: args.url.clone(),
    };
    // create the ThoriumCluster controller to watch for resource changes
    Controller::new(clusters_api, Config::default().any_semantic())
        .shutdown_on_signal()
        .run(reconcile, error_policy, state.to_context())
        .filter_map(|x| async move { std::result::Result::ok(x) })
        .for_each(|_| futures::future::ready(()))
        .await;
}
