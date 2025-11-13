//! The watcher for changes in Thorium CRs

use futures::StreamExt;
use kube::api::ListParams;
use kube::runtime::Controller;
use kube::runtime::controller::Action;
use kube::runtime::finalizer::{Event as Finalizer, finalizer};
use kube::runtime::watcher::Config;
use kube::{Api, Client};
use std::sync::Arc;
use std::time::Duration;
use thorium::Error;

use crate::args::OperateCluster;
use crate::k8s::clusters::ClusterMeta;
use crate::k8s::controller::SharedInfo;
use crate::k8s::crds::{self, ThoriumCluster};
use crate::k8s::operate;

/// Controller state including kubeapi client and url
#[derive(Clone)]
pub struct State {
    /// kube API client
    client: Client,
    /// ingress route for Thorium API
    url: Option<String>,
    // The shared thorium operator info
    shared: Arc<SharedInfo>,
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
            Finalizer::Apply(_cluster) => {
                operate::apply(&meta, state.url.clone(), &state.shared).await
            }
            Finalizer::Cleanup(_cluster) => operate::cleanup(&meta).await,
        }
    })
    .await
    .map_err(|e| Error::new(format!("Finalizer error: {}", e)))
}

/// Watch our nodes for any changes
pub async fn start(client: Client, args: OperateCluster, shared: Arc<SharedInfo>) {
    // list ThoriumCluster resources
    let clusters_api: Api<ThoriumCluster> = Api::<ThoriumCluster>::all(client.clone());
    if let Err(e) = clusters_api.list(&ListParams::default().limit(1)).await {
        println!("Failed to list ThoriumCluster API: {}", e);
        std::process::exit(1);
    }
    let state = State {
        client: client.clone(),
        url: args.url.clone(),
        shared,
    };
    // create the ThoriumCluster controller to watch for resource changes
    Controller::new(clusters_api, Config::default().any_semantic())
        .shutdown_on_signal()
        .run(reconcile, error_policy, state.to_context())
        .filter_map(|x| async move { std::result::Result::ok(x) })
        .for_each(|_| futures::future::ready(()))
        .await;
}
