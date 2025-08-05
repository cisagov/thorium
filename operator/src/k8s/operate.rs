use kube::runtime::controller::Action;
use thorium::{Error, Thorium};
use tokio::time::Duration;

use crate::app;
use crate::k8s::{self, clusters::ClusterMeta};

const POD_WAIT_TIMEOUT: u64 = 300u64;
const APPLY_REQUEUE_SECS: u64 = 86400u64;

/// Create a ThoriumCluster
///
/// This creates a Thorium cluster from scratch using a ThoriumCluster CRD as defined
/// in the k8s API.
///
/// Arguments
///
/// * `meta` - Thorium cluster metadata being operated upon
/// * `url` - Override url for Kubernetes api service.
pub async fn apply(meta: &ClusterMeta, url: Option<String>) -> Result<Action, Error> {
    println!(
        "Applying {} ThoriumCluster in {} namespace",
        &meta.name, &meta.namespace
    );
    // create ThoriumCluster namespace if none
    k8s::namespaces::try_create(&meta.namespace).await?;
    // create or update ConfigMaps
    k8s::config_maps::create_or_update_all(&meta).await?;
    // create or update thorium config secret
    k8s::secrets::create_or_update_config(&meta).await?;
    // create or update registry secrets
    k8s::secrets::create_or_update_registry_auth(&meta).await?;
    // create required s3 buckets if not exists
    app::helpers::create_all_buckets(&meta).await?;
    // create or update API service
    k8s::services::create_or_update(&meta).await?;
    // create or update api deployment from CR
    k8s::deployments::deploy_api(&meta).await?;
    // build API url host string
    let host: String = app::helpers::get_thorium_host(&meta, url.as_ref());
    // wait for api pods to be up
    let result =
        k8s::deployments::timeout_wait_for_pods(&meta, "api", POD_WAIT_TIMEOUT, Some(&host)).await;
    // exit cluster update if pod update timed out
    if result.is_err() {
        println!("{}", result.expect_err("expected error but found ()"));
        println!("Error: Timed out waiting for API pod to be reachable, exiting cluster provision");
        // exit cluster provision operation early here and requeue
        return Ok(Action::requeue(Duration::from_secs(60)));
    }
    // create operator user and retrieve token
    let operator_token = app::users::create_operator(&meta, &host).await?;
    // build out operator thorium client
    let operator = Thorium::build(host.clone())
        .token(&operator_token)
        .build()
        .await?;
    // create thorium user using operator token
    let (thorium_password, thorium_token) =
        app::users::create(&meta, &operator, &host, "thorium").await?;
    // build out thorium user's thorium client
    let thorium = Thorium::build(host.clone())
        .token(&thorium_token)
        .build()
        .await?;
    // create thorium-kaboom user using operator token
    let (kaboom_password, _) = app::users::create(&meta, &thorium, &host, "thorium-kaboom").await?;
    // create keys.yml secret for thorium user
    k8s::secrets::create_keys(&meta, "thorium", &thorium_password, true, None).await?;
    // create keys.yml secret for thorium-kaboom user
    k8s::secrets::create_keys(
        &meta,
        "thorium-kaboom",
        &kaboom_password,
        false,
        Some("keys-kaboom"),
    )
    .await?;
    // init cluster system settings
    app::configure::init_settings(&thorium).await?;
    // deploy node provisioner pods
    k8s::nodes::deploy_provision_pods(&meta).await?;
    // create scaler deployments from CR
    k8s::deployments::deploy_scalers(&meta).await?;
    // create event handler deployment from CR
    k8s::deployments::deploy_event_handler(&meta).await?;
    // create event handler deployment from CR
    k8s::deployments::deploy_search_streamer(&meta).await?;
    // label nodes
    k8s::nodes::label_all_nodes(&meta).await?;
    // add nodes to Thorium for each k8s cluster
    app::nodes::add_nodes_to_thorium(&meta, &thorium).await?;
    // deploy version specific upgrades here if needed
    app::upgrades::handler(&meta).await?;
    // log completed ThoriumCluster instance
    println!("Completed creation of {} ThoriumCluster", &meta.name);
    // If no events were received, check back every 5 min
    Ok(Action::requeue(Duration::from_secs(APPLY_REQUEUE_SECS)))
}

/// Delete a ThoriumCluster
///
/// This deletes an existing Thorium cluster, leaving only certain artifacts behind for future
/// ThoriumCluster deployments.
///
/// Notes:
///   Not all cluster remnants are remove with this operation. Databases and database content persist
///   after k8s Thorium resources are cleaned up. User passwords, cluster and node settings will all
///   persist after you delete a ThoriumCluster resource. This also does not remove any on host files
///   such as those dropped into the /opt/thorium directory of each worker node. Since we don't delete
///   the thorium-operator user, we also choose not to delete the corresponding thorium-operator-pass
///   k8s secret. This will allow future reprovisioning of a new ThoriumCluster using the same DBs without
///   manual intervention. If you wipe out the DBs after this operation runs, you will need to manually
///   delete that secret; otherwise, provisioning with that user will fail. Finally, since some resources
///   may remain inside this namespace, we do not delete the namespace from k8s.
///
/// Arguments
///
/// * `meta` - Thorium cluster metadata being operated upon
/// * `url` - Override url for Kubernetes api service.
pub async fn cleanup(meta: &ClusterMeta) -> Result<Action, Error> {
    println!(
        "Deleting {} ThoriumCluster in {} namespace",
        &meta.name, &meta.namespace
    );
    // remove kubernetes node labels
    k8s::nodes::delete_node_labels(&meta).await?;
    // delete node provision pods
    k8s::nodes::cleanup_provision_pods(&meta).await?;
    // remove thorium component deployments
    k8s::deployments::delete(&meta).await?;
    // delete api service
    k8s::services::delete(&meta).await?;
    // remove secrets including thorium.yml and keys.yml
    k8s::secrets::delete(&meta).await?;
    // remove configmaps such as tracing.yml
    k8s::config_maps::delete(&meta).await?;
    // log completion of cluster deletion
    println!("Completed cleanup of {} ThoriumCluster", &meta.name);
    Ok(Action::await_change())
}
