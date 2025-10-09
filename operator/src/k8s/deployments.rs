use chrono;
use k8s_openapi::api::apps::v1::Deployment;
use kube::api::{DeleteParams, ListParams, Patch, PatchParams, PostParams};
use serde_json::Value;
use std::time;
use thorium::{Error, client::Basic};
use tokio;

use super::clusters::ClusterMeta;
use super::crds;

/// Build JSON template for api deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
async fn api_template(meta: &ClusterMeta) -> Option<Value> {
    let api_spec = meta.cluster.get_api_spec();
    // only include imagePullSecret if required
    let image_pull_secrets = if !meta.cluster.spec.registry_auth.is_none() {
        serde_json::json!([
            {
                "name": "registry-token"
            }
        ])
    } else {
        serde_json::json!([])
    };
    match api_spec {
        Some(api_spec) => Some(serde_json::json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "namespace": meta.cluster.metadata.namespace.clone(),
                "name": "api",
                "labels": {
                    "app": "api",
                    "version": meta.cluster.get_version(),
                }
            },
            "spec": {
                "replicas": api_spec.replicas.clone(),
                "selector": {
                    "matchLabels": {
                        "app": "api",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "api",
                            "version": meta.cluster.get_version(),
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "api",
                                "image": meta.cluster.get_image(),
                                "command": api_spec.cmd.clone(),
                                "args": api_spec.args.clone(),
                                "imagePullPolicy": meta.cluster.spec.image_pull_policy.clone(),
                                "resources": {
                                    "limits": crds::Resources::request_conv(&api_spec.resources).expect("failed to convert resources to valid request format"),
                                    "requests": crds::Resources::request_conv(&api_spec.resources).expect("failed to convert resources to valid request format"),
                                },
                                "env": api_spec.env.clone(),
                                "volumeMounts": [
                                    {
                                        "name": "config",
                                        "mountPath": "/conf/thorium.yml",
                                        "subPath": "thorium.yml"
                                    },
                                    {
                                        "name": "banner",
                                        "mountPath": "/app/banner.txt",
                                        "subPath": "banner.txt"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "secret": {
                                    "secretName": "thorium"
                                }
                            },
                            {
                                "name": "banner",
                                "configMap": {
                                    "name": "banner",
                                    "optional": true,
                                }
                            }
                        ],
                        "imagePullSecrets": image_pull_secrets
                    }
                }
            }
        })),
        None => None,
    }
}

/// Build JSON template for baremetal-scaler deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
async fn scaler_template(meta: &ClusterMeta) -> Option<Value> {
    let scaler_spec = meta.cluster.get_scaler_spec();
    // only include imagePullSecret if required
    match scaler_spec {
        Some(scaler_spec) => {
            let image_pull_secrets = if !meta.cluster.spec.registry_auth.is_none() {
                serde_json::json!([
                    {
                        "name": "registry-token"
                    }
                ])
            } else {
                serde_json::json!([])
            };
            let mut volumes = serde_json::json!([
                {
                    "name": "config",
                    "secret": {
                        "secretName": "thorium"
                    }
                },
                {
                    "name": "keys",
                    "secret": {
                        "secretName": "keys"
                    }
                }
            ]);
            let mut volume_mounts = serde_json::json!([
                {
                    "name": "config",
                    "mountPath": "/conf/thorium.yml",
                    "subPath": "thorium.yml"
                },
                {
                    "name": "keys",
                    "mountPath": "/keys/keys.yml",
                    "subPath": "keys.yml"
                },
            ]);
            // only include skopeo secret when registry auth is configured
            if !meta.cluster.spec.registry_auth.is_none() {
                volumes.as_array_mut().unwrap().push(serde_json::json!({
                    "name": "docker-skopeo",
                    "secret": {
                        "secretName": "docker-skopeo"
                    }
                }));
                volume_mounts
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!({
                        "name": "docker-skopeo",
                        "mountPath": "/root/.docker"
                    }));
            }
            if !scaler_spec.service_account {
                // if not using service account we must map in a user created kube-config secret
                volumes.as_array_mut().unwrap().push(serde_json::json!({
                    "name": "kube-config",
                    "secret": {
                        "secretName": "kube-config"
                    }
                }));
            };
            Some(serde_json::json!({
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "namespace": meta.cluster.metadata.namespace.clone(),
                    "name": "scaler",
                    "labels": {
                        "app": "scaler",
                        "version": meta.cluster.get_version(),
                    }
                },
                "spec": {
                    "replicas": 1,
                    "selector": {
                        "matchLabels": {
                            "app": "scaler",
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "scaler",
                                "version": meta.cluster.get_version(),
                            }
                        },
                        "spec": {
                            "serviceAccountName": if scaler_spec.service_account {
                                Some("thorium")
                            } else {
                                None
                            },
                            "automountServiceAccountToken": scaler_spec.service_account.clone(),
                            "containers": [
                                {
                                    "name": "scaler",
                                    "image": meta.cluster.get_image(),
                                    "imagePullPolicy": meta.cluster.spec.image_pull_policy.clone(),
                                    "command": scaler_spec.cmd.clone(),
                                    "args": scaler_spec.args.clone(),
                                    "resources": {
                                        "resources": {
                                            "limits": crds::Resources::request_conv(&scaler_spec.resources).expect("failed to convert resources to valid request format"),
                                            "requests": crds::Resources::request_conv(&scaler_spec.resources).expect("failed to convert resources to valid request format"),
                                        },
                                    },
                                    "env": scaler_spec.env.clone(),
                                    "volumeMounts": volume_mounts
                                }
                            ],
                            "volumes": volumes,
                            "imagePullSecrets": image_pull_secrets
                        }
                    }
                }
            }))
        }
        None => None,
    }
}

/// Build JSON template for baremetal-scaler deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
async fn baremetal_scaler_template(meta: &ClusterMeta) -> Option<Value> {
    let scaler_spec = meta.cluster.get_baremetal_scaler_spec();
    // only include imagePullSecret if required
    let image_pull_secrets = if !meta.cluster.spec.registry_auth.is_none() {
        serde_json::json!([
            {
                "name": "registry-token"
            }
        ])
    } else {
        serde_json::json!([])
    };
    match scaler_spec {
        Some(scaler_spec) => Some(serde_json::json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "namespace": meta.cluster.metadata.namespace.clone(),
                "name": "baremetal-scaler",
                "labels": {
                    "app": "baremetal-scaler",
                    "version": meta.cluster.get_version(),
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "baremetal-scaler",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "baremetal-scaler",
                            "version": meta.cluster.get_version(),
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "baremetal-scaler",
                                "image": meta.cluster.get_image(),
                                "imagePullPolicy": meta.cluster.spec.image_pull_policy.clone(),
                                "command": scaler_spec.cmd.clone(),
                                "args": scaler_spec.args.clone(),
                                "resources": {
                                    "limits": crds::Resources::request_conv(&scaler_spec.resources).expect("failed to convert resources to valid request format"),
                                    "requests": crds::Resources::request_conv(&scaler_spec.resources).expect("failed to convert resources to valid request format"),
                                },
                                "env": scaler_spec.env.clone(),
                                "volumeMounts": [
                                    {
                                        "name": "config",
                                        "mountPath": "/conf/thorium.yml",
                                        "subPath": "thorium.yml"
                                    },
                                    {
                                        "name": "keys",
                                        "mountPath": "/keys/keys.yml",
                                        "subPath": "keys.yml"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "secret": {
                                    "secretName": "thorium"
                                }
                            },
                            {
                                "name": "keys",
                                "secret": {
                                    "secretName": "keys"
                                }
                            }
                        ],
                        "imagePullSecrets": image_pull_secrets
                    }
                }
            }
        })),
        None => None,
    }
}

/// Build JSON template for event-handler deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
async fn event_handler_template(meta: &ClusterMeta) -> Option<Value> {
    let handler_spec = meta.cluster.get_event_handler_spec();
    // only include imagePullSecret if required
    let image_pull_secrets = if !meta.cluster.spec.registry_auth.is_none() {
        serde_json::json!([
            {
                "name": "registry-token"
            }
        ])
    } else {
        serde_json::json!([])
    };
    match handler_spec {
        Some(handler_spec) => Some(serde_json::json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "namespace": meta.cluster.metadata.namespace.clone(),
                "name": "event-handler",
                "labels": {
                    "app": "event-handler",
                    "version": meta.cluster.get_version(),
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "event-handler",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "event-handler",
                            "version": meta.cluster.get_version(),
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "event-handler",
                                "image": meta.cluster.get_image(),
                                "imagePullPolicy": meta.cluster.spec.image_pull_policy.clone(),
                                "command": handler_spec.cmd.clone(),
                                "args": handler_spec.args.clone(),
                                "resources": {
                                    "limits": crds::Resources::request_conv(&handler_spec.resources).expect("failed to convert resources to valid request format"),
                                    "requests": crds::Resources::request_conv(&handler_spec.resources).expect("failed to convert resources to valid request format"),
                                },
                                "env": handler_spec.env.clone(),
                                "volumeMounts": [
                                    {
                                        "name": "config",
                                        "mountPath": "/conf/thorium.yml",
                                        "subPath": "thorium.yml",
                                    },
                                    {
                                        "name": "keys",
                                        "mountPath": "/keys/keys.yml",
                                        "subPath": "keys.yml"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "secret": {
                                    "secretName": "thorium"
                                },
                            },
                            {
                                "name": "keys",
                                "secret": {
                                    "secretName": "keys"
                                }
                            }
                        ],
                        "imagePullSecrets": image_pull_secrets
                    }
                }
            }
        })),
        None => None,
    }
}

/// Build JSON template for search-streamer deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
async fn search_streamer_template(meta: &ClusterMeta) -> Option<Value> {
    let streamer_spec = meta.cluster.get_search_streamer_spec();
    // only include imagePullSecret if required
    let image_pull_secrets = if !meta.cluster.spec.registry_auth.is_none() {
        serde_json::json!([
            {
                "name": "registry-token"
            }
        ])
    } else {
        serde_json::json!([])
    };
    match streamer_spec {
        Some(streamer_spec) => Some(serde_json::json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "namespace": meta.cluster.metadata.namespace.clone(),
                "name": "search-streamer",
                "labels": {
                    "app": "search-streamer",
                    "version": meta.cluster.get_version(),
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "search-streamer",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "search-streamer",
                            "version": meta.cluster.get_version(),
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "search-streamer",
                                "image": meta.cluster.get_image(),
                                "imagePullPolicy": meta.cluster.spec.image_pull_policy.clone(),
                                "command": streamer_spec.cmd.clone(),
                                "args": streamer_spec.args.clone(),
                                "resources": {
                                    "limits": crds::Resources::request_conv(&streamer_spec.resources).expect("failed to convert resources to valid request format"),
                                    "requests": crds::Resources::request_conv(&streamer_spec.resources).expect("failed to convert resources to valid request format"),
                                },
                                "env": streamer_spec.env.clone(),
                                "volumeMounts": [
                                    {
                                        "name": "config",
                                        "mountPath": "/conf/thorium.yml",
                                        "subPath": "thorium.yml"
                                    },
                                    {
                                        "name": "keys",
                                        "mountPath": "/keys/keys.yml",
                                        "subPath": "keys.yml"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "secret": {
                                    "secretName": "thorium"
                                }
                            },
                            {
                                "name": "keys",
                                "secret": {
                                    "secretName": "keys"
                                }
                            }
                        ],
                        "imagePullSecrets": image_pull_secrets
                    }
                }
            }
        })),
        None => None,
    }
}

/// Build deployment templates for ThoriumCluster components
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
#[allow(dead_code)]
pub async fn get_templates(meta: &ClusterMeta) -> Result<Vec<Deployment>, Error> {
    let mut deployments: Vec<Deployment> = Vec::with_capacity(5);
    // add any api deployment templates
    if let Some(deployment) = api_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        deployments.push(deployment);
    }
    // add any scaler deployment templates
    if let Some(deployment) = scaler_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        deployments.push(deployment);
    }
    // add any baremetal scaler deployment templates
    if let Some(deployment) = baremetal_scaler_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        deployments.push(deployment);
    }
    // add any event handler deployment templates
    if let Some(deployment) = event_handler_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        deployments.push(deployment);
    }
    // add any search streamer deployment templates
    if let Some(deployment) = search_streamer_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        deployments.push(deployment);
    }
    Ok(deployments)
}

/// Create or update a k8s deployment within a given namespace
///
///  Arguments
///
/// * `deployment` - Deployment spec to create or patch if already exists
/// * `meta` - Thorium cluster client and metadata
pub async fn create_or_update(deployment: Deployment, meta: &ClusterMeta) -> Result<(), Error> {
    let params = PostParams::default();
    let name = deployment
        .metadata
        .name
        .clone()
        .expect("could not get cluster name from metadata");
    match meta.deploy_api.create(&params, &deployment).await {
        Ok(_) => {
            println!(
                "Deployment created {} in namespace {}",
                &name, &meta.namespace
            );
            Ok(())
        }
        Err(kube::Error::Api(error)) => {
            // do not panic if deployment exists
            if error.reason == "AlreadyExists" {
                let patch = serde_json::json!({
                    "spec": deployment.spec
                });
                let patch = Patch::Merge(&patch);
                let params: PatchParams = PatchParams::default();
                match meta.deploy_api.patch(&name, &params, &patch).await {
                    Ok(_) => {
                        println!(
                            "Patched {} deployment in namespace {}",
                            &name, &meta.namespace
                        );

                        restart(&name, meta).await?;
                        Ok(())
                    }
                    Err(error) => Err(Error::new(format!(
                        "Failed to patch {} deployment: {}",
                        &name, error
                    ))),
                }
            } else {
                Err(Error::new(format!(
                    "Failed to create {} deployment: {}",
                    &name, error
                )))
            }
        }
        Err(error) => Err(Error::new(format!(
            "Failed to create {} deployment: {}",
            &name, error
        ))),
    }
}

/// Delete a k8s deployment within a given namespace
///
///  Arguments
///
/// * `name` - Name of deployment to delete
/// * `meta` - Thorium cluster client and metadata
pub async fn delete_one(name: &str, meta: &ClusterMeta) -> Result<(), Error> {
    let params = DeleteParams::default();
    // delete the deployment by name
    match meta.deploy_api.delete(name, &params).await {
        Ok(_) => println!(
            "Deleted {} deployment from namespace {}",
            name, &meta.namespace
        ),
        Err(kube::Error::Api(error)) => {
            // don't panic if deployment doesn't exist, thats the desired state
            if error.code == 404 {
                println!(
                    "No {} deployment in namespace {} to delete, skipping cleanup",
                    &name, &meta.namespace
                );
                return Ok(());
            }
            return Err(Error::new(format!(
                "Failed things to delete {} deployment in namespace {}: {}",
                &name, &meta.namespace, error.message
            )));
        }
        Err(error) => {
            return Err(Error::new(format!(
                "Failed things to delete {} deployment in namespace {}: {}",
                &name, &meta.namespace, error
            )));
        }
    }
    Ok(())
}
/// Create or update the API deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn deploy_api(meta: &ClusterMeta) -> Result<(), Error> {
    // add any api deployment templates
    if let Some(deployment) = api_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        create_or_update(deployment, meta).await?;
    // component not present in cluster spec during upgrades, cleanup
    } else {
        delete_one("api", meta).await?;
    }
    Ok(())
}

/// Create or update the scaler deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn deploy_scalers(meta: &ClusterMeta) -> Result<(), Error> {
    // deploy any scaler from template
    if let Some(deployment) = scaler_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        create_or_update(deployment, meta).await?;
    // component not present in cluster spec during upgrades, cleanup
    } else {
        delete_one("scaler", meta).await?;
    }
    // deploy any baremetal scaler from template
    if let Some(deployment) = baremetal_scaler_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        create_or_update(deployment, meta).await?;
    // component not present in cluster spec during upgrades, cleanup
    } else {
        delete_one("baremetal-scaler", meta).await?;
    }
    Ok(())
}

/// Create or update the event-handler deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn deploy_event_handler(meta: &ClusterMeta) -> Result<(), Error> {
    // deploy any event handler from template
    if let Some(deployment) = event_handler_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        create_or_update(deployment, meta).await?;
    // component not present in cluster spec during upgrades, cleanup
    } else {
        delete_one("event-handler", meta).await?;
    }
    Ok(())
}

/// Create or update the search-streamer deployment
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn deploy_search_streamer(meta: &ClusterMeta) -> Result<(), Error> {
    // deploy any event handler from template
    if let Some(deployment) = search_streamer_template(meta).await {
        let deployment: Deployment = serde_json::from_value(deployment)?;
        create_or_update(deployment, meta).await?;
    // component not present in cluster spec during upgrades, cleanup
    } else {
        delete_one("event-handler", meta).await?;
    }
    Ok(())
}

/// Wait for pods with correct app label to be online
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `app_tag` - App name to wait for running state
/// * `host` - Optional hostname to ping to determine healthy state of pods after they are running
pub async fn wait_for_pods(
    meta: &ClusterMeta,
    app_tag: &str,
    host: Option<&String>,
) -> Result<(), Error> {
    let label_selector = format!("app={},version={}", app_tag, meta.cluster.get_version()).clone();
    let params = ListParams::default().labels(label_selector.as_str());
    // create some durations for sleeps
    let one_second = tokio::time::Duration::from_secs(1);
    let five_seconds = tokio::time::Duration::from_secs(5);
    // app pods may need some time to start spinning up
    tokio::time::sleep(five_seconds).await;
    // check if pods are running
    println!("Waiting for {} labeled pods to be running", &label_selector);
    loop {
        let mut running = true;
        // get a list of pods with the api label
        let pods = meta.pod_api.list(&params).await?;
        // check pods status
        for pod in &pods.items {
            let status = pod
                .status
                .as_ref()
                .map(|s| s.phase.clone().unwrap_or_default())
                .unwrap_or_default();
            if status != "Running" {
                running = false;
            }
        }
        // break if all pods were found running
        if running == true {
            println!(
                "All \"{}\" pods with labels \"{}\" are now running",
                app_tag, label_selector
            );
            break;
        }
        println!(
            "Waiting for \"{}\" pods with labels \"{}\" to be running...",
            app_tag, label_selector
        );
        tokio::time::sleep(one_second).await;
    }
    // if app is the API, then lets make sure the API is responding
    if app_tag == "api" && host.is_some() {
        let thorium_client = reqwest::Client::new();
        let basic = Basic::new(
            host.expect("expected host to be some url but found none"),
            &thorium_client,
        );
        // status check loop
        loop {
            // ping Thorium API for status
            match basic.health().await {
                Ok(response) => {
                    // API responds, return success
                    println!("Thorium API is up and returned response \"{}\"", response);
                    break;
                }
                Err(error) => {
                    println!("Waiting for API pods to respond: {}", error);
                    tokio::time::sleep(one_second).await;
                    continue;
                }
            }
        }
    }
    Ok(())
}

/// Wait for a set amount of time for pods to be running
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `app_tag` - App tag to use for retrieving API pod status
/// * `timeout_secs` - Number of seconds to wait before timing out
/// * `host` - Optional hostname to ping to determine healthy state of pods after they are running
pub async fn timeout_wait_for_pods(
    meta: &ClusterMeta,
    app_tag: &str,
    timeout_secs: u64,
    host: Option<&String>,
) -> Result<(), Error> {
    // build label_selector string
    let label_selector = format!("app={},version={}", app_tag, meta.cluster.get_version()).clone();
    let wait_future = wait_for_pods(meta, app_tag, host);
    // run wait for pods with specified timeout
    if let Err(_error) =
        tokio::time::timeout(time::Duration::from_secs(timeout_secs), wait_future).await
    {
        return Err(Error::new(format!(
            "\"{}\" pods with labels \"{}\" did not start within {} seconds",
            app_tag, label_selector, timeout_secs
        )));
    }
    Ok(())
}

/// Restart a Thorium component pod by deployment name
///
///  Arguments
///
/// * `name` - Deployment name to restart
/// * `meta` - Thorium cluster client and metadata
pub async fn restart(name: &str, meta: &ClusterMeta) -> Result<(), Error> {
    println!(
        "Restarting {} deployment in namespace {}",
        name, &meta.namespace
    );
    let params = PatchParams::default();
    // build an annotation with timestamp
    let patch = serde_json::json!({
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": chrono::Utc::now().to_rfc3339(),
                    }
                }
            }
        }
    });
    // attempt to patch deployment with timestamp annotation
    match meta
        .deploy_api
        .patch(name, &params, &Patch::Merge(&patch))
        .await
    {
        Ok(_) => println!(
            "Restarted {} deployment in {} namespace",
            name, &meta.namespace
        ),
        Err(error) => {
            eprintln!("Failed to restart {} deployment: {}", name, error);
        }
    }
    Ok(())
}

/// Delete thorium component deployments
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn delete(meta: &ClusterMeta) -> Result<(), Error> {
    println!("Cleaning up deployments using CRD");
    // delete each deployment from the namespace
    let params = DeleteParams::default();
    for deployment in meta.cluster.list_component_names().into_iter() {
        match meta.deploy_api.delete(&deployment, &params).await {
            Ok(_) => println!(
                "Deleted {} deployment from {} namespace",
                &deployment, &meta.namespace
            ),
            Err(kube::Error::Api(error)) => {
                // don't panic if pods don't exist, thats the desired state
                if error.code == 404 {
                    println!("No {} deployment to delete, skipping cleanup", &deployment);
                    continue;
                }
                return Err(Error::new(format!(
                    "Failed to delete {} deployment: {}",
                    &deployment, error.message
                )));
            }
            Err(error) => {
                return Err(Error::new(format!(
                    "Failed to delete {} deployment: {}",
                    &deployment, error
                )));
            }
        }
    }
    Ok(())
}
