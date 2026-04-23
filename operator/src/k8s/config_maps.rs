use k8s_openapi::api::core::v1::ConfigMap;
use kube::api::{DeleteParams, ObjectMeta, Patch, PatchParams, PostParams};
use std::collections::BTreeMap;
use thorium::{Error, conf::Tracing};

use super::clusters::ClusterMeta;

/// Create or update a ConfigMap
///
/// This creates a kubernetes ConfigMap in the ThoriumCluster namespace using a
/// preconstructed ConfigMap object.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `cm` - Kubernetes ConfigMap to create or patch
pub async fn create_or_update(meta: &ClusterMeta, cm: &ConfigMap) -> Result<(), Error> {
    // get name and namespace for logging/error handling
    let name = cm
        .metadata
        .name
        .clone()
        .expect("could not get ConfigMap name");
    // first attempt to create the ConfigMap
    let params = PostParams::default();
    match meta.cm_api.create(&params, &cm).await {
        Ok(_) => {
            println!(
                "Created {} ConfigMap in namespace {}",
                &name, &meta.namespace
            );
            Ok(())
        }
        Err(kube::Error::Api(error)) => {
            // do not panic if ConfigMap exists, patch it
            if error.reason == "AlreadyExists" {
                let patch = serde_json::json!({
                    "data": cm.data
                });
                let patch = Patch::Merge(&patch);
                let params: PatchParams = PatchParams::default();
                match meta.cm_api.patch(&name, &params, &patch).await {
                    Ok(_) => {
                        println!(
                            "Patched {} ConfigMap in namespace {}",
                            &name, &meta.namespace
                        );
                        Ok(())
                    }
                    Err(error) => Err(Error::new(format!(
                        "Failed to patch {} ConfigMap: {}",
                        &name, error
                    ))),
                }
            } else {
                Err(Error::new(format!(
                    "Failed to create {} ConfigMap: {}",
                    &name, error
                )))
            }
        }
        Err(error) => Err(Error::new(format!(
            "Failed to create {} ConfigMap: {}",
            &name, error
        ))),
    }
}

/// Create Thorium tracing ConfigMap
///
/// The tracing.yml ConfigMap is used by the agents and the Thorium reactor and it's
/// configuration is embedded in the ThoriumCluster definition (and thorium.yml configuration).
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `tracing` - Thorium's tracing configuration settings
async fn create_tracing_cm(meta: &ClusterMeta, tracing: &Tracing) -> Result<(), Error> {
    // Create tracing config from ThoriumCluster resource
    let tracing_cm_name = "tracing-conf".to_string();
    let tracing_yaml = serde_norway::to_string(&serde_json::json!(&tracing))
        .expect("thorium config could not be converted to yaml string");
    let mut data = BTreeMap::new();
    data.insert("tracing.yml".to_string(), tracing_yaml);
    // Create tracing ConfigMap object
    let tracing_conf = ConfigMap {
        // Metadata for the ConfigMap
        metadata: ObjectMeta {
            name: Some(tracing_cm_name.clone()),
            namespace: Some(meta.namespace.to_owned()),
            ..Default::default()
        },
        data: Some(data),
        ..Default::default()
    };
    // now actually create/update the tracing cm
    create_or_update(meta, &tracing_conf).await?;
    Ok(())
}

/// Create all ConfigMaps for a Thorium cluster
///
/// This creates all CMs for a ThoriumCluster deployment. Right now since most
/// configurations contain secrets, CMs are limited to the tracing.yml conf.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn create_or_update_all(meta: &ClusterMeta) -> Result<(), Error> {
    // create tracing config
    create_tracing_cm(meta, &meta.cluster.spec.config.thorium.tracing).await?;
    Ok(())
}

/// Cleanup Thorium config maps
///
/// This cleans up all ConfigMap resources from a Thorium cluster's namespace
/// and is used when deleting a ThoriumCluster resource from k8s. Right now
/// this only cleans up the tracing configuration since all other configs are
/// kubernetes secrets.
///
/// # Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn delete(meta: &ClusterMeta) -> Result<(), Error> {
    let params: DeleteParams = DeleteParams::default();
    // delete the Thorium secret
    let tracing_conf_name = "tracing-conf";
    match meta.cm_api.delete(tracing_conf_name, &params).await {
        Ok(_) => {
            println!("Deleted {} ConfigMap", &tracing_conf_name);
        }
        Err(kube::Error::Api(error)) => {
            // ConfigMap was not found, continue on
            if error.code == 404 {
                println!(
                    "ConfigMap {} does not exist, skipping deletion",
                    tracing_conf_name
                );
            } else {
                return Err(Error::new(format!(
                    "Could not delete {} ConfigMap: {}",
                    &tracing_conf_name, error
                )));
            }
        }
        Err(error) => {
            return Err(Error::new(format!(
                "Could not delete {} ConfigMap: {}",
                &tracing_conf_name, error
            )));
        }
    }
    Ok(())
}
