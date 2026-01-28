use k8s_openapi::api::core::v1::Service;
use kube::{
    api::{DeleteParams, Patch, PatchParams, PostParams},
    runtime::reflector::Lookup,
};
use serde_json::Value;
use thorium::Error;

use super::clusters::ClusterMeta;

/// Build a Thorium API service
///
/// This creates an API service JSON template that can be used to create a
/// kubernetes application service for the Thorium API.
fn api_service() -> Result<Service, serde_json::Error> {
    // build the template for a thorium ai service
    let template = serde_json::json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "thorium-api"
        },
        "spec": {
            "selector": {
                "app": "api"
            },
            "ports": [
                {
                    "name": "web",
                    "port": 80,
                    "targetPort": 80
                }
            ],
            "type": "ClusterIP"
        }
    });
    // parse our tempalte into a service
    serde_json::from_value(template)
}

/// Build a Thorium API MCP service
///
/// This creates an API service JSON template that can be used to create a
/// kubernetes application service for the Thorium MCP API.
fn mcp_service() -> Result<Service, serde_json::Error> {
    // build the template for a thorium mcp service
    let template = serde_json::json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "thorium-mcp"
        },
        "spec": {
            "selector": {
                "mcp": "enabled"
            },
            "ports": [
                {
                    "name": "web",
                    "port": 80,
                    "targetPort": 80
                }
            ],
            "type": "ClusterIP"
        }
    });
    // parse our tempalte into a service
    serde_json::from_value(template)
}

async fn update(meta: &ClusterMeta, name: &str, service: &Service) -> Result<(), Error> {
    // build the patch template to apply
    let patch = serde_json::json!({
        "spec": &service.spec
    });
    // parse our patch template into a path
    let patch = Patch::Merge(&patch);
    // use default patch params
    let params: PatchParams = PatchParams::default();
    // apply this patch
    match meta.service_api.patch(name, &params, &patch).await {
        Ok(_) => {
            // log that we patched this service
            println!("Patched {name} service in namespace {}", &meta.namespace);
            Ok(())
        }
        // an error occured during patching
        Err(error) => Err(Error::new(format!(
            "Failed to patch {name} service in namespace {}: {}",
            &meta.namespace, error
        ))),
    }
}

/// Create an API service from a template
///
/// This creates an API service so network traffic can be routed to the API
/// from internal or external locations (using an ingress proxy like Traefik).
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `service` - The service to update
pub async fn create_or_update(meta: &ClusterMeta, service: &Service) -> Result<(), Error> {
    // get this services name
    let name = match service.name() {
        Some(name) => name,
        None => {
            return Err(Error::new(format!(
                "Failed to get name of service: {service:?}"
            )));
        }
    };
    // use default post params
    let post_params = PostParams::default();
    // try to create this service
    match meta.service_api.create(&post_params, &service).await {
        // log that we successfully created this service
        Ok(_) => {
            println!("{name} service created in namespace {}", &meta.namespace);
            Ok(())
        }
        // we ran into an error creating this service
        // maybe it already exists and we need to patch it
        Err(kube::Error::Api(error)) => {
            // do not panic if service exists
            if error.reason == "AlreadyExists" {
                // this service already exists patch it
                update(meta, &name, service).await
            } else {
                // we ran into a different problem so catch fire
                Err(Error::new(format!(
                    "Failed to create {name} service in namespace {}: {}",
                    &meta.namespace, error
                )))
            }
        }
        // we ran into a different problem so catch fire
        Err(error) => Err(Error::new(format!(
            "Failed to create {name} service in namespace {}: {}",
            &meta.namespace, error
        ))),
    }
}

/// Create an API service from a template
///
/// This creates an API service so network traffic can be routed to the API
/// from internal or external locations (using an ingress proxy like Traefik).
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn create_or_update_all(meta: &ClusterMeta) -> Result<(), Error> {
    // build API service template
    let api_service = api_service()?;
    // create or update this ap service
    create_or_update(meta, &api_service).await?;
    // build mcp service template
    let mcp_service = mcp_service()?;
    // create or update this ap service
    create_or_update(meta, &mcp_service).await
}

/// Cleanup Thorium API service
///
/// This deletes a Thorium API service from kubernetes based on the default
/// service name thorium-api.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn delete(meta: &ClusterMeta) -> Result<(), Error> {
    let params: DeleteParams = DeleteParams::default();
    // delete the Thorium service
    let service_name = "thorium-api".to_string();
    match meta.service_api.delete(&service_name, &params).await {
        Ok(_) => println!("Deleted {} service", &service_name),
        Err(kube::Error::Api(error)) => {
            // service was not found, continue on
            if error.code == 404 {
                println!("Service {} does not exist, skipping deletion", service_name);
                return Ok(());
            }
            return Err(Error::new(format!(
                "Could not delete {} service: {}",
                service_name, error.message
            )));
        }
        Err(error) => {
            return Err(Error::new(format!(
                "Could not delete {} service: {}",
                service_name, error
            )));
        }
    }
    Ok(())
}
