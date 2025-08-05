use k8s_openapi::{api::core::v1::Secret, ByteString};
use kube::api::{DeleteParams, ObjectMeta, Patch, PatchParams, PostParams};
use kube::Api;
use rand::distr::Alphanumeric;
use rand::Rng;
use std::collections::BTreeMap;
use thorium::Error;

use super::clusters::ClusterMeta;

/// Create or update a Secret
///
/// This creates or optionally updates a kubernetes secret if it exists.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `secret` - Secret object to update in the kubernetes API
/// * `update` - Should the operator update this secret if it exists
pub async fn create_or_update(
    meta: &ClusterMeta,
    secret: &Secret,
    update: bool,
) -> Result<bool, Error> {
    // get name and namespace for logging/error handling
    let name = secret
        .metadata
        .name
        .as_ref()
        .expect("could not get secret name");
    // first attempt to create the ConfigMap
    let params = PostParams::default();
    match meta.secret_api.create(&params, &secret).await {
        Ok(_) => {
            println!("Created {} secret in namespace {}", name, &meta.namespace);
            Ok(true)
        }
        Err(kube::Error::Api(error)) => {
            // do not panic if ConfigMap exists, patch it
            if error.reason == "AlreadyExists" {
                if !update {
                    println!(
                        "Warning: secret {} in namespace {} already exists",
                        name, &meta.namespace
                    );
                    return Ok(false);
                }
                let patch = serde_json::json!({
                    "data": secret.data
                });
                let patch = Patch::Merge(&patch);
                let params: PatchParams = PatchParams::default();
                match meta.secret_api.patch(&name, &params, &patch).await {
                    Ok(_) => {
                        println!("Patched {} secret in namespace {}", name, &meta.namespace);
                        Ok(true)
                    }
                    Err(error) => Err(Error::new(format!(
                        "Failed to patch {} secret: {}",
                        name, error
                    ))),
                }
            } else {
                Err(Error::new(format!(
                    "Failed to create {} secret: {}",
                    name, error
                )))
            }
        }
        Err(error) => Err(Error::new(format!(
            "Failed to create {} secret: {}",
            name, error
        ))),
    }
}

/// Build a Secret object
///
///  Arguments
///
/// * `secret` - JSON string secret data
/// * `name` - Name of secret being created
/// * `key` - Key within secret to place secret data
/// * `namespace` - Namespace to create secret within
pub fn build_secret(secret: &str, name: &str, key: &str, namespace: &str) -> Secret {
    let secret = ByteString(secret.to_owned().into_bytes());
    let mut data = BTreeMap::new();
    data.insert(key.to_owned(), secret);
    // create tracing ConfigMap object
    let secret = Secret {
        // metadata for the ConfigMap
        metadata: ObjectMeta {
            name: Some(name.to_owned()),
            namespace: Some(namespace.to_owned()),
            ..Default::default()
        },
        data: Some(data),
        ..Default::default()
    };
    secret
}

/// Create the thorium config secret
///
/// This creates the thorium.yml config secret using the ThoriumCluster CRD.
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn create_thorium_config(meta: &ClusterMeta) -> Result<(), Error> {
    // Create Thorium config secret from ThoriumCluster resource
    let secret_yaml = serde_yaml::to_string(&serde_json::json!(&meta.cluster.spec.config))?;
    // build thorium config secret template
    let thorium_secret = build_secret(
        secret_yaml.as_ref(),
        "thorium",
        "thorium.yml",
        &meta.namespace,
    );
    // create thorium config secret in k8s
    create_or_update(meta, &thorium_secret, true).await?;
    Ok(())
}

/// Create a keys.yml secret for a user
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
/// * `username` - Name of user
/// * `password` - Password for user
/// * `k8s` - Whether this keys.yml secret will be used by a component inside k8s
/// * `secret_name` - Optional name of secret
pub async fn create_keys(
    meta: &ClusterMeta,
    username: &str,
    password: &str,
    k8s: bool,
    secret_name: Option<&str>,
) -> Result<(), Error> {
    // within k8s use the service hostname
    let mut host = format!("http://thorium-api.{}.svc.cluster.local", &meta.namespace);
    // if not in k8s, use the url provided in the CRD
    if k8s == false {
        let urls = meta.cluster.get_api_urls();
        if urls.is_some() && !urls.clone().unwrap().is_empty() {
            // we take first because we don't know which is correct, really this host field
            // is just filler and admins will update when they pull the secret for kaboom
            host = format!(
                "http://{}",
                urls.unwrap()
                    .first()
                    .expect("expected string but found urls vector empty")
            )
            .to_owned();
        }
    }
    let template = serde_json::json!({
        "api": host,
        "username": username,
        "password": password
    })
    .to_string();
    // build out secret name if provided
    let mut name = "keys".to_string();
    if secret_name.is_some() {
        name = secret_name
            .expect("expected secret name for key to be some")
            .to_owned();
    }
    // build a password secret
    let secret = build_secret(
        template.as_ref(),
        name.as_ref(),
        "keys.yml",
        &meta.namespace,
    );
    // actually create the secret in k8s
    create_or_update(meta, &secret, true).await?;
    Ok(())
}

/// Create a user password secret
///
/// User password secrets are created when creating a new Thorium user. This helps
/// the operator track the authentication information for its admin user account as
/// well as any additional users needed for Thorium operation. If the thorium-operator
/// account password secret is lost, external action will need to be taken to delete
/// that user account.
///
///  Arguments
///
/// * `username` - Name of the Thorium user
/// * `meta` - Thorium cluster client and metadata
pub async fn create_user_secret(
    username: &str,
    meta: &ClusterMeta,
) -> Result<Option<String>, Error> {
    // generate alphanumeric password for operator
    let password: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    // user password secrets are names based on the username
    let secret_name = format!("{username}-pass");
    // build a password secret
    let secret = build_secret(
        password.as_ref(),
        secret_name.as_ref(),
        username,
        &meta.namespace,
    );
    // create the secret but do not force update
    if !create_or_update(meta, &secret, false).await? {
        return Ok(None);
    }
    Ok(Some(password))
}

/// Get kubernetes secret by name
///
///  Arguments
///
/// * `secret_api` - API for interacting with kubernetes secrets
/// * `secret_name` - Name of secret to retrieve
pub async fn get_secret(
    secret_api: &Api<Secret>,
    secret_name: &String,
) -> Result<Option<Secret>, Error> {
    return match secret_api.get(secret_name).await {
        Ok(secret) => Ok(Some(secret)),
        Err(kube::Error::Api(error)) => {
            // do not panic when secret does not exists
            if error.reason == "NotFound" {
                Ok(None)
            } else {
                Err(Error::new(format!(
                    "Failed to get {} secret: {}",
                    secret_name, error
                )))
            }
        }
        Err(error) => Err(Error::new(format!(
            "Failed to get {} secret: {}",
            secret_name, error
        ))),
    };
}

/// Retrieve password from a user k8s secret
///
///  Arguments
///
/// * `username` - Name of the Thorium user
/// * `meta` - Thorium cluster client and metadata
pub async fn get_user_password(
    username: &str,
    meta: &ClusterMeta,
) -> Result<Option<String>, Error> {
    // get secret from k8s
    let secret_name = format!("{username}-pass");
    let secret = get_secret(&meta.secret_api, &secret_name).await?;
    if secret.is_none() {
        // secret was not found
        Ok(None)
    } else {
        // secret found, attempt to decode it
        let secret_data = secret
            .expect("expected secret to be some")
            .data
            .expect("expected secret data to be some");
        let byte_string = secret_data.get(username);
        if let Some(ByteString(raw_bytes)) = byte_string {
            let decoded =
                std::str::from_utf8(&raw_bytes).expect("decoding of secret was not valid utf8");
            return Ok(Some(decoded.to_owned()));
        }
        Ok(None)
    }
}

/// Create or update Thorium secrets
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn create_or_update_config(meta: &ClusterMeta) -> Result<(), Error> {
    // create thorium config secret
    create_thorium_config(meta).await?;
    Ok(())
}

/// Create registry tokens secret from ThoriumCluster CRD
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn create_or_update_registry_auth(meta: &ClusterMeta) -> Result<(), Error> {
    // check if registry auth tokens have been provided
    let registries = meta.cluster.spec.registry_auth.clone();
    if registries.is_none() {
        println!("No registry auth provided, skipping registry secret creation");
        return Ok(());
    }
    // build the url/token structure for the registry auth data
    // it should look like {}
    let mut auth_map: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (url, token) in registries.unwrap().iter() {
        let registry_auth = BTreeMap::from([("auth".to_string(), token.to_owned())]);
        auth_map.insert(url.to_owned(), registry_auth);
    }
    // build the registry auth secret data
    let template = serde_json::json!({"auths": auth_map}).to_string();
    // build the docker skopeo secret for scaler
    let skopeo_secret = build_secret(
        template.as_ref(),
        "docker-skopeo",
        "config.json",
        &meta.namespace,
    );
    // create the secret but do not force update
    println!("Creating skopeo registry secret");
    create_or_update(meta, &skopeo_secret, true).await?;
    // build a container pull secret
    let mut pull_secret = build_secret(
        template.as_ref(),
        "registry-token",
        ".dockerconfigjson",
        &meta.namespace,
    );
    // we need to change the default type since default is Opaque
    println!("Creating registry pull secret");
    pull_secret.type_ = Some("kubernetes.io/dockerconfigjson".to_string());
    // create the secret and force update
    create_or_update(meta, &pull_secret, true).await?;
    Ok(())
}

/// Cleanup Thorium secrets
///
///  Arguments
///
/// * `meta` - Thorium cluster client and metadata
pub async fn delete(meta: &ClusterMeta) -> Result<(), Error> {
    let params: DeleteParams = DeleteParams::default();
    // delete secrets from vector
    // Do not delete the "thorium-operator-pass" unless deleting the thorium-operator user from the Thorium API
    let secrets_names = vec![
        "thorium".to_string(),
        "keys".to_string(),
        "keys-kaboom".to_string(),
        "thorium-pass".to_string(),
        "thorium-kaboom-pass".to_string(),
        "docker-skopeo".to_string(),
    ];
    for secret_name in secrets_names.iter() {
        match meta.secret_api.delete(secret_name, &params).await {
            Ok(_) => {
                println!("Deleted {} secret", secret_name);
            }
            Err(kube::Error::Api(error)) => {
                // secret was not found, continue on
                if error.code == 404 {
                    println!("Secret {} does not exist, skipping deletion", secret_name);
                    continue;
                }
                return Err(Error::new(format!(
                    "Failed to delete {} secret: {}",
                    secret_name, error
                )));
            }
            Err(error) => {
                return Err(Error::new(format!(
                    "Failed to delete {} secret: {}",
                    secret_name, error
                )))
            }
        }
    }
    Ok(())
}
