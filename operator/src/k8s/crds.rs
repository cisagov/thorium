use k8s_openapi::{
    apiextensions_apiserver::pkg::apis::apiextensions::v1::CustomResourceDefinition,
    apimachinery::pkg::api::resource::Quantity,
};
use kube::{
    api::{Api, Patch, PatchParams},
    core::CustomResourceExt,
    runtime::{conditions, wait::await_condition},
    Client,
};
use kube_derive::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use thorium::Error;

/// A struct representing an environment variable
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct EnvVar {
    pub name: String,
    pub value: Option<String>,
}

/// A struct representing the cpu and memory resources of a container
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct Resources {
    pub cpu: u64,
    pub memory: u64,
}

/// Serde helper for default container resources
fn default_resources() -> Resources {
    Resources {
        // Amount of CPU cores in millicpus
        cpu: 2000,
        // Amount of RAM in mebibytes
        memory: 4096,
    }
}

/// COPIED FROM API without ephemeral/gpus
// used when casting to a quantity
macro_rules! quantity {
    ($($raw:tt)+) => {serde_json::from_value(json!($($raw)+))}
}
impl Resources {
    /// converts a resource request to a BTreeMap
    ///
    /// This will ignore any value that is None
    ///
    /// # Arguments
    ///
    /// * `raw` - The resource request to convert
    pub fn request_conv(raw: &Resources) -> Result<BTreeMap<String, Quantity>, Error> {
        // creat btreemap of requests
        let mut btree = BTreeMap::default();
        // build the resource request map
        btree.insert("cpu".to_owned(), quantity!(format!("{}m", raw.cpu))?);
        btree.insert("memory".to_owned(), quantity!(format!("{}Mi", raw.memory))?);
        Ok(btree)
    }
}

/// Serde helper for default environment variables
fn default_envs() -> Vec<EnvVar> {
    vec![
        EnvVar {
            name: "http_proxy".to_owned(),
            value: Some("".to_owned()),
        },
        EnvVar {
            name: "https_proxy".to_owned(),
            value: Some("".to_owned()),
        },
        EnvVar {
            name: "no_proxy".to_owned(),
            value: Some("localhost,cluster.local".to_owned()),
        },
        EnvVar {
            name: "HTTP_PROXY".to_owned(),
            value: Some("".to_owned()),
        },
        EnvVar {
            name: "HTTPS_PROXY".to_owned(),
            value: Some("".to_owned()),
        },
        EnvVar {
            name: "NO_PROXY".to_owned(),
            value: Some("localhost,cluster.local".to_owned()),
        },
    ]
}

/// Serde helper for default api container args (cmd in a Dockerfile)
fn default_api_cmd() -> Vec<String> {
    vec!["/app/thorium".to_owned()]
}

/// Serde helper for default api container cmd (entrypoint in a Dockerfile)
fn default_api_args() -> Vec<String> {
    vec!["--config".to_owned(), "/conf/thorium.yml".to_owned()]
}

/// Serde helper for default API memory and cpu resources
fn default_api_resources() -> Resources {
    Resources {
        cpu: 2000,
        memory: 8192,
    }
}

/// Thorium API spec
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct ThoriumApi {
    /// Number of API pods to scale the deployment
    pub replicas: u16,
    /// External URLs to map to API service
    pub urls: Option<Vec<String>>,
    /// Ports to listen from at those external URLs
    pub ports: Option<Vec<u16>>,
    /// Environment variables to apply to API container
    #[serde(default = "default_envs")]
    pub env: Vec<EnvVar>,
    /// Commands to run in API container
    #[serde(default = "default_api_cmd")]
    pub cmd: Vec<String>,
    // Args to pass to command in API container
    #[serde(default = "default_api_args")]
    pub args: Vec<String>,
    /// The CPU and Memory needed by the API
    #[serde(default = "default_api_resources")]
    pub resources: Resources,
}

/// Serde helper for default kube config path
fn default_scaler_envs() -> Vec<EnvVar> {
    let mut envs = default_envs();
    envs.push(EnvVar {
        name: "KUBECONFIG".to_owned(),
        value: Some("/root/.kube/config".to_owned()),
    });
    envs
}

/// Serde helper for default scaler container cmd (entrypoint in a Dockerfile)
fn default_scaler_cmd() -> Vec<String> {
    vec!["/app/thorium-scaler".to_owned()]
}

/// Serde helper for default scaler container args (cmd in a Dockerfile)
fn default_scaler_args() -> Vec<String> {
    vec![
        "--config".to_owned(),
        "/conf/thorium.yml".to_owned(),
        "--auth".to_owned(),
        "/keys/keys.yml".to_owned(),
    ]
}

/// Serde helper for default for whether the scaler uses a service account
fn default_service_account() -> bool {
    false
}

/// K8s scaler spec
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct ThoriumScaler {
    /// Environment variables to apply in container
    #[serde(default = "default_scaler_envs")]
    pub env: Vec<EnvVar>,
    /// Commands to run in scaler container
    #[serde(default = "default_scaler_cmd")]
    pub cmd: Vec<String>,
    // Args to pass to command in scaler container
    #[serde(default = "default_scaler_args")]
    pub args: Vec<String>,
    /// The CPU and Memory for a scaler container
    #[serde(default = "default_resources")]
    pub resources: Resources,
    /// whether to use a service account
    #[serde(default = "default_service_account")]
    pub service_account: bool,
}

/// Serde helper for default baremetal scaler container args (cmd in a Dockerfile)
fn default_baremetal_scaler_args() -> Vec<String> {
    let mut args = default_scaler_args();
    args.append(&mut vec!["--scaler".to_owned(), "bare-metal".to_owned()]);
    args
}

/// Baremetal scaler spec
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct ThoriumBaremetalScaler {
    /// Environment variables to apply in container
    #[serde(default = "default_scaler_envs")]
    pub env: Vec<EnvVar>,
    /// Commands to run in scaler container
    #[serde(default = "default_scaler_cmd")]
    pub cmd: Vec<String>,
    // Args to pass to command in scaler container
    #[serde(default = "default_baremetal_scaler_args")]
    pub args: Vec<String>,
    /// The CPU and Memory for a scaler container
    #[serde(default = "default_resources")]
    pub resources: Resources,
}

/// Serde helper for default search-streamer container cmd (entrypoint in a Dockerfile)
fn default_search_streamer_cmd() -> Vec<String> {
    vec!["/app/thorium-search-streamer".to_owned()]
}

/// Serde helper for default search-streamer container args (cmd in a Dockerfile)
fn default_search_streamer_args() -> Vec<String> {
    vec![
        "--config".to_owned(),
        "/conf/thorium.yml".to_owned(),
        "--keys".to_owned(),
        "/keys/keys.yml".to_owned(),
    ]
}

/// Search streamer spec
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct ThoriumSearchStreamer {
    /// Environment variables to apply in container
    #[serde(default = "default_envs")]
    pub env: Vec<EnvVar>,
    /// Commands to run in search-streamer container
    #[serde(default = "default_search_streamer_cmd")]
    pub cmd: Vec<String>,
    // Args to pass to command in search-streamer container
    #[serde(default = "default_search_streamer_args")]
    pub args: Vec<String>,
    /// The CPU and Memory needed by the search-streamer
    #[serde(default = "default_resources")]
    pub resources: Resources,
}

/// Serde helper for default event-handler container cmd (entrypoint in a Dockerfile)
fn default_event_handler_cmd() -> Vec<String> {
    vec!["/app/thorium-event-handler".to_owned()]
}

/// Serde helper for default event-handler container args (cmd in a Dockerfile)
fn default_event_handler_args() -> Vec<String> {
    vec![
        "--config".to_owned(),
        "/conf/thorium.yml".to_owned(),
        "--auth".to_owned(),
        "/keys/keys.yml".to_owned(),
    ]
}

/// Event handler spec
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Hash, Eq, PartialEq)]
pub struct ThoriumEventHandler {
    /// Environment variables to apply in container
    #[serde(default = "default_envs")]
    pub env: Vec<EnvVar>,
    /// Commands to run in event-handler container
    #[serde(default = "default_event_handler_cmd")]
    pub cmd: Vec<String>,
    // Args to pass to command in event-handler container
    #[serde(default = "default_event_handler_args")]
    pub args: Vec<String>,
    /// The CPU and Memory needed by the event-handler
    #[serde(default = "default_resources")]
    pub resources: Resources,
}

/// Thorium components to deploy
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema, Hash, Eq, PartialEq)]
pub struct ThoriumComponents {
    /// Thorium API
    api: Option<ThoriumApi>,
    /// The kubernetes scaler
    scaler: Option<ThoriumScaler>,
    /// The baremetal/kaboom scaler
    baremetal_scaler: Option<ThoriumBaremetalScaler>,
    /// Elastic search streamer
    search_streamer: Option<ThoriumSearchStreamer>,
    /// Event trigger/handler component of Thorium
    event_handler: Option<ThoriumEventHandler>,
}

/// Serde helper for default image pull policy for containers
fn default_pull_policy() -> String {
    "Always".to_string()
}

/// Serde helper for default image version to deploy
fn default_version() -> String {
    //String::from(env!("CARGO_PKG_VERSION"))
    "latest".to_owned()
}

pub const CRD_NAME: &str = "thoriumclusters.sandia.gov";

/// ThoriumCluster CRD definition
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "sandia.gov",
    version = "v1",
    kind = "ThoriumCluster",
    namespaced,
    doc = "Custom resource representing a ThoriumCluster"
)]
pub struct ThoriumClusterSpec {
    /// List of Thorium components to deploy
    pub components: ThoriumComponents,
    /// URL to base container image for Thorium
    pub registry: String,
    /// Version or tag of image, overrides version
    #[serde(default = "default_version")]
    pub version: String,
    /// Auth tokens for Thorium's container registries
    pub registry_auth: Option<BTreeMap<String, String>>,
    /// K8s image pull policies for Thorium components
    #[serde(default = "default_pull_policy")]
    pub image_pull_policy: String,
    /// Configuration options for Thorium components
    pub config: thorium::Conf,
}

/// Methods operating on a ThoriumCluster resource
impl ThoriumCluster {
    /// Get the full image path within the registry
    pub fn get_image(&self) -> String {
        format!("{}:{}", self.spec.registry, self.spec.version)
    }

    /// Get target Thorium image version
    pub fn get_version(&self) -> String {
        self.spec.version.clone()
    }

    /// Get the API component spec
    pub fn get_api_spec(&self) -> Option<&ThoriumApi> {
        if let Some(spec) = &self.spec.components.api {
            Some(&spec)
        } else {
            None
        }
    }

    /// Get API urls from component spec
    pub fn get_api_urls(&self) -> Option<Vec<String>> {
        let spec = self.get_api_spec();
        if spec.is_some() {
            spec.expect("expected api but got none").urls.clone()
        } else {
            None
        }
    }

    /// Get the scaler component spec
    pub fn get_scaler_spec(&self) -> Option<&ThoriumScaler> {
        if let Some(spec) = &self.spec.components.scaler {
            Some(&spec)
        } else {
            None
        }
    }

    /// Get the baremetal scaler component spec
    pub fn get_baremetal_scaler_spec(&self) -> Option<&ThoriumBaremetalScaler> {
        if let Some(spec) = &self.spec.components.baremetal_scaler {
            Some(&spec)
        } else {
            None
        }
    }

    /// Get the event handler component spec
    pub fn get_event_handler_spec(&self) -> Option<&ThoriumEventHandler> {
        if let Some(spec) = &self.spec.components.event_handler {
            Some(&spec)
        } else {
            None
        }
    }

    /// Get the search streamer component spec
    pub fn get_search_streamer_spec(&self) -> Option<&ThoriumSearchStreamer> {
        if let Some(spec) = &self.spec.components.search_streamer {
            Some(&spec)
        } else {
            None
        }
    }

    /// List the components in the ThoriumCluster
    pub fn list_component_names(&self) -> Vec<String> {
        // a list of component names
        let mut names: Vec<String> = Vec::new();
        if let Some(_) = self.spec.components.api {
            names.push("api".to_owned());
        }
        if let Some(_) = self.spec.components.scaler {
            names.push("scaler".to_owned());
        }
        if let Some(_) = self.spec.components.baremetal_scaler {
            names.push("baremetal-scaler".to_owned());
        }
        if let Some(_) = self.spec.components.search_streamer {
            names.push("search-streamer".to_owned());
        }
        if let Some(_) = self.spec.components.event_handler {
            names.push("event-handler".to_owned());
        }
        names
    }
}

/// Build ThoriumCluster stub for testing
#[allow(dead_code)]
pub async fn get_stub_resource() -> Result<ThoriumCluster, Error> {
    let raw_thorium_cluster_spec = json!({
        "name": "ThoriumExample",
        "nodes": ["server1", "server2", "server3"],
        "components": {
            "api": {"replicas": 1, "urls": ["some_url"], "ports": [80, 443]},
            "scaler": {},
            "baremetal_scaler": {},
            "search_streamer": {},
        },
        "registry": "url:port/path/to/image",
        "tag": "tag"
    });
    // build ThoriumCluster spec from json
    let thorium_cluster_spec: ThoriumClusterSpec =
        serde_json::from_value(raw_thorium_cluster_spec)?;
    // create the ThoriumCluster cr using the ThoriumCluster spec
    let thorium_cluster = ThoriumCluster::new("ThoriumProduction", thorium_cluster_spec);
    // print the ThoriumCluster as yaml
    println!(
        "{}",
        serde_yaml::to_string(&thorium_cluster)
            .expect("could not turn ThoriumCluster to YAML string")
    );
    Ok(thorium_cluster)
}

/// Create or update the ThoriumCluster CRD
pub async fn create_or_update(client: &Client) -> Result<(), Error> {
    let params = PatchParams::apply("thorium_cluster_apply").force();
    let crd_api: Api<CustomResourceDefinition> = Api::all(client.clone());
    // create the CRD for this operator version or patch it if it already exists
    crd_api
        .patch(CRD_NAME, &params, &Patch::Apply(ThoriumCluster::crd()))
        .await?;
    // wait for crd to be setup
    let established = await_condition(crd_api, CRD_NAME, conditions::is_crd_established());
    // timeout if CRD isn't setup in N seconds
    let result = tokio::time::timeout(tokio::time::Duration::from_secs(30), established).await;
    // ensure CRD is established before continuing on
    match result {
        Ok(_) => println!("ThoriumCluster CRD applied"),
        Err(_) => {
            return Err(Error::new(format!(
                "Timed out waiting for ThoriumCluster CRD to be established"
            )))
        }
    }
    Ok(())
}
