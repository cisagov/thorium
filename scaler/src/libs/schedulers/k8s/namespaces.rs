use k8s_openapi::api::core::v1::Namespace;
use kube::api::{Api, ListParams, ObjectList, PostParams};
use std::collections::HashSet;
use tracing::{event, instrument, Level, Span};

/// Wrapper for namespace commands in k8s
pub struct Namespaces {
    /// API client for calling namespace commands in k8s
    api: Api<Namespace>,
}

impl Namespaces {
    /// Build a new wrapper for k8s functions regarding namespaces
    ///
    /// # Arguments
    ///
    /// * `client` - Kubernetes client
    pub fn new(client: &kube::Client) -> Self {
        // get namespaces api
        let api: Api<Namespace> = Api::all(client.clone());
        Namespaces { api }
    }

    /// List all namespaces
    pub async fn list(&self) -> Result<ObjectList<Namespace>, kube::Error> {
        self.api.list(&ListParams::default()).await
    }

    /// Create a namespace
    ///
    /// # Arguments
    ///
    /// * `name` - The namespace to create
    #[instrument(name = "k8s::Namespaces::create", skip(self, bans))]
    pub async fn create(&self, name: &str, bans: &mut HashSet<String>) {
        // build create params
        let params = PostParams::default();
        // create namespace
        let mut ns = Namespace::default();
        ns.metadata.name = Some(name.to_owned());
        // log that we are trying to create a namespace
        match self.api.create(&params, &ns).await {
            Ok(_) => event!(Level::INFO, msg = "Created namespace", namespace = name),
            Err(err) => {
                // log that we failed to create this namespace and are banning it
                event!(
                    Level::ERROR,
                    msg = "Failed to create namespace",
                    namespace = name,
                    ban = name,
                    error = err.to_string()
                );
                // ban this namespace
                bans.insert(name.to_owned());
            }
        }
    }
}

impl std::fmt::Debug for Namespaces {
    /// Implement debug
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NameSpaces").finish_non_exhaustive()
    }
}
