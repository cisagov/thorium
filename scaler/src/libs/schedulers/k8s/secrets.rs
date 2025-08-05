use k8s_openapi::api::core::v1::{LocalObjectReference, Secret};
use kube::api::{Api, ListParams, ObjectList, PostParams};
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::str;
use thorium::models::ScrubbedUser;
use thorium::{Conf, Error};
use tracing::{event, instrument, Level};

/// Secrets wrapper for kubernetes
pub struct Secrets {
    /// Kubernetes client
    client: kube::Client,
    /// API for non-namespaced api calls
    api: Api<Secret>,
    /// Docker registry token if a docker config exists
    registry_token: Option<String>,
    /// The api url to set
    api_url: String,
    /// Thorium config
    conf: Conf,
}

impl Secrets {
    /// Init the secrets wrapper for k8s
    ///
    /// # Arguments
    ///
    /// * `client` - kubernetes client
    /// * `conf` - Thorium Config
    /// * `context` - The name of the context to create secrets for
    pub fn new(client: &kube::Client, conf: &Conf, context: &str) -> Self {
        // get secret api
        let api: Api<Secret> = Api::all(client.clone());
        // get path to docker config
        let path = dirs::home_dir()
            .expect("Failed to detect home dir")
            .join(".docker/config.json");
        // read in registry token if it exists
        let registry_token = if path.exists() {
            Some(std::fs::read_to_string(path).expect("Failed to read in docker config"))
        } else {
            None
        };
        // get the api url to use
        let api_url = match conf.thorium.scaler.k8s.api_url(context) {
            // use this custom url
            Some(api_url) => api_url.to_owned(),
            // build the in cluster k8s service url
            None => format!(
                "http://thorium-api.{}.svc.cluster.local.",
                &conf.thorium.namespace
            ),
        };
        Secrets {
            client: client.clone(),
            api,
            registry_token,
            api_url,
            conf: conf.clone(),
        }
    }

    /// Gets [`Secret`]s in a namespace in k8s by name
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to get a secret from
    /// * `name` - The name of the secret to get
    pub async fn get(&self, ns: &str, name: &str) -> Result<Secret, kube::Error> {
        // get a namespaced client
        let api: Api<Secret> = Api::namespaced(self.client.clone(), ns);
        // get this secret by name
        api.get(name).await
    }

    /// Gets the most recently created secret with a prefix in a namespace
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix a secret should start with
    /// * `ns` - The namespace to look for this secret in
    pub async fn latest(&self, prefix: &str, ns: &str) -> Result<Option<Secret>, kube::Error> {
        // list the secrets in this namespace
        let list = self.list(ns).await?;

        // add the '-' suffix kustomize adds
        let prefix = format!("{}-", prefix);

        // get youngest secret
        let youngest: Option<Secret> = list
            .into_iter()
            // filter out any secrets without names
            .filter(|secret| secret.metadata.name.is_some())
            // filter out any whose name doesn't start with our prefix
            .filter(|secret| secret.metadata.name.as_ref().unwrap().starts_with(&prefix))
            // filter out any without creation timestamps
            .filter(|secret| secret.metadata.creation_timestamp.is_some())
            // get the youngest
            .max_by_key(|secret| secret.metadata.creation_timestamp.as_ref().unwrap().clone());
        Ok(youngest)
    }

    /// List all [`Secret`]s in a namespace in k8s
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to list secrets from
    pub async fn list(&self, ns: &str) -> Result<ObjectList<Secret>, kube::Error> {
        // build list params
        let params = ListParams::default().fields(&format!("metadata.namespace=={}", ns));
        // get list of all secrets
        self.api.list(&params).await
    }

    /// Build Thorium registry token reference
    pub fn registry_token(&self) -> Vec<LocalObjectReference> {
        if self.registry_token.is_some() {
            // create registry token if it does not exist in this group
            let name = "thorium-registry-token".to_owned();
            let image_pull_secret = LocalObjectReference { name };
            vec![impage_pull_secret]
        } else {
            Vec::default()
        }
    }

    /// Generates a secret with a single file within it
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the secret
    /// * `filename` - The name of the file within this secret
    /// * `data` - The data to store in the file
    /// * `type_` - The type of secret this is
    fn generate(
        &self,
        name: String,
        filename: String,
        data: String,
        type_: Option<String>,
    ) -> Secret {
        // build default secret
        let mut secret = Secret::default();
        secret.metadata.name = Some(name);
        secret.type_ = type_;
        // inject Thorium registry token data
        let mut map = BTreeMap::default();
        map.insert(filename, data);
        secret.string_data = Some(map);
        secret
    }

    /// Creates a secret containing a single file in a specific namespace
    ///
    /// This will not recreate a secret if it already exists.
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to inject the secret into
    /// * `name` - The name of the secret to create
    /// * `filename` - The name of the file to store in this secret
    /// * `data` - The data in this file
    /// * `type_` - The type of secret to create
    pub async fn create(
        &self,
        ns: &str,
        name: &str,
        filename: &str,
        data: &str,
        type_: Option<String>,
    ) -> Result<(), Error> {
        // build list params
        let params = ListParams::default().fields(&format!("metadata.namespace=={}", ns));
        // get list of secrets in this namespace
        let secrets = self.api.list(&params).await?;
        // check if the secret was already created
        let created = secrets
            .iter()
            .any(|secret| secret.metadata.name == Some(name.to_owned()));
        // if secret was not already created create it
        if !created {
            let spec = self.generate(name.into(), filename.into(), data.into(), type_);
            let api: Api<Secret> = Api::namespaced(self.client.clone(), ns);
            let params = PostParams::default();
            api.create(&params, &spec).await?;
        }
        Ok(())
    }

    /// Replaces a secret by name
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace that the secret to replace is in
    /// * `name` - The name of the secret to replace
    /// * `filename` - The name of the file in the secret to replace
    /// * `data` - The data to replace in the secret
    /// * `type_` - The type of this secret
    pub async fn update(
        &self,
        ns: &str,
        name: &str,
        filename: &str,
        data: &str,
        secret: Secret,
        type_: Option<String>,
    ) -> Result<(), Error> {
        // build this secret
        let mut spec = self.generate(name.into(), filename.into(), data.into(), type_);
        // get a namespaced api client
        let api: Api<Secret> = Api::namespaced(self.client.clone(), ns);
        let params = PostParams::default();
        // set the current version in our replacement object
        spec.metadata.resource_version = secret.metadata.resource_version;
        // replace this secret by name in the target namespace
        api.replace(name, &params, &spec).await?;
        Ok(())
    }

    /// Checks if a user's secret in a namespace is correct and updates it if it's not
    ///
    /// If a user's secret does not exist this will create it.
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace this token is in
    /// * `user` - The user to check against
    #[instrument(name = "k8s::Secrets::check_secret", skip(self, user), fields(user = user.username))]
    pub async fn check_secret(&self, ns: &str, user: &ScrubbedUser) -> Result<(), Error> {
        // build the right name for this secret
        let secret_name = format!("thorium-{}-keys", &user.username);
        // try to get this users keys secret
        match self.get(ns, &secret_name).await {
            // this users secret exists validate it is correct
            Ok(secret) => {
                // unwrap data if we have some
                if let Some(data) = &secret.data {
                    // make sure this secret contains keys.yml file
                    if let Some(key) = data.get("keys.yml") {
                        // decode the secret and extract the token
                        let raw = String::from_utf8_lossy(&key.0);
                        // if the token in the secret is incorrect then update it
                        if !raw.contains(&user.token) {
                            self.update_user(ns, user, secret).await?;
                            event!(Level::INFO, msg = "Updated user keys",);
                        }
                    }
                }
            }
            // an error was returned if this secret doesn't exist then create it
            // if this was a different error then raise that error
            Err(err) => {
                match &err {
                    kube::Error::Api(api_err) => {
                        // this secret doesn't exist so create it
                        if api_err.code == 404 {
                            self.setup_user(ns, user).await?;
                            event!(Level::INFO, msg = "Setup user",);
                        } else {
                            return Err(Error::from(err));
                        }
                    }
                    _ => return Err(Error::from(err)),
                }
            }
        }
        Ok(())
    }

    /// Sets up an existing namespace to to be used by Thorium
    ///
    /// # Arguments
    ///
    /// * `ns` - The name of the namespace to setup
    /// * `bans` - The ban set to add any failed namespaces too
    #[instrument(name = "k8s::Secrets::setup_namespace", skip(self, bans))]
    pub async fn setup_namespace(&self, ns: &str, bans: &mut HashSet<String>) {
        // get our registry token if we have one
        if let Some(reg_token) = &self.registry_token {
            // create the secret for this namespaces registry token
            match self
                .create(
                    ns,
                    "thorium-registry-token",
                    ".dockerconfigjson",
                    reg_token,
                    Some("kubernetes.io/dockerconfigjson".to_owned()),
                )
                .await
            {
                Ok(_) => event!(Level::INFO, msg = "Setup registry token", namespace = ns),
                Err(err) => {
                    // log that we failed to create this namespace and are banning it
                    event!(
                        Level::ERROR,
                        msg = "Failed to setup registry token",
                        namespace = ns,
                        ban = ns,
                        error = err.to_string()
                    );
                    // ban this namespace
                    bans.insert(ns.to_owned());
                }
            }
        }
    }

    /// Updates a users secret in Thorium
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to save the secret to
    /// * `user` - The user to create a secret for
    /// * `secret` - The secret that already exists in k8s we are updating
    pub async fn update_user(
        &self,
        ns: &str,
        user: &ScrubbedUser,
        secret: Secret,
    ) -> Result<(), Error> {
        // generate the auth data for this service accounts secret
        let data = format!("api: \"{}\"\ntoken: \"{}\"", self.api_url, &user.token);
        // build the name for this users secret
        let name = format!("thorium-{}-keys", &user.username);
        // update the secret for this user
        self.update(ns, &name, "keys.yml", &data, secret, None)
            .await
    }

    /// Set up a users secret in Thorium
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to save the secret to
    /// * `user` - The user to create a secret for
    pub async fn setup_user(&self, ns: &str, user: &ScrubbedUser) -> Result<(), Error> {
        let data = format!("api: \"{}\"\ntoken: \"{}\"", self.api_url, &user.token);
        // build the name for this users secret
        let name = format!("thorium-{}-keys", &user.username);
        // update the secret for this user
        self.create(ns, &name, "keys.yml", &data, None).await
    }
}

impl std::fmt::Debug for Secrets {
    /// Implement debug for our secrets client
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Secrets")
            .field("api_url", &self.api_url)
            .finish_non_exhaustive()
    }
}
