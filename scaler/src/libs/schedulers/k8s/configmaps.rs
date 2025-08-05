use k8s_openapi::api::core::v1::ConfigMap;
use kube::api::{Api, ListParams, ObjectList, PostParams};
use std::collections::{BTreeMap, HashSet};
use thorium::models::ScrubbedUser;
use thorium::Error;
use tracing::{event, instrument, Level};

/// ConfigMap wrappers for kubernetes
pub struct ConfigMaps {
    /// Kubernetes client
    client: kube::Client,
    /// API for non-namespaced api calls
    api: Api<ConfigMap>,
}

impl ConfigMaps {
    /// Init the ConfigMap wrapper for k8s
    ///
    /// # Arguments
    ///
    /// * `client` - kubernetes client
    pub fn new(client: &kube::Client) -> Self {
        // get ConfigMap api
        let api: Api<ConfigMap> = Api::all(client.clone());
        ConfigMaps {
            client: client.clone(),
            api,
        }
    }

    /// Gets the most recently created [`ConfigMap`] with a prefix in a namespace
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix a configmap should start with
    /// * `ns` - The namespace to look for this secret in
    pub async fn latest(&self, prefix: &str, ns: &str) -> Result<Option<ConfigMap>, kube::Error> {
        // list the configmaps in this namespace
        let list = self.list(ns).await?;

        // add the '-' suffix kustomize adds
        let prefix = format!("{}-", prefix);

        // get youngest configmap
        let youngest: Option<ConfigMap> = list
            .into_iter()
            // filter out any config maps without names
            .filter(|cm| cm.metadata.name.is_some())
            // filter out any whose name doesn't start with our prefix
            .filter(|cm| cm.metadata.name.as_ref().unwrap().starts_with(&prefix))
            // filter out any without creation timestamps
            .filter(|cm| cm.metadata.creation_timestamp.is_some())
            // get the youngest
            .max_by_key(|cm| cm.metadata.creation_timestamp.as_ref().unwrap().clone());
        Ok(youngest)
    }

    /// List all [`ConfigMap`]s in a namespace in k8s
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to list config maps from
    pub async fn list(&self, ns: &str) -> Result<ObjectList<ConfigMap>, kube::Error> {
        // build list params
        let params = ListParams::default().fields(&format!("metadata.namespace=={}", ns));
        // get list of all secrets
        self.api.list(&params).await
    }

    /// Generates a config map with a single file within it
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the config map
    /// * `filename` - The name of the file within this config map
    /// * `data` - The data to store in the file
    fn generate(&self, name: String, filename: String, data: String) -> ConfigMap {
        // build default secret
        let mut secret = ConfigMap::default();
        secret.metadata.name = Some(name);
        // inject Thorium registry token data
        let mut map = BTreeMap::default();
        map.insert(filename, data);
        secret.data = Some(map);
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
    pub async fn create(
        &self,
        ns: &str,
        name: &str,
        filename: &str,
        data: &str,
    ) -> Result<(), Error> {
        // build list params
        let params = ListParams::default().fields(&format!("metadata.namespace=={}", ns));
        // get list of secrets in this namespace
        let config = self.api.list(&params).await?;
        // check if this config map was already created
        let created = config
            .iter()
            .any(|secret| secret.metadata.name == Some(name.to_owned()));
        // if this config was not already created create it
        if !created {
            let spec = self.generate(name.into(), filename.into(), data.into());
            let api: Api<ConfigMap> = Api::namespaced(self.client.clone(), ns);
            let params = PostParams::default();
            api.create(&params, &spec).await?;
        }
        Ok(())
    }

    /// Set up a users passwd config map in Thorium
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to save the secret to
    /// * `user` - The user to create a secret for
    /// * `bans` - The set of banned users to add too
    /// * `span` - The span to log traces under
    #[instrument(name = "k8s::ConfigMaps::setup_passwd", skip(self, user, bans), fields(user = user.username))]
    pub async fn setup_passwd(&self, ns: &str, user: &ScrubbedUser, bans: &mut HashSet<String>) {
        // generate the passwd data for this service accounts secret
        if let Some(unix) = &user.unix {
            let passwd = format!(
                "root:x:0:0:root:/root:/bin/bash\n\
            daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n\
            bin:x:2:2:bin:/bin:/usr/sbin/nologin\n\
            sys:x:3:3:sys:/dev:/usr/sbin/nologin\n\
            sync:x:4:65534:sync:/bin:/bin/sync\n\
            games:x:5:60:games:/usr/games:/usr/sbin/nologin\n\
            man:x:6:12:man:/var/cache/man:/usr/sbin/nologin\n\
            lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin\n\
            mail:x:8:8:mail:/var/mail:/usr/sbin/nologin\n\
            news:x:9:9:news:/var/spool/news:/usr/sbin/nologin\n\
            uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin\n\
            proxy:x:13:13:proxy:/bin:/usr/sbin/nologin\n\
            www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin\n\
            backup:x:34:34:backup:/var/backups:/usr/sbin/nologin\n\
            list:x:38:38:Mailing List Manager:/var/list:/usr/sbin/nologin\n\
            irc:x:39:39:ircd:/var/run/ircd:/usr/sbin/nologin\n\
            gnats:x:41:41:Gnats Bug-Reporting System (admin):/var/lib/gnats:/usr/sbin/nologin\n\
            nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n\
            _apt:x:100:65534::/nonexistent:/usr/sbin/nologin\n\
            {name}:*:{uid}:{gid}:{name}:/home/{name}:/usr/sbin/nologin",
                name = user.username,
                uid = unix.user,
                gid = unix.group
            );
            // build the name for this users secret
            let name = format!("thorium-{}-passwd", &user.username);
            // update the secret for this user
            match self.create(ns, &name, "passwd", &passwd).await {
                // log that we setup this users /etc/passwd
                Ok(_) => event!(Level::INFO, msg = "Setup /etc/passwd",),
                Err(err) => {
                    // log that we failed to setup this users /etc/passwd
                    event!(
                        Level::ERROR,
                        msg = "Failed to setup /etc/passwd",
                        error = err.msg()
                    );
                    // ban this user
                    bans.insert(user.username.to_owned());
                }
            }
        }
    }
}

impl std::fmt::Debug for ConfigMaps {
    /// Implement debug
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigMaps").finish_non_exhaustive()
    }
}
