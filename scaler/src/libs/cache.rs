//! A cache of data in Thorium

use chrono::prelude::*;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use thorium::conf::{BaseNetworkPolicy, Conf};
use thorium::models::{
    Image, ImageScaler, NetworkPolicy, NetworkPolicyListOpts, ScrubbedUser, SystemSettings,
};
use thorium::{Error, Keys, Thorium};
use tracing::{event, span, Level, Span};
use uuid::Uuid;

use crate::{raw_entry_map_insert, raw_entry_vec_push};

use super::{tasks::TaskResult, DockerInfo};

/// Hashmap of vectors of image info by group
pub type ImageInfoCache = HashMap<String, HashMap<String, Image>>;
/// Hashmap of vectors of docker info by group
pub type DockerInfoCache = HashMap<String, HashMap<String, DockerInfo>>;

/// Reads in the Thorium keys and refreshes the client if its needed
///
/// # Arguments
///
/// * `thorium` - The Thorium client to refresh
/// * `keys` - The keys to use when refreshing our client
/// * `span` - The span to log traces under
async fn refresh_client(
    thorium: &mut Arc<Thorium>,
    keys: &String,
    span: &Span,
) -> Result<(), Error> {
    // check to see if our Thorium client token will expire in the next week and refresh it early
    // we refresh it a week early since the Thorium client is shared and we may not get a
    // mutable reference on the first try
    if thorium.expires < Some(Utc::now() - chrono::Duration::weeks(1)) {
        // start our Thorium auth refresh span
        let span = span!(parent: span, Level::INFO, "Thorium Client Refresh");
        // reload auth keys
        let keys = Keys::new(keys).expect("Failed to load auth keys");
        // panic if username is not known
        assert!(
            keys.username.is_some() && keys.password.is_some(),
            "keys.yml must contain username/password"
        );
        // try to get a mutable reference to our Thorium client
        if let Some(thorium) = Arc::get_mut(thorium) {
            thorium
                .refresh(keys.username.unwrap(), keys.password.unwrap())
                .await?;
        } else {
            // keep trying to refresh our token until it expires then panic
            if thorium.expires > Some(Utc::now()) {
                event!(
                    parent: &span,
                    Level::WARN,
                    msg = "Failed to get mutable ref to client"
                );
            } else {
                event!(
                    parent: &span,
                    Level::ERROR,
                    msg = "Failed to refresh expired token"
                );
            }
        }
    }
    Ok(())
}

/// A cache of network policy info in Thorium
#[derive(Debug, Default)]
pub struct NetworkPolicyInfoCache {
    // Hash map of network policy info by id
    pub policies_by_id: HashMap<Uuid, NetworkPolicy>,
    /// Hash map of network policies' IDs by their group and name
    ids_by_group_name: HashMap<String, HashMap<String, Uuid>>,
    /// Hash map of network policies' IDs by their group and K8's name
    pub ids_by_group_k8s_name: HashMap<String, HashMap<String, Uuid>>,
    /// Hash map of network policy's IDs that should always be applied by group
    forced_ids_by_group: HashMap<String, Vec<Uuid>>,
    /// ID's of policies that were added this cache reload
    pub policies_added: Vec<Uuid>,
    /// K8's names of policies that were removed this cache reload,
    /// mapped by group/namespace
    pub policies_removed: HashMap<String, Vec<String>>,
}

/// A local cache of data in the Thorium DB
///
/// This cache is reset based on both time and events in the API itself. It
/// will reset every 10 minutes by default or whenever an update is made in
/// the API.
pub struct Cache {
    /// The Thorium config
    conf: Conf,
    /// A client for Thorium
    thorium: Arc<Thorium>,
    /// System settings for Thorium
    pub settings: SystemSettings,
    /// A map of users in Thorium
    pub users: HashMap<String, ScrubbedUser>,
    /// A set of groups in Thorium
    pub groups: HashSet<String>,
    /// A map of image info in Thorium
    pub images: ImageInfoCache,
    /// A cache of network policy info in Thorium
    pub network_policies: NetworkPolicyInfoCache,
    /// A map of docker image info for our images
    pub docker: DockerInfoCache,
    // The timestamp this cache will be invalidated and reloaded at
    pub expires: DateTime<Utc>,
    /// The path to reload our auth keys from
    pub auth_keys: String,
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache")
            .field("settings", &self.settings)
            .field("users", &self.users)
            .field("groups", &self.groups)
            .field("images", &self.images)
            .field("network_policies", &self.network_policies)
            .field("docker", &self.docker)
            .field("expires", &self.expires)
            .finish()
    }
}

impl Cache {
    /// Creates a new cache object
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `conf` - The Thorium config
    /// * `auth_keys` - The path to reload our auth keys from
    /// * `span` - The span to log traces under
    pub async fn new(
        thorium: Arc<Thorium>,
        conf: Conf,
        auth_keys: String,
        scaler: ImageScaler,
        span: &Span,
    ) -> Result<Cache, Error> {
        // create a new empty Cache instance
        let mut cache = Cache {
            conf,
            thorium,
            settings: SystemSettings::default(),
            users: HashMap::new(),
            groups: HashSet::new(),
            images: HashMap::new(),
            network_policies: NetworkPolicyInfoCache::default(),
            docker: HashMap::new(),
            expires: Utc::now(),
            auth_keys,
        };
        // load this cache with data
        cache.load_data(scaler, span).await?;
        Ok(cache)
    }

    /// Reload data on all users in Thorium
    async fn load_users(&mut self, span: &Span) -> Result<(), Error> {
        // start our user info reload span
        let span = span!(parent: span, Level::INFO, "Reloading User Info");
        // get details on all users
        let users = self.thorium.users.list_details().await?;
        // log how many users we got info on
        event!(parent: &span, Level::INFO, users = users.len());
        for user in users {
            // skip the Thorium system user
            if user.username != "thorium" {
                self.users.insert(user.username.clone(), user);
            }
        }
        Ok(())
    }

    /// Get Thorium image info on all the images in all groups we have users for
    async fn load_images(&mut self, scaler: ImageScaler, span: &Span) -> Result<(), Error> {
        // start our image info reload span
        let span = span!(parent: span, Level::INFO, "Reloading Image Info");
        // build a set of all groups we have users for
        for user in self.users.values() {
            for name in &user.groups {
                // skip any system groups
                if !["system", "thorium"].contains(&&name[..]) {
                    self.groups.insert(name.to_owned());
                }
            }
        }
        // crawl over these groups and get all of their image info
        for group in &self.groups {
            // create a cursor object for images in this group
            let mut cursor = self.thorium.images.list(group).details().exec().await?;
            // get a mutable reference to this groups entry into the image/docker map
            let image_map = self.images.entry(group.clone()).or_default();
            let docker_map = self.docker.entry(group.clone()).or_default();
            // loop over our cursor until its exhausted
            loop {
                // insert all of our images into this groups image map
                let images = cursor
                    .details
                    .par_drain(..)
                    // skip any images without docker urls set if this is a K8s scaler
                    .filter(|image| image.image.is_some() || scaler != ImageScaler::K8s)
                    .filter_map(|image| {
                        // skip any images that have bans
                        if !image.bans.is_empty() {
                            return None;
                        }
                        // skip any images that don't match our scaler
                        if image.scaler == scaler {
                            // if we are scaling k8s then get docker info
                            if scaler == ImageScaler::K8s {
                                match DockerInfo::inspect(
                                    &self.conf.thorium.scaler.crane,
                                    image.image.as_ref().unwrap(),
                                    &span,
                                ) {
                                    // docker image info retrieved successfully
                                    Ok(docker) => Some((Some(docker), image)),
                                    Err(e) => {
                                        event!(
                                            parent: &span,
                                            Level::WARN,
                                            msg = "Failed to inspect image",
                                            group = group,
                                            image = image.name,
                                            error = e.msg(),
                                        );
                                        None
                                    }
                                }
                            } else {
                                Some((None, image))
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<(Option<DockerInfo>, Image)>>();
                // crawl over our built images and insert them
                for (docker, image) in images {
                    if let Some(docker) = docker {
                        docker_map.insert(image.name.clone(), docker);
                    }
                    image_map.insert(image.name.clone(), image);
                }
                // if this cursor is exhausted that break
                if cursor.exhausted {
                    break;
                }
                // otherwise get the next page of the cursor
                cursor.next().await?;
            }
        }
        Ok(())
    }

    /// Get Thorium network policy info
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler's type
    /// * `span` - The span to log traces under
    async fn load_network_policies(
        &mut self,
        scaler: ImageScaler,
        span: &Span,
    ) -> Result<(), Error> {
        // start our user info reload span
        let span = span!(parent: span, Level::INFO, "Reloading Network Policy Info");
        if scaler != ImageScaler::K8s {
            // network policies only apply to the K8's scaler, so return early if we're not K8's
            return Ok(());
        }
        // create the maps containing our data
        let mut policies_by_id: HashMap<Uuid, NetworkPolicy> = HashMap::new();
        let mut ids_by_group_name: HashMap<String, HashMap<String, Uuid>> = HashMap::new();
        let mut ids_by_group_k8s_name: HashMap<String, HashMap<String, Uuid>> = HashMap::new();
        let mut forced_ids_by_group: HashMap<String, Vec<Uuid>> = HashMap::new();
        // list all of the network policies in all Thorium groups that we have
        let mut cursor = self
            .thorium
            .network_policies
            .list_details(&NetworkPolicyListOpts::default().groups(self.groups.clone()))
            .await?;
        loop {
            for policy in cursor.data.drain(..) {
                // add all of the policy's groups to the IDs by group/name map
                for group in &policy.groups {
                    raw_entry_map_insert!(ids_by_group_name, group, policy.name.clone(), policy.id);
                    raw_entry_map_insert!(
                        ids_by_group_k8s_name,
                        group,
                        policy.k8s_name.clone(),
                        policy.id
                    );
                }
                if policy.forced_policy {
                    // add the policy to forced cache if it should always be applied in its group(s)
                    for group in &policy.groups {
                        raw_entry_vec_push!(forced_ids_by_group, group, policy.id);
                    }
                }
                // add the policy to the policies by id map
                policies_by_id.insert(policy.id, policy);
            }
            if cursor.exhausted() {
                break;
            }
            cursor.refill().await?;
        }
        // log how many network policies we got info on
        event!(parent: &span, Level::INFO, network_policies = policies_by_id.len());
        // update our cache
        self.network_policies.policies_by_id = policies_by_id;
        self.network_policies.ids_by_group_name = ids_by_group_name;
        self.network_policies.ids_by_group_k8s_name = ids_by_group_k8s_name;
        self.network_policies.forced_ids_by_group = forced_ids_by_group;
        Ok(())
    }

    /// Load data into the cache
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler's type
    /// * `span` - The span to log traces under
    pub async fn load_data(&mut self, scaler: ImageScaler, span: &Span) -> Result<(), Error> {
        // reload the system settings
        self.settings = self.thorium.system.get_settings().await?;
        // reload all cache data
        self.load_users(span).await?;
        // now that we have users load all of the images for their groups
        self.load_images(scaler, span).await?;
        // load all the network policies from Thorium
        self.load_network_policies(scaler, span).await?;
        // set the new cache expiration timestamp
        self.expires =
            Utc::now() + chrono::Duration::seconds(self.conf.thorium.scaler.cache_lifetime as i64);
        Ok(())
    }

    /// Generates a new cache and loads data into it
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client to refresh if its close to expiring
    /// * `conf` - A Thorium config
    /// * `auth_keys` - The path to our Thorium auth info
    /// * `scaler` - The scaler we are using
    pub async fn refresh(
        stale_cache: Arc<Self>,
        mut thorium: Arc<Thorium>,
        scaler: ImageScaler,
    ) -> Result<TaskResult, Error> {
        // start our cache refresh span
        let span = span!(Level::INFO, "Refreshing Cache");
        // clone data from our stale cache that needs to be carried over
        let conf = stale_cache.conf.clone();
        let auth_keys = stale_cache.auth_keys.clone();
        // refresh our Thorium client if its about to expire
        refresh_client(&mut thorium, &auth_keys, &span).await?;
        // build a new cache
        let mut cache = Cache::new(thorium, conf, auth_keys, scaler, &span).await?;
        // calculate any important differences between the stale cache and the new cache
        // (what's been added, deleted, updated, etc.)
        cache.diff(&stale_cache);
        Ok(TaskResult::Cache(cache))
    }

    /// Computes differences between this cache and a stale cache and
    /// saves those differences in this cache
    ///
    /// # Arguments
    ///
    /// * `stale_cache` - The stale cache we're comparing the cache to
    fn diff(&mut self, stale_cache: &Self) {
        self.diff_network_policies(&stale_cache.network_policies);
    }

    /// Computes the network policies that have been added, removed, or updated
    /// compared to a stale cache and saves that info in this cache
    ///
    /// # Arguments
    ///
    /// * `stale_cache` - The stale cache of network policy info we're comparing
    ///                   the cache to
    fn diff_network_policies(&mut self, stale_cache: &NetworkPolicyInfoCache) {
        // check each policy in the new cache
        for (policy_id, policy) in &self.network_policies.policies_by_id {
            // check if this is a new/updated policy based on the contents of the old cache
            match stale_cache.policies_by_id.get(policy_id) {
                Some(old_policy) => {
                    if policy.needs_k8s_update(old_policy) {
                        // this policy was updated, so we need to delete and re-add it to K8's
                        for group in &old_policy.groups {
                            raw_entry_vec_push!(
                                self.network_policies.policies_removed,
                                group,
                                old_policy.k8s_name.clone()
                            );
                        }
                        self.network_policies.policies_added.push(policy.id);
                    }
                }
                // this is a completely new policy, so we need to add it to K8's
                None => {
                    self.network_policies.policies_added.push(policy.id);
                }
            }
        }
        // check each policy in the old cache
        for (id, policy) in &stale_cache.policies_by_id {
            // check if this policy no longer exists based on the contents of the new cache
            if !self.network_policies.policies_by_id.contains_key(id) {
                // this policy no longer exists, so add it to be deleted from all of its groups
                for group in &policy.groups {
                    raw_entry_vec_push!(
                        self.network_policies.policies_removed,
                        group,
                        policy.k8s_name.clone()
                    );
                }
            }
        }
    }

    /// Get a specific image if it exists
    pub fn get_image(&self, group: &str, name: &str, span: &Span) -> Option<&Image> {
        // get this groups images
        match self.images.get(group) {
            Some(image_map) => {
                // get the correct image from this groups images
                match image_map.get(name) {
                    Some(image) => Some(image),
                    None => {
                        // this image is not in our cache so log an error
                        // build our error message
                        let msg = format!(
                            "{group}:{name} not in image level of image cache. Maybe it's banned?"
                        );
                        event!(parent: span, Level::ERROR, msg=msg);
                        None
                    }
                }
            }
            None => {
                // this group is not in our cache so log an error
                // build our error message
                let msg = format!("{group}:{name} not in group level of image cache");
                event!(parent: span, Level::ERROR, msg=msg);
                None
            }
        }
    }

    /// Return the base network policies defined in the Thorium config
    pub fn conf_base_network_policies(&self) -> &[BaseNetworkPolicy] {
        &self.conf.thorium.base_network_policies
    }

    /// Attempt to retrieve a network policy in the given group with the given name
    ///
    /// # Arguments
    ///
    /// * `group` - The group the network policy should be in
    /// * `name` - The name of the network policy
    pub fn get_network_policy(
        &self,
        group: &str,
        policy_name: &str,
    ) -> Result<&NetworkPolicy, Error> {
        // get the policy's id from our map
        let policy_id = self
            .network_policies
            .ids_by_group_name
            .get(group)
            .ok_or(Error::new(format!(
                "{group}:{policy_name} not in group level of network policy cache"
            )))?
            .get(policy_name)
            .ok_or(Error::new(format!(
                "{group}:{policy_name} not in network policy level of network policy cache"
            )))?;
        // return the policy itself or an error
        self.network_policies
            .policies_by_id
            .get(policy_id)
            .ok_or(Error::new(format!(
                "policy with id '{policy_id}' not in network policy cache"
            )))
    }

    /// Get the list of policies that should always be applied in the given group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to get the policies for
    pub fn forced_network_policies(
        &self,
        group: &str,
    ) -> Result<Option<Vec<&NetworkPolicy>>, Error> {
        self.network_policies
            .forced_ids_by_group
            .get(group)
            .map(|ids| {
                ids.iter()
                    .map(|id| {
                        self.network_policies
                            .policies_by_id
                            .get(id)
                            .ok_or(Error::new(format!(
                                "forced network policy with id '{id}' not in network policy cache"
                            )))
                    })
                    .collect::<Vec<Result<&NetworkPolicy, Error>>>()
                    .into_iter()
                    .collect()
            })
            .transpose()
    }
}
