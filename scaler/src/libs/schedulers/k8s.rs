//! Schedules requisitions into pods on Kubernetes
//!
//! Pods in kubernetes are not job specific but instead are spawned to complete a specific type of
//! job. This means that the scheduler is no aware which job a pod will run or claim in advance.
//! This is done so we can amortize the cost of scheduling a pod across many jobs. It does however
//! cause a semantic gap where the scheduler is not aware of how far a pod is any specific job or
//! what a pod is doing at any given time.

use chrono::prelude::*;
use futures::{stream, StreamExt};
use hashbrown::HashMap;
use k8s_openapi::api::networking::v1::NetworkPolicy;
use kube::config::{KubeConfigOptions, Kubeconfig};
use std::collections::{BTreeMap, HashSet};
use std::convert::TryFrom;
use thorium::models::{ImageScaler, ScrubbedUser, SystemSettings, UserRole, WorkerDeleteMap};
use thorium::{Conf, Error, Thorium};
use tracing::{event, instrument, Level};
use uuid::Uuid;

pub mod cluster;
pub mod configmaps;
pub mod containers;
pub mod namespaces;
pub mod network_policies;
pub mod nodes;
pub mod pods;
pub mod secrets;
pub mod services;
pub mod volumes;

use cluster::Cluster;
use configmaps::ConfigMaps;
use containers::Containers;
use namespaces::Namespaces;
use network_policies::NetworkPolicies;
use nodes::Nodes;
use pods::Pods;
use secrets::Secrets;
use services::Services;
use volumes::{MountGen, Volumes};

use super::{Allocatable, AllocatableUpdate, Scheduler, Spawned, WorkerDeletion};
use crate::libs::scaler::ErrorOutKinds;
use crate::libs::{helpers, BanSets, Cache, Tasks};
use crate::{raw_entry_vec_extend, raw_entry_vec_push};

/// Get a client for our k8s clusters based off of a kube config
///
/// # Arguments
///
/// * `schedulers` - The map of schedulers this scaler can use
/// * `conf` - Thorium Config
async fn from_kubeconfig(
    schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>,
    conf: &Conf,
) -> Result<(), Error> {
    // try to load the kubeconfig from the environment
    let kube_conf = match Kubeconfig::from_env()? {
        Some(kube_conf) => kube_conf,
        None => return Err(Error::new("Failed to load k8s config")),
    };
    // get a ref to our k8s config
    let k8s_conf = &conf.thorium.scaler.k8s;
    // iterate over all contexts in this kube config and build a scheduler for them
    // ignoring any of the ignored clusters
    for context in &kube_conf.contexts {
        // check if this context should be ignored
        if k8s_conf.ignored_contexts.contains(&context.name) {
            continue;
        }
        // build the options for getting a specific clusters config
        let mut opts = KubeConfigOptions::default();
        // set the context to use
        opts.context = Some(context.name.clone());
        // get this clusters config
        let mut cluster_conf =
            kube::Config::from_custom_kubeconfig(kube_conf.clone(), &opts).await?;
        // set the tls server name if its set for this cluster
        cluster_conf.tls_server_name = k8s_conf
            .tls_server_name(&context.name)
            .map(String::to_owned);
        // disable certificate validation if requested
        cluster_conf.accept_invalid_certs = k8s_conf.accept_invalid_certs(&context.name);
        // create a client based on this config
        let client = kube::Client::try_from(cluster_conf)?;
        // get this clusters alias or use our context name
        let name = k8s_conf.cluster_name(&context.name);
        // setup k8s wrappers
        let cluster = Cluster::new(&client, conf, name, &context.name);
        let pods = Pods::new(&client, conf, name, &context.name);
        let nodes = Nodes::new(&client, conf, name, &context.name);
        let services = Services::new(&client);
        let namespaces = Namespaces::new(&client);
        let secrets = Secrets::new(&client, conf, &context.name);
        let network_policies = NetworkPolicies::new(&client, conf)?;
        let configs = ConfigMaps::new(&client);
        // build a k8s scheduler
        let k8s = K8s {
            name: name.to_owned(),
            cluster,
            pods,
            nodes,
            services,
            namespaces,
            secrets,
            network_policies,
            configs,
        };
        // add this k8s scheduler to our scheduler map
        schedulers.insert(name.to_owned(), Box::new(k8s));
    }
    Ok(())
}

/// Try to load a kubernetes client from a service account token
///
/// # Arguments
///
/// * `context_name` - The context name to use with service accounts
/// * `schedulers` - The map of schedulers this scaler can use
/// * `conf` - Thorium Config
async fn from_service_account(
    context_name: &String,
    schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>,
    conf: &Conf,
) -> Result<(), Error> {
    // create a client based on this config
    let client = kube::Client::try_default().await?;
    // get a ref to our k8s config
    let k8s_conf = &conf.thorium.scaler.k8s;
    // get this clusters alias or use our context name
    let name = k8s_conf.cluster_name(context_name);
    // setup k8s wrappers
    let cluster = Cluster::new(&client, conf, name, context_name);
    let pods = Pods::new(&client, conf, name, context_name);
    let nodes = Nodes::new(&client, conf, name, context_name);
    let services = Services::new(&client);
    let namespaces = Namespaces::new(&client);
    let secrets = Secrets::new(&client, conf, context_name);
    let network_policies = NetworkPolicies::new(&client, conf)?;
    let configs = ConfigMaps::new(&client);
    // build a k8s scheduler
    let k8s = K8s {
        name: name.to_owned(),
        cluster,
        pods,
        nodes,
        services,
        namespaces,
        secrets,
        network_policies,
        configs,
    };
    // add this k8s scheduler to our scheduler map
    schedulers.insert(name.to_owned(), Box::new(k8s));
    Ok(())
}

/// A Kubernetes scheduler for Thorium
pub struct K8s {
    /// The name of this cluster
    pub name: String,
    /// Cluster wrappers
    pub cluster: Cluster,
    /// Pod wrappers
    pub pods: Pods,
    /// Node wrappers
    #[allow(dead_code)]
    pub nodes: Nodes,
    /// services wrappers
    #[allow(dead_code)]
    pub services: Services,
    /// Namespace wrappers
    pub namespaces: Namespaces,
    /// Secrets wrappers
    pub secrets: Secrets,
    // Network policies wrappers
    pub network_policies: NetworkPolicies,
    /// configs wrappers
    pub configs: ConfigMaps,
}

impl K8s {
    /// Builds a new k8s wrapper
    ///
    /// # Arguments
    ///
    /// * `context_name` - The context name to use with service accounts
    /// * `schedulers` - The map of schedulers this scaler can use
    /// * `conf` - Thorium Config
    pub async fn new(
        context_name: &String,
        schedulers: &mut HashMap<String, Box<dyn Scheduler + Send>>,
        conf: &Conf,
    ) -> Result<(), Error> {
        // get the path to our kubeconfig if it exists
        if let Some(config_path) = std::env::var("KUBECONFIG").ok() {
            // check if a kubeconfig exists or not
            if tokio::fs::try_exists(&config_path).await? {
                // load our schedulers from our kubeconfig
                from_kubeconfig(schedulers, conf).await?;
                // return early
                return Ok(());
            }
        }
        // try to load a client from a service account
        from_service_account(context_name, schedulers, conf).await?;
        Ok(())
    }

    /// Set up a normal users groups
    ///
    /// # Arguments
    ///
    /// * `user` - The user to setup groups for
    /// * `namespaces` - The namespaces that currently exist in K8s
    /// * `checked` - The set of namespaces we have already initially setup
    #[instrument(name = "K8s::setup_user", skip_all, fields(user = user.username, namespaces_count = namespaces.len()))]
    pub async fn setup_user<'a>(
        &mut self,
        user: &'a ScrubbedUser,
        namespaces: &[String],
        checked: &mut HashSet<&'a String>,
        bans: &mut BanSets,
    ) {
        // make sure all of this users groups have namespaces
        for ns in &user.groups {
            // skip any namespaces we have already checked
            if !checked.contains(ns) {
                // create this namespace if it doesn't exist yet
                if !namespaces.contains(ns) {
                    self.namespaces.create(ns, &mut bans.groups).await;
                }
                // setup this new namespace
                self.secrets.setup_namespace(ns, &mut bans.groups).await;
                // add this namespace to our list already created namespaces
                checked.insert(ns);
            }
            // make sure this users secret is set up and correct
            if let Err(err) = self.secrets.check_secret(ns, user).await {
                // log that we failed to setup this users secret
                event!(
                    Level::ERROR,
                    msg = "Failed to setup users secret",
                    error = err.msg()
                );
                // add this user to our ban set
                bans.users.insert(user.username.clone());
            }
            // if this user has unix info then create their passwd file
            if user.unix.is_some() {
                // make sure this users passwd is set up and correct
                self.configs.setup_passwd(ns, user, &mut bans.users).await;
            }
        }
    }

    /// Set up an admins configs in all namespaces
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `user` - The user to setup groups for
    /// * `namespaces` - The namespaces that currently exist in K8s
    /// * `checked` - The set of namespaces we have already initially setup
    #[instrument(name = "K8s::setup_admin", skip_all, fields(user = user.username, namespaces_count = namespaces.len()))]
    pub async fn setup_admin<'a>(
        &mut self,
        cache: &'a Cache,
        user: &ScrubbedUser,
        namespaces: &[String],
        checked: &mut HashSet<&'a String>,
        bans: &mut BanSets,
    ) {
        // make sure all of this users groups have namespaces
        for ns in &cache.groups {
            // skip any namespaces we have already checked
            if !checked.contains(ns) {
                // create this namespace if it doesn't exist yet
                if !namespaces.contains(ns) {
                    self.namespaces.create(ns, &mut bans.groups).await;
                }
                // setup this new namespace
                self.secrets.setup_namespace(ns, &mut bans.groups).await;
                // add this namespace to our list already created namespaces
                checked.insert(ns);
            }
            // make sure this users secret is set up and correct
            if let Err(err) = self.secrets.check_secret(ns, user).await {
                // log that we failed to setup this users secret
                event!(
                    Level::ERROR,
                    msg = "Failed to setup users secret",
                    error = err.msg()
                );
                // add this user to our ban set
                bans.users.insert(user.username.clone());
            }
            // if this user has unix info then create their passwd file
            if user.unix.is_some() {
                // make sure this users passwd is set up and correct
                self.configs.setup_passwd(ns, user, &mut bans.users).await;
            }
        }
    }

    /// Setup all network policies in K8's, making sure any existing policies match
    /// those in the initial Thorium cache
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use while setting things up
    #[instrument(name = "Scheduler<K8s>::setup_network_policies", skip_all, err(Debug))]
    async fn setup_network_policies(&mut self, cache: &Cache) -> Result<(), Error> {
        // retrieve info on all Thorium network policies from K8's
        let (k8s_base_policies, k8s_policies) =
            self.get_all_thorium_network_policies(cache).await?;
        // setup base network policies, making sure everything in K8's matches our cache
        self.setup_base_network_policies(k8s_base_policies).await;
        // calculate which policies we need to delete and add
        let mut policies_to_remove: HashMap<String, Vec<String>> = HashMap::new();
        let mut policies_to_add: HashMap<String, Vec<(Uuid, NetworkPolicy)>> = HashMap::new();
        // create a cache of generated policy specs mapped by K8's name
        // to avoid generating multiple times
        let mut policy_specs_by_k8s_name: HashMap<String, NetworkPolicy> = HashMap::new();
        // see which K8's policies are in the cache and how they compare
        for (ns, k8s_policies) in k8s_policies {
            let network_policies = cache.network_policies.ids_by_group_k8s_name.get(&ns);
            let Some(network_policies) = network_policies else {
                // the cache is missing this entire namespace, so delete everything in the namespace
                policies_to_remove.insert(
                    ns,
                    k8s_policies
                        .into_iter()
                        .filter_map(|policy| policy.metadata.name)
                        .collect(),
                );
                continue;
            };
            // create a mutable copy of our K8's policies in this namespace to keep track
            // of which policies are already in K8's and which aren't
            let mut cached_policy_ids_k8s_name = network_policies.clone();
            for k8s_policy in k8s_policies {
                let Some(k8s_policy_name) = k8s_policy.metadata.name.as_ref() else {
                    // log that there's a policy in K8's that has no name;
                    // TODO: I don't think this is possible?
                    let msg = format!("A policy exists in namespace '{ns}' that has no name!'");
                    event!(Level::WARN, msg = msg);
                    continue;
                };
                // see if we have this policy cached
                if let Some(id) = cached_policy_ids_k8s_name.remove(k8s_policy_name) {
                    // now try to get the policy's info
                    if let Some(cached_policy) = cache.network_policies.policies_by_id.get(&id) {
                        // generate a policy spec to compare to (and possibly to deploy) or get a cached one
                        let (_, cached_policy_spec) = policy_specs_by_k8s_name
                            .raw_entry_mut()
                            .from_key(k8s_policy_name)
                            .or_insert(
                                k8s_policy_name.clone(),
                                NetworkPolicies::generate(
                                    cached_policy.clone(),
                                    std::iter::once(ns.clone()),
                                )
                                .remove(0)
                                .1,
                            );
                        if !network_policies::policies_equal(&k8s_policy, cached_policy_spec) {
                            // if the policies differ, we need to delete and re-add the policy to update it
                            raw_entry_vec_push!(policies_to_remove, &ns, k8s_policy_name.clone());
                            // clone our cached policy and set the correct namespace
                            let mut cached_policy_spec = cached_policy_spec.clone();
                            cached_policy_spec.metadata.namespace = Some(ns.clone());
                            raw_entry_vec_push!(policies_to_add, &ns, (id, cached_policy_spec));
                        }
                    } else {
                        // this policy's info is missing from the cache; log an error
                        let msg = format!("Network policy info for '{ns}:{k8s_policy_name}' with ID '{id}' is missing from the cache!");
                        event!(Level::ERROR, msg = msg);
                    }
                } else {
                    // this policy is not in the cache, so add it to be deleted
                    raw_entry_vec_push!(policies_to_remove, &ns, k8s_policy_name.clone());
                }
            }
            // now we should be left with only the policies that are in the cache but *not*
            // already in K8's, so make specs for each one and add them to our list
            let policy_specs_to_add =
                cached_policy_ids_k8s_name
                    .into_iter()
                    .filter_map(|(k8s_name, id)| {
                        let Some(cached_policy) = cache.network_policies.policies_by_id.get(&id)
                        else {
                            // this policy's info is missing from the cache; log an error
                            let msg = format!("Network policy info for '{ns}:{k8s_name}' with ID '{id}' is missing from the cache!");
                            event!(Level::ERROR, msg = msg);
                            // skip this policy
                            return None;
                        };
                        // generate a policy spec to compare to (and possibly to deploy) or get a cached one
                        let cached_policy_spec =
                            policy_specs_by_k8s_name.entry(k8s_name).or_insert(
                                NetworkPolicies::generate(
                                    cached_policy.clone(),
                                    std::iter::once(ns.clone()),
                                )
                                .remove(0)
                                .1,
                            );
                        // clone our spec and add the right namespace
                        let mut cached_policy_spec = cached_policy_spec.clone();
                        cached_policy_spec.metadata.namespace = Some(ns.clone());
                        Some((id, cached_policy_spec))
                    });
            // add all those specs to the list to add
            raw_entry_vec_extend!(policies_to_add, &ns, policy_specs_to_add);
        }
        // remove and add network policies
        self.remove_network_policies(policies_to_remove).await;
        self.add_network_policies(policies_to_add).await;
        Ok(())
    }

    /// Make sure base network policies are in all of our Thorium groups
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use while setting things up
    #[instrument(name = "Scheduler<K8s>::setup_base_network_policies", skip_all)]
    async fn setup_base_network_policies(
        &mut self,
        k8s_base_policies: Vec<(String, Vec<NetworkPolicy>)>,
    ) {
        // calculate which base policies we need to delete and add
        let mut delete_base_policies: HashMap<String, Vec<String>> = HashMap::new();
        let mut add_base_policies: HashMap<String, Vec<&NetworkPolicy>> = HashMap::new();
        // create a set of base policy names to keep track of which policies
        // already exist in each namespace
        let base_policies_set: HashSet<&String> = self
            .network_policies
            .cache
            .base_policy_specs
            .keys()
            .collect();
        for (ns, k8s_base_policies) in k8s_base_policies {
            let mut base_policies_set = base_policies_set.clone();
            for k8s_base_policy in k8s_base_policies {
                let Some(k8s_policy_name) = k8s_base_policy.metadata.name.as_ref() else {
                    // log that there's a base policy in K8's that has no name;
                    // TODO: I don't think this is possible?
                    let msg =
                        format!("A base policy exists in namespace '{ns}' that has no name!'");
                    event!(Level::WARN, msg = msg);
                    continue;
                };
                // try to remove this base policy from the set
                if base_policies_set.remove(k8s_policy_name) {
                    // get the cached base policy now that we know it exists in the cache
                    let cached_base_policy = self
                        .network_policies
                        .cache
                        .base_policy_specs
                        .get(k8s_policy_name)
                        .unwrap();
                    // make sure the policies are the same
                    if !network_policies::policies_equal(&k8s_base_policy, cached_base_policy) {
                        // the policies are not the same so we need to delete and re-add the base policy
                        raw_entry_vec_push!(delete_base_policies, &ns, k8s_policy_name.clone());
                        raw_entry_vec_push!(add_base_policies, &ns, cached_base_policy);
                    }
                } else {
                    // this is a dangling policy; delete it
                    raw_entry_vec_push!(delete_base_policies, &ns, k8s_policy_name.clone());
                }
            }
            // now we're left with base policies that don't already exist in the namespace,
            // so add them all to our list to be added
            let base_policies = base_policies_set.into_iter().map(|name| {
                self.network_policies
                    .cache
                    .base_policy_specs
                    .get(name)
                    .unwrap()
            });
            raw_entry_vec_extend!(add_base_policies, &ns, base_policies);
        }
        // remove base policies
        for (ns, base_policies) in delete_base_policies {
            let delete_errors = self.network_policies.delete_many(&ns, base_policies).await;
            // handle any errors
            for (policy_name, k8s_err) in delete_errors {
                match &k8s_err {
                    kube::Error::Api(kube_err) => {
                        // if we got a 404, the policy didn't exist in the first place, so we can ignore the error
                        if kube_err.code != 404 {
                            // log the error
                            let msg = format!(
                                "Failed to delete base network policy '{policy_name}' from namespace '{ns}': {k8s_err}"
                            );
                            event!(Level::ERROR, msg = msg);
                        }
                    }
                    _ => {
                        // log the error
                        let msg = format!(
                            "Failed to delete base network policy '{policy_name}' from namespace '{ns}': {k8s_err}"
                        );
                        event!(Level::ERROR, msg = msg);
                    }
                }
            }
        }
        // add base policies
        for (ns, base_policies) in add_base_policies {
            if self.deploy_base_policies(&ns, base_policies).await {
                // mark that this namespace has the base policies deployed if successful
                self.network_policies
                    .cache
                    .namespaces_with_base_policies
                    .insert(ns);
            }
        }
    }

    /// Sync any changes in our network policies since last cache reload to K8's
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use while setting things up
    #[instrument(name = "Scheduler<K8s>::sync_network_policies", skip_all, err(Debug))]
    pub async fn sync_network_policies(&mut self, cache: &Cache) -> Result<(), Error> {
        // extend our network policy cache with any important info calculated last cache reload
        self.network_policies.cache.update(cache);
        // add/remove base network policies to/from new/deleted groups
        self.sync_base_network_policies(cache).await;
        // remove network policies set to be removed
        let policies_to_remove: HashMap<String, Vec<String>> = self
            .network_policies
            .cache
            .policies_to_remove
            .drain()
            .collect();
        self.remove_network_policies(policies_to_remove).await;
        // create a map of policies to create by namespace
        let mut policy_specs_by_ns: HashMap<String, Vec<(Uuid, NetworkPolicy)>> = HashMap::new();
        for (id, add_policy) in self
            .network_policies
            .cache
            .policies_to_add
            .drain(..)
            .map(|id| (id, cache.network_policies.policies_by_id.get(&id)))
        {
            match add_policy {
                // generate the spec(s) for this policy
                Some(add_policy) => {
                    // generate policy specs for all of the policy's groups
                    let policy_specs =
                        NetworkPolicies::generate(add_policy.clone(), add_policy.groups.clone());
                    // extend our map with those specs
                    policy_specs_by_ns.extend(
                        policy_specs
                            .into_iter()
                            .map(|(ns, policy_spec)| (ns, vec![(add_policy.id, policy_spec)])),
                    );
                }
                // the policy is missing from the cache, so log a warning
                None => {
                    let msg = format!("A policy with id '{id}' was set to be deployed, but the policy is missing from the cache!");
                    event!(Level::ERROR, msg = msg);
                }
            }
        }
        self.add_network_policies(policy_specs_by_ns).await;
        Ok(())
    }

    /// Add/remove base network policies to/from new/deleted groups since last
    /// cache reload
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium to use while setting things up
    #[instrument(name = "Scheduler<K8s>::sync_base_network_policies", skip_all)]
    async fn sync_base_network_policies(&mut self, cache: &Cache) {
        // get a list of groups that don't have base network policies deployed yet
        let namespaces_missing_base_policies = cache
            .groups
            .iter()
            .filter(|group| {
                !self
                    .network_policies
                    .cache
                    .namespaces_with_base_policies
                    .contains(*group)
            })
            .collect::<Vec<&String>>();
        // add the base network policies to all groups that haven't had it added yet
        for ns in namespaces_missing_base_policies {
            if self
                .deploy_base_policies(ns, self.network_policies.cache.base_policy_specs.values())
                .await
            {
                // if the deployment was successful, add this namespace to our set of
                // namespaces base policies have been deployed to
                self.network_policies
                    .cache
                    .namespaces_with_base_policies
                    .insert(ns.clone());
            }
        }
    }

    /// Get all Thorium network policies from K8's
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of Thorium info
    #[instrument(
        name = "Scheduler<K8s>::get_all_thorium_network_policies",
        skip_all,
        err(Debug)
    )]
    async fn get_all_thorium_network_policies(
        &self,
        cache: &Cache,
    ) -> Result<K8sThoriumPolicies, Error> {
        // get any Thorium network policies from K8's in all namespaces
        let mut k8s_policies = helpers::assert_send_stream(
            stream::iter(cache.groups.iter())
                .map(|ns| async {
                    // list all policies for this namespace
                    let policies = self.network_policies.list(ns).await?;
                    Ok((ns.clone(), policies))
                })
                // list from 5 namespaces at a time
                .buffer_unordered(5),
        )
        .collect::<Vec<Result<_, kube::Error>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|(ns, policies)| {
            (
                ns,
                // filter out any non-Thorium network policies if there are any
                policies
                    .into_iter()
                    .filter(|policy| {
                        policy.metadata.labels.as_ref().is_some_and(|labels| {
                            // this is a thorium policy if it has label "thorium": "true"
                            labels.get("thorium").is_some_and(|key| key == "true")
                        })
                    })
                    .collect::<Vec<NetworkPolicy>>(),
            )
        })
        .collect::<Vec<(String, Vec<NetworkPolicy>)>>();
        // filter out only base network policies from what we grabbed from k8's
        let k8s_base_policies = k8s_policies
            .iter_mut()
            .map(|(ns, policies)| {
                (
                    ns.clone(),
                    policies
                        .extract_if(.., |policy| {
                            policy.metadata.labels.as_ref().is_some_and(|labels| {
                                // this is a base policy if it has the label "base_policy": "true"
                                labels.get("base_policy").is_some_and(|key| key == "true")
                            })
                        })
                        .collect::<Vec<NetworkPolicy>>(),
                )
            })
            .collect::<Vec<(String, Vec<NetworkPolicy>)>>();
        // return the base policies and regular policies
        Ok((k8s_base_policies, k8s_policies))
    }

    /// Attempt to remove the given network policies (mapped by namespace)
    /// from K8's; re-add any network policies that couldn't be removed
    /// to our cache to attempt to remove them next time
    ///
    /// # Arguments
    ///
    /// * `policies_to_remove` - The names of policies to remove mapped by namespace
    #[instrument(name = "Scheduler<K8s>::remove_network_policies", skip_all)]
    async fn remove_network_policies(&mut self, policies_to_remove: HashMap<String, Vec<String>>) {
        // create a map of policy names and their errors to keep track of
        // which policies failed to be deleted, mapped by namespace
        let mut errors: HashMap<String, Vec<(String, kube::Error)>> = HashMap::new();
        for (ns, policy_names) in policies_to_remove {
            let delete_errors = self.network_policies.delete_many(&ns, policy_names).await;
            errors.insert(ns, delete_errors);
        }
        for (ns, errors) in errors {
            for (policy_name, k8s_err) in errors {
                match &k8s_err {
                    kube::Error::Api(kube_err) => {
                        // if we got a 404, the policy was already deleted, so we can ignore the error
                        if kube_err.code != 404 {
                            // log the error
                            let msg = format!(
                                "Failed to delete network policy '{policy_name}': {k8s_err}"
                            );
                            event!(Level::ERROR, msg = msg);
                            // add the policy back to our list of policies to remove to try again next time
                            raw_entry_vec_push!(
                                self.network_policies.cache.policies_to_remove,
                                &ns,
                                policy_name
                            );
                        }
                    }
                    _ => {
                        // log the error
                        let msg =
                            format!("Failed to delete network policy '{policy_name}': {k8s_err}");
                        event!(Level::ERROR, msg = msg);
                        // add the policy back to our list of policies to remove to try again next time
                        raw_entry_vec_push!(
                            self.network_policies.cache.policies_to_remove,
                            &ns,
                            policy_name
                        );
                    }
                }
            }
        }
    }

    /// Attempt to add the given network policies (mapped by namespace)
    /// from K8's; re-add any network policies that couldn't be removed
    /// to our cache to attempt to add them next time
    ///
    /// # Arguments
    ///
    /// * `policies_to_add` - The names of policies to add mapped by namespace
    #[instrument(name = "Scheduler<K8s>::add_network_policies", skip_all)]
    async fn add_network_policies(
        &mut self,
        policies_to_add: HashMap<String, Vec<(Uuid, NetworkPolicy)>>,
    ) {
        // create a list of policy IDs, names, and their errors
        // to keep track of which policies failed to be created
        let mut errors: Vec<(Uuid, String, kube::Error)> = Vec::new();
        // deploy the network policies and save any errors
        for (ns, policies) in policies_to_add {
            self.network_policies
                .deploy(&ns, policies, &mut errors)
                .await;
        }
        // handle any errors
        for (policy_id, policy_name, k8s_err) in errors {
            match &k8s_err {
                kube::Error::Api(api_err) => {
                    // if the policy already exists (we got a 409 CONFLICT), ignore the error
                    if api_err.code != 409 {
                        // log the error
                        let msg =
                            format!("Failed to deploy network policy '{policy_name}': {k8s_err}");
                        event!(Level::ERROR, msg = msg);
                        // add the policy back to our list of policies to add to try again next time
                        self.network_policies.cache.policies_to_add.push(policy_id);
                    }
                }
                _ => {
                    // log the error
                    let msg = format!("Failed to deploy network policy '{policy_name}': {k8s_err}");
                    event!(Level::ERROR, msg = msg);
                    // add the policy back to our list of policies to add to try again next time
                    self.network_policies.cache.policies_to_add.push(policy_id);
                }
            }
        }
    }

    /// Deploy an iterator of network policies to K8's at the given namespace
    ///
    /// Returns true if the deployment had no errors and false if an error
    /// occurred
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to deploy to
    /// * `base_policies` - The base policies to deploy
    #[instrument(name = "Scheduler<K8s>::deploy_base_policies", skip_all)]
    async fn deploy_base_policies<'a, I>(&'a self, ns: &str, base_policies: I) -> bool
    where
        I: IntoIterator<Item = &'a NetworkPolicy> + Send,
        <I as IntoIterator>::IntoIter: std::marker::Send,
    {
        // track whether or not a deploy error occurred for this group
        let mut no_errors = true;
        // deploy all base network policies to the group
        let deploy_errs = self.network_policies.deploy_base(ns, base_policies).await;
        // handle any errors
        for (policy_name, k8s_err) in deploy_errs {
            match &k8s_err {
                kube::Error::Api(api_err) => {
                    // if the base policy already exists (we got a 409 CONFLICT), ignore the error
                    if api_err.code != 409 {
                        // log the error
                        let msg =
                                        format!("Failed to deploy base network policy '{policy_name}' to namespace '{ns}': {k8s_err}");
                        event!(Level::ERROR, msg = msg);
                        // mark that a deploy error occurred for this group
                        no_errors = false;
                    }
                }
                _ => {
                    // log the error
                    let msg = format!(
                                    "Failed to deploy base network policy '{policy_name}' to namespace '{ns}': {k8s_err}"
                                );
                    event!(Level::ERROR, msg = msg);
                    // mark that a deploy error occurred for this group
                    no_errors = false;
                }
            }
        }
        no_errors
    }
}

/// All of the info on Thorium network policies from K8's, including base policies
/// and regular policies, organized by namespace
type K8sThoriumPolicies = (
    Vec<(String, Vec<NetworkPolicy>)>,
    Vec<(String, Vec<NetworkPolicy>)>,
);

#[async_trait::async_trait]
impl Scheduler for K8s {
    /// Determine how long to wait before running a task again
    ///
    /// # Arguments
    ///
    /// * `task` - The task we want to run again
    fn task_delay(&self, task: &Tasks) -> i64 {
        // get how long to wait before executing this task again
        match task {
            Tasks::ZombieJobs => 30,
            Tasks::LdapSync => 600,
            Tasks::CacheReload => 600,
            Tasks::Resources => 120,
            Tasks::UpdateRuntimes => 300,
            Tasks::Cleanup => 25,
            Tasks::DecreaseFairShare => 600,
        }
    }

    /// Get the resources available in Kubernetes
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `resources` - The resources to update
    /// * `settings` - The current Thorium system settings
    #[instrument(name = "Scheduler<K8s>::resources_available", skip_all, fields(cluster = &self.name), err(Debug))]
    async fn resources_available(
        &mut self,
        thorium: &Thorium,
        _settings: &SystemSettings,
    ) -> Result<AllocatableUpdate, Error> {
        self.cluster.resources_available(thorium).await
    }

    /// Set up the K8's cluster before scheduling any jobs, ensuring the
    /// cluster's state is equivalent to the info in Thorium
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the cluster we are setting up
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    #[instrument(name = "Scheduler<K8s>::setup", skip(self, cache, bans), err(Debug))]
    async fn setup(&mut self, name: &str, cache: &Cache, bans: &mut BanSets) -> Result<(), Error> {
        // get a list of all namespaces that exist in k8s
        let namespaces = self
            .namespaces
            .list()
            .await?
            .into_iter()
            .filter_map(|ns| ns.metadata.name)
            .collect::<Vec<String>>();
        // track all namespaces we have created/checked in this setup
        let mut checked = HashSet::with_capacity(10);
        // crawl over all users and set them up in all namespaces/groups
        // and track any groups or users we need to ban due to setup problems
        for user in cache.users.values() {
            // setup this users groups
            self.setup_user(user, &namespaces, &mut checked, bans).await;
            // check if this user is an admin
            if user.role == UserRole::Admin {
                // setup this user to run jobs in all possible groups
                self.setup_admin(cache, user, &namespaces, &mut checked, bans)
                    .await;
            }
        }
        // setup network policies, making sure everything in K8's matches our cache
        self.setup_network_policies(cache).await?;
        Ok(())
    }

    /// Sync the K8's cluster to any updates in the cache since last reload
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the cluster we are setting up
    /// * `cache` - A cache of info from Thorium to use while setting things up
    /// * `bans` - The users and groups to ban due to setup errors
    #[instrument(
        name = "Scheduler<K8s>::sync_to_new_cache",
        skip(self, cache, bans),
        err(Debug)
    )]
    async fn sync_to_new_cache(
        &mut self,
        name: &str,
        cache: &Cache,
        bans: &mut BanSets,
    ) -> Result<(), Error> {
        // get a list of all namespaces that exist in k8s
        let namespaces = self
            .namespaces
            .list()
            .await?
            .into_iter()
            .filter_map(|ns| ns.metadata.name)
            .collect::<Vec<String>>();
        // track all namespaces we have created/checked in this setup
        let mut checked = HashSet::with_capacity(10);
        // crawl over all users and set them up in all namespaces/groups
        // and track any groups or users we need to ban due to setup problems
        for user in cache.users.values() {
            // setup this users groups
            self.setup_user(user, &namespaces, &mut checked, bans).await;
            // check if this user is an admin
            if user.role == UserRole::Admin {
                // setup this user to run jobs in all possible groups
                self.setup_admin(cache, user, &namespaces, &mut checked, bans)
                    .await;
            }
        }
        // sync network policies to any changes in the cache since last reload
        self.sync_network_policies(cache).await?;
        Ok(())
    }

    /// Allow the Thorium K8s scheduler to scale pods up and down
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium
    /// * `reqs` - The requisitions to scale or downscale
    async fn spawn(
        &mut self,
        cache: &Cache,
        spawn_map: &BTreeMap<DateTime<Utc>, Vec<Spawned>>,
    ) -> HashMap<String, Error> {
        // Track any failed scaling attempts
        let mut errors = HashMap::default();
        // build a map to store the new pod specs by namespace in
        let mut scales: HashMap<&String, Vec<_>> = HashMap::with_capacity(5);
        for spawns in spawn_map.values() {
            // crawl over our deadline groups
            for spawn in spawns {
                // generate a pod for this spawn
                match self.pods.generate(cache, spawn).await {
                    Ok(pod) => {
                        // get an entry to this pods namespaces spawn list
                        let entry = scales.entry(&spawn.req.group).or_default();
                        // add this pod to this namespaces spawn list
                        entry.push(pod);
                    }
                    // some error occurred during pod generation
                    Err(err) => {
                        errors.insert(spawn.name.clone(), err);
                    }
                }
            }
        }
        // crawl over our namespace sorted list of pods
        for (namespace, pods) in scales {
            // spawn our pods for this namespace in bulk
            self.pods.deploy(namespace, pods, &mut errors).await;
        }
        errors
    }

    /// Allow the Thorium K8s scheduler to scale pods down
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `cache` - A cache of info from Thorium
    /// * `scaledowns` - The requisitions to delete
    #[instrument(name = "Scheduler<K8s>::delete", skip_all)]
    async fn delete(
        &mut self,
        thorium: &Thorium,
        _cache: &Cache,
        scaledowns: Vec<Spawned>,
    ) -> Vec<WorkerDeletion> {
        // Track the results of our deletes
        let mut results = Vec::with_capacity(scaledowns.len());
        // build a map to store the downscales sorted by namespace
        let mut downscales: HashMap<String, Vec<Spawned>> = HashMap::with_capacity(5);
        // crawl over each nodes downscales
        for spawned in scaledowns {
            // get an entry to this namespaces pod list
            let entry = downscales.entry(spawned.req.group.clone()).or_default();
            // add this pod to this namespaces list of scale downs
            entry.push(spawned);
        }
        // scale down the requested pods by namespace
        for (namespace, spawns) in downscales {
            // Ask k8s to start deleting the requested pods if we have any
            if !spawns.is_empty() {
                self.pods
                    .delete_many(&namespace, &spawns, &mut results)
                    .await;
            }
            // if we have any pending deletes then validate they were deleted
            if !spawns.is_empty() || !self.pods.pending_deletes.is_empty() {
                // Validate what pods were deleted
                self.pods
                    .validate_deletes(&namespace, spawns, &mut results)
                    .await;
            }
        }
        // build the map of workers we want to delete
        let mut deletes = WorkerDeleteMap::with_capacity(results.len());
        // add all of the workers that we did delete
        for delete_res in &results {
            // if this deleted worked then add it to our delete map
            if let WorkerDeletion::Deleted(spawn) = delete_res {
                // add this deleted worked
                deletes.add_mut(&spawn.name);
            }
        }
        // delete the workers we could delete from K8s
        if let Err(error) = thorium
            .system
            .delete_workers(ImageScaler::K8s, &deletes)
            .await
        {
            // log that our deletes failed
            event!(Level::ERROR, error = error.to_string());
        }
        results
    }

    /// Clears out any failed or terminal resources in specified groups and return their names.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A client for the Thorium api
    /// * `allocatable` - The currently allocatable resources by this scaler
    /// * `groups` - The groups to clear failing resources from
    /// * `failed` - A set of failed workers to add too
    /// * `terminal` - A set of terminal workers to add too
    /// * `error_out` - The pods whose workers we should fail out instead of just resetting
    /// * `span` - The span to log traces under
    #[instrument(name = "Scheduler<K8s>::clear_terminal", skip_all)]
    async fn clear_terminal(
        &mut self,
        _thorium: &Thorium,
        _allocatable: &Allocatable,
        groups: &HashSet<String>,
        failed: &mut HashSet<String>,
        terminal: &mut HashSet<String>,
        error_out: &mut HashSet<ErrorOutKinds>,
    ) -> Result<(), Error> {
        // crawl all pods in each group and remove any dead or failing workers
        self.pods
            .clear_failing(groups, failed, terminal, error_out)
            .await?;
        Ok(())
    }
}

impl std::fmt::Debug for K8s {
    /// Implement debug for the kubernetes scheduler
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("K8s")
            .field("Name", &self.name)
            .finish_non_exhaustive()
    }
}
