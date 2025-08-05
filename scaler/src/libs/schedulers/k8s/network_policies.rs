use futures::stream::{self, StreamExt};
use hashbrown::HashMap;
use k8s_openapi::api::networking::v1::{
    NetworkPolicy, NetworkPolicyEgressRule, NetworkPolicyIngressRule, NetworkPolicySpec,
};
use kube::api::{Api, DeleteParams, ListParams, ObjectList, PostParams};
use serde_json::json;
use std::collections::HashSet;
use thorium::{conf::BaseNetworkPolicy, models::NetworkPolicyRule};
use uuid::Uuid;

use crate::{
    libs::{helpers, Cache},
    raw_entry_vec_extend, same,
};

/// Wrapper for network policies api routes in k8s
pub struct NetworkPolicies {
    /// Client to use for creating namespaced clients
    client: kube::Client,
    /// A cache of information related to network policies that needs to be
    /// retained between runs
    pub cache: NetworkPolicyCache,
}

impl NetworkPolicies {
    /// Build new wrapper for k8s functions regarding network policies
    ///
    /// # Arguments
    ///
    /// * `client` - Kubernetes client
    /// * `conf` - Thorium Config
    pub fn new(client: &kube::Client, conf: &thorium::Conf) -> Result<Self, thorium::Error> {
        // get client for creating namespaced clients with
        let client = client.clone();
        // attempt to generate K8's specs for each base policy defined in the config
        let base_policy_specs = conf.thorium.base_network_policies
            .iter()
            .map(|base_policy| {
                let base_policy_spec = Self::generate_base_policy(base_policy)?;
                // return the name as well to easily find the base policy later
                Ok((base_policy.name.clone(), base_policy_spec))
            })
            .collect::<Result<HashMap<String, NetworkPolicy>, thorium::Error>>()
            .map_err(|err| {
                thorium::Error::new(format!("One or more of the base network policies in the Thorium config is invalid! {err}"))
            })?;
        let cache = NetworkPolicyCache::default().base_policy_specs(base_policy_specs);
        Ok(Self { client, cache })
    }

    /// List [`NetworkPolicy`]'s in a namespace in k8s
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to list network policies from
    pub async fn list(&self, ns: &str) -> Result<ObjectList<NetworkPolicy>, kube::Error> {
        // get a namespaced client
        let api: Api<NetworkPolicy> = Api::namespaced(self.client.clone(), ns);
        // use default list params
        let list_params = ListParams::default();
        // list the network policies for this namespace
        api.list(&list_params).await
    }

    /// Generate a network policy spec from a Thorium network policy for each of the
    /// given namespaces
    ///
    /// Returns a list of tuples containing the namespace and the corresponding
    /// spec
    ///
    /// # Arguments
    ///
    /// * `thorium_policy` - The Thorium network policy to generate the spec from
    /// * `namespaces` - The namespace to generate specs for
    pub fn generate<I, T>(
        thorium_policy: thorium::models::NetworkPolicy,
        namespaces: I,
    ) -> Vec<(String, NetworkPolicy)>
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        // build network policy spec(s)
        let raw = json!({
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                // refrain from including a namespace;
                // we'll copy the spec and add a namespace for each group the policy is in
                //"namespace":
                "name": &thorium_policy.k8s_name,
                "labels": {
                    "thorium": "true",
                }
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        // match thorium-spawned pods that are labeled with this policy
                        "thorium": "true",
                        &thorium_policy.k8s_name: "true"
                    }
                },
            }
        });
        // cast this json into a k8's network policy;
        // unwrap the result because we're sure that it's valid
        let mut network_policy: NetworkPolicy = serde_json::from_value(raw).unwrap();
        // get a mutable reference to the spec
        let netpol_spec = network_policy
            .spec
            .get_or_insert(NetworkPolicySpec::default());
        // define the policy types we're applying to (ingress/egress)
        // based on the presence/absence of rules
        netpol_spec.policy_types = match (&thorium_policy.ingress, &thorium_policy.egress) {
            (None, None) => None,
            (None, Some(_)) => Some(vec!["Egress".to_string()]),
            (Some(_), None) => Some(vec!["Ingress".to_string()]),
            (Some(_), Some(_)) => Some(vec!["Ingress".to_string(), "Egress".to_string()]),
        };
        // overlay the Thorium network policy rules on top of the K8's network policy
        netpol_spec.ingress = thorium_policy
            .ingress
            .map(|mut ingress| ingress.drain(..).map(Into::into).collect());
        netpol_spec.egress = thorium_policy
            .egress
            .map(|mut egress| egress.drain(..).map(Into::into).collect());
        // return a policy spec for each given namespace
        namespaces
            .into_iter()
            .map(|ns| {
                let ns = ns.into();
                let mut policy_spec = network_policy.clone();
                policy_spec.metadata.namespace = Some(ns.clone());
                (ns, policy_spec)
            })
            .collect()
    }

    /// Attempt to generate a policy spec from the given Thorium [`BaseNetworkPolicy`]
    ///
    /// # Arguments
    ///
    /// * `base_policy`
    fn generate_base_policy(
        base_policy: &BaseNetworkPolicy,
    ) -> Result<NetworkPolicy, thorium::Error> {
        // build network policy spec(s)
        let raw = json!({
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                // refrain from including a namespace;
                // we'll copy the spec and add a namespace for each group the base policy needs to be in
                //"namespace":
                "name": &base_policy.name,
                "labels": {
                    "thorium": "true",
                    "base_policy": "true"
                }
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        // match all thorium-spawned pods
                        "thorium": "true",
                    }
                },
            }
        });
        // cast this json into a k8's network policy;
        // unwrap the result because we're sure that it's valid
        let mut network_policy: NetworkPolicy = serde_json::from_value(raw).unwrap();
        // get a mutable reference to the spec
        let netpol_spec = network_policy
            .spec
            .get_or_insert(NetworkPolicySpec::default());
        // define the policy types we're applying to (ingress/egress)
        // based on the presence/absence of rules
        netpol_spec.policy_types = match (&base_policy.ingress, &base_policy.egress) {
            (None, None) => None,
            (None, Some(_)) => Some(vec!["Egress".to_string()]),
            (Some(_), None) => Some(vec!["Ingress".to_string()]),
            (Some(_), Some(_)) => Some(vec!["Ingress".to_string(), "Egress".to_string()]),
        };
        // overlay the Thorium network policy rules on top of the K8's network policy
        netpol_spec.ingress = base_policy
            .ingress
            .as_ref()
            .map(|ingress| {
                ingress
                    .iter()
                    .cloned()
                    // attempt to convert our conf rule to a real thorium rule, then to a K8's rule
                    .map(|conf_rule| Ok(NetworkPolicyRule::try_from(conf_rule)?.into()))
                    // propagate any errors
                    .collect::<Result<Vec<NetworkPolicyIngressRule>, thorium::Error>>()
            })
            .transpose()?;
        netpol_spec.egress = base_policy
            .egress
            .as_ref()
            .map(|egress| {
                egress
                    .iter()
                    .cloned()
                    // attempt to convert our conf rule to a real thorium rule, then to a K8's rule
                    .map(|conf_rule| Ok(NetworkPolicyRule::try_from(conf_rule)?.into()))
                    // propagate any errors
                    .collect::<Result<Vec<NetworkPolicyEgressRule>, thorium::Error>>()
            })
            .transpose()?;
        // return our base K8's network policy spec
        Ok(network_policy)
    }

    /// Deploys network policies within a specific namespace
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to deploy the network policies to
    /// * `network_policies` - The network policies to deploy
    /// * `errors` - A list of errors and the underlying policy that caused that error
    pub async fn deploy<'a>(
        &self,
        ns: &str,
        network_policies: Vec<(Uuid, NetworkPolicy)>,
        errors: &mut Vec<(Uuid, String, kube::Error)>,
    ) {
        // get a namespaced client for our network policies' target namespace
        let api: Api<NetworkPolicy> = Api::namespaced(self.client.clone(), ns);
        // set the default params
        let params = PostParams::default();
        // deploy network policies 5 at a time
        let deploys = helpers::assert_send_stream(
            stream::iter(network_policies.iter())
                .map(|(_, netpol)| {
                    // get references to minimize cloning
                    let api_ref = &api;
                    let params_ref = &params;
                    async move { api_ref.create(params_ref, netpol).await }
                })
                .buffered(5),
        )
        .collect::<Vec<Result<_, _>>>()
        .await;
        // crawl our network policies and determine if any failures occurred
        for ((id, policy), deploy) in network_policies.into_iter().zip(deploys.into_iter()) {
            // check if our deployment failed
            if let Err(error) = deploy {
                // this attempt failed so track this error
                errors.push((id, policy.metadata.name.unwrap_or_default(), error));
            }
        }
    }

    /// Deploy base network policies to the given namespace
    ///
    /// Returns a list of errors and the names of policies the errors arose from
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to deploy the base network policies to
    /// * `base_policy_specs` - The specs of base policies to deploy
    pub async fn deploy_base<'a, I>(
        &'a self,
        ns: &str,
        base_policy_specs: I,
    ) -> Vec<(String, kube::Error)>
    where
        I: IntoIterator<Item = &'a NetworkPolicy> + Send,
        <I as IntoIterator>::IntoIter: std::marker::Send,
    {
        // get a namespaced client for our base network policies' target namespace
        let api: Api<NetworkPolicy> = Api::namespaced(self.client.clone(), ns);
        // set the default params
        let params = PostParams::default();
        // deploy base network policies 5 at a time
        helpers::assert_send_stream(
            stream::iter(base_policy_specs)
                .map(|netpol| {
                    // get references to minimize cloning
                    let api_ref = &api;
                    let params_ref = &params;
                    async move { (netpol, api_ref.create(params_ref, netpol).await) }
                })
                .buffered(5),
        )
        .collect::<Vec<(_, Result<_, _>)>>()
        .await
        .into_iter()
        // return the names of base policies that failed to deploy and their errors
        .filter_map(|(policy, deploy)| {
            deploy
                .err()
                .map(|err| (policy.metadata.name.clone().unwrap_or_default(), err))
        })
        .collect()
    }

    /// Delete network policies from K8's with the given names at the given namespace
    ///
    /// Returns a list of policies that had an error when deleting and their errors
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace the policy is in
    /// * `policy_names` - The names of the policies to delete
    pub async fn delete_many(
        &self,
        ns: &str,
        policy_names: Vec<String>,
    ) -> Vec<(String, kube::Error)> {
        // get a namespaced client for our network policies' target namespace
        let api: Api<NetworkPolicy> = Api::namespaced(self.client.clone(), ns);
        // build delete params
        let params = DeleteParams::default().grace_period(0);
        // delete the policies 5 at a time
        let deletes = helpers::assert_send_stream(
            stream::iter(policy_names.iter())
                .map(|policy_name| {
                    // get references to minimize cloning
                    let api_ref = &api;
                    let params_ref = &params;
                    async move { api_ref.delete(policy_name, params_ref).await }
                })
                .buffered(5),
        )
        .collect::<Vec<Result<_, _>>>()
        .await;
        // collect errors and the policy names connected to them
        policy_names
            .into_iter()
            .zip(deletes.into_iter())
            .filter_map(|(policy_name, delete)| delete.err().map(|err| (policy_name, err)))
            .collect()
    }
}

/// A cache of information related to network policies that needs to be
/// retained between runs
#[derive(Default)]
pub struct NetworkPolicyCache {
    /// K8's specs for all of the base network policies defined in the Thorium config;
    /// saved in the cache to avoid regenerating them each time we need to reference them,
    /// and mapped by their name to easily find
    pub base_policy_specs: HashMap<String, NetworkPolicy>,
    /// The set of namespaces that base network policies have already been
    /// successfully deployed to
    pub namespaces_with_base_policies: HashSet<String>,
    /// ID's of policies that still need to be added
    pub policies_to_add: Vec<Uuid>,
    /// K8's names of policies that still need to be removed,
    /// mapped by group/namespace
    pub policies_to_remove: HashMap<String, Vec<String>>,
}

impl NetworkPolicyCache {
    /// Add base network policy specs to the cache
    ///
    /// # Arguments
    ///
    /// * `base_policy_specs` - The base network policy specs to add
    fn base_policy_specs(mut self, base_policy_specs: HashMap<String, NetworkPolicy>) -> Self {
        self.base_policy_specs = base_policy_specs;
        self
    }

    /// Update the network policy cache with the contents of a [`Cache`] of Thorium data
    ///
    /// The reasoning here is there is certain info that can only be computed
    /// within the context of [`Cache::reload_data`] using the data from the
    /// previous (stale) cache (specifically, any policies that have been added,
    /// removed, or updated since last cache reload); we also retain info on
    /// policies that failed to be added/removed last time in this `NetworkPolicyCache`,
    /// so we need to combine the info from [`Cache`] and `NetworkPolicyCache` here.
    ///
    /// # Arguments
    ///
    /// * `thorium_cache` - A cache of info retrieved from Thorium
    pub fn update(&mut self, thorium_cache: &Cache) {
        // add any policies that have been added since last cache reload
        self.policies_to_add
            .extend(&thorium_cache.network_policies.policies_added);
        // add any policies that have been removed since last cache reload (in all of their namespaces)
        for (ns, policy_names) in &thorium_cache.network_policies.policies_removed {
            raw_entry_vec_extend!(self.policies_to_remove, ns, policy_names.clone());
        }
    }
}

/// Check if the policy in K8's is equal to the cached policy
///
/// Note that the cached policy does not yet have a namespace, so checks that
/// the K8's policy is in the proper namespace should occur outside of this
/// function; there's also a lot of metadata in the K8's policy that the cached one
/// will not have (creation date, etc.) that we don't compare
///
/// # Arguments
///
/// * `k8s_policy` - The policy retrieved from K8's
/// * `cached_policy` - The policy from the Thorium cache
pub(super) fn policies_equal(k8s_policy: &NetworkPolicy, cached_policy: &NetworkPolicy) -> bool {
    same!(k8s_policy.metadata.name, cached_policy.metadata.name);
    same!(k8s_policy.spec, cached_policy.spec);
    true
}
