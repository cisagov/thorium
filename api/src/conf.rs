//! The shared config for Thorium
use schemars::JsonSchema;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[cfg(feature = "api")]
use base64::Engine as _;

use crate::models::{
    Image, ImageScaler, NetworkPolicyCustomK8sRule, NetworkPolicyCustomLabel, NetworkPolicyRuleRaw,
    NetworkProtocol, UnixInfo,
};

/// Helps serde default a value to false
fn default_false() -> bool {
    false
}

/// Helps serde default a value to true
fn default_true() -> bool {
    true
}

/// Settings for filtering out certain nodes when setting up nodes to execute
/// Thorium pods
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct NodeFilters {
    /// Whether to filter out master nodes or not
    #[serde(default = "default_false")]
    pub master: bool,
    /// Custom labels to require nodes to have
    #[serde(default)]
    pub custom: Vec<String>,
}

impl Default for NodeFilters {
    /// Create default `NodeFilter` object
    fn default() -> Self {
        NodeFilters {
            master: false,
            custom: Vec::default(),
        }
    }
}

/// Helps serde default the retention time to 7 days
fn default_retention() -> u64 {
    604_800
}

/// Helps serde default how many results to retain for each group to 3
fn default_results_versions() -> usize {
    3
}

/// Retention settings for data in Thorium
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Retention {
    /// How long general job data should be retained in Thorium
    #[serde(default = "default_retention")]
    pub data: u64,
    /// How long job logs should be retained for from the moment each row is inserted
    #[serde(default = "default_retention")]
    pub logs: u64,
    /// How long notifications should be retained for from the moment they are inserted;
    /// notifications at the 'ERROR' level never expire by default
    #[serde(default = "default_retention")]
    pub notifications: u64,
    /// How many results to retain for each group
    #[serde(default = "default_results_versions")]
    pub results: usize,
}

impl Default for Retention {
    fn default() -> Self {
        Self {
            data: default_retention(),
            logs: default_retention(),
            notifications: default_retention(),
            results: default_results_versions(),
        }
    }
}

/// The credentials to use when listing group membership info from ldap
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct LdapCreds {
    /// The filters to append to uid=<username> when binding in ldap
    pub bind_filters: String,
    /// The user to bind as
    pub user: String,
    /// The password to use when binding
    pub password: String,
}

/// How to deserialize IDs from ldap
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub enum LdapIdCast {
    /// Read it as an int
    Int(usize),
    //// Deserialize it from a specific chunk of a SID that is base64 encoded
    Base64Sid(usize),
}

impl Default for LdapIdCast {
    // Set the default for LdapIdCast
    fn default() -> Self {
        LdapIdCast::Int(0)
    }
}

impl LdapIdCast {
    /// Deserialize an id from an ldap attr
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw ldap attribute to cast
    #[cfg(feature = "api")]
    #[tracing::instrument(name = "LdapIdCast::cast", err(Debug))]
    pub fn cast(&self, raw: Vec<String>) -> Result<u64, crate::utils::ApiError> {
        // use the correct deserializer
        match self {
            LdapIdCast::Int(_) => {
                if let Some(first) = raw.first() {
                    return Ok(first.parse::<u64>()?);
                }
            }
            LdapIdCast::Base64Sid(index) => {
                if let Some(first) = raw.first() {
                    // deserialize our base64 encoded side
                    let decoded_bytes = base64::engine::general_purpose::STANDARD.decode(first)?;
                    // make sure this is a version 1 sid
                    if decoded_bytes[0] != 1 {
                        // log an error that this SID is invalid
                        tracing::event!(
                            tracing::Level::ERROR,
                            error = "Unsupported SID version",
                            version = decoded_bytes[0]
                        );
                        // return an internal error
                        return crate::internal_err!("Failed to parse LDAP response".to_owned());
                    }
                    // check our sids length
                    if 4 * decoded_bytes[1] as usize != decoded_bytes[8..].len() {
                        // log an error that this SID is invalid
                        tracing::event!(tracing::Level::ERROR, error = "SID length mismatch");
                        // return an internal error
                        return crate::internal_err!("Failed to parse LDAP response".to_owned());
                    }
                    // extract this id from our SID
                    if let Some(chunk) = decoded_bytes[8..].chunks(4).nth(*index) {
                        // cast this slice to an array
                        match chunk.try_into() {
                            Ok(array_cast) => {
                                // cast this array to a u32
                                let u32_cast = u32::from_ne_bytes(array_cast);
                                // recast to a u64 and return
                                return Ok(u64::from(u32_cast));
                            }
                            Err(error) => {
                                // log an error that this SID is invalid
                                tracing::event!(tracing::Level::ERROR, error = error.to_string());
                            }
                        }
                    }
                    // return an internal error
                    return crate::internal_err!("Failed to parse LDAP response".to_owned());
                }
            }
        }
        // we failed to get a valid id so return an error
        crate::internal_err!("Failed to get unix info from ldap".to_owned())
    }
}

/// The different attribute types in ldap
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub enum LdapAttrs {
    String(String),
    Binary(String),
}

impl LdapAttrs {
    /// Extract this entry from a `SearchEntry`
    ///
    /// Binary attributes will be base64 encoded and then likely decoded later.
    ///
    /// # Arguments
    ///
    /// * `entry` - The entry to get our attributes from
    #[cfg(feature = "api")]
    pub fn get(&self, entry: &mut ldap3::SearchEntry) -> Option<Vec<String>> {
        match self {
            LdapAttrs::String(key) => entry.attrs.get(key).cloned(),
            LdapAttrs::Binary(key) => {
                // get our attribute
                match entry.bin_attrs.get(key) {
                    Some(raw_chunks) => {
                        // encode all of our binary chunks
                        let encoded = raw_chunks
                            .iter()
                            .map(|raw| base64::engine::general_purpose::STANDARD.encode(raw))
                            .collect::<Vec<String>>();
                        Some(encoded)
                    }
                    None => None,
                }
            }
        }
    }
}

/// The info needed to get a users unix user/group id
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct UnixId {
    /// The name of the attribute to extract our id from
    pub attr: LdapAttrs,
    /// The strategy to use when casting/deserializing this id
    #[serde(default)]
    pub cast: LdapIdCast,
}

/// Helps serde default the sync interval for LDAP time to 10 minutes
fn default_sync_interval() -> u64 {
    600
}

/// What to prepend to the username to bind too
fn default_user_prepend() -> String {
    "uid=".to_owned()
}

/// What to append to the username to bind too
fn default_user_append() -> String {
    String::new()
}

/// What to append to usernames in search filters
fn default_search_prepend() -> String {
    String::new()
}

/// LDAP authentication settings
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Ldap {
    /// The hostname ldap can be reached at including ldap:// or ldaps://
    pub host: String,
    /// What to prepend to the username to bind too
    #[serde(default = "default_user_prepend")]
    pub user_prepend: String,
    /// What to append to the username to bind too
    #[serde(default = "default_user_append")]
    pub user_append: String,
    /// The filters to append to uid=<username> when binding in ldap
    pub bind_filters: String,
    /// What to prepend to usernames in search filters
    #[serde(default = "default_search_prepend")]
    pub search_filter_prepend: String,
    /// The filters to append to cn=<group> when searching ldap
    pub scope: String,
    /// The info needed to extract a unix user id
    pub user_unix_id: UnixId,
    /// The info needed to extract a unix group id
    pub group_unix_id: UnixId,
    /// The attribute to use when getting group membership
    pub group_members_attr: String,
    /// The field to extract group membership usernames from
    pub group_member_field: Option<String>,
    /// How long to sleep between syncing data in seconds
    #[serde(default = "default_sync_interval")]
    pub sync_interval: u64,
    /// Verify that the TLS cert is valid or not
    #[serde(default = "default_true")]
    pub tls_verify: bool,
    // The credentials to use when listing group membership info
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials: Option<LdapCreds>,
}

/// Helps serde default the default user token expiration to 90
fn default_token_expire() -> u32 {
    90
}

/// Helps serde default the local user/group IDs to a sane default
fn default_local_user_ids() -> UnixInfo {
    UnixInfo {
        user: 1_879_048_192,
        group: 1_879_048_192,
    }
}

/// The email settings to use for verification emails
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct EmailVerification {
    /// The thorium base api url to use for email verification (e.g. https://thorium.sandia.gov/api)
    pub base_url: String,
    /// The smtp server to use when sending emails
    pub smtp_server: String,
    /// The email address to send verification emails from
    pub addr: String,
    /// The password for email verification
    pub password: String,
    /// The email regexes to restrict users too
    pub approved_emails: Vec<String>,
}

/// Authentication settings
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Auth {
    // How long a users token can live for in days
    #[serde(default = "default_token_expire")]
    pub token_expire: u32,
    /// The settings to use for ldap
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ldap: Option<Ldap>,
    /// The user/group unix IDs to use for local users
    #[serde(default = "default_local_user_ids")]
    pub local_user_ids: UnixInfo,
    /// The email settings to use
    pub email: Option<EmailVerification>,
}

impl Default for Auth {
    /// Create a default auth config
    fn default() -> Self {
        Auth {
            token_expire: default_token_expire(),
            ldap: None,
            local_user_ids: default_local_user_ids(),
            email: None,
        }
    }
}

/// Helps serde default the cpu weight to 2
fn default_cpu_weight() -> u64 {
    2
}

/// Helps serde default the memory weight to 1
fn default_memory_weight() -> u64 {
    1
}

/// The settings to use when calculating fair share costs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct FairShareWeights {
    /// The multiplier to apply to cpu costs
    #[serde(default = "default_cpu_weight")]
    pub cpu: u64,
    /// The multiplier to apply to memory costs
    #[serde(default = "default_memory_weight")]
    pub memory: u64,
}

impl Default for FairShareWeights {
    fn default() -> Self {
        FairShareWeights {
            cpu: default_cpu_weight(),
            memory: default_memory_weight(),
        }
    }
}

/// The host aliases to apply to all pods in K8s
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, Default)]
pub struct K8sHostAliases {
    /// The ip for a list of hostname aliases
    ip: String,
    /// The host aliases for this IP
    hostnames: Vec<String>,
}

/// Helps serde default the max positive sway to 50
fn default_max_sway() -> u64 {
    50
}

/// Helps serde default the dwell for the scaler to 5
fn default_dwell() -> u64 {
    5
}

/// Helps serde default the fair share divisor to return the entire cluster every 10 mins
fn default_fair_share_divisor() -> u64 {
    1
}

/// The settings for a single k8s cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct K8sCluster {
    /// The name to use for this cluster in Thorium
    pub alias: Option<String>,
    /// The nodes in the cluster to run Thorium jobs on (defaults to all)
    #[serde(default)]
    pub nodes: Vec<String>,
    /// Settings for filtering nodes from executing Thorium jobs
    #[serde(default = "NodeFilters::default")]
    pub filters: NodeFilters,
    /// Any groups to restrict to being to deploy jobs to this cluster
    #[serde(default)]
    pub groups: Vec<String>,
    /// The max positive sway when spawning pods
    #[serde(default = "default_max_sway")]
    pub max_sway: u64,
    /// The tls server name to use for cert validation
    #[serde(default)]
    pub tls_server_name: Option<String>,
    /// The url to use for the Thorium api instead of the in cluster service
    #[serde(default)]
    pub api_url: Option<String>,
    /// The host aliases to apply to pods in this cluster
    #[serde(default)]
    pub host_aliases: Vec<K8sHostAliases>,
    /// Whether this cluster uses an invalid certificate or not
    #[serde(default)]
    pub insecure: bool,
    /// Whether this cluster can run any image or is restricted
    #[serde(default)]
    pub restricted: bool,
    /// The restrictions for this cluster
    /// TODO: move this out of the config and into the api in a cleaner/less painful way
    #[serde(default)]
    pub image_restrictions: HashMap<String, HashMap<String, Vec<String>>>,
}

impl Default for K8sCluster {
    /// Create a default k8s cluster config
    fn default() -> Self {
        K8sCluster {
            alias: None,
            nodes: vec![],
            filters: NodeFilters::default(),
            groups: vec![],
            max_sway: default_max_sway(),
            tls_server_name: None,
            api_url: None,
            host_aliases: Vec::default(),
            insecure: false,
            restricted: false,
            image_restrictions: HashMap::default(),
        }
    }
}

/// The settings for all k8s clusters used by Thorium
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct K8s {
    /// The k8s clusters to be used by Thorium by their context name
    #[serde(default)]
    pub clusters: BTreeMap<String, K8sCluster>,
    /// The contexts to ignore when parsing our kube config
    #[serde(default)]
    pub ignored_contexts: HashSet<String>,
    /// How long at minimum to wait between scale attempts in seconds
    #[serde(default = "default_dwell")]
    pub dwell: u64,
    /// The settings to use when calculating fair share costs
    #[serde(default = "FairShareWeights::default")]
    pub fair_share: FairShareWeights,
    /// The divisor to use when calculating what % of resources to reduce fair share ranks by
    #[serde(default = "default_fair_share_divisor")]
    pub fair_share_divisor: u64,
}

impl Default for K8s {
    /// Create a default k8s config
    fn default() -> Self {
        K8s {
            clusters: BTreeMap::default(),
            ignored_contexts: HashSet::default(),
            dwell: default_dwell(),
            fair_share: FairShareWeights::default(),
            fair_share_divisor: default_fair_share_divisor(),
        }
    }
}

impl K8s {
    /// Get a clusters k8s alias if it has one otherwise use the context name
    ///
    /// # Arguments
    ///
    /// The name of the context to get our cluster name for
    pub fn cluster_name<'a>(&'a self, context_name: &'a String) -> &'a String {
        match self.clusters.get(context_name) {
            Some(cluster) => cluster.alias.as_ref().unwrap_or(context_name),
            None => context_name,
        }
    }

    /// Get the TLS server name for this cluster
    ///
    /// If a cluster cannot be found then this function defaults to None.
    ///
    /// # Arguments
    ///
    /// * `cluster` - The context name for this cluster
    pub fn tls_server_name(&self, context_name: &str) -> Option<&String> {
        match self.clusters.get(context_name) {
            Some(cluster) => cluster.tls_server_name.as_ref(),
            None => None,
        }
    }

    /// Determine if a cluster should accept invalid certificates
    ///
    /// If this cluster cannot be found then this function will default to false.
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster to check by its context name
    pub fn accept_invalid_certs(&self, context_name: &str) -> bool {
        // try to get our cluster
        match self.clusters.get(context_name) {
            Some(cluster) => {
                println!("INSECURE CERT SUPPORTED -> {}", cluster.insecure);
                cluster.insecure
            }
            None => false,
        }
    }

    /// Get the api url to use if one was set for this cluster
    ///
    /// If this cluster cannot be found then this function will default to None.
    ///
    /// # Arguments
    ///
    /// * `context_name` - The cluster to check by its context name
    pub fn api_url(&self, context_name: &str) -> Option<&String> {
        // try to get our cluster
        match self.clusters.get(context_name) {
            Some(cluster) => cluster.api_url.as_ref(),
            None => None,
        }
    }

    /// The host aliases for pods in a specific cluster
    ///
    /// If this cluster cannot be found then this function will default to an empty hashmap.
    ///
    /// # Arguments
    ///
    /// * `context_name` - The cluster to check by its context name
    pub fn host_aliases(&self, context_name: &str) -> Option<&Vec<K8sHostAliases>> {
        // try to get our cluster
        match self.clusters.get(context_name) {
            Some(cluster) => Some(&cluster.host_aliases),
            None => None,
        }
    }

    /// Get the image restrictions for our k8s clusters
    pub fn restrictions(&self, restrictions: &mut WorkerRestrictions) {
        // crawl over our baremetal clusters
        for (cluster_name, cluster) in &self.clusters {
            // get a reference to this clusters alias or name
            let cluster_name = cluster.alias.as_ref().unwrap_or(cluster_name);
            // if this cluster is restricted that add it to our restricted cluster set
            if cluster.restricted {
                // set this cluster to be restricted using either its real name or its alias
                restrictions.clusters.insert(cluster_name.clone());
            }
            // crawl over the nodes in this cluster
            for (node_name, group_restrictions) in &cluster.image_restrictions {
                // if this node has image restrictions then add those
                for (group, images) in group_restrictions {
                    // get an entry to this groups restriction map
                    let group_entry = restrictions.images.entry(group.clone()).or_default();
                    // crawl over this groups image restrictions
                    for image in images {
                        // get an entry to this images restrictions
                        let image_entry = group_entry.entry(image.clone()).or_default();
                        // get an entry to this clusters node restriction list
                        let cluster_entry = image_entry
                            .clusters
                            .entry(cluster_name.clone())
                            .or_default();
                        // add our node to this clusters node restriction list
                        cluster_entry.insert(node_name.clone());
                    }
                }
            }
        }
    }
}

/// A restriction on what images or groups a node can run jobs for
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, JsonSchema)]
pub struct BareMetalNodeSettings {
    /// The group specific images this node can run jobs for
    #[serde(default)]
    pub images: HashMap<String, Vec<String>>,
}

/// Helps serde default the user to run the agent as on bare metal nodes
fn default_bare_metal_user() -> String {
    "root".to_owned()
}

/// Helps serde default the path to the agent for the bare metal scaler
fn default_bare_metal_agent() -> String {
    "thorium-agent".to_owned()
}

/// The settings for a specific bare metal cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct BareMetalCluster {
    /// The user to log in to these nodes as
    #[serde(default = "default_bare_metal_user")]
    pub username: String,
    /// The nodes in the cluster to run Thorium jobs on by hostname/ip and any restrictions
    #[serde(default)]
    pub nodes: HashMap<String, BareMetalNodeSettings>,
    /// The max positive sway when configuring agents
    #[serde(default = "default_max_sway")]
    pub max_sway: u64,
    /// The path the bare metal scaler can find the Thorium agent at
    #[serde(default = "default_bare_metal_agent")]
    pub agent_path: String,
}

impl Default for BareMetalCluster {
    /// Create a default bare metal config
    fn default() -> Self {
        BareMetalCluster {
            username: default_bare_metal_user(),
            nodes: HashMap::default(),
            max_sway: default_max_sway(),
            agent_path: default_bare_metal_agent(),
        }
    }
}

/// The settings for a specific bare metal cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct BareMetal {
    /// The bare metal clusters to schedule Thorium jobs on
    pub clusters: HashMap<String, BareMetalCluster>,
    /// How long at minimum to wait between scale attempts in seconds
    #[serde(default = "default_dwell")]
    pub dwell: u64,
    /// The settings to use when calculating fair share costs
    #[serde(default = "FairShareWeights::default")]
    pub fair_share: FairShareWeights,
    /// The divisor to use when calculating what % of resources to reduce fair share ranks by
    #[serde(default = "default_fair_share_divisor")]
    pub fair_share_divisor: u64,
}

impl Default for BareMetal {
    /// Create a Thorium default bare metal config
    fn default() -> Self {
        BareMetal {
            clusters: HashMap::default(),
            dwell: default_dwell(),
            fair_share: FairShareWeights::default(),
            fair_share_divisor: default_fair_share_divisor(),
        }
    }
}

impl BareMetal {
    /// Get the image restrictions for our baremetal clusters
    pub fn restrictions(&self, restrictions: &mut WorkerRestrictions) {
        // crawl over our baremetal clusters
        for (cluster_name, cluster) in &self.clusters {
            // crawl over the nodes in this cluster
            for (node_name, node) in &cluster.nodes {
                // if this node has image restrictions then add those
                for (group, images) in &node.images {
                    // get an entry to this groups restriction map
                    let group_entry = restrictions.images.entry(group.clone()).or_default();
                    // crawl over this groups image restrictions
                    for image in images {
                        // get an entry to this images restrictions
                        let image_entry = group_entry.entry(image.clone()).or_default();
                        // get an entry to this clusters node restriction list
                        let cluster_entry = image_entry
                            .clusters
                            .entry(cluster_name.clone())
                            .or_default();
                        // add our node to this clusters node restriction list
                        cluster_entry.insert(node_name.clone());
                    }
                }
            }
        }
    }
}

/// The settings for scaling Windows workers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Windows {
    /// The windows clusters to schedule Thorium jobs on
    pub clusters: Vec<String>,
    /// How long at minimum to wait between scale attempts in seconds
    #[serde(default = "default_dwell")]
    pub dwell: u64,
    /// The settings to use when calculating fair share costs
    #[serde(default = "FairShareWeights::default")]
    pub fair_share: FairShareWeights,
    /// The divisor to use when calculating what % of resources to reduce fair share ranks by
    #[serde(default = "default_fair_share_divisor")]
    pub fair_share_divisor: u64,
}

impl Default for Windows {
    /// Create a Thorium default windows config
    fn default() -> Self {
        Windows {
            clusters: Vec::default(),
            dwell: default_dwell(),
            fair_share: FairShareWeights::default(),
            fair_share_divisor: default_fair_share_divisor(),
        }
    }
}
/// A restriction on what images or groups a kvm node can run jobs for
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, JsonSchema)]
pub struct KvmNodeSettings {
    /// The group specific images this node can run jobs for
    #[serde(default)]
    pub images: HashMap<String, Vec<String>>,
}

/// Helps serde default the path to the agent for the bare metal scaler
fn default_kvm_agent() -> String {
    "thorium-agent".to_owned()
}

/// The settings for a specific kvm cluster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct KvmCluster {
    /// The nodes in the cluster to run Thorium jobs on by hostname/ip and any restrictions
    #[serde(default)]
    pub nodes: HashMap<String, KvmNodeSettings>,
    /// The max positive sway when configuring agents
    #[serde(default = "default_max_sway")]
    pub max_sway: u64,
    /// The path the bare metal scaler can find the Thorium agent at
    #[serde(default = "default_kvm_agent")]
    pub agent_path: String,
}

/// The settings for scaling Kvm workers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Kvm {
    /// The kvm clusters to schedule Thorium jobs on
    pub clusters: HashMap<String, KvmCluster>,
    /// How long at minimum to wait between scale attempts in seconds
    #[serde(default = "default_dwell")]
    pub dwell: u64,
    /// The settings to use when calculating fair share costs
    #[serde(default = "FairShareWeights::default")]
    pub fair_share: FairShareWeights,
    /// The divisor to use when calculating what % of resources to reduce fair share ranks by
    #[serde(default = "default_fair_share_divisor")]
    pub fair_share_divisor: u64,
}

impl Kvm {
    /// Get the image restrictions for our kvm clusters
    pub fn restrictions(&self, restrictions: &mut WorkerRestrictions) {
        // crawl over our baremetal clusters
        for (cluster_name, cluster) in &self.clusters {
            // crawl over the nodes in this cluster
            for (node_name, node) in &cluster.nodes {
                // if this node has image restrictions then add those
                for (group, images) in &node.images {
                    // get an entry to this groups restriction map
                    let group_entry = restrictions.images.entry(group.clone()).or_default();
                    // crawl over this groups image restrictions
                    for image in images {
                        // get an entry to this images restrictions
                        let image_entry = group_entry.entry(image.clone()).or_default();
                        // get an entry to this clusters node restriction list
                        let cluster_entry = image_entry
                            .clusters
                            .entry(cluster_name.clone())
                            .or_default();
                        // add our node to this clusters node restriction list
                        cluster_entry.insert(node_name.clone());
                    }
                }
            }
        }
    }
}

impl Default for Kvm {
    /// Create a Thorium default kvm config
    fn default() -> Self {
        Kvm {
            clusters: HashMap::default(),
            dwell: default_dwell(),
            fair_share: FairShareWeights::default(),
            fair_share_divisor: default_fair_share_divisor(),
        }
    }
}

/// Helps serde default the ldap sync delay to 600 seconds
fn default_ldap_sync() -> u32 {
    600
}

/// Helps serde default the bare metal image runtime update to 300 seconds
fn default_image_runtimes() -> u32 {
    300
}

/// Helps serde default the zombie cleanup delay to 30 seconds
fn default_zombie_delay() -> u32 {
    30
}

/// Helps serde default the cache reload to 600 seconds
fn default_cache_reload() -> u32 {
    600
}

/// Helps serde default the resource count update to 120 seconds
fn default_resources() -> u32 {
    120
}

/// Helps serde default the orphaned resource cleaned to 25 seconds
fn default_cleanup() -> u32 {
    25
}

/// Helps serde default the decreasing fair share ranks to 600 seconds
fn default_decreasing_fair_share() -> u32 {
    600
}

/// The time delay between different tasks carried out in the scaler
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct ScalerTaskDelays {
    /// How long to wait between syncing ldap info
    #[serde(default = "default_ldap_sync")]
    pub ldap_sync: u32,
    /// How long to wait between updating our image runtimes
    #[serde(default = "default_image_runtimes")]
    pub image_runtimes: u32,
    /// How long to wait between checking for zombie jobs
    #[serde(default = "default_zombie_delay")]
    pub zombies: u32,
    /// How long to wait between reloading the scaler cache
    #[serde(default = "default_cache_reload")]
    pub cache_reload: u32,
    /// How long to wait between updating our resources count
    #[serde(default = "default_resources")]
    pub resources: u32,
    /// How long to wait between cleaning up any orphaned resources
    #[serde(default = "default_cleanup")]
    pub cleanup: u32,
    /// How long to wait between decreasing fair share ranks
    #[serde(default = "default_decreasing_fair_share")]
    pub decrease_fair_share: u32,
}

impl Default for ScalerTaskDelays {
    /// Create a new instance of the bare metal task delay struct
    fn default() -> ScalerTaskDelays {
        ScalerTaskDelays {
            ldap_sync: default_ldap_sync(),
            image_runtimes: default_image_runtimes(),
            zombies: default_zombie_delay(),
            cache_reload: default_cache_reload(),
            resources: default_resources(),
            cleanup: default_cleanup(),
            decrease_fair_share: default_decreasing_fair_share(),
        }
    }
}

/// Helps serde default the max positive sway to 100
fn default_cache_lifetime() -> u64 {
    600
}

/// Helps serde default the deadline window to 100,000
fn default_deadline_window() -> u64 {
    100_000
}

/// Whether an image has some restrictions or not
pub enum IsRestricted<'a> {
    /// This image is not restricted to any clusters/nodes
    No,
    /// This image can be spawned on this cluster on specific nodes
    Yes(&'a HashSet<String>),
    /// This image cannot be spawned on this cluster
    WrongCluster,
}

/// Any restrictions for what nodes images can be spawned on a specific cluster
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, JsonSchema)]
pub struct ClusterImageRestrictions {
    /// The clusters that have image restrictions for nodes
    pub clusters: HashMap<String, HashSet<String>>,
}

/// Any restrictions for what nodes images can be spawned on
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, JsonSchema)]
pub struct WorkerRestrictions {
    /// The clusters that are restricted to running only certain jobs
    pub clusters: HashSet<String>,
    /// The groups/images that have cluster/node preferences or restrictions
    pub images: HashMap<String, HashMap<String, ClusterImageRestrictions>>,
}

impl WorkerRestrictions {
    /// Get whether this image can be spawned on this cluster
    ///
    /// If an empty list of nodes is returned that means
    #[must_use]
    pub fn check<'a>(&'a self, cluster: &str, image: &Image) -> IsRestricted<'a> {
        // try to get the restrictions for this image
        if let Some(group_restrictions) = self.images.get(&image.group) {
            if let Some(image_restrictions) = group_restrictions.get(&image.name) {
                // this image has some restrictions
                // if our cluster isn't in the restriction map then we can't schedule to this cluster
                match image_restrictions.clusters.get(cluster) {
                    Some(nodes) => return IsRestricted::Yes(nodes),
                    None => return IsRestricted::WrongCluster,
                }
            }
        }
        // if this cluster is restricted then return that it cannot be scheduled
        if self.clusters.contains(cluster) {
            IsRestricted::WrongCluster
        } else {
            // this image has no restrictions on where it can be spawned
            IsRestricted::No
        }
    }

    /// Get the restrictions for this image across all clusters
    ///
    /// # Arguments
    ///
    /// * `image` - The image to get restrictions for
    #[must_use]
    pub fn get<'a>(&'a self, image: &Image) -> Option<&'a ClusterImageRestrictions> {
        // try to get the restrictions for this image
        if let Some(group_restrictions) = self.images.get(&image.group) {
            // get this images restrictions if it has any
            group_restrictions.get(&image.name)
        } else {
            // this group has no images with restrictions on where they can be spawned
            None
        }
    }
}

/// The default location for the crane binary
fn default_crane_path() -> PathBuf {
    PathBuf::from("/app/crane")
}

/// The setting for crane
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Crane {
    /// The path to find crane at
    #[serde(default = "default_crane_path")]
    pub path: PathBuf,
    /// Tell crane to skip SSL validation
    #[serde(default)]
    pub insecure: bool,
}

impl Default for Crane {
    /// Create a default crane object
    fn default() -> Self {
        Crane {
            path: default_crane_path(),
            insecure: false,
        }
    }
}

/// The settings for the Thorium scalers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Scaler {
    /// How long the cache should live for at most before being invalidated in seconds
    #[serde(default = "default_cache_lifetime")]
    pub cache_lifetime: u64,
    /// The number of deadlines to pull for one scale loop
    #[serde(default = "default_deadline_window")]
    pub deadline_window: u64,
    /// Specific settings for the K8s scaler
    #[serde(default)]
    pub k8s: K8s,
    /// Specific settings for the BareMetal scaler
    #[serde(default)]
    pub bare_metal: BareMetal,
    /// The windows specific settings
    #[serde(default)]
    pub windows: Windows,
    /// The kvm specific settings
    #[serde(default)]
    pub kvm: Kvm,
    /// The global scaler specific tasks
    #[serde(default)]
    pub tasks: ScalerTaskDelays,
    /// The crane specific settings
    #[serde(default)]
    pub crane: Crane,
}

impl Default for Scaler {
    fn default() -> Self {
        Scaler {
            cache_lifetime: default_cache_lifetime(),
            deadline_window: default_deadline_window(),
            k8s: K8s::default(),
            windows: Windows::default(),
            bare_metal: BareMetal::default(),
            kvm: Kvm::default(),
            tasks: ScalerTaskDelays::default(),
            crane: Crane::default(),
        }
    }
}

impl Scaler {
    /// Get the maximum amount of positive sway allowed for the configured scheduler
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler to get the max sway of
    pub fn max_sway(&self, scaler: ImageScaler, cluster: &str) -> u64 {
        match scaler {
            ImageScaler::K8s => self
                .k8s
                .clusters
                .get(cluster)
                .map_or_else(default_max_sway, |c| c.max_sway),
            ImageScaler::Windows => 2,
            ImageScaler::BareMetal => self
                .bare_metal
                .clusters
                .get(cluster)
                .map_or_else(default_max_sway, |c| c.max_sway),
            ImageScaler::Kvm => self
                .kvm
                .clusters
                .get(cluster)
                .map_or_else(default_max_sway, |c| c.max_sway),
            ImageScaler::External => default_max_sway(),
        }
    }

    /// Get the maximum amount of time between scale attempts allowed for the configured scheduler
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler to get the dwell setting for
    #[must_use]
    pub fn dwell(&self, scaler: ImageScaler) -> u64 {
        match scaler {
            ImageScaler::K8s => self.k8s.dwell,
            ImageScaler::Windows => self.windows.dwell,
            ImageScaler::BareMetal => self.bare_metal.dwell,
            ImageScaler::Kvm => self.kvm.dwell,
            ImageScaler::External => default_dwell(),
        }
    }

    /// Get the names of the clusters for a specific scaler type
    ///
    /// # Arguments
    ///
    /// * `scaler` - The type of scaler to get the cluster names for
    #[must_use]
    pub fn cluster_names(&'_ self, scaler: ImageScaler) -> Vec<&'_ String> {
        match scaler {
            ImageScaler::K8s => self.k8s.clusters.keys().collect(),
            ImageScaler::Windows => self.windows.clusters.iter().collect(),
            ImageScaler::BareMetal => self.bare_metal.clusters.keys().collect(),
            ImageScaler::Kvm => self.kvm.clusters.keys().collect(),
            ImageScaler::External => Vec::default(),
        }
    }

    /// Build a map of image restrictions by cluster and node
    ///
    /// # Arguments
    ///
    /// * `scaler` - The type of scaler to get node restrictions for
    #[must_use]
    pub fn restrictions(&self, scaler: ImageScaler) -> WorkerRestrictions {
        // start with a default restriction map
        let mut restrictions = WorkerRestrictions::default();
        match scaler {
            ImageScaler::Windows | ImageScaler::External => (),
            ImageScaler::K8s => self.k8s.restrictions(&mut restrictions),
            ImageScaler::BareMetal => self.bare_metal.restrictions(&mut restrictions),
            ImageScaler::Kvm => self.kvm.restrictions(&mut restrictions),
        };
        restrictions
    }

    /// Get the fair share weights for this scaler
    ///
    /// # Arguments
    ///
    /// * `scaler` - The type of scaler to get the fair share weights for
    #[must_use]
    pub fn fair_share_weights(&'_ self, scaler: ImageScaler) -> FairShareWeights {
        match scaler {
            ImageScaler::K8s => self.k8s.fair_share.clone(),
            ImageScaler::Windows => self.windows.fair_share.clone(),
            ImageScaler::BareMetal => self.bare_metal.fair_share.clone(),
            ImageScaler::Kvm => self.kvm.fair_share.clone(),
            ImageScaler::External => FairShareWeights::default(),
        }
    }

    /// Get the fair share divisor for this scaler
    ///
    /// # Arguments
    ///
    /// * `scaler` - The type of scaler to get the fair share divisor for
    #[must_use]
    pub fn fair_share_divisor(&'_ self, scaler: ImageScaler) -> u64 {
        match scaler {
            ImageScaler::K8s => self.k8s.fair_share_divisor,
            ImageScaler::Windows => self.windows.fair_share_divisor,
            ImageScaler::BareMetal => self.bare_metal.fair_share_divisor,
            ImageScaler::Kvm => self.kvm.fair_share_divisor,
            ImageScaler::External => default_fair_share_divisor(),
        }
    }
}

/// Returns the default number of workers for the search-streamer
fn default_search_streamer_workers() -> NonZeroUsize {
    NonZeroUsize::new(5).unwrap()
}

/// Returns the default number of chunks per worker in the search-streamer
fn default_search_streamer_chunks_per_worker() -> NonZeroUsize {
    NonZeroUsize::new(10_000).unwrap()
}

/// The settings for the search-streamer init phase
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct SearchStreamerInit {
    /// The number of chunks per worker to split the Scylla into when
    /// running the init phase
    ///
    /// In general, the more data is in Scylla, the greater the number of chunks
    /// it should be split up into. Having more chunks means if the search-streamer
    /// is interrupted while initiating indexes, it can resume the process at a more
    /// specific point and repeat less work; however for very small databases,
    /// having too many chunks will lead to many very sparse or empty chunks and
    /// impact performance.
    #[serde(default = "default_search_streamer_chunks_per_worker")]
    pub chunks_per_worker: NonZeroUsize,
}

impl Default for SearchStreamerInit {
    fn default() -> Self {
        Self {
            chunks_per_worker: default_search_streamer_chunks_per_worker(),
        }
    }
}

/// The settings for the search-streamer
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct SearchStreamer {
    /// The number of workers each streamer in the search-streamer has
    ///
    /// The more workers, the more data can potentially be held in memory
    /// and sent to Elasticsearch at once. Decrease the number of workers
    /// if the search-streamer runs out of memory or Elasticsearch complains.
    #[serde(default = "default_search_streamer_workers")]
    pub workers: NonZeroUsize,
    /// The settings for the search-streamer init phase
    #[serde(default)]
    pub init: SearchStreamerInit,
}

impl Default for SearchStreamer {
    fn default() -> Self {
        Self {
            workers: default_search_streamer_workers(),
            init: SearchStreamerInit::default(),
        }
    }
}

/// The settings for sending traces to stdout/stderr
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct TracingLocal {
    /// The log level to use for stdout/stderr
    pub level: LogLevel,
}

/// The different settings for external tracing services
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub enum TracingServices {
    /// send traces to a gRPC based service
    #[serde(alias = "grpc")]
    Grpc { endpoint: String, level: LogLevel },
}

impl Default for TracingLocal {
    /// Create a default Tracing Local config
    fn default() -> Self {
        TracingLocal {
            level: LogLevel::Info,
        }
    }
}

/// The tracing settings to use
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, Default)]
pub struct Tracing {
    /// The settings for sending traces to an external service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external: Option<TracingServices>,
    /// The settings for sending traces to stdout/stderr
    #[serde(default)]
    pub local: TracingLocal,
}

impl Tracing {
    /// Load a tracing config from a file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load this tracing config from
    pub fn from_file(path: &str) -> Result<Self, config::ConfigError> {
        config::Config::builder()
            // load from a file first
            .add_source(config::File::new(path, config::FileFormat::Yaml))
            // then overlay any environment args on top
            .add_source(config::Environment::with_prefix("TRACING").separator("__"))
            .build()?
            .try_deserialize()
    }
}

/// The log level to set
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy, JsonSchema)]
pub enum LogLevel {
    /// Do not log any info
    Off,
    /// Log at the error level
    Error,
    /// Log at the warning level
    Warn,
    /// Only Set up and up info
    Setup,
    /// Log at the info level
    Info,
    /// Log at the debug level
    Debug,
    /// Log at the tracing level
    Trace,
}

/// Default the log level to Info
impl Default for LogLevel {
    /// Set the default log level to info
    fn default() -> Self {
        LogLevel::Info
    }
}

/// Default the log level to Info
impl LogLevel {
    #[cfg(feature = "trace")]
    /// Cast this log level to a tracing filter
    #[must_use]
    pub fn to_filter(&self) -> tracing::metadata::LevelFilter {
        match self {
            LogLevel::Off => tracing_subscriber::filter::LevelFilter::OFF,
            LogLevel::Error => tracing_subscriber::filter::LevelFilter::ERROR,
            LogLevel::Warn | LogLevel::Setup => tracing_subscriber::filter::LevelFilter::WARN,
            LogLevel::Info => tracing_subscriber::filter::LevelFilter::INFO,
            LogLevel::Debug => tracing_subscriber::filter::LevelFilter::DEBUG,
            LogLevel::Trace => tracing_subscriber::filter::LevelFilter::TRACE,
        }
    }
}

impl std::fmt::Display for LogLevel {
    /// Allow the log level to be displayed
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LogLevel::Off => write!(f, "Off"),
            LogLevel::Error => write!(f, "Error"),
            LogLevel::Warn => write!(f, "Warn"),
            LogLevel::Setup => write!(f, "Setup"),
            LogLevel::Info => write!(f, "Info"),
            LogLevel::Debug => write!(f, "Debug"),
            LogLevel::Trace => write!(f, "Trace"),
        }
    }
}

/// Helps serde default the tags partition size to 3 minutes
fn default_tags_partition_size() -> u16 {
    180
}

/// Helps serde default the tags earliest to 01/01/2021
const fn default_tags_earliest() -> i64 {
    1_609_459_201
}

/// The settings for saving/listing tags in scylla
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Tags {
    /// The number of seconds each partition in the database should cover
    #[serde(default = "default_tags_partition_size")]
    pub partition_size: u16,
    /// The earliest date a tag will exist as a unix epoch
    #[serde(default = "default_tags_earliest")]
    pub earliest: i64,
}

impl Default for Tags {
    fn default() -> Self {
        Tags {
            partition_size: default_tags_partition_size(),
            earliest: default_tags_earliest(),
        }
    }
}

/// Helps serde default the files bucket to thorium-files
fn default_files_password() -> String {
    "SecretCornIsBest".to_owned()
}

/// Helps serde default the files bucket to thorium-files
fn default_files_bucket() -> String {
    "thorium-files".to_owned()
}

/// Helps serde default the files earliest to 01/01/2010
fn default_files_earliest() -> i64 {
    1_262_332_800
}

/// Helps serde default the files chunk size to 3 minutes
fn default_files_partition_size() -> u16 {
    180
}

/// The settings for saving/Carting files to the backend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Files {
    /// The password to use when encrypting carted files
    #[serde(default = "default_files_password")]
    pub password: String,
    /// The bucket to write carted files too
    #[serde(default = "default_files_bucket")]
    pub bucket: String,
    /// The earliest date a file will have a submission date for as a unix epoch
    #[serde(default = "default_files_earliest")]
    pub earliest: i64,
    /// The number of seconds each partition in the database should cover
    #[serde(default = "default_files_partition_size")]
    pub partition_size: u16,
}

impl Default for Files {
    fn default() -> Self {
        Files {
            password: default_files_password(),
            bucket: default_files_bucket(),
            earliest: default_files_earliest(),
            partition_size: default_files_partition_size(),
        }
    }
}

/// Helps serde default the results extra files bucket to thorium-result-files
fn default_results_bucket() -> String {
    "thorium-result-files".to_owned()
}

/// Helps serde default the results earliest to 01/01/2021
fn default_results_earliest() -> i64 {
    1_609_459_201
}

/// Helps serde default the results chunk size to 1 minute
fn default_results_partition_size() -> u16 {
    60
}

/// The settings for saving results to the backend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Results {
    /// The bucket to write extras files to
    #[serde(default = "default_results_bucket")]
    pub bucket: String,
    /// The earliest date a result will exist at as a unix epoch
    #[serde(default = "default_results_earliest")]
    pub earliest: i64,
    /// The number of seconds each partition in the database should cover
    #[serde(default = "default_results_partition_size")]
    pub partition_size: u16,
}

impl Default for Results {
    fn default() -> Self {
        Results {
            bucket: default_results_bucket(),
            earliest: default_results_earliest(),
            partition_size: default_results_partition_size(),
        }
    }
}

/// Helps serde default the ephemeral files bucket to thorium-ephemeral-files
fn default_ephemeral_bucket() -> String {
    "thorium-ephemeral-files".to_owned()
}

/// The settings for saving ephemeral files to the backend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Ephemeral {
    /// The bucket to write extras files to
    #[serde(default = "default_ephemeral_bucket")]
    pub bucket: String,
}

impl Default for Ephemeral {
    fn default() -> Self {
        Ephemeral {
            bucket: default_ephemeral_bucket(),
        }
    }
}

/// Helps serde default the comment attachments bucket to thorium-attachment-files
fn default_attachments_bucket() -> String {
    "thorium-attachment-files".to_owned()
}

/// The settings for saving attachments to the backend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Attachments {
    /// The bucket to write attachment files to
    #[serde(default = "default_attachments_bucket")]
    pub bucket: String,
}

impl Default for Attachments {
    fn default() -> Self {
        Attachments {
            bucket: default_attachments_bucket(),
        }
    }
}

/// Helps serde default the zipped repos bucket to thorium-repo-files
fn default_repos_bucket() -> String {
    "thorium-repos-files".to_owned()
}

/// Helps serde default the repos earliest to 01/01/2010
fn default_repos_earliest() -> i64 {
    946_684_801
}

/// Helps serde default the repos chunk size to 3 minutes
fn default_repos_partition_size() -> u16 {
    180
}

/// The settings for saving repos to the backend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Repos {
    /// The bucket to write zipped repositories to
    #[serde(default = "default_repos_bucket")]
    pub bucket: String,
    /// The earliest date a repo will have a commit for as a unix epoch
    #[serde(default = "default_repos_earliest")]
    pub earliest: i64,
    /// The number of seconds each partition in the database should cover
    #[serde(default = "default_repos_partition_size")]
    pub partition_size: u16,
}

impl Default for Repos {
    fn default() -> Self {
        Repos {
            bucket: default_repos_bucket(),
            earliest: default_repos_earliest(),
            partition_size: default_repos_partition_size(),
        }
    }
}

/// Helps serde default the events partition size to 10 seconds
fn default_events_partition_size() -> u16 {
    10
}

/// The max depth to trigger new triggers at
fn default_events_max_depth() -> u8 {
    5
}

/// The settings related to events
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Events {
    /// How long to hold onto events for regardless of if they are acted on or not
    #[serde(default = "default_retention")]
    pub retention: u64,
    /// The number of seconds each partition in the database should cover
    #[serde(default = "default_events_partition_size")]
    pub partition_size: u16,
    /// The max depth to trigger new triggers at
    #[serde(default = "default_events_max_depth")]
    pub max_depth: u8,
}

impl Default for Events {
    // Build a default instance of the events config
    fn default() -> Self {
        Events {
            retention: default_retention(),
            partition_size: default_events_partition_size(),
            max_depth: default_events_max_depth(),
        }
    }
}

/// Helps serde default the S3 region
fn default_s3_region() -> String {
    String::new()
}

/// The settings for saving/Carting files to the backend
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct S3 {
    /// The access key S3c should use to authenticate
    pub access_key: String,
    /// The secret token S3c should use when authenticating
    pub secret_token: String,
    /// The endpoint S3c should talk to
    pub endpoint: String,
    /// The region our s3 client should use
    #[serde(default = "default_s3_region")]
    pub region: String,
}

/// Helps serde default the max size an incoming json body can be in mebibytes
fn default_json_limit() -> u64 {
    1024
}

/// Helps serde default the max size a form (sans files) can be in mebibytes
fn default_form_limit() -> u64 {
    1024
}

/// Helps serde default the max size an incoming data/file can be in mebibytes
fn default_data_limit() -> u64 {
    1024
}

/// The request size limits to use in the API
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct RequestSizeLimits {
    /// The limit to apply to json bodies
    #[serde(default = "default_json_limit")]
    pub json: u64,
    /// The limit to apply to forms
    #[serde(default = "default_form_limit")]
    pub form: u64,
    /// The limit to apply to data/files
    #[serde(default = "default_data_limit")]
    pub data: u64,
}

impl Default for RequestSizeLimits {
    fn default() -> Self {
        RequestSizeLimits {
            json: default_json_limit(),
            form: default_form_limit(),
            data: default_data_limit(),
        }
    }
}

/// Helps serde default the path to our user facing docs
fn default_user_docs_path() -> PathBuf {
    PathBuf::from("docs/user")
}

/// Helps serde default the path to our developer focused docs
fn default_dev_docs_path() -> PathBuf {
    PathBuf::from("docs/dev")
}

/// Helps serde default the path to Thorctl
fn default_binaries_path() -> PathBuf {
    PathBuf::from("binaries")
}

/// Helps serde default the path resource not found
fn default_not_found_path() -> PathBuf {
    PathBuf::from("docs/user/static_resources/mascot.png")
}

/// The static assets this api should serve
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Assets {
    /// The path to our user focused docs
    #[serde(default = "default_user_docs_path")]
    pub user_docs: PathBuf,
    /// The path to our developer focused docs
    #[serde(default = "default_dev_docs_path")]
    pub dev_docs: PathBuf,
    /// The path to serve Thorium binaries from
    #[serde(default = "default_binaries_path")]
    pub binaries: PathBuf,
    /// The path to the 404 page to serve on docs/binary serving 404s
    #[serde(default = "default_not_found_path")]
    pub not_found: PathBuf,
}

impl Default for Assets {
    /// Create a default assets config
    fn default() -> Self {
        Assets {
            user_docs: default_user_docs_path(),
            dev_docs: default_dev_docs_path(),
            binaries: default_binaries_path(),
            not_found: default_not_found_path(),
        }
    }
}

/// A base network policy that should apply to *all* images in ALL of Thorium
///
/// Because network policies in K8's are *additive*, the base policy should be
/// fairly restrictive to allow for other network policies to open up access.
/// Alternatively, to bypass Thorium's network policy functionality altogether
/// and allow full access for all tools, you can provide a base network policy
/// with ingress and egress containing a rule with `allowed_all` set to `true`
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, JsonSchema)]
pub struct BaseNetworkPolicy {
    /// The name of the network policy
    ///
    /// The policy name must be a valid name in K8's unlike regular network policies
    /// (lowercase, alphanumeric except for -'s)
    ///
    /// This name only needs to be unique among other base network policies, meaning
    /// regular policies can have this same name if desired. This is because regular
    /// policies have UUID's appended to them to prevent name collisions in K8's
    pub name: String,
    /// The rules for ingress to all tools in Thorium
    ///
    /// If None, all ingress traffic is allowed; if empty, no ingress traffic is allowed
    #[serde(default)]
    pub ingress: Option<Vec<NetworkPolicyRuleRaw>>,
    /// The rules for egress from all tools in Thorium
    ///
    /// If None, all egress traffic is allowed; if empty, no egress traffic is allowed
    #[serde(default)]
    pub egress: Option<Vec<NetworkPolicyRuleRaw>>,
}

impl Default for BaseNetworkPolicy {
    /// Provide a default base network policy
    ///
    /// The default base network policy in Thorium blocks all ingress traffic
    /// except from the Thorium API and blocks all egress traffic except to the
    /// Thorium API on all ports and to the K8's `CoreDNS` and `NodeLocalDNS`
    /// services and all local addresses on UDP port 53 which might be needed to
    /// resolve the Thorium API
    fn default() -> Self {
        Self {
            // give a name to our default policy; other policies can have this name because
            // we'll match with the value "default" rather than the value "true" as we do for
            // regular policies
            name: "thorium-default".to_string(),
            // allow in from the Thorium API
            ingress: Some(vec![NetworkPolicyRuleRaw::default().custom_rule(
                NetworkPolicyCustomK8sRule {
                    // select the Thorium namespace by its kubernetes metadata name
                    namespace_labels: Some(vec![NetworkPolicyCustomLabel::new(
                        "kubernetes.io/metadata.name",
                        "thorium",
                    )]),
                    // select the Thorium API by its label in a default deployment
                    pod_labels: Some(vec![NetworkPolicyCustomLabel::new("app", "api")]),
                },
            )]),
            // allow out to the Thorium API on all ports
            egress: Some(vec![
                NetworkPolicyRuleRaw::default().custom_rule(NetworkPolicyCustomK8sRule {
                    // select the Thorium namespace by its kubernetes metadata name
                    namespace_labels: Some(vec![NetworkPolicyCustomLabel::new(
                        "kubernetes.io/metadata.name",
                        "thorium",
                    )]),
                    // select the Thorium API by its label in a default deployment
                    pod_labels: Some(vec![NetworkPolicyCustomLabel::new("app", "api")]),
                }),
                // allow out to K8's/link-local DNS on port 53
                NetworkPolicyRuleRaw::default()
                    // allow link-local
                    .ip_block("169.254.0.0/16", None)
                    .ip_block("fe80::/10", None)
                    .custom_rule(NetworkPolicyCustomK8sRule {
                        // select the kube-system namespace by its kubernetes metadata name
                        namespace_labels: Some(vec![NetworkPolicyCustomLabel::new(
                            "kubernetes.io/metadata.name",
                            "kube-system",
                        )]),
                        // select the CoreDNS service by its special label
                        pod_labels: Some(vec![NetworkPolicyCustomLabel::new(
                            "k8s-app", "kube-dns",
                        )]),
                    })
                    .custom_rule(NetworkPolicyCustomK8sRule {
                        // select the kube-system namespace by its kubernetes metadata name
                        namespace_labels: Some(vec![NetworkPolicyCustomLabel::new(
                            "kubernetes.io/metadata.name",
                            "kube-system",
                        )]),
                        // select the NodeLocalDNS service by its special label
                        pod_labels: Some(vec![NetworkPolicyCustomLabel::new(
                            "k8s-app",
                            "node-local-dns",
                        )]),
                    })
                    .port(53, None, Some(NetworkProtocol::UDP)),
            ]),
        }
    }
}

/// Provide a default base network policy if none are given
fn default_base_network_policies() -> Vec<BaseNetworkPolicy> {
    vec![BaseNetworkPolicy::default()]
}

/// Serde default listen interface for the API
fn default_api_interface() -> String {
    "0.0.0.0".to_owned()
}

/// Serde default listen port for the API
fn default_api_port() -> u16 {
    80
}

/// Serde default kubernetes namespace for Thorium pods
fn default_namespace() -> String {
    "thorium".to_owned()
}

/// Provide a default set of namespaces to not allow Thorium to create
fn default_namespace_blacklist() -> HashSet<String> {
    [
        default_namespace(),
        "scylla".to_string(),
        "scylla-operator".to_string(),
        "cert-manager".to_string(),
        "redis".to_string(),
        "elastic-system".to_string(),
        "jaeger".to_string(),
        "quickwit".to_string(),
    ]
    .into_iter()
    .collect()
}

/// Thorium configs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Thorium {
    /// The interface to bind onto
    #[serde(default = "default_api_interface")]
    pub interface: String,
    /// The port to bind to
    #[serde(default = "default_api_port")]
    pub port: u16,
    /// The namespace to use in the backend
    #[serde(default = "default_namespace")]
    pub namespace: String,
    /// The secret key used for generating secrets
    /// and for bootstrapping with the operator
    /// make sure this is a longer string and its protected
    pub secret_key: String,
    /// The tracing settings to use
    #[serde(default)]
    pub tracing: Tracing,
    /// The settings for tags
    #[serde(default)]
    pub tags: Tags,
    /// The settings for saving/Carting files
    #[serde(default)]
    pub files: Files,
    /// The settings for results
    #[serde(default)]
    pub results: Results,
    /// The settings for ephemeral files
    #[serde(default)]
    pub ephemeral: Ephemeral,
    /// The settings for attachments
    #[serde(default)]
    pub attachments: Attachments,
    /// The settings for repos
    #[serde(default)]
    pub repos: Repos,
    /// The settings related to events
    #[serde(default)]
    pub events: Events,
    /// Base network policies that should be applied to *all* tools in Thorium
    ///
    /// If none are supplied, a default policy will be applied instead (see
    /// [`BaseNetworkPolicy::default`])
    ///
    /// These policies are not stored in the database, and are simply applied to
    /// every tool in every group by the scaler; [`thorium::models::NetworkPolicy`]s
    /// with the same name as `BaseNetworkPolicy`s are valid, as the two policy types
    /// are named differently in K8's. See [`BaseNetworkPolicy`] for more info
    #[serde(default = "default_base_network_policies")]
    pub base_network_policies: Vec<BaseNetworkPolicy>,
    /// The settings used to write objects to s3
    pub s3: S3,
    /// how long data of various types should be retained in Thorium
    #[serde(default)]
    pub retention: Retention,
    /// The settings to use to configure CORS
    #[serde(default)]
    pub cors: Cors,
    /// The authentication settings to use
    #[serde(default)]
    pub auth: Auth,
    /// The settings for the scaler
    #[serde(default)]
    pub scaler: Scaler,
    /// The settings for the search-streamer
    #[serde(default)]
    pub search_streamer: SearchStreamer,
    /// The request size limits to use in the API
    #[serde(default)]
    pub request_size_limits: RequestSizeLimits,
    /// The path to the Thorium docs to serve
    #[serde(default)]
    pub assets: Assets,
    /// A list of namespaces/groups that cannot be created by Thorium or its users
    #[serde(default = "default_namespace_blacklist")]
    pub namespace_blacklist: HashSet<String>,
}

/// Cross origin request settings
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, JsonSchema)]
pub struct Cors {
    /// Whether to allow CORS requests from any domain
    #[serde(default = "default_false")]
    pub insecure: bool,
    /// The domains to allow cross origin requests from
    #[serde(default)]
    pub domains: Vec<String>,
}

/// Redis settings
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Redis {
    /// The host redis is reachable at
    pub host: String,
    /// The port redis is bound to
    pub port: u16,
    /// The number of connections to have in the connection pool
    pub pool_size: Option<u32>,
    /// A username to use if redis has authentication enabled
    pub username: Option<String>,
    /// A password to use if redis has authentication enabled
    pub password: Option<String>,
}

/// Helps serde default the amount of time for scylla to get setup
fn default_scylla_setup_time() -> u32 {
    120
}

/// The authentication settings to use with scylla
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct ScyllaAuth {
    /// The username to use when authenticating
    pub username: String,
    /// The password to use when authenticating
    pub password: String,
}

/// Scylla settings
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Scylla {
    /// The list of nodes to connect to
    pub nodes: Vec<String>,
    /// The replication factor to use
    pub replication: u64,
    /// The amount of time to wait for scylla to get setup
    #[serde(default = "default_scylla_setup_time")]
    pub setup_time: u32,
    /// The auth creds to use when authenticating to scylla
    pub auth: Option<ScyllaAuth>,
}

/// Adds a "thorium_" namespace to given elastic index
fn elastic_namespace(index: &str) -> String {
    format!("thorium_{index}")
}

/// Helps serde default the index to query for results in Elastic
fn default_elastic_sample_results_index() -> String {
    elastic_namespace("sample_results")
}

/// Helps serde default the index to query for results in Elastic
fn default_elastic_repo_results_index() -> String {
    elastic_namespace("repo_results")
}

/// Helps serde default the index to query for sample tags in Elastic
fn default_elastic_sample_tags_index() -> String {
    elastic_namespace("sample_tags")
}

/// Helps serde default the index to query for repo tags in Elastic
fn default_elastic_repo_tags_index() -> String {
    elastic_namespace("repo_tags")
}

/// Helps serde default Elastic indexes' max analyzed offset
///
/// Default to 1,000,000 characters according to the
/// [Elastic docs](https://www.elastic.co/docs/reference/elasticsearch/index-settings/index-modules)
fn default_max_analyzed_offset() -> u32 {
    1_000_000
}

/// Scylla settings
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Elastic {
    /// The node to connect to
    pub node: String,
    /// The username to use when authenticating
    pub username: String,
    /// The password to use when authenticating
    pub password: String,
    /// The options for results in elastic
    #[serde(default)]
    pub results: ElasticResults,
    /// The options for tags in elastic
    #[serde(default)]
    pub tags: ElasticTags,
    /// The max analyzed offset for highlighting configure in the index
    /// and for every query
    ///
    /// This avoids errors when fields are larger than the configured maximum
    /// and truncates highlighting instead
    #[serde(default = "default_max_analyzed_offset")]
    pub max_analyzed_offset: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct ElasticResults {
    /// The name of the sample results index in elastic
    #[serde(default = "default_elastic_sample_results_index")]
    pub samples: String,
    /// The name of the repo results index in elastic
    #[serde(default = "default_elastic_repo_results_index")]
    pub repos: String,
}

impl Default for ElasticResults {
    fn default() -> Self {
        Self {
            samples: default_elastic_sample_results_index(),
            repos: default_elastic_repo_results_index(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct ElasticTags {
    /// The name of the sample tags index in elastic
    #[serde(default = "default_elastic_sample_tags_index")]
    pub samples: String,
    /// The name of the repo tags index in elastic
    #[serde(default = "default_elastic_repo_tags_index")]
    pub repos: String,
}

impl Default for ElasticTags {
    fn default() -> Self {
        Self {
            samples: default_elastic_sample_tags_index(),
            repos: default_elastic_repo_tags_index(),
        }
    }
}

/// configs for Thorium
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema)]
pub struct Conf {
    /// Allow scylla nodes to easily be overwritten with a single node for testing
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scylla_override: Option<String>,
    /// Thorium configs
    pub thorium: Thorium,
    /// Redis settings
    pub redis: Redis,
    /// Scylla settings
    pub scylla: Scylla,
    // Elastic Search settings
    pub elastic: Elastic,
}

impl Conf {
    /// Creates a new [Conf] object
    ///
    /// # Arguments
    ///
    /// * `path` - The path to use when reading the config file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        let mut conf: Conf = config::Config::builder()
            // load from a file first
            .add_source(config::File::from(path.as_ref()).format(config::FileFormat::Yaml))
            // then overlay any environment args on top
            .add_source(
                config::Environment::with_prefix("thorium")
                    .prefix_separator("__")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()?;
        // allow the override of the scylla node list to make testing easier
        if let Some(node) = conf.scylla_override.take() {
            conf.scylla.nodes = node
                .split(',')
                .map(std::borrow::ToOwned::to_owned)
                .collect();
        }
        Ok(conf)
    }

    /// Change the namespace for this config
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace for this config
    #[must_use]
    pub fn namespace<T: Into<String>>(mut self, namespace: T) -> Self {
        // update this configs namespace
        self.thorium.namespace = namespace.into();
        self
    }
}
