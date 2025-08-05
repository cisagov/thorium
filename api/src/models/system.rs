use chrono::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::ops::Add;
use std::path::PathBuf;
use uuid::Uuid;

use crate::models::conversions;
use crate::{matches_adds, matches_removes, matches_update};

use super::{
    Group, GroupStats, Image, ImageScaler, InvalidEnum, Pipeline, Requisition, Resources, User,
};

/// The default IFF to use when initializing Thorium
pub const DEFAULT_IFF: &str = "Thorium";
/// The Redis key that signals whether the K8's cache needs to be updated
pub const K8S_CACHE_KEY: &str = "k8s_cache";
/// The Redis key that signals whether the bare metal cache needs to be updated
pub const BARE_METAL_CACHE_KEY: &str = "bare_metal_cache";
/// The Redis key that signals whether the Windows cache needs to be updated
pub const WINDOWS_CACHE_KEY: &str = "windows_cache";
/// The Redis key that signals whether the KVM cache needs to be updated
pub const KVM_CACHE_KEY: &str = "kvm_cache";
/// The Redis key that signals whether the external cache needs to be updated
pub const EXTERNAL_CACHE_KEY: &str = "external_cache";

/// The query params for getting system info
#[derive(Deserialize, Serialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemInfoParams {
    /// Whether to reset any system info flags
    pub reset: Option<ImageScaler>,
}

/// Info about data in the current backend
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemInfo {
    /// Whether the k8s scaler cache needs to be updated
    pub k8s_cache: bool,
    /// Whether the bare_metal scaler cache needs to be updated
    pub bare_metal_cache: bool,
    /// Whether the windows scaler cache needs to be updated
    pub windows_cache: bool,
    /// Whether this kvm scaler cache needs to be updated
    pub kvm_cache: bool,
    /// Whether the external scaler cache needs to be updated
    pub external_cache: bool,
}

impl SystemInfo {
    /// Check whether our scalers cache needs to be updated
    #[must_use]
    pub fn expired_cache(&self, scaler: ImageScaler) -> bool {
        match scaler {
            ImageScaler::K8s => self.k8s_cache,
            ImageScaler::BareMetal => self.bare_metal_cache,
            ImageScaler::Windows => self.windows_cache,
            ImageScaler::Kvm => self.kvm_cache,
            ImageScaler::External => self.external_cache,
        }
    }
}

/// The number of jobs running and in the deadline queue for each scaler
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ScalerStats {
    /// The number of jobs currently in the deadline queue
    pub deadlines: i64,
    /// The number of jobs currently running
    pub running: i64,
}

impl ScalerStats {
    #[must_use]
    pub fn new(deadlines: i64, running: i64) -> Self {
        ScalerStats { deadlines, running }
    }
}

/// A map of spawned requisitions
pub type SpawnMap<'a> = HashMap<&'a String, BTreeMap<u64, Vec<(Requisition, u64)>>>;

/// Statistics about the current state of Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemStats {
    /// The total number of deadlines currently in the system across all scalers
    pub deadlines: i64,
    /// The total number of running jobs currently in the system across all scalers
    pub running: i64,
    /// The number of users currently in the system
    pub users: i64,
    /// The stats for jobs under the k8s scaler
    pub k8s: ScalerStats,
    /// The stats for jobs under the baremetal scaler
    pub baremetal: ScalerStats,
    /// The stats for jobs under the external scaler
    pub external: ScalerStats,
    /// Detailed stats reports for each group
    pub groups: HashMap<String, GroupStats>,
}

impl SystemStats {
    /// gets a total count for the number of stages in use by all groups, pipelines and users
    #[must_use]
    pub fn total(&self) -> usize {
        // add all the number of stage up for each group, user and pipeline
        self.groups.values().map(|map| map.total()).sum()
    }

    /// Builds a vector of unique names for each of the group/user/pipeline/image combos
    #[must_use]
    pub fn unique(&self) -> HashSet<String> {
        // build an empty list to store our unique names
        let mut unique = HashSet::with_capacity(self.groups.len() * 10);
        // crawl over each group
        for (group, pipeline_map) in &self.groups {
            // crawl over each pipeline and its user map
            for (pipeline, user_map) in &pipeline_map.pipelines {
                // crawl over each user and its stage map
                for (user, stage_map) in &user_map.stages {
                    // crawl over each stage
                    for (stage, _) in stage_map.iter() {
                        unique.insert(format!("{group}:{pipeline}:{user}:{stage}"));
                    }
                }
            }
        }
        unique
    }

    /// Gets this users image requests sorted from small to large
    #[must_use]
    pub fn users_jobs(&self) -> SpawnMap {
        // build a map of all users jobs
        let mut map: SpawnMap = HashMap::default();
        // crawl over each group
        for (group, pipeline_map) in &self.groups {
            // crawl over each pipeline and its user map
            for (pipeline, user_map) in &pipeline_map.pipelines {
                // crawl over the pipelines by user
                for (stage, pipe_stats) in &user_map.stages {
                    // add these stages to our map based on the number of created + started jobs
                    for (user, stage_stats) in pipe_stats {
                        // get an entry to this users map
                        let user_entry = map.entry(user).or_default();
                        // build a requisition for this image
                        let req = Requisition::new(user, group, pipeline, stage);
                        // get an entry to this images rank group and add it
                        let entry = user_entry.entry(stage_stats.running).or_default();
                        entry.push((req, stage_stats.created));
                    }
                }
            }
        }
        map
    }
}

// TODO: remove once serde allows for default values or we move this to a helper function
/// Helps serde default a value to true
const fn default_true() -> bool {
    true
}

/// The params for resetting system settings
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemSettingsResetParams {
    /// Whether to run the consistency scan after resetting system settings
    #[serde(default = "default_true")]
    pub scan: bool,
}

impl Default for SystemSettingsResetParams {
    /// Provide default `SystemSettingsResetParams`
    ///
    /// The consistency scan is run by default
    fn default() -> Self {
        Self { scan: true }
    }
}

impl SystemSettingsResetParams {
    /// Skip the consistency scan after resetting system settings
    #[must_use]
    pub fn no_scan(mut self) -> Self {
        self.scan = false;
        self
    }
}

/// The params for updating system settings
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemSettingsUpdateParams {
    /// Whether to run the consistency scan after updating system settings
    #[serde(default = "default_true")]
    pub scan: bool,
}

impl Default for SystemSettingsUpdateParams {
    /// Provide default `SystemSettingsUpdateParams`
    ///
    /// The consistency scan is run by default
    fn default() -> Self {
        Self { scan: true }
    }
}

impl SystemSettingsUpdateParams {
    /// Skip the consistency scan after updating system settings
    #[must_use]
    pub fn no_scan(mut self) -> Self {
        self.scan = false;
        self
    }
}

/// An update to the host path whitelist in Thorium [`SystemSettings`]
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct HostPathWhitelistUpdate {
    /// The set of paths to add to the whitelist
    #[serde(default)]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub add_paths: HashSet<PathBuf>,
    /// The set of paths to remove from the whitelist
    #[serde(default)]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub remove_paths: HashSet<PathBuf>,
}

impl HostPathWhitelistUpdate {
    /// Add a path to add to to the host path whitelist
    ///
    /// # Arguments
    ///
    /// * `paths` - The paths to add
    #[must_use]
    pub fn add_path<T: Into<PathBuf>>(mut self, path: T) -> Self {
        self.add_paths.insert(path.into());
        self
    }

    /// Add multiple paths to add to the host path whitelist
    ///
    /// # Arguments
    ///
    /// * `path` - The path to add
    #[must_use]
    pub fn add_paths<I, T>(mut self, paths: I) -> Self
    where
        T: Into<PathBuf>,
        I: IntoIterator<Item = T>,
    {
        self.add_paths.extend(paths.into_iter().map(Into::into));
        self
    }

    /// Remove a path to add to to the host path whitelist
    ///
    /// # Arguments
    ///
    /// * `path` - The path to remove
    #[must_use]
    pub fn remove_path<T: Into<PathBuf>>(mut self, path: T) -> Self {
        self.remove_paths.insert(path.into());
        self
    }

    /// Add multiple paths to remove from the host path whitelist
    ///
    /// # Arguments
    ///
    /// * `paths` - The paths to remove
    #[must_use]
    pub fn remove_paths<I, T>(mut self, paths: I) -> Self
    where
        T: Into<PathBuf>,
        I: IntoIterator<Item = T>,
    {
        self.remove_paths.extend(paths.into_iter().map(Into::into));
        self
    }
}

/// An update to Thorium's dynamic [`SystemSettings`]
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemSettingsUpdate {
    /// The amount of millicpu to reserve for things outside of Thorium
    pub reserved_cpu: Option<String>,
    /// The amount of memory to reserve for things outside of Thorium
    pub reserved_memory: Option<String>,
    /// The amount of ephemeral storage to reserve for things outside of Thorium
    pub reserved_storage: Option<String>,
    /// The amount of millicpu to use in the fair share pass if possible
    pub fairshare_cpu: Option<String>,
    /// The amount of memory to use in the fair share pass if possible
    pub fairshare_memory: Option<String>,
    /// The amount of ephemeral storage to use in the fair share pass if possible
    pub fairshare_storage: Option<String>,
    /// An update to the host path whitelist
    #[serde(default)]
    pub host_path_whitelist: HostPathWhitelistUpdate,
    #[serde(default)]
    pub clear_host_path_whitelist: bool,
    /// Allow users to create any host path
    pub allow_unrestricted_host_paths: Option<bool>,
}

impl SystemSettingsUpdate {
    /// Sets the amount of reserved CPU in millicpu or cores that cannot be used by Thorium
    ///
    /// If you place an m at the end it will be in millicpu while no unit will be treated as whole
    /// cores.
    ///
    /// # Arguments
    ///
    /// * `cpu` - The amount of cpu to reserve
    #[must_use]
    pub fn reserved_cpu<T: Into<String>>(mut self, cpu: T) -> Self {
        self.reserved_cpu = Some(cpu.into());
        self
    }

    /// Sets the amount of reserved memory that cannot be used by Thorium
    ///
    /// If no unit is specified then bytes are assumed.
    ///
    /// # Arguments
    ///
    /// * `memory` - The amount of memory to reserve
    #[must_use]
    pub fn reserved_memory<T: Into<String>>(mut self, memory: T) -> Self {
        self.reserved_memory = Some(memory.into());
        self
    }

    /// Sets the amount of reserved ephemeral storage that cannot be used by Thorium
    ///
    /// If no unit is specified then bytes are assumed.
    ///
    /// # Arguments
    ///
    /// * `reserved_storage` - The amount of ephemeral storage to reserve
    #[must_use]
    pub fn reserved_storage<T: Into<String>>(mut self, storage: T) -> Self {
        self.reserved_storage = Some(storage.into());
        self
    }

    /// Sets the amount of CPU in millicpu or cores that will be fairly shared if possible
    ///
    /// If you place an m at the end it will be in millicpu while no unit will be treated as whole
    /// cores.
    ///
    /// # Arguments
    ///
    /// * `cpu` - The amount of cpu to use for fair share
    #[must_use]
    pub fn fairshare_cpu<T: Into<String>>(mut self, cpu: T) -> Self {
        self.fairshare_cpu = Some(cpu.into());
        self
    }

    /// Sets the amount of memory that will be fairly shared if possible
    ///
    /// If no unit is specified then bytes are assumed.
    ///
    /// # Arguments
    ///
    /// * `memory` - The amount of memory to use for fair share
    #[must_use]
    pub fn fairshare_memory<T: Into<String>>(mut self, memory: T) -> Self {
        self.fairshare_memory = Some(memory.into());
        self
    }

    /// Sets the amount of ephemeral storage that will be fairly shared if possible
    ///
    /// If no unit is specified then bytes are assumed.
    ///
    /// # Arguments
    ///
    /// * `storage` - The amount of ephemeral storage to use for fair share
    #[must_use]
    pub fn fairshare_storage<T: Into<String>>(mut self, storage: T) -> Self {
        self.fairshare_storage = Some(storage.into());
        self
    }

    /// Sets a new host path whitelist to replace existing one
    ///
    /// # Arguments
    ///
    /// * `host_path_whitelist` - The new whitelist to set
    #[must_use]
    pub fn host_path_whitelist(mut self, host_path_whitelist: HostPathWhitelistUpdate) -> Self {
        self.host_path_whitelist = host_path_whitelist;
        self
    }

    /// Clear the host path whitelist
    ///
    /// Overrides any other add/remove settings
    #[must_use]
    pub fn clear_host_path_whitelist(mut self) -> Self {
        self.clear_host_path_whitelist = true;
        self
    }

    /// Set whether to allow all host paths, ignoring the host path whitelist
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set
    #[must_use]
    pub fn allow_unrestricted_host_paths(mut self, value: bool) -> Self {
        self.allow_unrestricted_host_paths = Some(value);
        self
    }
}

/// Settings that can be dynamically changed in Thorium
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SystemSettings {
    /// The amount of millicpu to reserve for things outside of Thorium
    pub reserved_cpu: u64,
    /// The amount of memory to reserve for things outside of Thorium
    pub reserved_memory: u64,
    /// The amount of ephemeral storage to reserve for things outside of Thorium
    pub reserved_storage: u64,
    /// The amount of millicpu to use in the fair share pass if possible
    pub fairshare_cpu: u64,
    /// The amount of memory to use in the fair share pass if possible
    pub fairshare_memory: u64,
    /// The amount of ephemeral storage to use in the fair share pass if possible
    pub fairshare_storage: u64,
    /// A whitelist of host paths users can mount in their tools with no admin intervention
    ///
    /// Users can mount to any of the paths in the whitelist or to a path whose parent is
    /// in the whitelist.
    #[serde(default)]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub host_path_whitelist: HashSet<PathBuf>,
    /// Allow users to create any host path, ignoring the whitelist; defaults to false
    #[serde(default)]
    pub allow_unrestricted_host_paths: bool,
}

impl PartialEq<SystemSettingsUpdate> for SystemSettings {
    /// Ensure all updates in a [`SystemSettingsUpdate`] were set
    ///
    /// # Arguments
    ///
    /// * `update` - The update to compare against
    #[rustfmt::skip]
    fn eq(&self, update: &SystemSettingsUpdate) -> bool {
        matches_update!(self.reserved_cpu, update.reserved_cpu, conversions::cpu);
        matches_update!(self.reserved_memory, update.reserved_memory, conversions::storage);
        matches_update!(self.reserved_storage, update.reserved_storage, conversions::storage);
        matches_update!(self.fairshare_cpu, update.fairshare_cpu, conversions::cpu);
        matches_update!(self.fairshare_memory, update.fairshare_memory, conversions::storage);
        matches_update!(self.fairshare_storage, update.fairshare_storage, conversions::storage);
        matches_adds!(self.host_path_whitelist, update.host_path_whitelist.add_paths);
        matches_removes!(self.host_path_whitelist, update.host_path_whitelist.remove_paths);
        matches_update!(self.allow_unrestricted_host_paths, update.allow_unrestricted_host_paths);
        true
    }
}

/// A struct containing a full backup of users/groups/images/pipelines of the server
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Backup {
    /// The system settings
    pub settings: SystemSettings,
    /// The users for this server
    pub users: Vec<User>,
    /// The groups on this server
    pub groups: Vec<Group>,
    /// The images on this server
    pub images: Vec<Image>,
    /// The pipelines on this server
    pub pipelines: Vec<Pipeline>,
}

/// An update for a specific streamer info
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StreamerInfoUpdate {
    /// The new most recent timestamp a streamer has tried to stream
    pub latest: Option<DateTime<Utc>>,
    /// A list of new cursor IDs this streamer is crawling
    pub add_cursors: Vec<Uuid>,
    /// a list of exhausted cursors this streamer is no longer using
    pub remove_cursors: Vec<Uuid>,
}

/// The information needed to register a node
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NodeRegistration {
    /// The cluster this node is in
    pub cluster: String,
    /// This nodes name
    pub name: String,
    /// The amount of resources this node has
    pub resources: Resources,
}

impl NodeRegistration {
    /// Create a new node registration object
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster this node is in
    /// * `name` - The name of this node
    /// * `resources` - The resources this node has
    pub fn new<C: Into<String>, N: Into<String>>(
        cluster: C,
        name: N,
        resources: Resources,
    ) -> Self {
        NodeRegistration {
            cluster: cluster.into(),
            name: name.into(),
            resources,
        }
    }
}

/// The current health of this node
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreJson))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum NodeHealth {
    /// This node is healthy and able to schedule jobs
    Healthy,
    /// This node is unhealthy and cannot schedule new jobs
    Unhealthy,
    /// This node has been deactivated
    Disabled(Option<String>),
    /// This node has been registered but has never completed a health check
    Registered,
}

impl std::fmt::Display for NodeHealth {
    /// Allow [`NodeHealth`] to be displayed
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NodeHealth::Healthy => write!(f, "Healthy"),
            NodeHealth::Unhealthy => write!(f, "Unhealthy"),
            NodeHealth::Disabled(_) => write!(f, "Disabled"),
            NodeHealth::Registered => write!(f, "Registered"),
        }
    }
}

/// Information regarding a single node in use by Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Node {
    /// The cluster this node is in
    pub cluster: String,
    /// This nodes name
    pub name: String,
    /// This nodes current health
    pub health: NodeHealth,
    /// The amount of resources this node has in total
    pub resources: Resources,
    /// The workers currently assigned to this node
    pub workers: HashMap<String, Worker>,
    /// The last time this node completed a health check
    pub heart_beat: Option<DateTime<Utc>>,
}

// A heartbeat for a nodes info
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NodeUpdate {
    /// The updated health of this node
    pub health: NodeHealth,
    /// The updated resources to set
    pub resources: Resources,
    /// Whether this update is a heart beat or not
    #[serde(default)]
    pub heart_beat: bool,
}

impl NodeUpdate {
    /// Create a new node update
    ///
    /// * `health` - The new health to set
    /// * `resources` - The new resources this node has available
    #[must_use]
    pub fn new(health: NodeHealth, resources: Resources) -> Self {
        NodeUpdate {
            health,
            resources,
            heart_beat: false,
        }
    }

    /// Set that this update should update the heart beat timestamp
    #[must_use]
    pub fn heart_beat(mut self) -> Self {
        // set our heart beat flag
        self.heart_beat = true;
        self
    }

    /// Set that this update should update the heart beat timestamp
    pub fn heart_beat_mut(&mut self) {
        // set our heart beat flag
        self.heart_beat = true;
    }
}

/// Helps serde default the node list scalers to all possible options
#[must_use]
pub fn default_scalers() -> Vec<ImageScaler> {
    vec![
        ImageScaler::K8s,
        ImageScaler::Windows,
        ImageScaler::BareMetal,
        ImageScaler::External,
    ]
}

/// The parameters for a node get request
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NodeGetParams {
    /// The scalers to limit our worker listing too
    #[serde(default = "default_scalers")]
    pub scalers: Vec<ImageScaler>,
}

impl Default for NodeGetParams {
    /// Create a default [`NodeGetParams`]
    fn default() -> Self {
        NodeGetParams {
            scalers: Vec::default(),
        }
    }
}
impl NodeGetParams {
    /// Add a new scaler to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler to add
    #[must_use]
    pub fn scaler(mut self, scaler: ImageScaler) -> Self {
        // add our scaler
        self.scalers.push(scaler);
        self
    }

    /// Adds new scalers to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler to add
    #[must_use]
    pub fn scalers(mut self, scaler: &[ImageScaler]) -> Self {
        // add our scalers
        self.scalers.extend(scaler);
        self
    }
}

/// Helps serde default the node list limit to 50
fn default_list_page_size() -> usize {
    50
}

/// The parameters for a node list request
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NodeListParams {
    /// The cursor id to user if one exists
    pub cursor: Option<Uuid>,
    /// The max amount of data to retrieve on a single page
    #[serde(default = "default_list_page_size")]
    pub page_size: usize,
    /// The max amount of nodes to list in this cursor
    pub limit: Option<usize>,
    /// The clusters to limit our listing too
    #[serde(default)]
    pub clusters: Vec<String>,
    /// The nodes to limit our listing too
    #[serde(default)]
    pub nodes: Vec<String>,
    /// The scalers to limit our worker listing too
    #[serde(default = "default_scalers")]
    pub scalers: Vec<ImageScaler>,
}

impl Default for NodeListParams {
    /// Create a default [`NodeListParams`]
    fn default() -> Self {
        NodeListParams {
            cursor: None,
            page_size: default_list_page_size(),
            limit: None,
            clusters: Vec::default(),
            nodes: Vec::default(),
            scalers: Vec::default(),
        }
    }
}

impl NodeListParams {
    /// Set the cursor for listing the next page of nodes
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor to set
    #[must_use]
    pub fn cursor(mut self, cursor: Uuid) -> Self {
        // set the cursor id
        self.cursor = Some(cursor);
        self
    }

    /// Set the max number of nodes to list on a single page
    ///
    /// # Arguments
    ///
    /// * `page_size` - The page size to set
    #[must_use]
    pub fn page_size(mut self, page_size: usize) -> Self {
        // set the page_size
        self.page_size = page_size;
        self
    }

    /// Set the max number of nodes to list in this cursor
    ///
    /// # Arguments
    ///
    /// * `limit` - The limit to set
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        // set the limit
        self.limit = Some(limit);
        self
    }

    /// Add a new cluster to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster to add
    #[must_use]
    pub fn cluster<C: Into<String>>(mut self, cluster: C) -> Self {
        // add our cluster
        self.clusters.push(cluster.into());
        self
    }

    /// Adds new clusters to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `cluster` - The clusters to add
    #[must_use]
    pub fn clusters<C: Into<String>>(mut self, cluster: Vec<C>) -> Self {
        // add our clusters
        self.clusters.extend(cluster.into_iter().map(Into::into));
        self
    }

    /// Add a new node to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `node` - The node to add
    #[must_use]
    pub fn node<N: Into<String>>(mut self, node: N) -> Self {
        // add our node
        self.nodes.push(node.into());
        self
    }

    /// Adds new nodes to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `node` - The nodes to add
    #[must_use]
    pub fn nodes<N: Into<String>>(mut self, node: impl Iterator<Item = N>) -> Self {
        // add our nodes
        self.nodes.extend(node.into_iter().map(Into::into));
        self
    }

    /// Add a new scaler to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler to add
    #[must_use]
    pub fn scaler(mut self, scaler: ImageScaler) -> Self {
        // add our scaler
        self.scalers.push(scaler);
        self
    }

    /// Adds new scalers to limit our listing too
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler to add
    #[must_use]
    pub fn scalers(mut self, scaler: &[ImageScaler]) -> Self {
        // add our scalers
        self.scalers.extend(scaler);
        self
    }
}

/// A list of nodes and their clusters
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(
    feature = "scylla-utils",
    derive(scylla::DeserializeRow, scylla::SerializeRow)
)]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NodeListLine {
    /// The cluster this node is from
    pub cluster: String,
    /// The name of this node
    pub node: String,
}

/// The different types of pools in the scaler
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(valuable::Valuable))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum Pools {
    /// The fair share pool
    FairShare,
    /// The deadline pool
    Deadline,
}

impl Pools {
    /// Cast our pool kind to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Pools::FairShare => "FairShare",
            Pools::Deadline => "Deadline",
        }
    }
}

#[cfg(feature = "client")]
impl TryFrom<&str> for Pools {
    type Error = crate::client::Error;

    /// Get our pool kind from a str
    ///
    /// # Arguments
    ///
    /// * `raw` - The str to get our pool from
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "FairShare" => Ok(Pools::FairShare),
            "Deadline" => Ok(Pools::Deadline),
            _ => Err(crate::client::Error::new(format!(
                "Unknown pool kind: {value}",
            ))),
        }
    }
}

/// The info needed to register this worker
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct WorkerRegistration {
    /// The cluster this worker is assigned to
    pub cluster: String,
    /// The node this worker is on
    pub node: String,
    /// The unique name of this worker
    pub name: String,
    /// The user this worker is executing a job for
    pub user: String,
    /// The group this worker is executing a job in
    pub group: String,
    /// The pipeline this worker is executing a job in
    pub pipeline: String,
    /// The stage this worker is executing a job for
    pub stage: String,
    /// The resources used to spawn this worker
    pub resources: Resources,
    /// The pool this worker was spawned in
    pub pool: Pools,
}

impl WorkerRegistration {
    /// Create a new worker registration object
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster this worker is assigned to
    /// * `node` - The node this worker is on
    /// * `name` - The unique name for this worker
    pub fn new<
        C: Into<String>,
        N: Into<String>,
        T: Into<String>,
        U: Into<String>,
        G: Into<String>,
        P: Into<String>,
        S: Into<String>,
    >(
        cluster: C,
        node: N,
        name: T,
        user: U,
        group: G,
        pipeline: P,
        stage: S,
        resources: Resources,
        pool: Pools,
    ) -> Self {
        WorkerRegistration {
            cluster: cluster.into(),
            node: node.into(),
            name: name.into(),
            user: user.into(),
            group: group.into(),
            pipeline: pipeline.into(),
            stage: stage.into(),
            resources,
            pool,
        }
    }
}

impl From<Worker> for WorkerRegistration {
    /// Convert a worker to a worker registration object
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker object to convert
    fn from(worker: Worker) -> Self {
        WorkerRegistration {
            cluster: worker.cluster,
            node: worker.node,
            name: worker.name,
            user: worker.user,
            group: worker.group,
            pipeline: worker.pipeline,
            stage: worker.stage,
            resources: worker.resources,
            pool: worker.pool,
        }
    }
}

/// The possible statuses of workers
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum WorkerStatus {
    /// This worker is being setup
    Spawning,
    /// This worker is currently executing jobs
    Running,
    /// This worker is being ordered to shutdown
    Shutdown,
}

impl WorkerStatus {
    /// Convert this status to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            WorkerStatus::Spawning => "Spawning",
            WorkerStatus::Running => "Running",
            WorkerStatus::Shutdown => "Shutdown",
        }
    }
}

impl Display for WorkerStatus {
    /// Display this worker's status
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter in use
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // write this as a str
        write!(f, "{}", self.as_str())
    }
}

// only implement the CQL value conversion if the API feature is enabled
impl TryFrom<&String> for WorkerStatus {
    type Error = InvalidEnum;
    // try to convert our string to a [`WorkerStatus`]
    fn try_from(raw: &String) -> Result<Self, Self::Error> {
        match raw.as_str() {
            "Spawning" => Ok(WorkerStatus::Spawning),
            "Running" => Ok(WorkerStatus::Running),
            "Shutdown" => Ok(WorkerStatus::Shutdown),
            _ => Err(InvalidEnum(format!("Unknown Worker Status: {raw}",))),
        }
    }
}

/// The current active job info for a worker
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ActiveJob {
    /// The reaction this worker is executing a job in
    pub reaction: Uuid,
    /// The job this worker is executing
    pub job: Uuid,
}

/// A active worker for a specific cluster and node
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Worker {
    /// The cluster this worker is assigned to
    pub cluster: String,
    /// The node this worker is on
    pub node: String,
    /// The scaler this worker was spawned under
    pub scaler: ImageScaler,
    /// The name of this worker
    pub name: String,
    /// The user this worker is executing a job for
    pub user: String,
    /// The group this worker is executing a job in
    pub group: String,
    /// The pipeline this worker is executing a job in
    pub pipeline: String,
    /// The stage this worker is executing a job for
    pub stage: String,
    /// The current status of this worker
    pub status: WorkerStatus,
    /// When this worker was spawned
    pub spawned: DateTime<Utc>,
    /// The last time this worker checked in with Thorium
    pub heart_beat: Option<DateTime<Utc>>,
    /// The resources used to spawn this worker
    pub resources: Resources,
    /// The pool this worker was spawned in
    pub pool: Pools,
    /// The current active job info for this worker if it has one
    pub active: Option<ActiveJob>,
}

/// A list of all active workers in Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkerList {
    /// The currently active workers for this scaler
    pub workers: Vec<Worker>,
}

/// The workers to add or remove from scylla
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct WorkerRegistrationList {
    /// The workers to add or remove from scylla
    pub workers: Vec<WorkerRegistration>,
}

impl WorkerRegistrationList {
    /// Adds a new worker to this registration list
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster this worker is assigned to
    /// * `node` - The node this worker is on
    /// * `name` - The unique name for this worker
    #[must_use]
    pub fn add<
        C: Into<String>,
        N: Into<String>,
        T: Into<String>,
        U: Into<String>,
        G: Into<String>,
        P: Into<String>,
        S: Into<String>,
    >(
        mut self,
        cluster: C,
        node: N,
        name: T,
        user: U,
        group: G,
        pipeline: P,
        stage: S,
        resources: Resources,
        pool: Pools,
    ) -> Self {
        // build our worker
        let worker = WorkerRegistration {
            cluster: cluster.into(),
            node: node.into(),
            name: name.into(),
            user: user.into(),
            group: group.into(),
            pipeline: pipeline.into(),
            stage: stage.into(),
            resources,
            pool,
        };
        // add this worker
        self.workers.push(worker);
        self
    }

    /// Adds a new worker to this registration list
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to add or remove
    pub fn add_mut(&mut self, worker: WorkerRegistration) {
        self.workers.push(worker);
    }
}

impl Add<WorkerRegistration> for WorkerRegistrationList {
    type Output = Self;
    /// Adds a new worker to add or remove
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to add or remove
    fn add(mut self, worker: WorkerRegistration) -> Self {
        self.workers.push(worker);
        self
    }
}

impl From<WorkerList> for WorkerRegistrationList {
    /// Convert a list of workers to a list of workers to add or remove
    ///
    /// # Arguments
    ///
    /// * `workers` - The workers to add or remove
    fn from(list: WorkerList) -> Self {
        // convert our list of workers to a list of worker registrations
        let workers = list
            .workers
            .into_iter()
            .map(WorkerRegistration::from)
            .collect::<Vec<WorkerRegistration>>();
        // build our worker registration list object
        WorkerRegistrationList { workers }
    }
}

/// The updates to apply to a worker
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct WorkerUpdate {
    /// The new status to set
    pub status: WorkerStatus,
}

impl WorkerUpdate {
    /// Create a new worker update
    ///
    /// # Arguments
    ///
    /// * `status` - The new status to set
    #[must_use]
    pub fn new(status: WorkerStatus) -> Self {
        WorkerUpdate { status }
    }
}

/// A map of clusters to delete workers from
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct WorkerDeleteMap {
    /// The names of the workers to delete
    pub workers: Vec<String>,
}

impl WorkerDeleteMap {
    /// Create a worker map with some capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - The capacity to allocate
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        // instance our vec with the right capacity
        let workers = Vec::with_capacity(capacity);
        WorkerDeleteMap { workers }
    }

    /// Add a worker to be deleted
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the worker to delete
    #[must_use]
    pub fn add<W: Into<String>>(mut self, name: W) -> Self {
        // add our worker
        self.workers.push(name.into());
        self
    }

    /// Add a worker to be deleted
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the worker to delete
    pub fn add_mut<W: Into<String>>(&mut self, name: W) {
        // add our worker
        self.workers.push(name.into());
    }
}

impl From<HashMap<String, Worker>> for WorkerDeleteMap {
    /// Build a worker delete map from a map of workers
    fn from(workers: HashMap<String, Worker>) -> Self {
        // build a vec of deletes for our map
        let workers = workers.keys().cloned().collect();
        // build our worker delete map
        WorkerDeleteMap { workers }
    }
}

/// A worker to delete
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct WorkerDelete {
    /// The cluster this worker is from
    pub cluster: String,
    /// The node this worker is on
    pub node: String,
    /// The name of this worker
    pub name: String,
}

impl WorkerDelete {
    /// Add a worker to be deleted
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster to delete a worker from
    /// * `node` - The node to delete a worker from
    /// * `name` - The name of the worker to delete
    pub fn new<C: Into<String>, N: Into<String>, W: Into<String>>(
        cluster: C,
        node: N,
        name: W,
    ) -> Self {
        // build our worker delete
        WorkerDelete {
            cluster: cluster.into(),
            node: node.into(),
            name: name.into(),
        }
    }
}

/// The parameters for a worker list request
#[derive(Serialize, Deserialize, Debug)]
pub struct WorkerListParams {
    /// The cursor id to user if one exists
    pub cursor: Option<Uuid>,
    /// The max amount of data to retrieve on a single page
    #[serde(default = "default_list_page_size")]
    pub page_size: usize,
    /// The max amount of workers to list in this cursor
    pub limit: Option<usize>,
    /// The clusters to limit our listing too
    #[serde(default)]
    pub clusters: Vec<String>,
    /// The scalers to limit our worker listing too
    #[serde(default = "default_scalers")]
    pub scalers: Vec<ImageScaler>,
}

/// The different components in Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum SystemComponents {
    Api,
    Scaler(ImageScaler),
    EventHandler,
    SearchStreamer,
}
