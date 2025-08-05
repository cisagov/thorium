//! The arguments for the backups and data restorations in Thorium

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use thorium::models::{
    HostPathWhitelistUpdate, SystemSettingsResetParams, SystemSettingsUpdate,
    SystemSettingsUpdateParams,
};

/// Provide a default admin config path
fn default_cluster_conf_path() -> PathBuf {
    let mut default_admin_path = PathBuf::from(".");
    default_admin_path.push("thorium.yml");
    default_admin_path
}

/// Provide a default config path
fn default_ctl_conf_path() -> PathBuf {
    let mut default_config_path = dirs::home_dir().unwrap_or_default();
    default_config_path.push(".thorium");
    default_config_path.push("config.yml");
    default_config_path
}

/// The arguments for the backups and data restorations in Thorium
#[derive(Parser, Debug, Clone)]
#[clap(version, author)]
pub struct Args {
    /// The number of workers to use
    #[clap(short, long, default_value = "10")]
    pub workers: usize,
    /// The sub command to execute
    #[clap(subcommand)]
    pub cmd: SubCommands,
    /// The path to the config for the Thorium cluster to backup
    #[clap(short, long, default_value = default_cluster_conf_path().into_os_string())]
    pub cluster_conf: PathBuf,
    /// The path to load our Thorctl config from
    #[clap(long, default_value = default_ctl_conf_path().into_os_string())]
    pub ctl_conf: PathBuf,
}

/// The sub commands for our backup tool
#[derive(Parser, Debug, Clone)]
pub enum SubCommands {
    /// Backup a Thorium cluster
    #[clap(subcommand)]
    Backup(BackupSubCommands),
    /// Edit Thorium system settings
    #[clap(subcommand)]
    Settings(SettingsSubCommands),
    /// Provision Thorium resources including nodes
    #[clap(subcommand)]
    Provision(ProvisionSubCommands),
    /// Census commands in Thorium
    #[clap(subcommand)]
    Census(CensusSubCommands),
}

/// The backup specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum BackupSubCommands {
    /// Take a new backup
    #[clap(version, author)]
    New(NewBackup),
    /// Scrub a backup for bitrot
    Scrub(ScrubBackup),
    /// Restore a backup to a Thorium cluster
    #[clap(version, author)]
    Restore(RestoreBackup),
}

/// Define the default backup components
fn default_backup_components() -> Vec<BackupComponents> {
    // return all by default
    vec![BackupComponents::All]
}

/// The tables that can be backed up
// JEHAMZA_TODO: wish this was dynamic...
#[derive(Debug, Clone, Copy, strum::Display, ValueEnum, PartialEq, Eq, Hash)]
#[strum(serialize_all = "kebab-case")]
pub enum BackupComponents {
    All,
    CommentAttachments,
    Comments,
    Commitish,
    CommitishList,
    Nodes,
    Redis,
    RepoData,
    RepoList,
    ResultFiles,
    Results,
    ResultsStream,
    S3Ids,
    S3IdsObjects,
    SamplesList,
    Tags,
}

/// Backup a Thorium cluster
#[derive(Parser, Debug, Clone)]
pub struct NewBackup {
    /// The components to backup
    #[clap(value_enum, default_values_t = default_backup_components(), value_delimiter = ',')]
    pub components: Vec<BackupComponents>,
    /// Where to store our backups
    #[clap(short, long, default_value = "ThoriumBackups")]
    pub output: PathBuf,
    /// The chunk multiplier to use with our worker count
    #[clap(short, long, default_value = "100")]
    pub multiplier: u64,
}

/// Scrub a backup for bitrot
#[derive(Parser, Debug, Clone)]
pub struct ScrubBackup {
    /// The path to the backup to scrub
    #[clap(short, long)]
    pub backup: PathBuf,
}

/// Restore a backup to a specific Thorium cluster
#[derive(Parser, Debug, Clone)]
pub struct RestoreBackup {
    /// The path to the backup to restore
    #[clap(short, long)]
    pub backup: PathBuf,
}

/// The settings specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum SettingsSubCommands {
    /// Print the current Thorium system settings
    #[clap(version, author)]
    Get,
    /// Update Thorium system settings
    #[clap(version, author)]
    Update(UpdateSettings),
    /// Reset Thorium system settings to default
    #[clap(version, author)]
    Reset(ResetSettings),
    /// Run a manual consistency scan based on the current Thorium system settings
    #[clap(version, author)]
    Scan,
}

#[derive(Parser, Debug, Clone)]
pub struct UpdateSettings {
    /// Forego the automatic consistency scan after the settings update
    #[clap(long)]
    pub no_scan: bool,
    /// The options to set when updating Thorium `SystemSettings` (see [`thorium::models::SystemSettings`])
    #[clap(flatten)]
    pub settings_opts: SettingsOpts,
}

/// The set of possible updates to Thorium `SystemSettings` where at least one is set (see [`thorium::models::SystemSettings`])
#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = true)]
pub struct SettingsOpts {
    /// The amount of millicpu to reserve for things outside of Thorium
    #[clap(long)]
    pub reserved_cpu: Option<String>,
    /// The amount of memory to reserve for things outside of Thorium
    #[clap(long)]
    pub reserved_memory: Option<String>,
    /// The amount of ephemeral storage to reserve for things outside of Thorium
    #[clap(long)]
    pub reserved_storage: Option<String>,
    /// The amount of millicpu to use in the fair share pass if possible
    #[clap(long)]
    pub fairshare_cpu: Option<String>,
    /// The amount of memory to use in the fair share pass if possible
    #[clap(long)]
    pub fairshare_memory: Option<String>,
    /// The amount of ephemeral storage to use in the fair share pass if possible
    #[clap(long)]
    pub fairshare_storage: Option<String>,
    /// A list host paths to add to the host path whitelist
    #[clap(short = 'a', long, value_delimiter = ',')]
    pub host_path_whitelist_add: Vec<PathBuf>,
    /// A list of host paths to remove from the host path whitelist
    #[clap(short = 'r', long, value_delimiter = ',')]
    pub host_path_whitelist_remove: Vec<PathBuf>,
    /// Clear the host path whitelist
    #[clap(long)]
    pub clear_host_path_whitelist: bool,
    /// Allow users to create any host path, ignoring the whitelist
    #[clap(long)]
    pub allow_unrestricted_host_paths: Option<bool>,
}

impl UpdateSettings {
    /// Create a [`SystemSettingsUpdate`] from `self`
    pub fn to_settings_update(&self) -> SystemSettingsUpdate {
        // create the host path whitelist update
        let host_path_whitelist_update = HostPathWhitelistUpdate::default()
            .add_paths(self.settings_opts.host_path_whitelist_add.iter())
            .remove_paths(self.settings_opts.host_path_whitelist_remove.iter());
        // create a settings update from the options
        SystemSettingsUpdate {
            reserved_cpu: self.settings_opts.reserved_cpu.clone(),
            reserved_memory: self.settings_opts.reserved_memory.clone(),
            reserved_storage: self.settings_opts.reserved_storage.clone(),
            fairshare_cpu: self.settings_opts.fairshare_cpu.clone(),
            fairshare_memory: self.settings_opts.fairshare_memory.clone(),
            fairshare_storage: self.settings_opts.fairshare_storage.clone(),
            host_path_whitelist: host_path_whitelist_update,
            clear_host_path_whitelist: self.settings_opts.clear_host_path_whitelist,
            allow_unrestricted_host_paths: self.settings_opts.allow_unrestricted_host_paths,
        }
    }

    /// Create [`SystemSettingsUpdateParams`] from `self`
    pub fn to_params(&self) -> SystemSettingsUpdateParams {
        SystemSettingsUpdateParams {
            scan: !self.no_scan,
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct ResetSettings {
    /// Forego the automatic consistency scan after the settings reset
    #[clap(long)]
    pub no_scan: bool,
}

impl ResetSettings {
    /// Create [`SystemSettingsResetParams`] from `self`
    pub fn to_params(&self) -> SystemSettingsResetParams {
        SystemSettingsResetParams {
            scan: !self.no_scan,
        }
    }
}

/// Provision servers used by Thorium
#[derive(Parser, Debug, Clone)]
pub enum ProvisionSubCommands {
    /// Provision k8s or baremetal servers
    #[clap(version, author)]
    Node(ProvisionNode),
}

/// Provision a worker node for Thorium
#[derive(Parser, Debug, Clone)]
pub struct ProvisionNode {
    /// Target a k8s worker node
    #[clap(long, default_value = "true")]
    pub k8s: bool,
    /// Target a k8s worker node
    #[clap(short, long, default_value = "false")]
    pub baremetal: bool,
    /// Path to API keys file
    #[clap(short, long)]
    pub keys: String,
}

/// The census specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum CensusSubCommands {
    /// Take a new census
    #[clap(version, author)]
    New(NewCensus),
}

/// The different kinds of censuses to take
#[derive(Parser, Debug, Copy, Clone, clap::ValueEnum, PartialEq, Eq)]
pub enum CensusKinds {
    /// Take a census of all data
    All,
    /// Take a census of tag data
    Tags,
    /// Take a census of tag data case-insensitive
    TagsCaseInsensitive,
    /// Take a census of files data
    Files,
    /// Take a census of repo data
    Repos,
    /// Take a census of commitish data
    Commitishes,
}

impl std::fmt::Display for CensusKinds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            &CensusKinds::All => write!(f, "All"),
            &CensusKinds::Tags => write!(f, "Tags"),
            &CensusKinds::TagsCaseInsensitive => write!(f, "TagsCaseInsensitive"),
            &CensusKinds::Files => write!(f, "Files"),
            &CensusKinds::Repos => write!(f, "Repos"),
            &CensusKinds::Commitishes => write!(f, "Commitishes"),
        }
    }
}

/// Take a new census
#[derive(Parser, Debug, Clone)]
pub struct NewCensus {
    /// The data we should take a census of
    #[clap(default_value = "all")]
    pub census_kinds: Vec<CensusKinds>,
    /// The chunk multiplier to use with our worker count
    #[clap(short, long, default_value = "100")]
    pub multiplier: u64,
    /// Whether this census should only report counts but not save them to the DB
    #[clap(short, long)]
    pub dry_run: bool,
}
