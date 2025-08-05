//! Arguments for cluster-related Thorctl commands

use clap::Parser;
use std::path::PathBuf;
use thorium::models::{ImageScaler, NodeListParams};

/// The settings for logging into Thorium
#[derive(Parser, Debug)]
pub struct Login {
    /// The url for Thorium
    pub url: String,
    /// The user to login as
    #[clap(short, long)]
    pub user: Option<String>,
    /// The user's password if they want to pass it in insecurely. Only use this for non-interactive environments.
    #[clap(short, long)]
    pub password: Option<String>,
    /// Trust invalid certificates
    #[clap(long)]
    pub invalid_certs: Option<bool>,
    /// Trust invalid hostnames
    #[clap(long)]
    pub invalid_hostnames: Option<bool>,
    /// The path to any certificate authorities to trust
    #[clap(long)]
    pub certificate_authorities: Option<Vec<PathBuf>>,
    /// Clear any existing Thorctl settings in the config file given by `--config` and start from scratch
    #[clap(long)]
    pub clear_settings: bool,
}

/// The commands for getting cluster information in Thorium
#[derive(Parser, Debug)]
pub enum Clusters {
    /// Show the status of nodes in Thorium including available resources
    ///
    /// Statistics on available resources are not necessarily live values and
    /// are updated by the scaler semi-frequently (currently every 2 minutes).
    /// CPU is represented in mCPU/milliCPU (1000 = 1 CPU core) and memory/ephemeral
    /// storage are represented in MiB/mebibytes.
    #[clap(version, author)]
    Status(ClusterStatus),
    /// Show the status of individual workers in Thorium
    #[clap(version, author)]
    Workers(ClusterWorkers),
}

/// A command to show the current cluster status
#[derive(Parser, Debug)]
pub struct ClusterStatus {
    /// The internal sub clusters to show
    #[clap(short, long)]
    pub clusters: Vec<String>,
}

impl From<&ClusterStatus> for NodeListParams {
    fn from(cmd: &ClusterStatus) -> Self {
        // build our node list params
        NodeListParams::default().clusters(cmd.clusters.clone())
    }
}

/// A command to show the current cluster status
#[derive(Parser, Debug)]
pub struct ClusterWorkers {
    /// The internal sub clusters to show
    #[clap(short, long)]
    pub clusters: Vec<String>,
    /// The scalers to list workers from
    #[clap(short, long, ignore_case = true)]
    pub scalers: Vec<ImageScaler>,
}

impl From<&ClusterWorkers> for NodeListParams {
    fn from(cmd: &ClusterWorkers) -> Self {
        // build our node list params
        NodeListParams::default()
            .clusters(cmd.clusters.clone())
            .scalers(&cmd.scalers)
    }
}
