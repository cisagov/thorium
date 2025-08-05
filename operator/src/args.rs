/// The arguments for operating a Thorium cluster
use clap::Parser;

/// The arguments for the operator
#[derive(Parser, Debug, Clone)]
#[clap(version, author)]
pub struct Args {
    /// The sub command to execute
    #[clap(subcommand)]
    pub cmd: SubCommands,
}

/// The sub commands for cluster operation
#[derive(Parser, Debug, Clone)]
pub enum SubCommands {
    /// Operate a Thorium k8s cluster
    Operate(OperateCluster),
}

/// Operate a thorium cluster arguments
#[derive(Parser, Debug, Clone)]
pub struct OperateCluster {
    /// Thorium URL when not running local to k8s
    #[clap(short, long)]
    pub url: Option<String>,
}
