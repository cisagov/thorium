//! The arguments for auto-volatility3

use bytesize::ByteSize;
use clap::Parser;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use uuid::Uuid;

/// Provide a default config path
fn default_config_path() -> PathBuf {
    let mut default_config_path = dirs::home_dir().unwrap_or_default();
    default_config_path.push(".thorium");
    default_config_path.push("config.yml");
    default_config_path
}

/// The arguments for auto-volatility3
#[derive(Parser, Debug)]
pub struct Args {
    /// The path to the memory dump to scan
    pub target: PathBuf,
    /// The reaction id to create sub reactions for
    #[clap(short, long)]
    pub reaction: Uuid,
    /// Our current job
    #[clap(short, long)]
    pub job: Uuid,
    /// The ephemeral files to upload as results
    #[clap(short, long)]
    pub ephemeral: Vec<PathBuf>,
    /// The group for reactions being created
    #[clap(short, long)]
    pub group: String,
    /// Checkpoint info that is not used
    #[clap(long)]
    pub checkpoint: Option<String>,
    /// The path to a thorctl conf to use when talking to Thorium
    #[clap(long, default_value = default_config_path().into_os_string())]
    pub conf: PathBuf,
    /// The path to the keys to use when talking to thorium
    #[clap(long, conflicts_with = "config")]
    pub keys: Option<String>,
}
