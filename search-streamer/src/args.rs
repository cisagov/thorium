//! The command line args for the search streamer

use clap::Parser;

/// The command line args for the search streamer
#[derive(Parser, Debug, Clone)]
#[clap(version, author)]
pub struct Args {
    /// The path to the thorium config
    #[clap(long, default_value = "thorium.yml")]
    pub config: String,
    /// The path to the single user auth keys to use in place of the thorctl config
    #[clap(short, long, default_value = "keys.yml")]
    pub keys: String,
    /// Delete existing indexes if they exist and reindex based on data currently
    /// in the database
    #[clap(long)]
    pub reindex: bool,
}
