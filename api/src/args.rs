use clap::Parser;

/// The command line args passed to the Thorium API
#[derive(Parser, Debug)]
#[clap(version, author)]
pub struct Args {
    /// The path to load the config file from
    #[clap(short, long, default_value = "thorium.yml")]
    pub config: String,
    /// Put the api in benchmarking mode (doesn't do anything yet)
    #[clap(long)]
    pub bench: bool,
}
