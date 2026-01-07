//! The arguments for AI features in Thorctl

use clap::Parser;

/// The commands to perform AI based commands in Thorctl
#[derive(Parser, Debug)]
pub enum Ai {
    /// Chat with an AI
    Chat(Chat),
    /// Summarize data in thorium
    Summary(Summary),
}

/// Chat with an AI
#[derive(Parser, Debug)]
pub struct Chat {
    /// Whether to log messages to/from the AI
    #[clap(long)]
    pub debug: bool,
}

/// Summarize some data in Thorium
#[derive(Parser, Debug)]
pub struct Summary {
    /// A sample or repo to summarize in Thorium
    pub target: String,
    /// Whether to log messages to/from the AI
    #[clap(long)]
    pub debug: bool,
}
