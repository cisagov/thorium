//! The arguments for parsing an email

use clap::Parser;
use std::path::PathBuf;

/// The arguments for parsing an email
#[derive(Debug, Parser)]
pub struct Args {
    /// The path to the email to parse
    #[arg(required = true)]
    pub path: PathBuf,
    /// The path to write results too
    #[arg(short, long, default_value = "/tmp/thorium/results")]
    pub output: PathBuf,
    /// The path to write tags too
    #[arg(short, long, default_value = "/tmp/thorium/tags")]
    pub tags_output: PathBuf,
    /// The path to write attachments too
    #[arg(short, long, default_value = "/tmp/thorium/children/carved/unknown")]
    pub attachments_output: PathBuf,
}
