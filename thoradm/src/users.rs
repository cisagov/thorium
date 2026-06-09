//! The user account management features for Thoradm

mod core;
mod dedupe;
mod scan;

use crate::args::{Args, UsersSubCommands};
use crate::Error;

/// Handle the user management command
///
/// # Arguments
///
/// * `cmd` - The user management subcommand to execute
/// * `args` - The Thoradm args
pub async fn handle(cmd: &UsersSubCommands, args: &Args) -> Result<(), Error> {
    // dispatch to the correct user subcommand handler
    match cmd {
        UsersSubCommands::Scan(opts) => scan::scan(opts, args).await,
        UsersSubCommands::Dedupe(opts) => dedupe::dedupe(opts, args).await,
    }
}
