//! Handle tree related Thorctl commands

use crate::Error;
use crate::args::Args;
use crate::args::trees::Trees;
use crate::handlers::update;
use crate::utils;

pub mod delete;

/// Handle all tree commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The trees command to execute
pub async fn handle(args: &Args, cmd: &Trees) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        update::ask_update(&thorium).await?;
    }
    // call the right trees handler
    match cmd {
        Trees::Delete(cmd) => delete::delete(&thorium, args, cmd).await,
    }
}
