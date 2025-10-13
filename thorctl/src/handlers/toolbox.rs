//! Handles toolbox commands

use thorium::Error;

mod import;
mod manifest;
mod shared;
mod update;

use crate::args::Args;
use crate::args::toolbox::Toolbox;
use crate::utils;

pub async fn handle(args: &Args, toolbox: &Toolbox) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        crate::handlers::update::ask_update(&thorium).await?;
    }
    match toolbox {
        Toolbox::Import(cmd) => import::import(thorium, conf, cmd).await,
        Toolbox::Update(cmd) => update::update(thorium, conf, cmd).await,
    }
}
