//! Performs admin tasks for Thorium

#![feature(round_char_boundary)]

mod args;
mod backup;
mod census;
mod error;
mod provision;
mod settings;
mod shared;
mod users;

use clap::Parser;

pub use error::Error;

#[tokio::main]
async fn main() {
    // load command line args
    let args = args::Args::parse();
    // execute the right handler
    if let Err(err) = match &args.cmd {
        args::SubCommands::Backup(backup_cmd) => backup::handle(backup_cmd, &args).await,
        args::SubCommands::Settings(settings_cmd) => settings::handle(settings_cmd, &args).await,
        args::SubCommands::Provision(provision_args) => provision::handle(provision_args).await,
        args::SubCommands::Census(census_cmd) => census::handle(census_cmd, &args).await,
        args::SubCommands::Users(users_cmd) => users::handle(users_cmd, &args).await,
    } {
        eprintln!("{err}");
        // TODO: return the proper exit code based on the error
        std::process::exit(1);
    }
}
