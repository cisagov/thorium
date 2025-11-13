//! Operate a Thorium k8s cluster
mod app;
#[allow(dead_code)]
#[allow(unused)]
mod args;
mod k8s;

use clap::Parser;
use k8s::controller;

#[tokio::main]
async fn main() {
    // load command line args
    let args = args::Args::parse();
    // execute the right handler
    match args.cmd {
        // start backing up data
        args::SubCommands::Operate(operate_args) => controller::run(operate_args).await,
    }
}
