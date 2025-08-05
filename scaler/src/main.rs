#![feature(hash_set_entry)]
#![feature(btree_extract_if)]

use clap::Parser;

mod args;
mod libs;
use libs::Scaler;

/// The Thorium agent
#[tokio::main]
async fn main() {
    // install a crypto provider for rustls
    // Rustls will complain if this is not run but we can ignore any errors
    // https://github.com/rustls/rustls/issues/1938
    let _ = rustls::crypto::ring::default_provider().install_default();
    // get command line args
    let args = args::Args::parse();
    // try to load a config file
    let conf = thorium::Conf::new(&args.config).expect("Failed to load config");
    // generate a name for this scaler based on what it schedules
    let name = format!("Thorium{}Scaler", args.scaler);
    // setup our tracer
    let trace_provider = thorium::utils::trace::setup(&name, &conf.thorium.tracing);
    // setup scaler
    let mut scaler = Scaler::new(args).await.expect("Scaler failed to initialize");
    // start scaler
    scaler.start().await.expect("Scaler crashed");
    // shutdown our trace provider
    thorium::utils::trace::shutdown(trace_provider);
}
