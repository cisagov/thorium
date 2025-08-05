//! Handles spawning containers directly for windows nodes
//!
//! This support could likely be extended to linux k8s and baremetal nodes but
//! for k8s nodes would come at the cost of everything k8s buys us.
//!
//! The Thorium reactor is only supported on Linux and Windows
#![feature(btree_extract_if)]

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "windows"))] {
        // add dependencies
        use clap::Parser;
        use tracing::{span, Level};

        pub use libs::Reactor;
    }
}

// place modules outside cfg_if to avoid macro expansion inside macros error:
// <https://github.com/rust-lang/rust/issues/52234>
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod args;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod libs;

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::main]
async fn main() {
    // parse our args
    let args = args::Args::parse();
    // build the name for this reactor based on type
    let trace_name = format!("Thorium{}Reactor", args.scaler);
    // setup our tracers/subscribers
    let trace_provider = thorium::utils::trace::from_file(&trace_name, &args.trace);
    // build and start this nodes reactor
    let reactor = match Reactor::new(args).await {
        Ok(reactor) => reactor,
        Err(err) => {
            // start our reactor build failure span
            span!(Level::ERROR, "Reactor Build Failure", err = err.msg());
            panic!("Reactor Build Error: {:#?}", err);
        }
    };
    if let Err(err) = reactor.start().await {
        // start our reactor failure span
        span!(Level::ERROR, "Reactor Failure", err = err.msg());
        panic!("Error: {:#?}", err);
    }
    // export any remaining traces and shutdown this provider
    thorium::utils::trace::shutdown(trace_provider);
}

#[cfg(target_os = "macos")]
fn main() {
    eprintln!("The Thorium Reactor is not supported on macOS!");
}
