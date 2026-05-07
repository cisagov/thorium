//! Handles spawning containers directly for windows nodes
//!
//! This support could likely be extended to linux k8s and baremetal nodes but
//! for k8s nodes would come at the cost of everthing k8s buys us.
//!
//! The Thorium reactor is only supported on Linux and Windows

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "windows"))] {
        // add dependencies
        use clap::Parser;

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
use libs::Error;

#[cfg(target_os = "linux")]
use libs::launchers::BareMetal;
#[cfg(target_os = "windows")]
use libs::launchers::Windows;
#[cfg(all(target_os = "linux", feature = "kvm"))]
use libs::launchers::kvm::Kvm;

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::main]
async fn main() -> Result<(), Error> {
    // parse our args
    let args = args::Args::parse();
    // build the name for this reactor based on type
    let trace_name = format!("Thorium{}Reactor", args.scaler);
    // setup our tracers/subscribers
    let trace_provider = thorium::utils::trace::from_file(&trace_name, &args.trace);
    // run the reactor
    if let Err(err) = run(args).await {
        // shutdown the trace provider first
        thorium::utils::trace::shutdown(trace_provider);
        // then propagate the error
        return Err(err);
    }
    // shut down our trace provider if we're shutting down cleanly
    thorium::utils::trace::shutdown(trace_provider);
    Ok(())
}

/// Builds and runs the reactor
///
/// # Arguments
///
/// * `args` - The arguments to the reactor
#[cfg(any(target_os = "linux", target_os = "windows"))]
async fn run(args: args::Args) -> Result<(), Error> {
    // build and start this nodes reactor
    match &args.launchers {
        #[cfg(target_os = "linux")]
        args::Launchers::BareMetal => {
            let node = args.node()?;
            let launcher = BareMetal::new(&args.cluster, node);
            let reactor = Reactor::new(args, launcher).await?;
            reactor.start().await
        }
        #[cfg(target_os = "windows")]
        args::Launchers::Windows => {
            let launcher = Windows::default();
            let reactor = Reactor::new(args, launcher).await?;
            reactor.start().await
        }
        #[cfg(feature = "kvm")]
        args::Launchers::Kvm(kvm) => {
            let launcher = Kvm::new(kvm.clone().into(), &args.agents_dir).await?;
            let reactor = Reactor::new(args, launcher).await?;
            reactor.start().await
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    eprintln!("The Thorium Reactor is not supported on MacOS!");
}
