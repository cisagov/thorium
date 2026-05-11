use clap::Parser;

mod args;
mod libs;
use libs::Worker;
use thorium::Error;
use tracing::instrument;

/// The Thorium agent main loop
#[tokio::main]
async fn main() -> Result<(), Error> {
    // load command line args
    let args = args::Args::parse();
    // build our agent name by what scaler we are claiming jobs for
    let trace_name = format!("Thorium{}Agent", args.env.kind());
    // setup our tracers/subscribers
    let trace_provider = thorium::utils::trace::from_file(&trace_name, &args.trace);
    let launch_result = launch(args).await;
    // export any remaining traces and shutdown this provider
    thorium::utils::trace::shutdown(trace_provider);
    launch_result
}

/// Launch the Thorium agent
///
/// # Arguments
///
/// * `args` - Arguments to the agent
#[instrument(name = "launch", skip_all, err(Debug))]
async fn launch(args: args::Args) -> Result<(), Error> {
    // build and execute worker
    let mut worker = Worker::new(args).await?;
    worker.start().await
}
