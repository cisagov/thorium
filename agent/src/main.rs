use clap::Parser;

mod args;
mod libs;
use libs::Worker;
use tracing::{event, span, Level};

/// The Thorium agent main loop
#[tokio::main]
async fn main() {
    // load command line args
    let args = args::Args::parse();
    // build our agent name by what scaler we are claiming jobs for
    let trace_name = format!("Thorium{}Agent", args.env.kind());
    // setup our tracers/subscribers
    let trace_provider = thorium::utils::trace::from_file(&trace_name, &args.trace);
    // start our worker launch span
    let span = span!(Level::INFO, "Worker Launch");
    // build and execute worker
    match Worker::new(args).await {
        Ok(mut worker) => match worker.start().await {
            Ok(()) => (),
            Err(error) => {
                // log that this worker died while executing jobs
                event!(
                    parent: &span,
                    Level::INFO,
                    msg = "Worker Failed",
                    error = error.msg()
                );
            }
        },
        Err(error) => {
            // log that this worker died while executing jobs
            event!(
                parent: &span,
                Level::INFO,
                msg = "Worker Creation Failed",
                error = error.msg()
            );
        }
    }
    // export any remaining traces and shutdown this provider
    thorium::utils::trace::shutdown(trace_provider);
}
