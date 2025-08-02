//! The Thorium event handler

use clap::Parser;

mod args;
mod libs;

use libs::EventController;

#[tokio::main]
async fn main() {
    // get command line args
    let args = args::Args::parse();
    // try to load a config file
    let conf = thorium::Conf::new(&args.config).expect("Failed to load config");
    // setup our tracer
    let trace_provider = thorium::utils::trace::setup("ThoriumEventHandler", &conf.thorium.tracing);
    // build our event controller
    let controller = EventController::new(args, conf)
        .await
        .expect("Failed to start event controller");
    // start our event handler workers
    controller.start().await;
    // export any remaining traces and shutdown this provider
    thorium::utils::trace::shutdown(trace_provider);
}
