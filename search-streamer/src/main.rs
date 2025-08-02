#![feature(round_char_boundary)]

use clap::Parser;
use std::sync::Arc;
use thorium::{Conf, Error, Thorium};

mod args;
mod events;
mod index;
mod init;
mod monitor;
mod msg;
mod sources;
mod stores;
mod streamer;
mod utils;
mod worker;

use args::Args;
use sources::{Results, Tags};
use stores::Elastic;
use streamer::SearchStreamer;
use tracing::instrument;

#[instrument(name = "search_streamer::main", err(Debug))]
#[tokio::main]
async fn main() -> Result<(), Error> {
    // get our command line args
    let args = Args::parse();
    // load our config
    let conf = Conf::new(&args.config)
        .map_err(|err| Error::new(format!("Failed to load Thorium config: {err}")))?;
    // setup our tracer
    let trace_provider =
        thorium::utils::trace::setup("ThoriumSearchStreamer", &conf.thorium.tracing);
    // get a Thorium client
    let thorium = Arc::new(
        Thorium::from_key_file(&args.keys)
            .await
            .expect("Failed to create Thorium client"),
    );
    // get a scylla client
    let scylla = Arc::new(utils::get_scylla_client(&conf).await?);
    // get a redis client
    let redis = utils::get_redis_client(&conf)?;
    let redis_conn = redis
        .get_multiplexed_tokio_connection()
        .await
        .map_err(|err| {
            Error::new(format!(
                "Error creating Redis multiplexed connection: {err}"
            ))
        })?;
    // build our streamers
    let results_streamer = SearchStreamer::<Results, Elastic>::new(
        thorium.clone(),
        scylla.clone(),
        redis_conn.clone(),
        &args,
        conf.clone(),
    );
    let tags_streamer = SearchStreamer::<Tags, Elastic>::new(
        thorium.clone(),
        scylla.clone(),
        redis_conn.clone(),
        &args,
        conf.clone(),
    );
    // start our streamers
    // TODO: controller paradigm
    tokio::try_join!(results_streamer.start(), tags_streamer.start()).map(|_| ())?;
    // shutdown our trace provider if we shutdown
    thorium::utils::trace::shutdown(trace_provider);
    Ok(())
}
