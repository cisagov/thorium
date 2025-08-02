//! Some utilities to help thoradm do what it needs to do

use std::time::Duration;

use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use thorium::Conf;

use crate::Error;

/// Build a scylla client for a specific cluster
///
/// # Arguments
///
/// * `config` - The config for this Thorium cluster
pub async fn get_scylla_client(config: &Conf) -> Result<Session, Error> {
    // start building our scylla client
    let mut session = SessionBuilder::new();
    // if we have auth info for scylla then add that
    if let Some(creds) = &config.scylla.auth {
        // inject our creds
        session = session.user(&creds.username, &creds.password);
    }
    // set our request timeout
    let session = session.connection_timeout(Duration::from_secs(config.scylla.setup_time as u64));
    // build a scylla session
    let scylla = config
        .scylla
        .nodes
        .iter()
        .fold(session, |builder, node| builder.known_node(node))
        .build()
        .await?;
    Ok(scylla)
}
