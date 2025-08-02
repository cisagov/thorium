//! Utility functions for search-streamer

use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use std::time::Duration;
use thorium::{Conf, Error};

/// Build a scylla client for a specific cluster
///
/// # Arguments
///
/// * `config` - The config for this Thorium cluster
pub async fn get_scylla_client(conf: &Conf) -> Result<Session, Error> {
    // start building our scylla client
    let mut session = SessionBuilder::new();
    // if we have auth info for scylla then add that
    if let Some(creds) = &conf.scylla.auth {
        // inject our creds
        session = session.user(&creds.username, &creds.password);
    }
    // set our request timeout
    let session =
        session.connection_timeout(Duration::from_secs(u64::from(conf.scylla.setup_time)));
    // build a scylla session
    let scylla = conf
        .scylla
        .nodes
        .iter()
        .fold(session, |builder, node| builder.known_node(node))
        .build()
        .await
        .map_err(|err| Error::new(format!("Error connecting to Scylla: {err}")))?;
    Ok(scylla)
}

/// Setup a connection pool to the redis backend
///
/// # Arguments
///
/// * `config` - The config for the Thorium API
pub fn get_redis_client(conf: &Conf) -> Result<redis::Client, Error> {
    // get redis config
    let redis = &conf.redis;
    // build url to server using authentication if its configured
    let url = match (&redis.username, &redis.password) {
        // redis with username/password auth setup
        (Some(user), Some(password)) => format!(
            "redis://{}:{}@{}:{}/",
            user, password, redis.host, redis.port
        ),
        (None, Some(password)) => format!(
            "redis://default:{}@{}:{}/",
            password, redis.host, redis.port
        ),
        (None, None) => format!("redis://{}:{}/", redis.host, redis.port),
        _ => {
            return Err(Error::new(
                "Redis Setup Error: Password must be set if username is set",
            ))
        }
    };
    // build Redis client
    let client = redis::Client::open(url)
        .map_err(|err| Error::new(format!("Error connecting to Redis: {err}")))?;
    Ok(client)
}
