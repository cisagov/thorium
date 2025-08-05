//! Setup Scylla for Thorium

use chrono::prelude::*;
use chrono::Duration;
use futures::{poll, task::Poll};
use scylla::client::session::Session;
use scylla::client::session_builder::{GenericSessionBuilder, SessionBuilder};
use std::time::Duration as StdDuration;

mod comments;
mod commitishes;
mod events;
mod logs;
mod network_policies;
mod nodes;
mod notifications;
mod repos;
mod results;
mod s3;
mod samples;
mod tags;
mod tools;

use comments::CommentsPreparedStatements;
use commitishes::CommitishesPreparedStatements;
use events::EventsPreparedStatements;
use logs::LogsPreparedStatements;
use network_policies::NetworkPoliciesPreparedStatements;
use nodes::NodesPreparedStatements;
use notifications::NotificationsPreparedStatements;
use repos::ReposPreparedStatements;
use results::ResultsPreparedStatements;
use s3::S3PreparedStatements;
use samples::SamplesPreparedStatements;
use tags::TagsPreparedStatements;
//use tools::ToolsPreparedStatements;

use crate::{setup, Conf};

/// The different groups of prepared statements for scylla
pub struct ScyllaPreparedStatements {
    /// The comments related prepared statements
    pub comments: CommentsPreparedStatements,
    /// The commitishes related prepared statements
    pub commitishes: CommitishesPreparedStatements,
    /// The events related prepared statements
    pub events: EventsPreparedStatements,
    /// The logs related prepared statements
    pub logs: LogsPreparedStatements,
    /// The network policies related prepared statements
    pub network_policies: NetworkPoliciesPreparedStatements,
    /// The nodes related prepared statements
    pub nodes: NodesPreparedStatements,
    /// The notifications related prepared statements
    pub notifications: NotificationsPreparedStatements,
    /// The repos related prepared statements
    pub repos: ReposPreparedStatements,
    /// The results related prepared statements
    pub results: ResultsPreparedStatements,
    /// The s3 related prepared statements
    pub s3: S3PreparedStatements,
    /// The samples related prepared statements
    pub samples: SamplesPreparedStatements,
    /// The tags related prepared statements
    pub tags: TagsPreparedStatements,
}

impl ScyllaPreparedStatements {
    /// Create our scylla prepared statements
    ///
    /// # Arguments
    ///
    /// * `session` - A scylla session
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // setup our prepared statements
        let comments = CommentsPreparedStatements::new(session, config).await;
        let commitishes = CommitishesPreparedStatements::new(session, config).await;
        let events = EventsPreparedStatements::new(session, config).await;
        let logs = LogsPreparedStatements::new(session, config).await;
        let network_policies = NetworkPoliciesPreparedStatements::new(session, config).await;
        let nodes = NodesPreparedStatements::new(session, config).await;
        let notifications = NotificationsPreparedStatements::new(session, config).await;
        let repos = ReposPreparedStatements::new(session, config).await;
        let results = ResultsPreparedStatements::new(session, config).await;
        let s3 = S3PreparedStatements::new(session, config).await;
        let samples = SamplesPreparedStatements::new(session, config).await;
        let tags = TagsPreparedStatements::new(session, config).await;
        // build our grouped prepared statement object
        ScyllaPreparedStatements {
            comments,
            commitishes,
            events,
            logs,
            network_policies,
            nodes,
            notifications,
            repos,
            results,
            s3,
            samples,
            tags,
        }
    }
}

/// The scylla client and prepared statements
pub struct Scylla {
    /// The scylla session object
    pub session: Session,
    /// prepared statements for scylla
    pub prep: ScyllaPreparedStatements,
}

impl Scylla {
    /// Create a new scylla client
    ///
    /// # Arguments
    ///
    /// * `config` - The Thorium config
    pub async fn new(config: &Conf) -> Self {
        // loop and try to complete this future
        for _ in 0..4 {
            // get the correct timeout for scylla
            let timeout = Utc::now() + Duration::seconds(i64::from(config.scylla.setup_time));
            // get a clone of our config and log
            let config_clone = config.clone();
            // build the future for our setup
            let mut future = tokio::spawn(async move { build(config_clone).await });
            // timeout appears to just hang so were going to check it manually
            loop {
                // check if this future has completed yet
                if let Poll::Ready(join_result) = poll!(&mut future) {
                    // if this future has errored out then panic with that error
                    match join_result {
                        Ok(client) => return client,
                        // there was an error so print it and try again
                        Err(err) => {
                            // print our error and try again
                            setup!(
                                config.thorium.tracing.local.level,
                                format!("Scylla setup error {:#?}", err)
                            );
                            // try to connect to scylla again
                            break;
                        }
                    }
                }
                // check if we are past our timeout yet
                if Utc::now() > timeout {
                    setup!(
                        config.thorium.tracing.local.level,
                        format!(
                            "Failed to connect to scylla in {} seconds",
                            config.scylla.setup_time
                        )
                    );
                    break;
                }
                // sleep for 3 seconds
                tokio::time::sleep(StdDuration::from_millis(100)).await;
            }
        }
        // panic if we fail to connect
        panic!("Failed to connect/setup Scylla");
    }
}

/// Create a new session to scylla
///
/// Arguments
///
/// * `config` - The thorium config
pub async fn new_session(config: &Conf) -> Session {
    // connecting to scylla
    setup!(
        config.thorium.tracing.local.level,
        format!("Connecting to scylla at {}", config.scylla.nodes.join(", "))
    );
    // start building our scylla client
    let mut session = SessionBuilder::new();
    // if we have auth info for scylla then add that
    if let Some(creds) = &config.scylla.auth {
        setup!(
            config.thorium.tracing.local.level,
            format!("Authenticating to Scylla as {}", creds.username)
        );
        // inject our creds
        session = session.user(&creds.username, &creds.password);
    }
    // set our request timeout
    let session =
        session.connection_timeout(StdDuration::from_secs(u64::from(config.scylla.setup_time)));
    // build our session
    config
        .scylla
        .nodes
        .iter()
        .fold(session, GenericSessionBuilder::known_node)
        .build()
        .await
        .expect("Failed to build scylla session")
}

/// Set up a keyspace for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_keyspace(session: &Session, config: &Conf) {
    // build keyspace create command
    let keyspace_cmd = format!(
        "CREATE KEYSPACE IF NOT EXISTS {ns} WITH REPLICATION = \
            {{'class' : 'NetworkTopologyStrategy', 'replication_factor': {repl_factor}}}",
        ns = &config.thorium.namespace,
        repl_factor = &config.scylla.replication
    );
    // setup scylla keyspaces
    session
        .query_unpaged(keyspace_cmd, &[])
        .await
        .expect("Failed to setup keyspace");
}

/// Build a session and setup tables/materialized views/prepared statements
async fn build(config: Conf) -> Scylla {
    // Create a new session for scylla
    let session = new_session(&config).await;
    // setup our keyspace if it doesn't already exist
    setup_keyspace(&session, &config).await;
    // get our tables/materialized views and prepared statements
    let prep = ScyllaPreparedStatements::new(&session, &config).await;
    // build our scylla client
    Scylla { session, prep }
}
