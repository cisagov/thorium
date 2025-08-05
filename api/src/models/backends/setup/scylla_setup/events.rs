//! Set up the events tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for events
pub struct EventsPreparedStatements {
    /// Insert an event
    pub insert: PreparedStatement,
    /// List ties in an event cursor
    pub list_ties: PreparedStatement,
    /// Get the actually event data for a cursor
    pub list_pull: PreparedStatement,
    /// Delete an event
    pub delete: PreparedStatement,
}

impl EventsPreparedStatements {
    /// Build a new commitishes prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the events tables
        setup_events(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list_pull = list_pull(session, config).await;
        let delete = delete(session, config).await;
        // build our prepared statement object
        EventsPreparedStatements {
            insert,
            list_ties,
            list_pull,
            delete,
        }
    }
}

/// Set up the event table
///
/// A table to tracks events in Thorium.
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_events(session: &Session, config: &Conf) {
    // build cmd for events table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.events (\
            event_type TEXT, \
            year INT, \
            bucket INT, \
            timestamp TIMESTAMP, \
            id UUID, \
            parent UUID, \
            user TEXT, \
            trigger_depth SMALLINT, \
            data TEXT, \
            PRIMARY KEY ((event_type, year, bucket), timestamp, id))
            WITH default_time_to_live = {ttl}",
        ns = &config.thorium.namespace,
        ttl = config.thorium.events.retention,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add the events table");
}

/// build the events insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build events insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.events \
                (event_type, year, bucket, timestamp, id, parent, user, trigger_depth, data) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla events insert statement")
}

/// Gets any remaining rows from past ties in listing events
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build events list ties prepared statement
    session
        .prepare(format!(
            "SELECT id, timestamp, parent, user, trigger_depth, data \
                FROM {}.events \
                WHERE event_type = ? \
                AND year = ? \
                AND bucket = ? \
                AND timestamp = ? \
                AND id <= ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla event list ties statement")
}

/// Pull the data needed to list events
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build events list ties prepared statement
    session
        .prepare(format!(
            "SELECT id, timestamp, parent, user, trigger_depth, data \
                FROM {}.events \
                WHERE event_type = ? \
                AND year = ? \
                AND bucket in ? \
                AND timestamp < ? \
                AND timestamp > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla event list pull statement")
}

/// build the events delete prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build events insert prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.events \
                WHERE event_type = ? \
                AND year = ? \
                AND bucket = ? \
                AND timestamp in ? \
                AND id in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla events insert statement")
}
