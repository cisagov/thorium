//! Set up the stage logs tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for tags
pub struct LogsPreparedStatements {
    /// Insert a log line
    pub insert: PreparedStatement,
    /// Get log lines
    pub get: PreparedStatement,
}

impl LogsPreparedStatements {
    /// Build a new logs prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the logs tables
        setup_log_table(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        // setup our prepared statement object
        LogsPreparedStatements { insert, get }
    }
}

/// Set up a log table for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_log_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.logs (\
        reaction UUID,
        stage TEXT,
        bucket INT,
        position BIGINT,
        line TEXT,
        PRIMARY KEY ((reaction, stage, bucket), position))
        WITH default_time_to_live = {ttl}",
        ns = &config.thorium.namespace,
        ttl = &config.thorium.retention.logs
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add log table");
}

/// build the log insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build log insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.logs \
                (reaction, stage, bucket, position, line) \
                VALUES (?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla log insert statement")
}

/// build the log get prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build log get prepared statement
    session
        .prepare(format!(
            "SELECT line FROM {}.logs \
                WHERE reaction = ? AND stage = ? AND bucket in ? AND position >= ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla log get statement")
}
