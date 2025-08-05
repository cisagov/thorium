//! Set up the commitishes tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statement for commitishes
pub struct CommitishesPreparedStatements {
    /// Insert a commitish
    pub insert: PreparedStatement,
    /// Insert a commitish into the list table
    pub insert_list: PreparedStatement,
    /// Get the commitish data key
    pub get_data: PreparedStatement,
    /// Check if a commitish exists
    pub exists: PreparedStatement,
    /// Get the repo data from a commitish
    pub get_repo_data: PreparedStatement,
    /// Get the number of repo datas that have commits tied to them
    pub get_repo_data_count: PreparedStatement,
    /// Get the ties for listing commitishes
    pub list_ties: PreparedStatement,
    /// Get the data for a page for a commitish cursor
    pub list_pull: PreparedStatement,
}

impl CommitishesPreparedStatements {
    /// Build a new commitishes prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the commitishes tables
        setup_commitishes_table(session, config).await;
        setup_commitishes_list_table(session, config).await;
        // set up the commitishes materialized view
        setup_committed_repo_data_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let insert_list = insert_list(session, config).await;
        let get_data = get_data(session, config).await;
        let exists = exists(session, config).await;
        let get_repo_data = get_repo_data(session, config).await;
        let get_repo_data_count = get_repo_data_count(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list_pull = list_pull(session, config).await;
        // build our prepared statement object
        CommitishesPreparedStatements {
            insert,
            insert_list,
            get_data,
            exists,
            get_repo_data,
            get_repo_data_count,
            list_ties,
            list_pull,
        }
    }
}

/// Set up the commitish table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_commitishes_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.commitishes (\
            kind TEXT,
            group TEXT,
            repo TEXT,
            key TEXT,
            timestamp TIMESTAMP,
            data TEXT,
            repo_data TEXT,
            PRIMARY KEY ((kind, group, repo, key)))",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add commitish table");
}

/// Set up the repo commitish list table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_commitishes_list_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.commitish_list (\
            kind TEXT,
            group TEXT,
            year INT,
            bucket INT,
            repo TEXT,
            timestamp TIMESTAMP,
            key TEXT,
            repo_data TEXT,
            PRIMARY KEY ((kind, group, year, bucket, repo), timestamp, key)) \
            WITH CLUSTERING ORDER BY (timestamp DESC, key DESC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add commitish list table");
}

/// Set up the repo data actually tied to a commitish materialized view for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_committed_repo_data_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.committed_repo_data AS \
            SELECT group, repo, repo_data, key FROM {ns}.commitish_list \
            WHERE kind IS NOT NULL \
            AND group IS NOT NULL \
            AND repo IS NOT NULL \
            AND repo_data IS NOT NULL \
            AND key IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND timestamp IS NOT NULL \
            PRIMARY KEY ((repo_data, year, bucket), kind, group, repo, key, timestamp)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add commitish repo data  materialized view");
}

/// build the commitish insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build commitish insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.commitishes \
                (kind, group, repo, key, timestamp, data, repo_data) \
                VALUES (?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla commitish insert statement")
}

/// build the commitish data get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_data(session: &Session, config: &Conf) -> PreparedStatement {
    // build the commitish details get prepared statement
    session
        .prepare(format!(
            "SELECT data \
                FROM {}.commitishes \
                WHERE kind = ? AND group IN ? AND repo = ? AND key = ? \
                LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla commitish details get statement")
}

/// build the commitish list insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert_list(session: &Session, config: &Conf) -> PreparedStatement {
    // build the commitish list insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.commitish_list \
                (kind, group, year, bucket, repo, timestamp, key, repo_data) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla commitish list insert statement")
}

/// build the commit exists get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn exists(session: &Session, config: &Conf) -> PreparedStatement {
    // build commit exists get prepared statement
    session
        .prepare(format!(
            "SELECT key \
                FROM {}.commitishes \
                WHERE kind = ? AND group IN ? AND repo = ? AND key = ? \
                PER PARTITION LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla commitish exists get statement")
}

/// build the commit repo data get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_repo_data(session: &Session, config: &Conf) -> PreparedStatement {
    // build the commit repo data get prepared statement
    session
        .prepare(format!(
            "SELECT kind, repo_data \
                FROM {}.commitishes \
                WHERE kind IN ? AND group IN ? AND repo = ? AND key = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla commitish repo data get statement")
}

/// build the committed repo data commit prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_repo_data_count(session: &Session, config: &Conf) -> PreparedStatement {
    // build committed repo data count prepared statement
    session
        .prepare(format!(
            "SELECT repo_data \
                    FROM {}.committed_repo_data \
                    WHERE year = ? AND bucket in ? AND repo_data = ? \
                    PER PARTITION LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla committed repo data count statement")
}

/// Gets any remaining rows from past ties in listing repo commits
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo commit list ties prepared statement
    session
        .prepare(format!(
            "SELECT kind, group, key, timestamp \
                FROM {}.commitish_list \
                WHERE kind in ? \
                AND group = ? \
                AND year = ? \
                AND bucket = ? \
                AND repo = ? \
                AND timestamp = ? \
                AND key <= ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo commit list ties statement")
}

/// Pulls the data for listing commits in Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo commit list pull prepared statement
    session
        .prepare(format!(
            "SELECT kind, group, key, timestamp \
                FROM {}.commitish_list \
                WHERE kind in ? \
                AND group = ? \
                AND year = ? \
                AND bucket in ? \
                AND repo = ? \
                AND timestamp < ? \
                AND timestamp > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo commit list pull statement")
}
