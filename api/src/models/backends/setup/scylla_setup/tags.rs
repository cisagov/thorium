//! Set up the samples tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for tags
pub struct TagsPreparedStatements {
    /// Insert a tag
    pub insert: PreparedStatement,
    /// Get just the basic tag values
    pub get: PreparedStatement,
    /// Get the entire tag row
    pub get_rows: PreparedStatement,
    /// Delete a tag row
    pub delete: PreparedStatement,
    /// List ties from a previous tag cursor
    pub list_ties: PreparedStatement,
    /// Pull tag rows for a specific cursor page
    pub list_pull: PreparedStatement,
    /// List ties from a previous tag cursor regardless of key/value case
    pub list_ties_case_insensitive: PreparedStatement,
    /// Pull tag rows for a specific cursor page regardless of key/value case
    pub list_pull_case_insensitive: PreparedStatement,
}

impl TagsPreparedStatements {
    /// Build a new tags prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the tags tables
        setup_tags_table(session, config).await;
        // set up the tags materialized view
        setup_tags_by_item_mat_view(session, config).await;
        setup_tags_case_insensitve_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let get_rows = get_rows(session, config).await;
        let delete = delete(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list_pull = list_pull(session, config).await;
        let list_ties_case_insensitive = list_ties_case_insensitive(session, config).await;
        let list_pull_case_insensitive = list_pull_case_insensitive(session, config).await;
        // build our prepared statement object
        TagsPreparedStatements {
            insert,
            get,
            get_rows,
            delete,
            list_ties,
            list_pull,
            list_ties_case_insensitive,
            list_pull_case_insensitive,
        }
    }
}

///// Setup all required tags tables and prepared statements
//pub fn setup(session: &Session, config: &Conf)

/// Set up the tags table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_tags_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.tags (\
            type TEXT,
            group TEXT,
            item TEXT,
            year INT,
            bucket INT,
            key TEXT,
            value TEXT,
            uploaded TIMESTAMP,
            key_lower TEXT,
            value_lower TEXT,
            PRIMARY KEY ((type, group, year, bucket, key, value), uploaded, item, key_lower, value_lower)) \
            WITH CLUSTERING ORDER BY (uploaded DESC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add tags table");
}

/// Create the materialized view for getting all tags for a specific hash
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_tags_by_item_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.tags_by_item AS \
            SELECT type, group, item, year, bucket, key, value, key_lower, value_lower, uploaded FROM {ns}.tags \
            WHERE type IS NOT NULL \
            AND group IS NOT NULL \
            AND item IS NOT NULL \
            AND year IS NOT NULL
            AND bucket IS NOT NULL \
            AND key IS NOT NULL \
            AND value IS NOT NULL \
            AND uploaded IS NOT NULL \
            AND key_lower IS NOT NULL \
            AND value_lower IS NOT NULL \
            PRIMARY KEY ((type, item), group, year, bucket, uploaded, key, value, key_lower, value_lower)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add tags search materialized view");
}

/// Create the materialized view for case-insensitve queries
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_tags_case_insensitve_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.tags_case_insensitive AS \
            SELECT type, group, item, year, bucket, key_lower, value_lower, uploaded, key, value FROM {ns}.tags \
            WHERE type IS NOT NULL \
            AND group IS NOT NULL \
            AND item IS NOT NULL \
            AND year IS NOT NULL
            AND bucket IS NOT NULL \
            AND key_lower IS NOT NULL \
            AND value_lower IS NOT NULL \
            AND uploaded IS NOT NULL \
            AND key IS NOT NULL \
            AND value IS NOT NULL \
            PRIMARY KEY ((type, group, year, bucket, key_lower, value_lower), uploaded, item, key, value)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add tags case-insensitive materialized view");
}

/// build the tags insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.tags \
                (type, group, item, year, bucket, key, value, uploaded, key_lower, value_lower) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla tags insert statement")
}

/// build the tags get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags get prepared statement
    session
        .prepare(format!(
            "SELECT group, item, key, value \
                FROM {}.tags_by_item \
                WHERE type = ? AND group IN ? AND item = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla tags get statement")
}

/// build the tags rows get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_rows(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags get prepared statement
    session
        .prepare(format!(
            "SELECT group, year, bucket, key, value, uploaded, item \
                FROM {}.tags_by_item \
                WHERE type = ? AND group IN ? AND item = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla tags rows get statement")
}

/// Deletes a tag row from scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags insert prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.tags \
                WHERE type = ? \
                AND group = ? \
                AND year = ? \
                AND bucket = ? \
                AND key = ? \
                AND value = ? \
                AND uploaded = ? \
                AND item = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla tags delete statement")
}

/// Gets any remaining rows from past ties in listing items by tags
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build list tag ties prepared statement
    session
        .prepare(format!(
            "SELECT group, item, uploaded \
                FROM {}.tags \
                WHERE type = ? \
                AND group = ? \
                AND year = ? \
                AND bucket = ? \
                AND key = ? \
                AND value = ? \
                AND uploaded = ? \
                AND item >= ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla list tag ties statement")
}

/// Pull the data needed to list items by tags
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build list tag pull prepared statement
    session
        .prepare(format!(
            "SELECT group, item, uploaded \
                FROM {}.tags \
                WHERE type = ? \
                AND group = ? \
                AND year = ? \
                AND bucket in ? \
                AND key = ? \
                AND value = ? \
                AND uploaded < ? \
                AND uploaded > ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla list tag pull statement")
}

/// Gets any remaining rows from past ties in listing items by tags regardless of
/// key/value case
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties_case_insensitive(session: &Session, config: &Conf) -> PreparedStatement {
    // build list tag ties case insensitive prepared statement
    session
        .prepare(format!(
            "SELECT group, item, uploaded \
                FROM {}.tags_case_insensitive \
                WHERE type = ? \
                AND group = ? \
                AND year = ? \
                AND bucket = ? \
                AND key_lower = ? \
                AND value_lower = ? \
                AND uploaded = ? \
                AND item >= ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla list tag ties case insensitive statement")
}

/// Pull the data needed to list items by tags regardless of key/value case
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull_case_insensitive(session: &Session, config: &Conf) -> PreparedStatement {
    // build list tag pull case insensitive prepared statement
    session
        .prepare(format!(
            "SELECT group, item, uploaded \
                FROM {}.tags_case_insensitive \
                WHERE type = ? \
                AND group = ? \
                AND year = ? \
                AND bucket in ? \
                AND key_lower = ? \
                AND value_lower = ? \
                AND uploaded < ? \
                AND uploaded > ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla list tag pull case insensitive statement")
}
