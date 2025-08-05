//! Set up the comments tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for tags
pub struct CommentsPreparedStatements {
    /// Insert a comment
    pub insert: PreparedStatement,
    /// Get a comment
    pub get: PreparedStatement,
    /// Delete a comment
    pub delete: PreparedStatement,
    /// Check if a comment exists
    pub exists: PreparedStatement,
}

impl CommentsPreparedStatements {
    /// Build a new comments prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the comments tables
        setup_comments_table(session, config).await;
        // set up the comments materialized views
        setup_comments_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let delete = delete(session, config).await;
        let exists = exists(session, config).await;
        // build our prepared statement object
        CommentsPreparedStatements {
            insert,
            get,
            delete,
            exists,
        }
    }
}

/// Set up the comments table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_comments_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.comments (\
            group TEXT,
            sha256 TEXT,
            uploaded TIMESTAMP,
            id UUID,
            author TEXT,
            comment TEXT,
            files TEXT,
            PRIMARY KEY ((group, sha256), uploaded, id)) \
            WITH CLUSTERING ORDER BY (uploaded ASC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add comments table");
}

/// Set up the repo materialized view for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_comments_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.comments_by_id AS \
            SELECT group, sha256, uploaded, id FROM {ns}.comments \
            WHERE id IS NOT NULL \
            AND sha256 IS NOT NULL \
            AND group IS NOT NULL \
            AND uploaded IS NOT NULL \
            PRIMARY KEY (id, sha256, group, uploaded)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add comments materialized view");
}

/// build the comments insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build comments insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.comments \
                (group, sha256, uploaded, id, author, comment, files) \
                VALUES (?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla comments insert statement")
}

/// Gets all comments for a sample from scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build comments insert prepared statement
    session
        .prepare(format!(
            "SELECT group, sha256, uploaded, id, author, comment, files \
                FROM {}.comments \
                WHERE group IN ? AND sha256 = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla comments get statement")
}

/// Deletes a comment row from scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build comment delete prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.comments \
                WHERE group = ? \
                AND sha256 = ? \
                AND uploaded = ? \
                AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla comment delete statement")
}

/// Checks if more comments exist for a sample in scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn exists(session: &Session, config: &Conf) -> PreparedStatement {
    // build comment exists prepared statement
    session
        .prepare(format!(
            "SELECT id  \
                FROM {}.comments_by_id \
                WHERE id IN ? \
                GROUP BY id",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla comment exists statement")
}
