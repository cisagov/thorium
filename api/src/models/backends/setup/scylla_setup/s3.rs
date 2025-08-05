//! Set up the s3 IDs tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for s3 IDs
pub struct S3PreparedStatements {
    /// Insert an s3 id
    pub insert: PreparedStatement,
    /// Check if an s3 id already exists
    pub id_exists: PreparedStatement,
    /// Check if an object in s3 already exists
    pub object_exists: PreparedStatement,
    /// Get the s3 id for an object in s3
    pub get: PreparedStatement,
    /// Delete an s3 id
    pub delete: PreparedStatement,
}

impl S3PreparedStatements {
    /// Build a new s3 IDs prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // setup our tables
        setup_s3_ids_table(session, config).await;
        // setup our materialized views
        setup_s3_sha256s_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let id_exists = id_exists(session, config).await;
        let object_exists = object_exists(session, config).await;
        let get = get(session, config).await;
        let delete = delete(session, config).await;
        // build our prepared statement object
        S3PreparedStatements {
            insert,
            id_exists,
            object_exists,
            get,
            delete,
        }
    }
}

/// Set up the s3 sample IDs table for Thorium
///
/// This is the ground truth table for all samples
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_s3_ids_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.s3_ids (\
            type TEXT,
            id UUID,
            sha256 TEXT,
            PRIMARY KEY ((type, id), sha256))",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add s3 ids table");
}

/// Create the materialized view for listings samples by group
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_s3_sha256s_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.s3_sha256s AS \
            SELECT type, id, sha256 FROM {ns}.s3_ids \
            WHERE type IS NOT NULL \
            AND id IS NOT NULL \
            AND sha256 IS NOT NULL \
            PRIMARY KEY (sha256, type, id)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add s3 sha256s materialized view");
}

/// build the s3 id insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build s3 id insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.s3_ids \
                (type, id, sha256) \
                VALUES (?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare s3 ids insert statement")
}

/// build the s3 id exists prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn id_exists(session: &Session, config: &Conf) -> PreparedStatement {
    // build s3 id exists prepared statement
    session
        .prepare(format!(
            "SELECT id  \
                FROM {}.s3_ids \
                WHERE type = ? AND id = ? \
                LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare s3 ids exists statement")
}

/// build the s3 object exists prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn object_exists(session: &Session, config: &Conf) -> PreparedStatement {
    // build s3 object exists prepared statement
    session
        .prepare(format!(
            "SELECT sha256  \
                FROM {}.s3_sha256s \
                WHERE type = ? AND sha256 = ? \
                LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare s3 object exists statement")
}

/// build the s3 object id get prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build s3 object id prepared statement
    session
        .prepare(format!(
            "SELECT id  \
                FROM {}.s3_sha256s \
                WHERE type = ? AND sha256 = ? \
                LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare s3 object id statement")
}

/// build the s3 id delete prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build s3 id delete prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.s3_ids WHERE type = ? and id in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare s3 ids insert statement")
}
