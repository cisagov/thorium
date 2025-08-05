//! Set up the samples tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for Samples
pub struct SamplesPreparedStatements {
    /// Insert a sample
    pub insert: PreparedStatement,
    /// Get a sample
    pub get: PreparedStatement,
    /// Get multiple samples
    pub get_many: PreparedStatement,
    /// Make sure a sample is visible by at least one group in a set
    pub auth: PreparedStatement,
    /// Delete a specific sample submission from a single group
    pub delete: PreparedStatement,
    /// Delete a submission id from many groups
    pub delete_multiple_groups: PreparedStatement,
    /// Get the basic submission info for a specific sha256
    pub get_basic_submission_info: PreparedStatement,
    /// List the ties for a sample cursor
    pub list_ties: PreparedStatement,
    /// Get a page of data for a sample cursor
    pub list_pull: PreparedStatement,
}

impl SamplesPreparedStatements {
    /// Build a new samples prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // setup our tables
        setup_samples_list_table(session, config).await;
        // setup our materialized view
        setup_samples_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let get_many = get_many(session, config).await;
        let auth = auth(session, config).await;
        let delete = delete(session, config).await;
        let delete_multiple_groups = delete_multiple_groups(session, config).await;
        let get_basic_submission_info = get_basic_submission_info(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list_pull = list_pull(session, config).await;
        // build our prepared statement object
        SamplesPreparedStatements {
            insert,
            get,
            get_many,
            auth,
            delete,
            delete_multiple_groups,
            get_basic_submission_info,
            list_ties,
            list_pull,
        }
    }
}

/// Set up the samples table for Thorium
///
/// This is the ground truth table for all samples
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_samples_list_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.samples_list (\
            group TEXT, \
            year INT, \
            bucket INT, \
            uploaded TIMESTAMP, \
            id UUID, \
            sha256 TEXT, \
            sha1 TEXT, \
            md5 TEXT, \
            name TEXT, \
            description TEXT, \
            submitter TEXT, \
            origin TEXT, \
            PRIMARY KEY ((group, year, bucket), uploaded, id)) \
            WITH CLUSTERING ORDER BY (uploaded DESC, id DESC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add samples table");
}

/// Create the materialized view for listings samples by group
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_samples_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!("CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.samples AS \
            SELECT group, year, bucket, uploaded, id, sha256, sha1, md5, name, description, submitter, origin FROM {ns}.samples_list \
            WHERE group IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND uploaded IS NOT NULL \
            AND id IS NOT NULL \
            AND sha256 IS NOT NULL \
            PRIMARY KEY (sha256, group, uploaded, id, year, bucket)
            WITH CLUSTERING ORDER BY (uploaded DESC, id DESC)",
            ns=&config.thorium.namespace,
        );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add samples group materialized view");
}

/// build the sample insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build samples insert prepared statement
    session
            .prepare(format!(
                "INSERT INTO {}.samples_list \
                (group, year, bucket, sha256, sha1, md5, id, name, description, submitter, origin, uploaded) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla sample insert statement")
}

/// build the sample get prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build samples get prepared statement
    session
        .prepare(format!(
            "SELECT sha256, sha1, md5, id, name, description, group, submitter, origin, uploaded \
                FROM {}.samples \
                WHERE sha256 = ? AND group in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample get statement")
}

/// build the sample get many prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_many(session: &Session, config: &Conf) -> PreparedStatement {
    // build samples get many prepared statement
    session
        .prepare(format!(
            "SELECT sha256, sha1, md5, id, name, description, group, submitter, origin, uploaded \
                FROM {}.samples \
                WHERE sha256 in ? AND group in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample get many statement")
}

/// Counts the number of submissions for a sample this user can see
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn auth(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags insert prepared statement
    session
        .prepare(format!(
            "SELECT sha256 \
                FROM {}.samples \
                WHERE sha256 in ? AND group in ? \
                GROUP BY sha256",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample auth statement")
}

/// Deletes a sample submission row from scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags insert prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.samples_list \
                WHERE group = ? \
                AND year = ? \
                AND bucket = ? \
                AND uploaded = ? \
                AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample delete statement")
}

/// Deletes a sample submission row from scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete_multiple_groups(session: &Session, config: &Conf) -> PreparedStatement {
    // build tags insert prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.samples_list \
                WHERE group in ? \
                AND year = ? \
                AND bucket = ? \
                AND uploaded = ? \
                AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample delete groups statement")
}

/// Get the submissions with the given sha256, including IDs, groups, and submitters
///
/// This is primarily used to prune unnecessary metadata after a submission is deleted
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_basic_submission_info(session: &Session, config: &Conf) -> PreparedStatement {
    // build sample submission exists prepared statement
    session
        .prepare(format!(
            "SELECT group, submitter, id  \
                FROM {}.samples \
                WHERE sha256 = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample submission exists statement")
}

/// Gets any remaining rows from past ties in listing samples
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build samples list ties prepared statement
    session
        .prepare(format!(
            "SELECT group, sha256, id, uploaded \
                FROM {}.samples_list \
                WHERE group = ? \
                AND year = ? \
                AND bucket = ? \
                AND uploaded = ? \
                AND id <= ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample list ties statement")
}

/// Pull the data needed to list samples
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build samples list ties prepared statement
    session
        .prepare(format!(
            "SELECT group, sha256, id, uploaded \
                FROM {}.samples_list \
                WHERE group = ? \
                AND year = ? \
                AND bucket in ? \
                AND uploaded < ? \
                AND uploaded > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla sample list pull statement")
}
