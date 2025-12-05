//! Setup the associations table in scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for Associations
pub struct AssociationsPreparedStatements {
    /// Insert an association into scylla
    pub insert: PreparedStatement,
    /// Delete an association
    pub delete: PreparedStatement,
    /// List the ties for a associations cursor
    pub list_ties: PreparedStatement,
    /// Get a page of data for a associations cursor
    pub list_pull: PreparedStatement,
}

impl AssociationsPreparedStatements {
    /// Build a new associations prepared statements struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // setup our tables
        setup_associations_table(session, config).await;
        //setup_associations_to_table(session, config).await;
        // build our prepared statements
        let insert = insert(session, config).await;
        let delete = delete(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list_pull = list_pull(session, config).await;
        // build our prepared statements object
        AssociationsPreparedStatements {
            insert,
            delete,
            list_ties,
            list_pull,
        }
    }
}

/// The the associations from table for Thorium
///
/// This table contains the associations from a specific object or entity
async fn setup_associations_table(session: &Session, config: &Conf) {
    // build the command for creating this table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.associations ( \
          group TEXT, \
          year INT, \
          bucket INT, \
          created TIMESTAMP, \
          direction TEXT, \
          kind TEXT, \
          source TEXT, \
          target TEXT, \
          submitter TEXT, \
          extra_source TEXT, \
          extra_target TEXT, \
          PRIMARY KEY ((group, year, bucket, source), created, target, direction))
          WITH CLUSTERING ORDER BY (created DESC)",
        ns = &config.thorium.namespace
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("Failed to add assocations table");
}

/// build the associations insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build associations_from insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.associations \
                (group, year, bucket, created, direction, kind, source, target, submitter, extra_source, extra_target) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare associations insert statement")
}

/// build the associations insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build associations_from insert prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.associations \
                WHERE group in ? \
                AND year = ? \
                AND bucket = ? \
                AND source = ? \
                AND created = ? \
                AND target = ? \
                AND direction = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare associations delete statement")
}

/// Gets any remaining rows from past ties in listing associations
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build associations list ties prepared statement
    session
        .prepare(format!(
            "SELECT group, kind, source, target, created, submitter, direction, extra_source, extra_target \
                FROM {}.associations \
                WHERE group = ? \
                AND year = ? \
                AND bucket = ? \
                AND source = ?
                AND created = ? \
                AND target <= ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla associations list ties statement")
}

/// Pull the data needed to list associations
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build associations list ties prepared statement
    session
        .prepare(format!(
            "SELECT group, kind, source, target, submitter, created, direction, extra_source, extra_target \
                FROM {}.associations \
                WHERE group = ? \
                AND year = ? \
                AND bucket in ? \
                AND source = ?
                AND created < ? \
                AND created > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla associations list pull statement")
}
