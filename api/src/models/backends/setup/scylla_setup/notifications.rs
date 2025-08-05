//! Set up the notifications tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for notifications
pub struct NotificationsPreparedStatements {
    /// Insert a new notification
    pub insert: PreparedStatement,
    /// Insert a new notification that doesn't expire
    pub insert_no_expire: PreparedStatement,
    /// Get all notifications for a specific entity
    pub get: PreparedStatement,
    /// Delete a notification
    pub delete: PreparedStatement,
    /// Delete all notifications for a specific entity
    pub delete_all: PreparedStatement,
}

impl NotificationsPreparedStatements {
    /// Build a new notifications prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the notifications table
        setup_notifications_table(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let insert_no_expire = insert_no_expire(session, config).await;
        let get = get(session, config).await;
        let delete = delete(session, config).await;
        let delete_all = delete_all(session, config).await;
        // build our prepared statement object
        NotificationsPreparedStatements {
            insert,
            insert_no_expire,
            get,
            delete,
            delete_all,
        }
    }
}

/// Set up a notifications table for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_notifications_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.notifications (\
            kind TEXT, \
            key TEXT, \
            created TIMESTAMP, \
            id UUID, \
            msg TEXT, \
            level TEXT, \
            ban_id UUID, \
            PRIMARY KEY ((kind, key), created, id))
            WITH default_time_to_live = {ttl}",
        ns = &config.thorium.namespace,
        ttl = &config.thorium.retention.notifications
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add notifications table");
}

/// Inserts a new image log into scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build notification insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.notifications \
                (kind, key, created, id, msg, level, ban_id) \
                VALUES (?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla notification insert statement")
}

/// Inserts a new notification into scylla that will not expire
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert_no_expire(session: &Session, config: &Conf) -> PreparedStatement {
    // build notification no expire insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.notifications \
                (kind, key, created, id, msg, level, ban_id) \
                VALUES (?, ?, ?, ?, ?, ?, ?) \
                USING TTL 0",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla notification insert no expire statement")
}

/// Gets all notifications for a given entity from scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build notifications get prepared statement
    session
        .prepare(format!(
            "SELECT key, created, id, msg, level, ban_id \
                 FROM {}.notifications \
                 WHERE kind = ? AND key = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla notifications get statement")
}

/// Deletes a specific notification
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build notification delete prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.notifications \
                WHERE kind = ? \
                AND key = ? \
                AND created = ? \
                AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla notification delete statement")
}

/// Deletes all notifications for a given entity
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete_all(session: &Session, config: &Conf) -> PreparedStatement {
    // build notifications delete all prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.notifications \
                WHERE kind = ? AND key = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla notifications delete all statement")
}
