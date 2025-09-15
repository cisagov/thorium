//! Setup the entities tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statments for entities
pub struct EntitiesPreparedStatements {
    /// Insert a new entity
    pub insert: PreparedStatement,
    /// Get rows of an entity from a name and list of groups
    pub get: PreparedStatement,
    /// Get rows of entities from a list of names and a list of groups
    pub get_many: PreparedStatement,
    /// Delete rows of an entity from a kind, name, and list of groups
    pub delete: PreparedStatement,
    /// Check if an entity with the given name exists in multiple groups
    pub exists_groups: PreparedStatement,
    /// Get info for listing entities
    pub list_pull: PreparedStatement,
    /// Get the remaining rows from a tie where rows have the same primary key
    /// except for their ids
    pub list_ties: PreparedStatement,
    /// Gets entities' names from their ids
    ///
    /// Used for supplementing name data when listing by tag
    pub get_names_kinds_by_ids: PreparedStatement,
}

impl EntitiesPreparedStatements {
    /// Build a new entities prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // setup the entities tables
        setup_entities_table(session, config).await;
        // setup material views
        setup_entities_id_mat_view(session, config).await;
        setup_entities_name_mat_view(session, config).await;
        // setup prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let get_many = get_many(session, config).await;
        let delete = delete(session, config).await;
        let exists_groups = exists_groups(session, config).await;
        let list_pull = list_pull(session, config).await;
        let list_ties = list_ties(session, config).await;
        let get_names_kinds_by_ids = get_names_kinds_by_ids(session, config).await;
        Self {
            insert,
            get,
            get_many,
            delete,
            exists_groups,
            list_pull,
            list_ties,
            get_names_kinds_by_ids,
        }
    }
}

/// Setup the entities table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_entities_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.entities (\
            kind TEXT,
            group TEXT,
            year INT,
            bucket INT,
            created TIMESTAMP,
            id UUID,
            name TEXT,
            submitter TEXT,
            kind_data TEXT,
            description TEXT,
            image TEXT,
            PRIMARY KEY ((kind, group, year, bucket), created, id))",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to create entities table");
}

/// Setup an entities by id material view for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_entities_id_mat_view(session: &Session, config: &Conf) {
    // create entities by name material view
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.entities_by_id AS \
            SELECT id, group, kind, created, year, bucket, name, submitter, kind_data, description, image FROM {ns}.entities \
            WHERE id IS NOT NULL \
            AND group IS NOT NULL \
            AND kind IS NOT NULL \
            AND created IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND name IS NOT NULL
            PRIMARY KEY (id, group, kind, created, year, bucket, name)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add entities by id materialized view");
}

/// Setup an entities by name material view for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_entities_name_mat_view(session: &Session, config: &Conf) {
    // create entities by name material view
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.entities_by_name AS \
            SELECT name, group, kind, created, year, bucket, id, submitter, kind_data, description, image FROM {ns}.entities \
            WHERE name IS NOT NULL \
            AND group IS NOT NULL \
            AND kind IS NOT NULL \
            AND created IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND id IS NOT NULL
            PRIMARY KEY (name, group, kind, created, year, bucket, id)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add entities by name materialized view");
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
            "INSERT INTO {}.entities \
                (kind, group, year, bucket, created, id, name, submitter, kind_data, description, image) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity insert statement")
}

/// Gets a single entity by its name and a list of groups it may or
/// may not be in along with all of its info
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT id, group, kind, created, name, submitter, kind_data, description, image \
                FROM {}.entities_by_id \
                WHERE id = ? \
                AND group in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity get by id statement")
}

/// Gets many entities by their names and groups it may or
/// may not be in along with all of its info
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_many(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT id, group, kind, created, name, submitter, kind_data, description, image \
                FROM {}.entities_by_id \
                WHERE id in ? \
                AND group in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity get many by name statement")
}

/// Delete rows from the entities table
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "DELETE FROM {}.entities \
                WHERE kind = ? \
                AND group in ? \
                AND year = ? \
                AND bucket = ? \
                AND created = ? \
                AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity delete statement")
}

/// Check if an entity exists in multiple groups
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn exists_groups(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT group \
                FROM {}.entities_by_id \
                WHERE id = ? \
                AND group in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entities exists statement")
}

/// Gets any remaining rows from past ties in listing repos
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo repo list ties prepared statement
    session
        .prepare(format!(
            "SELECT kind, group, created, id, name \
                FROM {}.entities \
                WHERE kind in ? \
                AND group = ? \
                AND year = ? \
                AND bucket = ? \
                AND created = ? \
                AND id <= ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity list ties statement")
}

/// Pulls the data for listing entities in Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build entity list pull prepared statement
    session
        .prepare(format!(
            "SELECT kind, group, created, id, name \
                FROM {}.entities \
                WHERE kind in ? \
                AND group = ? \
                AND year = ? \
                AND bucket in ? \
                AND created < ? \
                AND created > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity list pull statement")
}

/// Gets entities' names from their ids
///
/// Used for supplementing name data when listing by tag
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_names_kinds_by_ids(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT id, name, kind FROM {}.entities_by_id \
                WHERE id in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla entity get by name statement")
}
