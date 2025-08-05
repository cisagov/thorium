//! Set up the nodes tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for nodes
pub struct NodesPreparedStatements {
    /// Insert a node
    pub insert: PreparedStatement,
    /// Get a node
    pub get: PreparedStatement,
    /// Get multiple nodes
    pub get_many: PreparedStatement,
    /// update a node
    pub update: PreparedStatement,
    /// Update the heart beat value for a node
    pub update_heart_beat: PreparedStatement,
    /// Get the ties when listing nodes
    pub list_ties: PreparedStatement,
    /// Get all the nodes for a specific cluster
    pub list: PreparedStatement,
    /// Get the ties when listing node details for a specific cluster
    pub list_details_ties: PreparedStatement,
    /// Get the next page of node details
    pub list_details: PreparedStatement,
}

impl NodesPreparedStatements {
    /// Build a new nodes prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the node table
        setup_node_table(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let get_many = get_many(session, config).await;
        let update = update(session, config).await;
        let update_heart_beat = update_heart_beat(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list = list(session, config).await;
        let list_details_ties = list_details_ties(session, config).await;
        let list_details = list_details(session, config).await;
        // build our prepared statement object
        NodesPreparedStatements {
            insert,
            get,
            get_many,
            update,
            update_heart_beat,
            list_ties,
            list,
            list_details_ties,
            list_details,
        }
    }
}

/// Set up the nodes table
///
/// This table tracks nodes and their health
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_node_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.nodes (\
            cluster TEXT,
            node TEXT,
            health TEXT,
            resources TEXT,
            heart_beat TIMESTAMP,
            PRIMARY KEY ((cluster), node))",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add node table");
}

/// build the node register insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build node register prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.nodes \
                (cluster, node, health, resources) \
                VALUES (?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node register insert statement")
}

/// build the node get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build node get prepared statement
    session
        .prepare(format!(
            "SELECT cluster, node, health, resources, heart_beat \
                FROM {}.nodes \
                WHERE cluster = ? AND node = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node get statement")
}

/// build the node get many prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_many(session: &Session, config: &Conf) -> PreparedStatement {
    // build node get many prepared statement
    session
        .prepare(format!(
            "SELECT cluster, node, health, resources, heart_beat \
                FROM {}.nodes \
                WHERE cluster in ? AND node in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node get many statement")
}

/// build the node update without the updating the heart beat
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn update(session: &Session, config: &Conf) -> PreparedStatement {
    // build node update prepared statement
    session
        .prepare(format!(
            "UPDATE {}.nodes \
                SET health = ?, resources = ? \
                WHERE cluster = ? AND node = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node update statement")
}

/// build the node update with updating the heart beat
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn update_heart_beat(session: &Session, config: &Conf) -> PreparedStatement {
    // build node update heart beat prepared statement
    session
        .prepare(format!(
            "UPDATE {}.nodes \
                SET health = ?, resources = ?, heart_beat = ?\
                WHERE cluster = ? AND node = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node heart beat update statement")
}

/// build the node list ties prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build node list ties prepared statement
    session
        .prepare(format!(
            "SELECT cluster, node \
                FROM {}.nodes \
                WHERE cluster = ? AND node > ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node list ties statement")
}

/// build the node list prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list(session: &Session, config: &Conf) -> PreparedStatement {
    // build node list prepared statement
    session
        .prepare(format!(
            "SELECT cluster, node \
                FROM {}.nodes \
                WHERE cluster = ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node list statement")
}

/// build the node list details ties prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_details_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build node list details ties prepared statement
    session
        .prepare(format!(
            "SELECT cluster, node, health, resources, heart_beat \
                FROM {}.nodes \
                WHERE cluster = ? AND node > ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node list details ties statement")
}

/// build the node list details prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_details(session: &Session, config: &Conf) -> PreparedStatement {
    // build node list details prepared statement
    session
        .prepare(format!(
            "SELECT cluster, node, health, resources, heart_beat \
                FROM {}.nodes \
                WHERE cluster = ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla node list details statement")
}
