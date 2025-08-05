//! Set up the network policies tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for network policies
pub struct NetworkPoliciesPreparedStatements {
    /// Insert a network policy
    pub insert: PreparedStatement,
    /// Get a network policy
    pub get: PreparedStatement,
    /// Pull the first page of network policies for a cursor
    pub list_pull: PreparedStatement,
    /// Pull more pages of network policies for a cursor
    pub list_pull_more: PreparedStatement,
    /// Get the ties for a network policy cursor
    pub list_ties: PreparedStatement,
    /// Get multiple network policies
    pub get_many: PreparedStatement,
    /// Get the default network policies
    pub get_default: PreparedStatement,
    /// Check if a network policy exists
    pub exists: PreparedStatement,
    /// Delete a network policy
    pub delete: PreparedStatement,
    /// Delete all network policies in a group
    pub delete_all_in_group: PreparedStatement,
}

impl NetworkPoliciesPreparedStatements {
    /// Build a new network policies prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the network policies table
        setup_network_policies_table(session, config).await;
        // set up the network policies materialized views
        setup_network_policies_default_mat_view(session, config).await;
        setup_network_policies_name_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let list_pull = list_pull(session, config).await;
        let list_pull_more = list_pull_more(session, config).await;
        let list_ties = list_ties(session, config).await;
        let get_many = get_many(session, config).await;
        let get_default = get_default(session, config).await;
        let exists = exists(session, config).await;
        let delete = delete(session, config).await;
        let delete_all_in_group = delete_all_in_group(session, config).await;
        // build our prepared statement object
        NetworkPoliciesPreparedStatements {
            insert,
            get,
            list_pull,
            list_pull_more,
            list_ties,
            get_many,
            get_default,
            exists,
            delete,
            delete_all_in_group,
        }
    }
}

/// Set up a network policies table for Thorium
///
/// This is the ground truth for all network policies
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_network_policies_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.network_policies (\
            group TEXT, \
            name TEXT, \
            id UUID, \
            k8s_name TEXT, \
            created TIMESTAMP, \
            ingress TEXT, \
            egress TEXT, \
            forced_policy BOOLEAN, \
            default_policy BOOLEAN, \
            PRIMARY KEY ((group), name))",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add network policies table");
}

/// Set up a network policies by name material view for Thorium
///
/// Allows callers to get all of a network policy's rows by its name more easily
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_network_policies_name_mat_view(session: &Session, config: &Conf) {
    // create network policies by name material view
    let table_create = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.network_policies_by_name AS \
            SELECT name, group, id, k8s_name, created, ingress, egress, forced_policy, default_policy FROM {ns}.network_policies \
            WHERE name IS NOT NULL \
            AND group IS NOT NULL \
            PRIMARY KEY (name, group)",
            ns = &config.thorium.namespace,
        );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add network policies by name materialized view");
}

/// Set up a network policies by name material view for Thorium
///
/// Allows callers to get all of a network policy's rows by its name more easily
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_network_policies_default_mat_view(session: &Session, config: &Conf) {
    // create network policies by name material view
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.network_policies_default AS \
            SELECT group, name, id, default_policy FROM {ns}.network_policies \
            WHERE group IS NOT NULL \
            AND name IS NOT NULL \
            AND default_policy = true \
            PRIMARY KEY (group, name, default_policy)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add network policies default materialized view");
}

/// Creates a network policy
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    session
            .prepare(format!(
                "INSERT INTO {}.network_policies \
                (group, name, id, k8s_name, created, ingress, egress, forced_policy, default_policy) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla network policy insert statement")
}

/// List network policies by group
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT group, name, id \
                FROM {}.network_policies \
                WHERE group = ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policies pull statement")
}

/// List network policies by group starting from a given name
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull_more(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT group, name, id \
                FROM {}.network_policies \
                WHERE group = ? \
                AND name > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policies pull more statement")
}

/// List network policies by group, getting info from ties from the previous query
///
/// A tie occurs when groups have different policies with the same name
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT group, name, id \
                FROM {}.network_policies \
                WHERE group = ? \
                AND name = ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policies ties statement")
}

/// Gets a single network policy by its name and a list of groups it may or
/// may not be in
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    session
            .prepare(format!(
                "SELECT group, name, id, k8s_name, created, ingress, egress, forced_policy, default_policy \
                FROM {}.network_policies_by_name \
                WHERE name = ? \
                AND group in ?",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla network policy get by name statement")
}

/// Gets many network policies from a list of names and groups
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_many(session: &Session, config: &Conf) -> PreparedStatement {
    session
            .prepare(format!(
                "SELECT group, name, id, k8s_name, created, ingress, egress, forced_policy, default_policy \
                FROM {}.network_policies_by_name \
                WHERE name in ? \
                AND group in ?",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla network policy get many statement")
}

/// Get all network policies in a group that should be applied by default if none are specified
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_default(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT group, name, id \
                FROM {}.network_policies_default \
                WHERE group = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policies get default statement")
}

/// Check if a network policy exists in a group
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn exists(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "SELECT name, group \
                FROM {}.network_policies_by_name \
                WHERE name = ? \
                AND group = ? \
                LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policy exists statement")
}

/// Delete a network policy from several groups
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "DELETE FROM {}.network_policies \
                WHERE group in ? \
                AND name = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policy delete statement")
}

/// Delete all network policy rows from a group
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete_all_in_group(session: &Session, config: &Conf) -> PreparedStatement {
    session
        .prepare(format!(
            "DELETE FROM {}.network_policies WHERE group = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla network policy delete all group statement")
}
