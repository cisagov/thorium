//! Set up the tools tables/prepared statements in Scylla
//!
//! This isn't used yet but it will be soon.

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for tools
#[allow(dead_code)]
pub struct ToolsPreparedStatements {
    /// Insert a tool
    pub insert: PreparedStatement,
    /// Insert a tool into the tool list table
    pub insert_list: PreparedStatement,
    ///// Get a tool
    ////pub get: PreparedStatement,
}

#[allow(dead_code)]
impl ToolsPreparedStatements {
    /// Build a new tools prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // setup our tables
        setup_tools_table(session, config).await;
        setup_tools_list_table(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let insert_list = insert_list(session, config).await;
        ToolsPreparedStatements {
            insert,
            insert_list,
        }
    }
}

/// Set up the tools tables
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_tools_table(session: &Session, config: &Conf) {
    // build the cmd for creating the tools table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.tools (\
            id UUID, \
            name TEXT, \
            creator TEXT, \
            version TEXT, \
            scaler TEXT, \
            image TEXT, \
            lifetime TEXT, \
            timeout BIGINT, \
            resources TEXT, \
            spawn_limit TEXT, \
            env TEXT, \
            volumes TEXT, \
            args TEXT, \
            modifiers TEXT, \
            description TEXT, \
            security_context TEXT, \
            collect_logs BOOLEAN, \
            generator BOOLEAN, \
            dependencies TEXT, \
            display_type TEXT, \
            output_collection TEXT, \
            clean_up TEXT, \
            kvm TEXT, \
            bans TEXT, \
            network_policies TEXT, \
            created TIMESTAMP, \
            PRIMARY KEY (id))",
        ns = &config.thorium.namespace,
    );
    // create the tools table
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add tools table");
}

/// Set up the tools listing tables
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_tools_list_table(session: &Session, config: &Conf) {
    // build the cmd for creating the tools table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.tools_list (\
            group TEXT, \
            name TEXT, \
            created TIMESTAMP, \
            id UUID, \
            PRIMARY KEY ((group, name), created, id))",
        ns = &config.thorium.namespace,
    );
    // create the tools table
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add tools list table");
}

/// Inserts a new tool into scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build tools insert prepared statement
    session
            .prepare(format!(
                "INSERT INTO {}.tools \
                (id, name, creator, version, scaler, image, lifetime, timeout, resources, spawn_limit, env, volumes, args, modifiers, description, security_context, collect_logs, generator, dependencies, display_type, output_collection, clean_up, kvm, bans, network_policies) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla tools insert statement")
}

/// Inserts a new tool list entry into scylla
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert_list(session: &Session, config: &Conf) -> PreparedStatement {
    // build tool list insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.tools_list \
                (group, name, created, id) \
                VALUES (?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla tools list insert statement")
}
