//! Set up the results tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for results
pub struct ResultsPreparedStatements {
    /// Insert a result
    pub insert: PreparedStatement,
    /// Get a result
    pub get: PreparedStatement,
    /// Get when a result was uploaded
    pub get_uploaded_by_id: PreparedStatement,
    /// Get the id for a result by key, kind, group, and tool
    pub get_id: PreparedStatement,
    /// Get a results metadata by kind, group, and key
    pub get_with_key: PreparedStatement,
    /// Get a result metadata by kind, group, key, and tool
    pub get_with_key_and_tool: PreparedStatement,
    /// Get the uploaded timestamps for results
    pub get_uploaded: PreparedStatement,
    /// Get the IDs for results by kind, key, and id (used for counting)
    pub count: PreparedStatement,
    /// Delete a result
    pub delete: PreparedStatement,
    /// Update the children for a result
    pub update_children: PreparedStatement,
    /// Insert data into the results stream
    pub insert_stream: PreparedStatement,
    /// Delete data from the results stream
    pub delete_stream: PreparedStatement,
}

impl ResultsPreparedStatements {
    /// Build a new results prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the results tables
        setup_results_table(session, config).await;
        setup_results_stream_table(session, config).await;
        // set up the results materialized views
        setup_results_auth_mat_view(session, config).await;
        setup_results_auth_id_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let get = get(session, config).await;
        let get_uploaded_by_id = get_uploaded_by_id(session, config).await;
        let get_id = get_id(session, config).await;
        let get_with_key = get_with_key(session, config).await;
        let get_with_key_and_tool = get_with_key_and_tool(session, config).await;
        let get_uploaded = get_uploaded(session, config).await;
        let count = count(session, config).await;
        let delete = delete(session, config).await;
        let update_children = update_children(session, config).await;
        let insert_stream = insert_stream(session, config).await;
        let delete_stream = delete_stream(session, config).await;
        // setup our prepared statement object
        ResultsPreparedStatements {
            insert,
            get,
            get_uploaded_by_id,
            get_id,
            get_with_key,
            get_with_key_and_tool,
            get_uploaded,
            count,
            delete,
            update_children,
            insert_stream,
            delete_stream,
        }
    }
}

/// Set up the results stream materialized view
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_results_stream_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.results_stream (\
            kind TEXT,
            group TEXT,
            year INT,
            bucket INT,
            uploaded TIMESTAMP,
            id UUID,
            key TEXT,
            tool TEXT,
            tool_version TEXT,
            display_type TEXT,
            cmd TEXT,
            PRIMARY KEY ((kind, group, year, bucket), uploaded, id, key)) \
            WITH CLUSTERING ORDER BY (uploaded DESC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add result stream table");
}

/// Set up the results table for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_results_table(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.results (\
            id UUID,
            uploaded TIMESTAMP,
            tool TEXT,
            tool_version TEXT,
            cmd TEXT,
            result TEXT,
            files Set<TEXT>,
            display_type TEXT,
            children Map<TEXT, UUID>,
            PRIMARY KEY (id, uploaded)) \
            WITH CLUSTERING ORDER BY (uploaded DESC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add results table");
}

/// Set up the results authorization table for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_results_auth_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    // build cmd for materialized view insert
    let table_create = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.results_auth AS \
            SELECT kind, group, year, bucket, uploaded, id, key, tool, tool_version, display_type, cmd FROM {ns}.results_stream \
            WHERE kind IS NOT NULL \
            AND group IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND uploaded IS NOT NULL \
            AND id IS NOT NULL \
            AND key IS NOT NULL \
            AND tool IS NOT NULL \
            PRIMARY KEY (key, kind, group, tool, id, year, bucket, uploaded)",
            ns = &config.thorium.namespace,
        );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add results auth materialized view");
}

/// Set up the results authorization local index for Thorium
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_results_auth_id_mat_view(session: &Session, config: &Conf) {
    // build cmd for materialized view insert
    let table_create = format!(
        "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.results_ids AS \
            SELECT kind, group, year, bucket, uploaded, id, key, tool FROM {ns}.results_stream \
            WHERE kind IS NOT NULL \
            AND group IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND uploaded IS NOT NULL \
            AND id IS NOT NULL \
            AND key IS NOT NULL \
            PRIMARY KEY (key, id, kind, group, year, bucket, uploaded)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add results id materialized view");
}

/// build the result insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build results insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.results \
                (id, uploaded, tool, tool_version, cmd, result, files, display_type) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla result insert statement")
}

/// build the result get prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build results get prepared statement
    session
        .prepare(format!(
            "SELECT id, tool, tool_version, cmd, uploaded, result, files, display_type, children \
                FROM {}.results \
                WHERE id in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla result get statement")
}

/// build the result get uploaded timestamp prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_uploaded_by_id(session: &Session, config: &Conf) -> PreparedStatement {
    // build results get uploaded timestamp prepared statement
    session
        .prepare(format!(
            "SELECT id, uploaded \
                FROM {}.results \
                WHERE id in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla result get uploaded timestamp statement")
}

/// build the results auth prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_id(session: &Session, config: &Conf) -> PreparedStatement {
    // build results auth prepared statement
    session
        .prepare(format!(
            "SELECT id \
                FROM {}.results_auth \
                WHERE key = ? AND kind = ? AND group in ? AND tool = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla results auth statement")
}

/// build the results auth IDs prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_with_key(session: &Session, config: &Conf) -> PreparedStatement {
    // build results auth IDs prepared statement
    session
        .prepare(format!(
            "SELECT id, tool, display_type, cmd, group, uploaded \
                FROM {}.results_auth \
                WHERE kind = ? AND group in ? AND key = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla results auth ids statement")
}

/// build the results auth IDs restricted by tools prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_with_key_and_tool(session: &Session, config: &Conf) -> PreparedStatement {
    // build results auth IDs restricted by tool prepared statement
    session
        .prepare(format!(
            "SELECT id, tool, display_type, cmd, group, uploaded \
                FROM {}.results_auth \
                WHERE kind = ? AND group in ? AND key = ? AND tool in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla results auth ids restricted by tools statement")
}

/// build the results auth latest prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_uploaded(session: &Session, config: &Conf) -> PreparedStatement {
    // build results auth latest prepared statement
    session
        .prepare(format!(
            "SELECT key, uploaded \
                FROM {}.results_auth \
                WHERE kind = ? AND group = ? AND key in ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla results auth latest statement")
}

/// build the result count prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn count(session: &Session, config: &Conf) -> PreparedStatement {
    // build results count prepared statement
    session
        .prepare(format!(
            "SELECT id \
                FROM {}.results_ids \
                WHERE kind = ? AND key = ? AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla results count statement")
}

/// build the result delete prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn delete(session: &Session, config: &Conf) -> PreparedStatement {
    // build results count prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.results WHERE id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla results delete statement")
}

/// build the result children prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn update_children(session: &Session, config: &Conf) -> PreparedStatement {
    // build results insert prepared statement
    session
        .prepare(format!(
            "UPDATE {}.results \
                SET children[?] = ? \
                WHERE id = ? AND uploaded = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla update result children statement")
}

/// build the result stream insert prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn insert_stream(session: &Session, config: &Conf) -> PreparedStatement {
    // build results stream insert prepared statement
    session
            .prepare(format!(
                "INSERT INTO {}.results_stream \
                (kind, group, year, bucket, key, tool, tool_version, display_type, uploaded, cmd, id) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla result stream insert statement")
}

/// build the result stream delete prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn delete_stream(session: &Session, config: &Conf) -> PreparedStatement {
    // build results stream delete prepared statement
    session
            .prepare(format!(
                "DELETE FROM {}.results_stream \
                WHERE kind = ? AND group = ? AND year = ? AND bucket = ? AND uploaded = ? and id = ?",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla result stream delete statement")
}
