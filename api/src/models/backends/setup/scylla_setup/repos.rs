//! Set up the repos tables/prepared statements in Scylla

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;

use crate::Conf;

/// The prepared statements for repos
pub struct ReposPreparedStatements {
    /// Insert a new repo
    pub insert: PreparedStatement,
    /// Update the default checkout for a repo
    pub update_default_checkout: PreparedStatement,
    /// Update the earliest commit timestamp for this repo
    pub update_earliest: PreparedStatement,
    /// Get a repo
    pub get: PreparedStatement,
    /// Get multiple repos
    pub get_many: PreparedStatement,
    /// Check if this repo is visible by any group in a set
    pub auth: PreparedStatement,
    /// Insert a repos data hash
    pub insert_data: PreparedStatement,
    /// Get a repos data hash
    pub get_data: PreparedStatement,
    /// Check if a repos data exists
    pub data_exists: PreparedStatement,
    /// Delete a repos data
    pub delete_data: PreparedStatement,
    /// List the ties for repo list cursor page
    pub list_ties: PreparedStatement,
    /// Get the data for page of repos in a cursor
    pub list_pull: PreparedStatement,
}

impl ReposPreparedStatements {
    /// Build a new repos prepared statement struct
    ///
    /// # Arguments
    ///
    /// * `sessions` - The scylla session to use
    /// * `config` - The Thorium config
    pub async fn new(session: &Session, config: &Conf) -> Self {
        // set up the repos tables
        setup_repos_data_table(session, config).await;
        setup_repos_list_table(session, config).await;
        // set up the repos materialized view
        setup_repos_mat_view(session, config).await;
        // setup our prepared statements
        let insert = insert(session, config).await;
        let update_default_checkout = update_default_checkout(session, config).await;
        let update_earliest = update_earliest(session, config).await;
        let get = get(session, config).await;
        let get_many = get_many(session, config).await;
        let auth = auth(session, config).await;
        let insert_data = insert_data(session, config).await;
        let get_data = get_data(session, config).await;
        let data_exists = data_exists(session, config).await;
        let delete_data = delete_data(session, config).await;
        let list_ties = list_ties(session, config).await;
        let list_pull = list_pull(session, config).await;
        // build our prepared statement object
        ReposPreparedStatements {
            insert,
            update_default_checkout,
            update_earliest,
            get,
            get_many,
            auth,
            insert_data,
            get_data,
            data_exists,
            delete_data,
            list_ties,
            list_pull,
        }
    }
}

/// Set up the repo data table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_repos_data_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.repo_data (\
            hash TEXT,
            repo TEXT,
            PRIMARY KEY ((repo), hash))",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add repo data table");
}

/// Set up the repo list table for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_repos_list_table(session: &Session, config: &Conf) {
    // build cmd for tag table
    let table_create = format!(
        "CREATE TABLE IF NOT EXISTS {ns}.repos_list (\
            group TEXT,
            year INT,
            bucket INT,
            uploaded TIMESTAMP,
            id UUID,
            url TEXT,
            provider TEXT,
            user TEXT,
            name TEXT,
            creator TEXT,
            scheme TEXT,
            default_checkout TEXT,
            earliest TIMESTAMP,
            PRIMARY KEY ((group, year, bucket), uploaded, id)) \
            WITH CLUSTERING ORDER BY (uploaded DESC, id DESC)",
        ns = &config.thorium.namespace,
    );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add repos list table");
}

/// Set up the repo materialized view for Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `config` - The Thorium config
async fn setup_repos_mat_view(session: &Session, config: &Conf) {
    // build cmd for table insert
    let table_create = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS {ns}.repos AS \
            SELECT group, year, bucket, uploaded, id, url, provider, user, name, creator, scheme, default_checkout, earliest FROM {ns}.repos_list \
            WHERE group IS NOT NULL \
            AND year IS NOT NULL \
            AND bucket IS NOT NULL \
            AND uploaded IS NOT NULL \
            AND ID IS NOT NULL \
            AND url IS NOT NULL \
            PRIMARY KEY (url, group, uploaded, year, bucket, id)
            WITH CLUSTERING ORDER BY (uploaded DESC, year DESC, bucket DESC, id DESC)",
            ns = &config.thorium.namespace,
        );
    session
        .query_unpaged(table_create, &[])
        .await
        .expect("failed to add repos materialized view");
}

/// build the repos insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert(session: &Session, config: &Conf) -> PreparedStatement {
    // build repos insert prepared statement
    session
            .prepare(format!(
                "INSERT INTO {}.repos_list \
                (group, year, bucket, uploaded, id, url, provider, user, name, creator, scheme, default_checkout) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla repos insert statement")
}

/// Build the repos update default checkout prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn update_default_checkout(session: &Session, config: &Conf) -> PreparedStatement {
    // build repos update default checkout prepared statement
    session
        .prepare(format!(
            "UPDATE {}.repos_list \
                SET default_checkout = ? \
                WHERE group = ? AND year = ? AND bucket = ? AND uploaded = ? AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repos update default checkout statement")
}

/// Build the repos update earliest prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn update_earliest(session: &Session, config: &Conf) -> PreparedStatement {
    // build repos update earliest prepared statement
    session
        .prepare(format!(
            "UPDATE {}.repos_list \
                SET earliest = ? \
                WHERE group = ? AND year = ? AND bucket = ? AND uploaded = ? AND id = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repos update earliest statement")
}

/// build the repos get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get(session: &Session, config: &Conf) -> PreparedStatement {
    // build repos get prepared statement
    session
            .prepare(format!(
                "SELECT group, provider, user, name, url, id, creator, uploaded, scheme, default_checkout, earliest \
                FROM {}.repos \
                WHERE url = ? AND group in ?",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla repos get statement")
}

/// build the repos get many prepared statement
///
/// # Arguments
///
/// * `session` - The scylla session to use
/// * `config` - The Thorium config
async fn get_many(session: &Session, config: &Conf) -> PreparedStatement {
    // build repos get many prepared statement
    session
            .prepare(format!(
                "SELECT group, provider, user, name, url, id, creator, uploaded, scheme, default_checkout, earliest \
                FROM {}.repos \
                WHERE url in ? AND group in ?",
                &config.thorium.namespace
            ))
            .await
            .expect("Failed to prepare scylla repo get many statement")
}

/// build the repos auth prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn auth(session: &Session, config: &Conf) -> PreparedStatement {
    // build repos auth prepared statement
    session
        .prepare(format!(
            "SELECT url \
                FROM {}.repos \
                WHERE url in ? and group in ?
                group by url
                per partition limit 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repos auth statement")
}

/// build the repo data insert prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn insert_data(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo data insert prepared statement
    session
        .prepare(format!(
            "INSERT INTO {}.repo_data \
                (repo, hash) \
                VALUES (?, ?)",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo data insert statement")
}

/// build the repo data get prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn get_data(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo data get prepared statement
    session
        .prepare(format!(
            "SELECT hash  \
                FROM {}.repo_data \
                WHERE repo = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo data get statement")
}

/// build the repo data exists prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn data_exists(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo data exists prepared statement
    session
        .prepare(format!(
            "SELECT hash  \
                FROM {}.repo_data \
                WHERE repo = ? AND hash = ? \
                LIMIT 1",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo data exists statement")
}

/// build the repo data delete prepared statement
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn delete_data(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo data delete prepared statement
    session
        .prepare(format!(
            "DELETE FROM {}.repo_data \
                WHERE hash = ? \
                AND repo = ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo data delete statement")
}

/// Gets any remaining rows from past ties in listing repos
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_ties(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo list ties prepared statement
    session
        .prepare(format!(
            "SELECT group, url, id, uploaded \
                FROM {}.repos_list \
                WHERE group = ? \
                AND year = ? \
                AND bucket = ? \
                AND uploaded = ? \
                AND id <= ? \
                LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo list ties statement")
}

/// Pulls the data for listing repos in Thorium
///
/// # Arguments
///
/// * `sessions` - The scylla session to use
/// * `conf` - The Thorium config
async fn list_pull(session: &Session, config: &Conf) -> PreparedStatement {
    // build repo list pull prepared statement
    session
        .prepare(format!(
            "SELECT group, url, id, uploaded \
                FROM {}.repos_list \
                WHERE group = ? \
                AND year = ? \
                AND bucket in ? \
                AND uploaded < ? \
                AND uploaded > ? \
                PER PARTITION LIMIT ?",
            &config.thorium.namespace
        ))
        .await
        .expect("Failed to prepare scylla repo list pull statement")
}
