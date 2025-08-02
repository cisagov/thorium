//! Some utilities to help thoradm do what it needs to do

use scylla::client::session::Session;
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;

/// Drop a materialized view in scylla
///
/// # Arguments
///
/// * `name` - The name of the materialized view to delete
/// * `cluster` - The cluster we are deleting a materialized view from
/// * `scylla` - The scylla session to use
pub async fn drop_materialized_view(
    ns: &str,
    name: &str,
    scylla: &Session,
) -> Result<QueryResult, ExecutionError> {
    // Drop the target table
    let table_drop = format!(
        "drop materialized view if exists {ns}.{name}",
        ns = ns,
        name = name
    );
    // execute our query
    scylla.query_unpaged(table_drop, &[]).await
}

/// The shared functions across all traits
pub trait Utils {
    /// The name of the table we are operating on
    fn name() -> &'static str;

    /// Get the pretty name of the table
    fn pretty_name() -> String {
        Self::name().replace('_', " ")
    }
}
