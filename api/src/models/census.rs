//! The traits used for taking a census of data in Thorium

use scylla::client::session::Session;
use scylla::deserialize::row::DeserializeRow;
use scylla::errors::PrepareError;
use scylla::statement::prepared::PreparedStatement;
use std::fmt::Debug;

/// The census keys for both the count and stream key
pub struct CensusKeys {
    /// The count key
    pub count: String,
    /// The stream key
    pub stream: String,
    /// The bucket for these keys
    pub bucket: i32,
}

/// The trait used for taking a census of data
pub trait CensusSupport: 'static + Send {
    /// The type returned by our prepared statement
    type Row: 'static + Send + Debug + for<'frame, 'metadata> DeserializeRow<'frame, 'metadata>;

    /// Build the prepared statement for getting partition count info
    #[allow(async_fn_in_trait)]
    async fn scan_prepared_statement(
        scylla: &Session,
        ns: &str,
    ) -> Result<PreparedStatement, PrepareError>;

    /// Get the count for this partition
    fn get_count(row: &Self::Row) -> i64;

    /// Get the bucket for this partition
    fn get_bucket(row: &Self::Row) -> i32;

    /// Build the count key for this partition from a census scan row
    fn count_key_from_row(namespace: &str, row: &Self::Row, grouping: i32) -> String;

    /// Build the sorted set key for this census operation from a census scan row
    fn stream_key_from_row(namespace: &str, row: &Self::Row) -> String;
}
