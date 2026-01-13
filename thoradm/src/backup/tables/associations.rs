//! Backup and restore support for associations

use ahash::AHasher;
use chrono::{DateTime, Utc};
use futures::{StreamExt, stream::FuturesUnordered};
use indicatif::ProgressBar;
use rkyv::{Archive, CheckBytes, Deserialize, Serialize, string::ArchivedString};
use scylla::{
    DeserializeRow,
    client::session::Session,
    errors::{ExecutionError, PrepareError},
    statement::prepared::PreparedStatement,
};
use std::{hash::Hasher, sync::Arc};
use thorium::{
    Conf,
    models::{AssociationKind, Directionality},
};

use crate::Error;
use crate::backup::{Backup, Restore, Scrub, Utils};

/// A single row from the associations table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Association {
    /// The group this association is for
    pub group: String,
    /// The year the association was created
    pub year: i32,
    /// The bucket the association is in
    pub bucket: i32,
    /// When this association was created
    pub created: DateTime<Utc>,
    /// The direction for this association
    pub direction: Directionality,
    /// The kind of association this is
    pub kind: AssociationKind,
    /// The source for this association
    pub source: String,
    /// The other serialized data this association is with
    pub other: String,
    /// Who created this association
    pub submitter: String,
    /// Any extra info needed for the source column in this row
    pub extra_source: Option<String>,
    /// Any extra info needed for the target column in this row
    pub extra_other: Option<String>,
}

impl Utils for Association {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "associations"
    }
}

#[async_trait::async_trait]
impl Backup for Association {
    /// The prepared statement to use when retrieving data from Scylla
    ///
    /// # Arguments
    ///
    /// * `scylla` - The scylla session to build a prepared statement with
    /// * `ns` - The namespace for this prepared statement
    async fn prepared_statement(
        scylla: &Session,
        ns: &str,
    ) -> Result<PreparedStatement, PrepareError> {
        // build logs get prepared statement
        scylla
            .prepare(format!(
                "SELECT group, year, bucket, created, direction, kind, source, target, submitter, extra_source, extra_target \
                FROM {}.{} \
                WHERE token(group, year, bucket, source) >= ? AND token(group, year, bucket, source) <= ?",
                    ns,
                    Self::name(),
                ))
                .await
    }

    /// Hash this partitions info to see if we have changed partitions
    fn hash_partition(&self) -> u64 {
        // build a new hasher
        let mut hasher = AHasher::default();
        // ingest our partition key
        hasher.write(self.group.as_bytes());
        hasher.write_i32(self.year);
        hasher.write_i32(self.bucket);
        hasher.write(self.source.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for Association {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for Association {
    // The partition size is constant, so the partition config is just
    // the size itself
    type PartitionConf = u16;

    /// The steps to once run before restoring data
    async fn prep(_scylla: &Session, _ns: &str) -> Result<(), ExecutionError> {
        // no prep steps needed
        Ok(())
    }

    /// The prepared statement to use when restoring data to scylla
    ///
    /// # Arguments
    ///
    /// * `scylla` - A scylla client
    /// * `ns` - The namespace in scylla this table is from
    async fn prepared_statement(
        scylla: &Session,
        ns: &str,
    ) -> Result<PreparedStatement, PrepareError> {
        scylla
            .prepare(format!(
                "INSERT INTO {}.{} \
                (group, year, bucket, created, direction, kind, source, target, submitter, extra_source, extra_target) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ns,
                Self::name(),
            ))
            .await
    }

    /// Get the partition size for this data type
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    fn partition_conf(config: &Conf) -> u16 {
        config.thorium.associations.partition_size
    }

    /// Restore a single partition
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer we should restore data from
    /// * `scylla` - The client to use when talking to scylla
    /// * `rows_restored` - The number of rows that have been restored
    /// * `partition_size` - The partition size to use when restoring data
    /// * `progress` - The bar to report progress with
    /// * `prepared` - The prepared statement to inject data with
    async fn restore<'a>(
        buffer: &'a [u8],
        scylla: &Arc<Session>,
        partition_size: &u16,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<Association>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            let direction: Directionality = row.direction.deserialize(&mut rkyv::Infallible)?;
            let kind: AssociationKind = row.kind.deserialize(&mut rkyv::Infallible)?;
            // deserialize this rows uploaded timestamp
            let created = row.created.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(created, row.year, *partition_size);
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.group.as_str(),
                    row.year,
                    bucket,
                    created,
                    direction,
                    kind,
                    row.source.as_str(),
                    row.other.as_str(),
                    row.submitter.as_str(),
                    row.extra_source.as_ref().map(ArchivedString::as_str),
                    row.extra_other.as_ref().map(ArchivedString::as_str),
                ),
            );
            // add this to our futures
            futures.push(query);
            // if we have 1000 futures then wait for at least 500 of them to complete
            if futures.len() > 1000 {
                // poll our futures until one is complete
                while let Some(query_result) = futures.next().await {
                    // raise any errors
                    query_result?;
                    // increment our restored row count
                    *rows_restored += 1;
                    // set our current row count progress message
                    progress.set_message(rows_restored.to_string());
                    // if we have less then 100 future to go then refill our future set
                    if futures.len() < 100 {
                        break;
                    }
                }
            }
        }
        // poll our futures until one is complete
        while let Some(query_result) = futures.next().await {
            // raise any errors
            query_result?;
            // increment our restored row count
            *rows_restored += 1;
            // set our current row count progress message
            progress.set_message(rows_restored.to_string());
        }
        Ok(())
    }
}
