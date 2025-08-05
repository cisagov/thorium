//! Backup and restore support for commits

use ahash::AHasher;
use bytecheck::CheckBytes;
use chrono::prelude::*;
use futures::stream::{FuturesUnordered, StreamExt};
use indicatif::ProgressBar;
use rkyv::{Archive, Deserialize, Serialize};
use scylla::client::session::Session;
use scylla::errors::{ExecutionError, PrepareError};
use scylla::statement::prepared::PreparedStatement;
use scylla::DeserializeRow;
use std::hash::Hasher;
use std::sync::Arc;
use thorium::models::CommitishKinds;
use thorium::Conf;

use crate::args::BackupComponents;
use crate::backup::{utils, Backup, Restore, Scrub, Utils};
use crate::Error;

/// A single line of stage logs
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Commitish {
    /// The kind of commitish this row represents
    pub kind: CommitishKinds,
    /// The group this commitish is in
    pub group: String,
    /// The repo this commitish is for
    pub repo: String,
    /// The key for this commitish
    pub key: String,
    /// The data for this commitish
    pub data: String,
    /// The hash for this commits repo data blob
    pub repo_data: String,
    /// When this commit was added to its repo
    pub timestamp: DateTime<Utc>,
}

impl Utils for Commitish {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "commitishes"
    }
}

#[async_trait::async_trait]
impl Backup for Commitish {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::Commitish
    }

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
        // build tags get prepared statement
        scylla
            .prepare(format!(
                "SELECT kind, group, repo, key, data, repo_data, timestamp \
                FROM {}.{} \
                Where token(kind, group, repo, key) >= ? AND token(kind, group, repo, key) <= ?",
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
        hasher.write(self.repo.as_bytes());
        hasher.write(self.key.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the tags table
impl Scrub for Commitish {}

/// Implement restore support for the tags table
#[async_trait::async_trait]
impl Restore for Commitish {
    /// The steps to once run before restoring data
    async fn prep(_scylla: &Session, _ns: &str) -> Result<(), ExecutionError> {
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
                (kind, group, repo, key, timestamp, data, repo_data) \
                VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    fn partition_size(config: &Conf) -> u16 {
        config.thorium.repos.partition_size
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
        _partition_size: u16,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<Commitish>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize this rows uploaded timestamp
            let timestamp: DateTime<Utc> = row.timestamp.deserialize(&mut rkyv::Infallible)?;
            // deserialize our commitish kind
            let kind: CommitishKinds = row.kind.deserialize(&mut rkyv::Infallible)?;
            // restore this row back to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    kind,
                    row.group.as_str(),
                    row.repo.as_str(),
                    row.key.as_str(),
                    timestamp,
                    row.data.as_str(),
                    row.repo_data.as_str(),
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
                    // if we have less than 100 future to go then refill our future set
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

/// A single row from the commit list table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct CommitishList {
    /// The kind of commitish this row represents
    pub kind: CommitishKinds,
    /// The group this commitish is in
    pub group: String,
    /// The year this commitish was added to its repo
    pub year: i32,
    /// The bucket for this commitish
    pub bucket: i32,
    /// The repo this commitish is for
    pub repo: String,
    /// When this commitish was added to its repo
    pub timestamp: DateTime<Utc>,
    /// The key of this commitish
    pub key: String,
    /// The hash for this commitish's repo data blob
    pub repo_data: String,
}

impl Utils for CommitishList {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "commitish_list"
    }
}

#[async_trait::async_trait]
impl Backup for CommitishList {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::CommitishList
    }

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
        // build tags get prepared statement
        scylla
            .prepare(format!(
                "SELECT kind, group, year, bucket, repo, timestamp, key, repo_data \
                FROM {}.{} \
                Where token(kind, group, year, bucket, repo) >= ? AND token(kind, group, year, bucket, repo) <= ?",
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
        hasher.write(self.repo.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the tags table
impl Scrub for CommitishList {}

/// Implement restore support for the tags table
#[async_trait::async_trait]
impl Restore for CommitishList {
    /// The steps to once run before restoring data
    ///
    /// # Arguments
    ///
    /// * `scylla` - The scylla client to use
    /// * `ns` - The namespace for any tables to remove/prep
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "committed_repo_data", scylla).await?;
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
                (kind, group, year, bucket, repo, timestamp, key, repo_data) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
    fn partition_size(config: &Conf) -> u16 {
        config.thorium.repos.partition_size
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
        partition_size: u16,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<CommitishList>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize this rows uploaded timestamp
            let timestamp = row.timestamp.deserialize(&mut rkyv::Infallible)?;
            // deserialize our commitish kind
            let kind: CommitishKinds = row.kind.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(timestamp, row.year, partition_size);
            // restore this row back to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    kind,
                    row.group.as_str(),
                    row.year,
                    bucket,
                    row.repo.as_str(),
                    timestamp,
                    row.key.as_str(),
                    row.repo_data.as_str(),
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
                    // if we have less than 100 future to go then refill our future set
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
