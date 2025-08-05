//! Backup and restore support for the samples list table

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
use thorium::Conf;
use uuid::Uuid;

use crate::args::BackupComponents;
use crate::backup::{utils, Backup, Restore, Scrub, Utils};
use crate::Error;

/// The samples list table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct SamplesList {
    /// The group this sample is in
    pub group: String,
    /// The year this sample is from
    pub year: i32,
    /// the bucket this sample is in
    pub bucket: i32,
    /// The sha256 of this sample
    pub sha256: String,
    /// The sha1 of this sample
    pub sha1: String,
    /// The md5 of this sample
    pub md5: String,
    /// A UUID for this submission
    pub id: Uuid,
    /// The name of this sample if one was specified
    pub name: Option<String>,
    /// A description for this sample
    pub description: Option<String>,
    /// The user who submitted this sample
    pub submitter: String,
    /// Where this sample originates from if anywhere in serial form
    pub origin: Option<String>,
    // When this sample was uploaded
    pub uploaded: DateTime<Utc>,
}

impl Utils for SamplesList {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "samples_list"
    }
}

/// Implement backup support for the samples list table
#[async_trait::async_trait]
impl Backup for SamplesList {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::SamplesList
    }

    /// The prepared statement to use when retrieving data from Scylla
    ///
    /// # Arguments
    ///
    /// * `scylla` - A scylla client
    /// * `ns` - The namespace in scylla this table is from
    async fn prepared_statement(
        scylla: &Session,
        ns: &str,
    ) -> Result<PreparedStatement, PrepareError> {
        // build samples insert prepared statement
        scylla
            .prepare(format!(
                "Select group, year, bucket, sha256, sha1, md5, id, name, description, submitter, origin, uploaded \
                From {}.{} \
                Where token(group, year, bucket) >= ? AND token(group, year, bucket) <= ?",
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
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for SamplesList {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for SamplesList {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "samples", scylla).await?;
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
                (group, year, bucket, sha256, sha1, md5, id, name, description, submitter, origin, uploaded) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        config.thorium.files.partition_size
    }

    /// Restore a single partition
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer we should restore data from
    /// * `scylla` - The client to use when talking to scylla
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
        let rows = rkyv::check_archived_root::<Vec<SamplesList>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize this rows uploaded timestamp
            let uploaded = row.uploaded.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(uploaded, row.year, partition_size);
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.group.as_str(),
                    row.year,
                    bucket,
                    row.sha256.as_str(),
                    row.sha1.as_str(),
                    row.md5.as_str(),
                    row.id,
                    row.name.as_ref().map(|name| name.as_str()),
                    row.description
                        .as_ref()
                        .map(|description| description.as_str()),
                    row.submitter.as_str(),
                    row.origin.as_ref().map(|origin| origin.as_str()),
                    uploaded,
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
