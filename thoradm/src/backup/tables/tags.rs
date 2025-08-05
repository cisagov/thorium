//! Backup and restore support for tags

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
use thorium::models::backends::TagSupport;
use thorium::models::{TagRequest, TagType};
use thorium::Conf;

use crate::args::BackupComponents;
use crate::backup::{utils, Backup, Restore, Scrub, Utils};
use crate::Error;

/// A single line of stage logs
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Tag {
    /// The type of object this tag is for
    pub tag_type: TagType,
    /// The group this tag is a part of
    pub group: String,
    /// The year this tag was submitted
    pub year: i32,
    /// The bucket this tag was submitted in
    pub bucket: i32,
    /// The key for this tag
    pub key: String,
    /// The value for this tag
    pub value: String,
    /// The timestamp this tag was submitted
    pub uploaded: DateTime<Utc>,
    /// The item we are getting tags for
    pub item: String,
}

impl Utils for Tag {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "tags"
    }
}

#[async_trait::async_trait]
impl Backup for Tag {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::Tags
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
                "SELECT type, group, year, bucket, key, value, uploaded, item \
                FROM {}.{} \
                Where token(type, group, year, bucket, key, value) >= ? AND token(type, group, year, bucket, key, value) <= ?",
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
        hasher.write(self.tag_type.as_str().as_bytes());
        hasher.write(self.group.as_bytes());
        hasher.write_i32(self.year);
        hasher.write_i32(self.bucket);
        hasher.write(self.key.as_bytes());
        hasher.write(self.value.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the tags table
impl Scrub for Tag {}

/// Implement restore support for the tags table
#[async_trait::async_trait]
impl Restore for Tag {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "tags_by_item", scylla).await?;
        utils::drop_materialized_view(ns, "tags_case_insensitive", scylla).await?;
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
                (type, group, item, year, bucket, key, value, uploaded, key_lower, value_lower) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        config.thorium.tags.partition_size
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
        let rows = rkyv::check_archived_root::<Vec<Tag>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize our tag type
            let tag_type: TagType = row.tag_type.deserialize(&mut rkyv::Infallible)?;
            // deserialize this rows uploaded timestamp
            let uploaded = row.uploaded.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(uploaded, row.year, partition_size);
            // get the lowercase versions of key/value
            let key_lower = row.key.to_lowercase();
            let value_lower = row.value.to_lowercase();
            // restore this row back to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    tag_type,
                    row.group.as_str(),
                    row.item.as_str(),
                    row.year,
                    bucket,
                    row.key.as_str(),
                    row.value.as_str(),
                    uploaded,
                    key_lower,
                    value_lower,
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

impl<T: TagSupport> Utils for TagRequest<T> {
    /// The name of the table we are operating on
    fn name() -> &'static str {
        "tags"
    }
}
