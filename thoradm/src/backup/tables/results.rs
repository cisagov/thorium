//! Backup and restore support for the samples list table

use ahash::AHasher;
use bytecheck::CheckBytes;
use chrono::prelude::*;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use indicatif::ProgressBar;
use rkyv::{Archive, Deserialize, Serialize};
use scylla::client::session::Session;
use scylla::errors::{ExecutionError, PrepareError};
use scylla::statement::prepared::PreparedStatement;
use scylla::DeserializeRow;
use std::collections::HashMap;
use std::hash::Hasher;
use std::path::PathBuf;
use std::sync::Arc;
use thorium::models::OutputDisplayType;
use thorium::Conf;
use uuid::Uuid;

use crate::args::BackupComponents;
use crate::backup::S3Backup;
use crate::backup::S3Restore;
use crate::backup::{utils, Backup, Restore, Scrub, Utils};
use crate::Error;

/// The samples list table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Output {
    /// The unique id for this result
    pub id: Uuid,
    /// The tool or pipeline this result comes from
    pub tool: String,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// When this result was uploaded
    pub uploaded: DateTime<Utc>,
    /// The result
    pub result: String,
    /// An optional file tied to this result
    pub files: Option<Vec<String>>,
    /// The display type of this tool output
    pub display_type: OutputDisplayType,
    /// The children that were found when generating this result
    pub children: Option<HashMap<String, Uuid>>,
}

impl Utils for Output {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "results"
    }
}

/// Implement backup support for the results table
#[async_trait::async_trait]
impl Backup for Output {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::Results
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
        // build output get prepared statement
        scylla
            .prepare(format!(
                "SELECT id, tool, cmd, uploaded, result, files, display_type, children \
                FROM {}.{} \
                Where token(id) >= ? AND token(id) <= ?",
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
        hasher.write(self.id.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the results table
impl Scrub for Output {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for Output {
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
                (id, uploaded, tool, cmd, result, files, display_type) \
                VALUES (?, ?, ?, ?, ?, ?, ?)",
                ns,
                Self::name()
            ))
            .await
    }

    /// Get the partition size for this data type
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    fn partition_size(config: &Conf) -> u16 {
        config.thorium.results.partition_size
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
        let rows = rkyv::check_archived_root::<Vec<Output>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize our display type
            let display_type: OutputDisplayType =
                row.display_type.deserialize(&mut rkyv::Infallible)?;
            // deserialize this rows uploaded timestamp
            let uploaded: DateTime<Utc> = row.uploaded.deserialize(&mut rkyv::Infallible)?;
            // restore this row to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.id,
                    uploaded,
                    row.tool.as_str(),
                    row.cmd.as_ref().map(|cmd| cmd.as_str()),
                    row.result.as_str(),
                    row.files.as_ref().map(|files| {
                        files
                            .iter()
                            .map(|file| file.as_str())
                            .collect::<Vec<&str>>()
                    }),
                    display_type,
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

#[async_trait::async_trait]
impl S3Backup for Output {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::ResultFiles
    }

    /// Get the result files and where to write them off to disk at
    ///
    /// # Arguments
    ///
    /// * `conf` - The config for this Thorium cluster
    /// * `root` - The root path to write objects too
    /// * `chars` - The number of characters to use in order to partition this data on disk
    /// * `buffer` - The slice of data to deserialize and crawl for s3 backup objects
    async fn paths(
        conf: &Conf,
        root: &PathBuf,
        chars: usize,
        buffer: &[u8],
    ) -> Result<Vec<(String, String, PathBuf)>, Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<Output>>(buffer)?;
        // assume each sample will have at least 3 result files
        let mut downloads = Vec::with_capacity(3);
        // crawl our comments and build a list of attachments to write off to disk
        for row in rows.iter() {
            // skip anay rows without files
            if let Some(files) = row.files.as_ref() {
                // crawl the result files in this result
                for file in files.iter() {
                    // get the bucket for result files
                    let bucket = conf.thorium.results.bucket.clone();
                    // build the key for this result file
                    let key = format!("{}/{}", row.id, file);
                    // clone our root path
                    let mut path = root.clone();
                    // get the sub folder for this key
                    path.push(&key[..chars]);
                    // add the key to this path
                    path.push(&key);
                    // add this attachment to our download list
                    downloads.push((bucket, key, path));
                }
            }
        }
        Ok(downloads)
    }
}

impl S3Restore for Output {
    /// Get the bucket and s3 path for this file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file we are restoring
    /// * `conf` - The Thorium for this cluster
    fn parse(path: &PathBuf, conf: &Conf) -> Result<(String, String), Error> {
        // get the bucket for this object
        let bucket = conf.thorium.results.bucket.clone();
        // build an iterator over this paths components
        let chunks = path.components();
        // skip the first 3 components
        let mut chunks = chunks.skip(4);
        // get the sha256 this comment attachment is for
        let result_id = match chunks.next() {
            Some(result_id) => result_id.as_os_str().to_string_lossy(),
            None => return Err(Error::new("result ID is not in path")),
        };
        // get the rest of the result file path
        let file_path = chunks.collect::<PathBuf>().into_os_string();
        // make sure that our file path is not empty
        if file_path.is_empty() {
            return Err(Error::new("Result file path is empty"));
        }
        // build the s3 path for this result file
        let s3_path = format!("{}/{}", result_id, file_path.to_string_lossy());
        Ok((bucket, s3_path))
    }
}

/// The samples list table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct OutputStream {
    /// The kind of result this is for
    pub kind: String,
    /// The group this result is in
    pub group: String,
    /// The year this result was added
    pub year: i32,
    /// The bucket this result is in
    pub bucket: i32,
    /// The key for this result
    pub key: String,
    /// The tool this result comes from
    pub tool: String,
    /// The display type of this tool output
    pub display_type: OutputDisplayType,
    /// The timestamp for this tool
    pub uploaded: DateTime<Utc>,
    /// The command that was run to generate these results
    pub cmd: Option<String>,
    /// The unique id for this result
    pub id: Uuid,
}

impl Utils for OutputStream {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "results_stream"
    }
}

/// Implement backup support for the results table
#[async_trait::async_trait]
impl Backup for OutputStream {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::ResultsStream
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
        // build output get prepared statement
        scylla
            .prepare(format!(
                "SELECT kind, group, year, bucket, key, tool, display_type, uploaded, cmd, id \
                FROM {}.{} \
                Where token(kind, group, year, bucket) >= ? AND token(kind, group, year, bucket) <= ?",
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
        hasher.write(self.kind.as_bytes());
        hasher.write(self.group.as_bytes());
        hasher.write_i32(self.year);
        hasher.write_i32(self.bucket);
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the results table
impl Scrub for OutputStream {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for OutputStream {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "results_auth", scylla).await?;
        utils::drop_materialized_view(ns, "results_ids", scylla).await?;
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
                (kind, group, year, bucket, key, tool, display_type, uploaded, cmd, id) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ns,
                Self::name()
            ))
            .await
    }

    /// Get the partition size for this data type
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    fn partition_size(config: &Conf) -> u16 {
        config.thorium.results.partition_size
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
        let rows = rkyv::check_archived_root::<Vec<OutputStream>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize our display type
            let display_type: OutputDisplayType =
                row.display_type.deserialize(&mut rkyv::Infallible)?;
            // deserialize this rows uploaded timestamp
            let uploaded = row.uploaded.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(uploaded, row.year, partition_size);
            // restore this row to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.kind.as_str(),
                    row.group.as_str(),
                    row.year,
                    bucket,
                    row.key.as_str(),
                    row.tool.as_str(),
                    display_type,
                    uploaded,
                    row.cmd.as_ref().map(|cmd| cmd.as_str()),
                    row.id,
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
