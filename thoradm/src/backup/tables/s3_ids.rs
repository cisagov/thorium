//! Backup and restore support for s3_ids

use ahash::AHasher;
use bytecheck::CheckBytes;
use futures::stream::{FuturesUnordered, StreamExt};
use indicatif::ProgressBar;
use rkyv::{Archive, Deserialize, Serialize};
use scylla::client::session::Session;
use scylla::errors::{ExecutionError, PrepareError};
use scylla::statement::prepared::PreparedStatement;
use scylla::DeserializeRow;
use std::hash::Hasher;
use std::path::PathBuf;
use std::sync::Arc;
use thorium::models::{ArchivedS3Objects, S3Objects};
use thorium::Conf;
use uuid::Uuid;

use crate::args::BackupComponents;
use crate::backup::{utils, Backup, Restore, S3Backup, S3Restore, Scrub, Utils};
use crate::Error;

/// A single line of stage logs
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct S3Id {
    /// The type of object this id is for
    pub object_type: S3Objects,
    /// The id of this object
    pub id: Uuid,
    /// The sha256 of this object
    pub sha256: String,
}

impl Utils for S3Id {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "s3_ids"
    }
}

#[async_trait::async_trait]
impl Backup for S3Id {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::S3Ids
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
        // build logs get prepared statement
        scylla
            .prepare(format!(
                "SELECT type, id, sha256 \
                FROM {}.{} \
                Where token(type, id) >= ? AND token(type, id) <= ?",
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
        hasher.write(self.object_type.as_str().as_bytes());
        hasher.write(self.id.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for S3Id {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for S3Id {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "s3_sha256s", scylla).await?;
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
                (type, id, sha256) \
                VALUES (?, ?, ?)",
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
    fn partition_size(_config: &Conf) -> u16 {
        0
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
        let rows = rkyv::check_archived_root::<Vec<S3Id>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize our object type
            let object_type: S3Objects = row.object_type.deserialize(&mut rkyv::Infallible)?;
            // restore this row back to scylla
            let query =
                scylla.execute_unpaged(prepared, (object_type, row.id, row.sha256.as_str()));
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
impl S3Backup for S3Id {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::S3IdsObjects
    }

    /// Get the s3 urls and where to write them off to disk at
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
        let rows = rkyv::check_archived_root::<Vec<S3Id>>(buffer)?;
        // crawl our rows and build the paths to write to s3
        let paths = rows
            .iter()
            .map(|row| {
                // clone our root path
                let mut path = root.clone();
                // get this rows bucket
                match row.object_type {
                    ArchivedS3Objects::File => {
                        // get the bucket for files
                        let bucket = conf.thorium.files.bucket.clone();
                        // get the key for this file in s3
                        let key = row.id.to_string();
                        // nest this in either a file folder or a repos folder
                        path.push("files");
                        // get the sub folder for this path
                        path.push(&key[..chars]);
                        // add the sha256 to this path
                        path.push(&key);
                        (bucket, key, path)
                    }
                    ArchivedS3Objects::Repo => {
                        // get the bucket for files
                        let bucket = conf.thorium.repos.bucket.clone();
                        // get the key for this file in s3
                        let key = row.id.to_string();
                        // nest this in either a file folder or a repos folder
                        path.push("repos");
                        // get the sub folder for this path
                        path.push(&key[..chars]);
                        // add the sha256 to this path
                        path.push(&key);
                        (bucket, key, path)
                    }
                }
            })
            .collect();
        Ok(paths)
    }
}

impl S3Restore for S3Id {
    /// Get the bucket and s3 path for this file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file we are restoring
    /// * `conf` - The Thorium for this cluster
    fn parse<'a>(path: &PathBuf, conf: &'a Conf) -> Result<(String, String), Error> {
        // build an iterator over this paths components
        let chunks = path.components();
        // skip the first 3 components
        let mut chunks = chunks.skip(3);
        // get the bucket for this sample
        let bucket = match chunks.next().map(|comp| comp.as_os_str().to_str()) {
            Some(Some("files")) => conf.thorium.files.bucket.clone(),
            Some(Some("repos")) => conf.thorium.repos.bucket.clone(),
            _ => return Err(Error::new("Unknown s3 id object type")),
        };
        // skip the partitioning component
        let mut chunks = chunks.skip(1);
        // get our s3 path
        let s3_path = match chunks.next() {
            Some(s3_path) => s3_path.as_os_str().to_string_lossy().to_string(),
            None => return Err(Error::new("S3 ids path is not long enough")),
        };
        Ok((bucket, s3_path))
    }
}
