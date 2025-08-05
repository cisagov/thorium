//! Backup and restore support for comments

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
use std::collections::HashMap;
use std::hash::Hasher;
use std::path::PathBuf;
use std::sync::Arc;
use thorium::Conf;
use uuid::Uuid;

use crate::args::BackupComponents;
use crate::backup::{utils, Backup, Restore, S3Backup, S3Restore, Scrub, Utils};
use crate::Error;

/// A single line of stage logs
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Comment {
    /// The group to share this comment with
    pub group: String,
    /// The sha256 this comment was for
    pub sha256: String,
    /// When this comment was uploaded
    pub uploaded: DateTime<Utc>,
    /// The uuid for this comment
    pub id: Uuid,
    /// The author for this comment
    pub author: String,
    /// The comment for this file
    pub comment: String,
    /// Any paths in s3 to files/attachments for this comment
    pub files: String,
}

impl Utils for Comment {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "comments"
    }
}

#[async_trait::async_trait]
impl Backup for Comment {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::Comments
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
                "SELECT group, sha256, uploaded, id, author, comment, files \
                FROM {}.{} \
                Where token(group, sha256) >= ? AND token(group, sha256) <= ?",
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
        hasher.write(self.sha256.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for Comment {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for Comment {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "comments_by_id", scylla).await?;
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
                (group, sha256, uploaded, id, author, comment, files) \
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
    fn partition_size(_config: &Conf) -> u16 {
        0
    }

    /// Restore a single partition
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer we should restore data from
    /// * `scylla` - The client to use when talking to scylla
    /// * `partition_size` - The partition size to use when restoring data
    /// * `rows_restored` - The number of rows that have been restored
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
        let rows = rkyv::check_archived_root::<Vec<Comment>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize this rows uploaded timestamp
            let uploaded: DateTime<Utc> = row.uploaded.deserialize(&mut rkyv::Infallible)?;
            // restore this row to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.group.as_str(),
                    row.sha256.as_str(),
                    uploaded,
                    row.id,
                    row.author.as_str(),
                    row.comment.as_str(),
                    row.files.as_str(),
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
impl S3Backup for Comment {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::CommentAttachments
    }

    /// Get the comment attachments and where to write them off to disk at
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
        let rows = rkyv::check_archived_root::<Vec<Comment>>(buffer)?;
        // start with an empty list since most comments probably won't have attachments
        let mut downloads = Vec::default();
        // crawl our comments and build a list of attachments to write off to disk
        for row in rows.iter() {
            // deserialize our files map
            let files: HashMap<String, Uuid> = serde_json::from_str(row.files.as_str())?;
            // crawl the attachments for this comment
            for (_, id) in files.iter() {
                // get the bucket for comment attachments
                let bucket = conf.thorium.attachments.bucket.clone();
                // get the key for this attachment
                let key = format!("{}/{}/{}", row.sha256, row.id, id);
                // clone our root path
                let mut path = root.clone();
                // get the sub folder for this key
                path.push(&row.sha256[..chars]);
                // add the rest of the key to this path
                path.push(row.sha256.as_str());
                path.push(row.id.to_string());
                path.push(id.to_string());
                // add this attachment to our download list
                downloads.push((bucket, key, path));
            }
        }
        Ok(downloads)
    }
}

impl S3Restore for Comment {
    /// Get the bucket and s3 path for this file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file we are restoring
    /// * `conf` - The Thorium for this cluster
    fn parse(path: &PathBuf, conf: &Conf) -> Result<(String, String), Error> {
        // get the bucket for this object
        let bucket = conf.thorium.attachments.bucket.clone();
        // build an iterator over this paths components
        let chunks = path.components();
        // skip the first 3 components
        let mut chunks = chunks.skip(4);
        // get the sha256 this comment attachment is for
        let sha256 = match chunks.next() {
            Some(sha256) => sha256.as_os_str().to_string_lossy(),
            None => return Err(Error::new("sha256 is not in path")),
        };
        // get the id of this attachments comment
        let comment_id = match chunks.next() {
            Some(comment_id) => comment_id.as_os_str().to_string_lossy(),
            None => return Err(Error::new("sha256 is not in path")),
        };
        // get this attachments id
        let attachment_id = match chunks.next() {
            Some(attachment_id) => attachment_id.as_os_str().to_string_lossy(),
            None => return Err(Error::new("sha256 is not in path")),
        };
        // build the s3 path for this comment attachment
        let s3_path = format!("{}/{}/{}", sha256, comment_id, attachment_id);
        Ok((bucket, s3_path))
    }
}
