//! Backup and restore support for repos

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

/// A single line of stage logs
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct RepoList {
    /// The group this repo is in
    pub group: String,
    /// The year this repo was uploaded
    pub year: i32,
    /// The bucket this repo is in
    pub bucket: i32,
    /// When this repo was uploaded
    pub uploaded: DateTime<Utc>,
    /// The id for this repo
    pub id: Uuid,
    /// The user that added this repo to Thorium
    pub creator: String,
    /// The name of this repo
    pub name: String,
    /// Where this repo comes from (github/gitlab/...)
    pub provider: String,
    /// The scheme to use with this repo
    pub scheme: String,
    /// The url of this repo
    pub url: String,
    /// The user that uploaded this repo to its provider
    pub user: String,
}

impl Utils for RepoList {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "repos_list"
    }
}

#[async_trait::async_trait]
impl Backup for RepoList {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::RepoList
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
        // build repos get prepared statement
        scylla
            .prepare(format!(
                "SELECT group, year, bucket, uploaded, id, creator, name, provider, scheme, url, user \
                FROM {}.{} \
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

/// Implement scrub support for the repos table
impl Scrub for RepoList {}

/// Implement restore support for the repos table
#[async_trait::async_trait]
impl Restore for RepoList {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "repos", scylla).await?;
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
                (group, year, bucket, uploaded, id, url, provider, user, name, creator, scheme) \
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
    fn partition_size(config: &Conf) -> u16 {
        config.thorium.repos.partition_size
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
        partition_size: u16,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<RepoList>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize this rows uploaded timestamp
            let uploaded = row.uploaded.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(uploaded, row.year, partition_size);
            // restore this row back to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.group.as_str(),
                    row.year,
                    bucket,
                    uploaded,
                    row.id,
                    row.url.as_str(),
                    row.provider.as_str(),
                    row.user.as_str(),
                    row.name.as_str(),
                    row.creator.as_str(),
                    row.scheme.as_str(),
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

/// A single line of stage logs
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct RepoData {
    /// The url of the repo this data blob is for
    pub repo: String,
    /// The sha256 of this data blob
    pub hash: String,
}

impl Utils for RepoData {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "repo_data"
    }
}

#[async_trait::async_trait]
impl Backup for RepoData {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::RepoData
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
        // build repos get prepared statement
        scylla
            .prepare(format!(
                "SELECT repo, hash \
                FROM {}.{} \
                Where token(repo) >= ? AND token(repo) <= ?",
                ns,
                Self::name()
            ))
            .await
    }

    /// Hash this partitions info to see if we have changed partitions
    fn hash_partition(&self) -> u64 {
        // build a new hasher
        let mut hasher = AHasher::default();
        // ingest our partition key
        hasher.write(self.repo.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the repos table
#[async_trait::async_trait]
impl Scrub for RepoData {}

/// Implement restore support for the repos table
#[async_trait::async_trait]
impl Restore for RepoData {
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
                (repo, hash) \
                VALUES (?, ?)",
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
        let rows = rkyv::check_archived_root::<Vec<RepoData>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // restore this row back to scylla
            let query = scylla.execute_unpaged(prepared, (row.repo.as_str(), row.hash.as_str()));
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
