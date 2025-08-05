//! Backup and restore support for nodes

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
use thorium::models::{NodeHealth, Resources};
use thorium::Conf;

use crate::args::BackupComponents;
use crate::backup::{Backup, Restore, Scrub, Utils};
use crate::Error;

/// A single node in a Thorium cluster
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Node {
    /// The cluster this node is in
    pub cluster: String,
    /// The name of this node
    pub node: String,
    /// The current health of this node
    pub health: NodeHealth,
    /// When this node last checked in
    pub heart_beat: Option<DateTime<Utc>>,
    /// The resources this node has
    pub resources: Resources,
}

impl Utils for Node {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "nodes"
    }
}

#[async_trait::async_trait]
impl Backup for Node {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::Nodes
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
                "SELECT cluster, node, health, heart_beat, resources \
                FROM {}.{} \
                Where token(cluster) >= ? AND token(cluster) <= ?",
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
        hasher.write(self.cluster.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for Node {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for Node {
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
                (cluster, node, health, heart_beat, resources) \
                VALUES (?, ?, ?, ?, ?)",
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
        let rows = rkyv::check_archived_root::<Vec<Node>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize our node health
            let health: NodeHealth = row.health.deserialize(&mut rkyv::Infallible)?;
            // deserialize our node health
            let heart_beat: Option<DateTime<Utc>> =
                row.heart_beat.deserialize(&mut rkyv::Infallible)?;
            // deserialize our resources
            let resources: Resources = row.resources.deserialize(&mut rkyv::Infallible)?;
            // insert this row
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.cluster.as_str(),
                    row.node.as_str(),
                    health,
                    heart_beat,
                    resources,
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
