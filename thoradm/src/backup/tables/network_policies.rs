//! Backup and restore support for network policies

use ahash::AHasher;
use chrono::{DateTime, Utc};
use futures::{StreamExt, stream::FuturesUnordered};
use indicatif::ProgressBar;
use rkyv::{Archive, CheckBytes, Deserialize, Serialize};
use scylla::{
    DeserializeRow,
    client::session::Session,
    errors::{ExecutionError, PrepareError},
    statement::prepared::PreparedStatement,
};
use std::{hash::Hasher, sync::Arc};
use thorium::Conf;
use uuid::Uuid;

use crate::{
    Error,
    args::BackupComponents,
    backup::{Backup, Restore, Scrub, Utils, utils},
};

/// A single line from the network policy table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct NetworkPolicy {
    /// The group the network policy is in
    pub group: String,
    /// The network policy's name
    pub name: String,
    /// The network policy's ID
    pub id: Uuid,
    /// The network policy's unique name in K8's
    pub k8s_name: String,
    /// The date the network policy is created
    pub created: DateTime<Utc>,
    /// The ingress settings for this network policy
    pub ingress: String,
    /// The egress settings for this network policy
    pub egress: String,
    /// Whether this network policy is forced in its group
    pub forced_policy: bool,
    /// Whether this network policy is default in its group
    pub default_policy: bool,
}

impl Utils for NetworkPolicy {
    fn name() -> &'static str {
        "network_policies"
    }
}

#[async_trait::async_trait]
impl Backup for NetworkPolicy {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents {
        BackupComponents::NetworkPolicies
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
        // build network policies get prepared statement
        scylla
            .prepare(format!(
                "SELECT group, name, id, k8s_name, created, ingress, egress, forced_policy, default_policy \
                FROM {}.{} \
                WHERE token(group) >= ? AND token(group) <= ?",
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
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the network policies table
impl Scrub for NetworkPolicy {}

/// Implement restore support for the network policies table
#[async_trait::async_trait]
impl Restore for NetworkPolicy {
    // Network policies are not partitioned by time, so just return unit type
    type PartitionConf = ();

    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "network_policies_by_name", scylla).await?;
        utils::drop_materialized_view(ns, "network_policies_default", scylla).await?;
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
                (group, name, id, k8s_name, created, ingress, egress, forced_policy, default_policy) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ns,
                Self::name(),
            ))
            .await
    }

    /// Get the partition size for this data type
    ///
    /// Network policies are not partitioned by time, so we just return
    /// unit type here
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    fn partition_conf(_config: &Conf) -> Self::PartitionConf {}

    /// Restore a single partition
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer we should restore data from
    /// * `scylla` - The client to use when talking to scylla
    /// * `rows_restored` - The number of rows that have been restored
    /// * `partition_conf` - The config defining the partition size to use when restoring data
    /// * `progress` - The bar to report progress with
    /// * `prepared` - The prepared statement to inject data with
    async fn restore<'a>(
        buffer: &'a [u8],
        scylla: &Arc<Session>,
        _partition_conf: &Self::PartitionConf,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<NetworkPolicy>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            // deserialize this rows created timestamp
            let created: DateTime<Utc> = row.created.deserialize(&mut rkyv::Infallible)?;
            // restore this row back to scylla
            let query = scylla.execute_unpaged(
                prepared,
                (
                    row.group.as_str(),
                    row.name.as_str(),
                    row.id,
                    row.k8s_name.as_str(),
                    created,
                    row.ingress.as_str(),
                    row.egress.as_str(),
                    row.forced_policy,
                    row.default_policy,
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
