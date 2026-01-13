//! Backup and restore support for notifications

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
use thorium::{
    Conf,
    models::{NotificationLevel, NotificationType},
};
use uuid::Uuid;

use crate::Error;
use crate::backup::{Backup, Restore, Scrub, Utils};

/// A single row from the notifications table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Notification {
    /// The kind of notification this is
    pub kind: NotificationType,
    /// The key to the item this notification relates to
    pub key: String,
    /// The time this notification was created
    pub created: DateTime<Utc>,
    /// The notification's unique ID
    pub id: Uuid,
    /// The notification's message
    pub msg: String,
    /// The notification's level
    pub level: NotificationLevel,
    /// The id of a ban this notification is referencing if there is one
    pub ban_id: Option<Uuid>,
}

impl Utils for Notification {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "notifications"
    }
}

#[async_trait::async_trait]
impl Backup for Notification {
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
                "SELECT kind, key, created, id, msg, level, ban_id \
                FROM {}.{} \
                WHERE token(kind, key) >= ? AND token(kind, key) <= ?",
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
        hasher.write(self.kind.as_str().as_bytes());
        hasher.write(self.key.as_bytes());
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for Notification {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for Notification {
    // notifications are not partitioned
    type PartitionConf = ();

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
                (kind, key, created, id, msg, level, ban_id) \
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
    fn partition_conf(_config: &Conf) {
        // notifications are not partitioned
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
        _partition_size: &(),
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<Notification>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            let kind: NotificationType = row.kind.deserialize(&mut rkyv::Infallible)?;
            let level: NotificationLevel = row.level.deserialize(&mut rkyv::Infallible)?;
            // deserialize this rows uploaded timestamp
            let created: DateTime<Utc> = row.created.deserialize(&mut rkyv::Infallible)?;
            let query = scylla.execute_unpaged(
                prepared,
                (
                    kind,
                    row.key.as_str(),
                    created,
                    row.id,
                    row.msg.as_str(),
                    level,
                    row.ban_id.as_ref(),
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
