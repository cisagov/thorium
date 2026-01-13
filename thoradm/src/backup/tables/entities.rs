//! Backup and restore support for entities

use ahash::AHasher;
use chrono::{DateTime, Utc};
use futures::{StreamExt, stream::FuturesUnordered};
use indicatif::ProgressBar;
use rkyv::{Archive, CheckBytes, Deserialize, Serialize, string::ArchivedString};
use scylla::{
    DeserializeRow,
    client::session::Session,
    errors::{ExecutionError, PrepareError},
    statement::prepared::PreparedStatement,
};
use std::{hash::Hasher, sync::Arc};
use thorium::{Conf, models::EntityKinds};
use uuid::Uuid;

use crate::Error;
use crate::backup::{Backup, Restore, Scrub, Utils, utils};

/// A single row from the entities table
#[derive(Debug, Archive, Serialize, Deserialize, DeserializeRow)]
#[archive_attr(derive(Debug, CheckBytes))]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct Entity {
    /// The entity's kind as a raw string
    pub kind: EntityKinds,
    /// The group the entity is available to
    pub group: String,
    /// The year this entity is from
    pub year: i32,
    /// The bucket this entity is in
    pub bucket: i32,
    /// The time this policy was created
    pub created: DateTime<Utc>,
    /// The entity's unique ID
    pub id: Uuid,
    /// The name of the entity
    pub name: String,
    /// The user who originally submitted the entity
    pub submitter: String,
    /// The data specific to the entity's kind
    pub kind_data: String,
    /// The entity's description
    pub description: Option<String>,
    /// The path to this entities image if it has one
    pub image: Option<String>,
}

impl Utils for Entity {
    /// The name of the table we are backing up
    fn name() -> &'static str {
        "entities"
    }
}

#[async_trait::async_trait]
impl Backup for Entity {
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
                "SELECT kind, group, year, bucket, created, id, name, submitter, kind_data, description, image \
                FROM {}.{} \
                WHERE token(kind, group, year, bucket) >= ? AND token(kind, group, year, bucket) <= ?",
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
        hasher.write(self.group.as_bytes());
        hasher.write_i32(self.year);
        hasher.write_i32(self.bucket);
        // finish this hash and get its value
        hasher.finish()
    }
}

/// Implement scrub support for the samples list table
impl Scrub for Entity {}

/// Implement restore support for the samples list table
#[async_trait::async_trait]
impl Restore for Entity {
    // The partition size is constant, so the partition config is just
    // the size itself
    type PartitionConf = u16;

    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError> {
        // drop the materialized views for this table
        utils::drop_materialized_view(ns, "entities_by_name", scylla).await?;
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
                (kind, group, year, bucket, created, id, name, submitter, kind_data, description, image) \
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
    fn partition_conf(config: &Conf) -> u16 {
        config.thorium.entities.partition_size
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
        partition_size: &u16,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error> {
        // cast our buffer to its archived type
        let rows = rkyv::check_archived_root::<Vec<Entity>>(buffer)?;
        // build a set of futures
        let mut futures = FuturesUnordered::new();
        // build our queries to insert this partitions rows
        for row in rows.iter() {
            let kind: EntityKinds = row.kind.deserialize(&mut rkyv::Infallible)?;
            // deserialize this rows uploaded timestamp
            let created = row.created.deserialize(&mut rkyv::Infallible)?;
            // calculate the new bucket
            let bucket = thorium::utils::helpers::partition(created, row.year, *partition_size);
            let query = scylla.execute_unpaged(
                prepared,
                (
                    kind,
                    row.group.as_str(),
                    row.year,
                    bucket,
                    created,
                    row.id,
                    row.name.as_str(),
                    row.submitter.as_str(),
                    row.kind_data.as_str(),
                    row.description.as_ref().map(ArchivedString::as_str),
                    row.image.as_ref().map(ArchivedString::as_str),
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
