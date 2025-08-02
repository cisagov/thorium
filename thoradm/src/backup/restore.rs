//! The worker that handles restoring data to a Thorium cluster

use indicatif::ProgressBar;
use kanal::{AsyncReceiver, AsyncSender};
use rkyv::validation::validators::DefaultValidator;
use rkyv::Archive;
use scylla::client::session::Session;
use scylla::errors::{ExecutionError, PrepareError};
use scylla::statement::prepared::PreparedStatement;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use thorium::Conf;

use super::{ArchiveReader, MonitorUpdate, Utils};
use crate::Error;

/// The worker that handles restoring data to a Thorium cluster
pub struct RestoreWorker<R: Restore> {
    /// The scylla client to connect with
    scylla: Arc<Session>,
    /// The prepared statement to use
    prepared: PreparedStatement,
    /// The type we are restoring
    phantom: PhantomData<R>,
    /// The kanal channel workers should send restore updates over
    updates: AsyncSender<MonitorUpdate>,
    /// The progress bar to track progress with
    progress: ProgressBar,
    /// Track the total number of rows restored
    rows_restored: usize,
    /// The partition size to use for this table
    partition_size: u16,
}

impl<R: Restore> RestoreWorker<R> {
    /// Create a new restore worker
    ///
    /// # Arguments
    ///
    /// * `scylla` - The scylla client to use when restoring data
    /// * `conf` - A Thorium config
    /// * `updates` - The kanal channel to send restore updates over
    /// * `progress` - The progress bar to track progress with
    pub async fn new(
        scylla: &Arc<Session>,
        conf: &Conf,
        updates: AsyncSender<MonitorUpdate>,
        progress: ProgressBar,
    ) -> Result<Self, Error> {
        // get our namespace
        let namespace = &conf.thorium.namespace;
        // get our prepared statement
        let prepared = R::prepared_statement(&scylla, namespace).await?;
        // get our partition size
        let partition_size = R::partition_size(conf);
        // build our restore worker
        let worker = RestoreWorker {
            scylla: scylla.clone(),
            prepared,
            phantom: PhantomData::default(),
            updates,
            progress,
            rows_restored: 0,
            partition_size,
        };
        Ok(worker)
    }

    /// Restore all data in an archive
    ///
    /// # Arguments
    ///
    /// * `orders` - The channel to receive map paths on
    pub async fn restore(mut self, orders: AsyncReceiver<PathBuf>) -> Result<Self, Error>
    where
        <R as Archive>::Archived:
            for<'a> bytecheck::CheckBytes<DefaultValidator<'a>> + std::fmt::Debug,
    {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            let map_path = match orders.recv().await {
                Ok(path) => path,
                Err(kanal::ReceiveError::Closed) => break,
                Err(kanal::ReceiveError::SendClosed) => break,
            };
            // build our reader for this archive
            let mut reader = ArchiveReader::new(map_path).await?;
            // we split the read and the deserialization step into two functions
            // work around lifetime and mutability issues.
            // crawl over the partitions in this archive and restore them
            while let Some(restore_slice) = reader.next().await? {
                // get the number of rows we have restored so far
                let restored = self.rows_restored;
                // restore this slices data to scylla
                R::restore(
                    restore_slice,
                    &self.scylla,
                    self.partition_size,
                    &mut self.rows_restored,
                    &mut self.progress,
                    &self.prepared,
                )
                .await?;
                // set our new writer position
                self.progress.inc(restore_slice.len() as u64);
                // build the progress update for this partition
                let update = MonitorUpdate::Update {
                    items: self.rows_restored - restored,
                    bytes: restore_slice.len() as u64,
                };
                // send our progress update to the global tracker
                self.updates.send(update).await?;
            }
        }
        Ok(self)
    }

    /// Shutdown this worker
    pub fn shutdown(self) {
        // shutdown our progress bar
        self.progress.finish();
    }
}

#[async_trait::async_trait]
pub trait Restore: Utils + std::fmt::Debug + 'static + Send + Archive {
    /// The steps to once run before restoring data
    async fn prep(scylla: &Session, ns: &str) -> Result<(), ExecutionError>;

    /// The prepared statement to use when restoring data to scylla
    async fn prepared_statement(
        scylla: &Session,
        ns: &str,
    ) -> Result<PreparedStatement, PrepareError>;

    /// Get the partition size for this data type
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    fn partition_size(config: &Conf) -> u16;

    /// Restore a single partition
    async fn restore<'a>(
        buffer: &'a [u8],
        scylla: &Arc<Session>,
        partition_size: u16,
        rows_restored: &mut usize,
        progress: &mut ProgressBar,
        prepared: &PreparedStatement,
    ) -> Result<(), Error>;
}
