//! The worker that handles backing up Thorium data

use data_encoding::HEXLOWER;
use futures::stream::StreamExt;
use indicatif::ProgressBar;
use kanal::{AsyncReceiver, AsyncSender};
use rkyv::ser::serializers::AllocSerializer;
use rkyv::{AlignedVec, Archive, Serialize};
use scylla::client::session::Session;
use scylla::deserialize::row::DeserializeRow;
use scylla::errors::PrepareError;
use scylla::statement::prepared::PreparedStatement;
use sha2::{Digest, Sha256};
use std::io::IoSlice;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use super::{MonitorUpdate, PartitionArchive, Utils};
use crate::args::BackupComponents;
use crate::Error;

/// An archived partition that is ready to be written to disk
#[derive(Debug)]
pub struct PendingArchive {
    /// The number of rows in this partition
    rows: usize,
    /// The bytes for this archive
    bytes: AlignedVec,
}

/// Ensure that all bytes in a vectored write are written
async fn write_all_vectored<'a>(
    file: &mut File,
    mut segments: &mut [IoSlice<'a>],
    mut total: usize,
) -> Result<(), Error> {
    while total > 0 {
        // write all of our archives in one go
        let data_wrote = file.write_vectored(segments).await?;
        // check if we wrote all of our requested bytes
        if data_wrote != total {
            // advance our segment slices
            IoSlice::advance_slices(&mut segments, data_wrote);
        }
        // subtract any already written bytes
        total -= data_wrote;
    }
    Ok(())
}

/// The current archive we are writing too
#[derive(Debug)]
pub struct ArchiveWriter {
    /// The name of this archive
    pub name: Uuid,
    /// The path to write data too
    data_path: PathBuf,
    /// The path to write map info too
    map_path: PathBuf,
    /// The progress bar to update
    pub progress: ProgressBar,
    /// The buffers we have to write to disk still
    pending: Vec<PendingArchive>,
    /// The buffers for our map entries we have to write to disk still
    pending_maps: Vec<AlignedVec>,
    /// The current size of our pending buffers
    pending_bytes: usize,
    /// The number of bytes we have/will write to our current archive file
    written: usize,
    /// The file to write our archived data too
    data_file: Option<File>,
    /// The file to write our map data too
    map_file: Option<File>,
}

impl ArchiveWriter {
    /// Create a new archive writer
    ///
    /// # Arguments
    ///
    /// * `data_path` - The folder to write archive data too
    /// * `map_path` - The folder to write archive map info too
    /// * `progress` - The progress bar to update
    pub async fn new(
        data_path: &PathBuf,
        map_path: &PathBuf,
        progress: ProgressBar,
    ) -> Result<Self, Error> {
        // build our archive writer
        let writer = ArchiveWriter {
            name: Uuid::new_v4(),
            data_path: data_path.clone(),
            map_path: map_path.clone(),
            progress,
            pending: Vec::with_capacity(100),
            pending_maps: Vec::with_capacity(100),
            pending_bytes: 0,
            written: 0,
            data_file: None,
            map_file: None,
        };
        Ok(writer)
    }

    /// Add a partitions archived buffer to our pending buffer vec
    ///
    /// # Arguments
    ///
    /// * `partition` - The hash of the partition we are adding to this archive
    /// * `rows` - The number of rows we archived in this partition
    /// * `sha256` - The sha256 for the archive we archived
    /// * `bytes` - The bytes we are archiving
    pub fn add(&mut self, partition: u64, rows: usize, sha256: String, bytes: AlignedVec) -> bool {
        // calculate where this partitions archived data ends
        let end = self.written + bytes.len();
        // build the map for this partition
        let map = PartitionArchive::new(self.written, end, partition, sha256);
        // archive this partition and log its size
        let archived_map = rkyv::to_bytes::<_, 1024>(&map).unwrap();
        // panic if our len is greater than 96 bytes
        if archived_map.len() != 96 {
            panic!("entry len is {}", archived_map.len());
        }
        // build the pending archive object
        let archive = PendingArchive {
            rows,
            //sha256,
            bytes,
        };
        // add this buffers size to our pending size
        self.pending_bytes += archive.bytes.len();
        // set our new progress bar position
        self.progress.inc(archive.bytes.len() as u64);
        // add this archive to our pending bytes
        self.pending.push(archive);
        // add this map to our pending maps
        self.pending_maps.push(archived_map);
        // update the number of bytes we have written
        self.written = end;
        // determine if we have enough bytes to write this archive to disk
        self.pending_bytes > 1_048_576
    }

    /// Open a file handle to our data file
    async fn open_handles(&mut self) -> Result<(), Error> {
        // loop until we have found an unused id
        loop {
            // cast our archive id to a string
            let id = self.name.to_string();
            // build the new data path
            self.data_path.push(&id);
            // make sure this data file doesn't already exist
            if tokio::fs::try_exists(&self.data_path).await? {
                // generate a random uuid for this archive
                self.name = Uuid::new_v4();
                // pop this used id from our data path
                self.data_path.pop();
                // this data file already exists so try a new random id
                continue;
            }
            // open a file handle to the data file
            self.data_file = Some(File::create(&self.data_path).await?);
            // build the new map path
            self.map_path.push(&id);
            self.map_path.set_extension("thoriummap");
            // open a file handle to the data file
            self.map_file = Some(File::create(&self.map_path).await?);
            // reset our file paths
            self.data_path.pop();
            self.map_path.pop();
            // we have opened our file handles
            break;
        }
        Ok(())
    }

    /// Write any pending archived data to disk
    ///
    /// # Arguments
    ///
    /// * `update_tx` - The channel to send updates on
    pub async fn archive(
        &mut self,
        update_tx: &mut AsyncSender<MonitorUpdate>,
    ) -> Result<(), Error> {
        // if we have no pending bytes then just return
        if self.pending_bytes == 0 {
            return Ok(());
        }
        // get our open data file handle
        let (mut data_file, mut map_file) = match (self.data_file.as_mut(), self.map_file.as_mut())
        {
            (Some(data_file), Some(map_file)) => (data_file, map_file),
            _ => {
                // open our new file handles
                self.open_handles().await?;
                // get our opened file handles
                (
                    self.data_file.as_mut().unwrap(),
                    self.map_file.as_mut().unwrap(),
                )
            }
        };
        // build a list of IO slices to write our pending data
        let mut archive_slices = self
            .pending
            .iter()
            .map(|archive| IoSlice::new(&archive.bytes))
            .collect::<Vec<IoSlice>>();
        // write all of our data
        write_all_vectored(&mut data_file, &mut archive_slices[..], self.pending_bytes).await?;
        // build a list of IO slices to write our pending maps
        let mut map_slices = self
            .pending_maps
            .iter()
            .map(|map| IoSlice::new(&map))
            .collect::<Vec<IoSlice>>();
        // get the length of our map slices
        let map_len = map_slices.iter().fold(0, |acc, map| acc + map.len());
        // write all of our map info
        write_all_vectored(&mut map_file, &mut map_slices[..], map_len).await?;
        // build and send the updates for all of our written archives
        for archive in self.pending.drain(..) {
            // build the update for our monitor
            let update = MonitorUpdate::Update {
                items: archive.rows,
                bytes: archive.bytes.len() as u64,
            };
            // send our update to our controller
            update_tx.send(update).await?;
        }
        // clear our map updates
        self.pending_maps.clear();
        // flush our map and data files
        data_file.flush().await?;
        map_file.flush().await?;
        // clear our pending bytes
        self.pending_bytes = 0;
        // if we have written 10GiB worth of data then split this archive off into a new file next time
        if self.written >= 10_737_418_240 {
            // set our file to None so we make a new file on the next write
            self.data_file = None;
            // set our file to None so we make a new file on the next write
            self.map_file = None;
            // reset our written bytes
            self.written = 0;
        }
        Ok(())
    }
}

/// A single backup worker for Thorium
pub struct BackupWorker<T: Backup> {
    /// The scylla client to connect with
    scylla: Arc<Session>,
    /// The prepared statement to use
    prepared: PreparedStatement,
    /// The kanal channel workers should send backup updates over
    updates: AsyncSender<MonitorUpdate>,
    /// Our current partitions key
    partition: Option<u64>,
    /// The rows for our current partition
    rows: Vec<T>,
    /// The sha256 hasher we are using for partition hashes
    hasher: Sha256,
    /// The archive we are writing too
    writer: ArchiveWriter,
    /// The current number of rows this worker has backed up
    rows_backed_up: u64,
    /// The progress bar to write error messages with
    progress: ProgressBar,
}

impl<T: Backup> BackupWorker<T> {
    /// Create a new backup worker
    ///
    /// # Arguments
    ///
    /// * `scylla` - The scylla client to use when backing up data
    /// * `namespace` - The namespace for this backup
    /// * `updates` - The channel to send partition archive updates on
    /// * `data_path` - The path to write archive data too
    /// * `map_path` - The path to write map data too
    pub async fn new(
        scylla: &Arc<Session>,
        namespace: &str,
        updates: AsyncSender<MonitorUpdate>,
        data_path: &PathBuf,
        map_path: &PathBuf,
        progress: ProgressBar,
    ) -> Result<Self, Error> {
        // get our prepared statement
        let prepared = T::prepared_statement(&scylla, namespace).await?;
        // build a new archive writer
        let writer = ArchiveWriter::new(data_path, map_path, progress.clone()).await?;
        // build our backup worker
        let worker = BackupWorker {
            scylla: scylla.clone(),
            prepared,
            updates,
            partition: None,
            rows: Vec::with_capacity(1000),
            writer,
            rows_backed_up: 0,
            hasher: Sha256::new(),
            progress,
        };
        Ok(worker)
    }

    /// Archive our current rows
    ///
    /// # Arguments
    ///
    /// * `flush` - Whether we should flush map info to disk regardless
    async fn archive(&mut self, flush: bool) -> Result<(), Error> {
        // only archive data if we have some
        if !self.rows.is_empty() {
            // get our current partition
            if let Some(partition) = self.partition {
                // archive our current rows
                let archived_bytes = rkyv::to_bytes::<_, 1024>(&self.rows).unwrap();
                // hash our bytes
                self.hasher.update(&archived_bytes);
                // finalize our hash and cast it to a string
                let sha256 = HEXLOWER.encode(&self.hasher.finalize_reset());
                // get the number of rows we are archiving
                let row_count = self.rows.len();
                // add this archive and check if we have enough pending bytes to write them to disk
                if flush
                    || self
                        .writer
                        .add(partition, row_count, sha256, archived_bytes)
                {
                    // we have enough pending bytes so write our archived data to disk
                    self.writer.archive(&mut self.updates).await?;
                }
                // empty our current rows
                self.rows.clear();
            } else {
                return Err(Error::new("Tried to archive with no partition set"));
            }
        }
        Ok(())
    }

    /// Check if we started a new partition with this row
    ///
    /// This will flush the old partition to disk.
    ///
    /// # Arguments
    ///
    /// * `row` - The row to check
    async fn check_row(&mut self, row: T) -> Result<(), Error> {
        // Hash this rows partition key
        let hash = row.hash_partition();
        // check our hash to see if this partition changed
        match self.partition {
            // we don't yet have a partition hash so set it
            None => self.partition = Some(hash),
            // we are still in the same partition
            Some(current) if current == hash => (),
            // we have entered a new partition
            _ => {
                // archive our existing partition
                self.archive(false).await?;
                // update our partition hash
                self.partition = Some(hash)
            }
        }
        // add our new row
        self.rows.push(row);
        Ok(())
    }

    /// Start backing up data and streaming it to archives
    ///
    /// # Arguments
    ///
    /// * `orders` - The channel to receive map paths on
    pub async fn start(mut self, orders: AsyncReceiver<(i64, i64)>) -> Result<Self, Error> {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            let (start, end) = match orders.recv().await {
                Ok(path) => path,
                Err(kanal::ReceiveError::Closed) => break,
                Err(kanal::ReceiveError::SendClosed) => break,
            };
            // build and execute our paged query
            let rows_stream = self
                .scylla
                .execute_iter(self.prepared.clone(), &(start, end))
                .await?;
            // build a typed iter for these rows
            let mut typed_stream = match rows_stream.rows_stream::<T>() {
                Ok(typed_stream) => typed_stream,
                Err(error) => {
                    // build our error message
                    let msg = format!("Failed to set type for row stream: with {:#?}", error);
                    // log that we failed to cast this row
                    self.progress.println(msg.clone());
                    // continue to the next row
                    continue;
                }
            };
            // crawl over our typed stream
            while let Some(typed_row) = typed_stream.next().await {
                // error out if any of our typed rows fail
                let typed_row = match typed_row {
                    Ok(typed_row) => typed_row,
                    Err(error) => {
                        // build our error message
                        let msg = format!("Failed to backup row: with {:#?}", error);
                        // log that we failed to cast this row
                        self.progress.println(msg.clone());
                        // continue to the next row
                        continue;
                    }
                };
                // increment our row count
                self.rows_backed_up += 1;
                // set our current row count progress message
                self.writer
                    .progress
                    .set_message(self.rows_backed_up.to_string());
                // flush completed partitions to disk if necessary
                self.check_row(typed_row).await?;
            }
        }
        // archive any remaining data
        self.archive(true).await?;
        Ok(self)
    }

    /// Shutdown this worker
    pub fn shutdown(self) {
        // shutdown our progress bar
        self.writer.progress.finish();
    }
}

#[async_trait::async_trait]
pub trait Backup:
    Utils
    + std::fmt::Debug
    + 'static
    + Send
    + Archive
    + Serialize<AllocSerializer<1024>>
    + for<'frame, 'metadata> DeserializeRow<'frame, 'metadata>
{
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents;

    /// The prepared statement to use when retrieving data from Scylla
    async fn prepared_statement(
        scylla: &Session,
        ns: &str,
    ) -> Result<PreparedStatement, PrepareError>;

    /// Hash this rows partitions key to see if we have changed partitions
    fn hash_partition(&self) -> u64;
}
