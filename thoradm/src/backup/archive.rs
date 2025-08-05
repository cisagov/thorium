//! The archives of backed up data

use bytecheck::CheckBytes;
use bytes::BytesMut;
use rkyv::{Archive, Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};

use crate::Error;

/// A single archive of a partition
#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive_attr(derive(Debug, CheckBytes))]
pub struct PartitionArchive {
    /// What byte this partitions archive starts at
    pub start: u64,
    /// What byte this partitions archive stops at
    pub end: u64,
    /// The hash for this partition
    pub partition: u64,
    /// The sha256 for this partition
    pub sha256: String,
}

impl PartitionArchive {
    /// Create a new partition archive
    ///
    /// # Arguments
    ///
    /// * `start` - What byte this partition archive starts at
    /// * `end` - What byte this partitions archive stops as
    /// * `partition` - The hash for this partition
    /// * `sha256` - The sha256 for this partition
    pub fn new<T: Into<String>>(start: usize, end: usize, partition: u64, sha256: T) -> Self {
        PartitionArchive {
            start: start as u64,
            end: end as u64,
            partition,
            sha256: sha256.into(),
        }
    }
}

/// The archive to read data from
pub struct ArchiveReader {
    /// The map to read our archive map from
    map_reader: BufReader<File>,
    /// The file to read data from
    data_reader: BufReader<File>,
    /// The current buffer to map info into
    pub map_buffer: BytesMut,
    /// The current buffer to read data into
    pub data_buffer: BytesMut,
}

impl ArchiveReader {
    /// Create a new archive reader
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the map file to read in
    pub async fn new(mut path: PathBuf) -> Result<Self, Error> {
        // get the id for our archive
        let id = match path.file_stem() {
            Some(stem) => stem.to_owned(),
            None => {
                return Err(Error::new(format!(
                    "Failed to get archive id from {:?}",
                    path
                )))
            }
        };
        // open our map file
        let map_reader = BufReader::new(File::open(&path).await?);
        // pop our map file and map dir
        path.pop();
        path.pop();
        // add our data path and id
        path.push("data");
        path.push(id);
        // open our archive file
        let data_reader = BufReader::new(File::open(&path).await?);
        // create a bytesmut object for our map info
        let map_buffer = BytesMut::zeroed(96);
        // create a bytesmut object at least 1MiB big
        let data_buffer = BytesMut::zeroed(1_048_576);
        // create our archive reader
        let reader = ArchiveReader {
            //map,
            map_reader,
            data_reader,
            map_buffer,
            data_buffer,
        };
        Ok(reader)
    }

    /// Read in the bytes for the next partition
    pub async fn next(&mut self) -> Result<Option<&[u8]>, Error> {
        // try to read the next partitions info from our map
        // read the next row to add to our archive map
        if let Err(error) = self.map_reader.read_exact(&mut self.map_buffer).await {
            // if this is the end of the file then we reached the end of the map
            match error.kind() {
                std::io::ErrorKind::UnexpectedEof => return Ok(None),
                // some other error occurred
                _ => return Err(Error::from(error)),
            }
        }
        // deserialize this archive map entry
        let archived_entry = rkyv::check_archived_root::<PartitionArchive>(&self.map_buffer)?;
        // calculate the size of this partitions data
        let partition_len = (archived_entry.end - archived_entry.start) as usize;
        // extend our buffer to fit this partitions data if required
        // this currently never frees space but it likely should
        if self.data_buffer.len() < partition_len {
            // resize our buffer to fit
            self.data_buffer.resize(partition_len, 0x0);
        }
        // read this partitions data in
        self.data_reader
            .read_exact(&mut self.data_buffer[..partition_len])
            .await?;
        //Ok(Some(partition_len))
        Ok(Some(&self.data_buffer[..partition_len]))
    }

    /// Read in the bytes for the next partition and return its hash
    pub async fn next_partition(&mut self) -> Result<Option<(&[u8], PartitionArchive)>, Error> {
        // try to read the next partitions info from our map
        // read the next row to add to our archive map
        if let Err(error) = self.map_reader.read_exact(&mut self.map_buffer[..]).await {
            // if this is the end of the file then we reached the end of the map
            match error.kind() {
                std::io::ErrorKind::UnexpectedEof => return Ok(None),
                // some other error occurred
                _ => return Err(Error::from(error)),
            }
        }
        // deserialize this archived map entry
        let archived_entry = rkyv::check_archived_root::<PartitionArchive>(&self.map_buffer)?;
        // get the original type
        let entry: PartitionArchive = archived_entry.deserialize(&mut rkyv::Infallible)?;
        // calculate the size of this partitions data
        let partition_len = entry.end - entry.start;
        // extend our buffer to fit this partitions data if required
        // this currently never frees space but it likely should
        if self.data_buffer.len() < partition_len as usize {
            // resize our buffer to fit
            self.data_buffer.resize(partition_len as usize, 0x0);
        }
        // read this partitions data in
        self.data_reader
            .read_exact(&mut self.data_buffer[..partition_len as usize])
            .await?;
        Ok(Some((&self.data_buffer[..partition_len as usize], entry)))
    }
}
