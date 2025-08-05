//! Backup/Restore data in S3

use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_s3::config::Credentials;
use aws_sdk_s3::primitives::{ByteStream, SdkBody};
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::Client;
use bytes::{Buf, BytesMut};
use futures::stream::FuturesOrdered;
use futures::StreamExt;
use indicatif::ProgressBar;
use kanal::{AsyncReceiver, AsyncSender};
use num_format::{Locale, ToFormattedString};
use rkyv::validation::validators::DefaultValidator;
use rkyv::Archive;
use std::marker::PhantomData;
use std::path::PathBuf;
use thorium::Conf;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;
use tokio_util::io::ReaderStream;

use crate::args::BackupComponents;
use crate::Error;

use super::{ArchiveReader, MonitorUpdate, Utils};

/// The different s3 monitor updates
pub enum S3MonitorUpdate {
    /// A single object was downloaded
    Update(usize),
    /// All operations have completed
    Finished,
}

/// The s3 backup monitor
pub struct S3Monitor {
    /// The total number of objects we have backed up
    total_objects: u64,
    /// The receiver to pull updates from
    receiver: AsyncReceiver<S3MonitorUpdate>,
    /// The progress bar to report updates on
    progress: ProgressBar,
}

impl S3Monitor {
    /// Create a new s3 monitor
    ///
    /// # Arguments
    ///
    /// * `receiver` - The channel to receive updates on
    /// * `progress` - The bar to report progress on
    pub fn new(receiver: AsyncReceiver<S3MonitorUpdate>, progress: ProgressBar) -> Self {
        S3Monitor {
            total_objects: 0,
            receiver,
            progress,
        }
    }

    /// Start monitoring and reporting updates
    pub async fn start(mut self) -> Result<(), Error> {
        // track the number of updates since our last message update
        let mut since_msg_update = 0;
        // loop forever until the finished update is received
        loop {
            // handle this update
            match self.receiver.recv().await? {
                // update our progress bar with this new info
                S3MonitorUpdate::Update(amt) => {
                    // update our progress bar
                    self.progress.inc(amt as u64);
                    // update our total objects count
                    self.total_objects += 1;
                }
                S3MonitorUpdate::Finished => {
                    // set our message
                    self.progress.set_message(self.total_objects.to_string());
                    // stop monitoring for updates
                    break;
                }
            }
            // update our message if its been a while since we did or we have more no more updates
            if since_msg_update >= 100 || self.receiver.is_empty() {
                // set our new message
                self.progress
                    .set_message(self.total_objects.to_formatted_string(&Locale::en));
                // reset our messages since message update counter
                since_msg_update = 0;
            } else {
                // increment our messages since update counter
                since_msg_update += 1;
            }
        }
        // shutdown our progress bar
        self.progress.finish();
        Ok(())
    }
}

/// Helps download a single file and write it to disk
async fn download_file_helper(
    s3: Client,
    progress: &ProgressBar,
    bucket: &String,
    key: &String,
    path: &PathBuf,
) -> Result<Option<usize>, Error> {
    // create our sub dirs if we have any
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    // track the total bytes we have written
    let mut written = 0;
    // check if this file already exists
    if !tokio::fs::try_exists(path).await? {
        // start downloading this file from s3
        let mut object = s3.get_object().bucket(bucket).key(key).send().await?;
        // this file does not already exist so create it
        let mut file = File::create(path).await?;
        // stream this object to disk
        while let Some(buff) = object.body.try_next().await? {
            file.write_all(&buff).await?;
            // increment our total bytes written
            written += buff.len();
            // update our progress bar
            progress.inc(buff.len() as u64);
        }
    }
    Ok(Some(written))
}

/// Download a single file and write it to disk
async fn download_file(
    s3: Client,
    progress: ProgressBar,
    bucket: String,
    key: String,
    path: PathBuf,
) -> Option<usize> {
    // download this file and log any errors
    match download_file_helper(s3, &progress, &bucket, &key, &path).await {
        Ok(written) => written,
        Err(error) => {
            // log this error
            progress.println(format!("{bucket}/{key} -> {error:#?}"));
            // delete the file we ran into an error on if it exists
            match tokio::fs::try_exists(&path).await {
                Ok(exists) => {
                    // if this file already exists then delete it
                    if exists {
                        if let Err(error) = tokio::fs::remove_file(&path).await {
                            progress.println(format!("Failed to delete {path:#?} with {error}"));
                        }
                    }
                }
                // we failed to check if this file exists
                Err(error) => progress.println(format!(
                    "Failed to check if path {path:#?} exists with {error}"
                )),
            }
            // no bytes were written
            None
        }
    }
}

pub struct S3BackupWorker<S: S3Backup> {
    /// The type we are backing up
    phantom: PhantomData<S>,
    /// The config for this Thorium cluster
    conf: Conf,
    /// The kanal channel workers should send backup updates over
    updates: AsyncSender<MonitorUpdate>,
    /// The progress bar to track progress with
    progress: ProgressBar,
    /// the path to our object directory
    object_path: PathBuf,
    /// The s3 client to download files with
    pub s3: Client,
    /// The currently active downloads to monitor
    pub active: FuturesOrdered<JoinHandle<Option<usize>>>,
    /// Track the number of objects we have backed up
    pub backed_up: u64,
}

impl<S: S3Backup> S3BackupWorker<S> {
    /// Create a new backup worker
    ///
    /// # Arguments
    ///
    /// * `object_path` - The path to our object directory
    /// * `config` - The config for this Thorium cluster
    /// * `updates` - The channel to send partition archive updates on
    /// * `progress` - The progress bar for this worker
    pub fn new(
        conf: &Conf,
        updates: &AsyncSender<MonitorUpdate>,
        progress: ProgressBar,
        object_path: &PathBuf,
    ) -> Self {
        // get our s3 conf
        let s3_conf = &conf.thorium.s3;
        // get our s3 credentials
        let creds = Credentials::new(
            &s3_conf.access_key,
            &s3_conf.secret_token,
            None,
            None,
            "Thorium",
        );
        // build our s3 s3_config
        let s3_config = aws_sdk_s3::config::Builder::new()
            .endpoint_url(&s3_conf.endpoint)
            .region(aws_types::region::Region::new(s3_conf.region.clone()))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            .force_path_style(true)
            .build();
        // build our s3 client
        let s3 = Client::from_conf(s3_config);
        // build our worker
        S3BackupWorker {
            phantom: PhantomData,
            conf: conf.clone(),
            updates: updates.clone(),
            progress,
            object_path: object_path.clone(),
            s3,
            active: FuturesOrdered::new(),
            backed_up: 0,
        }
    }

    /// Check on our pending futures and wait if we have enough in flight
    pub async fn check_active(&mut self) {
        // check if we need to spawn any more workers
        if self.active.len() > 10 {
            // we have 10 active workers already so wait for a worker to finish
            if let Some(join) = self.active.next().await {
                // check if we had a join error
                match join {
                    Ok(written) => {
                        if let Some(written) = written {
                            // update our backed up count
                            self.backed_up += 1;
                            // set our new message
                            self.progress.set_message(self.backed_up.to_string());
                            // build our monitor update
                            let update = MonitorUpdate::Update {
                                items: 1,
                                bytes: written as u64,
                            };
                            // send an update to our monitor
                            if let Err(error) = self.updates.send(update).await {
                                self.progress.println(format!(
                                    "Failed to send monitor update: wrote {written} bytes with: {error:#?}"
                                ));
                            }
                        }
                    }
                    // log this join error
                    Err(error) => self.progress.println(format!("JoinError: {error:#?}")),
                }
            }
        }
    }

    /// Wait for all active tasks to complete
    pub async fn flush_active(&mut self) {
        // wait for all active tasks to complete
        while let Some(join) = self.active.next().await {
            // check if we had a join error
            match join {
                Ok(written) => {
                    if let Some(written) = written {
                        // update our backed up count
                        self.backed_up += 1;
                        // set our new message
                        self.progress.set_message(self.backed_up.to_string());
                        // build our monitor update
                        let update = MonitorUpdate::Update {
                            items: 1,
                            bytes: written as u64,
                        };
                        // send an update to our monitor
                        if let Err(error) = self.updates.send(update).await {
                            self.progress.println(format!(
                                "Failed to send monitor update: wrote {written} bytes with: {error:#?}"
                            ));
                        }
                    }
                }
                // log this join error
                Err(error) => self.progress.println(format!("JoinError: {error:#?}")),
            }
        }
    }

    // Crawl the data in this archive and backup any relevant s3 data to disk
    pub async fn backup(mut self, orders: AsyncReceiver<PathBuf>) -> Result<Self, Error>
    where
        <S as Archive>::Archived:
            for<'a> bytecheck::CheckBytes<DefaultValidator<'a>> + std::fmt::Debug,
    {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            let map_path = match orders.recv().await {
                Ok(path) => path,
                Err(kanal::ReceiveError::Closed | kanal::ReceiveError::SendClosed) => break,
            };
            // build our reader for this archive
            let mut reader = ArchiveReader::new(map_path).await?;
            // we split the read and the deserialization step into two functions
            // work around lifetime and mutability issues.
            // crawl over the partitions in this archive and back up its s3 data
            while let Some(backup_slice) = reader.next().await? {
                // check if we already have enough active workers
                self.check_active().await;
                // backup this partitions s3 data to disk
                for (bucket, url, path) in
                    S::paths(&self.conf, &self.object_path, 6, backup_slice).await?
                {
                    // clone any info from this worker
                    let s3 = self.s3.clone();
                    let progress = self.progress.clone();
                    // start downloading this file
                    let handle = tokio::spawn(async move {
                        download_file(s3, progress, bucket.clone(), url, path).await
                    });
                    // add this to our futures
                    self.active.push_back(handle);
                }
            }
        }
        // wait for all active downloads to finish
        self.flush_active().await;
        Ok(self)
    }

    /// Shutdown this worker
    pub fn shutdown(self) {
        // shutdown our progress bar
        self.progress.finish();
    }
}

#[async_trait::async_trait]
pub trait S3Backup: Utils + std::fmt::Debug + 'static + Send + Archive {
    /// Return the corresponding backup component for the implementor
    fn backup_component() -> BackupComponents;

    /// Get the s3 urls and where to write them off to disk at
    ///
    /// # Arguments
    ///
    /// * `conf` - The config for this Thorium cluster
    /// * `root` - The root path to write objects too
    /// * `chars` - The number of characters to use in order to partition this data on disk
    /// * `buffer` - The slice of data to deserialize and crawl for s3 backup objects
    async fn paths<'a>(
        conf: &'a Conf,
        root: &PathBuf,
        chars: usize,
        buffer: &[u8],
    ) -> Result<Vec<(String, String, PathBuf)>, Error>;
}

/// A sub worker for this upload worker
pub struct UploadSubWorker<R: S3Restore> {
    /// The type we are backing up
    phantom: PhantomData<R>,
    /// The Thorium config for the cluster we are restoring
    conf: Conf,
    /// The s3 client to use
    s3: Client,
    /// The progress bar to track progress with
    progress: ProgressBar,
    /// The kanal channel workers should send backup updates over
    updates: AsyncSender<S3MonitorUpdate>,
    /// The buffer to store our chunks in
    buffer: BytesMut,
    /// Track the parts we have uploaded
    parts: Vec<CompletedPart>,
}

impl<R: S3Restore> UploadSubWorker<R> {
    /// Create a new upload sub worker
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config for the cluster we are restoring
    /// * `s3` - The s3 client to upload objects with
    /// * `progress` - The progress bar to track progress with
    /// * `updates` - The channel to send monitor updates over
    pub fn new(
        conf: &Conf,
        s3: &Client,
        progress: &ProgressBar,
        updates: &AsyncSender<S3MonitorUpdate>,
    ) -> Self {
        // create our upload sub worker
        UploadSubWorker {
            phantom: PhantomData,
            conf: conf.clone(),
            s3: s3.clone(),
            progress: progress.clone(),
            updates: updates.clone(),
            buffer: BytesMut::with_capacity(5_242_880),
            parts: Vec::with_capacity(10),
        }
    }

    /// Stream a single object to s3
    async fn stream_object(
        &mut self,
        path: &PathBuf,
        bucket: &str,
        key: &str,
        upload_id: &str,
    ) -> Result<(), Error> {
        // make sure we do not have any lingering parts from old uploads
        self.parts.clear();
        // open a file handle to this file
        let mut stream = ReaderStream::new(File::open(path).await?);
        // track the current part numbers
        let mut part_num = 1;
        // start reading this file into our buffer
        while let Some(data) = stream.next().await {
            // get the bytes we read
            let data = data?;
            // write this data to our buffer
            self.buffer.extend_from_slice(&data);
            // determine how much data we have to send to s3
            if self.buffer.remaining() >= 5_242_880 {
                // build our byte stream
                let byte_stream = ByteStream::from(SdkBody::from(&self.buffer[..]));
                // write this buffer to s3
                let part = self
                    .s3
                    .upload_part()
                    .bucket(bucket)
                    .key(key)
                    .upload_id(upload_id)
                    .body(byte_stream)
                    .part_number(part_num)
                    .send()
                    .await?;
                // add this chunk to our parts list
                self.parts.push(
                    CompletedPart::builder()
                        .e_tag(part.e_tag.unwrap_or_default())
                        .part_number(part_num)
                        .build(),
                );
                // increment our progress bar
                self.progress.inc(self.buffer.remaining() as u64);
                // build the update to send to our global monitor
                let update = S3MonitorUpdate::Update(self.buffer.remaining());
                // send an update to the global bar
                self.updates.send(update).await?;
                // reset our packable and writable number of bytes to 0
                self.buffer.clear();
                // increment our part number
                part_num += 1;
            }
        }
        // build our byte stream
        let byte_stream = ByteStream::from(SdkBody::from(&self.buffer[..]));
        // write this buffer to s3
        let part = self
            .s3
            .upload_part()
            .bucket(bucket)
            .key(key)
            .upload_id(upload_id)
            .body(byte_stream)
            .part_number(part_num)
            .send()
            .await?;
        // add this chunk to our parts list
        self.parts.push(
            CompletedPart::builder()
                .e_tag(part.e_tag.unwrap_or_default())
                .part_number(part_num)
                .build(),
        );
        // build our complete multipart upload object
        let completed_parts = CompletedMultipartUpload::builder()
            .set_parts(Some(self.parts.clone()))
            .build();
        // finish this multipart upload
        self.s3
            .complete_multipart_upload()
            .bucket(bucket)
            .key(key)
            .multipart_upload(completed_parts)
            .upload_id(upload_id)
            .send()
            .await?;
        // increment our progress bar
        self.progress.inc(self.buffer.remaining() as u64);
        // build the update to send to our global monitor
        let update = S3MonitorUpdate::Update(self.buffer.remaining());
        // send an update to the global bar
        self.updates.send(update).await?;
        // reset our packable and writable number of bytes to 0
        self.buffer.clear();
        Ok(())
    }

    /// Upload a file to s3
    pub async fn upload_helper(&mut self, path: &PathBuf) -> Result<(), Error> {
        // get the bucket to right data too
        let (bucket, key) = R::parse(path, &self.conf)?;
        // initiate a multipart upload to s3
        let init = self
            .s3
            .create_multipart_upload()
            .bucket(&bucket)
            .key(&key)
            .content_type("application/octet-stream")
            .send()
            .await?;
        // get our upload id
        let upload_id = match init.upload_id() {
            Some(upload_id) => upload_id,
            None => return Err(Error::new("Failed to get upload id")),
        };
        // upload this file to s3
        if let Err(error) = self.stream_object(path, &bucket, &key, &upload_id).await {
            // clear our buffer
            self.buffer.clear();
            // log this error
            self.progress
                .println(format!("Failed to restore object: {:#?}", error));
            // delete our multipart upload
            self.s3
                .abort_multipart_upload()
                .upload_id(upload_id)
                .send()
                .await?;
        }
        Ok(())
    }

    /// Uploads a file to s3
    pub async fn upload(mut self, path: PathBuf) -> Self {
        // try to upload this file
        if let Err(error) = self.upload_helper(&path).await {
            // log this error
            self.progress.println(format!(
                "Failed to abort multipart upload for {path:?} : {error:#?}"
            ));
        }
        self
    }
}

pub struct S3RestoreWorker<R: S3Restore> {
    /// The type we are backing up
    phantom: PhantomData<R>,
    /// The Thorium config for the cluster we are restoring
    conf: Conf,
    /// The kanal channel workers should send backup updates over
    updates: AsyncSender<S3MonitorUpdate>,
    /// The progress bar to track progress with
    progress: ProgressBar,
    /// The s3 client to download files with
    s3: Client,
    /// The workers waiting for active tasks
    workers: Vec<UploadSubWorker<R>>,
    /// The currently active downloads to monitor
    active: FuturesOrdered<JoinHandle<UploadSubWorker<R>>>,
    /// Track the number of objects we have restored
    restored: u64,
}

impl<R: S3Restore> S3RestoreWorker<R> {
    /// Create a new restore worker
    ///
    /// # Arguments
    ///
    /// # Arguments
    ///
    /// * `config` - The config for this Thorium cluster
    /// * `updates` - The channel to send partition archive updates on
    /// * `progress` - The progress bar for this worker
    pub fn new(conf: &Conf, updates: &AsyncSender<S3MonitorUpdate>, progress: ProgressBar) -> Self {
        // get our s3 conf
        let s3_conf = &conf.thorium.s3;
        // get our s3 credentials
        let creds = Credentials::new(
            &s3_conf.access_key,
            &s3_conf.secret_token,
            None,
            None,
            "Thorium",
        );
        // build our s3 s3_config
        let s3_config = aws_sdk_s3::config::Builder::new()
            .endpoint_url(&s3_conf.endpoint)
            .region(aws_types::region::Region::new(s3_conf.region.clone()))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            .force_path_style(true)
            .build();
        // build our s3 client
        let s3 = Client::from_conf(s3_config);
        // create a sub worker
        let workers = (0..10)
            .into_iter()
            .map(|_| UploadSubWorker::new(conf, &s3, &progress, updates))
            .collect();
        // build our worker
        S3RestoreWorker {
            phantom: PhantomData::default(),
            conf: conf.clone(),
            updates: updates.clone(),
            progress,
            s3,
            workers,
            active: FuturesOrdered::new(),
            restored: 0,
        }
    }

    /// Check on our pending futures and wait if we have enough in flight
    pub async fn check_active(&mut self) -> UploadSubWorker<R> {
        // try to get an unused worked
        match self.workers.pop() {
            Some(worker) => worker,
            // we don't yet have a worker so wait for one to finish
            None => {
                // we have 10 active workers already so wait for a worker to finish
                if let Some(join) = self.active.next().await {
                    // check if we had a join error
                    match join {
                        Ok(sub_worker) => {
                            // update our restored count
                            self.restored += 1;
                            // set our new message
                            self.progress.set_message(self.restored.to_string());
                            // return this worker
                            sub_worker
                        }
                        Err(error) => {
                            // log this join error
                            self.progress.println(format!("JoinError: {error:#?}"));
                            // make a new sub worker
                            UploadSubWorker::new(
                                &self.conf,
                                &self.s3,
                                &self.progress,
                                &self.updates,
                            )
                        }
                    }
                } else {
                    // make a new sub worker
                    UploadSubWorker::new(&self.conf, &self.s3, &self.progress, &self.updates)
                }
            }
        }
    }

    /// Wait for all active tasks to complete
    pub async fn flush_active(&mut self) {
        // wait for all active tasks to complete
        while let Some(join) = self.active.next().await {
            // check if we had a join error
            match join {
                Ok(_) => {
                    // update our restored count
                    self.restored += 1;
                    // set our new message
                    self.progress.set_message(self.restored.to_string());
                }
                // log this join error
                Err(error) => self.progress.println(format!("JoinError: {error:#?}")),
            }
        }
    }

    /// Start restoring objects
    ///
    /// # Arguments
    ///
    /// * `orders` - The s3 objects to restore
    pub async fn restore(mut self, orders: AsyncReceiver<PathBuf>) -> Result<Self, Error> {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            let path = match orders.recv().await {
                Ok(path) => path,
                Err(kanal::ReceiveError::Closed | kanal::ReceiveError::SendClosed) => break,
            };
            // check if we already have enough active workers
            let worker = self.check_active().await;
            // start uploading this file
            let handle = tokio::spawn(async move { worker.upload(path).await });
            // add this to our futures
            self.active.push_back(handle);
        }
        // wait for all active downloads to finish
        self.flush_active().await;
        // close our progress bar
        self.progress.finish();
        Ok(self)
    }
}

/// The trait for restoring s3 objects
pub trait S3Restore: Utils + std::fmt::Debug + 'static + Send {
    /// Get the bucket and s3 path for this file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file we are restoring
    /// * `conf` - The Thorium for this cluster
    fn parse(path: &PathBuf, conf: &Conf) -> Result<(String, String), Error>;
}
