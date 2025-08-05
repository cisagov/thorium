//! A worker that streams data to a search store

use chrono::prelude::*;
use futures::StreamExt;
use kanal::{AsyncReceiver, AsyncSender};
use scylla::client::session::Session;
use std::convert::Into;
use std::sync::Arc;
use thorium::models::SearchEventType;
use thorium::Error;
use tracing::instrument;

use crate::events::CompactEvent;
use crate::index::{IndexMapping, IndexTyped};
use crate::msg::{Job, JobStatus};
use crate::sources::DataSource;
use crate::stores::{SearchStore, StoreLookup};

pub struct StreamWorker<D: DataSource, S: SearchStore> {
    /// A client for scylla
    scylla: Arc<Session>,
    /// The channel to pull jobs from
    jobs_rx: AsyncReceiver<Job<D>>,
    /// The channel to send progress updates on
    progress_tx: AsyncSender<JobStatus>,
    /// The source of data to index
    source: D,
    /// The store for data
    store: S,
}

impl<D: DataSource, S: SearchStore> StreamWorker<D, S>
where
    D::IndexType: IndexMapping<S>,
{
    /// Create a new stream worker
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium conf to use
    /// * `scylla` - A scylla client
    /// * `jobs_rx` - The channel to poll for jobs
    /// * `progress_tx` - The channel to send progress to the monitor on
    /// * `source` - An instance of the data we're pulling from in scylla
    /// * `store` - The search store to stream to
    pub fn new(
        scylla: Arc<Session>,
        jobs_rx: AsyncReceiver<Job<D>>,
        progress_tx: AsyncSender<JobStatus>,
        source: D,
        store: S,
    ) -> Self {
        Self {
            scylla,
            jobs_rx,
            progress_tx,
            source,
            store,
        }
    }

    /// Handle an init job
    ///
    /// # Arguments
    ///
    /// * `event` - The event to handle
    #[instrument(
        name = "Worker::handle_init",
        skip(self),
        fields(store = S::STORE_NAME, data = D::DATA_NAME),
        err(Debug)
    )]
    async fn handle_init(&mut self, start: i64, end: i64) -> Result<(), Error> {
        // enumerate the items we need to pull data for
        let resp = self
            .scylla
            .execute_iter(self.source.enumerate_prepared().clone(), (start, end))
            .await
            .map_err(|err| Error::new(format!("Failed to enumerate data from Scylla: {err}")))?;
        // chunk into how many things we want to pull and send to the store concurrently
        let mut typed_stream = resp.rows_stream::<D::InitRow>()?.chunks(D::INIT_CONCURRENT);
        while let Some(rows) = typed_stream.next().await {
            // check for any errors getting the rows
            let rows = rows.into_iter().collect::<Result<Vec<_>, _>>()?;
            // convert our init rows into info required to pull a bundle
            let bundle_info: Vec<<D as DataSource>::InitInfo> =
                rows.into_iter().map(Into::into).collect();
            // pull data and bundle it together
            let bundles = self.source.bundle_init(bundle_info, &self.scylla).await?;
            for (index_type, bundles) in bundles {
                // send the data bundles to the search store
                self.send_docs(index_type, bundles).await?;
            }
        }
        // tell the monitor an init job was completed
        self.progress_tx
            .send(JobStatus::InitComplete { start, end })
            .await?;
        Ok(())
    }

    /// Handle an event job
    ///
    /// # Arguments
    ///
    /// * `event` - The event to handle
    #[instrument(
        name = "Worker::handle_event",
        skip_all,
        fields(store = S::STORE_NAME, data = D::DATA_NAME)
        err(Debug)
    )]
    async fn handle_event(&mut self, compacted_event: D::CompactEvent) -> Result<(), Error> {
        // get the type from the event that maps to the corresponding index for this event
        let index_type = compacted_event.index_type();
        // determine the event's type
        match compacted_event.get_type() {
            // the item was modified, so re-stream it for its groups
            SearchEventType::Modified => {
                // get data for the item the event is referring to and bundle it up
                let bundles = self
                    .source
                    .bundle_event(compacted_event, &self.scylla)
                    .await?;
                // send the bundles to the search store
                self.send_docs(index_type, bundles).await?;
            }
            // the item was deleted, so delete it in all its groups
            SearchEventType::Deleted => {
                // get the IDs to the event's respective documents
                let store_ids = compacted_event
                    .store_ids()
                    .into_iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>();
                // delete the documents with those id's
                self.store
                    .delete(index_type.map_index(), &store_ids)
                    .await?;
            }
        }
        Ok(())
    }

    /// Send data bundles to the search store
    ///
    /// # Arguments
    ///
    /// * `index` - The type defining which index to send the data to
    /// * `bundles` - The bundles to send
    #[instrument(
        name = "Worker::send_docs",
        skip_all,
        fields(store = S::STORE_NAME, data = D::DATA_NAME),
        err(Debug)
    )]
    async fn send_docs(
        &mut self,
        index_type: D::IndexType,
        bundles: Vec<D::DataBundle>,
    ) -> Result<(), Error> {
        // get the current timestamp
        let now = Utc::now();
        // serialize the bundles to JSON values to stream
        let values = D::to_values(&bundles, &index_type, now)?;
        // send this data to our search store
        self.store.create(index_type.map_index(), values).await?;
        Ok(())
    }

    /// Poll our job queue and stream data to our search store
    #[instrument(name = "Worker::start", skip_all, fields(store = S::STORE_NAME, data = D::DATA_NAME), err(Debug))]
    pub async fn start(mut self) -> Result<(), Error> {
        // pop messages in our queue until it closes
        while let Ok(msg) = self.jobs_rx.recv().await {
            // handle this message
            match msg {
                // handle an init job; if an init job fails, we error out completely
                Job::Init { start, end } => self.handle_init(start, end).await?,
                // handle an event job; if an event job fails, we tell the monitor to tell the Thorium API
                Job::Event { compacted_event } => {
                    // get a copy of the event's IDs before we move the compacted event
                    let ids = compacted_event.get_ids();
                    match self.handle_event(compacted_event).await {
                        Ok(()) => {
                            self.progress_tx
                                .send(JobStatus::EventComplete { ids })
                                .await?;
                        }
                        Err(error) => {
                            self.progress_tx
                                .send(JobStatus::EventError { error, ids })
                                .await?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
