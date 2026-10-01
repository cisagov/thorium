//! A worker that streams data to a search store

use chrono::prelude::*;
use futures::StreamExt;
use kanal::{AsyncReceiver, AsyncSender};
use scylla::client::session::Session;
use std::convert::Into;
use std::sync::Arc;
use thorium::Error;
use thorium::models::SearchEventType;
use tracing::instrument;

use crate::events::CompactEvent;
use crate::index::{IndexMapping, IndexTyped};
use crate::msg::{Job, JobStatus};
use crate::sources::DataSource;
use crate::stores::{SearchStore, StoreIdentifiable, StoreLookup};

/// Pick the documents a Modified event leaves with no data to re-index
///
/// When the last tags of an item in a group are deleted, re-reading that group returns zero rows,
/// so no bundle (and no document) is produced for it. The documents of every event id that
/// produced no bundle must be deleted, otherwise they keep listing the deleted data forever.
///
/// # Arguments
///
/// * `event_ids` - The ids of every document the Modified event refers to
/// * `bundle_ids` - The ids of the documents that were re-bundled with fresh data
fn stale_modified_store_ids(event_ids: &[String], bundle_ids: &[String]) -> Vec<String> {
    // any document that was not re-bundled no longer has any data
    event_ids
        .iter()
        .filter(|id| !bundle_ids.contains(id))
        .cloned()
        .collect()
}

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
    /// The config for the search-streamer
    conf: thorium::conf::SearchStreamer,
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
        conf: thorium::conf::SearchStreamer,
    ) -> Self {
        Self {
            scylla,
            jobs_rx,
            progress_tx,
            source,
            store,
            conf,
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
                // get the ids to every document this event refers to before we consume it
                let event_ids = compacted_event
                    .store_ids()
                    .into_iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>();
                // get data for the item the event is referring to and bundle it up
                let bundles = self
                    .source
                    .bundle_event(compacted_event, &self.scylla)
                    .await?;
                // get the ids of the documents that still have data
                let bundle_ids = bundles
                    .iter()
                    .map(|bundle| bundle.as_store_id().to_string())
                    .collect::<Vec<_>>();
                // find any documents left with no data to re-index
                let stale_ids = stale_modified_store_ids(&event_ids, &bundle_ids);
                self.send_docs(index_type, bundles).await?;
                // delete any documents that no longer have any data
                if !stale_ids.is_empty() {
                    self.store
                        .delete(index_type.map_index()?, &stale_ids)
                        .await?;
                }
            }
            // the item was deleted, so delete it in all its groups
            SearchEventType::Deleted => {
                // get the ids to the event's respective documents
                let store_ids = compacted_event
                    .store_ids()
                    .into_iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>();
                // delete the documents with those id's
                self.store
                    .delete(index_type.map_index()?, &store_ids)
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
        let index = index_type.map_index()?;
        // serialize the bundles to JSON values to stream
        let values = D::to_values(
            bundles,
            &index_type,
            now,
            self.conf.max_document_size.into(),
        )?;
        // send this data to our search store
        self.store.create(index, values).await?;
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
                    // get a copy of the event's ids before we move the compacted event
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deleting the last remaining tag(s) of an item in a group emits a Modified search event.
    /// Re-reading the group from Scylla returns zero rows, so no `TagBundle` is produced for it,
    /// and every document the event refers to that produced no bundle is selected for deletion.
    /// This keeps the existing `{item}-{group}` document (still listing the deleted tags) from
    /// staying searchable in Elasticsearch forever.
    #[test]
    fn deleting_last_tag_removes_stale_doc() {
        // the event refers to the item's documents in two groups
        let event_ids = vec!["sha256-corn".to_owned(), "sha256-wheat".to_owned()];
        // only the corn group still has tags, the wheat group's last tag was deleted
        let bundle_ids = vec!["sha256-corn".to_owned()];
        // pick the documents left with no data
        let stale = stale_modified_store_ids(&event_ids, &bundle_ids);
        // the wheat document must be deleted so the deleted tags stop matching searches
        assert_eq!(
            stale,
            vec!["sha256-wheat".to_owned()],
            "a document whose group has no remaining data must be deleted"
        );
    }

    /// Documents that were re-bundled with fresh data are overwritten, not deleted, so live
    /// documents are never removed.
    #[test]
    fn rebundled_docs_are_not_deleted() {
        // every group the event refers to still has tags
        let event_ids = vec!["sha256-corn".to_owned()];
        let bundle_ids = vec!["sha256-corn".to_owned()];
        // nothing should be deleted
        assert!(stale_modified_store_ids(&event_ids, &bundle_ids).is_empty());
    }
}
