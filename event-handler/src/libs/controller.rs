//! The controller for handling events in Thorium
use futures_locks::RwLock;
use std::ops::DerefMut;
use std::sync::Arc;
use std::time::Duration;
use thorium::models::{EventCacheStatusOpts, EventType};
use thorium::{Conf, Error, Thorium};
use tokio::task::JoinHandle;

use super::cache::TriggerCache;
use super::worker::EventWorker;
use crate::args::Args;

/// The controller for handling events in Thorium
pub struct EventController {
    /// A Thorium client
    thorium: Arc<Thorium>,
    /// A shared trigger cache
    triggers: Arc<RwLock<TriggerCache>>,
    /// The different worker handles
    handles: Vec<JoinHandle<Result<(), Error>>>,
    /// The max depth to check for new triggers at
    max_depth: u8,
}

impl EventController {
    /// Create a new event handler controller
    ///
    /// # Arguments
    ///
    /// * `args` - The command line args passed to the event handler
    /// * `conf` - The Thorium Config
    pub async fn new(args: Args, conf: Conf) -> Result<Self, Error> {
        // build a Thorium client
        let thorium = Arc::new(Thorium::from_key_file(&args.auth).await?);
        // get our max depth
        let max_depth = conf.thorium.events.max_depth;
        // get a new trigger cache object
        let triggers = TriggerCache::new(&thorium, max_depth).await?;
        // build our handler
        let handler = EventController {
            thorium,
            triggers: Arc::new(RwLock::new(triggers)),
            handles: Vec::with_capacity(1),
            max_depth,
        };
        Ok(handler)
    }

    /// Spawn all of our workers
    pub async fn spawn(&mut self) {
        // create and spawn our one and only worker
        // well need to figure out how to properly wrap this so
        // we retain worker type on failure but for now we just
        // have one worker so ¯\_(ツ)_/¯
        let worker = EventWorker::new(&self.thorium, &self.triggers, EventType::ReactionTrigger);
        // spawn our only event worker
        let handle = tokio::task::spawn(worker.start());
        // add this to our task list
        self.handles.push(handle);
    }

    /// Check our caches status
    async fn check_cache_status(&mut self) -> Result<(), Error> {
        // set the options for getting our event cache info
        let opts = EventCacheStatusOpts::default().reset();
        // check if our cache needs to be updated
        let status = self.thorium.events.get_cache_status(&opts).await?;
        // if our cache needs to be updated then build a new one
        if status.triggers {
            // get a new trigger cache object
            let triggers = TriggerCache::new(&self.thorium, self.max_depth).await?;
            // get a writable lock on our trigger cache
            let mut lock = self.triggers.write().await;
            // get our trigger cache
            let cache = lock.deref_mut();
            // replace our trigger cache
            let _ = std::mem::replace(cache, triggers);
            // drop our lock
            drop(lock);
        }
        Ok(())
    }

    /// Check if any of our tasks have failed
    pub async fn check_tasks(&mut self) -> Result<(), Error> {
        // check all spawned tasks
        for handle in self.handles.iter_mut() {
            // check if this handle has finished
            if handle.is_finished() {
                // this handle has finish check what happened
                // in the future we should respawn just the failed task
                // but we only have one task so lets just catch fire
                handle.await??;
            }
        }
        Ok(())
    }

    /// Start handling events of all event types
    pub async fn start(mut self) {
        // spawn all of our workers
        self.spawn().await;
        // loop forever checking for task failures or if we should update our trigger cache
        loop {
            // check our cache status
            self.check_cache_status().await.unwrap();
            // check if any of our tasks have failed
            self.check_tasks().await.unwrap();
            // sleep for 5 seconds
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }
}
