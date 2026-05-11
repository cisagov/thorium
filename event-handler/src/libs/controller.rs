//! The controller for handling events in Thorium
use kanal::AsyncSender;
use linearize::StaticMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use strum::IntoEnumIterator;
use thorium::models::{EventCacheStatusOpts, ScrubbedUser};
use thorium::{Conf, Error, Thorium};
use tokio::task::JoinHandle;

use super::cache::TriggerCache;
use crate::args::Args;
use crate::libs::workers::{WorkerCacheUpdates, WorkerMsg};

use super::stats::EventWorkerStatChannels;
use super::workers::{EventWorkerKinds, SigmaRuleContext, WorkerCacheUpdateKinds};

/// Get info on all users in Thorium
///
/// # Arguments
///
/// * `thorium` - a Thorium client
async fn get_users(thorium: &Thorium) -> Result<HashMap<String, ScrubbedUser>, Error> {
    // get all users in Thorium
    let list = thorium.users.list_details().await?;
    // create a user map preallocated for the number of users we found
    let mut users = HashMap::with_capacity(list.len());
    // convert this user list into a map of users by name
    for user in list {
        // add this user to our map
        users.insert(user.username.clone(), user);
    }
    // get all pipelines in Thorium so we can
    Ok(users)
}

/// The shared cache/context for different workers
pub struct EventWorkerCache {
    /// A shared trigger cache
    pub triggers: TriggerCache,
    /// A shared sigma context
    pub sigma: SigmaRuleContext,
    /// The users we know about
    pub users: HashMap<String, ScrubbedUser>,
    /// The config for Thorium
    pub conf: Conf,
}

impl EventWorkerCache {
    /// Create a new cache
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorium config
    /// * `thorium` - A Thorium api client
    pub async fn new(conf: Conf, thorium: &Thorium) -> Result<Self, Error> {
        // get our max depth
        let max_depth = conf.thorium.events.max_depth;
        // get a new trigger cache object
        let triggers = TriggerCache::new(thorium, max_depth).await?;
        // build a new sigma rule context
        let sigma = SigmaRuleContext::new(thorium).await?;
        // get all of the users currently in Thorium
        let users = get_users(thorium).await?;
        // build our cache
        let cache = EventWorkerCache {
            triggers,
            sigma,
            users,
            conf,
        };
        Ok(cache)
    }

    /// rebuild our triggers cache if its stale
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium api client
    /// * `updated` - The kinds of cache data that has been updated
    async fn rebuild_stale_triggers(
        &mut self,
        thorium: &Arc<Thorium>,
        updated: &mut Vec<WorkerCacheUpdateKinds>,
    ) -> Result<(), Error> {
        // set the options for getting our event cache info
        let opts = EventCacheStatusOpts::default().reset();
        // check if our cache needs to be updated
        let status = thorium.events.get_cache_status(&opts).await?;
        // if our cache needs to be updated then build a new one
        if status.triggers {
            // get a new trigger cache object
            self.triggers = TriggerCache::new(thorium, self.conf.thorium.events.max_depth).await?;
        }
        // we updated our reaction triggers
        updated.push(WorkerCacheUpdateKinds::ReactionTriggers);
        Ok(())
    }

    /// Rebuild any stale parts of our cache
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium api client
    pub async fn rebuild_stale(
        &mut self,
        thorium: &Arc<Thorium>,
    ) -> Result<Vec<WorkerCacheUpdateKinds>, Error> {
        // track cache data if any was updated
        let mut updated = Vec::default();
        // rebuild our triggers cache if its stale
        self.rebuild_stale_triggers(thorium, &mut updated).await?;
        Ok(updated)
    }
}

/// A channel and task join handle
type ChannelAndHandle = (AsyncSender<WorkerMsg>, JoinHandle<Result<(), Error>>);

/// A controller for handling different event workers in Thorium
pub struct EventWorkerController {
    /// A Thorium client
    thorium: Arc<Thorium>,
    /// The shared cache across different workers
    shared_cache: EventWorkerCache,
    /// The stats channels for different worker kinds
    stats_channels: EventWorkerStatChannels,
    /// The different stats reporter handles
    stats_handles: StaticMap<EventWorkerKinds, Option<JoinHandle<Result<(), Error>>>>,
    /// The different worker handles
    handles: StaticMap<EventWorkerKinds, Vec<ChannelAndHandle>>,
}

impl EventWorkerController {
    /// Create a new controller
    ///
    /// # Arguments
    ///
    /// * `args` - The command line args passed to the event handler
    /// * `conf` - The Thorium Config
    pub async fn new(args: Args, conf: Conf) -> Result<Self, Error> {
        // build a Thorium client
        let thorium = Arc::new(Thorium::from_key_file(&args.auth).await?);
        // build a new shared cache
        let shared_cache = EventWorkerCache::new(conf, &thorium).await?;
        // build a new controller
        let controller = EventWorkerController {
            thorium,
            shared_cache,
            stats_channels: EventWorkerStatChannels::default(),
            stats_handles: StaticMap::default(),
            handles: StaticMap::default(),
        };
        Ok(controller)
    }

    /// Spawn a stat worker for each kind
    fn spawn_stats_worker(&mut self) {
        // spawn the stat worker for each event worker kind
        for kind in EventWorkerKinds::iter() {
            // spawn a starts worker for all valid worker kinds
            let handle = self.stats_channels.spawn_worker(kind);
            // keep track of this stats handle
            self.stats_handles[kind] = Some(handle);
        }
    }

    // check if our stats workers failed
    async fn check_stats(&mut self) -> Result<(), Error> {
        // check the stat worker for each event worker kind
        for kind in EventWorkerKinds::iter() {
            // get this worker kind stats worker handle
            match &mut self.stats_handles[kind] {
                Some(handle) => {
                    // check if this stats handler has finished
                    if handle.is_finished() {
                        // catch fire if our stats handler failed
                        handle.await??;
                    }
                }
                None => return Err(Error::new(format!("Missing stats worker for {kind}"))),
            }
        }
        Ok(())
    }

    /// Respawn any missing workers
    async fn respawn(&mut self) -> Result<(), Error> {
        // for the time being only spawn a single worker of each type
        // TODO we should spawn multiple workers in the future
        for kind in EventWorkerKinds::iter() {
            // count how many workers we need to respawn
            // TODO actually support more then one worker
            let needed = 1 - self.handles[kind].len();
            // spawn the correct number of workers
            for _ in 0..needed {
                // spawn one of this worker kind
                let (msg_tx, handle) = kind
                    .spawn(&self.shared_cache, &self.stats_channels, &self.thorium)
                    .await;
                // keep track of this worker
                self.handles[kind].push((msg_tx, handle));
            }
        }
        Ok(())
    }

    /// Check and rebuld any stale cache
    ///
    /// This will also inform all workers of their new cache items
    async fn check_cache(&mut self) -> Result<(), Error> {
        // rebuild any parts of our cache that are stale
        let update_kinds = self.shared_cache.rebuild_stale(&self.thorium).await?;
        // step over the updates that occured
        for update_kind in update_kinds {
            // step over our worker kinds and see if they are subscribed to this update
            for (worker_kind, handles) in &self.handles {
                // determine if this worker is subscribed to this kind of update
                if worker_kind.is_subscribed(update_kind) {
                    // build the update message to send
                    let cache_update = WorkerCacheUpdates::new(&self.shared_cache, update_kind);
                    // wrap our update
                    let wrapped = WorkerMsg::Update(cache_update);
                    // send this update to all of this kind of workers
                    for (msg_tx, _) in handles {
                        // send this update message to this worker
                        msg_tx.send(wrapped.clone()).await.unwrap();
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if any of our tasks have failed
    async fn check_tasks(&mut self) -> Result<(), Error> {
        // check all of the different worker kinds
        for kind in EventWorkerKinds::iter() {
            // track the handles to remove
            let mut to_remove = Vec::default();
            // check all spawnwed tasks
            for (index, (_, handle)) in self.handles[kind].iter_mut().enumerate() {
                // check if this handle has finished
                if handle.is_finished() {
                    // TODO we can't just catch fire here since we have multiple worker kinds now
                    handle.await??;
                    // keep track of a handle that we need to remove
                    to_remove.push(index);
                }
            }
            // remove any handles that have reached a terminal state
            // traverse them in reverse to ensure our indexes remain valid
            for index in to_remove.into_iter().rev() {
                // remove and drop this handle
                self.handles[kind].swap_remove(index);
            }
        }
        Ok(())
    }

    /// Start handling events of all event types
    pub async fn start(mut self) -> Result<(), Error> {
        // spawn all of our stats workers
        self.spawn_stats_worker();
        // spawn all of our workers
        self.respawn().await?;
        // loop forever checking for task failures or if we should update our trigger cache
        loop {
            // check our cache status and rebuild anything that is stale
            self.check_cache().await?;
            // check if any of our tasks have failed
            self.check_tasks().await?;
            // check if our stats workers have failed
            self.check_stats().await?;
            // sleep for 5 seconds
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }
}
