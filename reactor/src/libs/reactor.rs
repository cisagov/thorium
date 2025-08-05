//! Handles spawning containers directly for windows nodes
//!
//! This support could likely be extended to linux k8s and baremetal nodes but
//! for k8s nodes would come at the cost of everything k8s buys us.

use chrono::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use thorium::models::{Component, NodeGetParams, Worker, WorkerStatus};
use thorium::{Error, Keys, Thorium};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::{event, instrument, span, Level, Span};

use super::keys;
use super::launchers::{self, Launcher};
use super::tasks::{self, Tasks};
use crate::args::Args;

/// Adds a task back into our task queue at the right time
macro_rules! add_task {
    ($reactor:expr, $task:expr) => {{
        // get the datetime to start this task at
        let mut start = crate::from_now!($task.delay() as i64);
        // increase by 1 until we have found an open slot to start this job
        loop {
            // determine if a task already exists for this date
            if $reactor.tasks.get(&start).is_none() {
                break;
            }
            // increment start by 1 and try again
            start = start + chrono::Duration::seconds(1);
        }
        $reactor.tasks.insert(start, $task)
    }};
}

/// Check if an operation failed and if so who we should ban
macro_rules! try_ban {
    ($operation:expr, $user:expr, $bans:expr, $span:expr) => {
        match $operation {
            Ok(user) => user,
            Err(error) => {
                // log this error
                event!(
                    parent: &$span,
                    Level::ERROR,
                    error = true,
                    error_msg = error.to_string()
                );
                // add this user to our ban set
                $bans.insert($user.to_owned());
                // skip to the next user
                continue;
            }
        }
    };
}

/// The daemon that will monitor this nodes worker and spawn them
pub struct Reactor {
    /// The client used to talk to Thorium
    pub thorium: Thorium,
    /// The name of the cluster this node is in
    pub cluster: String,
    /// The name of this node
    pub name: String,
    /// A map of currently active workers on this node
    pub active: HashMap<String, Worker>,
    /// A queue of tasks to complete sorted by the time to start executing them
    tasks: BTreeMap<DateTime<Utc>, Tasks>,
    /// Allows the agent to poll the system for info
    system: System,
    /// The launcher to use when launching jobs
    launcher: Box<dyn Launcher>,
    /// The args used to start this reactor
    args: Args,
    /// Stop spawning new agents as an update is needed
    halt_spawning: bool,
    /// shutdown this reactor and exit
    shutdown: bool,
}

impl Reactor {
    /// Create a new reactor
    ///
    /// # Arguments
    ///
    /// * `args` - The args passed to this reactor
    pub async fn new(args: Args) -> Result<Self, Error> {
        // create a new Thorium client
        let thorium = Thorium::from_key_file(&args.keys).await?;
        // get this nodes name
        let name = args.node()?;
        // setup our task queue
        let tasks = Tasks::setup_queue();
        // configure our system poller to listen to specific info
        let refresh = RefreshKind::default()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        // set up a system poller
        let system = System::new_with_specifics(refresh);
        // build our launcher
        let launcher = launchers::new(&args);
        // build our reactor
        let reactor = Reactor {
            thorium,
            cluster: args.cluster.clone(),
            name,
            active: HashMap::default(),
            tasks,
            system,
            launcher,
            args,
            halt_spawning: false,
            shutdown: false,
        };
        Ok(reactor)
    }

    /// Check if we need to spawn and execute any tasks
    async fn spawn_tasks(&mut self, span: &Span) -> Result<(), Error> {
        // start our spawning tasks span
        let span = span!(parent: span, Level::INFO, "Spawning Tasks");
        // get the current timestamp
        let now = Utc::now();
        // track the tasks we completed
        let mut completed = Vec::default();
        // get any tasks we want to spawn and build a list of completed blocking tasks to rerun again
        for (_, task) in self.tasks.extract_if(.., |time, _| time < &now) {
            // log that we are spawning a task
            event!(parent: &span, Level::INFO, task = task.as_str());
            // spawn or execute this task
            match task {
                Tasks::Resources => {
                    // update this nodes resource in Thorium
                    tasks::update_resources(
                        &self.cluster,
                        &self.name,
                        &self.thorium,
                        &mut self.system,
                        &span,
                    )
                    .await?;
                    // add this task to be readded to our task queue
                    completed.push(Tasks::Resources);
                }
            }
        }
        // add any blocking completed tasks back to our task list
        for task in completed {
            add_task!(self, task);
        }
        Ok(())
    }

    /// Check if we need to spawn any new workers
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    async fn poll(&mut self, span: &Span) -> Result<HashMap<String, Worker>, Error> {
        // start our reactor check loop
        let span = span!(parent: span, Level::INFO, "Poll Thorium For Node Changes");
        // use default params for this node
        let params = NodeGetParams::default().scaler(self.args.scaler);
        // get the current desired state for this node
        let mut info = self
            .thorium
            .system
            .get_node(&self.cluster, &self.name, &params)
            .await?;
        // check if any of our workers have completed yet
        self.launcher
            .check(&self.thorium, &mut info, &mut self.active, &span)
            .await?;
        // determine if any workers need to be shut down
        let (mut workers, shutdowns): (HashMap<String, Worker>, HashMap<String, Worker>) = info
            .workers
            .into_iter()
            .partition(|(_, worker)| worker.status != WorkerStatus::Shutdown);
        // if we have workers to shutdown then do that
        if !shutdowns.is_empty() {
            // downselect to just our workers names
            let names = shutdowns.into_iter().map(|(name, _)| name).collect();
            // shutdown these workers
            self.launcher.shutdown(&self.thorium, names, &span).await?;
        }
        // compare to our currently active workers and determine what needs to be spawned still
        workers.retain(|name, _| !self.active.contains_key(name));
        // log how many changes are needed if any are
        if !workers.is_empty() {
            event!(parent: &span, Level::INFO, changes = workers.len());
        }
        Ok(workers)
    }

    /// Make sure that the keys for our target workers are loaded
    ///
    /// # Arguments
    ///
    /// * `changes` - The changes to the current workers to apply this loop
    /// * `span` - The span to log traces under
    async fn setup_keys(&mut self, changes: &mut HashMap<String, Worker>, span: &Span) {
        // start our reactor check loop
        let span = span!(parent: span, Level::INFO, "Setup Client Keys");
        // get a list of active users to write keys for
        let mut users = self
            .active
            .iter()
            .map(|(_, worker)| &worker.user)
            .collect::<HashSet<_>>();
        // check our new containers users keys too
        users.extend(changes.iter().map(|(_, worker)| &worker.user));
        // track the users we should ban
        let mut bans: HashSet<String> = HashSet::default();
        // try to setup all of our users tokens
        for name in users {
            // get this users info
            let user = try_ban!(self.thorium.users.get(&name).await, name, bans, span);
            // build the path to store this users keys at
            let path = keys::path(&user.username);
            // check if this users keys are already set
            if !try_ban!(keys::exists(&path, &user.token).await, name, bans, span) {
                // make sure all of our parent paths exists
                if let Some(parent) = path.parent() {
                    try_ban!(tokio::fs::create_dir_all(parent).await, name, bans, span);
                }
                // we need to create this users keys since they don't exist
                // build the keys object for this user
                let keys = Keys::new_token(&self.thorium.host, &user.token);
                // serialize our keys
                let serialized = try_ban!(serde_yaml::to_string(&keys), name, bans, span);
                // write our serialized keys to disk
                let mut file = try_ban!(File::create(&path).await, name, bans, span);
                let write = file.write_all(serialized.as_bytes());
                try_ban!(write.await, name, bans, span);
            }
        }
        // drop any workers that we failed to setup keys for
        changes.retain(|_, worker| !bans.contains(&worker.user));
    }

    /// Launcher all of our jobs
    async fn launch(&mut self, new: HashMap<String, Worker>, span: &Span) {
        // start our reactor check loop
        let span = span!(
            parent: span,
            Level::INFO,
            "Launch Workers",
            workers = new.len()
        );
        // only launch new workers if spawning hasn't been halted
        if !self.halt_spawning {
            // launch each of our jobs
            for (name, worker) in new {
                // launch this workers job
                // we already log errors so we just care about successes
                match self.launcher.launch(&self.thorium, &worker, &span).await {
                    // add this active worker to our active workers set
                    Ok(_) => {
                        self.active.insert(name, worker);
                    }
                    // we failed to spawn this workers
                    Err(error) => {
                        event!(
                            parent: &span,
                            Level::ERROR,
                            error = true,
                            error_msg = error.to_string()
                        );
                    }
                };
            }
        }
    }

    /// Check if we need an update or not and apply it if possible
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    #[instrument(name = "Reactor::needs_update", skip_all, err(Debug))]
    async fn needs_update(&mut self) -> Result<(), Error> {
        // Get the current Thorium version
        let version = self.thorium.updates.get_version().await?;
        // get the current version
        let current = env!("CARGO_PKG_VERSION");
        // log our current versions
        event!(
            Level::INFO,
            reactor = current,
            api = version.thorium.to_string()
        );
        // compare to our version and see if its different
        if version.thorium != semver::Version::parse(current)? {
            // start our update needed span
            event!(Level::INFO, update_needed = true,);
            // set the halt spawning flag so we stop spawning new agents
            self.halt_spawning = true;
            // only update if we have no active jobs
            if self.active.is_empty() {
                // update our agents
                self.thorium
                    .updates
                    .update_other(Component::Agent, "/opt/thorium/thorium-agent")
                    .await?;
                // update ourselves
                self.thorium.updates.update(Component::Reactor).await?;
                // shutdown this reactor
                self.shutdown = true;
            } else {
                event!(Level::INFO, msg = "Cannot update with active jobs");
            }
        }
        Ok(())
    }

    /// Start polling Thorium for changes to apply to this node
    pub async fn start(mut self) -> Result<(), Error> {
        // start our initial setup span
        let init_span = span!(Level::INFO, "Initial Setup");
        // apply any needed updates
        self.needs_update().await?;
        // update this nodes resource info
        tasks::update_resources(
            &self.cluster,
            &self.name,
            &self.thorium,
            &mut self.system,
            &init_span,
        )
        .await?;
        // drop our span so that the loop isn't included
        drop(init_span);
        // loop forever getting the desired state and applying it
        loop {
            // start our reactor check loop
            let span = span!(Level::INFO, "Reactor Loop");
            // apply any needed updates
            self.needs_update().await?;
            // check if we have any tasks that need to be spawned
            self.spawn_tasks(&span).await?;
            // check for changes in this node
            let mut changes = self.poll(&span).await?;
            // make sure our users keys are setup
            self.setup_keys(&mut changes, &span).await;
            // spawn our changes
            self.launch(changes, &span).await;
            // drop our span so that the sleep isn't included in timing info
            drop(span);
            // shutdown this reactor if needed
            if self.shutdown {
                break;
            }
            // sleep for the configured dwell between scale attempts
            let dwell = std::time::Duration::from_secs(2);
            tokio::time::sleep(dwell).await;
        }
        Ok(())
    }
}
