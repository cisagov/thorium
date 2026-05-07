//! Handles spawning containers directly for windows nodes
//!
//! This support could likely be extended to linux k8s and baremetal nodes but
//! for k8s nodes would come at the cost of everthing k8s buys us.

use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use thorium::models::{Arch, Component, NodeGetParams, Os, Version, Worker, WorkerStatus};
use thorium::{Keys, Thorium};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tracing::{Level, error, event, instrument};

use super::keys;
use super::launchers::Launcher;
use super::tasks::{self, Tasks};
use crate::Error;
use crate::args::Args;

/// Adds a task back into our task queue at the right time
macro_rules! add_task {
    ($reactor:expr, $task:expr) => {{
        // get the datetime to start this task at
        let mut start = crate::from_now!($task.delay());
        // increase by 1 until we have found an open slot to start this job
        loop {
            // determine if a task already exists for this date
            if $reactor.tasks.get(&start).is_none() {
                break;
            }
            // increment start by 1 and try again
            start += Duration::from_secs(1);
        }
        $reactor.tasks.insert(start, $task)
    }};
}

/// Check if an operation failed and if so who we should ban
macro_rules! try_ban {
    ($operation:expr, $user:expr, $bans:expr) => {
        match $operation {
            Ok(user) => user,
            Err(error) => {
                // log this error
                event!(Level::ERROR, error = true, error_msg = error.to_string());
                // add this user to our ban set
                $bans.insert($user.to_owned());
                // skip to the next user
                continue;
            }
        }
    };
}

/// The daemon that will monitor this nodes worker and spawn them
pub struct Reactor<L: Launcher> {
    /// The client used to talk to Thorium
    pub thorium: Arc<Thorium>,
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
    launcher: Arc<L>,
    /// The args used to start this reactor
    args: Args,
    /// Stop spawning new agents as an update is needed
    halt_spawning: bool,
    /// shutdown this reactor and exit
    shutdown: bool,
}

impl<L: Launcher> Reactor<L> {
    /// Create a new reactor
    ///
    /// # Arguments
    ///
    /// * `args` - The args passed to this reactor
    #[instrument(name = "Reactor::new", skip_all)]
    pub async fn new(args: Args, launcher: L) -> Result<Self, Error> {
        // create a new Thorium client
        let thorium = Thorium::from_key_file(&args.keys).await?;
        // get this nodes name
        let name = args.node()?;
        // setup our task queue
        let tasks = Tasks::setup_queue(&args);
        // configure our system poller to listen to specific info
        let refresh = RefreshKind::default()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        // setup a system poller
        let system = System::new_with_specifics(refresh);
        // build our reactor
        let reactor = Reactor {
            thorium: Arc::new(thorium),
            cluster: args.cluster.clone(),
            name,
            active: HashMap::default(),
            tasks,
            system,
            launcher: Arc::new(launcher),
            args,
            halt_spawning: false,
            shutdown: false,
        };
        Ok(reactor)
    }

    /// Check if we need to spawn and execute any tasks
    #[instrument(name = "Reactor::spawn_tasks", skip_all)]
    async fn spawn_tasks(&mut self) -> Result<(), Error> {
        // get the current timestamp
        let now = Utc::now();
        // track the tasks we completed
        let mut completed = Vec::default();
        // get any tasks we want to spawn and build a list of completed blocking tasks to rerun again
        let tasks_due: Vec<_> = self.tasks.extract_if(.., |time, _| time < &now).collect();
        for (_, task) in tasks_due {
            // log that we are spawning a task
            event!(Level::INFO, task = task.as_str());
            // spawn or execute this task
            match task {
                Tasks::Update => self.check_update().await?,
                Tasks::Resources => {
                    tasks::update_resources(
                        &self.cluster,
                        &self.name,
                        &self.thorium,
                        &mut self.system,
                    )
                    .await?;
                }
            }
            completed.push(task);
        }
        // add any blocking completed tasks back to our task list
        for task in completed {
            add_task!(self, task);
        }
        Ok(())
    }

    /// Check if we need to spawn any new workers
    ///
    /// Returns a map of the new workers we need to spawn
    #[instrument(name = "Reactor::poll", skip_all)]
    async fn poll(&mut self) -> Result<HashMap<String, Worker>, Error> {
        // use default params for this node
        let params = NodeGetParams::default().scaler(self.args.scaler);
        // get the current desired state for this node
        let mut info = self
            .thorium
            .system
            .get_node(&self.cluster, &self.name, &params)
            .await?;
        // get a mutable reference to the launcher from its Arc;
        // the only thing using this Arc was the launches themselves which are complete,
        // so this should never error
        let launcher_mut = Arc::get_mut(&mut self.launcher)
            .expect("Launcher Arc cannot be mutated because it is in use elsewhere");
        // reconcile the current worker state in the reactor and the API
        launcher_mut
            .reconcile(&self.thorium, &mut info, &mut self.active)
            .await?;
        // extract any workers that the API has registered for shutdown
        let (mut workers, shutdowns): (HashMap<String, Worker>, HashMap<String, Worker>) = info
            .workers
            .into_iter()
            .partition(|(_, worker)| worker.status != WorkerStatus::Shutdown);
        // shutdown those workers
        if !shutdowns.is_empty() {
            // downselect to just our workers names
            let names = shutdowns.into_keys().collect();
            launcher_mut.shutdown(&self.thorium, names).await?;
        }
        // compare to our currently active workers and determine what needs to be spawned still
        workers.retain(|name, _| !self.active.contains_key(name));
        // log how many changes are needed if any are
        if !workers.is_empty() {
            event!(Level::INFO, changes = workers.len());
        }
        Ok(workers)
    }

    /// Make sure that the the keys for our target workers are loaded
    ///
    /// # Arguments
    ///
    /// * `changes` - The changes to the current workers to apply this loop
    #[instrument(name = "Reactor::setup_keys", skip_all)]
    async fn setup_keys(&mut self, changes: &mut HashMap<String, Worker>) {
        // get a list of active users to write keys for
        let mut users = self
            .active
            .values()
            .map(|worker| &worker.user)
            .collect::<HashSet<_>>();
        // check our new containers users keys too
        users.extend(changes.values().map(|worker| &worker.user));
        // track the users we should ban
        let mut bans: HashSet<String> = HashSet::default();
        // try to setup all of our users tokens
        for name in users {
            // get this users info
            let user = try_ban!(self.thorium.users.get(name).await, name, bans);
            // build the path to store this users keys at
            let path = keys::path(&user.username);
            // check if this users keys are already set
            if !try_ban!(keys::exists(&path, &user.token).await, name, bans) {
                // make sure all of our parent paths exists
                if let Some(parent) = path.parent() {
                    try_ban!(tokio::fs::create_dir_all(parent).await, name, bans);
                }
                // we need to create this users keys since they don't exist
                // build the keys object for this user
                let keys = Keys::new_token(&self.thorium.host, &user.token);
                // serialize our keys
                let serialized = try_ban!(serde_norway::to_string(&keys), name, bans);
                // write our serialized keys to disk
                let mut file = try_ban!(File::create(&path).await, name, bans);
                let write = file.write_all(serialized.as_bytes());
                try_ban!(write.await, name, bans);
            }
        }
        // drop any workers that we failed to setup keys for
        changes.retain(|_, worker| !bans.contains(&worker.user));
    }

    /// Launch all of our jobs
    ///
    /// # Arguments
    ///
    /// * `new` - The new workers to launch
    #[instrument(name = "Reactor::launch", skip_all, fields(workers = new.len()))]
    async fn launch(&mut self, new: HashMap<String, Worker>) {
        // only launch new workers if spawning hasn't been halted
        if !self.halt_spawning {
            let mut join_set = tokio::task::JoinSet::new();
            // only launch up to 10 workers at a time
            let semaphore = Arc::new(Semaphore::new(10));
            for (name, worker) in new {
                event!(Level::DEBUG, msg = "Launching worker", worker = name);
                let semaphore = semaphore.clone();
                let launcher = self.launcher.clone();
                let thorium = self.thorium.clone();
                join_set.spawn(async move {
                    // clone the worker's name in case we error out
                    let worker_name = worker.name.clone();
                    // get a permit from the semaphore
                    match semaphore.acquire().await {
                        Ok(_permit) => launcher
                            .launch(&thorium, worker)
                            .await
                            .map_err(|err| (worker_name, err)),
                        Err(err) => Err((
                            worker_name,
                            Error::with_context("Error acquiring semaphore permit", err),
                        )),
                    }
                });
            }
            // log errors and unzip workers and launches into separate collections
            let (workers, launches): (Vec<_>, Vec<_>) = join_set
                .join_all()
                .await
                .into_iter()
                // log all errors and only get the successes
                .filter_map(|res| match res {
                    Ok(success) => Some(success),
                    Err((worker_name, err)) => {
                        error!(
                            msg = "Error launching worker",
                            worker = worker_name,
                            err = ?err
                        );
                        None
                    }
                })
                .unzip();
            // zip worker *references* and launches back up for the to launcher resolve
            let launches = workers.iter().zip(launches.into_iter()).collect();
            // get a mutable reference to the launcher from its Arc;
            // the only thing using this Arc was the launches themselves which are complete,
            // so this should never error
            let launcher_mut = Arc::get_mut(&mut self.launcher)
                .expect("Launcher Arc cannot be mutated because it is in use elsewhere");
            launcher_mut.resolve_launches(launches);
            // add the active workers to our active map
            self.active.extend(
                workers
                    .into_iter()
                    .map(|worker| (worker.name.clone(), worker)),
            );
        }
    }

    /// Check if we need an update or not and apply it if possible
    ///
    /// If an update is needed, the reactor will first update the Linux and Windows agents,
    /// then update itself and mark itself for shutdown. The reactor must be deployed to
    /// auto-restart (systemd service, K8s, etc.) in order to come back after update.
    #[instrument(name = "Reactor::check_update", skip_all, err(Debug))]
    async fn check_update(&mut self) -> Result<(), Error> {
        // Get the current Thorium version
        let version = self.thorium.updates.get_version().await?;
        // check if the agents need updating
        self.check_update_agents(&version).await?;
        if !self.args.skip_self_update {
            // get the current version
            let current = env!("CARGO_PKG_VERSION");
            // compare to our version and see if its different
            if version.thorium != semver::Version::parse(current)? {
                // set the halt spawning flag so we stop spawning new agents
                self.halt_spawning = true;
                // we need to update the reactor log the version difference
                event!(
                    Level::WARN,
                    reactor = current,
                    api = version.thorium.to_string(),
                    update_needed = true,
                );
                // update ourselves
                self.thorium.updates.update(Component::Reactor).await?;
                // shutdown this reactor; the reactor must deployed with some mechanism to auto-restart
                // in order to continue after update
                self.shutdown = true;
            }
        }
        Ok(())
    }

    /// Check if the agents need an update and apply if no jobs are running
    ///
    /// # Arguments
    ///
    /// * `api_version` - The version returned from the API
    #[instrument(skip_all, fields(api_version = api_version.thorium.to_string()))]
    async fn check_update_agents(&mut self, api_version: &Version) -> Result<(), Error> {
        let agent_path = PathBuf::from(&self.args.agents_dir).join("thorium-agent");
        // check the version of the Linux thorium agent
        let output = tokio::process::Command::new(&agent_path)
            .arg("--version")
            .output()
            .await
            .map_err(|err| {
                Error::with_context(
                    format!(
                        "Error checking version of Thorium agent at '{}'",
                        agent_path.display()
                    ),
                    err,
                )
            })?;
        // make sure the command exited cleanly
        if !output.status.success() {
            // form an error message, either stderr, the exit code, or an unknown error
            let err_msg = (!output.stderr.is_empty())
                .then(|| String::from_utf8(output.stderr).ok())
                .flatten()
                .or_else(|| {
                    output
                        .status
                        .code()
                        .map(|exit_code| format!("exit code {exit_code}"))
                })
                .unwrap_or("An unknown error occurred".to_string());
            return Err(Error::new(format!(
                "Checking for Thorium agent version exited in error: {err_msg}"
            )));
        }
        // get a string from the output
        let version_stdout = String::from_utf8(output.stdout).map_err(|err| {
            Error::with_context(
                "Error checking Thorium agent version: agent command output is not valid UTF-8",
                err,
            )
        })?;
        // extract just the version from the output, which should be the last word
        let version_raw = version_stdout.trim().rsplit(' ').next().ok_or(Error::new(
            "Error checking for Thorium agent version: agent command output is empty!",
        ))?;
        let version = semver::Version::parse(version_raw)
            .map_err(|err| {
                Error::with_context(
                    format!("Error checking Thorium agent version: agent output '{version_raw}' is not a valid semver version"),
                    err
                )
            })?;
        if version != api_version.thorium {
            // we need to update the agents; log the version difference
            event!(
                Level::INFO,
                agent = version_raw,
                api = api_version.thorium.to_string(),
                update_needed = true,
            );
            // update our agents
            self.thorium
                .updates
                .update_specific(
                    Os::Linux,
                    Arch::X86_64,
                    Component::Agent,
                    format!("{}/thorium-agent", self.args.agents_dir),
                )
                .await?;
            // assume the Windows agent also needs updating
            self.thorium
                .updates
                .update_specific(
                    Os::Windows,
                    Arch::X86_64,
                    Component::Agent,
                    format!("{}/thorium-agent.exe", self.args.agents_dir),
                )
                .await?;
        }
        Ok(())
    }

    /// Start polling Thorium for changes to apply to this node
    pub async fn start(mut self) -> Result<(), Error> {
        if !self.args.skip_update {
            // apply any needed updates
            self.check_update().await?;
        }
        // update this nodes resource info
        tasks::update_resources(&self.cluster, &self.name, &self.thorium, &mut self.system).await?;
        let dwell = std::time::Duration::from_secs(self.args.dwell_time.into());
        // loop forever getting the desired state and applying it
        loop {
            // check if we have any tasks that need to be spawned
            self.spawn_tasks().await?;
            // check for changes in this node
            let mut changes = self.poll().await?;
            // make sure our users keys are setup
            self.setup_keys(&mut changes).await;
            // spawn our changes
            self.launch(changes).await;
            // shutdown this reactor if needed
            if self.shutdown {
                break;
            }
            // sleep for the configured dwell between scale attempts
            tokio::time::sleep(dwell).await;
        }
        Ok(())
    }
}
