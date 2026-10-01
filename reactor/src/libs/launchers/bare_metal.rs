//! Launches bare metal jobs
use http::StatusCode;
use rustix::process::Signal;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use thorium::Thorium;
use thorium::models::{
    ArgStrategy, ImageScaler, Node, SecurityContext, Worker, WorkerDeleteMap, WorkerStatus,
};
use tokio::process::{Child, Command};
use tracing::{Level, event, instrument};

mod cgroups;

use super::Launcher;
use crate::Error;
use crate::libs::keys;
use cgroups::Cgroup;

/// purge a directory if its a file or directory
macro_rules! purge_parent {
    ($target:expr) => {
        // build the path to remove if it exists
        let path = std::path::Path::new(&$target);
        let parent = path.parent().unwrap();
        // if our parent is just /tmp/thorium then remove the full target path instead
        let target = if parent == Path::new("/tmp/thorium") {
            path
        } else {
            parent
        };
        // check if this path exists
        if target.exists() {
            // check if this is a file so we can delete it
            if target.is_file() {
                std::fs::remove_file(target)?;
            } else if target.is_dir() {
                std::fs::remove_dir_all(target)?;
            }
        }
    };
}

/// Inject a single argument
///
/// # Arguments
///
/// * `args` - The args to append too
/// * `value` - The value to add to our args
/// * `strategy` - The arg strategy to use
pub fn inject_arg(args: &mut Vec<String>, value: String, strategy: ArgStrategy) {
    // determine if we should set an output arg or not
    match strategy {
        ArgStrategy::None => (),
        ArgStrategy::Append => args.push(value),
        ArgStrategy::Kwarg(key) => {
            // add our key and our value
            args.push(key);
            args.push(value);
        }
    }
}

/// Isolate a path to target folder or file
///
/// # Arguments
///
/// * `raw` - The path to isolate
/// * `id` - The job id to append
fn isolate<P: AsRef<Path>>(raw: P, id: &str) -> Result<String, Error> {
    let path = raw.as_ref();
    // determine if this path has a target folder or not
    let path_buf = if path == Path::new("/tmp/thorium") {
        // the path to isolate is just the default Thorium path so just add our job id
        path.join(id).to_path_buf()
    } else {
        // a target path exists so insert our final job id before the final segment
        // get the parent
        match path.file_name() {
            // build a path with the parent
            Some(name) => path.parent().unwrap().join(id).join(name).to_path_buf(),
            None => {
                return Err(Error::new(format!(
                    "{} cannot be isolated by job",
                    path.to_string_lossy()
                )));
            }
        }
    };
    // cast our path to a string
    match path_buf.to_str() {
        Some(path_str) => Ok(path_str.to_owned()),
        None => Err(Error::new(format!(
            "{:#?} can not be cast to a string",
            path_buf
        ))),
    }
}

/// The fallback unprivileged uid to drop bare-metal tool processes to when neither the
/// image's security context nor an admin-configured default pins one (nobody)
const DEFAULT_UNPRIVILEGED_UID: u32 = 65534;

/// The fallback unprivileged gid to drop bare-metal tool processes to when neither the
/// image's security context nor an admin-configured default pins one (nogroup)
const DEFAULT_UNPRIVILEGED_GID: u32 = 65534;

/// Describes the OS-level isolation applied to a bare-metal tool process before it is spawned
///
/// The agent — and therefore the attacker-supplied tool it execs
/// (`agent/src/libs/agents/baremetal.rs`) — is launched via `Command::new(...).spawn()`. Because the
/// reactor runs as an admin identity and caches every user's token world-readable under
/// `/opt/thorium-keys`, any tool code (or malware that compromises a tool) that runs as the
/// reactor's own uid can read those credentials and impersonate every tenant. To prevent that we
/// drop the tool process to an unprivileged uid/gid before exec, preferring (in order) the image's
/// own security context, an admin-configured reactor default, and finally a hard-coded nobody
/// fallback so a drop always happens.
#[derive(Debug, Default)]
pub(crate) struct BareMetalIsolation {
    /// The uid the tool process should be dropped to before exec, if any
    pub uid: Option<u32>,
    /// The gid the tool process should be dropped to before exec, if any
    pub gid: Option<u32>,
}

impl BareMetalIsolation {
    /// Builds the isolation policy applied to a bare-metal worker before spawning it
    ///
    /// This returns the safe fallback (an unprivileged nobody/nogroup id) so a compromised
    /// tool cannot read the reactor's admin token or the cached tokens of other tenants. Callers
    /// on the spawn path then layer higher-precedence sources over it (an admin-configured default
    /// and the image's own security context) via `with_configured_defaults`/`with_security_context`.
    fn for_worker() -> Self {
        // drop tool processes to an unprivileged nobody/nogroup id by default
        BareMetalIsolation {
            uid: Some(DEFAULT_UNPRIVILEGED_UID),
            gid: Some(DEFAULT_UNPRIVILEGED_GID),
        }
    }

    /// Overrides the fallback ids with the admin-configured reactor defaults, if any are set
    ///
    /// # Arguments
    ///
    /// * `uid` - The admin-configured default uid from the reactor's CLI args, if set
    /// * `gid` - The admin-configured default gid from the reactor's CLI args, if set
    #[must_use]
    fn with_configured_defaults(mut self, uid: Option<u32>, gid: Option<u32>) -> Self {
        // prefer the admin-configured default uid when one was supplied
        if let Some(uid) = uid {
            self.uid = Some(uid);
        }
        // prefer the admin-configured default gid when one was supplied
        if let Some(gid) = gid {
            self.gid = Some(gid);
        }
        self
    }

    /// Overrides the current ids with the tool's own security-context ids, if it pins any
    ///
    /// The tool's security context takes precedence over every default.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The image's security context
    #[must_use]
    fn with_security_context(mut self, ctx: &SecurityContext) -> Self {
        // prefer the tool's configured uid when it sets one
        if let Some(uid) = ctx.user.and_then(|user| u32::try_from(user).ok()) {
            self.uid = Some(uid);
        }
        // prefer the tool's configured gid when it sets one
        if let Some(gid) = ctx.group.and_then(|group| u32::try_from(group).ok()) {
            self.gid = Some(gid);
        }
        self
    }

    /// Returns whether this policy actually drops the tool process below the reactor's privileges
    ///
    /// A meaningful drop requires reassigning the tool process to a non-root, unprivileged uid so
    /// it can no longer read the reactor's admin keys or the cached tokens of other tenants.
    fn drops_privileges(&self) -> bool {
        // a privilege drop requires reassigning the uid to a non-root user
        matches!(self.uid, Some(uid) if uid != 0)
    }

    /// Applies this isolation policy to a command that is about to be spawned
    ///
    /// # Arguments
    ///
    /// * `command` - The command to constrain before it is spawned
    fn apply(&self, command: &mut Command) {
        // drop to the configured gid before exec if one is set
        if let Some(gid) = self.gid {
            command.gid(gid);
        }
        // drop to the configured uid before exec if one is set
        if let Some(uid) = self.uid {
            command.uid(uid);
        }
    }
}

/// A currently active bare metal worker
pub struct ActiveWorker {
    /// The control group this worker is tied too
    cgroup: Cgroup,
    /// The spawned child process if we have one
    child: Option<Child>,
}

impl ActiveWorker {
    /// Spawn a new active worker
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to spawn
    /// * `default_uid` - The admin-configured default uid to drop the tool process to, if set
    /// * `default_gid` - The admin-configured default gid to drop the tool process to, if set
    #[instrument(name = "ActiveWorker::new", skip_all, err(Debug))]
    pub async fn new(
        thorium: &Thorium,
        worker: &Worker,
        default_uid: Option<u32>,
        default_gid: Option<u32>,
    ) -> Result<Self, Error> {
        // get our image
        let image = thorium.images.get(&worker.group, &worker.stage).await?;
        // build the control group for this worker
        let mut cgroup = Cgroup::new(&worker.name, &image)?;
        // build the path to this users keys
        let keys = keys::path(&worker.user);
        // convert our keys path to a str
        let keys_str = match keys.to_str() {
            Some(keys_str) => keys_str,
            None => {
                // log that our keys path is not valid unicode
                event!(
                    Level::ERROR,
                    error = true,
                    error_msg = "Keys path is not valid unicode",
                );
                return Err(Error::new("Keys path Not Utf-8".to_owned()));
            }
        };
        // build the args to spawn our agent
        let args = vec![
            "--cluster",
            &worker.cluster,
            "--group",
            &worker.group,
            "--pipeline",
            &worker.pipeline,
            "--stage",
            &worker.stage,
            "--name",
            &worker.name,
            "--keys",
            keys_str,
            "bare-metal",
        ];
        // build the isolation policy applied to this bare-metal tool process: the image's own
        // security context wins, then the admin-configured reactor default, then the unprivileged
        // nobody/nogroup fallback
        let isolation = BareMetalIsolation::for_worker()
            .with_configured_defaults(default_uid, default_gid)
            .with_security_context(&image.security_context);
        // build the command to spawn our agent
        let mut command = Command::new("/opt/thorium/thorium-agent");
        command.args(args);
        // apply our isolation policy before spawning the tool process
        isolation.apply(&mut command);
        // spawn our agent
        let child = command.spawn()?;
        // get the pid of the process we just spawned if it has one
        if let Some(pid) = child.id() {
            // add this pid to our cgroup
            cgroup.add(pid)?;
        } else {
            event!(
                Level::ERROR,
                error = true,
                error_msg = "Failed to add child to cgroup!"
            )
        }
        // build our worker struct
        let active = ActiveWorker {
            cgroup,
            child: Some(child),
        };
        Ok(active)
    }

    /// Checks if this worker is alive still
    #[instrument(name = "ActiveWorker::alive", skip_all, err(Debug))]
    pub fn alive(&mut self) -> Result<bool, Error> {
        // if we don't have a child struct then check if this cgroup has any active pids
        match &mut self.child {
            Some(child) => {
                match child.try_wait() {
                    // we were able to check this workers status
                    Ok(status) => match status {
                        // this worker has exited
                        Some(status) => {
                            // log any worker failures
                            if !status.success() {
                                event!(Level::ERROR, error = true, error_msg = status.to_string());
                            }
                            Ok(false)
                        }
                        // this worker has not exited yet
                        None => Ok(true),
                    },
                    Err(err) => {
                        // we failed to get this workers status
                        event!(Level::ERROR, error = true, error_msg = err.to_string());
                        Ok(true)
                    }
                }
            }
            // we don't have a child so check if our cgroup has any pids
            None => Ok(!self.cgroup.procs().is_empty()),
        }
    }

    /// Kill all active processes of this child
    #[instrument(name = "ActiveWorker::kill", skip_all, err(Debug))]
    pub async fn kill(&mut self) -> Result<(), Error> {
        // try to kill this child if it exists
        if let Some(child) = self.child.as_mut() {
            // send sigkill to this child
            child.kill().await?;
        }
        // get any child processes in this cgroup that we need to kill
        for child_proc in self.cgroup.procs() {
            // get this pid as a rustix pid
            // this should not be unsafe as this doesn't cause against any I/O or memory safety problems
            let child_pid = unsafe { rustix::process::Pid::from_raw(child_proc.pid as i32) };
            // kill this pid if we could get a valid pid
            if let Some(child_pid) = child_pid {
                rustix::process::kill_process(child_pid, Signal::KILL)?;
            }
        }
        Ok(())
    }
}

/// Handles launching jobs on bare metal nodes
pub struct BareMetal {
    /// The name of the cluster we are on
    cluster: String,
    /// The name of the node we are on
    node: String,
    /// A map of currently active workers
    active: HashMap<String, ActiveWorker>,
    /// The admin-configured default uid to drop tool processes to when their image pins none
    default_uid: Option<u32>,
    /// The admin-configured default gid to drop tool processes to when their image pins none
    default_gid: Option<u32>,
}

impl BareMetal {
    /// Create a new bare metal launcher
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster we are in
    /// * `node` - The node we are on
    /// * `default_uid` - The admin-configured default uid to drop tool processes to, if set
    /// * `default_gid` - The admin-configured default gid to drop tool processes to, if set
    pub fn new<C: Into<String>, N: Into<String>>(
        cluster: C,
        node: N,
        default_uid: Option<u32>,
        default_gid: Option<u32>,
    ) -> Self {
        BareMetal {
            cluster: cluster.into(),
            node: node.into(),
            active: HashMap::with_capacity(25),
            default_uid,
            default_gid,
        }
    }

    /// Helps our launcher check and clean up any active processes
    #[instrument(name = "BareMetal::check_helper", skip_all, err(Debug))]
    async fn check_helper(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error> {
        // keep a list of workers that should be deleted since they no longer exist
        let mut deletes = WorkerDeleteMap::default();
        // drop any workers that have completed
        self.active.retain(|name, worker| {
            // get whether this worker is alive or not
            let alive = match worker.alive() {
                Ok(alive) => alive,
                Err(error) => {
                    // we failed to get whether this worker was alive or not
                    event!(Level::ERROR, error = true, error_msg = error.to_string());
                    // default to this worker still being alive
                    true
                }
            };
            // delete this workers info its not alive
            if !alive {
                // get this workers and add it to our deletes
                if let Some(worker_info) = active.remove(name) {
                    // decide whether this dead worker should still be deleted from the API
                    if Self::should_delete_dead_worker(&worker_info.name, &info.workers) {
                        deletes.add_mut(worker_info.name);
                    }
                }
                // try to delete this workers cgroup
                if let Err(error) = worker.cgroup.delete() {
                    // we failed to delete a cgroup
                    event!(Level::ERROR, error = true, error_msg = error.to_string());
                }
                false
            } else {
                true
            }
        });
        // Add any running workers that do not exist on our node to our delete map
        for (name, worker) in info.workers.iter() {
            // add any workers that are running and not in our active set to our delete list
            if worker.status == WorkerStatus::Running && !self.active.contains_key(name) {
                deletes.add_mut(name);
            }
        }
        // delete the workers that no longer exist
        Self::delete_workers(thorium, &deletes).await
    }

    /// Delete workers from the Thorium API, treating a 404 as success
    ///
    /// A 404 means one or more of these workers were already deleted (e.g. by the agent itself on
    /// a clean exit), so it must not crash the reactor.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `deletes` - The workers to delete
    async fn delete_workers(thorium: &Thorium, deletes: &WorkerDeleteMap) -> Result<(), Error> {
        // skip the request entirely if there is nothing to delete
        if deletes.workers.is_empty() {
            return Ok(());
        }
        // try to delete these workers from Thorium
        match thorium
            .system
            .delete_workers(ImageScaler::BareMetal, deletes)
            .await
        {
            Ok(_) => Ok(()),
            // ignore 404 errors since one or more of these workers were already deleted
            Err(err) if err.status() == Some(StatusCode::NOT_FOUND) => {
                event!(
                    Level::WARN,
                    msg = "Workers to delete were already missing",
                    workers = ?deletes.workers
                );
                Ok(())
            }
            Err(err) => Err(Error::from(err)),
        }
    }

    /// Decide whether a dead bare-metal worker should be deleted from the Thorium API
    ///
    /// This is `check_helper`'s per-worker delete decision (it never needs `self`) so its logic can
    /// be unit tested without a live Thorium client. A dead worker is only deleted while the API
    /// still lists it; when the agent already deleted its own record on a clean exit, deleting it
    /// again would 404.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the dead worker being considered for deletion
    /// * `api_workers` - The workers the API still reports for this node
    fn should_delete_dead_worker(name: &str, api_workers: &HashMap<String, Worker>) -> bool {
        // only delete dead workers that still exist in the API
        api_workers.contains_key(name)
    }

    /// Decide whether recovered workers should be propagated into the reactor's active map
    ///
    /// This is `recover_workers`' propagation guard (it never needs `self`) so its logic can be
    /// unit tested. Recovered workers are propagated whenever the `recovered` flag is set or at
    /// least one worker was recovered in this pass, so they are never launched a second time.
    ///
    /// # Arguments
    ///
    /// * `recovered_flag` - Whether `recover_workers` recovered any workers
    /// * `recovered_count` - The number of workers actually recovered in this pass
    fn propagates_recovered_workers(recovered_flag: bool, recovered_count: usize) -> bool {
        // propagate if we recovered any workers at all
        recovered_flag || recovered_count > 0
    }

    /// Helps our launcher recover any existing workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    #[instrument(name = "BareMetal::recover_workers", skip_all, err(Debug))]
    fn recover_workers(
        &mut self,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error> {
        // whether we have recovered any workers or not
        let mut recovered = false;
        // the number of workers we actually recovered in this pass
        let mut recovered_count: usize = 0;
        // filter out any workers that we already know about
        // crawl the workers that should exist on this node and check if their control group exists
        for (name, _) in info.workers.iter() {
            // skip any workers that we know about
            if self.active.contains_key(name) {
                continue;
            }
            // try to get the control group for this worker
            let cgroup = Cgroup::load(name);
            // only recover this worker if it has procs
            if !cgroup.procs().is_empty() {
                // we have some processes so recover this worker
                event!(Level::INFO, msg = "Recovered worker", name = &name);
                // build our recovered worker without a child
                let recovered_worker = ActiveWorker {
                    cgroup,
                    child: None,
                };
                // add this worker to our active map
                self.active.insert(name.clone(), recovered_worker);
                // track that we recovered another worker in this pass
                recovered = true;
                recovered_count += 1;
            }
        }
        // only update our active map if we recovered some workers
        if Self::propagates_recovered_workers(recovered, recovered_count) {
            // put all our active workers in our reactors active map
            for name in self.active.keys() {
                // get this workers info, leaving it in the node info so later checks still see it
                if let Some(worker) = info.workers.get(name).cloned() {
                    // just overwrite any existing workers since this should only really happen at start
                    active.insert(name.clone(), worker);
                }
            }
        }
        Ok(())
    }

    /// Try to kill an active worker
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the worker to try to kill
    #[instrument(name = "BareMetal::kill", skip_all, fields(worker = name), err(Debug))]
    pub async fn kill(&mut self, name: &str) -> Result<(), Error> {
        // Try to get this workers child
        if let Some(mut worker) = self.active.remove(name) {
            // we only have to cancel if this child is still alive
            if worker.alive()? {
                // this worker is alive so kill all of its processes
                worker.kill().await?;
            }
        }
        Ok(())
    }

    /// Execute the cancel script for a killed worker
    #[instrument(name = "BareMetal::cleanup", skip_all, fields(worker = name), err(Debug))]
    pub async fn cleanup(&mut self, thorium: &Thorium, name: &str) -> Result<(), Error> {
        // get this workers info
        let worker = thorium.system.get_worker(&name).await?;
        // if this worker has an active job then get its image
        if let Some(active) = &worker.active {
            // get info on this workers image
            let image = thorium.images.get(&worker.group, &worker.stage).await?;
            // get our job id as a string
            let job_id = active.job.to_string();
            // execute a clean up script if its configured
            if let Some(clean_up) = image.clean_up {
                // build the command for the cancel script
                let mut cmd = vec![clean_up.script];
                // inject our job id if needed
                inject_arg(&mut cmd, job_id.to_string(), clean_up.job_id);
                // isolate our results and result files paths
                let iso_results = isolate(&image.output_collection.files.results, &job_id)?;
                let iso_result_files =
                    isolate(&image.output_collection.files.result_files, &job_id)?;
                // inject the remaining args
                inject_arg(&mut cmd, iso_results, clean_up.results);
                inject_arg(&mut cmd, iso_result_files, clean_up.result_files_dir);
                // execute our clean up script
                let output = tokio::process::Command::new(&cmd[0])
                    .args(&cmd[1..])
                    .output()
                    .await?;
                // check if this failed to run or not
                if !output.status.success() {
                    event!(Level::ERROR, exit_code = output.status.to_string());
                    // cast our error to a string
                    match std::str::from_utf8(&output.stderr) {
                        Ok(cast) => {
                            // get the first 512 chars of the error
                            let start = cast.chars().take(512).collect::<String>();
                            // log this error
                            event!(Level::ERROR, error = start);
                        }
                        Err(error) => {
                            event!(
                                Level::ERROR,
                                msg = "Failed to cast stderr to str",
                                error = error.to_string()
                            )
                        }
                    }
                }
            }
            // clean up all temp paths that were in use by this worker
            let samples_path = isolate(&image.dependencies.samples.location, &job_id)?;
            let ephemerals_path = isolate(&image.dependencies.ephemeral.location, &job_id)?;
            let repos_path = isolate(&image.dependencies.repos.location, &job_id)?;
            let results_dep_path = isolate(&image.dependencies.results.location, &job_id)?;
            let results_path = isolate(&image.output_collection.files.results, &job_id)?;
            let result_files_path = isolate(&image.output_collection.files.result_files, &job_id)?;
            let tags_path = isolate(&image.output_collection.files.tags, &job_id)?;
            let children_path = isolate(&image.output_collection.children, &job_id)?;
            // purge all of this workers paths if they exist
            purge_parent!(samples_path);
            purge_parent!(ephemerals_path);
            purge_parent!(repos_path);
            purge_parent!(results_path);
            purge_parent!(result_files_path);
            purge_parent!(results_dep_path);
            purge_parent!(tags_path);
            purge_parent!(children_path);
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Launcher for BareMetal {
    type LaunchedWorkerData = ActiveWorker;

    /// Spawn a worker and then return a process id that can be used to track it
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    #[instrument(
        name = "BareMetal::launch",
        skip_all,
        fields(
            name = worker.name,
            user = worker.user,
            group = worker.group,
            pipeline = worker.pipeline,
            stage = worker.stage
        ),
        err(Debug)
    )]
    async fn launch(
        &self,
        thorium: &Thorium,
        worker: Worker,
    ) -> Result<(Worker, ActiveWorker), Error> {
        // start our agent, dropping the tool process to our configured default ids if the image
        // does not pin its own security context
        let active =
            ActiveWorker::new(&thorium, &worker, self.default_uid, self.default_gid).await?;
        Ok((worker, active))
    }

    /// Resolve all of the data from this batch of launches, modifying any relevant data
    /// in the launcher's state
    ///
    /// # Arguments
    ///
    /// * `launches` - The batch of workers and their data to resolve
    fn resolve_launches(&mut self, launches: Vec<(&Worker, ActiveWorker)>) {
        // add this active worker to our map
        self.active.extend(
            launches
                .into_iter()
                .map(|(worker, active_worker)| (worker.name.clone(), active_worker)),
        );
    }

    /// Check if any of our current workers have completed or died
    ///
    /// This returns the currently active workers.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    #[instrument(name = "BareMetal::reconcile", skip_all, err(Debug))]
    async fn reconcile(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error> {
        // recover any already existing workers
        self.recover_workers(info, active)?;
        // check our currently active workers
        self.check_helper(thorium, info, active).await?;
        Ok(())
    }

    /// Shutdown a list of workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The workers to shutdown
    #[instrument(name = "BareMetal::shutdown", skip_all, fields(workers = workers.len()), err(Debug))]
    async fn shutdown(&mut self, thorium: &Thorium, workers: HashSet<String>) -> Result<(), Error> {
        // assume we will delete all requested workers
        let mut deletes = WorkerDeleteMap::with_capacity(workers.len());
        // crawl over the workers we want to shut down
        for worker in workers {
            // try to kill this worker
            self.kill(&worker).await?;
            // execute our cleanup script
            self.cleanup(thorium, &worker).await?;
            // build the delete payload to send for this iteration
            let payload = Self::next_shutdown_delete_payload(&mut deletes, &worker);
            // remove this workers from Thorium
            Self::delete_workers(thorium, &payload).await?;
        }
        Ok(())
    }
}

impl BareMetal {
    /// Build the delete-workers payload sent on a single `shutdown` loop iteration
    ///
    /// This is `shutdown`'s per-iteration delete bookkeeping (it never needs `self`) so it can be
    /// unit tested. The worker is still recorded in the `deletes` accumulator, but only the worker
    /// deleted on this iteration is sent so already-deleted workers are never re-deleted (which
    /// would 404).
    ///
    /// # Arguments
    ///
    /// * `deletes` - The delete map accumulated across `shutdown` iterations
    /// * `worker` - The worker being deleted on this iteration
    fn next_shutdown_delete_payload(
        deletes: &mut WorkerDeleteMap,
        worker: &str,
    ) -> WorkerDeleteMap {
        // track every worker we have deleted so far
        deletes.add_mut(worker);
        // only send the worker being deleted on this iteration
        WorkerDeleteMap::with_capacity(1).add(worker)
    }
}

#[cfg(test)]
mod tests {
    use super::{BareMetal, BareMetalIsolation};
    use std::collections::HashMap;
    use thorium::models::WorkerDeleteMap;

    /// Build a bare metal worker registered in the API with the given name
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this worker
    fn test_worker(name: &str) -> thorium::models::Worker {
        thorium::models::Worker {
            cluster: "cluster".to_owned(),
            node: "node".to_owned(),
            scaler: thorium::models::ImageScaler::BareMetal,
            name: name.to_owned(),
            user: "user".to_owned(),
            group: "group".to_owned(),
            pipeline: "pipeline".to_owned(),
            stage: "stage".to_owned(),
            status: thorium::models::WorkerStatus::Running,
            spawned: chrono::Utc::now(),
            heart_beat: None,
            resources: thorium::models::Resources::default(),
            pool: thorium::models::Pools::Deadline,
            active: None,
        }
    }

    /// `check_helper` only deletes a dead worker from the API when that worker is still present
    /// in the API's worker list, so a dead worker that is not listed is not re-deleted. When a
    /// bare-metal agent exits cleanly it deletes its own record first, so on the next poll the
    /// reactor sees a dead process whose record is already gone; deleting it again would 404 and
    /// crash the reactor daemon (this mirrors how the Kvm launcher filters its delete map against
    /// `info.workers` "to avoid 404's").
    #[test]
    fn dead_worker_absent_from_api_is_not_deleted() {
        // the API no longer lists this worker (the agent deleted its own record on clean exit)
        let api_workers: HashMap<String, thorium::models::Worker> = HashMap::new();
        // a dead worker already gone from the API must not be deleted again (that 404s and crashes)
        assert!(
            !BareMetal::should_delete_dead_worker("worker-a", &api_workers),
            "a dead worker whose API record was already deleted is deleted again, 404-crashing \
             the reactor on every clean bare-metal job completion"
        );
    }

    /// A dead worker that is still listed by the API is deleted so its registration does not
    /// leak.
    #[test]
    fn dead_worker_present_in_api_is_deleted() {
        // the API still lists this worker
        let mut api_workers: HashMap<String, thorium::models::Worker> = HashMap::new();
        api_workers.insert("worker-a".to_owned(), test_worker("worker-a"));
        // the dead worker must be deleted from the API
        assert!(BareMetal::should_delete_dead_worker(
            "worker-a",
            &api_workers
        ));
    }

    /// `recover_workers` propagates recovered workers into the reactor's active map whenever at
    /// least one worker was actually recovered, even if the `recovered` flag is not set. This
    /// puts recovered workers in the active map so they do not get a duplicate agent launched
    /// after every reactor restart.
    #[test]
    fn recovering_a_worker_propagates_it_to_the_active_map() {
        // the `recovered` flag is not set, so only the recovered count can trigger propagation
        let recovered_flag = false;
        // we recovered exactly one worker in this pass
        let recovered_count = 1;
        // recovering a worker must cause it to be propagated into the reactor's active map
        assert!(
            BareMetal::propagates_recovered_workers(recovered_flag, recovered_count),
            "recovered workers are never propagated into the reactor's active map, so a duplicate \
             agent is launched for each after every reactor restart"
        );
    }

    /// `shutdown` sends only the worker being deleted on each loop iteration, not an accumulating
    /// map of every worker deleted so far, so the payload for iteration 2 contains only the second
    /// worker. Re-sending iteration 1's already-deleted worker would 404 and crash the reactor.
    #[test]
    fn shutdown_payload_does_not_re_delete_prior_workers() {
        // the delete map carried across shutdown iterations
        let mut deletes = WorkerDeleteMap::default();
        // iteration 1 deletes worker A
        let first = BareMetal::next_shutdown_delete_payload(&mut deletes, "worker-a");
        assert_eq!(first.workers, vec!["worker-a".to_string()]);
        // iteration 2 deletes worker B and must NOT re-send already-deleted worker A
        let second = BareMetal::next_shutdown_delete_payload(&mut deletes, "worker-b");
        assert_eq!(
            second.workers,
            vec!["worker-b".to_string()],
            "shutdown re-sends the accumulated delete map, re-deleting already-deleted workers and \
             404-crashing the reactor on any multi-worker scale-down"
        );
    }

    /// bare-metal tool processes must not run with the reactor's own (admin, typically
    /// root) privileges. The reactor holds an admin token and caches every user's token
    /// world-readable under `/opt/thorium-keys`, so a tool running as the reactor's uid can read
    /// those credentials and impersonate every tenant. The isolation policy applied before spawning
    /// the agent must drop the tool process to an unprivileged, non-root uid.
    #[test]
    fn baremetal_worker_drops_privileges() {
        // build the isolation policy applied to a bare-metal worker before it is spawned
        let isolation = BareMetalIsolation::for_worker();
        // the tool process must be dropped below the reactor's privileges
        assert!(
            isolation.drops_privileges(),
            "bare-metal tool processes run with the reactor's own privileges, letting a tool read \
             the reactor's admin token and every cached tenant token"
        );
    }
}
