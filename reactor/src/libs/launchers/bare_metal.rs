//! Launches bare metal jobs
use rustix::process::Signal;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use thorium::models::{ArgStrategy, ImageScaler, Node, Worker, WorkerDeleteMap, WorkerStatus};
use thorium::{Error, Thorium};
use tokio::process::{Child, Command};
use tracing::{event, span, Level, Span};

mod cgroups;

use super::Launcher;
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
                )))
            }
        }
    };
    // cast our path to a string
    match path_buf.to_str() {
        Some(path_str) => Ok(path_str.to_owned()),
        None => Err(Error::new(format!(
            "{:#?} cannot be cast to a string",
            path_buf
        ))),
    }
}

/// A currently active bare metal worker
struct ActiveWorker {
    /// The control group this worker is tied too
    cgroup: Cgroup,
    /// The spawned child process if we have one
    child: Option<Child>,
}

impl ActiveWorker {
    /// Spawn a new active worker
    pub async fn new(thorium: &Thorium, worker: &Worker, span: &Span) -> Result<Self, Error> {
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
                    parent: span,
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
        // spawn our agent
        let child = Command::new("/opt/thorium/thorium-agent")
            .args(args)
            .spawn()?;
        // get the pid of the process we just spawned if it has one
        if let Some(pid) = child.id() {
            // add this pid to our cgroup
            cgroup.add(pid)?;
        } else {
            event!(
                parent: span,
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
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    pub fn alive(&mut self, span: &Span) -> Result<bool, Error> {
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
                                event!(
                                    parent: span,
                                    Level::ERROR,
                                    error = true,
                                    error_msg = status.to_string()
                                );
                            }
                            Ok(false)
                        }
                        // this worker has not exited yet
                        None => Ok(true),
                    },
                    Err(err) => {
                        // we failed to get this workers status
                        event!(
                            parent: span,
                            Level::ERROR,
                            error = true,
                            error_msg = err.to_string()
                        );
                        Ok(true)
                    }
                }
            }
            // we don't have a child so check if our cgroup has any pids
            None => Ok(!self.cgroup.procs().is_empty()),
        }
    }

    /// Kill all active processes of this child
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    pub async fn kill(&mut self, span: &Span) -> Result<(), Error> {
        // start our kill worker span
        span!(parent: span, Level::INFO, "Kill Worker");
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
}

impl BareMetal {
    /// Create a new bare metal launcher
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster we are in
    /// * `node` - The node we are on
    pub fn new<C: Into<String>, N: Into<String>>(cluster: C, node: N) -> Self {
        BareMetal {
            cluster: cluster.into(),
            node: node.into(),
            active: HashMap::with_capacity(25),
        }
    }

    /// Helps our launcher check and clean up any active processes
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `active` - The names of the currently active workers in the reactor
    /// * `span` - The span to log traces under
    async fn check_helper(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
        span: &Span,
    ) -> Result<(), Error> {
        // keep a list of workers that should be deleted since they no longer exist
        let mut deletes = WorkerDeleteMap::default();
        // drop any workers that have completed
        self.active.retain(|name, worker| {
            // get whether this worker is alive or not
            let alive = match worker.alive(span) {
                Ok(alive) => alive,
                Err(error) => {
                    // we failed to get whether this worker was alive or not
                    event!(
                        parent: span,
                        Level::ERROR,
                        error = true,
                        error_msg = error.to_string()
                    );
                    // default to this worker still being alive
                    true
                }
            };
            // delete this workers info its not alive
            if !alive {
                // get this workers and add it to our deletes
                if let Some(info) = active.remove(name) {
                    deletes.add_mut(info.name);
                }
                // try to delete this workers cgroup
                if let Err(error) = worker.cgroup.delete() {
                    // we failed to delete a cgroup
                    event!(
                        parent: span,
                        Level::ERROR,
                        error = true,
                        error_msg = error.to_string()
                    );
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
        thorium
            .system
            .delete_workers(ImageScaler::BareMetal, &deletes)
            .await?;
        Ok(())
    }

    /// Helps our launcher recover any existing workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    /// * `span` - The span to log traces under
    fn recover_workers(
        &mut self,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
        span: &Span,
    ) -> Result<(), Error> {
        // whether we have recovered any workers or not
        let recovered = false;
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
                event!(
                    parent: span,
                    Level::INFO,
                    msg = "Recovered worker",
                    name = &name
                );
                // build our recovered worker without a child
                let recovered = ActiveWorker {
                    cgroup,
                    child: None,
                };
                // add this worker to our active map
                self.active.insert(name.clone(), recovered);
            }
        }
        // only update our active map if we recovered some workers
        if recovered {
            // put all our active workers in our reactors active map
            for name in self.active.keys() {
                // get this workers info
                if let Some(worker) = info.workers.remove(name) {
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
    /// * `span` - The span to log traces under
    pub async fn kill(&mut self, name: &str, span: &Span) -> Result<(), Error> {
        // start our cancel worker span
        let span = span!(parent: span, Level::INFO, "Cancel Worker", worker = name);
        // Try to get this workers child
        if let Some(mut worker) = self.active.remove(name) {
            // we only have to cancel if this child is still alive
            if worker.alive(&span)? {
                // this worker is alive so kill all of its processes
                worker.kill(&span).await?;
            }
        }
        Ok(())
    }

    /// Execute the cancel script for a killed worker
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `name` - The name of the worker to run the clean up script for
    /// * `span` - The span to log traces under
    pub async fn cleanup(
        &mut self,
        thorium: &Thorium,
        name: &str,
        span: &Span,
    ) -> Result<(), Error> {
        // start our clean up span
        let span = span!(parent: span, Level::INFO, "Clean Up Worker", worker = name);
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
                    event!(parent: &span, Level::ERROR, exit_code = output.status.to_string());
                    // cast our error to a string
                    match std::str::from_utf8(&output.stderr) {
                        Ok(cast) => {
                            // get the first 512 chars of the error
                            let start = cast.chars().take(512).collect::<String>();
                            // log this error
                            event!(parent: &span, Level::ERROR, error = start);
                        }
                        Err(error) => {
                            event!(parent: &span, Level::ERROR, msg = "Failed to cast stderr to str", error = error.to_string())
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
    /// Spawn a worker and then return a process id that can be used to track it
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    /// * `span` - The span to log traces under
    async fn launch(
        &mut self,
        thorium: &Thorium,
        worker: &Worker,
        span: &Span,
    ) -> Result<(), Error> {
        // start our worker launch span
        let span = span!(
            parent: span,
            Level::INFO,
            "Launching Worker",
            name = worker.name,
            user = worker.user,
            group = worker.group,
            pipeline = worker.pipeline,
            stage = worker.stage
        );
        // start our agent
        let active = ActiveWorker::new(thorium, worker, &span).await?;
        // add this active worker to our map
        self.active.insert(worker.name.clone(), active);
        Ok(())
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
    /// * `span` - The span to log traces under
    async fn check(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
        span: &Span,
    ) -> Result<(), Error> {
        // stat our check active containers span
        let span = span!(parent: span, Level::INFO, "Checking Active Workers");
        // recover any already existing workers
        self.recover_workers(info, active, &span)?;
        // check our currently active workers
        self.check_helper(thorium, info, active, &span).await?;
        Ok(())
    }

    /// Shutdown a list of workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The workers to shutdown
    /// * `span` - The span to log traces under
    async fn shutdown(
        &mut self,
        thorium: &Thorium,
        workers: HashSet<String>,
        span: &Span,
    ) -> Result<(), Error> {
        // start our shutdown workers span
        let span = span!(parent: span, Level::INFO, "Shutdown Workers", workers = workers.len());
        // assume we will delete all requested workers
        let mut deletes = WorkerDeleteMap::with_capacity(workers.len());
        // crawl over the workers we want to shut down
        for worker in workers {
            // try to kill this worker
            match self.kill(&worker, &span).await {
                Ok(_) => {
                    // execute our cancel script
                    match self.cleanup(thorium, &worker, &span).await {
                        // add this worker to our delete map
                        Ok(_) => deletes.add_mut(&worker),
                        // log that we failed to clean up a worker
                        Err(error) => {
                            event!(parent: &span, Level::ERROR, worker = worker, error = error.to_string())
                        }
                    }
                }
                Err(err) => {
                    // log that we failed to shut down a worker
                    event!(parent: &span, Level::ERROR, error = err.to_string());
                }
            }
        }
        // remove this workers from Thorium
        thorium
            .system
            .delete_workers(ImageScaler::BareMetal, &deletes)
            .await?;
        Ok(())
    }
}
