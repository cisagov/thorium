//! Laucnhes KVM vms for Thorium

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use http::StatusCode;
use thorium::Thorium;
use thorium::models::{ImageScaler, Node, Worker, WorkerDeleteMap, WorkerStatus};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{Level, event, instrument};
use virt::domain::Domain;

use super::Launcher;
use crate::Error;

mod kvm_worker;
mod os_detector;
mod virt_async;

use kvm_worker::{AgentState, Created, KvmWorker, Launched, RunningKvmWorker};
use os_detector::OsDetector;
use virt_async::LibvirtClient;

/// The prefix to prepend to Thorium-managed domains
const WORKER_DOMAIN_NAME_PREFIX: &str = "thorium_";

/// The default timeout for QEMU agent commands in seconds
const QEMU_AGENT_TIMEOUT_SECONDS: i32 = 60;

/// No domain can live for longer than three hours; this is a fail-safe in case of a
/// hanging agent and the scaler fails to deregister a hanging worker for us
pub const DOMAIN_TIMEOUT: Duration = Duration::from_hours(3);

/// A domain where we have lost track of the agent and can't track its
/// progress should be destroyed in this time
pub const UNRESPONSIVE_DOMAIN_TIMEOUT: Duration = Duration::from_mins(10);

/// The maximum number of times we can try to launch a worker
pub const MAX_LAUNCH_ATTEMPTS: u8 = 10;

/// Roughly the amount of time we wait before launching the first time, allowing the domain
/// to boot and settle
const FIRST_LAUNCH_ATTEMPT_DWELL: Duration = Duration::from_mins(2);

/// Roughly the amount of time we wait in between launch attempts
const LAUNCH_ATTEMPT_DWELL: Duration = Duration::from_mins(1);

/// Attempt to proceed the worker to its next state,
/// destroying the worker if an error occurs
macro_rules! destroy_on_err {
    // async case
    ($client:expr, $recv:ident . $method:ident ( $($arg:expr),* $(,)? ) . await ;) => {{
        let old_state = $recv.clone();
        match $recv.$method($($arg),*).await {
            Ok(kvm_worker) => Ok(kvm_worker),
            Err(err) => {
                // first try to destroy the worker
                if let Err(destroy_err) = old_state.destroy($client).await {
                    // we failed to destroy the worker, so log the error
                    event!(Level::ERROR, error = destroy_err.to_string());
                }
                // propagate the original error
                Err(err)
            },
        }
    }};
}

/// A launcher for KVM-based jobs
///
/// The KVM launcher creates a domain for each worker from a golden image disk,
/// transfers the Thorium agent and the user's keys over with the
/// [QEMU guest agent's file functionality](<https://qemu-project.gitlab.io/qemu/interop/qemu-ga-ref.html#command-QGA-qapi-schema.guest-file-open>),
/// then launches the Thorium agent using QEMU guest agent
/// [guest-exec](<https://qemu-project.gitlab.io/qemu/interop/qemu-ga-ref.html#command-QGA-qapi-schema.guest-exec>).
pub struct Kvm {
    /// The kvm specific args
    args: crate::args::Kvm,
    /// An async-friendly libvirt client that avoids blocking the executor
    libvirt_client: LibvirtClient,
    // The KVM workers in flight
    running_workers: HashMap<String, RunningKvmWorker>,
    /// The path to the Thorium agent binaries on disk
    agent_paths: AgentPaths,
    /// A detector that attempts to determine the OS of guests so the correct Thorium agent is
    /// transferred and executed
    ///
    /// The detector contains a [`quick_cache::sync::Cache`] wrapped in an [`std::sync::Arc`] for
    /// cheap cloning. The cache stores OS info for images that haven't changed, avoiding unnecessary I/O.
    /// If there's a cache miss, workers will wait for a single worker to determine the OS, and then
    /// use the cached value.
    os_detector: OsDetector,
}

impl Kvm {
    /// Create a new kvm connector
    ///
    /// # Arguments
    ///
    /// * `args` - The args for the kvm launcher
    #[instrument(name = "Kvm::new", skip_all, err(Debug))]
    pub async fn new(args: crate::args::Kvm, agents_dir: &str) -> Result<Self, Error> {
        // create our libvirt threadpool
        let libvirt_client = LibvirtClient::new(&args.socket, args.virt_threads)?;
        // create our temporary directory in case it doesn't exist
        tokio::fs::create_dir_all(&args.temp_dir)
            .await
            .map_err(|err| {
                Error::with_context(
                    format!(
                        "Error creating temporary directory '{}'",
                        args.temp_dir.display()
                    ),
                    err,
                )
            })?;
        let agent_paths = AgentPaths {
            linux: PathBuf::from(agents_dir).join("thorium-agent"),
            windows: PathBuf::from(agents_dir).join("thorium-agent.exe"),
        };
        agent_paths.check_agents_exist().await?;
        let os_detector = OsDetector::new(args.os_detector_cache_size.into());
        // build our kvm launcher
        Ok(Self {
            args,
            libvirt_client,
            running_workers: HashMap::new(),
            agent_paths,
            os_detector,
        })
    }

    /// Check the status of a running [`KvmWorker`], attempting to launch if
    /// ready or marking for shutdown on exit, error, or timeout
    ///
    /// # Arguments
    ///
    /// * `worker_name` - The name of the worker in Thorium
    /// * `worker` - The worker's data in Thorium
    /// * `shutdowns` - The set of workers marked for shutdown
    /// * `launch_handles` - A set of handles to attempted worker launches
    /// * `worker_delete_map` - A map of workers to manually delete from the API
    /// * `active` - The reactor's master map of active Thorium workers
    #[instrument(skip_all, fields(worker = worker_name))]
    async fn reconcile_running_worker(
        &mut self,
        worker_name: &str,
        worker: &Worker,
        shutdowns: &mut HashSet<String>,
        launch_handles: &mut JoinSet<LaunchResult>,
        worker_delete_map: &mut WorkerDeleteMap,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error> {
        // see if we have a running KVM worker cached for this worker
        match self.running_workers.get_mut(worker_name) {
            // we have a KVM worker, but the agent hasn't been launched yet;
            // check if it's ready to be launched
            Some(RunningKvmWorker::Created(created)) => {
                if created.has_reached_max_attempts() {
                    // this worker has reached the maximum number of launch attempts;
                    // mark it for shutdown
                    event!(
                        Level::ERROR,
                        msg = "Worker has reached maximum attempted launches!",
                        worker = worker_name,
                        max_attepts = MAX_LAUNCH_ATTEMPTS
                    );
                    shutdowns.insert(worker_name.to_string());
                } else if created.ready_to_launch()
                    // gain ownership of the worker to try to launch it
                    && let Some(RunningKvmWorker::Created(created)) =
                        self.running_workers.remove(worker_name)
                {
                    // spawn worker in parallel; we don't need to worry about rate-limiting
                    // because the client threadpool will do that for us
                    let libvirt_client = self.libvirt_client.clone();
                    let worker = worker.clone();
                    let agent_paths = self.agent_paths.clone();
                    launch_handles.spawn(Self::launch_kvm_worker(
                        libvirt_client,
                        worker,
                        created,
                        agent_paths,
                    ));
                }
            }
            Some(RunningKvmWorker::Launched(launched)) => {
                if !launched.is_health_check_due() {
                    // if we're not due for a health check yet, just return early
                    return Ok(());
                }
                let agent_state = launched
                    .agent_state(&self.libvirt_client)
                    .await
                    // if we failed to get the state, assume the agent is missing;
                    // the error was already logged on function return
                    .unwrap_or(AgentState::Missing);
                // schedule the next health check
                launched.schedule_next_health_check();
                match agent_state {
                    AgentState::Missing => {
                        // shut down this unresponsive domain if it's reached the unresponsive timeout
                        if launched.timed_out_unresponsive() {
                            event!(
                                Level::ERROR,
                                msg = "KvmWorker timed out in response time",
                                worker = worker_name,
                                timeout =
                                    format!("{}mins", UNRESPONSIVE_DOMAIN_TIMEOUT.as_secs() / 60)
                            );
                            shutdowns.insert(worker_name.to_string());
                        }
                    }
                    AgentState::Running => {
                        // shut down this domain if it has reached the fail-safe maximum timeout
                        if launched.timed_out() {
                            event!(
                                Level::ERROR,
                                msg = "KvmWorker timed out",
                                worker = worker_name,
                                timeout = format!("{}hrs", DOMAIN_TIMEOUT.as_secs() / 3600)
                            );
                            shutdowns.insert(worker_name.to_string());
                        }
                    }
                    AgentState::Exited => {
                        // the agent exited for some reason; shut down the worker
                        event!(
                            Level::ERROR,
                            msg = "Agent exited with unknown status",
                            worker = worker_name
                        );
                        shutdowns.insert(worker_name.to_string());
                    }
                    AgentState::Errored {
                        exitcode,
                        stdout,
                        stderr,
                    } => {
                        // the agent errored out; shut down the worker
                        event!(
                            Level::ERROR,
                            msg = "Agent errored out!",
                            worker = worker_name,
                            exitcode = exitcode,
                            trimmed_stderr = stderr.as_ref().map(|s| last_chars(s, 100)),
                            trimmed_stdout = stdout.as_ref().map(|s| last_chars(s, 100))
                        );
                        shutdowns.insert(worker_name.to_string());
                    }
                    AgentState::Completed => {
                        // the agent completed successfully but has not yet cleaned up its worker;
                        // just shut it down now and delete the worker manually
                        event!(
                            Level::WARN,
                            msg = "Agent completed without deleting worker",
                            worker = worker_name
                        );
                        shutdowns.insert(worker_name.to_string());
                        worker_delete_map.add_mut(worker_name);
                    }
                }
            }
            None => {
                // this domain isn't in our running KVM worker list;
                // we should reclaim it
                self.reclaim_worker(worker_name, worker, active);
            }
        }
        Ok(())
    }

    /// Reclaim a running worker which we aren't currently tracking
    ///
    /// # Arguments
    ///
    /// * `worker_name` - The name of the worker
    /// * `worker` - The Thorium worker to reclaim
    /// * `active` - The map of active workers for the higher-level reactor to track
    fn reclaim_worker(
        &mut self,
        worker_name: &str,
        worker: &Worker,
        active: &mut HashMap<String, Worker>,
    ) {
        event!(Level::DEBUG, msg = "reclaimed worker", worker = worker_name);
        // add this worker to the reactor's active map if it's not already
        if !active.contains_key(worker_name) {
            active.insert(worker_name.to_string(), worker.clone());
        }
        // create a new KVM worker for this running domain, assuming that it's
        // already been launched
        self.running_workers.insert(
            worker_name.to_string(),
            RunningKvmWorker::Launched(
                KvmWorker::init(worker_name.to_string())
                    .mock_overlay(&self.args)
                    .mock_define()
                    .mock_create()
                    .mock_launch(),
            ),
        );
    }

    /// Launch a [`KvmWorker`]
    ///
    /// # Arguments
    ///
    /// * `client` - A client to libvirt
    /// * `worker` - the Thorium worker to launch
    /// * `kvm_worker` - The created [`KvmWorker`] to attempt to proceed to the [`Launched`] state
    /// * `agent_paths` - The path to the Thorium agent binaries on disk
    #[instrument(skip_all, fields(worker = worker.name))]
    async fn launch_kvm_worker(
        client: LibvirtClient,
        worker: Worker,
        kvm_worker: KvmWorker<Created>,
        agent_paths: AgentPaths,
    ) -> LaunchResult {
        // lookup the KvmWorker's domain
        let domain_name = kvm_worker.domain_name().to_string();
        let domain = match client
            .with_conn(move |conn| Domain::lookup_by_name(conn, &domain_name))
            .await
        {
            Ok(domain) => domain,
            Err(err) => return Err((worker.name.clone(), LaunchError(kvm_worker, err.into()))),
        };
        // attempt to launch the worker
        match kvm_worker
            .launch(&client, &domain, &worker, &agent_paths)
            .await
        {
            Ok(kvm_worker) => Ok((worker.name.clone(), kvm_worker)),
            Err(err) => Err((worker.name.clone(), err)),
        }
    }
}

#[async_trait::async_trait]
impl Launcher for Kvm {
    // The result of launching a Thorium worker is a `KvmWorker` in the `Created` state
    type LaunchedWorkerData = KvmWorker<Created>;

    /// Spawn a worker
    ///
    /// Notably for KVM, we don't actually fully "launch" the worker (run the Thorium agent)
    /// at this point to give the domain time to start. The worker will be fully launched
    /// on future polls of the worker's state (see [`Self::reconcile`]).
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    #[instrument(skip_all, fields(worker = worker.name))]
    async fn launch(
        &self,
        thorium: &Thorium,
        worker: Worker,
    ) -> Result<(Worker, KvmWorker<Created>), Error> {
        // get refs to our args and libvirt client
        let args = &self.args;
        let client = &self.libvirt_client;
        // get the image info for this worker
        let image = thorium.images.get(&worker.group, &worker.stage).await?;
        // initiate a KVM worker
        let kvm_worker = KvmWorker::init(worker.name.clone());
        // overlay the worker
        let kvm_worker =
            destroy_on_err!(client, kvm_worker.overlay(args, &image, &self.os_detector).await;)?;
        // define the worker
        let (kvm_worker, domain) =
            destroy_on_err!(client, kvm_worker.define(client, args, &worker, &image).await;)?;
        // create the worker
        let kvm_worker = destroy_on_err!(client, kvm_worker.create(client, &domain).await;)?;
        // return the worker and the created KVM worker;
        // we'll finish launching the KVM worker later once we give the domain time to settle
        Ok((worker, kvm_worker))
    }

    fn resolve_launches(&mut self, launches: Vec<(&Worker, KvmWorker<Created>)>) {
        // add the created worker to our map
        self.running_workers.extend(
            launches
                .into_iter()
                .map(|(worker, created)| (worker.name.clone(), RunningKvmWorker::Created(created))),
        );
    }

    #[instrument(name = "Kvm::reconcile", skip_all, err(Debug))]
    async fn reconcile(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error> {
        let mut all_workers: HashSet<String> = HashSet::new();
        let mut shutdowns: HashSet<String> = HashSet::new();
        let mut worker_delete_map = WorkerDeleteMap::default();
        let mut launch_handles = JoinSet::new();
        // crawl over all domains on this node
        let flags = virt::sys::VIR_CONNECT_LIST_DOMAINS_INACTIVE
            | virt::sys::VIR_CONNECT_LIST_DOMAINS_ACTIVE;
        for domain in self
            .libvirt_client
            .with_conn(move |conn| conn.list_all_domains(flags))
            .await?
        {
            let domain_name = self
                .libvirt_client
                .with_domain(&domain, virt::domain::Domain::get_name)
                .await?;
            // make sure this is a Thorium-managed domain
            let Some(worker_name) = domain_name.strip_prefix(WORKER_DOMAIN_NAME_PREFIX) else {
                continue;
            };
            // keep track of all workers registered, active or not
            all_workers.insert(worker_name.to_string());
            // check if this domain is even active
            let is_active = self
                .libvirt_client
                .with_domain(&domain, virt::domain::Domain::is_active)
                .await?;
            if !is_active {
                // the domain is not even running, so let's just remove it;
                // we'll recreate one on a future loop if needed
                shutdowns.insert(worker_name.to_string());
                continue;
            }
            // try to get info on this worker from the API state
            let Some(worker) = info.workers.get(worker_name) else {
                // this worker no longer exists in the API; we can safely shut this down
                shutdowns.insert(worker_name.to_string());
                continue;
            };
            // we need to reconcile this running worker's state;
            // ignore errors in doing so; they will be logged, we'll move on to the
            // next worker, and we can try to reconcile this worker again on next poll
            let _ = self
                .reconcile_running_worker(
                    worker_name,
                    worker,
                    &mut shutdowns,
                    &mut launch_handles,
                    &mut worker_delete_map,
                    active,
                )
                .await;
        }
        // check whether everything launched
        for launch_result in launch_handles.join_all().await {
            match launch_result {
                Ok((worker_name, launched)) => {
                    self.running_workers
                        .insert(worker_name, RunningKvmWorker::Launched(launched));
                }
                Err((worker_name, LaunchError(created_worker, err))) => {
                    // we failed to launch this worker; add it back to our list and we'll
                    // try again later; the worker is destroyed after a certain number of attempts
                    event!(
                        Level::WARN,
                        msg = "Error launching worker",
                        worker = worker_name,
                        err = err.to_string()
                    );
                    self.running_workers
                        .insert(worker_name, RunningKvmWorker::Created(created_worker));
                }
            }
        }
        // figure out which workers were missing completely
        let missing = info
            .workers
            .keys()
            .filter(|worker| !all_workers.contains(*worker))
            .collect::<HashSet<_>>();
        // only shutdown workers we've flagged that aren't already set to shutdown;
        // those workers will be shutdown later in the poll loop anyway
        shutdowns.retain(|worker_to_shutdown| {
            info.workers
                .get(worker_to_shutdown)
                .is_none_or(|worker| worker.status != WorkerStatus::Shutdown)
        });
        self.shutdown(thorium, shutdowns.clone()).await?;
        // remove everything we shutdown and whatever was missing from our active map
        active.retain(|worker_name, _| {
            !shutdowns.contains(worker_name) && !missing.contains(worker_name)
        });
        // also remove missing workers from our kvm workers map
        self.running_workers
            .retain(|worker_name, _| !missing.contains(worker_name));
        // only delete Thorium workers whose info we've confirmed is in the API
        // to avoid 404's as far as possible
        worker_delete_map
            .workers
            .retain(|delete_worker| info.workers.contains_key(delete_worker));
        // if we have workers to delete from the API, do that
        if !worker_delete_map.workers.is_empty() &&
            let Err(err) = thorium
                .system
                .delete_workers(ImageScaler::Kvm, &worker_delete_map)
                .await
                // ignore 404 errors – one or more of the workers were deleted as some point;
                // if there was more than one worker to delete, we'll leave some dangling
                // workers in the API that may get spawned again next loop; that's probably fine
                // since something went wrong with these workers in the first place for us to
                // delete them manually, and it may be good to re-run.
                && err
                    .status()
                    .is_none_or(|status| status != StatusCode::NOT_FOUND)
        {
            return Err(Error::from(err));
        }
        Ok(())
    }

    #[instrument(skip_all, fields(num_workers = workers.len()), err(Debug))]
    async fn shutdown(
        &mut self,
        _thorium: &Thorium,
        workers: HashSet<String>,
    ) -> Result<(), Error> {
        // shutdown all workers in parallel
        let mut join_set = JoinSet::new();
        // only shut down up to 10 workers at a time
        let semaphore = Arc::new(Semaphore::new(10));
        for worker in workers {
            // either retrieve the associated KVM worker to destroy or just
            // mock one
            let kvm_worker = match self.running_workers.remove(&worker) {
                Some(kvm_worker) => kvm_worker,
                None => {
                    // log a warning that we're destroying a worker we didn't even have claimed
                    event!(
                        Level::WARN,
                        msg = "Shutting down unclaimed worker",
                        worker = worker
                    );
                    RunningKvmWorker::Launched(
                        KvmWorker::init(worker.clone())
                            .mock_overlay(&self.args)
                            .mock_define()
                            .mock_create()
                            .mock_launch(),
                    )
                }
            };
            // clone a libvirt client for this worker; client is only an Arc wrapper,
            // so cloning is very cheap
            let client = self.libvirt_client.clone();
            let semaphore = semaphore.clone();
            join_set.spawn(async move {
                match semaphore.acquire().await {
                    Ok(_permit) => kvm_worker
                        .destroy(&client)
                        .await
                        // return the worker's name along with the error
                        .map_err(|err| (worker, err)),
                    Err(err) => Err((
                        worker,
                        Error::with_context("Error acquiring semaphore permit", err),
                    )),
                }
            });
        }
        // handle any shutdown errors
        for shutdown_res in join_set.join_all().await {
            if let Err((worker, err)) = shutdown_res {
                event!(
                    Level::ERROR,
                    msg = "Error shutting down KVM worker",
                    worker = worker,
                    err = err.to_string()
                );
            }
        }
        Ok(())
    }
}

/// The error returned from launching, including the worker that attempted
/// to launch
pub struct LaunchError(KvmWorker<Created>, Error);

impl fmt::Debug for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // only format the error part
        fmt::Debug::fmt(&self.1, f)
    }
}

/// The result of launching a KVM worker, either the name of the worker and
/// launched worker or the name of the worker and the launch error that occurred
type LaunchResult = Result<(String, KvmWorker<Launched>), (String, LaunchError)>;

/// The path to the Thorium agent binaries on disk
#[derive(Debug, Clone)]
pub struct AgentPaths {
    /// The path to the linux Thorium agent
    linux: PathBuf,
    /// The path to the Windows Thorium agent
    windows: PathBuf,
}

impl AgentPaths {
    /// Verifies that both of the agent binaries exist where they should
    #[instrument(name = "AgentPaths::check_agents_exist", skip_all, err(Debug))]
    async fn check_agents_exist(&self) -> Result<(), Error> {
        if !tokio::fs::try_exists(&self.linux).await.map_err(|err| {
            Error::with_context(
                format!(
                    "Error checking that Thorium agent binary exists at '{}'",
                    self.linux.display()
                ),
                err,
            )
        })? {
            return Err(Error::new(format!(
                "Thorium agent for Linux is missing at path '{}'",
                self.linux.display()
            )));
        }
        if !tokio::fs::try_exists(&self.windows).await.map_err(|err| {
            Error::with_context(
                format!(
                    "Error checking that Thorium agent binary exists at '{}'",
                    self.windows.display()
                ),
                err,
            )
        })? {
            return Err(Error::new(format!(
                "Thorium agent for Windows is missing at path '{}'",
                self.windows.display()
            )));
        }
        Ok(())
    }
}

/// Return the last `num_chars` from the string `s`
///
/// # Arguments
///
/// * `s` - The string to take chars from
/// * `num_chars` - The number of chars to take
fn last_chars(s: &str, num_chars: usize) -> String {
    s.chars()
        .rev()
        .take(num_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}
