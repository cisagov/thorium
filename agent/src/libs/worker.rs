use std::time::Duration;
use thorium::Error;
use thorium::Thorium;
use thorium::models::{StageLogsAdd, WorkerStatus};
use tokio::task::JoinHandle;
use tracing::{Level, event, instrument, span};
use uuid::Uuid;

use super::agents::{self, Agent};
use super::{Lifetime, Target};
use crate::args::Args;

/// Whether this worker should exit or look for more jobs
pub enum ClaimJobStatus {
    /// The active job this worker claimed
    ActiveJob {
        job_id: Uuid,
        handle: JoinHandle<()>,
    },
    /// This worker did not claim a job but does not have to immeidately exit
    DidNotClaim,
    /// This worker needs to exit as it needs to update or has exceeded its lifetime
    ExitWhenPossible,
}

/// A worker used to execute jobs in Thorium
pub struct Worker {
    /// A client for Thorium
    pub thorium: Thorium,
    /// The target job types to claim
    pub target: Target,
    /// The command line args passed to the agent
    pub args: Args,
    /// The node this worker is on
    pub node: String,
    /// This workers lifetime
    pub lifetime: Lifetime,
    /// Stop claiming new jobs as an update is needed
    pub halt_claiming: bool,
}

impl Worker {
    /// Build a new worker
    ///
    /// # Arguments
    ///
    /// * `args` - Arguments passed to the agent
    #[instrument(name = "Worker::new", skip_all, err(Debug))]
    pub async fn new(args: Args) -> Result<Self, Error> {
        // load our Thorium client
        let thorium = Thorium::from_key_file(&args.keys).await?;
        // get the targets for this image
        let target = args.target(&thorium).await?;
        // set up lifetime
        let lifetime = Lifetime::new(&target);
        // get the node we are running on
        let node = args.node()?;
        // build our worker
        let worker = Worker {
            thorium,
            target,
            args,
            node,
            lifetime,
            halt_claiming: false,
        };
        Ok(worker)
    }

    /// Check if we need an update or not and apply it if possible
    #[instrument(name = "Worker::needs_update", skip_all, err(Debug))]
    async fn needs_update(&mut self) -> Result<(), Error> {
        // Get the current Thorium version
        let version = self.thorium.updates.get_version().await?;
        // get the current version
        let current = env!("CARGO_PKG_VERSION");
        // compare to our version and see if its different
        if version.thorium != semver::Version::parse(current)? {
            // start our update needed span
            event!(
                Level::INFO,
                update_neede = true,
                current = current,
                new = version.thorium.to_string()
            );
            // set the halt spawning flag so we stop spawning new agents
            self.halt_claiming = true;
        }
        Ok(())
    }

    /// Claims and executes jobs on a worker
    async fn claim_jobs(&mut self) -> ClaimJobStatus {
        // if we have exceeded our lifetime or need to halt claiming then exit when possible
        if self.lifetime.exceeded() || self.halt_claiming {
            return ClaimJobStatus::ExitWhenPossible;
        }
        // get any jobs if they exist
        let mut jobs = match self
            .target
            .thorium
            .jobs
            .claim(
                &self.target.group,
                &self.target.pipeline,
                &self.target.stage,
                &self.args.cluster,
                &self.node,
                &self.target.name,
                1,
            )
            .await
        {
            Ok(jobs) => jobs,
            Err(error) => {
                // start our jobs claim error span
                span!(
                    Level::ERROR,
                    "Failed To Claim Jobs",
                    user = self.target.user.username,
                    group = self.target.group,
                    pipeline = self.target.pipeline,
                    image = self.target.stage,
                    name = self.target.name,
                    error = error.msg()
                );
                // return false since we didn't claim any jobs
                return ClaimJobStatus::DidNotClaim;
            }
        };
        // agents will only ever execute a single job or less
        debug_assert!(jobs.len() <= 1);
        // either execute any claimed jobs or immediately return false if we claimed no jobs
        match jobs.pop() {
            Some(job) => {
                // start our spawn jobs span
                let span = span!(Level::INFO, "Spawning Job");
                // log this job is going to be spawned
                event!(
                    parent: &span,
                    Level::INFO,
                    reaction = job.reaction.to_string(),
                    job = job.id.to_string(),
                    user = self.target.user.username,
                    group = self.target.group,
                    pipeline = self.target.pipeline,
                    image = self.target.stage,
                    name = self.target.name,
                );
                // increment our job counter
                self.lifetime.claimed_job();
                // get this jobs reaction and job id
                let job_id = job.id;
                // build the path to write this jobs logs to
                let log_path = format!("/tmp/{}-thorium.log", job.id);
                // build an agent for this job
                match Agent::new(self, &self.target, job) {
                    // agent successfully built so start executing it
                    Ok(agent) => {
                        // try to spawn this worker
                        let handle =
                            tokio::spawn(async move { agents::execute(agent, log_path).await });
                        ClaimJobStatus::ActiveJob { job_id, handle }
                    }
                    // we ran into a problem building our agent
                    Err(error) => {
                        // log this error to our tracer
                        event!(parent: &span, Level::ERROR, error = error.msg());
                        // build the error log to send to Thorium
                        let mut logs = StageLogsAdd::default();
                        logs.add(format!("Spawn Error: {error:#?}"));
                        // send our error logs to Thorium
                        if let Err(error) = self
                            .target
                            .thorium
                            .jobs
                            .error(&job_id, &StageLogsAdd::default())
                            .await
                        {
                            // log that we failed to update our stage logs in thorium
                            event!(
                                parent: &span,
                                Level::ERROR,
                                msg = "Failed to send stage logs",
                                error = error.msg()
                            );
                        }
                        // delete this log file
                        if let Err(error) = tokio::fs::remove_file(log_path).await {
                            // log this error to our tracer
                            event!(
                                parent: &span,
                                Level::ERROR,
                                msg = "Failed to remove log file",
                                error = error.to_string()
                            );
                        }
                        // We ran into a problem and didn't claim a job
                        ClaimJobStatus::DidNotClaim
                    }
                }
            }
            // no job was claimed
            None => ClaimJobStatus::DidNotClaim,
        }
    }

    /// check the process of any active jobs and if necessary continue executing them
    ///
    /// # Arguments
    ///
    /// * `job_id` - The id for the job to wait for
    /// * `handle` - The handle to the task that is executing this job
    ///
    /// # Returns
    ///
    /// Returns true if this job completes successfully and false if it failed.
    #[instrument(name = "Worker::wait_for_job", skip_all, fields(job = job_id.to_string()))]
    async fn await_job_completion(&mut self, job_id: Uuid, handle: JoinHandle<()>) -> bool {
        // wait for our job to complete
        match handle.await {
            Ok(()) => {
                // log that our job completed
                event!(Level::INFO, status = "Completed");
                // return true that our job didn't fail
                true
            }
            Err(error) => {
                // log that we failed this job
                event!(
                    Level::ERROR,
                    user = &self.target.user.username,
                    group = &self.target.group,
                    pipeline = &self.target.pipeline,
                    image = &self.target.stage,
                    error = error.to_string()
                );
                // return false since our job ran into an external error and we should exit
                false
            }
        }
    }

    /// Starts the worker loop
    #[instrument(name = "Worker::start", skip_all, err(Debug))]
    pub async fn start(&mut self) -> Result<(), Error> {
        // apply any needed updates
        self.needs_update().await?;
        // tell Thorium we are running
        self.target.update_worker(WorkerStatus::Running).await?;
        // track how long this work should sit in limbo before exiting without a job to claim
        let mut limbo = self.args.limbo;
        loop {
            // apply any needed updates
            self.needs_update().await?;
            // try and claim enough jobs to fill any open job slots
            match self.claim_jobs().await {
                // we have an active job so wait 25ms before checking if this job finished yet
                ClaimJobStatus::ActiveJob { job_id, handle } => {
                    // reset our limbo
                    limbo = self.args.limbo;
                    // block until our active job completes
                    if !self.await_job_completion(job_id, handle).await {
                        break;
                    }
                }
                // we did not claim a job so check if we have exceeded our limbo
                ClaimJobStatus::DidNotClaim => {
                    // log how much limbo we have left
                    event!(Level::INFO, limbo_left = limbo);
                    // if we have no more limbo left then exit
                    if limbo == 0 {
                        break;
                    }
                    // otherwise decrement our limbo
                    limbo -= 1;
                    // sleep for 1 second before looking for another job
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                ClaimJobStatus::ExitWhenPossible => break,
            }
        }
        // tell Thorium this worker is exiting
        self.target.remove_worker(&self.args).await?;
        Ok(())
    }
}
