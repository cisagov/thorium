use futures::{poll, task::Poll};
use std::time::Duration;
use thorium::models::{StageLogsAdd, WorkerStatus};
use thorium::Error;
use thorium::Thorium;
use tracing::{event, instrument, span, Level};

use super::agents::{self, Agent};
use super::{CurrentTarget, Lifetime, Target};
use crate::args::Args;

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
                update_needed = true,
                current = current,
                new = version.thorium.to_string()
            );
            // set the halt spawning flag so we stop spawning new agents
            self.halt_claiming = true;
            // only update if we have no active jobs
            if self.target.active.is_some() {
                event!(Level::INFO, msg = "Cannot update with active jobs");
            }
        }
        Ok(())
    }

    /// Claims and executes jobs on a worker
    async fn claim_jobs(&mut self) -> bool {
        // skip any targets with active jobs
        if self.target.active.is_some() || self.lifetime.exceeded() || self.halt_claiming {
            return false;
        }
        // get any jobs if they exist
        let jobs = match self
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
                return false;
            }
        };
        // either execute any claimed jobs or immediately return false if we claimed no jobs
        if !jobs.is_empty() {
            // start our spawn jobs span
            let span = span!(Level::INFO, "Spawning Jobs");
            // crawl over our prefetched jobs and execute them
            for job in jobs.into_iter() {
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
                let job_id = job.id.clone();
                // build the path to write this jobs logs to
                let log_path = format!("/tmp/{}-thorium.log", job.id);
                // build an agent for this job
                match Agent::new(&self, &self.target, job) {
                    // agent successfully built so start executing it
                    Ok(agent) => {
                        // try to spawn this worker
                        let handle =
                            tokio::spawn(async move { agents::execute(agent, log_path).await });
                        // build our active target info
                        let current = CurrentTarget::new(job_id, handle);
                        // set our targets active job
                        self.target.active = Some(current);
                    }
                    // we ran into a problem building our agent
                    Err(error) => {
                        // log this error to our tracer
                        event!(parent: &span, Level::ERROR, error = error.msg());
                        // build the error log to send to Thorium
                        let mut logs = StageLogsAdd::default();
                        logs.add(format!("Spawn Error: {:#?}", error));
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
                        };
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
                    }
                };
            }
        }
        // determine if any jobs were claimed or not
        self.target.active.is_some()
    }

    /// check the process of any active jobs and if necessary continue executing them
    async fn check_jobs(&mut self) {
        // try to get our active jobs handle
        if let Some(mut active) = self.target.active.take() {
            // check if this future has completed
            if let Poll::Ready(join_result) = poll!(&mut active.handle) {
                // fail any jobs where we can't get a join handle to them anymore
                if let Err(error) = join_result {
                    // start our Job execution failure span
                    let span = span!(Level::ERROR, "Job Poll Failure");
                    // log that we failed this job
                    event!(
                        parent: &span,
                        Level::ERROR,
                        user = &self.target.user.username,
                        group = &self.target.group,
                        pipeline = &self.target.pipeline,
                        image = &self.target.stage,
                        job = active.job.to_string(),
                        error = error.to_string()
                    );
                    // add this error to our logs
                    let mut logs = StageLogsAdd::default();
                    logs.add(format!("POLL ERROR: {:#?}", error));
                    // tell Thorium that we failed this job
                    if let Err(error) = self.target.thorium.jobs.error(&active.job, &logs).await {
                        event!(
                            parent: &span,
                            Level::ERROR,
                            msg = "Failed to send stage logs",
                            error = error.msg()
                        );
                    }
                }
            } else {
                // this job hasn't completed so keep tracking it
                let _ = self.target.active.insert(active);
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
        loop {
            // check if any of our spawned jobs have completed
            self.check_jobs().await;
            // apply any needed updates
            self.needs_update().await?;
            // try and claim enough jobs to fill any open job slots
            if !self.claim_jobs().await {
                // check if we have any active jobs or not
                if self.target.active.is_none() {
                    event!(Level::INFO, active = false);
                    break;
                }
            } else {
                // if we claimed a job then skip our sleep
                continue;
            }
            // sleep before trying to claim more jobs or checking our current ones
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // tell Thorium this worker is exiting
        self.target.remove_worker(&self.args).await?;
        Ok(())
    }
}
