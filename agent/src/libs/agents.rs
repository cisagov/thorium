//! A trait for executing a Thorium job in a specific environment

use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use thorium::models::{GenericJob, Image, StageLogsAdd};
use thorium::{Error, Thorium};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::time::{Duration, Instant};
use tracing::{event, instrument, Level};

pub mod baremetal;
pub mod k8s;
mod registry;
mod setup;

pub use baremetal::BareMetal;
pub use k8s::K8s;

use crate::args::Envs;
use crate::libs::children::Children;
use crate::libs::{results, tags, Target};
use crate::{from_now, log, Worker};

use super::results::RawResults;
use super::tags::TagBundle;

// log at most .10 mebibytes
const MAX_LOG: usize = 104_858;
const MAX_BATCHES: usize = 10;

/// Check if a subprocess child has completed or not
///
/// # Arguments
///
/// * `child` - The child to check against
async fn check_child(child: &mut Child) -> Result<JobStatus, Error> {
    // check if this sub process has finished yet
    match child.try_wait() {
        // get our exit code on macOS
        #[cfg(target_os = "macos")]
        Ok(Some(status)) => {
            // get and set the return code
            let code = match status.code() {
                Some(code) => Some(code),
                // the proc was killed by a signal so assume we failed
                None => Some(-1),
            };
            // check if an error occurred or not
            if code == Some(0) {
                Ok(JobStatus::Finished(code))
            } else {
                Ok(JobStatus::Failed(code))
            }
        }
        // get our exit code on linux
        #[cfg(target_os = "linux")]
        Ok(Some(status)) => {
            // get and set the return code
            let code = match status.code() {
                Some(code) => Some(code),
                // the proc was killed by a signal
                None => status.signal(),
            };
            // check if an error occurred or not
            if code == Some(0) {
                Ok(JobStatus::Finished(code))
            } else {
                Ok(JobStatus::Failed(code))
            }
        }
        // get our exit code on windows
        #[cfg(target_os = "windows")]
        Ok(Some(status)) => {
            // get our exit code
            let code = status.code();
            // check if an error occurred or not
            if code == Some(0) {
                Ok(JobStatus::Finished(code))
            } else {
                Ok(JobStatus::Failed(code))
            }
        }
        // this job is not done yet so sleep for 100ms
        Ok(None) => {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(JobStatus::OnGoing)
        }
        // something went wrong with this job
        Err(e) => Err(Error::from(e)),
    }
}

/// The different types of in flight jobs being executed by a Thorium agent
pub enum JobStatus {
    /// This job has finished executing
    Finished(Option<i32>),
    /// This job failed
    Failed(Option<i32>),
    /// This job is ongoing
    OnGoing,
}

/// The different types of in flight jobs being executed by a Thorium agent
pub enum InFlight {
    /// A subprocess command child currently being executed
    Child(Child),
}

impl InFlight {
    /// Check if this in flight job has completed or not
    pub async fn finished(&mut self) -> Result<JobStatus, Error> {
        // check if our job has finished yet
        match self {
            InFlight::Child(child) => check_child(child).await,
        }
    }

    /// Cancel this in flight job
    pub async fn cancel(&mut self) -> Result<(), Error> {
        // cancel our in flight job
        match self {
            InFlight::Child(child) => Ok(child.kill().await?),
        }
    }
}

fn get_executor(
    worker: &Worker,
    target: &Target,
    job: &GenericJob,
    sender: &Sender<String>,
) -> Result<Box<dyn AgentExecutor + Send + Sync>, Error> {
    // instance the correct agent
    match &worker.args.env {
        Envs::K8s(args) => Ok(Box::new(K8s::new(args, target, sender.clone())?)),
        Envs::BareMetal(_) => Ok(Box::new(BareMetal::new(target, job, sender.clone())?)),
        // we can use the k8s executor for containers on window
        Envs::Windows(args) => Ok(Box::new(K8s::from_windows(args, target, sender.clone())?)),
        // we can use the k8x executor for kvm vms
        Envs::Kvm(_) => Ok(Box::new(K8s::from_kvm(target, sender.clone())?)),
    }
}

pub struct Agent {
    /// A client to Thorium
    pub thorium: Thorium,
    /// The image we are executing a job for
    pub image: Image,
    /// The Job this results_path agent is executing
    pub job: GenericJob,
    /// The stage logs to send to Thorium
    stage_logs: StageLogsAdd,
    /// A receiver for a channel of logs to add for this job
    receiver: Receiver<String>,
    /// A sender for a channel of logs to add for this job
    pub sender: Sender<String>,
    /// The agent executor this agent is using
    pub executor: Box<dyn AgentExecutor + Send + Sync>,
    /// Whether this job should be failed or not
    pub completed: bool,
    /// How long this job took to run
    pub runtime: Option<u64>,
    /// A map of repos to their checked out commits
    commits: HashMap<String, String>,
}

impl Agent {
    /// Creates a new instance of the right agent
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker that will be executing our jobs
    /// * `target` - The target job types for this worker
    /// * `job` - The job to execute
    pub fn new(worker: &Worker, target: &Target, job: GenericJob) -> Result<Self, Error> {
        // create a channel to send logs to this agent
        let (sender, receiver) = crossbeam::channel::unbounded();
        // instance our executor
        let executor = get_executor(worker, target, &job, &sender)?;
        let agent = Agent {
            thorium: worker.thorium.clone(),
            image: target.image.clone(),
            job,
            stage_logs: StageLogsAdd::default(),
            receiver,
            sender,
            executor,
            completed: false,
            runtime: None,
            commits: HashMap::default(),
        };
        Ok(agent)
    }

    /// Send any logs in our channel to Thorium
    pub async fn send_channel_logs(&mut self) -> Result<(), Error> {
        // track how much data we are sending in this logs request
        let mut size = 0;
        // track how many full batches we are sending in this loop
        let mut batches_sent = 0;
        // consume everything in our channel and add it to our logs object
        for line in self.receiver.try_iter() {
            // add this lines length to our total log size
            size += line.len();
            // add this log to our logs to send to Thorium
            self.stage_logs.add(line);
            // if we are above our max log length then send our current logs
            if size >= MAX_LOG {
                // send the logs we have currently buffered
                self.thorium
                    .reactions
                    .add_stage_logs(
                        &self.job.group,
                        &self.job.reaction,
                        &self.job.stage,
                        &self.stage_logs,
                    )
                    .await?;
                // empty our stage logs
                self.stage_logs.logs.truncate(0);
                // increment how many batches are sent
                batches_sent += 1;
                // if we have sent our max number of batches then stop sending logs for a bit
                if batches_sent >= MAX_BATCHES {
                    return Ok(());
                }
            }
        }
        // if any logs still need to be sent then send them
        if !self.stage_logs.logs.is_empty() {
            // send the logs we have currently buffered
            self.thorium
                .reactions
                .add_stage_logs(
                    &self.job.group,
                    &self.job.reaction,
                    &self.job.stage,
                    &self.stage_logs,
                )
                .await?;
            // empty our stage logs
            self.stage_logs.logs.truncate(0);
            self.stage_logs.logs.shrink_to_fit();
        }
        Ok(())
    }

    /// Send any logs in our log file to Thorium
    pub async fn send_file_logs(&mut self, reader: &mut BufReader<File>) -> Result<(), Error> {
        // track how much data we are sending in this logs request
        let mut size = 0;
        // track how many full batches we are sending in this loop
        let mut batches_sent = 0;
        // get the current lines from our log file
        let mut lines = reader.lines();
        // consume any valid lines and send our log file to Thorium
        while let Ok(Some(line)) = lines.next_line().await {
            // add this lines length to our total log size
            size += line.len();
            // add this log to our logs to send to Thorium
            self.stage_logs.add(line);
            // if we are above our max log length then send our current logs
            if size >= MAX_LOG {
                // send the logs we have currently buffered
                self.thorium
                    .reactions
                    .add_stage_logs(
                        &self.job.group,
                        &self.job.reaction,
                        &self.job.stage,
                        &self.stage_logs,
                    )
                    .await?;
                // empty our stage logs
                self.stage_logs.logs.truncate(0);
                // increment how many batches are sent
                batches_sent += 1;
                // if we have sent our max number of batches then stop sending logs for a bit
                if batches_sent >= MAX_BATCHES {
                    return Ok(());
                }
            }
        }
        // if any logs still need to be sent then send them
        if !self.stage_logs.logs.is_empty() {
            // send the logs we have currently buffered
            self.thorium
                .reactions
                .add_stage_logs(
                    &self.job.group,
                    &self.job.reaction,
                    &self.job.stage,
                    &self.stage_logs,
                )
                .await?;
            // empty our stage logs
            self.stage_logs.logs.truncate(0);
            self.stage_logs.logs.shrink_to_fit();
        }
        Ok(())
    }

    /// Wait for a job to finish executing
    ///
    /// # Arguments
    ///
    /// * `in_flight`: The info about our active job
    /// * `reader` - The reader to pull logs from
    #[instrument(name = "agents::monitor", skip_all, err(Debug))]
    async fn monitor(
        &mut self,
        mut in_flight: InFlight,
        reader: &mut BufReader<File>,
    ) -> Result<JobStatus, Error> {
        // get timestamps to track how long this job has been running for
        let start = Instant::now();
        // get time job should be killed at if we have a timeout set
        let timeout = self.image.timeout.map(|seconds| from_now!(start, seconds));
        // get the duration to sleep between checks
        let sleep = Duration::from_millis(100);
        // wait for this job to finish executing
        loop {
            // send any logs in our log file
            self.send_file_logs(reader).await?;
            // check if this job has finished executing or not yet
            match in_flight.finished().await? {
                JobStatus::Finished(code) => {
                    // get the amount of time this job took to run
                    let runtime = Instant::now() - start;
                    self.runtime = Some(runtime.as_secs());
                    // log our job finished
                    event!(Level::INFO, msg = "Job Finished", code = code);
                    return Ok(JobStatus::Finished(code));
                }
                JobStatus::Failed(code) => {
                    // log our job failed
                    event!(Level::INFO, msg = "Job Failed", code = code);
                    return Ok(JobStatus::Failed(code));
                }
                JobStatus::OnGoing => (),
            };
            // check if we this job should be timed out or not
            if timeout.is_some() && Some(Instant::now()) > timeout {
                // log this timeout
                event!(Level::INFO, msg = "Job timed out");
                self.stage_logs
                    .add("Execution time limit exceeded".to_string());
                // cancel this job
                in_flight.cancel().await?;
                return Ok(JobStatus::Failed(None));
            }
            // sleep for 100 ms before checking if this job has finished  again
            tokio::time::sleep(sleep).await;
        }
    }

    /// Tell Thorium this job completed and to proceed
    #[instrument(name = "agents::proceed", skip_all, err(Debug))]
    pub async fn proceed(&self) -> Result<(), Error> {
        // if  a runtime was not present then fail this job
        match self.runtime {
            Some(runtime) => {
                self.thorium
                    .jobs
                    .proceed(&self.job, &self.stage_logs, runtime)
                    .await?
            }
            None => {
                self.thorium
                    .jobs
                    .error(&self.job.id, &self.stage_logs)
                    .await?
            }
        };
        Ok(())
    }

    /// Tell Thorium this job failed with an error message
    ///
    /// # Arguments
    ///
    /// * `reader` - The reader to pull file based logs from
    /// * `err` - The error to report back to the API
    #[instrument(name = "agents::error", skip(self, reader), err(Debug))]
    pub async fn error(&mut self, reader: &mut BufReader<File>, err: &Error) -> Result<(), Error> {
        // send any logs in our logs channel
        self.send_channel_logs().await?;
        // send any remaining logs from our log file
        self.send_file_logs(reader).await?;
        // add this error message to our log channel
        self.stage_logs.add(format!("Error: {err}"));
        self.thorium
            .jobs
            .error(&self.job.id, &self.stage_logs)
            .await?;
        Ok(())
    }

    /// Tell Thorium this job failed with an error message and ignore any file errors
    ///
    /// # Arguments
    ///
    /// * `err` - The error to report back to the API
    pub async fn error_channel_only(&mut self, err: &Error) -> Result<(), Error> {
        // send any logs in our logs channel
        self.send_channel_logs().await?;
        // add this error message to our log channel
        self.stage_logs.add(format!("Error: {err}"));
        self.thorium
            .jobs
            .error(&self.job.id, &self.stage_logs)
            .await?;
        Ok(())
    }
}

/// The methods used to launch and monitor a job in Thorium
#[async_trait::async_trait]
pub trait AgentExecutor {
    /// Get the paths to this executors current jobs results and result files
    fn result_paths(&self, image: &Image) -> (String, String);

    /// Set up the environment for executing a single job in Thorium
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to set up a job for
    /// * `job` - The job we are setting up for
    /// * `commits` - The commit that each repo is checked out too
    async fn setup(
        &mut self,
        image: &Image,
        job: &GenericJob,
        commits: &mut HashMap<String, String>,
    ) -> Result<(), Error>;

    /// Execute a single job in Thorium
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to execute a job for
    /// * `job` - The job we are executing
    /// * `log_file` - Where to write job logs to
    async fn execute(
        &mut self,
        image: &Image,
        job: &GenericJob,
        log_file: &str,
    ) -> Result<InFlight, Error>;

    /// Collect any result from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect results for
    async fn results(&mut self, image: &Image) -> Result<RawResults, Error>;

    /// Collect any tags from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect tags for
    /// * `job` - The job we are executing
    /// * `outputs` - The results to extract tags from
    async fn tags(
        &mut self,
        image: &Image,
        job: &GenericJob,
        outputs: &RawResults,
    ) -> Result<TagBundle, Error>;

    /// Collect any children files from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect children for
    async fn children(&mut self, image: &Image) -> Result<Children, Error>;

    /// Clean up after this job
    async fn clean_up(&mut self, image: &Image, job: &GenericJob) -> Result<(), Error>;
}

// checks if any action failed and logs its
#[macro_export]
macro_rules! check {
    ($fut:expr) => {
        match $fut {
            Ok(_) => (),
            Err(error) => tracing::event!(tracing::Level::ERROR, error = error.to_string()),
        }
    };
}

/// Create the log file for this job
///
/// # Arguments
///
/// * `log_path`
#[instrument(name = "agents::create_log_file", skip_all, err(Debug))]
async fn create_log_file<P: AsRef<Path>>(log_path: P) -> Result<File, Error> {
    // get our log path as a path
    let path = log_path.as_ref();
    // get our parent path if one exists
    if let Some(parent) = path.parent() {
        // create any required parent dirs
        tokio::fs::create_dir_all(parent).await?;
    }
    // delete our log file just in case it already exists for some reason
    if path.exists() {
        tokio::fs::remove_file(&path).await?;
    }
    // create a file to buffer our logs in
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .await?;
    Ok(file)
}

/// Executes a claimed job with the correct agent
///
/// # Arguments
///
/// * `agent` - The agent that is executing this job
/// * `log_path` - Where to write log files too
/// * `span` - The span to log traces under
#[instrument(name = "agents::sub_execute", skip(agent, reader), err(Debug))]
pub async fn sub_execute(
    agent: &mut Agent,
    reader: &mut BufReader<File>,
    log_path: &String,
) -> Result<(), Error> {
    // setup to execute this job
    agent
        .executor
        .setup(&agent.image, &agent.job, &mut agent.commits)
        .await?;
    // start executing this job
    let in_flight = agent
        .executor
        .execute(&agent.image, &agent.job, log_path)
        .await?;
    // send any logs in our logs channel
    agent.send_channel_logs().await?;
    // wait for this job to finish executing
    let status = agent.monitor(in_flight, reader).await?;
    // send any remaining logs from our log file
    agent.send_file_logs(reader).await?;
    // if this job finished successfully then look for results
    let code = match status {
        // this job successfully completed its job
        JobStatus::Finished(code) => {
            // collect any results from this job
            let raw_results = agent.executor.results(&agent.image).await?;
            // collect any tags from our results or disk
            let tag_bundle = agent
                .executor
                .tags(&agent.image, &agent.job, &raw_results)
                .await?;
            // send any logs in our logs channel
            agent.send_channel_logs().await?;
            // submit our results and tags
            let results = results::submit(
                &agent.thorium,
                &raw_results,
                &agent.job,
                &agent.image,
                &mut agent.sender,
            )
            .await?;
            // send any logs in our logs channel
            agent.send_channel_logs().await?;
            // submit any tags we found
            tags::submit(&agent.thorium, tag_bundle, &agent.job, &mut agent.sender).await?;
            // send any logs in our logs channel
            agent.send_channel_logs().await?;
            // collect any children files
            let mut children = agent.executor.children(&agent.image).await?;
            // send any logs in our logs channel
            agent.send_channel_logs().await?;
            // submit any collected children
            children
                .submit(
                    &agent.thorium,
                    &agent.job,
                    &results,
                    agent.job.trigger_depth,
                    &agent.commits,
                    &agent.image,
                    &mut agent.sender,
                )
                .await?;
            // mark this job as completed
            agent.completed = true;
            code
        }
        JobStatus::Failed(code) => code,
        JobStatus::OnGoing => {
            return Err(Error::new(format!("Job {} is still ongoing", agent.job.id)))
        }
    };
    // log that we have finished this job
    event!(Level::INFO, msg = "Finished job", code = code);
    // add the return code to our logs if it exists
    match code {
        Some(code) => agent.sender.send(format!("Return Code: {code}"))?,
        None => agent.sender.send("Return Code: None".to_string())?,
    };
    // send any remaining channel logs
    agent.send_channel_logs().await?;
    agent.executor.clean_up(&agent.image, &agent.job).await?;
    Ok(())
}

/// Executes a claimed job with the correct agent and logs any errors
#[instrument(
    name = "agents::execute",
    skip(agent),
    fields(
        image = agent.image.name,
        job = agent.job.id.to_string(),
        reaction = agent.job.reaction.to_string(),
        group = agent.job.group,
        pipeline = agent.job.pipeline,
        stage = agent.job.stage,
        creator = agent.job.creator,
    ))]
pub async fn execute(mut agent: Agent, log_path: String) {
    // create the file to write logs too
    let log_file = match create_log_file(&log_path).await {
        Ok(log_file) => log_file,
        Err(error) => {
            // build the error message to send
            let msg = format!("Failed to setup log file at {log_path}");
            // log this error event
            event!(Level::INFO, msg = &msg);
            // send our error message
            log!(agent.sender, msg);
            // error our this job
            check!(agent.error_channel_only(&error).await);
            // return early
            return;
        }
    };
    // get a bufreader to our log file
    let mut reader = BufReader::new(log_file);
    // try executing this job and report any failures if the occur
    match sub_execute(&mut agent, &mut reader, &log_path).await {
        // job completed so mark it as complete and proceed
        Ok(()) => {
            event!(Level::INFO, msg = "Proceeding with reaction");
            check!(agent.proceed().await);
            // delete this jobs log file
            if let Err(error) = tokio::fs::remove_file(log_path).await {
                // log this error but continue on since it probably doesn't matter
                event!(Level::INFO, msg = error.to_string());
            }
        }
        // job failed so make it as failed and send any error logs
        Err(error) => {
            event!(Level::INFO, msg = "Job failed", error = error.to_string());
            // clean up our failed job
            check!(agent.executor.clean_up(&agent.image, &agent.job).await);
            // error out this job
            check!(agent.error(&mut reader, &error).await);
            // delete this jobs log file
            if let Err(error) = tokio::fs::remove_file(log_path).await {
                // log this error but continue on since it probably doesn't matter
                event!(Level::INFO, msg = error.to_string());
            }
        }
    }
}
