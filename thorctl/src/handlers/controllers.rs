//! The controller for workers in Thorctl
use crate::args::Args;

use super::progress::{BarKind, MultiBar};
use super::{JobMsg, MonitorHandler, MonitorMsg, Worker, WorkerWrapper};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use kanal::{AsyncReceiver, AsyncSender};
use owo_colors::OwoColorize;
use thorium::{CtlConf, Error, Thorium};
use tokio::task::JoinHandle;

/// The controller for our workers
pub struct Controller<W: Worker> {
    /// The channel to send jobs to our workers on
    pub jobs: AsyncSender<JobMsg<W>>,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<W::Monitor>>,
    /// The recieve channel for this controller (used to ensure our channel doesn't close)
    monitor_rx: AsyncReceiver<MonitorMsg<W::Monitor>>,
    /// The handle for our monitor
    pub monitor: JoinHandle<()>,
    /// The progress bar to log updates on
    pub multi: MultiBar,
    /// The futures for our workers
    active: FuturesUnordered<JoinHandle<()>>,
}

impl<W: Worker> Controller<W> {
    /// Create a new controller
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to set for the monitor bar
    /// * `thorium` - A Thorium client
    /// * `workers` - The number of workers to spawn
    /// * `conf` - The config for Thorctl
    /// * `args` - The arguments that were passed to Thorctl
    /// * `cmd` - The specific args for this controllers workers
    pub async fn spawn(
        msg: &str,
        thorium: &Thorium,
        workers: usize,
        conf: &CtlConf,
        args: &Args,
        cmd: &W::Cmd,
    ) -> Self {
        // build a new multiprogress bar
        let multi = MultiBar::default();
        // build our channel for sending/receiving messages
        let (jobs_tx, jobs_rx) = kanal::unbounded_async();
        // build our channel for sending/receiving monitor updates on
        let (monitor_tx, monitor_rx) = kanal::unbounded_async();
        // spawn our global monitor
        let monitor = MonitorHandler::spawn(msg, monitor_rx.clone(), &multi);
        // build our controller
        let mut controller = Controller {
            jobs: jobs_tx.clone(),
            monitor_tx,
            monitor_rx,
            monitor,
            multi,
            active: FuturesUnordered::default(),
        };
        // spawn our workers
        controller
            .spawn_workers(thorium, workers, &jobs_tx, &jobs_rx, conf, args, cmd)
            .await;
        // return our controller
        controller
    }

    /// Spawn N new workers under this controller
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The number of workers to spawn
    /// * `jobs_tx` - The channel to send new jobs on
    /// * `jobs_rx` - The channel to receive new jobs on
    /// * `conf` - The config for Thorctl
    /// * `args` - The arguments that were passed to Thorctl
    /// * `cmd` - The specific args for this controllers workers
    async fn spawn_workers(
        &mut self,
        thorium: &Thorium,
        workers: usize,
        jobs_tx: &AsyncSender<JobMsg<W>>,
        jobs_rx: &AsyncReceiver<JobMsg<W>>,
        conf: &CtlConf,
        args: &Args,
        cmd: &W::Cmd,
    ) {
        for i in 0..workers {
            // clone our command
            let cmd = cmd.clone();
            // build our worker spawn message
            let msg = format!("Spawning 🦀: {i}");
            // setup this workers progress bar
            let bar = self.multi.add(&msg, BarKind::Timer);
            // build our inner worker
            let inner = W::init(thorium, conf, bar, args, cmd, &self.monitor_tx).await;
            // build our outer worker
            let worker = WorkerWrapper::<W>::new(jobs_rx, jobs_tx, inner);
            // start this worker
            let handle = tokio::spawn(async move { worker.start().await });
            // add this tank to our futures set
            self.active.push(handle);
        }
    }

    /// Log an error message
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to log
    pub fn error(&mut self, msg: &str) {
        self.multi
            .error(&format!("{}: {}", "Error".bright_red(), msg))
            .unwrap_or_else(|_| panic!("Failed to log error: {msg}"));
    }

    /// Print an info message
    ///
    /// # Arguments
    ///
    /// * `msg` - The info message to print
    #[allow(dead_code)]
    pub fn info(&self, msg: &str) {
        self.multi
            .error(&format!("{}: {}", "Info".bright_blue(), msg))
            .unwrap_or_else(|_| panic!("Failed to log info: {msg}"));
    }

    /// Add a job to our queue
    ///
    /// # Arguments
    ///
    /// * `job` - The job to send to our workers
    pub async fn add_job(&mut self, job: W::Job) -> Result<(), kanal::SendError> {
        // add a job to our queue
        self.jobs.send(JobMsg::Job(job)).await?;
        // tell our monitor about our new job
        self.monitor_tx.send(MonitorMsg::Extend(1)).await?;
        Ok(())
    }

    /// Wait for all workers to finish
    pub async fn finish(mut self) -> Result<(), Error> {
        // send our complete msg
        self.jobs.send(JobMsg::Finished).await.unwrap();
        // wait for all of our workers to complete
        while let Some(ret) = self.active.next().await {
            // log any errors
            if let Err(error) = ret {
                self.error(&error.to_string());
            }
        }
        // tell our monitor to complete
        self.monitor_tx.send(MonitorMsg::Finished).await?;
        // wait for our monitor to complete
        self.monitor.await?;
        Ok(())
    }
}
