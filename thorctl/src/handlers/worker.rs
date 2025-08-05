//! The worker trait for Thorctl

use crate::args::Args;

use super::progress::Bar;
use super::{Monitor, MonitorMsg};
use kanal::{AsyncReceiver, AsyncSender};
use thorium::{CtlConf, Thorium};

/// The trait for what workers should do
#[async_trait::async_trait]
pub trait Worker: Send + 'static {
    /// The cmd part of args for this specific worker
    type Cmd: Clone + Send;

    /// The type of jobs to receive
    type Job: Send;

    /// The global monitor to use
    type Monitor: Monitor;

    /// Initialize our worker
    async fn init(
        thorium: &Thorium,
        conf: &CtlConf,
        bar: Bar,
        args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self;

    /// Log an info message
    #[allow(dead_code)]
    fn info<T: AsRef<str>>(&mut self, msg: T);

    /// Start claiming and executing jobs
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to start
    async fn execute(&mut self, job: Self::Job);
}

/// The messages to send new jobs to workers with
pub enum JobMsg<W: Worker> {
    /// A new job to execute
    Job(W::Job),
    /// There are no more jobs so this worker should shutdown
    Finished,
}

/// A worker in Thorctl
///
/// This wraps our worker to remove the need for individual workers to implement
/// the job queue logic on their own.
pub(crate) struct WorkerWrapper<W: Worker> {
    /// The channel to get jobs on
    jobs_rx: AsyncReceiver<JobMsg<W>>,
    /// The channel to rebroadcast the finish message on
    jobs_tx: AsyncSender<JobMsg<W>>,
    /// The actual worker executing jobs
    inner: W,
}

impl<W: Worker> WorkerWrapper<W> {
    /// Create a default new worker
    pub fn new(
        jobs_rx: &AsyncReceiver<JobMsg<W>>,
        jobs_tx: &AsyncSender<JobMsg<W>>,
        inner: W,
    ) -> Self {
        WorkerWrapper {
            jobs_rx: jobs_rx.clone(),
            jobs_tx: jobs_tx.clone(),
            inner,
        }
    }

    /// Start executing jobs
    pub async fn start(mut self) {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            match self.jobs_rx.recv().await {
                Ok(JobMsg::Job(job)) => self.inner.execute(job).await,
                Ok(JobMsg::Finished) => {
                    // forward our finished message to another worker
                    self.jobs_tx
                        .send(JobMsg::Finished)
                        .await
                        .expect("Failed to forward finished message");
                    // exit this worker task
                    break;
                }
                Err(kanal::ReceiveError::Closed | kanal::ReceiveError::SendClosed) => break,
            }
        }
    }
}
