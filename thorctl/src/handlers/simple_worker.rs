//! A generic worker for "run one job at a time" Thorctl commands
//!
//! The image/pipeline import and export commands all share the same shape: a
//! progress bar, a Thorium client, a command struct, and a per-item unit of
//! work. Implementing the full [`Worker`] trait for each one duplicates the
//! identical monitor/init/execute plumbing, so this module implements that
//! plumbing once and lets each command provide only its actual work via
//! [`SimpleJob`].

use async_trait::async_trait;
use colored::Colorize;
use kanal::AsyncSender;
use thorium::{CtlConf, Error, Thorium};

use super::progress::{Bar, BarKind, MultiBar};
use super::{Monitor, MonitorMsg, Worker};
use crate::args::Args;

/// A monitor that counts completed jobs on a single bounded bar
///
/// Every simple worker reports the same thing - "one more job finished" - so
/// they share this monitor rather than each defining an identical one.
pub(crate) struct CountMonitor;

impl Monitor for CountMonitor {
    type Update = ();

    fn build_bar(multi: &MultiBar, msg: &str) -> Bar {
        multi.add(msg, BarKind::Bound(0))
    }

    fn apply(bar: &Bar, _: Self::Update) {
        bar.inc(1);
    }
}

/// A single unit of work run by a [`SimpleWorker`]
///
/// Implementors provide only how to initialize themselves and how to process
/// one job; [`SimpleWorker`] handles the bar/monitor bookkeeping shared by all
/// of these commands.
#[async_trait]
pub(crate) trait SimpleJob: Send + Sized + 'static {
    /// The command args for this job
    type Cmd: Clone + Send + Sync;

    /// The type of job processed (e.g. an image name or a pre-fetched pipeline)
    type Job: Send;

    /// The label to display on the progress bar for a given job
    fn label(job: &Self::Job) -> String;

    /// Initialize the job worker
    async fn init(thorium: &Thorium, conf: &CtlConf, args: &Args, cmd: Self::Cmd) -> Self;

    /// Process a single job
    async fn run(&mut self, bar: &Bar, job: Self::Job) -> Result<(), Error>;
}

/// A [`Worker`] adapter that runs a [`SimpleJob`]
///
/// This implements [`Worker`] a single time for all simple jobs, removing the
/// boilerplate `Monitor`/`init`/`execute` impls each command would otherwise
/// duplicate.
pub(crate) struct SimpleWorker<J: SimpleJob> {
    /// The inner job logic
    inner: J,
    /// The progress bar for this worker
    ///
    /// The wrapper owns the bar (rather than the inner job) because
    /// `Bar::rename` takes `&mut self` while [`SimpleJob::run`] only needs
    /// shared access - keeping them on separate fields lets `execute` rename the
    /// bar and then lend it to `run` without fighting the borrow checker.
    bar: Bar,
    /// The channel used to report completed jobs to the global monitor
    monitor_tx: AsyncSender<MonitorMsg<CountMonitor>>,
}

#[async_trait]
impl<J: SimpleJob> Worker for SimpleWorker<J> {
    type Cmd = J::Cmd;
    type Job = J::Job;
    type Monitor = CountMonitor;

    async fn init(
        thorium: &Thorium,
        conf: &CtlConf,
        bar: Bar,
        args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        let inner = J::init(thorium, conf, args, cmd).await;
        SimpleWorker {
            inner,
            bar,
            monitor_tx: updates.clone(),
        }
    }

    fn info<T: AsRef<str>>(&mut self, msg: T) {
        self.bar.info(msg);
    }

    async fn execute(&mut self, job: Self::Job) {
        // name the bar after the job we're about to process
        self.bar.rename(J::label(&job));
        self.bar.refresh("", BarKind::Timer);
        // run the actual import/export logic, surfacing any error on the bar
        if let Err(error) = self.inner.run(&self.bar, job).await {
            self.bar
                .error(format!("{}: {}", "Error".bright_red(), error));
        }
        // advance the global progress bar whether or not the job succeeded so it
        // always reaches completion
        if let Err(error) = self.monitor_tx.send(MonitorMsg::Update(())).await {
            self.bar
                .error(format!("{}: {}", "Error".bright_red(), error));
        }
        self.bar.finish_and_clear();
    }
}
