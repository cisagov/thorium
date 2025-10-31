use colored::Colorize;
use http::StatusCode;
use kanal::AsyncSender;
use std::sync::Arc;
use thorium::models::Reaction;
use thorium::{CtlConf, Thorium};

use crate::Args;
use crate::args::reactions::GetReactions;
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use crate::handlers::{Monitor, MonitorMsg, Worker};

/// The files download monitor
pub struct ReactionsDeleteMonitor;

impl Monitor for ReactionsDeleteMonitor {
    /// The update type to use
    type Update = ();

    /// build this monitors progress bar
    ///
    /// # Arguments
    ///
    /// * `multi` - The multibar to add a bar too
    /// * `msg`- The message to set for our monitor bar
    fn build_bar(multi: &MultiBar, msg: &str) -> Bar {
        multi.add(msg, BarKind::Bound(0))
    }

    /// Apply an update to our global progress bar
    ///
    /// # Arguments
    ///
    /// * `bar` - The bar to apply updates too
    /// * `update` - The update to apply
    fn apply(bar: &Bar, _: Self::Update) {
        bar.inc(1);
    }
}

macro_rules! check {
    ($bar:expr, $func:expr) => {
        match $func {
            Ok(output) => output,
            Err(error) => {
                // log this error
                $bar.error(format!("{}: {}", "Error".bright_red(), error));
                // return early
                return;
            }
        }
    };
}

pub struct ReactionsDeleteWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: GetReactions,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<ReactionsDeleteMonitor>>,
}

#[async_trait::async_trait]
impl Worker for ReactionsDeleteWorker {
    /// The cmd part of args for this specific worker
    type Cmd = GetReactions;

    /// The type of jobs to recieve
    type Job = Reaction;

    /// The global monitor to use
    type Monitor = ReactionsDeleteMonitor;

    /// Initialize our worker
    async fn init(
        thorium: &Thorium,
        _conf: &CtlConf,
        bar: Bar,
        _args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        ReactionsDeleteWorker {
            thorium: Arc::new(thorium.clone()),
            bar,
            cmd: cmd.clone(),
            monitor_tx: updates.clone(),
        }
    }

    /// Log an info message
    fn info<T: AsRef<str>>(&mut self, msg: T) {
        self.bar.info(msg);
    }

    /// Start claiming and executing jobs
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to start
    async fn execute(&mut self, job: Self::Job) {
        // set that we are tarring this repository
        self.bar.refresh("", BarKind::Unbound);
        // delete our reactions
        if let Err(error) = self.thorium.reactions.delete(&job.group, &job.id).await {
            // set our bars name to our reaction id and group
            self.bar.rename(format!("{}:{}", job.group, job.id));
            // log our error
            self.bar.error(error.to_string());
        }
        // send an update to our monitor
        if let Err(error) = self.monitor_tx.send(MonitorMsg::Update(())).await {
            // log this io error
            self.bar
                .error(format!("{}: {}", "Error".bright_red(), error));
        }
        // finish our progress bar
        self.bar.finish_and_clear();
    }
}
