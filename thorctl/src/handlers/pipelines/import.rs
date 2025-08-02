use colored::Colorize;
use kanal::AsyncSender;
use std::sync::Arc;
use thorium::models::PipelineRequest;
use thorium::{CtlConf, Error, Thorium};

use crate::args::pipelines::ImportPipelines;
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::Args;

/// The pipeline export monitor
pub struct PipelineImportMonitor;

impl Monitor for PipelineImportMonitor {
    /// The update type to use
    type Update = ();

    /// build this monitors progress bar
    fn build_bar(multi: &MultiBar, msg: &str) -> Bar {
        multi.add(msg, BarKind::Bound(0))
    }

    /// Apply an update to our global progress bar
    fn apply(bar: &Bar, _: Self::Update) {
        bar.inc(1);
    }
}

pub struct PipelineImportWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: ImportPipelines,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<PipelineImportMonitor>>,
}

impl PipelineImportWorker {
    /// Import an pipeline from a specific group by name
    pub async fn import(&mut self, name: &str) -> Result<(), Error> {
        // log that we are exporting this pipelines config
        self.bar.set_message("Importing Pipeline");
        // add our pipeline name
        let file_path = self.cmd.import.join(format!("{name}.json"));
        // try to load this pipeline from disk
        let pipeline_str = tokio::fs::read_to_string(&file_path).await?;
        // parse this pipeline request
        let mut pipeline_req: PipelineRequest = serde_json::from_str(&pipeline_str)?;
        // update the group for this pipeline
        pipeline_req.group = self.cmd.group.clone();
        // create this pipeline in Thorium
        self.thorium.pipelines.create(&pipeline_req).await?;
        Ok(())
    }
}

/// The trait for what workers should do
#[async_trait::async_trait]
impl Worker for PipelineImportWorker {
    /// The cmd part of args for this specific worker
    type Cmd = ImportPipelines;

    /// The type of jobs to recieve
    type Job = String;

    /// The global monitor to use
    type Monitor = PipelineImportMonitor;

    /// Initialize our worker
    async fn init(
        thorium: &Thorium,
        _conf: &CtlConf,
        bar: Bar,
        _args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        // create this pipeline export worker
        PipelineImportWorker {
            thorium: Arc::new(thorium.clone()),
            bar,
            cmd: cmd.clone(),
            monitor_tx: updates.clone(),
        }
    }

    /// Log an info message
    fn info<T: AsRef<str>>(&mut self, msg: T) {
        self.bar.info(msg)
    }

    /// Start claiming and executing jobs
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to start
    async fn execute(&mut self, job: Self::Job) {
        // set that we are tarring this repository
        self.bar.rename(job.clone());
        self.bar.refresh("", BarKind::Timer);
        // export this pipeline
        if let Err(error) = self.import(&job).await {
            // log this io error
            self.bar
                .error(format!("{}: {}", "Error".bright_red(), error));
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
