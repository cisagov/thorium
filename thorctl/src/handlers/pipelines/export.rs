use colored::Colorize;
use kanal::AsyncSender;
use std::sync::Arc;
use thorium::models::PipelineRequest;
use thorium::{CtlConf, Error, Thorium};

use crate::args::pipelines::ExportPipelines;
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::Args;

/// The pipeline export monitor
pub struct PipelineExportMonitor;

impl Monitor for PipelineExportMonitor {
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

pub struct PipelineExportWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: ExportPipelines,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<PipelineExportMonitor>>,
}

impl PipelineExportWorker {
    /// Export an pipeline from a specific group by name
    pub async fn export(&mut self, name: &str) -> Result<(), Error> {
        // log that we are exporting this pipelines config
        self.bar.set_message("Exporting Config");
        // create our export folder if it doesn't already exist
        tokio::fs::create_dir_all(&self.cmd.output).await?;
        // get this pipelines data
        let pipeline = self.thorium.pipelines.get(&self.cmd.group, name).await?;
        // build the path to write our exported pipeline info to
        let mut export_path = self.cmd.output.clone();
        // build the file name to write this pipelines exported config too
        export_path.push(format!("{}.json", &pipeline.name));
        // conver this pipeline into an pipeline request
        let pipeline_req = PipelineRequest::from(pipeline);
        // serialize this pipelines request
        let serialized = serde_json::to_string_pretty(&pipeline_req)?;
        // write this pipeline request to disk
        tokio::fs::write(&export_path, &serialized).await?;
        Ok(())
    }
}

/// The trait for what workers should do
#[async_trait::async_trait]
impl Worker for PipelineExportWorker {
    /// The cmd part of args for this specific worker
    type Cmd = ExportPipelines;

    /// The type of jobs to recieve
    type Job = String;

    /// The global monitor to use
    type Monitor = PipelineExportMonitor;

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
        PipelineExportWorker {
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
        if let Err(error) = self.export(&job).await {
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
