//! Image export support for thorctl

use colored::Colorize;
use kanal::AsyncSender;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use thorium::models::ImageRequest;
use thorium::{CtlConf, Error, Thorium};
use tokio::fs::File;
use tokio::process::Command;

use crate::args::images::ExportImages;
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::Args;

/// The image export monitor
pub struct ImageExportMonitor;

impl Monitor for ImageExportMonitor {
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

pub struct ImageExportWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: ExportImages,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<ImageExportMonitor>>,
}

impl ImageExportWorker {
    /// Export a single images info docker image
    async fn export_docker(
        &mut self,
        name: &str,
        image_url: &str,
        mut export_path: PathBuf,
    ) -> Result<(), Error> {
        // log that we are exporting this images config
        self.bar.set_message("Pulling image");
        // build the arguments to pull this image
        let pull_args = ["pull", image_url];
        // pull this images docker info
        Command::new("docker").args(pull_args).output().await?;
        // log that we are exporting this images config
        self.bar.set_message("Saving image");
        // build the arguments to save this file to disk
        let save_args = ["save", image_url];
        // pull this images data and feed it into a pipe
        let mut save_child = Command::new("docker")
            .args(save_args)
            .stdout(Stdio::piped())
            .spawn()?;
        // get the pipe for saves stdout
        let save_stdout: Stdio = save_child.stdout.take().unwrap().try_into().unwrap();
        // build the path to save our docker image too
        export_path.push(format!("{name}.tar.gz"));
        // create the file to write our docker image too or truncate it if it already exists
        let gz_file = File::create(&export_path).await?.into_std().await;
        // build the name of the file to save our docker image too
        // build the command to write the data form our docker pull pipe to disk
        let mut gzip_child = Command::new("gzip")
            .arg("--stdout")
            .stdin(save_stdout)
            .stdout(gz_file)
            .spawn()?;
        save_child.wait().await?;
        gzip_child.wait().await?;
        Ok(())
    }

    /// Export an image from a specific group by name
    pub async fn export(&mut self, name: &str) -> Result<(), Error> {
        // log that we are exporting this images config
        self.bar.set_message("Exporting config");
        // create our export folder if it doesn't already exist
        tokio::fs::create_dir_all(&self.cmd.output).await?;
        // get this images data
        let image = self.thorium.images.get(&self.cmd.group, name).await?;
        // build the path to write our exported image info to
        let mut export_path = self.cmd.output.clone();
        // build the file name to write this images exported config too
        export_path.push(format!("{}.json", &image.name));
        // conver this image into an image request
        let image_req = ImageRequest::from(image.clone());
        // serialize this images request
        let serialized = serde_json::to_string_pretty(&image_req)?;
        // write this image request to disk
        tokio::fs::write(&export_path, &serialized).await?;
        // pop our file name from our path
        export_path.pop();
        // save this images docker file to disk
        if let Some(image_url) = &image.image {
            self.export_docker(&image.name, image_url, export_path)
                .await?;
        }
        Ok(())
    }
}

/// The trait for what workers should do
#[async_trait::async_trait]
impl Worker for ImageExportWorker {
    /// The cmd part of args for this specific worker
    type Cmd = ExportImages;

    /// The type of jobs to recieve
    type Job = String;

    /// The global monitor to use
    type Monitor = ImageExportMonitor;

    /// Initialize our worker
    async fn init(
        thorium: &Thorium,
        _conf: &CtlConf,
        bar: Bar,
        _args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        // create this image export worker
        ImageExportWorker {
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
        // export this image
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
