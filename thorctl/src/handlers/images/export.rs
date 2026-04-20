//! Image export support for thorctl

use colored::Colorize;
use kanal::AsyncSender;
use std::path::PathBuf;
use thorium::{CtlConf, Error, Thorium};

use crate::Args;
use crate::args::images::ExportImages;
use crate::handlers::container;
use crate::handlers::progress::{Bar, BarKind};
use crate::handlers::{MonitorMsg, SimpleMonitor, Worker};

/// A worker that exports a single image's container tarball
pub struct ImageExportWorker {
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for this image export
    pub cmd: ExportImages,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<SimpleMonitor>>,
}

impl ImageExportWorker {
    /// Export a single image's container image to a gzipped tarball
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the image being exported (used for the tarball filename)
    /// * `image_url` - The container url to pull and save
    /// * `export_path` - The directory to write the `<name>.tar.gz` tarball into
    async fn export_container(
        &mut self,
        name: &str,
        image_url: &str,
        mut export_path: PathBuf,
    ) -> Result<(), Error> {
        container::pull(image_url, &self.bar).await?;
        export_path.push(format!("{name}.tar.gz"));
        container::save(image_url, &export_path, &self.bar).await?;
        Ok(())
    }

    /// Export a single image's container tarball
    ///
    /// Only the container pull/save runs here; the config file is written separately
    /// by `images::export` so concurrent tarball workers never race on prompting
    /// over an on-disk config conflict. The container url is supplied by the caller
    /// (captured when it wrote the config) so the worker doesn't re-fetch the image.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the image being exported
    /// * `image_url` - The container url to pull and save
    pub async fn export(&mut self, name: &str, image_url: &str) -> Result<(), Error> {
        let images_dir = self.cmd.output.join("images");
        tokio::fs::create_dir_all(&images_dir)
            .await
            .map_err(|e| Error::new(format!("Failed to create export directory: {e}")))?;
        self.bar.set_message("Exporting image");
        self.export_container(name, image_url, images_dir).await?;
        Ok(())
    }
}

/// The trait for what workers should do
#[async_trait::async_trait]
impl Worker for ImageExportWorker {
    /// The cmd part of args for this specific worker
    type Cmd = ExportImages;

    /// The type of jobs to recieve: the image name paired with its container url
    type Job = (String, String);

    /// The global monitor to use
    type Monitor = SimpleMonitor;

    /// Initialize our worker
    ///
    /// # Arguments
    ///
    /// * `_thorium` - The Thorium client (unused by this worker)
    /// * `_conf` - The Thorctl config (unused by this worker)
    /// * `bar` - The progress bar this worker logs progress with
    /// * `_args` - The shared Thorctl args (unused by this worker)
    /// * `cmd` - The export images command being executed
    /// * `updates` - The channel to send monitor updates on
    async fn init(
        _thorium: &Thorium,
        _conf: &CtlConf,
        bar: Bar,
        _args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        // create this image export worker
        ImageExportWorker {
            bar,
            cmd: cmd.clone(),
            monitor_tx: updates.clone(),
        }
    }

    /// Log an info message
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to log
    fn info<T: AsRef<str>>(&mut self, msg: T) {
        self.bar.info(msg)
    }

    /// Start claiming and executing jobs
    ///
    /// # Arguments
    ///
    /// * `job` - The image name paired with its container url to export
    async fn execute(&mut self, job: Self::Job) {
        let (name, url) = job;
        // set that we are tarring this repository
        self.bar.rename(name.clone());
        self.bar.refresh("", BarKind::Timer);
        // export this image
        if let Err(error) = self.export(&name, &url).await {
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
