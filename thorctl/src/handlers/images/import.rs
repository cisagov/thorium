//! Image export support for thorctl

use colored::Colorize;
use http::StatusCode;
use kanal::AsyncSender;
use std::process::Output;
use std::sync::Arc;
use thorium::models::{ImageRequest, ImageScaler, ImageUpdate};
use thorium::{CtlConf, Error, Thorium};
use tokio::process::Command;

use crate::args::images::ImportImages;
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::Args;

/// The image export monitor
pub struct ImageImportMonitor;

impl Monitor for ImageImportMonitor {
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

fn check_command(bar: &Bar, output: Output, msg: &str) -> Result<(), Error> {
    // check if this command failed
    if !output.status.success() {
        // cast this line to a string
        let lines_str = String::from_utf8_lossy(&output.stderr);
        // log each line on its own line
        for line in lines_str.lines().filter(|line| !line.is_empty()) {
            // log this line
            bar.error(line);
        }
        // return an error for this docker cmd
        return Err(Error::new(msg));
    }
    Ok(())
}

fn override_registry(image: &ImageRequest, override_opt: &Option<String>) -> Option<String> {
    match (&override_opt, &image.image) {
        (Some(registry), Some(old_url)) => {
            // try to split our old image url into the path and the domain
            let url_path = match old_url.split_once('/') {
                Some((_, old_path)) => old_path,
                None => &old_url,
            };
            // build our new url for the new registry
            Some(format!("{registry}/{url_path}"))
        }
        _ => None,
    }
}

pub struct ImageImportWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: ImportImages,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<ImageImportMonitor>>,
}

impl ImageImportWorker {
    /// Load an image request from disk
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the image whose config we are loading from disk
    async fn load_request(&self, name: &str) -> Result<ImageRequest, Error> {
        // add our image name
        let file_path = self.cmd.import.join(format!("{name}.json"));
        // try to load this image from disk
        let image_str = match tokio::fs::read_to_string(&file_path).await {
            Ok(image_str) => image_str,
            Err(error) => {
                // log that we failed to load an image at some path
                self.bar
                    .error(format!("Failed to read image data at {file_path:?}"));
                // reraise our error
                return Err(Error::from(error));
            }
        };
        // parse this image request
        let mut image_req: ImageRequest = match serde_json::from_str(&image_str) {
            Ok(image) => image,
            Err(error) => {
                // log that we failed to load an image at some path
                self.bar.error(format!(
                    "Faile to parse image data for {name} with {error:?}"
                ));
                // reraise our error
                return Err(Error::from(error));
            }
        };
        // update the group for this image
        image_req.group = self.cmd.group.clone();
        Ok(image_req)
    }

    /// Load an image to our local docker cache
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the image we are loading from disk
    async fn load_image(&self, name: &str) -> Result<(), Error> {
        // add our image name
        let file_path = self.cmd.import.join(format!("{name}.tar.gz"));
        // log that we are loading this image
        self.bar.set_message("loading image");
        // load this image from disk
        let output = Command::new("sh")
            .arg("-c")
            .arg("docker")
            .arg("load")
            .arg("<")
            .arg(file_path)
            .output()
            .await?;
        // make sure this command succeeded
        check_command(&self.bar, output, "Failed to load docker image")?;
        Ok(())
    }

    /// If we are uploading this image to a different registry then retag it
    ///
    /// # Arguments
    ///
    /// * `image_url` - Our original image url that may need to be retagged
    async fn retag_if_needed(&self, image: &mut ImageRequest) -> Result<(), Error> {
        // override our push registry if needed
        if let Some(retag_url) = override_registry(&image, &self.cmd.registry) {
            // log that we are exporting this images config
            self.bar.set_message("retagging image");
            // get our old url
            let old_url = image.image.as_ref().unwrap();
            // build the arguments to retag this image
            let retag_args = ["tag", &old_url, &retag_url];
            // retag this image
            let output = Command::new("docker").args(retag_args).output().await?;
            // make sure this command succeeded
            check_command(&self.bar, output, "Failed to retag docker image")?;
            // update our images url
            image.image = Some(retag_url);
        }
        Ok(())
    }

    /// Push a tools docker image if we have one set
    ///
    /// # Arguments
    ///
    /// * `image` - The image request to push to a docker registry
    async fn push_image(&self, image: &ImageRequest) -> Result<(), Error> {
        // only upload this docker image if there is one
        if let Some(image_url) = &image.image {
            // log that we are exporting this images config
            self.bar.set_message("Uploading image");
            // build the arguments to push this image
            let push_args = ["push", image_url];
            // push this images docker info
            let output = Command::new("docker").args(push_args).output().await?;
            // make sure this command succeeded
            check_command(&self.bar, output, "Failed to push docker image")?;
        }
        Ok(())
    }

    /// Migrate or create an image if it doesn't exist
    ///
    /// # Arguments
    ///
    /// * `image_req` - The image request to migrate or create
    async fn migrate_or_create(&self, image_req: &ImageRequest) -> Result<(), Error> {
        // import this image into thorium or just update its registry if needed
        if self.cmd.migrate_registry {
            // only update images with an image url
            if let Some(image_url) = &image_req.image {
                // build the image update to apply
                let update = ImageUpdate::default().image(image_url);
                // update this image
                match self
                    .thorium
                    .images
                    .update(&image_req.group, &image_req.name, &update)
                    .await
                {
                    Ok(_) => return Ok(()),
                    Err(error) => {
                        // check if this image just doesn't exist
                        if error.status() != Some(StatusCode::NOT_FOUND) {
                            // something else went wrong that we can't handle
                            return Err(error);
                        }
                    }
                }
            }
        }
        // if we aren't doing a migration or this image didn't already exist then create it
        if let Err(error) = self.thorium.images.create(&image_req).await {
            // check if this failed beacuse this image already exists
            if error.status() != Some(StatusCode::CONFLICT) {
                // something other then this image already existing caused a failure
                // raise this error
                return Err(error);
            }
        }
        Ok(())
    }

    /// Import an image to Thorium
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the image to import
    async fn import(&self, name: &str) -> Result<(), Error> {
        // try to load this images request from disk
        let mut image_req = self.load_request(name).await?;
        // handle docker specific needs if this image is under the k8s scaler
        if image_req.scaler == ImageScaler::K8s {
            // only perform docker load/retag/push if we are pushing
            if !self.cmd.skip_push {
                // load our docker image if this is a docker based tool
                self.load_image(name).await?;
                // retag this image if we are pushing to a different registry
                self.retag_if_needed(&mut image_req).await?;
                // push our docker image
                self.push_image(&image_req).await?;
            }
            // if set override our registry in Thorium
            if let Some(registry_override) =
                override_registry(&image_req, &self.cmd.registry_override)
            {
                // overrride our image url
                image_req.image = Some(registry_override);
            }
            // migrate this image to a new registry or create it
            self.migrate_or_create(&image_req).await?;
        }
        Ok(())
    }
}

/// The trait for what workers should do
#[async_trait::async_trait]
impl Worker for ImageImportWorker {
    /// The cmd part of args for this specific worker
    type Cmd = ImportImages;

    /// The type of jobs to recieve
    type Job = String;

    /// The global monitor to use
    type Monitor = ImageImportMonitor;

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
        ImageImportWorker {
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
        // set that we are importing this image
        self.bar.rename(job.clone());
        self.bar.refresh("", BarKind::Timer);
        // import this image
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
