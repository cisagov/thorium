//! Downloads files for thorctl

use colored::Colorize;
use itertools::Itertools;
use kanal::AsyncSender;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use thorium::models::{CarvedOrigin, FileDownloadOpts, Origin, Sample, SubmissionChunk};
use thorium::{CtlConf, Error, Thorium};

use crate::args::files::{DownloadFiles, FileDownloadOrganization};
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::utils;
use crate::Args;

/// The files download monitor
pub struct FilesDownloadMonitor;

impl Monitor for FilesDownloadMonitor {
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

pub struct FilesDownloadWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: DownloadFiles,
    /// The base output path to download repos too
    pub base: PathBuf,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<FilesDownloadMonitor>>,
}

impl FilesDownloadWorker {
    /// get the path to save this submissions file to based on provenance
    ///
    /// # Arguments
    ///
    /// * `target` - The submission chunk to build the provenance path for
    fn get_provenance_paths(&self, target: &SubmissionChunk) -> Result<PathBuf, Error> {
        match &target.origin {
            Origin::Downloaded { url, name } => {
                // if the users specified a name then use that
                let hostname = match name {
                    Some(name) => name,
                    None => utils::get_hostname(url)?,
                };
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("Downloaded");
                // add our hostname onto this path
                save_path.push(hostname);
                Ok(save_path)
            }
            Origin::Unpacked { tool, parent, .. } => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("Unpacked");
                // add the parent sha256
                save_path.push(parent);
                // add the tool that unpacked this or set Unknown
                match tool {
                    Some(tool) => save_path.push(tool),
                    None => save_path.push("Unknown"),
                }
                Ok(save_path)
            }
            Origin::Transformed { tool, parent, .. } => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("Transformed");
                // add the parent sha256
                save_path.push(parent);
                // add the tool that transformed this or set Unknown
                match tool {
                    Some(tool) => save_path.push(tool),
                    None => save_path.push("Unknown"),
                }
                Ok(save_path)
            }
            Origin::Wire {
                sniffer,
                source,
                destination,
            } => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("Wire");
                // add the sniffer
                save_path.push(sniffer);
                // add the source this file came from
                match source {
                    Some(source) => save_path.push(source),
                    None => save_path.push("Unknown"),
                }
                // add the destination this file was going too
                match destination {
                    Some(destination) => save_path.push(destination),
                    None => save_path.push("Unknown"),
                }
                Ok(save_path)
            }
            Origin::Incident {
                incident, machine, ..
            } => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("Incident");
                // add the sniffer
                save_path.push(incident);
                // if machine is set then set that
                match machine {
                    Some(machine) => save_path.push(machine),
                    None => save_path.push("Unknown"),
                }
                Ok(save_path)
            }
            Origin::MemoryDump { parent, .. } => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("MemoryDump");
                // add the sniffer
                save_path.push(parent);
                Ok(save_path)
            }
            Origin::Source {
                repo,
                commit,
                system,
                supporting,
                ..
            } => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("Source");
                // add the repo this was built from
                save_path.push(repo);
                // add the commit this was built from
                save_path.push(commit);
                // add the build system that was used
                save_path.push(system);
                // add whether this is a supporting build file or not
                if *supporting {
                    save_path.push("Supporting");
                }
                Ok(save_path)
            }
            Origin::None => {
                // clone our base path
                let mut save_path = self.base.clone();
                // add the origin type
                save_path.push("None");
                Ok(save_path)
            }
            Origin::Carved {
                parent,
                tool,
                carved_origin,
                ..
            } => {
                // add the origin type
                let mut save_path = self.base.join("Carved");
                // add the carved origin type
                match carved_origin {
                    CarvedOrigin::Pcap { .. } => {
                        save_path.push("Pcap");
                    }
                    CarvedOrigin::Unknown => save_path.push("Unknown"),
                }
                // add the parent SHA256
                save_path.push(parent);
                // add the tool that carved this file or set to unknown
                match tool {
                    Some(tool) => save_path.push(tool),
                    None => save_path.push("Unknown"),
                }
                Ok(save_path)
            }
        }
    }

    /// Get the name to write this file as in a human friendly format
    ///
    /// # Arguments
    ///
    /// * `sample` - The sample details to get our human readable name from
    pub fn friendly_name<'a>(
        &self,
        sample: &Sample,
        name: &'a Option<String>,
    ) -> (String, Option<&'a OsStr>) {
        // get the this submissions file name if it exists
        match name {
            Some(name) => {
                // extract the file name and extension from this name if it exists
                let path = Path::new(name);
                let base = path.file_prefix().map_or(sample.sha256.clone(), |name| {
                    name.to_string_lossy().into_owned()
                });
                let ext = path.extension();
                (base, ext)
            }
            None => (sample.sha256.clone(), None),
        }
    }

    /// Add the name from a specific submission to save this file as onto a path
    fn add_file_name_specific(
        &self,
        sample: &mut Sample,
        target: &SubmissionChunk,
        output: &mut PathBuf,
    ) {
        // if a human friendly name is requested then use that
        let (name, ext) = if self.cmd.friendly {
            // get the friendly name for this submission
            self.friendly_name(sample, &target.name)
        } else {
            (sample.sha256.clone(), None)
        };
        // add any tags and add this file name onto out path
        self.add_nametags(sample, name, ext, output);
    }

    /// Add the name from a specific submission to save this file as onto a path
    fn add_file_name(&self, sample: &mut Sample, output: &mut PathBuf) {
        // if a human friendly name is requested then use that
        if self.cmd.friendly {
            // find the first submission with a file name if one exists
            let target = sample.submissions.iter().find_map(|sub| sub.name.clone());
            // get the friendly name for this submission
            let (name, ext) = self.friendly_name(sample, &target);
            // add any tags and add this file name onto out path
            self.add_nametags(sample, name, ext, output);
        } else {
            // add any tags and add just the sha256 as our file name
            self.add_nametags(sample, sample.sha256.clone(), None, output);
        };
    }

    /// Extend our file name with any tags based on args
    fn add_nametags(
        &self,
        sample: &mut Sample,
        name: String,
        ext: Option<&OsStr>,
        output: &mut PathBuf,
    ) {
        // extract any tags to add
        let tags = self
            .cmd
            .nametags
            .iter()
            .filter_map(|nametag| sample.tags.get(nametag))
            .flat_map(|tag_map| tag_map.keys())
            .join("_");
        // if the file extension exists then add that to our human readable name
        match ext {
            Some(ext) => {
                // if tags is empty then dont add an extra '_'
                if tags.is_empty() {
                    // build the file name with tags to add to our path
                    let full_name = format!("{}.{}", name, ext.to_string_lossy());
                    // add our file name onto our path
                    output.push(full_name);
                } else {
                    // build the file name with tags to add to our path
                    let full_name = format!("{}_{}.{}", name, tags, ext.to_string_lossy());
                    // add our file name onto our path
                    output.push(full_name);
                }
            }
            None => {
                // if tags is empty then dont add an extra '_'
                if tags.is_empty() {
                    // extend our path with our full name
                    output.push(name);
                } else {
                    output.push(format!("{name}_{tags}"))
                }
            }
        }
    }

    /// Set up the paths for downloading this file based on this submission chunk
    async fn setup_organization(&self, sample: &mut Sample) -> Result<PathBuf, Error> {
        // get the path to write this file too based on our args
        match self.cmd.organization {
            FileDownloadOrganization::Simple => {
                // the simple structure just places all files in a single dir
                let mut output = self.base.clone();
                // add our file name onto this output path
                self.add_file_name(sample, &mut output);
                Ok(output)
            }
            FileDownloadOrganization::Provenance => {
                // get the next submission for this sample if one exists
                if let Some(submission) = sample.submissions.pop() {
                    // extend the output path based on how this submissions origin
                    let mut output = self.get_provenance_paths(&submission)?;
                    // add our file name onto this output path
                    self.add_file_name_specific(sample, &submission, &mut output);
                    // create this paths parents if it doesn't exist
                    if let Some(parent) = output.parent() {
                        tokio::fs::create_dir_all(parent).await?;
                    }
                    Ok(output)
                } else {
                    Err(Error::new(format!("{} has no submissions?", sample.sha256)))
                }
            }
        }
    }

    /// Download this file for the first submission chunk
    async fn download(&self, sample: &Sample, output: &PathBuf) -> Result<(), Error> {
        // set the file download opts to use
        let mut opts = FileDownloadOpts::default()
            .uncart_by_value(self.cmd.uncarted)
            .progress(self.bar.bar.clone());
        // download this file and uncart it
        self.thorium
            .files
            .download(&sample.sha256, &output, &mut opts)
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Worker for FilesDownloadWorker {
    /// The cmd part of args for this specific worker
    type Cmd = DownloadFiles;

    /// The type of jobs to receive
    type Job = String;

    /// The global monitor to use
    type Monitor = FilesDownloadMonitor;

    /// Initialize our worker
    async fn init(
        thorium: &Thorium,
        _conf: &CtlConf,
        bar: Bar,
        _args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        // if no output path was specified then use our current path
        let base = match &cmd.output {
            Some(output) => PathBuf::from_str(output).expect("Failed to cast output to a path"),
            None => std::env::current_dir().expect("Failed to get current directory"),
        };
        FilesDownloadWorker {
            thorium: Arc::new(thorium.clone()),
            bar,
            cmd: cmd.clone(),
            base,
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
        // set this progress bars name
        self.bar.rename(job.clone());
        // set that we are tarring this repository
        self.bar.refresh("", BarKind::UnboundIO);
        // get info on this sample
        let mut sample = check!(self.bar, self.thorium.files.get(&job).await);
        // create the required organization structure for downloading this file
        let output = check!(self.bar, self.setup_organization(&mut sample).await);
        // download this file
        check!(self.bar, self.download(&sample, &output).await);
        // if this file needs to be copied to other paths on disk then do that
        if self.cmd.organization.may_copy() {
            // iterate over the other submissions and copy them to the required positions
            while !sample.submissions.is_empty() {
                // create the required organization structure for downloading this file
                let target = check!(self.bar, self.setup_organization(&mut sample).await);
                // copy our original file to its new position
                if let Err(error) = tokio::fs::copy(&output, &target).await {
                    // log this error
                    self.bar
                        .error(format!("{}: {}", "Error".bright_red(), error));
                    // try the next target
                    continue;
                }
            }
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
