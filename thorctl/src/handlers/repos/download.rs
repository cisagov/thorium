//! Allow thorctl to download repos from Thorium

use async_walkdir::{DirEntry, WalkDir};
use futures::StreamExt;
use kanal::AsyncSender;
use owo_colors::OwoColorize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thorium::models::{RepoDownloadOpts, UntarredRepo};
use thorium::{CtlConf, Error, Thorium};

use super::progress::{Bar, BarKind, MultiBar};
use crate::args::repos::{DownloadRepos, RepoDownloadOrganization, RepoTarget};
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::Args;

/// The repo ingest monitor
pub struct RepoDownloadMonitor;

impl Monitor for RepoDownloadMonitor {
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

/// Delete a file if it exists
///
/// # Arguments
///
/// * `base` - The base path to our repo
/// * `nested` - The nested path to the file to delete
macro_rules! prune_file {
    ($base:expr, $nested:expr) => {
        // add our nested path
        $base.push($nested);
        // remove our file if it exists
        if tokio::fs::try_exists($base).await? {
            tokio::fs::remove_file($base).await?;
        }
        // pop our nested path
        $base.pop();
    };
}

/// Delete a dir if it exists
///
/// # Arguments
///
/// * `base` - The base path to our repo
/// * `nested` - The nested path to the dir to delete
macro_rules! prune_dir {
    ($base:expr, $nested:expr) => {
        // add our nested path
        $base.push($nested);
        // remove our dir if it exists
        if tokio::fs::try_exists($base).await? {
            tokio::fs::remove_dir_all($base).await?;
        }
        // pop our nested path
        $base.pop();
    };
}

pub struct DownloadWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The arguments for downloading repos
    pub cmd: DownloadRepos,
    /// The minimum and max sizes to retain if set
    pub size_bounds: (Option<u64>, Option<u64>),
    /// The base output path to download repos too
    pub base: PathBuf,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<RepoDownloadMonitor>>,
}

impl DownloadWorker {
    /// Set up the paths for this repo download based on user provided args
    async fn setup_organization(&self, target: &RepoTarget) -> Result<PathBuf, Error> {
        // setup any folders needed for the desired organizational structure
        match self.cmd.organization {
            RepoDownloadOrganization::Simple => {
                // make the base dir
                tokio::fs::create_dir_all(&self.base).await?;
                // return our base path
                Ok(self.base.clone())
            }
            RepoDownloadOrganization::Provenance => {
                // make the base dir
                tokio::fs::create_dir_all(&self.base).await?;
                // split our repo url  by '/'
                let split = target.url.split('/').take(2);
                // clone our base path
                let mut path = self.base.clone();
                // make folders for the
                for item in split {
                    // extend our base path with our sources name
                    path.push(item);
                    // create our source directory
                    if let Err(error) = tokio::fs::create_dir(&path).await {
                        // skip any already exists errors
                        if error.kind() != std::io::ErrorKind::AlreadyExists {
                            return Err(Error::from(error));
                        }
                    }
                }
                Ok(path)
            }
        }
    }

    /// Determine if this file should be deleted or not
    async fn should_prune(&self, entry: &DirEntry) -> Result<bool, Error> {
        // only check files
        if entry.file_type().await?.is_file() {
            // do 2 instead of 1 for # of entries otherwise we just get the entire string
            if let Some(ext) = entry.file_name().to_string_lossy().rsplitn(2, '.').next() {
                // check if this extension is in our prune set
                if self.cmd.prune_extensions.iter().any(|item| item == ext) {
                    return Ok(true);
                }
                // check if we have any extensions to retain
                if !self.cmd.retain_extensions.is_empty() {
                    if !self.cmd.retain_extensions.iter().any(|item| item == ext) {
                        return Ok(true);
                    }
                }
            }
            // check this files size if we have min/max size settings
            if self.size_bounds.0.is_some() || self.size_bounds.1.is_some() {
                // get this files size
                let size = entry.metadata().await?.len();
                // check if this file should be pruned
                let size_check = match self.size_bounds {
                    (Some(min), Some(max)) => min > size || size > max,
                    (Some(min), None) => min > size,
                    (None, Some(max)) => size > max,
                    (None, None) => false,
                };
                return Ok(size_check);
            }
        }
        Ok(false)
    }

    /// Prune any unneeded data in Thorium
    async fn prune(&self, repo: &UntarredRepo) -> Result<(), Error> {
        // clone the path to our repo
        let mut base = repo.path.clone();
        // track whether we pruned any data and need to check for empty dirs
        let mut check_empty_dirs = false;
        // remove git files if needed
        if self.cmd.prune_git {
            // remove our git files
            prune_dir!(&mut base, ".git");
            prune_dir!(&mut base, ".github");
            prune_file!(&mut base, ".gitignore");
            prune_file!(&mut base, ".gitlab-ci.yml");
            prune_file!(&mut base, ".gitmodules");
            prune_file!(&mut base, ".gitattributes");
            // check for empty dirs
            check_empty_dirs = true;
        }
        // crawl over all files and remove any that are not wanted
        if !self.cmd.prune_extensions.is_empty() || !self.cmd.retain_extensions.is_empty() {
            // build a walker over all files in the untarred repo
            let mut walker = WalkDir::new(&base);
            // crawl over our the files in this repo
            while let Some(entry) = walker.next().await {
                // get this entry
                let entry = entry.unwrap();
                // check if we should delete this entry
                if self.should_prune(&entry).await? {
                    // get the path to this file
                    tokio::fs::remove_file(&entry.path()).await?;
                }
            }
            // check for empty dirs
            check_empty_dirs = true;
        }
        // if needed check and delete any empty dirs
        if check_empty_dirs {
            // build a walker over all files in the untarred repo
            let mut walker = WalkDir::new(&base);
            // have a sorted set of directories by length
            let mut dirs = BTreeMap::new();
            // crawl over our the files in this repo
            while let Some(entry) = walker.next().await {
                // get this entry
                let entry = entry.unwrap();
                // only check directories
                if entry.file_type().await?.is_dir() {
                    // get this paths entry
                    let path = entry.path();
                    // get the length of this path
                    let length = path.as_os_str().len();
                    // add the path to this directory
                    dirs.insert(length, path);
                }
            }
            // crawl over all dirs in reverse and delete them if they are empty
            for path in dirs.values().rev() {
                // check if this directory contains any files
                if path.read_dir()?.next().is_none() {
                    // delete this empty directory
                    tokio::fs::remove_dir(path).await?;
                }
            }
            // check if this repo is empty
            if base.read_dir()?.next().is_none() {
                // get the name of this repo
                if let Some(name) = base.file_name() {
                    // log that this repo was entirely pruned
                    self.bar.info(format!("{name:?} was entirely pruned"));
                    // delete this empty directory
                    tokio::fs::remove_dir(base).await?;
                }
            }
        }
        Ok(())
    }

    /// Check if this repo has already been downloaded
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo url to download
    /// * `output` - The directory this repo might have been downloaded too
    pub async fn exists(&self, repo: &str, output: &PathBuf) -> bool {
        // clone our output path
        let mut output = output.clone();
        // get the name of our repo
        let repo_path = Path::new(repo);
        let repo_name = match repo_path.file_name() {
            Some(file_name) => file_name.to_string_lossy().to_string(),
            None => return false,
        };
        // build the path to this repos cart file
        output.push(&repo_name);
        // add the right extension for carted files
        if self.cmd.carted {
            // set the .tar.cart extension
            output.as_mut_os_string().push(".tar.cart");
        }
        // check if this file exists
        tokio::fs::try_exists(&output)
            .await
            .is_ok_and(|inner| inner)
    }

    /// Download a repo to disk
    pub async fn download(&self, mut job: RepoTarget, output: &PathBuf) -> Result<(), Error> {
        // build the opts for downloading this repo
        let mut opts = RepoDownloadOpts::default().progress(self.bar.bar.clone());
        // if a commitish is set then set that
        if let Some(commitish) = job.commitish.take() {
            opts.commitish = Some(commitish);
        }
        // if a commitish kind is set then set that
        if let Some(commitish_kind) = job.kind.take() {
            opts.kinds = vec![commitish_kind];
        }
        if self.cmd.carted {
            // download and leave this repo carted + tarred
            self.thorium
                .repos
                .download(&job.url, &opts, &output)
                .await?;
        } else {
            // download, uncart and untar this repo
            match self
                .thorium
                .repos
                .download_unpack(&job.url, &opts, &output)
                .await
            {
                Ok(untarred) => {
                    // prune this repo if required
                    if let Err(error) = self.prune(&untarred).await {
                        // log this error and delete the downloaded repo if it exists
                        self.bar
                            .error(format!("{}: {}", "Error".bright_red(), error));
                        // delete this downloaded repo
                        if tokio::fs::try_exists(&untarred.path)
                            .await
                            .is_ok_and(|exists| exists)
                        {
                            // try to delete this repo
                            if let Err(error) = tokio::fs::remove_dir_all(&untarred.path).await {
                                // log this error and delete the downloaded repo if it exists
                                self.bar
                                    .error(format!("{}: {}", "Error".bright_red(), error));
                            }
                        }
                    }
                }
                Err(error) => {
                    // log this error
                    self.bar
                        .error(format!("{}: {}", "Error".bright_red(), error));
                    // get the name of this repo
                    if let Some((_, name)) = job.url.rsplit_once('/') {
                        // build the path to the repo we tried to download
                        let path = output.join(name);
                        // delete this downloaded repo
                        if tokio::fs::try_exists(&path)
                            .await
                            .is_ok_and(|exists| exists)
                        {
                            // try to delete this repo
                            if let Err(error) = tokio::fs::remove_dir_all(&path).await {
                                // log this error and delete the downloaded repo if it exists
                                self.bar
                                    .error(format!("{}: {}", "Error".bright_red(), error));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Worker for DownloadWorker {
    /// The cmd part of args for this specific worker
    type Cmd = DownloadRepos;

    /// The type of jobs to receive
    type Job = RepoTarget;

    /// The global monitor to use
    type Monitor = RepoDownloadMonitor;

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
            Some(output) => output.to_owned(),
            None => std::env::current_dir().expect("Failed to get current directory"),
        };
        // get our minimum and max sizes bounds if they are set
        let size_bounds = cmd
            .convert_retain_sizes()
            .expect("Failed to get size bounds");
        DownloadWorker {
            thorium: Arc::new(thorium.clone()),
            bar,
            cmd: cmd.clone(),
            size_bounds,
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
        self.bar.rename(job.url.clone());
        // set that we are tarring this repository
        self.bar.refresh("", BarKind::UnboundIO);
        // set up any organizational structure
        let output = match self.setup_organization(&job).await {
            Ok(output) => output,
            Err(error) => {
                // log this error
                self.bar
                    .error(format!("{}: {}", "Error".bright_red(), error));
                // return early
                return;
            }
        };
        // if we should skip existing repos then check if this repo already exists
        let skip = if self.cmd.skip_existing {
            // check if this repo exists already
            if self.exists(&job.url, &output).await {
                // log that this repo already exists
                self.bar.info("Skipping already downloaded repo");
                true
            } else {
                false
            }
        } else {
            false
        };
        // if we aren't skipping this download then download this repo
        if !skip {
            // try to download this repo
            if let Err(error) = self.download(job, &output).await {
                // log this error
                self.bar
                    .error(format!("{}: {}", "Error".bright_red(), error));
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
