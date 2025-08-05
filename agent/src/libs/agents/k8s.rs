//! Implements the Thorium agent for containers running in K8s

use crossbeam::channel::Sender;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thorium::models::{DependencyPassStrategy, GenericJob, Image};
use thorium::{Error, Thorium};
use tokio::process::Command;
use tracing::{event, instrument, Level};

use super::{registry, setup, AgentExecutor, InFlight};
use crate::args;
use crate::libs::children::{self, Children};
use crate::libs::{results, tags, RawResults, TagBundle, Target};
use crate::{build_path_args, deserialize, log, purge};

/// An execution of a single job in k8s
pub struct K8s {
    /// A client to Thorium
    pub thorium: Thorium,
    /// A sender for a channel of logs to add for this job
    pub logs: Sender<String>,
    /// The original entrypoint for the image we are in
    entrypoint: Vec<String>,
    /// The original command for the image we are in
    cmd: Vec<String>,
    /// The paths to any downloaded sample files
    samples: Vec<PathBuf>,
    /// The paths to any downloaded ephemeral files
    ephemerals: Vec<PathBuf>,
    /// The paths to any downloaded repos
    repos: Vec<PathBuf>,
    /// The paths to any downloaded results
    results: Vec<PathBuf>,
    /// The paths to any downloaded tags
    tags: Vec<PathBuf>,
    /// The paths to any downloaded children
    children: Vec<PathBuf>,
    /// whether this is a windows container or not
    pub windows: bool,
}

impl K8s {
    /// Create a new k8s agent for executing a single job
    ///
    /// # Arguments
    ///
    /// * `args` - The args to build this agent with
    /// * `target` - The target this agent will be executing
    /// * `logs` - Where to send logs
    pub fn new(args: &args::K8s, target: &Target, logs: Sender<String>) -> Result<Self, Error> {
        // deserialize entrypoint and cmd
        let entrypoint = deserialize!(&args.entrypoint);
        let cmd = deserialize!(&args.cmd);
        let k8s = K8s {
            thorium: target.thorium.clone(),
            logs,
            entrypoint,
            cmd,
            samples: Vec::default(),
            ephemerals: Vec::default(),
            repos: Vec::default(),
            results: Vec::default(),
            tags: Vec::default(),
            children: Vec::default(),
            windows: false,
        };
        Ok(k8s)
    }

    /// Build a k8s agent from a windows config
    ///
    /// The k8s agent also works for windows containers jobs
    ///
    /// # Arguments
    ///
    /// * `args` - The args to build this agent with
    /// * `target` - The target this agent will be executing
    /// * `logs` - Where to send logs
    pub fn from_windows(
        args: &args::Windows,
        target: &Target,
        logs: Sender<String>,
    ) -> Result<Self, Error> {
        // deserialize entrypoint and cmd
        let entrypoint = deserialize!(&args.entrypoint);
        let cmd = deserialize!(&args.cmd);
        let k8s = K8s {
            thorium: target.thorium.clone(),
            logs,
            entrypoint,
            cmd,
            samples: Vec::default(),
            ephemerals: Vec::default(),
            repos: Vec::default(),
            results: Vec::default(),
            tags: Vec::default(),
            children: Vec::default(),
            windows: true,
        };
        Ok(k8s)
    }

    /// Build a k8s agent from a kvm config
    ///
    /// The k8s agent also works for kvm jobs
    ///
    /// # Arguments
    ///
    /// * `args` - The args to build this agent with
    /// * `target` - The target this agent will be executing
    /// * `logs` - Where to send logs
    pub fn from_kvm(target: &Target, logs: Sender<String>) -> Result<Self, Error> {
        // either our entrypoint or command must be set
        let (entrypoint, cmd) = match (&target.image.args.entrypoint, &target.image.args.command) {
            (Some(entrypoint), Some(cmd)) => (entrypoint.clone(), cmd.clone()),
            (Some(entrypoint), None) => (entrypoint.clone(), Vec::default()),
            (None, Some(cmd)) => (Vec::default(), cmd.clone()),
            (None, None) => return Err(Error::new("Entrypoint or cmd must be set!")),
        };
        // deserialize entrypoint and cmd
        let k8s = K8s {
            thorium: target.thorium.clone(),
            logs,
            entrypoint,
            cmd,
            samples: Vec::default(),
            ephemerals: Vec::default(),
            repos: Vec::default(),
            results: Vec::default(),
            tags: Vec::default(),
            children: Vec::default(),
            windows: true,
        };
        Ok(k8s)
    }
}

#[async_trait::async_trait]
impl AgentExecutor for K8s {
    /// Get the paths to this executors current jobs results and result files
    fn result_paths(&self, image: &Image) -> (String, String) {
        // get our paths
        let results = image.output_collection.files.results.clone();
        let result_files = image.output_collection.files.result_files.clone();
        (results, result_files)
    }

    /// Set up the environment for executing a single job in Thorium
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to set up a job for
    /// * `job` - The job we are setting up for
    /// * `commits` - The commit that each repo is checked out too
    #[instrument(
        name = "AgentExecutor<K8s>::setup",
        skip_all,
        field(
            samples = image.dependencies.samples.location,
            ephemeral = image.dependencies.ephemeral.location,
            repos = image.dependencies.repos.location,
            results = image.dependencies.results.location,
            tags = image.dependencies.tags.location,
            collect_results = image.output_collection.files.results,
            collect_result_files = image.output_collection.files.result_files,
            collect_tags = image.output_collection.files.tags,
        ),
        err(Debug))]
    async fn setup(
        &mut self,
        image: &Image,
        job: &GenericJob,
        commits: &mut HashMap<String, String>,
    ) -> Result<(), Error> {
        // setup dependency base paths
        std::fs::create_dir_all(&image.dependencies.samples.location)?;
        std::fs::create_dir_all(&image.dependencies.ephemeral.location)?;
        std::fs::create_dir_all(&image.dependencies.repos.location)?;
        std::fs::create_dir_all(&image.dependencies.children.location)?;
        std::fs::create_dir_all(&image.dependencies.tags.location)?;
        // setup result base paths
        std::fs::create_dir_all(&image.output_collection.files.result_files)?;
        // get the parent to our results path
        let results_path = Path::new(&image.output_collection.files.results);
        if let Some(result_parent) = results_path.parent() {
            // log that we are creating our results file parent path
            event!(
                Level::INFO,
                msg = "Creating results parent dir",
                // &* is to deref the cow and then get a ref to the underlying str
                results_parent = &*result_parent.to_string_lossy(),
            );
            std::fs::create_dir_all(result_parent)?;
        }
        // build the paths for storing children files
        children::setup(&image.output_collection.children).await?;
        // download any data required for this job
        self.samples = setup::download_samples(
            &self.thorium,
            image,
            job,
            &image.dependencies.samples.location,
            &mut self.logs,
        )
        .await?;
        self.ephemerals = setup::download_ephemeral(
            &self.thorium,
            image,
            job,
            &image.dependencies.ephemeral.location,
            &mut self.logs,
        )
        .await?;
        setup::download_parent_ephemeral(
            &mut self.ephemerals,
            &self.thorium,
            image,
            job,
            &image.dependencies.ephemeral.location,
            &mut self.logs,
        )
        .await?;
        self.repos = setup::download_repos(
            &self.thorium,
            image,
            job,
            &image.dependencies.repos.location,
            commits,
            &mut self.logs,
        )
        .await?;
        self.results = setup::download_results(
            &self.thorium,
            image,
            job,
            &image.dependencies.results.location,
            &mut self.logs,
        )
        .await?;
        // only download tags if its enabled
        if image.dependencies.tags.enabled {
            // download tags for our samples/repos
            self.tags = setup::download_tags(
                &self.thorium,
                image,
                job,
                &image.dependencies.tags.location,
                &mut self.logs,
            )
            .await?;
        }
        // only download children if its enabled
        if image.dependencies.children.enabled {
            // download any prior children
            self.children = setup::download_children(
                &self.thorium,
                image,
                job,
                &image.dependencies.children.location,
                &mut self.logs,
            )
            .await?;
        }
        Ok(())
    }

    /// Start executing a single job in Thorium
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to execute a job for
    /// * `job` - The job we are executing
    /// * `log_file` - Where to write job logs to
    #[instrument(
        name = "AgentExecutor<K8s>::execute",
        skip(self, image, job),
        err(Debug)
    )]
    async fn execute(
        &mut self,
        image: &Image,
        job: &GenericJob,
        log_path: &str,
    ) -> Result<InFlight, Error> {
        // get the correct way to pass our dependency paths to our job
        let sample_args = build_path_args!(job.samples, self.samples, image.dependencies.samples);
        let ephemeral_args =
            build_path_args!(job.ephemeral, self.ephemerals, image.dependencies.ephemeral);
        let repo_args = build_path_args!(job.repos, self.repos, image.dependencies.repos, url);
        let result_args = build_path_args!(
            image.dependencies.results.images,
            self.results,
            image.dependencies.results
        );
        let tag_args = build_path_args!(self.tags, image.dependencies.tags);
        let children_args = build_path_args!(self.children, image.dependencies.children);
        // build command to execute
        let built = registry::Cmd::new(image, job, &self.entrypoint, &self.cmd)
            .build(
                sample_args,
                ephemeral_args,
                repo_args,
                result_args,
                tag_args,
                children_args,
                &image.output_collection.files.results,
                image,
                job,
                &mut self.logs,
            )
            .await?;
        log!(self.logs, built.join(" "));
        // create a file to buffer our logs in
        let log_file = std::fs::File::create(log_path)?;
        // if we are in a windows container then adjust our built command
        let built = if self.windows {
            // prepend the cmd.exe command
            let mut prepended = vec!["C:\\Windows\\system32\\cmd.exe".to_owned(), "/C".to_owned()];
            prepended.extend(built.into_iter());
            prepended
        } else {
            built
        };
        event!(Level::INFO, cmd = built.join(" "));
        // execute built command
        let cmd_spawn = Command::new(built[0].clone())
            .args(&built[1..])
            .stdout(log_file.try_clone()?)
            .stderr(log_file)
            .spawn();
        // log any errors
        match cmd_spawn {
            Ok(child) => Ok(InFlight::Child(child)),
            // we failed to execute this entrypoint/command
            Err(error) => {
                // log this was a entrypoint/command execution error
                log!(self.logs, "Failed to execute entrypoint/command");
                // return our error
                Err(Error::from(error))
            }
        }
    }

    /// Collect any result files from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect results for
    /// * `job` - The job to collecting results for
    #[instrument(name = "AgentExecutor<K8s>::results", skip_all, err(Debug))]
    async fn results(&mut self, image: &Image) -> Result<RawResults, Error> {
        // collect any results from the default location
        results::collect(
            image,
            &image.output_collection.files.results,
            &image.output_collection.files.result_files,
            &mut self.logs,
        )
        .await
    }

    /// Collect any tags from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect tags for
    /// * `job` - The job we are executing
    /// * `raw` - The results to extract tags from
    #[instrument(name = "AgentExecutor<K8s>::tags", skip_all, err(Debug))]
    async fn tags(
        &mut self,
        image: &Image,
        job: &GenericJob,
        raw: &RawResults,
    ) -> Result<TagBundle, Error> {
        // all results are the same so get the first result
        tags::collect(
            job,
            raw,
            &image.output_collection,
            &image.output_collection.files.tags,
            &mut self.logs,
        )
        .await
    }

    /// Collect any children files from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect children for
    #[instrument(name = "AgentExecutor<K8s>::children", skip_all, err(Debug))]
    async fn children(&mut self, image: &Image) -> Result<Children, Error> {
        // collect any children from the default location
        Children::collect(&image.output_collection.children, &mut self.logs).await
    }

    /// Clean up after this job
    ///
    /// # Arguments
    ///
    /// * `image` - The image we are cleaning up a job for
    #[instrument(name = "AgentExecutor<K8s>::clean_up", skip_all, err(Debug))]
    async fn clean_up(&mut self, image: &Image, _: &GenericJob) -> Result<(), Error> {
        // purge any dependency paths
        purge!(image.dependencies.samples.location);
        purge!(image.dependencies.ephemeral.location);
        purge!(image.dependencies.results.location);
        purge!(image.dependencies.repos.location);
        purge!(image.dependencies.tags.location);
        // remove any results files/dirs
        purge!(image.output_collection.files.results);
        purge!(image.output_collection.files.result_files);
        purge!(image.output_collection.files.tags);
        // remove any children files/dirs
        purge!(image.output_collection.children);
        Ok(())
    }
}
