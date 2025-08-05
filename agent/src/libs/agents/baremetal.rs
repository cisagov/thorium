//! Implements the Thorium agent for bare-metal jobs

use crossbeam::channel::Sender;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thorium::models::{DependencyPassStrategy, GenericJob, Image};
use thorium::{Error, Thorium};
use tokio::process::Command;
use tracing::{event, instrument, Level};

use super::{registry, setup, AgentExecutor, InFlight};
use crate::libs::children::{self, Children};
use crate::libs::{results, tags, RawResults, TagBundle, Target};
use crate::{build_path_args, log, purge, purge_parent};

/// Isolate a path to target folder or file
///
/// # Arguments
///
/// * `raw` - The path to isolate
/// * `id` - The job id to append
fn isolate<P: AsRef<Path>>(raw: P, id: &str) -> Result<PathBuf, Error> {
    let path = raw.as_ref();
    // determine if this path has a target folder or not
    if path == Path::new("/tmp/thorium") {
        // the path to isolate is just the default Thorium path so just add our job id
        Ok(path.join(id).to_path_buf())
    } else {
        // a target path exists so insert our final job id before the final segment
        // get the parent
        match path.file_name() {
            // build a path with the parent
            Some(name) => Ok(path.parent().unwrap().join(id).join(name).to_path_buf()),
            None => Err(Error::new(format!(
                "{} cannot be isolated by job",
                path.to_string_lossy()
            ))),
        }
    }
}

/// An execution of a single job in k8s
pub struct BareMetal {
    /// A client to Thorium
    pub thorium: Thorium,
    /// A sender for a channel of logs to add for this job
    pub logs: Sender<String>,
    /// The path to write sample dependencies to
    pub samples_path: PathBuf,
    /// The path to write ephemeral dependencies to
    pub ephemerals_path: PathBuf,
    /// The path to write repo dependencies to
    pub repos_path: PathBuf,
    /// The path to write result dependencies to
    pub results_dep_path: PathBuf,
    /// The path to write tag dependencies to
    pub tags_dep_path: PathBuf,
    /// The path to write children dependencies to
    pub children_dep_path: PathBuf,
    /// The path to write results to
    pub results_path: PathBuf,
    /// The path to write result files to
    pub result_files_path: PathBuf,
    /// The path to write tags to
    pub tags_path: PathBuf,
    /// The path to write children to
    pub children_path: PathBuf,
    /// The paths to any downloaded sample files
    samples: Vec<PathBuf>,
    /// The paths to any downloaded ephemeral files
    ephemerals: Vec<PathBuf>,
    /// The paths to any downloaded repos
    repos: Vec<PathBuf>,
    /// The paths to any downloaded repos
    results: Vec<PathBuf>,
    /// The paths to any downloaded tags
    tags: Vec<PathBuf>,
    /// The paths to any downloaded children
    children: Vec<PathBuf>,
}

impl BareMetal {
    /// Create a new k8s agent for executing a single job
    pub fn new(target: &Target, job: &GenericJob, logs: Sender<String>) -> Result<Self, Error> {
        // get our job id as a string
        let id = job.id.to_string();
        // build the paths setup
        let samples_path = isolate(&target.image.dependencies.samples.location, &id)?;
        let ephemerals_path = isolate(&target.image.dependencies.ephemeral.location, &id)?;
        let repos_path = isolate(&target.image.dependencies.repos.location, &id)?;
        let results_dep_path = isolate(&target.image.dependencies.results.location, &id)?;
        let tags_dep_path = isolate(&target.image.dependencies.tags.location, &id)?;
        let children_dep_path = isolate(&target.image.dependencies.children.location, &id)?;
        let results_path = isolate(&target.image.output_collection.files.results, &id)?;
        let result_files_path = isolate(&target.image.output_collection.files.result_files, &id)?;
        let tags_path = isolate(&target.image.output_collection.files.tags, &id)?;
        let children_path = isolate(&target.image.output_collection.children, &id)?;
        // build our bare-metal object
        let bare_metal = BareMetal {
            thorium: target.thorium.clone(),
            logs,
            samples_path,
            ephemerals_path,
            repos_path,
            results_dep_path,
            tags_dep_path,
            children_dep_path,
            results_path,
            result_files_path,
            tags_path,
            children_path,
            samples: Vec::default(),
            ephemerals: Vec::default(),
            repos: Vec::default(),
            results: Vec::default(),
            tags: Vec::default(),
            children: Vec::default(),
        };
        Ok(bare_metal)
    }
}

/// The methods used to launch and monitor a job in Thorium
#[async_trait::async_trait]
impl AgentExecutor for BareMetal {
    /// Get the paths to this executors current jobs results and result files
    fn result_paths(&self, _: &Image) -> (String, String) {
        // get our paths as strings
        let results = self.results_path.to_string_lossy().to_string();
        let result_files = self.result_files_path.to_string_lossy().to_string();
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
        name = "AgentExecutor<BareMetal>::setup",
        skip_all,
        field(
            samples = self.samples_path.to_string_lossy(),
            ephemeral = self.ephemeral_path.to_string_lossy(),
            repos = self.repos_path.to_string_lossy(),
            results = self.results_dep_path.to_string_lossy(),
            tags = self.tags_dep_path.to_string_lossy(),
            collect_results = self.results_path.to_string_lossy(),
            collect_result_files = self.result_files_path.to_string_lossy(),
            collect_tags = self.tags_path.to_string_lossy(),
        ),
        err(Debug))]
    async fn setup(
        &mut self,
        image: &Image,
        job: &GenericJob,
        commits: &mut HashMap<String, String>,
    ) -> Result<(), Error> {
        // purge any paths that might contain dependencies
        purge!(self.samples_path);
        purge!(self.ephemerals_path);
        purge!(self.repos_path);
        // purge any paths that might contain results
        purge!(self.results_path);
        purge!(self.result_files_path);
        purge!(self.tags_path);
        purge!(self.children_path);
        // setup dependency base paths that are isolated by job IDs
        std::fs::create_dir_all(&self.samples_path)?;
        std::fs::create_dir_all(&self.ephemerals_path)?;
        std::fs::create_dir_all(&self.repos_path)?;
        // setup results/tags/children paths
        std::fs::create_dir_all(&self.results_path.parent().unwrap())?;
        std::fs::create_dir_all(&self.result_files_path)?;
        std::fs::create_dir_all(&self.tags_path)?;
        // build the paths for storing children files
        children::setup(&self.children_path).await?;
        // download any data required for this job
        self.samples = setup::download_samples(
            &self.thorium,
            image,
            job,
            &self.samples_path,
            &mut self.logs,
        )
        .await?;
        self.ephemerals = setup::download_ephemeral(
            &self.thorium,
            image,
            job,
            &self.ephemerals_path,
            &mut self.logs,
        )
        .await?;
        setup::download_parent_ephemeral(
            &mut self.ephemerals,
            &self.thorium,
            image,
            job,
            &self.ephemerals_path,
            &mut self.logs,
        )
        .await?;
        self.repos = setup::download_repos(
            &self.thorium,
            image,
            job,
            &self.repos_path,
            commits,
            &mut self.logs,
        )
        .await?;
        self.results = setup::download_results(
            &self.thorium,
            image,
            job,
            &self.results_dep_path,
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
                &self.tags_dep_path,
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
                &self.children_dep_path,
                &mut self.logs,
            )
            .await?;
        }
        Ok(())
    }

    /// Execute a single job in Thorium
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to execute a job for
    /// * `job` - The job we are executing
    /// * `log_file` - Where to write job logs to
    #[instrument(
        name = "AgentExecutor<BareMetal>::execute",
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
        // build the path to our results dir
        let output = self
            .results_path
            .parent()
            .unwrap()
            .to_string_lossy()
            .to_string();
        // start building the command we want to execute
        let cmd = match (&image.args.entrypoint, &image.args.command) {
            (Some(ep), Some(cmd)) => registry::Cmd::new(image, job, ep, cmd),
            (Some(ep), None) => registry::Cmd::new(image, job, ep, &[]),
            _ => {
                return Err(Error::new(
                    "Bare Metal jobs require an entrypoint!".to_owned(),
                ))
            }
        };
        // build the command we want to execute
        let built = cmd
            .build(
                sample_args,
                ephemeral_args,
                repo_args,
                result_args,
                tag_args,
                children_args,
                &output,
                image,
                job,
                &mut self.logs,
            )
            .await?;
        // cast our command to a str
        let built_str = built.join(" ");
        // log the command we are executing
        log!(self.logs, built_str);
        event!(Level::INFO, cmd = built_str);
        // open a file handle to this file
        let log_file = std::fs::File::create(log_path)?;
        // execute built command
        let child = Command::new(built[0].clone())
            .args(&built[1..])
            .stdout(log_file.try_clone()?)
            .stderr(log_file)
            .spawn()?;
        Ok(InFlight::Child(child))
    }

    /// Collect any result from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect results for
    /// * `job` - The job to collecting results for
    #[instrument(name = "AgentExecutor<BareMetal>::results", skip_all, err(Debug))]
    async fn results(&mut self, image: &Image) -> Result<RawResults, Error> {
        // collect any results from the default location
        results::collect(
            image,
            &self.results_path,
            &self.result_files_path,
            &mut self.logs,
        )
        .await
    }

    /// Collect any tags from a completed job and submit both them and any results
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect tags for
    /// * `job` - The job we are executing
    /// * `raw` - The results to extract tags from
    #[instrument(name = "AgentExecutor<BareMetal>::tags", skip_all, err(Debug))]
    async fn tags(
        &mut self,
        image: &Image,
        job: &GenericJob,
        raw: &RawResults,
    ) -> Result<TagBundle, Error> {
        tags::collect(
            job,
            &raw,
            &image.output_collection,
            &self.tags_path,
            &mut self.logs,
        )
        .await
    }

    /// Collect any children files from a completed job
    ///
    /// # Arguments
    ///
    /// * `image` - The Image to collect children for
    #[instrument(name = "AgentExecutor<BareMetal>::children", skip_all, err(Debug))]
    async fn children(&mut self, _: &Image) -> Result<Children, Error> {
        // collect any children from the default location
        Children::collect(&self.children_path, &mut self.logs).await
    }

    /// Clean up after this job
    #[instrument(name = "AgentExecutor<BareMetal>::clean_up", skip_all, err(Debug))]
    async fn clean_up(&mut self, _: &Image, _: &GenericJob) -> Result<(), Error> {
        // remove any paths for this job
        purge_parent!(self.samples_path);
        purge_parent!(self.ephemerals_path);
        purge_parent!(self.repos_path);
        purge_parent!(self.results_dep_path);
        purge_parent!(self.tags_dep_path);
        purge_parent!(self.results_path);
        purge_parent!(self.result_files_path);
        purge_parent!(self.tags_path);
        purge_parent!(self.children_path);
        Ok(())
    }
}
