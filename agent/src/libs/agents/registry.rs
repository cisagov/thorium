//! Inspects docker images in registries

use crossbeam::channel::Sender;
use path_clean::PathClean;
use std::path::PathBuf;
use thorium::{
    models::{ArgStrategy, GenericJob, GenericJobKwargs, GenericJobOpts, Image, KwargDependency},
    Error,
};
use tracing::instrument;

use crate::log;

/// The overlayed command to execute
#[derive(Debug)]
pub struct Cmd {
    /// The job specified positional args
    pub positionals: Vec<String>,
    /// The job specified keyword args
    pub kwargs: GenericJobKwargs,
    /// The job specified switch args
    pub switches: Vec<String>,
    /// The job specified options
    pub opts: GenericJobOpts,
    /// The source command from inspecting the docker image
    pub src: Vec<String>,
    /// The command built from overlaying our values on top of the source command
    pub built: Vec<String>,
}

impl Cmd {
    /// Build a new command object
    ///
    /// # Arguments
    ///
    /// * `image` - The image that we are building a command for
    /// * `job` - The job to build a command to execute from
    /// * `entrypoint` - The original entrypoint for our container
    /// * `cmd` - The original command for our container
    pub fn new(image: &Image, job: &GenericJob, entrypoint: &[String], cmd: &[String]) -> Self {
        // build our command object
        let mut cmd = Cmd {
            positionals: job.args.positionals.clone(),
            kwargs: job.args.kwargs.clone(),
            switches: job.args.switches.clone(),
            opts: job.args.opts.clone(),
            src: cmd.to_owned(),
            built: entrypoint.to_owned(),
        };
        // if this job is a generator then inject in the job and reaction id kwarg
        if job.generator {
            cmd.kwargs.insert("--job".into(), vec![job.id.to_string()]);
            cmd.kwargs
                .insert("--reaction".into(), vec![job.reaction.to_string()]);
        }
        // add any repo kwargs
        cmd.inject_repo_kwargs(image, job);
        cmd
    }

    /// Add the repo and commit kwargs to this command if required
    ///
    /// # Arguments
    ///
    /// * `image` - The image that we are building a command for
    /// * `job` - The job to build a command to execute from
    fn inject_repo_kwargs(&mut self, image: &Image, job: &GenericJob) {
        // add the repo kwargs if its set
        if let Some(repo_kwarg) = &image.args.repo {
            // crawl the repo dependencies for this job and build a list of repos
            let mut repos = Vec::with_capacity(job.repos.len());
            for repo_dependency in &job.repos {
                // add this repos url
                repos.push(repo_dependency.url.clone());
            }
            // add this kwarg to our command
            self.kwargs.insert(repo_kwarg.to_owned(), repos);
        }
        // add the commit kwargs if its set
        if let Some(commit_kwarg) = &image.args.commit {
            // crawl the repo dependencies for this job and build a list of commits
            let mut repos = Vec::with_capacity(job.repos.len());
            for repo_dependency in &job.repos {
                // get this repo dependencies commitish
                if let Some(commitish) = &repo_dependency.commitish {
                    // add this repos commit
                    repos.push(commitish.clone());
                }
            }
            // add this kwarg to our command
            self.kwargs.insert(commit_kwarg.to_owned(), repos);
        }
    }

    /// Overlays positional args from the job and source into the built command
    fn inject_positionals(&mut self) {
        // check if we should override all positional args
        // if we are overriding them then don't bother injecting the args from
        // the docker file
        // keep track of number of args to remove from list
        let mut consumed = 0;
        // add args until we get to an arg that starts with a '-'
        for arg in self.src.iter() {
            if !arg.starts_with('-') {
                // only inject if override is disabled
                if !self.opts.override_positionals {
                    self.built.push(arg.to_owned());
                }
                consumed += 1;
            } else {
                // break out of loop since we hit a kwarg
                break;
            }
        }
        // remove all consumed args
        if consumed > 0 {
            self.src.drain(0..consumed);
        }

        // append all custom positional args if any were set
        if !self.positionals.is_empty() {
            self.built.append(&mut self.positionals);
        }
    }

    /// Expands a string into a key/value if it is a joint kwarg
    ///
    /// A joint kwarg is a key=value string
    ///
    /// # Arguments
    ///
    /// * `arg` - The string to expand if its a joint kwarg
    fn expander(arg: String) -> (String, Option<String>) {
        // check if this is a joint arg
        if arg.contains('=') {
            // split into a tuple containing the key and the value
            let (key, value) = arg.split_at(arg.find('=').unwrap());
            (key.to_owned(), Some(value[1..].to_owned()))
        } else {
            // this isn't a joint arg
            (arg, None)
        }
    }

    /// Overlays kwargs from the job and source into the built command
    fn inject_kwargs(&mut self) {
        // check if we should override all kwargs
        // if we are overriding them then don't bother injecting the args from
        // the docker file
        if !self.opts.override_kwargs {
            // if we are currently wiping args we have replaced
            let mut wipe = false;
            // add all args and overlay any user specified kwargs
            for arg in self.src.drain(0..) {
                // check if this is a key or a value
                if arg.starts_with('-') {
                    // reset wipe as we hit a new kwarg
                    wipe = false;
                    // expand this arg if its a joint arg
                    let (key, value) = Self::expander(arg);
                    // inject value source docker config
                    self.built.push(key.clone());
                    // check if this arg should be overridden
                    if self.kwargs.contains_key(&key) {
                        // enable wipe until hit another kwarg
                        wipe = true;
                        // override value if one was set
                        let mut new_value = self.kwargs.remove(&key).unwrap();
                        // override value with our own value
                        self.built.append(&mut new_value);
                    } else if value.is_some() {
                        // push value from docker info
                        self.built.push(value.unwrap());
                    }

                // values for kwarg
                } else {
                    // if wipe is false append value
                    if !wipe {
                        self.built.push(arg);
                    }
                }
            }
        }
        // append all left over custom kwargs args if any were set
        for (key, mut values) in self.kwargs.drain() {
            self.built.push(key);
            self.built.append(&mut values);
        }
    }

    /// Overlays switch args from the job and source into the built command
    pub fn inject_switches(&mut self) {
        // inject all switches if any were requested
        if !self.switches.is_empty() {
            self.built.append(&mut self.switches);
        }
    }

    /// inject any paths to files downloaded by the agent
    pub fn inject_paths(&mut self, mut paths: Vec<String>, kwarg: &Option<String>) {
        // inject in the paths to our samples if we have any
        if !paths.is_empty() {
            // inject in the right commands based on if we have a kwarg or not
            if let Some(kwarg) = &kwarg {
                // we have a kwarg arg to inject our sample inputs to
                // get an entry to insert our args into so we can append to user passed args
                let entry = self.kwargs.entry(kwarg.to_owned()).or_default();
                entry.append(&mut paths);
            } else {
                // no kwarg was set so just append this path to the built command as positionals
                self.built.append(&mut paths);
            }
        }
    }

    /// inject any paths to files downloaded by the agent
    ///
    /// # Arguments
    ///
    /// * `paths` - The paths to inject
    /// * `kwarg` - The kwarg settings for results
    /// * `logs` - The channel to send logs to
    pub async fn inject_result_paths(
        &mut self,
        mut paths: Vec<String>,
        kwarg: &KwargDependency,
        logs: &mut Sender<String>,
    ) -> Result<(), Error> {
        // inject in the paths to our samples if we have any
        if !paths.is_empty() {
            // inject our result paths based on our kwarg dependency settings
            match kwarg {
                KwargDependency::List(key) => {
                    // we have a kwarg arg to inject our sample inputs to
                    // get an entry to insert our result args into so we can append to user passed args
                    let entry = self.kwargs.entry(key.to_owned()).or_default();
                    entry.append(&mut paths);
                }
                KwargDependency::Map(map) => {
                    // we have specific kwargs for each result
                    for (tool_name, key) in map {
                        // check for each path whether there is a sub-directory for our tool,
                        // and if there is, add it to kwargs at this key
                        let mut found_paths = Vec::new();
                        for path in &paths {
                            let tool_path = PathBuf::from(path).join(tool_name);
                            if tokio::fs::try_exists(&tool_path)
                                .await
                                .map_err(|err| Error::new(
                                    format!(
                                        "Error checking if results for tool '{tool_name}' exist in path '{path}': {err}"
                                    ))
                                )?
                            {
                                found_paths.push(tool_path.to_string_lossy().to_string()) ;
                            } else {
                                log!(logs, "Results for tool '{}' not found in {}! Not adding to kwarg '{}'...", tool_name, path, key);
                            }
                        }
                        // add the paths were found to the kwargs only if we found any
                        if !found_paths.is_empty() {
                            let entry = self.kwargs.entry(key.to_owned()).or_default();
                            entry.append(&mut found_paths);
                        }
                    }
                }
                KwargDependency::None => {
                    self.built.append(&mut paths);
                }
            }
        }
        Ok(())
    }

    /// Inject a single argument
    pub fn overwrite_arg(&mut self, value: &str, strategy: &ArgStrategy) {
        // determine if we should set an output arg or not
        match strategy {
            ArgStrategy::None => (),
            ArgStrategy::Append => self.built.push(value.to_owned()),
            ArgStrategy::Kwarg(key) => {
                self.kwargs.insert(key.to_owned(), vec![value.to_owned()]);
            }
        }
    }

    /// Inject our reaction id if this image requires it
    ///
    /// # Arguments
    ///
    /// * `image` - The image we are running jobs for
    /// * `job` - The job to pull our reaction id from
    fn inject_reaction_id(&mut self, image: &Image, job: &GenericJob) {
        // inject our reaction id if we have a kwarg set for it
        if let Some(key) = &image.args.reaction {
            // get an entry to this kwargs values or create a default
            let entry = self.kwargs.entry(key.to_owned()).or_default();
            // add our reaction id as a string
            entry.push(job.reaction.to_string());
        }
    }

    /// Check if the source command is empty or only
    /// invokes a shell
    #[instrument(name = "Cmd::src_empty_or_shell", skip_all)]
    fn built_empty_or_shell(&self) -> bool {
        let prefixes = ["", "/bin", "/usr/bin", "/usr/local/bin"];
        let shells = ["sh", "bash", "zsh"];
        // create a list of paths to shells from all combinations of shells and prefixes
        let mut shell_paths: Vec<PathBuf> = Vec::new();
        for prefix in prefixes {
            for shell in shells {
                shell_paths.push([prefix, shell].iter().collect());
            }
        }
        // ensure src isn't empty
        match self.built.first() {
            Some(first_arg) => {
                if self.built.len() == 1 {
                    // clean the src cmd path, removing extra /'s or \'s, .'s, and unnecessary ..'s
                    let cmd_path = PathBuf::from(first_arg).clean();
                    // src only invokes a shell if its path is one of the shell paths and it's the only argument
                    shell_paths.contains(&cmd_path)
                } else {
                    false
                }
            }
            None => true,
        }
    }

    /// Builds the overlayed command to execute
    ///
    /// # Arguments
    ///
    /// * `samples` - The paths to inject to any downloaded samples
    /// * `ephemeral` - The paths to inject to any downloaded ephemeral files
    /// * `repos` - The paths to inject to any downloaded repos
    /// * `output` - The path to retrieve tool output from
    /// * `image` - The image in use for this job
    /// * `job` - The job we are building a command for
    /// * `logs` - The channel to send logs to
    #[instrument(name = "Cmd::build", skip_all, err(Debug))]
    pub async fn build(
        mut self,
        samples: Vec<String>,
        ephemeral: Vec<String>,
        repos: Vec<String>,
        results: Vec<String>,
        tags: Vec<String>,
        children: Vec<String>,
        output: &str,
        image: &Image,
        job: &GenericJob,
        logs: &mut Sender<String>,
    ) -> Result<Vec<String>, Error> {
        // if command override is specified short circuit to that
        if let Some(override_cmd) = self.opts.override_cmd {
            return Ok(override_cmd);
        }
        // inject our command if it exists
        self.built.append(&mut self.src);
        // throw an error if the src command is empty to avoid simply running the sample naively
        if self.built_empty_or_shell() {
            return Err(Error::new(
                "The image entrypoint cannot be empty or only invoke a shell",
            ));
        }
        // inject in the paths to our samples and ephemeral files if we have any
        self.inject_paths(samples, &image.dependencies.samples.kwarg);
        self.inject_paths(ephemeral, &image.dependencies.ephemeral.kwarg);
        self.inject_paths(repos, &image.dependencies.repos.kwarg);
        self.inject_result_paths(results, &image.dependencies.results.kwarg, logs)
            .await?;
        self.inject_paths(tags, &image.dependencies.tags.kwarg);
        self.inject_paths(children, &image.dependencies.children.kwarg);
        // inject in our reaction id if its requested
        self.inject_reaction_id(image, job);
        // add our output path if its set
        self.overwrite_arg(output, &image.args.output);
        // overlay custom args on top of the docker images entrypoint/cmd
        // inject any positional args
        self.inject_positionals();
        // inject any kwargs
        self.inject_kwargs();
        // inject any switches
        self.inject_switches();
        Ok(self.built)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::prelude::*;
    use std::collections::{HashMap, HashSet};
    use thorium::models::{
        ChildFilters, CommitishKinds, Dependencies, GenericJob, GenericJobArgs, Image, ImageArgs,
        ImageVersion, JobStatus, OutputCollection, OutputDisplayType, RepoDependency, Resources,
        ResultDependencySettings, SecurityContext,
    };
    use uuid::Uuid;

    fn generate_job() -> GenericJob {
        // generate a test job with empty args
        GenericJob {
            reaction: Uuid::new_v4(),
            id: Uuid::new_v4(),
            group: "TestGroup".into(),
            pipeline: "TestPipeline".into(),
            stage: "TestStage".into(),
            creator: "mcarson".into(),
            args: GenericJobArgs::default(),
            status: JobStatus::Running,
            deadline: Utc::now(),
            parent: None,
            generator: false,
            samples: vec!["sample1".into(), "sample2".into()],
            ephemeral: vec!["file.txt".into(), "other.txt".into()],
            parent_ephemeral: HashMap::default(),
            repos: vec![RepoDependency {
                url: "github.com/curl/curl".into(),
                commitish: Some("master".into()),
                kind: Some(CommitishKinds::Branch),
            }],
            trigger_depth: None,
        }
    }

    fn generate_image() -> Image {
        // generate a test image
        Image {
            group: "TestGroup".into(),
            name: "TestImage".into(),
            version: Some(ImageVersion::SemVer(
                semver::Version::parse("1.0.0").unwrap(),
            )),
            creator: "mcarson".into(),
            image: Some("alpine:latest".into()),
            scaler: thorium::models::ImageScaler::K8s,
            lifetime: None,
            timeout: None,
            resources: Resources::default(),
            spawn_limit: thorium::models::SpawnLimits::Unlimited,
            env: HashMap::default(),
            args: ImageArgs::default(),
            runtime: 600.0,
            volumes: Vec::default(),
            modifiers: None,
            description: None,
            security_context: SecurityContext::default(),
            used_by: Vec::default(),
            collect_logs: true,
            generator: false,
            dependencies: Dependencies::default(),
            display_type: OutputDisplayType::default(),
            output_collection: OutputCollection::default(),
            child_filters: ChildFilters::default(),
            clean_up: None,
            kvm: None,
            bans: HashMap::default(),
            network_policies: HashSet::default(),
        }
    }

    /// Initializes a Vec of String
    macro_rules! vec_string {
        ($($remaining:expr),*) => {
            vec![$($remaining.to_string()),*]
        }
    }

    /// Initialize of slice of String
    macro_rules! slice_string {
        ($($remaining:expr),*) => {
            &[$($remaining.to_string()),*]
        }
    }

    /// Test a barebones job with no overlays
    #[tokio::test]
    async fn empty() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let job = generate_job();
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(built, vec_string!["/usr/bin/python3", "corn.py"]);
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!["/usr/bin/python3", "corn.py", "pos1", "pos2"]
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!("/usr/bin/python3", "corn.py", "--1", "1")
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_samples() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let job = generate_job();
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!["/usr/bin/python3", "corn.py", "sample1", "sample2"]
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_samples() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "sample1",
                "sample2",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_samples() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "sample1",
                "sample2",
                "--1",
                "1"
            )
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_samples_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.samples.kwarg = Some("--inputs".into());
        let job = generate_job();
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--inputs",
                "sample1",
                "sample2"
            )
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_samples_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.samples.kwarg = Some("--inputs".into());
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "--inputs",
                "sample1",
                "sample2"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_samples_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.samples.kwarg = Some("--inputs".into());
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--inputs", vec!["sample0"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--inputs",
                "sample0",
                "sample1",
                "sample2"
            )
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_ephemerals() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let job = generate_job();
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!["/usr/bin/python3", "corn.py", "file.txt", "other.txt"]
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_ephemerals() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "file.txt",
                "other.txt",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_ephemerals() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "file.txt",
                "other.txt",
                "--1",
                "1"
            )
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_ephemerals_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        let job = generate_job();
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--ephemeral",
                "file.txt",
                "other.txt"
            )
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_ephemerals_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "--ephemeral",
                "file.txt",
                "other.txt"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_ephemerals_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--ephemeral", vec!["first.txt"]);
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--ephemeral",
                "first.txt",
                "file.txt",
                "other.txt"
            )
        );
    }

    /// Test a generator job with keyword args
    #[tokio::test]
    async fn generator_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // make this job a generator
        job.generator = true;
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        // build our command
        let built = cmd
            .build(
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        // convert built to a set so that order does not matter
        let built_set: HashSet<_> = built.iter().collect();
        // get this jobs id
        let id = job.id.to_string();
        let reaction = job.reaction.to_string();
        // build the expected args
        let args = vec_string!(
            "/usr/bin/python3",
            "corn.py",
            "--1",
            "1",
            "--reaction",
            &reaction,
            "--job",
            &id
        );
        // convert args to a set so that order does not matter
        let args_set: HashSet<_> = args.iter().collect();
        // make sure our command has the same number of args before we cast it to a set
        assert_eq!(built.len(), args.len());
        // make sure we have the same args
        assert_eq!(built_set, args_set);
    }

    /// Test a job with switch overlays
    #[tokio::test]
    async fn switches() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.switch("--corn").switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!("/usr/bin/python3", "corn.py", "--corn", "--beans")
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--1", vec!["1"])
            .switch("--corn")
            .switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "--1",
                "1",
                "--corn",
                "--beans"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_samples() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--1", vec!["1"])
            .switch("--corn")
            .switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "sample1",
                "sample2",
                "pos1",
                "pos2",
                "--1",
                "1",
                "--corn",
                "--beans"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_samples_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.samples.kwarg = Some("--inputs".into());
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--inputs", vec!["sample0"])
            .switch("--corn")
            .switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "--inputs",
                "sample0",
                "sample1",
                "sample2",
                "--corn",
                "--beans"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_ephemeral() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--1", vec!["1"])
            .switch("--corn")
            .switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "file.txt",
                "other.txt",
                "pos1",
                "pos2",
                "--1",
                "1",
                "--corn",
                "--beans"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_ephemeral_kwargs() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--ephemeral", vec!["first.txt"])
            .switch("--corn")
            .switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                vec![],
                job.ephemeral.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "--ephemeral",
                "first.txt",
                "file.txt",
                "other.txt",
                "--corn",
                "--beans"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_all() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--1", vec!["1"])
            .switch("--corn")
            .switch("--beans");
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                job.ephemeral.clone(),
                vec!["curl".to_owned()],
                vec!["other-tool".to_owned()],
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "sample1",
                "sample2",
                "file.txt",
                "other.txt",
                "curl",
                "other-tool",
                "pos1",
                "pos2",
                "--1",
                "1",
                "--corn",
                "--beans"
            )
        );
    }

    // Test a job where the image has result dependencies set to 'Map', but
    // the dependencies only have results from one of the images
    #[tokio::test]
    async fn results_map() {
        // create a temporary log channel
        let (mut logs_tx, _logs_rx) = crossbeam::channel::unbounded::<String>();
        // generate an image
        let mut image = generate_image();
        // give the image result dependencies configured to map to kwargs
        image.dependencies = image.dependencies.results(
            ResultDependencySettings::default()
                .images(vec!["image1", "image2"])
                .kwarg(KwargDependency::Map(
                    [
                        ("image1".to_string(), "--image1-results".to_string()),
                        ("image2".to_string(), "--image2--results".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                )),
        );
        // generate a job
        let job = generate_job();
        // add paths for each of our samples/repos as result dependencies
        let test_dir = PathBuf::from("/tmp/thorium/testing");
        let results_dir = test_dir.join("prior-results");
        let results = job
            .samples
            .iter()
            .map(|sample| results_dir.join(sample))
            .chain(job.repos.iter().map(|repo| results_dir.join(&repo.url)))
            .collect::<Vec<PathBuf>>();
        // create sub-directories in the results dir for image1, but not for image2
        for dir in &results {
            let dir = dir.join("image1");
            tokio::fs::create_dir_all(&dir).await.unwrap();
        }
        // convert to String's
        let results = results
            .into_iter()
            .map(|dir| dir.to_string_lossy().into())
            .collect();
        let cmd = Cmd::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        );
        let built = cmd
            .build(
                job.samples.clone(),
                vec![],
                vec![],
                results,
                vec![],
                vec![],
                "results",
                &image,
                &job,
                &mut logs_tx,
            )
            .await
            .unwrap();
        assert_eq!(
            built,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "sample1",
                "sample2",
                "--image1-results",
                "/tmp/thorium/testing/prior-results/sample1/image1",
                "/tmp/thorium/testing/prior-results/sample2/image1",
                "/tmp/thorium/testing/prior-results/github.com/curl/curl/image1"
            )
        );
        // remove the test directory
        tokio::fs::remove_dir_all(&test_dir).await.unwrap();
    }
}
