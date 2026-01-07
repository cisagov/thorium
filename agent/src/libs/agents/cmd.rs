//! Inspects docker images in registries

use crossbeam::channel::Sender;
use path_clean::PathClean;
use std::path::PathBuf;
use thorium::{
    Error,
    models::{
        ArgStrategy, ChildrenDependencySettings, DependencyPassStrategy,
        EphemeralDependencySettings, GenericJob, GenericJobKwargs, GenericJobOpts, Image,
        KwargDependency, OutputHandler, RepoDependency, RepoDependencySettings,
        ResultDependencySettings, SampleDependencySettings, TagDependencySettings,
        images::CacheDependencySettings,
    },
};
use tracing::instrument;

use crate::libs::DownloadedCache;
use crate::log;

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

/// A builder for commands in Thorium
#[derive(Debug)]
pub struct CmdBuilder {
    /// The job specified positional args
    positionals: Vec<String>,
    /// The job specified keyword args
    kwargs: GenericJobKwargs,
    /// The job specified switch args
    switches: Vec<String>,
    /// The job specified options
    opts: GenericJobOpts,
    /// The original entry point for this worker
    entrypoint: Vec<String>,
    /// The original command for this worker
    cmd: Vec<String>,
    /// The number of positional args to skip when overriding positionals
    allowable_positionals: u8,
}

impl CmdBuilder {
    /// Build a new command object
    ///
    /// # Arguments
    ///
    /// * `image` - The image we are executing a job for
    /// * `job` - The job to build a command to execute from
    /// * `entrypoint` - The original entrypoint for our container
    /// * `cmd` - The original command for our container
    pub fn new(image: &Image, job: &GenericJob, entrypoint: &[String], command: &[String]) -> Self {
        // clone our kwargs so we can add generator info to them if needed
        let mut kwargs = job.args.kwargs.clone();
        // use our images entrypoint if we have an override set
        let entrypoint = match &image.args.entrypoint {
            Some(entrypoint) => entrypoint.clone(),
            None => entrypoint.to_owned(),
        };
        // use our images cmd if we have an override set
        let cmd = match &image.args.command {
            Some(command) => command.clone(),
            None => command.to_owned(),
        };
        // if this job is a generator then inject in the job and reaction id kwarg
        if job.generator {
            kwargs.insert("--job".into(), vec![job.id.to_string()]);
            kwargs.insert("--reaction".into(), vec![job.reaction.to_string()]);
        }
        // if this job has a reaction kwarg set then add that
        if let Some(reaction_kwarg) = &image.args.reaction {
            // inject and replace any existing reaction id args
            kwargs.insert(reaction_kwarg.into(), vec![job.reaction.to_string()]);
        }
        // build our command object
        CmdBuilder {
            positionals: job.args.positionals.clone(),
            kwargs,
            switches: job.args.switches.clone(),
            opts: job.args.opts.clone(),
            entrypoint,
            cmd,
            allowable_positionals: 1,
        }
    }

    /// Store some arguments as either positional args or kwargs
    fn add_maybe_kwargs(&mut self, kwarg: Option<&String>, mut args: Vec<String>) {
        // if a kwarg was set then add it as a kwarg
        match kwarg {
            // since we have a kwarg arg add these args under that kwarg
            Some(kwarg) => {
                // get an entry to this kwargs values
                let entry = self.kwargs.entry(kwarg.clone()).or_default();
                // add the values for this kwarg
                entry.append(&mut args);
            }
            None => {
                self.positionals.append(&mut args);
            }
        }
    }

    /// Inject a single argument
    pub fn add_arg_by_strategy(&mut self, value: &str, strategy: &ArgStrategy) {
        // determine if we should set an output arg or not
        match strategy {
            ArgStrategy::None => (),
            ArgStrategy::Append => self.positionals.push(value.to_owned()),
            ArgStrategy::Kwarg(key) => {
                self.kwargs.insert(key.to_owned(), vec![value.to_owned()]);
            }
        }
    }

    /// Add either positional or keyword args for ephemerals
    ///
    /// # Arguments
    ///
    /// * `sha256s` - The sha256s for the ephemerals we have as dependencies
    /// * `paths` - The paths to any downloaded sha256s
    /// * `settings` - The settings to use for adding ephemeral dependencies to our command
    pub fn add_ephemeral(
        mut self,
        names: &[String],
        paths: &[PathBuf],
        settings: &EphemeralDependencySettings,
    ) -> Self {
        // get the ephemeral args formatted correctly
        let args = match settings.strategy {
            // convert the paths to our ephemerals to strings
            DependencyPassStrategy::Paths => paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            // just return the names we already have
            DependencyPassStrategy::Names => names.to_vec(),
            // just return a list containing our directory name
            DependencyPassStrategy::Directory => {
                vec![settings.location.clone()]
            }
            // don't return anything and short circuit
            DependencyPassStrategy::Disabled => return self,
        };
        // add this arg as a kwarg if we kwarg settings
        self.add_maybe_kwargs(settings.kwarg.as_ref(), args);
        self
    }

    /// Add either positional or keyword args for samples
    ///
    /// # Arguments
    ///
    /// * `paths` - The paths to any downloaded sha256s
    /// * `settings` - The settings to use for adding sample dependencies to our command
    pub fn add_samples(mut self, paths: &[PathBuf], settings: &SampleDependencySettings) -> Self {
        // get the samples args formatted correctly
        let args = match settings.strategy {
            // convert the paths to our samples to strings
            DependencyPassStrategy::Paths => paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            // just return the names we already have
            DependencyPassStrategy::Names => paths
                .iter()
                .filter_map(|path| path.file_name())
                .map(|fname| fname.to_string_lossy().into_owned())
                .collect(),
            // just return a list containing our directory name
            DependencyPassStrategy::Directory => {
                vec![settings.location.clone()]
            }
            // don't return anything and short circuit
            DependencyPassStrategy::Disabled => return self,
        };
        // add this arg as a kwarg if we kwarg settings
        self.add_maybe_kwargs(settings.kwarg.as_ref(), args);
        self
    }

    /// Add either positional or keyword args for repos
    ///
    /// # Arguments
    ///
    /// * `image` - The image we are executing
    /// * `repos` - The repos we have as dependencies
    /// * `paths` - The paths to any downloaded repos
    pub fn add_repos(mut self, image: &Image, repos: &[RepoDependency], paths: &[PathBuf]) -> Self {
        // get the repos args formatted correctly
        let args = match image.dependencies.repos.strategy {
            // convert the paths to our repos to strings
            DependencyPassStrategy::Paths => paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            // just return the urls we already have
            DependencyPassStrategy::Names => repos.iter().map(|repo| repo.url.clone()).collect(),
            // just return a list containing our directory name
            DependencyPassStrategy::Directory => {
                vec![image.dependencies.repos.location.clone()]
            }
            // don't return anything and short circuit
            DependencyPassStrategy::Disabled => return self,
        };
        // add this arg as a kwarg if we kwarg settings
        self.add_maybe_kwargs(image.dependencies.repos.kwarg.as_ref(), args);
        // if we have a repo kwarg set then add that
        if let Some(repo_kwarg) = &image.args.repo {
            // get an entry to this repo_kwargs values
            let entry = self.kwargs.entry(repo_kwarg.clone()).or_default();
            // add each repo url to our kwargs
            entry.extend(repos.iter().map(|repo| repo.url.clone()));
        }
        // if we have a commit kwarg set then add that
        if let Some(commit_kwarg) = &image.args.commit {
            // get an entry to this commit_kwargs values
            let entry = self.kwargs.entry(commit_kwarg.clone()).or_default();
            // add each commit url to our kwargs
            entry.extend(repos.iter().filter_map(|repo| repo.commitish.clone()));
        }
        self
    }

    /// Add either positional or keyword args for result dependencies
    ///
    /// # Arguments
    ///
    /// * `images` - The images for the results we have as dependencies
    /// * `paths` - The paths to any downloaded results
    /// * `settings` - The settings to use for adding result dependencies to our command
    /// * `logs` - The channel to send logs too
    pub async fn add_results(
        mut self,
        images: &[String],
        paths: &[PathBuf],
        settings: &ResultDependencySettings,
        logs: &mut Sender<String>,
    ) -> Result<Self, Error> {
        // get the results args formatted correctly
        let mut args = match settings.strategy {
            // convert the paths to our results to strings
            DependencyPassStrategy::Paths => paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            // just return the names we already have
            DependencyPassStrategy::Names => images.to_vec(),
            // just return a list containing our directory name
            DependencyPassStrategy::Directory => {
                vec![settings.location.clone()]
            }
            // don't return anything and short circuit
            DependencyPassStrategy::Disabled => return Ok(self),
        };
        // add this arg as a kwarg if we kwarg settings
        match &settings.kwarg {
            KwargDependency::List(key) => {
                // get an entry to insert our result args into so we can append to user passed args
                let entry = self.kwargs.entry(key.to_owned()).or_default();
                entry.append(&mut args);
            }
            KwargDependency::Map(map) => {
                // we have specific kwargs for each result
                for (tool_name, key) in map {
                    // check for each path whether there is a sub-directory for our tool,
                    // and if there is, add it to kwargs at this key
                    let mut found_paths = Vec::new();
                    for path in &args {
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
                self.positionals.append(&mut args);
            }
        }
        Ok(self)
    }

    /// Add either positional or keyword args for tags
    ///
    /// # Arguments
    ///
    /// * `paths` - The paths to any downloaded tags
    /// * `settings` - The settings to use for adding tag dependencies to our command
    pub fn add_tags(mut self, paths: &[PathBuf], settings: &TagDependencySettings) -> Self {
        // get the tags args formatted correctly
        let args = match settings.strategy {
            // convert the paths to our tags to strings
            DependencyPassStrategy::Paths => paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            // get the file names for all of our tag files
            DependencyPassStrategy::Names => paths
                .iter()
                .filter_map(|path| path.file_name())
                .map(|name| name.to_string_lossy().to_string())
                .collect(),
            // just return a list containing our directory name
            DependencyPassStrategy::Directory => {
                vec![settings.location.clone()]
            }
            // don't return anything and short circuit
            DependencyPassStrategy::Disabled => return self,
        };
        // add this arg as a kwarg if we kwarg settings
        self.add_maybe_kwargs(settings.kwarg.as_ref(), args);
        self
    }

    /// Add either positional or keyword args for children
    ///
    /// # Arguments
    ///
    /// * `paths` - The paths to any downloaded children
    /// * `settings` - The settings to use for adding tag dependencies to our command
    pub fn add_children(
        mut self,
        paths: &[PathBuf],
        settings: &ChildrenDependencySettings,
    ) -> Self {
        // get the children args formatted correctly
        let args = match settings.strategy {
            // convert the paths to our children to strings
            DependencyPassStrategy::Paths => paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect(),
            // get the file names for all of our tag files
            DependencyPassStrategy::Names => paths
                .iter()
                .filter_map(|path| path.file_name())
                .map(|name| name.to_string_lossy().to_string())
                .collect(),
            // just return a list containing our directory name
            DependencyPassStrategy::Directory => {
                vec![settings.location.clone()]
            }
            // don't return anything and short circuit
            DependencyPassStrategy::Disabled => return self,
        };
        // add this arg as a kwarg if we kwarg settings
        self.add_maybe_kwargs(settings.kwarg.as_ref(), args);
        self
    }

    /// Add our cache args
    pub fn add_cache(
        mut self,
        downloaded: &DownloadedCache,
        settings: &CacheDependencySettings,
    ) -> Self {
        // if we had any cache data to download get its location
        if let Some(location) = &downloaded.generic {
            // we have cache data so get the arg value properly formatted
            let args = match settings.generic.strategy {
                // convert the paths to our children to strings
                DependencyPassStrategy::Paths => vec![location.to_string_lossy().to_string()],
                // get the file names for all of our tag files
                DependencyPassStrategy::Names => {
                    // get the name for our cache location if we have one
                    // it should be impossible to not have one
                    match location.file_name() {
                        Some(name_path) => {
                            // convert our name to a string
                            let name = name_path.to_string_lossy().to_string();
                            // wrap our name in a vec
                            vec![name]
                        }
                        // just short circuit if we don't have a name somehow
                        None => return self,
                    }
                }
                // just return a list containing our directory name
                DependencyPassStrategy::Directory => {
                    // get the parent folder for our cache
                    match location.parent() {
                        Some(parent) => vec![parent.to_string_lossy().to_string()],
                        // just short circuit if we don't have a parent
                        None => return self,
                    }
                }
                // don't return anything and short circuit
                DependencyPassStrategy::Disabled => return self,
            };
            // add this arg as a kwarg if we kwarg settings
            self.add_maybe_kwargs(settings.generic.kwarg.as_ref(), args);
        }
        self
    }

    // modify our args for running in windows
    pub fn windows(mut self, is_windows: bool) -> Self {
        // if this is not a windows agent then don't do anything
        if !is_windows {
            return self;
        }
        // if we have an override command then just modify that
        if let Some(override_cmd) = &mut self.opts.override_cmd {
            // build the windows prepend command
            let mut prepended = vec!["C:\\Windows\\system32\\cmd.exe".to_owned(), "/C".to_owned()];
            // add our built commands back in
            prepended.append(override_cmd);
            // now replace our prependend override command
            *override_cmd = prepended;
        } else {
            // we don't have an override but we do need to override our entrypoint
            // build the windows prepend command
            let mut prepended = vec!["C:\\Windows\\system32\\cmd.exe".to_owned(), "/C".to_owned()];
            // add our built commands back in
            prepended.append(&mut self.entrypoint);
            // replace our entrypoint
            self.entrypoint = prepended;
        }
        self
    }

    /// Check if the source command is empty or only invokes a shell
    #[instrument(name = "Cmd::not_empty_or_just_shell", skip_all)]
    fn not_empty_or_just_shell(&self) -> Result<(), Error> {
        // make sure we have an entrypoint or command
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
        let should_error = match self.entrypoint.first() {
            Some(first_arg) => {
                if self.entrypoint.len() == 1 {
                    // clean the src cmd path, removing extra /'s or \'s, .'s, and unnecessary ..'s
                    let cmd_path = PathBuf::from(first_arg).clean();
                    // src only invokes a shell if its path is one of the shell paths and it's the only argument
                    shell_paths.contains(&cmd_path)
                } else {
                    false
                }
            }
            None => true,
        };
        // raise an error if this is not a valid entrypoint/command combo
        if should_error {
            Err(Error::new(
                "The image entrypoint cannot be empty or only invoke a shell",
            ))
        } else {
            // this is a valid entrypoint/cmd combo
            Ok(())
        }
    }

    /// Overlays kwargs from the job and source into the built command
    fn scan_args(&mut self, built: &mut Vec<String>) {
        // track if we are in a kwarg or not
        let mut in_kwarg = false;
        // if we are currently wiping args we have replaced
        let mut wipe = false;
        // add all args and overlay any user specified kwargs
        for arg in self.entrypoint.drain(0..) {
            // check if this is a key or a value
            if arg.starts_with('-') {
                // expand this arg if its a joint arg
                let (key, value) = expander(arg);
                // update our in kwarg tracker
                in_kwarg = value.is_none();
                // check if this is a kwarg that we want to replace
                if self.kwargs.contains_key(&key) {
                    // this is a kwarg that we want to replace
                    // if this is a concatenated "<key>=<value>" kwarg then we don't need to set wipe
                    wipe = value.is_none();
                    // get the values we want to override this with
                    let new_values = self.kwargs.remove(&key).unwrap();
                    // for each of our values add our kwarg
                    // if there are none then we want to wipe this kwarg when its found
                    for new_value in new_values {
                        // if our old key/value was joined with an '=' then do that here as well
                        if value.is_some() {
                            // build out concatenated key/value and add it
                            built.push(format!("{key}={new_value}"));
                        } else {
                            // add our key
                            built.push(key.clone());
                            // override value with our own value
                            built.push(new_value);
                        }
                    }
                } else {
                    // if kwarg overrides are on and this isn't a kwarg we passed then drop it
                    if self.opts.override_kwargs {
                        // set wipe to true if this is not a concatenated kwarg
                        wipe = value.is_none();
                        // skip to the next arg
                        continue;
                    }
                    // reset wipe as this is a kwarg that we don't want to change
                    wipe = false;
                    // readd our kwarg
                    // if our old key/value was joined with an '=' then do that here as well
                    match value {
                        // rebuild our concatenated key/value and add it
                        Some(value) => built.push(format!("{key}={value}")),
                        // this was originally two seperate values so keep it that way
                        None => {
                            // add our keywarg/switch
                            built.push(key.clone());
                        }
                    }
                }
            } else {
                // if wipe is false then add this other arg
                if !wipe && (in_kwarg || !self.opts.override_positionals)
                    // if we are overriding positionals then only allow one positional from entrypoint and cmd each
                    || (self.opts.override_positionals && self.allowable_positionals > 0)
                {
                    // add this non kwarg argument
                    built.push(arg);
                    // if we are overriding positonals then decrement allowable positionals
                    if self.opts.override_positionals && !wipe && !in_kwarg {
                        self.allowable_positionals -= 1;
                    }
                }
                // reset in kwarg and wipe to false so we only ever assume that a kwarg has one value
                in_kwarg = false;
                wipe = false;
            }
        }
        // swap our kwargs btree map with an empty one
        let kwarg_map = std::mem::take(&mut self.kwargs);
        // append all left over custom kwargs args if any were set
        for (key, values) in kwarg_map {
            // add our kwargs
            for value in values {
                // add our key
                built.push(key.clone());
                // add our value
                built.push(value);
            }
        }
        // append any switches that we still have
        built.append(&mut self.switches);
        // append any positional args that we have
        built.append(&mut self.positionals);
    }

    /// Build the final command to execute
    pub fn build(
        mut self,
        image: &Image,
        isolated_results: Option<&String>,
    ) -> Result<Vec<String>, Error> {
        // if our command is overridden then just return that
        if let Some(override_cmd) = self.opts.override_cmd {
            return Ok(override_cmd);
        }
        // add our output arg at the end to make sure it comes last if its a positional
        match image.output_collection.handler {
            OutputHandler::Files => {
                // use either our images output arg or an isolated version if we have one
                let results_path =
                    isolated_results.unwrap_or(&image.output_collection.files.results);
                // add our output arg if we need too
                self.add_arg_by_strategy(results_path, &image.args.output);
            }
        }
        // calculate how much space our kwargs will take
        let kwarg_capacity = self
            .kwargs
            .values()
            .fold(0, |acc, vals| acc + (vals.len() * 2));
        // calculate how large our command vec will be
        let capacity = self.entrypoint.len()
            + self.cmd.len()
            + kwarg_capacity
            + self.switches.len()
            + self.positionals.len();
        // if we have a command then increment our allowed positionals to 2
        self.allowable_positionals += 1;
        // add our cmd onto our entrypoint
        self.entrypoint.append(&mut self.cmd);
        // throw an error if the src command is empty to avoid simply running the sample naively
        self.not_empty_or_just_shell()?;
        // instance a command with approximately enough space for our fully built command
        let mut cmd = Vec::with_capacity(capacity);
        // start building our command
        // the command will be <entrypoint> <cmd> <kwargs> <switches> <positionals> <output positional>
        self.scan_args(&mut cmd);
        Ok(cmd)
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
            repos: vec![
                RepoDependency {
                    url: "github.com/curl/curl".into(),
                    commitish: Some("main".into()),
                    kind: Some(CommitishKinds::Branch),
                },
                RepoDependency {
                    url: "github.com/notcurl/notcurl".into(),
                    commitish: Some("main".into()),
                    kind: Some(CommitishKinds::Branch),
                },
            ],
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
        // generate an image
        let image = generate_image();
        // generate a job
        let job = generate_job();
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        assert_eq!(cmd, vec_string!["/usr/bin/python3", "corn.py"]);
    }

    /// Test a barebones job with no overlays
    #[tokio::test]
    async fn empty_output_positional() {
        // generate an image
        let mut image = generate_image();
        // set our output to be a positional arg
        image.args.output = ArgStrategy::Append;
        // generate a job
        let job = generate_job();
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        assert_eq!(
            cmd,
            vec_string!["/usr/bin/python3", "corn.py", "/tmp/thorium/results"]
        );
    }

    /// Test a barebones job with no overlays
    #[tokio::test]
    async fn empty_output_kwarg() {
        // generate an image
        let mut image = generate_image();
        // set our output to be a positional arg
        image.args.output = ArgStrategy::Kwarg("-o".to_owned());
        // generate a job
        let job = generate_job();
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        assert_eq!(
            cmd,
            vec_string!["/usr/bin/python3", "corn.py", "-o", "/tmp/thorium/results"]
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!["/usr/bin/python3", "corn.py", "pos1", "pos2"]
        );
    }

    /// Test a job with an entrypoint/command set in image settings
    #[tokio::test]
    async fn entrypoint_command() {
        // generate an image
        let mut image = generate_image();
        // set this images arg overlays
        image.args.entrypoint = Some(vec!["/usr/bin/bash".to_owned()]);
        image.args.command = Some(vec!["woot.sh".to_owned()]);
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        job.args.opts.override_positionals = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "old1", "old2"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(cmd, vec_string!["/usr/bin/bash", "woot.sh", "pos1", "pos2"]);
    }

    /// Test a job with a reaction kwarg set in image settings
    #[tokio::test]
    async fn reaction() {
        // generate an image
        let mut image = generate_image();
        // set this images arg overlays
        image.args.reaction = Some("--reaction".to_owned());
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        job.args.opts.override_positionals = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "old1", "old2"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string![
                "/usr/bin/python3",
                "corn.py",
                "--reaction",
                &job.reaction.to_string(),
                "pos1",
                "pos2"
            ]
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn repo_commit() {
        // generate an image
        let mut image = generate_image();
        // set this images arg overlays
        image.args.repo = Some("--repo".to_owned());
        image.args.commit = Some("--commit".to_owned());
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        job.args.opts.override_positionals = true;
        // build paths to our repos
        let repo_paths = vec![PathBuf::from("/tmp/repo1"), PathBuf::from("/tmp/repo2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_repos(&image, &job.repos, &repo_paths)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--commit",
                "main",
                "--commit",
                "main",
                "--repo",
                "github.com/curl/curl",
                "--repo",
                "github.com/notcurl/notcurl",
                "pos1",
                "pos2",
                "/tmp/repo1",
                "/tmp/repo2"
            )
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn override_positionals() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        job.args.opts.override_positionals = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "old1", "old2"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!["/usr/bin/python3", "corn.py", "pos1", "pos2"]
        );
    }

    /// Test a job with positional overlays with overridin positionals and kwargs
    #[tokio::test]
    async fn override_positionals_kwargs() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        job.args.opts.override_positionals = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "--keep=this", "old1", "old2"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!["/usr/bin/python3", "corn.py", "--keep=this", "pos1", "pos2"]
        );
    }

    /// Test a job with positional overlays with overridin positionals and kwargs
    #[tokio::test]
    async fn override_positionals_kwargs_split() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        // enable positional overrides
        job.args.opts.override_positionals = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "--keep", "this", "old1", "old2"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string![
                "/usr/bin/python3",
                "corn.py",
                "--keep",
                "this",
                "pos1",
                "pos2"
            ]
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(cmd, vec_string!("/usr/bin/python3", "corn.py", "--1", "1"));
    }

    /// Test a job with keyword args where we override kwargs
    #[tokio::test]
    async fn kwargs_override() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // enable kwarg overrides
        job.args.opts.override_kwargs = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "--drop=this"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(cmd, vec_string!("/usr/bin/python3", "corn.py", "--1", "1"));
    }

    /// Test a job with keyword args where we override kwargs
    #[tokio::test]
    async fn kwargs_override_with_positionals() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // enable kwarg overrides
        job.args.opts.override_kwargs = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "--drop=this", "pos1"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!("/usr/bin/python3", "corn.py", "pos1", "--1", "1")
        );
    }

    /// Test a job with keyword args where we override kwargs
    #[tokio::test]
    async fn kwargs_override_with_positionals_split() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // enable kwarg overrides
        job.args.opts.override_kwargs = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py", "--drop", "this", "pos1"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!("/usr/bin/python3", "corn.py", "pos1", "--1", "1")
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_samples() {
        // generate an image
        let image = generate_image();
        // generate a job
        let job = generate_job();
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string![
                "/usr/bin/python3",
                "corn.py",
                "/tmp/sample1",
                "/tmp/sample2"
            ]
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_samples() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "/tmp/sample1",
                "/tmp/sample2"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_samples() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--1",
                "1",
                "/tmp/sample1",
                "/tmp/sample2"
            )
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_samples_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for samples
        image.dependencies.samples.kwarg = Some("--inputs".into());
        // generate a job
        let job = generate_job();
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--inputs",
                "/tmp/sample1",
                "--inputs",
                "/tmp/sample2"
            )
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_samples_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for samples
        image.dependencies.samples.kwarg = Some("--inputs".into());
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--inputs",
                "/tmp/sample1",
                "--inputs",
                "/tmp/sample2",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_samples_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for samples
        image.dependencies.samples.kwarg = Some("--inputs".into());
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--inputs", vec!["sample0"]);
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--inputs",
                "sample0",
                "--inputs",
                "/tmp/sample1",
                "--inputs",
                "/tmp/sample2"
            )
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_ephemerals() {
        // generate an image
        let image = generate_image();
        // generate a job
        let job = generate_job();
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string![
                "/usr/bin/python3",
                "corn.py",
                "/tmp/file.txt",
                "/tmp/other.txt"
            ]
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_ephemerals() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "pos1",
                "pos2",
                "/tmp/file.txt",
                "/tmp/other.txt"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_ephemerals() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--1", vec!["1"]);
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--1",
                "1",
                "/tmp/file.txt",
                "/tmp/other.txt"
            )
        );
    }

    /// Test a barebones job with samples but no overlays
    #[tokio::test]
    async fn empty_ephemerals_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for ephemeral files
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        // generate a job
        let job = generate_job();
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--ephemeral",
                "/tmp/file.txt",
                "--ephemeral",
                "/tmp/other.txt"
            )
        );
    }

    /// Test a job with positional overlays
    #[tokio::test]
    async fn positionals_ephemerals_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for ephemeral files
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        // generate a job
        let mut job = generate_job();
        // build stage args with just positionals
        job.args = job.args.positionals(vec!["pos1", "pos2"]);
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--ephemeral",
                "/tmp/file.txt",
                "--ephemeral",
                "/tmp/other.txt",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with keyword args
    #[tokio::test]
    async fn kwargs_ephemerals_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for ephemeral files
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--ephemeral", vec!["first.txt"]);
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--ephemeral",
                "first.txt",
                "--ephemeral",
                "/tmp/file.txt",
                "--ephemeral",
                "/tmp/other.txt"
            )
        );
    }

    /// Test a generator job with keyword args
    #[tokio::test]
    async fn generator_kwargs() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.kwarg("--nums", vec!["1", "2"]);
        // make this job a generator
        job.generator = true;
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        // convert built to a set so that order does not matter
        let built_set: HashSet<_> = cmd.iter().collect();
        // get this jobs id
        let id = job.id.to_string();
        let reaction = job.reaction.to_string();
        // build the expected args
        let args = vec_string!(
            "/usr/bin/python3",
            "corn.py",
            "--nums",
            "1",
            "--nums",
            "2",
            "--reaction",
            &reaction,
            "--job",
            &id
        );
        // convert args to a set so that order does not matter
        let args_set: HashSet<_> = args.iter().collect();
        // make sure our command has the same number of args before we cast it to a set
        assert_eq!(cmd.len(), args.len());
        // make sure we have the same args
        assert_eq!(built_set, args_set);
    }

    /// Test a job with switch overlays
    #[tokio::test]
    async fn switches() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job.args.switch("--corn").switch("--beans");
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!("/usr/bin/python3", "corn.py", "--corn", "--beans")
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build stage args with kwargs, switches, and positionals
        job.args = job
            .args
            .kwarg("--1", vec!["1"])
            .switch("--corn")
            .switch("--beans")
            .positionals(vec!["pos1", "pos2"]);
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--1",
                "1",
                "--corn",
                "--beans",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_samples() {
        // generate an image
        let image = generate_image();
        // generate a job
        let mut job = generate_job();
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build stage args with kwargs, switches, and positionals
        job.args = job
            .args
            .kwarg("--1", vec!["1"])
            .switch("--corn")
            .switch("--beans")
            .positionals(vec!["pos1", "pos2"]);
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--1",
                "1",
                "--corn",
                "--beans",
                "pos1",
                "pos2",
                "/tmp/sample1",
                "/tmp/sample2"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_samples_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for samples
        image.dependencies.samples.kwarg = Some("--inputs".into());
        // generate a job
        let mut job = generate_job();
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--inputs", vec!["sample0"])
            .switch("--corn")
            .switch("--beans");
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--inputs",
                "sample0",
                "--inputs",
                "/tmp/sample1",
                "--inputs",
                "/tmp/sample2",
                "--corn",
                "--beans",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_ephemeral() {
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
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--1",
                "1",
                "--corn",
                "--beans",
                "pos1",
                "pos2",
                "/tmp/file.txt",
                "/tmp/other.txt"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_ephemeral_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        // set a kwarg for ephemeral files
        image.dependencies.ephemeral.kwarg = Some("--ephemeral".into());
        // generate a job
        let mut job = generate_job();
        // build stage args with just kwargs
        job.args = job
            .args
            .positionals(vec!["pos1", "pos2"])
            .kwarg("--ephemeral", vec!["first.txt"])
            .switch("--corn")
            .switch("--beans");
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--ephemeral",
                "first.txt",
                "--ephemeral",
                "/tmp/file.txt",
                "--ephemeral",
                "/tmp/other.txt",
                "--corn",
                "--beans",
                "pos1",
                "pos2"
            )
        );
    }

    /// Test a job with positional, kwarg, and switches
    #[tokio::test]
    async fn combined_all() {
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
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build paths to our repos
        let repo_paths = vec![PathBuf::from("/tmp/repo1"), PathBuf::from("/tmp/repo2")];
        // build paths to our samples
        let ephem_paths = vec![
            PathBuf::from("/tmp/file.txt"),
            PathBuf::from("/tmp/other.txt"),
        ];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        //<<<<<<< HEAD
        .add_samples(&sample_paths, &image.dependencies.samples)
        .add_repos(&image, &job.repos, &repo_paths)
        //=======
        //        .add_samples(&sample_paths, &image.dependencies.samples)
        //        .add_repos(&job.repos, &repo_paths, &image.dependencies.repos)
        //>>>>>>> bd24ea94 (feat(agent): Added support for passing samples to jobs by file name)
        .add_ephemeral(&job.ephemeral, &ephem_paths, &image.dependencies.ephemeral)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--1",
                "1",
                "--corn",
                "--beans",
                "pos1",
                "pos2",
                "/tmp/sample1",
                "/tmp/sample2",
                "/tmp/repo1",
                "/tmp/repo2",
                "/tmp/file.txt",
                "/tmp/other.txt"
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
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // add paths for each of our samples/repos as result dependencies
        let test_dir = PathBuf::from("/tmp/thorium/testing");
        let results_dir = test_dir.join("prior-results");
        let result_paths = job
            .samples
            .iter()
            .map(|sample| results_dir.join(sample))
            .chain(job.repos.iter().map(|repo| results_dir.join(&repo.url)))
            .collect::<Vec<PathBuf>>();
        // create sub-directories in the results dir for image1, but not for image2
        for dir in &result_paths {
            let dir = dir.join("image1");
            tokio::fs::create_dir_all(&dir).await.unwrap();
        }
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string!["corn.py"],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .add_results(
            &image.dependencies.results.images,
            &result_paths,
            &image.dependencies.results,
            &mut logs_tx,
        )
        .await
        .unwrap()
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "--image1-results",
                "/tmp/thorium/testing/prior-results/sample1/image1",
                "--image1-results",
                "/tmp/thorium/testing/prior-results/sample2/image1",
                "--image1-results",
                "/tmp/thorium/testing/prior-results/github.com/curl/curl/image1",
                "--image1-results",
                "/tmp/thorium/testing/prior-results/github.com/notcurl/notcurl/image1",
                "/tmp/sample1",
                "/tmp/sample2"
            )
        );
        // remove the test directory
        tokio::fs::remove_dir_all(&test_dir).await.unwrap();
    }

    /// Test a job with a subcommand and kwarg overlays
    #[tokio::test]
    async fn subcommands_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.samples.kwarg = Some("--crop".into());
        let job = generate_job();
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string![
                "corn.py",
                "planter",
                "--crop=corn",
                "--other=StillHere",
                "start"
            ],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our overlayed command
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "planter",
                "--crop=/tmp/sample1",
                "--crop=/tmp/sample2",
                "--other=StillHere",
                "start"
            )
        );
    }

    /// Test a job with nested subcommands and kwarg overlays
    #[tokio::test]
    async fn subcommands_nested_kwargs() {
        // generate an image and set the kwarg to pass in samples with
        let mut image = generate_image();
        image.dependencies.samples.kwarg = Some("--crop".into());
        let job = generate_job();
        // build paths to our samples
        let sample_paths = vec![PathBuf::from("/tmp/sample1"), PathBuf::from("/tmp/sample2")];
        // build the command to execute
        let cmd = CmdBuilder::new(
            &image,
            &job,
            slice_string!["/usr/bin/python3"],
            slice_string![
                "corn.py",
                "planter",
                "--crop=corn",
                "start",
                "--fertilizer=blue",
                "nested"
            ],
        )
        .add_samples(&sample_paths, &image.dependencies.samples)
        .build(&image, None)
        .unwrap();
        // validate our args
        assert_eq!(
            cmd,
            vec_string!(
                "/usr/bin/python3",
                "corn.py",
                "planter",
                "--crop=/tmp/sample1",
                "--crop=/tmp/sample2",
                "start",
                "--fertilizer=blue",
                "nested"
            )
        );
    }
}
