//! Wrappers for interacting with jobs within Thorium with different backends
//! Currently only Redis is supported

use chrono::prelude::*;
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

use super::{ImageScaler, Reaction, RepoDependency, SystemComponents};
use crate::{matches_adds, matches_opt, matches_removes, matches_removes_map, same};

/// A list of job IDs with a cursor
#[derive(Serialize, Debug)]
pub struct JobList {
    /// The cursor used to page through job IDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of job IDs
    pub names: Vec<Uuid>,
}

/// A list of job details with a cursor
#[derive(Serialize, Debug)]
pub struct JobDetailsList {
    /// A cursor used to page through job details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of job details
    pub details: Vec<RawJob>,
}

/// Helps serde default the job list limit
fn default_job_list_limit() -> u64 {
    10_000
}

/// The query params for a map request
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct JobListOpts {
    /// The number of objects to skip in the stream
    #[serde(default)]
    pub skip: u64,
    /// The minimum number of objects to map
    #[serde(default = "default_job_list_limit")]
    pub limit: u64,
}

impl Default for JobListOpts {
    fn default() -> Self {
        JobListOpts {
            skip: 0,
            limit: default_job_list_limit(),
        }
    }
}

/// A single job to reset
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RunningJob {
    /// The uuidv4 of the job to reset
    pub job_id: Uuid,
    /// The container/node that is working on this job
    pub worker: String,
}

/// The requestor for a job reset
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum JobResetRequestor {
    /// This job was reset by a Thorium system component
    Component(SystemComponents),
    /// This job was reset directly by a user
    User,
}

/// A list of jobs to reset
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct JobResets {
    /// The component doing this resets
    pub requestor: JobResetRequestor,
    /// The reason for this request
    pub reason: String,
    /// The scaler to reset jobs for
    pub scaler: ImageScaler,
    /// The jobs to reset
    pub jobs: Vec<Uuid>,
}

impl JobResets {
    /// Create a new job reset request
    ///
    /// # Arguments
    ///
    /// * `reason` - The reason to reset this request
    pub fn new<T: Into<String>>(scaler: ImageScaler, reason: T) -> Self {
        JobResets {
            requestor: JobResetRequestor::User,
            reason: reason.into(),
            scaler,
            jobs: Vec::default(),
        }
    }

    /// Create a new job reset request
    ///
    /// # Arguments
    ///
    /// * `reason` - The reason to reset this request
    pub fn with_capacity<T: Into<String>>(scaler: ImageScaler, reason: T, capacity: usize) -> Self {
        JobResets {
            requestor: JobResetRequestor::User,
            reason: reason.into(),
            scaler,
            jobs: Vec::with_capacity(capacity),
        }
    }

    /// Set this reset requestor to a component
    ///
    /// # Arguments
    ///
    /// * `component` - The component that is requesting this reset
    pub fn as_component(mut self, component: SystemComponents) -> Self {
        // set our updated requestor
        self.requestor = JobResetRequestor::Component(component);
        self
    }

    /// Add a job to to be reset
    ///
    /// # Arguments
    ///
    /// * `job` - The job to be reset
    pub fn add(mut self, job: Uuid) -> Self {
        self.jobs.push(job);
        self
    }
}

/// The different possible statuses for a reaction
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "trace", derive(valuable::Valuable))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum JobStatus {
    /// This job has been created but is not running
    Created,
    /// This job has been claimed and is running
    Running,
    /// This job has completed
    Completed,
    /// This job has failed due to an error
    Failed,
    /// This job has returned to Thorium to be respawned (generators)
    Sleeping,
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            JobStatus::Created => write!(f, "Created"),
            JobStatus::Running => write!(f, "Running"),
            JobStatus::Completed => write!(f, "Completed"),
            JobStatus::Failed => write!(f, "Failed"),
            JobStatus::Sleeping => write!(f, "Sleeping"),
        }
    }
}

/// The different possible statuses for a Job handle command
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum JobHandleStatus {
    /// The current job completed but the reaction is still waiting for other jobs to complete
    Waiting,
    /// This was the final job to complete in this stage of our reaction and we are proceeding
    Proceeding,
    /// This job has completed
    Completed,
    /// This job was successfully errored out
    Errored,
    /// This job is a generator and is being set to sleep
    Sleeping,
    /// This job has been checkpointed
    Checkpointed,
}

/// response for handling Job command
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct HandleJobResponse {
    /// status of executed command
    pub status: JobHandleStatus,
}

/// A checkpoint string for a job
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Checkpoint {
    pub data: String,
}

/// A raw job that Thorium will execute
///
/// This should be cast to either a GenericJob or another known job
/// before being handed to the user.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RawJob {
    /// The group this job is in
    pub group: String,
    /// The pipeline this job is for
    pub pipeline: String,
    /// The reaction this job is apart of
    pub reaction: Uuid,
    /// The stage of the pipeline this job is for
    pub stage: String,
    /// The user who created the reaction for this job
    pub creator: String,
    /// The uuidv4 of this job
    pub id: Uuid,
    /// The arguments for this job
    pub args: String,
    /// The current status of the job
    pub status: JobStatus,
    /// The time this job must be started by
    pub deadline: DateTime<Utc>,
    /// The container/node that is working on this job
    pub worker: Option<String>,
    /// The parent reaction to this jobs reaction if it exists
    pub parent: Option<Uuid>,
    /// Whether this raw-job is a generator or not
    pub generator: bool,
    /// What scaler is responsible for scaling this image
    pub scaler: ImageScaler,
    /// A list of sample sha256s to download before executing this job
    pub samples: Vec<String>,
    /// A list of ephemeral files to download before executing this job
    pub ephemeral: Vec<String>,
    /// A list of ephemeral files from any parent reactions and what parent reaction its tied to
    pub parent_ephemeral: HashMap<String, Uuid>,
    /// A list of repos to download before executing this reactions jobs
    pub repos: Vec<RepoDependency>,
    /// The trigger depth for this job if one was set
    pub trigger_depth: Option<u8>,
}

/// Keyword args for generic jobs
pub type GenericJobKwargs = HashMap<String, Vec<String>>;

/// Helps serde default a value to false
fn default_false() -> bool {
    false
}

/// Options for a generic job
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "trace", derive(valuable::Valuable))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GenericJobOpts {
    /// Whether to always override all positional args in the original image
    #[serde(default = "default_false")]
    pub override_positionals: bool,
    /// Whether to always override all keyword args in the original image
    #[serde(default = "default_false")]
    pub override_kwargs: bool,
    /// The cmd to override the original cmd from the image with in its entirety
    pub override_cmd: Option<Vec<String>>,
}

impl Default for GenericJobOpts {
    /// Create a GenericJobOpts object with default values
    fn default() -> Self {
        GenericJobOpts {
            override_positionals: false,
            override_kwargs: false,
            override_cmd: None,
        }
    }
}

impl GenericJobOpts {
    /// Creates a new [`GenericJobOpts`] object
    ///
    /// Overriding positionals and kwargs will effective remove them from the source images docker
    /// command. Overriding the cmd will effectively replace the source images docker command.
    ///
    /// # Arguments
    ///
    /// * `positionals` - Whether to override the positional args or not
    /// * `kwargs` - Whether to override the kwargs or not
    /// * `cmd` - The command to override the source docker command with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobOpts;
    ///
    /// // create options to override the original docker command
    /// let opts = GenericJobOpts::new(false, false, Some(vec!("./harvest".into(), "corn".into())));
    /// ```
    pub fn new(positionals: bool, kwargs: bool, cmd: Option<Vec<String>>) -> Self {
        GenericJobOpts {
            override_positionals: positionals,
            override_kwargs: kwargs,
            override_cmd: cmd,
        }
    }
}

/// Arguments for a [`GenericJob`]
///
/// # Examples
///
/// ```
/// use thorium::models::{GenericJobOpts, GenericJobArgs};
///
/// // Create args to harvest corn in a specific field
/// // This also sets options to disable the original positional args
/// let args = GenericJobArgs::default()
///     .positionals(vec!("corn"))
///     .kwarg("field", vec!("west-3"))
///     .switches(vec!("--combine"))
///     .opts(GenericJobOpts::new(true, false, None));
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(valuable::Valuable))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GenericJobArgs {
    /// The positional arguments to overlay onto the original cmd
    #[serde(default)]
    pub positionals: Vec<String>,
    /// The keyword arguments to overlay onto the original cmd
    #[serde(default)]
    pub kwargs: GenericJobKwargs,
    /// The switch arguments to overlay onto the original cmd
    #[serde(default)]
    pub switches: Vec<String>,
    /// The options to apply to this generic job
    #[serde(default = "GenericJobOpts::default")]
    pub opts: GenericJobOpts,
}

impl GenericJobArgs {
    /// Sets the positional args for a GenericJob
    pub fn positionals<T: Into<String>>(mut self, positionals: Vec<T>) -> Self {
        // cast these values to strings
        let positionals = positionals.into_iter().map(|val| val.into()).collect();
        self.positionals = positionals;
        self
    }

    /// Adds a keyword arg to this job
    pub fn kwarg<K: Into<String>, V: Into<String>>(mut self, key: K, values: Vec<V>) -> Self {
        // convert our kwargs to strings
        let converted = values.into_iter().map(|value| value.into()).collect();
        self.kwargs.insert(key.into(), converted);
        self
    }

    /// Overwrites all kwargs with a new [`HashMap`]
    pub fn set_kwargs(mut self, kwargs: GenericJobKwargs) -> Self {
        self.kwargs = kwargs;
        self
    }

    /// Adds a switch to this job
    pub fn switch<T: Into<String>>(mut self, switch: T) -> Self {
        self.switches.push(switch.into());
        self
    }

    /// Sets the switches for a GenericJob
    pub fn switches<T: Into<String>>(mut self, switches: Vec<T>) -> Self {
        // cast these values to strings
        let switches = switches.into_iter().map(|val| val.into()).collect();
        self.switches = switches;
        self
    }

    /// sets the options for GenericJobArgs
    pub fn opts(mut self, opts: GenericJobOpts) -> Self {
        self.opts = opts;
        self
    }

    /// Cast all of the args to a Vector
    pub fn to_vec(&self) -> Vec<String> {
        // figure out how large our vec should be
        let size = self.positionals.len() + self.kwargs.len() + self.switches.len();
        let mut casts = Vec::with_capacity(size);
        // start with our positionals and switches
        casts.extend_from_slice(&self.positionals);
        casts.extend_from_slice(&self.switches);
        // crawl over our kwarg keys
        for (key, values) in self.kwargs.iter() {
            // combine and cast our kwargs
            casts.extend(values.iter().map(|val| format!("{}={}", key, val)));
        }
        casts
    }
}

/// checks that a job matches its reaction request
impl PartialEq<GenericJobArgsUpdate> for GenericJobArgs {
    fn eq(&self, update: &GenericJobArgsUpdate) -> bool {
        // make sure our kwargs were updated
        let kwarg_check = update
            .kwargs
            .iter()
            .all(|(key, value)| &self.kwargs[key] == value);
        same!(kwarg_check, true);
        // make sure our adds/removes were added
        matches_removes_map!(self.kwargs, update.remove_kwargs);
        matches_removes!(self.switches, update.remove_switches);
        matches_adds!(self.switches, update.add_switches);
        // if opts are set then make sure they were applied
        if let Some(opts) = &update.opts {
            same!(&self.opts, opts);
        }
        true
    }
}

/// Updates the arguments for a [`GenericJob`]
///
/// # Examples
///
/// ```
/// use thorium::models::{GenericJobOpts, GenericJobArgsUpdate};
///
/// // Create args to harvest corn in a specific field
/// // This also sets options to disable the original positional args
/// let args = GenericJobArgsUpdate::default()
///     .positionals(vec!("soy"))
///     .kwarg("--field", vec!("west-4"))
///     .remove_kwarg("--tractor")
///     .add_switches(vec!("--combine"))
///     .remove_switches(vec!("--plow"))
///     .opts(GenericJobOpts::new(true, false, None));
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GenericJobArgsUpdate {
    /// The positional arguments to replace the original args with
    #[serde(default)]
    pub positionals: Vec<String>,
    /// The keyword arguments to overlay onto the original kwargs for this job
    #[serde(default)]
    pub kwargs: GenericJobKwargs,
    /// The kwargs arguments to remove from the original kwargs for this job
    #[serde(default)]
    pub remove_kwargs: Vec<String>,
    /// The switch arguments to add to the original switches for this job
    #[serde(default)]
    pub add_switches: Vec<String>,
    /// The switch arguments to remove from the original switches for this job
    #[serde(default)]
    pub remove_switches: Vec<String>,
    /// The options to replace for this generic job
    pub opts: Option<GenericJobOpts>,
}

impl GenericJobArgsUpdate {
    /// Sets the positional args for a GenericJob
    ///
    /// # Arguments
    ///
    /// * `positionals` - The positional args for this job
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobArgsUpdate;
    ///
    /// let args = GenericJobArgsUpdate::default()
    ///     .positionals(vec!("soy"));
    /// ```
    pub fn positionals<T: Into<String>>(mut self, positionals: Vec<T>) -> Self {
        // cast these values to strings
        let positionals = positionals.into_iter().map(|val| val.into()).collect();
        self.positionals = positionals;
        self
    }

    /// Adds a keyword arg to this job
    ///
    /// # Arguments
    ///
    /// * `key` - The key for this keyword arg
    /// * `value` - The value for this keyword arg
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobArgsUpdate;
    ///
    /// let args = GenericJobArgsUpdate::default()
    ///     .kwarg("--field", vec!("west-4", "west-5"));
    /// ```
    pub fn kwarg<T: Into<String>>(mut self, key: T, values: Vec<T>) -> Self {
        // convert our kwargs to strings
        let converted = values.into_iter().map(|value| value.into()).collect();
        self.kwargs.insert(key.into(), converted);
        self
    }

    /// Removes a kwarg from this job
    ///
    /// # Arguments
    ///
    /// * `remove` - The kwarg to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobArgsUpdate;
    ///
    /// let args = GenericJobArgsUpdate::default()
    ///     .remove_kwarg("--tractor");
    /// ```
    pub fn remove_kwarg<T: Into<String>>(mut self, remove: T) -> Self {
        self.remove_kwargs.push(remove.into());
        self
    }

    /// Adds a switch to this job
    ///
    /// # Arguments
    ///
    /// * `switch` - The switch to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobArgsUpdate;
    ///
    /// let args = GenericJobArgsUpdate::default()
    ///     .switch("--combine");
    /// ```
    pub fn switch<T: Into<String>>(mut self, switch: T) -> Self {
        self.add_switches.push(switch.into());
        self
    }

    /// Adds new switches for a GenericJob
    ///
    /// # Arguments
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobArgsUpdate;
    ///
    /// let args = GenericJobArgsUpdate::default()
    ///     .add_switches(vec!("--combine"));
    /// ```
    pub fn add_switches<T: Into<String>>(mut self, adds: Vec<T>) -> Self {
        // cast these values to strings
        let adds = adds.into_iter().map(|val| val.into()).collect();
        self.add_switches = adds;
        self
    }

    /// Removes switches from a GenericJob's args
    ///
    /// # Arguments
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::GenericJobArgsUpdate;
    ///
    /// let args = GenericJobArgsUpdate::default()
    ///     .remove_switches(vec!("--plow"));
    /// ```
    pub fn remove_switches<T: Into<String>>(mut self, removes: Vec<T>) -> Self {
        // cast these values to strings
        let removes = removes.into_iter().map(|val| val.into()).collect();
        self.remove_switches = removes;
        self
    }

    /// Sets the options for GenericJobArgs completely overwriting the original options
    ///
    /// # Arguments
    ///
    /// * `opts` - The updated options for this job
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GenericJobOpts, GenericJobArgsUpdate};
    ///
    /// // Create args to harvest corn in a specific field
    /// // This also sets options to disable the original positional args
    /// let args = GenericJobArgsUpdate::default()
    ///     .opts(GenericJobOpts::new(true, false, None));
    /// ```
    pub fn opts(mut self, opts: GenericJobOpts) -> Self {
        self.opts = Some(opts);
        self
    }
}

/// A job that Thorium will execute
///
/// Currently this is the only job we support but in the future I would like to
/// have a way to allow users to tell Thorium how to bounds check jobs. That
/// will likely be its own type.
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GenericJob {
    /// The reaction this job is apart of
    pub reaction: Uuid,
    /// The uuidv4 of this job
    pub id: Uuid,
    /// The group this job is in
    pub group: String,
    /// The pipeline this job is for
    pub pipeline: String,
    /// The stage of the pipeline this job is for
    pub stage: String,
    /// The user who created the reaction for this job
    pub creator: String,
    /// The arguments for this job
    pub args: GenericJobArgs,
    /// The current status of the job
    pub status: JobStatus,
    /// The time this job must be started by
    pub deadline: DateTime<Utc>,
    /// The parent reaction to this jobs reaction if it exists
    pub parent: Option<Uuid>,
    /// Whether this job is a generator or not
    pub generator: bool,
    /// A list of sample sha256s to download before executing this job
    pub samples: Vec<String>,
    /// A list of ephemeral files to download before executing this job
    pub ephemeral: Vec<String>,
    /// A list of ephemeral files from any parent reactions and what parent reaction its tied to
    pub parent_ephemeral: HashMap<String, Uuid>,
    /// A list of repos to download before executing this reactions jobs
    pub repos: Vec<RepoDependency>,
    /// The trigger depth for this job if one was set
    pub trigger_depth: Option<u8>,
}

/// checks that a vector of jobs matches a reaction request
impl PartialEq<Reaction> for &Vec<GenericJob> {
    fn eq(&self, react: &Reaction) -> bool {
        // return false if any job does not equal this reaction
        !self.is_empty() && self.iter().all(|job| job == react)
    }
}

/// checks that a job matches its reaction request
impl PartialEq<Reaction> for GenericJob {
    fn eq(&self, react: &Reaction) -> bool {
        // make sure group, pipeline and reaction id match
        same!(self.group, react.group);
        same!(self.pipeline, react.pipeline);
        same!(self.reaction, react.id);
        // make sure our job id is in this reactions job IDs
        same!(react.jobs.contains(&self.id), true);
        // if there are no args specified then make sure our args are empty/defaults
        let mut args = react
            .args
            .get(&self.stage)
            .unwrap_or(&GenericJobArgs::default())
            .clone();
        // inject in any updated checkpointed args
        if let Some(checkpoint) = self.args.kwargs.get("--checkpoint") {
            args.kwargs
                .insert("--checkpoint".to_owned(), checkpoint.to_owned());
        }
        same!(&self.args, &args);
        //make sure this jobs parent reaction is set
        matches_opt!(&self.parent, &react.parent);
        // make sure our trigger depth matches
        same!(&self.trigger_depth, &react.trigger_depth);
        true
    }
}

/// The information in a specific job claim status queue
#[cfg(feature = "api")]
#[derive(Debug, Serialize, Deserialize)]
pub struct JobReactionIds {
    /// The id of this claimed job
    pub job: Uuid,
    /// The id of the reaction containing this job
    pub reaction: Uuid,
}

#[cfg(feature = "api")]
impl JobReactionIds {
    /// Create a new job claim object
    ///
    /// # Arguments
    ///
    /// * `job` - This jobs id
    /// * `reaction` - The reaction containing this jobs id
    pub fn new(job: Uuid, reaction: Uuid) -> Self {
        JobReactionIds { job, reaction }
    }
}
