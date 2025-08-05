//! Wrappers for interacting with reactions within Thorium with different backends
//! Currently only Redis is supported

use base64::Engine as _;
use chrono::prelude::*;
use std::collections::HashMap;
use uuid::Uuid;

// only support tokio for file reads in tokio mode
#[cfg(feature = "tokio-models")]
use tokio::{fs::File, io::AsyncReadExt};

use super::{
    GenericJobArgs, GenericJobArgsUpdate, JobHandleStatus, RepoDependency, RepoDependencyRequest,
};
use crate::{matches_adds, matches_removes, matches_vec, same};

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use serde_json::Value;

        use super::GenericJobOpts;
        use crate::{deserialize_value, bad, utils::ApiError};

        /// Generic job args containing non statically typed kwargs
        ///
        /// This is used to let users pass in either String or Vec<String> values to kwargs
        #[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
        pub struct RawGenericJobArgs {
            /// The positional arguments to overlay onto the original cmd
            #[serde(default)]
            pub positionals: Vec<String>,
            /// The keyword arguments to overlay onto the original cmd
            #[serde(default)]
            pub kwargs: HashMap<String, Value>,
            /// The switch arguments to overlay onto the original cmd
            #[serde(default)]
            pub switches: Vec<String>,
            /// The options to apply to this generic job
            #[serde(default = "GenericJobOpts::default")]
            pub opts: GenericJobOpts,
        }

        impl TryFrom<RawGenericJobArgs> for GenericJobArgs {
            type Error = ApiError;
            /// Converts [`RawGenericJobArgs`] to [`GenericJobArgs`]
            ///
            /// # Arguments
            ///
            /// * `raw` - The raw generic job args to convert
            fn try_from(raw: RawGenericJobArgs) -> Result<Self, ApiError> {
                // build a hashmap of the right size
                let mut kwargs = HashMap::with_capacity(raw.kwargs.len());
                // crawl over the raw kwarg values and convert them to vectors
                for (key, values) in raw.kwargs {
                    // if this value is a string then wrap it in a vector and insert it
                    if let Some(value) = values.as_str() {
                        kwargs.insert(key, vec!(value.to_owned()));
                        continue;
                    }
                    // if this value is a list then just insert it
                    if values.is_array() {
                        kwargs.insert(key, deserialize_value!(values));
                        continue
                    }
                    // the value was not a string or an array so throw an error
                    return bad!(format!("The kwarg value {:?} for {} is not a string or array", values, key));
                }
                // build our converted job args structure
                let converted = GenericJobArgs {
                    positionals: raw.positionals,
                    kwargs,
                    switches: raw.switches,
                    opts: raw.opts,
                };
                Ok(converted)
            }
        }

        // The arguments for all images in a reaction
        pub type RawReactionArgs = HashMap<String, RawGenericJobArgs>;

        /// A request to create a new reaction
        #[derive(Serialize, Deserialize, Debug, Clone)]
        pub struct RawReactionRequest {
            /// The group the reaction is in
            pub group: String,
            /// The pipeline this reaction is build around
            pub pipeline: String,
            /// The args to overlay on top of the args for images in this reaction
            pub args: RawReactionArgs,
            /// The number of seconds we have to meet this reactions SLA.
            pub sla: Option<u64>,
            /// The tags this reaction can be listed under
            #[serde(default)]
            pub tags: Vec<String>,
            /// The parent reaction to set if this is a sub reaction
            pub parent: Option<Uuid>,
            /// A list of sample sha256s to download before executing this reaction's jobs
            #[serde(default)]
            pub samples: Vec<String>,
            /// A map of ephemeral buffers to download before executing this reaction's jobs
            #[serde(default)]
            pub buffers: HashMap<String, String>,
            /// Any repos to download before executing this reaction's jobs
            #[serde(default)]
            pub repos: Vec<RepoDependencyRequest>,
            /// This reactions depth in triggers if this reaction was caused by a trigger
            pub trigger_depth: Option<u8>,
        }

        impl TryFrom<RawReactionRequest> for ReactionRequest {
            type Error = ApiError;
            /// Converts [`RawReactionRequest`] to [`ReactionRequest`]
            ///
            /// # Arguments
            ///
            /// * `raw` - The raw reaction request to convert
            fn try_from(raw: RawReactionRequest) -> Result<Self, Self::Error> {
                // build a hashmap to store our job args
                let mut args = HashMap::with_capacity(raw.args.len());
                // crawl over all job args and convert them
                for (stage, raw) in raw.args {
                    args.insert(stage, GenericJobArgs::try_from(raw)?);
                }
                // build the converted reaction request
                let converted = ReactionRequest {
                    group: raw.group,
                    pipeline: raw.pipeline,
                    args,
                    sla: raw.sla,
                    tags: raw.tags,
                    parent: raw.parent,
                    samples: raw.samples,
                    buffers: raw.buffers,
                    repos: raw.repos,
                    trigger_depth: raw.trigger_depth,
                };
                Ok(converted)
            }
        }
    }
}

/// A response containing the Reaction id
#[derive(Serialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ReactionIdResponse {
    /// The uuidv4 of a reaction
    pub id: Uuid,
}

/// The response from creating reactions in bulk
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct BulkReactionResponse {
    /// Any errors that occurred while creating reactions
    pub errors: HashMap<usize, String>,
    /// The successfully created reactions
    pub created: Vec<Uuid>,
}

impl BulkReactionResponse {
    /// Create a new reaction response with a starting capacity for created reactions
    ///
    /// # Arguments
    ///
    /// * `capacity` - The capacity to allocate
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        BulkReactionResponse {
            errors: HashMap::default(),
            created: Vec::with_capacity(capacity),
        }
    }
}

/// A response for handling the reaction command
#[derive(Serialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct HandleReactionResponse {
    /// The status of the executed command
    pub status: ReactionStatus,
}

/// The arguments for all images in a reaction
pub type ReactionArgs = HashMap<String, GenericJobArgs>;

/// A request to create a new reaction
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ReactionRequest {
    /// The group the reaction is in
    pub group: String,
    /// The pipeline this reaction is build around
    pub pipeline: String,
    /// The args to overlay on top of the args for images in this reaction
    pub args: ReactionArgs,
    /// The number of seconds we have to meet this reactions SLA.
    pub sla: Option<u64>,
    /// The tags this reaction can be listed under
    #[serde(default)]
    pub tags: Vec<String>,
    /// The parent reaction to set if this is a sub reaction
    pub parent: Option<Uuid>,
    /// A list of sample sha256s to download before executing this reaction's job
    #[serde(default)]
    pub samples: Vec<String>,
    /// A map of ephemeral buffers to download before executing this reaction's jobs
    #[serde(default)]
    pub buffers: HashMap<String, String>,
    /// Any repos to download before executing this reaction's jobs
    #[serde(default)]
    pub repos: Vec<RepoDependencyRequest>,
    /// This reactions depth in triggers if this reaction was caused by a trigger
    pub trigger_depth: Option<u8>,
}

impl ReactionRequest {
    /// Creates a [`ReactionRequest`] for a new reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction should be in
    /// * `pipeline` - The pipeline this reaction should be based on
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ReactionRequest, GenericJobArgs};
    ///
    /// // build args for all images/stages
    /// // plant corn
    /// let plant_args = GenericJobArgs::default()
    ///     .positionals(vec!("corn"));
    /// // use a combine to harvest
    /// let harvest_args = GenericJobArgs::default()
    ///     .switches(vec!("--combine"));
    /// // create a reaction with an SLA of 1 week
    /// let request = ReactionRequest::new("Corn", "harvest")
    ///     .sla(604800)
    ///     .args("plant", plant_args)
    ///     .args("harvest", harvest_args)
    ///     .tag("CornPlants")
    ///     .sample("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f");
    /// ```
    pub fn new<T: Into<String>>(group: T, pipeline: T) -> Self {
        ReactionRequest {
            group: group.into(),
            pipeline: pipeline.into(),
            sla: None,
            args: HashMap::default(),
            tags: Vec::default(),
            parent: None,
            samples: Vec::default(),
            buffers: HashMap::default(),
            repos: Vec::default(),
            trigger_depth: None,
        }
    }

    /// Add args for an image/stage in the reaction
    ///
    /// # Arguments
    ///
    /// * `image` - The name of the image/stage these args are for
    /// * `args` - The args for this image/stage
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ReactionRequest, GenericJobArgs};
    ///
    /// // build args for all images/stages
    /// // plant corn
    /// let plant_args = GenericJobArgs::default()
    ///     .positionals(vec!("corn"));
    /// // use a combine to harvest
    /// let harvest_args = GenericJobArgs::default()
    ///     .switches(vec!("--combine"));
    /// // create a reaction
    /// let request = ReactionRequest::new("Corn", "harvest")
    ///     .args("plant", plant_args)
    ///     .args("harvest", harvest_args);
    /// ```
    #[must_use]
    pub fn args<T: Into<String>>(mut self, image: T, args: GenericJobArgs) -> Self {
        self.args.insert(image.into(), args);
        self
    }

    /// Set the SLA for a reaction
    ///
    /// # Arguments
    ///
    /// * `sla` - The sla to set for a reaction
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction with an SLA of 1 week
    /// let request = ReactionRequest::new("Corn", "harvest").sla(604800);
    /// ```
    #[must_use]
    pub fn sla(mut self, sla: u64) -> Self {
        self.sla = Some(sla);
        self
    }

    /// Add a search tag
    ///
    /// # Arguments
    ///
    /// * `tag` - A tag to allow this job to be searched by
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction
    /// let request = ReactionRequest::new("Corn", "harvest").tag("CornPlants");
    /// ```
    #[must_use]
    pub fn tag<T: Into<String>>(mut self, tag: T) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Adds search tags
    ///
    /// # Arguments
    ///
    /// * `tags` - The tags to allow this job to be searched by
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction
    /// let request = ReactionRequest::new("Corn", "harvest")
    ///     .tags(vec!("CornPlants", "Plants"));
    /// ```
    #[must_use]
    pub fn tags<T: Into<String>>(mut self, tags: Vec<T>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Sets a parent reaction
    ///
    /// # Arguments
    ///
    /// * `parent` - The uuid of the reaction that is spawning this sub reaction
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    /// use uuid::Uuid;
    ///
    /// // get the uuid of the parent reaction
    /// let parent = Uuid::new_v4();
    /// // create a sub reaction with an SLA of 1 day
    /// let request = ReactionRequest::new("Combine", "fill_gas")
    ///     .parent(parent);
    /// ```
    #[must_use]
    pub fn parent<T: Into<Uuid>>(mut self, parent: T) -> Self {
        self.parent = Some(parent.into());
        self
    }

    /// Adds a sample to download when running this reactions jobs
    ///
    /// # Arguments
    ///
    /// * `sample` - The sha256 of the sample to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction
    /// let request = ReactionRequest::new("Corn", "harvest")
    ///     .sample("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f");
    /// ```
    #[must_use]
    pub fn sample<T: Into<String>>(mut self, sample: T) -> Self {
        self.samples.push(sample.into());
        self
    }

    /// Adds samples to download when running this reactions jobs
    ///
    /// # Arguments
    ///
    /// * `samples` - The sha256s of the samples to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction
    /// let request = ReactionRequest::new("Corn", "harvest")
    ///     .samples(vec!(
    ///         "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f",
    ///         "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f"));
    /// ```
    #[must_use]
    pub fn samples<T: Into<String>>(mut self, samples: Vec<T>) -> Self {
        self.samples.extend(samples.into_iter().map(Into::into));
        self
    }

    /// Adds a file to a reaction request
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to pass in
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// # async fn exec() -> Result<(), std::io::ApiError> {
    /// // create a reaction with that depends on a file by path
    /// let request = ReactionRequest::new("Combine", "fill_gas")
    ///     .file("reactions.rs");
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await.unwrap()
    /// # });
    /// ```
    #[cfg(feature = "tokio-models")]
    pub async fn file<T: AsRef<Path> + Into<Box<dyn std::error::ApiError + Send + Sync>>>(
        mut self,
        path: T,
    ) -> Result<Self, std::io::ApiError> {
        // try to get the name of this file
        let name = match path.as_ref().file_name() {
            Some(name) => name,
            // if we can't get a file name then assume this is a directory and error
            None => return Err(std::io::ApiError::new(ApiErrorKind::IsADirectory, path)),
        };
        // cast our name to a string lossily
        let clean = name.to_string_lossy().into_owned();
        // open a handle to our file
        let mut file = File::open(&path).await?;
        // try to read in our file
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).await?;
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(&buffer);
        self.buffers.insert(clean, encoded);
        Ok(self)
    }

    /// Adds a buffer to a reaction request
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the buffer to pass in
    /// * `buffer` - The buffer to download before running this reactions jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction with that depends on a file by path
    /// let request = ReactionRequest::new("Combine", "fill_gas")
    ///     .buffer("buffer.txt", "I am a buffer");
    /// ```
    #[must_use]
    pub fn buffer<T: Into<String>, B: AsRef<[u8]>>(mut self, name: T, buffer: B) -> Self {
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(&buffer);
        self.buffers.insert(name.into(), encoded);
        self
    }

    /// Adds a repo to a reaction request
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the buffer to pass in
    /// * `repo` - The repo dependency request to add to this reaction
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ReactionRequest, RepoDependencyRequest};
    ///
    /// // create a reaction with that depends on a repository
    /// let request = ReactionRequest::new("Combine", "fill_gas")
    ///     .repo(RepoDependencyRequest::new("github.com/rustlang/rust").commitish("main"));
    /// ```
    #[must_use]
    pub fn repo(mut self, repo: RepoDependencyRequest) -> Self {
        self.repos.push(repo);
        self
    }

    /// Set a trigger depth for this reaction
    ///
    /// # Arguments
    ///
    /// * `trigger_depth` - The trigger depth to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ReactionRequest;
    ///
    /// // create a reaction with that depends on a repository
    /// let request = ReactionRequest::new("Combine", "fill_gas").trigger_depth(3);
    /// ```
    #[must_use]
    pub fn trigger_depth(mut self, trigger_depth: u8) -> Self {
        self.trigger_depth = Some(trigger_depth);
        self
    }
}

/// Helps serde default the reaction list limit to 50
fn default_list_limit() -> usize {
    50
}

/// The parameters for a reaction list request
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ReactionListParams {
    /// The cursor id to use if one exists
    #[serde(default)]
    pub cursor: usize,
    /// The max amount of reactions to return in on request
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

impl Default for ReactionListParams {
    fn default() -> Self {
        Self {
            cursor: usize::default(),
            limit: default_list_limit(),
        }
    }
}

impl ReactionListParams {
    /// Set the limit in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `limit` - The limit to set
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// A list of reaction names with a cursor
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ReactionList {
    /// A cursor used to page through reaction names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of reaction names
    pub names: Vec<String>,
}

/// A list of reaction details with a cursor
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ReactionDetailsList {
    /// A cursor used to page through reaction details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of reaction details
    pub details: Vec<Reaction>,
}

/// A timestamped log line
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct StageLogLine {
    /// The position of this line when multiple lines have the same timestamp
    pub index: u64,
    /// The line of log data for this timestamp/index
    pub line: String,
}

impl StageLogLine {
    /// Turn a vector of strings into a [`Vec<StageLog>`]
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw strings to turn into stage log lines
    /// * `start` - The first line these logs should be inserted at (line 10 would be 10)
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::StageLogLine;
    ///
    /// // build stage log lines vector
    /// let (lines, end) = StageLogLine::new(vec!("line1", "line2", "line3"), 0);
    /// ```
    #[must_use]
    pub fn new<T: Into<String>>(raw: Vec<T>, start: u64) -> (Vec<Self>, u64) {
        // determine end index value
        let end = start + raw.len() as u64;
        // build the stage log line vector
        let lines = raw
            .into_iter()
            .enumerate()
            .map(|(i, line)| StageLogLine {
                index: i as u64 + start,
                line: line.into(),
            })
            .collect();
        (lines, end)
    }
}

/// A list of log lines to append to a stages logs
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct StageLogsAdd {
    /// The current log index to use as a start for newly added logs
    #[serde(skip)]
    pub index: u64,
    /// The log lines to append to our logs in the backend
    #[serde(default)]
    pub logs: Vec<StageLogLine>,
    /// The return to code to set if one has been returned
    pub return_code: Option<i32>,
}

impl StageLogsAdd {
    /// Adds new logs to be saved to an existing `StageLogsAdd`
    ///
    /// # Arguments
    ///
    /// * `logs` - The logs to save for this job
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::StageLogsAdd;
    ///
    /// let mut logs = StageLogsAdd::default();
    /// let index = logs.add_logs(vec!("line1", "line2", "line3"));
    /// ```
    pub fn add<T: Into<String>>(&mut self, line: T) {
        // build a single stage log line
        let line = StageLogLine {
            index: self.index,
            line: line.into(),
        };
        // update our index
        self.index += 1;
        // add our logs
        self.logs.push(line);
    }

    /// Adds new logs to be saved to an existing [`StageLogsAdd`]
    ///
    /// # Arguments
    ///
    /// * `logs` - The logs to save for this job
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::StageLogsAdd;
    ///
    /// let mut logs = StageLogsAdd::default();
    /// let index = logs.add_logs(vec!("line1", "line2", "line3"));
    /// ```
    pub fn add_logs<T: Into<String>>(&mut self, logs: Vec<T>) {
        // convert our new logs to strings
        let raw: Vec<String> = logs.into_iter().map(Into::into).collect();
        // convert to a vec of stage log lines
        let (logs, end) = StageLogLine::new(raw, self.index);
        // update our index
        self.index = end;
        // add our logs
        self.logs.extend(logs);
    }

    /// Adds new logs to be saved
    ///
    /// # Arguments
    ///
    /// * `logs` - The logs to save for this job
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::StageLogsAdd;
    ///
    /// let logs = StageLogsAdd::default()
    ///     .logs(vec!("line1", "line2", "line3"));
    /// ```
    #[must_use]
    pub fn logs<T: Into<String>>(mut self, logs: Vec<T>) -> Self {
        self.add_logs(logs);
        self
    }

    /// Sets the return code to set
    ///
    /// # Arguments
    ///
    /// * `code` - The return code to set for this job
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::StageLogsAdd;
    ///
    /// let logs = StageLogsAdd::default().code(0);
    /// ```
    #[must_use]
    pub fn code(mut self, code: i32) -> Self {
        // set our return code
        self.return_code = Some(code);
        self
    }

    /// Sets the index to use when adding logs
    ///
    /// # Arguments
    ///
    /// * `index` - The index to start with when adding new logs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::StageLogsAdd;
    ///
    /// let logs = StageLogsAdd::default().index(32);
    /// ```
    #[must_use]
    pub fn index(mut self, index: u64) -> Self {
        // set our index
        self.index = index;
        self
    }
}

/// The logs for a specific stage within a reaction
///
/// This does not have a cursor because the cursor is just the number of log lines to skip
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct StageLogs {
    /// The log lines for a specific stage within a reaction
    pub logs: Vec<String>,
}

/// The different possible statuses for a reaction
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash, enum_utils::FromStr)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum ReactionStatus {
    /// This reaction is created, but is not yet running
    Created,
    /// At least one stage of this reaction has started
    Started,
    /// This reaction has completed
    Completed,
    /// This reaction has failed due to an error
    Failed,
}

impl std::fmt::Display for ReactionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ReactionStatus::Created => write!(f, "Created"),
            ReactionStatus::Started => write!(f, "Started"),
            ReactionStatus::Completed => write!(f, "Completed"),
            ReactionStatus::Failed => write!(f, "Failed"),
        }
    }
}

impl From<JobHandleStatus> for ReactionStatus {
    /// Convert a [`JobHandleStatus`] into its correct `ReactionStatus`
    ///
    /// # Arguments
    ///
    /// * `job_status` - The [`JobHandleStatus`] to convert
    fn from(job_status: JobHandleStatus) -> Self {
        match job_status {
            JobHandleStatus::Waiting
            | JobHandleStatus::Proceeding
            | JobHandleStatus::Sleeping
            | JobHandleStatus::Checkpointed => ReactionStatus::Started,
            JobHandleStatus::Completed => ReactionStatus::Completed,
            JobHandleStatus::Errored => ReactionStatus::Failed,
        }
    }
}

/// A reaction built around a pipeline
///
/// This is used to track jobs across a single run of a pipeline
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Reaction {
    /// The uuidv4 that identifies this reaction
    pub id: Uuid,
    /// The group this reaction is in
    pub group: String,
    /// The creator of this reaction
    pub creator: String,
    /// The pipeline this reaction is built around
    pub pipeline: String,
    /// The current status of this reaction
    pub status: ReactionStatus,
    /// The current stage of this reaction
    pub current_stage: u64,
    /// The current stages progress
    pub current_stage_progress: u64,
    /// The current stages length,
    pub current_stage_length: u64,
    /// The args for this reaction (passed to all jobs)
    pub args: ReactionArgs,
    /// The timestamp this reactions SLA expires at
    pub sla: DateTime<Utc>,
    /// The uuidv4's of the jobs in this pipeline
    pub jobs: Vec<Uuid>,
    /// The tags this reaction can be listed under
    pub tags: Vec<String>,
    /// The parent reaction for this reaction if its a sub reaction
    pub parent: Option<Uuid>,
    /// The number of subreactions for this reaction
    pub sub_reactions: u64,
    /// The number of completed subreactions for this reaction
    pub completed_sub_reactions: u64,
    /// Whether the currently active stage of this reaction contains generators or not
    pub generators: Vec<Uuid>,
    /// A list of sample sha256s to download before executing this reactions jobs
    pub samples: Vec<String>,
    /// A list of ephemeral files to download
    pub ephemeral: Vec<String>,
    /// A list of ephemeral files from any parent reactions and what parent reaction its tied to
    pub parent_ephemeral: HashMap<String, Uuid>,
    /// A list of repos to download before executing this reactions jobs
    pub repos: Vec<RepoDependency>,
    /// This reactions depth in triggers if this reaction was caused by a trigger
    pub trigger_depth: Option<u8>,
}

impl PartialEq<ReactionRequest> for Reaction {
    /// Check if a [`ReactionRequest`] and a [`Reaction`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The `ReactionRequest` to compare against
    fn eq(&self, request: &ReactionRequest) -> bool {
        // build a corrected list of tags for this reaction
        let mut corrected = request.tags.clone();
        corrected.append(&mut request.samples.clone());
        corrected.push(self.creator.clone());
        // make sure all fields are the same
        same!(self.group, request.group);
        same!(self.pipeline, request.pipeline);
        matches_vec!(self.tags, corrected);
        same!(self.args, request.args);
        matches_vec!(self.samples, request.samples);
        // make sure we have the same number of ephemeral files
        same!(self.ephemeral.len(), request.buffers.len());
        // make sure our reaction depth is the same
        same!(self.trigger_depth, request.trigger_depth);
        true
    }
}

/// The response given when creating a reaction
#[derive(Serialize, Deserialize, Debug)]
pub struct ReactionCreation {
    /// The uuidv4 of the created reaction
    pub id: Uuid,
}

/// A Reaction expiration object
#[derive(Serialize, Deserialize, Debug)]
pub struct ReactionExpire {
    /// The cmd to use to expire this data
    pub cmd: String,
    /// The list to remove this id from
    pub list: String,
    // The id to remove
    pub id: String,
}

impl ReactionExpire {
    pub fn new<T: Into<String>, U: Into<String>>(cmd: T, list: U, id: &Uuid) -> Self {
        ReactionExpire {
            cmd: cmd.into(),
            list: list.into(),
            id: id.to_string(),
        }
    }
}

// The arguments for all images in a reaction
pub type ReactionArgsUpdate = HashMap<String, GenericJobArgsUpdate>;

/// A Reaction arg update request that allows stages to change later stages inputs
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ReactionUpdate {
    /// A hashmap of updated args to overlay
    #[serde(default)]
    pub args: ReactionArgsUpdate,
    /// The new tags this reaction should be listed under
    #[serde(default)]
    pub add_tags: Vec<String>,
    /// The tags this reaction should no longer be listed under
    #[serde(default)]
    pub remove_tags: Vec<String>,
    /// The ephemeral files to add to this reaction
    #[serde(default)]
    pub ephemeral: HashMap<String, String>,
}

impl ReactionUpdate {
    /// Adds a stage args update to this reaction update
    ///
    /// # Arguments
    ///
    /// * `stage` - The name of the stage these updated args are for
    /// * `update` - The update to apply to this stages args
    ///
    ///  # Example
    ///
    ///  ```
    ///  use thorium::models::{ReactionUpdate, GenericJobArgsUpdate};
    ///
    ///  // build an update for a specific stage
    ///  let harvester_update = GenericJobArgsUpdate::default()
    ///     .positionals(vec!("soy"))
    ///     .kwarg("--field", vec!("west-4"));
    ///  // build a reaction update with this stage update
    ///  let update = ReactionUpdate::default()
    ///     .arg("harvest", harvester_update);
    ///  ```
    #[must_use]
    pub fn arg<T: Into<String>>(mut self, stage: T, update: GenericJobArgsUpdate) -> Self {
        // insert new stage args update
        self.args.insert(stage.into(), update);
        self
    }

    /// Adds a new tag to list/find this reaction with
    ///
    /// # Arguments
    ///
    /// * `tag` - The new tag for this reaction
    ///
    ///  # Example
    ///
    ///  ```
    ///  use thorium::models::{ReactionUpdate, GenericJobArgsUpdate};
    ///
    ///  // build a reaction update with a new tag
    ///  let update = ReactionUpdate::default()
    ///     .tag("SoyHarvester");
    ///  ```
    #[must_use]
    pub fn tag<T: Into<String>>(mut self, tag: T) -> Self {
        // add new tag
        self.add_tags.push(tag.into());
        self
    }

    /// Removes a tag from this reaction
    ///
    /// # Arguments
    ///
    /// * `tag` - The tag to remove from this reaction
    ///
    ///  # Example
    ///
    ///  ```
    ///  use thorium::models::{ReactionUpdate, GenericJobArgsUpdate};
    ///
    ///  // build a reaction update to remove a tag
    ///  let update = ReactionUpdate::default()
    ///     .remove_tag("BananaHarvester");
    ///  ```
    #[must_use]
    pub fn remove_tag<T: Into<String>>(mut self, tag: T) -> Self {
        // add tag to be removed
        self.remove_tags.push(tag.into());
        self
    }
}

impl PartialEq<ReactionUpdate> for Reaction {
    /// Check if a [`ReactionRequest`] and a [`Reaction`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The [`ReactionRequest`] to compare against
    fn eq(&self, request: &ReactionUpdate) -> bool {
        // make sure all fields are the same
        matches_adds!(self.tags, request.add_tags);
        matches_removes!(self.tags, request.remove_tags);
        let args_check = request
            .args
            .iter()
            .all(|(stage, arg)| &self.args[stage] == arg);
        same!(args_check, true);
        true
    }
}
