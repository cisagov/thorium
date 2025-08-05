//! Wrappers for interacting with status logs within Thorium with different backends
//! Currently only Redis is supported

use chrono::DateTime;
use std::collections::HashMap;

use super::jobs::JobResetRequestor;

/// Actions that could occur in the status log from a Job object
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum JobActions {
    /// Created a Job
    Created,
    /// Job has begun execution
    Running,
    /// Job has been reset
    Reset(JobResetRequestor),
    /// Job completed
    Completed,
    /// Job has ran into an error
    Errored,
}

/// Actions that could occur in the status log from a Reaction object
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ReactionActions {
    /// Reaction created
    Created,
    /// Reaction completed
    Completed,
    /// The reaction is proceeding to the next group of stages/images
    Proceeding,
    /// Any jobs for this reaction will not be spawned
    Disabled,
    /// Jobs for this reaction will be spawned/executed
    Enabled,
    /// One of the jobs for this reaction has ran into an error and this reaction has failed
    Failed,
}

/// A request for a status update to place into the status log
pub struct StatusRequest {
    /// The group the pipeline/reaction this status update is for
    pub group: String,
    /// The pipeline this status update is for
    pub pipeline: String,
    /// The reaction this status update is for
    pub reaction: String,
    /// The action that occurred
    pub action: Actions,
    /// The update that was applied
    pub update: HashMap<String, String>,
}

/// Actions that could occur in the status log
#[derive(Debug, Serialize, Deserialize, strum::Display)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum Actions {
    /// Reaction created
    ReactionCreated,
    /// Reaction completed
    ReactionCompleted,
    /// The reaction is proceeding to the next group of stages/images
    ReactionProceeding,
    /// Jobs for this reaction will be spawned/executed
    ReactionEnabled,
    /// Any jobs for this reaction will not be spawned
    ReactionDisabled,
    /// One of the jobs for this reaction has ran into an error and this reaction has failed
    ReactionFailed,
    /// Created a Job
    JobCreated,
    /// Job has begun execution
    JobRunning,
    /// Job has been reset
    JobReset(JobResetRequestor),
    /// Job completed
    JobCompleted,
    /// Job has ran into an error
    JobFailed,
}

/// An individual status update
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct StatusUpdate {
    /// The group the pipeline/reaction this status update is for
    pub group: String,
    /// The pipeline this status update is for
    pub pipeline: String,
    /// The reaction this status update is for
    pub reaction: String,
    /// The action that occurred in this update
    pub action: Actions,
    /// The timestamp this occurred
    pub timestamp: DateTime<chrono::Utc>,
    /// A message or reason why this action occurred
    pub msg: Option<String>,
    /// The update that occurred
    pub update: HashMap<String, String>,
}
