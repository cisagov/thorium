//! Wrappers for interacting with deadlines within Thorium with different backends
//! Currently only Redis is supported

use chrono::prelude::*;
use uuid::Uuid;

use super::RawJob;

/// A deadline for when a job must be started by
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Deadline {
    /// The group this job is in
    pub group: String,
    /// The pipeline this job is from
    pub pipeline: String,
    /// The stage this job is for
    pub stage: String,
    /// The user that created this job
    pub creator: String,
    /// The job this is for
    pub job_id: Uuid,
    /// The reaction this job is apart of
    pub reaction: Uuid,
    /// The timestamp the job must be started by
    pub deadline: chrono::DateTime<Utc>,
}

impl Deadline {
    /// Returns a string that denotes the job class/key this deadline is for
    ///
    /// This class/key is tied to the creator, group, pipeline, and stage for this deadline.
    pub fn key(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.creator, self.group, self.pipeline, self.stage
        )
    }
}

impl From<&RawJob> for Deadline {
    /// Cast a reference to a RawJob into deadline
    ///
    /// # Arguments
    ///
    /// * `job` - A reference to a RawJob
    fn from(job: &RawJob) -> Self {
        Deadline {
            group: job.group.to_owned(),
            pipeline: job.pipeline.to_owned(),
            stage: job.stage.to_owned(),
            creator: job.creator.to_owned(),
            job_id: job.id.to_owned(),
            reaction: job.reaction.to_owned(),
            deadline: job.deadline,
        }
    }
}

impl From<RawJob> for Deadline {
    /// Cast a RawJob into deadline
    ///
    /// # Arguments
    ///
    /// * `job` - A RawJob
    fn from(job: RawJob) -> Self {
        Deadline {
            group: job.group,
            pipeline: job.pipeline,
            stage: job.stage,
            creator: job.creator,
            job_id: job.id,
            reaction: job.reaction,
            deadline: job.deadline,
        }
    }
}

// A partial deadline fragment to help with casting from StreamObjs to deadlines
#[derive(Deserialize)]
#[allow(dead_code)]
pub(super) struct DeadlineFragment {
    /// The group this job is in
    pub group: String,
    /// The pipeline this job is from
    pub pipeline: String,
    /// The stage this job is for
    pub stage: String,
    /// The user that created this job
    pub creator: String,
    /// The job this is for
    pub job_id: Uuid,
    /// The reaction this job is apart of
    pub reaction: Uuid,
}
