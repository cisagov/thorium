//! Wrappers for interacting with jobs within Thorium with different backends
//! Currently only Redis is supported

use chrono::prelude::*;
use std::collections::HashMap;
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::db;
use crate::models::{
    Checkpoint, GenericJob, GenericJobArgs, Group, ImageJobInfo, ImageScaler, JobDetailsList,
    JobHandleStatus, JobList, JobResets, JobStatus, Pipeline, RawJob, Reaction, RunningJob,
    StageLogsAdd, Stream, StreamObj, User, WorkerName,
};
use crate::utils::{ApiError, Shared};
use crate::{
    deserialize, deserialize_ext, deserialize_opt, extract, is_admin, not_found, serialize,
};

impl JobList {
    /// Creates new raw jobs list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used to page to the next list of jobs
    /// * `names` - A list of job IDs
    pub(super) fn new(cursor: Option<usize>, names: Vec<Uuid>) -> Self {
        JobList { cursor, names }
    }
}

impl JobDetailsList {
    /// Creates a new image details list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used to page to the next list of jobs
    /// * `details` - A list of job details
    pub(super) fn new(cursor: Option<usize>, details: Vec<RawJob>) -> Self {
        JobDetailsList { cursor, details }
    }
}

impl TryFrom<&StreamObj> for RunningJob {
    type Error = ApiError;
    /// Cast a StreamObj to a RunningJob
    ///
    /// # Arguments
    ///
    /// * `obj` - The StreamObj to cast
    fn try_from(obj: &StreamObj) -> Result<Self, Self::Error> {
        // ingest data from stream object
        // error can't be deserialized so were ok wrapping this
        Ok(deserialize!(&obj.data))
    }
}

impl RawJob {
    /// Builds a new job object
    ///
    /// # Arguments
    ///
    /// * `reaction` - The Reaction this job is a part of
    /// * `stage` - The name of the stage for this job
    /// * `deadline` - The deadline this must be executed by
    /// * `info` - A hashmap of ImageJobInfo including this jobs image
    pub async fn build<'a>(
        reaction: &Reaction,
        stage: &str,
        deadline: DateTime<Utc>,
        info: &HashMap<&'a String, ImageJobInfo>,
    ) -> Result<RawJob, ApiError> {
        // get args
        let args = match reaction.args.get(stage) {
            Some(args) => serde_json::to_string(args)?,
            None => "{}".to_owned(),
        };

        // create job object
        let cast = RawJob {
            reaction: reaction.id,
            id: Uuid::new_v4(),
            group: reaction.group.clone(),
            pipeline: reaction.pipeline.clone(),
            stage: stage.to_string(),
            creator: reaction.creator.clone(),
            args,
            status: JobStatus::Created,
            deadline,
            worker: None,
            parent: reaction.parent,
            generator: info[&stage.to_owned()].generator,
            scaler: info[&stage.to_owned()].scaler,
            samples: reaction.samples.clone(),
            ephemeral: reaction.ephemeral.clone(),
            parent_ephemeral: reaction.parent_ephemeral.clone(),
            repos: reaction.repos.clone(),
            trigger_depth: reaction.trigger_depth,
        };
        Ok(cast)
    }

    /// Try to cast a HashMap of strings into a RawJob
    ///
    /// # Arguments
    ///
    /// * `raw` - The HashMap to cast into a RawJob
    #[instrument(name = "RawJob::from_data", skip_all, err(Debug))]
    pub fn from_data(mut raw: HashMap<String, String>) -> Result<Self, ApiError> {
        // error if hashmap does not contain the required values
        if !raw.contains_key("reaction") || !raw.contains_key("id") {
            // check if this job was malformed or is just missing
            if raw.is_empty() {
                // this job is just empty
                event!(Level::ERROR, empty = true);
            } else {
                // this job contains data but not the required values
                // log the data that was missing required values
                event!(Level::ERROR, malformed = true, data = format!("{raw:?}"));
            }
            // tell this user this job couldn't be found
            return not_found!("Job not found".to_owned());
        }
        // cast our raw data to a Job
        let job = RawJob {
            reaction: Uuid::parse_str(&raw["reaction"])?,
            id: Uuid::parse_str(&raw["id"])?,
            group: extract!(raw, "group"),
            pipeline: extract!(raw, "pipeline"),
            stage: extract!(raw, "stage"),
            creator: extract!(raw, "creator"),
            args: extract!(raw, "args"),
            status: deserialize_ext!(raw, "status"),
            deadline: deserialize_ext!(raw, "deadline"),
            worker: deserialize_ext!(raw, "worker"),
            parent: deserialize_opt!(raw, "parent", Uuid::parse_str),
            generator: deserialize_ext!(raw, "generator", false),
            scaler: deserialize_ext!(raw, "scaler", ImageScaler::default()),
            samples: deserialize_ext!(raw, "samples", Vec::default()),
            ephemeral: deserialize_ext!(raw, "ephemeral", Vec::default()),
            parent_ephemeral: deserialize_ext!(raw, "parent_ephemeral", HashMap::default()),
            repos: deserialize_ext!(raw, "repos", Vec::default()),
            trigger_depth: deserialize_opt!(raw, "trigger_depth"),
        };
        Ok(job)
    }

    /// Gets a job object from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting a job
    /// * `id` - The id of the job to retrieve
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "RawJob::get", skip_all, err(Debug))]
    pub async fn get(user: &User, id: &Uuid, shared: &Shared) -> Result<(Group, RawJob), ApiError> {
        // get job from the backend
        let job = db::jobs::get(id, shared).await?;
        // ensure we have access to this jobs group
        let group = Group::authorize(user, &job.group, shared).await?;
        Ok((group, job))
    }

    /// Lists all job details in a list of jobs
    ///
    /// # Arguments
    ///
    /// * `jobs` - The job IDs to get the details for
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "RawJob::list_details", skip_all, err(Debug))]
    pub async fn list_details(jobs: JobList, shared: &Shared) -> Result<JobDetailsList, ApiError> {
        // get the details on this jobs in our list
        db::jobs::list_details(jobs, shared).await
    }

    /// Proceeds with a job
    ///
    /// This will set the jobs status to complete and check if all other jobs in that stage have
    /// completed. If they have it will then create the next stage of the reactions jobs.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is proceeding with a job
    /// * `group` - The name of the group this job is tied to
    /// * `runtime` - The amount of time it took to execute this job in seconds
    /// * `logs` - The logs to save for this job
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "RawJob::proceed", skip_all, err(Debug))]
    pub async fn proceed<'a>(
        self,
        user: &User,
        group: &Group,
        runtime: u64,
        logs: StageLogsAdd,
        shared: &Shared,
    ) -> Result<JobHandleStatus, ApiError> {
        // make sure this user can proceed with jobs from this group
        group.editable(user)?;
        // use correct backend to handle starting job
        db::jobs::proceed(self, runtime, logs, shared).await
    }

    /// ApiErrors out a job
    ///
    /// This will set the jobs status to error and fail out the rest of the pipeline.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is erroring out a job
    /// * `group` - The name of the group this job is tied to
    /// * `logs` - The logs to save for this job
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "RawJob::error", skip_all, err(Debug))]
    pub async fn error(
        self,
        user: &User,
        group: &Group,
        logs: StageLogsAdd,
        shared: &Shared,
    ) -> Result<JobHandleStatus, ApiError> {
        // make sure this user can error out jobs from this group
        group.editable(user)?;
        // use correct backend to handle starting job
        db::jobs::error(self, logs, shared).await
    }

    /// Checkpoints a job
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is erroring out a job
    /// * `group` - The name of the group this job is tied to
    /// * `checkpoint` - The checkpoint to set
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Job::checkpoint", skip(self, user, group, shared), err(Debug))]
    pub async fn checkpoint(
        mut self,
        user: &User,
        group: &Group,
        checkpoint: Checkpoint,
        shared: &Shared,
    ) -> Result<JobHandleStatus, ApiError> {
        // make sure this user can checkpoint jobs from this group
        group.editable(user)?;
        // inject in this jobs new checkpoint arg
        let mut args: GenericJobArgs = deserialize!(&self.args);
        args.kwargs
            .insert("--checkpoint".to_owned(), vec![checkpoint.data]);
        self.args = serialize!(&args);
        // update this jobs args in redis
        db::jobs::set_args(&self, shared).await
    }

    /// Sets a job status as sleeping in redis
    ///
    /// This does not complete a job and complete must still be called.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is sleeping this job
    /// * `group` - The name of the group this job is tied to
    /// * `checkpoint` - The checkpoint to set
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Job::sleep", skip(self, user, group, shared), err(Debug))]
    pub async fn sleep(
        self,
        user: &User,
        group: &Group,
        checkpoint: Checkpoint,
        shared: &Shared,
    ) -> Result<JobHandleStatus, ApiError> {
        // make sure this user can sleep generators from this group
        group.editable(user)?;
        // use correct backend to handle starting job
        db::jobs::sleep(self, checkpoint, shared).await
    }

    /// Resets jobs in bulk
    ///
    /// This will set all of the jobs statuses back to created except for any
    /// completed or failed jobs.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is erroring out a job
    /// * `resets` - The jobs to reset
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Job::bulk_reset", skip_all, err(Debug))]
    pub async fn bulk_reset(
        user: &User,
        resets: JobResets,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // make sure we are an admin
        is_admin!(user);
        // use correct backend to handle starting job
        db::jobs::bulk_reset(resets, shared).await
    }

    /// Lists running jobs between two timestamps
    ///
    /// This reads jobs from the running jobs stream and can only be called by an admin.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is performing this request
    /// * `scaler` - The scaler to get running jobs for
    /// * `start` - The timestamp in seconds (unix epoch) to start reading running jobs from
    /// * `end` - The timestamp in seconds (unix epoch) to stop reading running jobs at
    /// * `skip` - The number of entries in the running stream to skip
    /// * `limit` - The most running jobs to return at once
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Job::runner", skip(user, shared), err(Debug))]
    pub async fn running(
        user: &User,
        scaler: ImageScaler,
        start: i64,
        end: i64,
        skip: u64,
        limit: u64,
        shared: &Shared,
    ) -> Result<Vec<RunningJob>, ApiError> {
        // make sure we are an admin
        is_admin!(user);
        // cast our scaler to a string
        let ns = scaler.as_str();
        // read objects from streams
        let objects: Vec<StreamObj> = Stream::read(
            user, "system", &ns, "running", start, end, skip, limit, shared,
        )
        .await?;
        // cast stream objects to running jobs
        let running: Vec<RunningJob> = objects
            .iter()
            .map(RunningJob::try_from)
            .filter_map(Result::ok)
            .collect();
        Ok(running)
    }

    /// Serializes the parts of a job that go into the deadline stream
    pub fn stream_data(&self) -> String {
        // cast deadline data to stream data without timestamp
        // were using a format macro so we get a consistent order
        format!("{{\"group\":\"{}\",\"pipeline\":\"{}\",\"stage\":\"{}\",\"creator\":\"{}\",\"job_id\":\"{}\",\"reaction\":\"{}\"}}",
            self.group,
            self.pipeline,
            self.stage,
            self.creator,
            self.id,
            self.reaction)
    }
}

impl GenericJob {
    /// Claims a requested number of pending generic jobs
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is claiming a job
    /// * `group` - The name of the group this job is tied to
    /// * `pipeline` - The name of the pipeline is tied to
    /// * `stage` - The name of the stage for this job
    /// * `limit` - The number of jobs to claim at once
    /// * `worker` - The worker that is claiming jobs
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[allow(clippy::too_many_arguments)]
    #[instrument(
        name = "GenericJob::claim",
        skip(user, group, pipeline, worker, shared),
        err(Debug)
    )]
    pub async fn claim(
        user: &User,
        group: &Group,
        pipeline: &Pipeline,
        stage: &str,
        limit: usize,
        worker: &WorkerName,
        shared: &Shared,
    ) -> Result<Vec<GenericJob>, ApiError> {
        // make sure this user can claim jobs from this group
        group.editable(user)?;
        // claim job from backend if one exists
        let raw_claims = db::jobs::claim(user, pipeline, stage, limit, worker, shared).await?;
        // cast claims to GenericJobs
        let claims = raw_claims
            .into_iter()
            .map(GenericJob::try_from)
            .filter_map(Result::ok)
            .collect();
        Ok(claims)
    }

    /// Gets a job object from the backend
    ///
    /// # Arguments
    ///
    /// * `_group` - The Group the job is in (this is passed in for auth purposes)
    /// * `id` - The id of the job to retrieve
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "GenericJob::claim", skip_all, err(Debug))]
    pub async fn get(_group: &Group, id: &Uuid, shared: &Shared) -> Result<Self, ApiError> {
        let raw = db::jobs::get(id, shared).await;
        GenericJob::try_from(raw?)
    }
}

impl TryFrom<RawJob> for GenericJob {
    type Error = ApiError;

    /// Try to ast a RawJob into a GenericJob
    ///
    /// # Arguments
    ///
    /// * `raw` - The RawJob to cast to a GenericJob
    fn try_from(raw: RawJob) -> Result<Self, Self::Error> {
        // cast to GenericJob
        let cast = GenericJob {
            reaction: raw.reaction,
            id: raw.id,
            group: raw.group,
            pipeline: raw.pipeline,
            stage: raw.stage,
            creator: raw.creator,
            args: deserialize!(&raw.args),
            status: raw.status,
            deadline: raw.deadline,
            parent: raw.parent,
            generator: raw.generator,
            samples: raw.samples,
            ephemeral: raw.ephemeral,
            parent_ephemeral: raw.parent_ephemeral,
            repos: raw.repos,
            trigger_depth: raw.trigger_depth,
        };
        Ok(cast)
    }
}

//impl TryFrom<HashMap<String, String>> for RawJob {
//    type Error = ApiError;
//
//}
