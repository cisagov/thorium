use axum::http::StatusCode;
use bb8_redis::redis::cmd;
use chrono::prelude::*;
use std::collections::{HashMap, HashSet};
use tracing::{event, instrument, span, Level};
use uuid::Uuid;

use super::keys::{images::ImageKeys, jobs::JobKeys, reactions::ReactionKeys, streams::StreamKeys};
use super::{logs, reactions, streams, system};
use crate::models::{
    Checkpoint, GenericJobArgs, ImageScaler, JobActions, JobDetailsList, JobHandleStatus, JobList,
    JobReactionIds, JobResets, JobStatus, Pipeline, RawJob, Reaction, ReactionStatus, RunningJob,
    StageLogsAdd, StatusRequest, StatusUpdate, StreamObj, User, Worker, WorkerName,
};
use crate::utils::{ApiError, Shared};
use crate::{
    conflict, conn, deserialize, force_serialize, internal_err, not_found, query, serialize,
};

/// Builds the status queue function call
macro_rules! status_queue {
    ($job:expr, $status:expr, $shared:expr) => {
        JobKeys::status_queue(
            &$job.group,
            &$job.pipeline,
            &$job.stage,
            &$job.creator,
            $status,
            $shared,
        )
    };
    ($job:expr, $user:expr, $status:expr, $shared:expr) => {
        JobKeys::status_queue(
            &$job.group,
            &$job.pipeline,
            &$job.stage,
            &$user,
            $status,
            $shared,
        )
    };
}

/// Builds a [`redis::Pipeline`] with commands to create a [`RawJob`] in Redis
///
/// # Arguments
///
/// * `pipe` - The Redis [`redis::Pipeline`] to build on top of
/// * `job` - The job object to add to redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::build", skip_all, err(Debug))]
pub async fn build<'a>(
    pipe: &'a mut redis::Pipeline,
    cast: &'a RawJob,
    shared: &'a Shared,
) -> Result<&'a mut redis::Pipeline, ApiError> {
    // build job keys
    let keys = JobKeys::new(cast, shared);
    // cast our job claim data
    let job_claim = JobReactionIds::new(cast.id, cast.reaction);
    // cast to stream object
    let stream_obj = StreamObj::from(cast);
    // build pipeline to add job id to the right sorted sets
    let pipe = pipe
        // user requested job
        .cmd("hsetnx").arg(&keys.data).arg("reaction").arg(&cast.reaction.to_string())
        .cmd("hsetnx").arg(&keys.data).arg("id").arg(&cast.id.to_string())
        .cmd("hsetnx").arg(&keys.data).arg("group").arg(&cast.group)
        .cmd("hsetnx").arg(&keys.data).arg("pipeline").arg(&cast.pipeline)
        .cmd("hsetnx").arg(&keys.data).arg("stage").arg(&cast.stage)
        .cmd("hsetnx").arg(&keys.data).arg("creator").arg(&cast.creator)
        .cmd("hsetnx").arg(&keys.data).arg("scaler").arg(&serialize!(&cast.scaler))
        .cmd("hsetnx").arg(&keys.data).arg("args").arg(&cast.args)
        .cmd("hsetnx").arg(&keys.data).arg("status").arg(&serialize!(&cast.status))
        .cmd("hsetnx").arg(&keys.data).arg("deadline").arg(&serialize!(&cast.deadline))
        .cmd("hsetnx").arg(&keys.data).arg("worker").arg(&serialize!(&cast.worker))
        .cmd("sadd").arg(&ReactionKeys::jobs(&cast.group, &cast.reaction, shared)).arg(&cast.id.to_string())
        .cmd("zadd").arg(&keys.status).arg(cast.deadline.timestamp()).arg(&serialize!(&job_claim))
        .cmd("zadd").arg(&StreamKeys::system_scaler(cast.scaler, "deadlines", shared))
            .arg(stream_obj.timestamp).arg(stream_obj.data);
    // inject the parent field if this job has a parent
    if let Some(parent) = cast.parent {
        pipe.cmd("hsetnx").arg(&keys.data).arg("parent").arg(parent.to_string());
    }
    // if this image is a generator then increment our reactions current active generators
    if cast.generator {
        // build key to reaction data
        let reaction_key = ReactionKeys::generators(&cast.group, &cast.reaction, shared);
        // increment number of active generators by 1
        pipe.cmd("sadd").arg(reaction_key).arg(&cast.id.to_string())
            .cmd("hset").arg(&keys.data).arg("generator").arg(serialize!(&true));
    } else {
        // This job is not a generator
        pipe.cmd("hset").arg(&keys.data).arg("generator").arg(serialize!(&false));
    }
    // if this job has samples then serialize and save those
    if !cast.samples.is_empty() {
        pipe.cmd("hset").arg(&keys.data).arg("samples").arg(serialize!(&cast.samples));
    }
    // if this job has ephemeral files then serialize and save those
    if !cast.ephemeral.is_empty() {
        pipe.cmd("hset").arg(&keys.data).arg("ephemeral").arg(serialize!(&cast.ephemeral));
    }
    // if this job has repos then serialize and save those
    if !cast.repos.is_empty() {
        pipe.cmd("hset").arg(&keys.data).arg("repos").arg(serialize!(&cast.repos));
    }
    // if this job has a trigger depth then set it
    if let Some(trigger_depth) = &cast.trigger_depth {
        pipe.cmd("hsetnx").arg(&keys.data).arg("trigger_depth").arg(trigger_depth);
    }
    // create status log for this job
    let update_cast = StatusUpdate::new(StatusRequest::from_job(cast, JobActions::Created), None);
    logs::build(pipe, &[update_cast], shared)?;
    Ok(pipe)
}

/// Gets a job from the redis backend
///
/// # Arguments
///
/// * `id` - The id of the job to retrieve
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::jobs::get", skip_all, err(Debug))]
pub async fn get(id: &Uuid, shared: &Shared) -> Result<RawJob, ApiError> {
    // build key to job data
    let data_key = JobKeys::data(id, shared);
    // get all keys containing our job object
    let raw: HashMap<String, String> = query!(cmd("hgetall").arg(&data_key), shared).await?;
    if raw.contains_key("id") {
        RawJob::from_data(raw)
    } else {
        not_found!(format!("job {} has no data", &id))
    }
}

/// Get what scaler each job is under
///
/// # Arguments
///
/// * `ids` - The IDs of the job to get the scalers for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::jobs::get_scalers", skip_all, fields(ids_len = ids.len()), err(Debug))]
pub async fn get_scalers(
    ids: Vec<Uuid>,
    shared: &Shared,
) -> Result<HashMap<ImageScaler, Vec<Uuid>>, ApiError> {
    // build a pipeline to get these jobs scalers
    let mut pipe = redis::pipe();
    // add the command for each job
    for job in &ids {
        // build key to job data
        let data_key = JobKeys::data(job, shared);
        // get this jobs scaler
        pipe.cmd("hget").arg(data_key).arg("scaler");
    }
    // get all of our jobs scalers
    let scalers: Vec<Option<String>> = pipe.query_async(conn!(shared)).await?;
    // assume we only have one scaler for these jobs
    let mut scaler_map = HashMap::with_capacity(1);
    // build the map of scalers for these jobs
    for (job, scaler) in ids.into_iter().zip(scalers.iter()) {
        // check if this job has data in redis
        match scaler {
            Some(scaler) => {
                // cast our scaler to its enum
                let scaler = match serde_json::from_str(scaler) {
                    Ok(scaler) => scaler,
                    Err(error) => {
                        // log that we failed to deserialize scaler
                        event!(
                            Level::ERROR,
                            msg = "Failed to deserialize scaler",
                            job = job.to_string(),
                            scaler,
                            error = error.to_string()
                        );
                        // return an internal error
                        return internal_err!();
                    }
                };
                // get an entry to this scalers jobs
                let entry: &mut Vec<Uuid> = scaler_map.entry(scaler).or_default();
                // add our job
                entry.push(job);
            }
            None => event!(
                Level::ERROR,
                msg = "Scaler not found",
                job = job.to_string()
            ),
        }
    }
    Ok(scaler_map)
}

/// Get the worker for a set of jobs
///
/// # Arguments
///
/// * `jobs` - The jobs to get workers for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::get_worker", skip_all, err(Debug))]
async fn get_workers<'a>(jobs: &'a HashSet<&Uuid>, shared: &Shared) -> Result<HashMap<&'a Uuid, String>, ApiError> {
     // assume all jobs have a worker assigned to them
     let mut map = HashMap::with_capacity(jobs.len());
     // build a pipeline to get the workers for all jobs
    let mut pipe = redis::pipe();
    // get each jobs worker
    for job in jobs.iter() {
        // build the key to this jobs data
        let key = JobKeys::data(job, shared);
        // add the command to get this jobs data
        pipe.cmd("hget").arg(key).arg("worker");
    }
    // execute this redis pipeline
    let workers: Vec<Option<String>> = pipe.query_async(conn!(shared)).await?;
    // build the map of workers
    for (job, worker) in jobs.iter().zip(workers.into_iter()) {
        // if a worker was set then add it our map
        if let Some(worker) = worker {
            // try to deserialize this worker
            let worker = deserialize!(&worker);
            // add this jobs worker
            map.insert(*job, worker);
        }
    }
     Ok(map)  
}

/// Prune a dangling job
///
/// # Arguments
///
/// * `scaler` - The scaler the target jobs were spawned under
/// * `worker` - The worker that is trying to claim jobs
/// * `dest` - The running job stream to remove dangling jobs from
/// * `job` - The id of the dangling job
/// * `reaction` - The id for the reaction for this job
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::prune_dangling", skip_all, err(Debug))]
async fn prune_dangling(
    scaler: ImageScaler,
    worker: &Worker,
    dest: &str,
    job: Uuid,
    reaction: Uuid,
    shared: &Shared
) -> Result<(), ApiError> {
    // build a redis pipeline to prune this dangling job
    let mut pipe = redis::pipe();
    // build the json objects to remove from the deadlines and running sorted sets
    let deadline_obj = format!(
        "{{\"group\":\"{}\",\"pipeline\":\"{}\",\"stage\":\"{}\",\"creator\":\"{}\",\"job_id\":\"{}\",\"reaction\":\"{}\"}}",
        worker.group,
        worker.pipeline,
        worker.stage,
        worker.user,
        job,
        reaction);
    // build the object to remove from the running stream
    let running_obj = format!("{{\"job_id\":\"{job}\",\"worker\":\"{}\"}}", worker.name);
    // build the entry for the status stream
    let status_entry = JobReactionIds::new(job, reaction);
    // remove this jobs data
    pipe.cmd("del").arg(JobKeys::data_str(&job, shared))
        .cmd("zrem").arg(&dest).arg(serialize!(&status_entry))
        .cmd("zrem")
            .arg(&StreamKeys::system_scaler(scaler, "deadlines", shared)).arg(deadline_obj)
        .cmd("zrem")
            .arg(&StreamKeys::system_scaler(scaler, "running", shared)).arg(running_obj);
    // execute this pipeline
    let _:() = pipe.query_async(conn!(shared)).await?;
    Ok(())
}

/// Response from Redis when claiming jobs
pub type JobData = (HashMap<String, String>, bool, bool, bool, bool);

/// Pops a requested number of jobs from the job queue
///
/// # Arguments
///
/// * `scaler` - The scaler the target jobs were spawned under
/// * `worker` - The worker that is claiming jobs
/// * `src` - The created job stream to claim jobs from
/// * `dest` - The running job stream to place now running jobs into
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::pop_job", skip(worker, shared), err(Debug))]
pub async fn pop_job(scaler: ImageScaler, worker: &Worker, src: &str, dest: &str, shared: &Shared) -> Result<Option<RawJob>, ApiError> {
    // keep trying to claim a job until we get a valid one or our queue is empty
    loop {
        // claim the job with the lowest score
        let raw_claim: Vec<(String, f64)> = query!(cmd("zpopmin").arg(src), shared).await?;
        // if we claimed a job then update its data
        if let Some((raw, score)) = raw_claim.first() {
            // deserialize our job claim data
            let job_info: JobReactionIds = deserialize!(raw);
            // get claimed job data
            let mut pipe = redis::pipe();
            // get all of this jobs data
            pipe.cmd("hgetall").arg(JobKeys::data_str(&job_info.job, shared))
                // set this jobs status to running
                .cmd("hset").arg(JobKeys::data_str(&job_info.job, shared))
                    .arg("status").arg(force_serialize!(&JobStatus::Running))
                // set the worker for this job
                .cmd("hset").arg(JobKeys::data_str(&job_info.job, shared))
                    .arg("worker").arg(force_serialize!(&Some(&worker.name)))
                // add this to the correct destination status queue
                .cmd("zadd").arg(dest).arg(score).arg(&raw)
                // add this job to the running jobs stream
                .cmd("zadd")
                    .arg(StreamKeys::system_scaler(scaler, "running", shared))
                    .arg(Utc::now().timestamp())
                    .arg(force_serialize!(&serde_json::json!({"job_id": &job_info.job, "worker": &worker.name})));
            // execute this jobs query
            let job_data: JobData = pipe.atomic().query_async(conn!(shared)).await?;
            // if this jobs data is missing then delete it and try again
            if job_data.0.is_empty() {
                // log that we found a job that is missing data
                event!(Level::ERROR, msg = "Missing job data", job = job_info.job.to_string(), reaction = job_info.reaction.to_string());
                // delete our dangling job
                prune_dangling(scaler, worker, dest, job_info.job, job_info.reaction, shared).await?;
                // try to claim another job
                continue;
            }
            // convert our job claim to a raw job
            let job = RawJob::from_data(job_data.0)?;
            return Ok(Some(job));
        }
        // we didn't get a job so break
        break;
    }
    Ok(None)
}

/// Updates the reaction that contains this job
///
/// # Arguments
///
/// * `pipe` - The redis pipeline to add commands too
/// * `job` - The newly claimed job
/// * `reaction` - The reaction to update
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::update_reaction", skip_all, err(Debug))]
async fn update_reaction<'a>(pipe: &'a mut redis::Pipeline, job: &RawJob, reaction: &Reaction, shared: &Shared) -> Result<(), ApiError> {
    // get the timestamp for this reactions sla
    let timestamp = reaction.sla.timestamp();
    // if this reactions status is created then move it to the started set
    if reaction.status == ReactionStatus::Created {
        // build key to this jobs reactions data
        let react_key = ReactionKeys::data(&job.group, &job.reaction, shared);
        // set the status of this reaction to running
        let id = reaction.id.to_string();
        // update this reactions data
        pipe.cmd("hset").arg(&react_key).arg("status")
            .arg(serialize!(&ReactionStatus::Started))
            // move from created group set to started group set
            .cmd("zrem").arg(ReactionKeys::group_set(&job.group, &ReactionStatus::Created, shared))
                .arg(&id)
            .cmd("zadd").arg(ReactionKeys::group_set(&job.group, &ReactionStatus::Started, shared))
                .arg(timestamp).arg(&id)
            // move from created pipeline set to started pipeline set
            .cmd("srem").arg(ReactionKeys::status(&job.group, &job.pipeline, &ReactionStatus::Created, shared))
                .arg(&id)
            .cmd("sadd").arg(ReactionKeys::status(&job.group, &job.pipeline, &ReactionStatus::Started, shared))
                .arg(&id);
        if let Some(parent) = reaction.parent.as_ref() {
            // build key to the our sub reaction status sets
            // the old status is always Started because we
            let old_status = ReactionKeys::sub_status_set(&reaction.group, parent, &ReactionStatus::Created, shared);
            let new_status = ReactionKeys::sub_status_set(&reaction.group, parent, &ReactionStatus::Started, shared);
            // move from old sub reaction status list to new sub reaction status list
            pipe.cmd("srem").arg(old_status).arg(&id)
                .cmd("sadd").arg(new_status).arg(&id);
            }
    }
    Ok(())
}

/// Claims a job from the redis backend
///
/// # Arguments
///
/// * `user` - The user that is claiming jobs
/// * `pipeline` - The pipeline to retrieve a job from
/// * `stage` - The name of the stage to retrieve a job for within the pipeline
/// * `limit` - The max number of jobs to claim
/// * `worker` - The worker that is claiming jobs
/// * `external` - Whether this job is external or not
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::claim", skip_all, err(Debug))]
pub async fn claim(
    user: &User,
    pipeline: &Pipeline,
    stage: &str,
    limit: usize,
    worker: &WorkerName,
    shared: &Shared,
) -> Result<Vec<RawJob>, ApiError> {
    // build keys to data/queues
    let image_key = ImageKeys::data(&pipeline.group, stage, shared);
    // get if the scaler for this image
    let scaler: String = query!(cmd("hget").arg(&image_key).arg("scaler"), shared).await?;
    // if a scaler was defined then get it otherwise
    let scaler = deserialize!(&scaler);
    // get our current workers info
    let worker = super::system::get_worker(&worker.name, shared).await?;
    // build the status queues
    let src = JobKeys::status_queue(&pipeline.group, &pipeline.name, stage, &user.username, &JobStatus::Created, shared);
    let dest = JobKeys::status_queue(&pipeline.group, &pipeline.name, stage, &user.username, &JobStatus::Running, shared);
    // build alist of jobs we have claimed
    let mut claimed = Vec::with_capacity(limit);
    // claim up to the requested number of jobs
    'claim_loop: loop {
        // keep trying to claim a job until we get a valid one or there are no jobs to claim
        let (reaction, job) = loop {
            // try to claim a job
            let job = match pop_job(scaler, &worker, &src, &dest, shared).await? {
                Some(job) => job,
                // there are not jobs to claim so stop trying
                None => break 'claim_loop,
            };
            // try to get this jobs reaction data
            // get this jobs reaction info
            match reactions::get(&job.group, &job.reaction, shared).await {
                Ok(reaction) => break (reaction, job),
                // if this is a 404 then remove this job from queues and try again
                Err(error) => {
                    if error.code == StatusCode::NOT_FOUND {
                    // log that we found a job that is missing data
                    event!(Level::ERROR, msg = "Missing reaction data", job = job.id.to_string());
                        // prune this dangling job
                        prune_dangling(scaler, &worker, &dest, job.id,  job.reaction, shared).await?;
                        // try to claim another job
                        continue;
                    }
                }
            }
        };
        // update all claimed job logs
        let update_cast = StatusUpdate::new(StatusRequest::claim_job(&job, &worker.name), None);
        // update this workers current job
        system::update_worker_job(&worker, &job.reaction, &job.id, shared).await?;
        // build the redis pipeline and execute it
        let mut pipe = redis::pipe();
        // update this jobs reaction data
        update_reaction(&mut pipe, &job, &reaction, shared).await?;
            // add the status updates to our redis pipeline
        let _: () = logs::build(&mut pipe, &[update_cast], shared)?
            .atomic()
            .query_async(conn!(shared)).await?;
        // log the job that we claimed
        event!(Level::INFO, job = job.id.to_string());
        // add our claimed job
        claimed.push(job);
        // if we have claimed enough jobs then stop looping
        if claimed.len() == limit {
            break;
        }
    }
    Ok(claimed)
}

/// Updates the args for a job
///
/// # Arguments
///
/// * `job` - The job to update the args for
/// * `shared` - Shared Thorium objects
pub async fn set_args(job: &RawJob, shared: &Shared) -> Result<JobHandleStatus, ApiError> {
    // build key to this jobs data
    let key = JobKeys::data(&job.id, shared);
    // set the updated args for this job
    let _: () = query!(cmd("hset").arg(key).arg("args").arg(&job.args), shared).await?;
    Ok(JobHandleStatus::Checkpointed)
}

/// Sets a jobs status to be sleeping
///
/// This is used to let generator jobs return to Thorium and later be respawned. They must still be
/// completed by the agent to be slept (or manually if running outside of the agent).
///
/// # Arguments
///
/// * `job` - The job to sleep
/// * `checkpoint` - The checkpoint to set
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn sleep(
    job: RawJob,
    checkpoint: Checkpoint,
    shared: &Shared,
) -> Result<JobHandleStatus, ApiError> {
    // error on already completed jobs
    if job.status != JobStatus::Running {
        return conflict!(format!("job {} must be running to sleep", &job.id));
    }
    // build key to this jobs data
    let key = JobKeys::data(&job.id, shared);
    // update this jobs status to be Sleeping
    let mut pipe = redis::pipe();
    pipe.cmd("hset").arg(&key).arg("status").arg(serialize!(&JobStatus::Sleeping));
    // inject in this jobs new checkpoint arg
    let mut args: GenericJobArgs = deserialize!(&job.args);
    args.kwargs.insert("--checkpoint".to_owned(), vec!(checkpoint.data));
    pipe.cmd("hset").arg(&key).arg("args").arg(serialize!(&args));
    // execute this redis pipeline
    let _: () = pipe.query_async(conn!(shared)).await?;
    // current stage not yet complete wait
    Ok(JobHandleStatus::Sleeping)
}

/// Proceeds with a running job
///
/// This marks a running job as complete and will continue a reaction if the current stage of that reaction
/// is complete. This will also update the images average runtime. For sleeping images it will
/// simply remove it from the running jobs and deadline stream
///
/// # Arguments
///
/// * `job` - The job to proceed with
/// * `runtime` - The time it took to execute this job
/// * `logs` - Any logs to save for this job
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::proceed", skip_all, err(Debug))]
pub async fn proceed(
    job: RawJob,
    runtime: u64,
    logs: StageLogsAdd,
    shared: &Shared,
) -> Result<JobHandleStatus, ApiError> {
    // error on jobs that are not running or sleeping and set the correct destination status
    let status = match job.status {
        JobStatus::Running => JobStatus::Completed,
        JobStatus::Sleeping => JobStatus::Sleeping,
        _ => return {
            // log that our job had an incorrect status
            event!(
                Level::ERROR,
                msg = "Invalid Status",
                job = job.id.to_string(),
                status = job.status.to_string(),
            );
            conflict!(format!("job {} must be running or sleeping to proceed", &job.id))
        },
    };
    // cast to stream object
    let stream_obj = StreamObj::from(&job);
    // build key to the time to complete queue for this image
    let ttc_key = ImageKeys::ttc_queue(&job.group, &job.stage, shared);
    // build key to reaction data
    let reaction_data = ReactionKeys::data(&job.group, &job.reaction, shared);
    // build the status queues keys for non external jobs
    let src = JobKeys::status_queue(&job.group, &job.pipeline, &job.stage, &job.creator, &JobStatus::Running, shared);
    let dest = JobKeys::status_queue(&job.group, &job.pipeline, &job.stage, &job.creator, &status, shared);
    // cast our job claim data
    let job_info = serialize!(&JobReactionIds::new(job.id, job.reaction));
    // start building the redis pipeline for proceeding with this job
    let mut pipe = redis::pipe();
    // add running job specific commands if our status is running
    if job.status == JobStatus::Running {
        // increment progress for this reactions current stage
        pipe.cmd("hincrby").arg(&reaction_data).arg("current_stage_progress").arg(1)
            // get this reactions current stage length
            .cmd("hget").arg(&reaction_data).arg("current_stage_length")
            // update job status to completed
            .cmd("hset").arg(JobKeys::data(&job.id, shared)).arg("status")
                .arg(serialize!(&JobStatus::Completed))
            // push this jobs time to complete into the time to complete queue
            .cmd("lpush").arg(&ttc_key).arg(runtime)
            // ensure that our time to completion list does not grow past 10k times
            .cmd("ltrim").arg(ttc_key).arg(0).arg(10_000)
            // move this job to the correct status queues
            .cmd("zrem").arg(src).arg(&job_info)
            .cmd("zadd").arg(dest).arg(job.deadline.timestamp()).arg(&job_info);
            // add status log updates
            let update_cast = StatusUpdate::new(StatusRequest::from_job(&job, JobActions::Completed), None);
            logs::build(&mut pipe, &[update_cast], shared)?;
            // if this job is a generator then also remove it from the generator set
            if job.generator {
                // build key to generator list
                let gens = ReactionKeys::generators(&job.group, &job.reaction, shared);
                pipe.cmd("srem").arg(gens).arg(&job.id.to_string());
            }
    }
    // inject commands shared between sleeping/running jobs
    // remove job from deadlines stream
    pipe.cmd("zrem").arg(StreamKeys::system_scaler(job.scaler, "deadlines", shared))
            .arg(stream_obj.data)
        // remove job from running stream
        .cmd("zrem").arg(StreamKeys::system_scaler(job.scaler, "running", shared))
            .arg(force_serialize!(&serde_json::json!({"job_id": job.id, "worker": job.worker})));
    // save this jobs logs to the backend
    reactions::add_stage_logs(&job.reaction, &job.stage, logs, shared).await?;
    // execute redis pipeline
    // use the correct response
    let should_proceed = if job.status == JobStatus::Running {
        // execute our query and get the response based on if the job is a generator or not
        let progress = if job.generator {
            // execute the query with our job generator srem
            let full: (u64, u64, u64, u64, bool, u64, u64, u64, u64, u64, u64) = pipe.atomic().query_async(conn!(shared)).await?;
            // downselect to just the first two values
            (full.0, full.1)
        } else {
            // execute the query without the job generator srem
            let full: (u64, u64, u64, u64, bool, u64, u64, u64, u64, u64) = pipe.atomic().query_async(conn!(shared)).await?;
            // downselect to just the first two values
            (full.0, full.1)
        };
        // check if we should proceed or not
        progress.0 >= progress.1
    } else {
        // execute our query
        let _: () = pipe.atomic().query_async(conn!(shared)).await?;
        // check if we should proceed or not
        job.status == JobStatus::Sleeping
    };
    // check if we have completed all parts of the current stage or if this is a sleeping job
    //if job.status == JobStatus::Sleeping || progress[0] >= progress[1] {
    if should_proceed {
        // get reaction data
        let reaction = reactions::get(&job.group, &job.reaction, shared).await?;
        // proceed with this reaction
        reactions::proceed(reaction, shared).await
    } else {
        // current stage not yet complete wait
        Ok(JobHandleStatus::Waiting)
    }
}

/// ApiErrors out a job
///
/// This updates the jobs status to error and will fail out the rest of the pipeline.
///
/// # Arguments
///
/// * `job` - The job to mark as errored
/// * `logs` - Any logs to save for this job
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::error", skip_all, err(Debug))]
pub async fn error<'a>(
    job: RawJob,
    logs: StageLogsAdd,
    shared: &Shared,
) -> Result<JobHandleStatus, ApiError> {
    // error on non running jobs
    if job.status != JobStatus::Running {
            // log that our job had an incorrect status
            event!(
                Level::ERROR,
                msg = "Invalid Status",
                job = job.id.to_string(),
                status = job.status.to_string(),
            );
        return conflict!(format!("job {} must be running to error", &job.id));
    }
    // build the status queues keys for non external jobs
    let src = JobKeys::status_queue(&job.group, &job.pipeline, &job.stage, &job.creator, &JobStatus::Running, shared);
    let dest = JobKeys::status_queue(&job.group, &job.pipeline, &job.stage, &job.creator, &JobStatus::Failed, shared);
    // cast to stream object
    let stream_obj = StreamObj::from(&job);
    // cast our job claim data
    let job_claim = serialize!(&JobReactionIds::new(job.id, job.reaction));
    // start building the redis pipeline for erroring out this job
    let mut pipe = redis::pipe();
    let _: () = pipe
        // remove from deadlines stream
        .cmd("zrem").arg(StreamKeys::system_scaler(job.scaler, "deadlines", shared))
            .arg(stream_obj.timestamp).arg(stream_obj.data)
        // remove from running jobs stream
        .cmd("zrem").arg(StreamKeys::system_scaler(job.scaler, "running", shared))
            .arg(force_serialize!(&serde_json::json!({"job_id": job.id, "worker": job.worker})))
        // update status to error
        .cmd("hset").arg(JobKeys::data(&job.id, shared)).arg("status")
                .arg(serialize!(&JobStatus::Failed))
        // move to the correct status queue
        .cmd("zrem").arg(src).arg(&job_claim)
        .cmd("zadd").arg(dest).arg(job.deadline.timestamp()).arg(&job_claim)
        .query_async(conn!(shared))
        .await?;

    // if this job is a generator then remove it from this reactions generators
    if job.generator {
        // build key to this reactions generator set
        let gen_key = ReactionKeys::generators(&job.group, &job.reaction, shared);
        pipe.cmd("srem").arg(gen_key).arg(&job.id.to_string());
    }
    // save this jobs logs to scylla
    reactions::add_stage_logs(&job.reaction, &job.stage, logs, shared).await?;
    // create and save status log
    let update_cast = StatusUpdate::new(StatusRequest::from_job(&job, JobActions::Errored), None);
    logs::build(&mut pipe, &[update_cast], shared)?;
    // execute redis pipeline
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    // error out reaction as well
    let reaction = reactions::get(&job.group, &job.reaction, shared).await?;
    reactions::fail(reaction, shared).await?;
    Ok(JobHandleStatus::Errored)
}

/// Find entries in a stream with some uuid
///
/// # Arguments
///
/// * `stream` - The stream to search
/// * `filters` - The different uuids to search for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::find_in_running", skip(uuids, shared), fields(uuids = uuids.len()), err(Debug))]
pub async fn find_in_running(stream: &str, uuids: &HashSet<&Uuid>, shared: &Shared) -> Result<HashMap<Uuid, String>, ApiError> {
    // track the uuids that we have found
    let mut found = HashMap::with_capacity(uuids.len());
    // track our current position in the deadline stream
    let mut pos = 0;
    // build the score settings to use when reading
    let max = i64::MAX;
    // read through this stream until we have found all uuids or this stream is exhausted
    'outer: loop {
        // start a span for this iteration of searching
        let span = span!(Level::INFO, "db::jobs::find_in_running::iteration", pos);
        // read the next 10k items in the queue
        let chunk = streams::read_no_scores_by_key(stream, 0, max, pos, 10_000, shared).await?;
        // get how many items are in this stream chunk
        let chunk_len = chunk.len();
        // step over each row in this chunk and parse them
        for entry in chunk {
            // try to parse this chunk
            let running: RunningJob = deserialize!(&entry);
            // check if this is one of the uuids we were trying to find
            if uuids.contains(&running.job_id) {
                // we found this jobs running entry so add it to our found map
                found.insert(running.job_id, entry);
                // check if we found all entries
                if found.len() == uuids.len() {
                    break 'outer;
                }
            }
        }
        // if less than 10k items are returned then this stream has been exhausted
        if chunk_len < 10_000 {
            break 'outer;
        }
        // increment pos
        pos += chunk_len as u64;
        // drop our span to exit it
        drop(span);
    }
    Ok(found)
}

/// Resets jobs to be rerun in bulk
///
/// # Arguments
///
/// * `resets` - The jobs to reset
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::bulk_reset", skip_all, fields(resets = resets.jobs.len()), err(Debug))]
pub async fn bulk_reset(
    resets: JobResets,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build list of jobs based on reset list
    let job_list = JobList::new(None, resets.jobs.clone());
    // get data on all jobs to reset
    let mut jobs = RawJob::list_details(job_list, shared).await?;
    // build a set of job details
    let found: HashSet<&Uuid> = jobs.details.iter().map(|job| &job.id).collect();
    // get a list of jobs whose data cannot be retrieved
    let missing: HashSet<&Uuid> = resets.jobs.iter().filter(|id| !found.contains(id)).collect();
    // drop any details that have a terminal status
    jobs.details.retain(|job| job.status != JobStatus::Completed && job.status != JobStatus::Failed);
    // build a redis pipeline to reset jobs
    let mut pipe = redis::pipe();
    // crawl over jobs and build their reset commands
    for job in &jobs.details {
        // cast the id for this job to a string
        let job_id = job.id.to_string();
        // log that we are resetting this job
        event!(Level::INFO, job=&job_id, old_status=job.status.to_string());
        // build the status queues keys for non external jobs
        let src = JobKeys::status_queue(&job.group, &job.pipeline, &job.stage, &job.creator, &job.status, shared);
        let dest = JobKeys::status_queue(&job.group, &job.pipeline, &job.stage, &job.creator, &JobStatus::Created, shared);
        // cast our job claim data
        let job_claim = serialize!(&JobReactionIds::new(job.id, job.reaction));
        // add the command shared by internal and external jobs
        // update jobs status
        pipe.cmd("hset").arg(JobKeys::data(&job.id, shared))
                .arg("status").arg(force_serialize!(&JobStatus::Created))
            // add to created jobs set
            .cmd("zadd").arg(status_queue!(job, &JobStatus::Created, shared))
                .arg(job.deadline.timestamp()).arg(&job_claim)
            // move these jobs to the correct status queue
            .cmd("zrem").arg(src).arg(&job_claim)
            .cmd("zadd").arg(dest).arg(job.deadline.timestamp()).arg(&job_claim)
            // remove from running stream
            .cmd("zrem").arg(StreamKeys::system_scaler(job.scaler, "running", shared))
                .arg(force_serialize!(&serde_json::json!({"job_id": job.id, "worker": job.worker})))
                // add to deadlines queue if its not already added
                .cmd("zadd").arg(StreamKeys::system_scaler(job.scaler, "deadlines", shared)).arg(job.deadline.timestamp())
                    .arg(StreamObj::from(job).data);
    }
    // if we missing jobs then try to get there data if possible
    if !missing.is_empty() {
        // build the key to the running job stream
        let stream_key = StreamKeys::stream("system", resets.scaler.as_str(), "running", shared);
        // check to see if these jobs still have a worker set to avoid an expensive scan
        let workers = get_workers(&missing, shared).await?;
        // remove the jobs whose workers we could find from the running stream
        for (job, worker) in workers.iter() {
            // build the object to remove from the running stream
            let entry = format!("{{\"job_id\":\"{job}\",\"worker\":\"{worker}\"}}");
            // add the command to remove this job from the running stream
            pipe.cmd("zrem").arg(&stream_key).arg(entry);
        }
        // build a hashset of the jobs would couldn't find workers for
        let no_workers: HashSet<&Uuid> = missing.iter()
            .filter(|worker| !workers.contains_key(**worker))
            .map(|worker| *worker)
            .collect();
        // get the orphaned entries in the running job stream if there is any
        if !no_workers.is_empty() {
            // find the entries for these jobs in the running stream
            let entry_map = find_in_running(&stream_key, &no_workers, shared).await?;
            // remove these missing entries from the running jobs stream
            for (_, entry) in &entry_map {
                // add the command to remove this job from the running stream
                pipe.cmd("zrem").arg(&stream_key).arg(entry);
            }
        }
        // delete the orphaned jobs data if it exists
        for orphan in &missing {
            // log that we found a job that no longer has data
            event!(Level::ERROR, msg = "Deleting Job with no data", job = orphan.to_string());
            //. build the key to this jobs data
            let key = JobKeys::data(*orphan, shared);
            // add the command to delete this jobs data
            pipe.cmd("del").arg(key);
        }
    }
    // execute built redis pipeline
    let _:() = pipe.query_async(conn!(shared)).await?;
    Ok(())
}

/// Gets details on a list of job IDs
///
/// # Arguments
///
/// * `jobs` - The list of jobs to get details for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::jobs::list_details", skip_all, err(Debug))]
pub async fn list_details(
    jobs: JobList,
    shared: &Shared,
) -> Result<JobDetailsList, ApiError> {
    // get job data
    let raw: Vec<HashMap<String, String>> = jobs.names.iter()
        .fold(redis::pipe().atomic(), |pipe, id|
            pipe.cmd("hgetall").arg(&JobKeys::data(id, shared)))
        .query_async(conn!(shared))
        .await?;
    // build a vec to store our raw job details in
    let mut details = Vec::with_capacity(raw.len());
    // cast each job id in this list
    for raw_data in raw {
        // try to cast this raw data
        match RawJob::from_data(raw_data) {
            Ok(job) => details.push(job),
            // we failed to get this jobs details
            Err(error) => {
                // log that we failed to get this jobs details and ignore it
                event!(Level::ERROR, msg = error.msg);
            }
        }
    }
    // cast to group details list
    let details_list = JobDetailsList::new(jobs.cursor, details);
    Ok(details_list)
}
