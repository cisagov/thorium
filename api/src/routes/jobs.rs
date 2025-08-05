use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::{get, patch, post};
use axum::Router;
use tracing::instrument;
use utoipa::OpenApi;
use uuid::Uuid;

use super::OpenApiSecurity;

use crate::models::{
    Checkpoint, CommitishKinds, Deadline, GenericJob, GenericJobArgs, GenericJobOpts,
    HandleJobResponse, ImageScaler, JobHandleStatus, JobListOpts, JobResetRequestor, JobResets,
    JobStatus, Pipeline, RawJob, RepoDependency, RunningJob, StageLogLine, StageLogsAdd,
    SystemComponents, User, WorkerName,
};
use crate::utils::{ApiError, AppState};

/// Claims a job from a specific pipeline and stage for a worker
///
/// # Arguments
///
/// * `user` - The user that is claiming a job
/// * `group` - The group the target pipeline is in
/// * `pipeline` - The name of the pipeline to claim from
/// * `stage` - The specific stage to claim from
/// * `worker` - The name of the worker that is claiming this job
/// * `limit` - The max number of jobs to claim
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = "/api/jobs/claim/:group/:pipeline/:stage/:cluster/:node/:worker/:limit",
    params(
        ("group" = String, Path, description = "The group the target pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline to claim from"),
        ("stage" = String, Path, description = "The specific stage to claim from"),
        ("cluster" = String, Path, description = "The name of the cluster of the worker that is claiming this job"),
        ("node" = String, Path, description = "The name of the node of the worker that is claiming this job"),
        ("worker" = String, Path, description = "The name of the worker that is claiming this job"),
        ("limit" = usize, Path, description = "The max number of jobs to claim"),
    ),
    responses(
        (status = 200, description = "Slept the specified job", body = Vec<GenericJob>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::claim", skip_all, err(Debug))]
async fn claim(
    user: User,
    Path((group, pipeline, stage, cluster, node, worker, limit)): Path<(
        String,
        String,
        String,
        String,
        String,
        String,
        usize,
    )>,
    State(state): State<AppState>,
) -> Result<Json<Vec<GenericJob>>, ApiError> {
    // get pipeline
    let (group, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // build the primary keys to the worker that is claiming this job
    let worker = WorkerName::new(cluster, node, worker);
    // claim jobs if any exist
    let claims = GenericJob::claim(
        &user,
        &group,
        &pipeline,
        &stage,
        limit,
        &worker,
        &state.shared,
    )
    .await?;
    // return claimed jobs
    Ok(Json(claims))
}

/// Proceed with this job a worker has just completed
///
/// # Arguments
///
/// * `user` - The user that is proceeding with this job
/// * `id` - The uuid of the job that was completed
/// * `runtime` - The amount of time in seconds this job took to complete
/// * `state` - Shared Thorium objects
/// * `logs` - Any logs to append to this jobs stage logs
#[utoipa::path(
    post,
    path = "/api/jobs/handle/:id/proceed/:runtime",
    params(
        ("id" = Uuid, Path, description = "The uuid of the job that was completed"),
        ("runtime" = u64, Path, description = "The amount of time in seconds this job took to complete"),
        ("logs" = StageLogsAdd, description = "Any logs to append to this jobs stage logs"),
    ),
    responses(
        (status = 202, description = "Proceeding with specified job", body = HandleJobResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::proceed", skip_all, err(Debug))]
async fn proceed(
    user: User,
    Path((id, runtime)): Path<(Uuid, u64)>,
    State(state): State<AppState>,
    Json(logs): Json<StageLogsAdd>,
) -> Result<Response, ApiError> {
    // get job object
    let (group, job) = RawJob::get(&user, &id, &state.shared).await?;
    // proceed with job
    let status = job
        .proceed(&user, &group, runtime, logs, &state.shared)
        .await?;
    // build response
    let response = Json(HandleJobResponse { status });
    Ok((StatusCode::ACCEPTED, response).into_response())
}

/// ApiError out this job that has just failed
///
/// # Arguments
///
/// * `user` - The user that is erroring out this job
/// * `id` - The uuid of the job that was failed
/// * `state` - Shared Thorium objects
/// * `logs` - Any logs to append to this jobs stage logs
#[utoipa::path(
    post,
    path = "/api/jobs/handle/:id/error",
    params(
        ("id" = Uuid, Path, description = "The uuid of the job that was completed"),
        ("logs" = StageLogsAdd, description = "Any logs to append to this jobs stage logs"),
    ),
    responses(
        (status = 202, description = "Handle specified failed job", body = HandleJobResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::error", skip_all, fields(job = id.to_string()), err(Debug))]
async fn error(
    user: User,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(logs): Json<StageLogsAdd>,
) -> Result<Response, ApiError> {
    // get job object
    let (group, job) = RawJob::get(&user, &id, &state.shared).await?;
    // error out this job
    let status = job.error(&user, &group, logs, &state.shared).await?;
    // build response
    let response = Json(HandleJobResponse { status });
    Ok((StatusCode::ACCEPTED, response).into_response())
}

/// Sleep this generator job
///
/// Only generator jobs should be slept.
///
/// # Arguments
///
/// * `user` - The user that is sleeping this job
/// * `id` - The uuid of the job that is going to sleep
/// * `state` - Shared Thorium objects
/// * `checkpoint` - A checkpoint object to use when waking this job
#[utoipa::path(
    post,
    path = "/api/jobs/handle/:id/sleep",
    params(
        ("id" = Uuid, Path, description = "The uuid of the job that was completed"),
        ("checkpoint" = Checkpoint, description = "A checkpoint object to use when waking this job"),
    ),
    responses(
        (status = 202, description = "Slept the specified job", body = HandleJobResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::sleep", skip_all, fields(job = id.to_string()), err(Debug))]
async fn sleep(
    user: User,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(checkpoint): Json<Checkpoint>,
) -> Result<Response, ApiError> {
    // get job object
    let (group, job) = RawJob::get(&user, &id, &state.shared).await?;
    // sleep this job and set an optional checkpoint
    let status = job.sleep(&user, &group, checkpoint, &state.shared).await?;
    // build response
    let response = Json(HandleJobResponse { status });
    Ok((StatusCode::ACCEPTED, response).into_response())
}

/// Checkpoint generator job
///
/// Only generator jobs should use checkpoints.
///
/// # Arguments
///
/// * `user` - The user that is checkpointing this job
/// * `id` - The uuid of the job that is being checkpointed
/// * `state` - Shared Thorium objects
/// * `checkpoint` - A checkpoint object to use when waking this job
#[utoipa::path(
    post,
    path = "/api/jobs/handle/:id/checkpoint",
    params(
        ("id" = Uuid, Path, description = "The uuid of the job that was completed"),
        ("checkpoint" = Checkpoint, description = "A checkpoint object to use when waking this job"),
    ),
    responses(
        (status = 202, description = "Slept the specified job", body = HandleJobResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::checkpoint", skip_all, fields(job = id.to_string()), err(Debug))]
async fn checkpoint(
    user: User,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(checkpoint): Json<Checkpoint>,
) -> Result<Response, ApiError> {
    // get job object
    let (group, job) = RawJob::get(&user, &id, &state.shared).await?;
    // checkpoint this job
    let status = job
        .checkpoint(&user, &group, checkpoint, &state.shared)
        .await?;
    // build response
    let response = Json(HandleJobResponse { status });
    Ok((StatusCode::ACCEPTED, response).into_response())
}

/// Resets jobs in bulk
///
/// # Arguments
///
/// * `user` - The user that is resetting these jobs
/// * `state` - Shared Thorium objects
/// * `resets` - The jobs to reset
#[utoipa::path(
    post,
    path = "/api/jobs/bulk/reset",
    params(
        ("resets" = JobResets, description = "The jobs to reset"),
    ),
    responses(
        // TODO_UTOIPA: this func doesn't seem to return *anything*, a sanity check here would be good
        (status = 200, description = "Reset the specified job(s)"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::bulk_reset", skip_all, err(Debug))]
async fn bulk_reset(
    user: User,
    State(state): State<AppState>,
    Json(resets): Json<JobResets>,
) -> Result<StatusCode, ApiError> {
    // proceed with job
    RawJob::bulk_reset(&user, resets, &state.shared).await?;
    // return a 200
    Ok(StatusCode::OK)
}

/// List jobs in the deadline stream by when they must be started by.
///
/// # Arguments
///
/// * `user` - The user that is listing jobs
/// * `scaler` - The scaler to get deadlines for
/// * `start` - The starting timestamp in epoch time to list from
/// * `end` - The ending timestamp in epoch time to list to
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/jobs/deadlines/:scaler/:start/:end",
    params(
        ("scalar" = ImageScaler, Path, description = "The scaler to get deadlines for"),
        ("start" = i64, Path, description = "The starting timestamp in epoch time to list from"),
        ("end" = i64, Path, description = "The ending timestamp in epoch time to list to"),
        ("params" = JobListOpts, Query, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "Slept the specified job", body = Vec<Deadline>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::read_deadlines", skip_all, err(Debug))]
async fn read_deadlines(
    user: User,
    Path((scaler, start, end)): Path<(ImageScaler, i64, i64)>,
    Query(params): Query<JobListOpts>,
    State(state): State<AppState>,
) -> Result<Json<Vec<Deadline>>, ApiError> {
    // read from the deadline stream
    let deadlines = Deadline::read(
        &user,
        scaler,
        start,
        end,
        params.skip,
        params.limit,
        &state.shared,
    )
    .await?;
    Ok(Json(deadlines))
}

/// List jobs in the running stream by when they were started
///
/// # Arguments
///
/// * `user` - The user that is listing jobs
/// * `scaler` - The scaler to get running jobs for
/// * `start` - The starting timestamp in epoch time to list from
/// * `end` - The ending timestamp in epoch time to list to
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/jobs/bulk/running/:scaler/:start/:end",
    params(
        ("scalar" = ImageScaler, Path, description = "The scaler to get running jobs for"),
        ("start" = i64, Path, description = "The starting timestamp in epoch time to list from"),
        ("end" = i64, Path, description = "The ending timestamp in epoch time to list to"),
        ("params" = JobListOpts, Query, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "List of running jobs", body = Vec<RunningJob>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::jobs::bulk_running", skip_all, err(Debug))]
async fn bulk_running(
    user: User,
    Path((scaler, start, end)): Path<(ImageScaler, i64, i64)>,
    Query(params): Query<JobListOpts>,
    State(state): State<AppState>,
) -> Result<Json<Vec<RunningJob>>, ApiError> {
    // read from the deadline stream
    let running = RawJob::running(
        &user,
        scaler,
        start,
        end,
        params.skip,
        params.limit,
        &state.shared,
    )
    .await?;
    Ok(Json(running))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(claim, proceed, error, sleep, checkpoint, bulk_reset, read_deadlines, bulk_running),
    components(schemas(Checkpoint, CommitishKinds, Deadline, GenericJob, GenericJobArgs, GenericJobOpts, HandleJobResponse, ImageScaler, JobHandleStatus, JobListOpts, JobResetRequestor, JobResets, JobHandleStatus, JobStatus, RepoDependency, RunningJob, StageLogLine, StageLogsAdd, SystemComponents)),
    modifiers(&OpenApiSecurity),
)]
pub struct JobApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(JobApiDocs::openapi())
}

/// Add the streams routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route(
            "/api/jobs/claim/{group}/{pipeline}/{stage}/{cluster}/{node}/{worker}/{limit}",
            patch(claim),
        )
        .route("/api/jobs/handle/{id}/proceed/{runtime}", post(proceed))
        .route("/api/jobs/handle/{id}/error", post(error))
        .route("/api/jobs/handle/{id}/sleep", post(sleep))
        .route("/api/jobs/handle/{id}/checkpoint", post(checkpoint))
        .route("/api/jobs/bulk/reset", post(bulk_reset))
        .route(
            "/api/jobs/deadlines/{scaler}/{start}/{end}",
            get(read_deadlines),
        )
        .route(
            "/api/jobs/bulk/running/{scaler}/{start}/{end}",
            get(bulk_running),
        )
}
