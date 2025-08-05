use std::collections::HashMap;

use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use axum_extra::body::AsyncReadBody;
use tracing::{instrument, span, Level, Span};
use utoipa::OpenApi;
use uuid::Uuid;

use super::OpenApiSecurity;
use crate::bad;
use crate::models::{
    Actions, BulkReactionResponse, CommitishKinds, Group, HandleReactionResponse, ImageScaler,
    JobResetRequestor, Pipeline, Reaction, ReactionDetailsList, ReactionIdResponse, ReactionList,
    ReactionListParams, ReactionRequest, ReactionStatus, ReactionUpdate, RepoDependency,
    RepoDependencyRequest, StageLogLine, StageLogs, StageLogsAdd, StatusUpdate, SystemComponents,
    User,
};
use crate::utils::{ApiError, AppState};

/// Creates a new reaction
///
/// # Arguments
///
/// * `user` - The user that is creating this reaction
/// * `state` - Shared Thorium objects
/// * `req` - The reaction request used to create a reaction
#[utoipa::path(
    post,
    path = "/api/reactions/",
    params(
        ("req" = ReactionRequest, description = "The reaction request used to create a reaction"),
    ),
    responses(
        (status = 200, description = "Pipeline created", body = ReactionIdResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::create", skip_all, err(Debug))]
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(req): Json<ReactionRequest>,
) -> Result<Json<ReactionIdResponse>, ApiError> {
    // get pipeline
    let (group, pipeline) = Pipeline::get(&user, &req.group, &req.pipeline, &state.shared).await?;
    // refrain from running the reaction if the pipeline has a ban
    if !pipeline.bans.is_empty() {
        return bad!(format!(
            "The reaction cannot be created because pipeline '{}' in group '{}' has one or more bans! \
            See the pipeline's notifications for more details.",
            req.pipeline,
            req.group,
        ));
    }
    // build reaction object and inject it into the backend
    let reaction = Reaction::create(&user, &group, &pipeline, req, &state.shared).await?;
    Ok(Json(ReactionIdResponse { id: reaction.id }))
}

/// Creates new reactions in bulk
///
/// # Arguments
///
/// * `user` - The user that is creating these reactions
/// * `state` - Shared Thorium objects
/// * `req` - The reactions to create in bulk
#[utoipa::path(
    post,
    path = "/api/reactions/bulk/",
    params(
        ("reqs" = Vec<ReactionRequest>, description = "The reactions to create in bulk"),
    ),
    responses(
        (status = 200, description = "Pipeline created", body = BulkReactionResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::create_bulk", skip_all, err(Debug))]
async fn create_bulk(
    user: User,
    State(state): State<AppState>,
    Json(reqs): Json<Vec<ReactionRequest>>,
) -> Result<Json<BulkReactionResponse>, ApiError> {
    // create reactions in bulk
    let response = Reaction::create_bulk(&user, reqs, &state.shared).await?;
    Ok(Json(response))
}

/// Creates new reactions in bulk
///
/// # Arguments
///
/// * `user` - The user that is creating these reactions
/// * `state` - Shared Thorium objects
/// * `req` - The reactions to create in bulk
#[instrument(name = "routes::reactions::create_bulk_by_user", skip_all, err(Debug))]
async fn create_bulk_by_user(
    user: User,
    State(state): State<AppState>,
    Json(reqs): Json<HashMap<String, Vec<ReactionRequest>>>,
) -> Result<Json<HashMap<String, BulkReactionResponse>>, ApiError> {
    // create reactions in bulk
    let response = Reaction::create_bulk_by_user(&user, reqs, &state.shared).await?;
    Ok(Json(response))
}

/// Gets info about a specific reaction by id
///
/// # Arguments
///
/// * `user` - The user that is getting info about a reaction
/// * `group` - The group this reaction is in
/// * `id` - The uuid of the reaction to get info for
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/:group/:id",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("id" = Uuid, Path, description = "The uuid of the reaction to get info for"),
    ),
    responses(
        (status = 200, description = "Returned reaction", body = Reaction),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn get_reaction(
    user: User,
    Path((group, id)): Path<(String, Uuid)>,
    State(state): State<AppState>,
) -> Result<Json<Reaction>, ApiError> {
    // start our get reaction span
    let span = span!(Level::INFO, "Get Reaction Route");
    // get reaction from backend
    let (_, reaction) = Reaction::get(&user, &group, &id, &state.shared, &span).await?;
    Ok(Json(reaction))
}

/// Handle a command for a reaction
///
/// This can be used to proceed or fail a reaction
///
/// # Arguments
///
/// * `user` - The user that is giving a command for this reaction
/// * `group` - The group this reaction is in
/// * `id` - The uuid of the reaction to handle a command for
/// * `cmd` - The command to execute (proceed || fail)
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/reactions/handle/:group/:id/:cmd",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("id" = Uuid, Path, description = "The uuid of the reaction to handle a command for"),
        ("cmd" = String, Path, description = "The command to execute (proceed || fail)")
    ),
    responses(
        (status = 202, description = "Returned reaction", body = HandleReactionResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::handle", skip_all, err(Debug))]
async fn handle(
    user: User,
    Path((group, id, cmd)): Path<(String, Uuid, String)>,
    State(state): State<AppState>,
) -> Result<Response, ApiError> {
    // get our current span
    let span = Span::current();
    // get reaction object
    let (group, reaction) = Reaction::get(&user, &group, &id, &state.shared, &span).await?;
    // call correct handler
    let status = match cmd.as_ref() {
        "proceed" => reaction.proceed(&user, &group, &state.shared).await?,
        "fail" => reaction.fail(&user, &group, &state.shared).await?,
        _ => return bad!(format!("{} is not a known handler", cmd)),
    };
    // return response
    let response = Json(HandleReactionResponse { status });
    Ok((StatusCode::ACCEPTED, response).into_response())
}

/// Get the status logs for a reaction
///
/// These are not stdout/stderr logs but are instead logs of high level status changes.
///
/// # Arguments
///
/// * `user` - The user that is getting status logs
/// * `group` - The group this reaction is in
/// * `id` - The uuid of the reaction to get status logs for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/logs/:group/:id",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("id" = Uuid, Path, description = "The uuid of the reaction to get status logs for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Returned reaction", body = Vec<StatusUpdate>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn logs(
    user: User,
    Path((group, id)): Path<(String, Uuid)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<Vec<StatusUpdate>>, ApiError> {
    // start our handle reaction command span
    let span = span!(Level::INFO, "Get Reaction Logs Route");
    // get reaction object
    let (_, reaction) = Reaction::get(&user, &group, &id, &state.shared, &span).await?;
    // get the reaction logs
    let logs = reaction
        .logs(params.cursor, params.limit, &state.shared, &span)
        .await?;
    Ok(Json(logs))
}

/// Adds new stdout/stderr logs for a specific stage
///
/// # Arguments
///
/// * `user` - The user that is adding new stage logs
/// * `group` - The group this reaction is in
/// * `id` - The uuid of the reaction to add stage logs
/// * `stage` - The stage these logs are for
/// * `state` - Shared Thorium objects
/// * `logs` - The stdout/stderr logs to add
#[utoipa::path(
    post,
    path = "/api/reactions/logs/:group/:id/:stage",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("id" = Uuid, Path, description = "The uuid of the reaction to add stage logs"),
        ("stage" = String, Path, description = "The stage these logs are for"),
        ("logs" = StageLogsAdd, description = "The stdout/stderr logs to add")
    ),
    responses(
        (status = 204, description = "Stage logs added"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::add_stage_logs", skip_all, err(Debug))]
async fn add_stage_logs(
    user: User,
    Path((group, id, stage)): Path<(String, Uuid, String)>,
    State(state): State<AppState>,
    Json(logs): Json<StageLogsAdd>,
) -> Result<StatusCode, ApiError> {
    // get our current span
    let span = Span::current();
    // get reaction object
    let (_, reaction) = Reaction::get(&user, &group, &id, &state.shared, &span).await?;
    // append stage logs
    reaction.add_stage_logs(&stage, logs, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get the stdout/stderr logs for a specific stage in a reaction
///
/// # Arguments
///
/// * `user` - The user that is getting stage logs
/// * `group` - The group this reaction is in
/// * `id` - The uuid of the reaction to get stage logs for
/// * `stage` - The stage to get logs from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/logs/:group/:id/:stage",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("id" = Uuid, Path, description = "The uuid of the reaction to get stage logs for"),
        ("stage" = String, Path, description = "The stage to get logs from"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Logs for the requested reaction stage", body = StageLogs),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::stage_logs", skip_all, err(Debug))]
async fn stage_logs(
    user: User,
    Path((group, id, stage)): Path<(String, Uuid, String)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<StageLogs>, ApiError> {
    // get our current span
    let span = Span::current();
    // get reaction object
    let (_, reaction) = Reaction::get(&user, &group, &id, &state.shared, &span).await?;
    // get stage logs
    let logs = reaction
        .stage_logs(&stage, params.cursor, params.limit, &state.shared)
        .await?;
    Ok(Json(logs))
}

/// Lists reactions for a specific pipeline
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `pipeline` - The pipeline to list reactions from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/list/:group/:pipeline/",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("pipeline" = String, Path, description = "The pipeline to list reactions from"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reactions for this pipeline", body = ReactionList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list", skip_all, err(Debug))]
async fn list(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionList>, ApiError> {
    // get pipeline data
    let (_, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // list reactions in a group
    let names = Reaction::list(&pipeline, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(names))
}

/// Lists reactions with details for a specific pipeline
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `pipeline` - The pipeline to list reactions from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/list/:group/:pipeline/details/",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("pipeline" = String, Path, description = "The pipeline to list reactions from"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reaction details for this pipeline", body = ReactionDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list_details", skip_all, err(Debug))]
async fn list_details(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionDetailsList>, ApiError> {
    // get pipeline data
    let (_, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // list reactions in a group with details
    let details = Reaction::list(&pipeline, params.cursor, params.limit, &state.shared)
        .await?
        .details(&group, &state.shared)
        .await?;
    Ok(Json(details))
}

/// Lists reactions for a specific pipeline and status
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `pipeline` - The pipeline to list reactions from
/// * `status` - The status to list reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/status/:group/:pipeline/:status/",
    params(
        ("group" = String, Path, description = "The group to list reactions from"),
        ("pipeline" = String, Path, description = "The pipeline to list reactions from"),
        ("status" = ReactionStatus, Path, description = "The status to list reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reactions for this specific pipeline and status", body = ReactionList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list_status", skip_all, err(Debug))]
async fn list_status(
    user: User,
    Path((group, pipeline, status)): Path<(String, String, ReactionStatus)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionList>, ApiError> {
    // get the pipeline/group these reactions are in
    let (group, pipe) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // list reactions in a group with this status
    let reactions = Reaction::list_status(
        &group,
        &pipe,
        &status,
        params.cursor,
        params.limit,
        &state.shared,
    )
    .await?;
    Ok(Json(reactions))
}

/// Lists reactions with details for a specific pipeline and status
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `pipeline` - The pipeline to list reactions from
/// * `status` - The status to list reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/status/:group/:pipeline/:status/details/",
    params(
        ("group" = String, Path, description = "The group to list reactions from"),
        ("pipeline" = String, Path, description = "The pipeline to list reactions from"),
        ("status" = ReactionStatus, Path, description = "The status to list reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reaction details for this specific pipeline and status", body = ReactionDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list_status_details", skip_all, err(Debug))]
async fn list_status_details(
    user: User,
    Path((group, pipeline, status)): Path<(String, String, ReactionStatus)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionDetailsList>, ApiError> {
    // get the pipeline/group these reactions are in
    let (group, pipe) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // list reactions in a group
    let reactions = Reaction::list_status(
        &group,
        &pipe,
        &status,
        params.cursor,
        params.limit,
        &state.shared,
    )
    .await?;
    // get details on these reactions
    let details = reactions.details(&group.name, &state.shared).await?;
    Ok(Json(details))
}

/// Lists reactions with a specific tag
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `tag` - The tag to list reactions from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/tag/:group/:tag/",
    params(
        ("group" = String, Path, description = "The group to list reactions from"),
        ("tag" = String, Path, description = "The tag to list reactions from"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reactions with the specified tag", body = ReactionList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list_tag", skip_all, err(Debug))]
async fn list_tag(
    user: User,
    Path((group, tag)): Path<(String, String)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionList>, ApiError> {
    // get the group these reactions are in
    let group = Group::get(&user, &group, &state.shared).await?;
    // list reactions in a group
    let names =
        Reaction::list_tag(&group, &tag, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(names))
}

/// Lists reaction details with a specific tag
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `tag` - The tag to list reactions from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/tag/:group/:tag/details/",
    params(
        ("group" = String, Path, description = "The group to list reactions from"),
        ("tag" = String, Path, description = "The tag to list reactions from"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reaction details for the specified tag", body = ReactionDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list_tag_details", skip_all, err(Debug))]
async fn list_tag_details(
    user: User,
    Path((group, tag)): Path<(String, String)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionDetailsList>, ApiError> {
    // get the group these reactions are in
    let group = Group::get(&user, &group, &state.shared).await?;
    // list reactions with a specific tag
    let reactions =
        Reaction::list_tag(&group, &tag, params.cursor, params.limit, &state.shared).await?;
    // get details on these reactions
    let details = reactions.details(&group.name, &state.shared).await?;
    Ok(Json(details))
}

/// Lists reactions for a specific group and status
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `status` - The status to list reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/group/:group/:status/",
    params(
        ("group" = String, Path, description = "The group to list reactions from"),
        ("status" = ReactionStatus, Path, description = "The status to list reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reactions for the specified group and status", body = ReactionList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::reactions::list_group_set", skip_all, err(Debug))]
async fn list_group_set(
    user: User,
    Path((group, status)): Path<(String, ReactionStatus)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionList>, ApiError> {
    // get the group these reactions are in
    let group = Group::get(&user, &group, &state.shared).await?;
    // list reactions in a group with a set status
    let names =
        Reaction::list_group_set(&group, &status, params.cursor, params.limit, &state.shared)
            .await?;
    Ok(Json(names))
}

/// Lists reactions with details for a specific group and status
///
/// # Arguments
///
/// * `user` - The user that is listing reactions
/// * `group` - The group to list reactions from
/// * `status` - The status to list reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/group/:group/:status/details/",
    params(
        ("group" = String, Path, description = "The group to list reactions from"),
        ("status" = ReactionStatus, Path, description = "The status to list reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Reaction details for the specified group and status", body = ReactionDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::reactions::list_group_set_details",
    skip_all,
    err(Debug)
)]
async fn list_group_set_details(
    user: User,
    Path((group, status)): Path<(String, ReactionStatus)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionDetailsList>, ApiError> {
    // get the group these reactions are in
    let group = Group::get(&user, &group, &state.shared).await?;
    // list reactions in a group with a set status
    let reactions =
        Reaction::list_group_set(&group, &status, params.cursor, params.limit, &state.shared)
            .await?;
    // get details on these reactions
    let details = reactions.details(&group.name, &state.shared).await?;
    Ok(Json(details))
}

/// Lists sub reactions for a specific parent reaction
///
/// # Arguments
///
/// * `user` - The user that is listing sub reactions
/// * `group` - The group to list sub reactions from
/// * `reaction` - The parent reaction to list sub reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/sub/:group/:reaction/",
    params(
        ("group" = String, Path, description = "The group to list sub reactions from"),
        ("reaction" = Uuid, Path, description = "The parent reaction to list sub reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Sub-reactions for the specified group and parent reaction", body = ReactionList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn list_sub(
    user: User,
    Path((group, reaction)): Path<(String, Uuid)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionList>, ApiError> {
    // start our list sub reactions span
    let span = span!(Level::INFO, "List Sub Reactions Route");
    // get reaction data
    let (_, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // list sub reactions
    let names = Reaction::list_sub(&reaction, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(names))
}

/// Lists sub reactions for a specific parent reaction and status
///
/// # Arguments
///
/// * `user` - The user that is listing sub reactions
/// * `group` - The group to list sub reactions from
/// * `reaction` - The parent reaction to list sub reactions for
/// * `status` - The status to list sub reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/sub/:group/:reaction/:status/",
    params(
        ("group" = String, Path, description = "The group to list sub reactions from"),
        ("reaction" = Uuid, Path, description = "The parent reaction to list sub reactions for"),
        ("status" = ReactionStatus, Path, description = "The status to list sub reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Sub-reactions for the specified group and parent reaction and status", body = ReactionList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn list_sub_status(
    user: User,
    Path((group, reaction, status)): Path<(String, Uuid, ReactionStatus)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionList>, ApiError> {
    // start our list sub reactions by status span
    let span = span!(Level::INFO, "List Sub Reactions By Status Route");
    // get reaction data
    let (_, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // list sub reactions
    let names = Reaction::list_sub_status(
        &reaction,
        &status,
        params.cursor,
        params.limit,
        &state.shared,
    )
    .await?;
    Ok(Json(names))
}

/// Lists sub reactions with details for a specific parent reaction
///
/// # Arguments
///
/// * `user` - The user that is listing sub reactions
/// * `group` - The group to list sub reactions from
/// * `reaction` - The parent reaction to list sub reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/sub/:group/:reaction/details/",
    params(
        ("group" = String, Path, description = "The group to list sub reactions from"),
        ("reaction" = Uuid, Path, description = "The parent reaction to list sub reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Sub-reaction details for the specified group and parent reaction", body = ReactionDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn list_sub_details(
    user: User,
    Path((group, reaction)): Path<(String, Uuid)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionDetailsList>, ApiError> {
    // start our list sub reaction details span
    let span = span!(Level::INFO, "List Sub Reaction Details Route");
    // get reaction data
    let (_, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // list reactions in a group with details
    let details = Reaction::list_sub(&reaction, params.cursor, params.limit, &state.shared)
        .await?
        .details(&group, &state.shared)
        .await?;
    Ok(Json(details))
}

/// Lists sub reactions with details for a specific parent reaction and status
///
/// # Arguments
///
/// * `user` - The user that is listing sub reactions
/// * `group` - The group to list sub reactions from
/// * `reaction` - The parent reaction to list sub reactions for
/// * `status` - The status to list sub reactions for
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/reactions/sub/:group/:reaction/:status/details/",
    params(
        ("group" = String, Path, description = "The group to list sub reactions from"),
        ("reaction" = Uuid, Path, description = "The parent reaction to list sub reactions for"),
        ("status" = ReactionStatus, Path, description = "The status to list sub reactions for"),
        ("params" = ReactionListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Sub-reaction details for the specified group and parent reaction and status", body = ReactionDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn list_sub_status_details(
    user: User,
    Path((group, reaction, status)): Path<(String, Uuid, ReactionStatus)>,
    Query(params): Query<ReactionListParams>,
    State(state): State<AppState>,
) -> Result<Json<ReactionDetailsList>, ApiError> {
    // start our list sub reaction details by status span
    let span = span!(Level::INFO, "List Sub Reaction Details By Status Route");
    // get reaction data
    let (_, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // list reactions in a group with details
    let details = Reaction::list_sub_status(
        &reaction,
        &status,
        params.cursor,
        params.limit,
        &state.shared,
    )
    .await?
    .details(&group, &state.shared)
    .await?;
    Ok(Json(details))
}

/// Updates a reaction
///
/// # Arguments
///
/// * `user` - The user that is updating this reaction
/// * `group` - The group this reaction is in
/// * `reaction` - The uuid of the reaction to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply to this reaction
#[utoipa::path(
    patch,
    path = "/api/reactions/:group/:id",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("reaction" = Uuid, Path, description = "The uuid of the reaction to update"),
        ("update" = ReactionUpdate, description = "The update to apply to this reaction")
    ),
    responses(
        (status = 200, description = "Updated reaction", body = Reaction),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn update(
    user: User,
    Path((group, reaction)): Path<(String, Uuid)>,
    State(state): State<AppState>,
    Json(update): Json<ReactionUpdate>,
) -> Result<Json<Reaction>, ApiError> {
    // start our update reaction span
    let span = span!(Level::INFO, "Update Reaction Route");
    // cast to to uuid then get reaction using id
    let (group, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // update our reaction
    let reaction = reaction
        .update(&user, &group, update, &state.shared)
        .await?;
    Ok(Json(reaction))
}

/// Deletes a reaction
///
/// This will only cancel any currently active pods if this is the only reaction causing that pod
/// type to be spawned.
///
/// # Arguments
///
/// * `user` - The user that is deleting this reaction
/// * `group` - The group this reaction is in
/// * `reaction` - The uuid of the reaction to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/reactions/:group/:id",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("reaction" = Uuid, Path, description = "The uuid of the reaction to update")
    ),
    responses(
        (status = 204, description = "Reaction deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn delete_reaction(
    user: User,
    Path((group, reaction)): Path<(String, Uuid)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // start our update reaction span
    let span = span!(Level::INFO, "Delete Reaction Route");
    // cast to to uuid then get reaction using id
    let (group, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // delete reaction from backend
    reaction.delete(&user, &group, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Downloads an ephemeral file for a reaction
///
/// # Arguments
///
/// * `user` - The user that is downloading this ephemeral file
/// * `group` - The group this reaction is in
/// * `reaction` - The uuid of the reaction to download an ephemeral file from
/// * `name` - The name of the ephemeral file
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/reactions/ephemeral/:group/:id/:name",
    params(
        ("group" = String, Path, description = "The group this reaction is in"),
        ("reaction" = Uuid, Path, description = "The uuid of the reaction to download an ephemeral file from"),
        ("name" = String, Path, description = "The name of the ephemeral file")
    ),
    responses(
        (status = 200, description = "Ephemeral file byte stream", body = Vec<u8>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn download_ephemeral(
    user: User,
    Path((group, reaction, name)): Path<(String, Uuid, String)>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // start our update reaction span
    let span = span!(Level::INFO, "Download Ephemeral File For Reaction Route");
    // cast to to uuid then get reaction using id
    let (_, reaction) = Reaction::get(&user, &group, &reaction, &state.shared, &span).await?;
    // start streaming an ephemeral file from s3
    let stream = reaction.download_ephemeral(&name, &state.shared).await?;
    // convert our byte stream to a streamable body
    let body = AsyncReadBody::new(stream.into_async_read());
    Ok(body)
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(create, create_bulk, get_reaction, update, delete_reaction, handle, logs, stage_logs, add_stage_logs,
          list, list_details, list_status, list_status_details, list_tag, list_tag_details, list_group_set,
          list_group_set_details, list_sub, list_sub_details, list_sub_status_details, list_sub_status,
          download_ephemeral),
    components(schemas(Actions, BulkReactionResponse, CommitishKinds, HandleReactionResponse, ImageScaler, JobResetRequestor, Reaction, ReactionIdResponse, ReactionList, ReactionDetailsList, ReactionListParams, ReactionRequest, ReactionStatus, ReactionUpdate, RepoDependency, RepoDependencyRequest, StageLogs, StageLogsAdd, StageLogLine, StatusUpdate, SystemComponents)),
    modifiers(&OpenApiSecurity),
)]
pub struct ReactionApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(ReactionApiDocs::openapi())
}

/// Add the streams routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/reactions/", post(create))
        .route("/api/reactions/bulk/", post(create_bulk))
        .route("/api/reactions/bulk/by/user/", post(create_bulk_by_user))
        .route(
            "/api/reactions/{group}/{id}",
            get(get_reaction).patch(update).delete(delete_reaction),
        )
        .route("/api/reactions/handle/{group}/{id}/{cmd}", post(handle))
        .route("/api/reactions/logs/{group}/{id}", get(logs))
        .route(
            "/api/reactions/logs/{group}/{id}/{stage}",
            get(stage_logs).post(add_stage_logs),
        )
        .route("/api/reactions/list/{group}/{pipeline}/", get(list))
        .route(
            "/api/reactions/list/{group}/{pipeline}/details/",
            get(list_details),
        )
        .route(
            "/api/reactions/status/{group}/{pipeline}/{status}/",
            get(list_status),
        )
        .route(
            "/api/reactions/status/{group}/{pipeline}/{status}/details/",
            get(list_status_details),
        )
        .route("/api/reactions/tag/{group}/{tag}/", get(list_tag))
        .route(
            "/api/reactions/tag/{group}/{tag}/details/",
            get(list_tag_details),
        )
        .route(
            "/api/reactions/group/{group}/{status}/",
            get(list_group_set),
        )
        .route(
            "/api/reactions/group/{group}/{status}/details/",
            get(list_group_set_details),
        )
        .route("/api/reactions/sub/{group}/{reaction}/", get(list_sub))
        .route(
            "/api/reactions/sub/{group}/{reaction}/details/",
            get(list_sub_details),
        )
        .route(
            "/api/reactions/sub/{group}/{reaction}/{status}/details/",
            get(list_sub_status_details),
        )
        .route(
            "/api/reactions/sub/{group}/{reaction}/{status}/",
            get(list_sub_status),
        )
        .route(
            "/api/reactions/ephemeral/{group}/{id}/{name}",
            get(download_ephemeral),
        )
}
