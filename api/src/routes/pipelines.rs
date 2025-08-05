use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, patch, post};
use axum::Router;
use tracing::instrument;
use uuid::Uuid;

use super::shared;
use crate::is_admin;
use utoipa::OpenApi;

use super::OpenApiSecurity;
use crate::models::pipelines::{BannedImageBan, GenericBan};
use crate::models::{
    EventTrigger, Group, Notification, NotificationParams, NotificationRequest, Pipeline,
    PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineDetailsList, PipelineKey,
    PipelineList, PipelineListParams, PipelineRequest, PipelineUpdate, TagType, User,
};
use crate::utils::{ApiError, AppState};

/// Creates a new pipeline in a group
///
/// # Arguments
///
/// * `user` - The user that is creating this pipeline
/// * `request` - The pipeline to create
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/pipelines/",
    params(
        ("request" = PipelineRequest, description = "The pipeline to create"),
    ),
    responses(
        (status = 204, description = "Pipeline created"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::pipelines::create", skip_all, err(Debug))]
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(request): Json<PipelineRequest>,
) -> Result<StatusCode, ApiError> {
    // create pipeline object
    Pipeline::create(&user, request, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Gets details about a pipeline
///
/// # Arguments
///
/// * `user` - The user that is getting details about this pipeline
/// * `group` - The group this pipeline is in
/// * `pipeline` - The name of the pipeline to get details about
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/pipelines/data/:group/:pipeline",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline to get details about"),
    ),
    responses(
        (status = 200, description = "Pipeline details", body = Pipeline),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::pipelines::get", skip_all, err(Debug))]
async fn get_pipeline(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Pipeline>, ApiError> {
    // get pipeline data
    let (_, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    Ok(Json(pipeline))
}

/// Lists pipelines in a group
///
/// # Arguments
///
/// * `user` - The user that is listing pipelines
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/pipelines/list/:group/",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("params" = PipelineListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Pipeline details", body = PipelineList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::pipelines::list",
    skip(user, params, state),
    err(Debug)
)]
async fn list(
    user: User,
    Path(group): Path<String>,
    Query(params): Query<PipelineListParams>,
    State(state): State<AppState>,
) -> Result<Json<PipelineList>, ApiError> {
    // authorize this user is apart of this group
    let group = Group::authorize(&user, &group, &state.shared).await?;
    // get list of pipelines in group
    let names = Pipeline::list(&group, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(names))
}

/// Lists pipelines in a group with details
///
/// # Arguments
///
/// * `user` - The user that is listing pipelines
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/pipelines/list/:group/details/",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("params" = PipelineListParams, Query, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Pipeline details", body = PipelineDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::pipelines::list_details",
    skip(user, params, state),
    err(Debug)
)]
async fn list_details(
    user: User,
    Path(group): Path<String>,
    Query(params): Query<PipelineListParams>,
    State(state): State<AppState>,
) -> Result<Json<PipelineDetailsList>, ApiError> {
    // authorize this user is apart of this group
    let group = Group::authorize(&user, &group, &state.shared).await?;
    // get list of pipelines in group
    let pipelines = Pipeline::list(&group, params.cursor, params.limit, &state.shared).await?;
    // get details these pipelines
    let details = pipelines.details(&group, &state.shared).await?;
    Ok(Json(details))
}

/// Updates a pipeline
///
/// # Arguments
///
/// * `user` - The user that is updating this pipeline
/// * `group` - The group this pipeline is in
/// * `pipeline` - The name of the pipeline to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply
#[utoipa::path(
    patch,
    path = "/api/pipelines/:group/:pipeline",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline to update"),
        ("update" = PipelineUpdate, description = "The update to apply")
    ),
    responses(
        (status = 204, description = "Pipeline updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::pipelines::update", skip_all, err(Debug))]
async fn update(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(update): Json<PipelineUpdate>,
) -> Result<StatusCode, ApiError> {
    // get pipeline and group
    let (group, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // update pipelines
    pipeline
        .update(update, &user, &group, &state.shared)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes a pipeline
///
/// This will also delete any reactions tied to this pipeline.
///
/// # Arguments
///
/// * `user` - The user that is deleting this pipeline
/// * `group` - The group this pipeline is in
/// * `pipeline` - The name of the pipeline to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/pipelines/:group/:pipeline",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline to delete"),
    ),
    responses(
        (status = 204, description = "Pipeline deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::pipelines::delete_pipeline",
    skip(user, state),
    err(Debug)
)]
async fn delete_pipeline(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // get pipeline
    let (group, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // delete pipeline from backend
    pipeline.delete(&user, &group, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Create a notification for a pipeline in scylla
///
/// # Arguments
///
/// * `user` - The user sending the request
/// * `group` - The group containing the pipeline
/// * `pipeline` - The name of the pipeline the notification will be created for
/// * `state` - Shared Thorium objects
/// * `params` - The notification params sent with the request
/// * `req` - The notification request that was sent
#[utoipa::path(
    post,
    path = "/api/pipelines/notifications/:group/:pipeline",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline the notification will be created for"),
        ("params" = NotificationParams, description = "The notification params sent with the request"),
    ),
    responses(
        (status = 204, description = "Pipeline notification created"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::pipelines::create_notification", skip_all, err(Debug))]
async fn create_notification(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    State(state): State<AppState>,
    params: NotificationParams,
    Json(req): Json<NotificationRequest<Pipeline>>,
) -> Result<StatusCode, ApiError> {
    // only admins can create pipeline notifications
    is_admin!(&user);
    // check that the pipeline exists
    let (_, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // generate the pipeline's key
    let key = PipelineKey::from(&pipeline);
    // create the notification
    shared::notifications::create_notification(pipeline, key, req, params, &state.shared).await
}

/// Get the all of the pipeline's notifications
///
/// # Arguments
///
/// * `user` - The user that is requesting the notifications
/// * `group` - The group containing the pipeline
/// * `pipeline` - The name of the pipeline whose notifications are being requested
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/pipelines/notifications/:group/:pipeline",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline whose notifications are being requested"),
    ),
    responses(
        //(status = 200, description = "Pipeline notifications", body = Vec<Notification<Pipeline>>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::pipelines::get_notifications", skip_all, err(Debug))]
async fn get_notifications(
    user: User,
    Path((group, pipeline)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<Notification<Pipeline>>>, ApiError> {
    // check that the pipeline exists and the user has access
    let (_, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // generate the pipeline's key
    let key = PipelineKey::from(&pipeline);
    // get all of the pipeline's notifications
    shared::notifications::get_notifications(pipeline, key, &state.shared).await
}

/// Delete a specific notification from a pipeline
///
/// # Arguments
///
/// * `user` - The user that is requesting the deletion
/// * `group` - The group containing the pipeline
/// * `pipeline` - The name of the pipeline whose log is being deleted
/// * `id` - The notification's unique ID
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/pipelines/notifications/:group/:pipeline/:id",
    params(
        ("group" = String, Path, description = "The group this pipeline is in"),
        ("pipeline" = String, Path, description = "The name of the pipeline whose log is being deleted"),
        ("id" = Uuid, Path, description = "The notification's unique ID"),
    ),
    responses(
        (status = 204, description = "Pipeline notification deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::pipelines::delete_notification", skip_all, err(Debug))]
async fn delete_notification(
    user: User,
    Path((group, pipeline, id)): Path<(String, String, Uuid)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // only admins can delete pipeline logs
    is_admin!(&user);
    // check that the pipeline exists
    let (_, pipeline) = Pipeline::get(&user, &group, &pipeline, &state.shared).await?;
    // generate the pipeline's key
    let key = PipelineKey::from(&pipeline);
    // delete the notification
    shared::notifications::delete_notification(pipeline, key, None, id, &state.shared).await
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(create, get_pipeline, list, list_details, update, delete_pipeline),
    components(schemas(BannedImageBan, EventTrigger, GenericBan, Pipeline, PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineDetailsList, PipelineList, PipelineListParams, PipelineRequest, PipelineUpdate, TagType)),
    modifiers(&OpenApiSecurity),
)]
pub struct PipelineApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(PipelineApiDocs::openapi())
}

/// Add the file routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/pipelines/", post(create))
        .route("/api/pipelines/data/{group}/{pipeline}", get(get_pipeline))
        .route("/api/pipelines/list/{group}/", get(list))
        .route("/api/pipelines/list/{group}/details/", get(list_details))
        .route(
            "/api/pipelines/{group}/{pipeline}",
            patch(update).delete(delete_pipeline),
        )
        .route(
            "/api/pipelines/notifications/{group}/{pipeline}",
            get(get_notifications).post(create_notification),
        )
        .route(
            "/api/pipelines/notifications/{group}/{pipeline}/{id}",
            delete(delete_notification),
        )
}
