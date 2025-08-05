use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, patch, post};
use axum::Router;
use tracing::instrument;

use utoipa::OpenApi;

use super::OpenApiSecurity;
// our imports
use crate::is_admin;
use crate::models::{
    Group, GroupAllowAction, GroupAllowed, GroupAllowedUpdate, GroupDetailsList, GroupList,
    GroupListParams, GroupMap, GroupRequest, GroupStats, GroupUpdate, GroupUsers,
    GroupUsersRequest, GroupUsersUpdate, PipelineStats, Roles, StageStats, User,
};
use crate::utils::{ApiError, AppState};

/// Creates a new group
///
/// # Arguments
///
/// * `user` - The user that is creating this group
/// * `group` - The group to create
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/groups/",
    params(
        ("group" = GroupRequest, description = "The group to create")
    ),
    responses(
        (status = 204, description = "Group created"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::create", skip_all, err(Debug))]
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(group): Json<GroupRequest>,
) -> Result<StatusCode, ApiError> {
    // create group
    Group::create(&user, group, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Gets details on a specific group
///
/// # Arguments
///
/// * `group` - The group to get details on
/// * `user` - The user that is creating this group
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/groups/:group/details",
    params(
        ("group" = String, Path, description = "The group to get details on")
    ),
    responses(
        (status = 200, description = "Group details", body = Group),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::get_group", skip_all, err(Debug))]
async fn get_group(
    user: User,
    Path(group): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Group>, ApiError> {
    // get this groups info
    let group = Group::get(&user, &group, &state.shared).await?;
    Ok(Json(group))
}

/// List groups
///
/// # Arguments
///
/// * `cursor` - The cursor value to determine where start listing groups at
/// * `limit` - The max number of groups to list at once
/// * `user` - The user that is creating this group
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/groups/",
    params(
        ("params" = GroupListParams, Query, description = "The query params for the groups to list")
    ),
    responses(
        (status = 200, description = "Group list", body = GroupList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::list", skip_all, err(Debug))]
async fn list(
    user: User,
    Query(params): Query<GroupListParams>,
    State(state): State<AppState>,
) -> Result<Json<GroupList>, ApiError> {
    // get vector of group names
    let groups = Group::list(&user, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(groups))
}

/// List groups with details
///
/// # Arguments
///
/// * `cursor` - The cursor value to determine where start listing groups at
/// * `limit` - The max number of groups to list at once
/// * `user` - The user that is creating this group
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/groups/details/",
    params(
        ("params" = GroupListParams, Query, description = "The query params for the groups to list")
    ),
    responses(
        (status = 200, description = "Group details list", body = GroupDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::list_details", skip_all, err(Debug))]
async fn list_details(
    user: User,
    Query(params): Query<GroupListParams>,
    State(state): State<AppState>,
) -> Result<Json<GroupDetailsList>, ApiError> {
    // get vector of group names
    let groups = Group::list_details(&user, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(groups))
}

/// Updates a group
///
/// # Arguments
///
/// * `user` - The user that is creating this group
/// * `group` - The name of the group to update
/// * `update` - The update to apply to this group
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = "/api/groups/:group",
    params(
        ("group" = String, Path, description = "The name of the group to update"),
        ("update" = GroupUpdate, description = "The update to apply to this group")
    ),
    responses(
        (status = 204, description = "Group updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::update", skip(user, state, update), err(Debug))]
async fn update(
    user: User,
    Path(group): Path<String>,
    State(state): State<AppState>,
    Json(update): Json<GroupUpdate>,
) -> Result<StatusCode, ApiError> {
    // get group
    let group = Group::get(&user, &group, &state.shared).await?;
    // update group
    group.update(update, &user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete group
///
/// # Arguments
///
/// * `user` - The user that is deleting this group
/// * `group` - The name of the group to delete
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/groups/:group",
    params(
        ("group" = String, Path, description = "The name of the group to delete")
    ),
    responses(
        (status = 204, description = "Group deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::delete_group", skip(user, state), err(Debug))]
async fn delete_group(
    user: User,
    Path(group): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // get group
    let group = Group::get(&user, &group, &state.shared).await?;
    // delete group
    group.delete(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Syncs all ldap metagroups and their users
///
/// # Arguments
///
/// * `user` - The user that is telling Thorium to sync with ldap
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/groups/sync/ldap",
    params(),
    responses(
        (status = 204, description = "Ldap metagroups and users sync'd"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::sync_ldap", skip(user, state), err(Debug))]
async fn sync_ldap(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // make sure we are an admin
    is_admin!(user);
    // sync all groups with ldap
    Group::sync_ldap(&state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Gets details on a specific group
///
/// # Arguments
///
/// * `group` - The group to get details on
/// * `user` - The user that is creating this group
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/groups/:group/stats",
    params(
        ("group" = String, Path, description = "The group to get details on"),
        ("params" = GroupListParams, Query, description = "The query params for the groups for which to get details")
    ),
    responses(
        (status = 200, description = "Group stats", body = GroupStats),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::groups::get_stats", skip_all, err(Debug))]
async fn get_stats(
    user: User,
    Query(params): Query<GroupListParams>,
    Path(group): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<GroupStats>, ApiError> {
    // get the group we are getting pipeline statuses for
    let group = Group::get(&user, &group, &state.shared).await?;
    // get the status object for this group
    let status = group
        .stats(params.cursor, params.limit, &state.shared)
        .await?;
    Ok(Json(status))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(create, list, get_group, list_details, update, delete_group, sync_ldap, get_stats),
    components(schemas(Group, GroupAllowed, GroupAllowedUpdate, GroupAllowAction, GroupDetailsList, GroupList, GroupListParams, GroupMap, GroupRequest, GroupStats, GroupUpdate, GroupUsersRequest, GroupUsers, GroupUsersUpdate, PipelineStats, Roles, StageStats)),
    modifiers(&OpenApiSecurity),
)]
pub struct GroupApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(GroupApiDocs::openapi())
}

/// Add the groups routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/groups/", post(create).get(list))
        .route("/api/groups/{group}/details", get(get_group))
        .route("/api/groups/details/", get(list_details))
        .route("/api/groups/{group}", patch(update).delete(delete_group))
        .route("/api/groups/sync/ldap", post(sync_ldap))
        .route("/api/groups/{group}/stats", get(get_stats))
}
