//! API routes for interacting with Thorium network policies

use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;
use tracing::instrument;
use utoipa::OpenApi;
use uuid::Uuid;

use super::OpenApiSecurity;
use crate::is_admin;
use crate::models::{
    ApiCursor, Group, IpBlock, IpBlockRaw, Ipv4Block, Ipv6Block, NetworkPolicy,
    NetworkPolicyCustomK8sRule, NetworkPolicyCustomLabel, NetworkPolicyListLine,
    NetworkPolicyListParams, NetworkPolicyPort, NetworkPolicyRequest, NetworkPolicyRule,
    NetworkPolicyRuleRaw, NetworkPolicyUpdate, NetworkProtocol, User,
};
use crate::utils::{ApiError, AppState};

/// The query params that may be required to identify a network policy
#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
struct NetworkPolicyParams {
    /// The network policy's ID, necessary if one or more distinct network policies
    /// share the same name
    id: Option<Uuid>,
}

/// Creates a new network policy
///
/// # Arguments
///
/// * `user` - The user that is creating this network policy
/// * `request` - The network policy request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/network-policies",
    params(
        ("request" = NetworkPolicyRequest, description = "The network policy request"),
    ),
    responses(
        (status = 204, description = "Network Policy created"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::network_policies::create", skip_all, err(Debug))]
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(request): Json<NetworkPolicyRequest>,
) -> Result<StatusCode, ApiError> {
    // only admins can create network policies
    is_admin!(user);
    // create the network policy
    NetworkPolicy::create(request, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Lists network policies
///
/// # Arguments
///
/// * `user` - The user that is listing network policies
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/network-policies/",
    params(
        ("params" = NetworkPolicyListParams, description = "The network policy request"),
    ),
    responses(
        (status = 200, description = "Network Policy list", body = ApiCursor<NetworkPolicyListLine>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::network_policies::list", skip_all, err(Debug))]
async fn list(
    user: User,
    params: NetworkPolicyListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<NetworkPolicyListLine>>, ApiError> {
    // get a list of network policies in these groups
    let cursor = NetworkPolicy::list(&user, params, false, &state.shared).await?;
    Ok(Json(cursor))
}

/// Lists network policies by creation date with details
///
/// Importantly, the `allowed groups` rules for both ingress and egress will
/// not include any groups the user is not apart of, even if the network policy
/// has allows those groups. This is to prevent leaking a group's existence to
/// users that don't have access to them. This does not occur if the user is an admin.
///
/// # Arguments
///
/// * `user` - The user that is listing network policies with details
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/network-policies/details/",
    params(
        ("params" = NetworkPolicyListParams, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "Network Policy list by creation date with details", body = ApiCursor<NetworkPolicy>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::network_policies::list_details", skip_all, err(Debug))]
#[axum_macros::debug_handler]
async fn list_details(
    user: User,
    params: NetworkPolicyListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<NetworkPolicy>>, ApiError> {
    // get a list of network policies in these groups
    let list = NetworkPolicy::list(&user, params, true, &state.shared).await?;
    // convert our list to a details list
    let mut cursor = list.details(&user, &state.shared).await?;
    if !user.is_admin() {
        // scrub sensitive details from the policy if the requesting user is not an admin
        for policy in &mut cursor.data {
            policy.scrub(&user);
        }
    }
    Ok(Json(cursor))
}

/// Get details on a single network policy
///
/// Importantly, the `allowed groups` rules for both ingress and egress will
/// not include any groups the user is not apart of, even if the network policy
/// has allows those groups. This is to prevent leaking a group's existence to
/// users that don't have access to them. This does not occur if the user is an admin.
///
/// # Arguments
///
/// * `user` - The user that is creating this network policy
/// * `request` - The network policy request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/network-policies/:name",
    params(
        ("name" = String, Path, description = "The network policy name"),
        ("params" = NetworkPolicyParams, Query, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "Network Policy details", body = NetworkPolicy),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::network_policies::get_network_policy",
    skip_all,
    err(Debug)
)]
async fn get_network_policy(
    user: User,
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(params): Query<NetworkPolicyParams>,
) -> Result<Json<NetworkPolicy>, ApiError> {
    // get the network policy
    let mut network_policy = NetworkPolicy::get(&name, params.id, &user, &state.shared).await?;
    // scrub sensitive details from the policy if the requesting user is not an admin
    network_policy.scrub(&user);
    Ok(Json(network_policy))
}

/// Delete a network policy from the given groups (or completely delete the network policy)
///
/// # Arguments
///
/// * `user` - The user that is creating this network policy
/// * `request` - The network policy request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = "/api/network-policies/:name",
    params(
        ("name" = String, Path, description = "The network policy name"),
        ("params" = NetworkPolicyParams, Query, description = "The query params to use for this request"),
        ("update" = NetworkPolicyUpdate, description = "The update to apply to this network policy"),
    ),
    responses(
        (status = 204, description = "Network Policy updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::network_policies::update", skip_all, err(Debug))]
async fn update(
    user: User,
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(params): Query<NetworkPolicyParams>,
    Json(update): Json<NetworkPolicyUpdate>,
) -> Result<StatusCode, ApiError> {
    // only admins can update network policies
    is_admin!(user);
    // check that the update is valid
    update.validate()?;
    // get the network policy
    let network_policy = NetworkPolicy::get(&name, params.id, &user, &state.shared).await?;
    // update the network policy
    network_policy.update(update, &user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a network policy from the given groups (or completely delete the network policy)
///
/// # Arguments
///
/// * `user` - The user that is creating this network policy
/// * `request` - The network policy request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/network-policies/:name",
    params(
        ("name" = String, Path, description = "The network policy name"),
        ("params" = NetworkPolicyParams, Query, description = "The query params to use for this request"),
    ),
    responses(
        (status = 204, description = "Network Policy deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::network_policies::delete", skip_all, err(Debug))]
async fn delete(
    user: User,
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(params): Query<NetworkPolicyParams>,
) -> Result<StatusCode, ApiError> {
    // only admins can delete network policies
    is_admin!(user);
    // get the network policy
    let network_policy = NetworkPolicy::get(&name, params.id, &user, &state.shared).await?;
    // delete the network policy
    network_policy.delete(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get a list of default network policies for the given group
///
/// # Arguments
///
/// * `user` - The user that is getting this network policy
/// * `group` - The group to get default network policies for
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/network-policies/default/:group/",
    params(
        ("group" = String, Path, description = "The group to get default network policies for"),
    ),
    responses(
        (status = 200, description = "Network Policy list", body = Vec<NetworkPolicyListLine>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::network_policies::get_all_default",
    skip_all,
    err(Debug)
)]
async fn get_all_default(
    user: User,
    Path(group): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<NetworkPolicyListLine>>, ApiError> {
    // see if the group exists and we have access to it
    let group = Group::get(&user, &group, &state.shared).await?;
    // get the default network policies
    let default_network_policies =
        NetworkPolicy::get_all_default(&group, &user, &state.shared).await?;
    // return the policies as a json list
    Ok(Json(default_network_policies))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(get_network_policy, update, delete, create, list, get_all_default, list_details),
    components(schemas(ApiCursor<NetworkPolicy>, ApiCursor<NetworkPolicyListLine>, IpBlock, IpBlockRaw, Ipv4Block, Ipv6Block, NetworkPolicy, NetworkPolicyCustomK8sRule, NetworkPolicyCustomLabel, NetworkPolicyListLine, NetworkPolicyListParams, NetworkPolicyParams, NetworkPolicyPort, NetworkPolicyRequest, NetworkPolicyRule, NetworkPolicyRuleRaw, NetworkPolicyUpdate, NetworkProtocol)),
    modifiers(&OpenApiSecurity),
)]
pub struct NetworkPolicyDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(NetworkPolicyDocs::openapi())
}

/// Add the network policies routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route(
            "/api/network-policies/{name}",
            get(get_network_policy).patch(update).delete(delete),
        )
        .route("/api/network-policies", post(create))
        .route("/api/network-policies/", get(list))
        .route(
            "/api/network-policies/default/{group}/",
            get(get_all_default),
        )
        .route("/api/network-policies/details/", get(list_details))
}
