//! The routes related to associations

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::post;
use tracing::instrument;

use crate::models::AuthedUser;
use crate::models::{AssociationDeleteRequest, AssociationRequest};
use crate::utils::{ApiError, AppState};

/// Associate an entity or object with another entity/object
///
/// # Arguments
///
/// * `user` - The user that is creating associations
/// * `state` - Shared Thorium objects
/// * `req` - The association request to apply
#[instrument(name = "routes::associations::create", skip_all, err(Debug))]
async fn create(
    user: AuthedUser,
    State(state): State<AppState>,
    Json(req): Json<AssociationRequest>,
) -> Result<StatusCode, ApiError> {
    // apply this association request to the requested data
    req.apply(&user, &state.shared).await?;
    // if this request was successful then always return a 204
    Ok(StatusCode::NO_CONTENT)
}

/// Delete associations between an entity or object and other entities/objects
///
/// # Arguments
///
/// * `user` - The user that is deleting associations
/// * `state` - Shared Thorium objects
/// * `req` - The association delete request to apply
#[utoipa::path(
    delete,
    path = "/api/associations/",
    request_body = AssociationDeleteRequest,
    responses(
        (status = 204, description = "Associations deleted"),
        (status = 400, description = "No targets were set"),
        (status = 401, description = "This user is not authorized to delete these associations"),
        (status = 404, description = "The source does not exist or no matching associations were found"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::associations::delete", skip_all, err(Debug))]
async fn delete(
    user: AuthedUser,
    State(state): State<AppState>,
    Json(req): Json<AssociationDeleteRequest>,
) -> Result<StatusCode, ApiError> {
    // delete the associations matching this request
    req.apply(&user, &state.shared).await?;
    // if this request was successful then always return a 204
    Ok(StatusCode::NO_CONTENT)
}

/// Add the associations routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router.route("/associations/", post(create).delete(delete))
}
