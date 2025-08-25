//! The routes related to associations

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::post;

use crate::models::{ApiCursor, Association, AssociationListParams, AssociationRequest, User};
use crate::utils::{ApiError, AppState};

/// Associate an entity or object with another entity/object
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(req): Json<AssociationRequest>,
) -> Result<StatusCode, ApiError> {
    // apply this association request to the requested data
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
    router.route("/associations/", post(create))
}
