//! Routes for tag search events

use axum::http::StatusCode;
use axum::routing::patch;
use axum::Router;
use axum::{extract::State, Json};
use tracing::instrument;
use utoipa::OpenApi;

use crate::models::{
    SearchEvent, SearchEventBackend, SearchEventPopOpts, SearchEventStatus, TagSearchEvent, User,
};
use crate::routes::docs::OpenApiSecurity;
use crate::utils::{ApiError, AppState};

/// Pops the given number of tag search events from the queue
///
/// # Arguments
///
/// * `user` - The user popping tag search events
/// * `params` - The search event pop params given by the user
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = format!("/api/search/events/{}/pop/", TagSearchEvent::url()),
    params(
        ("params" = SearchEventPopOpts, description = "The query params to use with this request"),
    ),
    responses(
        (status = 200, description = "Returned tag search events", body = Vec<TagSearchEvent>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::search::events::tags::pop", skip_all, err(Debug))]
async fn pop(
    user: User,
    params: SearchEventPopOpts,
    State(state): State<AppState>,
) -> Result<Json<Vec<TagSearchEvent>>, ApiError> {
    // pop some search events
    let events = TagSearchEvent::pop(&user, params.limit, &state.shared).await?;
    Ok(Json(events))
}

/// Clears specific tag search events from the in-flight queue
/// and re-adds failed events to the main queue
///
/// # Arguments
///
/// * `user` - The user sending a status update on tag search events
/// * `state` - Shared Thorium objects
/// * `status` - The status of the events to clear
#[utoipa::path(
    patch,
    path = format!("/api/search/events/{}/status/", TagSearchEvent::url()),
    request_body = SearchEventStatus,
    responses(
        (status = 204, description = "Status report handled successfully"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::search::events::tags::status", skip_all, err(Debug))]
async fn status(
    user: User,
    State(state): State<AppState>,
    Json(status): Json<SearchEventStatus>,
) -> Result<StatusCode, ApiError> {
    // handle search event status
    TagSearchEvent::status(&user, status, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Moves all tag search events in-flight to the regular queue
///
/// Should be called when the consumer is rebooted in case
/// in-flight events were not processed completely
///
/// # Arguments
///
/// * `user` - The user clearing events
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = format!("/api/search/events/{}/reset/", TagSearchEvent::url()),
    request_body = SearchEventStatus,
    responses(
        (status = 204, description = "All tags search events reset"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::search::events::tags::reset_all", skip_all, err(Debug))]
async fn reset_all(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // reset all in-flight search events
    TagSearchEvent::reset_all(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(pop, status, reset_all),
    components(schemas(TagSearchEvent, SearchEventPopOpts, SearchEventStatus)),
    modifiers(&OpenApiSecurity),
)]
pub struct TagSearchEventApiDocs;

/// Mount the functions to their respective routes at the URL for
/// the implementing type
///
/// # Arguments
///
/// * `router` The router to add routes to
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    let url = TagSearchEvent::url();
    router
        .route(&format!("/api/search/events/{url}/pop/"), patch(pop))
        .route(&format!("/api/search/events/{url}/status/"), patch(status))
        .route(
            &format!("/api/search/events/{url}/reset/"),
            patch(reset_all),
        )
}
