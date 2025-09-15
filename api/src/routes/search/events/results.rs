//! Routes for result search events

use axum::http::StatusCode;
use axum::routing::patch;
use axum::Router;
use axum::{extract::State, Json};
use tracing::instrument;
use utoipa::OpenApi;

use crate::models::{
    ResultSearchEvent, SearchEvent, SearchEventBackend, SearchEventPopOpts, SearchEventStatus, User,
};
use crate::routes::docs::OpenApiSecurity;
use crate::utils::{ApiError, AppState};

/// Pops the given number of result search events from the queue
///
/// # Arguments
///
/// * `user` - The user popping result search events
/// * `params` - The search event pop params given by the user
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = format!("/api/search/events/{}/pop/", ResultSearchEvent::url()),
    params(
        ("params" = SearchEventPopOpts, description = "The query params to use with this request"),
    ),
    responses(
        (status = 200, description = "Returned result search events", body = Vec<ResultSearchEvent>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::search::events::results::pop", skip_all, err(Debug))]
async fn pop(
    user: User,
    params: SearchEventPopOpts,
    State(state): State<AppState>,
) -> Result<Json<Vec<ResultSearchEvent>>, ApiError> {
    // pop some search events
    let events = ResultSearchEvent::pop(&user, params.limit, &state.shared).await?;
    Ok(Json(events))
}

/// Clears specific result search events from the in-flight queue
/// and re-adds failed events to the main queue
///
/// # Arguments
///
/// * `user` - The user sending a status update on result search events
/// * `state` - Shared Thorium objects
/// * `status` - The status of the events to clear
#[utoipa::path(
    patch,
    path = format!("/api/search/events/{}/status/", ResultSearchEvent::url()),
    request_body = SearchEventStatus,
    responses(
        (status = 204, description = "Status report handled successfully"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::search::events::results::status", skip_all, err(Debug))]
async fn status(
    user: User,
    State(state): State<AppState>,
    Json(status): Json<SearchEventStatus>,
) -> Result<StatusCode, ApiError> {
    // handle search event status
    ResultSearchEvent::status(&user, status, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Moves all result search events in-flight to the regular queue
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
    path = format!("/api/search/events/{}/reset/", ResultSearchEvent::url()),
    request_body = SearchEventStatus,
    responses(
        (status = 204, description = "All results search events reset"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::search::events::results::handle_reset_all",
    skip_all,
    err(Debug)
)]
async fn reset_all(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // reset all in-flight search events
    ResultSearchEvent::reset_all(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(pop, status, reset_all),
    components(schemas(ResultSearchEvent, SearchEventPopOpts, SearchEventStatus)),
    modifiers(&OpenApiSecurity),
)]
pub struct ResultSearchEventApiDocs;

/// Mount the functions to their respective routes at the URL for
/// the implementing type
///
/// # Arguments
///
/// * `router` The router to add routes to
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    let url = ResultSearchEvent::url();
    router
        .route(&format!("/api/search/events/{url}/pop/"), patch(pop))
        .route(&format!("/api/search/events/{url}/status/"), patch(status))
        .route(
            &format!("/api/search/events/{url}/reset/"),
            patch(reset_all),
        )
}
