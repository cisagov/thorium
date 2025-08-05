//! The routes supporting events in Thorium
use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, patch};
use axum::Router;
use tracing::instrument;
use utoipa::OpenApi;

use super::OpenApiSecurity;
use crate::models::{
    Event, EventCacheStatus, EventCacheStatusOpts, EventIds, EventPopOpts, EventType, User,
};
use crate::utils::{ApiError, AppState};

/// Pop and handle some events
///
/// # Arguments
//
///
/// * `user` - The user that is popping events
/// * `params` - The query params to use with this request
/// * `state` - Shared Thorium objects
#[instrument(name = "routes::events::pop", skip_all, err(Debug))]
#[utoipa::path(
    patch,
    path = "/api/events/pop/:kind",
    params(
        ("kind" = EventType, description = "The type of events to pop"),
        ("params" = EventPopOpts, description = "Query params for popping events")
    ),
    responses(
        (status = 200, description = "A List of popped events in Thorium", body = Vec<Event>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn pop(
    user: User,
    Path(kind): Path<EventType>,
    params: EventPopOpts,
    State(state): State<AppState>,
) -> Result<Json<Vec<Event>>, ApiError> {
    // pop some events
    let events = Event::pop(&user, kind, params.limit, &state.shared).await?;
    Ok(Json(events))
}

/// clear some events
///
/// # Arguments
//
///
/// * `user` - The user that is clearing events
/// * `kind` - The kind of events to clear
/// * `id_list` - The events to clear
/// * `state` - Shared Thorium objects
#[instrument(name = "routes::events::clear", skip_all, err(Debug))]
#[utoipa::path(
    patch,
    path = "/api/events/clear/:kind",
    params(
        ("kind" = EventType, description = "The type of events to clear"),
        ("id_list" = EventIds, description = "JSON-formatted list of event ids to clear")
    ),
    responses(
        (status = 200, description = "The requested events were cleared"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn clear(
    user: User,
    Path(kind): Path<EventType>,
    State(state): State<AppState>,
    Json(id_list): Json<EventIds>,
) -> Result<StatusCode, ApiError> {
    // clear these events
    Event::clear(&user, kind, &id_list.ids, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Reset all in flight events
///
/// # Arguments
//
///
/// * `user` - The user that is resetting events
/// * `kind` - The kind of events to reset
/// * `state` - Shared Thorium objects
#[instrument(name = "routes::events::rest_all", skip_all, err(Debug))]
#[utoipa::path(
    patch,
    path = "/api/events/reset/:kind",
    params(
        ("kind" = EventType, description = "The type of events to reset"),
    ),
    responses(
        (status = 200, description = "All events currently in process were cleared"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
async fn reset_all(
    user: User,
    Path(kind): Path<EventType>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // reset all in flight events events
    Event::reset_all(&user, kind, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// get the status of our event handler cache
///
/// This is used to determine if our local cache needs to be refreshed
#[utoipa::path(
    get,
    path = "/api/events/cache/status/",
    params(
        ("params" = EventCacheStatusOpts, Query, description = "Whether to reset any cache statuses")
    ),
    responses(
        (status = 200, description = "Event handler cache status", body = EventCacheStatus),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::events::get_cache_status", skip_all, err(Debug))]
async fn get_cache_status(
    user: User,
    Query(params): Query<EventCacheStatusOpts>,
    State(state): State<AppState>,
) -> Result<Json<EventCacheStatus>, ApiError> {
    // get our event watermark
    let status = Event::get_cache_status(&user, params.reset, &state.shared).await?;
    Ok(Json(status))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(pop, clear, reset_all, get_cache_status),
    components(schemas(Event, EventCacheStatus, EventCacheStatusOpts, EventType, EventPopOpts)),
    modifiers(&OpenApiSecurity),
)]
pub struct EventApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(EventApiDocs::openapi())
}

/// Add the events routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/events/pop/{kind}/", patch(pop))
        .route("/api/events/clear/{kind}/", delete(clear))
        .route("/api/events/reset/{kind}/", patch(reset_all))
        .route("/api/events/cache/status/", get(get_cache_status))
}
