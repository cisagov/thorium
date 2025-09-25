use crate::models::Version;
use crate::models::backends::system;
use crate::utils::{ApiError, AppState};
use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::get;
use tracing::{Level, event, instrument};
use utoipa::OpenApi;

use super::OpenApiSecurity;

/// API identification route
///
/// # Arguments
///
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/",
    responses(
        (status = 200, description = "Identify this API as the Thorium API", body = String, example = json!("Thorium"))
    )
)]
#[instrument(name = "routes::basic::identify", skip_all, err(Debug))]
pub async fn identify(State(state): State<AppState>) -> Result<String, ApiError> {
    system::iff(&state.shared).await
}

/// API banner display route
///
/// # Arguments
///
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/banner",
    responses(
        (status = 200, description = "Thorium banner message", body = String),
    )
)]
#[instrument(name = "routes::basic::banner", skip_all, err(Debug))]
pub async fn banner(State(state): State<AppState>) -> Result<String, ApiError> {
    Ok(state.shared.banner.clone())
}

/// API identification route
///
/// # Arguments
///
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/health",
    responses(
        (status = 204, description = "Thorium is healthy"),
        (status = 503, description = "Thorium is unhealthy"),
    )
)]
#[instrument(name = "routes::basic::health", skip_all)]
pub async fn health(State(state): State<AppState>) -> StatusCode {
    match system::health(&state.shared).await {
        Ok(healthy) => {
            // log our health
            event!(Level::INFO, healthy = healthy);
            // return 204 if we our healthy
            if healthy {
                return StatusCode::NO_CONTENT;
            }
        }
        // log this error
        Err(error) => event!(Level::ERROR, error = error.to_string()),
    }
    // default to a service unavailable error
    StatusCode::SERVICE_UNAVAILABLE
}

/// Return the current Thorium version
///
/// # Arguments
///
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/version",
    responses(
        (status = 200, description = "Return the current Thorium version", body = Version),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::basic::version", skip_all, err(Debug))]
pub async fn version() -> Result<Json<Version>, ApiError> {
    // get our version info
    let version = Version::new()?;
    Ok(Json(version))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(identify, banner, health, version),
    components(schemas(Version, ApiError)),
    modifiers(&OpenApiSecurity),
)]
pub struct BasicApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(BasicApiDocs::openapi())
}

/// Add the basic routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/", get(identify))
        .route("/banner", get(banner))
        .route("/version", get(version))
        .route("/health", get(health))
}
