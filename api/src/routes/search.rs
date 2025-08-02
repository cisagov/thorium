//! The search routes for Thorium

use axum::extract::{Json, State};
use axum::routing::get;
use axum::Router;
use tracing::instrument;
use utoipa::OpenApi;

use super::OpenApiSecurity;
use crate::models::backends;
use crate::models::elastic::ElasticIndex;
use crate::models::ElasticSearchParams;
use crate::models::{ApiCursor, ElasticDoc, User};
use crate::utils::{ApiError, AppState};

pub mod events;

/// Search results in elastic and return a list of sha256s
///
/// # Arguments
///
/// * `user` - The user that is listing submissions
/// * `params` - The query params to use with this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/search/",
    params(
        ("params" = ElasticSearchParams, description = "The query params to use with this request"),
    ),
    responses(
        (status = 200, description = "Returned Elasticsearch docs matching search", body = ApiCursor<ElasticDoc>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::search::search", skip_all, err(Debug))]
async fn search(
    user: User,
    params: ElasticSearchParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<ElasticDoc>>, ApiError> {
    // get a list of elastic documents in the given indexes based on the given search
    let cursor = backends::search::search(&user, params, &state.shared).await?;
    Ok(Json(cursor))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(search),
    components(schemas(ApiCursor<ElasticDoc>, ElasticDoc, ElasticIndex, ElasticSearchParams)),
    modifiers(&OpenApiSecurity),
)]
pub struct SearchApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(SearchApiDocs::openapi())
}

// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    let router = router.route("/api/search/", get(search));
    // mount search events routes
    events::mount(router)
}
