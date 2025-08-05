use axum::extract::{Json, Path, Query, State};
use axum::routing::get;
use axum::Router;
use tracing::{instrument, Span};
use utoipa::OpenApi;

use super::OpenApiSecurity;
use crate::models::{Group, Stream, StreamDepth, User};
use crate::utils::{ApiError, AppState};

/// Gets the number of objects between two points in a stream
///
/// # Arguments
///
/// * `user` - The user that is getting a stream depth
/// * `group` - The group this stream is in
/// * `stream` - The name of the stream to count objects in
/// * `start` - The starting point in an epoch timestamp to count objects at
/// * `end` - The ending point in an epoch timestamp to count objects at
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/streams/depth/:group/:namespace/:stream/:start/:end",
    params(
        ("group" = String, Path, description = "The group this stream is in"),
        ("namespace" = String, Path, description = "The namespace this stream is in inside this group"),
        ("stream" = String, Path, description = "The name of the stream to count objects in"),
        ("start" = i64, Path, description = "The starting point in an epoch timestamp to count objects at"),
        ("end" = i64, Path, description = "The ending point in an epoch timestamp to count objects at"),
    ),
    responses(
        (status = 200, description = "The number of objects in the stream between start and end time", body = StreamDepth),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::gstreams::depth", skip_all, err(Debug))]
pub async fn depth(
    user: User,
    Path((group, namespace, stream, start, end)): Path<(String, String, String, i64, i64)>,
    State(state): State<AppState>,
) -> Result<Json<StreamDepth>, ApiError> {
    // get group
    let group = Group::get(&user, &group, &state.shared).await?;
    // read from the deadline stream
    let depth = Stream::depth(&group, &namespace, &stream, start, end, &state.shared).await?;
    Ok(Json(depth))
}

/// Gets the number of objects between two points in a stream in a range smaller chunks
///
/// If the split does not fit evenly into the overall range then the last chunk will be smaller.
///
/// # Arguments
///
/// * `user` - The user that is getting stream depths
/// * `group` - The group this stream is in
/// * `stream` - The name of the stream to count objects in
/// * `start` - The starting point in an epoch timestamp to count objects at
/// * `end` - The ending point in an epoch timestamp to count objects at
/// * `split` - How many seconds each chunk should cover
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/streams/depth/:group/:namespace/:stream/:start/:end/:split",
    params(
        ("group" = String, Path, description = "The group this stream is in"),
        ("namespace" = String, Path, description = "The namespace this stream is in inside this group"),
        ("stream" = String, Path, description = "The name of the stream to count objects in"),
        ("start" = i64, Path, description = "The starting point in an epoch timestamp to count objects at"),
        ("end" = i64, Path, description = "The ending point in an epoch timestamp to count objects at"),
        ("split" = i64, Path, description = "How many seconds each chunk should cover"),
    ),
    responses(
        (status = 200, description = "The number of objects in the stream between start and end time in specified chunks", body = Vec<StreamDepth>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::streams::depth_range", skip_all, err(Debug))]
pub async fn depth_range(
    user: User,
    Path((group, namespace, stream, start, end, split)): Path<(
        String,
        String,
        String,
        i64,
        i64,
        i64,
    )>,
    State(state): State<AppState>,
) -> Result<Json<Vec<StreamDepth>>, ApiError> {
    // get group
    let group = Group::get(&user, &group, &state.shared).await?;
    // read from the deadline stream
    let depths = Stream::depth_range(
        &group,
        &namespace,
        &stream,
        start,
        end,
        split,
        &state.shared,
    )
    .await?;
    Ok(Json(depths))
}

/// Helps serde default the map minimum to 500
fn default_map_min() -> u64 {
    500
}

/// The query params for a map request
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct MapParams {
    /// The minimum number of objects to map
    #[serde(default = "default_map_min")]
    pub min: u64,
}

/// Map a deadline stream to get a rough idea of what it looks like.
///
/// # Arguments
///
/// * `user` - The user that is mapping this stream
/// * `group` - The group this stream is in
/// * `namespace` - The namespace for this stream
/// * `stream` - The name of the stream to map
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/streams/map/:group/:namespace/:stream",
    params(
        ("group" = String, Path, description = "The group this stream is in"),
        ("namespace" = String, Path, description = "The namespace for this stream"),
        ("stream" = String, Path, description = "The name of the stream to map"),
        ("params" = MapParams, Query, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "StreamDepth map", body = Vec<StreamDepth>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::streams::map", skip_all, err(Debug))]
pub async fn map(
    user: User,
    Path((group, namespace, stream)): Path<(String, String, String)>,
    Query(params): Query<MapParams>,
    State(state): State<AppState>,
) -> Result<Json<Vec<StreamDepth>>, ApiError> {
    // get our current span
    let span = Span::current();
    // map the stream
    let map = Stream::map(
        &user,
        &group,
        &namespace,
        &stream,
        params.min,
        &state.shared,
        &span,
    )
    .await?;
    Ok(Json(map))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(depth, depth_range, map),
    components(schemas(MapParams, StreamDepth)),
    modifiers(&OpenApiSecurity),
)]
pub struct StreamApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(StreamApiDocs::openapi())
}

/// Add the streams routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route(
            "/api/streams/depth/{group}/{namespace}/{stream}/{start}/{end}",
            get(depth),
        )
        .route(
            "/api/streams/depth/{group}/{namespace}/{stream}/{start}/{end}/{split}",
            get(depth_range),
        )
        .route("/api/streams/map/{group}/{namespace}/{stream}", get(map))
}
