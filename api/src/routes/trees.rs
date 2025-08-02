//! The route for building a relationship tree out of data in Thorium

use axum::extract::{Json, Path, State};
use axum::routing::{patch, post};
use axum::Router;
use tracing::instrument;
use uuid::Uuid;

use crate::models::{Tree, TreeGrowQuery, TreeParams, TreeQuery, User};
use crate::utils::{ApiError, AppState};

/// Get info on a specific sample by sha256
///
/// # Arguments
///
/// * `user` - The user that is getting info for a specific sha256
/// * `sha256` - The sha256 to get info about
/// * `state` - Shared Thorium objects
//#[utoipa::path(
//    get,
//    path = "/api/files/sample/:sha256",
//    params(
//        ("sha256" = String, Path, description = "Sha256 of the sample to get info on")
//    ),
//    responses(
//        (status = 200, description = "Return a sample in Thorium", body = Sample),
//        (status = 401, description = "This user is not authorized to access this route"),
//    ),
//    security(
//        ("basic" = []),
//    )
//)]
#[instrument(name = "routes::trees::start_tree", skip_all, err(Debug))]
async fn start_tree(
    user: User,
    params: TreeParams,
    State(state): State<AppState>,
    Json(query): Json<TreeQuery>,
) -> Result<Json<Tree>, ApiError> {
    // build a tree from our params
    let mut tree = Tree::from_query(&user, query, &state.shared).await?;
    // grow this tree to the desired depth
    tree.grow(&user, &params, &state.shared).await?;
    // save this tree
    tree.save(&user, &state.shared).await?;
    // empty our sent vec
    tree.sent.clear();
    // return our built tree
    Ok(Json(tree))
}

/// Continue to grow a tree based on some growable nodes
#[instrument(name = "routes::trees::grow_tree", skip_all, err(Debug))]
async fn grow_tree(
    user: User,
    params: TreeParams,
    Path(cursor): Path<Uuid>,
    State(state): State<AppState>,
    Json(query): Json<TreeGrowQuery>,
) -> Result<Json<Tree>, ApiError> {
    // load our existing tree
    let mut tree = Tree::load(&user, &cursor, &state.shared).await?;
    // set our growable nodes
    tree.growable = query.growable;
    // grow this tree
    let added = tree.grow(&user, &params, &state.shared).await?;
    // save the latest info on this tree
    tree.save(&user, &state.shared).await?;
    // trim to only the new info for this tree
    tree.trim(added);
    // empty our sent vec
    tree.sent.clear();
    Ok(Json(tree))
}

/// Add the tree routes to our router
///
/// # Arguments
///
/// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/trees/", post(start_tree))
        .route("/api/trees/{cursor}", patch(grow_tree))
}
