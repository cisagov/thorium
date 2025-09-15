//! The route for building a relationship tree out of data in Thorium

use axum::Router;
use axum::extract::{Json, Path, State};
use axum::routing::{patch, post};
use tracing::instrument;
use uuid::Uuid;

use crate::models::{Tree, TreeGrowQuery, TreeParams, TreeQuery, User};
use crate::utils::{ApiError, AppState};

/// Start building a tree of data in Thorium from some starting points
///
/// # Arguments
///
/// * `user` - The user that is building a tree
/// * `params` - The params for building this tree
/// * `state` - Shared Thorium objects
/// * `query` - The query to use to start this tree
#[utoipa::path(
    post,
    path = "/api/trees/",
    params(
        ("params" = TreeParams, description = "The params for starting a new tree"),
        ("query" = TreeQuery, description = "The query to use to start a new tree")
    ),
    responses(
        (status = 200, description = "A new tree", body = Tree),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
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
    // clear any non user facing info
    tree.clear_non_user_facing();
    // return our built tree
    Ok(Json(tree))
}

/// Continue to grow a tree based on some growable nodes
/// Start building a tree of data in Thorium from some starting points
///
/// # Arguments
///
/// * `user` - The user that is building a tree
/// * `params` - The params for building this tree
/// * `state` - Shared Thorium objects
/// * `query` - The query to use to start this tree
#[utoipa::path(
    patch,
    path = "/api/trees/:cursor",
    params(
        ("cursor" = Uuid, description = "The id of an existing tree to grow"),
        ("params" = TreeParams, description = "The params for growing an existing tree"),
        ("query" = TreeGrowQuery, description = "The query to use to grow an existing tree")
    ),
    responses(
        (status = 200, description = "A tree grown from an existing tree", body = Tree),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
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
    tree.growable = query.growable.clone();
    // grow this tree
    let added = tree.grow(&user, &params, &state.shared).await?;
    // save the latest info on this tree
    tree.save(&user, &state.shared).await?;
    // trim to only the new info for this tree
    tree.trim(query.growable, added);
    // clear any non user facing info
    tree.clear_non_user_facing();
    Ok(Json(tree))
}

/// Add the tree routes to our router
///
/// # Arguments
///
/// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/trees/", post(start_tree))
        .route("/trees/{cursor}", patch(grow_tree))
}
