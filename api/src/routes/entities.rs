//! Routes for entities

use axum::extract::{Multipart, Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use axum_extra::body::AsyncReadBody;
use tracing::instrument;
use uuid::Uuid;

use super::shared::graphics;
use crate::models::backends::{GraphicSupport, TagSupport};
use crate::models::{
    ApiCursor, Entity, EntityListLine, EntityListParams, EntityResponse, TagDeleteRequest,
    TagRequest, User,
};
use crate::not_found;
use crate::utils::{ApiError, AppState};

/// Creates a new entity
///
/// # Arguments
///
/// * `user` - The user that is creating this entity
/// * `request` - The entity request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/entities/",
    responses(
        (status = 200, description = "Entity created", body = EntityResponse),
        (status = 401, description = "This user is not authorized to create entities in all given groups"),
        (status = 404, description = "One or more of the given groups does not exist"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::entities::create", skip_all, err(Debug))]
#[axum_macros::debug_handler]
async fn create(
    user: User,
    State(state): State<AppState>,
    form: Multipart,
) -> Result<Json<EntityResponse>, ApiError> {
    // create the entity
    let resp = Entity::create(&user, form, &state.shared).await?;
    Ok(Json(resp))
}

/// Creates a new entity
///
/// # Arguments
///
/// * `user` - The user that is creating this entity
/// * `request` - The entity request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/entities/:id",
    params(
        ("id" = Uuid, Path, description = "The entity's id"),
    ),
    responses(
        (status = 200, description = "Entity details", body = Entity),
        (status = 401, description = "This user is not authorized to access this route"),
        (status = 404, description = "A entity with the given name (and ID) does not exist in the user's groups"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::entities::get", skip_all, err(Debug))]
async fn get(
    user: User,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<Entity>, ApiError> {
    // get the entity from the backend
    let entity = Entity::get(&user, id, &state.shared).await?;
    // return the entity as JSON
    Ok(Json(entity))
}

/// Lists entities by the given parameters
///
/// # Arguments
///
/// * `user` - The user that is listing entities
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
//#[utoipa::path(
//get,
//path = "/api/entities/",
//params(
//("params" = EntityListParams, description = "Query params to use for this entity list request"),
//),
//responses(
//(status = 200, description = "JSON-formatted cursor response containing the names and ID's of entities as well as their upload timestamps", body = ApiCursor<EntityListLine>),
//(status = 401, description = "This user is not authorized to access this route"),
//),
//security(
//("basic" = []),
//)
//)]
#[instrument(name = "routes::entities::list", skip_all, err(Debug))]
async fn list(
    user: User,
    params: EntityListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<EntityListLine>>, ApiError> {
    // list entities
    let cursor = Entity::list(&user, params, false, &state.shared).await?;
    // return the cursor
    Ok(Json(cursor))
}

/// Lists entities and their details by the given parameters
///
/// # Arguments
///
/// * `user` - The user that is listing entity details
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/entities/details",
    params(
        ("params" = EntityListParams, description = "Query params to use for this entity list request"),
    ),
    responses(
        (status = 200, description = "JSON-formatted cursor response containing the entities' details", body = ApiCursor<Entity>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::entities::list_details", skip_all, err(Debug))]
async fn list_details(
    user: User,
    params: EntityListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<Entity>>, ApiError> {
    // list entities
    let list = Entity::list(&user, params, false, &state.shared).await?;
    // convert the list to a details list
    let cursor = list.details(&user, &state.shared).await?;
    // return the cursor
    Ok(Json(cursor))
}

/// Update an entity
///
/// # Arguments
///
/// * `user` - The user that is updating an entity
/// * `request` - The entity update request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = "/api/entities/:id",
    params(
        ("id" = Uuid, Path, description = "The entity id"),
    ),
    responses(
        (status = 204, description = "Entity updated"),
        (status = 401, description = "This user is not authorized to edit this entity or the group does not allow editing entities"),
        (status = 404, description = "One or more of the given groups does not exist or the entity does not exist"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::entities::update", skip_all, err(Debug))]
async fn update(
    user: User,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    form: Multipart,
) -> Result<StatusCode, ApiError> {
    // get the entity from the backend
    let entity = Entity::get(&user, id, &state.shared).await?;
    // update the entity
    entity.update(&user, form, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Update an entity
///
/// # Arguments
///
/// * `user` - The user that is updating an entity
/// * `request` - The entity update request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/entities/:id",
    params(
        ("id" = Uuid, Path, description = "The entity id"),
    ),
    responses(
        (status = 204, description = "Entity deleted"),
        (status = 401, description = "This user is not authorized to delete this entity or the group does not allow editing entities"),
        (status = 404, description = "One or more of the given groups does not exist or the entity does not exist"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::entities::delete", skip_all, err(Debug))]
async fn delete(
    user: User,
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    // get the entity from the backend
    let entity = Entity::get(&user, id, &state.shared).await?;
    // update the entity
    entity.delete(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Adds new tags to a entity
///
/// # Arguments
///
/// * `user` - The user that is adding new tags
/// * `sha256` - The entity to add a tag too
/// * `state` - Shared Thorium objects
/// * `tags` - The new tags to apply
#[utoipa::path(
    post,
    path = "/api/entities/tags/:id",
    params(
        ("id" = Uuid, Path, description = "The ID of the entity to add tags too"),
        ("tags" = TagRequest<Entity>, description = "JSON-formatted tags to apply to entity")
    ),
    responses(
        (status = 204, description = "entity tags updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::entities::tag", skip_all, err(Debug))]
async fn tag(
    user: User,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(tags): Json<TagRequest<Entity>>,
) -> Result<StatusCode, ApiError> {
    // get the entity we are adding tags too
    let entity = Entity::get(&user, id, &state.shared).await?;
    // try to add the new tags for this entity
    entity.tag(&user, tags, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes tags from a entity
///
/// # Arguments
///
/// * `user` - The user that is deleting tags
/// * `sha256` - The entity to delete tags from
/// * `state` - Shared Thorium objects
/// * `tags_del` - The tags to delete and the groups to delete them from
#[utoipa::path(
    delete,
    path = "/api/entities/tags/:id",
    params(
        ("Id" = Uuid, Path, description = "The id for the entity to delete tags from"),
        ("tags_del" = TagDeleteRequest<Entity>, description = "JSON-formatted tags to delete")
    ),
    responses(
        (status = 204, description = "entity tags deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::delete_tags", skip_all, err(Debug))]
async fn delete_tags(
    user: User,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(tags_del): Json<TagDeleteRequest<Entity>>,
) -> Result<StatusCode, ApiError> {
    // get the entity we are deleting tags from
    let entity = Entity::get(&user, id, &state.shared).await?;
    // try to delete the tags for this entity
    entity.delete_tags(&user, tags_del, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get an entity's image
///
/// # Arguments
///
/// * `user` - The user that is updating an entity
/// * `request` - The entity update request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    patch,
    path = "/api/entities/:id/image",
    params(
        ("id" = Uuid, Path, description = "The entity's ID")
    ),
    responses(
        (status = 200, description = "The image was successfully retrieved"),
        (status = 404, description = "The entity does not exist or the entity does not have an image"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::entities::get_image",
    skip(user, state),
    fields(user = user.username),
    err(Debug)
)]
#[axum_macros::debug_handler]
async fn get_image(
    user: User,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // get our entity by id
    let entity = Entity::get(&user, id, &state.shared).await?;
    // check if this entity has a graphic
    match &entity.image {
        Some(image_path) => {
            // get our
            let get_object = entity.download_graphic(image_path, &state.shared).await?;
            // get headers for this image
            let headers = graphics::get_headers(&get_object, image_path);
            // convert the output body to a streamable body
            let body = AsyncReadBody::new(get_object.body.into_async_read());
            // stream our body with its headers back
            Ok((headers, body))
        }
        None => not_found!(format!("Entity with id '{id}' has no image")),
    }
}

/// Add the entities routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/entities/", post(create))
        .route("/entities/", axum::routing::get(list))
        .route("/entities/details/", axum::routing::get(list_details))
        .route(
            "/entities/{id}",
            axum::routing::get(get).patch(update).delete(delete),
        )
        .route("/entities/{id}/image", axum::routing::get(get_image))
        .route("/entities/tags/{id}", post(tag).delete(delete_tags))
}
