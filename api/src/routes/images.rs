use super::shared;
use crate::is_admin;
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, patch, post};
use axum::Router;
use tracing::instrument;
use utoipa::OpenApi;
use uuid::Uuid;

use super::OpenApiSecurity;
// our imports
use crate::models::images::{GenericBan, InvalidHostPathBan, InvalidUrlBan};
use crate::models::{
    ArgStrategy, AutoTag, AutoTagLogic, AutoTagUpdate, ChildFilters, ChildFiltersUpdate,
    ChildrenDependencySettings, ChildrenDependencySettingsUpdate, Cleanup, CleanupUpdate,
    ConfigMap, Dependencies, DependenciesUpdate, DependencyPassStrategy, DependencySettingsUpdate,
    EphemeralDependencySettings, EphemeralDependencySettingsUpdate, FilesHandler,
    FilesHandlerUpdate, Group, HostPath, HostPathTypes, Image, ImageArgs, ImageArgsUpdate,
    ImageBan, ImageBanKind, ImageBanUpdate, ImageDetailsList, ImageKey, ImageLifetime, ImageList,
    ImageListParams, ImageNetworkPolicyUpdate, ImageRequest, ImageScaler, ImageUpdate,
    ImageVersion, Kvm, KvmUpdate, KwargDependency, Notification, NotificationLevel,
    NotificationParams, NotificationRequest, OutputCollection, OutputCollectionUpdate,
    OutputDisplayType, OutputHandler, RepoDependencySettings, Resources, ResourcesRequest,
    ResourcesUpdate, ResultDependencySettings, ResultDependencySettingsUpdate,
    SampleDependencySettings, Secret, SecurityContext, SecurityContextUpdate, SpawnLimits,
    TagDependencySettings, TagDependencySettingsUpdate, User, Volume, VolumeTypes, NFS,
};
use crate::utils::{ApiError, AppState};

/// Creates a new image in a group
///
/// # Arguments
///
/// * `user` - The user that is creating this image
/// * `state` - Shared Thorium objects
/// * `image_request` - The image to create
#[utoipa::path(
    post,
    path = "/api/images/",
    params(
        ("image_request" = ImageRequest, description = "The image to create")
    ),
    responses(
        (status = 204, description = "Image created"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::create", skip_all, err(Debug))]
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(image_request): Json<ImageRequest>,
) -> Result<StatusCode, ApiError> {
    // create Image
    Image::create(&user, image_request, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Gets details about a specific image
///
/// # Arguments
///
/// * `user` - The user that is getting this images info
/// * `group` - The group this image is in
/// * `image` - The name of the image to get details about
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/images/data/:group/:image",
    params(
        ("group" = String, Path, description = "The group this image is in"),
        ("image" = String, Path, description = "The name of the image to get details about")
    ),
    responses(
        (status = 200, description = "Image details", body = Image),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::get_image", skip_all, err(Debug))]
async fn get_image(
    user: User,
    Path((group, image)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Image>, ApiError> {
    // get image
    let (_, image) = Image::get(&user, &group, &image, &state.shared).await?;
    Ok(Json(image))
}

/// Lists images in a group
///
/// # Arguments
///
/// * `user` - The user that is listing images
/// * `group` - The group to list images from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/images/:group/",
    params(
        ("group" = String, Path, description = "The group to list images from"),
        ("params" = ImageListParams, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Group images details", body = ImageList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::list", skip_all, err(Debug))]
async fn list(
    user: User,
    Path(group): Path<String>,
    params: ImageListParams,
    State(state): State<AppState>,
) -> Result<Json<ImageList>, ApiError> {
    // authorize this user is apart of this group
    let group = Group::authorize(&user, &group, &state.shared).await?;
    // get image names
    let names = Image::list(&group, params.cursor, params.limit, &state.shared).await?;
    Ok(Json(names))
}

/// Lists images in a group with details
///
/// # Arguments
///
/// * `user` - The user that is listing images
/// * `group` - The group to list images from
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/images/:group/details/",
    params(
        ("group" = String, Path, description = "The group to list images from"),
        ("params" = ImageListParams, description = "The query params to use for this request")
    ),
    responses(
        (status = 200, description = "Group images details", body = ImageDetailsList),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[axum_macros::debug_handler]
#[instrument(name = "routes::images::list_details", skip_all, err(Debug))]
async fn list_details(
    user: User,
    Path(group): Path<String>,
    params: ImageListParams,
    State(state): State<AppState>,
) -> Result<Json<ImageDetailsList>, ApiError> {
    // authorize this user is apart of this group
    let group = Group::authorize(&user, &group, &state.shared).await?;
    // get a list of images
    let images = Image::list(&group, params.cursor, params.limit, &state.shared).await?;
    // transform this list into a details list
    let details = images.details(&group, &state.shared).await?;
    Ok(Json(details))
}

/// Updates an image
///
/// # Arguments
///
/// * `user` - The user that is updating this image
/// * `group` - The group to update an image from
/// * `image` - The name of the image to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply
#[utoipa::path(
    patch,
    path = "/api/images/:group/:image",
    params(
        ("group" = String, Path, description = "The group to update an image from"),
        ("image" = String, Path, description = "The name of the image to update"),
        ("update" = ImageUpdate, description = "The update to apply")
    ),
    responses(
        (status = 204, description = "Image updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::update", skip_all, err(Debug))]
async fn update(
    user: User,
    Path((group, image)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(update): Json<ImageUpdate>,
) -> Result<StatusCode, ApiError> {
    // get image
    let (group, image) = Image::get(&user, &group, &image, &state.shared).await?;
    // update the image in the backend
    image.update(update, &user, &group, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes an image
///
/// An image cannot be in use by any pipelines when it is deleted.
///
/// # Arguments
///
/// * `user` - The user that is deleting this image
/// * `group` - The group to delete an image from
/// * `image` - The name of the image to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/images/:group/:image",
    params(
        ("group" = String, Path, description = "The group to delete an image from"),
        ("image" = String, Path, description = "The name of the image to delete")
    ),
    responses(
        (status = 204, description = "Image deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::delete_image", skip_all, err(Debug))]
async fn delete_image(
    user: User,
    Path((group, image)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // get image
    let (group, image) = Image::get(&user, &group, &image, &state.shared).await?;
    // delete image from backend
    image.delete(&user, &group, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Update the average runtimes for all images
///
/// # Arguments
///
/// * `user` - The user that is telling Thorium to update all image runtimes
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/images/runtimes/update",
    params(),
    responses(
        (status = 204, description = "Runtimes updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::runtimes_update", skip_all, err(Debug))]
async fn runtimes_update(
    user: User,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // crawl groups and calculate and update their images average runtimes
    Image::update_runtimes(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Create a notification for an image in scylla
///
/// # Arguments
///
/// * `user` - The user sending the request
/// * `group` - The group the image is in
/// * `image` - The name of the image for which the notification will be created
/// * `state` - Shared Thorium objects
/// * `params` - The notification params sent with the request
/// * `req` - The notification request that was sent
#[utoipa::path(
    post,
    path = "/api/images/notifications/:group/:image",
    params(
        ("group" = String, Path, description = "The group the image is in"),
        ("image" = String, Path, description = "The name of the image for which the notification will be created"),
        ("params" = NotificationParams, description = "The notification params sent with the request"),
        ("req" = NotificationRequest<Image>, description = "The notification request that was sent")
    ),
    responses(
        (status = 204, description = "Notification created"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::create_notification", skip_all, err(Debug))]
async fn create_notification(
    user: User,
    Path((group, image)): Path<(String, String)>,
    State(state): State<AppState>,
    params: NotificationParams,
    Json(req): Json<NotificationRequest<Image>>,
) -> Result<StatusCode, ApiError> {
    // only admins can create image notifications
    is_admin!(&user);
    // check that the image exists
    let (_, image) = Image::get(&user, &group, &image, &state.shared).await?;
    // generate the image's key
    let key = ImageKey::from(&image);
    // create the notification
    shared::notifications::create_notification(image, key, req, params, &state.shared).await
}

/// Get the all of the image's notifications
///
/// # Arguments
///
/// * `user` - The user that is requesting the notifications
/// * `group` - The group the image is in
/// * `image` - The name of the image whose notifications are being requested
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/images/notifications/:group/:image",
    params(
        ("group" = String, Path, description = "The group the image is in"),
        ("image" = String, Path, description = "The name of the image whose notifications are being requested"),
    ),
    responses(
        //(status = 200, description = "Notifications returned for image", body = Vec<Notification<Image>>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::get_notifications", skip_all, err(Debug))]
async fn get_notifications(
    user: User,
    Path((group, image)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<Notification<Image>>>, ApiError> {
    // check that the image exists and the user has access
    let (_, image) = Image::get(&user, &group, &image, &state.shared).await?;
    // generate the image's key
    let key = ImageKey::from(&image);
    // get all of the image's notifications
    shared::notifications::get_notifications(image, key, &state.shared).await
}

/// Delete a specific notification from an image
///
/// # Arguments
///
/// * `user` - The user that is requesting the deletion
/// * `group` - The group the image is in
/// * `image` - The name of the image whose log is being deleted
/// * `id` - The notification's unique ID
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/images/notifications/:group/:image/:id",
    params(
        ("group" = String, Path, description = "The group the image is in"),
        ("image" = String, Path, description = "The name of the image whose log is being deleted"),
        ("id" = Uuid, Path, description = "The notification's unique ID"),
    ),
    responses(
        (status = 204, description = "Notification deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::images::delete_notification", skip_all, err(Debug))]
async fn delete_notification(
    user: User,
    Path((group, image, id)): Path<(String, String, Uuid)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // only admins can delete image logs
    is_admin!(&user);
    // check that the image exists
    let (_, image) = Image::get(&user, &group, &image, &state.shared).await?;
    // generate the image's key
    let key = ImageKey::from(&image);
    // delete the notification
    shared::notifications::delete_notification(image, key, None, id, &state.shared).await
}

/// Add the images routes to our router
/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(create, get_image, list, list_details, update, delete_image, runtimes_update, get_notifications, create_notification, delete_notification),
    components(schemas(ArgStrategy, AutoTag, AutoTagLogic, AutoTagUpdate, ChildFilters, ChildFiltersUpdate, ChildrenDependencySettings, ChildrenDependencySettingsUpdate, Cleanup, CleanupUpdate, ConfigMap, Dependencies, DependenciesUpdate, DependencyPassStrategy, DependencySettingsUpdate, EphemeralDependencySettings, EphemeralDependencySettingsUpdate, FilesHandler, FilesHandlerUpdate, GenericBan, HostPath, HostPathTypes, Image, ImageArgs, ImageArgsUpdate, ImageBan, ImageBanKind, ImageBanUpdate, ImageDetailsList, ImageLifetime, ImageList, ImageListParams, ImageNetworkPolicyUpdate, ImageRequest, ImageScaler, ImageUpdate, ImageVersion, InvalidHostPathBan, InvalidUrlBan, Kvm, KvmUpdate, KwargDependency, NFS, Notification<Image>, NotificationLevel, NotificationParams, NotificationRequest<Image>, OutputCollection, OutputCollectionUpdate, OutputDisplayType, OutputHandler, RepoDependencySettings, Resources, ResourcesRequest, ResourcesUpdate, ResultDependencySettings, ResultDependencySettingsUpdate, SampleDependencySettings, Secret, SecurityContext, SecurityContextUpdate, SpawnLimits, TagDependencySettings, TagDependencySettingsUpdate, Volume, VolumeTypes)),
    modifiers(&OpenApiSecurity),
)]
pub struct ImageApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(ImageApiDocs::openapi())
}

/// Add the results routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/images/", post(create))
        .route("/api/images/data/{group}/{image}", get(get_image))
        .route("/api/images/{group}/", get(list))
        .route("/api/images/{group}/details/", get(list_details))
        .route(
            "/api/images/{group}/{image}",
            patch(update).delete(delete_image),
        )
        .route("/api/images/runtimes/update", post(runtimes_update))
        .route(
            "/api/images/notifications/{group}/{image}",
            get(get_notifications).post(create_notification),
        )
        .route(
            "/api/images/notifications/{group}/{image}/{id}",
            delete(delete_notification),
        )
}
