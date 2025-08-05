use axum::extract::{Json, Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, patch, post};
use axum::Router;
use tracing::{instrument, span, Level};
use utoipa::OpenApi;

use super::OpenApiSecurity;
use crate::models::images::{GenericBan, InvalidHostPathBan, InvalidUrlBan};
use crate::models::pipelines::BannedImageBan;
use crate::models::{
    ActiveJob, ApiCursor, ArgStrategy, AutoTag, AutoTagLogic, Backup, ChildFilters,
    ChildFiltersUpdate, ChildrenDependencySettings, Cleanup, ConfigMap, Dependencies,
    DependencyPassStrategy, EphemeralDependencySettings, EventTrigger, FilesHandler, Group,
    GroupAllowed, GroupStats, GroupUsers, HostPath, HostPathTypes, HostPathWhitelistUpdate, Image,
    ImageArgs, ImageBan, ImageBanKind, ImageBanUpdate, ImageLifetime, ImageScaler, ImageVersion,
    Kvm, KwargDependency, Node, NodeGetParams, NodeHealth, NodeListLine, NodeListParams,
    NodeRegistration, NodeUpdate, OutputCollection, OutputDisplayType, OutputHandler, Pipeline,
    PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineStats, Pools, Reaction,
    RepoDependencySettings, Resources, ResultDependencySettings, SampleDependencySettings,
    ScalerStats, Secret, SecurityContext, SpawnLimits, StageStats, SystemInfo, SystemInfoParams,
    SystemSettings, SystemSettingsResetParams, SystemSettingsUpdate, SystemSettingsUpdateParams,
    SystemStats, TagDependencySettings, TagType, Theme, UnixInfo, User, UserRole, UserSettings,
    Volume, VolumeTypes, Worker, WorkerDelete, WorkerDeleteMap, WorkerRegistration,
    WorkerRegistrationList, WorkerStatus, WorkerUpdate, NFS,
};
use crate::utils::{ApiError, AppState};

/// Initializes the current backend's system info
///
/// # Arguments
///
/// * `user` - The user that is initializing Thorium
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/system/init",
    params(),
    responses(
        (status = 204, description = "System info initialized"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::init", skip_all, err(Debug))]
async fn init(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // init our system info
    SystemInfo::init(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Gets information on the current backend system and data
///
/// If reset is set then any flags tied to cache clearing events will be reset.
///
/// # Arguments
///
/// * `user` - The user that is getting system info
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/system/",
    params(
        ("params" = SystemInfoParams, Query, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "Backend system data", body = SystemInfo),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::info", skip_all, err(Debug))]
async fn info(
    user: User,
    Query(params): Query<SystemInfoParams>,
    State(state): State<AppState>,
) -> Result<Json<SystemInfo>, ApiError> {
    // get our system info
    let info = SystemInfo::get(&user, params.reset, &state.shared).await?;
    Ok(Json(info))
}

/// Gets statistics on Thorium
///
/// # Arguments
///
/// * `user` - The user that is getting system stats
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/system/stats",
    params(),
    responses(
        (status = 200, description = "Thorium statistics", body = SystemStats),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::stats", skip_all, err(Debug))]
async fn stats(user: User, State(state): State<AppState>) -> Result<Json<SystemStats>, ApiError> {
    // start our get system stats route span
    let system_stats = SystemStats::get(&user, &state.shared).await?;
    Ok(Json(system_stats))
}

/// Gets the current dynamic system settings
///
/// # Arguments
///
/// * `user` - The user that is getting system settings
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/system/settings",
    params(),
    responses(
        (status = 200, description = "Current dynamic system settings", body = SystemSettings),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::settings", skip_all, err(Debug))]
async fn settings(
    user: User,
    State(state): State<AppState>,
) -> Result<Json<SystemSettings>, ApiError> {
    // get system settings
    let settings = SystemSettings::get(&user, &state.shared).await?;
    Ok(Json(settings))
}

/// Reset the system settings to their defaults
///
/// # Arguments
///
/// * `user` - The user that is resetting system settings
/// * `state` - Shared Thorium objects
/// * `params` - The settings reset params
#[utoipa::path(
    patch,
    path = "/api/system/settings/reset",
    params(
        ("params" = SystemSettingsResetParams, Query, description = "The settings reset params"),
    ),
    responses(
        (status = 204, description = "System settings reset to default"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::settings_reset", skip_all, err(Debug))]
async fn settings_reset(
    user: User,
    State(state): State<AppState>,
    Query(params): Query<SystemSettingsResetParams>,
) -> Result<StatusCode, ApiError> {
    // determine if a scan is needed before resetting
    let scan_needed = if params.scan {
        // retrieve the current system settings
        let old_settings = SystemSettings::get(&user, &state.shared).await?;
        let default_settings = SystemSettings::default();
        // a scan is needed if either the whitelist or allow_unrestricted_host_paths was changed
        (old_settings.host_path_whitelist != default_settings.host_path_whitelist)
            || (old_settings.allow_unrestricted_host_paths
                != default_settings.allow_unrestricted_host_paths)
    } else {
        // a scan was not requested so no scan is necessary
        false
    };
    // reset system settings
    SystemSettings::reset(&user, &state.shared).await?;
    if scan_needed {
        // perform a scan with default settings if one is needed
        SystemSettings::default()
            .consistency_scan(&user, &state.shared)
            .await?;
        // reset the scaler's cache after the scan
        SystemInfo::reset_cache(&user, &state.shared).await?;
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Updates the system settings
///
/// # Arguments
///
/// * `user` - The user that is update system settings
/// * `state` - Shared Thorium objects
/// * `params` - The settings update params
/// * `update` - The update to apply to our system settings
#[utoipa::path(
    patch,
    path = "/api/system/settings",
    params(
        ("params" = SystemSettingsUpdateParams, Query, description = "The settings update params"),
        ("update" = SystemSettingsUpdate, description = "The update to apply to system settings"),
    ),
    responses(
        (status = 204, description = "System settings updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::settings_update", skip_all, err(Debug))]
async fn settings_update(
    user: User,
    State(state): State<AppState>,
    Query(params): Query<SystemSettingsUpdateParams>,
    Json(update): Json<SystemSettingsUpdate>,
) -> Result<StatusCode, ApiError> {
    // determine if a scan is needed before updating
    let scan_needed = if params.scan {
        let old_settings = SystemSettings::get(&user, &state.shared).await?;
        // has allow changed?
        let allow_changed = update
            .allow_unrestricted_host_paths
            .is_some_and(|allow| old_settings.allow_unrestricted_host_paths != allow);
        // has the whitelist changed?
        let whitelist_edited = !update.host_path_whitelist.add_paths.is_empty()
            || !update.host_path_whitelist.remove_paths.is_empty()
            || update.clear_host_path_whitelist;
        // is the whitelist active?
        let whitelist_active = if let Some(new_allow) = &update.allow_unrestricted_host_paths {
            !new_allow
        } else {
            !old_settings.allow_unrestricted_host_paths
        };
        // TODO: unit test!!
        match (allow_changed, whitelist_edited, whitelist_active) {
            // scan if the whitelist was edited and is active or if allow unrestricted host paths has changed
            (_, true, true) | (true, _, _) => true,
            // no changes were made, so don't scan
            _ => false,
        }
    } else {
        // a scan was not requested so no scan is necessary
        false
    };
    // get the current system settings
    let settings = SystemSettings::get(&user, &state.shared).await?;
    // update system settings
    let updated_settings = settings.update(update, &user, &state.shared).await?;
    // do a consistency scan if requested and necessary
    if scan_needed {
        // perform scan
        updated_settings
            .consistency_scan(&user, &state.shared)
            .await?;
        // reset the scaler's cache after the scan
        SystemInfo::reset_cache(&user, &state.shared).await?;
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Performs a scan of Thorium data, checking that all data is compliant with current [`SystemSettings`]
/// and cleaning/marking/modifying data that isn't; additionally signals the scaler to refresh its cache
///
/// Currently this only applies to images with host path mounts that may not be on the configured
/// whitelist after a settings update; also applies to the pipelines that have those images
///
/// # Arguments
///
/// * `user` - The user that is telling Thorium to scan
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/system/settings/scan",
    params(),
    responses(
        (status = 204, description = "System scanned and scaler cache reset"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::consistency_scan", skip_all, err(Debug))]
async fn consistency_scan(
    user: User,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // get the current system settings
    let settings = SystemSettings::get(&user, &state.shared).await?;
    // do a scan for consistency with current settings
    settings
        .consistency_scan(&user, &state.shared)
        .await
        .map_err(|err| {
            ApiError::new(
                err.code,
                Some(format!("An error occurred while scanning: {err}")),
            )
        })?;
    // reset the scaler's cache after the scan
    SystemInfo::reset_cache(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Cleans up things in Thorium
///
/// This will clean up data that cannot be handled by Redis or Scylla's expire/TTL functionality.
///
/// # Arguments
///
/// * `user` - The user that is telling Thorium to cleanup
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/system/cleanup",
    params(),
    responses(
        (status = 204, description = "System data cleaned"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::cleanup", skip_all, err(Debug))]
async fn cleanup(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // clean up any expired reactions from status lists
    Reaction::expire_lists(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Invalidates the scalers cache
///
/// This will set the cache invalidation flag for the scaler to see on its next check.
///
/// # Arguments
///
/// * `user` - The user that is telling the scaler to invalidate its cache
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/system/cache/reset",
    params(),
    responses(
        (status = 204, description = "Scalars cache invalidated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::reset_cache", skip_all, err(Debug))]
async fn reset_cache(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // Set the scaler cache to be invalid
    SystemInfo::reset_cache(&user, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// builds a backup of data in Thorium
///
/// This will backup all data except reactions as those are large.
///
/// # Arguments
///
/// * `user` - The user that is backing up data
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/system/backup",
    params(),
    responses(
        (status = 200, description = "Backup of Thorium data", body = Backup),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::backup", skip_all, err(Debug))]
async fn backup(user: User, State(state): State<AppState>) -> Result<Json<Backup>, ApiError> {
    // build a backup object
    let backup = Backup::new(&user, &state.shared).await?;
    Ok(Json(backup))
}

/// Restores a backup
///
/// This will erase redis destroying any left over data to prevent orphaned data from a past
/// instance.
///
/// # Arguments
///
/// * `user` - The user that is restoring a backup of Thorium
/// * `state` - Shared Thorium objects
/// * `backup` - The backup to restore
#[utoipa::path(
    post,
    path = "/api/system/restore",
    params(
        ("backup" = Backup, description = "The backup to restore"),
    ),
    responses(
        (status = 204, description = "Backup restored; redis erased"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::restore", skip_all, err(Debug))]
async fn restore(
    user: User,
    State(state): State<AppState>,
    Json(backup): Json<Backup>,
) -> Result<StatusCode, ApiError> {
    // start our system restore route span
    let span = span!(Level::INFO, "System Restore Route");
    // restore from backup object
    backup.restore(&user, &state.shared, &span).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Register a new bare metal/windows node in Thorium
///
/// # Arguments
///
/// * `state` - Shared Thorium objects
/// * `registration` - This nodes registration info
#[utoipa::path(
    post,
    path = "/api/system/nodes/",
    params(
        ("backup" = NodeRegistration, description = "This nodes registration info"),
    ),
    responses(
        (status = 201, description = "New bare metal/windows node registered"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::get_register", skip_all, err(Debug))]
async fn register_node(
    user: User,
    State(state): State<AppState>,
    Json(registration): Json<NodeRegistration>,
) -> Result<StatusCode, ApiError> {
    // register this node if its not already registered
    Node::register(&user, &registration, &state.shared).await?;
    Ok(StatusCode::CREATED)
}

/// Gets a nodes info
///
/// # Arguments
///
/// * `user` - The user that is getting this nodes info
/// * `cluster` - The cluster this node is in
/// * `node` - The node this heart beat is from
/// * `state` - Shared Thorium objects
/// * `heartbeat` - The heart beat info for this node
#[utoipa::path(
    get,
    path = "/api/system/nodes/:cluster/:node",
    params(
        ("cluster" = String, Path, description = "The cluster this node is in"),
        ("node" = String, Path, description = "The node this heart beat is from"),
        ("params" = NodeGetParams, description = "The parameters for this node get request"),
    ),
    responses(
        (status = 200, description = "Node info", body = Node),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::get_node", skip_all, err(Debug))]
async fn get_node(
    user: User,
    params: NodeGetParams,
    Path((cluster, node)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Node>, ApiError> {
    // get this nodes info
    let node = Node::get(&user, &cluster, &node, params, &state.shared).await?;
    Ok(Json(node))
}

/// Updates a nodes info
///
/// # Arguments
///
/// * `user` - The user that is sending this heart beat
/// * `cluster` - The cluster this node is in
/// * `node` - The node this heart beat is from
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply to this nodes info
#[utoipa::path(
    patch,
    path = "/api/system/nodes/:cluster/:node",
    params(
        ("cluster" = String, Path, description = "The cluster this node is in"),
        ("node" = String, Path, description = "The node this heart beat is from"),
        ("update" = NodeUpdate, description = "The update to apply to this nodes info"),
    ),
    responses(
        (status = 201, description = "Node info updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::update_node", skip_all, err(Debug))]
async fn update_node(
    user: User,
    Path((cluster, node)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(update): Json<NodeUpdate>,
) -> Result<StatusCode, ApiError> {
    // user default node get params
    let params = NodeGetParams::default();
    // get this nodes info
    let node = Node::get(&user, &cluster, &node, params, &state.shared).await?;
    // apply this heart beat to our nodes info
    node.update(&update, &state.shared).await?;
    Ok(StatusCode::CREATED)
}

/// Lists node names
///
/// # Arguments
///
/// * `user` - The user that is getting this nodes info
/// * `params` - The params to use when listing node names
/// * `state` - Shared Thorium objects
/// * `heartbeat` - The heart beat info for this node
#[utoipa::path(
    get,
    path = "/api/system/nodes/",
    params(
        ("params" = NodeListParams, description = "The params to use when listing node names"),
    ),
    responses(
        (status = 200, description = "Nodes list", body = ApiCursor<NodeListLine>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::list_nodes", skip_all, err(Debug))]
async fn list_nodes(
    user: User,
    params: NodeListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<NodeListLine>>, ApiError> {
    // get a page of node names
    let cursor = Node::list(&user, params, &state.shared).await?;
    Ok(Json(cursor))
}

/// Lists node details
///
/// # Arguments
///
/// * `user` - The user that is getting this nodes info
/// * `params` - The params to use when listing node details
/// * `state` - Shared Thorium objects
/// * `heartbeat` - The heart beat info for this node
#[utoipa::path(
    get,
    path = "/api/system/nodes/details/",
    params(
        ("params" = NodeListParams, description = "The params to use when listing node details"),
    ),
    responses(
        (status = 200, description = "Nodes list", body = ApiCursor<Node>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::list_node_details", skip_all, err(Debug))]
async fn list_node_details(
    user: User,
    params: NodeListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<Node>>, ApiError> {
    // get a page of node names
    let cursor = Node::list_details(&user, params, &state.shared).await?;
    Ok(Json(cursor))
}

/// Registers a new worker within Thorium
///
/// # Arguments
///
/// * `user` - The user that is registering a new worker
/// * `scaler` - The scaler this worker is under
/// * `state` - Shared Thorium objects
/// * `worker` - The workers to register
#[utoipa::path(
    post,
    path = "/api/system/worker/:scaler_or_name",
    params(
        ("scaler" = ImageScaler, Path, description = "The scaler this worker is under"),
        ("workers" = WorkerRegistrationList, description = "The workers to register"),
    ),
    responses(
        (status = 204, description = "Node registered"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::register_worker", skip_all, err(Debug))]
async fn register_worker(
    user: User,
    Path(scaler): Path<ImageScaler>,
    State(state): State<AppState>,
    Json(workers): Json<WorkerRegistrationList>,
) -> Result<StatusCode, ApiError> {
    // add this new worker to our workers table
    workers.register(&user, scaler, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get info on a specific worker in Thorium
///
/// # Arguments
///
/// * `user` - The user that is registering a new worker
/// * `scaler` - The scaler this worker is under
/// * `cluster` - The cluster this worker is in
/// * `node` - The node is worker on
/// * `name` - The name of this worker
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/system/worker/:scaler_or_name",
    params(
        ("name" = String, Path, description = "The name of this worker"),
    ),
    responses(
        (status = 200, description = "Worker info", body = Worker),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::get_worker", skip_all, err(Debug))]
async fn get_worker(
    user: User,
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Worker>, ApiError> {
    // get this worker
    let worker = Worker::get(&user, &name, &state.shared).await?;
    Ok(Json(worker))
}

/// Updates a workers status within Thorium
///
/// # Arguments
///
/// * `user` - The user that is registering a new worker
/// * `scaler` - The scaler this worker is under
/// * `status` - The new status of the worker to set
/// * `state` - Shared Thorium objects
/// * `worker` - The worker to update
#[utoipa::path(
    patch,
    path = "/api/system/worker/:scaler_or_name",
    params(
        ("name" = String, Path, description = "The name of this worker"),
        ("update" = WorkerUpdate, description = "The name of this worker"),
    ),
    responses(
        (status = 204, description = "Worker updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::update_worker", skip_all, err(Debug))]
async fn update_worker(
    user: User,
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(update): Json<WorkerUpdate>,
) -> Result<StatusCode, ApiError> {
    // get this worker from scylla
    let worker = Worker::get(&user, &name, &state.shared).await?;
    // add this new worker to our workers table
    worker.update(&user, &update, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Removes a no longer active worker from Thorium
///
/// # Arguments
///
/// * `user` - The user that is removing an old worker
/// * `scaler` - The scaler this worker was under
/// * `state` - Shared Thorium objects
/// * `worker` - The worker to remove
#[utoipa::path(
    delete,
    path = "/api/system/worker/:scaler_or_name",
    params(
        ("scaler" = ImageScaler, Path, description = "The scaler this worker was under"),
        ("deletes" = WorkerDeleteMap, Path, description = "The worker to remove"),
    ),
    responses(
        (status = 204, description = "Worker(s) deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::system::delete_workers", skip_all, err(Debug))]
async fn delete_workers(
    user: User,
    Path(scaler): Path<ImageScaler>,
    State(state): State<AppState>,
    Json(deletes): Json<WorkerDeleteMap>,
) -> Result<StatusCode, ApiError> {
    // remove this new worker from our workers table
    deletes.delete(&user, scaler, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(init, info, stats, settings, settings_update, consistency_scan, settings_reset, cleanup, reset_cache, backup, restore, register_node, list_nodes, list_node_details, get_node, update_node, register_worker, delete_workers, get_worker, update_worker),
    components(schemas(ActiveJob, ApiCursor<NodeListLine>, ArgStrategy, AutoTag, AutoTagLogic, Backup, BannedImageBan, ChildFilters, ChildFiltersUpdate, ChildrenDependencySettings, Cleanup, ConfigMap, Dependencies, DependencyPassStrategy, EphemeralDependencySettings, EventTrigger, FilesHandler, GenericBan, Group, GroupAllowed, GroupStats, GroupUsers, HostPath, HostPathTypes, HostPathWhitelistUpdate, Image, ImageArgs, ImageBan, ImageBanKind, ImageBanUpdate, ImageLifetime, ImageScaler, ImageVersion, InvalidHostPathBan, InvalidUrlBan, Kvm, KwargDependency, NFS, Node, NodeGetParams, NodeHealth, NodeListLine, NodeListParams, NodeRegistration, NodeUpdate, OutputCollection, OutputDisplayType, OutputHandler, Pipeline, PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineStats, Pools, RepoDependencySettings, Resources, ResultDependencySettings, SampleDependencySettings, ScalerStats, Secret, SecurityContext, SpawnLimits, StageStats, SystemInfo, SystemInfoParams, SystemSettings, SystemSettingsUpdate, SystemSettingsResetParams, SystemSettingsUpdateParams, SystemStats, TagDependencySettings, TagType, Theme, UnixInfo, User, UserRole, UserSettings, Volume, VolumeTypes, Worker, WorkerDeleteMap, WorkerDelete, WorkerRegistration, WorkerRegistrationList, WorkerStatus, WorkerUpdate)),
    modifiers(&OpenApiSecurity),
)]
pub struct SystemApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(SystemApiDocs::openapi())
}

/// Add the system routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/system/init", post(init))
        .route("/api/system/", get(info))
        .route("/api/system/stats", get(stats))
        .route("/api/system/settings", get(settings).patch(settings_update))
        .route("/api/system/settings/scan", post(consistency_scan))
        .route("/api/system/settings/reset", patch(settings_reset))
        .route("/api/system/cleanup", post(cleanup))
        .route("/api/system/cache/reset", post(reset_cache))
        .route("/api/system/backup", get(backup))
        .route("/api/system/restore", post(restore))
        .route("/api/system/nodes/", post(register_node).get(list_nodes))
        .route("/api/system/nodes/details/", get(list_node_details))
        .route(
            "/api/system/nodes/{cluster}/{node}",
            get(get_node).patch(update_node),
        )
        .route(
            "/api/system/worker/{scaler_or_name}",
            post(register_worker)
                .delete(delete_workers)
                .get(get_worker)
                .patch(update_worker),
        )
}
