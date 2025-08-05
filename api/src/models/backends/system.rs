use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use futures::stream::{self, StreamExt};
use futures::TryStreamExt;
use itertools::Itertools;
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tracing::{instrument, span, Level, Span};
use uuid::Uuid;

use super::db::{self, SimpleCursorExt};
use crate::models::backends::NotificationSupport;
use crate::models::{
    conversions, ApiCursor, Backup, Group, GroupRequest, GroupUsersRequest, HostPath,
    HostPathWhitelistUpdate, Image, ImageBan, ImageBanKind, ImageBanUpdate, ImageKey, ImageScaler,
    Node, NodeGetParams, NodeListLine, NodeListParams, NodeRegistration, NodeRow, NodeUpdate,
    Pipeline, PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineKey, SystemInfo,
    SystemSettings, SystemSettingsUpdate, SystemStats, User, VolumeTypes, Worker, WorkerDeleteMap,
    WorkerRegistrationList, WorkerUpdate,
};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, deserialize, deserialize_ext, extract, is_admin, log_scylla_err, not_found, unauthorized,
    update,
};

/// Check if Thorium is healthy
///
/// Currently this just checks redis but we should add scylla too.
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn health(shared: &Shared) -> Result<bool, ApiError> {
    db::system::health(shared).await
}

/// Returns a string denoting this server as a Thorium server
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn iff(shared: &Shared) -> Result<String, ApiError> {
    // get the system IFF string from the backend
    db::system::iff(shared).await
}

impl SystemInfo {
    /// Initialize the system info data with default values
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is Initializing system info
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "SystemInfo::init", skip_all, err(Debug))]
    pub async fn init(user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can get system info
        is_admin!(user);
        // get the system info from the backend
        db::system::init(shared).await?;
        // check if the system group already exists
        if !db::groups::exists(&["system".to_owned()], shared).await? {
            // the system group doesn't already exist so create a request to create it
            let req = GroupRequest::new("system")
                .owners(GroupUsersRequest::default().direct(&user.username));
            // save our group to the backend
            Group::create(user, req, shared).await?;
        }
        Ok(())
    }

    /// Gets system info on the data in the current backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting system info
    /// * `clear` - Whether to reset the flags denoting things like a stale cache
    /// * `shared` - Shared Thorium objects
    pub async fn get(
        user: &User,
        scaler: Option<ImageScaler>,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // only admins can get system info
        is_admin!(user);
        // get the system info from the backend
        db::system::get_info(scaler, shared).await
    }

    /// Sets the scaler cache reset flag
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting system info
    /// * `clear` - Whether to reset the flags denoting things like a stale cache
    /// * `shared` - Shared Thorium objects
    pub async fn reset_cache(user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can get system info
        is_admin!(user);
        // get the system info from the backend
        db::system::reset_cache(shared).await
    }
}

impl SystemStats {
    /// Gets statistics on the data in the current backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting system stats
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SystemStats::get", skip_all, err(Debug))]
    pub async fn get(user: &User, shared: &Shared) -> Result<Self, ApiError> {
        // build a map of groups status objects
        let mut group_status = HashMap::default();
        // crawl all groups visible to this user
        // TODO we need to make this a cursor instead of a hardcoded max limit
        for group in Group::list_details(user, 0, 5000, shared).await?.details {
            // get the status of this group
            let status = group.stats(0, 5000, shared).await?;
            // add it to our map of statuses
            group_status.insert(group.name, status);
        }
        // get the system statistics from the backend
        db::system::get_stats(group_status, shared).await
    }
}

impl HostPathWhitelistUpdate {
    /// Update the [`SystemSettings`] with the contents of this host path whitelist update
    ///
    /// # Arguments
    ///
    /// * `settings` - The settings to update
    pub fn update(self, settings: &mut SystemSettings) -> Result<(), ApiError> {
        // validate the host path whitelist update
        self.validate(settings)?;
        // add the requested paths
        for path in self.add_paths {
            settings.host_path_whitelist.insert(path);
        }
        // remove the requested paths
        for path in self.remove_paths {
            settings.host_path_whitelist.remove(&path);
        }
        Ok(())
    }

    /// Validate a `HostPathWhitelistUpdate` before updating
    ///
    /// # Arguments
    ///
    /// * `settings` - The current system settings we're validating against
    fn validate(&self, settings: &SystemSettings) -> Result<(), ApiError> {
        // make sure all of the paths to add to the whitelist are valid host paths
        let bad_add_host_paths: Vec<String> = self
            .add_paths
            .iter()
            .filter_map(|path| {
                (!HostPath::is_valid(path)).then_some(path.to_string_lossy().into_owned())
            })
            .collect();
        if !bad_add_host_paths.is_empty() {
            return bad!(format!(
                "Invalid host path(s) '{bad_add_host_paths:?}'! Host paths must be absolute \
                    and not contain any relative traversal ('.', '..', etc.)"
            ));
        }
        // make sure all of the paths to remove from the whitelist are IN the whitelist
        let bad_remove_host_paths: Vec<String> = self
            .remove_paths
            .iter()
            .filter_map(|path| {
                (!settings.host_path_whitelist.contains(path))
                    .then_some(path.to_string_lossy().into_owned())
            })
            .collect();
        if !bad_remove_host_paths.is_empty() {
            return bad!(format!(
                "Invalid host path(s) '{bad_remove_host_paths:?}'! The requested host paths to remove are \
                not in the current host path whitelist"
            ));
        }
        Ok(())
    }
}

impl SystemSettings {
    /// Reset dynamic Thorium settings to their defaults
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is resetting system settings
    /// * `shared` - Shared Thorium objects
    pub async fn reset(user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can reset system settings
        is_admin!(user);
        // reset the system settings in the backend
        db::system::reset_settings(shared).await
    }

    /// Get the currently configured system settings
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting system settings
    /// * `shared` - Shared Thorium objects
    pub async fn get(user: &User, shared: &Shared) -> Result<Self, ApiError> {
        // only admins can get system settings
        is_admin!(user);
        // get the system settings from the backend
        db::system::get_settings(shared).await
    }

    /// Update the current [`SystemSettings`]
    ///
    /// # Arguments
    ///
    /// * `settings` - The `SystemSettings` to update
    /// * `user` - The user that is getting system info
    /// * `shared` - Shared Thorium objects
    #[rustfmt::skip]
    pub async fn update(
        mut self,
        update: SystemSettingsUpdate,
        user: &User,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // only admins can update system settings
        is_admin!(user);
        // update the settings
        update!(self.reserved_cpu, update.reserved_cpu, conversions::cpu);
        update!(self.reserved_memory, update.reserved_memory, conversions::storage);
        update!(self.reserved_storage, update.reserved_storage, conversions::storage);
        update!(self.fairshare_cpu, update.fairshare_cpu, conversions::cpu);
        update!(self.fairshare_memory, update.fairshare_memory, conversions::storage);
        update!(self.fairshare_storage, update.fairshare_storage, conversions::storage);
        // update the host path whitelist
        update.host_path_whitelist.update(&mut self)?;
        // update the unrestricted host paths setting
        update!(
            self.allow_unrestricted_host_paths,
            update.allow_unrestricted_host_paths
        );
        // clear the whitelist if we're set to
        if update.clear_host_path_whitelist {
            self.host_path_whitelist.clear();
        }
        // update the system settings in the backend
        db::system::update_settings(&self, shared).await?;
        Ok(self)
    }

    /// A helper function for checking images in the consistency scan
    ///
    /// # Arguments
    ///
    /// * `image` - The image we're checking for consistency
    /// * `user` - The user who is running the scan
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SystemSettings::scan_helper_images", skip_all, err(Debug))]
    async fn scan_helper_images(
        &self,
        mut image: Image,
        user: &User,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the list of bans to add and remove
        let (bans_added, bans_removed) = if self.allow_unrestricted_host_paths {
            // add no bans and remove all host path related bans if we allow all host paths
            let added: Vec<ImageBan> = Vec::new();
            let removed: Vec<Uuid> = image
                .bans
                .iter()
                .filter_map(|(ban_id, ban)| match &ban.ban_kind {
                    ImageBanKind::InvalidHostPath(_) => Some(*ban_id),
                    _ => None,
                })
                .collect();
            (added, removed)
        } else {
            // create a set of already banned paths so we don't accidentally add the same ban twice
            let banned_paths: HashSet<&PathBuf> = image
                .bans
                .values()
                .filter_map(|ban| match &ban.ban_kind {
                    ImageBanKind::InvalidHostPath(ban_type) => Some(&ban_type.host_path),
                    _ => None,
                })
                .collect();
            // find any volumes that have bad host paths and add a ban for each one
            let added: Vec<ImageBan> = image
                .volumes
                .iter()
                .filter_map(|vol| match vol.archetype {
                    VolumeTypes::HostPath => {
                        // attempt to get the host path or skip if there isn't one set
                        let host_path = &vol.host_path.as_ref()?.path;
                        if self.is_whitelisted_host_path(host_path) {
                            // skip if the host path is whitelisted
                            None
                        } else {
                            let host_path = PathBuf::from(host_path);
                            if banned_paths.contains(&host_path) {
                                // if the path is already banned, don't add a new one
                                None
                            } else {
                                // otherwise create a new ban
                                Some(ImageBan::new(ImageBanKind::host_path(&vol.name, host_path)))
                            }
                        }
                    }
                    _ => None,
                })
                .collect();
            // find any bans with host paths that are now whitelisted
            let removed: Vec<Uuid> = image
                .bans
                .iter()
                .filter_map(|(ban_id, ban)| {
                    match &ban.ban_kind {
                        ImageBanKind::InvalidHostPath(ban) => {
                            if self.is_whitelisted_host_path(&ban.host_path) {
                                // if the ban's host path is now whitelisted, set it to be removed
                                Some(*ban_id)
                            } else {
                                None
                            }
                        }
                        // skip if the ban isn't host path related
                        _ => None,
                    }
                })
                .collect();
            (added, removed)
        };
        // if we have any bans to add/remove, update the image with those bans
        if !bans_added.is_empty() || !bans_removed.is_empty() {
            // create a ban update with the new bans
            let ban_update = ImageBanUpdate::default()
                .add_bans(bans_added.clone())
                .remove_bans(bans_removed.clone());
            // add the bans to the image
            ban_update.update(&mut image, user)?;
            db::images::update(&image, shared).await?;
            // update ban notifications
            image
                .update_ban_notifications(
                    &ImageKey::from(&image),
                    &bans_added,
                    &bans_removed,
                    shared,
                )
                .await?;
        }
        Ok(())
    }

    /// A helper function for checking pipelines in the consistency scan
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The pipeline we're checking for consistency
    /// * `user` - The user who is running the scan
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "SystemSettings::scan_helper_pipelines", skip_all, err(Debug))]
    async fn scan_helper_pipelines(
        &self,
        mut pipeline: Pipeline,
        user: &User,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get a list of the pipeline's images
        let images = pipeline.order.iter().flatten().unique();
        // get a map of images that already have a ban in the pipeline and their ban IDs
        let current_bans_images = pipeline
            .bans
            .values()
            .filter_map(|ban| match &ban.ban_kind {
                PipelineBanKind::BannedImage(ban_kind) => Some((&ban_kind.image, &ban.id)),
                _ => None,
            })
            .collect::<HashMap<&String, &Uuid>>();
        // calculate which pipeline bans to add/remove
        let (bans_added, bans_removed) = stream::iter(images)
            .map(Ok::<&std::string::String, ApiError>)
            .try_fold(
                (Vec::new(), Vec::new()),
                |(mut bans_added, mut bans_removed), image| {
                    let group = &pipeline.group;
                    let current_bans_images = &current_bans_images;
                    async move {
                        // get the image's bans
                        let bans = db::images::get_bans(group, image, shared).await?;
                        // determine if the image is banned
                        let image_banned = !bans.is_empty();
                        // determine if the image already has a ban in the pipeline's ban list
                        let image_pipeline_ban = current_bans_images.get(image);
                        match (image_banned, image_pipeline_ban) {
                            // the image is banned and the pipeline does not have it, so add a new ban
                            (true, None) => {
                                bans_added
                                    .push(PipelineBan::new(PipelineBanKind::image_ban(image)));
                            }
                            // the image is no longer banned but the pipeline still has it as a ban, so remove it
                            (false, Some(pipeline_ban)) => bans_removed.push(**pipeline_ban),
                            // either the image is banned and already in the pipeline bans or it isn't and it's not in the
                            // pipeline, so do nothing
                            _ => (),
                        }
                        Ok((bans_added, bans_removed))
                    }
                },
            )
            .await?;
        // update the pipeline's bans
        let ban_update = PipelineBanUpdate::default()
            .add_bans(bans_added.clone())
            .remove_bans(bans_removed.clone());
        ban_update.update(&mut pipeline, user)?;
        db::pipelines::update(&pipeline, &[], &[], shared).await?;
        // update the pipeline's ban notifications
        pipeline
            .update_ban_notifications(
                &PipelineKey::from(&pipeline),
                &bans_added,
                &bans_removed,
                shared,
            )
            .await?;
        Ok(())
    }

    /// Performs a scan of Thorium data, checking that all data is compliant with the settings in `self`
    /// and cleaning/marking/modifying data that isn't
    ///
    /// Currently this only applies to images with host path mounts that may not be on the configured
    /// whitelist after a settings update, as well as to pipelines those images may be a part of
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is telling Thorium to scan
    #[instrument(name = "SystemSettings::consistency_scan", skip_all, err(Debug))]
    pub async fn consistency_scan(&self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can perform a scan
        is_admin!(user);
        // update all images
        stream::iter(db::images::backup(shared).await?)
            .map(Ok)
            .try_for_each_concurrent(1000, |image| self.scan_helper_images(image, user, shared))
            .await?;
        // update all pipelines
        stream::iter(db::pipelines::backup(shared).await?)
            .map(Ok)
            .try_for_each_concurrent(1000, |pipeline| {
                self.scan_helper_pipelines(pipeline, user, shared)
            })
            .await?;
        Ok(())
    }

    /// Checks if the given path or any of its parents are in the whitelist
    ///
    /// # Arguments
    ///
    /// * `path` - The path to check if it or any of its parents are on the whitelist
    pub fn is_whitelisted_host_path<T: Into<PathBuf>>(&self, path: T) -> bool {
        // convert to a PathBuf
        let path: PathBuf = path.into();
        if self.host_path_whitelist.contains(&path) {
            // return true if this path is in the whitelist
            true
        } else if let Some(parent) = path.parent() {
            // recursively check the parent directory
            self.is_whitelisted_host_path(parent)
        } else {
            // return false if we're out of parents to check
            false
        }
    }
}

impl Backup {
    /// Gets a backup of the current server
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting a system backup
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Backup::new", skip_all, err(Debug))]
    pub async fn new(user: &User, shared: &Shared) -> Result<Self, ApiError> {
        // only admins can backup data
        is_admin!(user);
        // get the current system settings
        let settings = SystemSettings::get(user, shared).await?;
        // get lists of users/groups/images/pipelines to backup
        let users = db::users::backup(shared).await?;
        let groups = db::groups::backup(shared).await?;
        let images = db::images::backup(shared).await?;
        let pipelines = db::pipelines::backup(shared).await?;
        let backup = Backup {
            settings,
            users,
            groups,
            images,
            pipelines,
        };
        Ok(backup)
    }

    /// Wipes the backend and restores a backup
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is restoring Thorium data
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    pub async fn restore(&self, user: &User, shared: &Shared, span: &Span) -> Result<(), ApiError> {
        // start our system restore span
        let span = span!(
            parent: span,
            Level::INFO,
            "System Restore",
            user = user.username
        );
        // only admins can restore data
        is_admin!(user);
        // wipe backends now
        db::system::wipe(shared).await?;
        // initialize the system and restore settings
        db::system::init(shared).await?;
        db::system::restore_settings(&self.settings, shared).await?;
        // restore user/group/pipeline data
        db::users::restore(&self.users, shared).await?;
        db::groups::restore(&self.groups, shared, &span).await?;
        db::images::restore(&self.images, shared).await?;
        db::pipelines::restore(&self.pipelines, shared).await?;
        Ok(())
    }
}

impl Node {
    /// Register a new node
    ///
    /// This will not reregister existing nodes
    ///
    /// # Arguments
    ///
    /// * `node` - The node registration info
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "Node::register", skip_all, fields(cluster = node.cluster, node = node.name), err(Debug))]
    pub async fn register(
        user: &User,
        node: &NodeRegistration,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // only admins can register nodes
        is_admin!(user);
        // TODO check if node is already registered
        db::system::register(node, shared).await
    }

    /// Gets a nodes info
    ///
    /// # Arguments
    ///
    /// * `user` - The user getting a nodes info
    /// * `cluster` - The cluster this node is in
    /// * `node` - The name of this node
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Node::get", skip_all, err(Debug))]
    pub async fn get(
        user: &User,
        cluster: &str,
        node: &str,
        mut params: NodeGetParams,
        shared: &Shared,
    ) -> Result<Node, ApiError> {
        // only admins can get node info
        is_admin!(user);
        // if no scalers were added then expand them to all
        params.default_scalers();
        // get this nodes info
        db::system::get_node(cluster, node, &params, shared).await
    }

    /// Updates a nodes info
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this node
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "Node::update", skip(self, shared), fields(cluster = self.cluster, node = self.name), err(Debug))]
    pub async fn update(&self, update: &NodeUpdate, shared: &Shared) -> Result<(), ApiError> {
        // update this nodes info
        db::system::update_node(self, update, shared).await
    }

    /// Lists node names
    ///
    /// # Arguments
    ///
    /// * `_user` - The user that is listing node names (kept to ensure auth)
    /// * `params` - The params for listing node names
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Node::list", skip_all, err(Debug))]
    pub async fn list(
        _user: &User,
        mut params: NodeListParams,
        shared: &Shared,
    ) -> Result<ApiCursor<NodeListLine>, ApiError> {
        // if we don't have any clusters defined then add all known clusters
        params.default_clusters(shared);
        if params.nodes.is_empty() {
            // if we have no nodes specified, get a chunk of the node names list
            let scylla_cursor = db::system::list_nodes(params, shared).await?;
            // convert our scylla cursor to a user facing one
            Ok(ApiCursor::from(scylla_cursor))
        } else {
            // if we have nodes specified then just get info on those
            let rows = db::system::get_node_rows(&params.clusters, &params.nodes, shared).await?;
            // turn our rows into node list lines
            let nodes = rows
                .into_iter()
                .map(|row| NodeListLine::from(row))
                .collect();
            // turn our nodes into a cursor object
            let cursor = ApiCursor {
                cursor: None,
                data: nodes,
            };
            Ok(cursor)
        }
    }

    /// Lists node details
    ///
    /// # Arguments
    ///
    /// * `_user` - The user that is listing node details (kept to ensure auth)
    /// * `params` - The params for listing nodes
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Node::list_details", skip_all, err(Debug))]
    pub async fn list_details(
        _user: &User,
        mut params: NodeListParams,
        shared: &Shared,
    ) -> Result<ApiCursor<Node>, ApiError> {
        // if we don't have any clusters defined then add all known clusters
        params.default_clusters(shared);
        if params.nodes.is_empty() {
            // if we don't have nodes, get a chunk of the node details list
            db::system::list_node_details(params, shared).await
        } else {
            // if we have nodes specified then just get info on those
            let rows = db::system::get_node_rows(&params.clusters, &params.nodes, shared).await?;
            // create an empty user facing cursor to store this cursors data
            let mut cursor = ApiCursor::empty(rows.len());
            // get all of our nodes workers 25 at a time
            let nodes = stream::iter(rows)
                .map(|row| db::system::get_workers_for_node(row, &params.scalers, shared))
                .buffered(25)
                .collect::<Vec<Result<Node, ApiError>>>()
                .await
                .into_iter()
                .filter_map(|res| log_scylla_err!(res));
            // add this nodes data
            cursor.data.extend(nodes);
            Ok(cursor)
        }
    }
}

impl TryFrom<NodeRow> for Node {
    type Error = ApiError;
    /// Try to convert a [`NodeRow`] into a [`Node`]
    ///
    /// # Arguments
    ///
    /// * `row` - The row to convert
    fn try_from(row: NodeRow) -> Result<Node, Self::Error> {
        // deserialize our health
        let health = deserialize!(&row.health);
        // build our resources objects
        let resources = deserialize!(&row.resources);
        // build our node struct
        let node = Node {
            cluster: row.cluster,
            name: row.node,
            health,
            resources,
            workers: HashMap::default(),
            heart_beat: row.heart_beat,
        };
        Ok(node)
    }
}

#[async_trait::async_trait]
impl SimpleCursorExt for NodeListLine {
    /// Query scylla for the next page of data for this simple cursor
    ///
    /// # Arguments
    ///
    /// * `partition` - The partition to query data for
    /// * `tie` - The cluster key to use when breaking ties
    /// * `limit` - The max amount of data to retrieve at once
    /// * `shared` - Shared Thorium objects
    async fn query(
        partition: &str,
        tie: &Option<String>,
        limit: usize,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError> {
        // if we have a node tie then query for the next page of data from our tie
        match tie {
            // we have a node name to skip too
            Some(node) => {
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.nodes.list_ties,
                        (partition, node, limit as i32),
                    )
                    .await
            }
            // we do not have a node name to skip too
            None => {
                shared
                    .scylla
                    .session
                    .execute_unpaged(&shared.scylla.prep.nodes.list, (partition, limit as i32))
                    .await
            }
        }
    }

    /// Gets the cluster key to start the next page of data after
    fn get_tie(&self) -> Option<String> {
        Some(self.node.clone())
    }
}

#[async_trait::async_trait]
impl SimpleCursorExt for NodeRow {
    /// Query scylla for the next page of data for this simple cursor
    ///
    /// # Arguments
    ///
    /// * `partition` - The partition to query data for
    /// * `tie` - The cluster key to use when breaking ties
    /// * `limit` - The max amount of data to retrieve at once
    /// * `shared` - Shared Thorium objects
    async fn query(
        partition: &str,
        tie: &Option<String>,
        limit: usize,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError> {
        // if we have a node tie then query for the next page of data from our tie
        match tie {
            // we have a node name to skip too
            Some(node) => {
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.nodes.list_details_ties,
                        (partition, node, limit as i32),
                    )
                    .await
            }
            // we do not have a node name to skip too
            None => {
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.nodes.list_details,
                        (partition, limit as i32),
                    )
                    .await
            }
        }
    }

    /// Gets the cluster key to start the next page of data after
    fn get_tie(&self) -> Option<String> {
        Some(self.node.clone())
    }
}

impl<S> FromRequestParts<S> for NodeGetParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl<S> FromRequestParts<S> for NodeListParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl NodeListParams {
    /// Expand our params to sane defaults if they came in empty
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub fn default_expand(&mut self, shared: &Shared) {
        // expand our clusters
        self.default_clusters(shared);
        // expand our scalers
        self.default_scalers();
    }

    /// if we don't have any clusters then use all the ones we know about
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub fn default_clusters(&mut self, shared: &Shared) {
        // if we dont have clusters defined then add in all known ones
        if self.clusters.is_empty() {
            // get our k8s cluster names or aliased names
            for (name, cluster) in &shared.config.thorium.scaler.k8s.clusters {
                // use either our cluster name or our alias
                match &cluster.alias {
                    Some(alias) => self.clusters.push(alias.clone()),
                    None => self.clusters.push(name.clone()),
                };
            }
            // get our bare metal cluster names
            let names = shared.config.thorium.scaler.bare_metal.clusters.keys();
            // add in our bare metal clusters
            self.clusters.extend(names.cloned());
            // get our windows cluster names
            let names = shared.config.thorium.scaler.windows.clusters.iter();
            // add in our windows clusters
            self.clusters.extend(names.cloned());
        }
    }

    /// If we don't have any scalers then use all the ones we know about
    pub fn default_scalers(&mut self) {
        // if we don't have any scalers set then add all of them
        if self.scalers.is_empty() {
            // add all of our scalers
            self.scalers = vec![
                ImageScaler::K8s,
                ImageScaler::BareMetal,
                ImageScaler::Windows,
                ImageScaler::External,
            ];
        }
    }
}

impl NodeGetParams {
    /// If we don't have any scalers then use all the ones we know about
    pub fn default_scalers(&mut self) {
        // if we don't have any scalers set then add all of them
        if self.scalers.is_empty() {
            // add all of our scalers
            self.scalers = vec![
                ImageScaler::K8s,
                ImageScaler::BareMetal,
                ImageScaler::Windows,
                ImageScaler::External,
            ];
        }
    }
}

impl Worker {
    /// Gets info on a specific worker
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting this workers info
    /// * `name` - The name of the worker to get
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Worker::get", skip(user, shared), err(Debug))]
    pub async fn get(user: &User, name: &str, shared: &Shared) -> Result<Worker, ApiError> {
        // get this worker
        let worker = db::system::get_worker(name, shared).await?;
        // make sure this user can see this worker
        if !user.is_admin() && !user.groups.contains(&worker.group) {
            not_found!(format!("Worker {} does not exist", name))
        } else {
            Ok(worker)
        }
    }

    /// Updates a worker's status in Scylla
    ///
    /// # Arguments
    ///
    /// * `_` - The user that is updating this workers status
    /// * `scaler` - The scaler this worker is under
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Worker::update", skip_all, err(Debug))]
    pub async fn update(
        &self,
        user: &User,
        update: &WorkerUpdate,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // only the owner of this worker or admins can update it
        if !user.is_admin() && user.username != self.user {
            return unauthorized!();
        }
        // add this worker to our workers table in scylla
        db::system::update_worker(self, update, shared).await
    }
}

impl TryFrom<HashMap<String, String>> for Worker {
    type Error = ApiError;

    /// Try to cast a [`HashMap`] of strings into an [`Worker`]
    ///
    /// # Arguments
    ///
    /// * `raw` - The `HashMap` to cast into a worker
    fn try_from(mut map: HashMap<String, String>) -> Result<Self, Self::Error> {
        // build a worker from our map of data
        let worker = Worker {
            cluster: extract!(map, "cluster"),
            node: extract!(map, "node"),
            scaler: deserialize_ext!(map, "scaler"),
            name: extract!(map, "name"),
            user: extract!(map, "user"),
            group: extract!(map, "group"),
            pipeline: extract!(map, "pipeline"),
            stage: extract!(map, "stage"),
            status: deserialize_ext!(map, "status"),
            spawned: deserialize_ext!(map, "spawned"),
            heart_beat: deserialize_ext!(map, "heart_beat", None),
            resources: deserialize_ext!(map, "resources"),
            pool: deserialize_ext!(map, "pool"),
            active: deserialize_ext!(map, "active", None),
        };
        Ok(worker)
    }
}

impl WorkerRegistrationList {
    /// Adds new workers to Scylla
    ///
    /// # Arguments
    ///
    /// * `_` - The user that is registering new workers
    /// * `scaler` - The scaler this worker is under
    /// * `shared` - Shared Thorium objects
    pub async fn register(
        &self,
        _: &User,
        scaler: ImageScaler,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // TODO ensure all of these nodes exist
        // add this worker to our workers table in scylla
        db::system::register_workers(scaler, self, shared).await
    }
}

impl WorkerDeleteMap {
    /// Deletes workers from scylla
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting workers
    /// * `scaler` - The scaler that we are deleting workers from
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "WorkerDeleteMap::delete", skip(self, user, shared), fields(user = user.username, count = self.workers.len()), err(Debug))]
    pub async fn delete(
        self,
        user: &User,
        scaler: ImageScaler,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // if this user isn't an admin then make sure they are only deleting their own workers
        db::system::can_delete_workers(user, &self, shared).await?;
        // delete the specified workers
        db::system::delete_workers(scaler, self, shared).await?;
        Ok(())
    }
}

impl From<NodeRow> for NodeListLine {
    /// cast a node row to a node list line
    ///
    /// # Arguments
    ///
    /// * `row` - The node row to convert
    fn from(row: NodeRow) -> Self {
        NodeListLine {
            cluster: row.cluster,
            node: row.node,
        }
    }
}
