//! Wrappers for interacting with images within Thorium with different backends
//! Currently only Redis is supported

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use crate::models::backends::{db, NotificationSupport};
use crate::models::system::{
    BARE_METAL_CACHE_KEY, EXTERNAL_CACHE_KEY, K8S_CACHE_KEY, KVM_CACHE_KEY, WINDOWS_CACHE_KEY,
};
use crate::models::{
    ChildFilters, ChildFiltersUpdate, Cleanup, CleanupUpdate, Dependencies, DependenciesUpdate,
    Group, GroupAllowAction, Image, ImageArgs, ImageArgsUpdate, ImageBan, ImageBanKind,
    ImageBanUpdate, ImageDetailsList, ImageKey, ImageList, ImageListParams,
    ImageNetworkPolicyUpdate, ImageRequest, ImageScaler, ImageUpdate, Kvm, KvmUpdate,
    NetworkPolicy, OutputCollection, OutputDisplayType, PipelineBan, PipelineBanKind,
    PipelineBanUpdate, PipelineKey, Resources, ResourcesRequest, ResourcesUpdate, SecurityContext,
    SecurityContextUpdate, SpawnLimits, SystemSettings, User,
};
use crate::utils::{bounder, ApiError, Shared};
use crate::{
    bad, can_delete, can_develop, conflict, deserialize_ext, deserialize_opt, extract,
    internal_err, is_admin, not_found, update, update_clear, update_opt, update_opt_empty,
};

impl Resources {
    /// Create an internal default for newly created images without a defined resources block
    #[must_use]
    pub fn internal_default() -> Self {
        Resources {
            cpu: 1000,
            memory: 4096,
            ephemeral_storage: 0,
            worker_slots: 0,
            nvidia_gpu: 0,
            amd_gpu: 0,
        }
    }
}

impl TryFrom<ResourcesRequest> for Resources {
    type Error = ApiError;

    /// Try to convert a resources request to a resources object
    fn try_from(req: ResourcesRequest) -> Result<Self, ApiError> {
        // bound cpu
        let cpu = bounder::image_cpu(&req.cpu)?;
        // bound memory
        let memory = bounder::image_storage(&req.memory)?;
        // bound ephemeral storage
        let ephemeral_storage = match req.ephemeral_storage {
            Some(val) => bounder::image_storage(&val)?,
            None => 0,
        };
        let resources = Resources {
            cpu,
            memory,
            ephemeral_storage,
            nvidia_gpu: req.nvidia_gpu,
            amd_gpu: req.amd_gpu,
            worker_slots: 1,
        };
        Ok(resources)
    }
}

impl ImageList {
    /// Creates new image list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used to page to the next list of images
    /// * `names` - The list of image names
    pub(super) fn new(cursor: Option<usize>, names: Vec<String>) -> Self {
        ImageList { cursor, names }
    }

    /// Turns a [`ImageList`] into a [`ImageDetailsList`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group these reactions are from
    /// * `shared` - Shared objects in Thorium
    pub(crate) async fn details(
        self,
        group: &Group,
        shared: &Shared,
    ) -> Result<ImageDetailsList, ApiError> {
        // use correct backend to list reaction details
        let details = db::images::list_details(&group.name, &self.names, shared).await?;
        // cast to reaction details list
        let details_list = ImageDetailsList::new(self.cursor, details);
        Ok(details_list)
    }
}

impl ImageDetailsList {
    /// Creates a new image details list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - Cursor used to page to the next list of images
    /// * `details` - A list of image details
    pub(super) fn new(cursor: Option<usize>, details: Vec<Image>) -> Self {
        ImageDetailsList { cursor, details }
    }
}

/// Check that all raw regex filters are valid regular expressions
///
/// # Arguments
///
/// * `filters` - The filters to check
fn validate_regex_filters<'a, I>(filters: I) -> Result<(), ApiError>
where
    I: Iterator<Item = &'a String>,
{
    // validate child filters
    let filter_errors: Vec<(&String, regex::Error)> = filters
        // iterate only over unique filters in case we have duplicates;
        // we want to avoid compiling more regexes than we need to
        .unique()
        .filter_map(|raw_regex| Regex::new(raw_regex).err().map(|err| (raw_regex, err)))
        .collect();
    if !filter_errors.is_empty() {
        return bad!(format!(
            "One or more filter regular expressions is invalid: {filter_errors:?}"
        ));
    }
    Ok(())
}

impl ChildFilters {
    /// Check that all given child filters are valid
    fn validate(&self) -> Result<(), ApiError> {
        validate_regex_filters(
            self.mime
                .iter()
                .chain(self.file_name.iter())
                .chain(self.file_extension.iter()),
        )?;
        Ok(())
    }
}

impl ImageRequest {
    /// Cast an `ImageRequest` to a bounds checked [`Image`]
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is casting this request to an image
    /// * `settings` - The Thorium [`SystemSettings`]
    pub fn cast(self, user: &User, settings: &SystemSettings) -> Result<Image, ApiError> {
        // make sure our resource requests are valid
        let resources = Resources::try_from(self.resources)?;
        // validate all volumes
        for vol in &self.volumes {
            vol.validate(user, settings)?;
        }
        // make sure all child filters are valid regular expressions
        self.child_filters.validate()?;
        // if any security context options were set then make sure we are an admin
        if self.security_context.is_some() {
            // make sure we are an admin
            is_admin!(user);
        }
        // cast to an Image
        let image = Image {
            group: self.group,
            name: self.name,
            version: self.version,
            image: self.image,
            creator: user.username.clone(),
            lifetime: self.lifetime,
            timeout: self.timeout,
            resources,
            spawn_limit: self.spawn_limit,
            scaler: self.scaler,
            runtime: 600.0,
            volumes: self.volumes,
            env: self.env,
            args: self.args,
            modifiers: self.modifiers,
            description: self.description,
            security_context: self.security_context.unwrap_or_default(),
            used_by: Vec::default(),
            collect_logs: self.collect_logs,
            generator: self.generator,
            dependencies: self.dependencies,
            display_type: self.display_type,
            output_collection: self.output_collection,
            child_filters: self.child_filters,
            clean_up: self.clean_up,
            kvm: self.kvm,
            bans: HashMap::default(),
            network_policies: self.network_policies,
        };
        Ok(image)
    }
}

impl ImageArgsUpdate {
    /// Update an images args
    pub fn update(mut self, image: &mut Image) {
        // apply any updates to this image
        update_opt_empty!(image.args.entrypoint, self.entrypoint);
        update_clear!(image.args.entrypoint, self.clear_entrypoint);
        update_opt_empty!(image.args.command, self.command);
        update_clear!(image.args.command, self.clear_command);
        update_opt!(image.args.reaction, self.reaction);
        update_clear!(image.args.reaction, self.clear_reaction);
        update_opt!(image.args.repo, self.repo);
        update_clear!(image.args.repo, self.clear_repo);
        update_opt!(image.args.commit, self.commit);
        update_clear!(image.args.commit, self.clear_commit);
        update!(image.args.output, self.output);
    }
}

/// Updates a resource request or limit
macro_rules! update_resource {
    ($orig:expr, $update:expr, $translator:expr) => {
        if let Some(update) = $update.take() {
            $orig = $translator(&update)?;
        }
    };
}

impl ResourcesUpdate {
    /// Update an images resources
    ///
    /// # Arguments
    ///
    /// * `image` - The image to update
    pub fn update(mut self, image: &mut Image) -> Result<(), ApiError> {
        // update this images resources
        update_resource!(image.resources.cpu, self.cpu, bounder::image_cpu);
        update_resource!(image.resources.memory, self.memory, bounder::image_storage);
        update_resource!(
            image.resources.ephemeral_storage,
            self.ephemeral_storage,
            bounder::image_storage
        );
        update!(image.resources.nvidia_gpu, self.nvidia_gpu);
        update!(image.resources.amd_gpu, self.amd_gpu);
        Ok(())
    }
}

impl SecurityContextUpdate {
    /// Update an images security context
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is applying this update
    /// * `image` - The image to apply this update too
    pub fn update(mut self, user: &User, image: &mut Image) -> Result<(), ApiError> {
        // only admins can update security contexts
        is_admin!(user);
        // update security_context settings if its set
        update_opt!(image.security_context.user, self.user);
        update_opt!(image.security_context.group, self.group);
        // update privilege escalation if its set
        update!(
            image.security_context.allow_privilege_escalation,
            self.allow_privilege_escalation
        );
        // clear any requested values
        update_clear!(image.security_context.user, self.clear_user);
        update_clear!(image.security_context.group, self.clear_group);
        Ok(())
    }
}

impl DependenciesUpdate {
    /// Update an images depdendencies
    ///
    /// # Arguments
    ///
    /// * `image` - The image to apply this update too
    pub fn update(mut self, image: &mut Image) {
        // update any dependency settings
        // sample settings
        update!(image.dependencies.samples.location, self.samples.location);
        update_opt!(image.dependencies.samples.kwarg, self.samples.kwarg);
        update_clear!(image.dependencies.samples.kwarg, self.samples.clear_kwarg);
        update!(image.dependencies.samples.strategy, self.samples.strategy);
        // ephemeral settings
        update!(
            image.dependencies.ephemeral.location,
            self.ephemeral.location
        );
        update_opt!(image.dependencies.ephemeral.kwarg, self.ephemeral.kwarg);
        update_clear!(
            image.dependencies.ephemeral.kwarg,
            self.ephemeral.clear_kwarg
        );
        update!(
            image.dependencies.ephemeral.strategy,
            self.ephemeral.strategy
        );
        image
            .dependencies
            .ephemeral
            .names
            .retain(|name| !self.ephemeral.remove_names.contains(name));
        image
            .dependencies
            .ephemeral
            .names
            .extend(self.ephemeral.add_names);
        // results settings
        update!(image.dependencies.results.location, self.results.location);
        update!(image.dependencies.results.kwarg, self.results.kwarg);
        update!(image.dependencies.results.strategy, self.results.strategy);
        // update results images
        image
            .dependencies
            .results
            .images
            .retain(|image| !self.results.remove_images.contains(image));
        image
            .dependencies
            .results
            .images
            .extend(self.results.add_images);
        // update results names
        image
            .dependencies
            .results
            .names
            .retain(|name| !self.results.remove_names.contains(name));
        image
            .dependencies
            .results
            .names
            .extend(self.results.add_names);
        // repos settings
        update!(image.dependencies.repos.location, self.repos.location);
        update_opt!(image.dependencies.repos.kwarg, self.repos.kwarg);
        update_clear!(image.dependencies.repos.kwarg, self.repos.clear_kwarg);
        update!(image.dependencies.repos.strategy, self.repos.strategy);
        // tags settings
        update!(image.dependencies.tags.enabled, self.tags.enabled);
        update!(image.dependencies.tags.location, self.tags.location);
        update_opt!(image.dependencies.tags.kwarg, self.tags.kwarg);
        update_clear!(image.dependencies.tags.kwarg, self.tags.clear_kwarg);
        update!(image.dependencies.tags.strategy, self.tags.strategy);
        // Children settings
        update!(image.dependencies.children.enabled, self.children.enabled);
        update!(image.dependencies.children.location, self.children.location);
        update_opt!(image.dependencies.children.kwarg, self.children.kwarg);
        update_clear!(image.dependencies.children.kwarg, self.children.clear_kwarg);
        update!(image.dependencies.children.strategy, self.children.strategy);
        // update children images
        image
            .dependencies
            .children
            .images
            .retain(|image| !self.children.remove_images.contains(image));
        image
            .dependencies
            .children
            .images
            .extend(self.children.add_images);
    }
}

impl ChildFiltersUpdate {
    /// Update an image's child filters
    ///
    /// # Errors
    ///
    /// Returns a 400 BAD REQUEST error if any regular expressions to add
    /// are invalid or if any filters to remove are not in the image
    ///
    /// # Arguments
    ///
    /// `child_filters` - The image's child filters to update
    pub fn update(self, child_filters: &mut ChildFilters) -> Result<(), ApiError> {
        // first validate all filters we want to add
        validate_regex_filters(
            self.add_mime
                .iter()
                .chain(self.add_file_name.iter())
                .chain(self.add_file_extension.iter()),
        )?;
        // make sure all the mime filters we want to add are already in the image
        let missing_filters: Vec<&String> =
            self.remove_mime.difference(&child_filters.mime).collect();
        if !missing_filters.is_empty() {
            return bad!(format!(
                "Image is missing one or more mime child filters to be removed: {missing_filters:?}"
            ));
        }
        // make sure all the file name filters we want to add are already in the image
        let missing_filters: Vec<&String> = self
            .remove_file_name
            .difference(&child_filters.file_name)
            .collect();
        if !missing_filters.is_empty() {
            return bad!(format!(
                "Image is missing one or more file name child filters to be removed: {missing_filters:?}"
            ));
        }
        // make sure all the file extension filters we want to add are already in the image
        let missing_filters: Vec<&String> = self
            .remove_file_extension
            .difference(&child_filters.file_extension)
            .collect();
        if !missing_filters.is_empty() {
            return bad!(format!(
                "Image is missing one or more file extension child filters to be removed: {missing_filters:?}"
            ));
        }
        // add all filters
        child_filters.mime.extend(self.add_mime);
        child_filters.file_name.extend(self.add_file_name);
        child_filters.file_extension.extend(self.add_file_extension);
        // remove filters that are in the remove sets
        child_filters.mime.retain(|f| !self.remove_mime.contains(f));
        child_filters
            .file_name
            .retain(|f| !self.remove_file_name.contains(f));
        child_filters
            .file_extension
            .retain(|f| !self.remove_file_extension.contains(f));
        // update submit non-matches setting
        update!(child_filters.submit_non_matches, self.submit_non_matches);
        Ok(())
    }
}

impl CleanupUpdate {
    /// Update an images clean up script
    ///
    /// # Arguments
    ///
    /// * `image` - The image to apply this update too
    pub fn update(self, image: &mut Image) -> Result<(), ApiError> {
        // if the update is to clear this images clean up settings then just do that
        if self.clear {
            image.clean_up = None;
        } else {
            // check if we have an existing clean up script
            let clean_up = match image.clean_up.as_mut() {
                Some(clean_up) => {
                    // if a new clean up scrip was set then set that
                    update!(clean_up.script, self.script);
                    // return our existing clean up config
                    clean_up
                }
                None => {
                    // we don't already have an existing clean up config
                    // so make sure this config sets a target script
                    match self.script {
                        Some(script) => image.clean_up.insert(Cleanup::new(script)),
                        None => {
                            return bad!(
                                "A clean up script must be set to update clean up settings!"
                                    .to_owned()
                            )
                        }
                    }
                }
            };
            // update our clean up settings
            update!(clean_up.job_id, self.job_id);
            update!(clean_up.results, self.results);
            update!(clean_up.result_files_dir, self.result_files_dir);
        }
        Ok(())
    }
}

impl KvmUpdate {
    /// Updates an images kvm settings
    ///
    /// # Arguments
    ///
    /// * `image` - The image to apply this update too
    pub fn update(self, image: &mut Image) -> Result<(), ApiError> {
        // extract any existing kvm settings
        let kvm = match image.kvm.take() {
            Some(mut kvm) => {
                // apply any updates
                update!(kvm.xml, self.xml);
                update!(kvm.qcow2, self.qcow2);
                // return our updated kvm settings
                kvm
            }
            None => {
                // we have no existing settings so make sure all required options are set
                match (self.xml, self.qcow2) {
                    (Some(xml), Some(qcow2)) => Kvm { xml, qcow2 },
                    // we have no updates to apply
                    (None, None) => return Ok(()),
                    _ => return bad!("xml and qcow2 must both be set".to_owned()),
                }
            }
        };
        // set our new settings
        image.kvm = Some(kvm);
        Ok(())
    }
}

impl ImageBanUpdate {
    /// Updates an image's ban list
    ///
    /// # Arguments
    ///
    /// * `image` - The image to apply this update to
    /// * `user` - The user attempting to apply this update
    pub fn update(self, image: &mut Image, user: &User) -> Result<(), ApiError> {
        // exit immediately if there are no bans to add/remove
        if self.is_empty() {
            return Ok(());
        }
        // check that the user is an admin and return unauthorized if not
        is_admin!(user);
        // check that all the bans in the list of bans to be removed actually exist
        for ban_remove in &self.bans_removed {
            if !image.bans.contains_key(ban_remove) {
                return not_found!(format!(
                    "The ban with id '{ban_remove}' is not contained in the image ban list",
                ));
            }
        }
        // check that all bans to be added don't already exist
        for ban_add in &self.bans_added {
            if image.bans.contains_key(&ban_add.id) {
                return bad!(format!("A ban with id '{}' already exists. Bans cannot be updated, only added or removed.", ban_add.id));
            }
        }
        // add the requested bans
        for ban in self.bans_added {
            image.bans.insert(ban.id, ban);
        }
        // remove the requested bans
        for ban_id in &self.bans_removed {
            image.bans.remove(ban_id);
        }
        Ok(())
    }
}

impl ImageNetworkPolicyUpdate {
    /// Updates an image's network policies
    ///
    /// # Arguments
    ///
    /// * `image` - The image to apply this update to
    /// * `user` - The user attempting to apply this update
    /// * `shared` - Shared Thorium objects
    pub async fn update(self, image: &mut Image, shared: &Shared) -> Result<(), ApiError> {
        // return an error if the user is trying to add network policies and the image is
        // not scaled by K8's
        if image.scaler != ImageScaler::K8s && !self.policies_added.is_empty() {
            return bad!(
                "Network policies can only be applied to images scaled in K8s!".to_string()
            );
        }
        // check that the image has all of the network policies to be removed
        let bad_removes: Vec<&String> = self
            .policies_removed
            .iter()
            .filter(|remove| !image.network_policies.contains(*remove))
            .collect();
        if !bad_removes.is_empty() {
            return bad!(format!("The image does not have one or more of the network policies to be removed: {bad_removes:?}"));
        }
        // check that the image doesn't already have any of the added network policies already
        let bad_adds: Vec<&String> = self
            .policies_added
            .iter()
            .filter(|add| image.network_policies.contains(*add))
            .collect();
        if !bad_adds.is_empty() {
            return bad!(format!("The image already has one or more of the network policies to be added: {bad_adds:?}"));
        }
        // check that all of the added network policies actually exist
        let image_group_slice = &[image.group.clone()];
        let missing_policies =
            NetworkPolicy::exists_all(self.policies_added.iter(), image_group_slice, shared)
                .await?;
        if !missing_policies.is_empty() {
            return not_found!(format!("One or more of the added network policies does not exist in the image's group: {missing_policies:?}"));
        }
        // update the used_by data for the network policies
        NetworkPolicy::set_used_by(
            &image.group,
            self.policies_added.iter(),
            self.policies_removed.iter(),
            &image.name,
            shared,
        )
        .await?;
        // perform the update
        for added in self.policies_added {
            image.network_policies.insert(added);
        }
        for removed in &self.policies_removed {
            image.network_policies.remove(removed);
        }
        Ok(())
    }
}

impl Image {
    /// Creates an image in the backend from an imageRequest
    ///
    /// # Arguments
    ///
    /// * `user` - The user creating this image
    /// * `request` - The image request to build this image from
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::create", skip(request, shared), err(Debug))]
    pub async fn create(
        user: &User,
        mut request: ImageRequest,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // ensure name is alphanumeric and lowercase
        bounder::string_lower(&request.name, "name", 1, 25)?;
        // ensure the all volume names are alphanumeric and lowercase
        request
            .volumes
            .iter()
            .map(|vol| bounder::string_lower(&vol.name, "volume name", 1, 25))
            .collect::<Result<Vec<_>, ApiError>>()?;
        // authorize this user is apart of this group
        let group = Group::authorize(user, &request.group, shared).await?;
        // make sure we can create images in this group
        group.allowable(GroupAllowAction::Images)?;
        // make sure we can create images in this group
        group.developer(user, request.scaler)?;
        // now that permissions have been checked, check if the image already exists
        if db::images::exists_authenticated(&request.name, &group, shared).await? {
            return conflict!(format!(
                "Image {} already exists in group {}",
                &request.name, &request.group
            ));
        }
        // build the message to trace after we create this image
        let msg = format!("Created image {}:{}", &request.group, &request.name);
        // strip our docker image path if it exists
        if let Some(image) = request.image.as_mut() {
            // strip our image to ensure there is no extra whitespace
            *image = image.trim().to_owned();
            // make sure this image is not empty
            if image.is_empty() {
                // this image is empty so throw an error
                return bad!("Image cannot be empty!".to_owned());
            }
        }
        match (request.network_policies.is_empty(), &request.scaler) {
            // if the image is scaled in K8's and no policies were provided, use default policies
            (true, ImageScaler::K8s) => {
                let default_policies =
                    match NetworkPolicy::get_all_default(&group, user, shared).await {
                        Ok(default_policies) => default_policies,
                        Err(err) => {
                            return internal_err!(format!(
                                "Unable to retrieve default network policies \
                                    in this image's group: {err}"
                            ));
                        }
                    };
                // set default policies for this image
                request.network_policies = default_policies
                    .into_iter()
                    .map(|policy| policy.name)
                    .collect();
            }
            // if the image is scaled in K8's and policies were provided, make sure all the policies are valid
            (false, ImageScaler::K8s) => {
                let image_group_slice = &[request.group.clone()];
                let missing_policies = NetworkPolicy::exists_all(
                    request.network_policies.iter(),
                    image_group_slice,
                    shared,
                )
                .await?;
                if !missing_policies.is_empty() {
                    return not_found!(format!("One or more of the network policies does not exist in the image's group: {missing_policies:?}"));
                }
            }
            // if the image is NOT scaled in K8's and policies were provided, return an error
            (false, _) => {
                return bad!(
                    "Network policies can only be applied to images scaled in K8s!".to_string()
                );
            }
            // if the image is NOT scaled in K8's and no policies were given, do nothing
            (true, _) => (),
        }
        // create the image in the backend
        let image = db::images::create(user, request, shared).await?;
        // add this image to the used by sets for image's network policies if it has any
        if !image.network_policies.is_empty() {
            if let Err(err) = NetworkPolicy::set_used_by(
                &image.group,
                image.network_policies.iter(),
                std::iter::empty::<&str>(),
                &image.name,
                shared,
            )
            .await
            {
                return internal_err!(format!(
                    "Unable to update 'used_by' sets for one or more network policies: {err}"
                ));
            }
        }
        // log that we created this image
        event!(Level::INFO, msg = &msg);
        Ok(image)
    }

    /// Gets an image by group and name from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user getting this image
    /// * `group` - The group the requested image is in
    /// * `image` - The name of the image to retrieve
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::get", skip(group, image, shared), err(Debug))]
    pub async fn get(
        user: &User,
        group: &str,
        image: &str,
        shared: &Shared,
    ) -> Result<(Group, Self), ApiError> {
        // authorize this user is apart of this group
        let group = Group::authorize(user, group, shared).await?;
        // get image from backend
        let image = db::images::get(&group.name, image, shared).await?;
        Ok((group, image))
    }

    /// Checks if an image exists in the backend with an already authenticated group
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the image to check for
    /// * `group` - The group the image to check
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::exists_authenticated", skip_all, err(Debug))]
    pub async fn exists_authenticated(
        name: &str,
        group: &Group,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // check if image exists in the correct backend
        if db::images::exists_authenticated(name, group, shared).await? {
            Ok(())
        } else {
            not_found!(format!("Image {} does not exist in {}", name, &group.name))
        }
    }

    /// Gets the scaler for a specific image
    ///
    /// # Arguments
    ///
    /// * `group` - The group the image to check
    /// * `name` - The name of the image to inspect
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::get_scaler", skip(group, shared), err(Debug))]
    pub async fn get_scaler(
        group: &Group,
        name: &str,
        shared: &Shared,
    ) -> Result<ImageScaler, ApiError> {
        // check if image exists in the correct backend
        db::images::get_scaler(&group.name, name, shared).await
    }

    /// Gets the bans for a specific image
    ///
    /// If the image has no bans, the bans can either be a `None`
    /// or an empty map
    ///
    /// # Arguments
    ///
    /// * `group` - The group the image to check
    /// * `name` - The name of the image to inspect
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::get_bans", skip(group, shared), err(Debug))]
    pub async fn get_bans(
        group: &Group,
        name: &str,
        shared: &Shared,
    ) -> Result<HashMap<Uuid, ImageBan>, ApiError> {
        // get the image's bans
        db::images::get_bans(&group.name, name, shared).await
    }

    /// List all images in a group
    ///
    /// # Arguments
    ///
    /// * `group` - The group the Pipeline is in
    /// * `cursor` - The cursor to use when paging through images
    /// * `limit` - The number of objects to try and return (not strongly enforced)
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::list", skip_all, err(Debug))]
    pub async fn list(
        group: &Group,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ImageList, ApiError> {
        // get all image names in group
        db::images::list(&group.name, cursor, limit, shared).await
    }

    /// Deletes an image from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting an image
    /// * `group` - The group that the image to delete is in
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::delete", skip_all, err(Debug))]
    pub async fn delete(self, user: &User, group: &Group, shared: &Shared) -> Result<(), ApiError> {
        // log the image we are deleting
        event!(Level::INFO, group = group.name, image = self.name);
        // make sure we have the privileges to delete this image
        can_delete!(self, group, user);
        // throw an error if this image is still being leveraged
        if !self.used_by.is_empty() {
            return conflict!(format!(
                "This image is still being used by {:?}",
                self.used_by
            ));
        }
        // delete image from backend
        db::images::delete(&self, shared).await?;
        // delete all of the image's notifications
        let key = ImageKey::from(&self);
        self.delete_all_notifications(&key, shared).await
    }

    /// Deletes all images in a group from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting all images within a group from the backend
    /// * `group` - The group that to delete images from
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::delete_all", skip_all, err(Debug))]
    pub async fn delete_all(user: &User, group: &Group, shared: &Shared) -> Result<(), ApiError> {
        // log the group we are deleting images in
        event!(Level::INFO, group = group.name);
        // make sure we are an owner of this group
        group.modifiable(user)?;
        // use correct backend for deleting images
        db::images::delete_all(group, shared).await
    }

    /// Updates an image
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to apply to this image
    /// * `user` - The user that is updating this image
    /// * `group` - The group the image to update is in
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::update", skip(self, user, group, shared), err(Debug))]
    pub async fn update(
        mut self,
        mut update: ImageUpdate,
        user: &User,
        group: &Group,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // log the image we are updating
        event!(Level::INFO, group = group.name, image = self.name);
        // make sure we can edit images in this group
        group.allowable(GroupAllowAction::Images)?;
        // make sure we can modify this image
        can_develop!(self.creator, group, self.scaler, user);
        if let Some(new_scaler) = update.scaler {
            // if we are modifying the scaler then make sure we can develop for that too
            group.developer(user, new_scaler)?;
            // if we're changing the scaler from K8's to non K8's and we have network policies,
            // throw an error
            if new_scaler != ImageScaler::K8s && !self.network_policies.is_empty() {
                return bad!("Cannot set the image's scaler to non-K8's while network policies are applied! \
                    First remove all of the image's network policies, then update the scaler type.".to_string());
            }
        }
        // remove any environment variables
        self.env.retain(|env, _| !update.remove_env.contains(env));
        // ensure the all new volume names are alphanumeric and lowercase
        update
            .add_volumes
            .iter()
            .map(|vol| bounder::string_lower(&vol.name, "volume name", 1, 25))
            .collect::<Result<Vec<_>, ApiError>>()?;
        // keep any volume not in the remove volume vector
        self.volumes
            .retain(|vol| !update.remove_volumes.contains(&vol.name));
        // add new environment variables
        self.env.extend(update.add_env);
        if !update.add_volumes.is_empty() {
            // only get settings and validate volumes if there are volumes to validate
            let settings = db::system::get_settings(shared).await?;
            // validate any new volumes
            for volume in &update.add_volumes {
                volume.validate(user, &settings)?;
            }
        }
        // add new volumes
        self.volumes.extend(update.add_volumes);
        // update and strip our image if its set
        if let Some(image) = &update.image {
            // strip of any trailing whitespace
            let image = image.trim();
            // make sure this image is not empty
            if image.is_empty() {
                // this image is empty so throw an error
                return bad!("Image cannot be empty!".to_owned());
            }
            // set our new validated image
            self.image = Some(image.to_owned());
        }
        // overlay update on the Image data
        update_opt!(self.version, update.version);
        update_opt!(self.timeout, update.timeout);
        update_opt_empty!(self.image, update.image);
        update!(self.scaler, update.scaler);
        update_opt!(self.lifetime, update.lifetime);
        update_opt_empty!(self.modifiers, update.modifiers);
        update_opt_empty!(self.description, update.description);
        // update our resource requirements if any updates were found
        if let Some(resources) = update.resources.take() {
            resources.update(&mut self)?;
        }
        // update our spawn limit
        update!(self.spawn_limit, update.spawn_limit);
        // clear fields if requested
        update_clear!(self.version, update.clear_version);
        update_clear!(self.image, update.clear_image);
        update_clear!(self.lifetime, update.clear_lifetime);
        update_clear!(self.description, update.clear_description);
        // update our images args if any updates were found
        if let Some(args) = update.args.take() {
            args.update(&mut self);
        }
        // update security_context settings if its set
        if let Some(security_context) = update.security_context.take() {
            security_context.update(user, &mut self)?;
        }
        update!(self.collect_logs, update.collect_logs);
        update!(self.generator, update.generator);
        // update any dependency settings
        update.dependencies.update(&mut self);
        // update display_type
        update!(self.display_type, update.display_type);
        // get the output collection settings if we have any
        if let Some(output_collection) = update.output_collection.take() {
            // update output collection settings
            self.output_collection.update(output_collection);
        }
        if let Some(child_filters) = update.child_filters.take() {
            // update child filters if we have an update
            child_filters.update(&mut self.child_filters)?;
        }
        // update our kvm settings if we have any updates
        update.kvm.update(&mut self)?;
        // save a copy of our bans before updating
        let mut bans_update = update.bans.clone();
        // check if we were banned before the update
        let banned_before = !self.bans.is_empty();
        // update our ban list if there are any to update
        update.bans.update(&mut self, user)?;
        // remove bans if their volumes have been removed
        let mut bans_removed = self
            .bans
            .extract_if(|_, ban| match &ban.ban_kind {
                ImageBanKind::InvalidHostPath(ban) => {
                    update.remove_volumes.contains(&ban.volume_name)
                }
                _ => false,
            })
            .map(|(id, _)| id)
            .collect::<Vec<Uuid>>();
        // update the image's network policies
        update.network_policies.update(&mut self, shared).await?;
        // save image to correct backend
        db::images::update(&self, shared).await?;
        // check if we are banned after the update
        let banned_after = !self.bans.is_empty();
        // update the image's notifications based on the updated bans
        bans_removed.append(&mut bans_update.bans_removed);
        let key = ImageKey::from(&self);
        self.update_ban_notifications(&key, &bans_update.bans_added, &bans_removed, shared)
            .await?;
        // update any of the pipelines bans the image is a part of if needed
        self.update_pipeline_bans(banned_before, banned_after, user, shared)
            .await?;
        Ok(self)
    }

    /// Calculates and updates all images in a group average runtime
    ///
    // loop until jobs in this stage are deleted
    /// * `user` - The user that is updating this image
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Image::runtimes_update", skip_all, err(Debug))]
    pub async fn update_runtimes(user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can update the runtimes for all images
        is_admin!(user);
        // iterate over all groups and images
        let mut cursor = 0;
        loop {
            // crawl over groups and update their images average runtimes
            let groups = Group::list_details(user, cursor, 1000, shared).await?;
            for group in groups.details {
                // update average runtime in the correct backend
                db::images::update_runtimes(&group, shared).await?;
            }
            // check if our cursor has been exhausted
            if groups.cursor.is_none() {
                break;
            }
            // update cursor
            cursor = groups.cursor.unwrap();
        }
        Ok(())
    }

    /// Update the bans of any pipelines this image is a part of if its banned
    /// status has changed
    ///
    /// If multiple images in the same pipeline have bans added/removed at the
    /// same time, there are possible race conditions where bans aren't
    /// updated correctly in the pipeline
    ///
    /// # Arguments
    ///
    /// * `banned_before` - Whether the image was banned before the update
    /// * `banned_after` - Whether the image is banned after the update
    /// * `user` - The user that originally updated the image; this will fail if the user is not
    ///            an admin, but the bans shouldn't have changed if the user isn't an admin anyway
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Image::update_pipeline_bans", skip_all, err(Debug))]
    pub async fn update_pipeline_bans(
        &self,
        banned_before: bool,
        banned_after: bool,
        user: &User,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        match (banned_before, banned_after) {
            // the image is no longer banned, so remove the respective ban from all of its pipelines
            (true, false) => {
                stream::iter(self.used_by.iter())
                    .map(Ok)
                    .try_for_each_concurrent(100, |pipeline| async {
                        // retrieve the pipeline
                        let mut pipeline =
                            db::pipelines::get(&self.group, pipeline, shared).await?;
                        // get the id's of bans that refer to this image
                        let bans_removed = pipeline
                            .bans
                            .iter()
                            .filter_map(|(id, ban)| {
                                if let PipelineBanKind::BannedImage(ban_kind) = &ban.ban_kind {
                                    // return only bans that refer to this image
                                    (ban_kind.image == self.name).then_some(*id)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<Uuid>>();
                        // update the pipeline ban list
                        let bans_update =
                            PipelineBanUpdate::default().remove_bans(bans_removed.clone());
                        bans_update.update(&mut pipeline, user)?;
                        db::pipelines::update(&pipeline, &[], &[], shared).await?;
                        // create an empty bans slice to help the compiler know its type
                        let bans_added: &[PipelineBan] = &[];
                        // update the pipeline's notifications
                        let key = PipelineKey::from(&pipeline);
                        pipeline
                            .update_ban_notifications(&key, bans_added, &bans_removed, shared)
                            .await?;
                        Ok(())
                    })
                    .await
            }
            // the image is newly banned, so add a ban to all of its pipelines
            (false, true) => {
                stream::iter(self.used_by.iter())
                    .map(Ok)
                    .try_for_each_concurrent(100, |pipeline| async {
                        // get the pipeline
                        let mut pipeline =
                            db::pipelines::get(&self.group, pipeline, shared).await?;
                        // create a new ban referring to this image
                        let ban = PipelineBan::new(PipelineBanKind::image_ban(&self.name));
                        // update the pipeline ban list
                        let bans_update = PipelineBanUpdate::default().add_ban(ban.clone());
                        bans_update.update(&mut pipeline, user)?;
                        db::pipelines::update(&pipeline, &[], &[], shared).await?;
                        // update the pipeline's notifications
                        let key = PipelineKey::from(&pipeline);
                        pipeline
                            .update_ban_notifications(&key, &[ban], &[], shared)
                            .await?;

                        Ok(())
                    })
                    .await
            }
            // the image's banned status hasn't changed, so do nothing
            _ => Ok(()),
        }
    }
}

impl TryFrom<(HashMap<String, String>, Vec<String>)> for Image {
    type Error = ApiError;

    /// Try to cast a [`HashMap`] of strings into an [`Image`]
    ///
    /// # Arguments
    ///
    /// * `raw` - The `HashMap` to cast into an image
    fn try_from(raw: (HashMap<String, String>, Vec<String>)) -> Result<Self, Self::Error> {
        // unwrap raw into the raw hashmap and the used_by vector
        let (mut map, used_by) = raw;
        // cast to an image struct
        let image = Image {
            group: extract!(map, "group"),
            name: extract!(map, "name"),
            version: deserialize_opt!(map, "version"),
            creator: extract!(map, "creator"),
            scaler: deserialize_ext!(map, "scaler", ImageScaler::default()),
            image: deserialize_ext!(map, "image", None),
            resources: deserialize_ext!(map, "resources", Resources::internal_default()),
            spawn_limit: deserialize_ext!(map, "spawn_limit", SpawnLimits::Unlimited),
            lifetime: deserialize_ext!(map, "lifetime", None),
            timeout: deserialize_ext!(map, "timeout", None),
            runtime: extract!(map, "runtime").parse::<f64>()?,
            volumes: deserialize_ext!(map, "volumes"),
            env: deserialize_ext!(map, "env"),
            args: deserialize_ext!(map, "args", ImageArgs::default()),
            modifiers: deserialize_ext!(map, "modifiers", None),
            description: deserialize_opt!(map, "description"),
            security_context: deserialize_ext!(map, "security_context", SecurityContext::default()),
            used_by,
            collect_logs: deserialize_ext!(map, "collect_logs", true),
            generator: deserialize_ext!(map, "generator", false),
            dependencies: deserialize_ext!(map, "dependencies", Dependencies::default()),
            display_type: deserialize_ext!(map, "display_type", OutputDisplayType::default()),
            output_collection: deserialize_ext!(
                map,
                "output_collection",
                OutputCollection::default()
            ),
            child_filters: deserialize_ext!(map, "child_filters", ChildFilters::default()),
            clean_up: deserialize_opt!(map, "clean_up"),
            kvm: deserialize_opt!(map, "kvm"),
            bans: deserialize_ext!(map, "bans", HashMap::default()),
            network_policies: deserialize_ext!(map, "network_policies", HashSet::default()),
        };
        Ok(image)
    }
}

impl<S> FromRequestParts<S> for ImageListParams
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

impl ImageScaler {
    /// Get the cache key for our image scaler
    #[must_use]
    pub fn cache_key(&self) -> &str {
        match self {
            ImageScaler::K8s => K8S_CACHE_KEY,
            ImageScaler::BareMetal => BARE_METAL_CACHE_KEY,
            ImageScaler::Windows => WINDOWS_CACHE_KEY,
            ImageScaler::External => EXTERNAL_CACHE_KEY,
            ImageScaler::Kvm => KVM_CACHE_KEY,
        }
    }
}
