//! Structs and code related to updating images

use thorium::Error;
use thorium::models::{
    BurstableResourcesUpdate, Image, ImageBanUpdate, ImageRequest, ImageUpdate, Resources,
    ResourcesRequest, ResourcesUpdate, SecurityContext, SecurityContextUpdate,
};

use crate::utils::diff;
use crate::{calc_remove_add_map, calc_remove_add_vec, set_clear, set_modified, set_modified_opt};

/// A pipeline that needs to be updated and how that update should
/// be performed
#[derive(Debug)]
pub struct ToolboxImageUpdate {
    /// The name of the image
    pub image_name: String,
    /// The name of the image's version
    pub image_version: String,
    /// The image's group
    pub group: String,
    /// The operation to perform to update the image
    pub op: ToolboxImageUpdateOp,
}

impl ToolboxImageUpdate {
    pub fn new<N, V, G>(image_name: N, image_version: V, group: G, op: ToolboxImageUpdateOp) -> Self
    where
        N: Into<String>,
        V: Into<String>,
        G: Into<String>,
    {
        Self {
            image_name: image_name.into(),
            image_version: image_version.into(),
            group: group.into(),
            op,
        }
    }
}

/// An operation that needs to be performed to update an image
#[derive(Debug)]
pub enum ToolboxImageUpdateOp {
    /// This is a brand new image that needs to be created
    Create(Box<ImageRequest>),
    /// This is an existing image that needs to be updated
    Update(Box<ImageUpdate>),
    /// Nothing needs to be done for this image
    Unchanged,
}

/// Calculate the updates for image resources
///
/// # Arguments
///
/// * `old` - The old image's resources that we're updating
/// * `new` - The new image's resources to update to
#[allow(clippy::needless_pass_by_value)]
fn calculate_resource_update(old: Resources, new: ResourcesRequest) -> Option<ResourcesUpdate> {
    // convert our resources request to a resoruces object
    let new_cast = Resources::from(new.clone());
    // if our resources are identical then no update is needed
    if old == new_cast {
        return None;
    }
    // start with an empty burstable resources struct
    let mut burstable = BurstableResourcesUpdate::default();
    // if cpu is different then set that
    if new_cast.burstable.cpu != old.burstable.cpu {
        // our burstable cpu is different so add that to our update
        burstable.cpu = Some(new.burstable.cpu);
    }
    // if memory is different then set that
    if new_cast.burstable.memory != old.burstable.memory {
        // our burstable memory is different so add that to our update
        burstable.memory = Some(new.burstable.memory);
    }
    let new: ResourcesRequest = new_cast.into();
    // convert resources to a request for comparison
    let old: ResourcesRequest = old.into();
    Some(ResourcesUpdate {
        cpu: set_modified!(old.cpu, new.cpu),
        memory: set_modified!(old.memory, new.memory),
        ephemeral_storage: set_modified!(old.ephemeral_storage, new.ephemeral_storage),
        nvidia_gpu: set_modified!(old.nvidia_gpu, new.nvidia_gpu),
        amd_gpu: set_modified!(old.amd_gpu, new.amd_gpu),
        burstable,
    })
}

/// Calculate the updates to a security context
///
/// # Arguments
///
/// * `old` - The old security context we're updating
/// * `new` - The new security context to update to
#[allow(clippy::needless_pass_by_value)]
fn calculate_security_context_update(
    old: SecurityContext,
    new: Option<SecurityContext>,
) -> Option<SecurityContextUpdate> {
    match new {
        // do nothing if nothing changed
        Some(new) if old == new => None,
        // build the correct update for this security contex5
        Some(new) => Some(SecurityContextUpdate {
            user: set_modified_opt!(old.user, new.user),
            group: set_modified_opt!(old.group, new.group),
            allow_privilege_escalation: set_modified!(
                old.allow_privilege_escalation,
                new.allow_privilege_escalation
            ),
            clear_user: set_clear!(old.user, new.user),
            clear_group: set_clear!(old.group, new.group),
        }),
        // the old security context is the default, so do nothing
        None if old == SecurityContext::default() => None,
        // if nothing was provided in the security context and we had a non-default one before
        // we should clear it
        None => Some(
            SecurityContextUpdate::default()
                .clear_user()
                .clear_group()
                .disallow_escalation(),
        ),
    }
}

/// Calculate what updates need to be made to a image (if any) based
/// on the image's current state and the request from the toolbox manifest
///
/// # Arguments
///
/// * `image` - The current state of the image in Thorium
/// * `req` - The image request from the toolbox manifest
pub fn calculate_image_update(mut image: Image, mut req: ImageRequest) -> Option<ImageUpdate> {
    // if the request is the same as the image, no update is needed
    if image == req {
        return None;
    }
    let (remove_volumes, add_volumes) =
        calc_remove_add_vec!(image.volumes, |vol| vol.name, req.volumes, |vol| vol);
    let (remove_env, add_env) = calc_remove_add_map!(image.env, req.env);
    // build the image update to send to the api
    let update = ImageUpdate {
        // calculate clears first before we move things
        clear_version: set_clear!(image.version, req.version),
        clear_image: set_clear!(image.image, req.image),
        clear_lifetime: set_clear!(image.lifetime, req.lifetime),
        clear_description: set_clear!(image.description, req.description),
        version: set_modified_opt!(image.version, req.version),
        // seems unused?
        external: None,
        image: set_modified_opt!(image.image, req.image),
        scaler: set_modified!(image.scaler, req.scaler),
        lifetime: set_modified_opt!(image.lifetime, req.lifetime),
        timeout: set_modified_opt!(image.timeout, req.timeout),
        resources: calculate_resource_update(image.resources, req.resources),
        spawn_limit: set_modified!(image.spawn_limit, req.spawn_limit),
        add_volumes,
        remove_volumes,
        add_env,
        remove_env,
        args: diff::images::calculate_image_args_update(image.args, req.args),
        modifiers: set_modified_opt!(image.modifiers, req.modifiers),
        description: set_modified_opt!(image.description, req.description),
        security_context: calculate_security_context_update(
            image.security_context,
            req.security_context,
        ),
        collect_logs: set_modified!(image.collect_logs, req.collect_logs),
        generator: set_modified!(image.generator, req.generator),
        dependencies: diff::images::calculate_dependencies_update(
            image.dependencies,
            req.dependencies,
        ),
        display_type: set_modified!(image.display_type, req.display_type),
        output_collection: diff::images::calculate_output_collection_update(
            image.output_collection,
            req.output_collection,
        ),
        child_filters: diff::images::calculate_child_filters_update(
            image.child_filters,
            req.child_filters,
        ),
        clean_up: diff::images::calculate_clean_up_update(image.clean_up, req.clean_up),
        kvm: diff::images::calculate_kvm_update(image.kvm, req.kvm),
        // bans aren't in a manifest, so we can just set default here
        bans: ImageBanUpdate::default(),
        network_policies: diff::images::calculate_network_policies_update(
            image.network_policies,
            req.network_policies,
        ),
    };
    Some(update)
}
