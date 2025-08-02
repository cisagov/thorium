//! Wrappers for interacting with network policies within Thorium with different backends
//!
//! Currently only Scylla is supported

use axum::extract::FromRequestParts;
use axum::http::{request::Parts, StatusCode};
use chrono::prelude::*;
use futures::{stream, Future, StreamExt, TryStreamExt};
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;
use std::collections::{HashMap, HashSet};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::db::{self, GroupedScyllaCursorSupport};
use crate::models::{
    ApiCursor, Group, ImageNetworkPolicyUpdate, ImageUpdate, NetworkPolicy, NetworkPolicyListLine,
    NetworkPolicyListParams, NetworkPolicyListRow, NetworkPolicyRequest, NetworkPolicyRow,
    NetworkPolicyRule, NetworkPolicyUpdate, User,
};
use crate::utils::{bounder, helpers};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, deserialize, for_groups, internal_err, not_found, unauthorized, update_return_old,
    update_take,
};

impl NetworkPolicy {
    /// Create a `NetworkPolicy` in Scylla
    ///
    /// # Arguments
    ///
    /// * `req` - The network policy create request
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::create", skip_all, err(Debug))]
    pub async fn create(req: NetworkPolicyRequest, shared: &Shared) -> Result<(), ApiError> {
        // make sure the name isn't empty
        if req.name.is_empty() {
            return bad!("Network policy cannot have an empty name!".to_string());
        }
        // make sure none of the groups have empty names
        if req.groups.iter().any(String::is_empty) {
            return bad!(format!(
                "Network policy cannot have any groups with empty names: {:?}",
                req.groups
            ));
        }
        // make sure the groups exist
        if !db::groups::exists(&req.groups, shared)
            .await
            .map_err(|err| {
                ApiError::new(
                    err.code,
                    Some(format!("Unable to verify that groups exist: {err}")),
                )
            })?
        {
            return not_found!(format!(
                "One or more of the specified groups doesn't exist!"
            ));
        }
        // check that a network policy with this name doesn't exist in any of the requested groups
        let exists_groups = db::network_policies::exists(&req.groups, &req.name, shared).await?;
        if !exists_groups.is_empty() {
            return bad!(format!(
                "Network policy already exists in groups: {exists_groups:?}"
            ));
        }
        // make sure the request's rules are valid
        req.validate_rules(shared).await?;
        // create the network policy
        db::network_policies::create(req, shared).await
    }

    /// List all network policies
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is listing network policies
    /// * `params` - The params to use when listing network policies
    /// * `dedupe` - Whether to dedupe when listing samples or not
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "NetworkPolicy::list", skip(user, shared), err(Debug))]
    pub async fn list(
        user: &User,
        mut params: NetworkPolicyListParams,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ApiCursor<NetworkPolicyListLine>, ApiError> {
        // authorize the groups to list network policies from
        user.authorize_groups(&mut params.groups, shared).await?;
        // get a chunk of the network policies list
        let scylla_cursor = db::network_policies::list(params, dedupe, shared).await?;
        // convert our scylla cursor to a user facing cursor
        Ok(ApiCursor::from(scylla_cursor))
    }

    /// Retrieve a `NetworkPolicy` with the given name and in the given group
    ///
    /// # Arguments
    ///
    /// * `name` - The network policy's name
    /// * `id` - The network policy's ID, needed when one or more distinct network policies
    ///          share the same name
    /// * `user` - The user attempting to get the network policy
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::get", skip_all, err(Debug))]
    pub async fn get(
        name: &str,
        id: Option<Uuid>,
        user: &User,
        shared: &Shared,
    ) -> Result<NetworkPolicy, ApiError> {
        // for users we can search their groups but for admins we need to get all groups
        // try to get this network policy if it exists
        match for_groups!(db::network_policies::get, user, shared, name, id)? {
            // this network policy exists return it
            Some(network_policy) => Ok(network_policy),
            // this sample does not exist return a 404
            None => not_found!(format!("network policy '{}' not found", name)),
        }
    }

    /// Retrieve all default `NetworkPolicy`s in a given group
    ///
    /// # Arguments
    ///
    /// * `group` - One of the group's the network policy is in
    /// * `user` - The user attempting to get the network policy
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::get_all_default", skip_all, err(Debug))]
    pub async fn get_all_default(
        group: &Group,
        user: &User,
        shared: &Shared,
    ) -> Result<Vec<NetworkPolicyListLine>, ApiError> {
        // make sure the user has view access to this group
        group.viewable(user)?;
        // get all of the default policies for this group
        db::network_policies::get_all_default(&group.name, shared).await
    }

    /// Checks that policies with the given names exist within the given groups
    ///
    /// Returns a list of policies that do not exist in all of the given groups
    ///
    /// # Arguments
    ///
    /// * `policy_names` - The names of policies to verify
    /// * `groups` - The groups to verify the policies are in
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::exists_all", skip_all, err(Debug))]
    pub async fn exists_all<I, T>(
        policy_names: I,
        groups: &[String],
        shared: &Shared,
    ) -> Result<Vec<String>, ApiError>
    where
        I: Iterator<Item = T>,
        T: AsRef<str>,
    {
        // convert iterator to owned Strings so they're Send
        let policy_names = policy_names
            .map(|p| p.as_ref().to_owned())
            .collect::<Vec<String>>();
        // TODO: kinda a goofy workaround, but it works; it unfortunately requires
        // the Item in Iterator to be Send even though we're using one thread,
        // but without asserting, we get a weird compiler error "FnOnce isn't general enough";
        // see description for `assert_send_stream`
        let missing_policies = helpers::assert_send_stream(
            stream::iter(policy_names.into_iter())
                // check if each policy exists concurrently
                .map(|policy_name: String| async move {
                    let exists = !db::network_policies::exists(groups, &policy_name, shared)
                        .await?
                        .is_empty();
                    // return the policy and whether or not the group exists
                    Ok((policy_name, exists))
                })
                .buffer_unordered(100),
        )
        .collect::<Vec<Result<(String, bool), ApiError>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<(String, bool)>, ApiError>>()
        // propagate any error
        .map_err(|err: ApiError| {
            ApiError::new(
                err.code,
                Some(format!(
                    "Unable to verify network policies: {}",
                    err.msg
                        .unwrap_or("an unknown Scylla error occurred".to_string())
                )),
            )
        })?
        .into_iter()
        // return the policy if it doesn't exist
        .filter_map(|(policy_name, exists)| (!exists).then_some(policy_name))
        .collect();
        // return the list of policies that don't exist
        Ok(missing_policies)
    }

    /// Update a network policy
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to the network policy
    /// * `user`- The user deleting the network policies
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::update", skip_all, err(Debug))]
    pub async fn update(
        self,
        mut update: NetworkPolicyUpdate,
        user: &User,
        shared: &Shared,
    ) -> Result<NetworkPolicy, ApiError> {
        // make sure the user is an admin
        if !user.is_admin() {
            return unauthorized!();
        }
        // log which network policy we're updating
        event!(
            Level::INFO,
            network_policy = self.name,
            "Updating network policy"
        );
        // before updating anything, get a list of all the images using this network policy,
        // mapped by group
        let mut used_by = Self::used_by(&self.groups, &self.name, shared).await?;
        // update the network policy's groups
        let mut network_policy = update.update_groups(self, shared).await?;
        // update the network policy's rules
        network_policy = update.update_rules(network_policy, shared).await?;
        // update network policy's other data
        update_take!(network_policy.forced_policy, update.forced_policy);
        update_take!(network_policy.default_policy, update.default_policy);
        let old_name = update_return_old!(network_policy.name, update.new_name);
        // make sure the new name is valid
        if old_name.is_some() {
            if network_policy.name.is_empty() {
                return bad!("Network policy cannot have an empty name!".to_string());
            }
            // check that a network policy with this name doesn't exist in any of the requested groups
            let exists_groups =
                db::network_policies::exists(&network_policy.groups, &network_policy.name, shared)
                    .await?;
            if !exists_groups.is_empty() {
                return bad!(format!(
                    "Network policy with name '{}' already exists in groups: {:?}",
                    network_policy.name, exists_groups
                ));
            }
            // generate a new k8s name from the new name
            network_policy.k8s_name =
                super::helpers::to_k8s_name(&network_policy.name, network_policy.id)?;
        }
        // update the network policy in the db's
        db::network_policies::update(&network_policy, &update, &old_name, shared).await?;
        // if we removed groups, delete the network policy from any images in those groups
        if !update.remove_groups.is_empty() {
            // get the used_by data for only the groups we removed;
            // extract the data from the map so we don't attempt to unnecessarily rename
            // the policy for images where the policy was already removed
            let used_by = used_by
                .extract_if(|group, _| update.remove_groups.contains(group))
                .collect();
            // create an image update to remove this network policy
            let image_update = ImageUpdate::default().network_policies(
                ImageNetworkPolicyUpdate::default()
                    .remove_policy(old_name.as_ref().unwrap_or(&network_policy.name)),
            );
            Self::update_images(image_update, used_by, user, shared).await?;
        }
        // if we changed the policy's name, update the policy name in any images that have it
        if let Some(old_name) = old_name {
            // create an image update to rename this network policy in these images
            let image_update = ImageUpdate::default().network_policies(
                ImageNetworkPolicyUpdate::default()
                    .add_policy(network_policy.name.clone())
                    .remove_policy(old_name),
            );
            Self::update_images(image_update, used_by, user, shared).await?;
        }
        // return the updated network policy
        Ok(network_policy)
    }

    /// Apply the given update to the images using this network policy
    ///
    /// # Arguments
    ///
    /// * `image_update` - The update to apply to the images
    /// * `used_by` - The images using this network policy mapped by group
    /// * `user` - The user performing the update
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::update_images", skip_all, err(Debug))]
    async fn update_images(
        image_update: ImageUpdate,
        mut used_by: HashMap<String, Vec<String>>,
        user: &User,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the actual group objects we'll need to update
        let groups = used_by.keys();
        let groups = db::groups::list_details(groups, shared).await?;
        // apply the update to all of the images
        for (group, images) in groups
            .into_iter()
            // map each group object to the list of used images using our map
            .filter_map(|group| used_by.remove(&group.name).map(|images| (group, images)))
        {
            stream::iter(images)
                .map(Ok::<_, ApiError>)
                .try_for_each_concurrent(25, |image| {
                    let group = &group;
                    let image_update = image_update.clone();
                    async move {
                        // get the image from the db
                        let image = db::images::get(&group.name, &image, shared).await?;
                        // update the image, removing the network policy
                        let _ = image.update(image_update, user, group, shared).await?;
                        Ok(())
                    }
                })
                .await?;
        }
        Ok(())
    }

    /// Delete a network policy; also removes the network policy from
    /// all images using it
    ///
    /// # Arguments
    ///
    /// * `user`- The user deleting the network policies
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::delete", skip_all, err(Debug))]
    pub async fn delete(self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        // make sure the user is an admin
        if !user.is_admin() {
            return unauthorized!();
        }
        // create a message containing the groups we're deleting from
        let msg = format!("deleting from '{:?}'", self.groups);
        // log which network policy we're deleting and which groups
        event!(Level::INFO, network_policy = self.name, msg = msg);
        // get lists of images using this network policy mapped by group
        let mut used_by = NetworkPolicy::used_by(&self.groups, &self.name, shared).await?;
        // get the actual group objects we'll need to update
        let groups = used_by.keys();
        let groups = db::groups::list_details(groups, shared).await?;
        // delete this network policy from all images using it in these groups
        let image_update = ImageUpdate::default()
            .network_policies(ImageNetworkPolicyUpdate::default().remove_policy(self.name.clone()));
        for (group, images) in groups
            .into_iter()
            // map each group object to the list of used images using our map
            .filter_map(|group| used_by.remove(&group.name).map(|images| (group, images)))
        {
            stream::iter(images)
                .map(Ok::<_, ApiError>)
                .try_for_each_concurrent(25, |image| {
                    let group = &group;
                    let image_update = image_update.clone();
                    async move {
                        // get the image from the db
                        let image = db::images::get(&group.name, &image, shared).await?;
                        // update the image, removing the network policy
                        let _ = image.update(image_update, user, group, shared).await?;
                        Ok(())
                    }
                })
                .await?;
        }
        // delete the network policy
        db::network_policies::delete(self, shared).await?;
        Ok(())
    }

    /// Remove all network policies from a group
    ///
    /// Does not remove network policies from images, because the images were
    /// previously deleted with the group
    ///
    /// # Arguments
    ///
    /// * `user`- The user deleting the network policies
    /// * `group` - The group to delete the policies from
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "NetworkPolicy::delete_all_group", skip_all, err(Debug))]
    pub async fn delete_all_group(
        user: &User,
        group: &Group,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // log the group we are deleting network policies in
        event!(Level::INFO, group = group.name);
        // make sure we are an owner of this group
        group.modifiable(user)?;
        // delete all network policy rows for a group
        db::network_policies::delete_all_group(group, shared).await?;
        Ok(())
    }

    /// Get a list of images in the given groups used by the given network policy
    ///
    /// Returns a map of groups to lists images in each group using the network policy
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to check for images that use the network policy
    /// * `policy_name` - The name of the network policy
    /// * `shared` - Shared Thorium objects
    pub async fn used_by(
        groups: &[String],
        policy_name: &str,
        shared: &Shared,
    ) -> Result<HashMap<String, Vec<String>>, ApiError> {
        db::network_policies::used_by(groups, policy_name, shared).await
    }

    /// Add an image that one or more network policies are used by
    ///
    /// # Arguments
    ///
    /// * `group` - One of the network policy's groups
    /// * `policy_names` - The network policies used by this image
    /// * `image` - The image that's using the network policy
    /// * `shared` - Shared Thorium objects
    pub async fn set_used_by<I, J, T, S>(
        group: &str,
        policies_added: I,
        policies_removed: J,
        image: &str,
        shared: &Shared,
    ) -> Result<(), ApiError>
    where
        I: Iterator<Item = T>,
        J: Iterator<Item = S>,
        T: AsRef<str>,
        S: AsRef<str>,
    {
        db::network_policies::set_used_by(group, policies_added, policies_removed, image, shared)
            .await
    }

    /// Scrub any details of a network policy that a user does not have access to see
    ///
    /// # Arguments
    ///
    /// * `user` - The user that requested details on this policy
    pub fn scrub(&mut self, user: &User) {
        // scrub various details from the network policy if the user is not an admin
        if !user.is_admin() {
            for rule in self
                .ingress
                .iter_mut()
                .chain(self.egress.iter_mut())
                .flatten()
            {
                // remove any allowed groups that the user is not a part of to avoid leaking a group's
                // existence to a user
                rule.allowed_groups
                    .retain(|group| user.groups.contains(group));
            }
        }
    }
}

/// Validate labels for ingress/egress
macro_rules! validate_labels {
    ($ingress_or_egress:expr) => {{
        // check the ingress/egress rules
        for rule in $ingress_or_egress {
            // check the group "labels"
            for group in &rule.allowed_groups {
                bounder::string_lower(group, "allowed group", 1, 63)?;
            }
            // check the tool "labels"
            for tool in &rule.allowed_tools {
                bounder::string_lower(tool, "allowed tool name", 1, 63)?;
            }
            // iterate and check any custom labels
            for label in rule.allowed_custom.iter().flat_map(|custom_rule| {
                custom_rule
                    .namespace_labels
                    .as_ref()
                    .map(|ns_labels| ns_labels.iter())
                    .into_iter()
                    .flatten()
                    .chain(
                        custom_rule
                            .pod_labels
                            .as_ref()
                            .map(|pod_labels| pod_labels.iter())
                            .into_iter()
                            .flatten(),
                    )
            }) {
                bounder::string_lower(&label.key, "custom label key", 1, 63)?;
                bounder::string_lower(&label.value, "custom label value", 1, 63)?;
            }
        }
        Ok::<(), ApiError>(())
    }};
}

/// Validate allowed groups for ingress/egress
macro_rules! validate_allowed_groups {
    ($ingress:expr, $egress:expr, $shared:expr) => {
        async {
            // get all network policy groups
            let all_allowed_groups: HashSet<String> = $ingress
                .map(|rule| rule.allowed_groups.iter())
                .chain($egress.map(|rule| rule.allowed_groups.iter()))
                .flatten()
                .cloned()
                .collect();
            // make sure none of the allowed groups have an empty name
            if all_allowed_groups.contains("") {
                return bad!(
                    "Network policy settings have one or more groups with empty names".to_string()
                );
            }
            // verify all allowed groups exist
            let all_allowed_groups = all_allowed_groups.into_iter().collect::<Vec<String>>();
            match db::groups::exists(&all_allowed_groups, $shared).await {
                Ok(exists) => {
                    if !exists {
                        return not_found!(format!(
                            "One or more of the specified allowed groups doesn't exist: {:?}",
                            all_allowed_groups
                        ));
                    }
                }
                Err(err) => {
                    return internal_err!(format!(
                        "Unable to verify that allowed groups exist: {err}"
                    ))
                }
            }
            Ok::<(), ApiError>(())
        }
    };
}

/// Cast a list of `NetworkPolicyRuleRaw` to `NetworkPolicyRule`, propagating
/// any errors that occur with a given message
macro_rules! cast_rules {
    ($rules:expr, $msg:expr) => {{
        let cast = $rules
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_, crate::Error>>()
            .map_err(|err| {
                ApiError::new(StatusCode::BAD_REQUEST, Some(format!("{}: {}", $msg, err)))
            })?;
        Ok::<Vec<NetworkPolicyRule>, ApiError>(cast)
    }};
}

/// Update a network policy's rules
macro_rules! update_rules {
    ($ingress_or_egress:expr, $add:expr, $remove:expr, $clear:expr, $deny_all:expr, $msg:expr) => {
        if $clear {
            // if we're set to allow all, just set to None
            $ingress_or_egress = None;
        } else if $deny_all {
            // if we're set to deny all, set to an empty vec
            $ingress_or_egress = Some(Vec::default());
        } else {
            // remove rules
            let mut removed_set: HashSet<Uuid> = $remove.iter().copied().collect();
            if let Some(rules) = $ingress_or_egress.as_mut() {
                rules.retain(|rule| !removed_set.remove(&rule.id));
            }
            if !removed_set.is_empty() {
                // return an error if any of the egress removes weren't found
                return not_found!(format!(
                    "One or more {} rules to be removed is not found in the policy: {:?}",
                    $msg, removed_set
                ));
            }
            // cast the raw rules to add
            let mut added = cast_rules!(
                $add.drain(..),
                format!("One or more {} rules to add are invalid", $msg)
            )?;
            // add rules if we have any
            if !added.is_empty() {
                $ingress_or_egress
                    .get_or_insert(Vec::new())
                    .append(&mut added);
            }
        }
    };
}

impl NetworkPolicyRequest {
    /// Check that the rules in a [`NetworkPolicyRequest`] are valid
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    async fn validate_rules(&self, shared: &Shared) -> Result<(), ApiError> {
        // get iterators over our ingress/egress rules
        let ingress = self.ingress.iter().flatten();
        let egress = self.egress.iter().flatten();
        // validate the rules' allowed groups
        validate_allowed_groups!(ingress.clone(), egress.clone(), shared).await?;
        // validate ingress labels
        validate_labels!(ingress)?;
        // validate egress labels
        validate_labels!(egress)?;
        // check the egress rules
        Ok(())
    }
}

impl NetworkPolicyUpdate {
    /// Check that the `NetworkPolicyUpdate` is valid
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.is_empty() {
            return bad!("The update in the request is empty!".to_string());
        }
        if self.clear_ingress && self.deny_all_ingress {
            return bad!(
                "The update cannot be set to both clear ingress rules and deny all ingress"
                    .to_string()
            );
        }
        if self.clear_egress && self.deny_all_egress {
            return bad!(
                "The update cannot be set to both clear egress rules and deny all egress"
                    .to_string()
            );
        }
        Ok(())
    }

    /// Update the network policy's groups
    ///
    /// # Arguments
    ///
    /// * `network_policy` - The network policy whose groups we're updating
    /// * `shared` - Shared Thorium objects
    async fn update_groups(
        &self,
        mut network_policy: NetworkPolicy,
        shared: &Shared,
    ) -> Result<NetworkPolicy, ApiError> {
        // make a set of groups to easily check contains and remove groups
        let mut groups_set: HashSet<String> = network_policy.groups.iter().cloned().collect();
        // make sure all of the added groups exist
        if !db::groups::exists(&self.add_groups, shared)
            .await
            .map_err(|err| {
                ApiError::new(
                    err.code,
                    Some(format!("Unable to verify that added groups exist: {err}")),
                )
            })?
        {
            return not_found!(format!(
                "One or more of the specified groups to add doesn't exist!"
            ));
        }
        for group in &self.add_groups {
            if groups_set.contains(group) {
                // error if policy is already in a group to be added
                return bad!(format!("Network policy already contains group '{group}'"));
            }
            groups_set.insert(group.clone());
        }
        for group in &self.remove_groups {
            if !groups_set.contains(group) {
                // error if policy is not in a group to be removed
                return bad!(format!(
                    "Network policy does not contain group to be removed '{group}'"
                ));
            }
            groups_set.remove(group);
        }
        if groups_set.is_empty() {
            return bad!(
                "You cannot delete all of a network policy's groups with an update request!"
                    .to_string()
            );
        }
        network_policy.groups = groups_set.into_iter().collect();
        Ok(network_policy)
    }

    /// Update the network policy's rules
    ///
    /// # Arguments
    ///
    /// * `network_policy` - The network policy whose rules we're updating
    /// * `shared` - Shared Thorium objects
    async fn update_rules(
        &mut self,
        mut network_policy: NetworkPolicy,
        shared: &Shared,
    ) -> Result<NetworkPolicy, ApiError> {
        // validate the new rules' allowed groups
        validate_allowed_groups!(self.add_ingress.iter(), self.add_egress.iter(), shared).await?;
        // validate new ingress labels
        validate_labels!(self.add_ingress.iter())?;
        // validate new egress labels
        validate_labels!(self.add_egress.iter())?;
        // update rules
        update_rules!(
            network_policy.ingress,
            self.add_ingress,
            self.remove_ingress,
            self.clear_ingress,
            self.deny_all_ingress,
            "ingress"
        );
        update_rules!(
            network_policy.egress,
            self.add_egress,
            self.remove_egress,
            self.clear_egress,
            self.deny_all_egress,
            "egress"
        );
        Ok(network_policy)
    }
}

impl TryFrom<NetworkPolicyRow> for NetworkPolicy {
    type Error = ApiError;

    fn try_from(row: NetworkPolicyRow) -> Result<Self, Self::Error> {
        Ok(Self {
            name: row.name,
            id: row.id,
            k8s_name: row.k8s_name,
            groups: vec![row.group],
            created: row.created,
            ingress: deserialize!(&row.ingress_raw),
            egress: deserialize!(&row.egress_raw),
            forced_policy: row.forced_policy,
            default_policy: row.default_policy,
            used_by: HashMap::default(),
        })
    }
}

impl From<NetworkPolicyListRow> for NetworkPolicyListLine {
    /// Convert a single [`NetworkPolicyListRow`] to a `NetworkPolicyListLine`
    ///
    /// # Arguments
    ///
    /// * `row` - The policy row to convert
    fn from(row: NetworkPolicyListRow) -> Self {
        Self {
            groups: vec![row.group],
            name: row.name,
            id: row.id,
        }
    }
}

impl NetworkPolicyRequest {
    /// Cast a `NetworkPolicyRequest` to a `NetworkPolicy`
    pub(super) fn cast(self) -> Result<NetworkPolicy, ApiError> {
        // generate a UUID
        let id = Uuid::new_v4();
        // generate a k8s name from our name
        let k8s_name = super::helpers::to_k8s_name(&self.name, id)?;
        Ok(NetworkPolicy {
            name: self.name,
            id,
            k8s_name,
            groups: self.groups,
            created: Utc::now(),
            // cast the raw ingress rules
            ingress: match self.ingress {
                Some(ingress) => Some(cast_rules!(
                    ingress,
                    "One or more ingress rules are invalid"
                )?),
                None => None,
            },
            // cast the raw egress rules
            egress: match self.egress {
                Some(egress) => Some(cast_rules!(egress, "One or more egress rules are invalid")?),
                None => None,
            },
            forced_policy: self.forced_policy,
            default_policy: self.default_policy,
            used_by: HashMap::default(),
        })
    }
}

// Implement cursor support for our network policies list
impl GroupedScyllaCursorSupport for NetworkPolicyListLine {
    /// The params to build this cursor from
    type Params = NetworkPolicyListParams;

    /// The extra info to filter with
    type ExtraFilters = ();

    /// The intermediary component type casted from a Scylla row that is used to build `Self`
    type RowType = NetworkPolicyListRow;

    /// The type of data our rows are grouped by (AKA the partition key)
    type GroupBy = String;

    /// The type `Self` is sorted by in Scylla (AKA the clustering key);
    /// the resulting list of `Self` will be returned ordered by this type
    type SortBy = String;

    /// The type used to break ties when the same instance of `SortBy` is found in multiple
    /// groups, but they are unique entities (i.e. the two entities have the same name, but
    /// none of the same groups)
    type SortTieBreaker = Uuid;

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &mut Self::Params) -> Option<Uuid> {
        params.cursor.take()
    }

    /// Get any group restrictions from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_groups(params: &mut Self::Params) -> HashSet<Self::GroupBy> {
        let groups = std::mem::take(&mut params.groups);
        groups.into_iter().collect()
    }

    /// Get extra filters from our params
    ///
    /// Unused for network policies
    ///
    /// # Arguments
    ///
    /// * `params` - The cursor params
    fn get_extra_filters(_params: &mut Self::Params) -> Self::ExtraFilters {}

    /// Get our the max number of rows to return
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_limit(params: &Self::Params) -> Result<i32, ApiError> {
        params.limit.try_into().map_err(|_| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                Some("The per-request limit is larger than the database supports!".to_string()),
            )
        })
    }

    /// Get the `GroupBy` value from a casted row
    ///
    /// # Arguments
    ///
    /// * `row` - The row to get the `GroupBy` value from
    fn row_get_group_by(row: &Self::RowType) -> Self::GroupBy {
        row.group.clone()
    }

    /// Get the `SortBy` value from a casted row
    ///
    /// # Arguments
    ///
    /// * `row` - The row to get the `SortBy` value from
    fn row_get_sort_by(row: &Self::RowType) -> Self::SortBy {
        row.name.clone()
    }

    /// Get the tie-breaking value from the casted row
    ///
    /// # Arguments
    ///
    /// * `row` - The row to get the `SortBy` value from
    fn row_get_tie_breaker(row: &Self::RowType) -> &Self::SortTieBreaker {
        &row.id
    }

    /// Return the value we're sorting by from `self`
    fn get_sort_by(&self) -> Self::SortBy {
        self.name.clone()
    }

    /// Get the tie-breaking value from `self`
    fn get_tie_breaker(&self) -> &Self::SortTieBreaker {
        &self.id
    }

    /// Convert `self` to a tie to re-retrieve later
    fn to_tie(self) -> (Self::SortBy, Vec<Self::GroupBy>) {
        (self.name, self.groups)
    }

    /// Add a row to `self`, probably by just adding the row's group
    ///
    /// # Arguments
    ///
    /// * `row` - The component row to add
    fn add_row(&mut self, row: Self::RowType) {
        self.groups.push(row.group);
    }

    /// Builds the query string for getting data from ties in the last query
    ///
    /// Ties occur when two groups have the same `SortBy` value and we weren't
    /// able to return them all last iteration. These would be skipped if we
    /// proceeded to query from our last `SortBy` value, so we need to get them
    /// explicitly
    ///
    /// # Arguments
    ///
    /// * `ties` - The ties to get data for
    /// * `_extra` - Any extra filters to apply to this query
    /// * `limit` - The max number of rows to return
    /// * `shared` - Shared Thorium objects
    fn ties_query(
        ties: &[(Self::SortBy, Vec<Self::GroupBy>)],
        _extra: &Self::ExtraFilters,
        limit: i32,
        shared: &Shared,
    ) -> Vec<impl Future<Output = Result<QueryResult, ExecutionError>>> {
        let mut futures = Vec::new();
        for (name, groups) in ties {
            for group in groups {
                futures.push(shared.scylla.session.execute_unpaged(
                    &shared.scylla.prep.network_policies.list_ties,
                    (group, name, limit),
                ));
            }
        }
        futures
    }

    /// Builds the query for getting the first page of values
    ///
    /// The Scylla query must have a `PER PARTITION LIMIT` equal to [`Self::limit`] as
    /// defined by [`Self::Params`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `_extra` - Any extra filters to apply to this query
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    fn pull(
        group: &Self::GroupBy,
        _extra: &Self::ExtraFilters,
        limit: i32,
        shared: &Shared,
    ) -> impl Future<Output = Result<QueryResult, ExecutionError>> {
        // execute the query
        shared.scylla.session.execute_unpaged(
            &shared.scylla.prep.network_policies.list_pull,
            (group, limit),
        )
    }

    /// Builds and executes the query for getting the next page of values
    ///
    /// The Scylla query must have a `PER PARTITION LIMIT` equal to `limit` as
    /// defined by `Self::Params`, as well as pull from everything greater than
    /// `current_sort_by` to work properly
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `_extra` - Any extra filters to apply to this query
    /// * `current_sort_by` - The current sort value we left off at
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    fn pull_more(
        group: &Self::GroupBy,
        _extra: &Self::ExtraFilters,
        current_sort_by: &Self::SortBy,
        limit: i32,
        shared: &Shared,
    ) -> impl Future<Output = Result<QueryResult, ExecutionError>> {
        // execute the query
        shared.scylla.session.execute_unpaged(
            &shared.scylla.prep.network_policies.list_pull_more,
            (group, current_sort_by, limit),
        )
    }
}

impl ApiCursor<NetworkPolicyListLine> {
    /// Turns a list of `NetworkPolicyListLine` into a list of [`NetworkPolicy`]
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting the details for this list
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "ApiCursor<NetworkPolicyListLine>::details",
        skip_all
        err(Debug)
    )]
    pub(crate) async fn details(
        self,
        user: &User,
        shared: &Shared,
    ) -> Result<ApiCursor<NetworkPolicy>, ApiError> {
        // build a list of network policies we want to retrieve
        let policy_names = self
            .data
            .into_iter()
            .map(|line| line.name)
            .collect::<Vec<String>>();
        // use correct backend to list sample details
        let data = for_groups!(
            db::network_policies::list_details,
            user,
            shared,
            policy_names
        )?;
        // build our new cursor object
        Ok(ApiCursor {
            cursor: self.cursor,
            data,
        })
    }
}

impl<S> FromRequestParts<S> for NetworkPolicyListParams
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
