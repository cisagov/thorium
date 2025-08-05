//! Wrappers for interacting with groups within Thorium with different backends
//! Currently only Redis is supported

use ldap3::{Scope, SearchEntry};
use std::collections::{HashMap, HashSet};
use tracing::{event, instrument, Level};

use super::db;
use super::db::groups::{MembersLists, RawGroupData};
use crate::models::groups::GroupUsers;
use crate::models::{
    Group, GroupAllowAction, GroupAllowed, GroupAllowedUpdate, GroupDetailsList, GroupList,
    GroupRequest, GroupStats, GroupUpdate, GroupUsersRequest, GroupUsersUpdate, ImageScaler,
    Pipeline, User,
};
use crate::utils::{bounder, ApiError, Shared};
use crate::{
    bad, conflict, deserialize_ext, deserialize_opt, ldap, not_found, unauthorized, unavailable,
    update, update_clear, update_opt_empty,
};

// Only build in when DB features are enabled
impl GroupRequest {
    /// Cast a GroupRequest to a Group
    ///
    /// # Arguments
    ///
    /// * `user` - The user creating this group
    /// * `shared` - Shared objects in Thorium
    pub async fn cast(mut self, user: &User, shared: &Shared) -> Result<Group, ApiError> {
        // bounds check string and ensure its alphanumeric and lowercase
        bounder::string_lower(&self.name, "group['name']", 1, 50)?;
        // make sure all users exist
        User::exists_many(&self.owners.direct, shared).await?;
        User::exists_many(&self.managers.direct, shared).await?;
        User::exists_many(&self.users.direct, shared).await?;
        User::exists_many(&self.monitors.direct, shared).await?;
        // get a list of analysts in Thorium
        let analysts = db::users::get_analysts(shared).await?;
        // inject creator as an owner if he is not already in there
        if !self.owners.direct.contains(&user.username) {
            self.owners.direct.insert(user.username.clone());
        }
        // cast our role requests
        let owners = GroupUsers::from(self.owners);
        let managers = GroupUsers::from(self.managers);
        let users = GroupUsers::from(self.users);
        let monitors = GroupUsers::from(self.monitors);
        // cast to group object
        let mut cast = Group {
            name: self.name,
            owners,
            managers,
            analysts,
            users,
            monitors,
            description: self.description,
            allowed: self.allowed,
        };
        // fix this groups roles if its needed
        cast.fix();
        Ok(cast)
    }
}

impl From<GroupUsersRequest> for GroupUsers {
    /// Cast a group users request to a group users
    ///
    /// # Arguments
    ///
    /// * `req` - The group users request to cast
    fn from(req: GroupUsersRequest) -> Self {
        GroupUsers {
            combined: req.direct.clone(),
            direct: req.direct,
            metagroups: req.metagroups,
        }
    }
}

impl GroupUsersUpdate {
    /// Apply this update to a groups direct users
    ///
    /// # Arguments
    ///
    /// * `group` - The group to apply updates too
    pub fn update_direct(
        &mut self,
        role: &mut GroupUsers,
        added: &mut HashSet<String>,
        removed: &mut HashSet<String>,
        valid: &HashSet<String>,
    ) -> Result<(), ApiError> {
        // make sure that all new users are valid
        if let Some(user) = added.iter().find(|user| !valid.contains(*user)) {
            return bad!(format!("{} is not a valid user.", user));
        }
        // remove any users that got removed and added to the same role
        self.direct_add
            .retain(|name| !self.direct_remove.contains(name));
        // add any new users to our added set
        added.extend(self.direct_add.iter().cloned());
        // track the users we are removing
        removed.extend(self.direct_remove.iter().cloned());
        // remove any users that we are adding to a new role
        removed.retain(|name| !self.direct_add.contains(name));
        // update our direct users
        role.direct.extend(self.direct_add.drain());
        role.direct
            .retain(|name| !self.direct_remove.contains(name));
        // remove any removed users from our combined set
        role.combined.retain(|user| !removed.contains(user));
        // add these names to our combined set
        role.combined
            .extend(role.direct.iter().map(|name| name.to_owned()));
        Ok(())
    }

    /// Apply this update to a groups metagroups
    ///
    /// # Arguments
    ///
    /// * `group` - The group to apply updates too
    pub fn update_metagroups(&mut self, role: &mut GroupUsers) {
        // update our metagroups
        role.metagroups.extend(self.metagroups_add.drain());
        role.metagroups
            .retain(|name| !self.metagroups_remove.contains(name));
    }
}

impl GroupAllowedUpdate {
    /// Apply this update to our group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to apply this update too
    pub fn update(&mut self, group: &mut Group) {
        // apply our updates
        update!(group.allowed.files, self.files);
        update!(group.allowed.repos, self.repos);
        update!(group.allowed.tags, self.tags);
        update!(group.allowed.images, self.images);
        update!(group.allowed.pipelines, self.pipelines);
        update!(group.allowed.reactions, self.reactions);
        update!(group.allowed.results, self.results);
        update!(group.allowed.comments, self.comments);
    }
}

impl GroupList {
    /// Creates a new group list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used page to the next list of groups
    /// * `names` - A list of group names
    pub(super) fn new(cursor: Option<usize>, names: Vec<String>) -> Self {
        GroupList { cursor, names }
    }
}

impl GroupDetailsList {
    /// Creates a new group details list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used page to the next list of groups
    /// * `details` - A list of group details
    pub(super) fn new(cursor: Option<usize>, details: Vec<Group>) -> Self {
        GroupDetailsList { cursor, details }
    }
}

/// Remove users or metagroups from any lower groups
///
/// # Arguments
///
/// * `target` - The users or metagroups to check
/// * `lower` - The lower roles to remove from
macro_rules! fix_roles {
    ($target:expr, $lower:expr) => {
        // crawl over each lower combined users and remove any duplicate users
        for lower in $lower.iter_mut() {
            lower
                .combined
                .retain(|name| !$target.combined.contains(name));
        }
        // crawl over each direct users and remove any duplicate users
        for lower in $lower.iter_mut() {
            lower.direct.retain(|name| !$target.direct.contains(name));
        }
        // crawl over each lower role and remove any duplicate metagroups
        for lower in $lower.iter_mut() {
            lower
                .metagroups
                .retain(|name| !$target.metagroups.contains(name));
        }
    };
}

impl Group {
    /// Creates a group object in the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user creating a group
    /// * `req` - The group request to use when creating this group
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::create", skip(user, shared), err(Debug))]
    pub async fn create(user: &User, req: GroupRequest, shared: &Shared) -> Result<Self, ApiError> {
        // ensure name is alphanumeric and lowercase
        bounder::string_lower(&req.name, "name", 1, 50)?;
        // error if ldap is not configured but ldap roles are
        if shared.config.thorium.auth.ldap.is_none()
            && (!req.owners.metagroups.is_empty()
                || !req.managers.metagroups.is_empty()
                || !req.users.metagroups.is_empty()
                || !req.monitors.metagroups.is_empty())
        {
            return unavailable!("Ldap is not configured".to_owned());
        }
        // check if this group is on the namespace blacklist
        if shared
            .config
            .thorium
            .namespace_blacklist
            .contains(&req.name)
        {
            // return an error if this group is in the blacklist
            return bad!(format!(
                "'{}' is a restricted name! Please choose a different name.",
                req.name
            ));
        }
        // check if this group already exists
        if db::groups::exists(&[req.name.clone()], shared).await? {
            return unauthorized!();
        }
        // add group to backend
        let group = db::groups::create(user, req, shared).await?;
        Ok(group)
    }

    /// Fix a group so that any users or metagroups that are in multiple roles are valid.
    ///
    /// If there is overlap we will use the highest valid permission
    fn fix(&mut self) {
        // remove our owners from any lower roles
        fix_roles!(
            self.owners,
            &mut [&mut self.managers, &mut self.users, &mut self.monitors]
        );
        // remove our managers from any lower roles
        fix_roles!(self.managers, &mut [&mut self.users, &mut self.monitors]);
        // remove our users from any lower roles
        fix_roles!(self.users, &mut [&mut self.monitors]);
    }

    /// Lists all group names that you have permission to see
    ///
    /// # Arguments
    ///
    /// * `user` - The user listing groups
    /// * `cursor` - The cursor to use as the start for paging
    /// * `limit` - The number of items to attempt to retrieve (not a hard limit for admins)
    /// * `shared` - Shared objects in Thorium
    pub async fn list(
        user: &User,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<GroupList, ApiError> {
        // if user is an admin well need to list all groups
        if user.is_admin() {
            // list all groups in the backend
            db::groups::list(cursor, limit, shared).await
        } else {
            // just use the list of groups the user is apart of
            let names = user
                .groups
                .iter()
                .skip(cursor)
                .take(limit)
                .map(|group| group.to_owned())
                .collect();

            // calculate new cursor
            let new_cursor = cursor + limit;
            // check if this was the last page
            if new_cursor > user.groups.len() {
                // set cursor to None since no more groups exist
                Ok(GroupList::new(None, names))
            } else {
                // set next cursor
                Ok(GroupList::new(Some(new_cursor), names))
            }
        }
    }

    /// Lists all groups that you have permission to see with details
    ///
    /// # Arguments
    ///
    /// * `user` - The user listing groups
    /// * `cursor` - The cursor to use as the start for paging
    /// * `limit` - The number of items to attempt to retrieve (not a hard limit for admins)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_details(
        user: &User,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<GroupDetailsList, ApiError> {
        // get vector of groups to get details on and the new cursor
        let list = Self::list(user, cursor, limit, shared).await?;
        // get details on the users groups
        let details = db::groups::list_details(list.names.iter(), shared).await?;
        // cast to group details list object
        let group_details = GroupDetailsList::new(list.cursor, details);
        Ok(group_details)
    }

    /// Checks if a array of groups exists in bulk
    ///
    /// If any of these groups do not exist an error will be returned
    ///
    /// # Arguments
    ///
    /// * `names` - An array of group names to check if they exist
    /// * `shared` - Shared objects in Thorium
    pub async fn exists(names: &[String], shared: &Shared) -> Result<(), ApiError> {
        // check if group exists
        let exists = db::groups::exists(names, shared).await?;
        if !exists {
            // build error message
            return not_found!(format!("groups {:?} must exist", &names));
        }
        Ok(())
    }

    /// Checks if a user is an owner in this group
    ///
    /// # Arguments
    ///
    /// * `user` - The user to check edit privileges for
    #[instrument(name = "Group::is_owner", skip_all, fields(user = &user.username), err(Debug))]
    pub fn is_owner(&self, user: &User) -> Result<(), ApiError> {
        // if user is an admin then pass check
        if user.is_admin() {
            return Ok(());
        }
        // check if user is an owner
        let has_role = self
            .owners
            .combined
            .iter()
            .any(|name| name == &user.username);
        // return unauthorized if user does not have any of the needed roles
        if !has_role {
            // this user doesn't have any roles in this group
            event!(Level::ERROR, msg = "not an owner",);
            return unauthorized!();
        }
        Ok(())
    }

    /// Checks if a user can delete arbitrary things in a group
    ///
    /// # Arguments
    ///
    /// * `user` - The user to check edit privileges for
    #[instrument(name = "Group::modifiable", skip_all, err(Debug))]
    pub fn modifiable(&self, user: &User) -> Result<(), ApiError> {
        // if user is an admin or a group owner/manager then pass check
        if user.is_admin()
            || self.owners.combined.contains(&user.username)
            || self.managers.combined.contains(&user.username)
        {
            return Ok(());
        }
        // otherwise, this user doesn't have any roles in this group, so return unauthorized
        event!(
            Level::ERROR,
            msg = "cannot modify arbitrary data in group",
            user = user.username
        );
        unauthorized!()
    }

    /// Checks if a user can create and edit images and pipelines in this group
    ///
    /// # Arguments
    ///
    /// * `user` - The user to check edit privileges for
    /// * `scaler` - The scaler to check this user can edit/create for
    #[instrument(name = "Group::developer", skip_all, err(Debug))]
    pub fn developer(&self, user: &User, scaler: ImageScaler) -> Result<(), ApiError> {
        // if user is an admin then pass check
        if user.is_admin() {
            return Ok(());
        }
        // make sure this user has the developer role
        if !user.is_developer(scaler) {
            // this user is not a developer
            event!(
                Level::ERROR,
                msg = "User is not a developer",
                user = user.username,
                scaler = scaler.to_string(),
            );
            return unauthorized!();
        }
        // check if user has any of the roles needed to edit data
        if !self.users.combined.contains(&user.username)
            && !self.managers.combined.contains(&user.username)
            && !self.owners.combined.contains(&user.username)
        {
            // this user doesn't have any roles in this group
            event!(
                Level::ERROR,
                msg = "cannot edit data in group",
                user = user.username
            );
            return unauthorized!();
        }
        Ok(())
    }

    /// Checks if a user can create and edit images and pipelines in this group
    ///
    /// # Arguments
    ///
    /// * `user` - The user to check edit privileges for
    /// * `scaler` - The scaler to check this user can edit/create for
    #[instrument(name = "Group::developer_many", skip(self, user), fields(group = self.name), err(Debug))]
    pub fn developer_many(&self, user: &User, scalers: &[ImageScaler]) -> Result<(), ApiError> {
        // if user is an admin then pass check
        if user.is_admin() {
            return Ok(());
        }
        // make sure this user has the developer role
        if !user.is_developer_many(scalers) {
            // this user is not a developer
            event!(
                Level::ERROR,
                msg = "User is not a developer",
                user = user.username,
            );
            return unauthorized!();
        }
        // check if user has any of the roles needed to edit data
        if !self.users.combined.contains(&user.username)
            && !self.managers.combined.contains(&user.username)
            && !self.owners.combined.contains(&user.username)
        {
            // this user doesn't have any roles in this group
            event!(
                Level::ERROR,
                msg = "cannot edit data in group",
                user = user.username
            );
            return unauthorized!();
        }
        Ok(())
    }

    /// Check if a group allows a certain datatype or not
    #[instrument(name = "Group::allowable", skip(self), fields(group = self.name), err(Debug))]
    pub fn allowable(&self, action: GroupAllowAction) -> Result<(), ApiError> {
        if self.allowed.is_allowed(action) {
            Ok(())
        } else {
            unauthorized!(format!(
                "{} does not allow for {} to be created!",
                self.name, action
            ))
        }
    }

    /// Checks if a user can edit things in this group
    ///
    /// # Arguments
    ///
    /// * `user` - The user to check edit privileges for
    #[instrument(name = "Group::editable", skip_all, fields(group = self.name), err(Debug))]
    pub fn editable(&self, user: &User) -> Result<(), ApiError> {
        // if user is an admin then pass check
        if user.is_admin() {
            return Ok(());
        }
        // check if user has any of the roles needed to edit data
        if !self.users.combined.contains(&user.username)
            && !self.analysts.contains(&user.username)
            && !self.managers.combined.contains(&user.username)
            && !self.owners.combined.contains(&user.username)
        {
            // this user doesn't have any roles in this group
            event!(
                Level::ERROR,
                msg = "cannot edit data in group",
                user = user.username
            );
            return unauthorized!();
        }
        Ok(())
    }

    /// Check if a user can see items in this group
    ///
    /// # Arguments
    ///
    /// * `user` - The user try to access data in this group
    #[instrument(name = "Group::viewable", skip_all, err(Debug))]
    pub fn viewable(&self, user: &User) -> Result<(), ApiError> {
        // if user is an admin then pass check
        if user.is_admin() {
            return Ok(());
        }
        // check if user has any of the roles needed to view data
        if !self.users.combined.contains(&user.username)
            && !self.managers.combined.contains(&user.username)
            && !self.analysts.contains(&user.username)
            && !self.owners.combined.contains(&user.username)
            && !self.monitors.combined.contains(&user.username)
        {
            // this user doesn't have any roles in this group
            event!(
                Level::ERROR,
                msg = "no roles in group",
                user = user.username,
                group = &self.name,
            );
            return unauthorized!();
        }
        Ok(())
    }

    /// Authorize a user as part of a group with the required permissions
    ///
    /// Arguments
    ///
    /// * `user` - The user to authorize
    /// * `name` - The name of the group to retrieve
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::authorize", skip(user, shared), err(Debug))]
    pub async fn authorize(user: &User, name: &str, shared: &Shared) -> Result<Group, ApiError> {
        // bypass member check if we are an admin
        // make sure user is apart of this group
        if !user.is_admin() && !user.groups.iter().any(|val| val == name) {
            event!(Level::WARN, "User not in group");
            // user is not a part of the requested group
            return unauthorized!();
        }
        // get group object from backend
        // error if it doesn't exist
        let group = match db::groups::get(name, shared).await {
            Ok(group) => group,
            Err(error) => {
                // log that this group doesn't exist
                event!(Level::ERROR, error = error.msg);
                // return our error
                return Err(error);
            }
        };
        // make sure the user can see data in this group
        group.viewable(user)?;
        Ok(group)
    }

    /// Authorize a user as part of all of these group with viewing permissions
    ///
    /// Arguments
    ///
    /// * `user` - The user to authorize
    /// * `name` - The names of the groups to authorize for
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::authorize_all", skip(user, shared), err(Debug))]
    pub async fn authorize_all(
        user: &User,
        names: &[String],
        shared: &Shared,
    ) -> Result<Vec<Group>, ApiError> {
        // bypass member check if we are an admin
        // make sure user is apart of this group
        if !user.is_admin() && !names.iter().all(|name| user.groups.contains(name)) {
            // user is not a part of the requested group
            return unauthorized!();
        }
        // get group object from backend
        // error if it doesn't exist
        let groups = db::groups::list_details(names.iter(), shared).await?;
        // make sure the user has some role in this group
        if user.is_admin() {
            // if we are an admin we need to do a second call to make sure these groups exist
            // this is because nonexistent groups will still return empty user lists
            if !db::groups::exists(names, shared).await? {
                // one or more of the groups don't exist throw an error
                return not_found!(format!("all of {:?} groups must exist", names));
            }
        } else {
            for group in &groups {
                group.viewable(user)?;
            }
            // log that this user is authorized to view this group
            let msg = format!("{} authorized for viewing {:?}", user.username, names);
            event!(Level::INFO, msg);
        }
        Ok(groups)
    }

    /// Authorize a user as part of all of these group with the required permissions and all groups
    /// allow this action
    ///
    /// Arguments
    ///
    /// * `user` - The user to authorize
    /// * `name` - The names of the groups to authorize for
    /// * `shared` - Shared objects in Thorium
    #[instrument(
        name = "Group::authorize_check_allow_all",
        skip(user, shared),
        err(Debug)
    )]
    pub async fn authorize_check_allow_all(
        user: &User,
        names: &[String],
        action: GroupAllowAction,
        shared: &Shared,
    ) -> Result<Vec<Group>, ApiError> {
        // bypass member check if we are an admin
        // make sure user is apart of this group
        if !user.is_admin() && !names.iter().all(|name| user.groups.contains(name)) {
            // user is not a part of the requested group
            return unauthorized!();
        }
        // get group object from backend
        // error if it doesn't exist
        let groups = db::groups::list_details(names.iter(), shared).await?;
        // make sure the user has some role in this group
        if !user.is_admin() {
            for group in groups.iter() {
                // make sure this user can see this group
                group.viewable(user)?;
                // make sure this group can perform this action
                group.allowable(action)?;
            }
            // log that this user is authorized to view group
            let msg = format!("{} authorized to view {:?}", user.username, names);
            event!(Level::INFO, msg);
        } else {
            // make sure all groups can perform this action even for admins
            for group in groups.iter() {
                group.allowable(action)?;
            }
            // if we are an admin we need to do a second call to make sure these groups exist
            // this is because nonexistent groups will still return empty user lists
            if !db::groups::exists(names, shared).await? {
                // one or more of the groups don't exist throw an error
                return not_found!(format!("all of {:?} groups must exist", names));
            }
        }
        Ok(groups)
    }

    /// Get a group object if that group exists
    ///
    /// # Arguments
    ///
    /// * `user` - The user to authorize to be able to view this group
    /// * `name` - The name of the group to get
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::get", skip(user, shared), err(Debug))]
    pub async fn get(user: &User, name: &str, shared: &Shared) -> Result<Self, ApiError> {
        // authorize user is apart of this group
        Self::authorize(user, name, shared).await
    }

    /// Diffs two group versions and finds which users to add or remove from group perms
    ///
    /// # Arguments
    ///
    /// * `old` - The old group to compare too
    #[instrument(name = "Group::diff", skip_all)]
    fn diff(&self, old: &Group, added: &mut HashSet<String>, removed: &mut HashSet<String>) {
        // get the difference between our new and old group
        let owners_add = self.owners.combined.difference(&old.owners.combined);
        let owners_remove = old.owners.combined.difference(&self.owners.combined);
        let managers_add = self.managers.combined.difference(&old.managers.combined);
        let managers_remove = old.managers.combined.difference(&self.managers.combined);
        let users_add = self.users.combined.difference(&old.users.combined);
        let users_remove = old.users.combined.difference(&self.users.combined);
        let monitors_add = self.monitors.combined.difference(&old.monitors.combined);
        let monitors_remove = old.monitors.combined.difference(&self.monitors.combined);
        // ignore any users that are direct add users
        let owners_remove = owners_remove.filter(|name| !old.owners.direct.contains(*name));
        let managers_remove = managers_remove.filter(|name| !old.managers.direct.contains(*name));
        let users_remove = users_remove.filter(|name| !old.users.direct.contains(*name));
        let monitors_remove = monitors_remove.filter(|name| !old.monitors.direct.contains(*name));
        // extend our add set
        added.extend(owners_add.cloned());
        added.extend(managers_add.cloned());
        added.extend(users_add.cloned());
        added.extend(monitors_add.cloned());
        // extend our remove set
        removed.extend(owners_remove.cloned());
        removed.extend(managers_remove.cloned());
        removed.extend(users_remove.cloned());
        removed.extend(monitors_remove.cloned());
    }

    /// Updates the group
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to apply to this group
    /// * `user` - The user that is updating this group
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::update", skip_all, err(Debug))]
    pub async fn update(
        mut self,
        mut update: GroupUpdate,
        user: &User,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // error if ldap is not configured but new ldap roles are
        if shared.config.thorium.auth.ldap.is_none() && update.will_update_metagroups() {
            return unavailable!("Ldap is not configured".to_owned());
        }
        // list of users added/removed
        let mut added = HashSet::default();
        let mut removed = HashSet::default();
        // get a copy of our old group info
        let old = self.clone();
        // get a list of all valid users in Thorium to validate against
        let valid = HashSet::from_iter(db::users::list(shared).await?);
        // make sure we can modify this group with our role
        // if we are adding/removing owners we also need to be an owner
        if !update.owners.is_empty() {
            // make sure we are an owner
            self.is_owner(user)?;
            // error if we try to remove ourselves as owner
            if update.owners.direct_remove.contains(&user.username) {
                return conflict!("You cannot remove yourself as an owner".to_owned());
            }
            // update our owner roles
            update
                .owners
                .update_direct(&mut self.owners, &mut added, &mut removed, &valid)?;
            // error if the owner list will be empty after this
            if self.owners.direct.is_empty() && self.owners.metagroups.is_empty() {
                return conflict!("You cannot update a group to have no owners".to_owned());
            }
        } else {
            // only check manager if we are doing more than just removing ourselves
            if !update.removes_only_user(&user.username) {
                // we are removing more than just ourselves so check for arbitrary delete perms
                self.modifiable(user)?;
            }
        }
        // start with empty direct + metagroup user lists
        self.owners.combined.clear();
        self.managers.combined.clear();
        self.users.combined.clear();
        self.monitors.combined.clear();
        // apply the updates for this groups metagroups
        update.owners.update_metagroups(&mut self.owners);
        update.managers.update_metagroups(&mut self.managers);
        update.users.update_metagroups(&mut self.users);
        update.monitors.update_metagroups(&mut self.monitors);
        // get the ldap info for this group if we have any ldap info
        if self.metagroup_enabled() {
            // get the ldap info for this groups metagroups
            let ldap_info = LdapUserMap::new(&[&self], &valid, shared).await?;
            // apply this ldap info to our roles
            ldap_info.apply_to_role(&mut self.owners);
            ldap_info.apply_to_role(&mut self.managers);
            ldap_info.apply_to_role(&mut self.users);
            ldap_info.apply_to_role(&mut self.monitors);
        }
        // figure out which users to add or remove
        self.diff(&old, &mut added, &mut removed);
        // update this groups roles
        update
            .owners
            .update_direct(&mut self.owners, &mut added, &mut removed, &valid)?;
        update
            .managers
            .update_direct(&mut self.managers, &mut added, &mut removed, &valid)?;
        update
            .users
            .update_direct(&mut self.users, &mut added, &mut removed, &valid)?;
        update
            .monitors
            .update_direct(&mut self.monitors, &mut added, &mut removed, &valid)?;
        // fix any overlap in this updates roles
        self.fix();
        // update description
        update_opt_empty!(self.description, update.description);
        // clear description if flag is set
        update_clear!(self.description, update.clear_description);
        // update our allowed settings
        update.allowed.update(&mut self);
        // save updated group to the backend
        db::groups::update(&self, &added, &removed, shared).await?;
        Ok(self)
    }

    /// Deletes a group by name
    ///
    /// This will also delete all pipelines, images, and reactions within this group.
    ///
    /// # Arguments
    ///
    /// * `user` - The user to authorize to be able to delete this group
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::delete", skip_all, fields(group = &self.name), err(Debug))]
    pub async fn delete(self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        // make sure we are an owner of this group
        self.is_owner(user)?;
        // delete from backend
        db::groups::delete(user, &self, shared).await
    }

    /// Syncs all ldap metagroups and their users
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::sync_ldap", skip_all, err(Debug))]
    pub async fn sync_ldap(shared: &Shared) -> Result<(), ApiError> {
        // we are syncing all groups in Thorium so get the Thorium user since
        // they can see all groups.
        let admin = User::force_get("thorium", shared).await?;
        // list all of our groups and update them one at a time
        for group in db::groups::list_all(&admin, shared).await? {
            // make an empty group update to apply to this group
            let update = GroupUpdate::default();
            // apply this empty update to force an ldap update
            group.update(update, &admin, shared).await?;
        }
        Ok(())
    }

    /// Get stats on a group including its pipelines
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor to use when listing groups by status
    /// * `limit` - The max number of groups to return (soft limit)
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Group::stats", skip(shared), fields(group = &self.name), err(Debug))]
    pub async fn stats(
        &self,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<GroupStats, ApiError> {
        // get a list of the members of this group
        let members = self.members();
        // get a list of pipelines
        let pipelines = Pipeline::list(self, cursor, limit, shared)
            .await?
            .details(self, shared)
            .await?;
        // build an empty group status object
        let mut status = GroupStats {
            pipelines: HashMap::default(),
        };
        // crawl through the pipelines of this group and get their status
        for pipeline in pipelines.details.into_iter() {
            // get the status info for this pipeline
            let pipe_status = pipeline.status(&members, shared).await?;
            // add this pipeline status to our status map
            status.pipelines.insert(pipeline.name, pipe_status);
        }
        Ok(status)
    }
}

impl TryFrom<RawGroupData> for Group {
    type Error = ApiError;

    /// Cast raw data to a Group
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw data returned from the db including name, MembersLists,
    ///           and the data map and analysts
    fn try_from(raw: RawGroupData) -> Result<Self, Self::Error> {
        let (name, members, data, analysts) = raw;
        // build our owners object
        let owners = GroupUsers {
            combined: HashSet::from_iter(members.0),
            direct: HashSet::from_iter(members.1),
            metagroups: HashSet::from_iter(members.2),
        };
        // build our managers object
        let managers = GroupUsers {
            combined: HashSet::from_iter(members.3),
            direct: HashSet::from_iter(members.4),
            metagroups: HashSet::from_iter(members.5),
        };
        // build our users object
        let users = GroupUsers {
            combined: HashSet::from_iter(members.6),
            direct: HashSet::from_iter(members.7),
            metagroups: HashSet::from_iter(members.8),
        };
        // build our monitors object
        let monitors = GroupUsers {
            combined: HashSet::from_iter(members.9),
            direct: HashSet::from_iter(members.10),
            metagroups: HashSet::from_iter(members.11),
        };
        // cast to a group object
        let group = Group {
            name,
            owners,
            managers,
            analysts,
            users,
            monitors,
            description: deserialize_opt!(data, "description"),
            allowed: deserialize_ext!(data, "allowed", GroupAllowed::default()),
        };
        Ok(group)
    }
}

// The lifetime required for &HashSet appear to confuse the compiler so just
// write this tuple out manually instead of using a type
impl
    TryFrom<(
        String,
        MembersLists,
        HashMap<String, String>,
        &HashSet<String>,
    )> for Group
{
    type Error = ApiError;

    /// Cast raw data to a Group where analysts is a &hashset
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw data returned from the db including name, MembersLists,
    ///           and the data map, and analysts
    fn try_from(
        raw: (
            String,
            MembersLists,
            HashMap<String, String>,
            &HashSet<String>,
        ),
    ) -> Result<Self, Self::Error> {
        let (name, members, data, analysts) = raw;
        // build our owners object
        let owners = GroupUsers {
            combined: HashSet::from_iter(members.0),
            direct: HashSet::from_iter(members.1),
            metagroups: HashSet::from_iter(members.2),
        };
        // build our managers object
        let managers = GroupUsers {
            combined: HashSet::from_iter(members.3),
            direct: HashSet::from_iter(members.4),
            metagroups: HashSet::from_iter(members.5),
        };
        // build our users object
        let users = GroupUsers {
            combined: HashSet::from_iter(members.6),
            direct: HashSet::from_iter(members.7),
            metagroups: HashSet::from_iter(members.8),
        };
        // build our monitors object
        let monitors = GroupUsers {
            combined: HashSet::from_iter(members.9),
            direct: HashSet::from_iter(members.10),
            metagroups: HashSet::from_iter(members.11),
        };
        // cast to a group object
        let group = Group {
            name,
            owners,
            managers,
            analysts: analysts.clone(),
            users,
            monitors,
            description: deserialize_opt!(data, "description"),
            allowed: deserialize_ext!(data, "allowed", GroupAllowed::default()),
        };
        Ok(group)
    }
}

// Build the ldap filter for a specific key/value
//
// # Arguments
//
// * `arr` - The source array of objects
// * `key` - The name of the value to filter on
// * `target` - The target hashset to insert into
macro_rules! build_filter {
    ($arr:expr, $key:expr, $target:expr) => {
        for item in $arr.iter() {
            $target.insert(format!("({}={})", $key, item));
        }
    };
}

/// A map of users in ldap metagroups
#[derive(Debug)]
pub(super) struct LdapUserMap {
    users: HashMap<String, Vec<String>>,
}

impl LdapUserMap {
    /// Crawls all groups in Thorium for ldap metagroups and builds a metagroup user map
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to sync against LDAP
    /// * `valid_users` - A set of existing users in Thorium
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "LdapUserMap::new", skip_all, err(Debug))]
    pub async fn new(
        groups: &[&Group],
        valid_users: &HashSet<String>,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        if let Some(conf) = &shared.config.thorium.auth.ldap {
            // crawl over all groups and build a list of filters
            let mut group_filters = HashSet::new();
            // build a set of all ldap based groups to sync
            for group in groups.iter() {
                build_filter!(&group.owners.metagroups, "cn", group_filters);
                build_filter!(&group.managers.metagroups, "cn", group_filters);
                build_filter!(&group.users.metagroups, "cn", group_filters);
                build_filter!(&group.monitors.metagroups, "cn", group_filters);
            }
            // if we built no filters then return an empty user map
            if group_filters.is_empty() {
                return Ok(LdapUserMap {
                    users: HashMap::new(),
                });
            }
            // build the specific group filters
            let group_filters = itertools::join(&group_filters, "");
            let filter = format!("(&(|{})(objectClass=*))", group_filters);
            //  build an ldap connection
            //  we do this on demand instead of having it in shared because it needs to be mutable
            let (conn, mut ldap) = ldap!(conf).await?;
            ldap3::drive!(conn);
            // if we have an ldap service account then bind with that
            super::users::bind_to_ldap_system_user(&mut ldap, conf).await?;
            // start searching ldap
            let mut stream = ldap
                .streaming_search(&conf.scope, Scope::Subtree, &filter, vec!["*"])
                .await?;
            // crawl over ldap entries and get the users in these ldap metagroups
            let mut user_map = HashMap::new();
            while let Some(entry) = stream.next().await? {
                // cast this to a search entry
                let mut entry = SearchEntry::construct(entry);
                // get this groups name  and users if we can extract it
                if let (Some(name), Some(users)) = (
                    Self::group_name_extract(&entry.dn),
                    entry.attrs.remove(&conf.group_members_attr),
                ) {
                    // extract our usernames if necessary
                    let mut users = Self::group_member_extract(users, conf);
                    // trim the user list down to only valid Thorium users
                    users.retain(|name| valid_users.contains(name));
                    user_map.insert(name.to_owned(), users);
                }
            }
            // finish and abandon the search to prevent wasted resources on the ldap server
            stream.finish().await.success()?;
            let msgid = stream.ldap_handle().last_id();
            ldap.abandon(msgid).await?;
            ldap.unbind().await?;
            Ok(LdapUserMap { users: user_map })
        } else {
            unavailable!("Ldap is not configured!".to_owned())
        }
    }

    /// extract a group name from an ldap search entry
    ///
    /// # Arguments
    ///
    /// * `raw` - The utf-8 encoded ldap search entry
    fn group_name_extract(raw: &str) -> Option<&str> {
        // make this starts with a cn=
        if raw.starts_with("cn=") {
            // set the start to just past the cn=
            let mut start = 3;
            loop {
                // get the index of the first ','
                if let Some(end) = raw[start..].find(',') {
                    // check to see if this comma was escaped
                    if &raw[end - 1..end - 1] == "\\" {
                        // this comma was escaped skip it and keep looking
                        start = end;
                    } else {
                        // we found the end index of our group name return it
                        // add start to our end value since end is a relative index
                        return Some(&raw[3..start + end]);
                    }
                } else {
                    return None;
                }
            }
        } else {
            // this ldap entry does not start with cn= so ignore it
            None
        }
    }

    /// Extract a username from a group membership row
    #[instrument(name = "LdapUserMap::group_member_extract", skip_all)]
    fn group_member_extract<'a>(raw: Vec<String>, conf: &crate::conf::Ldap) -> Vec<String> {
        // get the field to extract our group members usernames from
        if let Some(key_name) = &conf.group_member_field {
            // build a list to store our extracted usernames
            let mut extracted = Vec::with_capacity(raw.len());
            // add the = to our field
            let key = format!("{}=", key_name);
            // crawl over our group member rows
            for row in raw.iter() {
                // try to extract the username field from this row
                if let Some(field) = row.split(',').find(|field| field.starts_with(&key)) {
                    // remove the field key form our username field
                    let username = field[key.len()..].to_owned();
                    // add our extracted username
                    extracted.push(username);
                } else {
                    // log that we failed to extract a username
                    event!(
                        Level::ERROR,
                        error = true,
                        error_msg = "Failed to extract username",
                        row
                    );
                }
            }
            extracted
        } else {
            raw
        }
    }

    /// Apply this update to a target roles groups
    pub fn apply_to_role(&self, role: &mut GroupUsers) {
        // crawl the metagroups that have this role and add them
        for metagroup in role.metagroups.iter() {
            // get the users in this metagroup
            if let Some(users) = self.users.get(metagroup) {
                // extend our combined users with the users in this metagroup
                role.combined
                    .extend(users.iter().map(|name| name.to_owned()));
            }
        }
    }
}
