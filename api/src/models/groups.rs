use std::collections::{HashMap, HashSet};

use super::PipelineStats;
use crate::{
    matches_adds, matches_clear, matches_clear_opt, matches_removes, matches_set,
    matches_update_opt, same,
};

/// The users and metagroups to add to a specific role in a group
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupUsersRequest {
    /// The users that were directly added to this group
    #[serde(default)]
    pub direct: HashSet<String>,
    /// The metagroups that should have this role
    #[serde(default)]
    pub metagroups: HashSet<String>,
}

impl GroupUsersRequest {
    /// Adds a direct user to this group
    pub fn direct<T: Into<String>>(mut self, user: T) -> Self {
        // cast and add this user
        self.direct.insert(user.into());
        self
    }

    /// Adds a direct user to this group
    pub fn metagroup<T: Into<String>>(mut self, metagroup: T) -> Self {
        // cast and add this metagroup
        self.metagroups.insert(metagroup.into());
        self
    }
}

/// The type of action to check if its allowed
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum GroupAllowAction {
    /// File actions
    Files,
    /// Repo actions
    Repos,
    /// Tag Actions
    Tags,
    /// Image actions
    Images,
    /// Pipeline actions
    Pipelines,
    /// Reaction actions
    Reactions,
    /// Result actions
    Results,
    /// Comments actions
    Comments,
}

impl std::fmt::Display for GroupAllowAction {
    /// Allow [`GroupAllowActions`] to be displayed
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Files => write!(f, "Files"),
            Self::Repos => write!(f, "Repos"),
            Self::Tags => write!(f, "Tags"),
            Self::Images => write!(f, "Images"),
            Self::Pipelines => write!(f, "Pipelines"),
            Self::Reactions => write!(f, "Reactions"),
            Self::Results => write!(f, "Results"),
            Self::Comments => write!(f, "Comments"),
        }
    }
}

/// The data that is allowed to be added/uploaded to a group
///
/// These permission are not retroactive.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupAllowed {
    /// Whether files are allowed to be added to this group
    pub files: bool,
    /// Whether repos are allowed to be added to this group
    pub repos: bool,
    /// Whether tags are allowed to be added to this group
    pub tags: bool,
    /// Whether images are allowed to be added to this group
    pub images: bool,
    /// Whether pipelines are allowed to be added to this group
    pub pipelines: bool,
    /// Whether reactions are allowed to be created in this group
    pub reactions: bool,
    /// Whether results are allowed to be added to this group
    pub results: bool,
    /// Whether comments are allowed to be added to this group
    pub comments: bool,
}

impl Default for GroupAllowed {
    /// All actions are allowed in a group by default
    fn default() -> Self {
        // create a group allowed structure where everything is allowed by default
        GroupAllowed {
            files: true,
            repos: true,
            tags: true,
            images: true,
            pipelines: true,
            reactions: true,
            results: true,
            comments: true,
        }
    }
}

impl GroupAllowed {
    /// Disable files being added to a group
    pub fn disable_files(mut self) -> Self {
        self.files = false;
        self
    }

    /// Enable files being added to a group
    pub fn enable_files(mut self) -> Self {
        self.files = true;
        self
    }

    /// Disable repos being added to a group
    pub fn disable_repos(mut self) -> Self {
        self.repos = false;
        self
    }

    /// Enable repos being added to a group
    pub fn enable_repos(mut self) -> Self {
        self.repos = true;
        self
    }

    /// Disable tags being added to a group
    pub fn disable_tags(mut self) -> Self {
        self.tags = false;
        self
    }

    /// Enable tags being added to a group
    pub fn enable_tags(mut self) -> Self {
        self.tags = true;
        self
    }

    /// Disable images being added to a group
    pub fn disable_images(mut self) -> Self {
        self.images = false;
        self
    }

    /// Enable images being added to a group
    pub fn enable_images(mut self) -> Self {
        self.images = true;
        self
    }

    /// Disable pipelines being added to a group
    pub fn disable_pipelines(mut self) -> Self {
        self.pipelines = false;
        self
    }

    /// Enable pipelines being added to a group
    pub fn enable_pipelines(mut self) -> Self {
        self.pipelines = true;
        self
    }

    /// Disable reactions being created in a group
    pub fn disable_reactions(mut self) -> Self {
        self.reactions = false;
        self
    }

    /// Enable reactions being created in a group
    pub fn enable_reactions(mut self) -> Self {
        self.reactions = true;
        self
    }

    /// Disable results being added to a group
    pub fn disable_results(mut self) -> Self {
        self.results = false;
        self
    }

    /// Enable results being added to a group
    pub fn enable_results(mut self) -> Self {
        self.results = true;
        self
    }

    /// Disable comments being added to a group
    pub fn disable_comments(mut self) -> Self {
        self.comments = false;
        self
    }

    /// Enable comments being added to a group
    pub fn enable_comments(mut self) -> Self {
        self.comments = true;
        self
    }

    /// Determine if this action is allowed
    pub fn is_allowed(&self, action: GroupAllowAction) -> bool {
        match action {
            GroupAllowAction::Files => self.files,
            GroupAllowAction::Repos => self.repos,
            GroupAllowAction::Tags => self.tags,
            GroupAllowAction::Images => self.images,
            GroupAllowAction::Pipelines => self.pipelines,
            GroupAllowAction::Reactions => self.reactions,
            GroupAllowAction::Results => self.results,
            GroupAllowAction::Comments => self.comments,
        }
    }
}

/// Group creation struct
///
/// Groups are how Thorium will let users permission their pipelines and reactions. In
/// order for another user to use or see your pipeline they must also be in the
/// same group as that pipeline. Users in groups must have one of the following roles:
///
/// | role | abilities |
/// | --- | ---------- |
/// | owners | delete entire group and modify roles |
/// | managers | modify non-owner roles and delete other users jobs/pipelines |
/// | users | create jobs and delete their own jobs/pipelines |
/// | monitors | monitor jobs and pipelines |
///
/// Groups can also be synced against LDAP groups by role but when a role is synced
/// against ldap you can only use LDAP metagroups for that role. This is to resolve
/// consistency problems between manually added users and ldap metagroups.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupRequest {
    /// The name of group
    pub name: String,
    /// Owners of this group.
    #[serde(default)]
    pub owners: GroupUsersRequest,
    /// Managers of this group.
    #[serde(default)]
    pub managers: GroupUsersRequest,
    /// Users of this group
    #[serde(default)]
    pub users: GroupUsersRequest,
    /// Reporters wv
    #[serde(default)]
    pub monitors: GroupUsersRequest,
    /// Group description
    pub description: Option<String>,
    /// The data that is allowed to be added to this group
    #[serde(default)]
    pub allowed: GroupAllowed,
}

impl GroupRequest {
    /// Creates a new [`GroupRequest`]
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the group to create
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupRequest, GroupUsersRequest};
    ///
    /// let request = GroupRequest::new("CornGroup")
    ///     .owners(GroupUsersRequest::default()
    ///                .direct("mcarson")
    ///                .metagroup("owners"))
    ///     .managers(GroupUsersRequest::default()
    ///                .direct("bob")
    ///                .metagroup("managers"))
    ///     .users(GroupUsersRequest::default()
    ///                .direct("sarah")
    ///                .metagroup("users"))
    ///     .monitors(GroupUsersRequest::default()
    ///                .direct("CornBot")
    ///                .metagroup("monitors"));
    /// ```
    pub fn new<T: Into<String>>(name: T) -> Self {
        GroupRequest {
            name: name.into(),
            owners: GroupUsersRequest::default(),
            managers: GroupUsersRequest::default(),
            users: GroupUsersRequest::default(),
            monitors: GroupUsersRequest::default(),
            description: None,
            allowed: GroupAllowed::default(),
        }
    }

    /// Sets the owners that should be specified in a [`GroupRequest`]
    ///
    /// Owners can delete the group and add/delete group managers.
    ///
    /// # Arguments
    ///
    /// * `users` - The users that should be owners in this new group
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupRequest, GroupUsersRequest};
    ///
    /// let request = GroupRequest::new("CornGroup")
    ///     .owners(GroupUsersRequest::default()
    ///                .direct("mcarson")
    ///                .metagroup("owners"));
    /// ```
    pub fn owners(mut self, users: GroupUsersRequest) -> Self {
        // inject owners
        self.owners = users;
        self
    }

    /// Sets the managers that should be specified in a [`GroupRequest`]
    ///
    /// Managers cannot delete the group but can add users and delete users jobs within this group.
    ///
    /// # Arguments
    ///
    /// * `users` - The users that should be managers in this new group
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupRequest, GroupUsersRequest};
    ///
    /// let request = GroupRequest::new("CornGroup")
    ///     .managers(GroupUsersRequest::default()
    ///                .direct("bob")
    ///                .metagroup("managers"));
    /// ```
    pub fn managers(mut self, users: GroupUsersRequest) -> Self {
        // inject managers
        self.managers = users;
        self
    }

    /// Sets the users that should be specified in a [`GroupRequest`]
    ///
    /// Users can create jobs and delete their own jobs but cannot add/delete users.
    ///
    /// # Arguments
    ///
    /// * `users` - The users that should be users in this new group
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupRequest, GroupUsersRequest};
    ///
    /// let request = GroupRequest::new("CornGroup")
    ///     .users(GroupUsersRequest::default()
    ///                .direct("sally")
    ///                .metagroup("users"));
    /// ```
    pub fn users(mut self, users: GroupUsersRequest) -> Self {
        // inject users
        self.users = users;
        self
    }

    /// Sets the monitors that should be specified in a [`GroupRequest`]
    ///
    /// Monitors cannot create/delete jobs/users but can monitor the jobs and users within a group
    ///
    /// # Arguments
    ///
    /// * `users` - The users that should be monitors in this new group
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupRequest, GroupUsersRequest};
    ///
    /// let request = GroupRequest::new("CornGroup")
    ///     .monitors(GroupUsersRequest::default()
    ///                .direct("sally")
    ///                .metagroup("users"));
    /// ```
    pub fn monitors(mut self, users: GroupUsersRequest) -> Self {
        // inject reporters
        self.monitors = users;
        self
    }

    /// Sets the description that should be specified in a [`GroupRequest`]
    ///
    /// # Arguments
    ///
    /// * `description` - The description of the new group
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupRequest, GroupUsersRequest};
    ///
    /// let request = GroupRequest::new("CornGroup")
    ///     .description("A corn group for corn things");
    /// ```
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Helps serde default the group list limit to 50
fn default_list_limit() -> usize {
    50
}

/// The parameters for a group list request
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupListParams {
    /// The cursor id to user if one exists
    #[serde(default)]
    pub cursor: usize,
    /// The max amount of groups to return in on request
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

/// List of group names with a cursor
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupList {
    /// Cursor used to page through group names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// List of group names
    pub names: Vec<String>,
}

/// List of group details with a cursor
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupDetailsList {
    /// Cursor used to page through group details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// List of group details
    pub details: Vec<Group>,
}

/// A hashmap version of a group details list
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupMap {
    /// Cursor used to page through group details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// List of group details
    pub details: HashMap<String, Group>,
}

/// Allow conversion of group details list to a hashmap
impl From<GroupDetailsList> for GroupMap {
    fn from(mut list: GroupDetailsList) -> Self {
        // extract cursor if one was set
        let cursor = list.cursor.take();
        // build hashmap from list
        let mut details = HashMap::with_capacity(list.details.len());
        list.details.into_iter().for_each(|val| {
            details.insert(val.name.clone(), val);
        });
        GroupMap { cursor, details }
    }
}

/// The users and metagroups to add or remove for a specific role in a group
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupUsersUpdate {
    /// The direct users to add
    #[serde(default)]
    pub direct_add: HashSet<String>,
    /// The direct users to remove
    #[serde(default)]
    pub direct_remove: HashSet<String>,
    /// The metagroups to add
    #[serde(default)]
    pub metagroups_add: HashSet<String>,
    /// The metagroups to remove
    #[serde(default)]
    pub metagroups_remove: HashSet<String>,
}

impl GroupUsersUpdate {
    /// Adds a direct user to this group role
    pub fn direct_add<T: Into<String>>(mut self, user: T) -> Self {
        // cast and add this user
        self.direct_add.insert(user.into());
        self
    }

    /// Removes a direct user from this group role
    pub fn direct_remove<T: Into<String>>(mut self, user: T) -> Self {
        // cast and add this user to remove
        self.direct_remove.insert(user.into());
        self
    }

    /// Adds a metagroup to this group role
    pub fn metagroup_add<T: Into<String>>(mut self, metagroup: T) -> Self {
        // cast and add this metagroup
        self.metagroups_add.insert(metagroup.into());
        self
    }

    /// Removes a metagroup from this group role
    pub fn metagroup_remove<T: Into<String>>(mut self, metagroup: T) -> Self {
        // cast and add this metagroup to remove
        self.metagroups_remove.insert(metagroup.into());
        self
    }

    /// Get the total number of changes this role update contains
    pub fn change_count(&self) -> usize {
        self.direct_add.len()
            + self.direct_remove.len()
            + self.metagroups_add.len()
            + self.metagroups_remove.len()
    }

    /// Checks whether this update changes this roles metagroup info
    pub fn updates_metagroups(&self) -> bool {
        !self.metagroups_add.is_empty() || !self.metagroups_remove.is_empty()
    }

    /// Whether this update is empty or not
    pub fn is_empty(&self) -> bool {
        // check if any changes are in this update
        self.direct_add.is_empty()
            && self.direct_remove.is_empty()
            && self.metagroups_add.is_empty()
            && self.metagroups_remove.is_empty()
    }
}

/// Helps default a serde value to false
/// TODO remove this when https://github.com/serde-rs/serde/issues/368 is resolved
fn default_as_false() -> bool {
    false
}

/// The updates to the data that is allowed to be added/uploaded to a group
///
/// These permission are not retroactive.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupAllowedUpdate {
    /// Whether files are allowed to be added to this group
    pub files: Option<bool>,
    /// Whether repos are allowed to be added to this group
    pub repos: Option<bool>,
    /// Whether tags are allowed to be added to this group
    pub tags: Option<bool>,
    /// Whether images are allowed to be added to this group
    pub images: Option<bool>,
    /// Whether pipelines are allowed to be added to this group
    pub pipelines: Option<bool>,
    /// Whether reactions are allowed to be created in this group
    pub reactions: Option<bool>,
    /// Whether results are allowed to be added to this group
    pub results: Option<bool>,
    /// Whether comments are allowed to be added to this group
    pub comments: Option<bool>,
}

impl GroupAllowedUpdate {
    /// Disable files being added to a group
    pub fn disable_files(mut self) -> Self {
        self.files = Some(false);
        self
    }

    /// Enable files being added to a group
    pub fn enable_files(mut self) -> Self {
        self.files = Some(true);
        self
    }

    /// Disable repos being added to a group
    pub fn disable_repos(mut self) -> Self {
        self.repos = Some(false);
        self
    }

    /// Enable repos being added to a group
    pub fn enable_repos(mut self) -> Self {
        self.repos = Some(true);
        self
    }

    /// Disable tags being added to a group
    pub fn disable_tags(mut self) -> Self {
        self.tags = Some(false);
        self
    }

    /// Enable tags being added to a group
    pub fn enable_tags(mut self) -> Self {
        self.tags = Some(true);
        self
    }

    /// Disable images being added to a group
    pub fn disable_images(mut self) -> Self {
        self.images = Some(false);
        self
    }

    /// Enable images being added to a group
    pub fn enable_images(mut self) -> Self {
        self.images = Some(true);
        self
    }

    /// Disable pipelines being added to a group
    pub fn disable_pipelines(mut self) -> Self {
        self.pipelines = Some(false);
        self
    }

    /// Enable pipelines being added to a group
    pub fn enable_pipelines(mut self) -> Self {
        self.pipelines = Some(true);
        self
    }

    /// Disable reactions being added to a group
    pub fn disable_reactions(mut self) -> Self {
        self.reactions = Some(false);
        self
    }

    /// Enable reactions being added to a group
    pub fn enable_reactions(mut self) -> Self {
        self.reactions = Some(true);
        self
    }

    /// Disable results being added to a group
    pub fn disable_results(mut self) -> Self {
        self.results = Some(false);
        self
    }

    /// Enable results being added to a group
    pub fn enable_results(mut self) -> Self {
        self.results = Some(true);
        self
    }

    /// Disable comments being added to a group
    pub fn disable_comments(mut self) -> Self {
        self.comments = Some(false);
        self
    }

    /// Enable comments being added to a group
    pub fn enable_comments(mut self) -> Self {
        self.comments = Some(true);
        self
    }

    /// Check if this update contains any changes
    pub fn is_empty(&self) -> bool {
        self.files.is_none()
            && self.repos.is_none()
            && self.tags.is_none()
            && self.images.is_none()
            && self.pipelines.is_none()
            && self.reactions.is_none()
            && self.results.is_none()
            && self.comments.is_none()
    }
}

/// An update for a group
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupUpdate {
    /// The updates to apply to the owners of this group
    #[serde(default)]
    pub owners: GroupUsersUpdate,
    /// The updates to apply to the managers of this group
    #[serde(default)]
    pub managers: GroupUsersUpdate,
    /// The updates to apply to the users of this group
    #[serde(default)]
    pub users: GroupUsersUpdate,
    /// The updates to apply to the monitors of this group
    #[serde(default)]
    pub monitors: GroupUsersUpdate,
    /// The updated description of this group
    #[serde(default)]
    pub description: Option<String>,
    /// Whether to clear the description or not
    #[serde(default = "default_as_false")]
    pub clear_description: bool,
    /// Update what is allowed in this group
    #[serde(default)]
    pub allowed: GroupAllowedUpdate,
}

impl GroupUpdate {
    /// Update the owners in this group
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to owners
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupUpdate, GroupUsersUpdate};
    ///
    /// GroupUpdate::default()
    ///     .owners(GroupUsersUpdate::default()
    ///         .direct_add("mcarson")
    ///         .direct_remove("not_michael")
    ///         .metagroup_add("new_metagroup")
    ///         .metagroup_remove("old_metagroup"));
    /// ```
    pub fn owners(mut self, update: GroupUsersUpdate) -> Self {
        // add the owners update
        self.owners = update;
        self
    }

    /// Update the managers in this group
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to managers
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupUpdate, GroupUsersUpdate};
    ///
    /// GroupUpdate::default()
    ///     .managers(GroupUsersUpdate::default()
    ///         .direct_add("mcarson")
    ///         .direct_remove("not_michael")
    ///         .metagroup_add("new_metagroup")
    ///         .metagroup_remove("old_metagroup"));
    /// ```
    pub fn managers(mut self, update: GroupUsersUpdate) -> Self {
        // add the managers update
        self.managers = update;
        self
    }

    /// Update the users in this group
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to users
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupUpdate, GroupUsersUpdate};
    ///
    /// GroupUpdate::default()
    ///     .users(GroupUsersUpdate::default()
    ///         .direct_add("mcarson")
    ///         .direct_remove("not_michael")
    ///         .metagroup_add("new_metagroup")
    ///         .metagroup_remove("old_metagroup"));
    /// ```
    pub fn users(mut self, update: GroupUsersUpdate) -> Self {
        // add the users update
        self.users = update;
        self
    }

    /// Update the monitors in this group
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to monitors
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupUpdate, GroupUsersUpdate};
    ///
    /// GroupUpdate::default()
    ///     .monitors(GroupUsersUpdate::default()
    ///         .direct_add("mcarson")
    ///         .direct_remove("not_michael")
    ///         .metagroup_add("new_metagroup")
    ///         .metagroup_remove("old_metagroup"));
    /// ```
    pub fn monitors(mut self, update: GroupUsersUpdate) -> Self {
        // add the monitors update
        self.monitors = update;
        self
    }

    /// Update the group description
    ///
    /// # Arguments
    ///
    /// * `description` - The new group description
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{GroupUpdate, GroupUsersUpdate};
    ///
    /// GroupUpdate::default()
    ///     .description("This is my new description");
    /// ```
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        // add the monitors update
        self.description = Some(description.into());
        self
    }

    /// Sets the clear description flag to true
    ///
    /// This will clear the group's current description and set it to None.
    ///
    /// ```
    /// use thorium::models::GroupUpdate;
    ///
    /// GroupUpdate::default().clear_description();
    /// ```
    pub fn clear_description(mut self) -> Self {
        self.clear_description = true;
        self
    }

    /// Check if this is update is empty
    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
            && self.managers.is_empty()
            && self.users.is_empty()
            && self.monitors.is_empty()
            && self.description.is_none()
            && !self.clear_description
            && self.allowed.is_empty()
    }

    /// Check if a group update just removes a user
    ///
    /// # Arguments
    ///
    /// * `user` - The user to check
    pub fn removes_only_user(&self, user: &str) -> bool {
        // crawl each of the roles and check them
        for role in &[&self.owners, &self.managers, &self.users, &self.monitors] {
            // get the number of changes in this role and check if its above 1 or not
            match role.change_count() {
                0 => (),
                1 => {
                    // there is only 1 change so make sure our remove username only contains the target user
                    if !role.direct_remove.contains(user) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        // we only removed ourselves from roles
        true
    }

    /// Check if this update contains an update to any roles metagroup info
    pub fn will_update_metagroups(&self) -> bool {
        self.owners.updates_metagroups()
            || self.managers.updates_metagroups()
            || self.users.updates_metagroups()
            || self.monitors.updates_metagroups()
    }
}

#[derive(PartialEq, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum Roles {
    /// Can delete the entire group and modify roles
    Owner,
    /// Can modify non-owner roles and delete users jobs/pipelines
    Manager,
    /// Can create jobs and delete their own jobs/pipelines
    Analyst,
    /// Can create jobs and delete their own jobs/pipelines
    User,
    /// Can monitor jobs and pipelines
    Monitor,
    /// When a user is not a member of this group and has no role
    NotAMember,
}

impl std::fmt::Display for Roles {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Roles::Owner => write!(f, "Owner"),
            Roles::Manager => write!(f, "Manager"),
            Roles::Analyst => write!(f, "Analyst"),
            Roles::User => write!(f, "User"),
            Roles::Monitor => write!(f, "Monitor"),
            Roles::NotAMember => write!(f, "Not a member"),
        }
    }
}

/// The different users and groups that have a specific role in a group
#[derive(Serialize, Deserialize, Clone, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupUsers {
    /// The combined direct users and members of metagroups for this role
    pub combined: HashSet<String>,
    /// The users that were directly added to this group
    pub direct: HashSet<String>,
    /// The metagroups that should have this role
    pub metagroups: HashSet<String>,
}

impl PartialEq<GroupUsersRequest> for GroupUsers {
    /// Check if a [`GroupUsersRequest`] and a [`GroupUsers`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The GroupUsersRequest to compare against
    fn eq(&self, request: &GroupUsersRequest) -> bool {
        // make sure our direct users match
        matches_set!(self.direct, request.direct);
        // make sure our metagroups match
        matches_set!(self.metagroups, request.metagroups);
        true
    }
}

impl PartialEq<GroupUsersUpdate> for GroupUsers {
    /// Check if a [`GroupUsersUpdate`] and a [`GroupUsers`] are equal
    ///
    /// # Arguments
    ///
    /// * `update` - The GroupUsersUpdate to compare against
    fn eq(&self, update: &GroupUsersUpdate) -> bool {
        // make sure we added all of our requested direct users
        matches_adds!(self.direct, update.direct_add);
        // make sure our we removed the requested direct users
        matches_removes!(self.direct, update.direct_remove);
        // make sure we added all of our requested metagroups
        matches_adds!(self.metagroups, update.metagroups_add);
        // make sure our we removed the requested metagroups
        matches_removes!(self.metagroups, update.metagroups_remove);
        true
    }
}

/// A group that contains pipelines, reactions, and images
///
/// Groups are how Thorium will let users permission their pipelines and reactions. In
/// order for another user to use or see your pipeline they must also be in the
/// same group as that pipeline. Users in groups must have one of the following roles:
///
/// | role | abilities |
/// | --- | ---------- |
/// | owners | delete entire group and modify roles |
/// | managers | modify non-owner roles and delete other users jobs/pipelines |
/// | analysts | Create jobs and delete their own jobs/pipelines |
/// | users | create jobs and delete their own jobs/pipelines |
/// | monitors | monitor jobs and pipelines |
#[derive(Serialize, Deserialize, Clone, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Group {
    /// The name of group
    pub name: String,
    /// Owners of this group.
    pub owners: GroupUsers,
    /// Managers of this group
    pub managers: GroupUsers,
    /// All analysts in Thorium
    #[serde(default)]
    pub analysts: HashSet<String>,
    /// Users of this group.
    pub users: GroupUsers,
    /// Reporters of this group.
    pub monitors: GroupUsers,
    /// Description of the group,
    pub description: Option<String>,
    /// The data that is allowed to be added to this group
    #[serde(default)]
    pub allowed: GroupAllowed,
}

impl Group {
    /// Count the total number of members in this group
    pub fn member_count(&self) -> usize {
        self.owners.combined.len()
            + self.managers.combined.len()
            + self.analysts.len()
            + self.users.combined.len()
            + self.monitors.combined.len()
    }

    /// Get the current role of this user
    ///
    /// # Arguments
    ///
    /// * `user` - The whose role we are trying to get
    pub fn role(&self, user: &String) -> Roles {
        if self.owners.combined.contains(user) {
            Roles::Owner
        } else if self.managers.combined.contains(user) {
            Roles::Manager
        } else if self.analysts.contains(user) {
            Roles::Analyst
        } else if self.users.combined.contains(user) {
            Roles::User
        } else if self.monitors.combined.contains(user) {
            Roles::Monitor
        } else {
            Roles::NotAMember
        }
    }

    /// Get the current role of this ldap metagroup
    pub fn ldap_role(&self, user: &String) -> Roles {
        if self.owners.metagroups.contains(user) {
            Roles::Owner
        } else if self.managers.metagroups.contains(user) {
            Roles::Manager
        } else if self.users.metagroups.contains(user) {
            Roles::User
        } else if self.monitors.metagroups.contains(user) {
            Roles::Monitor
        } else {
            Roles::NotAMember
        }
    }

    /// Build a vector of all members of this group
    pub fn members(&self) -> Vec<&String> {
        self.owners
            .combined
            .iter()
            .chain(&self.managers.combined)
            .chain(&self.analysts)
            .chain(&self.users.combined)
            .chain(&self.monitors.combined)
            .collect()
    }

    /// Check if this group has any metagroups based roles set
    pub fn metagroup_enabled(&self) -> bool {
        !self.owners.metagroups.is_empty()
            || !self.managers.metagroups.is_empty()
            || !self.users.metagroups.is_empty()
            || !self.monitors.metagroups.is_empty()
    }
}

impl PartialEq<GroupRequest> for Group {
    /// Check if a [`GroupRequest`] and a [`Group`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The GroupRequest to compare against
    fn eq(&self, request: &GroupRequest) -> bool {
        // make sure the name is the same
        same!(self.name, request.name);
        // make sure user types are the same
        same!(self.owners, request.owners);
        same!(self.managers, request.managers);
        same!(self.users, request.users);
        same!(self.monitors, request.monitors);
        same!(self.description, request.description);
        true
    }
}

impl PartialEq<GroupUpdate> for Group {
    /// Check if a [`Group`] contains all the updates from a [`GroupUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The GroupUpdate to compare against
    #[rustfmt::skip]
    fn eq(&self, update: &GroupUpdate) -> bool {
        // make sure all of our roles were updated
        same!(self.owners, update.owners);
        same!(self.managers, update.managers);
        same!(self.users, update.users);
        same!(self.monitors, update.monitors);
        matches_clear_opt!(self.description, update.description, update.clear_description);
        true
    }
}

/// A status summary for a specific group
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GroupStats {
    /// A map of status summaries for each pipeline in this group
    pub pipelines: HashMap<String, PipelineStats>,
}

impl GroupStats {
    /// gets a total count for the number of stages in use by all pipelines and users
    pub fn total(&self) -> usize {
        // add all the number of stage up for each user and pipeline
        self.pipelines.values().map(|map| map.total()).sum()
    }
}
