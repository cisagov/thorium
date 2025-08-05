//! Wrappers for interacting with users within Thorium with different backends

use chrono::prelude::*;
use schemars::JsonSchema;

use crate::{matches_vec, same};

/// The key used to bootstrap cluster when no admins are loaded
#[derive(Serialize, Deserialize, Debug)]
pub struct Key {
    /// The secret key used to bootstrap the cluster
    pub key: String,
}

/// The roles a user can have
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum UserRole {
    /// An admin can see all data in Thorium and perform any action
    Admin,
    /// An Analyst is given access to all samples in Thorium
    Analyst,
    /// A developer can create tools
    Developer {
        #[serde(default)]
        k8s: bool,
        #[serde(default)]
        bare_metal: bool,
        #[serde(default)]
        windows: bool,
        #[serde(default)]
        external: bool,
        #[serde(default)]
        kvm: bool,
    },
    /// A user can upload files and run jobs
    User,
}

impl Default for UserRole {
    /// Create a default user role of user
    fn default() -> Self {
        UserRole::User
    }
}

/// Data needed to register a user
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct UserCreate {
    /// The username of the user
    pub username: String,
    /// The password for the user
    pub password: String,
    /// This users email
    pub email: String,
    /// The role of this user
    #[serde(default)]
    pub role: UserRole,
    /// Whether this is a local account or not
    #[serde(default)]
    pub local: bool,
    /// The settings this user has set
    #[serde(default)]
    pub settings: UserSettings,
    /// Skip email verification (requires secret key)
    #[serde(default)]
    pub skip_verification: bool,
}

impl UserCreate {
    /// Create a [`UserCreate`] object
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user to create
    /// * `password` - The new users password
    /// * `email` - An email address for this user
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::UserCreate;
    ///
    /// UserCreate::new("mcarson", "Guest", "mcarson@sandia.gov");
    /// ```
    pub fn new<T: Into<String>, P: Into<String>, E: Into<String>>(
        username: T,
        password: P,
        email: E,
    ) -> Self {
        UserCreate {
            username: username.into(),
            password: password.into(),
            email: email.into(),
            role: UserRole::default(),
            local: false,
            settings: UserSettings::default(),
            skip_verification: false,
        }
    }

    /// Sets the users role
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{UserCreate, UserRole};
    ///
    /// UserCreate::new("mcarson", "Guest", "email@email.com")
    ///     .role(
    ///        UserRole::Developer {
    ///            k8s: true,
    ///            bare_metal: false,
    ///            windows: false,
    ///            external: false,
    ///            kvm: false,
    ///        });
    /// ```
    pub fn role(mut self, role: UserRole) -> Self {
        self.role = role;
        self
    }

    /// Sets a [`UserCreate`] to create an admin instead of a user
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::UserCreate;
    ///
    /// UserCreate::new("mcarson", "Guest", "email@email.com")
    ///     .admin();
    /// ```
    pub fn admin(mut self) -> Self {
        self.role = UserRole::Admin;
        self
    }

    /// Set the settings for this new user
    ///
    /// # Arguments
    ///
    /// * `settings` - The settings to set for this user
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{UserCreate, UserSettings, Theme};
    ///
    /// UserCreate::new("mcarson", "Guest", "email@email.com")
    ///    .settings(UserSettings::default().theme(Theme::Dark));
    /// ```
    pub fn settings(mut self, settings: UserSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Skip email verification when creating this user
    ///
    /// # Arguments
    ///
    /// * `settings` - The settings to set for this user
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{UserCreate, UserSettings, Theme};
    ///
    /// UserCreate::new("mcarson", "Guest", "email@email.com").skip_verification();
    /// ```
    pub fn skip_verification(mut self) -> Self {
        self.skip_verification = true;
        self
    }
}

/// An update for this user
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct UserUpdate {
    /// The updated password for a user
    pub password: Option<String>,
    /// The email address for this user
    pub email: Option<String>,
    /// The role to set for the user
    pub role: Option<UserRole>,
    /// The settings to set for this user
    pub settings: Option<UserSettingsUpdate>,
}

/// The info to inject about this user on a Unix/Linux system
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct UnixInfo {
    /// The unix user id of this user
    pub user: u64,
    /// The unix group id of this user
    pub group: u64,
}

/// The different themes supported in the webUI
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum Theme {
    /// A dark theme
    Dark,
    /// A light theme
    Light,
    /// A crab theme
    Crab,
    /// A dark blue theme
    Ocean,
    /// Use the theme closest to browser settings
    Automatic,
}

impl Default for Theme {
    /// Set the default theme to Automatic
    fn default() -> Theme {
        Theme::Automatic
    }
}

/// Any user specific settings
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct UserSettings {
    /// The theme this user uses in the webUI
    pub theme: Theme,
}

impl UserSettings {
    /// Update the theme for this user
    ///
    /// # Arguments
    ///
    /// * `theme` - The theme to set
    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }
}

/// The update to apply to a users settings
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct UserSettingsUpdate {
    /// The theme this user uses in the webUI
    pub theme: Option<Theme>,
}

impl UserSettingsUpdate {
    /// Update the theme for this user
    ///
    /// # Arguments
    ///
    /// * `theme` - The theme to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{UserSettingsUpdate, Theme};
    ///
    /// UserSettingsUpdate::default().theme(Theme::Dark);
    /// ```
    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = Some(theme);
        self
    }
}

/// A user within Thorium
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct User {
    /// The username of this user
    pub username: String,
    /// This users password if ldap is not being used
    pub password: Option<String>,
    /// This users email
    pub email: String,
    /// The role for this user (admin || user || service)
    pub role: UserRole,
    /// The groups this user is in
    pub groups: Vec<String>,
    /// The token for this user
    pub token: String,
    /// When this users token expires
    pub token_expiration: DateTime<Utc>,
    /// The info to inject about this user on unix/linux systems
    pub unix: Option<UnixInfo>,
    /// The settings this user has set
    pub settings: UserSettings,
    /// Whether this user has been verified already or not
    pub verified: bool,
    /// The verification token to check against if one has been set
    pub verification_token: Option<String>,
}

/// A user within Thorium that does not have its password
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ScrubbedUser {
    /// The username of this user
    pub username: String,
    /// The role for this user (admin || user)
    pub role: UserRole,
    /// This users email
    pub email: String,
    /// The groups this user is in
    pub groups: Vec<String>,
    /// The token for this user
    pub token: String,
    /// When this users token expires
    pub token_expiration: DateTime<Utc>,
    /// The info to inject about this user on unix/linux systems
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unix: Option<UnixInfo>,
    /// The settings this user has set
    pub settings: UserSettings,
    /// Whether this user is a local user or not
    pub local: bool,
    /// Whether this user has been verified already or not
    pub verified: bool,
}

impl PartialEq<ScrubbedUser> for ScrubbedUser {
    /// Check if a [`ScrubbedUser`] and a [`ScrubbedUser`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The ScrubbedUser to compare against
    fn eq(&self, request: &ScrubbedUser) -> bool {
        // make sure the username is the same
        same!(self.username, request.username);
        // make sure the role is the same
        same!(self.role, request.role);
        // make sure the email is the same
        same!(self.email, request.email);
        // make sure the group list is the same
        matches_vec!(self.groups, request.groups);
        // make sure the token and its expiration are the same
        same!(self.token, request.token);
        same!(self.token_expiration, request.token_expiration);
        // make sure our settings are the same
        same!(self.settings, request.settings);
        // make sure our local user info is the same
        same!(self.local, request.local);
        // make sure our verification is the same
        same!(self.verified, request.verified);
        true
    }
}

/// Response to a successful auth
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct AuthResponse {
    /// The token to use to talk to Thorium
    pub token: String,
    /// The date/time this token expires
    pub expires: DateTime<Utc>,
}
