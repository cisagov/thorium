//! The models for OAuth based auth

use chrono::prelude::*;
use uuid::Uuid;

use crate::models::AuthResponse;

use super::{UserRole, UserSettings};

#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OAuthCallbackParams {
    /// The code to verify this challenge
    pub code: String,
    /// The state for this OAuth authentication session
    pub state: String,
}

#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OAuthLinkParams {
    /// The username of the user we are adding an auth alias too
    pub username: String,
    /// The token for this oauth link request
    pub token: String,
}

/// The info for an authenticated user in OAuth or a registration session
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum OAuthMaybeAuthed {
    /// This user successfully authenticated and has a verified email
    Authed {
        /// The token to use to talk to Thorium
        token: String,
        /// The date/time this token expires
        expires: DateTime<Utc>,
    },
    /// This user successfully authenticated but needs to verify their email
    VerifyEmail(String),
    /// An Oauth registration session token
    NewUser(String),
}

/// An OAuth user creation session
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OAuthRegistrationSessionId {
    /// The id for this session
    pub id: Uuid,
}

/// Data needed to register a user
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OAuthUserCreate {
    /// The token for this oauth registration session
    pub session_token: String,
    /// The username of the user
    pub username: String,
    /// This users email
    pub email: String,
    /// The role of this user
    #[serde(default)]
    pub role: UserRole,
    /// The settings this user has set
    #[serde(default)]
    pub settings: UserSettings,
    /// Skip email verification (requires secret key)
    #[serde(default)]
    pub skip_verification: bool,
}

/// A username and session token to use to check if a username is taken or not
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OAuthUsernameCheck {
    /// The username that conflicted
    pub username: String,
    /// The token for this oauth registration session
    pub session_token: String,
}

impl OAuthUsernameCheck {
    /// Create a new [`OAuthUsernameCheck`]
    ///
    /// # Arguments
    ///
    /// * `username` - The username to check the availability of
    /// * `session_token` - An oauth registration session token
    pub fn new(username: impl Into<String>, session_token: impl Into<String>) -> Self {
        OAuthUsernameCheck {
            username: username.into(),
            session_token: session_token.into(),
        }
    }
}
