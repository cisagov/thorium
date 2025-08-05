//! Wrappers for interacting with users within Thorium with different backends
//! Currently only Redis is supported

use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::{Algorithm, Argon2, Version};
use axum::extract::{FromRef, FromRequestParts};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use chrono::prelude::*;
use headers::{Header, HeaderName, HeaderValue};
use ldap3::{Scope, SearchEntry};
use rand::prelude::*;
use std::collections::HashSet;
use std::str;
use tracing::{event, instrument, Level, Span};

use super::db;
use crate::conf::Ldap;
use crate::models::{
    AuthResponse, Group, ImageScaler, Key, ScrubbedUser, UnixInfo, User, UserCreate, UserRole,
    UserSettingsUpdate, UserUpdate,
};
use crate::utils::shared::EmailClient;
use crate::utils::{bounder, ApiError, AppState, Shared};
use crate::{bad, conflict, is_admin, ldap, unauthorized, unavailable};

/// The header name for our secret key
static SECRET_KEY_HEADER: HeaderName = HeaderName::from_static("secret-key");
static SECRET_KEY_HEADER_REF: &'static HeaderName = &SECRET_KEY_HEADER;

/// Return unauthorized if a function return an error
macro_rules! check_unauth {
    ($func:expr) => {
        match $func {
            Ok(val) => val,
            Err(_) => return crate::unauthorized!(),
        }
    };
}

impl Header for Key {
    fn name() -> &'static HeaderName {
        SECRET_KEY_HEADER_REF
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i HeaderValue>,
    {
        // get our header value
        let value = values.next().ok_or_else(headers::Error::invalid)?;
        // cast this header into a secret key
        let key = Key {
            key: value.to_str().unwrap().to_owned(),
        };
        Ok(key)
    }

    fn encode<E: Extend<HeaderValue>>(&self, values: &mut E) {
        // wrap our secret key
        let value = HeaderValue::from_str(&self.key);
        // add this to the header
        values.extend(value);
    }
}

/// Hashes a users password
#[macro_export]
macro_rules! hash_pw {
    ($raw:expr, $secret_key:expr) => {
        // hash this password with our salt
        Argon2::new_with_secret(
            $secret_key.as_bytes(),
            Algorithm::Argon2id,
            Version::V0x13,
            argon2::Params::default(),
        )?
        .hash_password($raw.as_bytes(), &SaltString::generate(&mut OsRng))?
        .to_string()
    };
}

/// generate a token
macro_rules! token {
    () => {{
        let mut rng = rand::rng();
        let token: [u8; 32] = rng.random();
        hex::encode(token)
    }};
}

/// get the time a new token should expire
macro_rules! token_expire {
    ($shared:expr) => {
        // update token expiration
        Utc::now() + chrono::Duration::days($shared.config.thorium.auth.token_expire as i64)
    };
}

/// Authenticate a user by token
///
/// # Arguments
///
/// * `token` - The token to authenticate with
/// * `shared` - Shared objects in Thorium
#[instrument(name = "backends::user::token_auth", skip_all, err(Debug))]
async fn token_auth<'a>(token: &str, shared: &Shared) -> Result<User, ApiError> {
    // get user
    let mut user = db::users::get_token(token, shared).await?;
    // throw unauthorized if token doesn't match
    // this should only happen if the token map is somehow wrong
    // which should never happen
    if user.token != token {
        event!(Level::ERROR, msg = "Token Map Corruption Likely");
        return unauthorized!();
    }
    // Check if this users token has expired
    if user.token_expiration < Utc::now() {
        // token is expired so generate a new one and bounce this user
        event!(Level::INFO, msg = "Regenerating Expired Token");
        user.regen_token(shared).await?;
        return unauthorized!();
    }
    // return authed user
    Ok(user)
}

/// Authenticate a user with basic auth stored in redis
///
/// # Arguments
///
/// * `possible` - The user data we are authenticating against
/// * `password` - The password to check against
/// * `shared` - Shared objects in Thorium
#[instrument(name = "backends::user::basic_auth_redis", skip_all, err(Debug))]
async fn basic_auth_redis(
    username: &str,
    password: &str,
    password_hash: &str,
    shared: &Shared,
) -> Result<(), ApiError> {
    // parse our password hash
    let parsed_hash = PasswordHash::new(password_hash)?;
    // get our key
    let secret_key = shared.config.thorium.secret_key.as_bytes();
    // build an argon hasher
    let argon = Argon2::new_with_secret(
        secret_key,
        Algorithm::Argon2id,
        Version::V0x13,
        argon2::Params::default(),
    )?;
    // verify this user provided the correct password
    match argon.verify_password(password.as_bytes(), &parsed_hash) {
        Ok(()) => Ok(()),
        Err(error) => {
            // log this authorization failure
            event!(Level::ERROR, user = username, error = error.to_string(),);
            // This user is not authorized to view this route/data
            unauthorized!()
        }
    }
}

/// Authenticate a user with basic auth using LDAP
///
/// # Arguments
///
/// * `username` - The name of the user to authenticate
/// * `password` - The password to authenticate with
/// * `shared` - Shared objects in Thorium
#[instrument(
    name = "backends::user::basic_auth_ldap",
    skip(password, shared),
    err(Debug)
)]
async fn basic_auth_ldap(
    username: &str,
    password: &str,
    shared: &Shared,
) -> Result<ldap3::Ldap, ApiError> {
    if let Some(ldap_conf) = &shared.config.thorium.auth.ldap {
        //  build an ldap connection
        //  we do this on demand instead of having it in shared because it needs to be mutable
        let (conn, mut ldap) = ldap!(ldap_conf).await?;
        ldap3::drive!(conn);
        // try to bind to ldap with this users creds
        let query = format!(
            "{}{username}{}",
            ldap_conf.user_prepend, ldap_conf.user_append
        );
        let res = ldap.simple_bind(&query, password).await?.success();
        // check if the bind failed or ot
        if let Err(err) = res {
            // log this auth failure
            event!(Level::ERROR, user = username, err = &err.to_string());
            // the bind failed return unauthorized
            return unauthorized!();
        }
        Ok(ldap)
    } else {
        unavailable!("ldap is not configured!".to_owned())
    }
}

/// Authenticate a user with basic auth stored in redis or using ldap
///
/// # Arguments
///
/// * `username` - The name of the user to authenticate
/// * `password` - The password to authenticate with
/// * `shared` - Shared objects in Thorium
#[instrument(
    name = "backends::users::password_auth",
    skip(password, shared),
    err(Debug)
)]
async fn password_auth(username: &str, password: &str, shared: &Shared) -> Result<User, ApiError> {
    // get the user doc we are authenticating against
    let mut possible = db::users::get(username, shared).await?;
    event!(
        Level::INFO,
        user = &possible.username,
        msg = "Attempting authentication",
    );
    // try to authenticate against redis or ldap based on if a password is set
    if let Some(password_hash) = &possible.password {
        // a password is set use basic auth
        basic_auth_redis(username, password, password_hash, shared).await?;
    } else {
        // no password was set so use ldap
        let mut ldap = basic_auth_ldap(username, password, shared).await?;
        // if no unix info is set then try to get it and save it
        if possible.unix.is_none() {
            // get this users unix info from ldap
            let unix = get_unix_info(
                username,
                shared.config.thorium.auth.ldap.as_ref().unwrap(),
                &mut ldap,
            )
            .await?;
            // unbind our ldap socket
            ldap.unbind().await?;
            // set our new unix info
            possible.unix = Some(unix);
            // save the updated user object to redis
            db::users::save(&possible, shared).await?;
        }
    }

    // check if our token is expired and regenerate it if it is
    if possible.token_expiration < Utc::now() {
        event!(Level::INFO, msg = "refreshing token");
        // token is expired so generate a new one and bounce this user
        possible.regen_token(shared).await?;
    }
    Ok(possible)
}

/// Build an ldap connection with our system user for listing info (not auth checks)
#[instrument(name = "backends::users::bind_to_ldap_system_user", skip_all, fields(creds_set = ldap_conf.credentials.is_some()), err(Debug))]
pub async fn bind_to_ldap_system_user(
    ldap: &mut ldap3::Ldap,
    ldap_conf: &Ldap,
) -> Result<(), ApiError> {
    // if we have any system creds set then use them
    if let Some(creds) = &ldap_conf.credentials {
        // try to bind to ldap with this users creds
        let query = format!(
            "{}{}{}",
            ldap_conf.user_prepend, creds.user, ldap_conf.user_append
        );
        let res = ldap.simple_bind(&query, &creds.password).await?.success();
        // check if the bind failed or ot
        if let Err(err) = res {
            // log this auth failure
            event!(Level::ERROR, user = &creds.user, err = &err.to_string());
            // the bind failed return unavailable
            return unavailable!(format!("Failed to bind with user {}", &creds.user));
        }
    }
    Ok(())
}

/// Pull the unix info about a user from ldap
///
/// # Arguments
///
/// * `username` - The username to get info
/// * `conf` - The Thorium Ldap config
/// * `ldap` - A bound and authenticated ldap connection
#[instrument(name = "backends::users::get_unix_info", skip(conf, ldap), err(Debug))]
async fn get_unix_info(
    username: &str,
    conf: &Ldap,
    ldap: &mut ldap3::Ldap,
) -> Result<UnixInfo, ApiError> {
    // build a filter to get info on this user
    let filter = format!(
        "(&({}{username})(objectClass=*))",
        conf.search_filter_prepend
    );
    // search for this users info in ldap
    let mut stream = ldap
        .streaming_search(&conf.scope, Scope::Subtree, &filter, vec!["*"])
        .await?;
    // crawl over the search entries for our user and pull their info
    if let Some(entry) = stream.next().await? {
        // try to cast this entry to strings instead of binary arrays
        let mut entry = SearchEntry::construct(entry);
        // get the group and user id if they exist
        let user = conf.user_unix_id.attr.get(&mut entry);
        let group = conf.group_unix_id.attr.get(&mut entry);
        // build UnixInfo object if they exist or error out
        if let (Some(user), Some(group)) = (user, group) {
            // cast our user and group IDs
            let user = conf.user_unix_id.cast.cast(user)?;
            let group = conf.group_unix_id.cast.cast(group)?;
            return Ok(UnixInfo { user, group });
        }
        // we didn't get the attrs we expected so log what we got to jaeger
        let ldap_str = format!("{:#?}", entry);
        event!(Level::ERROR, ldap_entry = ldap_str);
    }
    unavailable!(format!("Ldap did not return UNIX info for {}", username))
}

/// The different support auth methods
enum AuthMethods {
    /// Authenticate with a token
    Token(String),
    /// Authenticate with a password
    Password { username: String, password: String },
}

impl AuthMethods {
    /// Authenticates a user based on an auth header
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "User::authenticate", skip_all, err(Debug))]
    pub async fn authenticate(&self, shared: &Shared) -> Result<User, ApiError> {
        // try to authenticate this user
        let user = match self {
            Self::Token(token) => token_auth(token, shared).await,
            Self::Password { username, password } => {
                password_auth(username, password, shared).await
            }
        }?;
        // make sure this user has been verified
        if !user.verified {
            // our user has not been verified yet so reject this request
            return unauthorized!("Email has not been verified".to_owned());
        }
        Ok(user)
    }

    /// Build our auth method from a str
    ///
    /// # Arguments
    ///
    /// * `raw` - The str to build our auth method from
    /// * `span` - The span to log traces under
    #[instrument(name = "User::from_str", skip_all, err(Debug))]
    pub fn from_str(raw: &str) -> Result<Self, ApiError> {
        // get our current span
        let span = Span::current();
        // get the first index where a space exists
        if let Some(index) = raw.find(' ') {
            // make sure this isn't the last character
            if raw.len() < index + 1 {
                event!(parent: span, Level::ERROR, error = "Auth header is too short", len = raw.len(), index = index);
                return unauthorized!();
            }
            // try to get the correct auth method
            match &raw[..index] {
                "token" | "Token" | "bearer" | "Bearer" => Self::token(&raw[index + 1..]),
                "basic" | "Basic" => Self::password(&raw[index + 1..]),
                _ => {
                    event!(parent: span, Level::ERROR, error = "Unknown auth type", auth_type = &raw[..index]);
                    unauthorized!()
                }
            }
        } else {
            event!(parent: span, Level::ERROR, error = "Couldn't find space to split auth header");
            unauthorized!()
        }
    }

    /// Builds the token auth method
    ///
    /// # Arguments
    ///
    /// * `raw` - The str to pull our token from
    fn token(raw: &str) -> Result<Self, ApiError> {
        // try to decode this token value
        let decoded = b64_decode(raw)?;
        Ok(AuthMethods::Token(decoded))
    }

    /// Builds the password auth method
    ///
    /// # Arguments
    ///
    /// * `raw` - The str to pull our username and password from
    fn password(raw: &str) -> Result<Self, ApiError> {
        // try to decode this token value
        let decoded = b64_decode(raw)?;
        // find the first ':' that we should split on
        if let Some(index) = decoded.find(':') {
            // split this decoded string into a username and password
            let username = decoded[..index].to_owned();
            let password = decoded[index + 1..].to_owned();
            Ok(AuthMethods::Password { username, password })
        } else {
            // we couldn't find a ':' to split on
            unauthorized!()
        }
    }
}

impl UserSettingsUpdate {
    /// Apply any updated settings to this user
    ///
    /// # Arguments
    ///
    /// * `user` - The user to update
    pub fn apply(self, user: &mut User) {
        // update our theme if an update was set
        if let Some(theme) = self.theme {
            user.settings.theme = theme;
        }
    }
}

impl User {
    /// Creates a new user in the backend
    ///
    /// # Arguments
    ///
    /// * `req` - The user registration request
    /// * `key` - The secret key from the request
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "User::create", skip_all, fields(user = req.username, key_set = key.is_some()), err(Debug))]
    pub async fn create(
        mut req: UserCreate,
        key: Option<Key>,
        shared: &Shared,
    ) -> Result<User, ApiError> {
        // ensure username is alphanumeric
        bounder::string_lower(&req.username, "username", 1, 50)?;
        // make sure this isn't using any reserved usernames
        if req.username == "external" {
            return bad!("external is a reserved username".to_owned());
        }
        // make sure this user doesn't already exist
        if User::exists(&req.username, shared).await.is_ok() {
            return conflict!(format!("User {} already exists", req.username));
        }
        // only allow users with the secret key to create admins with this route
        if req.role == UserRole::Admin || req.local {
            // bounce users without the key
            if let Some(key) = key {
                // return unauthorized for users with invalid keys
                if shared.config.thorium.secret_key != key.key {
                    return unauthorized!();
                }
            } else {
                return unauthorized!();
            }
        }
        // if ldap is configured then authenticated against ldap and pull unix info
        let (password, unix) = match (&shared.config.thorium.auth.ldap, req.local) {
            // ldap config is set up and a local account was not requested
            (Some(conf), false) => {
                // authenticate against ldap
                let mut ldap = basic_auth_ldap(&req.username, &req.password, shared).await?;
                // get unix info for this user
                let unix = get_unix_info(&req.username, conf, &mut ldap).await?;
                // unbind our ldap socket
                ldap.unbind().await?;
                // sync this users data
                (None, Some(unix))
            }
            // ldap is not configured or a local account was requested
            (_, _) => {
                // get password from request and replace it with an empty str
                let pw = std::mem::take(&mut req.password);
                // get our secret key
                let key = &shared.config.thorium.secret_key;
                // hash password
                (Some(hash_pw!(pw, key)), None)
            }
        };
        // create user object
        let mut cast = User {
            username: req.username,
            password,
            email: req.email,
            groups: Vec::default(),
            role: req.role,
            token: token!(),
            unix,
            token_expiration: token_expire!(shared),
            settings: req.settings,
            verified: false,
            verification_token: None,
        };
        // send a verification email if needed
        match (req.skip_verification, &shared.email) {
            (true, _) | (_, None) => cast.verified = true,
            (false, Some(client)) => {
                // send our verification email
                cast.send_verification_email(client, shared).await?;
            }
        };
        // inject user into the backend
        let user = db::users::create(cast, shared).await?;
        // sync all groups in ldap if ldap is enabled
        if shared.config.thorium.auth.ldap.is_some() {
            // sync ldap data for all groups
            Group::sync_ldap(shared).await?;
        }
        Ok(user)
    }

    /// Send a verification email to an unverified user
    ///
    /// # Arguments
    ///
    /// *
    pub async fn send_verification_email(
        &mut self,
        client: &EmailClient,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // check if this user has an approved email
        if let Some(approved) = &client.approved {
            if !approved.is_match(&self.email) {
                // reject any emails that are not approved
                return unauthorized!(format!("{} is not an approved email", self.email));
            }
        }
        // if they are not yet verified then send the verification email
        if self.verified {
            // this user is already verified so return an error
            return bad!(format!(
                "{} has already verified their email",
                &self.username
            ));
        }
        // make sure we have email verification settings
        let email_conf = match &shared.config.thorium.auth.email {
            Some(email_conf) => email_conf,
            None => return unavailable!("Email verification is not enabled".to_owned()),
        };
        // generate a special token for our email verification
        let verification_token = token!();
        // save this verification token to redis
        db::users::set_verification_token(&self.username, &verification_token, shared).await?;
        // build our verification link to embed in the email
        let link = format!(
            "{}/users/verify/{}/email/{}",
            email_conf.base_url, self.username, verification_token
        );
        // update our user object
        self.verification_token = Some(verification_token);
        // build the subject for email verification email
        let subject = "🦀🎉 Welcome to Thorium 🎉🦀".to_owned();
        // build a body with our verification email
        let body = format!(
            "Please verify your Thorium account by clicking on the following link:\n\n{link}"
        );
        // send our verification email
        client.send(&self.email, subject, body).await
    }

    /// Verify an email for a user
    ///
    /// # Arguments
    ///
    /// * `verification_token` - The verification token to check
    pub async fn verify_email(
        &mut self,
        verification_token: &String,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // check if our verification token matches
        if Some(verification_token) == self.verification_token.as_ref() {
            // update our user object
            self.verified = true;
            self.verification_token = None;
            // clear this users verification token and set them as verified in redis
            db::users::clear_verification_token(&self.username, shared).await?;
            Ok(())
        } else {
            // this is the wrong verification token
            unauthorized!()
        }
    }

    /// Get a [`User`] by username
    ///
    /// This should only be used when absolutely necessary as it bypasses all authentication checks
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user to get
    /// * `shared` - Shared Thorium objects
    pub async fn force_get(username: &str, shared: &Shared) -> Result<User, ApiError> {
        // get user
        db::users::get(username, shared).await
    }

    /// Lists usernames
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub async fn list(&self, shared: &Shared) -> Result<Vec<String>, ApiError> {
        // list all usernames
        db::users::list(shared).await
    }

    /// Lists users with details
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub async fn list_details(&self, shared: &Shared) -> Result<Vec<ScrubbedUser>, ApiError> {
        // only admins can list user details
        is_admin!(self);
        // just get a backup since we want all users anyways
        let users = db::users::backup(shared).await?;
        // cast these users to scrubbed users
        let scrubbed = users.into_iter().map(ScrubbedUser::from).collect();
        Ok(scrubbed)
    }

    /// Checks if a vector of users exist
    ///
    /// # Arguments
    ///
    /// * `username` - the names of the users to check
    /// * `shared` - Shared Thorium objects
    pub async fn exists_many(usernames: &HashSet<String>, shared: &Shared) -> Result<(), ApiError> {
        // check if a set of users exists
        db::users::exists_many(usernames, shared).await
    }

    /// Checks if a users exists
    ///
    /// # Arguments
    ///
    /// * `username` - the names of the users to check
    /// * `shared` - Shared Thorium objects
    pub async fn exists(usernames: &str, shared: &Shared) -> Result<(), ApiError> {
        // check if this user exists
        db::users::exists(usernames, shared).await
    }

    /// Deletes a user from Thorium
    ///
    /// # Arguments
    ///
    /// * `user` - The user requesting this delete
    /// * `delete` - The username to delete
    /// * `shared` - Shared Thorium objects
    pub async fn delete(user: User, delete: &str, shared: &Shared) -> Result<(), ApiError> {
        // only admins or the user in question can delete their user
        // get the users data to delete
        let target = match (user.is_admin(), user.username == delete) {
            // were an admin deleting ourselves or we aren't and admin and were deleting ourselves
            (true | false, true) => user,
            // were an admin deleting another user
            (true, false) => Self::force_get(delete, shared).await?,
            // we aren't an admin and were aren't deleting ourselves
            _ => return unauthorized!(),
        };

        // delete user in the background
        db::users::delete(&target, shared).await
    }

    /// Checks if a user is an admin
    #[must_use]
    pub fn is_admin(&self) -> bool {
        self.role == UserRole::Admin
    }

    /// Check if this user is an admin or analyst
    #[must_use]
    pub fn is_admin_or_analyst(&self) -> bool {
        self.role == UserRole::Admin || self.role == UserRole::Analyst
    }

    /// Checks if a user is a developer
    ///
    /// # Arguments
    ///
    /// * `scaler` - The image scaler to check that we can develop for
    #[must_use]
    pub fn is_developer(&self, scaler: ImageScaler) -> bool {
        // make sure this user
        match self.role {
            // admins can develop for anything
            UserRole::Admin => true,
            // analysts can develop for anything
            UserRole::Analyst => true,
            // check if this developer can develop on all scalers
            UserRole::Developer {
                k8s,
                bare_metal,
                windows,
                external,
                kvm,
            } => {
                // check if this user can develop for this scaler
                match scaler {
                    ImageScaler::K8s => k8s,
                    ImageScaler::BareMetal => bare_metal,
                    ImageScaler::Windows => windows,
                    ImageScaler::External => external,
                    ImageScaler::Kvm => kvm,
                }
            }
            // this user cannot develop images/pipelines
            UserRole::User => false,
        }
    }

    /// Checks if a user is a developer
    ///
    /// # Arguments
    ///
    /// * `scaler` - The image scaler to check that we can develop for
    #[must_use]
    pub fn is_developer_many(&self, scalers: &[ImageScaler]) -> bool {
        // make sure this user
        match self.role {
            // admins can develop for anything
            UserRole::Admin => true,
            // this user can develop for anything
            UserRole::Analyst => true,
            // check if this developer can develop on all scalers
            UserRole::Developer {
                k8s,
                bare_metal,
                windows,
                external,
                kvm,
            } => {
                // check if this user can develop for this scaler
                for scaler in scalers {
                    // get the correct scaler permission
                    let permission = match scaler {
                        ImageScaler::K8s => k8s,
                        ImageScaler::BareMetal => bare_metal,
                        ImageScaler::Windows => windows,
                        ImageScaler::External => external,
                        ImageScaler::Kvm => kvm,
                    };
                    // if we don't have permission then short circuit
                    if !permission {
                        return false;
                    }
                }
                true
            }
            // this user cannot develop images/pipelines
            UserRole::User => false,
        }
    }

    /// generates a random token
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    fn gen_token(&mut self, shared: &Shared) {
        // update token and its expiration
        self.token = token!();
        self.token_expiration = token_expire!(shared);
    }

    /// Saves a users token into the backend
    ///
    /// # Arguments
    ///
    /// * `old` - The old token for this user
    /// * `shared` - Shared Thorium objects
    async fn save_token(&self, old: &str, shared: &Shared) -> Result<(), ApiError> {
        db::users::save_token(self, old, shared).await
    }

    /// Generate and save a new token for a user
    pub async fn regen_token(&mut self, shared: &Shared) -> Result<(), ApiError> {
        // get our old token
        let old = self.token.clone();
        // generate a new token
        self.gen_token(shared);
        // save our new token
        self.save_token(&old, shared).await?;
        Ok(())
    }

    /// Updates a user
    ///
    /// This will invalidate the user's current token if the
    /// password is updated.
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this user
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "User::update", skip_all, err(Debug))]
    pub async fn update(mut self, update: UserUpdate, shared: &Shared) -> Result<Self, ApiError> {
        // if we are updating our role make sure we are an admin
        if update.role.is_some() {
            // only admins can update roles
            is_admin!(self);
            // update our role
            crate::update!(self.role, update.role);
        }
        // check if we are updating their password
        if let Some(password) = &update.password {
            // disallow password updates for non local accounts
            if shared.config.thorium.auth.ldap.is_none() || self.password.is_some() {
                event!(Level::INFO, msg = "Updating password");
                // get our secret key
                let key = &shared.config.thorium.secret_key;
                // hash password and set a new token
                self.password = Some(hash_pw!(password, key));
                self.gen_token(shared);
            } else {
                return unavailable!("Cannot update password when ldap is enabled".to_string());
            }
        }
        // apply any settings updates
        if let Some(settings) = update.settings {
            settings.apply(&mut self);
        }
        // save update user to the backend
        db::users::save(&self, shared).await?;
        Ok(self)
    }

    /// Updates a logged-in user without further authentication
    ///
    /// This will invalidate the user's current token if the
    /// password is updated.
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this user
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "User::update_user", skip_all, err(Debug))]
    pub async fn update_user(
        &self,
        username: &str,
        update: UserUpdate,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // only admins can update other users
        is_admin!(self);
        // get info on the target user
        let mut target = User::force_get(username, shared).await?;
        // check if we are updating their password
        if let Some(password) = &update.password {
            // disallow password updates for non local accounts
            if shared.config.thorium.auth.ldap.is_none() || target.password.is_some() {
                event!(Level::INFO, msg = "Updating password");
                // clone key so it will live throughout async closure
                let key = &shared.config.thorium.secret_key;
                // hash password and set a new token
                target.password = Some(hash_pw!(password, key));
                target.gen_token(shared);
            } else {
                return unavailable!("Cannot update password when ldap is enabled".to_string());
            }
        }
        // update our role
        crate::update!(target.role, update.role);
        // apply any settings updates
        if let Some(settings) = update.settings {
            settings.apply(&mut target);
        }
        // save update user to the backend
        db::users::save(&target, shared).await?;
        Ok(())
    }

    /// Authenticate a user with the correct authentication method
    ///
    /// This gets the authorization data from the authorization header.
    ///
    /// # Arguments
    ///
    /// * `auth_header` - The auth header value to pull creds from
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "User::auth", skip_all, err(Debug))]
    async fn auth(auth_header: &str, shared: &Shared) -> Result<Self, ApiError> {
        // get our auth method
        let method = check_unauth!(AuthMethods::from_str(auth_header));
        // try to authenticate our user
        match method.authenticate(shared).await {
            Ok(user) => {
                event!(Level::INFO, user = &user.username);
                Ok(user)
            }
            Err(error) => {
                // we failed to auth this user due to an error
                event!(Level::ERROR, error = true, error_msg = error.to_string());
                Err(error)
            }
        }
    }

    /// Authorize this user can access some groups or gets the groups they can
    ///
    /// For admins this will get all groups in the cluster
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict this authorization check too
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Users::authorize_groups", skip_all, err(Debug))]
    pub async fn authorize_groups(
        &self,
        groups: &mut Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // for users we can search their groups but for admins we need to search all groups
        // if this user is not an admin then check if any groups were requested otherwise return
        // all the groups they are in
        if self.role == UserRole::Admin {
            // this user is an admin so they can access all groups
            // if they provided groups we just need to make sure they exist
            if !groups.is_empty() {
                // TODO: make sure groups exist
                return Ok(());
            }
            // no groups were provided so get 10000 groups
            // this is a bit of a hack and means if theres more than 10000 groups we silently miss things
            // TODO: Fix this
            let mut list = db::groups::list(0, 10000, shared).await?;
            // extend our groups list
            groups.append(&mut list.names);
            Ok(())
        } else {
            // this user is not an admin so make sure they can access any requested groups
            if groups.is_empty() {
                // no groups were provided so just default to the groups this user is a part of
                Group::authorize_all(self, &self.groups, shared).await?;
                // add our users groups
                groups.append(&mut self.groups.clone());
                Ok(())
            } else {
                Group::authorize_all(self, groups, shared).await?;
                Ok(())
            }
        }
    }

    /// Sync all ldap users info
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub async fn sync_all_unix_info(&self, shared: &Shared) -> Result<(), ApiError> {
        // only admins can sync unix info
        is_admin!(self);
        // get our ldap conf
        if let Some(conf) = &shared.config.thorium.auth.ldap {
            //  build an ldap connection
            let (conn, mut ldap) = ldap!(conf).await?;
            // drive this connection to completion
            ldap3::drive!(conn);
            // bind to our system user
            bind_to_ldap_system_user(&mut ldap, conf).await?;
            // get a list of all users
            for user in self.list_details(shared).await? {
                // if we don't have a password set then assume ldap and update our unix info
                let unix = if user.local {
                    // this is a local user so just use our default unix IDs
                    shared.config.thorium.auth.local_user_ids.clone()
                } else {
                    // this is an ldap based user so get updated info instead
                    get_unix_info(&user.username, conf, &mut ldap).await?
                };
                // update this users unix info
                db::users::update_unix_info(&user.username, &unix, shared).await?;
            }
            // unbind our ldap socket
            ldap.unbind().await?;
            Ok(())
        } else {
            bad!("LDAP is not configured!".to_owned())
        }
    }
}

/// Base64 decode a string
///
/// # Arguments
///
/// * `encoded` - A base64 encoded string to decode
fn b64_decode(encoded: &str) -> Result<String, ApiError> {
    // decode our base64'd bytes
    let decoded = base64::engine::general_purpose::STANDARD.decode(encoded)?;
    // convert our decoded bytes to a string
    let decoded_string = str::from_utf8(&decoded[..])?.to_owned();
    Ok(decoded_string)
}

pub struct AuthReject;

impl IntoResponse for AuthReject {
    fn into_response(self) -> Response {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

impl<S> FromRequestParts<S> for User
where
    AppState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = AuthReject;
    /// Gets an authenticated user from a request
    ///
    /// # Arguments
    ///
    /// * `parts` - The request parts to extract our secret key from
    /// * `state` - Shared Thorium objects
    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // get the shared app state
        let state = AppState::from_ref(state);
        // extract the authorization headers for this user
        if let Some(header_val) = parts.headers.get("authorization") {
            // try to cast our authorization header value to a str
            if let Ok(header_str) = header_val.to_str() {
                if let Ok(user) = User::auth(header_str, &state.shared).await {
                    return Ok(user);
                }
            }
        }
        // we failed to extract our auth info from our headers
        Err(AuthReject)
    }
}

impl From<User> for ScrubbedUser {
    fn from(user: User) -> Self {
        ScrubbedUser {
            username: user.username,
            email: user.email,
            role: user.role,
            groups: user.groups,
            token: user.token,
            token_expiration: user.token_expiration,
            unix: user.unix,
            settings: user.settings,
            local: user.password.is_some(),
            verified: user.verified,
        }
    }
}

impl From<User> for AuthResponse {
    /// Build an `AuthResponse` from a User
    ///
    /// # Arguments
    ///
    /// * `user` - The user to build an `AuthResponse` from
    fn from(user: User) -> Self {
        AuthResponse {
            token: user.token,
            expires: user.token_expiration,
        }
    }
}
