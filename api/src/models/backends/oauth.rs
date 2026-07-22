//! Support for OAuth in the API

use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::request::Parts;
use chrono::prelude::*;
use openidconnect::core::CoreAuthenticationFlow;
use openidconnect::{
    AccessTokenHash, AuthorizationCode, CsrfToken, Nonce, OAuth2TokenResponse, PkceCodeChallenge,
    Scope, TokenResponse,
};
use rand::prelude::*;
use std::collections::HashMap;
use tracing::{Level, event};
use url::Url;

use super::db;
use crate::models::{
    Group, OAuthCallbackParams, OAuthLinkParams, OAuthMaybeAuthed, OAuthUserCreate, User, UserRole,
};
use crate::utils::shared::OAuthClient;
use crate::utils::{ApiError, Shared};
use crate::{bad, conflict, token, token_expire, unauthorized, unavailable};

/// An OAuth user creation session
#[derive(Serialize, Deserialize, Debug)]
pub struct OAuthRegistrationSession {
    /// The token for this session
    pub token: String,
    /// The alias for this user in this OAuth provider
    pub alias: String,
}

impl OAuthRegistrationSession {
    /// Create a new [`OAuthRegistrationSession`]
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias for this user in a specific OAuth provider
    pub fn new(alias: impl Into<String>) -> Self {
        OAuthRegistrationSession {
            token: token!(),
            alias: alias.into(),
        }
    }
}

/// A fully authenticated user or a registration session if this is a new user
pub enum OAuthedMaybeUser {
    /// A fully authenticated user
    User(User),
    /// A registration session for a new user
    NewUser(OAuthRegistrationSession),
}

impl OAuthedMaybeUser {
    /// Get an existing user linked to this oauth provider
    ///
    ///  # Arguments
    ///
    /// * `provider` - The name of the provider this user authenticated through
    /// * `alias` - The authenticated alias for this user
    /// * `shared` - Shared Thorium objects
    pub async fn new(
        provider: &str,
        alias: &str,
        shared: &Shared,
    ) -> Result<OAuthedMaybeUser, ApiError> {
        // get this users name by alias if it exists
        match db::oauth::get_username_by_alias(provider, alias, shared).await? {
            // this user should already exist so get our user data
            Some(username) => Ok(Self::User(User::force_get(&username, shared).await?)),
            None => {
                // this is a new user or this user has not yet linked their account with this oauth provider
                // create a new registration session
                let session = OAuthRegistrationSession::new(alias);
                // save this registration session
                db::oauth::save_registration_session(provider, &session, shared).await?;
                // return our wrapped new user registration session
                Ok(Self::NewUser(session))
            }
        }
    }
}

impl From<OAuthedMaybeUser> for OAuthMaybeAuthed {
    fn from(maybe_user: OAuthedMaybeUser) -> Self {
        // build the correct response on if this is a new or existing user
        match maybe_user {
            OAuthedMaybeUser::User(user) => {
                // check if this user has a verified email or not
                if user.verified {
                    // this user has already verified their email
                    OAuthMaybeAuthed::Authed {
                        token: user.token,
                        expires: user.token_expiration,
                    }
                } else {
                    // this user needs to verify their email
                    OAuthMaybeAuthed::VerifyEmail(user.email)
                }
            }
            OAuthedMaybeUser::NewUser(session) => OAuthMaybeAuthed::NewUser(session.token),
        }
    }
}

impl OAuthClient {
    /// Generate a challenge for this OAuth provider
    ///
    /// # Arguments
    ///
    /// * `oauth_conf` - The config for oauth
    /// * `shared` - Shared Thorium objects
    pub async fn generate_challenge(&self, shared: &Shared) -> Result<Url, ApiError> {
        // generate a PKCE challenge.
        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();
        // generate the full authorization URL
        let builder = self
            .client
            .authorize_url(
                CoreAuthenticationFlow::AuthorizationCode,
                CsrfToken::new_random,
                Nonce::new_random,
            )
            // Set the PKCE code challenge.
            .set_pkce_challenge(pkce_challenge);
        // add the scopes that this provider needs
        let builder = self.scopes.iter().fold(builder, |builder, scope| {
            builder.add_scope(Scope::new(scope.clone()))
        });
        // build everything needed to start an OAuth authentication session
        let (auth_url, csrf_token, nonce) = builder.url();
        // save our nonce
        db::oauth::store_context(&self.name, &csrf_token, nonce, pkce_verifier, shared).await?;
        Ok(auth_url)
    }

    /// Authenticate a challenge response
    ///
    /// # Arguments
    ///
    /// * `code` - The code to verify
    /// * `csrf_token` - The CSRF token to use for protection
    /// * `shared` - Shared Thorium objects
    pub async fn auth_challenge(
        &self,
        code: AuthorizationCode,
        csrf_token: &CsrfToken,
        shared: &Shared,
    ) -> Result<OAuthedMaybeUser, ApiError> {
        // try to load the context for this
        let (nonce, pkce_verifier) =
            db::oauth::consume_context(&self.name, csrf_token, shared).await?;
        // get a token request for the challenge code from our client
        let token_req = self.client.exchange_code(code)?;
        // add our pkce verifier and  send this request
        let token_response = token_req
            .set_pkce_verifier(pkce_verifier)
            .request_async(&self.http_client)
            .await?;
        // get the id token so we can verify it
        let id_token = token_response
            .id_token()
            .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, None))?;
        // get this tokens verifier
        let id_token_verifier = self.client.id_token_verifier();
        // get the claims for this token
        let claims = id_token.claims(&id_token_verifier, &nonce)?;
        // get the accesss token for this claim
        if let Some(expected_access_token_hash) = claims.access_token_hash() {
            // get the access token that was supplied for the user
            let actual_access_token_hash = AccessTokenHash::from_token(
                token_response.access_token(),
                id_token.signing_alg()?,
                id_token.signing_key(&id_token_verifier)?,
            )?;
            // make sure this access token isn't for a different user
            if actual_access_token_hash != *expected_access_token_hash {
                // log an error that we found an invalid access token hash
                event!(Level::ERROR, msg = "OAuth found invalid access token hash");
                // return a 401
                return unauthorized!();
            }
        }
        // get the alias for this user from OAuth
        let alias = claims.subject().as_str();
        // get this authenticated users data if it this is a linked oauth account
        // or start a new registration flow
        OAuthedMaybeUser::new(&self.name, alias, shared).await
    }

    /// Check if a username is available in Thorium
    ///
    /// # Arguments
    ///
    /// * `session_token` - A valid registration token
    /// * `username` - The username to check
    /// * `shared` - Shared Thorium objects
    pub async fn username_available(
        &self,
        session_token: &str,
        username: &str,
        shared: &Shared,
    ) -> Result<bool, ApiError> {
        // make sure that this is a valid registration token
        db::oauth::validate_registration_session(&self.name, session_token, shared).await?;
        // now that we have confirmed we have a valid session check if this username is available
        // we are inverting this check so we have to unwrap invert then rewrap in Ok
        Ok(!User::exists(username, shared).await?)
    }
}

impl OAuthUserCreate {
    /// Send an existing user an oauth provider link email
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider we are trying to link this user too
    /// * `username` - The username for the user to link
    /// * `alias` - The alias for this user
    /// * `shared` - Shared Thorium objects
    pub async fn send_link_email(
        &self,
        provider: &str,
        username: &str,
        alias: &str,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get our oauth config or return an error
        let oauth = match &shared.config.thorium.auth.oauth {
            Some(oauth) => oauth,
            None => return unavailable!("OAuth is not configured!".to_owned()),
        };
        // get an email client
        let client = match &shared.email {
            Some(client) => client,
            None => return unavailable!("Email is not configured!".to_owned()),
        };
        // build a oauth link token
        let link_token = token!();
        // save our link token and the alias for this user
        db::oauth::save_link_token(provider, username, &link_token, alias, shared).await?;
        // build the base url for linking accounts
        let base = format!("{}/oauth/{provider}/link", oauth.redirect_base);
        // build our link to new OAuth provider link to embed in the email
        let link =
            Url::parse_with_params(&base, &[("username", username), ("token", &link_token)])?;
        // build the subject for email verification email
        let subject = format!("Link Thorium account to new OAuth provider: {provider}");
        // get the expiration of this link in a human readable format
        let ttl = crate::utils::helpers::human_duration(std::time::Duration::from_secs(
            oauth.link_expire,
        ));
        // build a body with our verification email
        let body = format!(
            "If you would like to link your Thorium account to the {provider} OAuth provider then click on the following link in the next {ttl}:\n\n{link}"
        );
        // send our verification email
        client.send(&self.email, subject, body).await
    }

    /// Register a new user from an OAuth registration session
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider to create a new registration session with
    /// * `shared` - Shared Thorium objects
    pub async fn register(self, provider: &str, shared: &Shared) -> Result<User, ApiError> {
        // make sure this is a valid provider
        if !shared.oauth.contains_key(provider) {
            // return unauthorized as this is not a valid OAuth provider
            return unauthorized!();
        }
        // get the session for this registration
        let session =
            db::oauth::consume_registration_session(provider, &self.session_token, shared).await?;
        // initialize this users alias map
        let mut aliases = HashMap::with_capacity(1);
        // check if this email is already in use
        let maybe_exists = db::users::get_username_for_email(&self.email, shared).await?;
        // if this email is already in use then this user must link their account instead
        if let Some(username) = maybe_exists {
            // check if this is a request to register a different user with this email
            if username != self.username {
                // email and usernames must both be unique
                // return a conflict error telling the user this email is already in use
                return conflict!("A different user with this email already exists! Emails must be unique for each user.".to_owned());
            }
            // create and send an oauth provider link email to this user
            self.send_link_email(provider, &username, &session.alias, shared)
                .await?;
            // return a 409 and tell the user they already have an account
            return conflict!("A user with this email already exists. Please check your email for a account link email!".to_owned());
        }
        // also validate that this username is not yet taken
        if User::exists(&self.username, shared).await? {
            // This username is already taken so return a conflict error
            return conflict!("This username is already taken!".to_owned());
        }
        // add this alias to this user
        aliases.insert(provider.to_owned(), session.alias);
        // this is a valid session so register this user
        // build the user object for this new user
        let mut cast = User {
            username: self.username,
            password: None,
            email: self.email,
            groups: Vec::default(),
            role: UserRole::User,
            token: token!(),
            unix: None,
            token_expiration: token_expire!(shared),
            settings: self.settings,
            verified: false,
            verification_token: None,
            verification_sent: None,
            aliases,
        };
        // send a verification email if needed
        match &shared.email {
            // send our verification email
            Some(email_client) => cast.send_verification_email(email_client, shared).await?,
            // just automatically verify this users email
            None => cast.verified = true,
        }
        // inject user into the backend
        let user = db::users::create(cast, shared).await?;
        // sync all groups in ldap if ldap is enabled
        if shared.config.thorium.auth.ldap.is_some() {
            // sync ldap data for all groups
            Group::sync_ldap(shared).await?;
        }
        Ok(user)
    }
}

impl OAuthLinkParams {
    /// Try to link an existing user to a new OAuth alias
    ///
    /// # Arguments
    ///
    /// * `provider` - The Oauth provider we are linking this account too
    /// * `shared` - Shared Thorium objects
    pub async fn link(&self, provider: String, shared: &Shared) -> Result<(), ApiError> {
        // get this links info if it exists
        let alias =
            db::oauth::consume_link_token(&provider, &self.username, &self.token, shared).await?;
        // get the user we want to add an alias too
        let mut user = User::force_get(&self.username, shared).await?;
        // add this alias to this user
        user.aliases.insert(provider, alias);
        // save this users info
        db::users::save(&user, shared).await?;
        // since we got this link through email we can also verify their email if its not yet verified
        if !user.verified {
            // clear this users verification token and set them as verified in redis
            db::users::clear_verification_token(&self.username, shared).await?;
        }
        Ok(())
    }

    /// Revoke an active account linking attempt
    ///
    /// # Arguments
    ///
    /// * `provider` - The Oauth provider we revoking an attempted account linking for
    /// * `shared` - Shared Thorium objects
    pub async fn revoke(&self, provider: String, shared: &Shared) -> Result<(), ApiError> {
        // consume and forget this account linking token
        db::oauth::consume_link_token(&provider, &self.username, &self.token, shared).await?;
        Ok(())
    }
}

impl<S> FromRequestParts<S> for OAuthCallbackParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new()
                .max_depth(5)
                .deserialize_str(query)?)
        } else {
            bad!("Missing code and state query params!".to_owned())
        }
    }
}

impl<S> FromRequestParts<S> for OAuthLinkParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new()
                .max_depth(5)
                .deserialize_str(query)?)
        } else {
            bad!("Missing token and username query params!".to_owned())
        }
    }
}
