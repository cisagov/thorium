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
use crate::utils::{ApiError, Shared, bounder};
use crate::{bad, conflict, token, token_expire, unauthorized, unavailable};

/// An OAuth user creation session
#[derive(Serialize, Deserialize, Debug)]
pub struct OAuthRegistrationSession {
    /// The token for this session
    pub token: String,
    /// The alias for this user in this OAuth provider
    pub alias: String,
    /// The verified email from the OIDC provider for this identity
    ///
    /// Defaulted so registration sessions written before this field existed still deserialize; a
    /// session with an empty email that does not require a user-supplied email is rejected as
    /// "no verified email" during registration.
    #[serde(default)]
    pub email: String,
    /// Whether the user must supply their own email at registration because the IdP returned none
    ///
    /// Defaulted so registration sessions written before this field existed still deserialize as
    /// `false` (the IdP-email path).
    #[serde(default)]
    pub email_required: bool,
}

impl OAuthRegistrationSession {
    /// Create a new [`OAuthRegistrationSession`] bound to an IdP-provided email
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias for this user in a specific OAuth provider
    /// * `email` - The verified email from the OIDC provider for this identity
    pub fn new(alias: impl Into<String>, email: impl Into<String>) -> Self {
        OAuthRegistrationSession {
            token: token!(),
            alias: alias.into(),
            email: email.into(),
            email_required: false,
        }
    }

    /// Create a new [`OAuthRegistrationSession`] that requires a user-supplied email
    ///
    /// This is used when the IdP returned no email claim but the provider is configured to accept a
    /// user-supplied email (via a non-empty `approved_emails` regex list). The email is left empty
    /// and provided by the user during registration.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias for this user in a specific OAuth provider
    pub fn new_email_required(alias: impl Into<String>) -> Self {
        OAuthRegistrationSession {
            token: token!(),
            alias: alias.into(),
            email: String::new(),
            email_required: true,
        }
    }
}

/// The outcome of deriving an email for a new user from an OIDC provider's claims
///
/// This decides how a new-user registration session is created based on what the IdP returned and
/// how the provider is configured.
pub enum RegistrationEmail<'a> {
    /// The IdP provided an email that is trusted and permitted for this provider
    Provided(&'a str),
    /// The IdP provided no email but this provider accepts a user-supplied email at registration
    UserRequired,
    /// The IdP provided no email and this provider does not accept a user-supplied email
    Missing,
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
    /// * `email` - The email outcome derived from the OIDC claims and provider config
    /// * `shared` - Shared Thorium objects
    pub async fn new(
        provider: &str,
        alias: &str,
        email: RegistrationEmail<'_>,
        shared: &Shared,
    ) -> Result<OAuthedMaybeUser, ApiError> {
        // get this users name by alias if it exists
        match db::oauth::get_username_by_alias(provider, alias, shared).await? {
            // this user should already exist so get our user data
            Some(username) => {
                // get this users data
                let user = User::force_get(&username, shared).await?;
                // reject disabled users before handing back a live token
                if user.role == UserRole::Disabled {
                    // this user has been disabled so reject this request
                    return unauthorized!("This user has been disabled".to_owned());
                }
                // existing linked users log in purely by alias so we do not require an email here
                Ok(Self::User(user))
            }
            None => {
                // this is a new user or this user has not yet linked their account with this oauth provider
                // build a registration session based on how we derived this identity's email so an
                // attacker can never register or trigger a link email with a self-asserted address
                let session = match email {
                    // the IdP provided a trusted, permitted email so bind it to the session
                    RegistrationEmail::Provided(email) => {
                        OAuthRegistrationSession::new(alias, email)
                    }
                    // the IdP provided no email but this provider accepts a user-supplied one, so
                    // start a session that requires the user to provide their email at registration
                    RegistrationEmail::UserRequired => {
                        OAuthRegistrationSession::new_email_required(alias)
                    }
                    // fail closed when the provider did not return an email and does not accept one
                    RegistrationEmail::Missing => {
                        return unauthorized!(
                            "This OAuth provider did not return a verified email".to_owned()
                        );
                    }
                };
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
            OAuthedMaybeUser::NewUser(session) => OAuthMaybeAuthed::NewUser {
                // hand back the registration token and whether the user must supply an email
                token: session.token,
                email_required: session.email_required,
            },
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
        // derive the verified email from the claims, only trusting it when the provider asserts it
        // is verified (or when this provider is explicitly configured as trusted)
        let verified_email = derive_verified_email(
            claims.email().map(|email| email.as_str()),
            claims.email_verified(),
            self.trust_unverified_email,
        );
        // decide how this new identity's email should be handled based on the claims and config
        let registration_email = match &verified_email {
            // the IdP returned a trusted email so it must be permitted for this provider
            Some(email) => {
                // reject an IdP email that is not permitted by this provider's approved regexes
                if !self.email_allowed(email) {
                    // log the rejection so this decision is auditable
                    event!(Level::WARN, msg = "OAuth email not permitted for this provider");
                    // fail closed as the IdP email is not on this provider's allow list
                    return unauthorized!(
                        "This email is not permitted for this OAuth provider".to_owned()
                    );
                }
                // the IdP email is trusted and permitted so bind it to the session
                RegistrationEmail::Provided(email)
            }
            // the IdP returned no trusted email so fall back to a user-supplied email if allowed
            None if self.accepts_user_email() => RegistrationEmail::UserRequired,
            // the IdP returned no email and this provider does not accept a user-supplied one
            None => RegistrationEmail::Missing,
        };
        // get this authenticated users data if it this is a linked oauth account
        // or start a new registration flow
        OAuthedMaybeUser::new(&self.name, alias, registration_email, shared).await
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

/// Validate a username presented during OAuth self-registration
///
/// OAuth registration must enforce the same username rules as `User::create` so an OAuth registrant
/// cannot claim a reserved identity (such as `external`, used for external-scaler attribution) or a
/// username carrying LDAP/redis/k8s metacharacters that later subsystems assume are absent.
/// This centralizes those checks so the OAuth path and the local path stay in sync.
///
/// # Arguments
///
/// * `username` - The username presented in the OAuth registration request
fn validate_oauth_username(username: &str) -> Result<(), ApiError> {
    // ensure username is lowercase-alphanumeric and within length bounds
    bounder::string_lower(username, "username", 1, 50)?;
    // reject the reserved `external` username here too, mirroring `User::create`, so an
    // OAuth registrant cannot claim the identity used for external-scaler attribution
    if username == "external" {
        return bad!("external is a reserved username".to_owned());
    }
    Ok(())
}

/// Derive the email we are willing to trust from an OIDC provider's claims
///
/// The registration email is never client-supplied; it comes from the provider's claims. We only
/// trust the email when the provider asserts it is verified (`email_verified == true`) so a new user
/// cannot register (or trigger a link email) with an unverified or attacker-controlled address. When
/// a provider is explicitly configured as trusted via `trust_unverified_email`, we also accept an
/// email whose `email_verified` claim is *absent* (some IdPs never emit it), but we never accept an
/// email the provider actively marks unverified (`email_verified == false`).
///
/// # Arguments
///
/// * `email` - The email claim returned by the provider, if any
/// * `email_verified` - The `email_verified` claim returned by the provider, if any
/// * `trust_unverified_email` - Whether this provider is configured to trust an absent verified flag
fn derive_verified_email(
    email: Option<&str>,
    email_verified: Option<bool>,
    trust_unverified_email: bool,
) -> Option<String> {
    // decide which emails we are willing to trust based on the verified flag and provider config
    match (email, email_verified) {
        // the provider returned an email and vouched that it is verified
        (Some(email), Some(true)) => Some(email.to_owned()),
        // the provider returned an email but omitted email_verified and this provider is configured
        // as trusted, so accept it; an explicit `false` still falls through to the reject arm below
        (Some(email), None) if trust_unverified_email => {
            // log that we trusted an unverified email so this decision is auditable
            event!(
                Level::WARN,
                msg = "Trusting OAuth email without email_verified per provider config"
            );
            Some(email.to_owned())
        }
        // an email was present but not verified so log it and treat it as absent
        (Some(_), _) => {
            // log that we dropped an unverified email
            event!(Level::WARN, msg = "OAuth email claim was not verified");
            None
        }
        // no email claim was returned at all
        _ => None,
    }
}

impl OAuthUserCreate {
    /// Send an existing user an oauth provider link email
    ///
    /// The email is always sent to the existing account's own stored address (never a
    /// client-supplied one), so an account-link link can only ever reach the real account owner.
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider we are trying to link this user too
    /// * `username` - The username for the user to link
    /// * `recipient` - The existing account's stored email to send the link to
    /// * `alias` - The OIDC identity/alias that would be linked to this account
    /// * `shared` - Shared Thorium objects
    pub async fn send_link_email(
        provider: &str,
        username: &str,
        recipient: &str,
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
        // build a body that names the exact IdP identity being linked so the recipient can tell
        // whether the request was theirs before clicking
        let body = format!(
            "If you would like to link your Thorium account to the {provider} identity '{alias}' then click on the following link in the next {ttl}:\n\n{link}\n\nIf you did not request this, you can ignore this email."
        );
        // send our verification email to the existing account's own stored address
        client.send(recipient, subject, body).await
    }

    /// Register a new user from an OAuth registration session
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider to create a new registration session with
    /// * `shared` - Shared Thorium objects
    pub async fn register(self, provider: &str, shared: &Shared) -> Result<User, ApiError> {
        // validate the username against the same rules the local registration path enforces
        validate_oauth_username(&self.username)?;
        // make sure this is a valid provider
        if !shared.oauth.contains_key(provider) {
            // return unauthorized as this is not a valid OAuth provider
            return unauthorized!();
        }
        // get the provider's client so we can enforce its approved email regexes
        let client = match shared.oauth.get(provider) {
            Some(client) => client,
            // return unauthorized as this is not a valid OAuth provider
            None => return unauthorized!(),
        };
        // get the session for this registration
        let session =
            db::oauth::consume_registration_session(provider, &self.session_token, shared).await?;
        // pull the verified alias and email off the session so we never trust a client email
        let OAuthRegistrationSession {
            alias,
            email: session_email,
            email_required,
            ..
        } = session;
        // resolve the final email for this account based on how the session was created
        let email = if email_required {
            // the IdP returned no email so this session requires a user-supplied email
            let email = match self.email {
                // the user supplied an email so we can continue validating it
                Some(email) => email,
                // fail closed as this provider requires a user-supplied email but none was given
                None => return bad!("This OAuth provider requires an email address".to_owned()),
            };
            // reject a user-supplied email that is not permitted by this provider's approved regexes
            if !client.email_allowed(&email) {
                // fail closed as the user-supplied email is not on this provider's allow list
                return unauthorized!(
                    "This email is not permitted for this OAuth provider".to_owned()
                );
            }
            email
        } else {
            // reject sessions that carry no email and do not require a user-supplied one (e.g. old
            // pre-migration sessions)
            if session_email.is_empty() {
                // fail closed as we cannot register a user without a verified email
                return unauthorized!(
                    "This OAuth provider did not return a verified email".to_owned()
                );
            }
            // the IdP provided this email so reject a user-supplied email that does not match it
            if let Some(user_email) = &self.email {
                if user_email != &session_email {
                    // fail closed as the user tried to register with a different email than the IdP's
                    return bad!(
                        "The supplied email must match the email from the OAuth provider".to_owned()
                    );
                }
            }
            // defensively re-validate the IdP email against this provider's approved regexes
            if !client.email_allowed(&session_email) {
                // fail closed as the IdP email is not on this provider's allow list
                return unauthorized!(
                    "This email is not permitted for this OAuth provider".to_owned()
                );
            }
            session_email
        };
        // initialize this users alias map
        let mut aliases = HashMap::with_capacity(1);
        // check if this verified email is already in use
        let maybe_exists = db::users::get_username_for_email(&email, shared).await?;
        // if this email is already in use then this user must link their account instead
        if let Some(username) = maybe_exists {
            // check if this is a request to register a different user with this email
            if username != self.username {
                // email and usernames must both be unique
                // return a conflict error telling the user this email is already in use
                return conflict!("A different user with this email already exists! Emails must be unique for each user.".to_owned());
            }
            // send an account-link email to the existing account's own stored address (which is this
            // verified email, since we matched the account by it) naming the identity being linked
            Self::send_link_email(provider, &username, &email, &alias, shared).await?;
            // return a 409 and tell the user they already have an account
            return conflict!("A user with this email already exists. Please check your email for a account link email!".to_owned());
        }
        // also validate that this username is not yet taken
        if User::exists(&self.username, shared).await? {
            // This username is already taken so return a conflict error
            return conflict!("This username is already taken!".to_owned());
        }
        // add this alias to this user
        aliases.insert(provider.to_owned(), alias);
        // this is a valid session so register this user
        // build the user object for this new user
        let mut cast = User {
            username: self.username,
            password: None,
            email,
            groups: Vec::default(),
            actual_groups: Vec::default(),
            role: UserRole::User,
            token: token!(),
            unix: None,
            token_expiration: token_expire!(shared),
            settings: self.settings,
            verified: false,
            verification_token: None,
            verification_sent: None,
            aliases,
            image: None,
        };
        // send a verification email, requiring email verification to be configured for OAuth so a
        // user-supplied email can never be auto-verified for an address the registrant does not own
        match &shared.email {
            // send our verification email
            Some(email_client) => cast.send_verification_email(email_client, shared).await?,
            // fail closed as OAuth registration requires email verification to be enabled
            None => {
                return unavailable!(
                    "Email verification must be enabled to register via OAuth".to_owned()
                );
            }
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
        // reject disabled users so they cannot link aliases or verify their email
        if user.role == UserRole::Disabled {
            // this user has been disabled so reject this request
            return unauthorized!("This user has been disabled".to_owned());
        }
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

#[cfg(test)]
mod tests {
    use super::{OAuthRegistrationSession, derive_verified_email, validate_oauth_username};

    #[test]
    fn oauth_registration_session_stores_verified_email() {
        // a session must retain the verified alias and email it was created with so registration
        // uses the provider-verified email rather than any client-supplied value
        let session = OAuthRegistrationSession::new("subject-123", "user@example.com");
        assert_eq!(session.alias, "subject-123");
        assert_eq!(session.email, "user@example.com");
    }

    #[test]
    fn oauth_registration_session_defaults_missing_email() {
        // a registration session serialized before the `email` field existed must still deserialize,
        // decoding to an empty email that the register path then rejects as "no verified email"
        let legacy = r#"{"token":"tok","alias":"subject-123"}"#;
        let session: OAuthRegistrationSession = serde_json::from_str(legacy).unwrap();
        assert_eq!(session.alias, "subject-123");
        assert!(
            session.email.is_empty(),
            "a legacy session without an email must decode to an empty (fail-closed) email"
        );
    }

    #[test]
    fn oauth_username_rejects_reserved_external() {
        // the reserved `external` username (external-scaler attribution) must be rejected
        // on the OAuth path exactly as `User::create` rejects it on the local path
        assert!(
            validate_oauth_username("external").is_err(),
            "OAuth registration accepted the reserved 'external' username"
        );
    }

    #[test]
    fn oauth_username_rejects_ldap_metacharacters() {
        // a username with LDAP filter metacharacters later injects into the admin LDAP sync filter
        assert!(
            validate_oauth_username("x)(objectclass=*").is_err(),
            "OAuth registration accepted an LDAP-metacharacter username"
        );
    }

    #[test]
    fn oauth_username_allows_normal_names() {
        // a well behaved lowercase-alphanumeric username must still be accepted
        assert!(
            validate_oauth_username("analyst1").is_ok(),
            "OAuth registration rejected a valid username"
        );
    }

    #[test]
    fn derive_verified_email_trusts_verified_claim() {
        // an email the provider vouches for (email_verified == true) is always trusted, regardless
        // of the trust_unverified_email flag
        assert_eq!(
            derive_verified_email(Some("user@example.com"), Some(true), false),
            Some("user@example.com".to_owned()),
            "a provider-verified email must be trusted"
        );
    }

    #[test]
    fn derive_verified_email_rejects_missing_flag_by_default() {
        // an email with no email_verified claim must be dropped when the provider is not configured
        // as trusted, preserving the fail-closed security default
        assert_eq!(
            derive_verified_email(Some("user@example.com"), None, false),
            None,
            "an unverified email must be rejected unless the provider is explicitly trusted"
        );
    }

    #[test]
    fn derive_verified_email_trusts_missing_flag_when_configured() {
        // an email with no email_verified claim is accepted only when the provider is explicitly
        // configured to trust it, covering IdPs that never emit email_verified
        assert_eq!(
            derive_verified_email(Some("user@example.com"), None, true),
            Some("user@example.com".to_owned()),
            "a trusted provider's email must be accepted when email_verified is absent"
        );
    }

    #[test]
    fn derive_verified_email_rejects_explicit_false_even_when_trusted() {
        // an email the provider actively marks unverified must never be trusted, even for a provider
        // configured with trust_unverified_email, since the IdP explicitly disclaims it
        assert_eq!(
            derive_verified_email(Some("user@example.com"), Some(false), true),
            None,
            "an explicitly unverified email must be rejected even for a trusted provider"
        );
    }

    #[test]
    fn derive_verified_email_rejects_missing_email() {
        // when the provider returns no email claim at all there is nothing to trust
        assert_eq!(
            derive_verified_email(None, None, true),
            None,
            "a missing email claim must yield no verified email"
        );
    }

    #[test]
    fn oauth_registration_session_email_required_carries_no_email() {
        // a session for a provider that returned no email must require a user-supplied email and
        // carry an empty email until the user provides one during registration
        let session = OAuthRegistrationSession::new_email_required("subject-123");
        assert_eq!(session.alias, "subject-123");
        assert!(
            session.email.is_empty(),
            "an email-required session must not carry an IdP email"
        );
        assert!(
            session.email_required,
            "a session created for a no-email provider must require a user-supplied email"
        );
    }

    #[test]
    fn oauth_registration_session_defaults_email_required_false() {
        // a session serialized before the `email_required` field existed must still deserialize,
        // defaulting to the IdP-email path (email_required == false) so it fails closed on empty email
        let legacy = r#"{"token":"tok","alias":"subject-123","email":"user@example.com"}"#;
        let session: OAuthRegistrationSession = serde_json::from_str(legacy).unwrap();
        assert_eq!(session.email, "user@example.com");
        assert!(
            !session.email_required,
            "a legacy session without email_required must default to the IdP-email path"
        );
    }

    #[test]
    fn approved_emails_regex_set_matches_expected_domains() {
        // the per-provider approved_emails regexes gate which emails may register; verify a
        // representative domain-restricting regex matches only that domain, mirroring how
        // OAuthClient::email_allowed applies the compiled RegexSet
        let approved = regex::RegexSet::new([r".*@sandia\.gov$"]).unwrap();
        assert!(
            approved.is_match("analyst@sandia.gov"),
            "an in-domain email must be permitted"
        );
        assert!(
            !approved.is_match("attacker@gmail.com"),
            "an out-of-domain email must be rejected"
        );
        assert!(
            !approved.is_match("analyst@sandia.gov.evil.com"),
            "an anchored regex must not match a look-alike domain suffix"
        );
    }
}
