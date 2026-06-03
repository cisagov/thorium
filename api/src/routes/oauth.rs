//! The routes to support oauth based authentication
use axum::Router;
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::Redirect;
use axum::routing::{get, post};
use openidconnect::{AuthorizationCode, CsrfToken};
use tracing::{Level, event, instrument};
use utoipa::OpenApi;

use super::OpenApiSecurity;

use crate::models::{
    AuthResponse, OAuthCallbackParams, OAuthLinkParams, OAuthMaybeAuthed,
    OAuthRegistrationSessionId, OAuthUserCreate, OAuthUsernameCheck,
};
use crate::unauthorized;
use crate::utils::{ApiError, AppState};

/// List all of the configured oauth providers
///
/// # Arguments
///
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/oauth/",
    responses(
        (status = 200, description = "The list of oauth providers", body=Vec<String>),
    ),
)]
#[instrument(name = "routes::oauth::list_providers", skip_all, err(Debug))]
async fn list_providers(State(state): State<AppState>) -> Result<Json<Vec<String>>, ApiError> {
    // get our oauth config or return an error
    let oauth = match &state.shared.config.thorium.auth.oauth {
        Some(oauth) => oauth,
        None => return unauthorized!("OAuth is not configured!".to_owned()),
    };
    // get our provider names
    let providers = oauth
        .providers
        .keys()
        .map(|provider| provider.to_owned())
        .collect::<Vec<String>>();
    Ok(Json(providers))
}

/// Generate an auth/challenge url against a specific oauth provider
///
/// This is the start of an OAuth flow
///
/// # Arguments
///
/// * `provider` - The provider this user wants to auth against
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/oauth/{name}/auth",
    responses(
        (status = 303, description = "Redirect to the OAuth provider"),
    ),
)]
#[instrument(name = "routes::oauth::generate_challenege", skip(state), err(Debug))]
async fn generate_challenge(
    Path(provider): Path<String>,
    State(state): State<AppState>,
) -> Result<Redirect, ApiError> {
    // try to get the requested provider
    let client = match state.shared.oauth.get(&provider) {
        Some(client) => client,
        None => return unauthorized!("Unknown OAuth provider".to_owned()),
    };
    // generate a challenge with this provider
    let auth_url = client.generate_challenge(&state.shared).await?;
    Ok(Redirect::to(auth_url.as_str()))
}

/// Verify a user was successfully authenticated against by OAuth provider
///
/// This is the second and final step of OAuth authorization (but not
/// the last step for new users who need to register).
///
/// # Arguments
///
/// * `params` - The params for verifying an OAuth callback
/// * `provider` - The provider this auth callback is from
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/oauth/{provider}/callback",
    params(
        ("params" = OAuthCallbackParams, description = "Query params for verifying an OAuth challenge"),
    ),
    responses(
        (status = 200, description = "An Oauth registration session", body=OAuthRegistrationSessionId),
    ),
)]
#[instrument(name = "routes::oauth::callback", skip(state), err(Debug))]
async fn callback(
    params: OAuthCallbackParams,
    Path(provider): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<OAuthMaybeAuthed>, ApiError> {
    // try to get the requested provider
    let client = match state.shared.oauth.get(&provider) {
        Some(client) => client,
        None => return unauthorized!("Unknown OAuth provider".to_owned()),
    };
    // convert our params to their wrapped types
    let code = AuthorizationCode::new(params.code);
    let csrf_token = CsrfToken::new(params.state);
    // verify this users code was valid for their OAuth challenge
    let maybe_user = client
        .auth_challenge(code, &csrf_token, &state.shared)
        .await?;
    // build the correct auth response based on if this is a authed user or a new user
    let maybe_authed = OAuthMaybeAuthed::from(maybe_user);
    // return our json serialized data
    Ok(Json(maybe_authed))
}

/// Register a user after a successful oauth authentication
///
/// This can only be called once per registration session. If it errors after getting
/// the session from redis then users will need to reauth against oauth.
///
/// # Arguments
///
/// * `provider` - The provider this user wants to register a new user against
/// * `state` - Shared Thorium objects
/// * `create_req` - The info needed to register this new user
#[utoipa::path(
    post,
    path = "/api/oauth/{provider}/register",
    responses(
        (status = 200, description = "A user authentication response", body=AuthResponse),
    ),
)]
#[instrument(name = "routes::oauth::register", skip(state), err(Debug))]
async fn register(
    Path(provider): Path<String>,
    State(state): State<AppState>,
    Json(create_req): Json<OAuthUserCreate>,
) -> Result<Json<AuthResponse>, ApiError> {
    // valid this registration session and try to create this user
    let user = create_req.register(&provider, &state.shared).await?;
    // build an auth response for this user
    Ok(Json(AuthResponse::from(user)))
}

/// link a user after a successful oauth authentication
///
/// This can only be called once per link attempt. If it errors after getting
/// the alias from redis then users will need to reauth against oauth.
///
/// # Arguments
///
/// * `params` - The parameters needed to link an existing user to a new OAuth provider
/// * `provider` - The name of the provider to link against
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/oauth/{provider}/link",
    params(
        ("params" = OAuthLinkParams, description = "Query params for linking an existing user to a new OAuth alias"),
    ),
    responses(
        (status = 204, description = "The OAuth alias was linked to the account"),
        (status = 401, description = "The link token is invalid, expired, or already used"),
    ),
)]
#[instrument(name = "routes::oauth::link", skip(state), err(Debug))]
async fn link(
    params: OAuthLinkParams,
    Path(provider): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // attempt the link; an invalid/expired/used token AND OAuth not being configured all surface
    // uniformly as a 401 so this unauthenticated endpoint never reveals whether OAuth is enabled
    match params.link(provider, &state.shared).await {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(error) => {
            // log the underlying reason (the client only ever sees the uniform 401)
            event!(Level::WARN, error = %error, msg = "OAuth account link failed");
            unauthorized!("This account-link has expired or was already used".to_owned())
        }
    }
}

/// Revoke an account linking attempt
///
/// # Arguments
///
/// * `params` - The params needed to revoke a link attempt
/// * `provider` - The provider we are revoking a link attempt for
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/oauth/{provider}/link",
    params(
        ("params" = OAuthLinkParams, description = "Query params for revoking an attempt to link a new OAuth alias"),
    ),
    responses(
        (status = 204),
    ),
)]
#[instrument(name = "routes::oauth::revoke_link_attempt", skip(state), err(Debug))]
async fn revoke_link_attempt(
    params: OAuthLinkParams,
    Path(provider): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // The user does not wish to proceed with this account linking so revoke the link
    params.revoke(provider, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Check if a username is available
///
/// Only users with a valid oauth registration session can perform this check.
/// This won't prevent username enumeration but it will at least mitigate it.
/// This does require a provider name but username are not provider specific.
/// The provider name is used to validate a registration session token.
///
/// # Arguments
///
/// * `provider` - The provider we have an active registration session for
/// * `state` - Shared Thorium objects
/// * `check` - The info needed to check if a username is available
#[utoipa::path(
    post,
    path = "/api/oauth/{name}/username/available",
    responses(
        (status = 204, description = "This username is available"),
        (status = 409, description = "This username is taken"),
    ),
)]
#[instrument(name = "routes::oauth::list_providers", skip_all, err(Debug))]
async fn username_available(
    Path(provider): Path<String>,
    State(state): State<AppState>,
    Json(check): Json<OAuthUsernameCheck>,
) -> Result<StatusCode, ApiError> {
    // try to get the requested provider
    let client = match state.shared.oauth.get(&provider) {
        Some(client) => client,
        None => return unauthorized!("Unknown OAuth provider".to_owned()),
    };
    // check if this username is available
    if client
        .username_available(&check.session_token, &check.username, &state.shared)
        .await?
    {
        // this username is available
        Ok(StatusCode::NO_CONTENT)
    } else {
        // this username is not available
        Ok(StatusCode::CONFLICT)
    }
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(list_providers, generate_challenge, callback, register, link, revoke_link_attempt, username_available),
    components(schemas(OAuthCallbackParams, OAuthMaybeAuthed, OAuthUserCreate, AuthResponse, OAuthLinkParams, OAuthUsernameCheck)),
    modifiers(&OpenApiSecurity),
)]
pub struct OAuthApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(OAuthApiDocs::openapi())
}

/// Add the oauth routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/oauth/", get(list_providers))
        .route("/oauth/{name}/auth", get(generate_challenge))
        .route("/oauth/{name}/callback", get(callback))
        .route("/oauth/{name}/register", post(register))
        .route("/oauth/{name}/link", get(link).delete(revoke_link_attempt))
        .route("/oauth/{name}/username/available", post(username_available))
}
