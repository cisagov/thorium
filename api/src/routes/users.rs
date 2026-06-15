use axum::Router;
use axum::extract::{DefaultBodyLimit, Json, Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum_extra::TypedHeader;
use axum_extra::body::AsyncReadBody;
use chrono::Utc;
use tracing::{Level, event, instrument};
use utoipa::OpenApi;

use super::OpenApiSecurity;
use super::shared::graphics;

// our imports
use crate::models::backends::GraphicSupport;
use crate::models::{
    AiEndpoint, AiEndpointUpdate, AiSettings, AiSettingsUpdate, AuthResponse, AuthedUser, Key,
    ScopedToken, ScopedTokenRequest, ScopedTokenUpdate, ScrubbedUser, Theme, UnixInfo, User,
    UserCreate, UserRole, UserSettings, UserSettingsUpdate, UserUpdate,
};
use crate::utils::{ApiError, AppState};
use crate::{bad, conflict, is_admin, not_found, unauthorized, unavailable};

/// The maximum size (bytes) accepted for a profile-icon upload. Static images are client-resized to a small
/// PNG, but animated GIFs and short video clips (MP4/WebM) are uploaded as-is since they can't be cheaply
/// downscaled in the browser, so the cap is sized to fit a short clip while still bounding what a user can
/// stream into the graphics bucket. This is applied only to the icon-upload route; the global body limit
/// stays disabled so large sample/repo uploads are unaffected.
const ICON_MAX_BYTES: usize = 10 << 20;

/// Creates a new user
///
/// # Arguments
///
/// * `key` - An optional secret key used for bootstrapping admins
/// * `state` - Shared Thorium objects
/// * `user_create` - The user to create
#[utoipa::path(
    post,
    path = "/api/users/",
    params(
        ("user_create" = UserCreate, description = "The user to create"),
    ),
    responses(
        (status = 200, description = "New user created", body=AuthResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::create", skip_all, fields(key_set = key.is_some()), err(Debug))]
async fn create(
    key: Option<TypedHeader<Key>>,
    State(state): State<AppState>,
    Json(user_create): Json<UserCreate>,
) -> Result<Json<AuthResponse>, ApiError> {
    // get our secret key if it exists
    let key = key.map(|header| header.0);
    // create a user
    let user = User::create(user_create, key, &state.shared).await?;
    // build our auth response
    let resp = AuthResponse::from(user);
    Ok(Json(resp))
}

/// The response to a resend-verification-email request.
///
/// Both arms carry a `Retry-After` header (in seconds): `Sent` reports the full cooldown window so the
/// UI can seed its countdown timer, and `Cooldown` reports the time remaining before another resend is
/// allowed.
pub(crate) enum ResendVerificationResponse {
    /// A new verification email was sent; carries the full cooldown window in seconds.
    Sent { retry_after: u64 },
    /// Rate-limited: a verification email was sent too recently; carries the remaining cooldown in seconds.
    Cooldown { retry_after: i64 },
}

impl IntoResponse for ResendVerificationResponse {
    fn into_response(self) -> Response {
        match self {
            ResendVerificationResponse::Sent { retry_after } => (
                StatusCode::OK,
                [(header::RETRY_AFTER, retry_after.to_string())],
            )
                .into_response(),
            ResendVerificationResponse::Cooldown { retry_after } => (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, retry_after.to_string())],
                format!("Cannot resend a verification email for another {retry_after} seconds"),
            )
                .into_response(),
        }
    }
}

/// Resend our verification email if we are not yet verified
#[utoipa::path(
    get,
    path = "/api/users/resend/verify/email/:username",
    params(
        ("username" = String, Path, description = "The user to resend verificaton email for"),
    ),
    responses(
        (status = 200, description = "Verification email resent"),
        (status = 401, description = "This user is not authorized to access this route"),
        (status = 429, description = "A verification email was sent too recently; see the Retry-After header"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(
    name = "routes::users::resent_email_verification",
    skip_all,
    err(Debug)
)]
async fn resend_email_verification(
    Path(username): Path<String>,
    State(state): State<AppState>,
) -> Result<ResendVerificationResponse, ApiError> {
    // get the target user
    let mut user = User::force_get(&username, &state.shared).await?;
    // make sure email verification is enabled and grab the resend cooldown window
    let rate_limit = match &state.shared.config.thorium.auth.email {
        Some(email_conf) => email_conf.rate_limit,
        None => return unavailable!("Email verification is not enabled!".to_owned()),
    };
    // an already-verified user can't (and doesn't need to) resend — return a clear conflict before
    // the cooldown check so the response isn't shadowed by a misleading "wait N seconds" message
    if user.verified {
        return conflict!(format!(
            "{} has already verified their email",
            user.username
        ));
    }
    // enforce the cooldown here so we can report the remaining time via the Retry-After header; this
    // lets the UI render an accurate countdown instead of scraping it out of an error message
    if let Some(sent) = user.verification_sent {
        let remaining = rate_limit as i64 - (Utc::now() - sent).num_seconds();
        if remaining > 0 {
            return Ok(ResendVerificationResponse::Cooldown {
                retry_after: remaining,
            });
        }
    }
    // send a new verification email and report the full cooldown window so the UI can seed its timer
    match &state.shared.email {
        Some(client) => {
            // send a verification email to this user
            user.send_verification_email(client, &state.shared).await?;
            Ok(ResendVerificationResponse::Sent {
                retry_after: rate_limit,
            })
        }
        None => unavailable!("Email verification is not enabled!".to_owned()),
    }
}

/// Verifies an email for a specific user
#[utoipa::path(
    get,
    path = "/api/users/verify/:username/email/:verification_token",
    params(
        ("username" = String, Path, description = "The user to resend verificaton email for"),
        ("verification_token" = String, Path, description = "The token to send in the verification email"),
    ),
    responses(
        (status = 204, description = "The user's email was verified"),
        (status = 401, description = "The verification token is invalid, expired, or already used"),
    ),
)]
#[instrument(name = "routes::users::verify_email", skip_all, err(Debug))]
async fn verify_email(
    Path((username, verification_token)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // attempt verification; a wrong/expired token or unknown user all surface uniformly as a 401
    // so this unauthenticated endpoint never reveals whether an account exists
    match User::force_get(&username, &state.shared).await {
        Ok(mut user) => match user.verify_email(&verification_token, &state.shared).await {
            Ok(()) => Ok(StatusCode::NO_CONTENT),
            Err(error) => {
                // log the underlying reason (the client only ever sees the uniform 401)
                event!(Level::WARN, error = %error, msg = "Email verification token rejected");
                unauthorized!("This verification link has expired or was already used".to_owned())
            }
        },
        Err(error) => {
            // log the underlying reason (the client only ever sees the uniform 401)
            event!(Level::WARN, error = %error, msg = "Email verification could not load user");
            unauthorized!("This verification link has expired or was already used".to_owned())
        }
    }
}

/// Authenticates a user
///
/// # Arguments
///
/// * `resp` - The response to return for this user
#[utoipa::path(
    post,
    path = "/api/users/auth",
    params(
        ("user" = User, description = "The user to create"),
    ),
    responses(
        (status = 200, description = "User authenticated", body=AuthResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::auth", skip_all, err(Debug))]
async fn auth(resp: AuthResponse) -> Result<Json<AuthResponse>, ApiError> {
    // build auth response
    Ok(Json(resp))
}

/// Gets info about a specific user
///
/// # Arguments
///
/// * `user` - The user that is requesting info about another user
/// * `username` - The user to get info about
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/user/:username",
    params(
        ("user" = User, description = "The user that is requesting info about another user"),
        ("username" = String, Path, description = "The user to get info about"),
    ),
    responses(
        (status = 200, description = "Requested user info", body=ScrubbedUser),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::get_user", skip_all, err(Debug))]
async fn get_user(
    user: AuthedUser,
    Path(username): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ScrubbedUser>, ApiError> {
    // if user is an admin then allow them to get any user
    if user.is_admin() && user.username != username {
        let requested = User::force_get(&username, &state.shared).await?;
        Ok(Json(ScrubbedUser::from(requested)))
    // were requesting info on ourselves so just return it
    } else if user.username == username {
        Ok(Json(ScrubbedUser::from(user.into_user())))
    // were not an admin and not asking about ourselves reject it
    } else {
        unauthorized!()
    }
}

/// Lists all users
///
/// # Arguments
///
/// * `user` - The user that is listing users
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/",
    params(
        ("user" = User, description = "The user that is listing users"),
    ),
    responses(
        (status = 200, description = "Requested user list", body=Vec<String>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::list", skip_all, err(Debug))]
async fn list(
    user: AuthedUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<String>>, ApiError> {
    // list all users
    let users = user.list(&state.shared).await?;
    Ok(Json(users))
}

/// Lists all users with details
///
/// # Arguments
///
/// * `user` - The user that is listing user details
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/details/",
    params(
        ("user" = User, description = "The user that is listing user details"),
    ),
    responses(
        (status = 200, description = "Requested user info list", body=Vec<ScrubbedUser>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::list_details", skip_all, err(Debug))]
async fn list_details(
    user: AuthedUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<ScrubbedUser>>, ApiError> {
    // list all users with details
    let details = user.list_details(&state.shared).await?;
    Ok(Json(details))
}

/// Gets info about the currently authenticated user (ourselves)
///
/// # Arguments
///
/// * `user` - The current user
#[utoipa::path(
    get,
    path = "/api/users/whoami",
    params(
        ("user" = User, description = "The current user"),
    ),
    responses(
        (status = 200, description = "Currently-authenticated user info", body=ScrubbedUser),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::info", skip_all, err(Debug))]
async fn info(user: AuthedUser) -> Result<Json<ScrubbedUser>, ApiError> {
    Ok(Json(ScrubbedUser::from(user.into_user())))
}

/// Updates our current user
///
/// # Arguments
///
/// * `user` - The user to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply to this user
#[utoipa::path(
    patch,
    path = "/api/users/",
    params(
        ("user" = User, description = "The user to update"),
        ("update" = UserUpdate, description = "The update to apply to this user"),
    ),
    responses(
        (status = 204, description = "User update applied"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::update", skip_all, err(Debug))]
async fn update(
    user: AuthedUser,
    State(state): State<AppState>,
    Json(update): Json<UserUpdate>,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user since scoped tokens cannot update accounts
    let user = user.require_full()?;
    // update user
    user.update(update, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Updates a different user
///
/// # Arguments
///
/// * `user` - The user to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply to this user
#[utoipa::path(
    patch,
    path = "/api/users/user/:username",
    params(
        ("name" = String, Path, description = "The name of the user to update"),
        ("user" = User, description = "The user applying the update"),
        ("update" = UserUpdate, description = "The update to apply to this user"),
    ),
    responses(
        (status = 204, description = "User update applied"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::update_user", skip_all, err(Debug))]
async fn update_user(
    user: AuthedUser,
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(update): Json<UserUpdate>,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user since scoped tokens cannot update accounts
    let user = user.require_full()?;
    // update user
    user.update_user(&name, update, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Logs a user out
///
/// # Arguments
///
/// * `user` - The user to logout
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/users/logout",
    params(
        ("user" = User, description = "The user to logout"),
    ),
    responses(
        (status = 204, description = "User logged out"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::logout", skip_all, err(Debug))]
async fn logout(user: AuthedUser, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // logout based on how this user authenticated
    match user {
        // this user authenticated with their primary token so rotate that
        AuthedUser::Full(mut user) => user.regen_token(&state.shared).await?,
        // this user authenticated with a scoped token so only rotate that token
        AuthedUser::Scoped(scoped) => scoped.logout(&state.shared).await?,
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Logs another user out by username
///
/// # Arguments
///
/// * `user` - The admin who is logging another user out
/// * `target` - The username to logout
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/logout/:target",
    params(
        ("target" = String, Path, description = "The username to logout"),
        ("user" = User, description = "The user forcing the logout"),
    ),
    responses(
        (status = 204, description = "User logged out"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::logout_user", skip_all, err(Debug))]
async fn logout_user(
    user: AuthedUser,
    Path(target): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user since scoped tokens cannot logout other users
    let user = user.require_full()?;
    // only admins can logout other users
    is_admin!(user);
    // try to get the other user
    let mut target = User::force_get(&target, &state.shared).await?;
    // generate and save a new token
    target.regen_token(&state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes a user by username
///
/// Only admins can delete users other then themselves.
///
/// # Arguments
///
/// * `user` - The user who is deleting another user
/// * `target` - The username to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/users/delete/:target",
    params(
        ("target" = String, Path, description = "The username to delete"),
        ("user" = User, description = "The user who is deleting another user"),
    ),
    responses(
        (status = 204, description = "User deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::delete_user", skip_all, err(Debug))]
async fn delete_user(
    user: AuthedUser,
    Path(target): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user since scoped tokens cannot delete users
    let user = user.require_full()?;
    // try to delete this user
    User::delete(user, &target, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Syncs all ldap metagroups and their users
///
/// # Arguments
///
/// * `user` - The user that is telling Thorium to sync with ldap
/// * `shared` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/users/sync/ldap",
    params(),
    responses(
        (status = 204, description = "All ldap metagroups and users synced"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::sync_ldap", skip_all, err(Debug))]
async fn sync_ldap(
    user: AuthedUser,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // sync all groups with ldap
    user.sync_all_unix_info(&state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Creates a new scoped token for the current user
///
/// Scoped tokens are limited to a subset of the current users groups and may
/// optionally expire on a set date.
///
/// # Arguments
///
/// * `user` - The user to create a scoped token for
/// * `state` - Shared Thorium objects
/// * `req` - The scoped token to create
#[utoipa::path(
    post,
    path = "/api/users/tokens/",
    params(
        ("req" = ScopedTokenRequest, description = "The scoped token to create"),
    ),
    responses(
        (status = 200, description = "New scoped token created", body=ScopedToken),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::create_scoped_token", skip_all, err(Debug))]
async fn create_scoped_token(
    user: AuthedUser,
    State(state): State<AppState>,
    Json(req): Json<ScopedTokenRequest>,
) -> Result<Json<ScopedToken>, ApiError> {
    // require a fully authed user since scoped tokens cannot manage scoped tokens
    let user = user.require_full()?;
    // create this scoped token
    let scoped = ScopedToken::create(&user, req, &state.shared).await?;
    Ok(Json(scoped))
}

/// Lists all of the current users scoped tokens
///
/// # Arguments
///
/// * `user` - The user to list scoped tokens for
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/tokens/",
    params(),
    responses(
        (status = 200, description = "The current users scoped tokens", body=Vec<ScopedToken>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::list_scoped_tokens", skip_all, err(Debug))]
async fn list_scoped_tokens(
    user: AuthedUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<ScopedToken>>, ApiError> {
    // require a fully authed user since scoped tokens cannot manage scoped tokens
    let user = user.require_full()?;
    // list this users scoped tokens
    let scoped = ScopedToken::list(&user, &state.shared).await?;
    Ok(Json(scoped))
}

/// Gets one of the current users scoped tokens by name
///
/// If this scoped tokens value has expired then it will be rotated and the
/// new value returned.
///
/// # Arguments
///
/// * `user` - The user to get a scoped token for
/// * `name` - The name of the scoped token to get
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/tokens/:name",
    params(
        ("name" = String, Path, description = "The name of the scoped token to get"),
    ),
    responses(
        (status = 200, description = "The requested scoped token", body=ScopedToken),
        (status = 401, description = "This user is not authorized to access this route"),
        (status = 404, description = "This scoped token does not exist"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::get_scoped_token", skip_all, err(Debug))]
async fn get_scoped_token(
    user: AuthedUser,
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ScopedToken>, ApiError> {
    // require a fully authed user since scoped tokens cannot manage scoped tokens
    let user = user.require_full()?;
    // get this scoped token
    let scoped = ScopedToken::get(&user, &name, &state.shared).await?;
    Ok(Json(scoped))
}

/// Updates one of the current users scoped tokens by name
///
/// Updates never change a scoped tokens value so activated tokens keep
/// working after an update.
///
/// # Arguments
///
/// * `user` - The user to update a scoped token for
/// * `name` - The name of the scoped token to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply to this scoped token
#[utoipa::path(
    patch,
    path = "/api/users/tokens/:name",
    params(
        ("name" = String, Path, description = "The name of the scoped token to update"),
        ("update" = ScopedTokenUpdate, description = "The update to apply to this scoped token"),
    ),
    responses(
        (status = 200, description = "The updated scoped token", body=ScopedToken),
        (status = 400, description = "This scoped token update is invalid"),
        (status = 401, description = "This user is not authorized to access this route"),
        (status = 404, description = "This scoped token does not exist"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::update_scoped_token", skip_all, err(Debug))]
async fn update_scoped_token(
    user: AuthedUser,
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(update): Json<ScopedTokenUpdate>,
) -> Result<Json<ScopedToken>, ApiError> {
    // require a fully authed user since scoped tokens cannot manage scoped tokens
    let user = user.require_full()?;
    // update this scoped token
    let scoped = ScopedToken::update(&user, &name, update, &state.shared).await?;
    Ok(Json(scoped))
}

/// Deletes one of the current users scoped tokens by name
///
/// # Arguments
///
/// * `user` - The user to delete a scoped token for
/// * `name` - The name of the scoped token to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/users/tokens/:name",
    params(
        ("name" = String, Path, description = "The name of the scoped token to delete"),
    ),
    responses(
        (status = 204, description = "Scoped token deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
        (status = 404, description = "This scoped token does not exist"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::delete_scoped_token", skip_all, err(Debug))]
async fn delete_scoped_token(
    user: AuthedUser,
    Path(name): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user since scoped tokens cannot manage scoped tokens
    let user = user.require_full()?;
    // delete this scoped token
    ScopedToken::delete(&user, &name, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Uploads or replaces the current user's profile icon
///
/// # Arguments
///
/// * `user` - The user setting their icon
/// * `state` - Shared Thorium objects
/// * `form` - The multipart form containing the image in an `image` field
#[utoipa::path(
    post,
    path = "/api/users/image",
    request_body(
        content = String,
        description = "Multipart form with an 'image' field containing the icon",
        content_type = "multipart/form-data",
    ),
    responses(
        (status = 204, description = "The profile icon was uploaded"),
        (status = 400, description = "No image field was provided"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[axum_macros::debug_handler]
#[instrument(name = "routes::users::upload_image", skip_all, fields(user = user.username), err(Debug))]
async fn upload_image(
    user: AuthedUser,
    State(state): State<AppState>,
    mut form: Multipart,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user
    let mut user = user.require_full()?;
    // find the image field in the multipart form and stream it to S3
    while let Some(field) = form.next_field().await? {
        if field.name() == Some("image") {
            user.set_image(field, &state.shared).await?;
            return Ok(StatusCode::NO_CONTENT);
        }
    }
    // no image field was present in the form
    bad!("Missing 'image' field in multipart form".to_owned())
}

/// Removes the current user's profile icon
///
/// # Arguments
///
/// * `user` - The user removing their icon
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/users/image",
    responses(
        (status = 204, description = "The profile icon was removed"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::delete_image", skip_all, fields(user = user.username), err(Debug))]
async fn delete_image(
    user: AuthedUser,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // require a fully authed user
    let mut user = user.require_full()?;
    // clear this user's icon from S3 and the user record
    user.delete_image(&state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Gets a user's profile icon
///
/// Any authenticated user may fetch any user's icon so avatars can be shown next to names.
///
/// # Arguments
///
/// * `user` - The requesting user (must be authenticated)
/// * `username` - The user whose icon to fetch
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/users/user/{username}/image",
    params(
        ("username" = String, Path, description = "The user whose icon to fetch"),
    ),
    responses(
        (status = 200, description = "The profile icon was retrieved"),
        (status = 404, description = "The user does not exist or has no icon"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::get_image", skip(_user, state), err(Debug))]
async fn get_image(
    _user: AuthedUser,
    Path(username): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // get the target user
    let target = User::force_get(&username, &state.shared).await?;
    // stream their icon back if they have one
    match &target.image {
        Some(image_path) => {
            // download the icon from S3 with its metadata; if the referenced object is missing (a dangling
            // reference) treat it as "no icon" (404) instead of surfacing a 400 with raw S3 internals
            let get_object = match target.download_graphic(image_path, &state.shared).await {
                Ok(get_object) => get_object,
                Err(_) => return not_found!(format!("User '{username}' has no image")),
            };
            // build response headers (content-type, disposition, length) from the object
            let headers = graphics::get_headers(&get_object, image_path);
            // stream the body back with its headers
            let body = AsyncReadBody::new(get_object.body.into_async_read());
            Ok((headers, body))
        }
        None => not_found!(format!("User '{username}' has no image")),
    }
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(list, create, update, resend_email_verification, verify_email, list_details, auth, get_user, update_user, info, logout, logout_user, delete_user, sync_ldap, create_scoped_token, list_scoped_tokens, get_scoped_token, update_scoped_token, delete_scoped_token, upload_image, delete_image, get_image),
    components(schemas(AuthResponse, ScrubbedUser, Theme, UnixInfo, User, UserCreate, UserRole, UserSettings, UserSettingsUpdate, UserUpdate, AiSettings, AiSettingsUpdate, AiEndpoint, AiEndpointUpdate, ScopedToken, ScopedTokenRequest, ScopedTokenUpdate)),
    modifiers(&OpenApiSecurity),
)]
pub struct UserApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(UserApiDocs::openapi())
}

/// Add the file routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/users/", get(list).post(create).patch(update))
        .route(
            "/users/resend/verify/email/{username}",
            get(resend_email_verification),
        )
        .route(
            "/users/verify/{username}/email/{verification_token}",
            get(verify_email),
        )
        .route("/users/details/", get(list_details))
        .route("/users/auth", post(auth))
        .route(
            "/users/tokens/",
            get(list_scoped_tokens).post(create_scoped_token),
        )
        .route(
            "/users/tokens/{name}",
            get(get_scoped_token)
                .patch(update_scoped_token)
                .delete(delete_scoped_token),
        )
        .route("/users/user/{username}", get(get_user).patch(update_user))
        .route("/users/whoami", get(info))
        // cap the icon upload body only (the global limit is disabled for large file uploads); delete carries
        // no body so it's left unwrapped
        .route(
            "/users/image",
            post(upload_image)
                .layer(DefaultBodyLimit::max(ICON_MAX_BYTES))
                .delete(delete_image),
        )
        .route("/users/user/{username}/image", get(get_image))
        .route("/users/logout", post(logout))
        .route("/users/logout/{target}", get(logout_user))
        .route("/users/delete/{target}", delete(delete_user))
        .route("/users/sync/ldap", post(sync_ldap))
}
