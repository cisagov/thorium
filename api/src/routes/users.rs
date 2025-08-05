use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::Redirect;
use axum::routing::{delete, get, post};
use axum::Router;
use axum_extra::TypedHeader;
use tracing::instrument;
use utoipa::OpenApi;

use super::OpenApiSecurity;

// our imports
use crate::models::{
    AuthResponse, Key, ScrubbedUser, Theme, UnixInfo, User, UserCreate, UserRole, UserSettings,
    UserSettingsUpdate, UserUpdate,
};
use crate::utils::{ApiError, AppState};
use crate::{is_admin, unauthorized, unavailable};

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
#[axum_macros::debug_handler]
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

/// Resend our verification email if we are not yet verified
#[utoipa::path(
    get,
    path = "/api/users/resend/verify/email/:username",
    params(
        ("username" = String, Path, description = "The user to resend verification email for"),
    ),
    responses(
        (status = 200, description = "Verification email resent"),
        (status = 401, description = "This user is not authorized to access this route"),
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
) -> Result<StatusCode, ApiError> {
    // get the target user
    let mut user = User::force_get(&username, &state.shared).await?;
    // send a new verification email if we have email an email client
    match &state.shared.email {
        Some(client) => {
            // send a verification email to this user
            user.send_verification_email(client, &state.shared).await?;
            Ok(StatusCode::OK)
        }
        None => unavailable!("Email verification is not enabled!".to_owned()),
    }
}

/// Verifies an email for a specific user
#[utoipa::path(
    get,
    path = "/api/users/verify/:username/email/:verification_token",
    params(
        ("username" = String, Path, description = "The user to resend verification email for"),
        ("verification_token" = String, Path, description = "The token to send in the verification email"),
    ),
    responses(
        (status = 303, description = "Redirect to main page after sending verification email"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::users::verify_email", skip_all, err(Debug))]
async fn verify_email(
    Path((username, verification_token)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Redirect, ApiError> {
    // get the target user
    let mut user = User::force_get(&username, &state.shared).await?;
    // try to verify this users email
    user.verify_email(&verification_token, &state.shared)
        .await?;
    // redirect back to the webUI
    Ok(Redirect::to("/"))
}

/// Authenticates a user
///
/// # Arguments
///
/// * `user` - The authenticated user
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
async fn auth(user: User) -> Result<Json<AuthResponse>, ApiError> {
    // build auth response
    let resp = AuthResponse::from(user);
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
    user: User,
    Path(username): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ScrubbedUser>, ApiError> {
    // if user is an admin then allow them to get any user
    if user.is_admin() && user.username != username {
        let requested = User::force_get(&username, &state.shared).await?;
        Ok(Json(ScrubbedUser::from(requested)))
    // were requesting info on ourselves so just return it
    } else if user.username == username {
        Ok(Json(ScrubbedUser::from(user)))
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
async fn list(user: User, State(state): State<AppState>) -> Result<Json<Vec<String>>, ApiError> {
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
    user: User,
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
async fn info(user: User) -> Result<Json<ScrubbedUser>, ApiError> {
    Ok(Json(ScrubbedUser::from(user)))
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
    user: User,
    State(state): State<AppState>,
    Json(update): Json<UserUpdate>,
) -> Result<StatusCode, ApiError> {
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
    user: User,
    Path(name): Path<String>,
    State(state): State<AppState>,
    Json(update): Json<UserUpdate>,
) -> Result<StatusCode, ApiError> {
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
async fn logout(mut user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // generate and save a new token
    user.regen_token(&state.shared).await?;
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
    user: User,
    Path(target): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
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
    user: User,
    Path(target): Path<String>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
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
async fn sync_ldap(user: User, State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    // sync all groups with ldap
    user.sync_all_unix_info(&state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(list, create, update, resend_email_verification, verify_email, list_details, auth, get_user, update_user, info, logout, logout_user, delete_user, sync_ldap),
    components(schemas(AuthResponse, ScrubbedUser, Theme, UnixInfo, User, UserCreate, UserRole, UserSettings, UserSettingsUpdate, UserUpdate)),
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
        .route("/api/users/", get(list).post(create).patch(update))
        .route(
            "/api/users/resend/verify/email/{username}",
            get(resend_email_verification),
        )
        .route(
            "/api/users/verify/{username}/email/{verification_token}",
            get(verify_email),
        )
        .route("/api/users/details/", get(list_details))
        .route("/api/users/auth", post(auth))
        .route(
            "/api/users/user/{username}",
            get(get_user).patch(update_user),
        )
        .route("/api/users/whoami", get(info))
        .route("/api/users/logout", post(logout))
        .route("/api/users/logout/{target}", get(logout_user))
        .route("/api/users/delete/{target}", delete(delete_user))
        .route("/api/users/sync/ldap", post(sync_ldap))
}
