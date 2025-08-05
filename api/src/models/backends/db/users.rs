use bb8_redis::redis::cmd;
use std::collections::{HashMap, HashSet};
use tracing::instrument;

use super::helpers;
use super::keys::{EventKeys, GroupKeys, SystemKeys, UserKeys};
use crate::models::{UnixInfo, User, UserRole, UserSettings};
use crate::utils::{ApiError, Shared};
use crate::{
    conn, deserialize_ext, deserialize_opt, extract, not_found, query, serialize, unauthorized,
};

/// Builds a user creation pipeline for Redis
///
/// This will give this user the user role in any groups that are in its group list. Currently only
/// service accounts should have any groups when being created.
///
/// # Arguments
///
/// * `pipe` - The redis pipeline to add onto
/// * `cast` - The user to create in redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub fn build(
    pipe: &mut redis::Pipeline,
    cast: &User,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build keys to user data
    let keys = UserKeys::new(cast, shared);
    // build the key to our event cache status flags
    let cache_status = EventKeys::cache(shared);
    // build pipeline to save a user into redis
    pipe.cmd("hsetnx").arg(&keys.data).arg("username").arg(&cast.username)
        .cmd("hsetnx").arg(&keys.data).arg("email").arg(&cast.email)
        .cmd("hsetnx").arg(&keys.data).arg("role").arg(serialize!(&cast.role))
        .cmd("sadd").arg(&keys.global).arg(&cast.username)
        .cmd("hsetnx").arg(&keys.data).arg("token").arg(&cast.token)
        .cmd("hsetnx").arg(&keys.data).arg("token_expiration")
            .arg(serialize!(&cast.token_expiration))
        .cmd("hset").arg(&SystemKeys::data(shared)).arg("scaler_cache").arg(true)
        .cmd("hsetnx").arg(&keys.tokens).arg(&cast.token).arg(&cast.username)
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        .cmd("hsetnx").arg(&keys.data).arg("settings").arg(serialize!(&cast.settings))
        .cmd("hsetnx").arg(&keys.data).arg("verified").arg(cast.verified);
    // if password is set then set that in redis
    if let Some(password) = &cast.password {
        pipe.cmd("hsetnx").arg(&keys.data).arg("password").arg(password);
    }
    // if unix info has been set then set that in redis
    if let Some(unix) = &cast.unix {
        pipe.cmd("hsetnx").arg(&keys.data).arg("unix").arg(serialize!(&unix));
    }
    // if this users role is analyst then add them to the analyst set
    if cast.role == UserRole::Analyst {
        // build the key to the analyst set
        let analyst_key = UserKeys::analysts(shared);
        // insert this user into the analyst set
        pipe.cmd("sadd").arg(analyst_key).arg(&cast.username);
    }
    // if a verification token has been set then set that in redis
    if let Some(verification_token) = &cast.verification_token {
        pipe.cmd("hsetnx").arg(&keys.data).arg("verification_token")
            .arg(verification_token);
    }
    Ok(())
}

/// Creates a user in Redis
///
/// This will give this user the user role in any groups that are in its group list. Currently only
/// service accounts should have any groups when being created.
///
/// # Arguments
///
/// * `cast` - The user to create in redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::create", skip_all, err(Debug))]
pub async fn create(cast: User, shared: &Shared) -> Result<User, ApiError> {
    // build pipeline to save a user into redis
    let mut pipe = redis::pipe();
    build(&mut pipe, &cast, shared)?;
    // try to save user into redis
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(cast)
}

/// Cast a hashmap and list of groups into a User
///
/// # Arguments
///
/// * `raw` - The hashmap to cast to a user
/// * `groups` - The list of groups this user is in
#[instrument(name = "db::users::cast", skip_all, err(Debug))]
pub(super) fn cast(
    mut raw: HashMap<String, String>,
    groups: Vec<String>,
) -> Result<User, ApiError> {
    // return 404 if hashmap is empty
    if raw.is_empty() {
        return not_found!("user not found".to_owned());
    }
    // get this users username
    let username = extract!(raw, "username");
    // cast to a User document
    let user = User {
        email: extract!(raw, "email", format!("{}@unknown.unknown", &username)),
        username,
        password: helpers::extract_opt(&mut raw, "password"),
        role: deserialize_ext!(raw, "role"),
        groups,
        unix: deserialize_opt!(raw, "unix"),
        token: extract!(raw, "token"),
        token_expiration: deserialize_ext!(raw, "token_expiration"),
        settings: deserialize_ext!(raw, "settings", UserSettings::default()),
        verified: helpers::extract_bool_default(&mut raw, "verified", true)?,
        verification_token: helpers::extract_opt(&mut raw, "verification_token"),
    };
    Ok(user)
}

/// The raw data needed to cast to a user doc
type UserData = (HashMap<String, String>, Vec<String>);

/// Gets a user from Redis
///
/// # Arguments
///
/// * `username` - The username of the user to retrieve
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::get", skip(shared), err(Debug))]
pub async fn get(username: &str, shared: &Shared) -> Result<User, ApiError> {
    // build keys to user data
    let data_key = UserKeys::data(username, shared);
    let groups_key = UserKeys::groups(username, shared);
    // build redis pipeline to get this users data
    let data: Result<UserData, _> = redis::pipe()
        .cmd("hgetall")
        .arg(&data_key)
        .cmd("smembers")
        .arg(&groups_key)
        .query_async(conn!(shared))
        .await;
    // return 404 if we ran into an error
    let mut user = match data {
        Ok((data, groups)) => cast(data, groups)?,
        Err(_) => return not_found!("user not found".to_owned()),
    };
    // if this user is an admin or analyst then replace their group list with all groups
    if user.is_admin_or_analyst() {
        // build the key to all groups in Thorium
        let groups_key = GroupKeys::set(shared);
        // get all groups in Thorium
        let all_groups = query!(cmd("smembers").arg(&groups_key), shared).await?;
        // replace our users groups with
        user.groups = all_groups;
    }
    Ok(user)
}

/// Get a list of users with details to backup
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::backup", skip_all, err(Debug))]
pub async fn backup(shared: &Shared) -> Result<Vec<User>, ApiError> {
    // build key to user global set
    let key = UserKeys::global(shared);
    // get a list of user names
    let names: Vec<String> = query!(cmd("smembers").arg(key), shared).await?;
    // build a pipeline to retrieve all of our users data
    let mut pipe = redis::pipe();
    names.iter().fold(&mut pipe, |pipe, name| {
        pipe.cmd("hgetall")
            .arg(UserKeys::data(name, shared))
            .cmd("smembers")
            .arg(UserKeys::groups(name, shared))
    });
    // get raw user data
    let raw: Vec<UserData> = pipe.query_async(conn!(shared)).await?;
    // cast to user docs
    raw.into_iter()
        .map(|(data, groups)| cast(data, groups))
        .collect::<Result<Vec<User>, _>>()
}

/// Adds a user to groups based on the user object
///
/// This is only used during the restoration of users
///
/// # Arguments
///
/// * `pipe` - The redis pipeline to add onto
/// * `user` - The user to restore
/// * `shared` - Shared Thorium objects
fn restore_groups<'a>(pipe: &'a mut redis::Pipeline, user: &User, shared: &Shared) {
    // Build the key to the groups this user is in
    let groups_key = UserKeys::groups(&user.username, shared);
    // add the sadd commands for this users group
    user.groups.iter().fold(pipe, |pipe, name| {
        pipe.cmd("sadd").arg(&groups_key).arg(name)
    });
}

/// Restore user data
///
/// # Arguments
///
/// * `users` - The list of users to restore
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::restore", skip_all, err(Debug))]
pub async fn restore(users: &[User], shared: &Shared) -> Result<(), ApiError> {
    // build the our redis pipeline
    let mut pipe = redis::pipe();
    // crawl over users and build the pipeline to restore each one
    users
        .iter()
        .map(|user| {
            // restore this users groups
            restore_groups(&mut pipe, user, shared);
            // restore the rest of the users data
            build(&mut pipe, user, shared)
        })
        .collect::<Result<Vec<()>, ApiError>>()?;
    // restore all user data
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Checks if a set of users exist in Redis
///
/// # Arguments
///
/// * `usernames` - The usernames to check the existence of
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::exists_many", skip(shared), err(Debug))]
pub async fn exists_many(usernames: &HashSet<String>, shared: &Shared) -> Result<(), ApiError> {
    // build key to users set
    let key = UserKeys::global(shared);
    // make sure all of these users exist
    let checks: Vec<bool> = usernames
        .iter()
        .fold(redis::pipe().atomic(), |pipe, name| {
            pipe.cmd("sismember").arg(&key).arg(name)
        })
        .query_async(conn!(shared))
        .await?;
    // error if any of the username checks failed
    if checks.iter().any(|x| x == &false) {
        not_found!(format!("{} must all be valid users", serialize!(usernames)))
    } else {
        Ok(())
    }
}

/// Checks if a user exists in Redis
///
/// # Arguments
///
/// * `usernames` - The usernames to check the existence of
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::exists", skip(shared), err(Debug))]
pub async fn exists(username: &str, shared: &Shared) -> Result<(), ApiError> {
    // build key to users set
    let key = UserKeys::global(shared);
    // make sure a user exists
    let check: (bool,) = redis::pipe()
        .cmd("sismember")
        .arg(&key)
        .arg(username)
        .query_async(conn!(shared))
        .await?;
    if check.0 {
        Ok(())
    } else {
        not_found!(format!("{} is not a valid user", username))
    }
}

/// Gets a list of all valid users
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::list", skip_all, err(Debug))]
pub async fn list(shared: &Shared) -> Result<Vec<String>, ApiError> {
    // build key to users set
    let key = UserKeys::global(shared);
    // get the usernames of all valid users
    let users = query!(cmd("smembers").arg(key), shared).await?;
    Ok(users)
}

/// Saves a new token for a user into Redis
///
/// # Arguments
///
/// * `user` - The user to update the token of
/// * `old` - This users old token
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::save_token", skip_all, fields(user = user.username), err(Debug))]
pub async fn save_token(user: &User, old: &str, shared: &Shared) -> Result<(), ApiError> {
    // build keys to user data/map
    let keys = UserKeys::new(user, shared);
    let system_map = SystemKeys::data(shared);
    let cache_status = EventKeys::cache(shared);
    // build pipeline to save a users token
    let _: () = redis::pipe().atomic()
        // update this users info
        .cmd("hset").arg(&keys.data).arg("token").arg(&user.token)
        .cmd("hset").arg(&keys.data).arg("token_expiration")
            .arg(serialize!(&user.token_expiration))
        // update the token map
        .cmd("hset").arg(&keys.tokens).arg(&user.token).arg(&user.username)
        .cmd("hdel").arg(&keys.tokens).arg(old)
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        .cmd("hset").arg(&system_map).arg("scaler_cache").arg("true")
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Gets a users token from Redis
///
/// # Arguments
///
/// * `token` - The token of the user to retrieve
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::get_token", skip_all, err(Debug))]
pub async fn get_token(token: &str, shared: &Shared) -> Result<User, ApiError> {
    // build key to username/token map
    let key = UserKeys::tokens(shared);
    // get username for this token if it exists
    let username: Option<String> = query!(cmd("hget").arg(&key).arg(token), shared).await?;
    // if a username was found get it otherwise return unauthorized
    match username {
        // get this users data
        Some(username) => get(&username, shared).await,
        None => unauthorized!(),
    }
}

/// Saves a users data in Redis
///
/// # Arguments
///
/// * `user` - The user data to save
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::save", skip_all, fields(user = user.username), err(Debug))]
pub async fn save(user: &User, shared: &Shared) -> Result<(), ApiError> {
    // build key to user data
    let data_key = UserKeys::data(&user.username, shared);
    // build the key to our event cache status flags
    let cache_status = EventKeys::cache(shared);
    // build pipeline to save user into redis
    let mut pipe = redis::pipe();
    // set values that will always exist
    pipe.cmd("hset").arg(&data_key).arg("username").arg(&user.username)
        .cmd("hset").arg(&data_key).arg("groups").arg(serialize!(&user.groups))
        .cmd("hset").arg(&data_key).arg("role").arg(serialize!(&user.role))
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        .cmd("hset").arg(&data_key).arg("settings").arg(serialize!(&user.settings));
    // if password is set then save that in redis
    if let Some(password) = &user.password {
        pipe.cmd("hset").arg(&data_key).arg("password").arg(password);
    }
    // save this users unix info if it is set
    if let Some(unix) = &user.unix {
        pipe.cmd("hset").arg(&data_key).arg("unix").arg(serialize!(unix));
    }
    // build the key to the analyst set
    let analyst_key = UserKeys::analysts(shared);
    // if this users role is analyst then add them to the analyst set
    if user.role == UserRole::Analyst {
        pipe.cmd("sadd").arg(analyst_key).arg(&user.username);
    } else {
        // make sure this user is not in the analyst set
        pipe.cmd("srem").arg(analyst_key).arg(&user.username);
    }
    // save user into redis
    let _: () = pipe.atomic()
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// Updates a specific users unix info
///
/// # Arguments
///
/// * `username` - The name of the user whose info we are updating
/// * `info` - The updated unix info to save
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::update_unix_info", skip(shared), err(Debug))]
pub async fn update_unix_info(username: &str, info: &UnixInfo, shared: &Shared) -> Result<(), ApiError> {
    // build key to user data
    let data_key = UserKeys::data(username, shared);
    // build a redis pipeline
    let mut pipe = redis::pipe();
    // set our updated unix info
    pipe.cmd("hset").arg(&data_key).arg("unix").arg(serialize!(info))
        .cmd("hset").arg(&data_key).arg("email").arg(format!("{}@unknown.unknown", username));
    // save user into redis
    let _: () = pipe.atomic()
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// Updates a specific users verification token
///
/// # Arguments
///
/// * `username` - The name of the user whose info we are updating
/// * `verification_token` - The verification token to save
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::set_verification_token", skip(verification_token, shared), err(Debug))]
pub async fn set_verification_token(username: &str, verification_token: &str, shared: &Shared) -> Result<(), ApiError> {
    // build key to user data
    let data_key = UserKeys::data(username, shared);
    // build a redis pipeline
    let mut pipe = redis::pipe();
    // set our updated verification token
    pipe.cmd("hset").arg(&data_key).arg("verification_token").arg(verification_token);
    // save user into redis
    let _: () = pipe.atomic()
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// clears a specific users verification token
///
/// # Arguments
///
/// * `username` - The name of the user whose info we are updating
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::clear_verification_token", skip(shared), err(Debug))]
pub async fn clear_verification_token(username: &str, shared: &Shared) -> Result<(), ApiError> {
    // build key to user data
    let data_key = UserKeys::data(username, shared);
    // build a redis pipeline
    let mut pipe = redis::pipe();
    // set our updated verification token
    pipe
        .cmd("hset").arg(&data_key).arg("verified").arg(true)
        .cmd("hdel").arg(&data_key).arg("verification_token");
    // save user into redis
    let _: () = pipe.atomic()
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// builds a delete user pipeline from Redis
///
/// # Arguments
///
/// * `user` - The user to delete from redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub fn build_delete(
    pipe: &mut redis::Pipeline,
    user: &User,
    shared: &Shared,
) {
    // remove this user from all of its groups and all possible roles
    for group in &user.groups {
        // add the commands for removing from the combined sets
        pipe.cmd("srem").arg(GroupKeys::combined(group, "owners", shared)).arg(&user.username)
            .cmd("srem").arg(GroupKeys::combined(group, "managers", shared)).arg(&user.username)
            .cmd("srem").arg(GroupKeys::combined(group, "users", shared)).arg(&user.username)
            .cmd("srem").arg(GroupKeys::combined(group, "monitors", shared)).arg(&user.username)
            // add the commands for removing from the direct sets
            .cmd("srem").arg(GroupKeys::direct(group, "owners", shared)).arg(&user.username)
            .cmd("srem").arg(GroupKeys::direct(group, "managers", shared)).arg(&user.username)
            .cmd("srem").arg(GroupKeys::direct(group, "users", shared)).arg(&user.username)
            .cmd("srem").arg(GroupKeys::direct(group, "monitors", shared)).arg(&user.username);
    }
    // build keys to user data
    let keys = UserKeys::new(user, shared);
    // remove from user and user sets/maps
    pipe.cmd("srem").arg(&keys.global).arg(&user.username)
        .cmd("del").arg(&keys.data)
        .cmd("del").arg(&keys.groups)
        .cmd("hdel").arg(&keys.tokens).arg(&user.token);
    // if this users role is analyst then add them to the analyst set
    if user.role == UserRole::Analyst {
        // build the key to the analyst set
        let analyst_key = UserKeys::analysts(shared);
        // make sure this user is not in the analyst set
        pipe.cmd("srem").arg(analyst_key).arg(&user.username);
    }
}

/// Delete a user from Redis
///
/// # Arguments
///
/// * `user` - The user to delete from redis
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::delete", skip_all, fields(user = user.username), err(Debug))]
pub async fn delete(user: &User, shared: &Shared) -> Result<(), ApiError> {
    // build pipeline to save a user into redis
    let mut pipe = redis::pipe();
    build_delete(&mut pipe, user, shared);
    // try to save user into redis
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Get all analysts in Thorium
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn get_analysts(shared: &Shared) -> Result<HashSet<String>, ApiError> {
    // build the key to analysts in Thorium
    let key = UserKeys::analysts(shared);
    // get all analysts in Thorium from redis
    let analysts: HashSet<String> = query!(cmd("smembers").arg(key), shared).await?;
    Ok(analysts)
}
