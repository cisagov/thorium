use bb8_redis::redis::cmd;
use chrono::prelude::*;
use std::collections::{HashMap, HashSet};
use tracing::{Level, event, instrument};

use super::helpers;
use super::keys::{EventKeys, GroupKeys, SystemKeys, UserKeys};
use crate::models::{ScopedToken, UnixInfo, User, UserRole, UserSettings};
use crate::utils::{ApiError, Shared};
use crate::{
    conflict, conn, deserialize, deserialize_ext, deserialize_opt, extract, not_found, query,
    serialize, unauthorized,
};

/// Builds a user creation pipeline for Redis
///
/// This will give this user the user role in any groups that are in its group list. Currently only
/// service accounts should have any groups when being created.
///
/// # Arguments
///
/// * `pipe` - The redis pipeline to add onto
/// * `keys` - The redis keys to save this users data too
/// * `cast` - The user to create in redis
/// * `set_username` - Wheether to also set a users username
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub fn build(
    pipe: &mut redis::Pipeline,
    keys: &UserKeys,
    cast: &User,
    set_username: bool,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build the key to our event cache status flags
    let cache_status = EventKeys::cache(shared);
    // build pipeline to save a user into redis
    pipe.cmd("hsetnx").arg(&keys.data).arg("email").arg(&cast.email)
        .cmd("hsetnx").arg(&keys.data).arg("role").arg(serialize!(&cast.role))
        .cmd("sadd").arg(&keys.global).arg(&cast.username)
        .cmd("hsetnx").arg(&keys.data).arg("token").arg(&cast.token)
        .cmd("hsetnx").arg(&keys.data).arg("token_expiration")
            .arg(serialize!(&cast.token_expiration))
        .cmd("hset").arg(SystemKeys::data(shared)).arg("scaler_cache").arg(true)
        .cmd("hsetnx").arg(&keys.by_token).arg(&cast.token).arg(&cast.username)
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        .cmd("hsetnx").arg(&keys.data).arg("settings").arg(serialize!(&cast.settings))
        .cmd("hsetnx").arg(&keys.data).arg("verified").arg(cast.verified)
        .cmd("hsetnx").arg(&keys.data).arg("aliases").arg(serialize!(&cast.aliases))
        // add this users email to username map
        .cmd("hsetnx").arg(&keys.by_email).arg(&cast.email).arg(&cast.username);
    // set a users username if enabled
    if set_username {
        pipe.cmd("hsetnx").arg(&keys.data).arg("username").arg(&cast.username);
    }
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
    // if a profile icon has been set then set that in redis
    if let Some(image) = &cast.image {
        pipe.cmd("hsetnx").arg(&keys.data).arg("image").arg(image);
    }
    // add any aliases for this user
    for (provider, alias) in &cast.aliases {
        // build the key to this providers alias map
        let alias_key = super::keys::oauth::alias_to_username(provider, shared);
        // add this alias to our alias map
        pipe.cmd("hsetnx").arg(alias_key).arg(alias).arg(&cast.username);
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
    // build keys to user data
    let keys = UserKeys::new(&cast, shared);
    // try to reserve this username
    let is_new: bool = redis::cmd("hsetnx").arg(&keys.data).arg("username").arg(&cast.username)
        .query_async(conn!(shared)).await?;
    // if this username wasn't yet taken then proceed otherwise abort
    if is_new {
        // build pipeline to save a user into redis
        let mut pipe = redis::pipe();
        // add the commands to create the rest of our user
        build(&mut pipe, &keys, &cast, false, shared)?;
        // try to save user into redis
        match pipe.atomic().exec_async(conn!(shared)).await {
            // we successfully created this user
            Ok(()) => Ok(cast),
            // we ran into a problem creating this user
            Err(error) => {
                // log that we are rolling back reserver this username
                event!(Level::INFO, msg="Rollback username reservation", username=&cast.username);
                // rollback reserving this username
                // its likely this will fail since it probably means redis is down
                pipe.cmd("hdel").arg(&keys.data).arg("username").arg(&cast.username)
                    .exec_async(conn!(shared)).await?;
                // reemit our error
                Err(error.into())
            }
        }
    } else {
        // this username is already taken so throw a conflict error
        conflict!(format!("Username {} is already taken", cast.username))
    }
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
        email: extract!(raw, "email"),
        username,
        password: helpers::extract_opt(&mut raw, "password"),
        role: deserialize_ext!(raw, "role"),
        groups,
        actual_groups: Vec::default(),
        unix: deserialize_opt!(raw, "unix"),
        token: extract!(raw, "token"),
        token_expiration: deserialize_ext!(raw, "token_expiration"),
        settings: deserialize_ext!(raw, "settings", UserSettings::default()),
        verified: helpers::extract_bool_default(&mut raw, "verified", true)?,
        verification_token: helpers::extract_opt(&mut raw, "verification_token"),
        verification_sent: deserialize_opt!(&mut raw, "verification_sent"),
        aliases: deserialize_ext!(raw, "aliases", HashMap::default()),
        image: helpers::extract_opt(&mut raw, "image"),
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
        // swap this users groups to the actual groups before replacing it
        std::mem::swap(&mut user.groups, &mut user.actual_groups);
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
            // build keys to user data
            let keys = UserKeys::new(user, shared);
            // restore this users groups
            restore_groups(&mut pipe, user, shared);
            // restore the rest of the users data
            build(&mut pipe, &keys, user, true, shared)
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
#[rustfmt::skip]
#[instrument(name = "db::users::exists", skip(shared), err(Debug))]
pub async fn exists(username: &str, shared: &Shared) -> Result<bool, ApiError> {
    // build key to users set
    let key = UserKeys::global(shared);
    // make sure a user exists
    let check: bool = redis::cmd("sismember").arg(&key).arg(username)
        .query_async(conn!(shared))
        .await?;
    Ok(check)
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
        .cmd("hset").arg(&keys.by_token).arg(&user.token).arg(&user.username)
        .cmd("hdel").arg(&keys.by_token).arg(old)
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        .cmd("hset").arg(&system_map).arg("scaler_cache").arg("true")
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// The different types of tokens a raw token value can match in Redis
pub enum TokenMatch {
    /// This token is a users primary token
    Primary(String),
    /// This token is a scoped token tied to a user
    Scoped(ScopedToken),
}

/// Gets the user or scoped token tied to a token value from Redis
///
/// This checks the primary and scoped token maps in a single round trip.
///
/// # Arguments
///
/// * `token` - The token value to look up
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::get_token_match", skip_all, err(Debug))]
pub async fn get_token_match(token: &str, shared: &Shared) -> Result<TokenMatch, ApiError> {
    // build keys to the primary and scoped token maps
    let primary_key = UserKeys::tokens(shared);
    let scoped_key = UserKeys::scoped_tokens(shared);
    // look for this token in both maps in a single round trip
    let (primary, scoped): (Option<String>, Option<String>) = redis::pipe()
        .cmd("hget")
        .arg(&primary_key)
        .arg(token)
        .cmd("hget")
        .arg(&scoped_key)
        .arg(token)
        .query_async(conn!(shared))
        .await?;
    // check if this token is a primary or scoped token
    match (primary, scoped) {
        // this token is a users primary token
        (Some(username), _) => Ok(TokenMatch::Primary(username)),
        // this token is a scoped token so get its data
        (None, Some(entry)) => {
            // deserialize the owner and name of this scoped token
            let (owner, name): (String, String) = deserialize!(&entry);
            // get this scoped tokens data
            let scoped = get_scoped_token(&owner, &name, shared).await?;
            Ok(TokenMatch::Scoped(scoped))
        }
        // this token doesn't exist so bounce this user
        (None, None) => unauthorized!(),
    }
}

/// Creates a scoped token in Redis
///
/// # Arguments
///
/// * `token` - The scoped token to create
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::create_scoped_token", skip_all, fields(user = token.owner, name = token.name), err(Debug))]
pub async fn create_scoped_token(token: &ScopedToken, shared: &Shared) -> Result<(), ApiError> {
    // build the keys to this users scoped token data and the scoped token map
    let data_key = UserKeys::scoped_data(&token.owner, shared);
    let map_key = UserKeys::scoped_tokens(shared);
    // try to reserve this scoped tokens name for this user
    let is_new: bool = redis::cmd("hsetnx").arg(&data_key).arg(&token.name).arg(serialize!(token))
        .query_async(conn!(shared)).await?;
    // if this name wasn't yet taken then proceed otherwise abort
    if is_new {
        // add this scoped token to the global scoped token map
        match redis::cmd("hset").arg(&map_key).arg(&token.token)
            .arg(serialize!(&(&token.owner, &token.name)))
            .query_async::<()>(conn!(shared)).await
        {
            // we successfully created this scoped token
            Ok(()) => Ok(()),
            // we ran into a problem creating this scoped token
            Err(error) => {
                // log that we are rolling back reserving this scoped tokens name
                event!(Level::INFO, msg = "Rollback scoped token reservation", name = &token.name);
                // rollback reserving this scoped tokens name
                // its likely this will fail since it probably means redis is down
                redis::cmd("hdel").arg(&data_key).arg(&token.name)
                    .query_async::<()>(conn!(shared)).await?;
                // reemit our error
                Err(error.into())
            }
        }
    } else {
        // this scoped token name is already taken so throw a conflict error
        conflict!(format!("Scoped token {} already exists", token.name))
    }
}

/// Gets a specific users scoped token from Redis
///
/// # Arguments
///
/// * `username` - The username of the user this scoped token is tied to
/// * `name` - The name of the scoped token to get
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::get_scoped_token", skip(shared), err(Debug))]
pub async fn get_scoped_token(
    username: &str,
    name: &str,
    shared: &Shared,
) -> Result<ScopedToken, ApiError> {
    // build the key to this users scoped token data
    let key = UserKeys::scoped_data(username, shared);
    // get this scoped tokens data if it exists
    let raw: Option<String> = query!(cmd("hget").arg(&key).arg(name), shared).await?;
    // deserialize this scoped token if it exists
    match raw {
        Some(raw) => Ok(deserialize!(&raw)),
        None => not_found!(format!("Scoped token {name} not found")),
    }
}

/// Lists all of a users scoped tokens from Redis
///
/// # Arguments
///
/// * `username` - The username of the user to list scoped tokens for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::list_scoped_tokens", skip(shared), err(Debug))]
pub async fn list_scoped_tokens(
    username: &str,
    shared: &Shared,
) -> Result<Vec<ScopedToken>, ApiError> {
    // build the key to this users scoped token data
    let key = UserKeys::scoped_data(username, shared);
    // get all of this users scoped tokens
    let raw: HashMap<String, String> = query!(cmd("hgetall").arg(&key), shared).await?;
    // deserialize each of this users scoped tokens
    let mut tokens = Vec::with_capacity(raw.len());
    for entry in raw.values() {
        // deserialize this scoped token
        tokens.push(deserialize!(entry));
    }
    Ok(tokens)
}

/// Saves a scoped tokens rotated value into Redis
///
/// # Arguments
///
/// * `token` - The scoped token to save
/// * `old` - This scoped tokens old value
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::save_scoped_token", skip_all, fields(user = token.owner, name = token.name), err(Debug))]
pub async fn save_scoped_token(
    token: &ScopedToken,
    old: &str,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build the keys to this users scoped token data and the scoped token map
    let data_key = UserKeys::scoped_data(&token.owner, shared);
    let map_key = UserKeys::scoped_tokens(shared);
    // build pipeline to save this scoped tokens new value
    let _: () = redis::pipe().atomic()
        // update this scoped tokens data
        .cmd("hset").arg(&data_key).arg(&token.name).arg(serialize!(token))
        // update the scoped token map
        .cmd("hset").arg(&map_key).arg(&token.token)
            .arg(serialize!(&(&token.owner, &token.name)))
        .cmd("hdel").arg(&map_key).arg(old)
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Updates a scoped tokens data in Redis without changing its value
///
/// This must be used instead of [`save_scoped_token`] when a scoped tokens
/// value is unchanged since that functions pipeline removes the old value
/// from the scoped token map after adding the new one. With an unchanged
/// value that would delete the entry that was just written and break auth
/// for this scoped token.
///
/// # Arguments
///
/// * `token` - The scoped token to update
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::update_scoped_token", skip_all, fields(user = token.owner, name = token.name), err(Debug))]
pub async fn update_scoped_token(token: &ScopedToken, shared: &Shared) -> Result<(), ApiError> {
    // build the key to this users scoped token data
    let data_key = UserKeys::scoped_data(&token.owner, shared);
    // update this scoped tokens data
    let _: () = redis::cmd("hset").arg(&data_key).arg(&token.name).arg(serialize!(token))
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Deletes a scoped token from Redis
///
/// # Arguments
///
/// * `token` - The scoped token to delete
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::users::delete_scoped_token", skip_all, fields(user = token.owner, name = token.name), err(Debug))]
pub async fn delete_scoped_token(token: &ScopedToken, shared: &Shared) -> Result<(), ApiError> {
    // build the keys to this users scoped token data and the scoped token map
    let data_key = UserKeys::scoped_data(&token.owner, shared);
    let map_key = UserKeys::scoped_tokens(shared);
    // build pipeline to delete this scoped token
    let _: () = redis::pipe().atomic()
        .cmd("hdel").arg(&data_key).arg(&token.name)
        .cmd("hdel").arg(&map_key).arg(&token.token)
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Gets the username associated with an email if it exists
///
/// # Arguments
///
/// * `email` - The email to get a username for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::get_username_for_email", skip_all, err(Debug))]
pub async fn get_username_for_email(
    email: &str,
    shared: &Shared,
) -> Result<Option<String>, ApiError> {
    // build key to username/token map
    let key = UserKeys::by_email(shared);
    // get username for this email if it exists
    let username: Option<String> = query!(cmd("hget").arg(&key).arg(email), shared).await?;
    Ok(username)
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
        .cmd("hset").arg(&data_key).arg("settings").arg(serialize!(&user.settings))
        .cmd("hset").arg(&data_key).arg("aliases").arg(serialize!(&user.aliases));
    // if password is set then save that in redis
    if let Some(password) = &user.password {
        pipe.cmd("hset").arg(&data_key).arg("password").arg(password);
    }
    // save this users unix info if it is set
    if let Some(unix) = &user.unix {
        pipe.cmd("hset").arg(&data_key).arg("unix").arg(serialize!(unix));
    }
    // save this users profile icon if set, otherwise remove any existing one
    if let Some(image) = &user.image {
        pipe.cmd("hset").arg(&data_key).arg("image").arg(image);
    } else {
        pipe.cmd("hdel").arg(&data_key).arg("image");
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
    // add any aliases for this user
    for (provider, alias) in &user.aliases {
        // build the key to this providers alias map
        let alias_key = super::keys::oauth::alias_to_username(provider, shared);
        // add this alias to our alias map
        pipe.cmd("hset").arg(alias_key).arg(alias).arg(&user.username);
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
    pipe.cmd("hset").arg(&data_key).arg("unix").arg(serialize!(info));
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
    pipe.cmd("hset").arg(&data_key).arg("verification_token").arg(verification_token)
        .cmd("hset").arg(&data_key).arg("verification_sent").arg(serialize!(&Utc::now()));
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
/// * `pipe` - The redis pipeline to add onto
/// * `user` - The user to delete from redis
/// * `scoped_tokens` - The scoped tokens to delete for this user
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub fn build_delete(
    pipe: &mut redis::Pipeline,
    user: &User,
    scoped_tokens: &[ScopedToken],
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
    // remove from user and and user sets/maps
    pipe.cmd("srem").arg(&keys.global).arg(&user.username)
        .cmd("del").arg(&keys.data)
        .cmd("del").arg(&keys.groups)
        .cmd("hdel").arg(&keys.by_token).arg(&user.token)
        .cmd("hdel").arg(&keys.by_email).arg(&user.email).arg(&user.username);
    // remove any aliases for this user
    for (provider, alias) in &user.aliases {
        // build the key to this providers alias map
        let alias_key = super::keys::oauth::alias_to_username(provider, shared);
        // remove this alias from our alias map
        pipe.cmd("hdel").arg(alias_key).arg(alias).arg(&user.username);
    }
    // if this users role is analyst then remove them from the analyst set
    if user.role == UserRole::Analyst {
        // build the key to the analyst set
        let analyst_key = UserKeys::analysts(shared);
        // make sure this user is not in the analyst set
        pipe.cmd("srem").arg(analyst_key).arg(&user.username);
    }
    // build the key to the global scoped token map
    let scoped_map = UserKeys::scoped_tokens(shared);
    // remove all of this users scoped tokens from the scoped token map
    for scoped in scoped_tokens {
        pipe.cmd("hdel").arg(&scoped_map).arg(&scoped.token);
    }
    // delete this users scoped token data
    pipe.cmd("del").arg(UserKeys::scoped_data(&user.username, shared));
}

/// Delete a user from Redis
///
/// # Arguments
///
/// * `user` - The user to delete from redis
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::users::delete", skip_all, fields(user = user.username), err(Debug))]
pub async fn delete(user: &User, shared: &Shared) -> Result<(), ApiError> {
    // get this users scoped tokens so we can remove them from the scoped token map
    let scoped_tokens = list_scoped_tokens(&user.username, shared).await?;
    // build pipeline to delete this user from redis
    let mut pipe = redis::pipe();
    build_delete(&mut pipe, user, &scoped_tokens, shared);
    // try to delete this user from redis
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
