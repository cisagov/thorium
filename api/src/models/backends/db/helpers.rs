use bb8_redis::{bb8, RedisConnectionManager};
use std::collections::HashMap;

use crate::utils::{ApiError, Shared};
use crate::{bad, unavailable};

/// Gets a connection from the connection pool
#[doc(hidden)]
#[macro_export]
macro_rules! conn {
    ($shared:expr) => {
        &mut *$crate::models::backends::db::helpers::get_conn($shared).await?
    };
}

/// Perform a non pipelined query to Redis
#[doc(hidden)]
#[macro_export]
macro_rules! query {
    ($cmd:expr, $shared:expr) => {
        $cmd.query_async(&mut *$crate::models::backends::db::helpers::get_conn($shared).await?)
    };
}

/// Perform a non pipelined query with no return value to Redis
#[doc(hidden)]
#[macro_export]
macro_rules! exec_query {
    ($cmd:expr, $shared:expr) => {
        $cmd.exec_async(&mut *$crate::models::backends::db::helpers::get_conn($shared).await?)
    };
}

/// Adds a pipelined hsetnx to a redis pipeline if an option is set and serialize
#[doc(hidden)]
#[macro_export]
macro_rules! hsetnx_opt_serialize {
    ($pipe:expr, $key:expr, $field:expr, $value:expr) => {
        // insert pipelined command if this value is set
        if let Some(val) = $value {
            $pipe
                .cmd("hsetnx")
                .arg($key)
                .arg($field)
                .arg(serialize!(val));
        }
    };
}

/// Adds a pipelined hset to a redis pipeline if an option is set and serialize
#[doc(hidden)]
#[macro_export]
macro_rules! hset_opt_serialize {
    ($pipe:expr, $key:expr, $field:expr, $value:expr) => {
        // insert pipelined command if this value is set
        if let Some(val) = $value {
            $pipe.cmd("hset").arg($key).arg($field).arg(serialize!(val));
        }
    };
}

/// Adds a pipelined hset to a redis pipeline if an option is set and serialize
///
/// Deletes if the option is none
#[doc(hidden)]
#[macro_export]
macro_rules! hset_del_opt_serialize {
    ($pipe:expr, $key:expr, $field:expr, $value:expr) => {
        // insert pipelined command if this value is set
        if let Some(val) = $value {
            $pipe.cmd("hset").arg($key).arg($field).arg(serialize!(val));
        } else {
            $pipe.cmd("hdel").arg($key).arg($field);
        }
    };
}

/// Gets a connection from the Redis connection pool
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn get_conn(
    shared: &Shared,
) -> Result<bb8::PooledConnection<'_, RedisConnectionManager>, ApiError> {
    // get connection from redis pool
    match shared.redis.get().await {
        Ok(conn) => Ok(conn),
        Err(error) => unavailable!(format!("Failed to get connection from pool: {:#?}", error)),
    }
}

/// Extracts a value from a hashmap or returns a helpful error
///
/// # Arguments
///
/// * `map` - The hashmap to extract from
/// * `key` - The key to extract
pub fn extract(map: &mut HashMap<String, String>, key: &str) -> Result<String, ApiError> {
    match map.remove(key) {
        Some(value) => Ok(value),
        None => bad!(format!("Failed to extract {}", key)),
    }
}

/// Coerces a string to a bool
///
/// # Arguments
///
/// * `key` - The name of the data that is being coerced
/// * `raw` - The string to coerce to a bool
pub fn coerce_bool(key: &str, raw: &str) -> Result<bool, ApiError> {
    match raw {
        "0" | "false" => Ok(false),
        "1" | "true" => Ok(true),
        val => bad!(format!("Failed to coerce {}({}) to bool", key, val)),
    }
}

/// Extracts a bool from a hashmap or returns a helpful error
///
/// # Arguments
///
/// * `map` - The hashmap to extract from
/// * `key` - The key to extract
pub fn extract_bool(map: &mut HashMap<String, String>, key: &str) -> Result<bool, ApiError> {
    match map.get(key) {
        Some(value) => coerce_bool(key, value),
        None => bad!(format!("Failed to extract {}", key)),
    }
}

/// Extracts a bool from a hashmap with a default or returns a helpful error
///
/// # Arguments
///
/// * `map` - The hashmap to extract from
/// * `key` - The key to extract
/// * `default` - The default to use if this bool is not set
pub fn extract_bool_default(
    map: &mut HashMap<String, String>,
    key: &str,
    default: bool,
) -> Result<bool, ApiError> {
    match map.get(key) {
        Some(value) => coerce_bool(key, value),
        None => Ok(default),
    }
}

/// Extracts an optional value from a hashmap or returns a helpful error
///
/// # Arguments
///
/// * `map` - hashmap to extract from
/// * `key` - key to extract
pub fn extract_opt(map: &mut HashMap<String, String>, key: &str) -> Option<String> {
    // short circuit and return None if a key does not exist
    if !map.contains_key(key) {
        return None;
    }

    // extract string
    let val = map.remove(key).unwrap();
    // cast to option wrapped string or None if its "null"
    match val.as_ref() {
        "null" | "None" => None,
        _ => Some(val),
    }
}

/// Check that a given value exists in the Redis set
/// at the given key
///
/// # Arguments
///
/// * `values` - The values to check for
/// * `set_key` - The key to the Redis set
/// * `shared` - The Thorium shared object
pub async fn exists<T>(value: T, set_key: T, shared: &Shared) -> Result<bool, ApiError>
where
    T: AsRef<str>,
{
    let exists: bool = redis::cmd("sismember")
        .arg(set_key.as_ref())
        .arg(value.as_ref())
        .query_async(conn!(shared))
        .await?;
    Ok(exists)
}

/// Check that all values in a given list exist in the Redis set
/// at the given key
///
/// # Arguments
///
/// * `values` - The values to check for
/// * `set_key` - The key to the Redis set
/// * `shared` - The Thorium shared object
pub async fn exists_all(
    values: &[String],
    set_key: &str,
    shared: &Shared,
) -> Result<bool, ApiError> {
    // check if every value is in the set in one Redis pipeline
    let checks: Vec<bool> = values
        .iter()
        .fold(redis::pipe().atomic(), |pipe, value| {
            pipe.cmd("sismember").arg(set_key).arg(value)
        })
        .query_async(conn!(shared))
        .await?;
    // check that all of the values exist
    let all_exist = !checks.contains(&false);
    Ok(all_exist)
}

/// Checks if a user is an admin or not and uses the right groups for the action requested
///
/// # Arguments
///
/// * `action` - The function to call
/// * `user` - The user that is performing this action
/// * `shared` - Shared objects in Thorium
/// * `args` - Any extra args to pass between groups and shared
#[doc(hidden)]
#[macro_export]
macro_rules! for_groups {
    ($action:expr, $user:expr, $shared:expr, $($args:expr),*) => {
        // for users we can search their groups but for admins we need to get all groups
        if $user.role == $crate::models::UserRole::Admin {
            // TODO: make this better then a guess of under 1000 groups
            let cursor = $crate::models::backends::backends_reexport::db::groups::list(0, 1000, $shared).await?;
            // search for this sample in all groups
            $action(&cursor.names, $($args,)* $shared).await
        } else {
            // search for this sample in all the user groups
            $action(&$user.groups, $($args,)* $shared).await
        }
    };
    // handle where no extra args are needed
    ($action:expr, $user:expr, $shared:expr) => {
        // for users we can search their groups but for admins we need to get all groups
        if $user.role == $crate::models::UserRole::Admin {
            // TODO make this better then a guess of under 1000 groups
            let cursor = $crate::models::backends::backends_reexport::db::groups::list(0, 1000, $shared).await?;
            // search for this sample in all groups
            $action(&cursor.names, $shared).await
        } else {
            // search for this sample in all the user groups
            $action(&$user.groups, $shared).await
        }
    }
}

/// Checks if two vectors are the same
#[doc(hidden)]
#[macro_export]
macro_rules! same_vec {
    ($left:expr, $right:expr) => {
        if $left.len() != $right.len() {
            false
        } else {
            // make sure that B contains all elements in A
            if !$left.iter().all(|x| $right.iter().any(|r| r == x))
                || !$right.iter().all(|x| $left.iter().any(|l| l == x))
            {
                false
            } else {
                true
            }
        }
    };
}
