//! Helper macros for the Thorium API

/// Return internal server error if a function returns an error
#[doc(hidden)]
#[macro_export]
macro_rules! check {
    ($func:expr) => {
        match $func {
            Ok(val) => val,
            Err(_) => return Outcome::Failure((Status::InternalServerError, ())),
        }
    };
}

/// Return internal server error if a function does not return `Outcome::Success`
#[doc(hidden)]
#[macro_export]
macro_rules! outcome {
    ($func:expr) => {
        match $func {
            Outcome::Success(val) => val,
            _ => return Outcome::Failure((Status::InternalServerError, ())),
        }
    };
}

/// Serialize data to a string
#[doc(hidden)]
#[macro_export]
macro_rules! serialize {
    ($data:expr) => {
        match serde_json::to_string($data) {
            Ok(serial) => serial,
            Err(e) => return $crate::bad!(format!("Failed to serialize data with error {}", e)),
        }
    };
}

/// Serialize data to a string
#[doc(hidden)]
#[macro_export]
macro_rules! serialize_opt {
    ($data:expr) => {
        match $data {
            Some(data) => match serde_json::to_string(data) {
                Ok(serial) => Some(serial),
                Err(e) => return $crate::bad!(format!("Failed to serialize data with error {}", e)),
            },
            None => None,
        }
    };
}

/// Serialize data to a string or panic trying
#[doc(hidden)]
#[macro_export]
macro_rules! force_serialize {
    ($data:expr) => {
        match serde_json::to_string($data) {
            Ok(serial) => serial,
            Err(e) => panic!("Failed to serialize data with error {}", e),
        }
    };
}

/// Deserialize data from a string
#[doc(hidden)]
#[macro_export]
macro_rules! deserialize {
    ($data:expr) => {
        match serde_json::from_str($data) {
            Ok(serial) => serial,
            Err(e) => return $crate::bad!(format!("Failed to deserialize data with error {}", e)),
        }
    };
    ($data:expr, $key:expr) => {
        match serde_json::from_str($data) {
            Ok(serial) => serial,
            Err(e) => {
                return $crate::bad!(format!("Failed to deserialize {} with error {}", $key, e))
            }
        }
    };
}

/// Extract a string from a map and deserialize it
#[doc(hidden)]
#[macro_export]
macro_rules! deserialize_ext {
    ($map:expr, $key:expr) => {
        match $map.get($key) {
            Some(data) => $crate::deserialize!(data, $key),
            None => return $crate::bad!(format!("Failed to extract {}", $key)),
        }
    };
    ($map:expr, $key:expr, $def:expr) => {
        match $map.get($key) {
            Some(data) => $crate::deserialize!(data, $key),
            None => $def,
        }
    };
}

/// Deserialize data from a string wrapped in an option
#[doc(hidden)]
#[macro_export]
macro_rules! deserialize_opt {
    ($data:expr) => {
        match $data {
            Some(data) => Some($crate::deserialize!(data)),
            None => None,
        }
    };
    ($map:expr, $key:expr) => {
        match $map.get($key) {
            Some(data) => Some($crate::deserialize!(data, $key)),
            None => None,
        }
    };
    ($map:expr, $key:expr, $func:expr) => {
        match $map.get($key) {
            Some(data) => Some($func(data)?),
            None => None,
        }
    };
}

/// Deserialize data from a string but don't force a return
#[doc(hidden)]
#[macro_export]
macro_rules! deserialize_internal {
    ($data:expr) => {
        match serde_json::from_str($data) {
            Ok(serial) => serial,
            Err(e) => $crate::bad!(format!("Failed to deserialize data with error {}", e)),
        }
    };
}

/// Deserialize data from a string
#[doc(hidden)]
#[macro_export]
macro_rules! deserialize_value {
    ($data:expr) => {
        match serde_json::from_value($data) {
            Ok(serial) => serial,
            Err(e) => return $crate::bad!(format!("Failed to deserialize data with error {}", e)),
        }
    };
    ($data:expr, $key:expr) => {
        match serde_json::from_value($data) {
            Ok(serial) => serial,
            Err(e) => {
                return $crate::bad!(format!("Failed to deserialize {} with error {}", $key, e))
            }
        }
    };
}

/// Extract a value from a map
#[doc(hidden)]
#[macro_export]
macro_rules! extract {
    ($map:expr, $key:expr) => {
        match $map.remove($key) {
            Some(value) => value,
            None => return $crate::bad!(format!("Failed to extract {}", $key)),
        }
    };
    ($map:expr, $key:expr, $default:expr) => {
        match $map.remove($key) {
            Some(value) => value,
            None => $default,
        }
    };
}

/// Extract a bool from a map
#[doc(hidden)]
#[macro_export]
macro_rules! extract_bool {
    ($map:expr, $key:expr) => {
        match $map.get($key) {
            Some(value) => match &value[..] {
                "0" => false,
                "1" => true,
                "false" => false,
                "true" => true,
                val => return $crate::bad!(format!("Failed to coerce {}({}) to a bool", $key, val)),
            },
            None => return $crate::bad!(format!("Failed to extract {}", $key)),
        }
    };
}

/// Coerces a string into a boolean
#[doc(hidden)]
#[macro_export]
macro_rules! coerce_bool {
    ($raw:expr, $key:expr) => {
        match $raw.as_ref() {
            "0" => false,
            "1" => true,
            "false" => false,
            "true" => true,
            val => return $crate::bad!(format!("Failed to coerce {}({}) to bool", $key, val)),
        }
    };
}

/// Return unauthorized if the specified user is not an admin
#[doc(hidden)]
#[macro_export]
macro_rules! is_admin {
    ($user:expr) => {
        if $user.role != $crate::models::UserRole::Admin {
            // log this user failed an is admin check
            tracing::event!(
                tracing::Level::ERROR,
                not_admin = true,
                user = &$user.username
            );
            return $crate::unauthorized!();
        }
    };
}

/// Return unauthorized when a user does not have the role needed modify things
#[doc(hidden)]
#[macro_export]
macro_rules! can_modify {
    // for some objects a group check doesn't make sense so just check against the creator
    ($username:expr, $user:expr) => {
        // make sure we own this object or are an admin
        if $username != $user.username && $user.role != $crate::models::UserRole::Admin {
            return $crate::unauthorized!();
        }
    };
    ($username:expr, $group:expr, $user:expr) => {
        // make sure we have delete capabilities in this group or we own this pipeline
        if $username != $user.username && $group.editable($user).is_err() {
            return $crate::unauthorized!();
        }
    };
}

/// Return unauthorized when a user does not have the role needed modify things
#[doc(hidden)]
#[macro_export]
macro_rules! can_develop {
    ($username:expr, $group:expr, $scaler:expr, $user:expr) => {
        // make sure we have delete capabilities in this group or we own this pipeline
        if $username != $user.username && $group.developer($user, $scaler).is_err() {
            return $crate::unauthorized!();
        }
    };
}

/// Return unauthorized when a user does not have the role needed modify things
#[doc(hidden)]
#[macro_export]
macro_rules! can_develop_many {
    ($username:expr, $group:expr, $scalers:expr, $user:expr) => {
        // make sure we have delete capabilities in this group or we own this pipeline
        if $username != $user.username && $group.developer_many($user, $scalers).is_err() {
            return $crate::unauthorized!();
        }
    };
}

/// Return unauthorized when a user does not have the role needed to delete things
#[doc(hidden)]
#[macro_export]
macro_rules! can_delete {
    ($item:expr, $group:expr, $user:expr) => {
        // make sure we have delete capabilities in this group or we own this pipeline
        if $item.creator != $user.username && $group.modifiable($user).is_err() {
            return $crate::unauthorized!();
        }
    };
}

/// Return unauthorized when a user does not have the role needed to create things in all groups
#[doc(hidden)]
#[macro_export]
macro_rules! can_create_all {
    ($groups:expr, $user:expr, $shared:expr) => {
        // make sure we have delete capabilities in this group or we own this pipeline
        if $user.role != $crate::models::UserRole::Admin {
            for group in $groups.iter() {
                if group.editable($user).is_err() {
                    return $crate::unauthorized!();
                }
            }
        }
    };
}

/// Update a value if the new value is not None
#[doc(hidden)]
#[macro_export]
macro_rules! update {
    ($orig:expr, $update:expr) => {
        if let Some(new) = $update {
            $orig = new;
        }
    };
    // map the updated value with a fallible mapping function before setting the value
    ($orig:expr, $update:expr, $map:expr) => {
        if let Some(new) = $update {
            $orig = $map(new)?;
        }
    };
}

/// Update a value if the new value is not None by taking it from the Option
/// instead of consuming it
#[doc(hidden)]
#[macro_export]
macro_rules! update_take {
    ($orig:expr, $update:expr) => {
        if let Some(new) = $update.take() {
            $orig = new;
        }
    };
}

/// Update a value if the new value is not None and return the old value
#[doc(hidden)]
#[macro_export]
macro_rules! update_return_old {
    ($orig:expr, $update:expr) => {
        $update.take().map(|new| std::mem::replace(&mut $orig, new))
    };
}

/// Update a value if the new value is not None but keep it wrapped it in Option
#[doc(hidden)]
#[macro_export]
macro_rules! update_opt {
    ($orig:expr, $update:expr) => {
        if let Some(new) = $update.take() {
            $orig = Some(new);
        }
    };
}

/// Update a value if the new value is not None and clear it if its empty
#[doc(hidden)]
#[macro_export]
macro_rules! update_opt_empty {
    ($orig:expr, $update:expr) => {
        // get our updated value
        if let Some(new) = $update.take() {
            // check if new is empty or not
            if !new.is_empty() {
                // new isn't empty so set it
                $orig = Some(new);
            } else {
                $orig = None;
            }
        }
    };
}

/// Set a value to None if clear is true
#[doc(hidden)]
#[macro_export]
macro_rules! update_clear {
    ($orig:expr, $clear:expr) => {
        if $clear == true {
            $orig = None;
        }
    };
}

/// Logs an error that would normally be lost by an iterator filter
///
/// # Arguments
///
/// * `res` - The result to check for an error to log
#[cfg(feature = "api")]
pub fn log_err<T>(res: Result<T, crate::utils::ApiError>) -> Option<T> {
    // log error if it exists
    match res {
        Ok(res) => Some(res),
        Err(error) => {
            // log this error
            tracing::event!(tracing::Level::ERROR, msg = error.msg);
            None
        }
    }
}

/// Logs an error that would normally be lost by an iterator filter
#[doc(hidden)]
#[macro_export]
macro_rules! log_err {
    ($result:expr) => {
        // log error if it exists
        match $result {
            Ok(res) => Some(res),
            Err(error) => {
                tracing::event!(tracing::Level::ERROR, msg = &error.msg);
                None
            }
        }
    };
}

/// Logs a scylla error that would normally be lost by an iterator filter
#[doc(hidden)]
#[macro_export]
macro_rules! log_scylla_err {
    ($result:expr) => {
        // log error if it exists
        match $result {
            Ok(res) => Some(res),
            Err(error) => {
                let error = $crate::utils::ApiError::from(error);
                tracing::event!(tracing::Level::ERROR, msg = &error.msg);
                None
            }
        }
    };
}

/// Attempt to cast values to a type and log any errors
#[doc(hidden)]
#[macro_export]
macro_rules! cast {
    ($src:expr, $func:expr) => {
        $src.into_iter()
            .map($func)
            .filter_map(|res| $crate::log_err!(res))
            .collect()
    };
    ($src:expr, $func:expr, $part:tt) => {
        $src.into_iter()
            .map(|data| $func(data.$part))
            .filter_map(|res| $crate::log_err!(res))
            .collect()
    };
}

/// Attempt to cast values to a type and log any errors
#[doc(hidden)]
#[macro_export]
macro_rules! cast_extra {
    ($src:expr, $func:expr, $extra:expr) => {
        $src.into_iter()
            .map(|item| $func(item, $extra))
            .filter_map(|res| $crate::log_err!(res))
            .collect()
    };
}

/// Create an ldap connection and client
#[doc(hidden)]
#[macro_export]
macro_rules! ldap {
    ($conf:expr) => {
        ldap3::LdapConnAsync::with_settings(
            ldap3::LdapConnSettings::new().set_no_tls_verify(!$conf.tls_verify),
            &$conf.host,
        )
    };
}

/// Validate that all vectors are disjoint
///
/// passing in two separate vectors of items will not just uniqueness within a vector while passing
/// in a vector of vectors will.
#[doc(hidden)]
#[macro_export]
macro_rules! disjoint {
    // make sure two vectors have no shared values
    ($left:expr, $right:expr) => {
        if $left.iter().map(|val| $right.contains(val)).any(|x| x) {
            return bad!(format!("{:?} and {:?} must be disjoint", $left, $right));
        }
    };
    // make sure a vector of vectors are disjoint
    ($vecs:expr) => {{
        // if we have less than 200 items then its faster to just naively check each of them
        let size = $vecs.iter().fold(0, |acc, g| acc + g.len());
        // but the heavier hashmap implementation does
        if size < 200 {
            // Ensure each group
            for (i, &item) in $vecs.iter().enumerate() {
                // make sure no duplicates exist within each vector
                for val in item.iter() {
                    if item.iter().filter(|n| *n == val).count() > 1 {
                        return bad!(format!("{} cannot be in {:?} multiple times", val, item));
                    }
                }
                // Is disjoint with those following
                for other in &$vecs[i + 1..] {
                    $crate::disjoint!(item, other);
                }
            }
        // we have more than 200 total items so insert all items into a hashmap and error on dups
        } else {
            // build a mapping of what values have been seen in what set/vect
            let mut map: std::collections::HashMap<&String, usize> = HashMap::with_capacity(size);
            for (i, set) in $vecs.iter().enumerate() {
                for item in set.iter() {
                    // if we already inserted this value before then throw an error
                    if map.insert(item, i).is_some() {
                        return bad!(format!("{:?} and {:?} must be disjoint", set, $vecs[i]));
                    }
                }
            }
        }
    }};
}

// converts a vec of references to owned objects
#[doc(hidden)]
#[macro_export]
macro_rules! owned_vec {
    ($refs:expr) => {
        $refs.iter().map(|item| String::from(*item)).collect()
    };
}

// parse a date from a string or set a default
#[doc(hidden)]
#[macro_export]
macro_rules! parse_date {
    ($raw:expr, $default:expr) => {
        match $raw {
            Some(raw) => {
                chrono::DateTime::parse_from_rfc3339(&raw)?.with_timezone(&chrono::offset::Utc)
            }
            None => $default,
        }
    };
    ($raw:expr) => {
        parse_date!($raw, chrono::offset::Utc::now())
    };
}

// get a UTC timestamp at the configured earliest time
#[doc(hidden)]
#[macro_export]
macro_rules! earliest {
    ($shared:expr, $type:ident) => {
        chrono::offset::Utc.timestamp($shared.config.thorium.$type.earliest, 0)
    };
}
