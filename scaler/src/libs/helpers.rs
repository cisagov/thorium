use futures::Stream;
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use rand::Rng;
use regex::Regex;
use thorium::Error;

/// Serialize a value to a string
#[macro_export]
macro_rules! serialize {
    ($data:expr) => {
        match serde_json::to_string($data) {
            Ok(serial) => serial,
            Err(e) => {
                return Err(Error::new(format!(
                    "Failed to serialize data with error {}",
                    e
                )))
            }
        }
    };
}

/// Serialize a value to a string and wrap it in single quotes
#[macro_export]
macro_rules! serialize_wrap {
    ($data:expr) => {
        match serde_json::to_string($data) {
            Ok(serial) => format!("'{}'", serial),
            Err(e) => {
                return Err(Error::new(format!(
                    "Failed to serialize data with error {}",
                    e
                )))
            }
        }
    };
}

/// Extract a value from a hashmap or throw an error
#[macro_export]
macro_rules! extract {
    ($map:expr, $key:expr) => {
        match $map.remove($key) {
            Some(val) => val,
            None => return Err(Error::new(format!("HashMap missing value {}", $key))),
        }
    };
}

/// Converts a cpu quantity to millicpu
///
/// # Arguments
///
/// * `raw` - Raw cpu value
pub fn cpu(raw: Option<&Quantity>) -> Result<u64, Error> {
    // if raw is None then return 0
    let raw = match raw {
        Some(raw) => raw,
        None => return Ok(0),
    };
    // cast quantity to string
    let raw: String = serde_json::from_value(serde_json::json!(raw))?;
    // convert our raw value to millicpu
    let converted = thorium::models::conversions::cpu(raw)?;
    Ok(converted)
}

/// Converts a storage quantity to mebibytes
///
/// # Arguments
///
/// * `raw` - Raw storage value
pub fn storage(raw: Option<&Quantity>) -> Result<u64, Error> {
    // if raw is None then return 0
    let raw = match raw {
        Some(raw) => raw,
        None => return Ok(0),
    };
    // cast quantity to string
    let raw: String = serde_json::from_value(serde_json::json!(raw))?;
    // convert our raw value to mebibytes
    let mebibytes = thorium::models::conversions::storage(raw)?;
    Ok(mebibytes)
}

/// Converts an integer quantity to a ``u64``
///
/// # Arguments
///
/// * `raw` - Raw integer value
pub fn u64_quantity(raw: Option<&Quantity>) -> Result<u64, Error> {
    // if raw is None then return 0
    let raw = match raw {
        Some(raw) => raw,
        None => return Ok(0),
    };
    // cast quantity to string
    let raw: String = serde_json::from_value(serde_json::json!(raw))?;
    // try to cast this quantity as a u64
    let worker_slots = raw.parse::<u64>()?;
    Ok(worker_slots)
}

/// Generates a random string from [a-z, 0-9]
///
/// # Arguments
///
/// * `len` - The length of the string to generate
pub fn gen_string(len: usize) -> String {
    // build charset to pull chars from
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz\
                           0123456789";
    // get some rng and build string 12 chars long
    let mut rng = rand::rng();
    (0..len)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// drops any return and logs any errors
/// This will log but suppress any errors
#[macro_export]
macro_rules! check {
    ($attempt:expr, $log:expr) => {
        match $attempt.await {
            Ok(_) => (),
            Err(e) => slog::error!($log, "{:#?}", e),
        }
    };
}

/// gets a timestamp N seconds from now
#[macro_export]
macro_rules! from_now {
    ($seconds:expr) => {
        chrono::Utc::now() + chrono::Duration::seconds($seconds)
    };
}

/// checks that two things are the same and returns false if not
#[macro_export]
macro_rules! same {
    ($left:expr, $right:expr) => {
        if $left != $right {
            return false;
        }
    };
}

/// push a value into a vec at the given map key without cloning the key using
/// the `RawEntryMut` API from hasbrown::HashMap
#[macro_export]
macro_rules! raw_entry_vec_push {
    ($map:expr, $key:expr, $value:expr) => {
        let (_key, vec) = $map
            .raw_entry_mut()
            .from_key($key)
            .or_insert($key.clone(), Vec::default());
        vec.push($value);
    };
}

/// extend values to a vec at the given map key without cloning the key using
/// the `RawEntryMut` API from hasbrown::HashMap
#[macro_export]
macro_rules! raw_entry_vec_extend {
    ($map:expr, $key:expr, $values:expr) => {
        let (_key, vec) = $map
            .raw_entry_mut()
            .from_key($key)
            .or_insert($key.clone(), Vec::default());
        vec.extend($values);
    };
}

/// insert a key/value pair to an inner map at the given map key without
/// cloning the key using the `RawEntryMut` API from hasbrown::HashMap
#[macro_export]
macro_rules! raw_entry_map_insert {
    ($map:expr, $key:expr, $inner_key:expr, $value:expr) => {
        let (_key, inner_map) = $map
            .raw_entry_mut()
            .from_key($key)
            .or_insert($key.into(), HashMap::default());
        inner_map.insert($inner_key, $value);
    };
}

/// extend an inner map at the given map key without
/// cloning the key using the `RawEntryMut` API from hasbrown::HashMap
#[macro_export]
macro_rules! raw_entry_map_extend {
    ($map:expr, $key:expr, $extend:expr) => {
        let (_key, inner_map) = $map
            .raw_entry_mut()
            .from_key($key)
            .or_insert($key.into(), HashMap::default());
        inner_map.extend($extend);
    };
}

/// Resolves `FnOnce` errors by asserting an iterator is Send
///
/// See <https://users.rust-lang.org/t/implementation-of-fnonce-is-not-general-enough-with-async-block/83427/3>
///
/// # Arguments
///
/// * `it` - The iterator to assert
pub fn assert_send_stream<R>(it: impl Send + Stream<Item = R>) -> impl Send + Stream<Item = R> {
    it
}
