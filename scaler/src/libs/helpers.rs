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

/// Bounds checks a cpu value and converts it to millicpu
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
    // try to cast this directly to a f64
    // This is because we assume that any f64 value is # of cores
    // if parse was successful then convert to millicpu
    if let Ok(cores) = raw.parse::<f64>() {
        return Ok((cores * 1000.0).ceil() as u64);
    }

    // f64 parse failed check if it ends in a millicpu unit
    if raw.ends_with('m') {
        // try to parse as millicpu
        let millicpu = raw[..raw.len() - 1].parse::<u64>();
        if millicpu.is_err() {
            return Err(Error::new(format!(
                "Invalid cpu value: {}",
                millicpu.unwrap()
            )));
        }
        return Ok(millicpu.unwrap());
    }
    // error if all of the cpu handlers failed
    Err(Error::new(format!("Failed to parse cpu value: {}", raw)))
}

/// Bounds checks an image storage value and converts it to
///
/// # Arguments
///
/// * `raw` - Raw cpu value
pub fn storage(raw: Option<&Quantity>) -> Result<u64, Error> {
    // if raw is None then return 0
    let raw = match raw {
        Some(raw) => raw,
        None => return Ok(0),
    };

    // cast quantity to string
    let raw: String = serde_json::from_value(serde_json::json!(raw))?;
    // try to cast this directly to a u64
    // This is because we assume that any u64 value is # of bytes
    // if parse was successful then convert to millicpu
    if let Ok(bytes) = raw.parse::<u64>() {
        // convert bytes to mebibytes
        return Ok((bytes as f64 / 1.049e+6).ceil() as u64);
    }

    // u64 failed parse check lets find first occurrence of a any valid char
    let unit_regex = Regex::new(r"[KMGTPE]").unwrap();
    // find index where unit starts
    let reg = match unit_regex.find(&raw) {
        Some(reg) => reg,
        None => return Err(Error::new(format!("failed to find parse {}", raw))),
    };
    // split raw based on where unit was found
    let (amt, unit) = raw.split_at(reg.start());
    // cast amt to u64
    let amt = amt.parse::<u64>()?;
    // convert to mebibytes
    let mebibytes = match unit {
        "K" => amt / 1049,
        "M" => (amt as f64 / 1.049).ceil() as u64,
        "G" => amt * 954,
        "T" => amt * 953674,
        "P" => (amt as f64 * 9.537e+8).ceil() as u64,
        "E" => (amt as f64 * 9.537e+11).ceil() as u64,
        "Ki" => amt / 1024,
        "Mi" => amt,
        "Gi" => amt * 1024,
        "Ti" => (amt as f64 * 1.049e+6).ceil() as u64,
        "Pi" => (amt as f64 * 1.074e+9).ceil() as u64,
        "Ei" => (amt as f64 * 1.1e+12).ceil() as u64,
        _ => {
            return Err(Error::new(format!(
                "Failed to parse storage value: {}",
                raw
            )))
        }
    };
    Ok(mebibytes)
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
