//! Bounds checking utilities for user input to Thorium

use std::collections::HashMap;

use regex::Regex;
use serde_json::Value;
use tracing::instrument;
use uuid::Uuid;

use super::{ApiError, Shared};
use crate::bad;
use crate::models::{EventTrigger, Group, Image, User};

/// Bounds check a string
///
/// This enforces a minimum and maximum size for a string.
///
/// # Arguments
///
/// * `input` - The string to bounds check
/// * `name` - The variable name to be bounds checked (for logging/errors)
/// * `min` - The minimum length of this string
/// * `max` - The maximum length of this string
pub fn string(input: &str, name: &'static str, min: usize, max: usize) -> Result<(), ApiError> {
    // bounds check length
    let input_len = input.len();
    if input_len < min || input_len > max {
        return bad!(format!(
            "{} must be between {} and  {} chars",
            name, min, max
        ));
    }

    // ensure this string is alpha numeric
    if !input
        .chars()
        .all(|chr| char::is_alphanumeric(chr) || chr == '-')
    {
        return bad!(format!(
            "{} must be only alphanumeric or '-' {}",
            name, input
        ));
    }
    Ok(())
}

/// Bounds check a lowercase string
///
/// This enforces a minimum and maximum size for a string and ensures it is all lowercase.
///
/// # Arguments
///
/// * `input` - The string to bounds check
/// * `name` - The variable name to be bounds checked (for logging/errors)
/// * `min` - The minimum length of this string
/// * `max` - The maximum length of this string
pub fn string_lower(
    input: &str,
    name: &'static str,
    min: usize,
    max: usize,
) -> Result<(), ApiError> {
    // bounds check length
    let input_len = input.len();
    if input_len < min || input_len > max {
        return bad!(format!(
            "{} must be between {} and  {} chars",
            name, min, max
        ));
    }

    // ensure this string is alpha numeric and lowercase or a -
    if !input
        .chars()
        .all(|chr| char::is_lowercase(chr) || char::is_numeric(chr) || chr == '-')
    {
        return bad!(format!(
            "{} must be only lowercase alphanumeric or '-' {}",
            name, input
        ));
    }
    Ok(())
}

/// Bounds check a file name
///
/// This enforces a minimum and maximum size for a string.
///
/// # Arguments
///
/// * `input` - The string to bounds check
/// * `name` - The variable name to be bounds checked (for logging/errors)
/// * `min` - The minimum length of this string
/// * `max` - The maximum length of this string
pub fn file_name(input: &str, name: &'static str, min: usize, max: usize) -> Result<(), ApiError> {
    // bounds check length
    let input_len = input.len();
    if input_len < min || input_len > max {
        return bad!(format!(
            "{} must be between {} and  {} chars",
            name, min, max
        ));
    }

    // ensure this string is alpha numeric
    if !input
        .chars()
        .all(|chr| char::is_alphanumeric(chr) || chr == '-' || chr == '.')
    {
        return bad!(format!(
            "{} must be only alphanumeric or '-'/'.' {}",
            name, input
        ));
    }
    Ok(())
}

/// Bounds check a JsonValue that should be cast as a string
///
/// This enforces a minimum and maximum size for a string
///
/// # Arguments
///
/// * `input` - The json value to cast as a string and bounds check
/// * `name` - The variable name to be bounds checked (for logging/errors)
/// * `min` - The minimum length of this string
/// * `max` - The maximum length of this string
pub fn string_json_value(
    input: &serde_json::Value,
    name: &'static str,
    min: usize,
    max: usize,
) -> Result<String, ApiError> {
    // cast to string string
    if !input.is_string() {
        return bad!(format!("{} must be a string - {:#?}", name, input));
    }
    let input_str = input.as_str().unwrap_or("");

    // bounds check length
    let input_len = input_str.len();
    if input_len < min || input_len > max {
        return bad!(format!(
            "{} must be between {} and  {} chars - {}",
            name, min, max, input
        ));
    }
    Ok(input_str.to_string())
}
/// Bounds check a number
///
/// This enforces a minimum and maximum value for a signed int.
///
/// # Arguments
///
/// * `input` - The signed into to bounds check
/// * `name` - The variable name to be bounds checked (for logging/errors)
/// * `min` - The minimum value for this int
/// * `max` - The maximum value for this int
pub fn number(input: i64, name: &'static str, min: i64, max: i64) -> Result<i64, ApiError> {
    // bounds check size
    if input < min || input > max {
        return bad!(format!(
            "{} must be between {} and  {} but is {}",
            name, min, max, input
        ));
    }
    Ok(input)
}

/// Bounds check a unsigned number
///
/// This enforces a minimum and maximum value for a unsigned int.
///
/// # Arguments
///
/// * `input` - The unsigned int to bounds check
/// * `name` - The variable name to be bounds checked (for logging/errors)
/// * `min` - The minimum value for this int
/// * `max` - The maximum value for this int
pub fn unsigned(input: u64, name: &'static str, min: u64, max: u64) -> Result<u64, ApiError> {
    // bounds check size
    if input < min || input > max {
        return bad!(format!(
            "{} must be between {} and  {} - {}",
            name, min, max, input
        ));
    }
    Ok(input)
}

/// Bounds checks an image cpu and converts it to millicpu
///
/// # Arguments
///
/// * `raw` - A raw cpu value
pub fn image_cpu(raw: &str) -> Result<u64, ApiError> {
    // try to cast this directly to a f64
    // This is because we assume that any f64 value is # of cores
    if let Ok(cores) = raw.parse::<f64>() {
        // if parse was successful then convert to millicpu
        return Ok((cores * 1000.0).ceil() as u64);
    }

    // f64 parse failed check if it ends in a millicpu unit
    if raw.ends_with('m') {
        // try to parse as millicpu
        let millicpu = raw[..raw.len() - 1].parse::<u64>();
        if millicpu.is_err() {
            return bad!(format!("Invalid cpu value: {}", millicpu.unwrap()));
        }
        return Ok(millicpu.unwrap());
    }
    // error if all of the cpu handlers failed
    bad!(format!("Failed to parse cpu value: {}", raw))
}

/// Bounds checks an image storage value and converts it to mebibytes
///
/// # Arguments
///
/// * `raw` - A raw storage value
pub fn image_storage(raw: &str) -> Result<u64, ApiError> {
    // try to cast this directly to a u64
    // This is because we assume that any u64 value is # of bytes
    if let Ok(bytes) = raw.parse::<u64>() {
        // if parse was successful then convert to mebibytes
        return Ok((bytes as f64 * 1.049e+6).ceil() as u64);
    }

    // u64 failed parse check lets find first occurrence of a any valid char
    let unit_regex = Regex::new(r"[KMGTPE]").unwrap();
    // find index where unit starts
    let reg = match unit_regex.find(&raw) {
        Some(reg) => reg,
        None => return bad!(format!("failed to find parse {}", raw)),
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
        _ => return bad!(format!("Failed to parse storage value: {}", raw)),
    };
    Ok(mebibytes)
}

/// Bounds check a pipeline order
///
/// This enforces that a pipeline orders stages are defined.
///
/// # Arguments
///
/// * `raw` - The raw pipeline order to bounds check
/// * `user` - The user that is creating/updating this pipeline
/// * `group` - The group this pipeline is in
/// * `shared` - Shared Thorium objects
#[instrument(name = "utils::bounder::pipeline_order", skip_all, err(Debug))]
pub async fn pipeline_order(
    raw: &Value,
    user: &User,
    group: &Group,
    shared: &Shared,
) -> Result<Vec<Vec<String>>, ApiError> {
    // make sure order is an array
    if !raw.is_array() {
        return bad!("order must be an array".to_string());
    }
    // cast raw order to Vec<Vec<String>>
    let mut cast = Vec::new();
    // iterate over stages in order
    for stage in raw.as_array().unwrap() {
        // handle stages with sub stages
        // cast to a vector if order is an array
        if stage.is_array() {
            // get sub stages
            let sub_stages = stage.as_array().unwrap();
            // make sure that we have some sub stages defined
            if sub_stages.is_empty() {
                return bad!("order cannot have an empty stage".to_owned());
            }
            // iterate over sub stages and bounds check them
            let mut inner_cast = Vec::new();
            for item in stage.as_array().unwrap() {
                // cast image name to string
                let item = string_json_value(item, "stage", 1, 255)?;
                // make sure image exists
                Image::exists_authenticated(&item, group, shared).await?;
                // get the scaler for this image
                let scaler = Image::get_scaler(group, &item, shared).await?;
                // make sure the image doesn't have any bans
                let bans = Image::get_bans(group, &item, shared).await?;
                if !bans.is_empty() {
                    return bad!(format!(
                        "Image '{item}' has one or more bans! See image details for more info."
                    ));
                }
                // make sure we can develop this image
                group.developer(user, scaler)?;
                // push into vec
                inner_cast.push(item);
            }
            cast.push(inner_cast)

        // handle stages with no sub stages
        } else {
            let mut inner_cast = Vec::new();
            // cast image name to string
            let stage = string_json_value(stage, "stage", 1, 255)?;
            // make sure image exists
            Image::exists_authenticated(&stage, group, shared).await?;
            // get the scaler for this image
            let scaler = Image::get_scaler(group, &stage, shared).await?;
            // make sure the image doesn't have any bans
            let bans = Image::get_bans(group, &stage, shared).await?;
            if !bans.is_empty() {
                return bad!(format!(
                    "Image '{stage}' has one or more bans! See image details for more info."
                ));
            }
            // make sure we can develop this image
            group.developer(user, scaler)?;
            inner_cast.push(stage);
            cast.push(inner_cast);
        }
    }

    // make sure order is not empty
    if cast.is_empty() {
        return bad!("order must not be empty".to_string());
    }
    Ok(cast)
}

/// Convert a string to a uuid
///
/// This will error on invalid uuidv4 inputs.
///
/// # Arguments
///
/// * `uuid` - A uuidv4 as a string
/// * `name` - The variable name to be bounds checked (for logging/errors)
pub fn uuid<'a>(uuid: &'a str, name: &'a str) -> Result<Uuid, ApiError> {
    // throw an error if an invalid uuid was passed
    match Uuid::parse_str(uuid) {
        Ok(valid) => Ok(valid),
        Err(_) => bad!(format!("{} must be a valid uuidv4", name)),
    }
}

/// Validate triggers
///
/// # Arguments
///
/// * `triggers` - The triggers to validate
pub fn triggers(triggers: &HashMap<String, EventTrigger>) -> Result<(), ApiError> {
    // make sure all new tag type event triggers have types
    for (name, trigger) in triggers.iter() {
        // make sure new tag triggers have tag types set
        match trigger {
            EventTrigger::NewSample => continue,
            EventTrigger::Tag { tag_types, .. } => {
                // make sure we have some tag type set
                if tag_types.is_empty() {
                    return bad!(format!("tag triggers must have tag types set: {}", name));
                }
            }
        }
    }
    Ok(())
}
