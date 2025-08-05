use regex::Regex;
use std::num::ParseIntError;

/// Converts a string into a `ConversionError`
macro_rules! err {
    ($msg:expr) => {
        Err(ConversionError::new($msg))
    };
}

/// An error that occurred while converting values
#[derive(Debug, Deserialize, Serialize)]
pub struct ConversionError {
    /// The message explaining the error that occurred
    pub msg: String,
}

impl From<std::num::ParseIntError> for ConversionError {
    fn from(error: std::num::ParseIntError) -> Self {
        ConversionError::new(format!("Failed to cast to int: {error}"))
    }
}

impl ConversionError {
    /// Create a new [`ConversionError`]
    ///
    /// # Arguments
    ///
    /// * `msg` - The error that occurred during this conversion
    #[must_use]
    pub fn new(msg: String) -> Self {
        ConversionError { msg }
    }
}

/// Bounds checks an image cpu and converts it to millicpu
///
/// # Arguments
///
/// * `raw` - A raw cpu value
pub fn cpu<T: AsRef<str>>(raw: T) -> Result<u64, ParseIntError> {
    let raw = raw.as_ref();
    // try to cast this directly to a f64
    // This is because we assume that any f64 value is # of cores
    if let Ok(cores) = raw.parse::<f64>() {
        // if parse was successful then convert to millicpu
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        return Ok((cores * 1000.0).ceil() as u64);
    }
    // f64 parse failed check if it ends in a millicpu unit
    match raw.strip_suffix('m') {
        // try to parse as millicpu
        Some(stripped) => stripped.parse::<u64>(),
        None => raw.parse::<u64>(),
    }
}

/// Bounds checks an image storage value and converts it to mebibytes
///
/// # Arguments
///
/// * `raw` - A raw storage value
pub fn storage<T: AsRef<str>>(raw: T) -> Result<u64, ConversionError> {
    let raw = raw.as_ref();
    // try to cast this directly to a u64
    // This is because we assume that any u64 value is # of bytes
    if let Ok(bytes) = raw.parse::<u64>() {
        // if parse was successful then convert to mebibytes
        return Ok(bytes * 1_048_576);
    }

    // u64 failed parse check lets find first occurrence of a any valid char
    let unit_regex = match Regex::new(r"[KMGTPE]") {
        Ok(regex) => regex,
        Err(error) => return err!(format!("Failed to compile regex: {error}")),
    };
    // find index where unit starts
    let Some(reg) = unit_regex.find(raw) else {
        return err!(format!("failed to find units in {}", raw));
    };
    // split raw based on where unit was found
    let (amt, unit) = raw.split_at(reg.start());
    // cast amt to u64
    let amt = amt.parse::<u64>()?;
    // convert to mebibytes using fixed point math
    let mebibytes = match unit {
        "K" => amt / 1049,
        "M" => amt * 1_000_000 / 1_048_576,
        "G" => amt * 954,
        "T" => amt * 953_674,
        "P" => amt * 1_000_000_000_000_000 / 1_048_576,
        "E" => amt * 1_000_000_000_000_000_000 / 104_875,
        "Ki" => amt / 1024,
        "Mi" => amt,
        "Gi" => amt * 1024,
        "Ti" => amt * 1_099_511_627_776 / 1_048_576,
        "Pi" => amt * 1_125_899_906_842_624 / 104_875,
        "Ei" => amt * 1_152_921_504_606_846_976 / 104_875,
        _ => return err!(format!("Failed to parse storage value: {}", raw)),
    };
    Ok(mebibytes)
}
