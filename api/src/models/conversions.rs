use serde::Deserialize;
use serde::Serializer;
use serde::de::{self, Deserializer, Visitor};
use std::num::ParseIntError;

/// Converts a string into a `ConversionError`
macro_rules! err {
    ($msg:expr) => {
        Err(ConversionError::new($msg))
    };
}

/// An error that occured while converting values
#[derive(Debug, Deserialize, Serialize)]
pub struct ConversionError {
    /// The message explaining the error that occured
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
    /// * `msg` - The error that occured during this conversion
    #[must_use]
    pub fn new(msg: String) -> Self {
        ConversionError { msg }
    }
}

/// Serialize a cpu value
pub fn serialize_cpu<S>(initial: &u64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // if this is an even core count then convert it to cores otherwise convert it to a millicpu string
    if initial.is_multiple_of(1000) {
        // convert our millicpu count to cores and serialize that
        s.serialize_u64(initial / 1000)
    } else {
        // just serialize this as millicpu
        s.serialize_str(&format!("{initial}m"))
    }
}

/// Serialize an optional cpu value
pub fn serialize_cpu_opt<S>(initial_opt: &Option<u64>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // get our option if it exists
    if let Some(initial) = initial_opt {
        serialize_cpu(initial, s)
    } else {
        s.serialize_none()
    }
}

/// The visitor for deserializing cpu values
struct MillicpuVisitor;

impl<'de> Visitor<'de> for MillicpuVisitor {
    /// The final type to store cpu values in millicpu as
    type Value = u64;

    /// Return an error when getting incorrect cpu values
    ///
    /// # Arguments
    ///
    /// * `formatter` - The formatter to write errors too
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("A cpu value in millicpu (\"2500m\") or cores (4, 2.5)")
    }

    /// Deserialize from ``i8`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i8`` value to convert from
    fn visit_i8<E>(self, value: i8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // multiple our number of cores by 1000
            Ok(cast) => Ok(cast * 1000),
            // throw an error on negative core amounts
            Err(_) => Err(E::custom(format!("Core count cannot be negative: {value}"))),
        }
    }

    /// Deserialize from ``i16`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i16`` value to convert from
    fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // multiple our number of cores by 1000
            Ok(cast) => Ok(cast * 1000),
            // throw an error on negative core amounts
            Err(_) => Err(E::custom(format!("Core count cannot be negative: {value}"))),
        }
    }

    /// Deserialize from ``i32`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i32`` value to convert from
    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // multiple our number of cores by 1000
            Ok(cast) => Ok(cast * 1000),
            // throw an error on negative core amounts
            Err(_) => Err(E::custom(format!("Core count cannot be negative: {value}"))),
        }
    }

    /// Deserialize from ``i64`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i64`` value to convert from
    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // multiple our number of cores by 1000
            Ok(cast) => Ok(cast * 1000),
            // throw an error on negative core amounts
            Err(_) => Err(E::custom(format!("Core count cannot be negative: {value}"))),
        }
    }

    /// Deserialize from ``u8`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u8`` value to convert from
    fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // multiple our value by 1000 to convert it to millicpu
        Ok(u64::from(value) * 1000)
    }

    /// Deserialize from ``u16`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u16`` value to convert from
    fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // multiple our value by 1000 to convert it to millicpu
        Ok(u64::from(value) * 1000)
    }

    /// Deserialize from ``u32`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u32`` value to convert from
    fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // multiple our value by 1000 to convert it to millicpu
        Ok(u64::from(value) * 1000)
    }

    /// Deserialize from ``u64`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u64`` value to convert from
    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // multiple our value by 1000 to convert it to millicpu
        Ok(value * 1000)
    }

    /// Deserialize from ``f32`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``f32`` value to convert from
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // make sure our cores value is positive
        if value < 0.0 {
            return Err(E::custom(format!("Core count cannot be negative: {value}")));
        }
        // convert our floating point core count to fixed point
        Ok((value * 1000.0_f32.ceil()) as u64)
    }

    /// Deserialize from ``f64`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``f64`` value to convert from
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // make sure our cores value is positive
        if value < 0.0 {
            return Err(E::custom(format!("Core count cannot be negative: {value}")));
        }
        // convert our floating point core count to fixed point
        Ok((value * 1000.0_f64.ceil()) as u64)
    }

    /// Deserialize from ``str`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``str`` value to convert from
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // check if our string is a millicpu value
        if value.ends_with('m') {
            // trim the m from our millicpu value
            match value.split_once('m') {
                Some((trimmed, _)) => {
                    // parse our millicpu value
                    trimmed
                        .parse::<u64>()
                        .map_err(|_| E::custom(format!("Invalid millicpu value: {value}")))
                }
                None => Err(E::custom(format!(
                    "Millicpu value cannot just be an 'm': {value}"
                ))),
            }
        } else if value.contains('.') {
            // try to cast our string to a f64
            let f64_cast = value
                .parse::<f64>()
                .map_err(|_| E::custom(format!("Invalid cpu value: {value}")))?;
            // convert our floating point core count to fixed point cpu value
            self.visit_f64(f64_cast)
        } else {
            // assume this is a fixed point core count
            let u64_cast = value
                .parse::<u64>()
                .map_err(|_| E::custom(format!("Invalid cpu value: {value}")))?;
            // convert our fixed point core count to millicpu
            self.visit_u64(u64_cast)
        }
    }

    /// Deserialize from ``String`` to a millicpu value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``String`` value to convert from
    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // just use our &str deserializer
        Self::visit_str(self, &value)
    }
}

/// Deserialize millicpu values from many initial types
///
/// This will convert any core count to a fixed point millicpu representation
pub fn deserialize_cpu<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    // deserialize our cpu value
    deserializer.deserialize_any(MillicpuVisitor)
}

/// Helps serde deserialize optional cpu values
#[derive(Debug, Deserialize)]
struct CpuOptHelper(#[serde(deserialize_with = "deserialize_cpu")] u64);

pub fn deserialize_cpu_opt<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    // deserialize our cpu value
    Option::<CpuOptHelper>::deserialize(deserializer).map(|cpu_opt| cpu_opt.map(|helper| helper.0))
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

/// Serialize a storage value
pub fn serialize_storage<S>(initial: &u64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // if this is an even multiple of 1024 then return this in Gi
    if initial.is_multiple_of(1024) {
        // calculate our total number of gibibytes
        let gibibytes = initial / 1024;
        // write this with the units
        s.serialize_str(&format!("{gibibytes}Gi"))
    } else {
        // this is not cleanly represented as a gibibyte so serialize it as Mebibytes
        s.serialize_str(&format!("{initial}Mi"))
    }
}

/// Serialize an optional storage value
pub fn serialize_storage_opt<S>(initial_opt: &Option<u64>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // get our option if it exists
    if let Some(initial) = initial_opt {
        serialize_storage(initial, s)
    } else {
        s.serialize_none()
    }
}

/// The visitor for deserializing storage values
struct MebibyteStorageVisitor;

impl<'de> Visitor<'de> for MebibyteStorageVisitor {
    /// The final type to store memeory/storage values in Mebibytes
    type Value = u64;

    /// Return an error when getting incorrect memory/storage values
    ///
    /// # Arguments
    ///
    /// * `formatter` - The formatter to write errors too
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str(
            "A storage/memory value with units (\"512Mi\", \"2Gi\") or mebibytes (2048, 4096)",
        )
    }

    /// Deserialize from ``i8`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i8`` value to convert from
    fn visit_i8<E>(self, value: i8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // Assume they gave us this size in Mebibytes
            Ok(cast) => Ok(cast),
            // throw an error on negative Memory/storage amounts
            Err(_) => Err(E::custom(format!(
                "Memory/storage amount cannot be negative: {value}"
            ))),
        }
    }

    /// Deserialize from ``i16`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i16`` value to convert from
    fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // Assume they gave us this size in Mebibytes
            Ok(cast) => Ok(cast),
            // throw an error on negative Memory/storage amounts
            Err(_) => Err(E::custom(format!(
                "Memory/storage amount cannot be negative: {value}"
            ))),
        }
    }

    /// Deserialize from ``i32`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i32`` value to convert from
    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // Assume they gave us this size in Mebibytes
            Ok(cast) => Ok(cast),
            // throw an error on negative Memory/storage amounts
            Err(_) => Err(E::custom(format!(
                "Memory/storage amount cannot be negative: {value}"
            ))),
        }
    }

    /// Deserialize from ``i64`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``i64`` value to convert from
    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // try to cast our integer to a u64
        match u64::try_from(value) {
            // Assume they gave us this size in Mebibytes
            Ok(cast) => Ok(cast),
            // throw an error on negative Memory/storage amounts
            Err(_) => Err(E::custom(format!(
                "Memory/storage amount cannot be negative: {value}"
            ))),
        }
    }

    /// Deserialize from ``u8`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u8`` value to convert from
    fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // Assume they gave us this size in Mebibytes
        Ok(u64::from(value))
    }

    /// Deserialize from ``u16`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u16`` value to convert from
    fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // Assume they gave us this size in Mebibytes
        Ok(u64::from(value))
    }

    /// Deserialize from ``u32`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u32`` value to convert from
    fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // Assume they gave us this size in Mebibytes
        Ok(u64::from(value))
    }

    /// Deserialize from ``u64`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``u64`` value to convert from
    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // Assume they gave us this size in Mebibytes
        Ok(value)
    }

    /// Deserialize from ``str`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``str`` value to convert from
    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // Convert this raw value to mebibytes
        storage(value).map_err(|error| E::custom(error.msg))
    }

    /// Deserialize from ``String`` to a memory/storage value
    ///
    /// # Arguments
    ///
    /// * `value` - The ``String`` value to convert from
    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        // just use our &str deserializer
        Self::visit_str(self, &value)
    }
}

/// Helper that parses the three accepted forms:
///   * a string ending with “m”  → strip the suffix and parse the number
///   * a plain string of a number → treat it as whole CPUs and multiply by 1000
///   * an integer               → treat it as whole CPUs and multiply by 1000
///   * a float                  → treat it as whole CPUs and multiply by 1000
pub fn deserialize_storage<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    // deserialize our cpu value
    deserializer.deserialize_any(MebibyteStorageVisitor)
}

/// Helps serde deserialize optional memory/storage values
#[derive(Debug, Deserialize)]
struct StorageOptHelper(#[serde(deserialize_with = "deserialize_storage")] u64);

pub fn deserialize_storage_opt<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    // deserialize our Storage value
    Option::<StorageOptHelper>::deserialize(deserializer)
        .map(|storage_opt| storage_opt.map(|helper| helper.0))
}

/// Bounds checks an image storage value and converts it to mebibytes
///
/// This assumes that any int without a unit is mebibytes.
///
///
/// # Arguments
///
/// * `raw` - A raw storage value
pub fn storage<T: AsRef<str>>(raw: T) -> Result<u64, ConversionError> {
    let raw = raw.as_ref();
    // try to cast this directly to a u64
    // This is because we assume that any u64 value is # of mebibytes
    if let Ok(bytes) = raw.parse::<u64>() {
        return Ok(bytes);
    }
    // find the first character at the end of our serialized raw
    let index = match raw.chars().rev().position(|char| char.is_ascii_digit()) {
        Some(index) => raw.len() - index,
        None => return err!(format!("failed to find units in {}", raw)),
    };
    // split raw based on where unit was found
    let (amt, unit) = raw.split_at(index);
    // cast amt to u64
    let amt = amt.parse::<u64>().map_err(|_| {
        ConversionError::new(format!(
            "Failed to parse integer and failed to detect unit: {raw}"
        ))
    })?;
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

/// Bounds checks an image storage value and converts it to mebibytes
///
/// This assumes that any int without a unit is bytes.
///
/// # Arguments
///
/// * `raw` - A raw storage value
pub fn storage_bytes_as_base<T: AsRef<str>>(raw: T) -> Result<u64, ConversionError> {
    let raw = raw.as_ref();
    // try to cast this directly to a u64
    // This is because we assume that any u64 value is # of bytes
    if let Ok(bytes) = raw.parse::<u64>() {
        // if parse was successful then convert to mebibytes
        return Ok(bytes / 1_048_576);
    }
    // find the first character at the end of our serialized raw
    let index = match raw.chars().rev().position(|char| char.is_ascii_digit()) {
        Some(index) => raw.len() - index,
        None => return err!(format!("failed to find units in {}", raw)),
    };
    // split raw based on where unit was found
    let (amt, unit) = raw.split_at(index);
    // cast amt to u64
    let amt = amt.parse::<u64>().map_err(|_| {
        ConversionError::new(format!(
            "Failed to parse integer and failed to detect unit: {raw}"
        ))
    })?;
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
