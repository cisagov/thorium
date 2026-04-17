/// The different types of s3 objects we should track
use std::str::FromStr;

use crate::models::InvalidEnum;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
pub enum S3Objects {
    /// A sample or file
    File,
    /// A zipped repo
    Repo,
}

impl S3Objects {
    /// Convert our s3 object into a str
    pub fn as_str(&self) -> &'static str {
        match self {
            S3Objects::File => "File",
            S3Objects::Repo => "Repo",
        }
    }
}

// To use the `{}` marker, the trait `fmt::Display` must be implemented
// manually for the type.
impl std::fmt::Display for S3Objects {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            S3Objects::File => write!(f, "File"),
            S3Objects::Repo => write!(f, "Repo"),
        }
    }
}

impl FromStr for S3Objects {
    type Err = InvalidEnum;

    /// Cast a str to an `S3Objects`
    ///
    /// # Arguments
    ///
    /// * `val` - The str to cast
    fn from_str(val: &str) -> Result<Self, Self::Err> {
        match val {
            "File" => Ok(S3Objects::File),
            "Repo" => Ok(S3Objects::Repo),
            _ => Err(InvalidEnum(format!("Unknown enum variant: {val}"))),
        }
    }
}
