//! The scylla utils/structs for exports
use chrono::prelude::*;
use scylla::DeserializeRow;
use std::str::FromStr;
use uuid::Uuid;

use crate::models::InvalidEnum;

/// The different types of export operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum ExportOps {
    /// Results are being exported
    Results,
}

impl ExportOps {
    /// Allow export ops to be serialized as a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            ExportOps::Results => "Results",
        }
    }
}

impl std::fmt::Display for ExportOps {
    /// Allow [`ExportOps`] to be displayed
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ExportOps {
    type Err = InvalidEnum;

    /// Convert this str to an [`ExportOps`]
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "Results" => Ok(ExportOps::Results),
            _ => Err(InvalidEnum(format!("Unknown ExportOps: {raw}"))),
        }
    }
}

/// An internal struct containing a single export row in Scylla
#[derive(Debug)]
pub struct ExportRow {
    /// The ID for this export operation
    pub id: Uuid,
    /// The name of this export operation
    pub name: String,
    /// The user that owns this export operation
    pub user: String,
    /// The starting timestamp to stream from (earliest)
    pub start: Option<DateTime<Utc>>,
    /// The current timestamp/progress for our cursor
    pub current: DateTime<Utc>,
    /// The ending timestamp to stream to (latest)
    pub end: DateTime<Utc>,
}

/// An internal struct containing a single export row in Scylla
#[derive(Debug)]
pub struct ExportIdRow {
    /// The group this export operation can be seen by
    pub group: String,
    /// The ID for this export operation
    pub id: Uuid,
}

/// An internal struct containing a single export cursor row in Scylla
#[derive(Debug)]
pub struct ExportCursorRow {
    /// This cursors ID in Thorium
    pub id: Uuid,
    /// The starting timestamp to stream from (earliest)
    pub start: DateTime<Utc>,
    /// The current timestamp/progress for our cursor
    pub current: DateTime<Utc>,
    /// The ending timestamp to stream to (latest)
    pub end: DateTime<Utc>,
    /// The export error this cursor is retrying if its tied to an error
    pub error: Option<Uuid>,
}

/// An internal struct containing a single export error row in Scylla
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct ExportErrorRow {
    /// The id for this error
    pub id: Uuid,
    /// The start of the chunk of data that we failed to export
    pub start: DateTime<Utc>,
    /// The end of the chunk of data that we failed to export
    pub end: DateTime<Utc>,
    /// The error number/code that occurred
    pub code: Option<i32>,
    /// A message explaining the error that occurred
    pub msg: String,
}
