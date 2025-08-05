//! The result specific scylla utils

use chrono::prelude::*;
use scylla::DeserializeRow;
use std::collections::HashMap;
use uuid::Uuid;

use crate::models::backends::OutputSupport;
use crate::models::{ImageVersion, OutputDisplayType};

/// A request to store the output or result of a tool in scylla
#[derive(Debug)]
pub struct OutputForm<O: OutputSupport> {
    /// The id for this result
    pub id: Uuid,
    /// The groups to share this result with
    pub groups: Vec<String>,
    /// The tool (image) this result comes from
    pub tool: String,
    /// The version of the tool (image) this result comes from
    pub tool_version: Option<ImageVersion>,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// The result to save
    pub result: String,
    /// The display type to use when rendering this result
    pub display_type: OutputDisplayType,
    /// Any files tied to this result
    pub files: Vec<String>,
    /// Any extra info thats needed in this result form
    pub extra: O::ExtraKey,
}

/// A request to store the output or result of a tool in scylla
#[derive(Debug)]
pub struct OutputFormBuilder<O: OutputSupport> {
    /// The id for this result
    pub id: Uuid,
    /// The groups to share this result with
    pub groups: Vec<String>,
    /// The tool (image) this result comes from
    pub tool: Option<String>,
    /// The version of the tool (image) this result comes from
    pub tool_version: Option<ImageVersion>,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// The result to save
    pub result: Option<String>,
    /// The display type to use when rendering this result
    pub display_type: Option<OutputDisplayType>,
    /// Any files tied to this result
    pub files: Vec<String>,
    /// Any extra info thats needed in this result form
    pub extra: Option<O::ExtraKey>,
}

impl<O: OutputSupport> Default for OutputFormBuilder<O> {
    /// Create a default output form builder
    fn default() -> Self {
        OutputFormBuilder {
            id: Uuid::new_v4(),
            groups: Vec::with_capacity(1),
            tool: None,
            tool_version: None,
            cmd: None,
            result: None,
            display_type: None,
            files: Vec::default(),
            extra: None,
        }
    }
}

/// A row from scylla containing a single id + group for a result from a tool
#[derive(Serialize, Deserialize, Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct OutputIdRow {
    /// The id for the result this ID row is tied to
    pub id: Uuid,
    /// The tool or pipeline this result comes from
    pub tool: String,
    /// The display type for this result
    pub display_type: OutputDisplayType,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// The group this auth row gives access to
    pub group: String,
    /// When this result was uploaded
    pub uploaded: DateTime<Utc>,
}

/// A row from scylla containing a single id + group for a result from a tool
#[derive(Serialize, Deserialize, Debug)]
pub struct OutputId {
    /// The id for the result this ID row is tied to
    pub id: Uuid,
    /// The tool or pipeline this result comes from
    pub tool: String,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// The group this auth row gives access to
    pub groups: Vec<String>,
    /// When this result was uploaded
    pub uploaded: DateTime<Utc>,
}

/// A row from scylla containing a single result for a tool
#[derive(Serialize, Deserialize, Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct OutputRow {
    /// The unique id for this result
    pub id: Uuid,
    /// The tool or pipeline this result comes from
    pub tool: String,
    /// The version of the tool or image this result came from
    pub tool_version: Option<ImageVersion>,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// When this result was uploaded
    pub uploaded: DateTime<Utc>,
    /// The result
    pub result: String,
    /// An optional file tied to this result
    pub files: Option<Vec<String>>,
    /// The display type of this tool output
    pub display_type: OutputDisplayType,
    /// The children that were found when generating this result
    pub children: Option<HashMap<String, Uuid>>,
}
