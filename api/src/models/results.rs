//! Results or output of tools for samples in Thorium
//!
//! This currently requires a sample to be saved into Thorium.

use chrono::prelude::*;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use uuid::Uuid;

use super::backends::OutputSupport;
use super::{Buffer, ImageVersion, InvalidEnum};
use crate::{
    matches_adds, matches_clear, matches_removes, matches_update, matches_update_opt, same,
};

#[cfg(feature = "client")]
use crate::{multipart_file, multipart_list, multipart_text};

#[cfg(feature = "client")]
use crate::client::Error;

/// The kind of results we are saving to the db
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum OutputKind {
    /// We are saving file results
    Files,
    /// We are saving repo results
    Repos,
}

impl OutputKind {
    /// Cast our result kind to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            OutputKind::Files => "Files",
            OutputKind::Repos => "Repos",
        }
    }
}

impl FromStr for OutputKind {
    type Err = InvalidEnum;

    /// Convert this str to an [`OutputKind`]
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "Files" => Ok(OutputKind::Files),
            "Repos" => Ok(OutputKind::Repos),
            _ => Err(InvalidEnum(format!("Unknown OutputKind: {raw}"))),
        }
    }
}

/// A single result for a single run of a tool with a specific command
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OutputRequest<O: OutputSupport> {
    /// The primary key, not including groups, to access this result with
    pub key: O::Key,
    /// The groups that can see this result
    pub groups: Vec<String>,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// The tool this result is from
    pub tool: String,
    /// The version of the tool this result is from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<ImageVersion>,
    /// The result
    pub result: String,
    /// Any files tied to this result
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<OnDiskFile>,
    /// Any buffers to upload as result files
    pub buffers: Vec<Buffer>,
    /// The display type of this result
    pub display_type: OutputDisplayType,
}

impl<O: OutputSupport> OutputRequest<O> {
    /// Creates a new output request
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool this result is from
    /// * `result` - The result to save
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String);
    /// ```
    pub fn new<T: Into<String>, R: Into<String>>(
        key: O::Key,
        tool: T,
        result: R,
        display_type: OutputDisplayType,
    ) -> Self {
        OutputRequest {
            key,
            groups: Vec::default(),
            cmd: None,
            result: result.into(),
            tool: tool.into(),
            tool_version: None,
            files: Vec::default(),
            buffers: Vec::default(),
            display_type,
        }
    }

    /// Adds a group to list of groups that can see this output
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .group("CornGroup");
    /// ```
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        // convert our group to a string and add it
        self.groups.push(group.into());
        self
    }

    /// Adds groups to the list of groups that can see this output
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .groups(vec!("CornGroup", "Harvesters"));
    /// ```
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // convert our groups to strings and add them
        self.groups.extend(groups.into_iter().map(Into::into));
        self
    }

    /// Sets the command that was used to generate this result
    ///
    /// # Arguments
    ///
    /// * `cmd` - The command to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .cmd("./Harvesters --field 1 --crop corn");
    /// ```
    #[must_use]
    pub fn cmd<T: Into<String>>(mut self, cmd: T) -> Self {
        // convert our command to a string and set it
        self.cmd = Some(cmd.into());
        self
    }

    /// Sets the version of the tool that was used to generate this result
    ///
    /// # Arguments
    ///
    /// * `tool_version` - The tool version to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ImageVersion, OutputRequest, OutputDisplayType, Sample};
    /// use semver::Version;
    ///
    /// let image_version = ImageVersion::SemVer(Version::parse("1.0.0").unwrap());
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .tool_version(image_version);
    /// ```
    #[must_use]
    pub fn tool_version(mut self, tool_version: ImageVersion) -> Self {
        self.tool_version = Some(tool_version);
        self
    }

    /// Adds a path to a file to upload with this result
    ///
    /// # Arguments
    ///
    /// * `path` - The path to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, OnDiskFile, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .file(OnDiskFile::new("/corn/bushel1"));
    /// ```
    #[must_use]
    pub fn file(mut self, file: OnDiskFile) -> Self {
        // convert our path and add it
        self.files.push(file);
        self
    }

    /// Adds paths to the list of paths to upload with this result
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to add
    /// * `trim_prefix` - The prefix to trim from the start of our file paths
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, OnDiskFile, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .files(vec!(OnDiskFile::new("/corn/bushel1"), OnDiskFile::new("/corn/bushel2")));
    /// ```
    #[must_use]
    pub fn files(mut self, files: Vec<OnDiskFile>) -> Self {
        // convert our paths and add them
        self.files.extend(files);
        self
    }

    /// Adds a buffer to upload with this result
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Buffer, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .buffer(Buffer::new("buffer").name("buffer.txt"));
    /// ```
    #[must_use]
    pub fn buffer(mut self, buffer: Buffer) -> Self {
        // add our buffer to our list of buffers to upload
        self.buffers.push(buffer);
        self
    }

    /// Adds multiple buffers to upload with this result
    ///
    /// # Arguments
    ///
    /// * `buffers` - The buffers to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Buffer, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .buffers(vec!(Buffer::new("buffer0"), Buffer::new("buffer1")));
    /// ```
    #[must_use]
    pub fn buffers(mut self, mut buffers: Vec<Buffer>) -> Self {
        // append our new buffers on to our list of buffers to upload
        self.buffers.append(&mut buffers);
        self
    }

    /// Sets the display type to use when rendering these results
    ///
    /// # Arguments
    ///
    /// * `display_type` - The display type to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputRequest, OutputDisplayType, Sample};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let req = OutputRequest::<Sample>::new(sha256, "CornHarvester", "Lots of Corn", OutputDisplayType::String)
    ///     .display_type(OutputDisplayType::Json);
    /// ```
    #[must_use]
    pub fn display_type(mut self, display_type: OutputDisplayType) -> Self {
        // convert our command to a string and set it
        self.display_type = display_type;
        self
    }

    /// Create a multipart form from this sample request
    #[cfg(feature = "client")]
    pub async fn to_form(mut self) -> Result<reqwest::multipart::Form, Error> {
        // build the form we are going to send
        // disable percent encoding, as the API natively supports UTF-8
        let form = reqwest::multipart::Form::new()
            .percent_encode_noop()
            // the tool that created this result
            .text("tool", self.tool)
            // the string to save for this result
            .text("result", self.result)
            // the display type to use when rendering these results
            .text("display_type", self.display_type);
        // add the groups to share this result with
        let form = multipart_list!(form, "groups", self.groups);
        // add the version of the tool that created this result if it was set and serialize it
        let form = match self.tool_version.take() {
            Some(tool_version) => form.text(
                "tool_version",
                serde_json::to_string(&tool_version).unwrap(),
            ),
            None => form,
        };
        // add the command that created this result if it was set
        let mut form = multipart_text!(form, "cmd", self.cmd);
        // add any files that were added by path
        for on_disk in self.files {
            // a path was set so read in that file and add it to the form
            form = multipart_file!(form, "files", on_disk.path, on_disk.trim_prefix);
        }
        // add any buffers that were added directly
        for buff in self.buffers {
            form = form.part("files", buff.to_part()?);
        }
        Ok(form)
    }
    ///// build the url to post results too
    /////
    ///// # Arguments
    /////
    ///// * `host` - The host to use in this url
    //pub fn build_post_url(&self, host: &str) -> String {
    //    O::build_post_url(&self.key, host)
    //}
}

/// Optional parameters for getting results
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ResultGetParams {
    /// Also show hidden results
    #[serde(default)]
    pub hidden: bool,
    /// Any tools to limit our results to
    #[serde(default)]
    pub tools: Vec<String>,
    /// Any groups to limit our results to
    #[serde(default)]
    pub groups: Vec<String>,
}

impl ResultGetParams {
    /// Return hidden results
    #[must_use]
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Adds a tool to list of tools to get results for
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultGetParams;
    ///
    /// ResultGetParams::default().tool("CornDetector");
    /// ```
    #[must_use]
    pub fn tool<T: Into<String>>(mut self, tool: T) -> Self {
        // convert our tool to a string and add it
        self.tools.push(tool.into());
        self
    }

    /// Adds tools to the list of tools to get results for
    ///
    /// # Arguments
    ///
    /// * `tools` - The tools to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultGetParams;
    ///
    /// ResultGetParams::default()
    ///     .tools(vec!("CornDetector", "AppleDetector"));
    /// ```
    #[must_use]
    pub fn tools<'a, T, I>(mut self, tools: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        // convert our tools to strings and add them
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Adds a group to the list of groups to get results from
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultGetParams;
    ///
    /// ResultGetParams::default().group("Detectors");
    /// ```
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        // convert our group to a string and add it
        self.groups.push(group.into());
        self
    }

    /// Adds groups to the list of groups to get results from
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultGetParams;
    ///
    /// ResultGetParams::default()
    ///     .groups(vec!("Detectors", "Harvesters"));
    /// ```
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // convert our groups to strings and add them
        self.groups.extend(groups.into_iter().map(Into::into));
        self
    }
}

/// An on-disk file to upload to Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OnDiskFile {
    /// The path to this file to upload
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub path: PathBuf,
    /// The prefix to optionally trim from our file
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub trim_prefix: Option<PathBuf>,
}

impl OnDiskFile {
    /// Create a new on disk file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to this the on disk file to upload
    pub fn new<T: Into<PathBuf>>(path: T) -> Self {
        OnDiskFile {
            path: path.into(),
            trim_prefix: None,
        }
    }

    /// The prefix to strip from our path when setting metadata in Thorium
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix to trim
    #[must_use]
    pub fn trim_prefix<T: Into<PathBuf>>(mut self, prefix: T) -> Self {
        // set our prefix to be trimmed
        self.trim_prefix = Some(prefix.into());
        self
    }
}

/// A response from creating a result
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OutputResponse {
    /// The id of the created response
    pub id: Uuid,
}

/// A single result for a single run of a tool with a specific command
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Output {
    /// The id for this result
    pub id: Uuid,
    /// The groups that can see this result
    pub groups: Vec<String>,
    /// The version of the tool that generated this result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<ImageVersion>,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// When this result was uploaded
    pub uploaded: DateTime<Utc>,
    /// Set to true if a deserialization failure occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deserialization_error: Option<String>,
    /// The result
    pub result: Value,
    /// Any files tied to this result
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    /// The display type of this tool output
    pub display_type: OutputDisplayType,
    /// The children that were found when generating this result
    pub children: HashMap<String, Uuid>,
}

#[cfg(any(feature = "api", feature = "client"))]
impl<O: OutputSupport> PartialEq<OutputRequest<O>> for Output {
    /// Check if a [`OutputRequest`] and a [`Output`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The `OutputRequest` to compare against
    fn eq(&self, request: &OutputRequest<O>) -> bool {
        // make sure all fields are the same
        same!(self.cmd, request.cmd);
        same!(self.result, request.result);
        // make sure the groups for this request were added
        matches_adds!(self.groups, request.groups);
        // make sure the on disk file info matches
        for on_disk in &request.files {
            // build the path to check for
            let path = match &on_disk.trim_prefix {
                Some(trim) => match on_disk.path.strip_prefix(trim) {
                    Ok(stripped) => stripped,
                    Err(_) => return false,
                },
                None => on_disk.path.as_path(),
            };
            // make sure this path is in our comment
            if !self.files.contains(&path.to_string_lossy().to_string()) {
                return false;
            }
        }
        // make sure our display type matches
        same!(self.display_type, request.display_type);
        true
    }
}

/// A map of results for tools
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OutputMap {
    /// a map of results by tool
    pub results: HashMap<String, Vec<Output>>,
}

impl<O: OutputSupport> PartialEq<OutputRequest<O>> for OutputMap {
    /// Check if a [`OutputRequest`] and a [`Output`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The `OutputRequest` to compare against
    fn eq(&self, request: &OutputRequest<O>) -> bool {
        // make sure this tool has a result
        if let Some(results) = self.results.get(&request.tool) {
            // make sure at least one of the results match
            results.iter().any(|result| *result == *request)
        } else {
            // this tool didn't have a result so return false
            false
        }
    }
}

/// A single result for a single run of a tool with a specific command
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OutputChunk {
    /// The id for this result
    pub id: Uuid,
    /// The command used to generate this result
    pub cmd: Option<String>,
    /// The version of the tool used to generate this result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<ImageVersion>,
    /// When this result was uploaded
    pub uploaded: DateTime<Utc>,
    /// Set to true if a deserialization failure occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deserialization_error: Option<String>,
    /// The result
    pub result: Value,
    /// Any files tied to this result
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    /// The children that were found when generating this result
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub children: HashMap<String, Uuid>,
}

/// The type of display class to use in the UI for this output
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy, clap::ValueEnum)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum OutputDisplayType {
    /// Render this output as json
    #[serde(rename = "JSON", alias = "Json")]
    Json,
    /// Render this output as a string
    String,
    /// Render this output as a table
    Table,
    /// Render this output as one or more images
    Image,
    /// Use a custom render class in the UI, class will be based on tool name
    Custom,
    /// Result to render is disassembly
    Disassembly,
    /// Result to render is HTML and may need sanitization before rendering
    #[serde(rename = "HTML", alias = "Html")]
    Html,
    /// Result to render is Markdown formatted
    Markdown,
    /// Do not return this output unless its requested
    Hidden,
    /// Result to render is XML formatted
    #[serde(rename = "XML", alias = "Xml")]
    Xml,
}

impl OutputDisplayType {
    /// Cast our output display type to a str
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputDisplayType::Json => "Json",
            OutputDisplayType::String => "String",
            OutputDisplayType::Table => "Table",
            OutputDisplayType::Image => "Image",
            OutputDisplayType::Custom => "Custom",
            OutputDisplayType::Disassembly => "Disassembly",
            OutputDisplayType::Html => "Html",
            OutputDisplayType::Markdown => "Markdown",
            OutputDisplayType::Hidden => "Hidden",
            OutputDisplayType::Xml => "Xml",
        }
    }
}

impl Default for OutputDisplayType {
    /// Create a default `OutputDisplayType` of Json
    fn default() -> Self {
        OutputDisplayType::Json
    }
}

impl TryFrom<&String> for OutputDisplayType {
    type Error = InvalidEnum;
    // try to convert our string to an [`OutputDisplayType`]
    fn try_from(raw: &String) -> Result<Self, Self::Error> {
        Self::from_str(raw)
    }
}

impl From<OutputDisplayType> for Cow<'static, str> {
    /// convert this display type into a copy on write str
    fn from(display_type: OutputDisplayType) -> Self {
        match display_type {
            OutputDisplayType::Json => Cow::Borrowed("Json"),
            OutputDisplayType::String => Cow::Borrowed("String"),
            OutputDisplayType::Table => Cow::Borrowed("Table"),
            OutputDisplayType::Image => Cow::Borrowed("Image"),
            OutputDisplayType::Custom => Cow::Borrowed("Custom"),
            OutputDisplayType::Disassembly => Cow::Borrowed("Disassembly"),
            OutputDisplayType::Html => Cow::Borrowed("Html"),
            OutputDisplayType::Markdown => Cow::Borrowed("Markdown"),
            OutputDisplayType::Hidden => Cow::Borrowed("Hidden"),
            OutputDisplayType::Xml => Cow::Borrowed("Xml"),
        }
    }
}

impl FromStr for OutputDisplayType {
    type Err = InvalidEnum;
    /// convert this str to an [`OutputDisplayType`]
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "Json" => Ok(OutputDisplayType::Json),
            "String" => Ok(OutputDisplayType::String),
            "Table" => Ok(OutputDisplayType::Table),
            "Image" => Ok(OutputDisplayType::Image),
            "Custom" => Ok(OutputDisplayType::Custom),
            "Disassembly" => Ok(OutputDisplayType::Disassembly),
            "Html" => Ok(OutputDisplayType::Html),
            "Markdown" => Ok(OutputDisplayType::Markdown),
            "Hidden" => Ok(OutputDisplayType::Hidden),
            "Xml" => Ok(OutputDisplayType::Xml),
            _ => Err(InvalidEnum(format!("Unknown OutputDisplayType: {raw}"))),
        }
    }
}

/// The different type of handlers for collecting results
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum OutputHandler {
    /// Collect output from a location on disk
    Files,
}

impl Default for OutputHandler {
    /// Create a default [`OutputHandler`] of stdout
    fn default() -> Self {
        OutputHandler::Files
    }
}

/// helps serde default the results path
fn default_results_path() -> String {
    "/tmp/thorium/results".into()
}

/// helps serde default the result files path
fn default_result_files_path() -> String {
    "/tmp/thorium/result-files".into()
}
/// helps serde default the tags path
fn default_tags_path() -> String {
    "/tmp/thorium/tags".into()
}

/// The settings for collecting results from a specific location on disk
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct FilesHandler {
    /// The location to look for small renderable results at on disk
    #[serde(default = "default_results_path")]
    pub results: String,
    /// The location to look for files that should be uploaded as result files
    #[serde(default = "default_result_files_path")]
    pub result_files: String,
    /// The location to load tags to set from
    #[serde(default = "default_tags_path")]
    pub tags: String,
    /// Any file names to restrict our handler to
    #[serde(default)]
    pub names: Vec<String>,
}

impl Default for FilesHandler {
    /// Create a default [`FilesHandler`]
    fn default() -> Self {
        FilesHandler {
            results: "/tmp/thorium/results".into(),
            result_files: "/tmp/thorium/result-files".into(),
            tags: "/tmp/thorium/tags".into(),
            names: Vec::default(),
        }
    }
}

impl FilesHandler {
    /// Set the results path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandler;
    ///
    /// FilesHandler::default().results("/data/results");
    /// ```
    #[must_use]
    pub fn results<T: Into<String>>(mut self, path: T) -> Self {
        self.results = path.into();
        self
    }

    /// Set the result files path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandler;
    ///
    /// FilesHandler::default().result_files("/data/result_files");
    /// ```
    #[must_use]
    pub fn result_files<T: Into<String>>(mut self, path: T) -> Self {
        self.result_files = path.into();
        self
    }

    /// Add a name to the list of results or result files to restrict to
    ///
    /// # Arguments
    ///
    /// * `name` - The name to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandler;
    ///
    /// FilesHandler::default().name("output.json");
    /// ```
    #[must_use]
    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        self.names.push(name.into());
        self
    }

    /// Add multiple names to the list of results or result files to restrict to
    ///
    /// # Arguments
    ///
    /// * `names` - The names to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandler;
    ///
    /// FilesHandler::default().names(vec!("output.json", "corn.png"));
    /// ```
    #[must_use]
    pub fn names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        self.names.extend(names.into_iter().map(Into::into));
        self
    }
}

impl PartialEq<FilesHandlerUpdate> for FilesHandler {
    /// Check if an [`FilesHandler`] contains all the updates from a [`FilesHandlerUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to compare against
    fn eq(&self, update: &FilesHandlerUpdate) -> bool {
        // make sure any updates were applied
        matches_update!(self.results, update.results);
        matches_update!(self.result_files, update.result_files);
        matches_adds!(self.names, update.add_names);
        // make sure we removed any requested names
        matches_removes!(self.names, update.remove_names);
        true
    }
}

/// The logic to use to determine whether to create this tag or not
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum AutoTagLogic {
    Exists,
    Equal(Value),
    Not(Value),
    Greater(Value),
    GreaterOrEqual(Value),
    LesserOrEqual(Value),
    Lesser(Value),
    In(Vec<Value>),
    NotIn(Vec<Value>),
}

impl Default for AutoTagLogic {
    /// Create a default [`AutoTagLogic`]
    fn default() -> Self {
        AutoTagLogic::Exists
    }
}

/// Settings for extracting a single tag from a result
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct AutoTag {
    /// The logic to use when deciding whether to apply this tag
    pub logic: AutoTagLogic,
    /// What to rename this tags key too
    pub key: Option<String>,
}

impl AutoTag {
    /// Set a custom logic for an auto tag
    ///
    /// # Arguments
    ///
    /// * `logic` - The logic to use when creating this tag automatically
    #[must_use]
    pub fn logic(mut self, logic: AutoTagLogic) -> Self {
        self.logic = logic;
        self
    }

    /// Set the key to rename this tag too when creating it
    ///
    /// # Arguments
    ///
    /// * `key` - The key to use instead of the one found in the result
    #[must_use]
    pub fn key(mut self, key: String) -> Self {
        self.key = Some(key);
        self
    }
}

impl PartialEq<AutoTagUpdate> for AutoTag {
    /// Check if an [`AutoTag`] contains all the updates from a [`AutoTagUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The auto tag settings update to compare against
    fn eq(&self, update: &AutoTagUpdate) -> bool {
        // makes sure the logic and key are the same
        matches_update!(self.logic, update.logic);
        matches_update_opt!(self.key, update.key);
        matches_clear!(self.key, update.clear_key);
        true
    }
}

/// Helps serde default the children collection path
fn default_children() -> String {
    "/tmp/thorium/children".to_owned()
}

/// The settings for collecting output
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OutputCollection {
    /// The handler used to collect output
    #[serde(default)]
    pub handler: OutputHandler,
    /// The file Handler settings
    #[serde(default)]
    pub files: FilesHandler,
    /// Where to look for child files to ingest,
    #[serde(default = "default_children")]
    pub children: String,
    /// Settings for automatically extracting a tag from results
    #[serde(default)]
    pub auto_tag: HashMap<String, AutoTag>,
    /// The groups we should restrict our result uploads too
    #[serde(default)]
    pub groups: Vec<String>,
}

impl Default for OutputCollection {
    /// Create a default `OutputCollection` object
    fn default() -> Self {
        OutputCollection {
            handler: OutputHandler::default(),
            files: FilesHandler::default(),
            children: "/tmp/thorium/children".to_owned(),
            auto_tag: HashMap::default(),
            groups: Vec::default(),
        }
    }
}

impl OutputCollection {
    /// Set the settings for the Files Handler
    ///
    /// # Arguments
    ///
    /// * `files` - The files handler settings to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputCollection, FilesHandler};
    ///
    /// OutputCollection::default()
    ///     .files(FilesHandler::default()
    ///         .results("/data/results")
    ///         .result_files("/data/result_files")
    ///         .names(vec!("output.json", "corn.png")));
    /// ```
    #[must_use]
    pub fn files(mut self, files: FilesHandler) -> Self {
        self.files = files;
        self
    }
}

impl PartialEq<OutputCollectionUpdate> for OutputCollection {
    /// Check if an [`OutputCollection`] contains all the updates from a [`OutputCollectionUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to compare against
    fn eq(&self, update: &OutputCollectionUpdate) -> bool {
        // make sure any updates were applied
        matches_update!(self.handler, update.handler);
        same!(self.files, update.files);
        true
    }
}

/// The settings for collecting results from a specific location on disk
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct FilesHandlerUpdate {
    /// The location to look for small renderable results at on disk
    pub results: Option<String>,
    /// The location to look for files that should be uploaded as result files
    pub result_files: Option<String>,
    /// The location to load tags to set from
    pub tags: Option<String>,
    /// Any new file names to restrict our handler to
    #[serde(default)]
    pub add_names: Vec<String>,
    /// Any file names to remove from the list of file names to restrict our handler to
    #[serde(default)]
    pub remove_names: Vec<String>,
    /// Whether to clear the list of files names to restrict our handler to
    #[serde(default)]
    pub clear_names: bool,
}

impl FilesHandlerUpdate {
    /// set the results path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().results("/data/results");
    /// ```
    #[must_use]
    pub fn results<T: Into<String>>(mut self, path: T) -> Self {
        self.results = Some(path.into());
        self
    }

    /// Set the result files path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().result_files("/data/result_files");
    /// ```
    #[must_use]
    pub fn result_files<T: Into<String>>(mut self, path: T) -> Self {
        self.result_files = Some(path.into());
        self
    }

    /// Set the tags path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().tags("/data/tags");
    /// ```
    #[must_use]
    pub fn tags<T: Into<String>>(mut self, path: T) -> Self {
        self.tags = Some(path.into());
        self
    }

    /// Add a name to the list of results or result files to restrict to
    ///
    /// # Arguments
    ///
    /// * `name` - The name to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().add_name("output.json");
    /// ```
    #[must_use]
    pub fn add_name<T: Into<String>>(mut self, name: T) -> Self {
        self.add_names.push(name.into());
        self
    }

    /// Add multiple names to the list of results or result files to restrict to
    ///
    /// # Arguments
    ///
    /// * `names` - The names to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().add_names(vec!("output.json", "corn.png"));
    /// ```
    #[must_use]
    pub fn add_names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        self.add_names.extend(names.into_iter().map(Into::into));
        self
    }

    /// Add a name to remove from the list of results or result files to restrict to
    ///
    /// # Arguments
    ///
    /// * `name` - The name to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().remove_name("output.json");
    /// ```
    #[must_use]
    pub fn remove_name<T: Into<String>>(mut self, name: T) -> Self {
        self.remove_names.push(name.into());
        self
    }

    /// Add multiple names to remove from the list of results or result files to restrict to
    ///
    /// # Arguments
    ///
    /// * `names` - The names to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::FilesHandlerUpdate;
    ///
    /// FilesHandlerUpdate::default().remove_names(vec!("output.json", "corn.png"));
    /// ```
    #[must_use]
    pub fn remove_names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        self.remove_names.extend(names.into_iter().map(Into::into));
        self
    }
}

impl PartialEq<FilesHandler> for FilesHandlerUpdate {
    /// Check if an [`FilesHandler`] contains all the updates from a [`FilesHandlerUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The updated files handler to compare against
    fn eq(&self, handler: &FilesHandler) -> bool {
        // make sure any updates were applied
        matches_update!(handler.results, self.results);
        matches_update!(handler.result_files, self.result_files);
        matches_adds!(handler.names, self.add_names);
        // make sure we removed any requested names
        matches_removes!(handler.names, self.remove_names);
        true
    }
}

/// The settings to update for extracting a single tag from a result
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct AutoTagUpdate {
    /// The logic to use when deciding whether to apply this tag
    pub logic: Option<AutoTagLogic>,
    /// What to rename this tags key too
    pub key: Option<String>,
    // Whether to clear the key value or not
    #[serde(default)]
    pub clear_key: bool,
    /// whether to delete this tag key or not
    #[serde(default)]
    pub delete: bool,
}

impl AutoTagUpdate {
    /// Set a custom logic for an auto tag
    ///
    /// # Arguments
    ///
    /// * `logic` - The logic to use when creating this tag automatically
    #[must_use]
    pub fn logic(mut self, logic: AutoTagLogic) -> Self {
        self.logic = Some(logic);
        self
    }

    /// Set the key to rename this tag too when creating it
    ///
    /// # Arguments
    ///
    /// * `key` - The key to use instead of the one found in the result
    #[must_use]
    pub fn key(mut self, key: String) -> Self {
        self.key = Some(key);
        self
    }

    /// Clear this auto tags key value
    #[must_use]
    pub fn clear_key(mut self) -> Self {
        self.clear_key = true;
        self
    }

    /// Set this auto tag to be deleted
    #[must_use]
    pub fn delete(mut self) -> Self {
        self.delete = true;
        self
    }
}

impl PartialEq<AutoTag> for AutoTagUpdate {
    /// Check if an [`AutoTag`] contains all the updates from a [`AutoTagUpdate`]
    ///
    /// # Arguments
    ///
    /// * `settings` - The auto tag settings to compare against
    fn eq(&self, settings: &AutoTag) -> bool {
        // makes sure the logic and key are the same
        matches_update!(settings.logic, self.logic);
        matches_update_opt!(settings.key, self.key);
        matches_clear!(settings.key, self.clear_key);
        true
    }
}

/// The settings for collecting output
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OutputCollectionUpdate {
    /// The handler used to collect output
    #[serde(default)]
    pub handler: Option<OutputHandler>,
    /// The file Handler settings
    #[serde(default)]
    pub files: FilesHandlerUpdate,
    /// Update settings for automatically extracting a tag from results
    #[serde(default)]
    pub auto_tag: HashMap<String, AutoTagUpdate>,
    /// Where to look for child files to ingest,
    #[serde(default)]
    pub children: Option<String>,
    /// The groups we should restrict our results uploads too
    #[serde(default)]
    pub groups: Vec<String>,
    /// Whether to clear the files handler settings
    #[serde(default)]
    pub clear_files: bool,
    /// Whether to clear the results groups restrictions or not
    #[serde(default)]
    pub clear_groups: bool,
}

impl OutputCollectionUpdate {
    /// Sets files handler settings to be cleared
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputCollectionUpdate, OutputHandler};
    ///
    /// OutputCollectionUpdate::default().handler(OutputHandler::Files);
    /// ```
    #[must_use]
    pub fn handler(mut self, handler: OutputHandler) -> Self {
        self.handler = Some(handler);
        self
    }

    /// Set the settings for the files handler
    ///
    /// # Arguments
    ///
    /// * `files` - The files handler settings to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputCollectionUpdate, FilesHandlerUpdate};
    ///
    /// OutputCollectionUpdate::default()
    ///     .files(FilesHandlerUpdate::default()
    ///         .results("/data/results")
    ///         .result_files("/data/result_files")
    ///         .add_names(vec!("output.json", "corn.png"))
    ///         .remove_names(vec!("output.txt", "soy.png")));
    /// ```
    #[must_use]
    pub fn files(mut self, files: FilesHandlerUpdate) -> Self {
        self.files = files;
        self
    }

    /// Adds an auto tag update
    ///
    /// # Arguments
    ///
    /// * `key` - The key to extract tags from
    /// * `auto_tag` - The auto tag setting to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputCollectionUpdate, AutoTagUpdate};
    ///
    /// OutputCollectionUpdate::default()
    ///    .auto_tag("Plant", AutoTagUpdate::default());
    /// ```
    #[must_use]
    pub fn auto_tag<T: Into<String>>(mut self, key: T, auto_tag: AutoTagUpdate) -> Self {
        self.auto_tag.insert(key.into(), auto_tag);
        self
    }

    /// Sets files handler settings to be cleared
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputCollectionUpdate};
    ///
    /// OutputCollectionUpdate::default().clear_files();
    /// ```
    #[must_use]
    pub fn clear_files(mut self) -> Self {
        self.clear_files = true;
        self
    }

    /// Sets the result groups restrictions to be cleared
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{OutputCollectionUpdate};
    ///
    /// OutputCollectionUpdate::default().clear_groups();
    /// ```
    #[must_use]
    pub fn clear_groups(mut self) -> Self {
        self.clear_groups = true;
        self
    }
}

impl PartialEq<OutputCollection> for OutputCollectionUpdate {
    /// Check if an [`OutputCollection`] contains all the updates from a [`OutputCollectionUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to compare against
    fn eq(&self, collection: &OutputCollection) -> bool {
        // make sure any updates were applied
        matches_update!(collection.handler, self.handler);
        same!(collection.files, self.files);
        // make sure that all auto tag updates are applied
        for (key, update) in &self.auto_tag {
            // determine if this update was properly applied
            if let Some(applied) = collection.auto_tag.get(key) {
                // return false if our updates were not applied
                if applied != update {
                    return false;
                }
            } else {
                // if this auto tag was not deleted then return false as its missing
                if !update.delete {
                    return false;
                }
            }
        }
        true
    }
}
