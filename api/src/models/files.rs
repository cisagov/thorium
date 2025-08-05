//! Tracks files uploaded to Thorium by users
//!
//! This allows Thorium to more easily generate and execute jobs based on data within Thorium but
//! also allows users to find and track their data. Tags are used to facilitate searching on the
//! files within Thorium.

use bytes::Bytes;
use cart_rs::UncartStream;
use chrono::prelude::*;
use indicatif::ProgressBar;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::hash::Hasher;
use std::net::IpAddr;
use std::path::Path;
use std::path::PathBuf;
use uuid::Uuid;

use super::{OnDiskFile, TreeSupport};
use crate::{matches_adds, matches_removes, matches_update_opt, same};

// api only imports
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use crate::utils::{ApiError, Shared};
        use super::{User, TagDeleteRequest};
        use std::str::FromStr;
    }
}

// client only imports
cfg_if::cfg_if! {
    if #[cfg(feature = "client")] {
        use tokio::fs::{File, OpenOptions};
        use tokio::io::BufReader;
        use crate::client::Error;
        use crate::{multipart_file, multipart_list, multipart_list_conv, multipart_text, multipart_text_to_string};
    }
}

// api/client imports
cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        use crate::models::scylla_utils::keys::KeySupport;
        use super::backends::{OutputSupport, TagSupport};
        use super::{OutputKind, TagRequest, TagType};
    }
}

// only support scylla and other api side only structs if the api features is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        /// A request to to set the origin for a submission
        ///
        /// This is only used internally to deserialize multipart forms
        #[derive(Debug, Default)]
        pub struct OriginForm {
            /// The type of origin this should be deserialized as
            pub origin_type: OriginTypes,
            /// The result IDs to add this child too
            pub result_ids: Vec<Uuid>,
            /// The url this was downloaded from
            pub url: Option<String>,
            /// The non url name of this website
            pub name: Option<String>,
            /// The tool that unpacked or transformed this sample
            pub tool: Option<String>,
            /// The sha256 of the sample this was unpacked/transformed from
            pub parent: Option<String>,
            /// The flags that were used to transform or build this sample
            pub flags: Vec<String>,
            /// The full command used to transform this sample
            pub cmd: Option<String>,
            /// The sniffer this sample came from
            pub sniffer: Option<String>,
            /// The source IP/hostname this sample came from when it was sniffed
            pub source: Option<String>,
            /// Where this sample was headed to when it was sniffed
            pub destination: Option<String>,
            /// The name or id of the incident this sample was from
            pub incident: Option<String>,
            /// The cover term for this incident
            pub cover_term: Option<String>,
            /// The mission team that handled this incident
            pub mission_team: Option<String>,
            /// The network this incident occurred on
            pub network: Option<String>,
            /// The IP or hostname of the machine this occurred on
            pub machine: Option<String>,
            /// The physical location of this incident
            pub location: Option<String>,
            /// The type of memory this memory dump originates from
            pub memory_type: Option<String>,
            /// The characteristics that were reconstructed in this memory dump
            pub reconstructed: Vec<String>,
            /// the base address for this memory dump
            pub base_addr: Option<String>,
            /// The repository this was built from
            pub repo: Option<String>,
            /// The branch, commit, or tag this child sample was built from
            pub commitish: Option<String>,
            /// The commit the repository was on
            pub commit: Option<String>,
            /// The build system that was used to build this
            pub system: Option<String>,
            /// Whether this is a supporting build file or a final build file
            pub supporting: Option<bool>,
            /// The source IP this sample was sent from
            pub src_ip: Option<IpAddr>,
            /// The destination IP this sample was going to
            pub dest_ip: Option<IpAddr>,
            /// The source port this sample was sent from
            pub src_port: Option<u16>,
            /// The destination port this sample was going to
            pub dest_port: Option<u16>,
            /// The type of protocol this sample was transported in
            pub proto: Option<PcapNetworkProtocol>,
        }

        /// A request to upload a sample to Thorium
        #[derive(Debug, Default)]
        pub struct SampleForm {
            /// The groups this sample is a part of
            pub groups: Vec<String>,
            /// A description for this sample
            pub description: Option<String>,
            /// The tags for this sample
            pub tags: HashMap<String, HashSet<String>>,
            /// The origin of this sample if one exists
            pub origin: OriginForm,
            /// An optional name of this file
            pub file_name: Option<String>,
            /// The trigger depth for this sample request
            pub trigger_depth: u8,
        }

        /// A request for a comment about a specific sample
        #[derive(Debug)]
        pub struct CommentForm {
            /// The ID to assign for this form
            pub id: Uuid,
            /// The groups to share this comment with
            pub groups: Vec<String>,
            /// The comment to save
            pub comment: String,
            /// Mappings of attachment file names to S3 UUID's
            pub attachments: HashMap<String, Uuid>,
        }

        impl Default for CommentForm {
            /// Create a default comment form
            fn default() -> Self {
                CommentForm {
                    id: Uuid::new_v4(),
                    groups: Vec::default(),
                    comment: String::default(),
                    attachments: HashMap::default()
                }
            }
        }
    }
}

/// A struct used for checking
///
/// This will be the same as `SampleRequest` but data will be a path instead of the actual contents
/// of the file.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SampleCheck {
    /// The sha256 of the file to check for
    pub sha256: String,
    /// The name of this file
    pub name: Option<String>,
    /// The groups this sample is a part of
    #[serde(default)]
    pub groups: Vec<String>,
    /// The origin of this sample if one exists
    pub origin: Option<OriginRequest>,
}

impl SampleCheck {
    /// Builds a new sample existence check object
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 to check against
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleCheck;
    ///
    /// SampleCheck::new("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f");
    /// ```
    pub fn new<T: Into<String>>(sha256: T) -> Self {
        SampleCheck {
            sha256: sha256.into(),
            name: None,
            groups: Vec::default(),
            origin: None,
        }
    }

    /// Adds a single group for this existence check
    ///
    /// # Arguments
    ///
    /// * `group` - The group to check against
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleCheck;
    ///
    /// SampleCheck::new("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f")
    ///     .group("corn");
    /// ```
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        // convert this group to a string and set it
        self.groups.push(group.into());
        self
    }

    /// Adds multiple groups for this existence check
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleCheck;
    ///
    /// SampleCheck::new("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f")
    ///     .groups(vec!("corn", "tacos"));
    /// ```
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // convert these groups  to strings and add them
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// Sets the origin for this existence check
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{SampleCheck, OriginRequest};
    ///
    /// SampleCheck::new("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f")
    ///     .origin(OriginRequest::downloaded("https://google.com", Some("google".to_string())));
    /// ```
    #[must_use]
    pub fn origin(mut self, origin: OriginRequest) -> Self {
        self.origin = Some(origin);
        self
    }
}

/// A struct used for checking if a submission already exists
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SampleCheckResponse {
    /// Whether this sample exists or not
    pub exists: bool,
    /// The id of the already created exactly matching submission object
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Uuid>,
}

/// A in memory buffer to upload
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Buffer {
    /// An optional file name to use for this buffer when written to disk
    pub name: Option<String>,
    /// The buffer to upload
    pub data: Vec<u8>,
}

impl Buffer {
    /// Creates a new buffer to upload
    ///
    /// # Arguments
    ///
    /// * `data` - The in memory buffer to upload
    pub fn new<T: Into<Vec<u8>>>(data: T) -> Self {
        Buffer {
            name: None,
            data: data.into(),
        }
    }

    /// Set a file name to use for this buffer when written to disk
    ///
    /// # Arguments
    ///
    /// * `name` - The file name to use when writing this buffe to disk
    #[must_use]
    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.name = Some(name.into());
        self
    }

    /// Create a multipart part from this buffer
    #[cfg(feature = "client")]
    pub fn to_part(self) -> Result<reqwest::multipart::Part, reqwest::Error> {
        // create a Part object containing this buffer as bytes
        let mut part = reqwest::multipart::Part::bytes(self.data)
            // set the mime string to form-data so the server doesn't corrupt the data
            .mime_str("multipart/form-data")?;
        // if a file name was set then add that info
        if let Some(name) = self.name {
            part = part.file_name(name);
        }
        Ok(part)
    }
}

/// A struct used for uploading samples to Thorium
#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SampleRequest {
    /// The groups this sample is a part of
    pub groups: Vec<String>,
    /// A description for this sample
    pub description: Option<String>,
    /// The tags for this sample
    pub tags: HashMap<String, HashSet<String>>,
    /// The origin of this sample if one exists
    pub origin: Option<OriginRequest>,
    /// The path to the file to upload if this sample is on disk
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub path: Option<PathBuf>,
    /// The data to upload directly
    pub data: Option<Buffer>,
    /// The trigger depth of this sample upload
    #[serde(default)]
    pub trigger_depth: u8,
}

impl SampleRequest {
    /// Creates a new sample request for target a file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to upload a file from
    /// * `groups` - The groups to upload this file too
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use thorium::models::{SampleRequest, OriginRequest};
    ///
    /// SampleRequest::new("files.rs", vec!("CornPeeps"))
    ///     .description("A wonderful picture of corn")
    ///     .tag("plant", "corn")
    ///     .origin(OriginRequest::downloaded("https://google.com", Some("google".to_string())));
    /// ```
    pub fn new<P: Into<PathBuf>, T: Into<String>>(path: P, groups: Vec<T>) -> Self {
        // convert out list of groups into strings
        let groups = groups.into_iter().map(Into::into).collect();
        SampleRequest {
            groups,
            description: None,
            tags: HashMap::default(),
            origin: None,
            path: Some(path.into()),
            data: None,
            trigger_depth: 0,
        }
    }

    /// Creates a new sample request for an in memory buffer
    ///
    /// # Arguments
    ///
    /// * `data` - The in memory buffer to upload
    /// * `groups` - The groups to upload this file too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{SampleRequest, OriginRequest};
    ///
    /// SampleRequest::new("data", vec!("CornPeeps"))
    ///     .description("A wonderful picture of corn")
    ///     .tag("plant", "corn")
    ///     .origin(OriginRequest::downloaded("https://google.com", Some("google".to_string())));
    /// ```
    #[must_use]
    pub fn new_buffer<T: Into<String>>(data: Buffer, groups: Vec<T>) -> Self {
        // convert out list of groups into strings
        let groups = groups.into_iter().map(|group| group.into()).collect();
        SampleRequest {
            groups,
            description: None,
            tags: HashMap::default(),
            origin: None,
            path: None,
            data: Some(data),
            trigger_depth: 0,
        }
    }

    /// Adds a description for this sample
    ///
    /// # Arguments
    ///
    /// * `description` - The description to set for this file
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleRequest;
    ///
    /// SampleRequest::new("/corn.jpeg", vec!("CornPeeps"))
    ///     .description("Lots of corn");
    /// ```
    #[must_use]
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        // convert this description to a string and set it
        self.description = Some(description.into());
        self
    }

    /// Adds a tag for this sample
    ///
    /// # Arguments
    ///
    /// * `key` - The key to set for this tag
    /// * `value` - The value to set for this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleRequest;
    ///
    /// SampleRequest::new("/corn.jpeg", vec!("CornPeeps"))
    ///     .tag("plant", "corn");
    /// ```
    #[must_use]
    pub fn tag<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        // get the vector of values for this tag or insert a default
        let values = self.tags.entry(key.into()).or_default();
        // insert our new tag
        values.insert(value.into());
        self
    }

    /// Adds multiple values for the same tag for this sample
    ///
    /// # Arguments
    ///
    /// * `key` - The key to set for this tag
    /// * `value` - The values to set for this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleRequest;
    ///
    /// SampleRequest::new("/corn.jpeg", vec!("CornPeeps"))
    ///     .tags("plant", vec!("corn", "oranges"));
    /// ```
    #[must_use]
    pub fn tags<T: Into<String>>(mut self, key: T, values: Vec<T>) -> Self {
        // get the vector of values for this tag or insert a default
        let entry = self.tags.entry(key.into()).or_default();
        // insert our new tags
        entry.extend(values.into_iter().map(|val| val.into()));
        self
    }

    /// Sets the origin for this sample upload
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin to set for this sample
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{SampleRequest, OriginRequest};
    ///
    /// SampleRequest::new("/corn.jpeg", vec!("CornPeeps"))
    ///     .origin(OriginRequest::downloaded("https://google.com", Some("google".to_string())));
    /// ```
    #[must_use]
    pub fn origin(mut self, origin: OriginRequest) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Create a multipart form from this sample request
    #[cfg(feature = "client")]
    pub async fn to_form(mut self) -> Result<reqwest::multipart::Form, Error> {
        // build the form we are going to send

        use crate::multipart_set;
        // disable percent encoding, as the API natively supports UTF-8
        let form = reqwest::multipart::Form::new().percent_encode_noop();
        let form = multipart_text!(form, "description", self.description);
        let mut form = multipart_list!(form, "groups", self.groups);
        // add any tags to this form
        for (key, mut values) in self.tags {
            // build the tag key for this tag
            let tag_key = format!("tags[{key}]");
            // add this tags list of values to our form
            form = multipart_set!(form, &tag_key, values);
        }
        // if an origin was set then set those fields
        let form = match self.origin.take() {
            Some(origin) => origin.extend_form(form),
            None => form,
        };
        // if a trigger depth was set then add that to our form
        let form = form.text("trigger_depth", format!("{}", self.trigger_depth));
        // read in this file if a path was set
        let form = if let Some(path) = self.path.take() {
            // a path was set so read in that file and add it to the form
            multipart_file!(form, "data", path)
        } else {
            // no path was set so a buffer must have been used
            form.part("data", self.data.unwrap().to_part()?)
        };
        Ok(form)
    }

    /// Set the trigger depth for this sample request
    ///
    /// # Arguments
    ///
    /// * `trigger_depth` - The trigger depth to set
    #[must_use]
    pub fn trigger_depth(mut self, trigger_depth: u8) -> Self {
        // update our trigger depth
        self.trigger_depth = trigger_depth;
        self
    }
}

impl std::fmt::Debug for SampleRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampleRequest")
            .field("groups", &self.groups)
            .field("description", &self.description)
            .field("origin", &self.origin)
            .field("path", &self.path)
            .field("data", &self.data.is_some())
            .finish()
    }
}

impl PartialEq<Sample> for SampleRequest {
    /// Check if a [`Sample`] and a [`SampleRequest`] are equal
    ///
    /// # Arguments
    ///
    /// * `req` - The Sample to compare against
    fn eq(&self, req: &Sample) -> bool {
        // find our submission in this sample
        // this is a bit unreliable as we don't have our username in the request
        req.submissions.iter().any(|sub| {
            sub.description == self.description
                && self.groups.iter().all(|group| sub.groups.contains(group))
                && sub.origin == self.origin
        })
    }
}

impl PartialEq<CommentRequest> for Sample {
    /// Check if a [`CommentRequest`] was added to a [`Sample`]
    ///
    /// # Arguments
    ///
    /// * `req` - The `CommentRequest` to compare against
    fn eq(&self, req: &CommentRequest) -> bool {
        self.comments.iter().any(|comment| comment == req)
    }
}

/// The response from a file submission
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SampleSubmissionResponse {
    /// The sha256 of this sample
    pub sha256: String,
    /// The sha1 of this sample
    pub sha1: String,
    /// The md5 of this sample
    pub md5: String,
    /// A UUID for this submission
    pub id: Uuid,
}

/// A tag object used to filter samples by when searching
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Tag {
    /// The sha256 of the sample this tag is for
    pub sha256: String,
    /// The groups this tag is a part of
    pub groups: Vec<String>,
    /// The values for this tag
    pub values: Vec<String>,
}

/// The different type of carved origins
#[derive(Debug)]
pub enum CarvedOriginTypes {
    /// The sample was carved from a packet capture
    Pcap,
    /// The sample was carved from an unknown or unspecified file type
    Unknown,
}

/// The different types of origins
#[derive(Debug)]
pub enum OriginTypes {
    /// This sample was downloaded from an external website
    Downloaded,
    /// This sample was unpacked from another sample
    Unpacked,
    /// This sample was unpacked from another sample
    Transformed,
    /// This sample comes from a sniffer/the wire
    Wire,
    /// This sample cames from an incident or engagement
    Incident,
    /// This sample comes from dumping memory while running a parent sample
    MemoryDump,
    /// This sample was built from source
    Source,
    /// This sample was statically carved out from another sample
    ///
    /// Unlike `Unpacked`, `Carved` describes a sample that is just
    /// a simple piece of another file, like a file from an archive or
    /// a packet capture. It's extraction can be easily replicated without
    /// any dynamic unpacking process.
    Carved(CarvedOriginTypes),
    /// This sample has no unique origin
    None,
}

impl Default for OriginTypes {
    /// Build a default origin type of None
    fn default() -> Self {
        OriginTypes::None
    }
}

/// A request to to set the origin for a submission
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct OriginRequest {
    /// The type of origin this should be deserialized as
    pub origin_type: String,
    /// The result IDs to add this child too
    pub result_ids: Vec<Uuid>,
    /// The url this was downloaded from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The non url name of this website
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The tool that unpacked or transformed this sample
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    ///  The sha256 of the sample this was unpacked/transformed from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// The flags that were used to transform or build this sample
    #[serde(default)]
    pub flags: Vec<String>,
    /// The full command used to transform this sample
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmd: Option<String>,
    /// The sniffer this sample came from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sniffer: Option<String>,
    /// The source IP/hostname this sample came from when it was sniffed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Where this sample was headed to when it was sniffed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination: Option<String>,
    /// The name or id of the incident this sample was from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incident: Option<String>,
    /// The cover term for this incident
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cover_term: Option<String>,
    /// The mission team that handled this incident
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mission_team: Option<String>,
    /// The network this incident occurred on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<String>,
    /// The IP or hostname of the machine this occurred on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
    /// The physical location of this incident
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// The type of memory this memory dump originates from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    /// The characteristics that were reconstructed in this memory dump
    #[serde(default)]
    pub reconstructed: Vec<String>,
    /// the base address for this memory dump
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_addr: Option<String>,
    /// The repository this was built from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// The branch, commit, or tag this child sample was built from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitish: Option<String>,
    /// The commit the repository was on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// The build system that was used to build this
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Whether this is a supporting build file or a final build file
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supporting: Option<bool>,
    /// The source IP this sample was sent from
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub src_ip: Option<IpAddr>,
    /// The destination IP this sample was going to
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub dest_ip: Option<IpAddr>,
    /// The source port this sample was sent from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub src_port: Option<u16>,
    /// The destination port this sample was going to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dest_port: Option<u16>,
    /// The type of protocol this sample was transported in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proto: Option<PcapNetworkProtocol>,
    /// If this is an origin update then set the origin to None
    #[serde(default)]
    pub clear_origin: bool,
}

impl OriginRequest {
    /// Builds a downloaded origin request
    ///
    /// # Arguments
    ///
    /// * `url` - The url this sample was downloaded from
    /// * `name` - The optional name of the site this was downloaded from
    pub fn downloaded<T: Into<String>>(url: T, name: Option<String>) -> Self {
        OriginRequest {
            origin_type: "Downloaded".to_owned(),
            result_ids: Vec::default(),
            url: Some(url.into()),
            name,
            tool: None,
            parent: None,
            flags: Vec::default(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            clear_origin: false,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds a unpacked origin request
    ///
    /// # Arguments
    ///
    /// * `parent` - The sha256 of the parent file this sample was unpacked from
    /// * `tool` - The optional tool that unpacked this sample
    pub fn unpacked<T: Into<String>>(parent: T, tool: Option<String>) -> Self {
        OriginRequest {
            origin_type: "Unpacked".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool,
            parent: Some(parent.into()),
            flags: Vec::default(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            clear_origin: false,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds a transformed origin request
    ///
    /// # Arguments
    ///
    /// * `tool` - The optional tool that unpacked this sample
    /// * `parent` - The sha256 of the parent file this sample was unpacked from
    /// * `flags` - The flags used to transform the parent file into this file
    /// * `cmd` - The cmd used to transform the parent file into this file
    pub fn transformed<T, F>(
        parent: T,
        tool: Option<String>,
        flags: Vec<F>,
        cmd: Option<String>,
    ) -> Self
    where
        T: Into<String>,
        F: Into<String>,
    {
        OriginRequest {
            origin_type: "Transformed".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool,
            parent: Some(parent.into()),
            flags: flags.into_iter().map(Into::into).collect(),
            cmd,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            clear_origin: false,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds a wire origin request
    ///
    /// # Arguments
    ///
    /// * `sniffer` - The sniffer that found this sample
    /// * `source` - The source IP/hostname for this sample
    /// * `destination` - The destination IP/hostname for this sample
    pub fn wire<T: Into<String>>(
        sniffer: T,
        source: Option<String>,
        destination: Option<String>,
    ) -> Self {
        OriginRequest {
            origin_type: "Wire".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool: None,
            parent: None,
            flags: Vec::default(),
            cmd: None,
            sniffer: Some(sniffer.into()),
            source,
            destination,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            clear_origin: false,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds an incident origin request
    ///
    /// # Arguments
    ///
    /// * `incident` - The name or other unique identifier for this incident
    /// * `cover_term` - The cover term for this incident
    /// * `mission_team` - The mission team for this incident
    /// * `network` - The name of the network this occurred on
    /// * `machine` - The machine this incident occurred on
    /// * `location` - The physical location of this incident (building/office)
    pub fn incident<T: Into<String>>(
        incident: T,
        cover_term: Option<String>,
        mission_team: Option<String>,
        network: Option<String>,
        machine: Option<String>,
        location: Option<String>,
    ) -> Self {
        OriginRequest {
            origin_type: "Incident".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool: None,
            parent: None,
            flags: Vec::default(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: Some(incident.into()),
            cover_term,
            mission_team,
            network,
            machine,
            location,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            clear_origin: false,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds a memory dump origin request
    ///
    /// # Arguments
    ///
    /// * `memory_type` - The type of memory this memory dump came from
    /// * `reconstructed` - The attributes that were reconstructed on this memory dump
    /// * `base_addr` - The base address of this memory dump
    pub fn memory_dump<P, T>(
        parent: P,
        memory_type: Option<String>,
        reconstructed: Vec<T>,
        base_addr: Option<String>,
    ) -> Self
    where
        P: Into<String>,
        T: Into<String>,
    {
        OriginRequest {
            origin_type: "MemoryDump".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool: None,
            parent: Some(parent.into()),
            flags: Vec::default(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type,
            reconstructed: reconstructed.into_iter().map(Into::into).collect(),
            base_addr,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            clear_origin: false,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds a source origin request
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo this file came from
    /// * `commitish` - The branch, commit, or tag this file was built from
    /// * `commit` - The commit this file came from
    /// * `flags` - The flags that were used to build this file
    /// * `system` - The build system that was used
    /// * `supporting` - Whether this is a supporting file or not
    pub fn source<R, C, D, F, S>(
        repo: R,
        commitish: Option<C>,
        commit: D,
        flags: impl Iterator<Item = F>,
        system: S,
        supporting: bool,
    ) -> Self
    where
        R: Into<String>,
        C: Into<String>,
        D: Into<String>,
        F: Into<String>,
        S: Into<String>,
    {
        OriginRequest {
            origin_type: "Source".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool: None,
            parent: None,
            flags: flags.into_iter().map(Into::into).collect(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: Some(repo.into()),
            commitish: commitish.map(|val| val.into()),
            commit: Some(commit.into()),
            system: Some(system.into()),
            clear_origin: false,
            supporting: Some(supporting),
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
        }
    }

    /// Builds a carved from packet capture origin request
    ///
    /// # Arguments
    ///
    /// * `parent` - The sample's parent SHA256
    /// * `tool` - The tool that carved this sample from the packet capture
    /// * `src_ip` - The source IP this file was sent from
    /// * `dest_ip` - The destination IP this file was sent to
    /// * `src_port` - The source port this file was sent from
    /// * `dest_port` - The destination port this file was sent to
    /// * `proto` - The protocol over which this file was sent
    /// * `url` - The URL this file was sent to/from
    pub fn carved_pcap<P: Into<String>>(
        parent: P,
        tool: Option<String>,
        src_ip: Option<IpAddr>,
        dest_ip: Option<IpAddr>,
        src_port: Option<u16>,
        dest_port: Option<u16>,
        proto: Option<PcapNetworkProtocol>,
        url: Option<String>,
    ) -> Self {
        Self {
            origin_type: "CarvedPcap".to_owned(),
            result_ids: Vec::default(),
            url,
            name: None,
            tool,
            parent: Some(parent.into()),
            flags: Vec::default(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            supporting: None,
            src_ip,
            dest_ip,
            src_port,
            dest_port,
            proto,
            clear_origin: false,
        }
    }

    /// Builds a carved origin request
    ///
    /// # Arguments
    ///
    /// * `parent` - The sample's parent SHA256
    /// * `tool` - The tool that carved this sample from the packet capture
    pub fn carved_unknown<P: Into<String>>(parent: P, tool: Option<String>) -> Self {
        Self {
            origin_type: "CarvedUnknown".to_owned(),
            result_ids: Vec::default(),
            url: None,
            name: None,
            tool,
            parent: Some(parent.into()),
            flags: Vec::default(),
            cmd: None,
            sniffer: None,
            source: None,
            destination: None,
            incident: None,
            cover_term: None,
            mission_team: None,
            network: None,
            machine: None,
            location: None,
            memory_type: None,
            reconstructed: Vec::default(),
            base_addr: None,
            repo: None,
            commitish: None,
            commit: None,
            system: None,
            supporting: None,
            src_ip: None,
            dest_ip: None,
            src_port: None,
            dest_port: None,
            proto: None,
            clear_origin: false,
        }
    }

    /// A result ID to this origin
    ///
    /// # Arguments
    ///
    /// * `result_id` - The result ID to add to this origin
    #[must_use]
    pub fn result_id(mut self, result_id: Uuid) -> Self {
        self.result_ids.push(result_id);
        self
    }

    /// Adds a list of result IDs to this origin
    ///
    /// # Arguments
    ///
    /// * `result_id` - The result ID to add to this origin
    #[must_use]
    pub fn result_ids(mut self, mut result_ids: Vec<Uuid>) -> Self {
        self.result_ids.append(&mut result_ids);
        self
    }

    /// Create a multipart part from this buffer
    ///
    /// # Arguments
    ///
    /// * `form` - The form to extend with our origin request info
    #[cfg(feature = "client")]
    pub fn extend_form(mut self, form: reqwest::multipart::Form) -> reqwest::multipart::Form {
        // set the type of origin this is
        let form = form.text("origin[origin_type]", self.origin_type);
        // add any values that were set
        let form = multipart_list_conv!(form, "origin[result_ids]", self.result_ids);
        let form = multipart_text!(form, "origin[url]", self.url);
        let form = multipart_text!(form, "origin[name]", self.name);
        let form = multipart_text!(form, "origin[tool]", self.tool);
        let form = multipart_text!(form, "origin[parent]", self.parent);
        let form = multipart_list!(form, "origin[flags]", self.flags);
        let form = multipart_text!(form, "origin[sniffer]", self.sniffer);
        let form = multipart_text!(form, "origin[source]", self.source);
        let form = multipart_text!(form, "origin[destination]", self.destination);
        let form = multipart_text!(form, "origin[incident]", self.incident);
        let form = multipart_text!(form, "origin[cover_term]", self.cover_term);
        let form = multipart_text!(form, "origin[mission_team]", self.mission_team);
        let form = multipart_text!(form, "origin[network]", self.network);
        let form = multipart_text!(form, "origin[machine]", self.machine);
        let form = multipart_text!(form, "origin[location]", self.location);
        let form = multipart_text!(form, "origin[memory_type]", self.memory_type);
        let form = multipart_list!(form, "origin[reconstructed]", self.reconstructed);
        let form = multipart_text!(form, "origin[base_addr]", self.base_addr);
        let form = multipart_text!(form, "origin[repo]", self.repo);
        let form = multipart_text!(form, "origin[commit]", self.commit);
        let form = multipart_text!(form, "origin[system]", self.system);
        let form = multipart_text_to_string!(form, "origin[supporting]", self.supporting);
        let form = multipart_text_to_string!(form, "origin[src_ip]", self.src_ip);
        let form = multipart_text_to_string!(form, "origin[dest_ip]", self.dest_ip);
        let form = multipart_text_to_string!(form, "origin[src_port]", self.src_port);
        let form = multipart_text_to_string!(form, "origin[dest_port]", self.dest_port);
        let form = multipart_text!(form, "origin[proto]", self.proto);
        // if the clear origin flag is set then add it to the form
        if self.clear_origin {
            form.text("origin[clear_origin]", "true")
        } else {
            form
        }
    }
}

/// The types of network protocols used in a packet capture
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum PcapNetworkProtocol {
    /// The TCP protocol
    #[serde(rename = "TCP", alias = "Tcp", alias = "tcp")]
    Tcp,
    /// The UDP protocol
    #[serde(rename = "UDP", alias = "Udp", alias = "udp")]
    Udp,
}

impl std::fmt::Display for PcapNetworkProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<PcapNetworkProtocol> for Cow<'static, str> {
    fn from(proto: PcapNetworkProtocol) -> Self {
        match proto {
            PcapNetworkProtocol::Tcp => Cow::Borrowed("TCP"),
            PcapNetworkProtocol::Udp => Cow::Borrowed("UDP"),
        }
    }
}

impl PcapNetworkProtocol {
    fn as_str(&self) -> &str {
        match self {
            PcapNetworkProtocol::Tcp => "TCP",
            PcapNetworkProtocol::Udp => "UDP",
        }
    }
}

#[cfg(feature = "api")]
impl FromStr for PcapNetworkProtocol {
    type Err = ApiError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "TCP" | "Tcp" | "tcp" => Ok(PcapNetworkProtocol::Tcp),
            "UDP" | "Udp" | "udp" => Ok(PcapNetworkProtocol::Udp),
            _ => crate::bad!(format!(
                "Invalid network protocol '{s}'! Valid network protocols \
                for packet captures are 'TCP/Tcp/tcp' and 'UDP/Udp/udp'"
            )),
        }
    }
}

/// The types of files a samples can be carved from
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum CarvedOrigin {
    /// The sample was carved from a packet capture
    Pcap {
        /// The source IP this sample was sent from
        #[cfg_attr(feature = "api", schema(value_type = String))]
        src_ip: Option<IpAddr>,
        /// The destination IP this sample was going to
        #[cfg_attr(feature = "api", schema(value_type = String))]
        dest_ip: Option<IpAddr>,
        /// The source port this sample was sent from
        src_port: Option<u16>,
        /// The destination port this sample was going to
        dest_port: Option<u16>,
        /// The type of protocol this sample was transported in
        proto: Option<PcapNetworkProtocol>,
        /// The URL this file was retrieved from or sent to
        url: Option<String>,
    },
    /// The sample was carved from an unknown or unspecified file type
    Unknown,
}

/// The different origin relationships for files
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "api", schema(example = json!(
    {
        "Unpacked": {
            "tool": "unzip",
            "parent": "c1d1b8bdc9197d38c6f392dd58d6bf7200a80a84e45af5eab91dbec3376ddd85",
            "dangling": false
        }
    }
)))]
pub enum Origin {
    /// This sample was downloaded from an external website
    Downloaded { url: String, name: Option<String> },
    /// This sample was unpacked from another sample
    Unpacked {
        /// The tool that unpacked this sample
        tool: Option<String>,
        /// The sha256 of the sample this was unpacked from
        parent: String,
        /// Whether this parent sample exists or not
        #[serde(default)]
        dangling: bool,
    },
    /// This sample is an output of a transformation of another sample
    Transformed {
        /// The tool that transformed this sample
        tool: Option<String>,
        /// The sha256 of the sample that was transformed
        parent: String,
        /// Whether this parent sample exists or not
        #[serde(default)]
        dangling: bool,
        /// The flags used to transform this sample
        flags: Vec<String>,
        /// The full command used to transform this sample
        cmd: Option<String>,
    },
    /// This sample comes from a sniffer/the wire
    Wire {
        /// The name of the sniffer that found this sample
        sniffer: String,
        /// The source of this sample on the wire
        source: Option<String>,
        /// The destination of this sample on the wire
        destination: Option<String>,
    },
    /// This sample cames from an incident or engagement
    Incident {
        /// The incident this sample comes from
        incident: String,
        /// The cover term used for this incident
        cover_term: Option<String>,
        /// The mission team involved in this incident
        mission_team: Option<String>,
        /// The network this sample was found on
        network: Option<String>,
        /// The machine this sample was found on
        machine: Option<String>,
        /// The location this sample is from
        location: Option<String>,
    },
    /// This sample comes from dumping memory while running a parent sample
    MemoryDump {
        /// The sample this file was a memory dump from
        parent: String,
        /// Whether this parent sample exists or not
        #[serde(default)]
        dangling: bool,
        /// The characteristics that were reconstructed in this memory dump
        reconstructed: Vec<String>,
        /// the base address for this memory dump
        base_addr: Option<String>,
    },
    /// This sample was built from source
    Source {
        /// The repository this was built from
        repo: String,
        /// The branch, commit, tag that this was built from
        commitish: Option<String>,
        /// The commit the repository was on
        commit: String,
        /// The flags used to build this
        flags: Vec<String>,
        /// The build system that was used to build this
        system: String,
        /// Whether this is a supporting build file or a final build file
        supporting: bool,
    },
    /// This sample was statically carved out from another sample
    ///
    /// Unlike `Unpacked`, `Carved` describes a sample that is just
    /// a simple piece of another file, like a file from an archive or
    /// a packet capture. It's extraction can be easily replicated without
    /// any dynamic unpacking process.
    Carved {
        /// The sample this file was carved from
        parent: String,
        /// The tool that carved out this sample
        tool: Option<String>,
        /// Whether this parent sample exists or not
        #[serde(default)]
        dangling: bool,
        /// The type of carved file this is
        carved_origin: CarvedOrigin,
    },
    /// This sample has no unique origin
    None,
}

/// add an origin value to tags
macro_rules! tag {
    ($tags:expr, $key:expr, $value:expr) => {
        $tags.entry($key.to_owned()).or_default().insert($value);
    };
}

/// Optionally add an origin value to tags
macro_rules! opt_tag {
    ($tags:expr, $key:expr, $value:expr) => {
        if let Some(value) = $value {
            $tags.entry($key.to_owned()).or_default().insert(value);
        }
    };
}

impl Origin {
    // get the tags to set for this origin
    pub fn get_tags(self, tags: &mut HashMap<String, HashSet<String>>) {
        // add the correct tags
        match self {
            Origin::Downloaded { url, name } => {
                tag!(tags, "Origin", "Downloaded".to_owned());
                tag!(tags, "DownloadedUrl", url);
                opt_tag!(tags, "DownloadName", name);
            }
            Origin::Unpacked { tool, parent, .. } => {
                tag!(tags, "Origin", "Unpacked".to_owned());
                opt_tag!(tags, "UnpackedTool", tool);
                tag!(tags, "Parent", parent);
            }
            Origin::Transformed {
                tool,
                parent,
                flags,
                cmd,
                ..
            } => {
                tag!(tags, "Origin", "Transformed".to_owned());
                opt_tag!(tags, "TransformedWith".to_owned(), tool);
                tag!(tags, "Parent", parent);
                // if any flags are set then tag them
                if !flags.is_empty() {
                    // get an entry to the flag tag values
                    let entry = tags.entry("TransformedFlags".to_owned()).or_default();
                    // add our flag values
                    entry.extend(flags);
                }
                opt_tag!(tags, "TransformedCmd".to_owned(), cmd);
            }
            Origin::Wire {
                sniffer,
                source,
                destination,
            } => {
                tag!(tags, "Origin", "Wire".to_owned());
                tag!(tags, "WireSniffer", sniffer);
                opt_tag!(tags, "WireSource".to_owned(), source);
                opt_tag!(tags, "WireDestination".to_owned(), destination);
            }
            Origin::Incident {
                incident,
                cover_term,
                mission_team,
                network,
                machine,
                location,
            } => {
                tag!(tags, "Origin", "Incident".to_owned());
                tag!(tags, "Incident", incident);
                opt_tag!(tags, "CoverTerm".to_owned(), cover_term);
                opt_tag!(tags, "MissionTeam".to_owned(), mission_team);
                opt_tag!(tags, "IncidentNetwork".to_owned(), network);
                opt_tag!(tags, "IncidentMachine".to_owned(), machine);
                opt_tag!(tags, "IncidentLocation".to_owned(), location);
            }
            Origin::MemoryDump {
                parent,
                reconstructed,
                base_addr,
                ..
            } => {
                tag!(tags, "Origin", "MemoryDump".to_owned());
                tag!(tags, "Parent", parent);
                // if any reconstruction methods are set then tag them
                if !reconstructed.is_empty() {
                    // get an entry to the reconstruction methods tag values
                    let entry = tags
                        .entry("MemoryDumpReconstructed".to_owned())
                        .or_default();
                    // add our reconstruction method values
                    entry.extend(reconstructed);
                }
                opt_tag!(tags, "MemoryDumpBaseAddr".to_owned(), base_addr);
            }
            Origin::Source {
                repo,
                commitish,
                commit,
                flags,
                system,
                supporting,
            } => {
                tag!(tags, "Origin", "Source".to_owned());
                tag!(tags, "Repo", repo);
                opt_tag!(tags, "Commitish", commitish);
                tag!(tags, "Commit", commit);
                // if any build flags are set then tag them
                if !flags.is_empty() {
                    // get an entry to the build flag tag values
                    let entry = tags.entry("BuildFlags".to_owned()).or_default();
                    // add our build flags values
                    entry.extend(flags);
                }
                tag!(tags, "BuildSystem", system);
                // add a flag denoting if this is a supporting file or not
                if supporting {
                    tag!(tags, "BuildSupporting", "True".to_owned());
                } else {
                    tag!(tags, "BuildSupporting", "False".to_owned());
                }
            }
            Origin::None => (),
            Origin::Carved {
                parent,
                tool,
                carved_origin,
                ..
            } => {
                tag!(tags, "Origin", "Carved".to_string());
                tag!(tags, "Parent", parent);
                opt_tag!(tags, "CarvedTool", tool);
                match carved_origin {
                    CarvedOrigin::Pcap {
                        src_ip,
                        dest_ip,
                        src_port,
                        dest_port,
                        proto,
                        url,
                    } => {
                        tag!(tags, "CarvedOrigin", "Pcap".to_string());
                        opt_tag!(tags, "SrcIp", src_ip.map(|v| v.to_string()));
                        opt_tag!(tags, "DestIp", dest_ip.map(|v| v.to_string()));
                        opt_tag!(tags, "SrcPort", src_port.map(|v| v.to_string()));
                        opt_tag!(tags, "DestPort", dest_port.map(|v| v.to_string()));
                        opt_tag!(tags, "Proto", proto.map(|v| v.to_string()));
                        opt_tag!(tags, "Url", url);
                    }
                    CarvedOrigin::Unknown => {
                        tag!(tags, "CarvedOrigin", "Unknown".to_string());
                    }
                }
            }
        }
    }

    /// Determine if this origins parent is a specific sha256
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the parent to check
    pub fn is_child_of(&self, sha256: &str) -> bool {
        match self {
            Origin::Downloaded { .. } => false,
            Origin::Unpacked { parent, .. } => parent == sha256,
            Origin::Transformed { parent, .. } => parent == sha256,
            Origin::Wire { .. } => false,
            Origin::Incident { .. } => false,
            Origin::MemoryDump { parent, .. } => parent == sha256,
            Origin::Source { .. } => false,
            Origin::Carved { parent, .. } => parent == sha256,
            Origin::None => false,
        }
    }
}

impl PartialEq<Option<OriginRequest>> for Origin {
    /// Check if a [`ImageRequest`] and a [`Image`] are equal
    ///
    /// # Arguments
    ///
    /// * `req` - The ImageRequest to compare against
    fn eq(&self, req: &Option<OriginRequest>) -> bool {
        // if an origin request was set then make sure our origin is set
        if let Some(req) = req {
            // match against the right type of origin
            match self {
                Origin::Downloaded { url, name } => {
                    same!(Some(url), req.url.as_ref());
                    same!(name, &req.name);
                }
                Origin::Unpacked { tool, parent, .. } => {
                    same!(tool, &req.tool);
                    same!(Some(parent), req.parent.as_ref());
                }
                Origin::Transformed {
                    tool,
                    parent,
                    flags,
                    cmd,
                    ..
                } => {
                    same!(tool, &req.tool);
                    same!(Some(parent), req.parent.as_ref());
                    same!(flags, &req.flags);
                    same!(cmd, &req.cmd);
                }
                Origin::Wire {
                    sniffer,
                    source,
                    destination,
                } => {
                    same!(Some(sniffer), req.sniffer.as_ref());
                    same!(source, &req.source);
                    same!(destination, &req.destination);
                }
                Origin::Incident {
                    incident,
                    cover_term,
                    mission_team,
                    network,
                    machine,
                    location,
                } => {
                    same!(Some(incident), req.incident.as_ref());
                    same!(cover_term, &req.cover_term);
                    same!(mission_team, &req.mission_team);
                    same!(network, &req.network);
                    same!(machine, &req.machine);
                    same!(location, &req.location);
                }
                Origin::MemoryDump {
                    parent,
                    reconstructed,
                    base_addr,
                    ..
                } => {
                    same!(Some(parent), req.parent.as_ref());
                    same!(reconstructed, &req.reconstructed);
                    same!(base_addr, &req.base_addr);
                }
                Origin::Source {
                    repo,
                    commitish,
                    commit,
                    flags,
                    system,
                    supporting,
                } => {
                    same!(Some(repo), req.repo.as_ref());
                    same!(commitish, &req.commitish);
                    same!(Some(commit), req.commit.as_ref());
                    same!(flags, &req.flags);
                    same!(Some(system), req.system.as_ref());
                    same!(Some(supporting), req.supporting.as_ref());
                }
                Origin::Carved {
                    parent,
                    tool,
                    carved_origin,
                    ..
                } => {
                    same!(Some(parent), req.parent.as_ref());
                    same!(tool, &req.tool);
                    match carved_origin {
                        CarvedOrigin::Pcap {
                            src_ip,
                            dest_ip,
                            src_port,
                            dest_port,
                            proto,
                            url,
                        } => {
                            same!(src_ip, &req.src_ip);
                            same!(dest_ip, &req.dest_ip);
                            same!(src_port, &req.src_port);
                            same!(dest_port, &req.dest_port);
                            same!(proto, &req.proto);
                            same!(url, &req.url);
                        }
                        CarvedOrigin::Unknown => (),
                    }
                }
                Origin::None => {
                    same!(true, req.clear_origin);
                }
            }
            true
        } else {
            // make sure that origin is none
            matches!(self, Origin::None)
        }
    }
}

/// A sample uploaded by a user to Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Submission {
    /// The sha256 of this sample
    pub sha256: String,
    /// The sha1 of this sample
    pub sha1: String,
    /// The md5 of this sample
    pub md5: String,
    /// A UUID for this submission
    pub id: Uuid,
    /// The name of this sample if one was specified
    pub name: Option<String>,
    /// A description for this sample
    pub description: Option<String>,
    /// The groups this submission is apart of
    pub groups: Vec<String>,
    /// The user who submitted this sample
    pub submitter: String,
    /// Where this sample originates from if anywhere in serial form
    pub origin: Option<String>,
    // When this sample was uploaded
    pub uploaded: DateTime<Utc>,
}

/// Updates a specific submission by submission id
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SubmissionUpdate {
    /// The ID of the submission to update
    pub id: Uuid,
    /// The name of this sample if one was specified
    pub name: Option<String>,
    /// The groups to add to this submission
    #[serde(default)]
    pub add_groups: Vec<String>,
    /// The groups to remove from this submission
    #[serde(default)]
    pub remove_groups: Vec<String>,
    /// The description to update with
    pub description: Option<String>,
    /// The updated origin for this sample
    pub origin: Option<OriginRequest>,
}

impl SubmissionUpdate {
    /// Create a new submission update
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the submission to update
    #[must_use]
    pub fn new(id: Uuid) -> Self {
        SubmissionUpdate {
            id,
            name: None,
            add_groups: Vec::default(),
            remove_groups: Vec::default(),
            description: None,
            origin: None,
        }
    }

    /// update the name of this submission
    ///
    /// # Arguments
    ///
    /// * `name` - The new name to set
    #[must_use]
    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        // convert name to a string and set it to be updated
        self.name = Some(name.into());
        self
    }

    /// update the description of this submission
    ///
    /// # Arguments
    ///
    /// * `description` - The new description to set
    #[must_use]
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        // convert description to a string and set it to be updated
        self.description = Some(description.into());
        self
    }

    /// updates the origin for this submission
    ///
    /// # Arguments
    ///
    /// * `origin` - The new origin to set
    #[must_use]
    pub fn origin(mut self, origin: OriginRequest) -> Self {
        self.origin = Some(origin);
        self
    }
}

impl PartialEq<SubmissionUpdate> for Sample {
    /// Check if a [`SubmissionUpdate`] was correctly applied
    ///
    /// # Arguments
    ///
    /// * `update` - The SampleUpdate to check
    fn eq(&self, update: &SubmissionUpdate) -> bool {
        // find our submission in this sample
        let find = self.submissions.iter().find(|sub| sub.id == update.id);
        // if we found a submission then make sure the update was applied
        if let Some(sub) = find {
            // make sure all updates were applied
            matches_update_opt!(sub.name, update.name);
            matches_update_opt!(sub.description, update.description);
            // check group updates
            matches_adds!(sub.groups, update.add_groups);
            matches_removes!(sub.groups, update.remove_groups);
            // manually check instead of using matches_update since PartialEq is implemented for
            // Option<OriginRequest> and not just OriginRequest
            if update.origin.is_some() {
                same!(sub.origin, update.origin);
            }
            // all updates were applied correctly
            true
        } else {
            // no submission found so update must have failed
            false
        }
    }
}

/// A cut down version of a submission object contain just unique info
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "api", schema(example = json!(
	{
		"id": "f085b909-a4b5-48d6-96da-5ed0b6a2a1e8",
		"name": "NotMalware.exe",
		"description": null,
		"groups": [
			"corn"
		],
		"submitter": "mcarson",
		"uploaded": "2024-01-31T21:24:00.007Z",
		"origin": "None"
	}
)))]
pub struct SubmissionChunk {
    /// A UUID for this submission
    pub id: Uuid,
    /// The name of this sample if one was specified
    pub name: Option<String>,
    /// A description for this sample
    pub description: Option<String>,
    /// The groups this submission is in
    pub groups: Vec<String>,
    /// The user who submitted this sample
    pub submitter: String,
    /// When this sample was uploaded
    pub uploaded: DateTime<Utc>,
    /// The origin of this sample if one was specified
    pub origin: Origin,
}

/// A map of tags for a specific sample or repo
///
/// The format is `HashMap<Key, HashMap<Value, HashSet<Groups>>>`. So if we had an Os tag with a value of
/// Win10 for the Corn and Oranges group then that would look like:
/// ```text
/// HashMap<"Os", HashMap<"Win10">, HashSet<"Corn", "Oranges">>
/// ```
pub type TagMap = HashMap<String, HashMap<String, HashSet<String>>>;

/// A request for a comment about a specific sample
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "api", schema(example = json!(
    {
		"groups": [
			"corn"
		],
		"uploaded": "2024-01-31T21:59:08.412Z",
		"id": "ba788031-3c3a-4e62-a158-71bbce73b25a",
		"author": "mcarson",
		"comment": "This is definitely not malware",
		"attachments": {}
	}
)))]
pub struct Comment {
    /// The groups to share this comment with
    pub groups: Vec<String>,
    /// When this comment was uploaded
    pub uploaded: DateTime<Utc>,
    /// The uuid for this comment
    pub id: Uuid,
    /// The author for this comment
    pub author: String,
    /// The comment for this file
    pub comment: String,
    /// Mappings of file names to their S3 UUID
    pub attachments: HashMap<String, Uuid>,
}

impl PartialEq<CommentRequest> for Comment {
    /// Check if a [`CommentRequest`] and a [`Comment`] are equal
    ///
    /// # Arguments
    ///
    /// * `req` - The CommentRequest to compare against
    fn eq(&self, req: &CommentRequest) -> bool {
        // if the groups were manually set then check them
        if !req.groups.is_empty() {
            same!(self.groups, req.groups);
        }
        // make sure the comment string is correct
        same!(self.comment, req.comment);
        // make sure the on disk file info matches
        for on_disk in req.files.iter() {
            // build the path to check for
            let path = match &on_disk.trim_prefix {
                Some(trim) => match on_disk.path.strip_prefix(trim) {
                    Ok(stripped) => stripped,
                    Err(_) => return false,
                },
                None => on_disk.path.as_path(),
            };
            // make sure this path is in our comment
            if !self
                .attachments
                .contains_key(&path.to_string_lossy().to_string())
            {
                return false;
            }
        }
        true
    }
}

/// A response from commenting on a file
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CommentResponse {
    /// The id for this comment
    pub id: Uuid,
}

/// An external combined view of all submissions rows a user can see
///
/// User will largely only know/use sample structs over Submission.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "api", schema(example = json!({
	"sha256": "a08a865e1b926ca5bbf5d6ee9e75d6e5cb11ee834a1397296ea9186f4d7331d8",
	"sha1": "cf7b6f343ac2e89cabbfbe70670aca19ac8b8be6",
	"md5": "bfdee9bf6aec2099f90b97957a54a7fe",
	"tags": {
		"FileSize": {
			"2 MB": [
				"corn",
			]
		},
		"Format": {
			"Win32 EXE": [
				"corn"
			]
		},
		"submitter": {
			"mcarson": [
				"corn"
			]
		}
	},
	"submissions": [
		{
			"id": "f085b909-a4b5-48d6-96da-5ed0b6a2a1e8",
			"name": "NotMalware.exe",
			"description": null,
			"groups": [
				"corn"
			],
			"submitter": "mcarson",
			"uploaded": "2024-01-31T21:24:00.007Z",
			"origin": "None"
		}
	],
	"comments": [
        {
			"groups": [
				"corn"
			],
			"uploaded": "2024-01-31T21:59:08.412Z",
			"id": "ba788031-3c3a-4e62-a158-71bbce73b25a",
			"author": "mcarson",
			"comment": "This is definitely not malware",
			"attachments": {}
		}
    ]
})))]
pub struct Sample {
    /// The sha256 of this sample
    pub sha256: String,
    /// The sha1 of this sample
    pub sha1: String,
    /// The md5 of this sample
    pub md5: String,
    /// The tags for this sample
    #[cfg_attr(feature = "api", schema(
        value_type = HashMap<String, HashMap<String, Vec<String>>>,
        example = json!({
			"groups": [
				"devops-testing",
				"testing"
			],
			"uploaded": "2024-06-18T21:59:08.412Z",
			"id": "ba788031-3c3a-4e62-a158-71bbce73b25a",
			"author": "mcarson",
			"comment": "this is a picture",
			"attachments": {}
		})
    ))]
    pub tags: TagMap,
    /// The different submissions for this sample
    pub submissions: Vec<SubmissionChunk>,
    /// Any comments for this sample
    pub comments: Vec<Comment>,
}

impl Sample {
    /// get the groups this sample is apart of
    #[must_use]
    pub fn groups(&self) -> HashSet<&str> {
        // crawl over the groups and build a deduped list of groups this sample is in
        let mut groups = HashSet::default();
        for sub in self.submissions.iter() {
            // crawl over the groups for this submission
            for group in sub.groups.iter() {
                // if this group is not already in our list then add it
                if !groups.contains(group.as_str()) {
                    groups.insert(group.as_str());
                }
            }
        }
        groups
    }

    /// Get the earliest time each group saw this sample
    #[must_use]
    pub fn earliest_owned(self) -> HashMap<String, DateTime<Utc>> {
        // get the earliest timestamp for each group that we can see this sample was submitted at
        let mut earliest = HashMap::with_capacity(3);
        // crawl over each submission and map the earliest time each group saw this sample
        for sub in self.submissions.into_iter().rev() {
            // crawl over each group in this submission
            for group in sub.groups {
                // if this group isn't already in our map then add its upload time
                earliest.entry(group).or_insert(sub.uploaded);
            }
        }
        earliest
    }

    /// Simplify this samples tag map to just key/values (no group info)
    pub fn simple_tags(&self) -> HashMap<&String, Vec<&String>> {
        // init our hashmap to be the correct size
        let mut simple = HashMap::with_capacity(self.tags.len());
        // crawl and add our tags
        for (key, value_map) in self.tags.iter() {
            // build a vec of our values
            let values = value_map.keys().collect::<Vec<&String>>();
            // insert our values
            simple.insert(key, values);
        }
        simple
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl KeySupport for Sample {
    /// The full key for this tag request
    type Key = String;

    /// The extra info stored in our tag request that gets added to our key
    type ExtraKey = ();

    /// Build the key to use as part of the partition key when storing this data in scylla
    ///
    /// # Arguments
    ///
    /// * `key` - The root part of this key
    /// * `_extra` - Any extra info required to build this key
    fn build_key(key: Self::Key, _extra: &Self::ExtraKey) -> String {
        key
    }

    /// Build a URL component composed of the key to access the resource
    ///
    /// # Arguments
    ///
    /// * `key` - The root part of this key
    /// * `extra` - Any extra info required to build this key
    fn key_url(key: &Self::Key, _extra: Option<&Self::ExtraKey>) -> String {
        // our key is just a String, so return that
        key.clone()
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl TagSupport for Sample {
    /// Get the tag kind to write to the DB
    fn tag_kind() -> TagType {
        TagType::Files
    }

    /// Get the earliest each group has seen this object
    fn earliest(&self) -> HashMap<&String, DateTime<Utc>> {
        // get the earliest timestamp for each group that we can see this sample was submitted at
        let mut earliest = HashMap::with_capacity(3);
        // crawl over each submission and map the earliest time each group saw this sample
        for sub in self.submissions.iter().rev() {
            // crawl over each group in this submission
            for group in sub.groups.iter() {
                // if this group isn't already in our map then add its upload time
                earliest.entry(group).or_insert(sub.uploaded);
            }
        }
        earliest
    }

    /// Add some tags to a sample
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating tags
    /// * `req` - The tag request to apply
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[tracing::instrument(name = "TagSupport<Sample>::tag", skip_all, fields(sha256 = self.sha256), err(Debug))]
    async fn tag(
        &self,
        user: &User,
        mut req: TagRequest<Sample>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // if groups were supplied then validate this sample is in them otherwise use defaults
        self.validate_check_allow_groups(
            user,
            &mut req.groups,
            super::GroupAllowAction::Tags,
            shared,
        )
        .await?;
        // get the earliest time this sample was uploaded for each group
        let earliest = self.earliest();
        // save our files tags to scylla
        super::backends::db::tags::create(user, self.sha256.clone(), req, &earliest, shared).await
    }

    /// Delete some tags from this sample
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting tags
    /// * `req` - The tags to delete
    /// * `shared` - Shared Thorium objects
    #[tracing::instrument(name = "TagSupport<Sample>::delete_tags", skip_all, fields(sha256 = self.sha256), err(Debug))]
    #[cfg(feature = "api")]
    async fn delete_tags(
        &self,
        user: &User,
        mut req: TagDeleteRequest<Sample>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // if groups were supplied then validate this sample is in them otherwise use defaults
        self.validate_groups(user, &mut req.groups, true, shared)
            .await?;
        // delete the requested tags for this sample if they exist
        super::backends::db::tags::delete(&self.sha256, &req, shared).await
    }

    /// Gets tags for a specific sample
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our returned tags too
    /// * `shared` - Shared Thorium objects
    #[tracing::instrument(name = "TagSupport<Sample>::get_tags", skip_all, fields(sha256 = self.sha256), err(Debug))]
    #[cfg(feature = "api")]
    async fn get_tags(&mut self, groups: &Vec<String>, shared: &Shared) -> Result<(), ApiError> {
        // get the requested tags
        super::backends::db::tags::get(TagType::Files, groups, &self.sha256, &mut self.tags, shared)
            .await
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl PartialEq<TagRequest<Sample>> for Sample {
    /// Check if a [`TagRequest`] was added to a [`Sample`]
    ///
    /// # Arguments
    ///
    /// * `req` - The `TagRequest` to compare against
    fn eq(&self, req: &TagRequest<Sample>) -> bool {
        // crawl over the tags we requested to be added
        for (key, values) in &req.tags {
            // make sure each tag was added
            if let Some(updated) = self.tags.get(key) {
                // crawl over the values we wanted to be added
                for value in values {
                    // make sure that each value was added
                    if let Some(groups) = updated.get(value) {
                        // make sure that the groups were added
                        matches_adds!(groups, req.groups);
                    } else {
                        // this value wasn't added so return false
                        return false;
                    }
                }
            } else {
                // this key wasn't added so return false
                return false;
            }
        }
        // the tag request was successfully completed
        true
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl OutputSupport for Sample {
    /// Get the tag kind to write to the DB
    fn output_kind() -> OutputKind {
        OutputKind::Files
    }

    /// Build a tag request for this output kind
    fn tag_req() -> TagRequest<Self> {
        TagRequest::<Sample>::default()
    }

    /// get our extra info
    ///
    /// # Arguments
    ///
    /// `extra` - The extra field to extract
    fn extract_extra(_: Option<Self::ExtraKey>) -> Self::ExtraKey {}

    /// Ensures any user requested groups are valid for this result.
    ///
    /// If no groups are specified then all groups we can see this object in will be returned.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is validating this object is in some groups
    /// * `groups` - The user specified groups to check against
    /// * `shared` - Shared objects in Thorium
    #[cfg(feature = "api")]
    async fn validate_groups_viewable(
        &self,
        user: &crate::models::User,
        groups: &mut Vec<String>,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // validate this objects groups
        self.validate_groups(user, groups, false, shared).await?;
        Ok(())
    }

    /// Ensures any user requested groups are valid and editable for this result.
    ///
    /// If no groups are specified then all groups we can see/edit this object in will be returned.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is validating this object is in some groups
    /// * `groups` - The user specified groups to check against
    /// * `shared` - Shared objects in Thorium
    #[cfg(feature = "api")]
    async fn validate_groups_editable(
        &self,
        user: &crate::models::User,
        groups: &mut Vec<String>,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // validate this objects groups
        self.validate_check_allow_groups(user, groups, super::GroupAllowAction::Results, shared)
            .await?;
        Ok(())
    }
}

impl PartialEq<SampleRequest> for Sample {
    /// Check if a [`SampleRequest`] and a [`Sample`] are equal
    ///
    /// # Arguments
    ///
    /// * `req` - The `SampleRequest` to compare against
    fn eq(&self, req: &SampleRequest) -> bool {
        // crawl over the tags we wish to set and check them
        if req.tags.iter().any(|(key, values)| {
            // make sure this tag was set
            if let Some(value_map) = self.tags.get(key) {
                // make sure our new values and the correct groups were set
                values.iter().all(|value| {
                    // if our value wasn't set then return false
                    if let Some(groups) = value_map.get(value) {
                        // make sure the correct groups were set
                        req.groups.iter().all(|item| groups.contains(item))
                    } else {
                        // the key exists but our value was not set
                        false
                    }
                })
            } else {
                // the new key does not exist in the tag map so
                false
            }
        }) {
            // this tag was not set or the correct groups were not set
            return false;
        }
        // find our submission in this sample
        // this is a bit unreliable as we don't have our username in the request
        self.submissions.iter().any(|sub| {
            sub.description == req.description
                && req.groups.iter().all(|group| sub.groups.contains(group))
                && sub.origin == req.origin
        })
    }
}

impl TreeSupport for Sample {
    /// Hash this child object
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed to set the hasher to use
    fn tree_hash(&self, seed: i64) -> u64 {
        // build a hasher
        let mut hasher = gxhash::GxHasher::with_seed(seed);
        // hash this samples sha
        hasher.write(self.sha256.as_bytes());
        // finalize our hasher
        let hash = hasher.finish();
        hash
    }

    /// Gather any initial nodes for a tree
    #[cfg(feature = "api")]
    async fn gather_initial(
        user: &User,
        query: &crate::models::TreeQuery,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<super::TreeNodeData>, crate::utils::ApiError> {
        // build a list of initial data
        let mut initial = Vec::with_capacity(query.samples.len());
        // get all our intial sample data
        // TODO do this in parallel
        for sha256 in &query.samples {
            // get this samples data
            let sample = Sample::get(user, sha256, shared).await?;
            // wrap this node in a tree node data object
            let node_data = super::TreeNodeData::Sample(sample);
            // add this tree node data object to our list
            initial.push(node_data);
        }
        Ok(initial)
    }

    /// Gather any children for this child node
    #[cfg(feature = "api")]
    async fn gather_children(
        &self,
        user: &User,
        shared: &crate::utils::Shared,
    ) -> Result<Vec<super::TreeNode>, crate::utils::ApiError> {
        // build the opts to get everything tagged with this parent hash
        let opts = FileListOpts::default().tag("Parent", &self.sha256);
        // gather the children for this sample
        let sha256s = Sample::list(user, opts, true, shared).await?;
        // convert this to a details list
        let mut cursor = sha256s.details(user, shared).await?;
        // build a list of related children
        let mut children = Vec::with_capacity(cursor.data.len());
        // wrap these samples in a tree node
        for sample in cursor.data.drain(..) {
            // get the origins that are related to our parent
            let relationships = sample
                .submissions
                .iter()
                .filter(|sub| sub.origin.is_child_of(&self.sha256))
                .map(|sub| super::TreeRelationships::Origin(sub.origin.clone()))
                .collect();
            // wrap this sample in a node data object
            let data = super::TreeNodeData::Sample(sample);
            // build the tree node for this child sample
            let node = super::TreeNode::new(relationships, data);
            // add this to our list of children nodes
            children.push(node);
        }
        Ok(children)
    }
}

#[derive(Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DeleteSampleParams {
    /// The groups to delete data from
    #[serde(default)]
    pub groups: Vec<String>,
}

/// The options used when downloading files
#[derive(Debug, Default)]
pub struct FileDownloadOpts {
    /// Whether this file uncarted while downloading
    pub uncart: bool,
    /// The progress bar to update
    pub progress: Option<ProgressBar>,
}

impl FileDownloadOpts {
    /// Uncart this file during the download
    pub fn uncart(mut self) -> Self {
        // set this file to be uncarted as its streamed to disk
        self.uncart = true;
        self
    }

    /// Set whether to uncart this file during the download
    ///
    /// # Arguments
    ///
    /// * `uncart` - Whether to uncart this file or not
    pub fn uncart_by_value(mut self, uncart: bool) -> Self {
        // set this file to be uncarted as its streamed to disk
        self.uncart = uncart;
        self
    }

    /// Add a progress to update with our download progress
    pub fn progress(mut self, progress: ProgressBar) -> Self {
        self.progress = Some(progress);
        self
    }
}

/// The carted data for a sample
#[derive(Debug, Clone)]
pub struct CartedSample {
    /// The path to the carted sample
    pub path: PathBuf,
}

#[cfg(feature = "client")]
impl CartedSample {
    /// Uncarts a sample into its unpacked bytes
    ///
    /// # Arguments
    ///
    /// * `path` - The path to uncart this file to
    pub async fn uncart<P: AsRef<Path>>(self, path: P) -> Result<(), Error> {
        // open our file
        let file = File::open(&self.path).await?;
        // get an async buf reader for our file
        let reader = BufReader::new(file);
        // start uncarting this stream of data
        let mut uncart = UncartStream::new(reader);
        // make a file to save the response too
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .await?;
        // write our uncarted stream to disk
        tokio::io::copy(&mut uncart, &mut file).await?;
        Ok(())
    }
}

/// An uncarted sample
#[cfg(feature = "client")]
#[derive(Debug)]
pub struct UncartedSample {
    /// The file handle containing our uncarted sample
    pub file: File,
}

/// A downloaded sample
pub enum DownloadedSample {
    /// An on disk sample that is still carted
    Carted(CartedSample),
    /// An on disk sample that is uncarted
    Uncarted(UncartedSample),
}

impl DownloadedSample {
    /// Uncarts this sample if it is carted to a target path
    ///
    /// Trying to uncart an already carted file will return an error
    ///
    /// # Arguments
    ///
    /// * `path` - The path to uncart this file too
    #[allow(dead_code)]
    async fn uncart_to<P: AsRef<Path>>(self, path: P) -> Result<Self, Error> {
        match self {
            DownloadedSample::Carted(carted) => {
                // uncart our sample
                carted.uncart(&path).await?;
                // get a handle to our newly uncarted file
                let file = File::open(path).await?;
                // return an uncarted version of this enum
                Ok(Self::Uncarted(UncartedSample { file }))
            }
            DownloadedSample::Uncarted(_) => Err(Error::new("Already uncarted")),
        }
    }

    /// Uncart this file in place to its current location
    ///
    /// This will temporarily create a new hidden file. Doing this on an already
    /// uncarted file is simply a no op.
    #[allow(dead_code)]
    async fn uncart(self) -> Result<Self, Error> {
        match self {
            DownloadedSample::Carted(carted) => {
                // get the current path to this sample
                let original = carted.path.clone();
                // build a hidden file name (on linux at least for now)
                let hidden = match original.file_name() {
                    Some(name) => {
                        // start with just a period as our name
                        let mut hidden = OsString::from(".");
                        // add the rest of our filename to make a hidden file on linux
                        // on windows it will just start with a period
                        hidden.push(name);
                        hidden
                    }
                    None => {
                        // return an error if we don't have a filename
                        return Err(Error::new(format!(
                            "File '{}' is missing a filename!",
                            original.to_string_lossy()
                        )));
                    }
                };
                // uncart our file to our hidden target
                carted.uncart(&hidden).await?;
                // move our uncarted file to our original path
                tokio::fs::rename(hidden, &original).await?;
                // get a file handle to our newly uncarted file
                let file = File::open(original).await?;
                Ok(Self::Uncarted(UncartedSample { file }))
            }
            DownloadedSample::Uncarted(uncarted) => Ok(Self::Uncarted(uncarted)),
        }
    }
}

/// An attachment to a result or comment
#[derive(Debug, Clone)]
pub struct Attachment {
    /// The attachment in bytes
    pub data: Bytes,
}

/// Default the file list limit to 50
fn default_list_limit() -> usize {
    50
}

#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct FileListParams {
    /// The groups to list data from
    #[serde(default)]
    pub groups: Vec<String>,
    /// When to start listing data at
    #[serde(default = "Utc::now")]
    pub start: DateTime<Utc>,
    /// When to stop listing data at
    pub end: Option<DateTime<Utc>>,
    /// The tags to filter on
    #[serde(default)]
    pub tags: HashMap<String, Vec<String>>,
    /// The cursor id to use if one exists
    pub cursor: Option<Uuid>,
    /// The max number of items to return in this response
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    /// Whether matching on tags should be case-insensitive
    pub tags_case_insensitive: bool,
}

impl Default for FileListParams {
    /// Create a default file list params
    fn default() -> Self {
        FileListParams {
            groups: Vec::default(),
            start: Utc::now(),
            end: None,
            tags: HashMap::default(),
            cursor: None,
            limit: default_list_limit(),
            tags_case_insensitive: false,
        }
    }
}

impl From<FileListOpts> for FileListParams {
    /// Convert `FileListOpts` into `FileListParams`
    fn from(opts: FileListOpts) -> Self {
        FileListParams {
            groups: opts.groups,
            start: opts.start.unwrap_or_else(|| Utc::now()),
            end: opts.end,
            tags: opts.tags,
            cursor: opts.cursor,
            limit: opts.limit.unwrap_or_else(|| default_list_limit()),
            tags_case_insensitive: opts.tags_case_insensitive,
        }
    }
}

impl FileListParams {
    /// Get the end timestamp or get a sane default
    #[cfg(feature = "api")]
    pub fn end(
        &self,
        shared: &crate::utils::Shared,
    ) -> Result<DateTime<Utc>, crate::utils::ApiError> {
        match self.end {
            Some(end) => Ok(end),
            None => match Utc.timestamp_opt(shared.config.thorium.files.earliest, 0) {
                chrono::LocalResult::Single(default_end) => Ok(default_end),
                _ => crate::internal_err!(format!(
                    "default earliest files timestamp is invalid or ambiguous - {}",
                    shared.config.thorium.files.earliest
                )),
            },
        }
    }
}

// A single sample submission line
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SampleListLine {
    /// The group this submission was apart of (used only for cursor generation)
    #[serde(skip_serializing, skip_deserializing)]
    pub groups: HashSet<String>,
    /// The sha256 of this sample
    pub sha256: String,
    /// The submission ID for this instance of this sample if is exposed by the listing op used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submission: Option<Uuid>,
    /// The timestamp this was last uploaded
    pub uploaded: DateTime<Utc>,
}

/// A request to add a comment to a sample
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CommentRequest {
    /// The SHA256 of the sample to comment on
    pub sha256: String,
    /// The groups this sample is a part of
    pub groups: Vec<String>,
    /// A description for this sample
    pub comment: String,
    /// The path to the file to upload if attachments are on disk
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<OnDiskFile>,
    /// The attachments to upload directly
    pub buffers: Vec<Buffer>,
}

impl CommentRequest {
    /// Creates a new request for a comment
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sample to comment on
    /// * `comment` - The comment to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::CommentRequest;
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment");
    /// ```
    pub fn new<S: Into<String>, C: Into<String>>(sha256: S, comment: C) -> Self {
        CommentRequest {
            sha256: sha256.into(),
            groups: Vec::default(),
            comment: comment.into(),
            files: Vec::default(),
            buffers: Vec::default(),
        }
    }

    /// Adds a single group to this comment request
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::CommentRequest;
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment")
    ///     .group("corn");
    /// ```
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        // convert this group to a string and set it
        self.groups.push(group.into());
        self
    }

    /// Adds multiple groups to this comment request
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to expose this comment to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::CommentRequest;
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment")
    ///     .groups(vec!("corn", "tacos"));
    /// ```
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // convert these groups  to strings and add them
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// Adds a path to an attachment to upload with this comment
    ///
    /// # Arguments
    ///
    /// * `path` - The path to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CommentRequest, OnDiskFile};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment")
    ///     .file(OnDiskFile::new("/corn/bushel1"));
    /// ```
    #[must_use]
    pub fn file(mut self, file: OnDiskFile) -> Self {
        // add our on disk file
        self.files.push(file);
        self
    }

    /// The files to
    ///
    /// # Arguments
    ///
    /// * `files` - The files to upload in this comment
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CommentRequest, OnDiskFile};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment")
    ///     .files(vec!(OnDiskFile::new("/corn/bushel1"), OnDiskFile::new("/corn/bushel2")));
    /// ```
    #[must_use]
    pub fn files(mut self, mut files: Vec<OnDiskFile>) -> Self {
        // add these new on disk files
        self.files.append(&mut files);
        self
    }

    /// Adds a buffer to upload with this comment as an attachment
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CommentRequest, Buffer};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment")
    ///     .buffer(Buffer::new("buffer").name("buffer.txt"));
    /// ```
    #[must_use]
    pub fn buffer(mut self, buffer: Buffer) -> Self {
        // add our buffer to our list of buffers to upload
        self.buffers.push(buffer);
        self
    }

    /// Adds multiple buffers to upload with this comment as attachments
    ///
    /// # Arguments
    ///
    /// * `buffers` - The buffers to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CommentRequest, Buffer};
    ///
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// let req = CommentRequest::new(sha256, "I am a comment")
    ///     .buffers(vec!(Buffer::new("buffer0"), Buffer::new("buffer1")));
    /// ```
    #[must_use]
    pub fn buffers(mut self, mut buffers: Vec<Buffer>) -> Self {
        // append our new buffers on to our list of buffers to upload
        self.buffers.append(&mut buffers);
        self
    }

    /// Create a multipart form from this comment request
    #[cfg(feature = "client")]
    pub async fn to_form(mut self) -> Result<reqwest::multipart::Form, Error> {
        // build the form we are going to send
        let form = reqwest::multipart::Form::new()
            // the tool that created this result
            .text("comment", self.comment);
        // add the groups to share this result with
        let mut form = multipart_list!(form, "groups", self.groups);
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
}

/// The query parameters used when deleting a comment
#[derive(Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DeleteCommentParams {
    /// The groups to delete the comment from
    /// (if empty, the comment will be deleted from all groups)
    #[serde(default)]
    pub groups: Vec<String>,
}

impl DeleteCommentParams {
    /// Adds a single group to the delete comment parameters
    ///
    /// # Arguments
    ///
    /// * `group` - The group to delete a comment from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DeleteCommentParams;
    ///
    /// let params = DeleteCommentParams::default().group("corn");
    /// ```
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        // convert this group to a string and set it
        self.groups.push(group.into());
        self
    }

    /// Adds multiple groups to the delete comment parameters
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to delete a comment from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DeleteCommentParams;
    ///
    /// let params = DeleteCommentParams::default().groups(vec!["corn", "tacos"]);
    /// ```
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // convert these groups  to strings and add them
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }
}

/// The options that you can set when listing files in Thorium
///
/// Currently this only supports single tag queries but when ES support is added multi tag queries
/// will be supported.
#[derive(Debug, Clone)]
pub struct FileListOpts {
    /// The cursor to use to continue this search
    pub cursor: Option<Uuid>,
    /// The latest date to start listing samples from
    pub start: Option<DateTime<Utc>>,
    /// The oldest date to stop listing samples from
    pub end: Option<DateTime<Utc>>,
    /// The max number of objects to retrieve on a single page
    pub page_size: usize,
    /// The total number of objects to return with this cursor
    pub limit: Option<usize>,
    /// The groups limit our search to
    pub groups: Vec<String>,
    /// The tags to filter on
    pub tags: HashMap<String, Vec<String>>,
    /// Whether matching on tags should be case-insensitive
    pub tags_case_insensitive: bool,
}

impl Default for FileListOpts {
    /// Build a default search
    fn default() -> Self {
        FileListOpts {
            start: None,
            cursor: None,
            end: None,
            page_size: 50,
            limit: None,
            groups: Vec::default(),
            tags: HashMap::default(),
            tags_case_insensitive: false,
        }
    }
}

impl FileListOpts {
    /// Restrict the file search to start at a specific date
    ///
    /// # Arguments
    ///
    /// * `start` - The date to start listing samples from
    #[must_use]
    pub fn start(mut self, start: DateTime<Utc>) -> Self {
        // set the date to start listing files at
        self.start = Some(start);
        self
    }

    /// Set the cursor to use when continuing this search
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor id to use for this search
    #[must_use]
    pub fn cursor(mut self, cursor: Uuid) -> Self {
        // set cursor for this search
        self.cursor = Some(cursor);
        self
    }

    /// Restrict the file search to stop at a specific date
    ///
    /// # Arguments
    ///
    /// * `end` - The date to stop listing samples at
    #[must_use]
    pub fn end(mut self, end: DateTime<Utc>) -> Self {
        // set the date to end listing files at
        self.end = Some(end);
        self
    }

    /// The max number of objects to retrieve in a single page
    ///
    /// # Arguments
    ///
    /// * `page_size` - The max number of documents to return in a single request
    #[must_use]
    pub fn page_size(mut self, page_size: usize) -> Self {
        // set the date to end listing files at
        self.page_size = page_size;
        self
    }

    /// Limit how many samples this search can return at once
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of objects to return over the lifetime of this cursor
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        // set the date to end listing files at
        self.limit = Some(limit);
        self
    }

    /// Limit what groups we search in
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our search to
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // set the date to end listing files at
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// List files that match a specific tag
    ///
    /// # Arguments
    ///
    /// * `key` - The tag key to match against
    /// * `value` - The tag value to match against
    #[must_use]
    pub fn tag<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        // get an entry into this tags value list
        let entry = self.tags.entry(key.into()).or_default();
        // add this tags value
        entry.push(value.into());
        self
    }

    /// List files that match a specific tag by ref
    ///
    /// # Arguments
    ///
    /// * `key` - The tag key to match against
    /// * `value` - The tag value to match against
    pub fn tag_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        // get an entry into this tags value list
        let entry = self.tags.entry(key.into()).or_default();
        // add this tags value
        entry.push(value.into());
    }

    /// Set for matching on tags to be case-insensitive
    #[must_use]
    pub fn tags_case_insensitive(mut self) -> Self {
        self.tags_case_insensitive = true;
        self
    }
}

/// Options for file deletion
#[derive(Debug, Clone, Default)]
pub struct FileDeleteOpts {
    /// The list of groups from which submission access will be removed
    pub groups: Vec<String>,
}

impl FileDeleteOpts {
    /// Adds groups from which submission access will be removed
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to lose access to the submission
    #[must_use]
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }
}

/// The parameters for downloading samples as zips
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ZipDownloadParams {
    /// The password to use to encrypt this zip
    pub password: Option<String>,
}

#[derive(Debug)]
#[cfg_attr(feature = "scylla-utils", derive(scylla::DeserializeRow))]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
pub struct SampleCensusRow {
    pub group: String,
    pub year: i32,
    pub bucket: i32,
    pub count: i64,
}

#[cfg(feature = "scylla-utils")]
impl super::CensusSupport for Sample {
    /// The type returned by our prepared statement
    type Row = SampleCensusRow;

    /// Build the prepared statement for getting partition count info
    async fn scan_prepared_statement(
        scylla: &scylla::client::session::Session,
        ns: &str,
    ) -> Result<scylla::statement::prepared::PreparedStatement, scylla::errors::PrepareError> {
        // build tags get partition count prepared statement
        scylla
            .prepare(format!(
                "SELECT group, year, bucket, count(*) \
                FROM {}.samples_list \
                WHERE token(group, year, bucket) >= ? AND token(group, year, bucket) <= ? \
                GROUP BY group, year, bucket",
                ns,
            ))
            .await
    }

    /// Get the count for this partition
    fn get_count(row: &SampleCensusRow) -> i64 {
        row.count
    }

    /// Get the bucket for this partition
    fn get_bucket(row: &SampleCensusRow) -> i32 {
        row.bucket
    }

    /// Build the count key for this partition
    fn count_key_from_row(namespace: &str, row: &Self::Row, grouping: i32) -> String {
        // build the key for this row
        format!(
            "{namespace}:census:samples:counts:{group}:{year}:{grouping}",
            namespace = namespace,
            group = row.group,
            year = row.year,
            grouping = grouping,
        )
    }

    /// Build the sorted set key for this census operation
    fn stream_key_from_row(namespace: &str, row: &Self::Row) -> String {
        // build the stream key for this row
        format!(
            "{namespace}:census:samples:stream:{group}:{year}",
            namespace = namespace,
            group = row.group,
            year = row.year,
        )
    }
}
