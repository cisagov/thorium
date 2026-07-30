//! The associations for objects in Thorium
//!
//! Associations are directional relationships between two objects in Thorium.

use chrono::prelude::*;
use std::str::FromStr;
use uuid::Uuid;

use crate::models::{Directionality, EntityKinds};

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum AssociationTarget {
    /// This assocation is associated with another entity
    Entity { id: Uuid, name: String },
    /// This association is associated with a file
    File(String),
    /// This association is associated with a repo
    Repo(String),
}

/// The different possible associations
#[derive(Debug, Serialize, Deserialize, Copy, Clone, Eq, PartialEq, Hash, strum::EnumString)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
pub enum AssociationKind {
    /// This file is associated with something else
    FileFor,
    /// This is documentation for something else
    DocumentationFor,
    /// This file or repo is or contains firmware for a device
    FirmwareFor,
    /// This file/repo/entity is associated with something else
    AssociatedWith,
    /// This was developed or created by
    DevelopedBy,
    /// This contains a CVE
    ContainsCVE,
    /// This contains a CWE
    ContainsCWE,
    /// This is based in specific countries
    BasedIn,
    /// This person was or is employed by
    EmployedBy,
    /// This is the parent company of another company
    ParentCompanyOf,
    /// This is used by a specific person or group
    UsedBy,
    /// This was used in a specific campaign or engagement
    UsedIn,
    /// This campaign was performed by
    PerformedBy,
    /// This filesystem was extracted/carved from
    FileSystemIn,
    /// This is a folder within a filesystem or another folder
    FolderIn,
    /// This is a file in a folder in a filesytem
    FileIn,
    /// A Process tree in or from something
    ProcessTreeIn,
    /// A Process in a process tree or a child process
    ChildProcess,
    /// Opens or receives data from a network connection
    HasNetworkConnection,
    /// A PE section within a file/binary
    SectionIn,
    /// A library imported by a file/binary
    ImportIn,
    /// A flag on some data in Thorium
    FlagFor,
    /// This entity was created by
    CreatedBy,
    /// A Sigma rule hit
    SigmaRuleHit,
    /// This was involved in something
    InvolvedIn,
}

impl std::fmt::Display for AssociationKind {
    /// Cleanly print an association kind
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // get the str for our enum variant
        let variant_str = self.as_str();
        // write this variant to our formatter
        write!(f, "{variant_str}")
    }
}

impl AssociationKind {
    /// Cast our AssocationKind to a str
    pub fn as_str(&self) -> &str {
        match self {
            AssociationKind::FileFor => "FileFor",
            AssociationKind::DocumentationFor => "DocumentationFor",
            AssociationKind::FirmwareFor => "FirmwareFor",
            AssociationKind::AssociatedWith => "AssociatedWith",
            AssociationKind::DevelopedBy => "DevelopedBy",
            AssociationKind::ContainsCVE => "ContainsCVE",
            AssociationKind::ContainsCWE => "ContainsCWE",
            AssociationKind::BasedIn => "BasedIn",
            AssociationKind::ParentCompanyOf => "ParentCompanyOf",
            AssociationKind::EmployedBy => "EmployedBy",
            AssociationKind::UsedBy => "UsedBy",
            AssociationKind::UsedIn => "UsedIn",
            AssociationKind::PerformedBy => "PerformedBy",
            AssociationKind::FileSystemIn => "FileSystemIn",
            AssociationKind::FolderIn => "FolderIn",
            AssociationKind::FileIn => "FileIn",
            AssociationKind::ProcessTreeIn => "ProcessTreeIn",
            AssociationKind::ChildProcess => "ChildProcess",
            AssociationKind::HasNetworkConnection => "HasNetworkConnection",
            AssociationKind::SectionIn => "SectionIn",
            AssociationKind::ImportIn => "ImportIn",
            AssociationKind::FlagFor => "FlagFor",
            AssociationKind::CreatedBy => "CreatedBy",
            AssociationKind::SigmaRuleHit => "SigmaRuleHit",
            AssociationKind::InvolvedIn => "InvolvedIn",
        }
    }

    /// The association kind for for an entity back to a sample/repo (not an entity)
    ///
    /// This is not exhaustive and is best effort
    pub fn to_parent_nonentity(kind: EntityKinds) -> Self {
        match kind {
            EntityKinds::WindowsProcessTree => Self::ProcessTreeIn,
            EntityKinds::WindowsProcess => Self::ChildProcess,
            EntityKinds::FileSystem => Self::FileSystemIn,
            EntityKinds::Folder => Self::FolderIn,
            EntityKinds::Flag => Self::FlagFor,
            EntityKinds::NetworkConnection => Self::HasNetworkConnection,
            EntityKinds::PeSection => Self::SectionIn,
            EntityKinds::PeImport => Self::ImportIn,
            _ => Self::AssociatedWith,
        }
    }
}

impl From<(EntityKinds, EntityKinds)> for AssociationKind {
    fn from(kinds: (EntityKinds, EntityKinds)) -> Self {
        match kinds {
            (EntityKinds::WindowsProcessTree, EntityKinds::WindowsProcess)
            | (EntityKinds::WindowsProcess, EntityKinds::WindowsProcess) => Self::ChildProcess,
            (EntityKinds::FileSystem, EntityKinds::Folder) => Self::FolderIn,
            (EntityKinds::SigmaRule, EntityKinds::Flag) => Self::CreatedBy,
            (EntityKinds::Flag, _) => Self::FlagFor,
            (EntityKinds::Vendor, EntityKinds::Device) => Self::DevelopedBy,
            (EntityKinds::Incident, _) | (EntityKinds::Other, _) | (_, EntityKinds::Other) => {
                Self::AssociatedWith
            }
            _ => Self::AssociatedWith,
        }
    }
}

/// The specific submissions tied to each side of an association
///
/// This lets an association point at the exact submission on either side when that side is a file/sample.
/// For example a `FileIn` association can record which submission placed a file in a folder, allowing that
/// submission's file name to be recovered losslessly.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct AssociatedSubmissions {
    /// The submission id for the source side of this association, if it is a file/sample
    pub source: Option<Uuid>,
    /// The submission id for the other side of this association, if it is a file/sample
    pub other: Option<Uuid>,
}

impl AssociatedSubmissions {
    /// Check if this has no submission ids set on either side
    #[must_use]
    pub fn is_empty(&self) -> bool {
        // we are empty if neither side has a submission id
        self.source.is_none() && self.other.is_none()
    }

    /// Get the opposite view of these submissions with source and other swapped
    ///
    /// This is used when storing/reading the inverted (other -> source) row for an association.
    #[must_use]
    pub fn swapped(&self) -> Self {
        AssociatedSubmissions {
            source: self.other,
            other: self.source,
        }
    }
}

/// An association with a specific piece of data
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Association {
    /// The kind of association this is
    pub kind: AssociationKind,
    /// The other data this directional association is with
    pub other: AssociationTarget,
    /// The creator of this association
    pub submitter: String,
    /// The groups for this association
    pub groups: Vec<String>,
    /// When this association was created
    pub created: DateTime<Utc>,
    /// The direction for this association
    pub direction: Directionality,
    /// The specific submissions tied to each side of this association if any are known
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submissions: Option<AssociatedSubmissions>,
}

/// A request to associate one piece of data with another
#[derive(Debug, Serialize, Deserialize)]
pub struct AssociationRequest {
    /// The kind of association to make
    pub kind: AssociationKind,
    /// The piece of data this association starts with
    pub source: AssociationTarget,
    /// The data this association is with
    pub targets: Vec<AssociationTarget>,
    /// The groups for this association
    #[serde(default)]
    pub groups: Vec<String>,
    /// Whether this is a bidirecitonal relationship or not
    pub is_bidirectional: bool,
    /// The specific submissions tied to each side of this association if any are known
    ///
    /// When more than one target is set the `other` submission is applied to every target, so this is only
    /// meaningful for single target requests (the common case for filesystem links).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submissions: Option<AssociatedSubmissions>,
}

impl AssociationRequest {
    /// Create a new association request
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of association request to create
    /// * `source` - Where this assocation comes from
    pub fn new(kind: AssociationKind, source: AssociationTarget) -> Self {
        // build this association request
        AssociationRequest {
            kind,
            source,
            targets: Vec::default(),
            groups: Vec::default(),
            is_bidirectional: false,
            submissions: None,
        }
    }

    /// Create a new association request with a preset capacity
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of association request to create
    /// * `source` - Where this assocation comes from
    /// * `capacity` - The capacity to set
    pub fn with_capacity(
        kind: AssociationKind,
        source: AssociationTarget,
        capacity: usize,
    ) -> Self {
        // build this association request
        AssociationRequest {
            kind,
            source,
            targets: Vec::with_capacity(capacity),
            groups: Vec::with_capacity(capacity),
            is_bidirectional: false,
            submissions: None,
        }
    }

    /// Add a target for this association
    ///
    /// # Arguments
    ///
    /// * `target` - The target for this association
    pub fn target(mut self, target: AssociationTarget) -> Self {
        // add our new target
        self.targets.push(target);
        self
    }

    /// Set the groups to use with this association
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to add to this request
    pub fn groups<I>(mut self, groups: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        // extend our groups with our new groups
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// Set this association to be bidirectional
    pub fn biderectional(mut self) -> Self {
        self.is_bidirectional = true;
        self
    }

    /// Set the submissions tied to each side of this association
    ///
    /// # Arguments
    ///
    /// * `submissions` - The submissions to tie to this association
    #[must_use]
    pub fn submissions(mut self, submissions: AssociatedSubmissions) -> Self {
        // set the submissions for this association
        self.submissions = Some(submissions);
        self
    }

    /// Tie a specific submission to the other/target side of this association
    ///
    /// This is a convenience for the common single target case (e.g. linking a file into a folder).
    ///
    /// # Arguments
    ///
    /// * `submission` - The submission id to tie to the other side of this association
    #[must_use]
    pub fn submission_other(mut self, submission: Uuid) -> Self {
        // get an entry to our submissions or insert a default and set the other side
        self.submissions.get_or_insert_with(AssociatedSubmissions::default).other = Some(submission);
        self
    }

    /// Tie a specific submission to the source side of this association
    ///
    /// # Arguments
    ///
    /// * `submission` - The submission id to tie to the source side of this association
    #[must_use]
    pub fn submission_source(mut self, submission: Uuid) -> Self {
        // get an entry to our submissions or insert a default and set the source side
        self.submissions.get_or_insert_with(AssociatedSubmissions::default).source = Some(submission);
        self
    }
}

pub trait AssociationSupport {
    /// Check if this assocition kind is valid
    ///
    /// False means this association is not valid for this entity.
    ///
    /// # Arguments
    ///
    /// * `association` - The association to check the validity off
    fn is_valid(association: &Association) -> bool;

    /// Save this association to the DB
    #[cfg(feature = "api")]
    fn save(&self, association: &Association) -> Result<(), crate::utils::ApiError>;
}

/// The options that you can set when listing associations in Thorium
///
/// Currently this only supports single tag queries but when ES support is added multi tag queries
/// will be supported.
#[derive(Debug, Clone)]
pub struct AssociationListOpts {
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
}

impl Default for AssociationListOpts {
    /// Build a default search
    fn default() -> Self {
        AssociationListOpts {
            start: None,
            cursor: None,
            end: None,
            page_size: 50,
            limit: None,
            groups: Vec::default(),
        }
    }
}

impl AssociationListOpts {
    /// Restrict the association search to start at a specific date
    ///
    /// # Arguments
    ///
    /// * `start` - The date to start listing samples from
    #[must_use]
    pub fn start(mut self, start: DateTime<Utc>) -> Self {
        // set the date to start listing associations at
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

    /// Restrict the association search to stop at a specific date
    ///
    /// # Arguments
    ///
    /// * `end` - The date to stop listing samples at
    #[must_use]
    pub fn end(mut self, end: DateTime<Utc>) -> Self {
        // set the date to end listing associations at
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
        // set the date to end listing associations at
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
        // set the date to end listing associations at
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
        // set the date to end listing associations at
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }
}

/// Default the association list limit to 50
fn default_list_limit() -> usize {
    50
}

#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct AssociationListParams {
    /// The groups to list data from
    #[serde(default)]
    pub groups: Vec<String>,
    /// When to start listing data at
    #[serde(default = "Utc::now")]
    pub start: DateTime<Utc>,
    /// When to stop listing data at
    pub end: Option<DateTime<Utc>>,
    /// The cursor id to use if one exists
    pub cursor: Option<Uuid>,
    /// The max number of items to return in this response
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

impl Default for AssociationListParams {
    /// Create a default file list params
    fn default() -> Self {
        AssociationListParams {
            groups: Vec::default(),
            start: Utc::now(),
            end: None,
            cursor: None,
            limit: default_list_limit(),
        }
    }
}

impl From<AssociationListOpts> for AssociationListParams {
    /// Convert `AssociationListOpts` into `AssociationListParams`
    fn from(opts: AssociationListOpts) -> Self {
        AssociationListParams {
            groups: opts.groups,
            start: opts.start.unwrap_or_else(Utc::now),
            end: opts.end,
            cursor: opts.cursor,
            limit: opts.limit.unwrap_or_else(default_list_limit),
        }
    }
}

impl AssociationListParams {
    /// Get the end timestamp or get a sane default
    #[cfg(feature = "api")]
    pub fn end(
        &self,
        shared: &crate::utils::Shared,
    ) -> Result<DateTime<Utc>, crate::utils::ApiError> {
        match self.end {
            Some(end) => Ok(end),
            None => match Utc.timestamp_opt(shared.config.thorium.associations.earliest, 0) {
                chrono::LocalResult::Single(default_end) => Ok(default_end),
                _ => crate::internal_err!(format!(
                    "default earliest association timestamp is invalid or ambigous - {}",
                    shared.config.thorium.associations.earliest
                )),
            },
        }
    }
}
