//! The scylla utils/structs for files

use chrono::prelude::*;
use scylla::DeserializeRow;
use uuid::Uuid;

/// An internal struct containing a single submission row in Scylla
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct SubmissionListRow {
    /// The group this submission was apart of (used only for cursor generation)
    pub group: String,
    /// The sha256 of this sample
    pub sha256: String,
    /// The submission ID for this instance of this sample
    pub submission: Uuid,
    /// The timestamp this was last uploaded
    pub uploaded: DateTime<Utc>,
}

/// An internal struct containing a single submission row in Scylla
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct SubmissionRow {
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
    /// The group this submission is for
    pub group: String,
    /// The user who submitted this sample
    pub submitter: String,
    /// Where this sample originates from if anywhere in serial form
    pub origin: Option<String>,
    // When this sample was uploaded
    pub uploaded: DateTime<Utc>,
}

/// An internal struct containing one instance or row of a Tag in scylla
#[derive(Serialize, Deserialize, Debug, Clone, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub(super) struct TagRow {
    /// The group this tag is a part of
    pub group: String,
    /// The sha256 of the sample this tag is for
    pub sha256: String,
    /// The key for this tag
    pub key: String,
    /// The value for this tag
    pub value: String,
}

/// An internal struct containing one instance or row of a Tag in scylla
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(super) struct FullTagRow {
    /// The group this tag is a part of
    pub group: String,
    /// The year this tag was submitted
    pub year: i32,
    /// The hour this tag was submitted
    pub hour: i32,
    /// The key for this tag
    pub key: String,
    /// The value for this tag
    pub value: String,
    /// The timestamp this tag was submitted
    pub uploaded: DateTime<Utc>,
    /// The sha256 of the sample this tag is for
    pub sha256: String,
}

/// A request for a comment about a specific sample
#[derive(Serialize, Deserialize, Debug, Clone, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct CommentRow {
    /// The group to share this comment with
    pub group: String,
    /// The sha256 of the sample this comment is for
    pub sha256: String,
    /// When this comment was uploaded
    pub uploaded: DateTime<Utc>,
    /// The uuid for this comment
    pub id: Uuid,
    /// the author of this comment
    pub author: String,
    /// The comment for this file
    pub comment: String,
    /// Any paths in s3 to files/attachments for this comment in serialized form
    pub files: String,
}
