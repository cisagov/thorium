//! A branch is a snapshot of a tree of commits for a repo

use chrono::prelude::*;

/// A branch for a git repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Branch {
    /// The name of this branch
    pub name: String,
    /// The groups this branch is visible too
    pub groups: Vec<String>,
    /// When this branch was last updated
    pub timestamp: DateTime<Utc>,
}

/// A branch for a git repo with detailed info
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct BranchDetails {
    /// The name of this branch
    pub name: String,
    /// The groups this branch is visible too
    pub groups: Vec<String>,
    /// The commit this branch is on
    pub commit: String,
    /// When this branch was last updated
    pub timestamp: DateTime<Utc>,
}
