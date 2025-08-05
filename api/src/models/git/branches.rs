//! A branch is a snapshot of a tree of commits for a repo

use chrono::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Branch {
    /// The name of this branch
    pub name: String,
    /// The timestamp for the most recent commit in this branch
    pub last_committed: DateTime<Utc>,
}
