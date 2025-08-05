//! A commit is a record of changes to a git repo

use chrono::prelude::*;
use gix::refs::Category;
use std::collections::HashMap;
use std::str::FromStr;
use uuid::Uuid;

#[cfg(feature = "scylla-utils")]
use crate::models::CensusKeys;
use crate::models::InvalidEnum;

/// Info about a single commit for a repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CommitRequest {
    /// The author of this commit
    pub author: String,
    /// When this commit was added
    pub timestamp: DateTime<Utc>,
    /// The topic of this commit
    pub topic: Option<String>,
    /// The description for this commit
    pub description: Option<String>,
    /// Whether this description was truncated or not
    pub truncated: bool,
}

impl CommitRequest {
    /// Create a new [`CommitRequest`]
    ///
    /// # Arguments
    ///
    /// * `commit` - The commit to ingest into Thorium
    #[cfg(feature = "client")]
    pub fn new(commit: git2::Commit) -> Self {
        // extract the hash, author, and timestamp for this commit
        let author = String::from_utf8_lossy(commit.author().email_bytes()).into_owned();
        let timestamp = Utc.timestamp_opt(commit.time().seconds(), 0).unwrap();
        // get the commit message for this commit if it exists
        let (topic, description) = if let Some(message) = commit.message() {
            // try to break this message down into a topic and description
            if let Some(index) = message.find("\n\n") {
                let (topic, description) = message.split_at(index);
                (Some(topic.to_owned()), Some(description[2..].to_owned()))
            } else {
                // if the message is less than 80 chars then assume we just have a topic
                if message.len() <= 80 {
                    (Some(message.to_owned()), None)
                } else {
                    // no topic was found so just use the first 80 chars as the topic
                    let topic = message.chars().take(80).collect::<String>();
                    (Some(topic), Some(message.to_owned()))
                }
            }
        } else {
            (None, None)
        };
        // build our commit request
        let mut req = CommitRequest {
            author,
            timestamp,
            topic,
            description,
            truncated: false,
        };
        // truncate our description if needed
        req.truncate_description(524_288);
        req
    }

    /// Build a new commit request based on a gix commit
    ///
    /// # Arguments
    ///
    /// * `commit` - The commit to ingest into Thorium
    #[cfg(feature = "client")]
    #[must_use]
    pub fn new_gix(commit: gix::Commit) -> Self {
        // decode our entire commit
        let decoded = commit.decode().unwrap();
        // extract the author
        let author = decoded.author().email.to_string();
        // get the timestamp for this commit
        let time = commit.time().unwrap();
        let timestamp = Utc
            .timestamp_opt(time.seconds + time.offset as i64, 0)
            .unwrap();
        // get the commit message for this commit if it exists
        let message = decoded.message();
        // get the first 80 chars of our topic
        let topic = message
            .title
            .to_string()
            .chars()
            .take(80)
            .collect::<String>();
        // get our commit description if it exists
        let description = match &message.body {
            Some(body) => Some(body.to_string()),
            None => None,
        };
        // build our commit request
        let mut req = CommitRequest {
            author,
            timestamp,
            topic: Some(topic),
            description,
            truncated: false,
        };
        // truncate our description if needed
        req.truncate_description(524_288);
        req
    }

    /// Truncate this [`CommitRequest`]'s description to some number of bytes
    ///
    /// # Arguments
    ///
    /// * `len` - The number of bytes to truncate this commits description too
    pub fn truncate_description(&mut self, len: usize) {
        // get our description if one is set
        if let Some(description) = self.description.as_mut() {
            // check if we need to truncate this description
            if description.len() > len {
                // we need to truncate so find the boundary we can safely do that at
                let boundary = description.floor_char_boundary(len);
                // truncate our description
                description.truncate(boundary);
                self.truncated = true;
            }
        }
    }
}

/// A request to store info about a single branch in a repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct BranchRequest {
    /// The commit this branch is on
    pub commit: String,
    /// When this branch was last updated
    pub timestamp: DateTime<Utc>,
}

impl BranchRequest {
    /// Build a new branch request based on a gix reference
    ///
    /// # Arguments
    ///
    /// * `commit` - The commit to ingest into Thorium
    #[cfg(feature = "client")]
    #[must_use]
    pub fn new_gix<'a>(commit: gix::Commit) -> Result<Self, crate::client::Error> {
        // get the timestamp for this commit
        let time = commit.time()?;
        let timestamp = Utc
            .timestamp_opt(time.seconds + time.offset as i64, 0)
            .unwrap();
        // build our branch request
        let req = BranchRequest {
            commit: commit.id().to_string(),
            timestamp,
        };
        Ok(req)
    }
}

/// A request to store info about a single tag in a repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GitTagRequest {
    /// The commit this tag is on
    pub commit: String,
    /// The person who created this tag if an author is set
    pub author: String,
    /// When this tag was last updated
    pub timestamp: DateTime<Utc>,
}

impl GitTagRequest {
    /// Build a new git tag request based on a gix reference
    ///
    /// # Arguments
    ///
    /// * `tag` - The tag to ingest into Thorium
    #[cfg(feature = "client")]
    #[must_use]
    pub fn new_gix<'a>(tag: gix::Tag) -> Result<Self, crate::client::Error> {
        // get our tag targets id
        let target_id = tag.target_id()?;
        // get the signature for this tag if it exists
        if let Some(signature) = tag.tagger()? {
            // get this tags author
            let author = signature.email.to_string();
            // get when this tag was created
            let git_time = signature.time()?;
            // cast our git time to a chrono timestamp
            let timestamp = Utc
                .timestamp_opt(git_time.seconds + git_time.offset as i64, 0)
                .unwrap();
            // build our tag request
            let tag_req = GitTagRequest {
                commit: target_id.to_string(),
                author,
                timestamp,
            };
            Ok(tag_req)
        } else {
            panic!("{:#?} has no signature?", tag.id);
        }
    }

    /// Build a new git tag request based on the commit from a lightweight tag
    ///
    /// # Arguments
    ///
    /// * `commit` - The tag to ingest into Thorium
    #[cfg(feature = "client")]
    #[must_use]
    pub fn from_commit<'a>(commit: gix::Commit) -> Result<Self, crate::client::Error> {
        // get this commits hash
        let commit_hash = commit.id().to_string();
        // get this commits author
        let author = commit.author()?.email.to_string();
        // get the timestamp for this commit
        let time = commit.time()?;
        let timestamp = Utc
            .timestamp_opt(time.seconds + time.offset as i64, 0)
            .unwrap();
        // build our tag request
        let tag_req = GitTagRequest {
            commit: commit_hash,
            author,
            timestamp,
        };
        Ok(tag_req)
    }
}

/// A single commitish
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum CommitishRequest {
    /// A request for a single commit for a repo
    Commit(CommitRequest),
    /// A request for a single branch for a repo
    Branch(BranchRequest),
    /// A request for a single tag for a repo
    Tag(GitTagRequest),
}

impl CommitishRequest {
    /// Cast this gix commitish to a commitish request
    ///
    /// # Arguments
    ///
    /// * `gix_ref` - The gix reference to cast
    #[cfg(feature = "client")]
    #[must_use]
    pub fn new_gix<'a>(
        gix_ref: &gix::Reference<'a>,
    ) -> Result<Option<(String, Self)>, crate::client::Error> {
        // get the category of this reference
        if let Some(cat) = gix_ref.name().category() {
            // get the object this reference points too
            let id = gix_ref.id();
            let object = id.object()?;
            // get the full name of this reference as a string
            let name = gix_ref.name().as_bstr().to_string();
            // trim the common stems
            let name = name
                .trim_start_matches("refs/remotes/origin/")
                .trim_start_matches("refs/tags/");
            // handle the different categories
            let commitish = match cat {
                Category::RemoteBranch => {
                    // cast our object to a commit
                    let commit = object.try_into_commit()?;
                    // build our branch request
                    Self::Branch(BranchRequest::new_gix(commit)?)
                }
                Category::Tag => {
                    // check if this is a lightweight tag or not
                    match object.kind {
                        // this is a lightweight tag so just use the commits info
                        gix::worktree::object::Kind::Commit => {
                            // cast this object into a commit
                            let commit = object.try_into_commit()?;
                            // build our tag request
                            Self::Tag(GitTagRequest::from_commit(commit)?)
                        }
                        // this is a full tag so use the tags info
                        gix::worktree::object::Kind::Tag => {
                            // cast our object to a tag
                            let tag = object.try_into_tag()?;
                            // build our tag request
                            Self::Tag(GitTagRequest::new_gix(tag)?)
                        }
                        // skip any other type of object
                        _ => return Ok(None),
                    }
                }
                _ => return Ok(None),
            };
            Ok(Some((name.to_string(), commitish)))
        } else {
            Ok(None)
        }
    }

    /// Get this commitish requests' timestamp
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            CommitishRequest::Commit(commit) => commit.timestamp,
            CommitishRequest::Branch(branch) => branch.timestamp,
            CommitishRequest::Tag(tag) => tag.timestamp,
        }
    }

    /// Get the kind of commitish this is
    pub fn kind(&self) -> CommitishKinds {
        match self {
            CommitishRequest::Commit(_) => CommitishKinds::Commit,
            CommitishRequest::Branch(_) => CommitishKinds::Branch,
            CommitishRequest::Tag(_) => CommitishKinds::Tag,
        }
    }
}

/// A map of commits for a repo and the data tied to them
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CommitishMapRequest {
    /// The groups to save these commits too
    pub groups: Vec<String>,
    /// The earliest possible commit for this repo
    pub earliest: Option<DateTime<Utc>>,
    /// Whether this is the final batch of commits or not
    pub end: bool,
    /// A map of commit hashes and their info
    pub commitishes: HashMap<String, CommitishRequest>,
}

/// Info on a single commit for a repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Commit {
    /// The hash for this commit
    pub hash: String,
    /// The groups this commit is visible too
    pub groups: Vec<String>,
    /// When this commit was added to this repo
    pub timestamp: DateTime<Utc>,
}

/// Info on a single commit for a repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CommitDetails {
    /// The hash for this commit
    pub hash: String,
    /// The groups this commit is visible too
    pub groups: Vec<String>,
    /// When this commit was added to this repo
    pub timestamp: DateTime<Utc>,
    /// The author of this commit
    pub author: String,
    /// The topic for this commit
    pub topic: Option<String>,
    /// The description for this commit
    pub description: Option<String>,
    /// Whether this description was truncated or not
    pub truncated: bool,
}

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

/// A tag for a git repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GitTag {
    /// The name of this tag
    pub name: String,
    /// The groups this tag is visible too
    pub groups: Vec<String>,
    /// When this tag was last updated
    pub timestamp: DateTime<Utc>,
}

/// A tag for a git repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GitTagDetails {
    /// The name of this tag
    pub name: String,
    /// The groups this tag is visible too
    pub groups: Vec<String>,
    /// The commit this tag is on
    pub commit: String,
    /// The author of this tag
    pub author: String,
    /// When this tag was last updated
    pub timestamp: DateTime<Utc>,
}

/// The different kinds of commitishes
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum Commitish {
    /// A commit for a git repo
    Commit(Commit),
    /// A branch for a git repo
    Branch(Branch),
    /// A tag for a git repo
    Tag(GitTag),
}

impl Commitish {
    /// Get the key for this commitish regardless of kind
    #[must_use]
    pub fn key(&self) -> &String {
        match self {
            Self::Commit(commit) => &commit.hash,
            Self::Branch(branch) => &branch.name,
            Self::Tag(tag) => &tag.name,
        }
    }

    /// Get the groups for this commitish regardless of kind
    #[must_use]
    pub fn groups(&self) -> &Vec<String> {
        match self {
            Self::Commit(commit) => &commit.groups,
            Self::Branch(branch) => &branch.groups,
            Self::Tag(tag) => &tag.groups,
        }
    }

    /// Get the timestamp for this commitish regardless of kind
    #[must_use]
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::Commit(commit) => commit.timestamp,
            Self::Branch(branch) => branch.timestamp,
            Self::Tag(tag) => tag.timestamp,
        }
    }

    /// Get the kind of commitish this is
    #[must_use]
    pub fn kind(&self) -> CommitishKinds {
        match self {
            Self::Commit(_) => CommitishKinds::Commit,
            Self::Branch(_) => CommitishKinds::Branch,
            Self::Tag(_) => CommitishKinds::Tag,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "scylla-utils", derive(scylla::DeserializeRow))]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
pub struct CommitishCensusRow {
    pub kind: CommitishKinds,
    pub group: String,
    pub year: i32,
    pub bucket: i32,
    pub repo: String,
    pub count: i64,
}

#[cfg(feature = "scylla-utils")]
impl crate::models::CensusSupport for Commitish {
    /// The type returned by our prepared statement
    type Row = CommitishCensusRow;

    /// Build the prepared statement for getting partition count info
    async fn scan_prepared_statement(
        scylla: &scylla::client::session::Session,
        ns: &str,
    ) -> Result<scylla::statement::prepared::PreparedStatement, scylla::errors::PrepareError> {
        // build tags get partition count prepared statement
        scylla
            .prepare(format!(
                "SELECT kind, group, year, bucket, repo, count(*) \
                FROM {}.commitish_list \
                WHERE token(kind, group, year, bucket, repo) >= ? AND token(kind, group, year, bucket, repo) <= ? \
                GROUP BY kind, group, year, bucket, repo",
                ns,
            ))
            .await
    }

    /// Get the count for this partition
    fn get_count(row: &CommitishCensusRow) -> i64 {
        row.count
    }

    /// Get the bucket for this partition
    fn get_bucket(row: &CommitishCensusRow) -> i32 {
        row.bucket
    }

    /// Build the count key for this partition
    fn count_key_from_row(namespace: &str, row: &Self::Row, grouping: i32) -> String {
        // build the key for this row
        format!(
            "{namespace}:census:commitish:counts:{kind}:{group}:{repo}:{year}:{grouping}",
            namespace = namespace,
            kind = row.kind,
            group = row.group,
            repo = row.repo,
            year = row.year,
            grouping = grouping,
        )
    }

    /// Build the sorted set key for this census operation
    fn stream_key_from_row(namespace: &str, row: &Self::Row) -> String {
        format!(
            "{namespace}:census:commitish:stream:{kind}:{group}:{repo}:{year}",
            namespace = namespace,
            kind = row.kind,
            group = row.group,
            repo = row.repo,
            year = row.year,
        )
    }
}

/// The different kinds of commitishes
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum CommitishDetails {
    /// A commit for a git repo
    Commit(CommitDetails),
    /// A branch for a git repo
    Branch(BranchDetails),
    /// A tag for a git repo
    Tag(GitTagDetails),
}

impl CommitishDetails {
    /// Get the key for this commitish regardless of kind
    #[must_use]
    pub fn key(&self) -> &String {
        match self {
            Self::Commit(commit) => &commit.hash,
            Self::Branch(branch) => &branch.name,
            Self::Tag(tag) => &tag.name,
        }
    }

    /// Get the groups for this commitish regardless of kind
    #[must_use]
    pub fn groups(&self) -> &Vec<String> {
        match self {
            Self::Commit(commit) => &commit.groups,
            Self::Branch(branch) => &branch.groups,
            Self::Tag(tag) => &tag.groups,
        }
    }

    /// Get the author for this commitish if one can be retrieved
    ///
    /// Branches will return None.
    #[must_use]
    pub fn author(&self) -> Option<&String> {
        match self {
            Self::Commit(commit) => Some(&commit.author),
            Self::Branch(_) => None,
            Self::Tag(tag) => Some(&tag.author),
        }
    }

    /// Get the author for this commitish if one can be retrieved
    ///
    /// Branches will return None.
    #[must_use]
    pub fn author_owned(self) -> Option<String> {
        match self {
            Self::Commit(commit) => Some(commit.author),
            Self::Branch(_) => None,
            Self::Tag(tag) => Some(tag.author),
        }
    }

    /// Get the timestamp for this commitish regardless of kind
    #[must_use]
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::Commit(commit) => commit.timestamp,
            Self::Branch(branch) => branch.timestamp,
            Self::Tag(tag) => tag.timestamp,
        }
    }

    /// Get the kind of commitish this is
    #[must_use]
    pub fn kind(&self) -> CommitishKinds {
        match self {
            Self::Commit(_) => CommitishKinds::Commit,
            Self::Branch(_) => CommitishKinds::Branch,
            Self::Tag(_) => CommitishKinds::Tag,
        }
    }
}

/// The options that you can set when listing repo commits in Thorium
#[derive(Debug, Clone)]
pub struct CommitListOpts {
    /// The cursor to use to continue this search
    pub cursor: Option<Uuid>,
    /// The latest date to start listing samples from
    pub start: Option<DateTime<Utc>>,
    /// The oldest date to stop listing samples from
    pub end: Option<DateTime<Utc>>,
    /// The max number of objects to retrieve on a single page
    pub page_size: usize,
    /// The limit to use when requesting data
    pub limit: Option<usize>,
    /// The groups limit our search to
    pub groups: Vec<String>,
    /// The kinds of commitishes to list
    pub kinds: Vec<CommitishKinds>,
}

impl Default for CommitListOpts {
    /// Build a default search
    fn default() -> Self {
        CommitListOpts {
            start: None,
            cursor: None,
            end: None,
            page_size: 50,
            limit: None,
            groups: Vec::default(),
            kinds: Vec::default(),
        }
    }
}

impl CommitListOpts {
    /// Restrict the file search to start at a specific date
    ///
    /// # Arguments
    ///
    /// * `start` - The date to start listing samples from
    pub fn start(mut self, start: DateTime<Utc>) -> Self {
        // set the date to start listing commits at
        self.start = Some(start);
        self
    }

    /// Set the cursor to use when continuing this search
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor id to use for this search
    pub fn cursor(mut self, cursor: Uuid) -> Self {
        // set cursor for this search
        self.cursor = Some(cursor);
        self
    }

    /// Restrict the commit search to stop at a specific date
    ///
    /// # Arguments
    ///
    /// * `end` - The date to stop listing samples at
    pub fn end(mut self, end: DateTime<Utc>) -> Self {
        // set the date to end listing commits at
        self.end = Some(end);
        self
    }

    /// The max number of objects to retrieve in a single page
    ///
    /// # Arguments
    ///
    /// * `page_size` - The max number of documents to return in a single request
    pub fn page_size(mut self, page_size: usize) -> Self {
        // set the date to end listing commits at
        self.page_size = page_size;
        self
    }

    /// Limit how many commits this search can return at once
    ///
    /// # Arguments
    ///
    /// * `limit` - The number of documents to return at once
    pub fn limit(mut self, limit: usize) -> Self {
        // set the date to end listing commits at
        self.limit = Some(limit);
        self
    }

    /// Limit what groups we search in
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our search to
    pub fn groups<T: Into<String>>(mut self, groups: Vec<T>) -> Self {
        // set the date to end listing commits at
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }
}

/// Default the list limit to 50
fn default_list_limit() -> usize {
    50
}

/// Help serde default to all commitish kinds
fn default_commitish_kinds() -> Vec<CommitishKinds> {
    CommitishKinds::all()
}

/// The options that you can set when listing repo commits in Thorium
#[derive(Deserialize, Debug)]
pub struct CommitishListParams {
    /// The cursor to use to continue this search
    pub cursor: Option<Uuid>,
    /// The latest date to start listing samples from
    #[serde(default = "Utc::now")]
    pub start: DateTime<Utc>,
    /// The oldest date to stop listing samples from
    pub end: Option<DateTime<Utc>>,
    /// The limit to use when requesting data
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    /// The groups limit our search to
    #[serde(default)]
    pub groups: Vec<String>,
    /// The kinds of commitishes to limit our search too
    #[serde(default = "default_commitish_kinds")]
    pub kinds: Vec<CommitishKinds>,
}

impl Default for CommitishListParams {
    /// Build a default search
    fn default() -> Self {
        CommitishListParams {
            start: Utc::now(),
            cursor: None,
            end: None,
            limit: default_list_limit(),
            groups: Vec::default(),
            kinds: CommitishKinds::all(),
        }
    }
}

impl CommitishListParams {
    /// Build a new default commit list params for some groups
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict this cursor too
    /// * `end` - The oldest timestamp to list commits at if one is known
    /// * `limit` - The max number of commits to return
    #[cfg(feature = "api")]
    pub fn new(groups: Vec<String>, end: Option<DateTime<Utc>>, limit: usize) -> Self {
        CommitishListParams {
            start: Utc::now(),
            cursor: None,
            end,
            limit,
            groups,
            kinds: CommitishKinds::all(),
        }
    }

    /// Get the end timestamp or get a sane default
    #[cfg(feature = "api")]
    pub fn end(
        &self,
        shared: &crate::utils::Shared,
    ) -> Result<DateTime<Utc>, crate::utils::ApiError> {
        match self.end {
            Some(end) => Ok(end),
            None => match Utc.timestamp_opt(shared.config.thorium.repos.earliest, 0) {
                chrono::LocalResult::Single(default_end) => Ok(default_end),
                _ => crate::internal_err!(format!(
                    "default earliest repos timestamp is invalid or ambiguous - {}",
                    shared.config.thorium.repos.earliest
                )),
            },
        }
    }
}

/// The different kind of commitishes
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "trace", derive(valuable::Valuable))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum CommitishKinds {
    /// A commit
    Commit,
    /// A branch
    Branch,
    /// A tag
    Tag,
}

impl CommitishKinds {
    // Build a list of all commitish kinds
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::Commit, Self::Branch, Self::Tag]
    }

    /// Cast our [`CommittishKind`] to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Commit => "Commit",
            Self::Branch => "Branch",
            Self::Tag => "Tag",
        }
    }
}

impl std::fmt::Display for CommitishKinds {
    /// Allows for commitish kinds to be displayed
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for CommitishKinds {
    type Err = InvalidEnum;

    /// Convert this str to an [`CommitishKinds`]
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "Commit" => Ok(CommitishKinds::Commit),
            "Branch" => Ok(CommitishKinds::Branch),
            "Tag" => Ok(CommitishKinds::Tag),
            _ => Err(InvalidEnum(format!("Unknown CommitishKinds: {raw}"))),
        }
    }
}
