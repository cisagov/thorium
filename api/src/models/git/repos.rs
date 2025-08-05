//! A repository is a collection of source code for a project

use chrono::prelude::*;
use indicatif::ProgressBar;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::path::PathBuf;
use uuid::Uuid;

use super::CommitishKinds;
use crate::models::{KeySupport, TagMap};

// api only imports
#[cfg(feature = "api")]
use crate::models::TagDeleteRequest;

// api/client imports
cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        use crate::models::{TagRequest, TagType, OutputKind};
        use crate::models::backends::{TagSupport, OutputSupport};
    }
}

// client only imports
cfg_if::cfg_if! {
    if #[cfg(feature = "client")] {
        use crate::Error;
        use crate::{multipart_file, multipart_list};
        use tokio::io::BufReader;
        use tokio::fs::File;
        use tokio_tar::Archive;
    }
}

// only support scylla and other api side only structs if the api features is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        /// A map of commits for a repo and the data tied to them
        #[derive(Debug, Default)]
        pub struct RepoDataForm{
            /// The groups to share this repos data with
            pub groups: Vec<String>,
        }
    }
}

/// The components of a repo URL
pub struct RepoUrlComponents {
    pub provider: String,
    pub user: String,
    pub name: String,
    pub scheme: RepoScheme,
}

impl RepoUrlComponents {
    /// Construct a repo URL from its components
    #[must_use]
    pub fn get_url(&self) -> String {
        format!("{}/{}/{}", self.provider, self.user, self.name)
    }
}

/// A Request for a new repo to be added to Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoRequest {
    // The groups to share this repo with
    pub groups: Vec<String>,
    /// The url this repo comes from
    pub url: String,
    /// The tags to save for this repo
    pub tags: HashMap<String, HashSet<String>>,
    /// The default checkout behavior for this repo
    pub default_checkout: Option<RepoCheckout>,
    /// The trigger depth of this sample upload
    #[serde(default)]
    pub trigger_depth: u8,
}

impl RepoRequest {
    /// Create a new repo request
    ///
    /// # Arguments
    ///
    /// * `url` - The url to this repo
    /// * `groups` - The groups to share this repo with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{RepoRequest, RepoCheckout};
    ///
    /// // This repos default branch is main
    /// let default_checkout = Some(RepoCheckout::branch("main"));
    /// RepoRequest::new("https://github.com/rust-lang/rust", vec!("CornPeeps"), default_checkout);
    /// ```
    pub fn new<U: Into<String>, G: Into<String>>(
        url: U,
        groups: Vec<G>,
        default_checkout: Option<RepoCheckout>,
    ) -> Self {
        // convert our groups to strings
        let groups = groups.into_iter().map(|item| item.into()).collect();
        RepoRequest {
            url: url.into(),
            groups,
            tags: HashMap::default(),
            default_checkout,
            trigger_depth: 0,
        }
    }

    /// Adds a tag for this repo
    ///
    /// # Arguments
    ///
    /// * `key` - The key to set for this tag
    /// * `value` - The value to set for this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{RepoRequest, RepoCheckout};
    ///
    /// // This repos default branch is main
    /// let default_checkout = Some(RepoCheckout::branch("main"));
    /// // Build this repo request with a custom tag
    /// RepoRequest::new("https://github.com/rust-lang/rust", vec!("CornPeeps"), Some(RepoCheckout::branch("main")))
    ///   .tag("plant", "corn");
    /// ```
    #[must_use]
    pub fn tag<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        // get the vector of values for this tag or insert a default
        let values = self.tags.entry(key.into()).or_default();
        // insert our new tag
        values.insert(value.into());
        self
    }

    /// Adds multiple values for the same tag for this repo
    ///
    /// # Arguments
    ///
    /// * `key` - The key to set for this tag
    /// * `value` - The values to set for this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{RepoRequest, RepoCheckout};
    ///
    /// // This repos default branch is main
    /// let default_checkout = Some(RepoCheckout::branch("main"));
    /// // Build this repo request with custom tags
    /// RepoRequest::new("https://github.com/rust-lang/rust", vec!("CornPeeps"), Some(RepoCheckout::branch("main")))
    ///   .tags("plant", vec!("corn", "oranges"));
    /// ```
    #[must_use]
    pub fn tags<T: Into<String>>(mut self, key: T, values: Vec<T>) -> Self {
        // get the vector of values for this tag or insert a default
        let entry = self.tags.entry(key.into()).or_default();
        // insert our new tags
        entry.extend(values.into_iter().map(|val| val.into()));
        self
    }

    /// Set the trigger depth for this repo request
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

/// The response returned from a repo create request
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoCreateResponse {
    /// The normalized repo URL
    pub url: String,
}

/// A single submission of a repository
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RepoSubmission {
    /// The group this repo submission is visible by
    pub groups: Vec<String>,
    /// Where this repo comes from (e.g. github.com)
    pub provider: String,
    /// The user that created this repo in the provider
    pub user: String,
    /// The name of this repo
    pub name: String,
    /// The full url for this repo
    pub url: String,
    /// The unique id for this repo
    pub id: Uuid,
    /// The user that added this repo to Thorium
    pub creator: String,
    /// When this repo was added to Thorium
    pub uploaded: DateTime<Utc>,
    // The scheme to use when cloning this repo
    pub scheme: RepoScheme,
    /// The default checkout behavior for this repo
    pub default_checkout: Option<RepoCheckout>,
    /// The earliest commit ever seen in this repo
    pub earliest: Option<DateTime<Utc>>,
}

/// The scheme to use when pulling this repos data
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum RepoScheme {
    /// Use https when pulling this repo
    Https,
    /// Use http when pulling this repo
    Http,
    /// Use https when pulling this repo with auth info
    HttpsAuthed { username: String, password: String },
    /// Use http when pulling this repo with auth info
    HttpAuthed { username: String, password: String },
    /// An authenticated scheme was used but not for this user
    ScrubbedAuth,
}

impl Default for RepoScheme {
    /// Create a default [`RepoScheme`] of [`RepoScheme::Https`]
    fn default() -> Self {
        RepoScheme::Https
    }
}

impl std::fmt::Display for RepoScheme {
    /// Display this repo scheme nicely
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // write this repo scheme correctly
        match self {
            RepoScheme::Https => write!(f, "https://"),
            RepoScheme::Http => write!(f, "http://"),
            RepoScheme::HttpsAuthed { username, password } => {
                write!(f, "https://{username}:{password}@")
            }
            RepoScheme::HttpAuthed { username, password } => {
                write!(f, "http://{username}:{password}@")
            }
            RepoScheme::ScrubbedAuth => write!(f, "ScrubbedAuth"),
        }
    }
}

/// A single submission of a repository
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoSubmissionChunk {
    /// The group this repo submission is visible by
    pub groups: Vec<String>,
    /// The unique id for this repo submission
    pub id: Uuid,
    /// The user that added this repo to Thorium
    pub creator: String,
    /// When this repo was added to Thorium
    pub uploaded: DateTime<Utc>,
    // The scheme to use when cloning this repo
    pub scheme: RepoScheme,
    /// The earliest commit ever seen in this repo
    pub earliest: Option<DateTime<Utc>>,
}

/// The commit or branch to checkout
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum RepoCheckout {
    /// Checkout a commit in detached mode
    Commit(String),
    /// Checkout a branch
    Branch(String),
    /// Checkout a tag
    Tag(String),
}

impl RepoCheckout {
    /// Checkout a commit in detached mode by default when unpacking this repo
    ///
    /// # Arguments
    ///
    /// * `commit` - The commit to checkout
    #[must_use]
    pub fn commit<T: Into<String>>(commit: T) -> Self {
        RepoCheckout::Commit(commit.into())
    }

    /// Checkout a branch by default when unpacking this repo
    ///
    /// # Arguments
    ///
    /// * `branch` - The branch to checkout
    #[must_use]
    pub fn branch<T: Into<String>>(branch: T) -> Self {
        RepoCheckout::Branch(branch.into())
    }

    /// Checkout a tag by default when unpacking this repo
    ///
    /// # Arguments
    ///
    /// * `tag` - The tag to checkout
    #[must_use]
    pub fn tag<T: Into<String>>(tag: T) -> Self {
        RepoCheckout::Tag(tag.into())
    }

    /// Get the value of this checkout target regardless of what type
    pub fn value(&self) -> &str {
        match self {
            Self::Commit(commit) => commit,
            Self::Branch(branch) => branch,
            Self::Tag(tag) => tag,
        }
    }
}

/// A git repository that is tracked by Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Repo {
    /// Where this repo comes from (e.g. github.com)
    pub provider: String,
    /// The user that created this repo in the provider
    pub user: String,
    /// The name of this repo
    pub name: String,
    /// The url for this repo
    pub url: String,
    /// The tags for this repo
    pub tags: TagMap,
    /// The default checkout behavior for this repo
    pub default_checkout: Option<RepoCheckout>,
    /// The submissions for this repo
    pub submissions: Vec<RepoSubmissionChunk>,
    /// The earliest commit ever seen in this repo
    pub earliest: Option<DateTime<Utc>>,
}

impl Repo {
    /// Add a repo submission to this repos submission list
    ///
    /// # Arguments
    ///
    /// * `sub` - The repo submission to add to this repo object
    pub(crate) fn add(&mut self, sub: RepoSubmission) {
        // downselect to just the fields for a submission chunk
        let chunk = RepoSubmissionChunk {
            id: sub.id,
            groups: sub.groups,
            creator: sub.creator,
            uploaded: sub.uploaded,
            scheme: sub.scheme,
            earliest: sub.earliest,
        };
        // update our default checkout if its different
        if self.default_checkout != sub.default_checkout {
            // update our repos checkout behavior
            self.default_checkout = sub.default_checkout.clone();
        }
        // update our earliest if its not set yet or this submission has an older timestamp
        match (self.earliest.as_mut(), &sub.earliest) {
            (Some(old), Some(new)) if new < old => *old = *new,
            (None, Some(new)) => self.earliest = Some(*new),
            _ => (),
        };
        self.submissions.push(chunk);
    }

    /// Get the groups this repo is apart of in all of its submissions
    #[must_use]
    pub fn groups(&self) -> HashSet<&String> {
        self.submissions
            .iter()
            // get the groups from each submission
            .flat_map(|sub| sub.groups.iter())
            // dedupe groups by collecting to a HashSet
            .collect()
    }

    /// Take the groups this repo is apart of in all of its submissions
    /// as owned Strings, replacing submissions' groups with empty Vec's
    #[must_use]
    pub fn groups_take(&mut self) -> HashSet<String> {
        self.submissions
            .iter_mut()
            // take the groups from each submission
            .flat_map(|s| std::mem::take(&mut s.groups))
            // dedupe groups by collecting to a HashSet
            .collect()
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

impl KeySupport for Repo {
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
impl TagSupport for Repo {
    /// Get the tag kind to write to the DB
    fn tag_kind() -> TagType {
        TagType::Repos
    }

    /// Get the earliest each group has seen this object
    fn earliest(&self) -> HashMap<&String, DateTime<Utc>> {
        // assume we have at least 3 groups to cut down on reallocations
        let mut earliest = HashMap::default();
        // crawl over each submission and find the earliest time a group saw this repo
        for sub in self.submissions.iter().rev() {
            // crawl over the groups in this submission
            for group in &sub.groups {
                // if this group isn't in our map then insert it
                earliest.entry(group).or_insert(sub.uploaded);
            }
        }
        earliest
    }

    /// Add some tags to a repo
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating tags
    /// * `req` - The tag request to apply
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[tracing::instrument(name = "TagSupport<Repo>::tag", skip_all, fields(repo = self.url), err(Debug))]
    async fn tag(
        &self,
        user: &crate::models::User,
        mut req: TagRequest<Repo>,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // if groups were supplied then validate this repo is in them otherwise use defaults

        use crate::models::GroupAllowAction;
        self.validate_check_allow_groups(user, &mut req.groups, GroupAllowAction::Tags, shared)
            .await?;
        // get the earliest time this repo was uploaded for each group
        let earliest = self.earliest();
        // save our repo's tags to scylla
        crate::models::backends::db::tags::create(user, self.url.clone(), req, &earliest, shared)
            .await
    }

    /// Delete some tags from this repo
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting tags
    /// * `req` - The tags to delete
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[tracing::instrument(name = "TagSupport<Repo>::delete_tags", skip_all, fields(repo = self.url), err(Debug))]
    async fn delete_tags(
        &self,
        user: &crate::models::User,
        mut req: TagDeleteRequest<Repo>,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // if groups were supplied then validate this repo is in them otherwise use defaults
        self.validate_groups(user, &mut req.groups, true, shared)
            .await?;
        // delete the requested tags for this repo if they exist
        crate::models::backends::db::tags::delete(&self.url, &req, shared).await
    }

    /// Gets tags for a specific repo
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to restrict our returned tags too
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[tracing::instrument(name = "TagSupport<Repo>::get_tags", skip_all, fields(repo = self.url), err(Debug))]
    async fn get_tags(
        &mut self,
        groups: &Vec<String>,
        shared: &crate::utils::Shared,
    ) -> Result<(), crate::utils::ApiError> {
        // get the requested tags
        crate::models::backends::db::tags::get(
            crate::models::TagType::Repos,
            groups,
            &self.url,
            &mut self.tags,
            shared,
        )
        .await
    }
}

#[cfg(any(feature = "api", feature = "client"))]
impl OutputSupport for Repo {
    /// Get the tag kind to write to the DB
    fn output_kind() -> OutputKind {
        OutputKind::Repos
    }

    /// Build a tag request for this output kind
    fn tag_req() -> TagRequest<Self> {
        TagRequest::<Repo>::default()
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
        self.validate_check_allow_groups(
            user,
            groups,
            crate::models::GroupAllowAction::Results,
            shared,
        )
        .await?;
        Ok(())
    }
}

impl From<RepoSubmission> for Repo {
    fn from(sub: RepoSubmission) -> Self {
        // build repo with just current submission
        let mut repo = Repo {
            provider: sub.provider.clone(),
            user: sub.user.clone(),
            name: sub.name.clone(),
            url: sub.url.clone(),
            submissions: Vec::with_capacity(1),
            default_checkout: None,
            tags: TagMap::default(),
            earliest: None,
        };
        // add our current submission to this repo
        repo.add(sub);
        repo
    }
}

impl PartialEq<RepoRequest> for Repo {
    /// Check if a [`RepoRequest`] and a [`Repo`] are equal
    ///
    /// # Arguments
    ///
    /// * `req` - The request to compare against
    fn eq(&self, req: &RepoRequest) -> bool {
        // make sure all tags in the request are set in the repo
        if !req.tags.iter().all(|(key, values)| {
            // make sure this tag was set
            if let Some(value_map) = self.tags.get(key) {
                // make sure all values were set with the correct groups
                values.iter().all(|value| {
                    // retrieve the groups set for this value
                    if let Some(groups) = value_map.get(value) {
                        // make sure the correct groups were set
                        req.groups
                            .iter()
                            .all(|req_group| groups.contains(req_group))
                    } else {
                        // the request tag key exists but our value was not set
                        false
                    }
                })
            } else {
                // the request tag key does not exist in the tag map
                false
            }
        }) {
            return false;
        }
        // the request url may have been modified when saved, so modify in the same way to compare
        let req_url_trimmed = req.url.trim_end_matches(".git");
        // the scheme may have been trimmed, so check only that the request url ends with the repo url;
        // this is a bit unreliable, but avoids having to parse the url here, which is fallible
        if !req_url_trimmed.ends_with(&self.url) {
            return false;
        }
        // find our submission in this repo
        // this is a bit unreliable as we don't have our username in the request
        self.submissions
            .iter()
            .any(|sub| req.groups.iter().all(|group| sub.groups.contains(group)))
    }
}

/// A tarred and downloaded repo
#[derive(Debug)]
pub struct TarredRepo {
    /// The name of the repo that is tarred up (not the git url)
    pub name: String,
    /// The repo that we tarred up
    pub repo: PathBuf,
}

impl TarredRepo {
    /// Untar this repo
    ///
    /// # Arguments
    ///
    /// * `path` - The dir to untar this repo too
    #[cfg(feature = "client")]
    pub async fn unpack<P: AsRef<Path>>(self, target: P) -> Result<UntarredRepo, Error> {
        // get a file handle to our tarred repo
        use git2::build::CheckoutBuilder;
        let file = File::open(&self.repo).await?;
        // build the uncart stream wrapper for this file
        let uncart = cart_rs::UncartStream::new(BufReader::new(file));
        // cast our target to a path
        let target = target.as_ref();
        // untar this repo to disk
        Archive::new(uncart).unpack(target).await?;
        // build the path to this repo
        let mut path = PathBuf::from(target);
        path.push(&self.name);
        // open our untarred repo as a git object
        let repo = git2::Repository::open(&path)?;
        // set our checkout options
        let mut checkout_opts = CheckoutBuilder::new();
        // ensure we match head under any circumstances
        checkout_opts.force();
        // checkout head to resolve any changes
        repo.checkout_head(Some(&mut checkout_opts))?;
        // build our untarred object
        Ok(UntarredRepo { path })
    }

    /// Add this zipped repo to a multipart form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to extend with our origin request info
    #[cfg(feature = "client")]
    pub async fn to_form(self, mut groups: Vec<String>) -> Result<reqwest::multipart::Form, Error> {
        // build the form we are going to send
        // disable percent encoding, as the API natively supports UTF-8
        let form = reqwest::multipart::Form::new().percent_encode_noop();
        // add the groups to share this repo data with
        let form = multipart_list!(form, "groups", groups);
        // add this zip file to our form
        let form = multipart_file!(form, "data", &self.repo);
        Ok(form)
    }
}

/// A response containing the sha256 of the repo data that was uploaded
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoDataUploadResponse {
    pub sha256: String,
}

// only include untarred repo support when the client feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "client")] {
        /// An untarred repository on disk
        pub struct UntarredRepo {
            /// The path to this untarred repo
            pub path: PathBuf,
        }

        impl UntarredRepo {
            /// Create a new untarred repo object by path
            pub fn new<P: Into<PathBuf>>(path: P) -> Result<Self, Error> {
                // cast to a path buf
                let path = path.into();
                Ok(UntarredRepo { path })
            }

            /// Get the remote url for this repo without the scheme or the ending .git
            ///
            /// If no remote name is provided then origin with be used.
            ///
            /// `name` - The remote to use
            pub fn remote(&self, name: Option<&str>) -> Result<String, Error> {
                // open our untarred repo as a git repo
                let repo = git2::Repository::open(&self.path)?;
                // get this repos remote for the requested origin
                let remote = repo.find_remote(name.unwrap_or("origin"))?;
                // cast this remote lossily to a utf8 string
                let url = String::from_utf8_lossy(remote.url_bytes()).into_owned();
                // TODO: replace by detecting the scheme
                let url = url.trim_start_matches("https://");
                let url = url.trim_start_matches("http://");
                // remove the trailing .git if it exists
                let url = url.trim_end_matches('/').trim_end_matches(".git");
                Ok(url.to_string())
            }

            /// Get this repos currently checked out commit
            pub fn commit(&self) -> Result<String, Error> {
                // open our repo
                let repo = gix::open(&self.path)?;
                // get our current commit head
                let head = repo.head_commit().expect("Failed to get head as a commit");
                // get the sha1 for this commit
                let commit = head.id().to_string();
                Ok(commit)
            }

            /// Tar up a repository to be uploaded to Thorium
            ///
            /// # Arguments
            ///
            /// * `target` - The parent directory to write this repo too
            pub async fn tar<P: Into<PathBuf>>(&self, target: P) -> Result<TarredRepo, Error> {
                // cast our target to a pathbuf
                let path = target.into();
                // create the file we want to write our tar file too
                let file = File::create(&path).await?;
                // Instance our tar builder
                let mut tar = tokio_tar::Builder::new(file);
                // ensure that we add symlinks instead of their contents to our repos
                tar.follow_symlinks(false);
                // get the name of our repo
                let name = match self.path.file_name() {
                    Some(file_name) => file_name.to_string_lossy().to_string(),
                    None => return Err(Error::new("Failed to get repository name".to_owned())),
                };
                // tar up our repo
                tar.append_dir_all(&name, &self.path).await?;
                // finish tarring our repo
                tar.finish().await?;
                // create our tarred repo object
                Ok(TarredRepo { name, repo: path } )
            }

            /// Find the default checkout branch for this repo
            ///
            /// This may return none if the repo has only been initialized.
            ///
            /// # Arguments
            ///
            /// * `preferred` - The branches we would prefer to checkout if they exist in prioritized order
            ///
            /// # Panics
            ///
            /// This will panic if any remote branches HEAD timestamp is not valid.
            pub fn find_default_checkout(&self, preferred: &[String]) -> Result<Option<RepoCheckout>, Error> {
                // open our repo
                let repo = gix::open(&self.path)?;
                // get an iterator over this repos references
                let refs = repo.references()?;
                // build a map of all branches and their last updated timestamp
                let mut branch_map = BTreeMap::default();
                // keep track of the preferred branches that we find
                let mut found_pref = Vec::with_capacity(preferred.len() - 1);
                // crawl over all branches and find the most likely default branch
                for info in refs.remote_branches()? {
                    // bubble up any errors from getting this references info
                    let mut info = info.map_err(|err| Error::new(err.to_string()))?;
                    // get the human friendly common name for this branch
                    let name = info.name().file_name().to_string();
                    // get when this branch was last updated
                    let object = info.peel_to_id_in_place()?.object()?;
                    let commit = object.to_commit_ref();
                    let time = commit.time();
                    let timestamp = Utc.timestamp_opt(time.seconds + time.offset as i64, 0).unwrap();
                    // if we found our top choice then short circuit and just use that
                    if Some(&name) == preferred.first() {
                        return Ok(Some(RepoCheckout::Branch(name)));
                    }
                    // if this is in our preferred list then track that
                    if preferred.contains(&name) {
                        found_pref.push(name);
                    } else {
                        // add this branch and its timestamp
                        branch_map.insert(timestamp, name);
                    }
                }
                // if this is in our preferred list then track that
                if let Some(name) = preferred.iter().find(|name| found_pref.contains(name)) {
                        return Ok(Some(RepoCheckout::Branch(name.to_owned())));
                }
                // otherwise use the most recent branch we found
                match branch_map.pop_first() {
                    Some((_, name)) => Ok(Some(RepoCheckout::Branch(name))),
                    None => Ok(None),
                }
            }

            /// Update our local copy of this repo
            ///
            /// This will leave the repo at whatever is the last reference to be fetched. Afterward you likely
            /// will need to checkout main or another branch.
            ///
            /// # Arguments
            ///
            /// * `repo` - The repo to fetch updates for
            /// * `git_conf` - The git settings to use when pulling repos
            pub fn fetch(&self, repo: &mut git2::Repository, git_conf: &Option<crate::client::conf::GitSettings>) -> Result<(), Error>{
                // build the options for this fetch
                let mut opts = git2::FetchOptions::new();
                // always pull all tags
                opts.download_tags(git2::AutotagOption::All);
                // build a new callback to setup auth if we have git settings
                if let Some(git_settings) = git_conf {
                    let mut callbacks = git2::RemoteCallbacks::new();
                    // set our credentials info
                    callbacks.credentials(|_url, username, _| {
                        git2::Cred::ssh_key(
                            username.unwrap_or("git"),
                            None,
                            std::path::Path::new(&git_settings.ssh_keys),
                            None,
                        )
                    });
                    opts.remote_callbacks(callbacks);
                }
                // get the origin remote
                let mut remote = repo.find_remote("origin")?;
                // fetch  from our remote
                remote.fetch::<&String>(&[], Some(&mut opts), None)?;
                // fast forward each branch in our repo
                for branch in repo.branches(Some(git2::BranchType::Remote))? {
                    // unwrap this branch
                    let (branch, _) = branch?;
                    // turn this branch into a reference
                    let mut reference = branch.into_reference();
                    // skip any symbolic references
                    if reference.kind() != Some(git2::ReferenceType::Symbolic) {
                        // get an annotated commit
                        let commit = repo.reference_to_annotated_commit(&reference)?;
                        // get this references name
                        let name = match reference.name() {
                            Some(name) => name.to_string(),
                            None => String::from_utf8_lossy(reference.name_bytes()).to_string(),
                        };
                        // get this references id
                        let id = commit.id();
                        // set this references target
                        reference.set_target(id, "")?;
                        // checkout this reference
                        repo.set_head(&name)?;
                        // checkout our head
                        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
                        // update our local branch
                        repo.branch_from_annotated_commit(name.trim_start_matches("refs/remotes/origin/"), &commit, true)?;
                    }
                }
                Ok(())
            }


            /// Checkout a specific commitish for this repo
            ///
            /// # Arguments
            ///
            /// `commitish` - The branch, commit, tag to checkout
            pub fn checkout(&self, commitish: &str) -> Result<(), Error> {
                // open our untarred repo as a git object
                let repo = git2::Repository::open(&self.path)?;
                // get the object and reference to checkout
                let (object, reference) = repo.revparse_ext(commitish)?;
                // checkout our object
                repo.checkout_tree(&object, None)?;
                // update our head correctly
                match reference {
                    Some(reference) => match reference.name() {
                        Some(name) => repo.set_head(name)?,
                        None => return Err(Error::new("Failed to get reference name")),
                    },
                    None => repo.set_head_detached(object.id())?,
                }
                Ok(())
            }

            /// Checkout a specific commit for an already opened repo
            ///
            /// # Arguments
            ///
            /// `repo` - The repo to checkout a commitish for
            /// `commitish` - The branch, commit, tag to checkout
            pub fn checkout_in_place(&self, repo: &mut git2::Repository, commitish: &str) -> Result<(), Error> {
                // get the object and reference to checkout
                let (object, reference) = repo.revparse_ext(commitish)?;
                // checkout our object
                repo.checkout_tree(&object, None)?;
                // update our head correctly
                match reference {
                    Some(reference) => match reference.name() {
                        Some(name) => repo.set_head(name)?,
                        None => return Err(Error::new("Failed to get reference name")),
                    },
                    None => repo.set_head_detached(object.id())?,
                }
                Ok(())
            }

        }
    }
}

/// A single repo submission line
#[derive(Serialize, Deserialize, Debug, Clone)]
/// Add FromRow support for scylla loading if API mode is enabled
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoListLine {
    /// The group this submission was apart of (used only for cursor generation)
    #[serde(skip_serializing, skip_deserializing)]
    pub groups: HashSet<String>,
    /// The url for this repo
    pub url: String,
    /// The submission ID for this instance of this repo if its exposed by this listing operation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submission: Option<Uuid>,
    /// The timestamp this was uploaded
    pub uploaded: DateTime<Utc>,
}

/// A request for a specific repo/commit to be downloaded executing a job
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoDependencyRequest {
    /// The url to the repo to download
    pub url: String,
    /// The branch, commit, or tag to checkout
    pub commitish: Option<String>,
    /// The kind of commitish to use for checkout
    pub kind: Option<CommitishKinds>,
}

impl RepoDependencyRequest {
    /// Build a repo dependency request
    ///
    /// # Arguments
    ///
    /// * `url` - The url of the repo
    pub fn new<T: Into<String>>(url: T) -> Self {
        RepoDependencyRequest {
            url: url.into(),
            commitish: None,
            kind: None,
        }
    }

    /// Set the commitish reference to be checked out.
    ///
    /// This can be a branch, commit, or tag.
    ///
    /// # Arguments
    ///
    /// * `commitish` - The commitish  reference to be checked out
    pub fn commitish<T: Into<String>>(mut self, commitish: T) -> Self {
        self.commitish = Some(commitish.into());
        self
    }

    /// Set the kind of commitish to be checked out
    ///
    /// If no kind is set Thorium will check against all kinds and fail any
    /// reactions where the commitish is ambiguous (i.e. A branch named main
    /// and a tag named main).
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of commitish to download
    pub fn kind(mut self, kind: CommitishKinds) -> Self {
        self.kind = Some(kind);
        self
    }
}

/// A specific repo/commit to download before executing a job
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "trace", derive(valuable::Valuable))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoDependency {
    /// The url to the repo to download
    pub url: String,
    /// The commitish to checkout
    pub commitish: Option<String>,
    /// The kind of commitish to use for checkout
    pub kind: Option<CommitishKinds>,
}

/// Default the list limit to 50
fn default_list_limit() -> usize {
    50
}

/// The options that you can set when listing repos in Thorium
#[derive(Debug, Clone)]
pub struct RepoListOpts {
    /// The cursor to use to continue this search
    pub cursor: Option<Uuid>,
    /// The latest date to start listing repos from
    pub start: Option<DateTime<Utc>>,
    /// The oldest date to stop listing repos from
    pub end: Option<DateTime<Utc>>,
    /// The max number of objects to retrieve on a single page
    pub page_size: usize,
    /// The limit to use when requesting data
    pub limit: Option<usize>,
    /// The groups limit our search to
    pub groups: Vec<String>,
    /// The tags to filter on
    pub tags: HashMap<String, Vec<String>>,
    /// Whether matching on tags should be case-insensitive
    pub tags_case_insensitive: bool,
}

impl Default for RepoListOpts {
    /// Build a default search
    fn default() -> Self {
        RepoListOpts {
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

impl RepoListOpts {
    /// Restrict the file search to start at a specific date
    ///
    /// # Arguments
    ///
    /// * `start` - The date to start listing repos from
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
    /// * `end` - The date to stop listing repos at
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

    /// List repos that match a specific tag
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

    /// Set for matching on tags to be case-insensitive
    #[must_use]
    pub fn tags_case_insensitive(mut self) -> Self {
        self.tags_case_insensitive = true;
        self
    }
}

/// The query params for listing results
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoListParams {
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

impl Default for RepoListParams {
    /// Create a default file list params
    fn default() -> Self {
        RepoListParams {
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

impl RepoListParams {
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

/// The query params for downloading a specific commit
#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoDownloadOpts {
    /// The specific branch, tag, or commit to download
    pub commitish: Option<String>,
    /// The kind of commitish to download
    #[serde(default = "CommitishKinds::all")]
    pub kinds: Vec<CommitishKinds>,
    /// The progress bar to update
    #[serde(skip_deserializing)]
    pub progress: Option<ProgressBar>,
}

impl Default for RepoDownloadOpts {
    // Build a default repo download params object
    fn default() -> Self {
        RepoDownloadOpts {
            commitish: None,
            kinds: Vec::default(),
            progress: None,
        }
    }
}

impl RepoDownloadOpts {
    /// Set the commitish to download
    ///
    /// # Arguments
    ///
    /// * `commitish` - The commitish to set
    pub fn commitish<T: Into<String>>(mut self, commitish: T) -> Self {
        self.commitish = Some(commitish.into());
        self
    }

    /// Add a commitish kind to check to be downloaded
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of commitish to add
    pub fn kinds(mut self, kind: CommitishKinds) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Add a progress to update with our download progress
    pub fn progress(mut self, progress: ProgressBar) -> Self {
        self.progress = Some(progress);
        self
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "scylla-utils", derive(scylla::DeserializeRow))]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
pub struct RepoCensusRow {
    pub group: String,
    pub year: i32,
    pub bucket: i32,
    pub count: i64,
}

#[cfg(feature = "scylla-utils")]
impl crate::models::CensusSupport for Repo {
    /// The type returned by our prepared statement
    type Row = RepoCensusRow;

    /// Build the prepared statement for getting partition count info
    async fn scan_prepared_statement(
        scylla: &scylla::client::session::Session,
        ns: &str,
    ) -> Result<scylla::statement::prepared::PreparedStatement, scylla::errors::PrepareError> {
        // build tags get partition count prepared statement
        scylla
            .prepare(format!(
                "SELECT group, year, bucket, count(*) \
                FROM {}.repos_list \
                WHERE token(group, year, bucket) >= ? AND token(group, year, bucket) <= ? \
                GROUP BY group, year, bucket",
                ns,
            ))
            .await
    }

    /// Get the count for this partition
    fn get_count(row: &RepoCensusRow) -> i64 {
        row.count
    }

    /// Get the bucket for this partition
    fn get_bucket(row: &RepoCensusRow) -> i32 {
        row.bucket
    }

    /// Build the count key for this partition
    fn count_key_from_row(namespace: &str, row: &Self::Row, grouping: i32) -> String {
        // build the key for this row
        format!(
            "{namespace}:census:repos:counts:{group}:{year}:{grouping}",
            namespace = namespace,
            group = row.group,
            year = row.year,
            grouping = grouping,
        )
    }

    /// Build the sorted set key for this census operation
    fn stream_key_from_row(namespace: &str, row: &Self::Row) -> String {
        format!(
            "{namespace}:census:repos:stream:{group}:{year}",
            namespace = namespace,
            group = row.group,
            year = row.year,
        )
    }
}
