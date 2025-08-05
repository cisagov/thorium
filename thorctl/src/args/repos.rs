//! Arguments for repo-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use std::{collections::HashMap, path::PathBuf};

use clap::builder::NonEmptyStringValueParser;
use clap::{Parser, ValueEnum};
use thorium::models::{CommitishKinds, RepoCheckout, RepoDependencyRequest, RepoRequest};
use thorium::Error;
use uuid::Uuid;

use super::traits::describe::{DescribeCommand, DescribeSealed};
use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};
use crate::handlers::repos::RepoBuild;
use crate::utils;

/// The commands to send to the repos task handler
#[derive(Parser, Debug)]
pub enum Repos {
    /// Get a list of repos and their details
    #[clap(version, author)]
    Get(GetRepos),
    /// Describe a specific repo, displaying all details
    Describe(DescribeRepos),
    /// List the commitishes (commits, branches, git tags) for a repo
    Commits(ListCommits),
    /// Ingest repositories by their url
    #[clap(version, author)]
    Ingest(IngestRepos),
    /// Update an already ingested repo
    #[clap(version, author)]
    Update(UpdateRepos),
    /// Download zipped repositories
    #[clap(version, author)]
    Download(DownloadRepos),
    /// Compile a list of repos
    #[clap(version, author)]
    Compile(CompileRepos),
    /// List the contributors in a repo
    #[clap(version, author)]
    Contributors(ContributorsRepos),
}

/// A command to get repo info
#[derive(Parser, Debug)]
pub struct GetRepos {
    /// Any groups to filter by when searching for repos
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for repos
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on repo tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///     Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///     (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S", verbatim_doc_comment)]
    pub date_fmt: String,
    /// The max number of repos to list
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving files
    ///     Note: This can lead to retrieving info for many millions of files
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of repos to retrieve per request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// Print the repos in URL alphabetical order rather than by group, then creation date
    ///     Note: Sorting can require many system resources for large amounts of repos
    #[clap(short, long, verbatim_doc_comment)]
    pub alpha: bool,
}

impl SearchParameterized for GetRepos {
    fn has_targets(&self) -> bool {
        // GetRepos should never have specific targets
        false
    }
    fn apply_to_all(&self) -> bool {
        // GetRepos has no explicit "--all" option
        false
    }
}
impl SearchSealed for GetRepos {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

#[derive(Parser, Debug)]
pub struct DescribeRepos {
    /// Any specific repos to describe
    pub repos: Vec<String>,
    /// The path to a file containing a list of repo URL's to describe separated by newlines
    #[clap(short, long)]
    pub repo_list: Option<PathBuf>,
    /// The path to the file to write output to; if not provided, details will be output to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Output details in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
    /// Any groups to filter by when searching for repos to describe
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for repos to describe
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on repo tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///     Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///     (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S", verbatim_doc_comment)]
    pub date_fmt: String,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// The max number of repos to describe
    #[clap(short, long, default_value = "10000")]
    pub limit: usize,
    /// Describe repos with no limit
    #[clap(long, conflicts_with = "limit", verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of repos to describe per API request
    #[clap(short, long, default_value = "100")]
    pub page_size: usize,
    /// Refrain from setting any filters when describing repos, attempting to describe all repos the user can view
    ///     Note: This will override any other search parameters set except for those associated with limit.
    ///           When combined with `--no-limit`, this will describe all files to which the current user
    ///           has access.
    #[clap(long, verbatim_doc_comment)]
    pub describe_all: bool,
}

impl SearchSealed for DescribeRepos {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

impl SearchParameterized for DescribeRepos {
    fn has_targets(&self) -> bool {
        !self.repos.is_empty() || self.repo_list.is_some()
    }

    fn apply_to_all(&self) -> bool {
        self.describe_all
    }
}

impl DescribeSealed for DescribeRepos {
    type Data = thorium::models::Repo;

    type Target<'a> = &'a str;

    type Cursor = thorium::models::Cursor<Self::Data>;

    fn raw_targets(&self) -> &[String] {
        &self.repos
    }

    fn condensed(&self) -> bool {
        self.condensed
    }

    fn out_path(&self) -> Option<&PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&PathBuf> {
        self.repo_list.as_ref()
    }

    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error> {
        // no parsing is required so just return the raw target
        Ok(raw)
    }

    async fn retrieve_data<'a>(
        &self,
        target: Self::Target<'a>,
        thorium: &thorium::Thorium,
    ) -> Result<Self::Data, thorium::Error> {
        thorium.repos.get(target).await
    }

    async fn retrieve_data_search(
        &self,
        thorium: &thorium::Thorium,
    ) -> Result<Vec<Self::Cursor>, thorium::Error> {
        Ok(vec![
            thorium.repos.list_details(&self.build_repo_opts()?).await?,
        ])
    }
}

impl DescribeCommand for DescribeRepos {}

/// Provide a default temp directory
fn default_temp_ingest_path() -> PathBuf {
    #[cfg(unix)]
    return PathBuf::from("/tmp/repo_ingests");
    #[cfg(target_os = "windows")]
    return std::env::temp_dir().join("repo_ingests");
}

/// A command to ingest repos directly from git repos
#[derive(Parser, Debug, Clone)]
pub struct IngestRepos {
    /// The urls to the repos to ingest
    pub urls: Vec<String>,
    /// The paths to already cloned repos to ingest from disk
    #[clap(short, long)]
    pub local: Vec<PathBuf>,
    /// The groups to add these repos to
    #[clap(short = 'G', long, value_delimiter = ',', required = true, value_parser = NonEmptyStringValueParser::new())]
    pub add_groups: Vec<String>,
    /// The tags to add to any repos ingested where key/value is separated by a delimiter
    #[clap(short = 'T', long)]
    pub add_tags: Vec<String>,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// Any files with lists of repos to ingest
    #[clap(short, long)]
    pub repos_list: Vec<PathBuf>,
    /// Where to temporarily store zipped repo files
    #[clap(long, default_value = default_temp_ingest_path().into_os_string())]
    pub temp: PathBuf,
    /// The branches to limit our commit crawlers too
    #[clap(short, long, value_delimiter = ',')]
    pub branches: Vec<String>,
    /// The remote git tags to limit our commit crawlers too
    #[clap(long)]
    pub remote_tags: Vec<String>,
    /// The branches to prefer when detecting default checkout behavior
    ///    These will be prioritized in the order they are specified.
    #[clap(
        short,
        long,
        value_delimiter = ',',
        verbatim_doc_comment,
        default_values = ["main", "Main", "master", "Master"],
    )]
    pub preferred_checkout_branches: Vec<String>,
    /// Ignore any repo data already ingested into Thorium and pull everything from source
    #[clap(short, long, default_value_t = false)]
    pub no_cache: bool,
}

impl IngestRepos {
    /// Build a [`RepoRequest`] from a cleaned remote URL and command options
    ///
    /// # Arguments
    ///
    /// * `url` - The cleaned repo remote URL
    pub fn build_req(&self, url: &str, default_checkout: Option<RepoCheckout>) -> RepoRequest {
        let mut req = RepoRequest::new(url, self.add_groups.clone(), default_checkout);
        // crawl over and split any tags
        for combined in &self.add_tags {
            // split this combined tag by our delimiter
            let split = combined.split(self.delimiter).collect::<Vec<&str>>();
            // add each of the split values
            for value in split.iter().skip(1) {
                req = req.tag(split[0], *value);
            }
        }
        req
    }

    /// Get the references to crawl
    pub fn references(&self) -> Vec<String> {
        // create our vector to store our references
        let mut refs = Vec::with_capacity(self.branches.len() + self.remote_tags.len());
        // build our branch references
        refs.extend(
            self.branches
                .iter()
                .map(|name| format!("refs/remotes/origin/{name}")),
        );
        // build our tag references
        refs.extend(
            self.remote_tags
                .iter()
                .map(|name| format!("refs/tags/{name}")),
        );
        refs
    }
}

impl From<&UpdateRepos> for IngestRepos {
    /// Create an [`IngestRepos`] cmd from an [`UpdateRepos`] one
    ///
    /// # Arguments
    ///
    /// * `update` - The update command to build our ingest command from
    fn from(update: &UpdateRepos) -> IngestRepos {
        IngestRepos {
            urls: update.urls.clone(),
            local: Vec::default(),
            add_groups: update.add_groups.clone(),
            add_tags: update.add_tags.clone(),
            delimiter: update.delimiter,
            repos_list: update.files.clone(),
            temp: update.temp.clone(),
            branches: update.branches.clone(),
            remote_tags: update.remote_tags.clone(),
            preferred_checkout_branches: update.preferred_checkout_branches.clone(),
            no_cache: update.no_cache,
        }
    }
}

/// A command to update already ingested repos
#[derive(Parser, Debug, Clone)]
pub struct UpdateRepos {
    /// The urls to the repos to update
    pub urls: Vec<String>,
    /// The groups to add these repos to; if none are provided, repos will be updated
    /// in all their groups
    #[clap(short = 'G', long, value_delimiter = ',', value_parser = NonEmptyStringValueParser::new())]
    pub add_groups: Vec<String>,
    /// The tags to add to any repos ingested where key/value is separated by a delimiter
    #[clap(short = 'T', long)]
    pub add_tags: Vec<String>,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// Any files with lists of repos to ingest
    #[clap(short, long)]
    pub files: Vec<PathBuf>,
    /// Where to temporarily store zipped repo files
    #[clap(long, default_value = "/tmp/repo_ingests")]
    pub temp: PathBuf,
    /// The branches to limit our commit crawlers too
    #[clap(short, long, value_delimiter = ',')]
    pub branches: Vec<String>,
    /// The remote git tags to limit our commit crawlers too
    #[clap(long)]
    pub remote_tags: Vec<String>,
    /// The branches to prefer when detecting default checkout behavior
    ///    These will be prioritized in the order they are specified.
    #[clap(
        short = 'P',
        long,
        value_delimiter = ',',
        verbatim_doc_comment,
        default_values = ["main", "Main", "master", "Master", "release", "Release", "dev",  "Dev"],
    )]
    pub preferred_checkout_branches: Vec<String>,
    /// Any groups to filter by when searching for repos to update
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for repos to update
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on repo tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///     Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///     (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S", verbatim_doc_comment)]
    pub date_fmt: String,
    /// The max number of repos to update when listing
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving repos
    ///     Note: This can lead to updating info for many millions of repos
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of repos to retrieve per request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// Update repos for all repos with no search filter
    ///     Note: This, combined with '--no-limit', will update ALL repos
    ///           to which you have access. Be careful!
    #[clap(long, default_value = "false", verbatim_doc_comment)]
    pub update_for_all: bool,
    /// Ignore any repo data already ingested into Thorium and pull everything from source
    #[clap(short, long, default_value_t = false)]
    pub no_cache: bool,
}

impl SearchParameterized for UpdateRepos {
    fn has_targets(&self) -> bool {
        !self.urls.is_empty() || !self.files.is_empty()
    }
    fn apply_to_all(&self) -> bool {
        self.update_for_all
    }
}
impl SearchSealed for UpdateRepos {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

/// The organization structure to use when downloading repos
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RepoDownloadOrganization {
    /// Download repos to a single folder organized by repo name
    #[default]
    Simple,
    /// Download repos and separate them by source (github/gitlab) and group/user
    Provenance,
}

impl std::fmt::Display for RepoDownloadOrganization {
    /// write our [`FileDownloadOrganization`] to this formatter
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl RepoDownloadOrganization {
    /// Cast a [`RepoDownloadOrganization`] to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            RepoDownloadOrganization::Simple => "Simple",
            RepoDownloadOrganization::Provenance => "Provenance",
        }
    }
}

/// A command to upload repos and their commits to Thorium
#[derive(Parser, Debug, Clone)]
pub struct DownloadRepos {
    // Any specific repos/commitishes to download
    pub repos: Vec<String>,
    /// Any files containing the repos to download delimited by newlines
    #[clap(short, long, value_delimiter = ',')]
    pub files: Vec<String>,
    /// Any groups to filter by when searching for repos
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for repos
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on repo tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///     Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///     (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S", verbatim_doc_comment)]
    pub date_fmt: String,
    /// The max number of repos to list
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving files
    ///     Note: This can lead to retrieving info for many millions of files
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of repos to retrieve per request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// Skip uncarting any downloaded repos
    #[clap(short, long)]
    pub carted: bool,
    /// The organizational file structure to use when downloading repos
    #[clap(long, default_value_t, ignore_case = true)]
    pub organization: RepoDownloadOrganization,
    /// Whether to remove git data when uncarting repos
    #[clap(long)]
    pub prune_git: bool,
    /// The extensions to prune when uncarting repos
    #[clap(long)]
    pub prune_extensions: Vec<String>,
    /// The extensions to keep when uncarting repos
    #[clap(long)]
    pub retain_extensions: Vec<String>,
    /// The minium size of a files to keep when uncarting repos
    #[clap(long)]
    pub retain_min_size: Option<String>,
    /// The max size of files to keep when uncarting repos
    #[clap(long)]
    pub retain_max_size: Option<String>,
    /// Skip any already existing repos
    #[clap(long)]
    pub skip_existing: bool,
    /// The folder to write downloaded repos too
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

impl DownloadRepos {
    /// Convert  our minimum size to bytes
    pub fn convert_retain_sizes(&self) -> Result<(Option<u64>, Option<u64>), Error> {
        // convert our minimum size if we have one
        let min = match &self.retain_min_size {
            Some(min) => Some(utils::convert_size_to_bytes(min)?),
            None => None,
        };
        // convert our maximum size if we have one
        let max = match &self.retain_max_size {
            Some(max) => Some(utils::convert_size_to_bytes(max)?),
            None => None,
        };
        Ok((min, max))
    }
}

impl SearchParameterized for DownloadRepos {
    fn has_targets(&self) -> bool {
        // GetRepos should never have specific targets
        false
    }
    fn apply_to_all(&self) -> bool {
        // GetRepos has no explicit "--all" option
        false
    }
}
impl SearchSealed for DownloadRepos {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

/// A command to compile certain repos in Thorium
#[derive(Parser, Debug)]
pub struct CompileRepos {
    /// This group allows the user set both repo and list, but also requires at least one is set
    #[clap(flatten)]
    pub group: CompileReposGroup,
    /// A branch, commit, or tag in the given <REPO> to compile
    #[clap(short, long, value_delimiter = ',', requires = "repo")]
    pub commitish: Vec<String>,
    /// the kind of commitish in the given <REPO> to compile
    #[clap(short, long)]
    pub kind: Option<CommitishKinds>,
    /// Any dependencies that need to be installed before building <REPO>
    #[clap(short, long, value_delimiter = ',', requires = "repo")]
    pub dependencies: Vec<String>,
    /// Any flags to set when building <REPO>
    #[clap(last = true, requires = "repo")]
    pub flags: Vec<String>,
    /// Batch these reactions together
    #[clap(short, long)]
    pub batch: Option<String>,
    /// Watch all spawned reactions progress and automatically batch them
    #[clap(short = 'W', long)]
    pub watch: bool,
}

impl CompileRepos {
    /// If batch or watch is enabled then get our batch name
    pub fn batch_name(&self) -> Option<String> {
        match (&self.batch, &self.watch) {
            (Some(batch), _) => Some(batch.to_owned()),
            (None, true) => Some(Uuid::new_v4().to_string()),
            _ => None,
        }
    }

    /// Get the set of builds the users is requesting to build
    pub async fn get_builds(&self) -> Result<Vec<RepoBuild>, Error> {
        // try to get the builds from file if provided
        let mut builds = if let Some(ref list) = self.group.list {
            RepoBuild::load(list).await?
        } else {
            Vec::with_capacity(std::cmp::max(self.commitish.len(), 1))
        };
        // if the user gave a repo, add that repo to the list of repo builds
        if let Some(ref repo) = self.group.repo {
            if self.commitish.is_empty() {
                // if no commits are given, just add one item to the list of repo build
                builds.push(RepoBuild {
                    repo: repo.clone(),
                    commitish: None,
                    dependencies: self.dependencies.clone(),
                    flags: self.flags.clone(),
                    cc: None,
                    cxx: None,
                    tags: HashMap::new(),
                    kind: None,
                });
            } else {
                // Add each commitish to the list of repo with the same dependencies and flags
                for commitish in &self.commitish {
                    builds.push(RepoBuild {
                        repo: repo.clone(),
                        commitish: Some(commitish.clone()),
                        kind: self.kind,
                        dependencies: self.dependencies.clone(),
                        flags: self.flags.clone(),
                        cc: None,
                        cxx: None,
                        tags: HashMap::new(),
                    });
                }
            }
        }
        Ok(builds)
    }
}

/// This struct is required to create an `ArgGroup` and set the `required` and `multiple` fields
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = true)]
pub struct CompileReposGroup {
    /// A repo to add to the list of repo to compile
    #[arg(short, long)]
    pub repo: Option<String>,
    /// The path to a list of repos to compile and any flags to use
    pub list: Option<String>,
}

/// A command to compile certain repos in Thorium
#[derive(Parser, Debug)]
pub struct ContributorsRepos {
    /// The repo in Thorium to get contributors for
    pub repo: String,
    /// Whether or not to tag this repo with its contributors
    #[clap(short, long)]
    pub tag: bool,
}

/// A command to list the commitishes for a repo in Thorium
#[derive(Parser, Debug)]
pub struct ListCommits {
    /// The repo we're listing commitishes for
    pub repo: String,
    /// The kinds of commitishes to list (e.g. `--kinds branch,tag`) [default: all]
    #[clap(short, long, value_delimiter = ',')]
    pub kinds: Vec<CommitishKinds>,
    /// Any groups to filter by when listing commitishes
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///
    /// Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    /// (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S")]
    pub date_fmt: String,
    /// The max number of commitishes to list
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving commitishes
    #[clap(long)]
    pub no_limit: bool,
    /// The number of commitishes to retrieve per request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepoTarget {
    /// The repo to pull for this reaction
    pub url: String,
    /// The branch, commit, or tag to checkout when pulling this repo
    pub commitish: Option<String>,
    /// The kind of commitish valid for this target
    pub kind: Option<CommitishKinds>,
}

impl RepoTarget {
    /// Create a new [`RepoTarget`] without a commitish
    ///
    /// # Arguments
    ///
    /// * `url` - The repo url
    pub fn new<T: Into<String>>(url: T) -> Self {
        Self {
            url: url.into(),
            commitish: None,
            kind: None,
        }
    }
}

impl TryFrom<&str> for RepoTarget {
    type Error = Error;
    /// Try to parse a repo target from a string
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut split = value.split(':');
        let url = split.next();
        let commitish_or_kind = split.next();
        let commitish = split.next();
        match (url, commitish_or_kind, commitish) {
            (Some(url), Some(kind), Some(commitish)) => {
                // get our kind from this str
                let kind = match CommitishKinds::from_str(kind, true) {
                    Ok(kind) => kind,
                    Err(_) => {
                        // build our error message
                        let msg = format!("Valid kinds are Branch, Tag, Commit - {}", value);
                        return Err(Error::new(msg));
                    }
                };
                Ok(RepoTarget {
                url: url.to_owned(),
                commitish: Some(commitish.to_owned()),
                kind: Some(kind),
            })
            },
            (Some(url), Some(commitish), None) => Ok(RepoTarget {
                url: url.to_owned(),
                commitish: Some(commitish.to_owned()),
                kind: None
            }),
            (Some(url), None, None) => Ok(RepoTarget {
                url: url.to_owned(),
                commitish: None,
                kind: None
            }),
            _ => Err(Self::Error::new(
                "Invalid repo format: repos must include only the repo's url and \
                optionally a commitish separated by a colon (i.e. '<REPO_URL>:<COMMITISH_KIND>:<BRANCH/COMMIT/TAG>'). \
                If the commitish is ambiguous (e.g. a branch and a tag have the same name), \
                the commitish kind must also be specified before the commitish separated with a colon. \
                An example of this would be github.com/rust-lang/rust:branch:stable."
                    .to_string(),
            )),
        }
    }
}

impl TryFrom<&String> for RepoTarget {
    type Error = Error;
    fn try_from(value: &String) -> Result<Self, Self::Error> {
        Self::try_from(value.as_str())
    }
}

impl From<RepoTarget> for RepoDependencyRequest {
    fn from(target: RepoTarget) -> Self {
        RepoDependencyRequest {
            url: target.url,
            commitish: target.commitish,
            kind: target.kind,
        }
    }
}
