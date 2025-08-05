//! Arguments for results-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use std::path::PathBuf;

use clap::Parser;
use thorium::models::OutputDisplayType;
use uuid::Uuid;

use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};

/// The commands to send to the results task handler
#[derive(Parser, Debug)]
pub enum Results {
    /// Get information on specific results
    #[clap(version, author)]
    Get(GetResults),
    /// Upload new results to Thorium
    #[clap(version, author)]
    Upload(UploadResults),
}

#[derive(Default, Debug, Clone, clap::ValueEnum)]
pub enum ResultsPostProcessing {
    #[default]
    Strip,
    Split,
    Full,
}

/// A command to get info on some reactions
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct GetResults {
    /// Any groups to restrict retrieved results to;
    /// if no groups are specified, results will be retrieved from all groups
    #[clap(short = 'G', long)]
    pub results_groups: Vec<String>,
    /// The tools to get results from
    #[clap(long, value_delimiter = ',')]
    pub tools: Vec<String>,
    /// How to process the results after downloading
    ///
    /// "strip" removes results metadata, "split" saves metadata to a separate file,
    /// and "full" leaves metadata and results in one combined file
    #[clap(long, value_enum, default_value_t, ignore_case = true)]
    pub post_processing: ResultsPostProcessing,
    /// Save results in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
    /// Any specific files to get results for
    #[clap(short, long)]
    pub files: Vec<String>,
    /// The path to a file containing a list of file SHA256's to download results for,
    /// delimited by newlines
    #[clap(long, verbatim_doc_comment)]
    pub file_list: Option<PathBuf>,
    /// Any specific repos + optionally commits to run these jobs on
    ///
    /// Note: Repo commits are formatted with a colon after the repo URL
    ///       (i.e. "<REPO-URL>:<COMMIT-HASH>)"
    #[clap(short, long)]
    pub repos: Vec<String>,
    /// Create reactions only for repos with the given search criteria (i.e. tags)
    #[clap(short = 'R', long)]
    pub repos_only: bool,
    /// The path to a file containing a list of repo URL's to create reactions for,
    /// delimited by newlines
    #[clap(long)]
    pub repo_list: Option<PathBuf>,
    /// Include repos in the search and run reactions on them
    #[clap(long)]
    pub include_repos: bool,
    /// Any groups to filter by when searching for samples/repos to get results from
    ///
    /// Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for samples/repos to get results from
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on sample/repo tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The delimiter character to use when splitting tags into key/values
    /// (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=")]
    pub delimiter: char,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///
    /// Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    ///          (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S")]
    pub date_fmt: String,
    /// Get results for all files with no search filter
    ///
    /// Note: This, combined with "--no-limit", will get results for ALL files to
    ///       which you have access. Be careful!
    #[clap(long, default_value = "false")]
    pub get_all: bool,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// The max number of total submissions to find in the search
    ///
    /// Note: Because one file may have several submissions (i.e. users upload the same file
    ///       to different groups), the number of results retrieved will likely be less than this limit
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Retrieve file results with no limit
    ///
    /// Note: Retrieving results based on tags with no limit can lead
    ///       to many results being retrieved. Be careful!
    #[clap(long, conflicts_with = "limit")]
    pub no_limit: bool,
    /// The number of results to get in one request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// The output directory to write these results too
    #[clap(short, long, default_value = "results")]
    pub output: String,
}

impl SearchParameterized for GetResults {
    fn has_targets(&self) -> bool {
        !self.files.is_empty()
            || self.file_list.is_some()
            || !self.repos.is_empty()
            || self.repo_list.is_some()
    }
    fn apply_to_all(&self) -> bool {
        self.get_all
    }
}
impl SearchSealed for GetResults {
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

/// A command to upload new results to Thorium
#[derive(Parser, Debug)]
pub struct UploadResults {
    // Any specific files to create results for
    pub targets: Vec<String>,
    /// The groups to add these result to
    #[clap(short = 'G', long, value_delimiter = ',')]
    pub result_groups: Vec<String>,
    /// The tool these results are for
    #[clap(short, long)]
    pub tool: String,
    /// The name of the file containing the results to display in the UI
    #[clap(short, long)]
    pub results: Option<String>,
    /// The display type to use when rendering these results
    #[clap(short, long, value_enum, ignore_case = true)]
    pub display_type: OutputDisplayType,
}
