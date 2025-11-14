//! Arguments for results-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use clap::Parser;
use clap::builder::NonEmptyStringValueParser;
use std::path::PathBuf;
use thorium::models::OutputDisplayType;
use uuid::Uuid;

use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};

/// The commands to send to the results task handler
#[derive(Parser, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Results {
    /// Get information on specific results
    #[clap(version, author)]
    Get(GetResults),
    /// Upload new results to Thorium
    #[command(
        about = "Upload new results to Thorium",
        long_about = r#"
Upload new results to Thorium

Examples:

# Upload a result for the file designated by SHA256;
# if a file, uploads file contents as a result;
# if a directory, uploads content from file designated by '--results' flag
# and uploads all other files as attachments
thorctl results upload --tools my-tool --display-type string ./abcd1234ef5678...

# Set a specific display type for the result (e.g. JSON)
thorctl results upload --tools my-tool --display-type json ./abcd1234ef5678...

# Upload multiple targets in one command
thorctl results upload --tools my-tool --display-type string ./sha1 ./sha2 ./sha3

# Upload a parent directory containing multiple SHA256 sub‑directories/files
#   ./results/
#   ├── a1b2c3d4e5f6...
#   │   └── results
#   ├── f6e5d4c3b2a1...
#   └── 1234567890ab...
#       └── results
thorctl results upload --tools my-tool --display-type string ./results

# Upload a SHA256 directory with per‑tool subdirectories
#   ./abcd1234ef5678...
#   ├── tool-a/
#   │   └── results
#   ├── tool-b/
#   │   └── results
#   └── tool-c/
#       └── results
thorctl results upload --display-type string ./abcd1234ef5678...

# Specify a custom results file name to upload as the main results (visible in the Web UI)
thorctl results upload --results my_result_file --display-type string ./abcd1234ef5678...

# Restrict results visibility to specific groups
thorctl results upload -G example-group -t my-tool --display-type string ./abcd1234ef5678...

# Perform a dry run to see what would be uploaded without actually uploading
thorctl results upload --display-type string --dry-run ./results
"#
    )]
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
    /// Note: Retrieiving results based on tags with no limit can lead
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
    fn get_search_params(&self) -> SearchParams<'_> {
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
    /// The files/directories containing results to upload; these should either be
    /// a parent directory containing sub-directories/files named as the SHA256
    /// for their respective file (e.g. 'results/<SHA256>') or the SHA256
    /// directories/files themselves (e.g. '<SHA256>').
    ///
    /// Files named as a SHA256 will be uploaded as a single result for the SHA256
    /// they are named for and associated with the tool(s) given by the `--tools/-t`
    /// flag.
    ///
    /// Directories named by SHA256 can either have results organized by the tools they
    /// should be associated with (e.g. '<SHA256>/my-tool/results') OR they can have
    /// results all together in the SHA256 directory (e.g. '<SHA256>/results'), in which
    /// case they will be associated to the tool(s) given by the `--tools/-t` flag.
    /// If any files are further nested within tool directories
    /// (e.g. '<SHA256>/my-tool/some-dir/results'), they will be uploaded recursively
    /// and all associated together.
    ///
    /// Any files/directories under the main target directory *not* named by SHA256
    /// will be ignored (e.g. 'results/my-file.txt')
    #[clap(required = true, value_parser = NonEmptyStringValueParser::new())]
    pub targets: Vec<String>,
    /// The groups to add these results to
    ///
    /// If no groups are provided, results will be visible to all groups that
    /// can view the file
    #[clap(short = 'G', long, value_delimiter = ',', value_parser = NonEmptyStringValueParser::new())]
    pub result_groups: Vec<String>,
    /// The tools these results are for
    ///
    /// Determines with which tools file not nested within tool directories will be
    /// associated (e.g. '<SHA256>/results'). Note that any tool may be given,
    /// even if the tool doesn't exist within the Thorium instance.
    #[clap(short, long, value_delimiter = ',', value_parser = NonEmptyStringValueParser::new())]
    pub tools: Vec<String>,
    /// The name of the file within results directories containing the main results
    /// to display in the UI
    ///
    /// All other files in the directory will be uploaded as result files. If any tool
    /// directory or the SHA256 directory is missing this main results file, an error
    /// will be logged and no results will be uploaded. To upload empty results, create
    /// an empty results file.
    #[clap(short, long, default_value = "results", value_parser = NonEmptyStringValueParser::new())]
    pub results: String,
    /// The display type to use when rendering results
    ///
    /// This is applied to *all* results for *all* tools
    #[clap(short, long, value_enum, ignore_case = true)]
    pub display_type: OutputDisplayType,
    /// Display which results will be uploaded and to which tools without actually uploading
    #[clap(long)]
    pub dry_run: bool,
}
