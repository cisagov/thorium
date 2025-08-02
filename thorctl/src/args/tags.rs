//! Arguments for tag-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use std::path::PathBuf;

use clap::{builder::NonEmptyStringValueParser, Parser};

use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};

/// The commands to send to the tags task handler
#[derive(Parser, Debug)]
pub enum Tags {
    /// Get a list of tags for a file/repo
    #[clap(version, author)]
    Get(GetTags),
    /// Adds tags to files/repos
    #[clap(version, author)]
    Add(AddTags),
    /// Delete tags from files/repos
    #[clap(version, author)]
    Delete(DeleteTags),
}

/// A command to get a list of tags for a file or repo
#[derive(Parser, Debug)]
pub struct GetTags {
    /// The sample SHA256 of the file or URL of the repo to list tags from
    #[clap(value_parser = NonEmptyStringValueParser::new())]
    pub sha256_or_repo: String,
    /// The path to a file to write the tags to
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Print the resulting tags JSON in a condensed format (no formatting/whitespace)
    #[clap(short, long)]
    pub condensed: bool,
}

/// A command to add tags to files/repos
#[derive(Parser, Debug)]
#[allow(clippy::struct_field_names)]
#[allow(clippy::struct_excessive_bools)]
pub struct AddTags {
    /// The tags to add to files/repos where key/value is separated by a delimiter
    #[clap(short = 'T', long, required = true)]
    pub add_tags: Vec<String>,
    /// The delimiter character to use when splitting tags into key/values
    /// (applies to both `--add-tags` and `--tags`)
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// The groups the tags should be visible to
    ///     Note: If no groups are given, the tags will be visible to all of the objects' groups
    #[clap(short = 'G', long, value_delimiter = ',', verbatim_doc_comment)]
    pub add_groups: Vec<String>,
    // Any specific files to tag
    //    Note: Explicitly specified files will always be tagged, even if they
    //          don't match the search criteria (i.e. don't have matching tags)
    #[clap(short, long, verbatim_doc_comment)]
    pub files: Vec<String>,
    // Any specific repos to tag
    //    Note: Explicitly specified repos will always be tagged, even if they
    //          don't match the search criteria (i.e. don't have matching tags)
    #[clap(short, long, verbatim_doc_comment)]
    pub repos: Vec<String>,
    /// Tag only repos matching the given search criteria
    #[clap(short = 'R', long)]
    pub repos_only: bool,
    /// Add tags to repos matching the given search criteria in addition to files
    #[clap(long)]
    pub include_repos: bool,
    /// Any groups to filter by when searching for files/repos to add tags to
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for samples/repos to add tags to
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on sample/repo tags should be case-insensitive
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
    /// Add the given tag(s) to all files/repos with no search filter
    ///     Note: This, combined with "--no-limit", will add the given tag(s) to ALL files/repos
    ///           to which you have access. Be careful!
    #[clap(long, default_value = "false", verbatim_doc_comment)]
    pub add_to_all: bool,
    /// The maximum number of objects to tag
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit on the number of objects that are tagged
    ///     Note: Tagging with no limit can lead to many more tags being created than
    ///           expected. Be careful!
    #[clap(long, conflicts_with = "limit", verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of objects to tag per request
    #[clap(long, default_value = "50")]
    pub page_size: usize,
}

impl SearchParameterized for AddTags {
    fn has_targets(&self) -> bool {
        !self.files.is_empty() || !self.repos.is_empty()
    }
    fn apply_to_all(&self) -> bool {
        self.add_to_all
    }
}
impl SearchSealed for AddTags {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: None,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

/// A command to add tags to files/repos
#[derive(Parser, Debug)]
#[allow(clippy::struct_field_names)]
#[allow(clippy::struct_excessive_bools)]
pub struct DeleteTags {
    /// The tags to delete from files/repos where key/value is separated by a delimiter
    #[clap(short = 'T', long, required = true)]
    pub delete_tags: Vec<String>,
    /// The delimiter character to use when splitting tags into key/values
    /// (applies to both `--delete-tags` and `--tags`)
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// The groups the tags should be deleted from
    ///     Note: If no groups are given, the tags will be deleted from all of the objects' groups
    #[clap(short = 'G', long, value_delimiter = ',', verbatim_doc_comment)]
    pub delete_groups: Vec<String>,
    // Any specific files to remove tags from
    //    Note: Explicitly specified files will always have their tags removed, even if they
    //          don't match the search criteria (i.e. don't have matching tags)
    #[clap(short, long, verbatim_doc_comment)]
    pub files: Vec<String>,
    // Any specific repos to remove tags from
    //    Note: Explicitly specified repos will always have their tags removed, even if they
    //          don't match the search criteria (i.e. don't have matching tags)
    #[clap(short, long, verbatim_doc_comment)]
    pub repos: Vec<String>,
    /// Delete tags only from repos matching the given search criteria
    #[clap(short = 'R', long)]
    pub repos_only: bool,
    /// Delete tags from repos matching the given search criteria in addition to files
    #[clap(long)]
    pub include_repos: bool,
    /// Any groups to filter by when searching for files/repos to delete tags from
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for samples/repos to delete tags from
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on sample/repo tags should be case-insensitive
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
    /// Delete the given tag(s) from all files/repos with no search filter
    ///     Note: This, combined with "--no-limit", will delete the given tag(s) from ALL files/repos
    ///           to which you have access. Be careful!
    #[clap(long, default_value = "false", verbatim_doc_comment)]
    pub delete_from_all: bool,
    /// The maximum number of objects to delete tags from
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit on the number of objects that have their tags removed
    ///     Note: Deleting tags with no limit can lead to many more tags being deleted than
    ///           expected. Be careful!
    #[clap(long, conflicts_with = "limit", verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of objects to delete tags from per request
    #[clap(long, default_value = "50")]
    pub page_size: usize,
}

impl SearchParameterized for DeleteTags {
    fn has_targets(&self) -> bool {
        !self.files.is_empty() || !self.repos.is_empty()
    }
    fn apply_to_all(&self) -> bool {
        self.delete_from_all
    }
}
impl SearchSealed for DeleteTags {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: None,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}
