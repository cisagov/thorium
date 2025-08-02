//! Arguments for file-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use clap::Parser;
use std::path::{Path, PathBuf};
use thorium::models::{OriginRequest, SampleRequest};
use uuid::Uuid;

use super::traits::describe::{DescribeCommand, DescribeSealed};
use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};

/// The commands to send to the files task handler
#[derive(Parser, Debug)]
pub enum Files {
    /// Upload some files and/or directories to Thorium
    #[clap(version, author)]
    Upload(UploadFiles),
    /// Download files from Thorium
    #[clap(version, author)]
    Download(DownloadFiles),
    /// Get information on files
    #[clap(version, author)]
    Get(GetFiles),
    /// Describe a particular file, displaying all details
    #[clap(version, author)]
    Describe(DescribeFiles),
    /// Delete file submissions
    #[clap(version, author)]
    Delete(DeleteFiles),
}

/// A command to upload some files to Thorium
#[derive(Parser, Debug)]
pub struct UploadFiles {
    /// The files and or folders to upload
    pub targets: Vec<String>,
    /// The groups to upload these files to
    #[clap(short = 'G', long, value_delimiter = ',', required = true)]
    pub file_groups: Vec<String>,
    /// The tags to add to any files uploaded where key/value is separated by a delimiter
    #[clap(short = 'T', long)]
    pub file_tags: Vec<String>,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=", verbatim_doc_comment)]
    pub delimiter: char,
    /// Any pipelines to immediately spawn for the files that are uploaded;
    /// pipelines are specified by their name + group, separated with ":"
    /// (i.e. <PIPELINE1>:<GROUP1>,<PIPELINE2>:<GROUP2>)
    #[clap(short, long, value_delimiter = ',')]
    pub pipelines: Option<Vec<String>>,
    /// Any regular expressions to use to determine which files to upload
    #[clap(short, long)]
    pub filter: Vec<String>,
    /// Any regular expressions to use to determine which files to skip
    #[clap(short, long)]
    pub skip: Vec<String>,
    /// Apply include/skip filters to directories as well as files
    #[clap(short = 'F', long, default_value = "false")]
    pub filter_dirs: bool,
    /// Include hidden directories/files
    #[clap(long, default_value = "false")]
    pub include_hidden: bool,
    /// The tags keys to use for each folder name starting at the root of the specified targets
    #[clap(long)]
    pub folder_tags: Vec<String>,
}

impl UploadFiles {
    /// Try to extract any origins from a file name
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to extract
    fn extract_origin(path: &Path) -> Option<OriginRequest> {
        // if we can extract a filename then check if it matches any of the origin endings
        if let Some(Some(name)) = path.file_name().map(|name| name.to_str()) {
            // if this sample ends with _unpacked then upload it with the unpacked origin
            if name.ends_with("_unpacked") {
                let end = name.len() - "_unpacked".len();
                Some(OriginRequest::unpacked(&name[0..end], None))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Build a sample upload request for a specific path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to upload
    pub fn build_req(&self, path: &Path) -> SampleRequest {
        let mut req = SampleRequest::new(path, self.file_groups.clone());
        // crawl over and split any tags
        for combined in &self.file_tags {
            // split this combined tag by our delimiter
            let split = combined.split(self.delimiter).collect::<Vec<&str>>();
            // add each of the split values
            for value in split.iter().skip(1) {
                req = req.tag(split[0], *value);
            }
        }
        // get our parent path if one exists
        if let Some(parent) = path.parent() {
            // crawl over any folder_tags and our path and build the tags
            for (key, value) in self.folder_tags.iter().zip(parent.iter().skip(1)) {
                // try to cast the folder name to a utf-8 string
                if let Some(name) = value.to_str() {
                    req = req.tag(key, name);
                }
            }
        }
        // extract any origins from this path
        req.origin = UploadFiles::extract_origin(path);
        req
    }
}

/// The organization structure to use when downloading files
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FileDownloadOrganization {
    /// Download files to a single folder organized by file name
    #[default]
    Simple,
    /// Download files and separate them by origin
    Provenance,
}

impl std::fmt::Display for FileDownloadOrganization {
    /// write our [`FileDownloadOrganization`] to this formatter
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FileDownloadOrganization {
    /// Cast a [`FileDownloadOrganization`] to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            FileDownloadOrganization::Simple => "Simple",
            FileDownloadOrganization::Provenance => "Provenance",
        }
    }

    /// Whether this organization structure may require copying data to more paths on disk
    pub fn may_copy(&self) -> bool {
        match self {
            FileDownloadOrganization::Simple => false,
            FileDownloadOrganization::Provenance => true,
        }
    }
}

/// A command to download some files from Thorium
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct DownloadFiles {
    /// The samples to download
    pub sha256s: Vec<String>,
    /// Download files uncarted rather than leaving them in the benign "Cart" format
    #[clap(short, long)]
    pub uncarted: bool,
    /// The path to download these files to
    #[clap(short, long)]
    pub output: Option<String>,
    /// Refrain from adding the ".cart" extension to the downloaded file
    /// (uncarted files never have the extension)
    #[clap(long, conflicts_with = "uncarted")]
    pub no_extension: bool,
    /// Any groups to filter by when searching for files
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for files
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on tags should be case-insensitive
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
    /// Download all files using no search filters
    ///     Note: This, combined with `--no-limit`, will download ALL files from Thorium
    ///           to which you have access. Be careful!
    #[clap(long, default_value = "false", verbatim_doc_comment)]
    pub download_all: bool,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// The max number of total submissions to find in the search
    ///     Note: Because one file may have several submissions (i.e. users upload the same file
    ///           to different groups), the number of files downloaded will likely be less than this limit
    #[clap(short, long, default_value = "50", verbatim_doc_comment)]
    pub limit: usize,
    /// Refrain from setting a limit when downloading files
    ///     Note: This can lead to downloading many millions of files
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of file to find in one request
    #[clap(short, long, default_value = "30")]
    pub page_size: usize,
    /// Try to give any downloaded files human friendly names
    #[clap(short, long)]
    pub friendly: bool,
    /// The tags to add to this file name when generating human friendly names
    #[clap(short, long)]
    pub nametags: Vec<String>,
    /// The organizational file structure to use when downloading repos
    #[clap(long, default_value_t, ignore_case = true)]
    pub organization: FileDownloadOrganization,
}

impl SearchParameterized for DownloadFiles {
    fn has_targets(&self) -> bool {
        !self.sha256s.is_empty()
    }
    fn apply_to_all(&self) -> bool {
        self.download_all
    }
}
impl SearchSealed for DownloadFiles {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
            tags_case_insensitive: self.tags_case_insensitive,
        }
    }
}

/// A command to get info on some files to Thorium
#[derive(Parser, Debug)]
pub struct GetFiles {
    /// Any groups to filter by when searching for files
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for files (<Key>=<Value>)
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The delimiter character to use when splitting tags into key/values
    ///    (i.e. <Key>=<VALUE>)
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
    /// The max number of file to list
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving files
    ///     Note: This can lead to retrieving info for many millions of files
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of file to list in one request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
}

impl SearchParameterized for GetFiles {
    fn has_targets(&self) -> bool {
        // GetFiles should never have specific targets
        false
    }
    fn apply_to_all(&self) -> bool {
        // GetFiles has no explicit "--all" option
        false
    }
}
impl SearchSealed for GetFiles {
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

/// A command to delete file submissions from Thorium
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct DeleteFiles {
    /// The SHA256s + optionally submissions to delete that overrides all search options
    ///     Note: each target should contain a SHA256 and 0 or more submission ID's separated by colons
    ///           (i.e. "<SHA256>:<SUBMISSION1>:<SUBMISSION2>")
    #[clap(verbatim_doc_comment)]
    pub targets: Vec<String>,
    /// Any users (submitters) to filter by when searching for files to delete;
    /// if no user is set, the current user is selected
    ///     Note: Only admins and group owners/managers can delete other users' submissions
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub users: Vec<String>,
    /// Delete samples/submissions from all users
    /// (in other words, don't filter files by their submitters)
    ///     Note: Only admins and group owners/managers can delete other users' submissions
    #[clap(long, conflicts_with = "users", verbatim_doc_comment)]
    pub all_users: bool,
    /// Any groups to filter by when searching for files to delete
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for files to delete
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on tags should be case-insensitive
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
    /// The max number of files to delete
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Delete files with no limit
    ///     Note: Deleting files based on tags with no limit can lead
    ///           to many files being deleted inadvertently. Be careful!
    #[clap(long, conflicts_with = "limit", verbatim_doc_comment)]
    pub no_limit: bool,
    /// The number of file to delete in one request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// Refrain from setting any filters when deleting files, attempting to delete any files the user can view
    ///     Note: This will override any other search parameters set except for those associated with limit.
    ///           When combined with `--no-limit`, this will attempt to delete all files to which the current user
    ///           has access. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub delete_all: bool,
    /// Allow deletion of files by supplying general search parameters (groups, tags, etc.)
    ///     Note: This can potentially delete ALL files in a Thorium instance. Be careful!
    #[clap(short, long, verbatim_doc_comment)]
    pub force: bool,
}

impl SearchParameterized for DeleteFiles {
    fn has_targets(&self) -> bool {
        !self.targets.is_empty()
    }
    fn apply_to_all(&self) -> bool {
        self.delete_all
    }
}
impl SearchSealed for DeleteFiles {
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

/// A command to describe a particular image in full
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct DescribeFiles {
    /// Any specific file SHA256's to describe
    pub files: Vec<String>,
    /// The path to the file containing a list of SHA256's to describe separated by newlines
    #[clap(short, long)]
    pub file_list: Option<PathBuf>,
    /// The path to the file to write output to; if not provided, details will be output to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Output details in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
    /// Any groups to filter by when searching for files to describe
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for files to describe
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on tags should be case-insensitive
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
    /// The max number of files to describe
    #[clap(short, long, default_value = "10000")]
    pub limit: usize,
    /// Describe files with no limit
    #[clap(long, conflicts_with = "limit")]
    pub no_limit: bool,
    /// The number of files to describe per API request
    #[clap(short, long, default_value = "100")]
    pub page_size: usize,
    /// Refrain from setting any filters when describing files, attempting to describe all files the user can view
    ///
    /// This will override any other search parameters set except for those associated with limit.
    /// When combined with `--no-limit`, this will describe all files to which the current user
    /// has access.
    #[clap(long)]
    pub describe_all: bool,
}

impl SearchSealed for DescribeFiles {
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

impl SearchParameterized for DescribeFiles {
    fn has_targets(&self) -> bool {
        !self.files.is_empty() || self.file_list.is_some()
    }

    fn apply_to_all(&self) -> bool {
        self.describe_all
    }
}

impl DescribeSealed for DescribeFiles {
    type Data = thorium::models::Sample;

    type Target<'a> = &'a str;

    type Cursor = thorium::models::Cursor<Self::Data>;

    fn raw_targets(&self) -> &[String] {
        &self.files
    }

    fn condensed(&self) -> bool {
        self.condensed
    }

    fn out_path(&self) -> Option<&PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&PathBuf> {
        self.file_list.as_ref()
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
        thorium.files.get(target).await
    }

    async fn retrieve_data_search(
        &self,
        thorium: &thorium::Thorium,
    ) -> Result<Vec<Self::Cursor>, thorium::Error> {
        // build our file list opts
        let opts = self.build_file_opts()?;
        // list details for these files
        let cursor = thorium.files.list_details(&opts).await?;
        Ok(vec![cursor])
    }
}

impl DescribeCommand for DescribeFiles {}
