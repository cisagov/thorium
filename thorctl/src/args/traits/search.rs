//! Traits and logic allowing handlers to perform a search based on params in a given command

use chrono::NaiveDateTime;
use uuid::Uuid;

use thorium::{
    models::{FileListOpts, NetworkPolicyListOpts, RepoListOpts},
    Error,
};

/// The parameters used for a cursor search in Thorium
pub struct SearchParams<'a> {
    /// The groups to filter by
    pub groups: &'a [String],
    /// The tags to filter by
    pub tags: &'a [String],
    /// Whether matching on tags should be case-insensitive
    pub tags_case_insensitive: bool,
    /// The delimiter for tags
    pub delimiter: char,
    /// The datetime to start searching
    pub start: &'a Option<String>,
    /// The datetime to stop searching
    pub end: &'a Option<String>,
    /// The format used for parsing the start and end datetimes
    pub date_fmt: &'a str,
    /// The cursor to resume from if one exists
    pub cursor: Option<Uuid>,
    /// The maximum number of objects to return
    pub limit: usize,
    /// Refrain from setting a limit for the maximum number of objects to return
    pub no_limit: bool,
    /// The number of objects to return per request
    pub page_size: usize,
}

/// A private trait preventing inner search business logic
/// from being called outside the module
pub trait SearchSealed {
    /// Build [`SearchParams`] from the implementor
    fn get_search_params(&self) -> SearchParams;
}

/// Describes a command that can build the parameters needed to search
#[allow(private_bounds)]
pub trait SearchParameterized: SearchSealed {
    /// Check if the implementor has specific targets
    fn has_targets(&self) -> bool;

    /// Check if the implementor has any parameters to perform a search
    fn has_parameters(&self) -> bool {
        let parameters = self.get_search_params();
        !parameters.tags.is_empty()
            || !parameters.groups.is_empty()
            || parameters.start.is_some()
            || parameters.end.is_some()
    }

    /// Checks that the implementor has a valid search configuration, specifically that it
    /// has specific targets, has at least one search parameter, or is explicitly set to
    /// apply to all. Also checks that "apply to all" is not set while also having parameters.
    /// Returns an error if the configuration is invalid
    fn validate_search(&self) -> Result<(), Error> {
        if !self.has_targets() && !self.has_parameters() && !self.apply_to_all() {
            return Err(Error::new(
                "Command must be given specific targets, have at \
                least one search parameter (tags, groups, start, end, etc.), \
                or must be explicitly set to apply to all!",
            ));
        }
        if self.apply_to_all() && self.has_parameters() {
            // return an error if the user set search parameters AND the apply to all option
            return Err(Error::new(
                "Command has one or more search filters set but is also set \
                    to apply to all! Please remove either the search filter(s) or the \
                    '--...-all' option.",
            ));
        }
        Ok(())
    }

    /// Apply the implementor's action to all with no search filter;
    /// this should not be true if search parameters are also given
    fn apply_to_all(&self) -> bool;

    /// Attempt to build [`FileListOpts`] based on search parameters from the trait implementor
    fn build_file_opts(&self) -> Result<FileListOpts, Error> {
        let params = self.get_search_params();
        let mut search = FileListOpts::default().page_size(params.page_size);
        // add a limit unless the "no_limit" flag is set
        if !params.no_limit {
            search = search.limit(params.limit);
        }
        // if a cursor was specified then set it
        if let Some(cursor) = params.cursor {
            search = search.cursor(cursor);
        }
        if self.apply_to_all() {
            // return the search with no filters if the command is set to apply to all
            return Ok(search);
        }
        search = search.groups(params.groups.to_vec());
        // set start/end times to search if provided
        if let Some(start) = &params.start {
            search = search.start(NaiveDateTime::parse_from_str(start, params.date_fmt)?.and_utc());
        }
        if let Some(end) = &params.end {
            search = search.end(NaiveDateTime::parse_from_str(end, params.date_fmt)?.and_utc());
        }
        // crawl over and split any tags
        for combined in params.tags {
            // split this combined tag by our delimiter
            match combined.split_once(params.delimiter) {
                // add this tag filter to our search
                Some((key, value)) => search = search.tag(key, value),
                None => {
                    // build an error message to return
                    let msg =
                        format!("{combined} is not a key/value pair. Please check your delimiter.");
                    // return our error
                    return Err(Error::new(msg));
                }
            }
        }
        // set whether the search on tags is case insensitive
        if params.tags_case_insensitive {
            search = search.tags_case_insensitive();
        }
        Ok(search)
    }

    /// Attempt to build [`RepoListOpts`] based on search parameters from the trait implementor
    fn build_repo_opts(&self) -> Result<RepoListOpts, Error> {
        let params = self.get_search_params();
        // start building our list options
        let mut search = RepoListOpts::default().page_size(params.page_size);
        // add a limit unless the "no_limit" flag is set
        if !params.no_limit {
            search = search.limit(params.limit);
        }
        // if a cursor was specified then set it
        if let Some(cursor) = params.cursor {
            search = search.cursor(cursor);
        }
        if self.apply_to_all() {
            // return the search with no filters if the command is set to apply to all
            return Ok(search);
        }
        search = search.groups(params.groups.to_vec());
        // set start/end times to search if provided
        if let Some(start) = &params.start {
            search = search.start(NaiveDateTime::parse_from_str(start, params.date_fmt)?.and_utc());
        }
        if let Some(end) = &params.end {
            search = search.end(NaiveDateTime::parse_from_str(end, params.date_fmt)?.and_utc());
        }
        // crawl over and split any tags
        for combined in params.tags {
            // split this combined tag by our delimiter
            match combined.split_once(params.delimiter) {
                // add this tag filter to our search
                Some((key, value)) => search = search.tag(key, value),
                None => {
                    // build an error message to return
                    let msg =
                        format!("{combined} is not a key/value pair. Please check your delimiter.");
                    // return our error
                    return Err(Error::new(msg));
                }
            }
        }
        // set whether the search on tags is case insensitive
        if params.tags_case_insensitive {
            search = search.tags_case_insensitive();
        }
        Ok(search)
    }

    fn build_network_policy_opts(&self) -> NetworkPolicyListOpts {
        // get our search params
        let params = self.get_search_params();
        NetworkPolicyListOpts {
            cursor: params.cursor,
            page_size: params.page_size,
            limit: (!params.no_limit).then_some(params.limit),
            groups: params.groups.to_vec(),
        }
    }
}
