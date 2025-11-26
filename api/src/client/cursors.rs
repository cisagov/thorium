use chrono::prelude::*;
use serde::Deserialize;

use super::Error;
use crate::models::StageLogs;
use crate::send_build;

#[cfg(feature = "sync")]
use super::RUNTIME;

#[derive(Deserialize)]
struct RawCursorData<T> {
    /// The optional cursor returned by api
    pub cursor: Option<u64>,
    /// The names returned by this cursor when details is false
    pub names: Option<Vec<String>>,
    /// The data returned by this cursor when details is true
    pub details: Option<Vec<T>>,
}

/// A cursor for basic searches with a single cursor value
pub struct Cursor<T>
where
    for<'de> T: Deserialize<'de>,
{
    /// The url used to build/rehydrate this cursor
    pub url: String,
    /// The reqwest client used get data
    client: reqwest::Client,
    /// token to use for auth
    token: String,
    /// The cursor we will use for the next hydration requestion
    pub cursor: u64,
    /// The amount of data to get per page of this cursor
    pub page_size: u64,
    /// The total amount of data to get over the lifetime of this cursor
    pub limit: Option<u64>,
    /// The current amount of data that has been retrieved from the server
    pub retrieved: u64,
    /// Whether the cursor has reached its limit or the API has run out of
    /// items to return
    pub exhausted: bool,
    /// The names returned by this cursor
    pub names: Vec<String>,
    /// The details returned by this cursor if details is enabled
    pub details: Vec<T>,
    /// Whether this cursor should retry on transient errors
    pub retry: bool,
}

impl<T> Cursor<T>
where
    for<'de> T: Deserialize<'de>,
{
    /// Build a new cursor object
    ///
    /// This should be built by the list methods on any of sub clients in this crate. You likely do
    /// not want to create it yourself.
    ///
    /// # Arguments
    ///
    /// * `url` - The url we will be using to build/rehydrate this cursor
    /// * `token` - The authentication token used for this cursor
    /// * `client` - The client this cursor should use
    #[must_use]
    pub fn new(url: String, token: &str, client: &reqwest::Client) -> Self {
        Cursor {
            url,
            client: client.clone(),
            token: token.to_owned(),
            cursor: 0,
            page_size: 50,
            retrieved: 0,
            limit: None,
            exhausted: false,
            names: Vec::default(),
            details: Vec::default(),
            retry: true,
        }
    }

    /// Sets the new cursor value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `cursor` - The new cursor value to use
    #[must_use]
    pub fn cursor(mut self, cursor: u64) -> Self {
        self.cursor = cursor;
        self
    }

    /// Sets the page size value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `page_size` - The page size to use
    #[must_use]
    pub fn page_size(mut self, page_size: u64) -> Self {
        self.page_size = page_size;
        self
    }

    /// Sets the new limit value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `limit` - The new limit value to use
    #[must_use]
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Transform this cursor into a details cursor by appending /details to the url
    ///
    /// Calling this multiple times will result in multiple /details being added and so should not
    /// be done.
    #[must_use]
    pub fn details(mut self) -> Self {
        self.url = format!("{}details/", self.url);
        self
    }

    /// Sets the new page value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `page` - The new page value to use
    #[must_use]
    pub fn retry(mut self, retry: bool) -> Self {
        self.retry = retry;
        self
    }

    /// Executes a newly created cursor returning it
    ///
    /// This just wraps next which takes a mutable reference.
    #[cfg_attr(feature = "sync", thorium_derive::blocking_fn)]
    pub async fn exec(mut self) -> Result<Self, Error> {
        self.next().await?;
        Ok(self)
    }

    /// Get the next page of data for this cursor
    #[cfg_attr(feature = "sync", thorium_derive::blocking_fn)]
    pub async fn next(&mut self) -> Result<(), Error> {
        // make sure our page is not larger then our limit
        let page_size = match self.limit {
            Some(limit) => std::cmp::min(self.page_size, limit - self.retrieved),
            None => self.page_size,
        };
        // retry sending our request for new data on transient errors if enabled
        let mut raw = loop {
            // build request
            let req = self
                .client
                .get(&self.url)
                .header("authorization", &self.token)
                .query(&[("cursor", self.cursor), ("limit", page_size)]);
            // send request and build a raw cursor
            match send_build!(self.client, req, RawCursorData<T>) {
                Ok(raw) => break raw,
                Err(error) => {
                    // if retry is enabled then check if we should retry or just fail
                    if self.retry {
                        // determine if this error could be transient or not
                        if error
                            .status()
                            .is_some_and(|status| status.is_server_error())
                        {
                            continue;
                        }
                    }
                    // return our error
                    return Err(error);
                }
            }
        };
        // extract data from our cursor
        self.names = raw.names.take().unwrap_or_default();
        self.details = raw.details.take().unwrap_or_default();
        // update the current amount of total data retrieved
        self.retrieved += (self.names.len() + self.details.len()) as u64;
        // update our cursor if we got a new cursor
        match (raw.cursor, self.limit) {
            // only set the new cursor if we haven't hit our limit
            (Some(cursor), Some(limit)) if self.retrieved < limit => self.cursor = cursor,
            // we have no limit, so set the new cursor
            (Some(cursor), None) => self.cursor = cursor,
            // otherwise this cursor is exhausted
            (_, _) => self.exhausted = true,
        }
        Ok(())
    }
}

/// A cursor for basic searches with of stage logs
pub struct LogsCursor {
    /// The url used to build/rehydrate this cursor
    pub url: String,
    /// The reqwest client used get data
    client: reqwest::Client,
    /// token to use for auth
    token: String,
    /// The cursor we will use for the next hydration requestion
    pub cursor: usize,
    /// The amount of data to get per page of this cursor
    pub page: usize,
    /// The total amount of data to get over the lifetime of this cursor
    pub limit: Option<usize>,
    /// The current amount of data that has been retrieved from the server
    pub retrieved: usize,
    /// Whether our cursor has been exhausted
    pub exhausted: bool,
    /// The current page of logs returned by this cursor
    pub logs: StageLogs,
}

impl LogsCursor {
    /// Build a new cursor object
    ///
    /// This should be built by the list methods on any of sub clients in this crate. You likely do
    /// not want to create it yourself.
    ///
    /// # Arguments
    ///
    /// * `url` - The url we will be using to build/rehydrate this cursor
    /// * `token` - The authentication token used for this cursor
    /// * `client` - The client this cursor should use
    pub fn new(url: String, token: &str, client: &reqwest::Client) -> Self {
        Self {
            url,
            client: client.clone(),
            token: token.to_owned(),
            cursor: 0,
            page: 50,
            retrieved: 0,
            limit: None,
            exhausted: false,
            logs: StageLogs { logs: Vec::new() },
        }
    }

    /// Sets the new cursor value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `cursor` - The new cursor value to use
    pub fn cursor(mut self, cursor: usize) -> Self {
        self.cursor = cursor;
        self
    }

    /// Sets the new page value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `page` - The new page value to use
    pub fn page(mut self, page: usize) -> Self {
        self.page = page;
        self
    }

    /// Sets the new limit value to use in the next request
    ///
    /// # Arguments
    ///
    /// * `limit` - The new limit value to use, if None, then no limit checks are performed
    pub fn limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Executes a newly created cursor returning it
    ///
    /// This just wraps next which takes a mutable reference.
    #[cfg_attr(feature = "sync", thorium_derive::blocking_fn)]
    pub async fn exec(mut self) -> Result<Self, Error> {
        self.next().await?;
        Ok(self)
    }

    /// Get the next page of data for this cursor
    #[cfg_attr(feature = "sync", thorium_derive::blocking_fn)]
    pub async fn next(&mut self) -> Result<(), Error> {
        // build request
        let req = self
            .client
            .get(&self.url)
            .header("authorization", &self.token)
            .query(&[("cursor", self.cursor), ("limit", self.page)]);
        // send request to get the requested logs
        self.logs = send_build!(self.client, req, StageLogs)?;
        // update the current amount of total data retrieved
        self.retrieved += self.logs.logs.len();
        // determine if the limit has been reached
        let limit_reached = if let Some(limit) = self.limit {
            self.retrieved >= limit
        } else {
            false
        };
        // Either update the cursor position
        self.cursor += self.logs.logs.len();
        // Inform the user that the cursor is currently exhausted
        if self.logs.logs.is_empty() || limit_reached {
            self.exhausted = true;
        }
        Ok(())
    }
}

/// Build a specific date for a file search restriction
pub struct SearchDate;

impl SearchDate {
    /// build a new search date for a specific year
    ///
    /// # Arguments
    ///
    /// * `year` - The year to restrict our search with
    /// * `end` - Whether this should be the start or end of the year
    pub fn year(year: u64, end: bool) -> Result<DateTime<Utc>, Error> {
        // build the rfc 3339 string for either the start or end of the year
        let raw = if end {
            format!("{year}-00-01T00:00:00-00:00")
        } else {
            format!("{year}-12-31T23:59:59-00:00")
        };
        Ok(DateTime::parse_from_rfc3339(&raw)?.with_timezone(&Utc))
    }

    /// build a new search date for a specific year and month
    ///
    /// # Arguments
    ///
    /// * `year` - The year to restrict our search with
    /// * `month` - The month to restrict our search with
    /// * `day` - The day to restrict our search with
    /// * `end` - Whether this should be the start or end of the year
    pub fn day(year: u64, month: u8, day: u8, end: bool) -> Result<DateTime<Utc>, Error> {
        // build the rfc 3339 string for either the start or end of the year
        let raw = if end {
            format!("{year}-{month}-{day}T00:00:00-00:00")
        } else {
            format!("{year}-{month}-{day}T23:59:59-00:00")
        };
        Ok(DateTime::parse_from_rfc3339(&raw)?.with_timezone(&Utc))
    }
}
