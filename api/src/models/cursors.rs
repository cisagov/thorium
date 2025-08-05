//! A cursor for data in Thorium

use serde::{Deserialize, Serialize};
use uuid::Uuid;

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        /// A cursor for data in Thorium used by the API
        #[derive(Debug, Serialize)]
        #[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
        pub struct ApiCursor<T>
        where
            for<'de> T: Deserialize<'de>,
            T: Serialize
        {
            /// The ID for this cursor if more data can be pulled
            #[serde(skip_serializing_if = "Option::is_none")]
            pub cursor: Option<Uuid>,
            /// A single page worth of data
            pub data: Vec<T>,
        }

        impl<T> ApiCursor<T>
        where
            for<'de> T: Deserialize<'de>,
            T: Serialize,
        {
            /// Create an empty api cursor
            ///
            /// # Arguments
            ///
            /// * `hint` - The hint for how much space to allocate
            pub fn empty(hint: usize) -> Self {
                ApiCursor { cursor: None, data: Vec::with_capacity(hint) }
            }
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "client")] {
        use crate::client::Error;
        use crate::{add_query, send_build};
        use chrono::prelude::*;

        /// Build a specific date for a file search restriction
        pub struct DateOpts;

        impl DateOpts {
            /// build a new search date for a specific year
            ///
            /// # Arguments
            ///
            /// * `year` - The year to restrict our search with
            /// * `end` - Whether this should be the start or end of the year
            pub fn year(year: u64, end: bool) -> Result<DateTime<Utc>, Error> {
                // build the rfc 3339 string for either the start or end of the year
                let raw = if end {
                    format!("{}-00-01T00:00:00-00:00", year)
                } else {
                    format!("{}-12-31T23:59:59-00:00", year)
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
                    format!("{}-{}-{}T00:00:00-00:00", year, month, day)
                } else {
                    format!("{}-{}-{}T23:59:59-00:00", year, month, day)
                };
                Ok(DateTime::parse_from_rfc3339(&raw)?.with_timezone(&Utc))
            }
        }

        /// The data that the cursor from the API alone will return
        #[derive(Deserialize)]
        struct CursorData<T>
        {
            /// The ID for this cursor if more data can be pulled
            pub cursor: Option<Uuid>,
            /// A single page worth of data
            pub data: Vec<T>,
        }

        /// The data that the cursor from the API alone will return
        pub struct Cursor<T>
        where
            for<'de> T: Deserialize<'de>,
        {
            /// The ID for this cursor if more data can be pulled
            pub id: Option<Uuid>,
            /// A single page worth of data
            pub data: Vec<T>,
            /// The amount of data to retrieve per page
            pub page_size: usize,
            /// Whether this cursor has been exhausted or not
            exhausted: bool,
            /// The url to get more data at
            url: String,
            /// The amount of data currently returned by this cursor
            gathered: usize,
            /// The total amount of data to return
            pub limit: Option<usize>,
            /// Whether this cursor should retry on transient errors
            pub retry: bool,
            /// The token to authenticate to Thorium with
            token: String,
            /// The max amount of data this cursor should gather
            /// A reqwest client used to get more data from the API
            client: reqwest::Client,
        }

        impl<T> Cursor<T>
        where
            for<'de> T: Deserialize<'de>,
            T: Serialize,
        {
            /// Create a new cursor object
            pub async fn new<U, A, Q>(
                url: U,
                page_size: usize,
                limit: Option<usize>,
                token: A,
                query: &Q,
                client: &reqwest::Client
            ) -> Result<Self, Error>
            where
                U: Into<String>,
                A: Into<String>,
                Q: Serialize + ?Sized
            {
                // cast our url and token to a String
                let url = url.into();
                let token = token.into();
                // build request
                let req = client
                    .get(&url)
                    .header("authorization", &token)
                    .query(query);
                // send request and build the data and id for this cursor
                let raw = send_build!(client, req, CursorData<T>)?;
                // get the length of the data we received
                let gathered = raw.data.len();
                // build our final cursor object
                let cursor = Cursor {
                    exhausted: raw.cursor.is_none(),
                    id: raw.cursor,
                    data: raw.data,
                    url,
                    page_size,
                    limit,
                    retry: true,
                    gathered,
                    token,
                    client: client.clone(),
                };
                Ok(cursor)
            }

            /// Sets the new page value to use in the next request
            ///
            /// # Arguments
            ///
            /// * `page` - The new page value to use
            pub fn retry(mut self, retry: bool) -> Self {
                self.retry = retry;
                self
            }


            /// Check if this cursor has either run out of data or retrieved all the requested data
            pub fn exhausted(&self) -> bool {
                // check if our cursor is exhausted
                if self.exhausted {
                    true
                } else {
                    // check if we have a limit defined since our cursor isn't exhausted
                    if let Some(limit) = self.limit {
                        // check if we have retrieved all requested data
                        self.gathered >= limit
                    } else {
                        false
                    }
                }
            }

            /// Get the size of the next page of our cursor
            pub fn next_page_size(&self) -> usize {
                match self.limit {
                    Some(limit) => std::cmp::min(self.page_size, limit - self.gathered),
                    None => self.page_size,
                }
            }

            /// Refill this cursor with new data
            pub async fn refill(&mut self) -> Result<(), Error> {
                // build our query params
                let mut query = vec![("limit", self.next_page_size().to_string())];
                add_query!(query, "cursor", self.id);
                // build request
                let raw = loop {
                    // build our request
                    let req = self
                        .client
                        .get(&self.url)
                        .header("authorization", &self.token)
                        .query(&query);
                    // send request and build a raw cursor
                    match send_build!(self.client, req, CursorData<T>) {
                        Ok(raw) => break raw,
                        Err(error) => {
                            // if retry is enabled then check if we should retry or just fail
                            if self.retry {
                                // determine if this error could be transient or not
                                if error
                                    .status()
                                    .map(|status| status.is_server_error())
                                    .unwrap_or(false)
                                {
                                    continue;
                                }
                            }
                            // return our error
                            return Err(error);
                        }
                    }
                };
                // update our cursors data
                self.id = raw.cursor;
                self.exhausted = self.id.is_none();
                self.gathered += raw.data.len();
                self.data = raw.data;
                Ok(())
            }
        }
    }
}
