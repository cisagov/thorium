//! The search support for the Thorium client

use super::{Error, SearchEvents};
use crate::models::{Cursor, ElasticDoc, ElasticSearchOpts};
use crate::{add_date, add_query, add_query_list};

pub mod events;

/// A handler for the search routes in Thorium
#[derive(Clone)]
pub struct Search {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
    /// The search events handler
    pub events: events::SearchEvents,
}

impl Search {
    /// Creates a new search handler
    ///
    /// Instead of directly creating this handler you likely want to simply create a
    /// `thorium::Thorium` and use the handler within that instead.
    ///
    /// # Arguments
    ///
    /// * `host` - url/ip of the thorium api
    /// * `token` - The token used for authentication
    /// * `client` - The reqwest client to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::Search;
    ///
    /// let client = reqwest::Client::new();
    /// let results = Search::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Search {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
            events: SearchEvents::new(host, token, client),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// A blocking handler for the results routes in Thorium
        #[derive(Clone)]
        #[allow(dead_code)]
        pub struct SearchBlocking {
            /// The host/url that Thorium can be reached at
            host: String,
            /// token to use for auth
            token: String,
            /// A reqwest client for reqwests
            client: reqwest::Client,
        }

        impl SearchBlocking {
            /// Creates a new blocking search handler
            ///
            /// Instead of directly creating this handler you likely want to simply create a
            /// `thorium::ThoriumBlocking` and use the handler within that instead.
            ///
            ///
            /// # Arguments
            ///
            /// * `host` - url/ip of the thorium api
            /// * `token` - The token used for authentication
            /// * `client` - The reqwest client to use
            ///
            /// # Examples
            ///
            /// ```
            /// use thorium::client::SearchBlocking;
            ///
            /// let results = SearchBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                SearchBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Search {
    /// Executes a full text search query in Thorium
    ///
    /// # Arguments
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::ElasticSearchOpts;
    /// # use thorium::Error;
    /// use uuid::Uuid;
    /// use chrono::prelude::*;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build our search options
    /// let opts = ElasticSearchOpts::new("pe32 AND x86_64");
    /// // send our search query
    /// let cursor = thorium.search.search(&opts).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn search(&self, opts: &ElasticSearchOpts) -> Result<Cursor<ElasticDoc>, Error> {
        // build the url for searching data in Thorium
        let url = format!("{}/api/search/", self.host);
        // get the correct page size if our limit is smaller then our page_size
        let page_size = opts.limit.map_or_else(
            || opts.page_size,
            |limit| std::cmp::min(opts.page_size, limit),
        );
        // build our query params
        let mut query = vec![
            ("query", opts.query.clone()),
            ("limit", page_size.to_string()),
        ];
        add_query_list!(query, "indexes[]", &opts.indexes);
        add_query_list!(query, "groups[]", &opts.groups);
        add_date!(query, "start", opts.start);
        add_date!(query, "end", opts.end);
        add_query!(query, "cursor", opts.cursor);
        // get the data for this request and create our cursor
        Cursor::new(
            &url,
            opts.page_size,
            opts.limit,
            &self.token,
            &query,
            &self.client,
        )
        .await
    }
}
