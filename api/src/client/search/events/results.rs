//! Interacts with result search events routes in Thorium

use crate::client::traits::GenericClient;
use crate::models::{ResultSearchEvent, SearchEventPopOpts, SearchEventStatus};

use super::{SearchEventsClient, SearchEventsClientHelper};

#[derive(Clone)]
pub struct ResultSearchEvents {
    host: String,
    /// token to use for auth
    token: String,
    client: reqwest::Client,
}

impl ResultSearchEvents {
    /// Creates a new result search events handler
    ///
    /// Instead of directly creating this handler you likely want to simply create a
    /// `thorium::Thorium` and use the handler within that instead.
    ///
    /// # Arguments
    ///
    /// * `host` - The url/ip of the Thorium api
    /// * `token` - The token used for authentication
    /// * `client` - The reqwest client to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::ResultSearchEvents;
    ///
    /// let client = reqwest::Client::new();
    /// let events = ResultSearchEvents::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        Self {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

impl GenericClient for ResultSearchEvents {
    fn base_url(&self) -> String {
        format!("{}/api/search/events", self.host)
    }

    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    fn token(&self) -> &str {
        &self.token
    }
}

impl SearchEventsClientHelper for ResultSearchEvents {
    type SearchEventHelper = ResultSearchEvent;
}

#[async_trait::async_trait]
impl SearchEventsClient for ResultSearchEvents {
    type SearchEvent = ResultSearchEvent;

    /// Pop some tag search events to handle
    ///
    /// # Arguments
    ///
    /// * `opts` - The parameters to use when popping search events
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::SearchEventsClient;
    /// use thorium::models::SearchEventPopOpts;
    /// # use thorium::Error;
    /// use chrono::prelude::*;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build our search options
    /// let opts = SearchEventPopOpts::default().limit(10);
    /// // pop the events
    /// let events = thorium.search.events.results.pop(&opts).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    async fn pop(&self, opts: &SearchEventPopOpts) -> Result<Vec<Self::SearchEvent>, crate::Error> {
        self.pop_generic(opts).await
    }

    /// Send the status of processed tag search events
    ///
    /// # Arguments
    ///
    /// * `status` - The status of the processed events (i.e. which ones succeeded and which failed)
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::SearchEventsClient;
    /// use thorium::models::SearchEventStatus;
    /// # use thorium::Error;
    /// use uuid::Uuid;
    /// use chrono::prelude::*;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // designate which events succeeded/failed
    /// let successes = vec![Uuid::new_v4()];
    /// let failures = vec![Uuid::new_v4()];
    /// let status = SearchEventStatus { successes, failures };
    /// // clear the events
    /// thorium.search.events.results.send_status(&status).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    async fn send_status(
        &self,
        status: &SearchEventStatus,
    ) -> Result<reqwest::Response, crate::Error> {
        self.send_status_generic(status).await
    }

    /// Reset all in-flight tag search events
    ///
    /// This function should be called when first initializing a consumer
    /// of tag search events to ensure no events were lost in-flight
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::SearchEventsClient;
    /// # use thorium::Error;
    /// use uuid::Uuid;
    /// use chrono::prelude::*;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // reset all in-flight tag search events
    /// thorium.search.events.results.reset_all().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    async fn reset_all(&self) -> Result<reqwest::Response, crate::Error> {
        self.reset_all_generic().await
    }
}
