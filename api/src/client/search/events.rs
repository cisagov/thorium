//! Exposes search events routes in Thorium

use results::ResultSearchEvents;
use tags::TagSearchEvents;

use crate::{
    client::traits::GenericClient,
    models::{SearchEvent, SearchEventPopOpts, SearchEventStatus},
    send, send_build, Error,
};

pub mod results;
pub mod tags;

#[derive(Clone)]
pub struct SearchEvents {
    pub tags: TagSearchEvents,
    pub results: ResultSearchEvents,
}

impl SearchEvents {
    /// Creates a new search events handler
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
    /// use thorium::client::SearchEvents;
    ///
    /// let client = reqwest::Client::new();
    /// let events = SearchEvents::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Self {
            tags: TagSearchEvents::new(host, token, client),
            results: ResultSearchEvents::new(host, token, client),
        }
    }
}

/// A helper trait containing generic implementations for [`SearchEventsClient`]
///
/// The functions are separated to allow for specific docs for each implementation
pub(super) trait SearchEventsClientHelper: GenericClient {
    /// The search event this client is interacting with
    type SearchEventHelper: SearchEvent;

    /// Pop some search events to handle
    ///
    /// # Arguments
    ///
    /// * `opts` - The parameters to use when popping search events
    async fn pop_generic(
        &self,
        opts: &SearchEventPopOpts,
    ) -> Result<Vec<Self::SearchEventHelper>, Error> {
        // build the url
        let url = format!(
            "{base}/{event_type}/pop/",
            base = self.base_url(),
            event_type = Self::SearchEventHelper::url()
        );
        // build our query opts
        let query = vec![("limit", opts.limit)];
        // build our request
        let req = self
            .client()
            .patch(&url)
            .query(&query)
            .header("authorization", self.token());
        // send this request
        send_build!(self.client(), req, Vec<Self::SearchEventHelper>)
    }

    /// Send the status of processed events
    ///
    /// # Arguments
    ///
    /// * `status` - The status of the processed events (i.e. which ones succeeded and which failed)
    async fn send_status_generic(
        &self,
        status: &SearchEventStatus,
    ) -> Result<reqwest::Response, Error> {
        // build the url
        let url = format!(
            "{base}/{event_type}/status/",
            base = self.base_url(),
            event_type = Self::SearchEventHelper::url()
        );
        // build our request
        let req = self
            .client()
            .patch(&url)
            .json(status)
            .header("authorization", self.token());
        // send this request
        send!(self.client(), req)
    }

    /// Reset all in-flight search events
    ///
    /// This function should be called when first initializing a consumer
    /// of search events to ensure no events were lost in-flight
    async fn reset_all_generic(&self) -> Result<reqwest::Response, Error> {
        // build the url
        let url = format!(
            "{base}/{event_type}/reset/",
            base = self.base_url(),
            event_type = Self::SearchEventHelper::url()
        );
        // build our request
        let req = self
            .client()
            .patch(&url)
            .header("authorization", self.token());
        // send this request
        send!(self.client(), req)
    }
}

/// Describes a client that is capable of interacting with search events for the
///
/// A client can implement these functions to provide specific docs for its implementation
///
/// We use the async_trait proc macro here to avoid compiler errors where
/// it's not sure the functions are `Send`
#[allow(async_fn_in_trait, private_bounds)]
#[async_trait::async_trait]
pub trait SearchEventsClient: SearchEventsClientHelper {
    type SearchEvent: SearchEvent;

    async fn pop(&self, opts: &SearchEventPopOpts) -> Result<Vec<Self::SearchEvent>, Error>;

    async fn send_status(&self, status: &SearchEventStatus) -> Result<reqwest::Response, Error>;

    async fn reset_all(&self) -> Result<reqwest::Response, Error>;
}
