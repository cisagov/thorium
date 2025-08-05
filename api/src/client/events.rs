//! Exposes events routes in Thorium

use crate::models::{
    Event, EventCacheStatus, EventCacheStatusOpts, EventIds, EventPopOpts, EventType,
};
use crate::{send, send_build, Error};

#[derive(Clone)]
pub struct Events {
    host: String,
    /// token to use for auth
    token: String,
    client: reqwest::Client,
}

impl Events {
    /// Creates a new events handler
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
    /// use thorium::client::Events;
    ///
    /// let client = reqwest::Client::new();
    /// let events = Events::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Events {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        #[derive(Clone)]
        pub struct EventsBlocking {
            host: String,
            /// token to use for auth
            token: String,
            client: reqwest::Client,
        }

        impl EventsBlocking {
            /// creates a new blocking events handler
            ///
            /// Instead of directly creating this handler you likely want to simply create a
            /// `thorium::ThoriumBlocking` and use the handler within that instead.
            ///
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
            /// use thorium::client::EventsBlocking;
            ///
            /// let events = EventsBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                EventsBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Events {
    /// Pop some events to handle
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of events to pop
    /// * `opts` - The parameters to use when popping events
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::events::pop", skip(self, opts), err(Debug))
    )]
    pub async fn pop(&self, kind: EventType, opts: &EventPopOpts) -> Result<Vec<Event>, Error> {
        // build the url for listing events
        let url = format!("{}/api/events/pop/{}/", self.host, kind);
        // build our query opts
        let query = vec![("limit", opts.limit)];
        // build our request
        let req = self
            .client
            .patch(&url)
            .query(&query)
            .header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, Vec<Event>)
    }

    /// Clear some events
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of events to clear
    /// * `ids` - The events IDs to clear
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::events::clear", skip(self, ids), err(Debug))
    )]
    pub async fn clear(&self, kind: EventType, ids: &EventIds) -> Result<reqwest::Response, Error> {
        // build the url for listing events
        let url = format!("{}/api/events/clear/{}/", self.host, kind);
        // build our request
        let req = self
            .client
            .delete(&url)
            .json(ids)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Reset all currently in flight events for a specific event kind
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of events to clear
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::events::reset_all", skip(self), err(Debug))
    )]
    pub async fn reset_all(&self, kind: EventType) -> Result<reqwest::Response, Error> {
        // build the url for listing events
        let url = format!("{}/api/events/reset/{}/", self.host, kind);
        // build our request
        let req = self.client.patch(&url).header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Get the event queue cache status
    ///
    /// # Arguments
    ///
    /// * `params` - The params for getting the status of the event cache
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::events::get_cache_status", skip(self), err(Debug))
    )]
    pub async fn get_cache_status(
        &self,
        opts: &EventCacheStatusOpts,
    ) -> Result<EventCacheStatus, Error> {
        // build the url for getting our event watermark
        let url = format!("{}/api/events/cache/status/", self.host);
        // build our query params
        let query = vec![("reset", opts.reset.to_string())];
        // build our request
        let req = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send this request and build our event watermark from the response
        send_build!(self.client, req, EventCacheStatus)
    }
}
