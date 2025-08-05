use chrono::prelude::*;

use super::Error;
use crate::models::StreamDepth;
use crate::send_build;

#[derive(Clone)]
pub struct Streams {
    host: String,
    /// token to use for auth
    token: String,
    client: reqwest::Client,
}

impl Streams {
    /// Creates a new streams handler
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
    /// use thorium::client::Streams;
    ///
    /// let client = reqwest::Client::new();
    /// let streams = Streams::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Streams {
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
        pub struct StreamsBlocking {
            host: String,
            /// token to use for auth
            token: String,
            client: reqwest::Client,
        }

        impl StreamsBlocking {
            /// creates a new blocking streams handler
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
            /// use thorium::client::StreamsBlocking;
            ///
            /// let streams = StreamsBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                StreamsBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Streams {
    /// Gets the number of objects between two points in a stream
    ///
    /// # Arguments
    ///
    /// * `group` - The group this stream is in
    /// * `namespace` - The namespace of the stream within this group
    /// * `stream` - The name of the stream to check
    /// * `start` - The timestamp to start counting at
    /// * `end` - The timestamp to stop counting at
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use chrono::prelude::*;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get the number of deadlines in the next hour
    /// let start = Utc::now();
    /// let end = start + chrono::Duration::hours(1);
    /// let depth = thorium.streams.depth("system", "k8s", "deadlines", &start, &end).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn depth(
        &self,
        group: &str,
        namespace: &str,
        stream: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Result<StreamDepth, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/streams/depth/{group}/{namespace}/{stream}/{start}/{end}",
            base = &self.host,
            group = group,
            namespace = namespace,
            stream = stream,
            start = start.timestamp(),
            end = end.timestamp()
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a stream depth from the response
        send_build!(self.client, req, StreamDepth)
    }

    /// Gets the number of objects between in even chunks of time between two timestamps
    ///
    /// # Arguments
    ///
    /// * `group` - The group this stream is in
    /// * `stream` - The name of the stream to check
    /// * `namespace` - The namespace of the stream within this group
    /// * `start` - The timestamp to start counting at
    /// * `end` - The timestamp to stop counting at
    /// * `split` - The size of chunk to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use chrono::prelude::*;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get the number of deadlines in the next hour in 5 minute splits
    /// let start = Utc::now();
    /// let end = start + chrono::Duration::hours(1);
    /// let split = chrono::Duration::minutes(5);
    /// let depths = thorium.streams.depth_range("system", "k8s", "deadlines", &start, &end, &split).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn depth_range(
        &self,
        group: &str,
        namespace: &str,
        stream: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        split: &chrono::Duration,
    ) -> Result<Vec<StreamDepth>, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/streams/depth/{group}/{namespace}/{stream}/{start}/{end}/{split}",
            base = &self.host,
            group = group,
            namespace = namespace,
            stream = stream,
            start = start.timestamp(),
            end = end.timestamp(),
            split = split.num_seconds()
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a vector of stream depths from the response
        send_build!(self.client, req, Vec<StreamDepth>)
    }
}
