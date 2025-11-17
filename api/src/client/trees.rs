//! Support the tree routes in the Thorium client

#[cfg(feature = "trace")]
use tracing::instrument;
use uuid::Uuid;

use super::Error;
use crate::models::{Tree, TreeGrowQuery, TreeOpts, TreeQuery};
use crate::{add_query, send_build};

#[derive(Clone)]
pub struct Trees {
    /// The host to talk to the Thorium api at
    host: String,
    /// The token to use for auth
    token: String,
    /// A client to use when making requests
    client: reqwest::Client,
}

impl Trees {
    /// Creates a new trees handler
    ///
    /// Instead of directly creating this handler you likely want to simply create a
    /// `thorium::Thorium` and use the handler within that instead.
    ///
    /// # Arguments
    ///
    /// * `host` - url/ip of the Thorium api
    /// * `token` - The token used for authentication
    /// * `client` - The reqwest client to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::Trees;
    ///
    /// let client = reqwest::Client::new();
    /// let trees = Trees::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build trees route handler
        Trees {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only inlcude blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        #[derive(Clone)]
        pub struct TreesBlocking {
            /// The host to talk to the Thorium api at
            host: String,
            /// The token to use for auth
            token: String,
            /// A client to use when making requests
            client: reqwest::Client,
        }

        impl TreesBlocking {
            /// creates a new blocking jobs handler
            ///
            /// Instead of directly creating this handler you likely want to simply create a
            /// `thorium::ThoriumBlocking` and use the handler within that instead.
            ///
            ///
            /// # Arguments
            ///
            /// * `host` - url/ip of the Thorium api
            /// * `token` - The token used for authentication
            /// * `client` - The reqwest client to use
            ///
            /// # Examples
            ///
            /// ```
            /// use thorium::client::TreesBlocking;
            ///
            /// let jobs = TreesBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                TreesBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Trees {
    /// Start a new tree of data in Thorium
    ///
    /// # Arguments
    ///
    /// * `opts` - The query params to use when creating this tree
    /// * `query` - The query to use to create this tree
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{SubmissionUpdate, TreeOpts, TreeQuery};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // use the default query params
    /// let opts = TreeOpts::default();
    /// // build our initial tree query
    /// let query = TreeQuery::default()
    ///   // have a sample to build a tree from
    ///   .sample("856926b48a936b50e92682807bdae12d5ce39abf509d4c0be82e1327b548705f");
    /// // Generate a tree in Thorium
    /// let tree = thorium.trees.start(&opts, &query).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Trees::start", skip_all, err(Debug))
    )]
    pub async fn start(&self, opts: &TreeOpts, query: &TreeQuery) -> Result<Tree, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/trees/", base = &self.host,);
        // build our query params
        let mut query_params = vec![("limit", opts.limit.to_string())];
        add_query!(query_params, "gather_parents", opts.gather_parents);
        add_query!(query_params, "gather_related", opts.gather_related);
        add_query!(
            query_params,
            "gather_tag_children",
            opts.gather_tag_children
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(query)
            .query(&query_params);
        // send this request and build a generic job from the response
        send_build!(self.client, req, Tree)
    }

    /// Grow an existing tree of data in Thorium
    ///
    /// # Arguments
    ///
    /// * `opts` - The query params to use when creating this tree
    /// * `query` - The query to use to create this tree
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{SubmissionUpdate, TreeOpts, TreeGrowQuery};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have an existing tree to grow from
    /// # let cursor = uuid::Uuid::new_v4();
    /// // use the default query params
    /// let opts = TreeOpts::default();
    /// // build our initial tree query
    /// let query = TreeGrowQuery::default()
    ///   // have a node to grow this tree from
    ///   .add_growable(1234);
    /// // Generate a tree in Thorium
    /// let tree = thorium.trees.grow(cursor, &opts, &query).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Trees::grow", skip_all, err(Debug))
    )]
    pub async fn grow(
        &self,
        cursor: Uuid,
        opts: &TreeOpts,
        query: &TreeGrowQuery,
    ) -> Result<Tree, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/trees/{cursor}",
            base = &self.host,
            cursor = cursor
        );
        // build our query params
        let mut query_params = vec![("limit", opts.limit.to_string())];
        add_query!(query_params, "gather_parents", opts.gather_parents);
        add_query!(query_params, "gather_related", opts.gather_related);
        add_query!(
            query_params,
            "gather_tag_children",
            opts.gather_tag_children
        );
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .json(query)
            .query(&query_params);
        // send this request and build a generic job from the response
        send_build!(self.client, req, Tree)
    }
}
