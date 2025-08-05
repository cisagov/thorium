use std::collections::HashMap;

use bytes::Bytes;
use uuid::Uuid;

use super::{Cursor, Error, LogsCursor};
use crate::models::{
    BulkReactionResponse, Reaction, ReactionCreation, ReactionListParams, ReactionRequest,
    ReactionStatus, ReactionUpdate, StageLogs, StageLogsAdd, StatusUpdate,
};
use crate::{send, send_build, send_bytes};

/// An async Reactions handler for the Thorium client
#[derive(Clone)]
pub struct Reactions {
    host: String,
    /// token to use for auth
    token: String,
    client: reqwest::Client,
}

impl Reactions {
    /// Creates a new reactions handler
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
    /// use thorium::client::Reactions;
    ///
    /// let client = reqwest::Client::new();
    /// let reactions = Reactions::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Reactions {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// A blocking Reactions handler for the Thorium client
        #[derive(Clone)]
        pub struct ReactionsBlocking {
            host: String,
            /// token to use for auth
            token: String,
            client: reqwest::Client,
        }

        impl ReactionsBlocking {
            /// creates a new blocking reactions handler
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
            /// use thorium::client::ReactionsBlocking;
            ///
            /// let reactions = ReactionsBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                ReactionsBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Reactions {
    /// Creates a [`Reaction`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `data` - The reaction request to use to create a reaction
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{ReactionRequest, GenericJobArgs};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the args for our corn stage
    /// let corn_args = GenericJobArgs::default()
    ///     .kwarg("type", vec!("corn"));
    /// // build the args for our soybean stage
    /// let soy_args = GenericJobArgs::default()
    ///     .kwarg("type", vec!("soybean"));
    /// // build a reaction request
    /// let react_req = ReactionRequest::new("Corn", "Harvest")
    ///     .sla(86400)
    ///     .args("CornHarvest", corn_args)
    ///     .args("SoyBeanHarvest", soy_args);
    /// // create a reaction in Thorium
    /// let react_create = thorium.reactions.create(&react_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::create", skip_all, err(Debug))
    )]
    pub async fn create(&self, data: &ReactionRequest) -> Result<ReactionCreation, Error> {
        // build request
        let req = self
            .client
            .post(format!("{}/api/reactions/", self.host))
            .header("authorization", &self.token)
            .json(&data);
        // send request and build a reaction creation
        send_build!(self.client, req, ReactionCreation)
    }

    /// Create [`Reaction`]s in bulk
    ///
    /// # Arguments
    ///
    /// * `reqs` - The reaction requests to create reactions in bulk
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{ReactionRequest, GenericJobArgs};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the args for our corn stage
    /// let corn_args = GenericJobArgs::default()
    ///     .kwarg("type", vec!("corn"));
    /// // build the args for our soybean stage
    /// let soy_args = GenericJobArgs::default()
    ///     .kwarg("type", vec!("soybean"));
    /// // build a reaction request
    /// let react_req = ReactionRequest::new("Corn", "Harvest")
    ///     .sla(86400)
    ///     .args("CornHarvest", corn_args)
    ///     .args("SoyBeanHarvest", soy_args);
    /// // This is going to use the same reaction request 10 times but it works the same with
    /// // different reaction requests
    /// let mut reqs = Vec::with_capacity(10);
    /// for _ in 0..10 {
    ///     reqs.push(react_req.clone());
    /// }
    /// // create 10 reactions in Thorium
    /// let react_creates = thorium.reactions.create_bulk(&reqs).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::create_bulk", skip_all, err(Debug))
    )]
    pub async fn create_bulk(
        &self,
        reqs: &[ReactionRequest],
    ) -> Result<BulkReactionResponse, Error> {
        // build request
        let req = self
            .client
            .post(format!("{}/api/reactions/bulk/", self.host))
            .header("authorization", &self.token)
            .json(&reqs);
        // send request and build a vector of reaction creations
        send_build!(self.client, req, BulkReactionResponse)
    }

    /// Create [`Reaction`]s in bulk for multiple users
    ///
    /// # Arguments
    ///
    /// * `reqs` - The reaction requests to create reactions in bulk for multiple users
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{ReactionRequest, GenericJobArgs};
    /// use std::collections::HashMap;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the args for our corn stage
    /// let corn_args = GenericJobArgs::default()
    ///     .kwarg("type", vec!("corn"));
    /// // build the args for our soybean stage
    /// let soy_args = GenericJobArgs::default()
    ///     .kwarg("type", vec!("soybean"));
    /// // build a reaction request
    /// let react_req = ReactionRequest::new("Corn", "Harvest")
    ///     .sla(86400)
    ///     .args("CornHarvest", corn_args)
    ///     .args("SoyBeanHarvest", soy_args);
    /// // Add this reaction, or multiple, for each user you want
    /// // in this example its the same reaction request but in reality it would likely be
    /// //different ones
    /// let mut map = HashMap::default();
    /// map.insert("mcarson".to_owned(), vec![react_req.clone()]);
    /// map.insert("alice".to_owned(), vec![react_req]);
    /// // create reactions for each user in the map
    /// let react_creates = thorium.reactions.create_bulk_by_user(&map).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "Thorium::Reactions::create_bulk_by_user",
            skip_all,
            err(Debug)
        )
    )]
    pub async fn create_bulk_by_user(
        &self,
        reqs: &HashMap<String, Vec<ReactionRequest>>,
    ) -> Result<HashMap<String, BulkReactionResponse>, Error> {
        // build request
        let req = self
            .client
            .post(format!("{}/api/reactions/bulk/by/user/", self.host))
            .header("authorization", &self.token)
            .json(&reqs);
        // send request and build a vector of reaction creations
        send_build!(self.client, req, HashMap<String, BulkReactionResponse>)
    }

    /// Gets details about a [`Reaction`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction is in
    /// * `id` - The id of the reaction to get details about
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have an id for a reaction you want to retrieve
    /// let id = Uuid::parse_str("d86ce41a-4a5b-43b5-aef9-bf90ff5d09ba")?;
    /// // get details on this reaction
    /// let reaction = thorium.reactions.get("Corn", &id).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::get", skip(self), fields(id = id.to_string()), err(Debug))
    )]
    pub async fn get(&self, group: &str, id: &Uuid) -> Result<Reaction, Error> {
        // build url
        let url = format!(
            "{host}/api/reactions/{group}/{id}",
            host = &self.host,
            group = group,
            id = id
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request and build a reaction
        send_build!(self.client, req, Reaction)
    }

    /// Sends logs for a specific stage in a [`Reaction`] to Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The group this job is within
    /// * `reaction` - The reaction this job is from
    /// * `stage` - The stage these logs come from
    /// * `logs` - The new logs to save
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::StageLogsAdd};
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have an id for a reaction you want to save logs for
    /// let id = Uuid::parse_str("d86ce41a-4a5b-43b5-aef9-bf90ff5d09ba")?;
    /// let logs = StageLogsAdd::default()
    ///     .logs(vec!("these", "are", "new", "logs"));
    /// // send the new logs to Thorium
    /// thorium.reactions.add_stage_logs("Corn", &id, "CornHarvest", &logs).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "Thorium::Reactions::add_stage_logs",
            skip(self, logs),
            fields(reaction = reaction.to_string()),
            err(Debug)
        )
    )]
    pub async fn add_stage_logs(
        &self,
        group: &str,
        reaction: &Uuid,
        stage: &str,
        logs: &StageLogsAdd,
    ) -> Result<reqwest::Response, Error> {
        // build url
        let url = format!(
            "{host}/api/reactions/logs/{group}/{reaction}/{stage}",
            host = &self.host,
            group = group,
            reaction = reaction,
            stage = stage
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(&logs);
        // send request
        send!(self.client, req)
    }

    /// Gets logs from a specific stage of a [`Reaction`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction is in
    /// * `id` - The id of the reaction to get details about
    /// * `stage` - The stage to get logs for
    /// * `params` - The params to set when retrieving logs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::ReactionListParams;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have an id for a reaction you want to retrieve
    /// let id = Uuid::parse_str("d86ce41a-4a5b-43b5-aef9-bf90ff5d09ba")?;
    /// // create params
    /// let params = ReactionListParams::default().limit(100_000);
    /// // get the logs for this reaction and stage
    /// let logs = thorium.reactions.logs("Corn", &id, "Harvest", &params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "Thorium::Reactions::logs",
            skip(self),
            fields(id = id.to_string()),
            err(Debug)
        )
    )]
    pub async fn logs(
        &self,
        group: &str,
        id: &Uuid,
        stage: &str,
        params: &ReactionListParams,
    ) -> Result<StageLogs, Error> {
        // build url
        let url = format!(
            "{host}/api/reactions/logs/{group}/{id}/{stage}",
            host = &self.host,
            group = group,
            id = id,
            stage = stage,
        );
        // build query
        let query = vec![
            ("cursor", params.cursor.to_string()),
            ("limit", params.limit.to_string()),
        ];
        // build request
        let req = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send request and build a reaction
        send_build!(self.client, req, StageLogs)
    }

    /// Gets a [`LogsCursor`] for a specific stage in a [`Reaction`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction is in
    /// * `id` - The id of the reaction to get details about
    /// * `stage` - The stage to get logs for
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have an id for a reaction you want to retrieve
    /// let id = Uuid::parse_str("d86ce41a-4a5b-43b5-aef9-bf90ff5d09ba")?;
    /// // get the logs for this reaction and stage
    /// let cursor = thorium.reactions.logs_cursor("Corn", &id, "Harvest");
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "Thorium::Reactions::logs_cursor",
            skip(self),
            fields(id = id.to_string()),
        )
    )]
    #[must_use]
    pub fn logs_cursor(&self, group: &str, id: &Uuid, stage: &str) -> LogsCursor {
        // build url
        let url = format!(
            "{host}/api/reactions/logs/{group}/{id}/{stage}",
            host = &self.host,
            group = group,
            id = id,
            stage = stage,
        );
        // build new cursor
        LogsCursor::new(url, &self.token, &self.client)
    }

    /// Gets status logs for a reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction is in
    /// * `id` - The id of the reaction to get details about
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have an id for a reaction you want to retrieve
    /// let id = Uuid::parse_str("d86ce41a-4a5b-43b5-aef9-bf90ff5d09ba")?;
    /// // get the status logs for this reaction
    /// let logs = thorium.reactions.status_logs("Corn", &id).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(
            name = "Thorium::Reactions::status_logs",
            skip(self),
            fields(id = id.to_string()),
            err(Debug)
        )
    )]
    pub async fn status_logs(&self, group: &str, id: &Uuid) -> Result<Vec<StatusUpdate>, Error> {
        // build url
        let url = format!(
            "{host}/api/reactions/logs/{group}/{id}",
            host = &self.host,
            group = group,
            id = id,
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request and build a reaction
        send_build!(self.client, req, Vec<StatusUpdate>)
    }

    /// Lists [`Reaction`] names in a group for a specific pipeline
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `pipeline` - The pipeline to list reactions from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 50 reaction names from Thorium (limit is weakly enforced)
    /// let cursor = thorium.reactions.list("Corn", "CornHarvest").limit(50).exec().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::list", skip(self))
    )]
    #[must_use]
    pub fn list(&self, group: &str, pipeline: &str) -> Cursor<Reaction> {
        // build url for listing reactions
        let url = format!(
            "{base}/api/reactions/list/{group}/{pipeline}/",
            base = self.host,
            group = group,
            pipeline = pipeline
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Lists [`Reaction`] names with a status in a group for a specific pipeline
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `pipeline` - The pipeline to list reactions from
    /// * `status` - The status reactions should have
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::ReactionStatus};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 50 pending reaction names from Thorium (limit is weakly enforced)
    /// let reactions = thorium.reactions
    ///     .list_status("Corn", "CornHarvest", &ReactionStatus::Created)
    ///     .limit(50)
    ///     .next().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::list_status", skip(self))
    )]
    #[must_use]
    pub fn list_status(
        &self,
        group: &str,
        pipeline: &str,
        status: &ReactionStatus,
    ) -> Cursor<Reaction> {
        // build url for listing reactions
        let url = format!(
            "{base}/api/reactions/status/{group}/{pipeline}/{status}/",
            base = self.host,
            group = group,
            pipeline = pipeline,
            status = status
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Lists [`Reaction`] names with a tag in a group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `tag` - The tag reactions should have
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 50 reaction names from Thorium (limit is weakly enforced) with the woot tag
    /// let reactions = thorium.reactions.list_tag("Corn", "woot").limit(50).next().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::list_tag", skip(self))
    )]
    #[must_use]
    pub fn list_tag(&self, group: &str, tag: &str) -> Cursor<Reaction> {
        // build url for listing reactions
        let url = format!(
            "{base}/api/reactions/tag/{group}/{tag}/",
            base = self.host,
            group = group,
            tag = tag
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Lists [`Reaction`] names with a set status in an entire group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `tag` - The tag reactions should have
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::ReactionStatus};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 50 reaction names from Thorium (limit is weakly enforced) with a status
    /// let reactions = thorium.reactions.list_group("Corn", &ReactionStatus::Started).limit(50).next().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::list_group", skip(self))
    )]
    #[must_use]
    pub fn list_group(&self, group: &str, status: &ReactionStatus) -> Cursor<Reaction> {
        // build url for listing reactions
        let url = format!(
            "{base}/api/reactions/group/{group}/{status}/",
            base = self.host,
            group = group,
            status = status
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Lists sub[`Reaction`] IDs for a parent reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group our parent reaction is in
    /// * `reaction` - The parent reaction to list sub reactions from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // in a real use case this would be an actual reaction uuid
    /// let reaction = Uuid::new_v4();
    /// // list up to 50 sub reaction IDs from Thorium (limit is weakly enforced)
    /// let reactions = thorium.reactions.list_sub("Corn", &reaction).limit(50).next().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::list_sub", skip(self), fields(reaction = reaction.to_string()))
    )]
    #[must_use]
    pub fn list_sub(&self, group: &str, reaction: &Uuid) -> Cursor<Reaction> {
        // build url for listing reactions
        let url = format!(
            "{base}/api/reactions/sub/{group}/{reaction}/",
            base = self.host,
            group = group,
            reaction = reaction
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Lists sub[`Reaction`] IDs for a parent reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group our parent reaction is in
    /// * `reaction` - The parent reaction to list sub reactions from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::ReactionStatus};
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // in a real use case this would be an actual reaction uuid
    /// let reaction = Uuid::new_v4();
    /// // list up to 50 sub reaction IDs from Thorium (limit is weakly enforced)
    /// let reactions = thorium.reactions.list_sub_status("Corn", &reaction, &ReactionStatus::Created)
    ///     .limit(50)
    ///     .next()
    ///     .await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::list_sub_status", skip(self), fields(reaction = reaction.to_string()))
    )]
    #[must_use]
    pub fn list_sub_status(
        &self,
        group: &str,
        reaction: &Uuid,
        status: &ReactionStatus,
    ) -> Cursor<Reaction> {
        // build url for listing reactions
        let url = format!(
            "{base}/api/reactions/sub/{group}/{reaction}/{status}/",
            base = self.host,
            group = group,
            reaction = reaction,
            status = status,
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Updates a [`Reaction`]s data
    ///
    /// This will naively update the arguments for stages that have already completed.
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `id` - The reaction to update
    /// * `update` - The updates to apply
    ///
    /// # Examples
    ///
    /// ```
    /// use uuid::Uuid;
    /// use thorium::{Thorium, models::ReactionUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a reaction update
    /// let update = ReactionUpdate::default()
    ///     .tag("NewCornTag");
    /// // update our reaction
    /// let reaction = Uuid::parse_str("e0ca2720-50e0-4103-a412-344bbb714240")?;
    /// let details = thorium.reactions.update("Corn", &reaction, &update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::update", skip(self, update), fields(id = id.to_string()), err(Debug))
    )]
    pub async fn update(
        &self,
        group: &str,
        id: &Uuid,
        update: &ReactionUpdate,
    ) -> Result<Reaction, Error> {
        // build url for updating a reaction
        let url = format!(
            "{base}/api/reactions/{group}/{id}",
            base = self.host,
            group = group,
            id = id
        );
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .json(update);
        // send request and build a reaction
        send_build!(self.client, req, Reaction)
    }

    /// Deletes a [`Reaction`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group to delete a reactions from
    /// * `id` - The reaction to delete
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // delete our reaction
    /// let reaction = Uuid::parse_str("e0ca2720-50e0-4103-a412-344bbb714240")?;
    /// let details = thorium.reactions.delete("Corn", &reaction).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::delete", skip(self), fields(id = id.to_string()), err(Debug))
    )]
    pub async fn delete(&self, group: &str, id: &Uuid) -> Result<reqwest::Response, Error> {
        // build url for deleting a reaction
        let url = format!(
            "{base}/api/reactions/{group}/{id}",
            base = self.host,
            group = group,
            id = id
        );
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token);
        // send request
        send!(self.client, req)
    }

    /// Downloads an ephemeral file for a  [`Reaction`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction is from
    /// * `id` - The reaction to download an ephemeral file for
    /// * `name` - The name of the ephemeral file to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // download an ephemeral file from this reaction
    /// let reaction = Uuid::parse_str("e0ca2720-50e0-4103-a412-344bbb714240")?;
    /// let file = thorium.reactions.download_ephemeral("Corn", &reaction, "file.txt").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Thorium::Reactions::download_ephemeral", skip(self), fields(id = id.to_string()), err(Debug))
    )]
    pub async fn download_ephemeral(
        &self,
        group: &str,
        id: &Uuid,
        name: &str,
    ) -> Result<Bytes, Error> {
        // build url for deleting a reaction
        let url = format!(
            "{base}/api/reactions/ephemeral/{group}/{id}/{name}",
            base = self.host,
            group = group,
            id = id,
            name = name,
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request
        send_bytes!(self.client, req)
    }
}
