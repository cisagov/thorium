use uuid::Uuid;

use super::traits::{GenericClient, NotificationsClient};
use super::{Cursor, Error};
use crate::models::{
    Notification, NotificationParams, NotificationRequest, Pipeline, PipelineKey, PipelineRequest,
    PipelineUpdate,
};
use crate::{send, send_build};

#[derive(Clone)]
pub struct Pipelines {
    host: String,
    /// token to use for auth
    token: String,
    client: reqwest::Client,
}

impl Pipelines {
    /// Creates a new pipelines handler
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
    /// use thorium::client::Pipelines;
    ///
    /// let client = reqwest::Client::new();
    /// let pipelines = Pipelines::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Pipelines {
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
        pub struct PipelinesBlocking {
            host: String,
            /// token to use for auth
            token: String,
            client: reqwest::Client,
        }

        impl PipelinesBlocking {
            /// creates a new blocking pipelines handler
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
            /// use thorium::client::PipelinesBlocking;
            ///
            /// let pipelines = PipelinesBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                PipelinesBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Pipelines {
    /// Creates a [`Pipeline`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `pipeline_request` - The pipeline request to use to create a pipeline
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::PipelineRequest};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a pipeline request
    /// let order = serde_json::json!(vec!(vec!("CornHarvest", "SoyBeanHarvest")));
    /// let pipe_req = PipelineRequest::new("Corn", "Harvest", order)
    ///     .sla(86400);
    /// // create a pipeline in Thorium
    /// thorium.pipelines.create(&pipe_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn create(&self, pipe_req: &PipelineRequest) -> Result<reqwest::Response, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/pipelines/", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(pipe_req);
        // send this request
        send!(self.client, req)
    }

    /// Gets details on a [`Pipeline`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group this pipeline is in
    /// * `pipeline` - The name of the pipeline to get details on
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
    /// // get details on a pipeline in Thorium
    /// let pipeline = thorium.pipelines.get("Corn", "CornHarvest").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get(&self, group: &str, pipeline: &str) -> Result<Pipeline, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/pipelines/data/{group}/{pipeline}",
            base = self.host,
            group = group,
            pipeline = pipeline
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a pipeline from the response
        send_build!(self.client, req, Pipeline)
    }

    /// Updates a [`Pipeline`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The group this pipeline is in
    /// * `pipeline` - The name of the pipeline to update
    /// * `update` - The update to apply to this pipeline
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::PipelineUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a pipeline update
    /// let update = PipelineUpdate::default()
    ///     .order(
    ///         vec![
    ///             vec!["CornPlanter".to_string(), "CornGrower".to_string()],
    ///             vec!["CornHarvester".to_string()],
    ///         ]
    ///     )
    ///     .sla(1000)
    ///     .description("Updated description");
    /// // update this pipeline in Thorium
    /// thorium.pipelines.update("corn", "harvest", &update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update(
        &self,
        group: &str,
        pipeline: &str,
        update: &PipelineUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a pipeline
        let url = format!(
            "{base}/api/pipelines/{group}/{pipeline}",
            base = self.host,
            group = group,
            pipeline = pipeline
        );
        // build request
        let req = self
            .client
            .patch(&url)
            .json(update)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Deletes a [`Pipeline`] from Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The group this pipeline is in
    /// * `pipeline` - The name of the pipeline to delete
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::PipelineUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // delete this pipeline from Thorium
    /// thorium.pipelines.delete("corn", "harvest").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete(&self, group: &str, pipeline: &str) -> Result<reqwest::Response, Error> {
        // build url for updating a pipeline
        let url = format!(
            "{base}/api/pipelines/{group}/{pipeline}",
            base = self.host,
            group = group,
            pipeline = pipeline
        );
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Lists [`Pipeline`]s in a group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list pipelines from
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
    /// // list up to 50 pipeline names in the Corn group (limit is weakly enforced)
    /// let pipelines = thorium.pipelines.list("Corn").limit(50).next().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[must_use]
    pub fn list(&self, group: &str) -> Cursor<Pipeline> {
        // build url for listing pipelines
        let url = format!(
            "{base}/api/pipelines/list/{group}/",
            base = self.host,
            group = group
        );
        Cursor::new(url, &self.token, &self.client)
    }

    /// Create a notification for a pipeline
    ///
    /// # Arguments
    ///
    /// * `group` - The group containing the pipeline
    /// * `pipeline` - The name of the pipeline
    /// * `req` - The request to create a notification
    /// * `params` - The params to send with this notification create request
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{NotificationRequest, NotificationParams, NotificationLevel};
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // create the request
    /// let req = NotificationRequest::new("This is an example notification!", NotificationLevel::Info);
    /// // set the params for the notification request
    /// // (Note: notifications below the `Error` level will expire automatically by default)
    /// let params = NotificationParams::default().expire(false);
    /// // create the notification for the 'harvest' pipeline in the 'corn' group
    /// thorium.pipelines.create_notification("corn", "harvest", &req, &params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn create_notification<S, T>(
        &self,
        group: S,
        pipeline: T,
        req: &NotificationRequest<Pipeline>,
        params: &NotificationParams,
    ) -> Result<reqwest::Response, Error>
    where
        S: Into<String>,
        T: Into<String>,
    {
        self.create_notification_generic(&PipelineKey::new(group, pipeline), req, params)
            .await
    }

    /// Gets all of a pipeline's notifications
    ///
    /// # Arguments
    ///
    /// * `group` - The group that the pipeline belongs to
    /// * `pipeline` - The pipeline whose notifications we're requesting
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
    /// // retrieve all of the notifications for the 'harvest' pipeline in the 'corn' group
    /// let logs = thorium.pipelines.get_notifications("corn", "harvest").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get_notifications<S, T>(
        &self,
        group: S,
        pipeline: T,
    ) -> Result<Vec<Notification<Pipeline>>, Error>
    where
        S: Into<String>,
        T: Into<String>,
    {
        self.get_notifications_generic(&PipelineKey::new(group, pipeline))
            .await
    }

    /// Deletes a pipeline notification
    ///
    /// # Arguments
    ///
    /// * `group` - The group that the pipeline belongs to
    /// * `pipeline` - The pipeline whose notification we're deleting
    /// * `id` - The id of the pipeline notification to delete
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
    /// // get the id of the notification
    /// let id = Uuid::new_v4();
    /// // delete the notification
    /// thorium.pipelines.delete_notification("corn", "harvest", &id).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete_notification<S, T>(
        &self,
        group: S,
        pipeline: T,
        id: &Uuid,
    ) -> Result<reqwest::Response, Error>
    where
        S: Into<String>,
        T: Into<String>,
    {
        self.delete_notification_generic(&PipelineKey::new(group, pipeline), id)
            .await
    }
}

impl GenericClient for Pipelines {
    /// Provide the base url to the pipelines route in the API
    fn base_url(&self) -> String {
        format!("{}/api/pipelines", self.host)
    }

    /// Provide the configured client from `self`
    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Provide the configured auth token from `self`
    fn token(&self) -> &str {
        &self.token
    }
}

impl NotificationsClient for Pipelines {
    /// The underlying type that has the Notifications (see [`crate::models::backends::NotificationSupport`])
    type NotificationSupport = Pipeline;
}
