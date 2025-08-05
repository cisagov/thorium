use super::Error;
use crate::models::{
    Backup, Cursor, ImageScaler, Node, NodeGetParams, NodeListLine, NodeListParams,
    NodeRegistration, NodeUpdate, SystemInfo, SystemSettings, SystemSettingsResetParams,
    SystemSettingsUpdate, SystemSettingsUpdateParams, SystemStats, Worker, WorkerDeleteMap,
    WorkerRegistrationList, WorkerUpdate,
};
use crate::{add_query, add_query_list, send, send_build};

/// system handler for the Thorium client
#[derive(Clone)]
pub struct System {
    /// url/ip of the Thorium ip
    host: String,
    /// token to use for auth
    token: String,
    /// reqwest client object
    client: reqwest::Client,
}

impl System {
    /// Creates a new system handler
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
    /// use thorium::client::System;
    ///
    /// let client = reqwest::Client::new();
    /// let systems = System::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build system route handler
        System {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// system handler for the Thorium client
        #[derive(Clone)]
        pub struct SystemBlocking {
            /// url/ip of the Thorium ip
            host: String,
            /// token to use for auth
            token: String,
            /// reqwest client object
            client: reqwest::Client,
        }

        impl SystemBlocking {
            /// creates a new blocking system handler
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
            /// use thorium::client::SystemBlocking;
            ///
            /// let system = SystemBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build system route handler
                SystemBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl System {
    /// Inits [`SystemInfo`] in Thorium
    ///
    /// This will overwrite the current system info if its called on a already initialized Thorium
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
    /// // initialize Thorium's system info
    /// thorium.system.init().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn init(&self) -> Result<reqwest::Response, Error> {
        // build url for initializing system settings
        let url = format!("{}/api/system/init", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Gets current [`SystemInfo`] for Thorium
    ///
    /// # Arguments
    ///
    /// * `reset` - Whether to reset the shared system flags during this request
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
    /// // initialize Thorium's system info
    /// thorium.system.init().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get_info(&self, reset: Option<ImageScaler>) -> Result<SystemInfo, Error> {
        // build url for getting system info
        let url = format!("{}/api/system/", self.host);
        // build request
        let req = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&[("reset", reset)]);
        // send this request and build a SystemInfo from the response
        send_build!(self.client, req, SystemInfo)
    }

    /// Sets the reset cache flag in the [`SystemInfo`] object in the API
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
    /// // initialize Thorium's system info
    /// thorium.system.reset_cache().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn reset_cache(&self) -> Result<reqwest::Response, Error> {
        // build url for resetting the system cache
        let url = format!("{}/api/system/cache/reset", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send this request and build a SystemInfo from the response
        send!(self.client, req)
    }

    /// Gets the current [`SystemSettings`] from Thorium
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
    /// // get system settings from Thorium
    /// thorium.system.get_settings().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get_settings(&self) -> Result<SystemSettings, Error> {
        // build url getting system settings
        let url = format!("{}/api/system/settings", self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a SystemSettings from the response
        send_build!(self.client, req, SystemSettings)
    }

    /// Resets [`SystemSettings`] in Thorium and optionally runs an automatic consistency scan to
    /// ensure data in Thorium adheres to the new settings (see [`System::consistency_scan`]);
    /// additionally signals the scaler to refresh its cache after the scan
    ///
    /// Because the scan can take awhile, you might instead opt to run
    /// with no scan first to reset the settings and then either
    /// fix inconsistent data manually or run [`System::consistency_scan`] at a
    /// later time.
    ///
    /// # Arguments
    ///
    /// * `params` - The system settings reset params
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::SystemSettingsResetParams;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // define our params; the consistency scan is run by default, or we can specify
    /// // no_scan to skip it
    /// let params = SystemSettingsResetParams::default().no_scan();
    /// // reset system settings in Thorium
    /// thorium.system.reset_settings(&params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn reset_settings(
        &self,
        params: &SystemSettingsResetParams,
    ) -> Result<reqwest::Response, Error> {
        // build url
        let url = format!("{}/api/system/settings/reset", self.host);
        // build our query
        let query = vec![("scan", params.scan.to_string())];
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send this request
        send!(self.client, req)
    }

    /// Updates the [`SystemSettings`] in Thorium and optionally runs an automatic consistency scan to
    /// ensure data in Thorium adheres to the new settings (see [`System::consistency_scan`]);
    /// additionally signals the scaler to fresh its cache after the scan
    ///
    /// Because the scan can take awhile, you might instead opt to run
    /// with no scan first to update the settings and then either
    /// fix inconsistent data manually or run [`System::consistency_scan`] at a
    /// later time.
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to apply to system settings in Thorium
    /// * `params` - The system settings update params
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{SystemSettingsUpdate, SystemSettingsUpdateParams};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build our system settings update
    /// let update = SystemSettingsUpdate::default()
    ///     // reserve 100 cores for use outside of Thorium
    ///     .reserved_cpu("100")
    ///     // fairly share 600 cores, 1 Ti of Ram, and 256 Gi of ephemeral storage
    ///     .fairshare_cpu("600")
    ///     .fairshare_memory("1Ti")
    ///     .fairshare_storage("256Gi");
    /// // build our params; the consistency scan is run by default or we can specify
    /// // no_scan to skip it
    /// let params = SystemSettingsUpdateParams::default().no_scan();
    /// // update system settings in Thorium
    /// thorium.system.update_settings(&update, &params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update_settings(
        &self,
        update: &SystemSettingsUpdate,
        params: &SystemSettingsUpdateParams,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating settings
        let url = format!("{}/api/system/settings", self.host);
        // build our query
        let query = vec![("scan", params.scan.to_string())];
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .json(update)
            .query(&query);
        // send this request
        send!(self.client, req)
    }

    /// Performs a scan of Thorium data, checking that all data is compliant with current Thorium
    /// [`SystemSettings`] and cleaning/marking/modifying data that isn't; additionally signals
    /// the scaler to refresh its cache after the scan
    ///
    /// Currently this only applies to images with host path mounts that may not be on the configured
    /// whitelist after a settings update.
    ///
    /// ```
    /// use thorium::Thorium ;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // perform a settings consistency scan
    /// thorium.system.consistency_scan().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn consistency_scan(&self) -> Result<reqwest::Response, Error> {
        // build url for updating settings
        let url = format!("{}/api/system/settings/scan", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Gets the current [`SystemStats`] from Thorium
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
    /// // get system stats from Thorium
    /// thorium.system.stats().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn stats(&self) -> Result<SystemStats, Error> {
        // build url for getting system stats
        let url = format!("{}/api/system/stats", self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a SystemStats from the response
        send_build!(self.client, req, SystemStats)
    }

    /// Cleans up reaction lists in Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::SystemSettingsUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // clean up reaction lists in Thorium
    /// thorium.system.cleanup().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn cleanup(&self) -> Result<reqwest::Response, Error> {
        // build url for cleaning up the system
        let url = format!("{}/api/system/cleanup", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Get a backup of data in Thorium
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
    /// // get a backup from Thorium
    /// let backup = thorium.system.backup().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn backup(&self) -> Result<Backup, Error> {
        // build url for backing up
        let url = format!("{}/api/system/backup", self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a Backup from the response
        send_build!(self.client, req, Backup)
    }

    /// restore a backup of data in Thorium
    ///
    /// # Arguments
    ///
    /// * `backup` - The backup to restore from
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
    /// // get a backup from Thorium or load an older one from disk somewhere
    /// let backup = thorium.system.backup().await?;
    /// // restore that backup
    /// thorium.system.restore(&backup).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn restore(&self, backup: &Backup) -> Result<reqwest::Response, Error> {
        // build url for restoring a backup
        let url = format!("{}/api/system/restore", self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(backup);
        // send this request
        send!(self.client, req)
    }

    /// Register new nodes
    ///
    /// # Arguments
    ///
    /// * `node` - The node to register
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{NodeRegistration, Resources};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get this nodes resources
    /// let resources = Resources::default();
    /// // Build the node registration object
    /// let node = NodeRegistration::new("CornCluster", "Corn1", resources);
    /// // register this node
    /// thorium.system.register_node(&node).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn register_node(&self, node: &NodeRegistration) -> Result<reqwest::Response, Error> {
        // build url for registering a node
        let url = format!("{}/api/system/nodes/", self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(node);
        // send this request
        send!(self.client, req)
    }

    /// Gets info on a specific node
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster this node is in
    /// * `node` - The node to get info on
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::NodeGetParams;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get info on the corn node
    /// thorium.system.get_node("CornCluster", "Corn1", &NodeGetParams::default()).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get_node(
        &self,
        cluster: &str,
        node: &str,
        params: &NodeGetParams,
    ) -> Result<Node, Error> {
        // build url for registering a node
        let url = format!("{}/api/system/nodes/{}/{}", self.host, cluster, node);
        // build empty query params
        let mut query = Vec::default();
        add_query_list!(query, "scalers[]", params.scalers);
        // build request
        let req = self
            .client
            .get(&url)
            .query(&query)
            .header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, Node)
    }

    /// Updates an existing node in Thorium
    ///
    /// # Arguments
    ///
    /// * `cluster` - The cluster of the node to update
    /// * `node` - The node to update
    /// * `update` - The update to apply
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{NodeUpdate, NodeHealth, Resources};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get this nodes new resources
    /// let resources = Resources::default();
    /// // Build the node update object
    /// let update = NodeUpdate::new(NodeHealth::Healthy, resources);
    /// // update this node
    /// thorium.system.update_node("CornCluster", "Corn1", &update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update_node(
        &self,
        cluster: &str,
        node: &str,
        update: &NodeUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a node
        let url = format!("{}/api/system/nodes/{}/{}", self.host, cluster, node);
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .json(update);
        // send this request
        send!(self.client, req)
    }

    /// List nodes in Thorium
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to use when listing nodes
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{NodeListParams};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Build params to list nodes from the CornCluster cluster
    /// let params = NodeListParams::default().cluster("CornCluster");
    /// // list the node names in this cluster
    /// thorium.system.list_nodes(&params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn list_nodes(&self, params: &NodeListParams) -> Result<Cursor<NodeListLine>, Error> {
        // build url for listing node names
        let url = format!("{}/api/system/nodes/", self.host);
        // build our query params
        let mut query = vec![("page_size", params.page_size.to_string())];
        add_query!(query, "cursor", params.cursor);
        add_query_list!(query, "clusters[]", params.clusters);
        add_query_list!(query, "scalers[]", params.scalers);
        // get the data for this request and create our cursor
        Cursor::new(
            &url,
            params.page_size,
            params.limit,
            &self.token,
            &query,
            &self.client,
        )
        .await
    }

    /// List nodes with details in Thorium
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to use when listing node details
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{NodeListParams};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Build params to list nodes from the CornCluster cluster
    /// let params = NodeListParams::default().cluster("CornCluster");
    /// // list the node details in this cluster
    /// thorium.system.list_node_details(&params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn list_node_details(&self, params: &NodeListParams) -> Result<Cursor<Node>, Error> {
        // build url for listing node details
        let url = format!("{}/api/system/nodes/details/", self.host);
        // build our query params
        let mut query = vec![("page_size", params.page_size.to_string())];
        add_query!(query, "cursor", params.cursor);
        add_query_list!(query, "clusters[]", params.clusters);
        add_query_list!(query, "scalers[]", params.scalers);
        // get the data for this request and create our cursor
        Cursor::new(
            &url,
            params.page_size,
            params.limit,
            &self.token,
            &query,
            &self.client,
        )
        .await
    }

    /// Register new workers for a specific scaler
    ///
    /// #Arguments
    ///
    /// * `scaler` - The scaler to add workers too
    /// * `workers` - The workers to register
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{ImageScaler, WorkerRegistrationList, Resources, Pools};
    /// use std::ops::Add;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the resources this worker needs
    /// let res = Resources::new(4000, 4096, 0, 1);
    /// // Build our worker registration list
    /// let list = WorkerRegistrationList::default()
    ///   .add("Corncluster", "Corn1", "Worker1", "user1", "group1", "pipe1", "stage1", res, Pools::Deadline);
    /// // register this worker
    /// thorium.system.register_workers(ImageScaler::K8s, &list).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn register_workers(
        &self,
        scaler: ImageScaler,
        workers: &WorkerRegistrationList,
    ) -> Result<reqwest::Response, Error> {
        // build url for registering a new worker
        let url = format!("{}/api/system/worker/{}", self.host, scaler);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(workers);
        // send this request
        send!(self.client, req)
    }

    /// Updates a workers current status
    ///
    /// #Arguments
    ///
    /// * `name`  - The name of this worker
    /// * `update` - The update to apply to this worker
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{ImageScaler, WorkerUpdate, WorkerStatus};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Build the update to apply to this worker
    /// let update = WorkerUpdate::new(WorkerStatus::Running);
    /// // update this k8s worker's status
    /// thorium.system.update_worker("Corn1", &update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update_worker(
        &self,
        name: &str,
        update: &WorkerUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for registering a new worker
        let url = format!("{}/api/system/worker/{}", self.host, name);
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .json(update);
        // send this request
        send!(self.client, req)
    }

    /// Gets info about a specific worker
    ///
    /// #Arguments
    ///
    /// * `name`  - The name of this worker
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::ImageScaler;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get info about this worker
    /// thorium.system.get_worker("Corn1").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get_worker(&self, name: &str) -> Result<Worker, Error> {
        // build url for registering a new worker
        let url = format!("{}/api/system/worker/{}", self.host, name);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, Worker)
    }

    /// Removes no longer active workers for a specific scaler
    ///
    /// #Arguments
    ///
    /// * `scaler` - The scaler to remove workers from
    /// * `workers` - The workers to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{ImageScaler, WorkerDeleteMap};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Build our worker delete map
    /// let deletes = WorkerDeleteMap::default()
    ///   .add("Worker1");
    /// // delete this worker
    /// thorium.system.delete_workers(ImageScaler::K8s, &deletes).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete_workers(
        &self,
        scaler: ImageScaler,
        workers: &WorkerDeleteMap,
    ) -> Result<reqwest::Response, Error> {
        // build url for registering a new worker
        let url = format!("{}/api/system/worker/{}", self.host, scaler);
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token)
            .json(workers);
        // send this request
        send!(self.client, req)
    }
}
