//! Groups are how Thorium will let users permission their pipelines and reactions. In
//! order for another user to use or see your pipeline they must also be in the
//! same group as that pipeline.

//! Before you can create anything in Thorium you need to either create or be apart of
//! the group you wish those images, pipelines, or reactions in.

use super::{Cursor, Error};
use crate::models::{Group, GroupRequest, GroupUpdate};
use crate::{send, send_build};

/// group handler for the Thorium client
#[derive(Clone)]
pub struct Groups {
    /// url/ip of the Thorium ip
    host: String,
    /// token to use for auth
    token: String,
    /// reqwest client object
    client: reqwest::Client,
}

impl Groups {
    /// Creates a new group handler
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
    /// use thorium::client::Groups;
    ///
    /// let client = reqwest::Client::new();
    /// let groups = Groups::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new<T: Into<String>>(host: T, token: T, client: &reqwest::Client) -> Self {
        // build basic route handler
        Groups {
            host: host.into(),
            token: token.into(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// group handler for the Thorium client
        #[derive(Clone)]
        pub struct GroupsBlocking {
            /// url/ip of the Thorium ip
            host: String,
            /// token to use for auth
            token: String,
            /// reqwest client object
            client: reqwest::Client,
        }

        impl GroupsBlocking {
            /// creates a new blocking group handler
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
            /// use thorium::client::GroupsBlocking;
            ///
            /// let groups = GroupsBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new<T: Into<String>>(host: T, token: T, client: &reqwest::Client) -> Self {
                // build basic route handler
                GroupsBlocking {
                    host: host.into(),
                    token: token.into(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Groups {
    /// Creates a new [`Group`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `blueprint` - group creation blueprint
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::GroupRequest};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build group request
    /// let req = GroupRequest::new("CornGroup");
    /// // create our group
    /// thorium.groups.create(&req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn create(&self, blueprint: &GroupRequest) -> Result<reqwest::Response, Error> {
        // build url for creating  a group
        let url = format!("{}/api/groups/", self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(&blueprint);
        // send this request
        send!(self.client, req)
    }

    /// Gets details about a [`Group`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The name of the group to get details about
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
    /// // get our groups data
    /// let group = thorium.groups.get("CornGroup").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get(&self, group: &str) -> Result<Group, Error> {
        // build url for listing groups
        let url = format!("{}/api/groups/{}/details", self.host, group);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a group from the response
        send_build!(self.client, req, Group)
    }

    /// Lists all groups in Thorium
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor denoting what page of groups to list
    /// * `limit` - The weakly enforced limit on groups to return
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
    /// // list the names of the groups we are in
    /// let groups = thorium.groups.list().exec().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub fn list(&self) -> Cursor<Group> {
        // build url for listing groups
        let url = format!("{}/api/groups/", self.host);
        Cursor::new(url, &self.token, &self.client).limit(500)
    }

    /// Deletes a [`Group`] from Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - name of the group to delete
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
    /// // deletes a group from Thorium
    /// thorium.groups.delete("NotCornGroup").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete(&self, group: &str) -> Result<reqwest::Response, Error> {
        // build url for deleting a group
        let url = format!("{}/api/groups/{}", self.host, group);
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Updates a [`Group`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The name of the group to update
    /// * `update` - The update to apply to this group
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{GroupUpdate, GroupUsersUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a group update to add a new user
    /// let update = GroupUpdate::default()
    ///     .users(GroupUsersUpdate::default().direct_add("bob"));
    /// // update a group
    /// thorium.groups.update("CornGroup", &update).await?;
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
        update: &GroupUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for listing groups
        let url = format!("{}/api/groups/{}", self.host, group);
        // build request
        let req = self
            .client
            .patch(&url)
            .json(update)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Sync all [`Group`] data with LDAP
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
    /// // sync all group data in Thorium
    /// thorium.groups.sync_ldap().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn sync_ldap(&self) -> Result<reqwest::Response, Error> {
        // build url for deleting a group
        let url = format!("{}/api/groups/sync/ldap", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }
}
