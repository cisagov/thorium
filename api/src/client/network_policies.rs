//! Client handler for network policies routes in Thorium

use uuid::Uuid;

use super::Error;
use crate::models::{
    Cursor, NetworkPolicy, NetworkPolicyListLine, NetworkPolicyListOpts, NetworkPolicyRequest,
    NetworkPolicyUpdate,
};
use crate::{add_query, add_query_list, send, send_build};

#[cfg(feature = "trace")]
use tracing::instrument;

// import our static runtime if we need a blocking client
#[cfg(feature = "sync")]
use super::RUNTIME;

/// A handler for the network policies routes in Thorium
#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
#[derive(Clone)]
pub struct NetworkPolicies {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
}

/// Create a new list cursor; helpful because the list and `list_details` routes
/// will likely always be identical except for their URL's
macro_rules! list_cursor {
    ($token:expr, $client:expr, $url:expr, $opts:expr) => {
        async {
            // get the correct page size if our limit is smaller then our page_size
            let page_size = $opts.limit.map_or_else(
                || $opts.page_size,
                |limit| std::cmp::min($opts.page_size, limit),
            );
            // build our query params
            let mut query = vec![("limit", page_size.to_string())];
            add_query_list!(query, "groups[]", &$opts.groups);
            add_query!(query, "cursor", &$opts.cursor);
            // get the data for this request and create our cursor
            Cursor::new($url, $opts.page_size, $opts.limit, $token, &query, $client).await
        }
    };
}

#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
impl NetworkPolicies {
    /// Creates a new network policies handler
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
    /// use thorium::client::NetworkPolicies;
    ///
    /// let client = reqwest::Client::new();
    /// let network_policies = NetworkPolicies::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build network policies route handler
        NetworkPolicies {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }

    /// Creates a [`NetworkPolicy`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `req` - The request to create a network policy
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    /// use thorium::models::{NetworkPolicyRequest, NetworkProtocol, NetworkPolicyRuleRaw};
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // create a policy request that allows in TCP on port 80 for IP's 10.0.0.0/8
    /// // except for 10.10.0.0/16;
    /// // the policy is called "allow-in-http" and is in the "corn" and "test" groups
    /// let req = NetworkPolicyRequest::new("allow-in-http", ["corn", "test"])
    ///     .add_ingress_rule(NetworkPolicyRuleRaw::default()
    ///         .ip_block(
    ///             "10.0.0.0/8",
    ///             Some(vec!["10.10.0.0/16"]),
    ///         )
    ///         .port(80, None, Some(NetworkProtocol::TCP))
    ///     );
    /// // create the network policy
    /// thorium.network_policies.create(req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::NetworkPolicies::create", skip_all, err(Debug))
    )]
    pub async fn create(&self, req: NetworkPolicyRequest) -> Result<reqwest::Response, Error> {
        // build url for creating a network policy
        let url = format!("{base}/api/network-policies", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(&req);
        // send this request
        send!(self.client, req)
    }

    /// Gets a [`NetworkPolicy`] from Thorium
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the network policy to delete
    /// * `id` - The network policy's unique ID, required if more than one distinct
    ///          network policies exist with the same name
    ///
    /// # Examples
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get a network policy called "default"
    /// let network_policy = thorium.network_policies.get("default", None).await?;
    /// // symbols/other languages for network policy names are also supported
    /// let network_policy2 = thorium.network_policies.get("cool😎policy", None).await?;
    /// // specify an ID if there are more than one distinct network policies with the same name
    /// let id = Uuid::from_str("7b27590a-dc19-4e4a-8898-cbe81a7cbb7b").unwrap();
    /// let network_policy3 = thorium.network_policies.get("not-unique", Some(id)).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::NetworkPolicies::get", skip_all, err(Debug))
    )]
    pub async fn get(&self, name: &str, id: Option<Uuid>) -> Result<NetworkPolicy, Error> {
        // build url for getting a network policy
        let url = format!("{base}/api/network-policies/{name}", base = self.host);
        // create a query with our ID in case we supplied one
        let mut query = vec![];
        add_query!(query, "id", id);
        // build request
        let req = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send this request
        send_build!(self.client, req, NetworkPolicy)
    }

    /// Updates a [`NetworkPolicy`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the network policy to delete
    /// * `id` - The network policy's unique ID, required if more than one distinct
    ///          network policies exist with the same name
    /// * `update` - The update to apply to the network policy
    ///
    /// # Examples
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// use thorium::Thorium;
    /// use thorium::models::NetworkPolicyUpdate;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // create an update to change a network policy's name to "new-name" and add it
    /// // to the "corn" group
    /// let update = NetworkPolicyUpdate::default().new_name("new-name").add_group("corn");
    /// // apply the update to a network policy called "default"
    /// thorium.network_policies.update("default", None, &update).await?;
    /// // symbols/other languages for network policy names are also supported
    /// thorium.network_policies.update("cool😎policy", None, &update).await?;
    /// // specify an ID if there are more than one distinct network policies with the same name
    /// let id = Uuid::from_str("7b27590a-dc19-4e4a-8898-cbe81a7cbb7b").unwrap();
    /// thorium.network_policies.update("not-unique", Some(id), &update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::NetworkPolicies::update", skip_all, err(Debug))
    )]
    pub async fn update(
        &self,
        name: &str,
        id: Option<Uuid>,
        update: &NetworkPolicyUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for getting a network policy
        let url = format!("{base}/api/network-policies/{name}", base = self.host);
        // create a query with our ID in case we supplied one
        let mut query = vec![];
        add_query!(query, "id", id);
        // build request
        let req = self
            .client
            .patch(&url)
            .header("authorization", &self.token)
            .query(&query)
            .json(&update);
        // send this request
        send!(self.client, req)
    }

    /// Deletes a [`NetworkPolicy`] from Thorium
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the network policy to delete
    /// * `id` - The network policy's unique ID, required if more than one distinct
    ///          network policies exist with the same name
    ///
    /// # Examples
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// use thorium::Thorium;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // delete a network policy called "default"
    /// thorium.network_policies.delete("default", None).await?;
    /// // symbols/other languages for network policy names are also supported
    /// thorium.network_policies.delete("cool😎policy", None).await?;
    /// // specify an ID if there are more than one distinct network policies with the same name
    /// let id = Uuid::from_str("7b27590a-dc19-4e4a-8898-cbe81a7cbb7b").unwrap();
    /// thorium.network_policies.delete("not-unique", Some(id)).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::NetworkPolicies::delete", skip_all, err(Debug))
    )]
    pub async fn delete(&self, name: &str, id: Option<Uuid>) -> Result<reqwest::Response, Error> {
        // build url for getting a network policy
        let url = format!("{base}/api/network-policies/{name}", base = self.host);
        // create a query with our ID in case we supplied one
        let mut query = vec![];
        add_query!(query, "id", id);
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send this request
        send!(self.client, req)
    }

    /// Gets the names/id's of all default network policies in a given group from Thorium
    ///
    /// Default network policies are automatically added to new images when no network policies
    /// are supplied
    ///
    /// # Arguments
    ///
    /// * `group` - The group to get default network policies for
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
    /// // get all default network policies from the group "corn"
    /// let default_network_policies = thorium.network_policies.get_all_default("corn").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::NetworkPolicies::get_all_default",
            skip_all,
            err(Debug)
        )
    )]
    pub async fn get_all_default<T: AsRef<str>>(
        &self,
        group: T,
    ) -> Result<Vec<NetworkPolicyListLine>, Error> {
        // build url for getting default network policies
        let url = format!(
            "{base}/api/network-policies/default/{group}/",
            base = self.host,
            group = group.as_ref()
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, Vec<NetworkPolicyListLine>)
    }

    /// Lists all network policies that meet some search criteria
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for this cursor
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, SearchDate};
    /// use thorium::models::NetworkPolicyListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a search to list network policies in the "corn" and "rust" groups
    /// let opts = NetworkPolicyListOpts::default()
    ///     .groups(vec!["corn", "rust"])
    ///     // limit it to 100 network policies
    ///     .limit(100);
    /// let cursor = thorium.network_policies.list(&opts).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::NetworkPolicies::list", skip_all, err(Debug))
    )]
    pub async fn list(
        &self,
        opts: &NetworkPolicyListOpts,
    ) -> Result<Cursor<NetworkPolicyListLine>, Error> {
        // build the url for listing files
        let url = format!("{}/api/network-policies/", self.host);
        // create the list cursor
        list_cursor!(&self.token, &self.client, &url, opts).await
    }

    /// Lists all network policies that meet some search criteria, including all their details
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for this cursor
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, SearchDate};
    /// use thorium::models::NetworkPolicyListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a search to list network policies in the "corn" and "rust" groups,
    /// // including all their details
    /// let opts = NetworkPolicyListOpts::default()
    ///     .groups(vec!["corn", "rust"])
    ///     // limit it to 100 network policies
    ///     .limit(100);
    /// let cursor = thorium.network_policies.list_details(&opts).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::NetworkPolicies::list_details", skip_all, err(Debug))
    )]
    pub async fn list_details(
        &self,
        opts: &NetworkPolicyListOpts,
    ) -> Result<Cursor<NetworkPolicy>, Error> {
        // build the url for listing files
        let url = format!("{}/api/network-policies/details/", self.host);
        // create the list cursor
        list_cursor!(&self.token, &self.client, &url, opts).await
    }
}
