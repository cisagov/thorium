use base64::Engine as _;

use super::{helpers, ClientSettings, Error};
use crate::models::{AuthResponse, ScrubbedUser, UserCreate, UserUpdate};
use crate::{send, send_build};

/// users handler for the Thorium client
#[derive(Clone)]
pub struct Users {
    /// url/ip of the Thorium ip
    host: String,
    /// token to use for auth
    token: String,
    /// reqwest client object
    client: reqwest::Client,
}

impl Users {
    /// Creates a new users handler
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
    /// use thorium::client::Users;
    ///
    /// let client = reqwest::Client::new();
    /// let users = Users::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Users {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// users handler for the Thorium client
        #[derive(Clone)]
        pub struct UsersBlocking {
            /// url/ip of the Thorium ip
            host: String,
            /// reqwest client object
            client: reqwest::Client,
            /// token to use for auth
            token: String,
        }

        impl UsersBlocking {
            /// creates a new blocking users handler
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
            /// use thorium::client::UsersBlocking;
            ///
            /// let users = UsersBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                UsersBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Users {
    /// Creates a [`User`] in Thorium
    ///
    /// When adding an admin you must pass in the secret key.
    ///
    /// # Arguments
    ///
    /// * `host` - The host (starting with http:// or https://) to reach Thorium at
    /// * `blueprint` - User creation blueprint
    /// * `key` - The secret key to use in order to add an admin
    /// * `settings` - The settings for this client
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::UserCreate;
    /// use thorium::client::{ClientSettings, Users};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // build a user create request
    /// let req = UserCreate::new("mcarson", "guest", "email@email.com");
    /// let settings = ClientSettings::default();
    /// // we don't have a secret key
    /// let secret: Option<String> = None;
    /// // create a user in Thorium
    /// Users::create("http://127.0.0.1", req, secret, &settings).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn create<K: AsRef<str>>(
        host: &str,
        blueprint: UserCreate,
        key: Option<K>,
        settings: &ClientSettings,
    ) -> Result<AuthResponse, Error> {
        // build url for creating a user
        let url = format!("{host}/api/users/");
        // get client
        let client = helpers::build_reqwest_client(settings).await?;
        // build request
        let mut req = client.post(&url).json(&blueprint);
        // inject key header if it exists
        if let Some(key) = key {
            req = req.header("secret-key", key.as_ref());
        }
        // send request
        send_build!(client, req, AuthResponse)
    }

    /// Updates a [`User`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `host` - The host (starting with http:// or https://) to reach Thorium at
    /// * `username` - Username of user to update
    /// * `update` - User update info
    /// * `settings` - The settings for this client
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{UserUpdate, UserRole};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // build a user update request
    /// let update = UserUpdate {
    ///     password: Some("password".to_owned()),
    ///     email: Some("email@email.com".to_owned()),
    ///     role: Some(UserRole::Admin),
    ///     settings: None,
    /// };
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // create a user in Thorium
    /// thorium.users.update("username", update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update(
        &self,
        username: &str,
        update: UserUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a user
        let url = format!("{}/api/users/user/{}", self.host, username);
        // build request
        let req = self
            .client
            .patch(&url)
            .json(&update)
            .header("authorization", &self.token);
        // send request
        send!(self.client, req)
    }

    /// Authenticates to Thorium using a username/password
    ///
    /// Instead of calling this you likely want to use the methods exposed by
    /// [`thorium::Thorium`].
    ///
    /// # Arguments
    ///
    /// * `host` - The host (starting with http:// or https://) to reach Thorium at
    /// * `username` - The user that is authenticating
    /// * `password` - The password to authenticate with
    /// * `client` - The client to authenticate with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::Users;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// let client = reqwest::Client::new();
    /// // authenticate to Thorium
    /// let auth_resp = Users::auth_basic("http://127.0.0.1", "mcarson", "secretCorn", &client).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn auth_basic(
        host: &str,
        username: &str,
        password: &str,
        client: &reqwest::Client,
    ) -> Result<AuthResponse, Error> {
        // build url for listing groups
        let url = format!("{host}/api/users/auth");
        // build basic auth object
        let joint = format!("{username}:{password}");
        // base64 encode creds
        let encoded = base64::engine::general_purpose::STANDARD.encode(joint.as_bytes());
        // build basic auth string
        let auth = format!("basic {encoded}");
        // build request
        let req = client.post(&url).header("Authorization", auth);
        // send request and build a reaction
        send_build!(client, req, AuthResponse)
    }

    /// Gets info on a specific [`User`]
    ///
    /// # Arguments
    ///
    /// * `username` - The user to get info on
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
    /// // get info on a user
    /// let user = thorium.users.get("mcarson").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get(&self, username: &str) -> Result<ScrubbedUser, Error> {
        // build url for getting a users data
        let url = format!("{}/api/users/user/{}", self.host, username);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request and build a reaction
        send_build!(self.client, req, ScrubbedUser)
    }

    /// Lists usernames in Thorium
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
    /// // get a list of users in Thorium
    /// let users = thorium.users.list().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn list(&self) -> Result<Vec<String>, Error> {
        // build url for getting a users data
        let url = format!("{}/api/users/", self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request and build a reaction
        send_build!(self.client, req, Vec<String>)
    }

    /// Gets info on all [`User`]s in Thorium
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
    /// // get info on all users
    /// let users = thorium.users.list().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn list_details(&self) -> Result<Vec<ScrubbedUser>, Error> {
        // build url for getting a users data
        let url = format!("{}/api/users/details/", self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request and build a reaction
        send_build!(self.client, req, Vec<ScrubbedUser>)
    }

    /// Gets info on our current user
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
    /// // get info on our current user
    /// let user = thorium.users.info().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn info(&self) -> Result<ScrubbedUser, Error> {
        // build url for getting a users data
        let url = format!("{}/api/users/whoami", self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send request and build a reaction
        send_build!(self.client, req, ScrubbedUser)
    }

    /// Log the current [`User`] out of Thorium invalidating their token
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
    /// // log ourselves out of Thorium
    /// thorium.users.logout().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn logout(&self) -> Result<reqwest::Response, reqwest::Error> {
        // build url for logging a user out
        let url = format!("{}/api/users/logout", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send request
        self.client.execute(req.build()?).await?.error_for_status()
    }

    /// Log a different [`User`] out of Thorium invalidating their token
    ///
    /// Only admins are allowed to log other users out.
    ///
    /// # Arguments
    ///
    /// * `user` - The user to logout of Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // log a different user out of Thorium
    /// thorium.users.logout_user("gachael").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn logout_user(&self, user: &str) -> Result<reqwest::Response, Error> {
        // build url for logging a user out
        let url = format!("{}/api/users/logout/{}", self.host, user);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send request
        send!(self.client, req)
    }

    /// Delete a [`User`] in Thorium
    ///
    /// Only admins are allowed to delete other users.
    ///
    /// # Arguments
    ///
    /// * `user` - The user to logout of Thorium
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
    /// // delete a different in Thorium
    /// thorium.users.delete("gachael").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete(&self, user: &str) -> Result<reqwest::Response, Error> {
        // build url for logging a user out
        let url = format!("{}/api/users/delete/{}", self.host, user);
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token);
        // send request
        send!(self.client, req)
    }
}
