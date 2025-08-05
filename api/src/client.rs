//! Asynchronous and synchronous clients for Thorium

use base64::Engine as _;
use chrono::prelude::*;
use std::path::{Path, PathBuf};

use crate::models::{self, AuthResponse, ScrubbedUser};

mod basic;
pub mod conf;
mod cursors;
mod error;
mod events;
mod files;
mod groups;
mod helpers;
mod images;
mod jobs;
mod keys;
mod network_policies;
mod pipelines;
mod reactions;
mod repos;
mod search;
mod streams;
mod system;
mod traits;
mod updates;
mod users;
mod utils;

pub use basic::Basic;
pub use conf::{ClientSettings, CtlConf};
pub use cursors::{Cursor, LogsCursor, SearchDate};
pub use error::Error;
pub use events::Events;
pub use files::Files;
pub use groups::Groups;
pub use images::Images;
pub use jobs::Jobs;
pub use keys::Keys;
pub use network_policies::NetworkPolicies;
pub use pipelines::Pipelines;
pub use reactions::Reactions;
pub use repos::Repos;
pub use search::events::results::ResultSearchEvents;
pub use search::events::tags::TagSearchEvents;
pub use search::events::{SearchEvents, SearchEventsClient};
pub use search::Search;
pub use streams::Streams;
pub use system::System;
pub use traits::ResultsClient;
pub use updates::Updates;
pub use users::Users;

// if the blocking client is enabled then also expose those
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        pub use basic::BasicBlocking;
        pub use files::FilesBlocking;
        pub use groups::GroupsBlocking;
        pub use images::ImagesBlocking;
        pub use jobs::JobsBlocking;
        pub use pipelines::PipelinesBlocking;
        pub use reactions::ReactionsBlocking;
        pub use repos::ReposBlocking;
        pub use search::SearchBlocking;
        pub use streams::StreamsBlocking;
        pub use system::SystemBlocking;
        pub use users::UsersBlocking;
        pub use events::EventsBlocking;
    }
}

/// Builds the Thorium client
#[derive(Debug, Clone)]
pub struct ThoriumClientBuilder {
    /// The host/domain the Thorium api can be found at
    host: String,
    /// The username to login with
    username: Option<String>,
    /// The password to login with
    password: Option<String>,
    /// A token to use instead of a username/password combo
    token: Option<String>,
    /// The settings for thorctls client
    pub settings: ClientSettings,
}

impl ThoriumClientBuilder {
    /// Sets a basic auth username/password combo to use for authentication
    ///
    /// # Arguments
    ///
    /// * `username` - The username to authenticate with
    /// * `password` - The password to authenticate with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    ///
    /// Thorium::build("http://127.0.0.1")
    ///     .basic_auth("user", "password");
    /// ```
    #[must_use]
    pub fn basic_auth<T: Into<String>>(mut self, username: T, password: T) -> Self {
        // set a username/password combo for basic auth
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }

    /// Sets a token to use for authentication instead of username/password
    ///
    /// # Arguments
    ///
    /// * `token` - The token to use to authenticate
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    ///
    /// Thorium::build("http://127.0.0.1").token("token");
    /// ```
    #[must_use]
    pub fn token<T: Into<String>>(mut self, token: T) -> Self {
        // Set a token for token based auth
        self.token = Some(token.into());
        self
    }

    /// Allow insecure invalid certificates to be trusted
    #[must_use]
    pub fn danger_accept_invalid_certs(mut self) -> Self {
        self.settings.invalid_certs = true;
        self
    }

    /// Allow insecure invalid hostnames to be trusted
    #[must_use]
    pub fn danger_accept_invalid_hostnames(mut self) -> Self {
        self.settings.invalid_hostnames = true;
        self
    }

    /// Adds a custom CA to be trusted
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the certificate authority to trust
    #[must_use]
    pub fn add_certificate_authority<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.settings.certificate_authorities.push(path.into());
        self
    }

    /// Load auth info from a key file on disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load auth info from
    pub fn from_keys(self, path: &str) -> Result<Self, Error> {
        // load auth keys
        let keys = Keys::new(path)?;
        // use the correct auth method based on what is defined in the config
        match (keys.username, keys.password, keys.token) {
            (Some(user), Some(pass), _) => Ok(self.basic_auth(user, pass)),
            (_, None, Some(token)) => Ok(self.token(token)),
            (_, _, _) => Err(Error::new("Either username/password or token must be set")),
        }
    }

    /// Load auth info and settings from a [`CtlConf`]
    ///
    /// # Arguments
    ///
    /// * `ctl_conf` - The `CtlConf` to build from
    pub fn from_ctl_conf(mut self, ctl_conf: CtlConf) -> Result<Self, Error> {
        // get our keys object from our ctl conf
        let keys = ctl_conf.keys;
        // save our settings from this ctl conf
        self.settings = ctl_conf.client;
        // use the correct auth method based on what is defined in the config
        match (keys.username, keys.password, keys.token) {
            (Some(user), Some(pass), _) => Ok(self.basic_auth(user, pass)),
            (_, None, Some(token)) => Ok(self.token(token)),
            (_, _, _) => Err(Error::new("Either username/password or token must be set")),
        }
    }

    /// Builds a client with the configured auth settings
    ///
    /// This will panic if username/password or auth are not set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// let thorium = Thorium::build("http://127.0.0.1")
    ///     .basic_auth("user", "password")
    ///     .build()
    ///     .await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn build(self) -> Result<Thorium, Error> {
        // build a client
        let client = helpers::build_reqwest_client(&self.settings).await?;
        // get token if we have a username/password and no token
        let (token, expires) = match self.token {
            Some(token) => (token, None),
            None => Thorium::auth(&self.host, self.username, self.password, &client).await?,
        };
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        // build token auth string
        let auth_str = format!("token {encoded}");
        // build handlers
        let basic = Basic::new(&self.host, &client);
        let jobs = Jobs::new(&self.host, &auth_str, &client);
        let reactions = Reactions::new(&self.host, &auth_str, &client);
        let pipelines = Pipelines::new(&self.host, &auth_str, &client);
        let groups = Groups::new(&self.host, &auth_str, &client);
        let images = Images::new(&self.host, &auth_str, &client);
        let streams = Streams::new(&self.host, &auth_str, &client);
        let users = Users::new(&self.host, &auth_str, &client);
        let system = System::new(&self.host, &auth_str, &client);
        let search = Search::new(&self.host, &auth_str, &client);
        let files = Files::new(&self.host, &auth_str, &client);
        let repos = Repos::new(&self.host, &auth_str, &client);
        let updates = Updates::new(&self.host, &auth_str, &client);
        let events = Events::new(&self.host, &auth_str, &client);
        let network_policies = NetworkPolicies::new(&self.host, &auth_str, &client);
        // build Thorium client
        let client = Thorium {
            basic,
            jobs,
            reactions,
            pipelines,
            groups,
            images,
            streams,
            users,
            system,
            search,
            files,
            repos,
            events,
            network_policies,
            host: self.host,
            auth_str,
            expires,
            updates,
            client,
        };
        Ok(client)
    }

    /// Builds a client with the configured auth settings
    ///
    /// This will panic if username/password or auth are not set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # fn exec() -> Result<(), Error> {
    /// let thorium = Thorium::build("http://127.0.0.1")
    ///     .basic_auth("user", "password")
    ///     .build_blocking()?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # exec();
    /// ```
    #[cfg(feature = "sync")]
    pub fn build_blocking(self) -> Result<ThoriumBlocking, Error> {
        // build a client
        let client = helpers::build_blocking_reqwest_client(&self.settings).await?;
        // get token if we have a username/password and no token
        let (token, expires) = match self.token {
            Some(token) => (token, None),
            None => ThoriumBlocking::auth(&self.host, self.username, self.password, &client)?,
        };
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        // build token auth string
        let auth_str = format!("token {encoded}");
        // build a client
        let client = helpers::build_blocking_reqwest_client(&self.settings)?;
        // build handlers
        let basic = BasicBlocking::new(&self.host, &client);
        let jobs = JobsBlocking::new(&self.host, &auth_str, &client);
        let reactions = ReactionsBlocking::new(&self.host, &auth_str, &client);
        let pipelines = PipelinesBlocking::new(&self.host, &auth_str, &client);
        let groups = GroupsBlocking::new(&self.host, &auth_str, &client);
        let images = ImagesBlocking::new(&self.host, &auth_str, &client);
        let streams = StreamsBlocking::new(&self.host, &auth_str, &client);
        let users = UsersBlocking::new(&self.host, &auth_str, &client);
        let system = SystemBlocking::new(&self.host, &auth_str, &client);
        let search = SearchBlocking::new(&self.host, &auth_str, &client);
        let files = FilesBlocking::new(&self.host, &auth_str, &client);
        let repos = ReposBlocking::new(&self.host, &auth_str, &client);
        let events = EventsBlocking::new(&self.host, &auth_str);
        let network_policies = NetworkPoliciesBlocking::new(&self.host, &auth_str);
        // build Thorium client
        let client = ThoriumBlocking {
            basic,
            jobs,
            reactions,
            pipelines,
            groups,
            images,
            streams,
            users,
            system,
            search,
            files,
            repos,
            events,
            network_policies,
            host: self.host,
            auth_str,
            expires,
            client,
        };
        Ok(client)
    }
}

/// An asynchronous client for Thorium
#[derive(Clone)]
pub struct Thorium {
    /// Handles basic routes in Thorium
    pub basic: Basic,
    /// Handles jobs routes in Thorium
    pub jobs: Jobs,
    /// Handles reactions routes in Thorium
    pub reactions: Reactions,
    /// Handles pipelines routes in Thorium
    pub pipelines: Pipelines,
    /// Handles groups routes in Thorium
    pub groups: Groups,
    /// Handles images routes in Thorium
    pub images: Images,
    /// Handles streams routes in Thorium
    pub streams: Streams,
    /// Handles users routes in Thorium
    pub users: Users,
    /// Handles system routes in Thorium
    pub system: System,
    /// Handles search routes in Thorium
    pub search: Search,
    /// Handles files routes in Thorium
    pub files: Files,
    /// Handles repos routes in Thorium
    pub repos: Repos,
    /// Handles binary update routes in Thorium
    pub updates: Updates,
    /// Handles event routes in Thorium
    pub events: Events,
    /// Handles network policies routes in Thorium
    pub network_policies: NetworkPolicies,
    /// The host/url to reach Thorium at
    pub host: String,
    /// The auth str to use when reverting from a masquerade
    auth_str: String,
    /// When our token expires if we have a token
    pub expires: Option<DateTime<Utc>>,
    // keep a copy of our client for faster masquerades and refreshes
    client: reqwest::Client,
}

/// An blocking client for Thorium
#[cfg(feature = "sync")]
#[derive(Clone)]
pub struct ThoriumBlocking {
    /// Handles basic routes in Thorium
    pub basic: BasicBlocking,
    /// Handles jobs routes in Thorium
    pub jobs: JobsBlocking,
    /// Handles reactions routes in Thorium
    pub reactions: ReactionsBlocking,
    /// Handles pipelines routes in Thorium
    pub pipelines: PipelinesBlocking,
    /// Handles groups routes in Thorium
    pub groups: GroupsBlocking,
    /// Handles images routes in Thorium
    pub images: ImagesBlocking,
    /// Handles streams routes in Thorium
    pub streams: StreamsBlocking,
    /// Handles users routes in Thorium
    pub users: UsersBlocking,
    /// Handles system routes in Thorium
    pub system: SystemBlocking,
    /// Handles search routes in Thorium
    pub search: SearchBlocking,
    /// Handles files routes in Thorium
    pub files: FilesBlocking,
    /// Handles repos routes in Thorium
    pub repos: ReposBlocking,
    /// Handles event routes in Thorium
    pub events: EventsBlocking,
    /// Handles network policies routes in Thorium
    pub network_policies: NetworkPoliciesBlocking,
    /// The host/url to reach Thorium at
    pub host: String,
    /// The auth str to use when reverting from a masquerade
    auth_str: String,
    /// When our token expires if we have a token
    pub expires: Option<DateTime<Utc>>,
    // keep a copy of our client for faster masquerades and refreshes
    client: reqwest::Client,
}

#[syncwrap::clone_impl]
impl Thorium {
    /// Create a new Thorium client builder
    ///
    /// This can use either username/password or token. When using a token, the client will not
    /// known when it expires.
    ///
    /// # Arguments
    ///
    /// * `host` - The host/url/ip the Thorium API can be reached at
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// let thorium = Thorium::build("http://127.0.0.1")
    ///     .basic_auth("user", "password")
    ///     .build()
    ///     .await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub fn build<T: Into<String>>(host: T) -> ThoriumClientBuilder {
        ThoriumClientBuilder {
            host: host.into(),
            username: None,
            password: None,
            token: None,
            settings: ClientSettings::default(),
        }
    }

    /// Authenticate using a username/password to Thorium
    ///
    /// # Arguments
    ///
    /// * `host` - The host/url/ip the Thorium API can be reached at
    /// * `username` - The username of the user to login as
    /// * `password` - The password to authenticate with
    /// * `client` - The client to authenticate with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// let client = reqwest::Client::new();
    /// let (token, expiration) = Thorium::auth("http://127.0.0.1",
    ///     Some("user".into()),
    ///     Some("pass".into()),
    ///     &client)
    ///     .await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn auth(
        host: &str,
        username: Option<String>,
        password: Option<String>,
        client: &reqwest::Client,
    ) -> Result<(String, Option<DateTime<Utc>>), Error> {
        // make sure both username and password are specified
        if username.is_none() || password.is_none() {
            panic!("Both username and password must be specified if token is not");
        }

        // create auth handler and get token
        let resp = Users::auth_basic(host, &username.unwrap(), &password.unwrap(), client).await?;
        Ok((resp.token, Some(resp.expires)))
    }

    /// Adds a new admin to Thorium using the secret key
    ///
    /// This is primarily used for bootstrapping a new Thorium install.
    ///
    /// # Arguments
    ///
    /// * `host` - The host/url/ip the Thorium API can be reached at
    /// * `username` - The username of the user to create
    /// * `password` - The password to set for the new user
    /// * `key` - The secret key used to authenticate to the Thorium API
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::ClientSettings;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// let settings = ClientSettings::default();
    /// let mut thorium = Thorium::bootstrap("http://127.0.0.1", "user", "password", "email@email.com", "secret", &settings).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn bootstrap<T: Into<String>>(
        host: &str,
        username: T,
        password: T,
        email: T,
        key: &str,
        settings: &ClientSettings,
    ) -> Result<AuthResponse, Error> {
        // build an admin user create request
        let bp = models::UserCreate::new(username.into(), password.into(), email.into())
            .admin()
            .skip_verification();
        Users::create(host, bp, Some(key), settings).await
    }
}

impl Thorium {
    /// Create a Thorium client from a path on disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load out Key from
    pub async fn from_key_file(path: &str) -> Result<Self, Error> {
        // load auth keys
        let keys = Keys::new(path)?;
        // build a Thorium client from keys
        Self::from_keys(keys).await
    }

    /// Create a `Thorium` client from a keys struct
    ///
    /// # Arguments
    ///
    /// * `keys` - The keys to create a client with
    pub async fn from_keys(keys: Keys) -> Result<Self, Error> {
        // create a Thorium client builder
        let builder = Self::build(keys.api);
        // use the correct auth method based on what is defined in the config
        let builder = match (keys.username, keys.password, keys.token) {
            (Some(user), Some(pass), None) => builder.basic_auth(user, pass),
            (_, _, Some(token)) => builder.token(token),
            (_, _, _) => return Err(Error::new("Either username/password or token must be set")),
        };
        // build a new Thorium client
        builder.build().await
    }

    /// Create a Thorium client from a [`CtlConf`] serialized to a file at the given path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load our `CtlConf` from
    pub async fn from_ctl_conf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        // load ctl conf
        let ctl_conf = CtlConf::from_path(&path)?;
        // build a client from the ctl conf
        Self::from_ctl_conf(ctl_conf).await
    }

    /// Attempt to create a `Thorium` client from the given [`CtlConf`]
    ///
    /// # Arguments
    ///
    /// * `conf` - The `CtlConf` to create a `Thorium` client from
    pub async fn from_ctl_conf(ctl_conf: CtlConf) -> Result<Self, Error> {
        // create a builder from the ctl conf and build
        Self::build(ctl_conf.keys.api.clone())
            .from_ctl_conf(ctl_conf)?
            .build()
            .await
    }

    /// Refresh this clients token
    ///
    /// This will invalidate the client current token and create a new one causing any other
    /// clients using this token to fail.
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user to login as
    /// * `password` - The password to authenticate with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // get Thorium client
    /// let mut thorium = Thorium::build("http://127.0.0.1")
    ///     .token("token")
    ///     .build()
    ///     .await?;
    /// // refresh our token
    /// thorium.refresh("user", "password").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn refresh<T: Into<String>>(
        &mut self,
        username: T,
        password: T,
    ) -> Result<(), Error> {
        // logout and invalidate our token
        self.users.logout().await?;
        // authenticate and get new token
        let (token, expiration) = Self::auth(
            &self.host,
            Some(username.into()),
            Some(password.into()),
            &self.client,
        )
        .await?;
        // update token expiration
        self.expires = expiration;
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        // build token auth string
        let auth_str = format!("token {encoded}");
        // update handlers
        self.basic = Basic::new(&self.host, &self.client);
        self.jobs = Jobs::new(&self.host, &auth_str, &self.client);
        self.reactions = Reactions::new(&self.host, &auth_str, &self.client);
        self.pipelines = Pipelines::new(&self.host, &auth_str, &self.client);
        self.groups = Groups::new(&self.host, &auth_str, &self.client);
        self.images = Images::new(&self.host, &auth_str, &self.client);
        self.streams = Streams::new(&self.host, &auth_str, &self.client);
        self.users = Users::new(&self.host, &auth_str, &self.client);
        self.system = System::new(&self.host, &auth_str, &self.client);
        self.files = Files::new(&self.host, &auth_str, &self.client);
        self.repos = Repos::new(&self.host, &auth_str, &self.client);
        self.events = Events::new(&self.host, &auth_str, &self.client);
        Ok(())
    }

    /// Masquerade as a another user
    ///
    /// # Arguments
    ///
    /// * `user` - The user to masquerade as
    pub fn masquerade(&mut self, user: &ScrubbedUser) {
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(user.token.as_bytes());
        // build token auth string
        let auth_str = format!("token {encoded}");
        // update handlers
        self.basic = Basic::new(&self.host, &self.client);
        self.jobs = Jobs::new(&self.host, &auth_str, &self.client);
        self.reactions = Reactions::new(&self.host, &auth_str, &self.client);
        self.pipelines = Pipelines::new(&self.host, &auth_str, &self.client);
        self.groups = Groups::new(&self.host, &auth_str, &self.client);
        self.images = Images::new(&self.host, &auth_str, &self.client);
        self.streams = Streams::new(&self.host, &auth_str, &self.client);
        self.users = Users::new(&self.host, &auth_str, &self.client);
        self.system = System::new(&self.host, &auth_str, &self.client);
        self.files = Files::new(&self.host, &auth_str, &self.client);
        self.repos = Repos::new(&self.host, &auth_str, &self.client);
        self.events = Events::new(&self.host, &auth_str, &self.client);
    }

    /// Revert back to our original user from a masquerade
    pub fn revert_masquerade(&mut self) {
        // update handlers
        self.basic = Basic::new(&self.host, &self.client);
        self.jobs = Jobs::new(&self.host, &self.auth_str, &self.client);
        self.reactions = Reactions::new(&self.host, &self.auth_str, &self.client);
        self.pipelines = Pipelines::new(&self.host, &self.auth_str, &self.client);
        self.groups = Groups::new(&self.host, &self.auth_str, &self.client);
        self.images = Images::new(&self.host, &self.auth_str, &self.client);
        self.streams = Streams::new(&self.host, &self.auth_str, &self.client);
        self.users = Users::new(&self.host, &self.auth_str, &self.client);
        self.system = System::new(&self.host, &self.auth_str, &self.client);
        self.files = Files::new(&self.host, &self.auth_str, &self.client);
        self.repos = Repos::new(&self.host, &self.auth_str, &self.client);
        self.events = Events::new(&self.host, &self.auth_str, &self.client);
    }
}

#[cfg(feature = "sync")]
impl ThoriumBlocking {
    /// Create a Thorium client from a path on disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load auth info from
    pub fn from_file(path: &str) -> Result<ThoriumBlocking, Error> {
        // load auth keys
        let keys = Keys::new(path)?;
        // create a Thorium client builder
        let builder = ThoriumBlocking::build(keys.api);
        // use the correct auth method based on what is defined in the config
        let builder = match (keys.username, keys.password, keys.token) {
            (Some(user), Some(pass), None) => builder.basic_auth(user, pass),
            (_, _, Some(token)) => builder.token(token),
            (_, _, _) => return Err(Error::new("Either username/password or token must be set")),
        };
        // build a new Thorium client
        builder.build_blocking()
    }

    /// Refresh this clients token
    ///
    /// This will invalidate the client current token and create a new one causing any other
    /// clients using this token to fail.
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user to login as
    /// * `password` - The password to authenticate with
    /// # Examples
    ///
    /// ```
    /// use thorium::ThoriumBlocking;
    /// # use thorium::Error;
    ///
    /// # fn exec() -> Result<(), Error> {
    /// // get Thorium client
    /// let mut thorium = ThoriumBlocking::build("http://127.0.0.1")
    ///     .token("token")
    ///     .build_blocking()?;
    /// // refresh our token
    /// thorium.refresh("user", "password")?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # exec();
    /// ```
    pub fn refresh<T: Into<String>>(&mut self, username: T, password: T) -> Result<(), Error> {
        // logout and invalidate our token
        self.users.logout()?;
        // authenticate and get new token
        let (token, expiration) =
            Self::auth(&self.host, Some(username.into()), Some(password.into()))?;
        // update token expiration
        self.expires = expiration;
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        // build token auth string
        let auth_str = format!("token {}", encoded);
        // update handlers
        self.basic = BasicBlocking::new(&self.host, &self.client);
        self.jobs = JobsBlocking::new(&self.host, &auth_str, &self.client);
        self.reactions = ReactionsBlocking::new(&self.host, &auth_str, &self.client);
        self.pipelines = PipelinesBlocking::new(&self.host, &auth_str, &self.client);
        self.groups = GroupsBlocking::new(&self.host, &auth_str, &self.client);
        self.images = ImagesBlocking::new(&self.host, &auth_str, &self.client);
        self.streams = StreamsBlocking::new(&self.host, &auth_str, &self.client);
        self.users = UsersBlocking::new(&self.host, &auth_str, &self.client);
        self.system = SystemBlocking::new(&self.host, &auth_str, &self.client);
        self.files = FilesBlocking::new(&self.host, &auth_str, &self.client);
        self.repos = ReposBlocking::new(&self.host, &auth_str, &self.client);
        self.events = EventsBlocking::new(&self.host, &auth_str, &self.client);
        Ok(())
    }

    /// Masquerade as a another user
    ///
    /// # Arguments
    ///
    /// * `user` - The user to masquerade as
    pub fn masquerade(&mut self, user: &ScrubbedUser) {
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(user.token.as_bytes());
        // build token auth string
        let auth_str = format!("token {}", encoded);
        // update handlers
        self.basic = BasicBlocking::new(&self.host, &self.client);
        self.jobs = JobsBlocking::new(&self.host, &auth_str, &self.client);
        self.reactions = ReactionsBlocking::new(&self.host, &auth_str, &self.client);
        self.pipelines = PipelinesBlocking::new(&self.host, &auth_str, &self.client);
        self.groups = GroupsBlocking::new(&self.host, &auth_str, &self.client);
        self.images = ImagesBlocking::new(&self.host, &auth_str, &self.client);
        self.streams = StreamsBlocking::new(&self.host, &auth_str, &self.client);
        self.users = UsersBlocking::new(&self.host, &auth_str, &self.client);
        self.system = SystemBlocking::new(&self.host, &auth_str, &self.client);
        self.files = FilesBlocking::new(&self.host, &auth_str, &self.client);
        self.repos = ReposBlocking::new(&self.host, &auth_str, &self.client);
        self.events = EventsBlocking::new(&self.host, &auth_str, &self.client);
    }

    /// Revert back to our original user from a masquerade
    pub fn revert_masquerade(&mut self) {
        // update handlers
        self.basic = BasicBlocking::new(&self.host, &self.client);
        self.jobs = JobsBlocking::new(&self.host, &self.auth_str, &self.client);
        self.reactions = ReactionsBlocking::new(&self.host, &self.auth_str, &self.client);
        self.pipelines = PipelinesBlocking::new(&self.host, &self.auth_str, &self.client);
        self.groups = GroupsBlocking::new(&self.host, &self.auth_str, &self.client);
        self.images = ImagesBlocking::new(&self.host, &self.auth_str, &self.client);
        self.streams = StreamsBlocking::new(&self.host, &self.auth_str, &self.client);
        self.users = UsersBlocking::new(&self.host, &self.auth_str, &self.client);
        self.system = SystemBlocking::new(&self.host, &self.auth_str, &self.client);
        self.files = FilesBlocking::new(&self.host, &self.auth_str, &self.client);
        self.repos = ReposBlocking::new(&self.host, &self.auth_str, &self.client);
        self.events = EventsBlocking::new(&self.host, &self.auth_str, &self.client);
    }
}
