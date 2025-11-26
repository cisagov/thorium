use super::Error;
use crate::send;

// import our static runtime if we need a blocking client
#[cfg(feature = "sync")]
use super::RUNTIME;

// import python bindings
#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};

#[cfg_attr(feature = "sync", thorium_derive::blocking_struct(python))]
#[derive(Clone)]
pub struct Basic {
    host: String,
    client: reqwest::Client,
}

#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
impl Basic {
    /// Creates a new async handler for basic routes in Thorium
    ///
    /// # Arguments
    ///
    /// * `host` - The host/url the Thorium api can be reached at
    /// * `client` - The reqwest client to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::Basic;
    ///
    /// let client = reqwest::Client::new();
    /// let basic = Basic::new("http://127.0.0.1", &client);
    /// ```
    #[must_use]
    pub fn new<T: Into<String>>(host: T, client: &reqwest::Client) -> Self {
        // build basic route handler
        Basic {
            host: host.into(),
            client: client.clone(),
        }
    }
}

#[cfg_attr(feature = "sync", thorium_derive::blocking_struct(python))]
impl Basic {
    /// Have the API identify itself with a static string
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
    /// let identity = thorium.basic.identify().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn identify(&self) -> Result<String, Error> {
        // build request
        let req = self.client.get(format!("{}/api", self.host));
        // send this request and build a string
        let val = send!(self.client, req)?.text().await?;
        Ok(val)
    }

    /// Check the health of the Thorium API
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
    /// let health = thorium.basic.health().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn health(&self) -> Result<bool, Error> {
        // build request
        let req = self.client.get(format!("{}/api/health", self.host));
        // send this request and build a string
        Ok(send!(self.client, req)?.status().is_success())
    }
}
