//! The client for associations in Thorium

#[cfg(feature = "trace")]
use tracing::instrument;

use super::Error;
use crate::models::AssociationRequest;
use crate::send;

// import our static runtime if we need a blocking client
#[cfg(feature = "sync")]
use super::RUNTIME;

/// A handler for the associations routes in Thorium
#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
#[derive(Clone)]
pub struct Associations {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
}

#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
impl Associations {
    /// Creates a new associations handler
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
    /// use thorium::client::Associations;
    ///
    /// let client = reqwest::Client::new();
    /// let associations = Associations::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build associations route handler
        Associations {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }

    /// Creates an [`Association`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `association_req` - The association request to use to add an association to Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    /// use thorium::models::{AssociationRequest, AssociationTarget, AssociationKind};
    /// use uuid::Uuid;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have a source for this association
    /// let source = AssociationTarget::File("63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned());
    /// // have a target for this association (use a real entity uuid instead)
    /// let target = AssociationTarget::Entity { id: Uuid::new_v4(), name: "USW-Pro-Max-24".to_owned() };
    /// // build the association request
    /// let association_req = AssociationRequest::new(AssociationKind::FirmwareFor, source)
    ///   .target(target);
    /// // try to create association in Thorium
    /// thorium.associations.create(&association_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Associations::create", skip_all, err(Debug))
    )]
    pub async fn create(
        &self,
        association_req: &AssociationRequest,
    ) -> Result<reqwest::Response, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/associations/", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .json(association_req)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }
}
