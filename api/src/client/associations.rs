//! The client for associations in Thorium

#[cfg(feature = "trace")]
use tracing::instrument;

use uuid::Uuid;

use super::Error;
use crate::models::{
    Association, AssociationDeleteRequest, AssociationListOpts, AssociationRequest, Cursor,
};
use crate::{add_date, add_query, add_query_list, send};

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

    /// Deletes associations in Thorium
    ///
    /// Only the submitter of an association, a manager/owner of its group, or an admin can
    /// delete an association.
    ///
    /// # Arguments
    ///
    /// * `delete_req` - The associations to delete
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{AssociationDeleteRequest, AssociationKind, AssociationTarget};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the source and target of the associations to delete
    /// let source = AssociationTarget::File("63b9b5a8c4d9f2b7cb5d2a1b5e8e1fd4f6d9e9a5c2b1a0f8e7d6c5b4a3928170".to_owned());
    /// let target = AssociationTarget::Repo("github.com/curl/curl".to_owned());
    /// // build our association delete request
    /// let delete_req = AssociationDeleteRequest::new(AssociationKind::AssociatedWith, source)
    ///   .target(target);
    /// // try to delete these associations in Thorium
    /// thorium.associations.delete(&delete_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Associations::delete", skip_all, err(Debug))
    )]
    pub async fn delete(
        &self,
        delete_req: &AssociationDeleteRequest,
    ) -> Result<reqwest::Response, Error> {
        // build url for deleting associations
        let url = format!("{base}/api/associations/", base = self.host);
        // build request
        let req = self
            .client
            .delete(&url)
            .json(delete_req)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Lists the associations for a file in Thorium
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the file to list associations for
    /// * `opts` - The options to use when listing associations
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::AssociationListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 100 associations for a file
    /// let opts = AssociationListOpts::default().limit(100);
    /// let sha256 = "63b9b5a8c4d9f2b7cb5d2a1b5e8e1fd4f6d9e9a5c2b1a0f8e7d6c5b4a3928170";
    /// let cursor = thorium.associations.list_file(sha256, &opts).await?;
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
            name = "Thorium::Associations::list_file",
            skip(self, opts),
            err(Debug)
        )
    )]
    pub async fn list_file(
        &self,
        sha256: &str,
        opts: &AssociationListOpts,
    ) -> Result<Cursor<Association>, Error> {
        // build the url for listing this files associations
        let url = format!("{base}/api/files/associations/{sha256}", base = self.host);
        // build our query params
        let query = list_query(opts);
        // get the data for this request and create our cursor
        Cursor::new(
            &url,
            opts.page_size,
            opts.limit,
            &self.token,
            &query,
            &self.client,
        )
        .await
    }

    /// Lists the associations for a repo in Thorium
    ///
    /// # Arguments
    ///
    /// * `repo` - The url of the repo to list associations for (without a scheme or `.git`)
    /// * `opts` - The options to use when listing associations
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::AssociationListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 100 associations for a repo
    /// let opts = AssociationListOpts::default().limit(100);
    /// let cursor = thorium.associations.list_repo("github.com/curl/curl", &opts).await?;
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
            name = "Thorium::Associations::list_repo",
            skip(self, opts),
            err(Debug)
        )
    )]
    pub async fn list_repo(
        &self,
        repo: &str,
        opts: &AssociationListOpts,
    ) -> Result<Cursor<Association>, Error> {
        // build the url for listing this repos associations
        let url = format!("{base}/api/repos/associations/{repo}", base = self.host);
        // build our query params
        let query = list_query(opts);
        // get the data for this request and create our cursor
        Cursor::new(
            &url,
            opts.page_size,
            opts.limit,
            &self.token,
            &query,
            &self.client,
        )
        .await
    }

    /// Lists the associations for an entity in Thorium
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the entity to list associations for
    /// * `opts` - The options to use when listing associations
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::AssociationListOpts;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list up to 100 associations for an entity
    /// let opts = AssociationListOpts::default().limit(100);
    /// let id = Uuid::parse_str("bf5b4ea2-7a43-4a69-8c41-1f3b2c6e5d8a").unwrap();
    /// let cursor = thorium.associations.list_entity(id, &opts).await?;
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
            name = "Thorium::Associations::list_entity",
            skip(self, opts),
            err(Debug)
        )
    )]
    pub async fn list_entity(
        &self,
        id: Uuid,
        opts: &AssociationListOpts,
    ) -> Result<Cursor<Association>, Error> {
        // build the url for listing this entities associations
        let url = format!("{base}/api/entities/associations/{id}", base = self.host);
        // build our query params
        let query = list_query(opts);
        // get the data for this request and create our cursor
        Cursor::new(
            &url,
            opts.page_size,
            opts.limit,
            &self.token,
            &query,
            &self.client,
        )
        .await
    }
}

/// Build the query params for listing associations
///
/// # Arguments
///
/// * `opts` - The options to use when listing associations
fn list_query(opts: &AssociationListOpts) -> Vec<(String, String)> {
    // get the correct page size if our limit is smaller then our page_size
    let page_size = opts.limit.map_or_else(
        || opts.page_size,
        |limit| std::cmp::min(opts.page_size, limit),
    );
    // build our query params
    let mut query = vec![("limit".to_owned(), page_size.to_string())];
    add_query_list!(query, "groups[]".to_owned(), opts.groups);
    add_date!(query, "start".to_owned(), opts.start);
    add_date!(query, "end".to_owned(), opts.end);
    add_query!(query, "cursor".to_owned(), opts.cursor);
    query
}
