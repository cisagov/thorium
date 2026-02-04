//! The client for entities in Thorium

#[cfg(feature = "trace")]
use tracing::instrument;
use uuid::Uuid;

use super::Error;
use crate::models::{Cursor, Entity, EntityListOpts, EntityRequest, EntityResponse, EntityUpdate};
use crate::{
    add_date, add_query, add_query_bool, add_query_list, add_query_list_clone, send, send_build,
};

// import our static runtime if we need a blocking client
#[cfg(feature = "sync")]
use super::RUNTIME;

/// A handler for the entities routes in Thorium
#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
#[derive(Clone)]
pub struct Entities {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
}

#[cfg_attr(feature = "sync", thorium_derive::blocking_struct)]
impl Entities {
    /// Creates a new entities handler
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
    /// use thorium::client::Entities;
    ///
    /// let client = reqwest::Client::new();
    /// let entities = Entities::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Entities {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }

    /// Creates an [`Entity`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `entity_req` - The entity request to use to add an entity to Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    /// use thorium::models::{EntityRequest, EntityMetadataRequest};
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create a Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have metadata for the entity to create
    /// let meta = EntityMetadataRequest::Other;
    /// // build the entity request
    /// let entity_req = EntityRequest::new("sponge", meta, vec!("bob".to_owned()));
    /// // try to create an entity in Thorium
    /// thorium.entities.create(entity_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Entities::create", skip_all, err(Debug))
    )]
    pub async fn create(&self, entity_req: EntityRequest) -> Result<EntityResponse, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/entities/", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .multipart(entity_req.to_form()?)
            .header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, EntityResponse)
    }

    /// Updates an [`Entity`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the entity to update
    /// * `update` - The update to apply to an entity
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    /// use thorium::models::EntityUpdate;
    /// use uuid::Uuid;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // update Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the entity update
    /// let update = EntityUpdate::default().group("woot");
    /// // try to update an entity in Thorium
    /// thorium.entities.update(Uuid::new_v4(), update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Entities::update", skip_all, err(Debug))
    )]
    pub async fn update(&self, id: Uuid, update: EntityUpdate) -> Result<reqwest::Response, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/entities/{id}", base = self.host);
        // build request
        let req = self
            .client
            .patch(&url)
            .multipart(update.to_form()?)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Lists all entities that meet some search criteria
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for this cursor
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, SearchDate};
    /// use thorium::models::EntityListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a search to list entities from 2020
    /// let search = EntityListOpts::default()
    ///     .start(SearchDate::year(2020, false)?)
    ///     .end(SearchDate::year(2020, true)?)
    ///     // limit it to 100 entities
    ///     .limit(100);
    /// // list the up to 100 entities from 2020
    /// thorium.entities.list(&search).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::entities::list", skip_all, err(Debug))
    )]
    pub async fn list(&self, opts: &EntityListOpts) -> Result<Cursor<Entity>, Error> {
        // build the url for listing entities
        let url = format!("{}/api/entities/", self.host);
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
        // add our tag query params
        for (key, values) in &opts.tags {
            // build the key for this tag param
            let query_key = format!("tags[{key}][]");
            // add this tag keys filters to our query params
            add_query_list_clone!(query, query_key, values);
        }
        add_query_bool!(
            query,
            "tags_case_insensitive".to_owned(),
            opts.tags_case_insensitive
        );
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

    /// Lists all entities that meet some search criteria with details
    ///
    /// # Arguments
    ///
    /// * `search` - The search criteria for this query
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, SearchDate};
    /// use thorium::models::EntityListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a search to list entities from 2020 with details
    /// let search = EntityListOpts::default()
    ///     .start(SearchDate::year(2020, false)?)
    ///     .end(SearchDate::year(2020, true)?)
    ///     // limit it to 100 entities
    ///     .limit(100);
    /// // list the up to 100 entities from 2020
    /// thorium.entities.list_details(&search).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::entities::list_details", skip_all, err(Debug))
    )]
    pub async fn list_details(&self, opts: &EntityListOpts) -> Result<Cursor<Entity>, Error> {
        // build the url for listing entities
        let url = format!("{}/api/entities/details/", self.host);
        // get the correct page size if our limit is smaller then our page_size
        let page_size = opts.limit.map_or_else(
            || opts.page_size,
            |limit| std::cmp::min(opts.page_size, limit),
        );
        // build our query params
        let mut query = vec![("limit".to_owned(), page_size.to_string())];
        add_query_list!(query, "groups[]".to_owned(), opts.groups);
        add_query!(query, "start".to_owned(), opts.start);
        add_query!(query, "end".to_owned(), opts.end);
        add_query!(query, "cursor".to_owned(), opts.cursor);
        // add our tag query params
        for (key, values) in &opts.tags {
            // build the key for this tag param
            let query_key = format!("tags[{key}][]");
            // add this tag keys filters to our query params
            add_query_list_clone!(query, query_key, values);
        }
        add_query_bool!(
            query,
            "tags_case_insensitive".to_owned(),
            opts.tags_case_insensitive
        );
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
