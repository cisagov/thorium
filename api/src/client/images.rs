use uuid::Uuid;

use super::traits::{GenericClient, NotificationsClient};
use super::{Cursor, Error};
use crate::models::{
    Image, ImageKey, ImageRequest, ImageUpdate, Notification, NotificationParams,
    NotificationRequest,
};
use crate::{send, send_build};

/// A handler for the image routes in Thorium
///
/// Images are used to define what each stage of a pipeline look like. Each stage
/// can have multiple images or a single image. Separating the image declarations
/// from pipeline declaration allows you to reuse images across pipelines
/// without having to redefine an image every time. This also makes updating
/// images easier as their is less duplicate information to update.
#[derive(Clone)]
pub struct Images {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
}

impl Images {
    /// Creates a new images handler
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
    /// use thorium::client::Images;
    ///
    /// let client = reqwest::Client::new();
    /// let images = Images::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Images {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// A blocking handler for the image routes in Thorium
        ///
        /// Images are used to define what each stage of a pipeline look like. Each stage
        /// can have multiple images or a single image. Separating the image declarations
        /// from pipeline declaration allows you to reuse images across pipelines
        /// without having to redefine an image every time. This also makes updating
        /// images easier as their is less duplicate information to update.
        #[derive(Clone)]
        pub struct ImagesBlocking {
            /// The host/url that Thorium can be reached at
            host: String,
            /// token to use for auth
            token: String,
            /// A reqwest client for reqwests
            client: reqwest::Client,
        }

        impl ImagesBlocking {
            /// creates a new blocking images handler
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
            /// use thorium::client::ImagesBlocking;
            ///
            /// let images = ImagesBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                ImagesBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Images {
    /// Creates an [`Image`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `image_req` - The image request to use to add an image to Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::ImageRequest};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the image request
    /// let image_req = ImageRequest::new("Corn", "harvester")
    ///     .image("Thorium:CornHarvester");
    /// // try to create image in Thorium
    /// thorium.images.create(&image_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn create(&self, image_req: &ImageRequest) -> Result<reqwest::Response, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/images/", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .json(&image_req)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Gets details about a specific [`Image`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The group this image is in
    /// * `image` - The name of the image to get
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
    /// // get details on this image
    /// let image = thorium.images.get("Corn", "CornHarvester").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get(&self, group: &str, image: &str) -> Result<Image, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/images/data/{group}/{image}",
            base = self.host,
            group = group,
            image = image
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a image from the response
        send_build!(self.client, req, Image)
    }

    /// Lists all images in a group
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list images from
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
    /// // list the first page of images in this group up to 50 images (weakly enforced limit)
    /// let images = thorium.images.list("Corn").next().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[must_use]
    pub fn list(&self, group: &str) -> Cursor<Image> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/images/{group}/",
            base = self.host,
            group = group
        );
        // build cursor
        Cursor::new(url, &self.token, &self.client)
    }

    /// Deletes an [`Image`] from Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The group this image is in
    /// * `name` - The name of the image to delete
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
    /// // delete an image from Thorium
    /// let images = thorium.images.delete("Corn", "CornHarvester").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete(&self, group: &str, name: &str) -> Result<reqwest::Response, Error> {
        // build url for deleting an image
        let url = format!(
            "{base}/api/images/{group}/{name}",
            base = self.host,
            group = group,
            name = name
        );
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Updates an [`Image`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `group` - The group this image is in
    /// * `name` - The name of the image to update
    /// * `update` - The update to apply to this image
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::ImageUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build an image update
    /// let update = ImageUpdate::default()
    ///     .image("Thorium:SuperCornHarvester")
    ///     .timeout(100);
    /// // update this image in Thorium
    /// let images = thorium.images.update("Corn", "CornHarvester", &update).await?;
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
        name: &str,
        update: &ImageUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating an image
        let url = format!(
            "{base}/api/images/{group}/{name}",
            base = self.host,
            group = group,
            name = name
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

    /// Updates all images runtimes
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
    /// // update all of the runtimes for images currently in Thorium
    /// let images = thorium.images.update_runtimes().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update_runtimes(&self) -> Result<reqwest::Response, Error> {
        // build url for updating an image
        let url = format!("{}/api/images/runtimes/update", self.host);
        // build request
        let req = self.client.post(&url).header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Create a notification for an image
    ///
    /// # Arguments
    ///
    /// * `group` - The group the image is in
    /// * `image` - The name of the image
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
    /// // create the notification for the 'harvester' image in the 'corn' group
    /// thorium.images.create_notification("corn", "harvester", &req, &params).await?;
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
        image: T,
        req: &NotificationRequest<Image>,
        params: &NotificationParams,
    ) -> Result<reqwest::Response, Error>
    where
        S: Into<String>,
        T: Into<String>,
    {
        self.create_notification_generic(&ImageKey::new(group, image), req, params)
            .await
    }

    /// Gets all of an image's notifications
    ///
    /// # Arguments
    ///
    /// * `group` - The group that the image belongs to
    /// * `image` - The image whose notifications we're requesting
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
    /// // retrieve all of the notifications for the 'harvester' image in the 'corn' group
    /// let logs = thorium.images.get_notifications("corn", "harvester").await?;
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
        image: T,
    ) -> Result<Vec<Notification<Image>>, Error>
    where
        S: Into<String>,
        T: Into<String>,
    {
        self.get_notifications_generic(&ImageKey::new(group, image))
            .await
    }

    /// Deletes an image notification
    ///
    /// # Arguments
    ///
    /// * `group` - The group that the image belongs to
    /// * `image` - The image whose notification we're deleting
    /// * `id` - The id of the image log to delete
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
    /// thorium.images.delete_notification("corn", "harvester", &id).await?;
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
        image: T,
        id: &Uuid,
    ) -> Result<reqwest::Response, Error>
    where
        S: Into<String>,
        T: Into<String>,
    {
        self.delete_notification_generic(&ImageKey::new(group, image), id)
            .await
    }
}

impl GenericClient for Images {
    /// Provide the base url to the images route in the API
    fn base_url(&self) -> String {
        format!("{}/api/images", self.host)
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

impl NotificationsClient for Images {
    /// The underlying type that has the Notifications (see [`crate::models::backends::NotificationSupport`])
    type NotificationSupport = Image;
}
