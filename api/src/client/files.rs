use cart_rs::UncartStream;
use futures::stream::StreamExt;
use futures::TryStreamExt;
use reqwest::StatusCode;
use std::path::{Path, PathBuf};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio_util::io::StreamReader;
use uuid::Uuid;

#[cfg(feature = "trace")]
use tracing::instrument;

use super::traits::{GenericClient, ResultsClient, ResultsClientHelper, TransferProgress};
use super::Error;
use crate::models::{
    Attachment, CartedSample, CommentRequest, CommentResponse, Cursor, DeleteCommentParams,
    DownloadedSample, FileDeleteOpts, FileDownloadOpts, FileListOpts, OutputMap, OutputRequest,
    OutputResponse, ResultGetParams, Sample, SampleCheck, SampleCheckResponse, SampleListLine,
    SampleRequest, SampleSubmissionResponse, SubmissionUpdate, TagDeleteRequest, TagRequest,
    UncartedSample,
};
use crate::{
    add_date, add_query, add_query_bool, add_query_list, add_query_list_clone, send, send_build,
    send_bytes,
};

/// A handler for the files routes in Thorium
#[derive(Clone)]
pub struct Files {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
}

impl Files {
    /// Creates a new files handler
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
    /// use thorium::client::Files;
    ///
    /// let client = reqwest::Client::new();
    /// let files = Files::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Files {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// A blocking handler for the files routes in Thorium
        #[derive(Clone)]
        pub struct FilesBlocking {
            /// The host/url that Thorium can be reached at
            host: String,
            /// token to use for auth
            token: String,
            /// A reqwest client for reqwests
            client: reqwest::Client,
        }
        impl FilesBlocking {
            /// creates a new blocking files handler
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
            /// use thorium::client::FilesBlocking;
            ///
            /// let files = FilesBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                FilesBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Files {
    /// Creates an [`Sample`] in Thorium by uploading a file
    ///
    /// # Arguments
    ///
    /// * `file_req` - The file request to use to add a file to Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    /// use thorium::models::SampleRequest;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build the file request
    /// let file_req = SampleRequest::new("corn.txt", vec!("plants".to_owned()));
    /// // try to create file in Thorium
    /// thorium.files.create(file_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::create", skip_all, err(Debug))
    )]
    pub async fn create(&self, file_req: SampleRequest) -> Result<SampleSubmissionResponse, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/files/", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .multipart(file_req.to_form().await?)
            .header("authorization", &self.token)
            // use a really long timeout for really large files
            // this is probably done better some other way
            // 86,400 seconds == a day
            .timeout(std::time::Duration::from_secs(86_400));
        // send this request
        send_build!(self.client, req, SampleSubmissionResponse)
    }

    /// Gets details about a specific [`Sample`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the file to get details on
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
    /// // get details on this file
    /// thorium.files.get("325030adff0665689b0360ac9c8398cd62a2377e98e06ad7d3914fabacb0daef").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::get", skip(self), err(Debug))
    )]
    pub async fn get(&self, sha256: &str) -> Result<Sample, Error> {
        // build url for getting info on a sample
        let url = format!(
            "{base}/api/files/sample/{sha256}",
            base = self.host,
            sha256 = sha256
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and build a sample from the response
        send_build!(self.client, req, Sample)
    }

    /// Deletes a file submission from Thorium
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the file to delete
    /// * `submission` - The submission to delete
    /// * `opts` - The options for this delete request
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::FileDeleteOpts;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have a sha256 and submission to delete
    /// let sha256 = "325030adff0665689b0360ac9c8398cd62a2377e98e06ad7d3914fabacb0daef";
    /// let submission = Uuid::new_v4();
    /// let groups: Vec<String> = vec!["my-test-group".to_owned(), "my-other-test-group".to_owned()];
    /// // Delete a file
    /// thorium.files.delete(sha256, &submission, &FileDeleteOpts::default().groups(groups)).await?;
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
            name = "Thorium::Files::delete",
            skip(self, submission, opts),
            fields(submission = submission.to_string()),
            err(Debug)
        )
    )]
    pub async fn delete(
        &self,
        sha256: &str,
        submission: &Uuid,
        opts: &FileDeleteOpts,
    ) -> Result<reqwest::Response, Error> {
        // build url for getting info on a sample
        let url = format!(
            "{base}/api/files/sample/{sha256}/{submission}",
            base = self.host,
            sha256 = sha256,
            submission = submission,
        );
        // build our query params
        let mut query = vec![];
        add_query_list!(query, "groups[]", &opts.groups);
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send this request and build a sample from the response
        send!(self.client, req)
    }

    /// Downloads a file in the CART format
    ///
    /// The options are not truly modified but updating a progress bar if one is set
    /// requires an &mut.
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the file to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::FileDownloadOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // have a sha245 to download and the path to write it too
    /// let sha256 = "325030adff0665689b0360ac9c8398cd62a2377e98e06ad7d3914fabacb0daef";
    /// let path = "file.cart";
    /// // use default options
    /// let mut opts = FileDownloadOpts::default();
    /// // download this file in CART format
    /// thorium.files.download(sha256, path, &mut opts).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::download", skip(self, path), fields(uncart = opts.uncart), err(Debug))
    )]
    pub async fn download<P: Into<PathBuf>>(
        &self,
        sha256: &str,
        path: P,
        opts: &mut FileDownloadOpts,
    ) -> Result<DownloadedSample, Error> {
        // build url for getting info on a sample
        let url = format!(
            "{base}/api/files/sample/{sha256}/download",
            base = self.host,
            sha256 = sha256
        );
        // build and send the request
        let resp = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .send()
            .await?;
        // make sure we got a 200
        match resp.status() {
            StatusCode::OK => {
                // convert our path to a path buf
                let path = path.into();
                // check if this file should be downloaded in an uncarted format or not
                if opts.uncart {
                    // get our response as a stream of bytes
                    let stream = resp.bytes_stream().map_err(|err| {
                        std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
                    });
                    // convert our async read to a buf reader
                    let reader = StreamReader::new(stream);
                    // start uncarting this stream of data
                    let mut uncart = UncartStream::new(reader);
                    // make a file to save the response too
                    let mut file = OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&path)
                        .await?;
                    // write our uncart stream to disk
                    match &mut opts.progress {
                        Some(bar) => {
                            // wrap this read so our progress bar is updated
                            tokio::io::copy(&mut bar.wrap_async_read(uncart), &mut file).await?
                        }
                        None => tokio::io::copy(&mut uncart, &mut file).await?,
                    };
                    Ok(DownloadedSample::Uncarted(UncartedSample { file }))
                } else {
                    // leave this file in a carted format
                    // make a file to save the response too
                    let mut file = OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&path)
                        .await?;
                    // get our response as a stream of bytes
                    let mut stream = resp.bytes_stream();
                    // crawl over this stream and write it to the file
                    while let Some(data) = stream.next().await {
                        // check if we had an error getting bytes
                        let data = data?;
                        // write this part of the stream to disk
                        file.write_all(&data).await?;
                        // update our progress bar if we have one
                        opts.update_progress_bytes(&data);
                    }
                    // build our carted sample object from the bytes
                    Ok(DownloadedSample::Carted(CartedSample { path }))
                }
            }
            // the response had an error status
            _ => Err(Error::from(resp)),
        }
    }

    /// Checks if a sample or submission exists
    ///
    /// # Arguments
    ///
    /// * `check` - The attributes to check for
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::SampleCheck};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get details on this file
    /// let check = SampleCheck::new("325030adff0665689b0360ac9c8398cd62a2377e98e06ad7d3914fabacb0daef");
    /// thorium.files.exists(&check).await?;
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
            name = "Thorium::Files::exists",
            skip_all,
            fields(sha256 = check.sha256),
            err(Debug)
        )
    )]
    pub async fn exists(&self, check: &SampleCheck) -> Result<SampleCheckResponse, Error> {
        // build url for getting info on a sample
        let url = format!("{}/api/files/exists", self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(check);
        // send this request and build a sample from the response
        send_build!(self.client, req, SampleCheckResponse)
    }

    /// Lists all files that meet some search criteria
    ///
    /// # Arguments
    ///
    /// * `opts` - The options for this cursor
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, SearchDate};
    /// use thorium::models::FileListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a search to list files from 2020
    /// let search = FileListOpts::default()
    ///     .start(SearchDate::year(2020, false)?)
    ///     .end(SearchDate::year(2020, true)?)
    ///     // limit it to 100 files
    ///     .limit(100);
    /// // list the up to 100 files from 2020
    /// thorium.files.list(&search).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::list", skip_all, err(Debug))
    )]
    pub async fn list(&self, opts: &FileListOpts) -> Result<Cursor<SampleListLine>, Error> {
        // build the url for listing files
        let url = format!("{}/api/files/", self.host);
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

    /// Lists all files that meet some search criteria with details
    ///
    /// # Arguments
    ///
    /// * `search` - The search criteria for this query
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, SearchDate};
    /// use thorium::models::FileListOpts;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a search to list files from 2020 with details
    /// let search = FileListOpts::default()
    ///     .start(SearchDate::year(2020, false)?)
    ///     .end(SearchDate::year(2020, true)?)
    ///     // limit it to 100 files
    ///     .limit(100);
    /// // list the up to 100 files from 2020
    /// thorium.files.list_details(&search).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::list_details", skip_all, err(Debug))
    )]
    pub async fn list_details(&self, opts: &FileListOpts) -> Result<Cursor<Sample>, Error> {
        // build the url for listing files
        let url = format!("{}/api/files/details/", self.host);
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

    /// Updates an [`Sample`] in Thorium
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the sample to update
    /// * `update` - The update to apply to this sample
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::SubmissionUpdate};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// # let id = uuid::Uuid::new_v4();
    /// // build a submission update for a specific submission
    /// let update = SubmissionUpdate::new(id).name("SuperCorn");
    /// // update this file in Thorium
    /// let files = thorium.files.update("856926b48a936b50e92682807bdae12d5ce39abf509d4c0be82e1327b548705f", &update).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::update", skip(self, update), err(Debug))
    )]
    pub async fn update(
        &self,
        sha256: &str,
        update: &SubmissionUpdate,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a file
        let url = format!(
            "{base}/api/files/sample/{sha256}",
            base = self.host,
            sha256 = sha256
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

    /// Adds new tags to a sample
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sample to add tags to
    /// * `tags` - The tag request to send
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::TagRequest};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a request to add tags to this sample
    /// let tag_req = TagRequest::default().add("plant", "corn");
    /// // add a tag to this file
    /// let files = thorium.files.tag("856926b48a936b50e92682807bdae12d5ce39abf509d4c0be82e1327b548705f", &tag_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::tag", skip(self, tags), err(Debug))
    )]
    pub async fn tag(
        &self,
        sha256: &str,
        tags: &TagRequest<Sample>,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a file
        let url = format!("{}/api/files/tags/{}", self.host, sha256);
        // build request
        let req = self
            .client
            .post(&url)
            .json(tags)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Deletes tags from a sample
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sample to delete tags from
    /// * `tags_del` - The delete tag request to send
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::{TagDeleteRequest, Sample}};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a request to delete a tag from this sample
    /// let tags_del = TagDeleteRequest::default().add("plant", "corn");
    /// // optionally specify group(s) to delete the tag from
    /// let tags_del = tags_del.group("example-group");
    /// // delete a tag from this sample
    /// let files = thorium.files.delete_tags("856926b48a936b50e92682807bdae12d5ce39abf509d4c0be82e1327b548705f", &tags_del).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Files::delete_tags", skip(self, tags_del), err(Debug))
    )]
    pub async fn delete_tags(
        &self,
        sha256: &str,
        tags_del: &TagDeleteRequest<Sample>,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a file
        let url = format!("{}/api/files/tags/{}", self.host, sha256);
        // build request
        let req = self
            .client
            .delete(&url)
            .json(tags_del)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Adds a new comment to a sample
    ///
    /// # Arguments
    ///
    /// * `comment_req` - The comment request to send
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::CommentRequest};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // the sample to add a comment to
    /// let sha256 = "856926b48a936b50e92682807bdae12d5ce39abf509d4c0be82e1327b548705f";
    /// // build a request to add a comment to this sample
    /// let comment_req = CommentRequest::new(sha256, "Corn is tasty");
    /// // comment on this file
    /// let files = thorium.files.comment(comment_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn comment(&self, comment_req: CommentRequest) -> Result<CommentResponse, Error> {
        // build url for commenting on a file
        let url = format!("{}/api/files/comment/{}", self.host, comment_req.sha256);
        // build request
        let req = self
            .client
            .post(&url)
            .multipart(comment_req.to_form().await?)
            .header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, CommentResponse)
    }

    /// Deletes a comment for a sample
    ///
    /// * `sha256` - The SHA256 of the file from which the comment will be deleted
    /// * `comment_id` - The UUID of the comment to delete
    /// * `params` - The parameters to use when deleting the comment
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::DeleteCommentParams;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // the sample to add a comment to
    /// let sha256 = "856926b48a936b50e92682807bdae12d5ce39abf509d4c0be82e1327b548705f";
    /// let id = Uuid::new_v4();
    /// // optionally add specific groups to delete the comment from
    /// let params = DeleteCommentParams::default().groups(vec!["corn", "taco"]);
    /// // delete the comment with the above id from the file with the above SHA256
    /// let files = thorium.files.delete_comment(&sha256, &id, &params).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn delete_comment<T: AsRef<str>>(
        &self,
        sha256: T,
        comment_id: &Uuid,
        params: &DeleteCommentParams,
    ) -> Result<reqwest::Response, Error> {
        // build url for deleting a comment
        let url = format!(
            "{}/api/files/comment/{}/{}",
            self.host,
            sha256.as_ref(),
            comment_id
        );
        // add groups to query if provided
        let mut query = vec![];
        add_query_list!(query, "groups[]", params.groups);
        // build request
        let req = self
            .client
            .delete(&url)
            .header("authorization", &self.token)
            .query(&query);
        // send this request
        send!(self.client, req)
    }

    /// Downloads a comment attachment
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the file with the comment
    /// * `comment` - The id of the comment to download an attachment from
    /// * `attachment` - The id of the attachment to download
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
    /// // have a sample with a comment and an attachment
    /// let sha256 = "325030adff0665689b0360ac9c8398cd62a2377e98e06ad7d3914fabacb0daef";
    /// let comment = Uuid::new_v4();
    /// let attachment = Uuid::new_v4();
    /// // download a comment attachment
    /// thorium.files.download_attachment(&sha256, &comment, &attachment).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn download_attachment<T: AsRef<str>>(
        &self,
        sha256: T,
        comment: &Uuid,
        attachment: &Uuid,
    ) -> Result<Attachment, Error> {
        // build url for getting a comment attachment
        let url = format!(
            "{base}/api/files/comment/download/{sha256}/{comment}/{attachment}",
            base = self.host,
            sha256 = sha256.as_ref(),
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request and read it as bytes
        let data = send_bytes!(self.client, req)?;
        // build our attachment object from the bytes
        Ok(Attachment { data })
    }
}

impl GenericClient for Files {
    /// Provide the base url to the files route in the API
    fn base_url(&self) -> String {
        format!("{}/api/files", self.host)
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

impl ResultsClientHelper for Files {
    /// The underlying type that has the results/outputs (see [`crate::models::results::OutputSupport`])
    type OutputSupport = Sample;
}

impl ResultsClient for Files {
    /// The underlying type that has the results/outputs (see [`crate::models::results::OutputSupport`])
    type OutputSupport = Sample;

    /// Creates an [`Output`] in Thorium for files
    ///
    /// # Arguments
    ///
    /// * `output_req` - The output request to use to add output from a tool
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::ResultsClient;
    /// use thorium::models::{OutputRequest, OutputDisplayType, Sample};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build our output request
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f".to_owned();
    /// let output_req = OutputRequest::<Sample>::new(sha256, "TestTool", "I am an output", OutputDisplayType::String);
    /// // try to create result in Thorium
    /// thorium.files.create_result(output_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    async fn create_result(
        &self,
        output_req: OutputRequest<Sample>,
    ) -> Result<OutputResponse, Error> {
        self.create_result_generic(output_req).await
    }

    /// Gets results for a specific file
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the sample to get results for
    /// * `params` - The params to use when getting this samples results
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::ResultsClient;
    /// use thorium::models::ResultGetParams;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // get any results for this hash
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// thorium.files.get_results(sha256, &ResultGetParams::default()).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    async fn get_results<T: AsRef<str>>(
        &self,
        sha256: T,
        params: &ResultGetParams,
    ) -> Result<OutputMap, Error> {
        self.get_results_generic(sha256, params).await
    }

    /// Downloads a result file
    ///
    /// # Arguments
    ///
    /// * `sha256` - The sha256 of the sample to get a result file for
    /// * `tool` - The tool to that made this result file
    /// * `result_id` - The uuid for this result
    /// * `path` - The path for the file to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::client::ResultsClient;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // the hash to get results for
    /// let sha256 = "63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f";
    /// // download an attachment from this result
    /// thorium.files.download_result_file(sha256, "tool", &Uuid::new_v4(), "crabs.png").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    async fn download_result_file<T, P>(
        &self,
        sha256: T,
        tool: &str,
        result_id: &Uuid,
        path: P,
    ) -> Result<Attachment, Error>
    where
        T: AsRef<str>,
        P: AsRef<Path>,
    {
        self.download_result_file_generic(sha256, tool, result_id, path)
            .await
    }
}
