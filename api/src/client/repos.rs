use cart_rs::UncartStream;
use futures::stream::StreamExt;
use futures::TryStreamExt;
use git2::build::CheckoutBuilder;
use reqwest::StatusCode;
use std::path::{Path, PathBuf};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio_tar::Archive;
use tokio_util::io::StreamReader;
use uuid::Uuid;

#[cfg(feature = "trace")]
use tracing::instrument;

use super::traits::{GenericClient, ResultsClient, ResultsClientHelper, TransferProgress};
use super::Error;
use crate::models::{
    Attachment, CommitListOpts, Commitish, CommitishDetails, CommitishMapRequest, Cursor,
    OutputMap, OutputRequest, OutputResponse, Repo, RepoCreateResponse, RepoDataUploadResponse,
    RepoDownloadOpts, RepoListLine, RepoListOpts, RepoRequest, ResultGetParams, TagDeleteRequest,
    TagRequest, TarredRepo, UntarredRepo,
};
use crate::{
    add_date, add_query, add_query_bool, add_query_list, add_query_list_clone, send, send_build,
};

/// repos handler for the Thorium client
#[derive(Clone)]
pub struct Repos {
    /// url/ip of the Thorium ip
    host: String,
    /// token to use for auth
    token: String,
    /// reqwest client object
    client: reqwest::Client,
}

impl Repos {
    /// Creates a new repos handler
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
    /// use thorium::client::Repos;
    ///
    /// let client = reqwest::Client::new();
    /// let repos = Repos::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Repos {
            host: host.to_owned(),
            client: client.clone(),
            token: token.to_owned(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        /// repos handler for the Thorium client
        #[derive(Clone)]
        pub struct ReposBlocking {
            /// url/ip of the Thorium ip
            host: String,
            /// token to use for auth
            token: String,
            /// reqwest client object
            client: reqwest::Client,
        }

        impl ReposBlocking {
            /// creates a new blocking repos handler
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
            /// use thorium::client::ReposBlocking;
            ///
            /// let repos = ReposBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                ReposBlocking {
                    host: host.to_owned(),
                    client: client.clone(),
                    token: token.to_owned(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Repos {
    /// Register a repository in Thorium
    ///
    /// # Arguments
    ///
    /// * `req` - The repo to add
    pub async fn create(&self, req: &RepoRequest) -> Result<RepoCreateResponse, Error> {
        // build url for adding commits to a repo
        let url = format!("{base}/api/repos/", base = self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(req)
            // use a really long timeout for really large repos
            // this is probably done better some other way
            // 86,400 seconds == a day
            .timeout(std::time::Duration::from_secs(86_400));
        // send this request
        send_build!(self.client, req, RepoCreateResponse)
    }

    /// Get info on a specific repository
    ///
    /// # Arguments
    ///
    /// * `repo` - The url of the repo to get info on
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::get", skip(self), err(Debug))
    )]
    pub async fn get(&self, repo: &str) -> Result<Repo, Error> {
        // build url for adding commits to a repo
        let url = format!(
            "{base}/api/repos/data/{repo}",
            base = self.host,
            repo = repo
        );
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, Repo)
    }

    /// Upload a repositories data
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to upload data for
    /// * `zip` - The tarred repo to save to s3
    /// * `groups` - The groups to share this repo with
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::upload", skip(self, tar), err(Debug))
    )]
    pub async fn upload(
        &self,
        repo: &str,
        tar: TarredRepo,
        groups: Vec<String>,
    ) -> Result<RepoDataUploadResponse, Error> {
        // build url for adding commits to a repo
        let url = format!(
            "{base}/api/repos/data/{repo}",
            base = self.host,
            repo = repo
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .multipart(tar.to_form(groups).await?);
        // send this request
        send_build!(self.client, req, RepoDataUploadResponse)
    }

    /// Adds commits to a repository in Thorium
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to add new commits too
    /// * `zip` - the sha256 of the data zip to add commits too
    /// * `map` - The map of commits that are contained in this zip
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::add_commits", skip(self, map), err(Debug))
    )]
    pub async fn add_commits(
        &self,
        repo: &str,
        zip: &str,
        map: &CommitishMapRequest,
    ) -> Result<reqwest::Response, Error> {
        // build url for adding commits to a repo
        let url = format!(
            "{base}/api/repos/commitishes/{zip}/{repo}",
            base = self.host,
            zip = zip,
            repo = repo,
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(map);
        // send this request
        send!(self.client, req)
    }

    ///// Adds new tags to a repo
    /////
    ///// # Arguments
    /////
    ///// * `repo` - The url of the repo to add tags too
    ///// * `tags` - The tag request to send
    /////
    ///// # Examples
    /////
    ///// ```
    ///// use thorium::{Thorium, models::TagRequest};
    ///// # use thorium::Error;
    /////
    ///// # async fn exec() -> Result<(), Error> {
    ///// // create Thorium client
    ///// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    ///// // build a request to add tags to this repo
    ///// let tag_req = TagRequest::default().add("Ferris", "IsCool");
    ///// // add a tag to this repo
    ///// thorium.repos.tag("github.com/rust-lang/rust", &tag_req).await?;
    ///// # // allow test code to be compiled but don't unwrap as no API instance would be up
    ///// # Ok(())
    ///// # }
    ///// # tokio_test::block_on(async {
    ///// #    exec().await
    ///// # });
    ///// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::tag", skip(self, tags), err(Debug))
    )]
    pub async fn tag(
        &self,
        repo: &str,
        tags: &TagRequest<Repo>,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a file
        let url = format!("{}/api/repos/tags/{}", self.host, repo);
        // build request
        let req = self
            .client
            .post(&url)
            .json(tags)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Deletes tags from a repo
    ///
    /// # Arguments
    ///
    /// * `url` - The URL of the repo to delete tags from
    /// * `tags_del` - The delete tag request to send
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::{TagDeleteRequest, Repo}};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a request to delete a tag from this repo
    /// let tags_del: TagDeleteRequest<Repo> = TagDeleteRequest::default().add("plant", "corn");
    /// // optionally specify group(s) to delete the tag from
    /// let tags_del = tags_del.group("example-group");
    /// // delete a tag from this repo
    /// thorium.repos.delete_tags("example.com/user/repo", &tags_del).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::delete_tags", skip(self, tags_del), err(Debug))
    )]
    pub async fn delete_tags(
        &self,
        url: &str,
        tags_del: &TagDeleteRequest<Repo>,
    ) -> Result<reqwest::Response, Error> {
        // build url for updating a file
        let url = format!("{}/api/repos/tags/{}", self.host, url);
        // build request
        let req = self
            .client
            .delete(&url)
            .json(tags_del)
            .header("authorization", &self.token);
        // send this request
        send!(self.client, req)
    }

    /// Downloads the zip for a specific repo
    ///
    /// If you are going to immediately unzip this repo then you want `download_unpack` instead.
    /// If a commit is passed this zip will contain that commit but it likely will still need to
    /// be checked out.
    ///
    /// # Arguments
    ///
    /// * `repo` - The url of the repo to download
    /// * `opts` - The options for downloading a repo
    /// * `path` - The path to download this zipped repo to
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::download", skip(self, opts, path), err(Debug))
    )]
    pub async fn download<P: Into<PathBuf>>(
        &self,
        repo: &str,
        opts: &RepoDownloadOpts,
        path: P,
    ) -> Result<TarredRepo, Error> {
        // get the name of our repo
        let repo_path = Path::new(repo);
        let repo_name = match repo_path.file_name() {
            Some(file_name) => file_name.to_string_lossy().to_string(),
            None => return Err(Error::new("Failed to get repository name".to_owned())),
        };
        // convert our path to a path buf
        let mut path: PathBuf = path.into();
        if path.is_dir() {
            // add repo name
            path.push(&repo_name);
            // append .tar.cart extension
            path.as_mut_os_string().push(".tar.cart");
        }
        // build url for adding commits to a repo
        let url = format!(
            "{base}/api/repos/download/{repo}",
            base = self.host,
            repo = repo
        );
        // build our query params
        let mut query = vec![];
        add_query!(query, "commitish", opts.commitish);
        add_query_list!(query, "kinds[]", opts.kinds);
        // build request
        let resp = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&query)
            .send()
            .await?;
        // make sure we got a 200
        match resp.status() {
            StatusCode::OK => {
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
                Ok(TarredRepo {
                    name: repo_name,
                    repo: path,
                })
            }
            // the response had an error status
            _ => Err(Error::from(resp)),
        }
    }

    /// Downloads a specific repo and untars it as we go
    ///
    /// If a commit is passed this will checkout our repo to the target commit.
    ///
    /// # Arguments
    ///
    /// * `repo` - The url of the repo to download
    /// * `opts` - The options for downloading a repo
    /// * `path` - The path to download this zipped repo to
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Repos::download_unpack",
            skip(self, opts, path),
            err(Debug)
        )
    )]
    pub async fn download_unpack<P: Into<PathBuf>>(
        &self,
        repo: &str,
        opts: &RepoDownloadOpts,
        path: P,
    ) -> Result<UntarredRepo, Error> {
        // build url for adding commits to a repo
        let url = format!(
            "{base}/api/repos/download/{repo}",
            base = self.host,
            repo = repo
        );
        // build our query params
        let mut query = vec![];
        add_query!(query, "commitish", opts.commitish);
        add_query_list!(query, "kinds[]", opts.kinds);
        // build request
        let resp = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&query)
            .send()
            .await?;
        // convert our path to a path buf
        let path = path.into();
        // get our response as a stream of bytes
        let stream = resp
            .bytes_stream()
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()));
        // convert our async read to a buf reader
        let reader = StreamReader::new(stream);
        // start uncarting this stream of data
        let uncart = UncartStream::new(reader);
        // untar this repo to disk
        Archive::new(uncart).unpack(&path).await?;
        // clone our target path to add in the repo name
        let mut target = path.clone();
        // get the final name of the repo
        match Path::new(repo).file_name() {
            Some(name) => target.push(name),
            None => {
                return Err(Error::new(format!(
                    "Failed to extract repo name from {repo}"
                )))
            }
        };
        // build our untarred repo object
        let untarred = UntarredRepo::new(target)?;
        // set our checkout options
        let mut checkout_opts = CheckoutBuilder::new();
        // ensure we match head under any circumstances
        checkout_opts.force();
        // open our untarred repo as a git repo
        let repo = git2::Repository::open(&untarred.path)?;
        // checkout head to resolve any changes
        repo.checkout_head(Some(&mut checkout_opts))?;
        // checkout the correct commit if one wa specified
        if let Some(commitish) = &opts.commitish {
            untarred.checkout(commitish)?;
        }
        Ok(untarred)
    }

    /// Lists repos in Thorium
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use when listing repos
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::list", skip(self), err(Debug))
    )]
    pub async fn list(&self, opts: &RepoListOpts) -> Result<Cursor<RepoListLine>, Error> {
        // build the url for listing repos
        let url = format!("{}/api/repos/", self.host);
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

    /// Lists repos details in Thorium
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use when listing repo details
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::list_details", skip(self), err(Debug))
    )]
    pub async fn list_details(&self, opts: &RepoListOpts) -> Result<Cursor<Repo>, Error> {
        // build the url for listing repo details
        let url = format!("{}/api/repos/details/", self.host);
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

    /// Lists commitishes for a specific repo
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to list commits for
    /// * `params` - The params to use when listing commits
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Repos::list_commitishes", skip(self), err(Debug))
    )]
    pub async fn list_commitishes(
        &self,
        repo: &str,
        opts: &CommitListOpts,
    ) -> Result<Cursor<Commitish>, Error> {
        // build the url for listing commits
        let url = format!("{}/api/repos/commitishes/{}", self.host, repo);
        // get the correct page size if our limit is smaller then our page_size
        let page_size = opts.limit.map_or_else(
            || opts.page_size,
            |limit| std::cmp::min(opts.page_size, limit),
        );
        // build our query params
        let mut query = vec![("limit", page_size.to_string())];
        add_query_list!(query, "groups[]", opts.groups);
        add_date!(query, "start", opts.start);
        add_date!(query, "end", opts.end);
        add_query!(query, "cursor", opts.cursor);
        add_query_list!(query, "kinds[]", opts.kinds);
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

    /// Lists commitishes with details for a specific repo
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to list commits for
    /// * `params` - The params to use when listing commits
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Repos::list_commitish_details",
            skip(self),
            err(Debug)
        )
    )]
    pub async fn list_commitish_details(
        &self,
        repo: &str,
        opts: &CommitListOpts,
    ) -> Result<Cursor<CommitishDetails>, Error> {
        // build the url for listing commit details
        let url = format!("{}/api/repos/commitish-details/{}", self.host, repo);
        // get the correct page size if our limit is smaller then our page_size
        let page_size = opts.limit.map_or_else(
            || opts.page_size,
            |limit| std::cmp::min(opts.page_size, limit),
        );
        // build our query params
        let mut query = vec![("limit", page_size.to_string())];
        add_query_list!(query, "groups[]", opts.groups);
        add_date!(query, "start", opts.start);
        add_date!(query, "end", opts.end);
        add_query!(query, "cursor", opts.cursor);
        add_query_list!(query, "kinds[]", opts.kinds);
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

impl GenericClient for Repos {
    /// Provide the base url to the repo routes in the API
    fn base_url(&self) -> String {
        format!("{}/api/repos", self.host)
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

impl ResultsClientHelper for Repos {
    /// The underlying type that has the results/outputs (see [`crate::models::results::OutputSupport`])
    type OutputSupport = Repo;
}

impl ResultsClient for Repos {
    /// The underlying type that has the results/outputs (see [`crate::models::results::OutputSupport`])
    type OutputSupport = Repo;

    /// Creates an [`Output`] in Thorium for repos
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
    /// use thorium::models::{OutputRequest, OutputDisplayType, Repo};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build our output request
    /// let repo = "github.com/rust-lang/rust".to_owned();
    /// let output_req = OutputRequest::<Repo>::new(repo, "TestTool", "I am an output", OutputDisplayType::String);
    /// // try to create result in thorium
    /// thorium.repos.create_result(output_req).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "ResultsClient<Repos>::create_results", skip_all, err(Debug))
    )]
    async fn create_result(
        &self,
        output_req: OutputRequest<Repo>,
    ) -> Result<OutputResponse, Error> {
        self.create_result_generic(output_req).await
    }

    /// Gets results for a specific repo
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to get results for
    /// * `params` - The params to use when getting this repo's results
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
    /// // get any results for this repo
    /// let repo = "github.com/user/repo";
    /// thorium.repos.get_results(repo, &ResultGetParams::default()).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "ResultsClient<Repos>::get_results", skip(self), fields(repo = repo.as_ref()), err(Debug))
    )]
    async fn get_results<T: AsRef<str>>(
        &self,
        repo: T,
        params: &ResultGetParams,
    ) -> Result<OutputMap, Error> {
        // trim any ending '/' from the repo URL
        let repo_trimmed = repo.as_ref().trim_end_matches('/');
        self.get_results_generic(repo_trimmed, params).await
    }

    /// Downloads a specific result file for a repo
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to get a result file for
    /// * `tool` - The tool to that made this result file
    /// * `result_id` - The uuid for this result
    /// * `path` - The path of the result file to download
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
    /// // the repo to get results for
    /// let repo = "github.com/user/repo";
    /// // download an attachment from this result
    /// thorium.repos.download_result_file(repo, "tool", &Uuid::new_v4(), "crabs.png").await?;
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
            name = "ResultsClient<Repos>::download_result_file",
            skip(self, repo, result_id, path),
            fields(repo = repo.as_ref())
            err(Debug)
        )
    )]
    async fn download_result_file<T, P>(
        &self,
        repo: T,
        tool: &str,
        result_id: &Uuid,
        path: P,
    ) -> Result<Attachment, Error>
    where
        T: AsRef<str>,
        P: AsRef<Path>,
    {
        // trim any ending '/' from the repo URL
        let repo_trimmed = repo.as_ref().trim_end_matches('/');
        self.download_result_file_generic(repo_trimmed, tool, result_id, path)
            .await
    }
}
