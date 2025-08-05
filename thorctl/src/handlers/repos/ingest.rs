//! A worker that ingests repos into Thorium
use chrono::prelude::*;
use futures::stream::{FuturesUnordered, StreamExt};
use git2::build::RepoBuilder;
use git2::{Cred, FetchOptions, ProxyOptions, RemoteCallbacks};
use gix::bstr::ByteSlice;
use gix::remote::Direction;
use gix::Id;
use kanal::{AsyncSender, Receiver, Sender};
use owo_colors::OwoColorize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thorium::client::conf::GitSettings;
use thorium::models::{
    CommitRequest, CommitishMapRequest, CommitishRequest, RepoDownloadOpts, UntarredRepo,
};
use thorium::Thorium;
use tokio::task::JoinHandle;
use url::Url;
use uuid::Uuid;

use super::progress::{Bar, BarKind, MultiBar};
use crate::args::{repos::IngestRepos, Args};
use crate::check;
use crate::handlers::{Monitor, MonitorMsg, Worker};
use crate::{CtlConf, Error};

/// Fix a remote to not be an ssh clone or contain an ending "/" or ".git"
///
/// # Arguments
///
/// * `remote` - The remote URL to fix
#[allow(clippy::needless_pass_by_value)]
fn fix_remote(remote: String) -> String {
    // trim ending /'s or ".git"
    let remote = remote.trim_end_matches('/').trim_end_matches(".git");
    // convert ssh clones to scheme-less http style remotes
    if let Some(fixed) = remote.strip_prefix("git@") {
        // replace the first : with a /
        fixed.replacen(':', "/", 1)
    } else {
        remote.to_string()
    }
}

/// Trim any extra paths from our repo url
///
/// # Arguments
///
/// * `raw` - The url to trim
fn trim_url(raw: &str) -> Result<(Url, String), Error> {
    // parse our url
    let mut url = Url::parse(raw)?;
    // get the total number of url segments
    let total_segments = match url.path_segments() {
        Some(segments) => segments.count(),
        None => return Err(Error::new(format!("{url} does not have any segments"))),
    };
    // get a mutable ref to our urls path
    {
        // just unwrap because if we were a base we would have returned does not have any segments
        let mut segments = url.path_segments_mut().unwrap();
        // pop all but the first 2 segments
        for _ in 2..total_segments {
            segments.pop();
        }
    }
    // get our url as a string
    let url_string = url.to_string();
    Ok((url, url_string))
}

// cast a repo url string to a url and get the repo name
///
/// # Arguments
///
/// * `raw` - The url to parse
/// * `conf` - The config for Thorctl
fn parse_url(raw: &str, conf: &CtlConf) -> Result<(String, String, String), Error> {
    // trim any trailing / from our repo url
    let raw = raw.trim_end_matches('/');
    // if this repo does not contain some scheme then add one
    let (url, thorium_url, clone_url) = match &raw[..4] {
        // this starts with a scheme
        "http" => {
            // get our trimmed url
            let (url, trimmed) = trim_url(raw)?;
            // get the scheme for this url
            let scheme = url.scheme();
            // build our thorium url
            let thorium_url = trimmed[scheme.len() + 3..].to_string();
            // if we have ssh keys set up then clone with ssh instead
            if conf.git.is_some() {
                // skip the opening ""://" and replace the first / with a :
                let replaced = trimmed[scheme.len() + 3..].replacen('/', ":", 1);
                // set the ssh scheme for our clone url
                let fixed = format!("git@{replaced}");
                // return the ssh clone urls
                (url, thorium_url, fixed)
            } else {
                // return the non ssh clone urls
                (url, thorium_url, trimmed)
            }
        }
        // this is an ssh clone so get the https version too
        "git@" => {
            // replace the first : with a /
            let thorium_url = raw[4..].replacen(':', "/", 1);
            // build our https clone url
            let fixed = format!("https://{}", &thorium_url);
            // get our trimmed url
            let (url, trimmed) = trim_url(&fixed)?;
            (url, thorium_url, trimmed)
        }
        // we don't have a scheme so just default to ssh
        _ => {
            // set the https scheme for a parseable url
            let parsable = format!("https://{raw}");
            // get our trimmed url
            let (url, trimmed) = trim_url(&parsable)?;
            // if we have ssh keys set up then clone with ssh instead
            if conf.git.is_some() {
                // replace the first / with a :
                let replaced = trimmed[8..].replacen('/', ":", 1);
                // set the ssh scheme for our clone url
                let fixed = format!("git@{replaced}");
                // return the ssh clone urls
                (url, raw.to_string(), fixed)
            } else {
                // return the non ssh clone urls
                (url, raw.to_string(), trimmed)
            }
        }
    };
    // try to get this repos name
    let name = match url.path_segments() {
        // skip the repo owner/group and get the repo name
        Some(mut segments) => match segments.nth(1) {
            Some(name) => name,
            None => return Err(Error::new(format!("{raw} does not contain a repo name"))),
        },
        None => return Err(Error::new(format!("{raw} does not contain a repo name"))),
    };
    // make sure our name is not empty
    if name.is_empty() {
        return Err(Error::new(format!("{raw} does not contain a repo name")));
    }
    // strip .git from our name if its set
    let name = if let Some(stripped) = name.strip_suffix(".git") {
        stripped.to_owned()
    } else {
        name.to_owned()
    };
    // strip .git from our thorium url if its set
    let thorium_url = if let Some(stripped) = thorium_url.strip_suffix(".git") {
        stripped.to_owned()
    } else {
        thorium_url.clone()
    };
    Ok((clone_url, thorium_url, name))
}

/// Help clone a repo with ssh keys
///
/// # Arguments
///
/// * `clone_url` - The url to the repo to clone
/// * `path` - The path to clone our repo into
/// * `bar` - The bar to log progress updates with
async fn clone_repo_https(clone_url: String, path: PathBuf, bar: &Bar) -> Result<(), Error> {
    // set that we are cloning this repo
    bar.refresh("Cloning With HTTP(S)", BarKind::UnboundIO);
    // spawn a blocking task to clone our repo
    tokio::task::spawn_blocking(move || {
        // build our proxy options
        let mut proxy = ProxyOptions::new();
        proxy.auto();
        // add our proxy options to our fetch options
        let mut fetch = FetchOptions::default();
        fetch.proxy_options(proxy);
        // build our repo builder and add our fetch options
        let mut repo = RepoBuilder::new();
        repo.fetch_options(fetch);
        // clone this repo
        repo.clone(&clone_url, &path)
    })
    // nesting result unwraps makes me cry but this all goes away when gitoxide can do what we need
    .await??;
    Ok(())
}

/// Help clone a repo with ssh keys
///
/// # Arguments
///
/// * `clone_url` - The url to clone
/// * `path` - The path to clone our repo into
/// * `git_conf` - The git settings to use
/// * `bar` - The progress bar to update
async fn clone_repo_ssh(
    clone_url: String,
    path: PathBuf,
    git_conf: GitSettings,
    bar: &Bar,
) -> Result<(), Error> {
    // set that we are cloning this repo
    bar.refresh("Cloning With SSH", BarKind::UnboundIO);
    // clone our bar so we update progress
    let cloned_bar = bar.clone();
    // spawn a blocking task to clone our repo
    tokio::task::spawn_blocking(move || {
        // build our proxy options
        let mut proxy = ProxyOptions::new();
        proxy.auto();
        // add our proxy options to our fetch options
        let mut fetch = FetchOptions::default();
        fetch.proxy_options(proxy);
        // build a new callback to setup
        let mut callbacks = RemoteCallbacks::new();
        // set our credentials info
        callbacks.credentials(|_url, username, _| {
            Cred::ssh_key(username.unwrap_or("git"), None, &git_conf.ssh_keys, None)
        });
        callbacks.transfer_progress(|stats| {
            // increment our progress bar
            cloned_bar.set_position(stats.received_bytes() as u64);
            // ¯\_(ツ)_/¯ all the examples always return true here
            true
        });
        // add our callbacks
        fetch.remote_callbacks(callbacks);
        // build our repo builder and add our fetch options
        let mut repo = RepoBuilder::new();
        repo.fetch_options(fetch);
        // clone this repo
        repo.clone(&clone_url, &path)
    })
    // nesting result unwraps makes me cry but this all goes away when gitoxide can do what we need
    .await??;
    Ok(())
}

/// Crawl all commits across all references
///
/// # Arguments
///
/// * `untarred` - The untarred repo to crawl for commits
/// * `tx` - The channel to send commits to ingest too
/// * `bar` - The bar to log progress with
fn crawl_all(
    untarred: UntarredRepo,
    tx: Sender<(String, CommitishRequest)>,
    bar: Bar,
) -> Result<(), Error> {
    // open our untarred repo as a git repo
    let repo = gix::open(&untarred.path)?;
    // track what commits we have already seen
    let mut commits: HashSet<_> = HashSet::with_capacity(1000);
    // get an iter over the references in this repo
    let refs = repo.references()?;
    // track the reference IDs
    let mut ref_ids = Vec::with_capacity(1);
    // crawl over these repos and get the non commit commitishes and reference IDs
    for refer in refs.all().unwrap() {
        if let Ok(refer_ok) = refer {
            // detect if this is a branch that peels to a commit
            if let gix::refs::TargetRef::Object(_) = refer_ok.target() {
                // get this commitish request
                if let Some((name, commitish_req)) = CommitishRequest::new_gix(&refer_ok)? {
                    // send this commitish to our ingestor
                    tx.send((name, commitish_req))?;
                    // get this references id
                    if let Some(id) = refer_ok.try_id() {
                        ref_ids.push(id);
                    }
                }
            }
        }
    }
    // build a revwalk over this repos commits
    for info in repo.rev_walk(ref_ids).all().unwrap() {
        // skip any infos that are not commits
        if let Ok(info) = info {
            // check if this has already been ingested
            if !commits.contains(&info.id) {
                // cast this info to a commit
                let commit = info.object().unwrap();
                // get this commits hash
                let hash = info.id.to_string();
                // convert our commit to a commit request
                let commit_req = CommitRequest::new_gix(commit);
                // send this commit to our ingestor
                tx.send((hash, CommitishRequest::Commit(commit_req)))?;
                // add this commit to our ingested commits
                commits.insert(info.id);
                // increment our bars length
                bar.inc_length(1);
            }
        }
    }
    Ok(())
}

/// Crawl all commits across specific references
///
/// # Arguments
///
/// * `untarred` - The untarred repo to crawl for commits
/// * `tx` - The channel to send commits to ingest too
/// * `references` - The references to scan
/// * `bar` - The bar to log progress with
fn crawl_specific(
    untarred: UntarredRepo,
    tx: Sender<(String, CommitishRequest)>,
    references: Vec<String>,
    bar: Bar,
) -> Result<(), Error> {
    // open our untarred repo as a git repo
    let repo = gix::open(untarred.path).unwrap();
    // track what commits we have already seen
    let mut commits: HashSet<_> = HashSet::with_capacity(1000);
    // build a list of reference IDs
    let ids = references
        .iter()
        .map(|refer| repo.find_reference(refer).map(|refer| refer.id()))
        .collect::<Result<Vec<Id>, gix::reference::find::existing::Error>>()?;
    // build a revwalk over this repos commits
    for info in repo.rev_walk(ids).all().unwrap() {
        // unwrap this error
        let info = info.unwrap();
        // check if this has already been ingested
        if !commits.contains(&info.id) {
            // cast this info to a commit
            let commit = info.object().unwrap();
            // get this commits hash
            let hash = info.id.to_string();
            // convert our commit to a commit request
            let commit_req = CommitRequest::new_gix(commit);
            // send this commit to our ingestor
            tx.send((hash, CommitishRequest::Commit(commit_req)))?;
            // add this commit to our ingested commits
            commits.insert(info.id);
            // increment our bars length
            bar.inc_length(1);
        }
    }
    Ok(())
}

/// A worker that is ingesting commitishes
pub struct CommitishIngestor {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The progress bars to log progress with
    bar: Bar,
    /// The cmd part of args for this specific worker
    pub cmd: IngestRepos,
    /// The commit maps to be uploading in parallel
    maps: Vec<CommitishMapRequest>,
    /// The futures for the commit maps we are currently uploading
    active: FuturesUnordered<JoinHandle<Result<CommitishMapRequest, (CommitishMapRequest, Error)>>>,
}

impl CommitishIngestor {
    /// Create a new ingestor
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `bar` - The progress bar to update
    /// * `cmd` - The command for ingesting repos
    pub fn new(thorium: &Arc<Thorium>, bar: &Bar, cmd: &IngestRepos) -> Self {
        // start with 10 maps initialized with capacity for 500 commits
        let maps = (0..9)
            .map(|_| CommitishMapRequest {
                groups: cmd.add_groups.clone(),
                earliest: None,
                end: false,
                commitishes: HashMap::with_capacity(500),
            })
            .collect();
        // build our commit ingestor
        CommitishIngestor {
            thorium: thorium.clone(),
            bar: bar.clone(),
            cmd: cmd.clone(),
            maps,
            active: FuturesUnordered::default(),
        }
    }

    /// Get a free map or wait for one to be freed
    async fn get_map(&mut self) -> CommitishMapRequest {
        // if we don't have any maps then wait for a future to finish
        match self.maps.pop() {
            Some(map) => map,
            None => match self.active.next().await {
                Some(Ok(Ok(mut map))) => {
                    // clear our map
                    map.commitishes.clear();
                    // return our cleared map
                    map
                }
                Some(Ok(Err((mut map, error)))) => {
                    // log this error
                    self.bar
                        .error(format!("{}: {}", "Error".bright_red(), error));
                    // clear our map
                    map.commitishes.clear();
                    // return our map
                    map
                }
                Some(Err(error)) => {
                    // log this error
                    self.bar
                        .error(format!("{}: {}", "Error".bright_red(), error));
                    // instance a new map
                    CommitishMapRequest {
                        groups: self.cmd.add_groups.clone(),
                        earliest: None,
                        end: false,
                        commitishes: HashMap::with_capacity(500),
                    }
                }
                None => {
                    // instance a new map
                    CommitishMapRequest {
                        groups: self.cmd.add_groups.clone(),
                        earliest: None,
                        end: false,
                        commitishes: HashMap::with_capacity(500),
                    }
                }
            },
        }
    }

    /// Wait for all currently active futures to complete
    async fn wait_for_all(&mut self) {
        // poll our futures until they are all complete
        while let Some(handle) = self.active.next().await {
            // check if an error occurred
            match handle {
                Ok(Ok(_map)) => (),
                Ok(Err((_map, error))) => {
                    // log this error
                    self.bar.error(error.to_string());
                }
                Err(error) => {
                    // log this error
                    self.bar.error(error.to_string());
                }
            }
        }
    }

    /// Start our commit ingestor worker
    ///
    /// # Arguments
    ///
    /// * `remote` - The remote we are ingesting commits for
    /// * `sha256` - The sha256 of our repo data
    /// * `rx` - The channel to get commits to ingest on
    fn start(
        mut self,
        remote: Arc<String>,
        sha256: Arc<String>,
        rx: Receiver<(String, CommitishRequest)>,
    ) -> Self {
        // get a handle to our current runtime
        let handle = tokio::runtime::Handle::current();
        // get the map to start add commits too
        let mut map = handle.block_on(self.get_map());
        // track the oldest commit we have seen and start with now
        let mut earliest = Utc::now();
        // loop over our commit ingests
        while let Ok((hash, commit_req)) = rx.recv() {
            // get our commit reqs timestamp
            let req_time = commit_req.timestamp();
            // update our oldest timestamp if this is older
            if req_time < earliest {
                // this timestamp is older so update it
                earliest = req_time;
            }
            // add this commit to our commit map
            map.commitishes.insert(hash, commit_req);
            // increment our progress bar
            self.bar.inc(1);
            // check if we have 500 commits buffered
            if map.commitishes.len() >= 500 {
                // get a new map
                let mut new_map = handle.block_on(self.get_map());
                // swap our maps
                std::mem::swap(&mut map, &mut new_map);
                // we have 300 commits in the buffer so spawn a task to send them to Thorium
                let local_client = self.thorium.clone();
                let local_remote = remote.clone();
                let local_sha256 = sha256.clone();
                let task = tokio::spawn(async move {
                    match local_client
                        .repos
                        .add_commits(&local_remote, &local_sha256, &new_map)
                        .await
                    {
                        Ok(_) => Ok(new_map),
                        Err(err) => Err((new_map, err)),
                    }
                });
                self.active.push(task);
            }
        }
        // wait for our current futures to complete
        handle.block_on(self.wait_for_all());
        // close our repo commit upload
        map.end = true;
        // set the oldest timestamp to check for
        map.earliest = Some(earliest);
        // send the final commitish update
        if let Err(error) = handle.block_on(self.thorium.repos.add_commits(&remote, &sha256, &map))
        {
            // log this error
            self.bar
                .error(format!("{}: {}", "Error".bright_red(), error));
        }
        self
    }
}

/// The repo ingest monitor
pub struct RepoIngestMonitor;

impl Monitor for RepoIngestMonitor {
    /// The update type to use
    type Update = ();

    /// build this monitors progress bar
    fn build_bar(multi: &MultiBar, msg: &str) -> Bar {
        multi.add(msg, BarKind::Bound(0))
    }

    /// Apply an update to our global progress bar
    fn apply(bar: &Bar, _: Self::Update) {
        bar.inc(1);
    }
}

// A worker that ingests repos into Thorium
pub struct IngestWorker {
    /// The Thorium client for this worker
    thorium: Arc<Thorium>,
    /// The config used by Thorctl
    pub conf: CtlConf,
    /// The progress bars to log progress with
    bar: Bar,
    /// The cmd part of args for this specific worker
    pub cmd: IngestRepos,
    /// The commit ingestor if we have one
    pub ingestor: Option<CommitishIngestor>,
    /// The channel to send monitor updates on
    pub monitor_tx: AsyncSender<MonitorMsg<RepoIngestMonitor>>,
    /// Whether to use existing thorium data as a cache
    pub no_cache: bool,
}

impl IngestWorker {
    /// Clone a repo from a remote url
    async fn remote_clone(&self, remote_url: &str, path: &PathBuf) -> Result<PathBuf, Error> {
        // update our bars message
        self.bar.set_message("Parsing URL");
        // try to parse our repos url
        let (clone_url, thorium_url, name) = parse_url(&remote_url, &self.conf)?;
        // set that we are trying to clone from Thorium
        self.bar.refresh("Cloning From Thorium", BarKind::Unbound);
        let dl_req = if self.no_cache {
            // if we are skipping then thorium cache then just return an error to
            // skip thoriums cache
            Err(Error::new(""))
        } else {
            // check if we have this repos data already in Thorium
            // try to start downloading this repo
            self.thorium
                .repos
                .download_unpack(&thorium_url, &RepoDownloadOpts::default(), &path)
                .await
        };
        // nest our repo with the right name to the target path
        let nested = path.join(name);
        // if we run into a problem then assume this repo doesn't exist in Thorium
        match dl_req {
            Ok(untarred) => {
                // open our cached repo
                let mut repo = git2::Repository::open(&untarred.path).unwrap();
                // fetch any updates to this repo
                untarred.fetch(&mut repo, &self.conf.git)?;
                // return the path to our repo
                Ok(nested)
            }
            // Assume we do not already have this repo ingested so get it directly
            Err(err) => {
                // we haven't set any groups to submit new repos to and this repo wasn't already submitted,
                // so return an error
                if self.cmd.add_groups.is_empty() {
                    return Err(Error::new(format!(
                        "Repo does not already exist in \
                        Thorium and no add groups were provided! - {err}"
                    )));
                }
                // clone our repo with the right helper
                match self.conf.git.clone() {
                    Some(git_conf) => {
                        clone_repo_ssh(clone_url, nested.clone(), git_conf, &self.bar).await?;
                    }
                    None => clone_repo_https(clone_url, nested.clone(), &self.bar).await?,
                };
                // return the path to our repo
                Ok(nested)
            }
        }
    }

    /// Clone a remote repo if it doesn't already exist
    async fn clone(&self, job: &IngestJob, root: &PathBuf) -> Result<PathBuf, Error> {
        // if this is a remote repo then clone it
        match job {
            IngestJob::Remote(remote_url) => self.remote_clone(remote_url, root).await,
            IngestJob::Local(path) => Ok(path.clone()),
        }
    }

    /// Uploads a repository and all of its commits to Thorium
    ///
    /// # Arguments
    ///
    /// * `repo` - The path to the repo to upload
    /// * `tarred` - The path to tar this repo to
    /// * `references` - The references to crawl commits for
    async fn upload<P: Into<PathBuf>>(
        &mut self,
        repo: P,
        tarred: &Path,
        references: Vec<String>,
    ) -> Result<(), thorium::Error> {
        // set that we are tarring this repository
        self.bar.refresh("Tarring Repository", BarKind::Timer);
        // make sure this is a valid repo
        let untarred = UntarredRepo::new(repo)?;
        // tar this repo up
        let tar = untarred.tar(tarred).await?;
        // get this repos remote url and fix it
        let remote = fix_remote(untarred.remote(None)?);
        // find our default checkout branch
        let default_checkout =
            untarred.find_default_checkout(&self.cmd.preferred_checkout_branches)?;
        // init this repo in Thorium
        let mut req = self.cmd.build_req(&remote, default_checkout);
        // if no add groups were provided, assume the repo was already submitted
        // and try to get the repo's current groups
        let repo_url = if self.cmd.add_groups.is_empty() {
            let mut repo_details = self.thorium.repos.get(&req.url).await.map_err(|err| {
                Error::new(format!(
                    "Repo does not already exist in \
                    Thorium and no add groups were provided! - {err}",
                ))
            })?;
            // set the request's groups to all the repo's groups we retrieved
            req.groups = repo_details.groups_take().into_iter().collect();
            // use the normalized URL key to the repo from the API
            repo_details.url
        } else {
            // upload our zipped repo and retrieve the normalized URL key to the repo from the API
            self.thorium.repos.create(&req).await?.url
        };
        // set that we are uploading this repository
        self.bar.refresh("Uploading Repository", BarKind::Timer);
        // upload the repo
        let resp = self
            .thorium
            .repos
            .upload(&repo_url, tar, self.cmd.add_groups.clone())
            .await?;
        // build a channel for our commit reqs
        let (tx, rx) = kanal::bounded(1000);
        // change our progress bar
        self.bar.refresh("Uploading Commits", BarKind::Unbound);
        // clone our progress bar for our crawlers
        let crawl_bar = self.bar.clone();
        // crawl this repo and upload its commits for all or some refs
        let crawler = if references.is_empty() {
            tokio::task::spawn_blocking(|| crawl_all(untarred, tx, crawl_bar))
        } else {
            tokio::task::spawn_blocking(|| crawl_specific(untarred, tx, references, crawl_bar))
        };
        // wrap our passable items in arcs
        let remote_arc = Arc::new(remote);
        let sha256_arc = Arc::new(resp.sha256);
        // get or instance an ingestor
        let ingestor = self
            .ingestor
            .take()
            .unwrap_or_else(|| CommitishIngestor::new(&self.thorium, &self.bar, &self.cmd));
        // start ingesting these commits
        let ingestor = tokio::task::spawn_blocking(|| ingestor.start(remote_arc, sha256_arc, rx));
        // wait for our crawler and ingestor to finish
        let (crawler_res, ingestor) = tokio::join!(crawler, ingestor);
        // log any crawler errors
        match crawler_res {
            Ok(inner) => {
                if let Err(error) = inner {
                    // log this error
                    self.bar
                        .error(format!("{}: {}", "Error".bright_red(), error));
                }
            }
            Err(error) => {
                // log this error
                self.bar
                    .error(format!("{}: {}", "Error".bright_red(), error));
            }
        }
        // log any ingestor errors and try to recover our ingestor
        match ingestor {
            Ok(ingestor) => self.ingestor = Some(ingestor),
            Err(error) => {
                // log this error
                self.bar
                    .error(format!("{}: {}", "Error".bright_red(), error));
            }
        }
        Ok(())
    }
}

/// Whether is a local or remote repo to ingest
#[derive(Clone, Debug)]
pub enum IngestJob {
    /// Ingest a remote repo by url
    Remote(String),
    /// Ingest an already cloned repo from disk
    Local(PathBuf),
}

impl IngestJob {
    /// Get if this is a remote job
    pub fn is_remote(&self) -> bool {
        match self {
            IngestJob::Remote(_) => true,
            IngestJob::Local(_) => false,
        }
    }
}

impl std::fmt::Display for IngestJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestJob::Remote(remote) => write!(f, "{remote}"),
            IngestJob::Local(remote) => write!(f, "{remote:?}"),
        }
    }
}

#[async_trait::async_trait]
impl Worker for IngestWorker {
    /// The command part of our args for this specific worker
    type Cmd = IngestRepos;

    /// The type of jobs to receive
    type Job = IngestJob;

    /// An update for the repo ingest monitor
    type Monitor = RepoIngestMonitor;

    /// Initialize our worker
    async fn init(
        thorium: &Thorium,
        conf: &CtlConf,
        bar: Bar,
        _args: &Args,
        cmd: Self::Cmd,
        updates: &AsyncSender<MonitorMsg<Self::Monitor>>,
    ) -> Self {
        // build our repo ingest worker
        IngestWorker {
            thorium: Arc::new(thorium.clone()),
            conf: conf.clone(),
            bar,
            cmd: cmd.clone(),
            ingestor: None,
            monitor_tx: updates.clone(),
            no_cache: cmd.no_cache,
        }
    }

    /// Log an info message
    fn info<T: AsRef<str>>(&mut self, msg: T) {
        self.bar.info(msg);
    }

    /// Start claiming and executing jobs
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to start
    async fn execute(&mut self, job: Self::Job) {
        // set this progress bars name
        self.bar.rename(job.to_string());
        // build a unique root target path for this repo
        let root = self.cmd.temp.join(Uuid::new_v4().to_string());
        // Clone or parse the target repo
        let nested = check!(self, self.clone(&job, &root).await, &root);
        // get a list of references to crawl if any were passed
        let refs = self.cmd.references();
        // create a path to tar the repo
        let tarred = nested.with_extension("tar");
        // ingest this repo into Thorium
        check!(self, self.upload(nested, &tarred, refs).await, &root);
        // only clean up local clone data if this is a remote clone
        if job.is_remote() {
            // clean up this repos dir if this is a remote
            if let Err(error) = tokio::fs::remove_dir_all(root).await {
                // log this io error
                self.bar.error(error.to_string());
            }
        }
        // send an update to our monitor
        if let Err(error) = self.monitor_tx.send(MonitorMsg::Update(())).await {
            // log this io error
            self.bar.error(error.to_string());
        }
        // finish our progress bar
        self.bar.finish_and_clear();
    }
}
