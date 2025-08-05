use chrono::NaiveDateTime;
use download::DownloadWorker;
use futures::StreamExt;
use itertools::Itertools;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thorium::models::{
    CommitListOpts, Commitish, CommitishKinds, GenericJobArgs, ReactionRequest, Repo,
    RepoDependencyRequest, TagRequest,
};
use thorium::Thorium;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_stream::wrappers::LinesStream;

mod download;
mod ingest;

use self::ingest::{IngestJob, IngestWorker};

use super::{progress, update, Controller};
use crate::args::repos::{
    CompileRepos, ContributorsRepos, DescribeRepos, DownloadRepos, GetRepos, IngestRepos,
    ListCommits, RepoTarget, Repos, UpdateRepos,
};
use crate::args::{Args, DescribeCommand, SearchParameterized};
use crate::utils;
use crate::{CtlConf, Error};

struct GetReposLine;

impl GetReposLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<56} | {:<20} | {:<38}",
            "REPO URL", "GROUP", "SUBMISSION",
        );
        println!("{:-<57}+{:-<22}+{:-<40}", "", "", "");
    }

    /// Print a repo's info
    ///
    /// # Arguments
    ///
    /// * `repo` - The repo to print
    pub fn print_repo(repo: &Repo) {
        for sub in &repo.submissions {
            for group in &sub.groups {
                println!("{:<56} | {:<20} | {}", repo.url, group, sub.id,);
            }
        }
    }
}

struct CommitishLine;

impl CommitishLine {
    /// Print this log line's header
    pub fn header() {
        println!("{:<6} | {:<40} | {:<20}", "KIND", "COMMITISH", "TIMESTAMP",);
        println!("{:-<7}+{:-<42}+{:-<22}", "", "", "");
    }

    /// Print a commitish's info
    ///
    /// # Arguments
    ///
    /// * `commitish` - The commitish to print
    pub fn print_commitish(commitish: &Commitish) {
        println!(
            "{:<6} | {:<40} | {}",
            // get the raw str for kind because Display seems to override margin settings
            commitish.kind().as_str(),
            commitish.key(),
            commitish.timestamp()
        );
    }
}

/// Get repo's info from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The repo get command to execute
async fn get(thorium: Thorium, cmd: &GetRepos) -> Result<(), Error> {
    GetReposLine::header();
    let opts = cmd.build_repo_opts()?;
    // get repos cursor
    let mut repos_cursor = thorium.repos.list_details(&opts).await?;
    // allocate a Vec for storing repos to be alphabetized
    let mut alpha_repos: Vec<Repo> = Vec::new();
    // print the repos from the cursor in alphabetical order
    loop {
        if cmd.alpha {
            // save the repos to be sorted later
            alpha_repos.append(&mut repos_cursor.data);
        } else {
            // otherwise just print immediately
            repos_cursor.data.iter().for_each(GetReposLine::print_repo);
        }
        if repos_cursor.exhausted() {
            break;
        }
        repos_cursor.refill().await?;
    }
    if cmd.alpha {
        // sort then print the repos if they should be alphabetized
        alpha_repos
            .iter()
            .sorted_unstable_by(|a, b| Ord::cmp(&a.url, &b.url))
            .for_each(GetReposLine::print_repo);
    }
    Ok(())
}

/// Describe repos by displaying/saving all of their JSON-formatted details
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The [`DescribeRepos`] command to execute
async fn describe(thorium: Thorium, cmd: &DescribeRepos) -> Result<(), Error> {
    cmd.describe(&thorium).await
}

/// List a repo's commitishes
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The list command to execute
async fn list_commits(thorium: Thorium, cmd: &ListCommits) -> Result<(), Error> {
    // try to buil commit list opts
    let opts = cmd.build_commit_opts()?;
    // get our cursor
    let mut cursor = thorium.repos.list_commitishes(&cmd.repo, &opts).await?;
    // print the header
    CommitishLine::header();
    loop {
        // print all commitishes
        cursor.data.iter().for_each(CommitishLine::print_commitish);
        // exit the loop if we're out of data
        if cursor.exhausted() {
            break;
        }
        // try to get more data
        cursor.refill().await?;
    }
    Ok(())
}

/// Ingest repos from a list of positional args
///
/// # Arguments
///
/// * `controller` - The controller for our workers
/// * `cmd` - The command to use to ingest repos
/// * `added` - A set of repos that we have already added
async fn ingest_positionals(
    controller: &mut Controller<IngestWorker>,
    cmd: &IngestRepos,
    added: &mut HashSet<String>,
) -> Result<(), Error> {
    // crawl over the urls passed as positional args
    for repo in &cmd.urls {
        // skip any repos already in our set
        if added.insert(repo.clone()) {
            // wrap this repo url in a Remote job
            let job = IngestJob::Remote(repo.to_owned());
            // add this repo to our jobs queue
            if let Err(error) = controller.add_job(job).await {
                // log this error
                controller.error(&error.to_string());
            }
        }
    }
    Ok(())
}

/// Ingest repos from files with new line separated lists
///
/// # Arguments
///
/// * `controller` - The controller for our workers
/// * `cmd` - The command to use to ingest repos
/// * `added` - A set of repos that we have already added
async fn ingest_files_lists(
    controller: &mut Controller<IngestWorker>,
    cmd: &IngestRepos,
    added: &mut HashSet<String>,
) -> Result<(), Error> {
    // read our files line by line
    for path in &cmd.repos_list {
        // open this file
        let file = tokio::fs::File::open(path).await?;
        // open this file in a bufreader
        let reader = tokio::io::BufReader::new(file);
        // build an iterator over this files lines
        let mut line_stream = LinesStream::new(reader.lines());
        // consume our line stream
        while let Some(line) = line_stream.next().await {
            // log that we failed to get this url
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    // log this error
                    controller.error(&error.to_string());
                    // skip to the next line
                    continue;
                }
            };
            // skip any repos already in our set
            if added.insert(line.clone()) {
                // wrap this repo url in a Remote job
                let job = IngestJob::Remote(line.to_owned());
                // add this line to our jobs queue
                if let Err(error) = controller.add_job(job).await {
                    // log this error
                    controller.error(&error.to_string());
                }
            }
        }
    }
    Ok(())
}

/// Ingest local repos from a list of positional args
///
/// # Arguments
///
/// * `controller` - The controller for our workers
/// * `cmd` - The command to use to ingest repos
/// * `added` - A set of repos that we have already added
async fn ingest_local_positionals(
    controller: &mut Controller<IngestWorker>,
    cmd: &IngestRepos,
    added: &mut HashSet<PathBuf>,
) -> Result<(), Error> {
    // crawl over the urls passed as positional args
    for repo in &cmd.local {
        // skip any repos already in our set
        if added.insert(repo.clone()) {
            // wrap this repo url in a Local job
            let job = IngestJob::Local(repo.to_owned());
            // add this repo to our jobs queue
            if let Err(error) = controller.add_job(job).await {
                // log this error
                controller.error(&error.to_string());
            }
        }
    }
    Ok(())
}

/// Ingest these repos by url
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use to ingest repos
/// * `args` - The args to Thorctl
/// * `conf` - The config for Thorctl
async fn ingest(
    thorium: Arc<Thorium>,
    cmd: &IngestRepos,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // create a new worker controller
    let mut controller = Controller::<IngestWorker>::spawn(
        "Ingesting Repos",
        &thorium,
        args.workers,
        conf,
        args,
        cmd,
    )
    .await;
    // track and remove any duplicate repos as those can cause hangs
    let mut added = HashSet::with_capacity(100);
    // ingest any remote repos
    // ingest any repos passed with positional args
    ingest_positionals(&mut controller, cmd, &mut added).await?;
    // ingest any repos from any lists in files
    ingest_files_lists(&mut controller, cmd, &mut added).await?;
    // track and remove any duplicate local repos as its wasteful to ingest them more than once
    let mut local_added = HashSet::with_capacity(100);
    // Add any locally cloned repos
    ingest_local_positionals(&mut controller, cmd, &mut local_added).await?;
    // wait for all our workers to complete
    controller.finish().await?;
    Ok(())
}

/// Update some already ingested repos
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use to update ingested repos
/// * `args` - The args to Thorctl
/// * `conf` - The config for Thorctl
async fn update(
    thorium: Arc<Thorium>,
    cmd: &UpdateRepos,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // build the ingest command for this update repos command
    let ingest_cmd = IngestRepos::from(cmd);
    // create a new worker controller
    let mut controller = Controller::<IngestWorker>::spawn(
        "Updating Repos",
        &thorium,
        args.workers,
        conf,
        args,
        &ingest_cmd,
    )
    .await;
    // track and remove any duplicate repos as those can cause hangs
    let mut added = HashSet::with_capacity(100);
    // ingest any repos passed with positional args
    ingest_positionals(&mut controller, &ingest_cmd, &mut added).await?;
    // ingest any repos from any lists in files
    ingest_files_lists(&mut controller, &ingest_cmd, &mut added).await?;
    // make sure we had at least some targets
    cmd.validate_search()?;
    // if we have any search params then  also crawl and update those
    if cmd.has_parameters() || cmd.apply_to_all() {
        // build the options for listing repos
        let opts = cmd.build_repo_opts()?;
        // get repos cursor
        let mut repos_cursor = thorium.repos.list(&opts).await?;
        // print the repos from the cursor in alphabetical order
        loop {
            // crawl over the repos and add them to our update queue
            for line in repos_cursor.data.drain(..) {
                // skip any repos we already added
                if added.insert(line.url.clone()) {
                    // all update repos will be a remote job
                    let job = IngestJob::Remote(line.url);
                    // add this repo to our jobs queue
                    if let Err(error) = controller.add_job(job).await {
                        // log this error
                        controller.error(&error.to_string());
                    }
                }
            }
            // if this cursor has been exhausted then stop crawling repos
            if repos_cursor.exhausted() {
                break;
            }
            // refill our cursors data
            if let Err(error) = repos_cursor.refill().await {
                // log this error and then stop listing data
                controller
                    .multi
                    .error(&format!("Error listing repos: {error:?}"))
                    .unwrap_or_else(|_| panic!("Failed to log error: {error:#?}"));
            }
        }
    }
    // wait for all our workers to complete
    controller.finish().await?;
    Ok(())
}

/// Download repos from Thorium
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use to download repos
async fn download(
    thorium: &Thorium,
    cmd: &DownloadRepos,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // create a new worker controller
    let mut controller = Controller::<DownloadWorker>::spawn(
        "Downloading Repos",
        thorium,
        args.workers,
        conf,
        args,
        cmd,
    )
    .await;
    // build the opts for listing repos
    let list_opts = cmd.build_repo_opts()?;
    // track and remove any duplicate repos as those can cause hangs
    let mut added = HashSet::with_capacity(100);
    // add any repos in our command
    for repo in &cmd.repos {
        // turn this repo into a repo target
        match RepoTarget::try_from(repo) {
            Ok(target) => {
                // check if this repo has already been added to be downloaded
                if added.insert(target.url.clone()) {
                    // try to add this download job
                    if let Err(error) = controller.add_job(target).await {
                        // log this error
                        controller.error(&error.to_string());
                    }
                }
            }
            // log this error
            Err(error) => controller.error(&error.to_string()),
        }
    }
    // add any repos from our file lists
    for file in &cmd.files {
        // open this file
        let file = File::open(file).await?;
        // wrap our file in a buf reader
        let reader = BufReader::new(file);
        // get a stream of lines from this file
        let mut stream = reader.lines();
        // read the repo targets from this file line by line
        while let Some(repo) = stream.next_line().await? {
            // build the repo target for this repo
            let target = RepoTarget::new(repo);
            // try to add this download job
            if let Err(error) = controller.add_job(target).await {
                // log this error
                controller.error(&error.to_string());
            }
        }
    }
    // skip listing repos if targets were specified
    if cmd.repos.is_empty() && cmd.files.is_empty() {
        // get a repos cursor
        let mut repos_cursor = thorium.repos.list(&list_opts).await?;
        // step over all repos and download them
        loop {
            // otherwise just print immediately
            for repo in repos_cursor.data.drain(..) {
                // check if this repo has already been added to be downloaded
                if added.insert(repo.url.clone()) {
                    // build the repo target for this repo
                    let target = RepoTarget::new(repo.url);
                    // try to add this download job
                    if let Err(error) = controller.add_job(target).await {
                        // log this error
                        controller.error(&error.to_string());
                    }
                }
            }
            // check if this cursor has been exhausted
            if repos_cursor.exhausted() {
                break;
            }
            repos_cursor.refill().await?;
        }
    }
    // wait for all our workers to complete
    controller.finish().await?;
    Ok(())
}

macro_rules! add_vec {
    ($args:expr, $key:expr, $vec:expr) => {
        // if this vec is not empty then add it as a kwarg
        if !$vec.is_empty() {
            $args.kwarg($key, $vec)
        } else {
            $args
        }
    };
}

macro_rules! add_option {
    ($args:expr, $key:expr, $opt:expr) => {
        // if this vec is not empty then add it as a kwarg
        if let Some(value) = $opt {
            $args.kwarg($key, vec![value])
        } else {
            $args
        }
    };
}

/// The options for a single repo build
#[derive(Deserialize, Debug)]
pub struct RepoBuild {
    /// The repo url to compile
    pub repo: String,
    /// The branch, commit, or tag to compile
    pub commitish: Option<String>,
    /// The kind of commitish to use
    pub kind: Option<CommitishKinds>,
    /// Any dependencies that need to be installed before building this repo
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Any flags to set when building this repo
    #[serde(default)]
    pub flags: Vec<String>,
    /// The C compiler to force
    #[serde(default)]
    pub cc: Option<String>,
    /// The c++ compiler to force
    #[serde(default)]
    pub cxx: Option<String>,
    /// The tags to apply to any compiled blobs
    #[serde(default)]
    pub tags: HashMap<String, Vec<String>>,
}

impl RepoBuild {
    /// Load a list of repo builds from a file
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Vec<RepoBuild>, Error> {
        // read in our repo build file as a string
        let raw = tokio::fs::read_to_string(&path).await?;
        // deserialize our list
        let build_list = serde_json::from_str(&raw)?;
        Ok(build_list)
    }

    /// Convert this repo build to a reaction request
    pub fn to_request(self, batch: &Option<String>) -> ReactionRequest {
        // build the args for the builder job
        let build_args = GenericJobArgs::default();
        // if any dependencies or flags  were specified then add them
        let build_args = add_vec!(build_args, "--dependencies", self.dependencies);
        let mut build_args = add_vec!(build_args, "--flags", self.flags);
        // build a list of all the tag args to specify
        let mut combined = Vec::with_capacity(self.tags.len());
        // add the built tags to this reaction
        if !self.tags.is_empty() {
            for (key, values) in self.tags {
                // build the combined values for each tag arg
                for value in values {
                    combined.push(format!("{key}={value}"));
                }
            }
            build_args.kwargs.insert("--tags".to_owned(), combined);
        }
        // add the compiler flags for this reaction
        let build_args = add_option!(build_args, "--CC", self.cc);
        let build_args = add_option!(build_args, "--CXX", self.cxx);
        let build_args = add_option!(build_args, "--tags", batch);
        // build the repo dependency request
        let repo = RepoDependencyRequest {
            url: self.repo,
            commitish: self.commitish,
            kind: self.kind,
        };
        // build our base reaction request
        let mut req = ReactionRequest::new("builder", "classifier")
            .args("classifier", build_args)
            .repo(repo);
        // if we have a batch name then inject it
        if let Some(batch) = batch {
            req = req.tag(batch);
        }
        req
    }
}

/// Load a list of repos from a file and compile them
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use to compile repos
async fn compile(thorium: Thorium, cmd: &CompileRepos) -> Result<(), Error> {
    // try to create the list of repos to build
    let builds = cmd.get_builds().await?;
    // get our batch name if needed
    let batch = cmd.batch_name();
    // convert our builds to bundles of reaction requests
    let reqs = builds
        .into_iter()
        .map(|item| item.to_request(&batch))
        .collect::<Vec<ReactionRequest>>();
    // send our requests 100 at a time
    for chunk in reqs.chunks(100) {
        // create this chunk of reactions
        thorium.reactions.create_bulk(chunk).await?;
    }
    // if we want to watch all our jobs complete then do that
    if cmd.watch {
        // make sure we have a batch name
        if let Some(batch) = batch {
            // close our create print
            // put an empty line of space between our create prints and our watch log
            println!("\n\tWATCHING REACTIONS\t\n");
            super::reactions::create::watch(
                thorium,
                &batch,
                std::iter::once("builder".to_string()),
            )
            .await?;
        }
    }
    Ok(())
}

/// prints out a single contributor and their commit count
macro_rules! contributor_print {
    ($author:expr, $count:expr) => {
        println!("{:<64} | {:<12} ", $author, $count)
    };
}

/// A single line for uploading a repo commit
struct ContributorLine;

impl ContributorLine {
    /// Print this log lines header
    pub fn header() {
        println!("{:64} | {:<12}", "CONTRIBUTOR", "COMMITS");
        println!("{:-<65}+{:-<14}", "", "");
    }

    /// Print a log line for a repo contributor
    ///
    /// # Arguments
    ///
    /// * `author` - The author that contributed these commits
    /// * `count` - The number of commits this author contributed
    pub fn line(author: &str, count: u64) {
        // log this line
        contributor_print!(author, count);
    }
}

/// Crawl our commits and build a map of contributor counts
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use when getting repo contributors
async fn map_contributors(
    thorium: &Thorium,
    cmd: &ContributorsRepos,
) -> Result<HashMap<String, u64>, Error> {
    // build a map to store contributor info
    let mut contributors = HashMap::with_capacity(100);
    // build the options for listing commits
    let opts = CommitListOpts::default().page_size(1000);
    // build a cursor to crawl this repos commits
    let mut cursor = thorium
        .repos
        .list_commitish_details(&cmd.repo, &opts)
        .await?;
    // crawl our cursor
    loop {
        // crawl the commits on this page
        for commitish in cursor.data.drain(..) {
            // get this commitishes author if one can be retrieved
            if let Some(author) = commitish.author_owned() {
                // get an entry to this contributors number of commits
                let entry = contributors.entry(author).or_insert(0);
                // increment this contributors number of commits
                *entry += 1;
            }
        }
        // check if this cursor has been exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of data
        cursor.refill().await?;
    }
    Ok(contributors)
}

/// Tag contributors of a repo
async fn tag_contributors(
    thorium: &Thorium,
    cmd: &ContributorsRepos,
    contributors: &HashMap<String, u64>,
) -> Result<(), Error> {
    // build a vec of all the contributors names
    let names = contributors
        .iter()
        .map(|(name, _)| name.to_owned())
        .collect();
    // build our tag request
    let tag_req = TagRequest::default().add_values("Contributor", names);
    // add our contributor tags to our repo
    thorium.repos.tag(&cmd.repo, &tag_req).await?;
    Ok(())
}

/// List the contributors for a repo
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `cmd` - The command to use when getting repo contributors
async fn contributors(thorium: &Thorium, cmd: &ContributorsRepos) -> Result<(), Error> {
    // get a map of contributors
    let contributors = map_contributors(thorium, cmd).await?;
    // tag our contributors if its enabled
    if cmd.tag {
        // add our contributor tags
        tag_contributors(thorium, cmd, &contributors).await?;
    }
    // print our contributor header
    ContributorLine::header();
    // crawl over and print our contributors
    for (author, count) in &contributors {
        ContributorLine::line(author, *count);
    }
    Ok(())
}

/// Handle all repos commands or print repo info
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The repos command to execute
pub async fn handle(args: &Args, cmd: &Repos) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        update::ask_update(&thorium).await?;
    }
    // call the right repos handler
    match cmd {
        Repos::Get(cmd) => get(thorium, cmd).await,
        Repos::Describe(cmd) => describe(thorium, cmd).await,
        Repos::Commits(cmd) => list_commits(thorium, cmd).await,
        Repos::Ingest(cmd) => ingest(Arc::new(thorium), cmd, args, &conf).await,
        Repos::Update(cmd) => update(Arc::new(thorium), cmd, args, &conf).await,
        Repos::Download(cmd) => download(&thorium, cmd, args, &conf).await,
        Repos::Compile(cmd) => compile(thorium, cmd).await,
        Repos::Contributors(cmd) => contributors(&thorium, cmd).await,
    }
}

impl ListCommits {
    /// Build [`CommitListOpts`] from a `ListCommits` command
    fn build_commit_opts(&self) -> Result<CommitListOpts, Error> {
        Ok(CommitListOpts {
            cursor: self.cursor,
            start: self
                .start
                .as_ref()
                .map(|start| {
                    NaiveDateTime::parse_from_str(start, &self.date_fmt).map(|t| t.and_utc())
                })
                .transpose()?,
            end: self
                .end
                .as_ref()
                .map(|end| NaiveDateTime::parse_from_str(end, &self.date_fmt).map(|t| t.and_utc()))
                .transpose()?,
            page_size: self.page_size,
            limit: (!self.no_limit).then_some(self.limit),
            groups: self.groups.clone(),
            // API defaults to all, so no need to handle cases where no kinds are passed
            kinds: self.kinds.clone(),
        })
    }
}
