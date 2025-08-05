//! Handle repo interactions with the backend

use axum::http::StatusCode;
use chrono::prelude::*;
use futures::stream::{self, StreamExt};
use itertools::Itertools;
use scylla::response::query_result::QueryResult;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BTreeMap, HashMap, HashSet};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::{ExistsCursor, ScyllaCursor};
use crate::models::backends::TagSupport;
use crate::models::{
    Commitish, CommitishDetails, CommitishKinds, CommitishListParams, CommitishMapRequest, Repo,
    RepoCheckout, RepoListLine, RepoListParams, RepoRequest, RepoRow, RepoScheme, RepoSubmission,
    RepoSubmissionChunk, RepoUrlComponents, TagRequest, TagType, User,
};
use crate::utils::{helpers, ApiError, Shared};
use crate::{
    bad, internal_err, log_scylla_err, not_found, same_vec, serialize, serialize_opt, unauthorized,
};

/// Check if a user already has a matching repo submission
///
/// # Arguments
///
/// * `user` - The user to check for submissions for
/// * `req` - The repo to save to the backend
/// * `scheme` - The scheme used to submit this repo
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::submission_exists", skip_all, err(Debug))]
pub async fn submission_exists(
    user: &User,
    req: &RepoRequest,
    scheme: &RepoScheme,
    shared: &Shared,
) -> Result<Option<Repo>, ApiError> {
    // try to get the base repo this submission is for
    match get_base(&req.groups, user, &req.url, shared).await? {
        // this repo already exists in Thorium check its submission
        Some(repo) => {
            // check our existing submissions
            if repo.submissions.iter().any(|sub| {
                sub.creator == user.username
                    && &sub.scheme == scheme
                    && same_vec!(&sub.groups, &req.groups)
            }) {
                // we have a matching submission so return that
                event!(Level::INFO, exists = true, matching = true);
                Ok(Some(repo))
            } else {
                // this repo exists but we don't have a matching submission
                // so we will have to pull it again anyways
                event!(Level::INFO, exists = true, matching = false);
                Ok(None)
            }
        }
        // this repo doesn't exist so this must be the first submission
        None => {
            event!(Level::INFO, exists = false, matching = false);
            Ok(None)
        }
    }
}

/// Update all existing submissions default checkout behavior for a specific repo
#[instrument(
    name = "db::repos::update_default_checkout",
    skip(repo, shared),
    err(Debug)
)]
async fn update_default_checkout(
    groups: &[String],
    checkout: &RepoCheckout,
    repo: &mut Repo,
    shared: &Shared,
) -> Result<(), ApiError> {
    // serialize our default checkout
    let default_checkout = serialize!(&checkout);
    // update each of this repos submissions
    for submission in &repo.submissions {
        // get this submissions year
        let year = submission.uploaded.year();
        // get the partition size for repos
        let chunk_size = shared.config.thorium.repos.partition_size;
        // get the partition to write this repo off too
        let bucket = helpers::partition(submission.uploaded, year, chunk_size);
        // update our default checkout behavior for all the groups this repo submission is in
        for group in repo.groups().iter().filter(|group| groups.contains(group)) {
            // log that we are updating this submissions default checkout for a specific group
            event!(
                Level::INFO,
                group = group,
                year,
                bucket,
                uploaded = submission.uploaded.to_string(),
                id = submission.id.to_string()
            );
            // update this repos submissions
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.repos.update_default_checkout,
                    (
                        &default_checkout,
                        group,
                        year,
                        bucket,
                        submission.uploaded,
                        submission.id,
                    ),
                )
                .await?;
        }
    }
    // update our default checkout behavior
    repo.default_checkout = Some(checkout.to_owned());
    Ok(())
}

/// Update any repo submissions earliest that are younger or don't exist
#[instrument(name = "db::repos::update_earliest", skip_all, err(Debug))]
async fn update_earliest(
    groups: &[String],
    earliest: DateTime<Utc>,
    repo: &mut Repo,
    shared: &Shared,
) -> Result<(), ApiError> {
    // check every submission for this repo
    for submission in &mut repo.submissions {
        // update each group in this submission
        // filter down to just groups we updated commitishes in
        for group in submission
            .groups
            .iter()
            .filter(|group| groups.contains(group))
        {
            // check if we have an existing earliest set
            if let Some(existing) = submission.earliest {
                // if our submissions earliest is older then this one then don't update it
                if existing < earliest {
                    event!(Level::INFO, msg = "new earliest is older then current",);
                    continue;
                }
            }
            // get this submissions year
            let year = submission.uploaded.year();
            // get the partition size for repos
            let chunk_size = shared.config.thorium.repos.partition_size;
            // get the partition to write this repo off too
            let bucket = helpers::partition(submission.uploaded, year, chunk_size);
            // we either don't already have an earliest set or this earliest is older
            // update this submission with our new earliest
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.repos.update_earliest,
                    (
                        earliest,
                        group,
                        year,
                        bucket,
                        submission.uploaded,
                        submission.id,
                    ),
                )
                .await?;
            // update this submissions earliest
            submission.earliest = Some(earliest);
        }
    }
    // update our repos earliest if needed
    match (repo.earliest.as_mut(), &earliest) {
        (Some(old), new) if new < old => *old = *new,
        (None, new) => repo.earliest = Some(*new),
        _ => (),
    };
    Ok(())
}

/// Save a repo to scylla
///
/// # Arguments
///
/// * `user` - The user who is saving this repo
/// * `req` - The repo to save to the backend
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::create", skip_all, err(Debug))]
pub async fn create(user: &User, req: RepoRequest, shared: &Shared) -> Result<String, ApiError> {
    // log the repo url we are creating
    event!(Level::INFO, repo = &req.url);
    // extract the required info from this repos url
    let url_components = RepoUrlComponents::parse(&req.url)?;
    // check if this user has already submitted this repo to these groups
    let repo = match submission_exists(user, &req, &url_components.scheme, shared).await? {
        // this submission already exists so just return our repo object
        Some(mut repo) => {
            event!(Level::INFO, msg = "Existing repo submission found");
            if let Some(default_checkout) = &req.default_checkout {
                // check if this request has an updated default checkout behavior
                if req.default_checkout != repo.default_checkout {
                    // this repo has a different default checkout so update it
                    update_default_checkout(&req.groups, default_checkout, &mut repo, shared)
                        .await?;
                }
            }
            repo
        }
        // this submission doesn't exist so create it and get an updated repo objects
        None => {
            // generate a unique id for new repo submissions
            let id = Uuid::new_v4();
            // get the current timestamp for this repos submission and its year and month
            let now = Utc::now();
            let year = now.year();
            // get the partition size for repos
            let chunk_size = shared.config.thorium.repos.partition_size;
            // get the partition to write this repo off too
            let bucket = helpers::partition(now, year, chunk_size);
            // serialize our scheme
            let scheme_raw = serialize!(&url_components.scheme);
            // serialize our default checkout behavior if we have one
            let default_checkout = serialize_opt!(&req.default_checkout);
            // get the current timestamp
            let now = Utc::now();
            // save this repo to scylla
            for group in &req.groups {
                event!(Level::INFO, msg = "Creating new submission", group = group);
                // save this repos submission to this group in scylla
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.repos.insert,
                        (
                            group,
                            year,
                            bucket,
                            now,
                            id,
                            // Reduce the url only to it's components: "provider/user/name"
                            &url_components.get_url(),
                            &url_components.provider,
                            &url_components.user,
                            &url_components.name,
                            &user.username,
                            &scheme_raw,
                            &default_checkout,
                        ),
                    )
                    .await?;
            }
            // create a list to store our keys
            let mut keys = Vec::with_capacity(req.groups.len());
            // build the keys to update
            super::keys::repos::census_keys(&mut keys, &req.groups, year, bucket, shared);
            // update our census info for these commitishes
            super::census::incr_cache(keys, shared).await?;
            // create a submission chunk based on what we just submitted
            let submission = RepoSubmissionChunk {
                groups: req.groups.clone(),
                id,
                creator: user.username.clone(),
                uploaded: now,
                scheme: url_components.scheme.clone(),
                earliest: None,
            };
            // attempt to retrieve this repo from the backend, verifying that it has the info we just saved
            get_updated_repo(
                user,
                submission,
                url_components,
                req.default_checkout,
                shared,
            )
            .await?
        }
    };
    // get the earliest time this repo was uploaded for each group
    let earliest = repo.earliest();
    // build a tag request
    let mut tag_req = TagRequest::<Repo>::default().groups(req.groups.clone());
    // move our tags over to our tag request
    tag_req.tags = req.tags;
    // save our files tags to scylla
    super::tags::create(user, repo.url.clone(), tag_req, &earliest, shared).await?;
    Ok(repo.url)
}

/// Retrieves a recently created/updated repo from the backend, verifying that
/// it has the info that was just added. If the repo cannot be immediately found,
/// a Repo with empty tags is constructed manually from the information provided.
///
/// # Arguments
///
/// * `user` - The user that created the repo submission in Thorium
/// * `submission` - The repo submission that was just created
/// * `url_components` - The components of the repo URL
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::get_updated_repo", skip_all, err(Debug))]
async fn get_updated_repo(
    user: &User,
    submission: RepoSubmissionChunk,
    url_components: RepoUrlComponents,
    default_checkout: Option<RepoCheckout>,
    shared: &Shared,
) -> Result<Repo, ApiError> {
    let url = url_components.get_url();
    match get(&user.groups, user, &url, shared).await {
        Ok(mut repo) => {
            // verify the repo has our recent submission and if not, add it
            if !repo.submissions.contains(&submission) {
                repo.submissions.push(submission);
            }
            Ok(repo)
        }
        Err(err) => {
            // if we get a NOT FOUND error retrieving the repo,
            // assume it hasn't been saved to the db yet and create it manually
            if err.code == StatusCode::NOT_FOUND {
                Ok(Repo {
                    provider: url_components.provider,
                    user: url_components.user,
                    name: url_components.name,
                    url,
                    tags: HashMap::default(),
                    default_checkout,
                    earliest: submission.earliest.clone(),
                    submissions: vec![submission],
                })
            } else {
                Err(err)
            }
        }
    }
}

/// Authorizes that a user has access to a list of repos
///
/// # Arguments
///
/// * `groups` - The groups the user is in
/// * `repos` - The repos to authorize we have access to
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::authorize", skip(shared), err(Debug))]
pub async fn authorize(
    groups: &Vec<String>,
    repos: &Vec<String>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // if we specified no groups then we do not have access to this repo
    if groups.is_empty() {
        return unauthorized!();
    }
    // check if our cartesian product will be under 100
    if groups.len() * repos.len() < 100 {
        // our cartesian product will be low enough so just query for them all at once
        // get the number of tools or default to 10
        // check if any of our groups have access to this repo
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.repos.auth, (repos, groups))
            .await?;
        // cast this query to a rows query
        let query_rows = query.into_rows_result()?;
        //  make sure we got back the same number of repos
        if query_rows.rows_num() != repos.len() {
            return unauthorized!();
        }
    } else {
        // track the repos we have authed
        let mut authed = HashSet::with_capacity(repos.len());
        // build an iter over the cartesian product of our groups and repos
        // doing this in a buffered_unordered closure would probably be faster
        // but that runs into lifetime errors currently :(
        for (groups_chunk, repos_chunk) in groups[..]
            .chunks(50)
            .cartesian_product(repos[..].chunks(50))
        {
            // send a query to check if we have access to these repos
            let query = shared
                .scylla
                .session
                .execute_unpaged(&shared.scylla.prep.repos.auth, (repos_chunk, groups_chunk))
                .await?;
            // cast this query to a rows query
            let query_rows = query.into_rows_result()?;
            // cast our rows to the right type
            for typed_row in query_rows.rows::<(String,)>()? {
                // check if we failed to cast this row
                let (repo,) = typed_row?;
                // add this repo to our authed repos
                authed.insert(repo);
            }
            // if we have authed all of our repos then return early
            if authed.len() == repos.len() {
                // we have authed all of the repos we wanted to auth against
                return Ok(());
            }
        }
    }
    // we have access to this repo
    Ok(())
}

/// Gets a base repo object without its tags
#[instrument(name = "db::repos::get_base", skip(user, shared), err(Debug))]
async fn get_base(
    groups: &Vec<String>,
    user: &User,
    url: &str,
    shared: &Shared,
) -> Result<Option<Repo>, ApiError> {
    // build a btree to sort our submissions
    let mut sorted: BTreeMap<DateTime<Utc>, Vec<RepoSubmission>> = BTreeMap::default();
    // if we have more than 100 groups then chunk it into bathes of 100  otherwise just get our info
    if groups.len() > 100 {
        // break our groups into chunks of 100
        for chunk in groups.chunks(100) {
            // turn our group chunk into a vec
            let chunk_vec = chunk.to_vec();
            // get this chunks data
            let query = shared
                .scylla
                .session
                .execute_unpaged(&shared.scylla.prep.repos.get, (url, chunk_vec))
                .await?;
            // cast this query to a rows query
            let query_rows = query.into_rows_result()?;
            // set the type for the rows returned by this query
            let typed_iter = query_rows.rows::<RepoRow>()?;
            // cast our rows into submission objects and add them to our btree
            typed_iter
                .filter_map(|res| log_scylla_err!(res))
                .for_each(|sub| {
                    // get an entry to the list for this timestamp
                    let entry = sorted.entry(sub.uploaded).or_default();
                    // crawl the entries for this timestamp and dedupe entries
                    if let Some(pos) = entry.iter().position(|item| sub.id == item.id) {
                        entry[pos].groups.push(sub.group);
                    } else {
                        // add this new submission to this timestamps list
                        match RepoSubmission::try_from((sub, user)) {
                            Ok(sub) => entry.push(sub),
                            Err(error) => event!(Level::ERROR, msg = &error.msg),
                        }
                    }
                });
        }
    } else {
        // get all our data in one go
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.repos.get, (url, groups))
            .await?;
        // cast this query to a rows query
        let query_rows = query.into_rows_result()?;
        // set the type for the rows returned by this query
        let typed_iter = query_rows.rows::<RepoRow>()?;
        // cast our rows into submission objects and add them to our btree
        typed_iter
            .filter_map(|res| log_scylla_err!(res))
            .for_each(|sub| {
                // get an entry to the list for this timestamp
                let entry = sorted.entry(sub.uploaded).or_default();
                // crawl the entries for this timestamp and dedupe entries
                if let Some(pos) = entry.iter().position(|item| sub.id == item.id) {
                    entry[pos].groups.push(sub.group);
                } else {
                    // add this new submission to this timestamps list
                    match RepoSubmission::try_from((sub, user)) {
                        Ok(sub) => entry.push(sub),
                        Err(error) => event!(Level::ERROR, msg = &error.msg),
                    }
                }
            });
    }
    // get an iter of these submissions in descending order
    let mut descending = sorted.into_iter().rev().flat_map(|(_, sub)| sub);
    // get the first submission object and cast it to a sample
    if let Some(sub) = descending.next() {
        // cast this submission object to a repo
        let mut repo = Repo::from(sub);
        // crawl the remaining submission objects and add them
        descending.for_each(|sub| repo.add(sub));
        // return our repo
        return Ok(Some(repo));
    };
    Ok(None)
}

/// Gets info about a specific repo with its tags
///
/// # Arguments
///
/// * `groups` - The groups to look for this repo in
/// * `user` - The user that is getting this repo
/// * `url` - The url for this repo
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::get", skip(user, shared), err(Debug))]
pub async fn get(
    groups: &Vec<String>,
    user: &User,
    url: &str,
    shared: &Shared,
) -> Result<Repo, ApiError> {
    // get our base repo object without tags
    match get_base(groups, user, url, shared).await? {
        Some(mut repo) => {
            // get the tags for this repo
            super::tags::get(TagType::Repos, groups, url, &mut repo.tags, shared).await?;
            // return our repo
            Ok(repo)
        }
        None => not_found!(format!("repo {} not found", url)),
    }
}

/// Get the details for specific repos
///
/// # Arguments
///
/// * `groups` - The groups to search in
/// * `repos` - The repos to retrieve submission docs for
/// * `shared` - Shared Thorium objects
/// * `req_id` - The uuid for this request
#[instrument(name = "db::repos::list_details", skip(user, shared), err(Debug))]
pub async fn list_details(
    groups: &Vec<String>,
    repos: Vec<String>,
    user: &User,
    shared: &Shared,
) -> Result<Vec<Repo>, ApiError> {
    // build a btreemap to store the order repos should be returned in
    let mut sorted: BTreeMap<DateTime<Utc>, Vec<String>> = BTreeMap::default();
    // build a hashmap to store our repo submissions in
    let mut map = HashMap::with_capacity(repos.len());
    // split our groups and repos into chunks of 50
    for (repos_chunk, groups_chunk) in repos.chunks(50).cartesian_product(groups.chunks(50)) {
        // turn our chunks into vecs
        let groups_vec = groups_chunk.to_vec();
        let repos_vec = repos_chunk.to_vec();
        // send a query to get this chunks data
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.repos.get_many, (repos_vec, groups_vec))
            .await?;
        // cast this query to a rows query
        let query_rows = query.into_rows_result()?;
        // crawl the rows returned by this query
        for cast in query_rows.rows::<RepoRow>()? {
            // get this row if possible
            let row = cast?;
            // get an entry to this submissions timestamp list
            let entry = sorted.entry(row.uploaded).or_default();
            // add this submission to the btreemap
            entry.push(row.url.clone());
            // try to get a mutable entry to this submissions repo
            match map.entry(row.url.clone()) {
                // we do not yet have an entry for this url
                Vacant(entry) => {
                    // create a repo entry for this submission
                    let repo = Repo::try_from((row, user))?;
                    // add this repo to the map
                    entry.insert(repo);
                }
                // we already have an entry for this submission
                Occupied(entry) => entry.into_mut().add_row(row, user)?,
            }
        }
    }
    // build our vec of repo details to return
    let mut details = Vec::with_capacity(map.len());
    // crawl the submissions in order and add them to our map
    for (_, repos) in sorted.into_iter().rev() {
        // crawl over the submission in this timestamp group
        for repo in repos {
            // try to pop this submission
            if let Some(mut repo) = map.remove(&repo) {
                // get the tags for this repo
                super::tags::get(TagType::Repos, &groups, &repo.url, &mut repo.tags, shared)
                    .await?;
                // add this repo to our vec
                details.push(repo);
            }
        }
    }
    Ok(details)
}

/// Save this zipped repo to s3 and add it to scylla
///
/// # Arguments
///
/// * `repo` - The repo to save data for
/// * `sha256` - The sha256 of the zipped repo we are saving
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::upload", skip(shared), err(Debug))]
pub async fn upload(repo: &str, sha256: &str, shared: &Shared) -> Result<(), ApiError> {
    // insert this repo data blob into the repo data table in scylla
    shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.repos.insert_data, (repo, &sha256))
        .await?;
    Ok(())
}

/// Delete a repo data blob from scylla and s3
///
/// # Arguments
///
/// * `repo` - The repo to delete a data blob from
/// * `hash` - The hash of the repo data blob to delete
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::delete_repo_data", skip(shared), err(Debug))]
async fn delete_repo_data(repo: &str, hash: &str, shared: &Shared) -> Result<(), ApiError> {
    // delete this object from s3
    shared.s3.repos.delete(hash).await?;
    // delete this repos data from scylla
    shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.repos.delete_data, (hash, repo))
        .await?;
    Ok(())
}

/// Prune any orphaned repo data blobs
///
/// # Arguments
///
/// * `repo` - The repo to prune orphaned data blobs from
/// * `hash` - The hash of the repo data zip that just had commits added to it
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::prune_data", skip(shared), err(Debug))]
async fn prune_data(
    repo: &str,
    hash: &str,
    oldest: Option<DateTime<Utc>>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // get the hashes of the data we have for this repo
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.repos.get_data, (repo,))
        .await?;
    // enable rows on this query response
    let query_rows = query.into_rows_result()?;
    // check if we got any rows
    if query_rows.rows_num() > 0 {
        // cast our rows to strings
        let mut hashes = query_rows
            .rows::<(String,)>()?
            .filter_map(|res| log_scylla_err!(res))
            .map(|tup| tup.0)
            .collect::<Vec<String>>();
        // remove the repo data hash we just uploaded
        if let Some(index) = hashes.iter().position(|item| item == hash) {
            // remove our current hash from our list of hashes to check
            hashes.swap_remove(index);
        }
        event!(Level::INFO, hashes = hashes.len());
        // get the earliest a commit should exist at if we weren't given an oldest hint
        let end = match oldest {
            // subtract 1 second to ensure that we don't miss our oldest commit
            Some(oldest) => oldest - chrono::Duration::seconds(1),
            None => Utc
                .timestamp_opt(shared.config.thorium.repos.earliest, 0)
                .unwrap(),
        };
        // get our partition size
        let partition_size = shared.config.thorium.repos.partition_size;
        // make sure each repo data blob has at least one commit tied to it
        for repo_data in hashes {
            // build the cursor to use when checking what repo data blobs are needed
            let cursor = ExistsCursor::new(Utc::now(), end, partition_size)?;
            // check if any commits are still backed by this repo
            if !cursor
                .exists(
                    &shared.scylla.prep.commitishes.get_repo_data_count,
                    &repo_data,
                    shared,
                )
                .await?
            {
                // this repo has no commits tied to it so prune it from scylla and s3
                delete_repo_data(repo, &repo_data, shared).await?;
            }
        }
    }
    Ok(())
}

/// Validate that a repos data blob actually exist
///
/// # Arguments
///
/// * `repo` - The repo to check if data exists for
/// * `data` - The sha256 of the data to check for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::repo_data_exists", skip(shared), err(Debug))]
pub async fn repo_data_exists(repo: &str, data: &str, shared: &Shared) -> Result<(), ApiError> {
    // try to get this repos data
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.repos.data_exists, (repo, data))
        .await?;
    // enable rows on this query response
    let query_rows = query.into_rows_result()?;
    // if we don't get any rows then this repo doesn't exist
    if query_rows.rows_num() == 0 {
        return not_found!(format!("{} does not have data {}", repo, data));
    }
    Ok(())
}

/// Add or update commitishes to this repo
///
/// # Arguments
///
/// * `repo` - The repo to save commitishes for
/// * `data` - The hash of the repo data zip containing these commits
/// * `groups` - The groups these commits are visible too
/// * `map` - The map of commits to save
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::repos::add_commitishes", skip(req, shared), err(Debug))]
pub async fn add_commitishes(
    repo: &mut Repo,
    repo_data: &str,
    req : CommitishMapRequest,
    shared: &Shared,
) -> Result<(), ApiError> {
    // log number of commits we are saving
    event!(Level::INFO, commitishes=&req.commitishes.len(), end = req.end);
    // make sure this repo data blob exists for this repo
    repo_data_exists(&repo.url, repo_data, shared).await?;
    // instance a list to store our census keys to update
    let mut keys = Vec::with_capacity(req.groups.len() * req.commitishes.len());
    // save the commit data into scylla
    for (key, commitish) in req.commitishes {
        // get this commitishes timestamp
        let timestamp = commitish.timestamp();
        // get the month and year of this commitish
        let year = timestamp.year();
        // get the partition size for repos
        let chunk_size = shared.config.thorium.repos.partition_size;
        // get the partition to write this repo off too
        let bucket = helpers::partition(timestamp, year, chunk_size);
        // serialize our data
        let (kind, commitish_data) = commitish.serialize_data()?;
        // crawl each group and insert into our commitish list table
        for group in &req.groups {
            // insert this commit into the commitish table in scylla
            shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.commitishes.insert,
                (kind, group, &repo.url, &key, timestamp, &commitish_data, &repo_data)
            ).await?;
            // insert this commit into the commitish list table in scylla for this group
            shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.commitishes.insert_list,
                (kind, group, year, bucket, &repo.url, timestamp, &key, &repo_data)
            ).await?;
        }
        // build the keys to update
        super::keys::commitishes::census_keys(&mut keys, &repo.url, kind, &req.groups, year, bucket, shared);
    }
    // update our census info for these commitishes
    super::census::incr_cache(keys, shared).await?;
    // prune any orphaned repo data blobs if this the final commit batch
    if req.end {
        // if we have an earliest set then try to update it
        if let Some(earliest) = req.earliest {
            // update our repos earliest timestamp if needed
            update_earliest(&req.groups, earliest, repo, shared).await?;
        }
        // this is the last commit batch so clear any orphans
        prune_data(&repo.url, repo_data, repo.earliest, shared).await?;
    }
    Ok(())
}

//// List commitishes for a specific repo
///
/// # Arguments
///
/// * `repo` - The repo to list commitishes from
/// * `params` - The query params to use when listing commitishes
/// * `dedupe` - Whether to dedupe any data that is returned
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[instrument(name = "db::repos::commitishes", skip(params, shared), err(Debug))]
pub async fn commitishes(
    repo: &Repo,
    mut params: CommitishListParams,
    dedupe: bool,
    shared: &Shared,
) -> Result<ScyllaCursor<Commitish>, ApiError> {
    // take our kinds from our params
    let kinds = std::mem::take(&mut params.kinds);
    // default our end date with this repos earliest commit timestamp if we have one
    if let Some(earliest) = repo.earliest {
        params.end.get_or_insert_with(|| earliest);
    }
    // get our target repos url
    let url = repo.url.clone();
    // get our cursor
    let mut cursor = ScyllaCursor::from_params_extra(params, (kinds, url), dedupe, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    Ok(cursor)
}

/// Cast this to a map of what groups each commits is visible too and their commitish kinds
///
/// # Arguments
///
/// * `list` - the commits to create a map for
#[instrument(name = "db::repos::commitish_group_map", skip_all)]
pub(super) fn commitish_group_map(
    list: Vec<Commitish>,
) -> HashMap<String, (CommitishKinds, Vec<String>)> {
    // create a map presized for our commits
    let mut map = HashMap::with_capacity(list.len());
    // crawl our data and insert into our map
    for commitish in list {
        // get the relevant info from each commitish
        let (key, kind, groups) = match commitish {
            Commitish::Commit(commit) => (commit.hash, CommitishKinds::Commit, commit.groups),
            Commitish::Branch(branch) => (branch.name, CommitishKinds::Branch, branch.groups),
            Commitish::Tag(tag) => (tag.name, CommitishKinds::Tag, tag.groups),
        };
        // insert our commitishes info
        map.insert(key, (kind, groups));
    }
    map
}

/// Helps the commitish details lister get details for a single commitish
async fn commitish_details_helper(
    repo: &str,
    commitish: Commitish,
    shared: &Shared,
) -> Result<CommitishDetails, ApiError> {
    // get our commitish's info
    let kind = commitish.kind();
    let key = commitish.key();
    let groups = commitish.groups();
    // query scylla for this commitishes data
    let query = shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.commitishes.get_data,
            (kind, groups, repo, key),
        )
        .await?;
    // enable rows on this response
    let query_rows = query.into_rows_result()?;
    // get this commitishes data or return an error
    match query_rows.maybe_first_row::<(String,)>()? {
        Some((data,)) => return commitish.to_details(&data),
        None => {
            // build our error string
            let msg = format!("{}:{} is missing commitish details", repo, commitish.key());
            // return this error
            internal_err!(msg)
        }
    }
}

//// Get details on a list of commitishes for a repo
///
/// # Arguments
///
/// * `groups` - The groups to search in
/// * `repo` - The repo to get commit details for
/// * `list` - The commits to get details for
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[instrument(name = "db::repos::commit_details", skip(list, shared), err(Debug))]
pub async fn commitish_details(
    groups: &[String],
    repo: &str,
    list: Vec<Commitish>,
    shared: &Shared,
) -> Result<Vec<CommitishDetails>, ApiError> {
    // log the number of commits we are getting details for
    event!(Level::INFO, commits = list.len());
    // build a details list for our samples
    let mut details: Vec<CommitishDetails> = Vec::with_capacity(list.len());
    // get the commitish details for each of our commitishes
    let mut details_stream = stream::iter(list)
        .map(|commitish| async move { commitish_details_helper(repo, commitish, shared).await })
        .buffered(10);
    // poll this stream for our commitish details
    while let Some(commitish_details) = details_stream.next().await {
        // raise any errors from deserializing our commits
        let commitish_details = commitish_details?;
        // add this commitish to our list
        details.push(commitish_details);
    }
    Ok(details)
}

/// Check if a repo has a specific commit
///
/// # Arguments
///
/// * `groups` - The groups to look for commits in
/// * `repo` - The repo to check if a commit exists in
/// * `kind` - The kind of commitish to check
/// * `commitish` - The commitish to check
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::commit_exists", skip(shared), err(Debug))]
pub async fn commitish_exists(
    groups: &HashSet<&String>,
    repo: &str,
    kind: CommitishKinds,
    commit: &str,
    shared: &Shared,
) -> Result<(), ApiError> {
    // if we have more than 100 groups then break them into chunks of 50
    if groups.len() > 100 {
        // cast our hashset to a vector so we can chunk it up
        let group_cast = groups.iter().map(|name| *name).collect::<Vec<&String>>();
        // chunk gour groups up into chunks of 100
        for group_chunk in group_cast.chunks(100) {
            // send this query to scylla
            let query = shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.commitishes.exists,
                    (kind, group_chunk, repo, commit),
                )
                .await?;
            // enable rows on this response
            let query_rows = query.into_rows_result()?;
            // get the first row since this is a count operation
            if query_rows.rows_num() > 0 {
                // at least one row was found so this commit exists
                return Ok(());
            }
        }
    } else {
        // send this query to scylla
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.commitishes.exists,
                (kind, groups, repo, commit),
            )
            .await?;
        // enable rows on this response
        let query_rows = query.into_rows_result()?;
        // get the first row since this is a count operation
        if query_rows.rows_num() > 0 {
            // at least one row was found so this commit exists
            return Ok(());
        }
    }
    not_found!(format!("Repo {} does not have commit {}", repo, commit))
}

/// Get the latest commit for a repo
///
/// # Arguments
///
/// * `groups` - The groups to look for commits in
/// * `repo` - The repo to get the latest commit for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::latest_commit", skip(shared), err(Debug))]
pub async fn latest_commit(
    groups: &HashSet<&String>,
    repo: &Repo,
    shared: &Shared,
) -> Result<String, ApiError> {
    // convert groups to owned strings
    let groups = groups
        .iter()
        .map(|item| item.to_string())
        .collect::<Vec<String>>();
    // build the params for listing this repos commits
    let params = CommitishListParams::new(groups, repo.earliest, 1);
    // build our extra filters to only search commits for this repo
    let extra = (vec![CommitishKinds::Commit], repo.url.to_owned());
    // get our cursor
    let mut cursor: ScyllaCursor<Commitish> =
        ScyllaCursor::from_params_extra(params, extra, false, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // get the first commits hash
    match cursor.data.into_iter().next() {
        Some(commitish) => {
            // make sure we got a commit
            match commitish {
                Commitish::Commit(commit) => Ok(commit.hash),
                // These error cases should never happen since we restrict our search to just commits
                Commitish::Branch(_) => {
                    internal_err!("Expected a commit and got a branch!".to_owned())
                }
                Commitish::Tag(_) => internal_err!("Expected a commit and got a tag!".to_owned()),
            }
        }
        None => not_found!(format!("No commits for {} were found!", repo.url)),
    }
}

/// Check if a commitish key is ambiguous
///
/// # Arguments
///
/// * `repo` - The repo to get the repo data hash for
/// * `commitish` - The commitish we are searching for
/// * `previous` - The previous commitish kind we found
/// * `query` - The query to check
#[instrument(name = "db::repos::is_ambiguous", skip(query), err(Debug))]
async fn is_ambiguous(
    repo: &str,
    commitish: &str,
    previous: &mut Option<CommitishKinds>,
    query: QueryResult,
) -> Result<String, ApiError> {
    // enable rows on this query response
    let query_rows = query.into_rows_result()?;
    // check if all of our queries are for the same kind
    if let Ok(mut rows) = query_rows.rows::<(CommitishKinds, String)>() {
        // track the first kind  and data hash we see
        let (first_kind, repo_data) = match rows.next() {
            Some(row) => row?,
            None => {
                return not_found!(format!(
                    "Commitish {} does not exist for {}",
                    commitish, repo
                ))
            }
        };
        // if we have a previous kind then confirm they are still the same
        match previous {
            Some(previous_kind) => {
                if *previous_kind != first_kind {
                    return bad!(format!(
                        "{} is ambiguous please restrict the commitish kind to download",
                        commitish
                    ));
                }
            }
            None => {
                *previous = Some(first_kind);
            }
        }
        // check if we find any other kinds
        while let Some(row) = rows.next() {
            // unwrap any errors from this row
            let (kind, _) = row?;
            // check if this kind is different
            if kind != first_kind {
                return bad!(format!(
                    "{} is ambiguous please restrict the commitish kind to download",
                    commitish
                ));
            }
        }
        // return the repo data we found
        Ok(repo_data)
    } else {
        not_found!(format!(
            "Commitish {} does not exist for {}",
            commitish, repo
        ))
    }
}

/// Get the repo data hash for a specific repo and commitish
///
/// # Arguments
///
/// * `groups` - The groups to search for this repos data
/// * `repo` - The repo to get the repo data hash for
/// * `commitish` - The commitish we are searching for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::repos::repo_data_hash", skip(shared), err(Debug))]
pub async fn repo_data_hash(
    groups: &HashSet<&String>,
    repo: &str,
    kinds: &Vec<CommitishKinds>,
    commitish: &str,
    shared: &Shared,
) -> Result<String, ApiError> {
    // track our previous kinds across checks
    let mut previous = None;
    // if we have more than 100 groups then break them into chunks of 50
    if groups.len() > 32 {
        // track the last repo_data hash
        let mut last_repo_data_hash = None;
        // cast our hashset to a vector so we can chunk it up
        let group_cast = groups.iter().map(|name| *name).collect::<Vec<&String>>();
        // chunk gour groups up into chunks of 100
        for group_chunk in group_cast.chunks(32) {
            // retrieve submissions from scylla
            let query = shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.commitishes.get_repo_data,
                    (kinds, group_chunk, repo, commitish),
                )
                .await?;
            // check if this commitish is ambiguous or not
            let repo_data = is_ambiguous(repo, commitish, &mut previous, query).await?;
            // set our repo data hash
            last_repo_data_hash = Some(repo_data);
        }
        // if we have a repo data hash then return it
        if let Some(repo_data) = last_repo_data_hash {
            return Ok(repo_data);
        }
    } else {
        // retrieve submissions from scylla
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.commitishes.get_repo_data,
                (kinds, groups, repo, commitish),
            )
            .await?;
        // check if this commitish is ambiguous or not
        return is_ambiguous(repo, commitish, &mut previous, query).await;
    }
    not_found!(format!(
        "Commitish {} does not exist for {}",
        commitish, repo
    ))
}

/// List repos for specific groups
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[instrument(name = "db::repos::list", skip_all, err(Debug))]
pub async fn list(
    params: RepoListParams,
    shared: &Shared,
) -> Result<ScyllaCursor<RepoListLine>, ApiError> {
    // get our cursor
    let mut cursor = ScyllaCursor::from_params(params, true, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    Ok(cursor)
}
