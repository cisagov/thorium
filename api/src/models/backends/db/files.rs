//! Saves files into the backend

use chrono::prelude::*;
use itertools::Itertools;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BTreeMap, HashMap, HashSet};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::ScyllaCursor;
use crate::models::backends::TagSupport;
use crate::models::{
    Comment, CommentForm, CommentRow, Event, FileListParams, ResultSearchEvent, Sample,
    SampleCheck, SampleCheckResponse, SampleForm, SampleListLine, SampleSubmissionResponse,
    Submission, SubmissionChunk, SubmissionRow, SubmissionUpdate, TagDeleteRequest, TagRequest,
    TagSearchEvent, User,
};
use crate::utils::s3::StandardHashes;
use crate::utils::{helpers, ApiError, Shared};
use crate::{
    conflict, for_groups, internal_err, log_scylla_err, not_found, same_vec, serialize,
    unauthorized,
};

/// Deletes a submission from multiple groups, breaking into chunks of 100 if > 100
macro_rules! delete_from_groups {
    ($shared:expr, $groups:expr, $year:expr, $bucket:expr, $uploaded:expr, $id:expr) => {
        // if we have less than 100 groups then just delete them in one go
        if $groups.len() <= 100 {
            // remove any requested submissions
            $shared
                .scylla
                .session
                .execute_unpaged(
                    &$shared.scylla.prep.samples.delete_multiple_groups,
                    (&$groups, $year, $bucket, $uploaded, $id),
                )
                .await?;
        } else {
            // we have more than 100 groups so break them into chunks of 100
            for chunk in $groups.chunks(100) {
                // copy this chunk into a vec
                let chunk_vec = chunk.to_vec();
                // remove this chunks submissions
                $shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &$shared.scylla.prep.samples.delete_multiple_groups,
                        (chunk_vec, $year, $bucket, $uploaded, $id),
                    )
                    .await?;
            }
        }
    };
}

/// Check if a specific submission already exists in the DB
///
/// # Arguments
///
/// * `user` - The user that is checking if a file exists or not
/// * `check` - The info to check with
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::exists", skip(user, shared), err(Debug))]
pub async fn exists(
    user: &User,
    check: &SampleCheck,
    shared: &Shared,
) -> Result<SampleCheckResponse, ApiError> {
    // get this sample if it already exists
    if let Some(sample) = for_groups!(get, user, shared, user, &check.sha256)? {
        // this sample already exists so check if we should patch one of our existing samples
        // this must be for the same groups, user, name, and origin otherwise we could be leaking info
        if let Some(sub) = sample.submissions.into_iter().find(|sub| {
            sub.submitter == user.username
                && sub.origin == check.origin
                && sub.name == check.name
                && same_vec!(&sub.groups, &check.groups)
        }) {
            // This submission already exists so just upload that
            return Ok(SampleCheckResponse {
                exists: true,
                id: Some(sub.id),
            });
        }
        return Ok(SampleCheckResponse {
            exists: true,
            id: None,
        });
    }
    Ok(SampleCheckResponse {
        exists: false,
        id: None,
    })
}

/// Check if a specific submission already exists in the DB
///
/// # Arguments
///
/// * `user` - The user that is checking if a file exists or not
/// * `check` - The info to check with
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::sha256_exists", skip(shared), err(Debug))]
pub async fn sha256_exists(
    groups: &[String],
    sha256: &str,
    shared: &Shared,
) -> Result<bool, ApiError> {
    // check if this sample exists in any of the groups we can see
    // break our group into chunks of 50 to stay under the cartesian product limit
    for groups_chunk in groups.chunks(50) {
        // build a query to check if the sample exists in this bucket of groups
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.samples.auth,
                (vec![sha256], groups_chunk),
            )
            .await?;
        // cast this query to a rows query
        let query_rows = query.into_rows_result()?;
        // if we got any sha256s back then this sample must exist and be visible to this user
        if query_rows.rows_num() > 0 {
            // this sample exists
            return Ok(true);
        }
    }
    Ok(false)
}

/// Adds a child sample to its originating result
///
/// # Arguments
///
/// * `sha256` - The sha256 that is being ingested as a child sample
/// * `submission` - The submission id of the child sample being ingested
/// * `results` - The result IDs to add this child under
/// * `shared` - Shared Thorium objects
/// * `req_id` - The uuid for this request
#[instrument(name = "db::files::add_child", skip(shared), err(Debug))]
async fn add_child(
    sha256: &str,
    submission: &Uuid,
    results: &[Uuid],
    shared: &Shared,
) -> Result<(), ApiError> {
    // skip updating result children if none are passed
    if !results.is_empty() {
        // instance an empty result map to insert any retrieved rows into
        let mut temp = HashMap::with_capacity(results.len());
        // execute our query
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.results.get_uploaded_by_id, (results,))
            .await?;
        // enable casting to types for this query
        let query_rows = query.into_rows_result()?;
        // set the type to cast this stream too
        let typed_iter = query_rows.rows::<(Uuid, DateTime<Utc>)>()?;
        // crawl and consume these typed rows
        for typed in typed_iter {
            // check if we ran into a problem casting this row
            let (id, timestamp) = typed?;
            // add this to our temp map
            temp.insert(id, timestamp);
        }
        // add this child to the target results
        for (result_id, uploaded) in temp.into_iter() {
            // update our child field in this result
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.results.update_children,
                    (&sha256, submission, result_id, uploaded),
                )
                .await?;
        }
    }
    Ok(())
}

/// Saves a file to the backend
///
/// # Arguments
///
/// * `user` - The user who is saving this file
/// * `upload` - The sample to save to the backend
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[rustfmt::skip]
#[instrument(name = "db::files::create", skip(user, form, shared), err(Debug))]
pub async fn create(
    user: &User,
    mut form: SampleForm,
    hashes: StandardHashes,
    shared: &Shared,
) -> Result<SampleSubmissionResponse, ApiError> {
    // get our origin if one was set
    let (origin, results) = form.origin.to_origin()?;
    // serialize the origin
    let origin_str = origin.serialize()?;
    // add the user that is uploading this as a tag
    form.tags.entry("submitter".to_owned()).or_default().insert(user.username.clone());
    // get the partition size to use for files and tags
    let chunk = shared.config.thorium.files.partition_size;
    // If this submission doesn't exist then use the current time as its upload time
    let now = Utc::now();
    // get this sample if it already exists and build an earliest seen map
    let mut earliest = if let Some(sample) = for_groups!(get, user, shared, user, &hashes.sha256)? {
        // this sample already exists so check if we should return a conflict error
        if let Some(sub)  = sample.submissions.iter()
            .find(|sub| sub.submitter == user.username && sub.origin == origin
                && sub.description == form.description && sub.name == form.file_name
                && same_vec!(&sub.groups, &form.groups)) {
                // upload all our user specified samples tags regardless of if this has been seen before or not
                // get the earliest time this repo was uploaded for each group
                let earliest = sample.earliest();
                // build a tag request
                let mut req = TagRequest::<Sample>::default().groups(form.groups.clone());
                // move our tags over to our tag request
                req.tags = form.tags;
                // save our files tags to scylla
                super::tags::create(
                    user,
                    hashes.sha256.clone(),
                    req,
                    &earliest,
                    shared,
                )
                .await?;
                // add this child sample to its result if any were set
                add_child(&hashes.sha256, &sub.id, &results, shared).await?;
                // this user has already uploaded this same submission so return a conflict error
                return conflict!(hashes.sha256);
        }
        // get the earliest timestamp for each group that we can see this sample was submitted at
        sample.earliest_owned()
    } else {
        // build a map of the current timestamp for each group
        HashMap::with_capacity(form.groups.len())
    };
    // add any new groups
    for group in form.groups.iter() {
        earliest.entry(group.to_owned()).or_insert_with(|| now);
    }
    // get the current year and month so we can bucket this sample by time
    let year = now.year();
    let bucket = helpers::partition(now, year, chunk);
    let id = Uuid::new_v4();
    // save submission objects into scylla
    // currently do it one at a time instead of with buffered_unordered to work around Fn Once
    for group in form.groups.iter() {
        shared.scylla.session.execute_unpaged(
            &shared.scylla.prep.samples.insert,
            (group, &year, bucket, &hashes.sha256, &hashes.sha1, &hashes.md5, &id, &form.file_name, &form.description, &user.username, &origin_str, now)
        ).await?;
    }
    // build the keys for this items census cache
    let keys = super::keys::samples::census_keys(&form.groups, year, bucket, shared);
    // update this samples census cache info
    super::census::incr_cache(keys, shared).await?;
    // add our origin tags to our tags map
    origin.get_tags(&mut form.tags);
    // build a tag request
    let mut req = TagRequest::<Sample>::default().groups(form.groups.clone());
    // move our tags over to our tag request
    req.tags = form.tags;
    // save our files tags to scylla
    super::tags::create_owned(
        user,
        hashes.sha256.clone(),
        req,
        &earliest,
        shared,
    )
    .await?;
    // add this child sample to its result if any were set
    add_child(&hashes.sha256, &id, &results, shared).await?;
    // create our new sample event
    let event = Event::new_sample(user, form.groups.clone(), hashes.sha256.clone(), form.trigger_depth);
    // save our event
    super::events::create(&event, shared).await?;
    // build our submission response object
    let resp = SampleSubmissionResponse { sha256: hashes.sha256, sha1: hashes.sha1, md5: hashes.md5, id };
    Ok(resp)
}

/// Authorizes that a user has access to a list of samples
///
/// # Arguments
///
/// * `groups` - The groups the user is in
/// * `sha256` - The Sha256 to authorize we have access to
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[instrument(name = "db::files::authorize", skip(shared), err(Debug))]
pub async fn authorize(
    groups: &Vec<String>,
    sha256s: &Vec<String>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // if we specified no groups then we do not have access to this sample
    if groups.is_empty() {
        return unauthorized!();
    }
    // check if our cartesian product will be under 100
    if groups.len() * sha256s.len() < 100 {
        // our cartesian product will be low enough so just query for them all at once
        // get the number of tools or default to 10
        // check if any of our groups have access to this sample
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.samples.auth, (sha256s, groups))
            .await?;
        // cast this to a rows query
        let query_rows = query.into_rows_result()?;
        //  make sure we got back the same number of sha256s
        if query_rows.rows_num() != sha256s.len() {
            return unauthorized!();
        }
    } else {
        // track the sha256s we have authed
        let mut authed = HashSet::with_capacity(sha256s.len());
        // build an iter over the cartesian product of our groups and sha256s
        // doing this in a buffered_unordered closure would probably be faster
        // but that runs into lifetime errors currently :(
        for (groups_chunk, sha256s_chunk) in groups[..]
            .chunks(50)
            .cartesian_product(sha256s[..].chunks(50))
        {
            // send a query to check if we have access to these samples
            let query = shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.samples.auth,
                    (sha256s_chunk, groups_chunk),
                )
                .await?;
            // enable casting to types for this query
            let query_rows = query.into_rows_result()?;
            // cast our rows to the right type
            for typed_row in query_rows.rows::<(String,)>()? {
                // check if we failed to cast this row
                let (sha256,) = typed_row?;
                // add this sha256 to our authed samples
                authed.insert(sha256);
            }
            // if we have authed all of our sha256s then return early
            if authed.len() == sha256s.len() {
                // we have authed all of the sha256s we wanted to auth against
                return Ok(());
            }
        }
    }
    // we have access to this sample
    Ok(())
}

/// Gets all the comments for a sample
///
/// # Arguments
///
/// * `groups` - The groups to restrict our returned comments too
/// * `sha256` - The sha256 to get comments for
/// * `list` - The vector to add our tags too
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::get_comments", skip(list, shared), err(Debug))]
async fn get_comments(
    groups: &Vec<String>,
    sha256: &str,
    list: &mut Vec<Comment>,
    shared: &Shared,
) -> Result<(), ApiError> {
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
                .execute_unpaged(&shared.scylla.prep.comments.get, (chunk_vec, sha256))
                .await?;
            // build a btreemap to collect all comments and then sort them
            let mut map: BTreeMap<DateTime<Utc>, HashMap<Uuid, Comment>> = BTreeMap::default();
            // enable casting to types for this query
            let query_rows = query.into_rows_result()?;
            // set the type to cast this stream too
            let typed_iter = query_rows.rows::<CommentRow>()?;
            // crawl over rows and add them to our tag map while logging any errors
            typed_iter
                .filter_map(|res| log_scylla_err!(res))
                .for_each(|comment| {
                    // get an entry to the nested map of comments by id
                    let id_map = map.entry(comment.uploaded).or_default();
                    // get our comments doc so we can add a group or insert it
                    match id_map.entry(comment.id) {
                        // this comment has already been added so just add a new group to it
                        Occupied(entry) => entry.into_mut().groups.push(comment.group),
                        Vacant(entry) => {
                            // try to turn this row into a comment
                            let res = Comment::try_from(comment);
                            // if we can deserialize this string then insert it
                            if let Some(cast) = log_scylla_err!(res) {
                                entry.insert(cast);
                            }
                        }
                    }
                });
            // flatten our map of comments and append it to our list of comments
            list.extend(map.into_iter().flat_map(|(_, map)| map.into_values()));
        }
    } else {
        // we have less than 100 groups so just get their data
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.comments.get, (groups, sha256))
            .await?;
        // enable casting to types for this query
        let query_rows = query.into_rows_result()?;
        // set the type to cast this stream too
        let typed_iter = query_rows.rows::<CommentRow>()?;
        // build a btreemap to collect all comments and then sort them
        let mut map: BTreeMap<DateTime<Utc>, HashMap<Uuid, Comment>> = BTreeMap::default();
        // crawl over rows and add them to our tag map while logging any errors
        typed_iter
            .filter_map(|res| log_scylla_err!(res))
            .for_each(|comment| {
                // get an entry to the nested map of comments by id
                let id_map = map.entry(comment.uploaded).or_default();
                // get our comments doc so we can add a group or insert it
                match id_map.entry(comment.id) {
                    // this comment has already been added so just add a new group to it
                    Occupied(entry) => entry.into_mut().groups.push(comment.group),
                    Vacant(entry) => {
                        // try to turn this row into a comment
                        let res = Comment::try_from(comment);
                        // if we can deserialize this string then insert it
                        if let Some(cast) = log_scylla_err!(res) {
                            entry.insert(cast);
                        }
                    }
                }
            });
        // flatten our map of comments and append it to our list of comments
        list.extend(map.into_iter().flat_map(|(_, map)| map.into_values()));
    }
    Ok(())
}

/// Gets all submissions for a specific set of groups and a sha256 and cast it to a sample
///
/// # Arguments
///
/// * `groups` - The groups to search in
/// * `sha256` - The sha256 to retrieve submission docs for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::get", skip(shared), err(Debug))]
pub async fn get(
    groups: &Vec<String>,
    user: &User,
    sha256: &str,
    shared: &Shared,
) -> Result<Option<Sample>, ApiError> {
    // build a btree to sort our submissions
    let mut sorted: BTreeMap<DateTime<Utc>, Vec<Submission>> = BTreeMap::default();
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
                .execute_unpaged(&shared.scylla.prep.samples.get, (sha256, chunk_vec))
                .await?;
            // enable casting to types for this query
            let query_rows = query.into_rows_result()?;
            // set the type to cast this stream too
            let typed_iter = query_rows.rows::<SubmissionRow>()?;
            // crawl over rows and add them to our tag map while logging any errors
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
                        entry.push(Submission::from(sub));
                    }
                });
        }
    } else {
        // get this chunks data
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.samples.get, (sha256, groups))
            .await?;
        // enable casting to types for this query
        let query_rows = query.into_rows_result()?;
        // set the type to cast this stream too
        let typed_iter = query_rows.rows::<SubmissionRow>()?;
        // crawl over rows and add them to our tag map while logging any errors
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
                    entry.push(Submission::from(sub));
                }
            });
    }
    // get an iter of these submissions in descending order
    let mut descending = sorted.into_iter().rev().flat_map(|(_, sub)| sub);
    // get the first submission object and cast it to a sample
    if let Some(sub) = descending.next() {
        // cast this submission object to a sample
        let mut sample = Sample::try_from_submission(groups, sub, shared).await?;
        // crawl the remaining submission objects and add them
        for sub in descending {
            // add this submission to our sample
            sample.add(groups, sub, shared).await?;
        }
        // get the tags and comments for this sample
        sample.get_tags(groups, shared).await?;
        get_comments(groups, sha256, &mut sample.comments, shared).await?;
        // return our sample
        return Ok(Some(sample));
    };
    Ok(None)
}

/// Deletes a submission from a sample
///
/// # Arguments
///
/// * `sample` - The sample to delete a submission from
/// * `submission` - The submission to delete
/// * `groups` - The groups to delete this submission from
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[instrument(name = "db::files::delete_submission", skip(shared), err(Debug))]
pub async fn delete_submission(
    sample: &Sample,
    sub: &SubmissionChunk,
    groups: &Vec<String>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // get the target submissions year/hour of being submitted
    let year = sub.uploaded.year();
    // get the chunk size for files
    let chunk_size = shared.config.thorium.files.partition_size;
    // determine what bucket to add data too
    let bucket = helpers::partition(sub.uploaded, year, chunk_size);
    // delete the submissions from the db
    delete_from_groups!(shared, groups, year, bucket, sub.uploaded, sub.id);
    // get the groups and submitters of other submissions for this sample
    let group_submitter_map = other_submissions(&sample.sha256, sub, groups, shared).await?;
    // prune submitter tags
    prune_submitter_tags(
        &sample.sha256,
        groups,
        &sub.submitter,
        &group_submitter_map,
        shared,
    )
    .await?;
    // if no other submissions exist then just clean up all data otherwise we
    // need to prune data from groups that no longer have access to prevent
    // leaking other groups also can see this sample
    let prunes: Vec<String> = if group_submitter_map.is_empty() {
        // no other groups have access to this sample so delete all of its data
        sample
            .groups()
            .into_iter()
            .map(|name| name.to_owned())
            .collect()
    } else {
        // other groups have access to this data so determine if there are any
        // groups that no longer have access and should have there access pruned
        groups
            .iter()
            .filter(|group| !group_submitter_map.contains_key(*group))
            .map(|name| name.to_owned())
            .collect()
    };
    // if any groups need their access pruned then do it
    if !prunes.is_empty() {
        // prune access for our target groups
        prune_access(sample, &prunes, shared).await?;
    }
    // build the ache keys for these submissions
    // There is no extra info for samples to pass so we just pass in an &()
    let keys = super::keys::samples::census_keys(groups, year, bucket, shared);
    // update this samples census cache info
    super::census::decr_cache(keys, shared).await?;
    Ok(())
}

/// Determine if any other submissions for this sha256 exist
/// and return a map of the submission groups and submitters
///
/// # Arguments
///
/// * `sha256` - The sha256 to check for other submissions for
/// * `sub` - The submission that was deleted
/// * `groups` - The groups that the submission was deleted from
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::other_submissions", skip(shared), err(Debug))]
async fn other_submissions(
    sha256: &str,
    sub: &SubmissionChunk,
    groups: &Vec<String>,
    shared: &Shared,
) -> Result<HashMap<String, HashSet<String>>, ApiError> {
    // get the groups and submitters for the sample
    let query = shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.samples.get_basic_submission_info,
            &(sha256,),
        )
        .await?;
    // enable casting to types for this query
    let query_rows = query.into_rows_result()?;
    // set the type to cast this stream too
    let typed_iter = query_rows.rows::<(String, String, Uuid)>()?;
    // size our map correctly for the number of rows returned
    let mut map: HashMap<String, HashSet<String>> = HashMap::with_capacity(query_rows.rows_num());
    // cast our rows to typed values
    for row in typed_iter {
        // try to cast our row to the group, submitter, and id
        let (group, submitter, id) = row?;
        // ignore row if it was supposed to be deleted, but the db
        // hasn't yet updated to reflect the delete
        if id == sub.id && groups.contains(&group) {
            continue;
        }
        // add this group's submitter to our map
        map.entry(group).or_default().insert(submitter);
    }
    return Ok(map);
}

/// Prune submitter tag if submitter no longer has a submission in the groups
///
/// # Arguments
///
/// * `sha256` - The sample sha256
/// * `groups` - The list of groups that were deleted from
/// * `submitter` - The submitter of the deleted submission
/// * `group_submitter_map` - A map of groups to submitters after deletion
/// * `shared` - Shared Thorium objects
#[instrument(
    name = "db::files::prune_submitter_tags",
    skip(group_submitter_map, shared),
    err(Debug)
)]
async fn prune_submitter_tags(
    sha256: &String,
    groups: &Vec<String>,
    submitter: &String,
    group_submitter_map: &HashMap<String, HashSet<String>>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // identify all groups to which the submitter no longer has submissions
    let mut dangling_groups: HashSet<&String> = HashSet::default();
    for group in groups {
        if let Some(submitters) = group_submitter_map.get(group) {
            if !submitters.contains(submitter) {
                dangling_groups.insert(group);
            }
        } else {
            dangling_groups.insert(group);
        }
    }
    // delete the submitter tag for the submitter and groups
    if !dangling_groups.is_empty() {
        let tag_deletes: TagDeleteRequest<Sample> = TagDeleteRequest::default()
            .groups(dangling_groups)
            .add("submitter", submitter);
        super::tags::delete(sha256, &tag_deletes, shared).await?;
    }
    Ok(())
}

/// Prune access to any sample info for the target groups
///
/// This will remove access to any tags, comments, and results.
///
/// # Arguments
///
/// * `sample` - The sample to remove access too
/// * `groups` - The groups that will get pruned
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::prune_access", skip(sample, shared), err(Debug))]
async fn prune_access(
    sample: &Sample,
    groups: &Vec<String>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // log the sha256 we are pruning access too
    event!(Level::INFO, sha256 = &sample.sha256);
    // build our tag delete object with empty groups to delete from all
    let mut tag_deletes: TagDeleteRequest<Sample> = TagDeleteRequest::default().groups(groups);
    // delete all tag rows for our groups + sample
    for (key, val_map) in &sample.tags {
        // add all values for this key
        tag_deletes.add_values_ref(key, val_map.keys());
    }
    // delete the requested tags for this repo if they exist
    super::tags::delete(&sample.sha256, &tag_deletes, shared).await?;
    // delete all comment rows for our groups + sample
    for comment in &sample.comments {
        // delete the rows for this comment
        delete_comment(&sample.sha256, groups, comment, shared).await?;
    }
    // prune comment attachments now that the comments are deleted
    prune_comment_attachments(&sample.comments, &sample.sha256, shared).await?;
    // create events to delete search info for this sample in the pruned groups
    let tag_event = TagSearchEvent::deleted::<Sample>(sample.sha256.clone(), groups.clone());
    let result_event = ResultSearchEvent::deleted::<Sample>(sample.sha256.clone(), groups.clone());
    if let Err(err) = tokio::try_join!(
        super::search::events::create(tag_event, shared),
        super::search::events::create(result_event, shared)
    ) {
        return internal_err!(format!("Failed to create search event: {err}"));
    }
    Ok(())
}

/// List samples for specific groups
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `dedupe` - Whether to dedupe submissions for the same sha256
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::list", skip(shared), err(Debug))]
pub async fn list(
    params: FileListParams,
    dedupe: bool,
    shared: &Shared,
) -> Result<ScyllaCursor<SampleListLine>, ApiError> {
    // get our cursor
    let mut cursor = ScyllaCursor::from_params(params, dedupe, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    Ok(cursor)
}

/// Gets details on a specific samples by group and sha256
///
/// # Arguments
///
/// * `groups` - The groups to search in
/// * `sha256` - The sha256 to retrieve submission docs for
/// * `shared` - Shared Thorium objects
/// * `req_id` - The uuid for this request
#[instrument(name = "db::files::list_details", skip(shared), err(Debug))]
pub async fn list_details(
    groups: &Vec<String>,
    sha256s: Vec<String>,
    shared: &Shared,
) -> Result<Vec<Sample>, ApiError> {
    // build a btreemap to store the order samples should be returned in
    let mut sorted: BTreeMap<DateTime<Utc>, Vec<String>> = BTreeMap::default();
    // build a hashmap to store our sha256s in
    let mut map = HashMap::with_capacity(sha256s.len());
    // split our groups and samples into chunks of 50
    for (sha256s_chunk, groups_chunk) in sha256s.chunks(50).cartesian_product(groups.chunks(50)) {
        // turn our chunks into vecs
        let groups_vec = groups_chunk.to_vec();
        let sha256s_vec = sha256s_chunk.to_vec();
        // send a query to get this chunks data
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.samples.get_many,
                (sha256s_vec, groups_vec),
            )
            .await?;
        // enable casting to types for this query
        let query_rows = query.into_rows_result()?;
        // crawl the rows returned by this query
        for cast in query_rows.rows::<SubmissionRow>()? {
            // get this row if possible
            let row = cast?;
            // get an entry to this submissions timestamp list
            let entry = sorted.entry(row.uploaded).or_default();
            // add this submission to the btreemap
            entry.push(row.sha256.clone());
            // try to get a mutable entry to this submissions sample
            match map.entry(row.sha256.clone()) {
                // we do not yet have an entry for this sha256
                Vacant(entry) => {
                    // create a sample entry for this submission
                    let sample = Sample::try_from(row)?;
                    // add this sample to the map
                    entry.insert(sample);
                }
                // we already have an entry for this submission
                Occupied(entry) => entry.into_mut().add_row(row)?,
            }
        }
    }
    // build our vec of sample details to return
    let mut details = Vec::with_capacity(map.len());
    // crawl the submissions in order and add them to our map
    for (_, sha256s) in sorted.into_iter().rev() {
        // crawl over the submission in this timestamp group
        for sha256 in sha256s {
            // try to pop this submission
            if let Some(mut sample) = map.remove(&sha256) {
                // get the tags for this sample
                sample.get_tags(&groups, shared).await?;
                // add this sample to our vec
                details.push(sample);
            }
        }
    }
    Ok(details)
}

/// Updates a submission object in scylla
///
/// # Arguments
///
/// * `sample` - The sample we are updating
/// * `update` - The update to apply to this submission object
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::update", skip_all, err(Debug))]
pub async fn update(
    sample: Sample,
    update: &SubmissionUpdate,
    shared: &Shared,
) -> Result<Sample, ApiError> {
    // log the sample we are updating
    event!(Level::INFO, sha256 = &sample.sha256);
    // get the submission to update if it exists
    if let Some(sub) = sample.submissions.iter().find(|sub| update.id == sub.id) {
        // serialize the new origin string
        let origin_str = sub.origin.serialize()?;
        // get the year and month this submission was originally added
        let year = sub.uploaded.year();
        // get the chunk size for files
        let chunk_size = shared.config.thorium.files.partition_size;
        // determine what bucket to add data too
        let bucket = helpers::partition(sub.uploaded, year, chunk_size);
        // update all of the submission rows for this submission
        for group in sub.groups.iter() {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.samples.insert,
                    (
                        group,
                        &year,
                        &bucket,
                        &sample.sha256,
                        &sample.sha1,
                        &sample.md5,
                        &sub.id,
                        &sub.name,
                        &sub.description,
                        &sub.submitter,
                        &origin_str,
                        sub.uploaded,
                    ),
                )
                .await?;
        }
        // if any groups are set to be removed then remove them
        if !update.remove_groups.is_empty() {
            delete_from_groups!(
                shared,
                update.remove_groups,
                year,
                bucket,
                sub.uploaded,
                sub.id
            );
        }
        Ok(sample)
    } else {
        not_found!(format!(
            "Submission {}:{} not found",
            sample.sha256, update.id
        ))
    }
}

/// Creates a comment for a specific sha256
///
/// # Arguments
///
/// * `user` - The user that is adding new comments
/// * `sha256` - The sha256 to comment on
/// * `form` - The comment to save to scylla
/// * `shared` - Shared objects in Thorium
#[instrument(
    name = "db::files::create_comment",
    skip(user, form, shared),
    err(Debug)
)]
pub async fn create_comment<'v>(
    user: &User,
    sha256: &str,
    form: &CommentForm,
    shared: &Shared,
) -> Result<(), ApiError> {
    // serialize our s3 paths
    let paths = serialize!(&form.attachments);
    // get the current timestamp
    let now = Utc::now();
    // create a comment row for each group
    for group in form.groups.iter() {
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.comments.insert,
                (
                    &group,
                    &sha256,
                    now,
                    &form.id,
                    &user.username,
                    &form.comment,
                    &paths,
                ),
            )
            .await?;
    }
    Ok(())
}

/// Prunes the attachments for the given list of comments if needed
///
/// Attachments are only pruned if the comment is no longer reachable (does not exist)
///
/// # Arguments
///
/// * `comments` - The comments to prune attachments for
/// * `sha256` - The SHA256 of the associated file
/// * `shared` - Shared Thorium objects
pub async fn prune_comment_attachments(
    comments: &[Comment],
    sha256: &str,
    shared: &Shared,
) -> Result<(), ApiError> {
    // downselect down to only those comments that have attachments
    let has_attachments = comments
        .iter()
        .filter(|comment| !comment.attachments.is_empty())
        .map(|comment| &comment.id)
        .collect::<Vec<&Uuid>>();
    // determine which comment attachments are still reachable (the comments exist)
    let reachable = comments_exist(&has_attachments, shared).await?;
    // downselect to comments that are unreachable
    let unreachable = comments
        .iter()
        .filter(|comment| !comment.attachments.is_empty())
        .filter(|comment| !reachable.contains(&comment.id));
    // crawl through and delete all of our comment attachments
    for comment in unreachable {
        // delete all of this comments files
        for s3_id in comment.attachments.values() {
            // build the path to save this attachment at in s3
            let s3_path = format!("{}/{}/{}", sha256, &comment.id, s3_id);
            shared.s3.attachments.delete(&s3_path).await?;
        }
    }
    Ok(())
}

/// Determines which of the given comments still exist in Thorium
///
/// # Arguments
///
/// * `ids` - The comment IDs to check for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::comment_exists", skip(shared), err(Debug))]
async fn comments_exist(ids: &[&Uuid], shared: &Shared) -> Result<HashSet<Uuid>, ApiError> {
    // build our hashset of comments
    let mut found = HashSet::with_capacity(ids.len());
    // break these IDs into chunks of 100
    for ids_chunk in ids.chunks(100) {
        // execute this query
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.comments.exists, (ids_chunk,))
            .await?;
        // enable casting to types for this query
        let query_rows = query.into_rows_result()?;
        // get the rows from this query if they were returned
        for cast in query_rows.rows::<(Option<Uuid>,)>()? {
            // raise instead of just logging errors as ignoring them can cause
            // dangling references in the Db
            let (id_opt,) = cast?;
            // skip any rows without IDs
            if let Some(id) = id_opt {
                // track that this comment id was still found
                found.insert(id);
            }
        }
    }
    Ok(found)
}

/// Deletes a comment for a specific sha256
///
/// This doesn't delete any of the comments attachments.
///
/// # Arguments
///
/// * `sha256` - The sha256 to delete the comment from
/// * `groups` - The groups to delete this comment for
/// * `comment` - The comment to delete
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::files::delete_comment", skip(comment, shared), err(Debug))]
pub async fn delete_comment(
    sha256: &str,
    groups: &[String],
    comment: &Comment,
    shared: &Shared,
) -> Result<(), ApiError> {
    // log the comment we are deleting
    event!(Level::INFO, comment = &comment.id.to_string());
    // delete this comment for each of our target groups
    for group in groups {
        // delete this tag row
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.comments.delete,
                (group, sha256, comment.uploaded, &comment.id),
            )
            .await?;
    }
    Ok(())
}
