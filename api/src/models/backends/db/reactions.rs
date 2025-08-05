use async_recursion::async_recursion;
use bb8_redis::redis::cmd;
use chrono::prelude::*;
use futures::future::try_join_all;
use futures::stream::{self, StreamExt};
use scylla::DeserializeRow;
use std::collections::HashMap;
use tracing::{event, instrument, span, Level, Span};
use uuid::Uuid;

use super::keys::{logs, ImageKeys, JobKeys, ReactionKeys, StreamKeys, SubReactionLists};
use super::{images, jobs, pipelines, streams};
use crate::models::{
    BulkReactionResponse, Group, JobHandleStatus, JobList, JobResetRequestor, JobResets, Pipeline,
    RawJob, Reaction, ReactionActions, ReactionExpire, ReactionList, ReactionRequest,
    ReactionStatus, StageLogs, StageLogsAdd, StatusRequest, StatusUpdate, SystemComponents, User,
};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, cast, conflict, conn, force_serialize, log_err, log_scylla_err, query, serialize,
};

/// build created status update from a reaction
macro_rules! status_create {
    ($cast:expr) => {
        StatusUpdate::new(
            StatusRequest::from_reaction($cast, ReactionActions::Created),
            None,
        )
    };
}

/// build created status update from a reaction
macro_rules! status_complete {
    ($cast:expr) => {
        StatusUpdate::new(
            StatusRequest::from_reaction($cast, ReactionActions::Completed),
            None,
        )
    };
}

/// build the pipeline cache key
macro_rules! pipe_key {
    ($reaction:expr) => {
        format!("{}:{}", $reaction.group, $reaction.pipeline)
    };
}

/// Builds a [`redis::Pipeline`] with commands to create a [`Reaction`] in Redis
///
/// # Arguments
///
/// * `pipe` - The Redis [`redis::Pipeline`] to build on top of
/// * `cast` - The [`Reaction`] to create in Redis
/// * `pipeline` - The [`Pipeline`] this [`Reaction`] is based on
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(
    name = "db::reactions::get_parent_ephemeral",
    skip_all,
    err(Debug)
)]
pub async fn build<'a>(
    pipe: &'a mut redis::Pipeline,
    cast: Reaction,
    pipeline: &Pipeline,
    shared: &Shared,
) -> Result<(Reaction, JobHandleStatus), ApiError> {
    // build reaction data keys
    let keys = ReactionKeys::new(&cast, shared);
    // build reaction status update
    let update = status_create!(&cast);
    // get the timestamp for this reaction
    let timestamp = cast.sla.timestamp();
    // create reaction
    let pipe = pipe
        // set reaction data
        .cmd("hsetnx").arg(&keys.data).arg("id").arg(cast.id.to_string())
        .cmd("hsetnx").arg(&keys.data).arg("group").arg(&cast.group)
        .cmd("hsetnx").arg(&keys.data).arg("pipeline").arg(&cast.pipeline)
        .cmd("hsetnx").arg(&keys.data).arg("creator").arg(&cast.creator)
        .cmd("hsetnx").arg(&keys.data).arg("status").arg(serialize!(&cast.status))
        .cmd("hsetnx").arg(&keys.data).arg("current_stage").arg(cast.current_stage)
        .cmd("hsetnx").arg(&keys.data).arg("current_stage_progress").arg(0)
        .cmd("hsetnx").arg(&keys.data).arg("current_stage_length").arg(cast.current_stage_length)
        .cmd("hsetnx").arg(&keys.data).arg("args").arg(serialize!(&cast.args))
        .cmd("hsetnx").arg(&keys.data).arg("sla").arg(serialize!(&cast.sla))
        .cmd("hsetnx").arg(&keys.data).arg("tags").arg(serialize!(&cast.tags))
        .cmd("hsetnx").arg(&keys.data).arg("sub_reactions").arg(cast.sub_reactions)
        .cmd("hsetnx").arg(&keys.data).arg("completed_sub_reactions")
            .arg(cast.completed_sub_reactions)
        .cmd("hsetnx").arg(&keys.data).arg("samples").arg(serialize!(&cast.samples))
        .cmd("hsetnx").arg(&keys.data).arg("ephemeral").arg(serialize!(&cast.ephemeral))
        .cmd("hsetnx").arg(&keys.data).arg("parent_ephemeral").arg(serialize!(&cast.parent_ephemeral))
        .cmd("hsetnx").arg(&keys.data).arg("repos").arg(serialize!(&cast.repos))
        // add to specific status set
        .cmd("sadd").arg(&ReactionKeys::status(&cast.group, &cast.pipeline, &cast.status, shared))
            .arg(&cast.id.to_string())
        // add this reaction to the specific group/pipeline/stage set
        .cmd("sadd").arg(&keys.set).arg(&cast.id.to_string())
        // add this reaction to the group wide sorted set
        .cmd("zadd").arg(&keys.group_set).arg(timestamp).arg(&cast.id.to_string())
        // push the reaction create status log update
        .cmd("rpush").arg(&keys.logs).arg(serialize!(&update));
    // if a parent was set then set that too
    if let Some(parent) = cast.parent.as_ref() {
        // get key to parent reactions sub set and sub status set
        let sub_key = ReactionKeys::sub_set(&cast.group, parent, shared);
        let status_key = ReactionKeys::sub_status_set(&cast.group, parent, &ReactionStatus::Created, shared);
        let parent_data = ReactionKeys::data(&cast.group, parent, shared);
        // set  parent field
        pipe.cmd("hsetnx").arg(&keys.data).arg("parent").arg(serialize!(&parent))
            // increment our parent sub reaction counter
            .cmd("hincrby").arg(&parent_data).arg("sub_reactions").arg(1)
            // add sub reaction to sub reaction set
            .cmd("sadd").arg(&sub_key).arg(cast.id.to_string())
            // add sub reaction to sub reaction status set
            .cmd("sadd").arg(&status_key).arg(cast.id.to_string());
    }
    // set our trigger depth info if it needs to be set
    if let Some(trigger_depth) = cast.trigger_depth {
        // set our trigger depth
        pipe.cmd("hsetnx").arg(&keys.data).arg("trigger_depth").arg(trigger_depth);
    }
    // add to any required tag lists
    let pipe = cast.tags.iter()
        .fold(pipe, |pipe, tag|
            pipe.cmd("sadd").arg(ReactionKeys::tag(&cast.group, tag, shared))
                .arg(&cast.id.to_string()));
    // create initial jobs
    react(pipe, pipeline, cast, shared).await
}

/// Recursively crawl parent reactions and build a list of their ephemeral files
///
/// # Arguments
///
/// * `group` - The name of the group our reactions are from
/// * `reaction` - The reaction to start our crawling from
/// * `map` - A map of ephemeral files and what reaction its from
/// * `shared` - Shared Thorium objects
#[async_recursion::async_recursion]
#[instrument(
    name = "db::reactions::get_parent_ephemeral",
    skip(map, shared),
    err(Debug)
)]
pub async fn get_parent_ephemeral(
    group: &str,
    reaction: &Option<Uuid>,
    mut map: HashMap<String, Uuid>,
    shared: &Shared,
) -> Result<HashMap<String, Uuid>, ApiError> {
    // get our parent reactions id if it has one
    if let Some(id) = reaction.as_ref() {
        // this reaction has a parent so get its info
        let parent = get(group, id, shared).await?;
        // add any of our parents ephemeral files to our map of ephemeral files
        for ephemeral in parent.ephemeral {
            // add this id and the parent reaction that's tied to it
            map.insert(ephemeral, *id);
        }
        // continue to recursively crawl any parent reactions
        get_parent_ephemeral(&parent.group, &parent.parent, map, shared).await
    } else {
        // this reaction has no parent return our list
        Ok(map)
    }
}

/// Creates a [`Reaction`] in redis
///
/// # Arguments
///
/// * `user` - The [`User`] creating this [`Reaction`]
/// * `request` - The [`ReactionRequest`] to add to the backend
/// * `pipeline` - The [`Pipeline`] this [`Reaction`] is based on
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::reactions::create", skip_all, err(Debug))]
pub async fn create(
    user: &User,
    request: ReactionRequest,
    pipeline: &Pipeline,
    shared: &Shared,
) -> Result<Reaction, ApiError> {
    // get any ephemeral files from any parent reactions
    let map = HashMap::default();
    let ephemeral = get_parent_ephemeral(&request.group, &request.parent, map, shared).await?;
    // cast to a reaction
    let (cast, _) = request.cast(user, pipeline, ephemeral, shared).await?;
    // build reaction creation pipeline
    let mut pipe = redis::pipe();
    let (reaction, _) = build(&mut pipe, cast.clone(), pipeline, shared).await?;
    // create reaction along with its jobs in redis
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(reaction)
}

/// Creates [`Reaction`]s in Redis in bulk
///
/// # Arguments
///
/// * `user` - The [`User`] creating these [`Reaction`]s
/// * `requests` - The [`Reaction`] requests to add to the backend
/// * `pipe_cache` - The pipeline cache these [`Reaction`]s are based on
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::reactions::create_bulk", skip_all, err(Debug))]
pub async fn create_bulk(
    user: &User,
    requests: Vec<ReactionRequest>,
    pipe_cache: &HashMap<String, Pipeline>,
    shared: &Shared,
) -> Result<BulkReactionResponse, ApiError> {
    // build a vector to store all of our casted reactions
    let mut casts: Vec<(Reaction, &Pipeline)> = Vec::with_capacity(requests.len());
    // build a response object allocated to the right size
    let mut response = BulkReactionResponse::with_capacity(requests.len());
    // try to cast all of our requests to a reaction
    for (index, req) in requests.into_iter().enumerate() {
        // get any ephemeral files from any parent reactions
        let map = HashMap::default();
        let ephemeral = get_parent_ephemeral(&req.group, &req.parent, map, shared).await?;
        // get a reference to pipeline data and request as a tuple
        if let Some(pipeline) = pipe_cache.get(&pipe_key!(req)) {
            // cast this request to a full reaction
            match req.cast(user, pipeline, ephemeral, shared).await {
                // we don't continue to track the index because any errors past this point
                // can lead to malformed redis command pipelines and so are fatal. These
                // errors should never occur though and when they are it likely means that
                // all redis operations will fail.
                Ok(cast) => casts.push(cast),
                Err(error) => {
                    // log this error
                    event!(Level::ERROR, error = error.to_string());
                    // add this error to our response
                    response.errors.insert(index, error.to_string());
                }
            }
        }
    }
    // build all reactions
    let mut pipe = redis::pipe();
    for (cast, pipeline) in casts.iter() {
        // add this reaction to our redis pipeline
        let (reaction, _) = build(&mut pipe, cast.clone(), pipeline, shared).await?;
        // add this newly created reactions id to our response object
        response.created.push(reaction.id)
    }
    // create all reactions along with their jobs in redis
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    // log how many reactions were created and how many ran into errors
    event!(
        Level::INFO,
        created = response.created.len(),
        errors = response.errors.len()
    );
    Ok(response)
}

/// Gets a reaction from the backend
///
/// # Arguments
///
/// * `group` - The group this reaction is from
/// * `id` - The id of the reaction
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::reactions::get", skip_all, err(Debug))]
pub async fn get(
    group: &str,
    id: &Uuid,
    shared: &Shared,
) -> Result<Reaction, ApiError> {
    // build key to reaction data
    let data_key = ReactionKeys::data(group, id, shared);
    let jobs_key = ReactionKeys::jobs(group, id, shared);
    let gen_key = ReactionKeys::generators(group, id, shared);
    // get reaction data
    let (raw, jobs, gens): (HashMap<String, String>, Vec<String>, Vec<String>) = redis::pipe()
        .cmd("hgetall").arg(&data_key)
        .cmd("smembers").arg(&jobs_key)
        .cmd("smembers").arg(&gen_key)
        .query_async(conn!(shared))
        .await?;
    // cast to reaction
    // return 404 if no data was retrieved
    let reaction = Reaction::try_from((raw, jobs, gens))?;
    Ok(reaction)
}

/// Lists reaction IDs in redis for a group
///
/// # Arguments
///
/// * `group` - The group this pipeline is in
/// * `pipeline` - The pipeline to list reactions for
/// * `cursor` - The cursor to use when paging through reactions
/// * `limit` - The number of reactions to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list(
    group: &str,
    pipeline: &str,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ReactionList, ApiError> {
    // get key to reaction set for this group/pipeline
    let key = ReactionKeys::set(group, pipeline, shared);
    // get list of reactions
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(ReactionList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(ReactionList::new(Some(new_cursor), names))
    }
}

/// Lists reaction IDs with a status in redis for a group
///
/// # Arguments
///
/// * `group` - The group this pipeline is in
/// * `pipeline` - The pipeline to list reactions for
/// * `status` - The status that listed reactions must have
/// * `cursor` - The cursor to use when paging through reactions
/// * `limit` - The number of reactions to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list_status(
    group: &str,
    pipeline: &str,
    status: &ReactionStatus,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ReactionList, ApiError> {
    // get key to reaction set for this group/pipeline
    let key = ReactionKeys::status(group, pipeline, status, shared);
    // get list of reactions
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(ReactionList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(ReactionList::new(Some(new_cursor), names))
    }
}

/// Lists reaction IDs by a tag
///
/// # Arguments
///
/// * `group` - The group this pipeline is in
/// * `tag` - The tag to list reactions for
/// * `cursor` - The cursor to use when paging through reactions
/// * `limit` - The number of reactions to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list_tag(
    group: &str,
    tag: &str,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ReactionList, ApiError> {
    // get key to reaction set for this group/pipeline
    let key = ReactionKeys::tag(group, tag, shared);
    // get list of reactions
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("count").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(ReactionList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(ReactionList::new(Some(new_cursor), names))
    }
}

/// Lists reaction IDs in the group wide status sorted set
///
/// # Arguments
///
/// * `group` - The group this pipeline is in
/// * `status` - The status that listed reactions must have
/// * `cursor` - The cursor to use when paging through reactions
/// * `limit` - The number of reactions to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn list_group_set(
    group: &str,
    status: &ReactionStatus,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ReactionList, ApiError> {
    // get key to reaction set for this group/pipeline
    let key = ReactionKeys::group_set(group, status, shared);
    // use a arbitrarily large value to attempt to read to
    let end: i64 = 999_999_999_999;
    // read the group status sorted set
    let data: Vec<(String, i64)> = query!(
        cmd("zrevrangebyscore").arg(key).arg(end).arg(cursor)
            .arg("LIMIT").arg(0).arg(limit).arg("WITHSCORES"),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if data.is_empty() {
        Ok(ReactionList::new(None, Vec::default()))
    } else {
        // more groups exist use new_cursor
        let cursor = data.last().unwrap().clone().1 as usize;
        // collapse to just the IDs
        let ids = data.into_iter().map(|pair| pair.0).collect();
        Ok(ReactionList::new(Some(cursor), ids))
    }
}

/// Lists sub reaction IDs in redis for a reaction
///
/// # Arguments
///
/// * `group` - The group this pipeline is in
/// * `reaction` - The reaction to list sub reactions for
/// * `cursor` - The cursor to use when paging through reactions
/// * `limit` - The number of reactions to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list_sub(
    group: &str,
    reaction: &Uuid,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ReactionList, ApiError> {
    // get key to reaction set for this group/pipeline
    let key = ReactionKeys::sub_set(group, reaction, shared);
    // get list of reactions
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(ReactionList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(ReactionList::new(Some(new_cursor), names))
    }
}

/// Lists sub reaction IDs in redis for a reaction
///
/// # Arguments
///
/// * `group` - The group this pipeline is in
/// * `reaction` - The reaction to list sub reactions for
/// * `status` - The status of sub reactions to list
/// * `cursor` - The cursor to use when paging through reactions
/// * `limit` - The number of reactions to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list_sub_status(
    group: &str,
    reaction: &Uuid,
    status: &ReactionStatus,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ReactionList, ApiError> {
    // get key to reaction set for this group/pipeline
    let key = ReactionKeys::sub_status_set(group, reaction, status, shared);
    // get list of reactions
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(ReactionList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(ReactionList::new(Some(new_cursor), names))
    }
}

/// A single redis response for reaction details
type ReactionData = (HashMap<String, String>, Vec<String>, Vec<String>);

/// Lists all reactions details in the redis backend for a group
///
/// # Arguments
///
/// * `group` - The group to retrieve reactions for
/// * `ids` - The reaction IDs to get details on
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::reactions::list_details", skip(ids, shared), err(Debug))]
pub async fn list_details(
    group: &str,
    ids: &[String],
    shared: &Shared,
) -> Result<Vec<Reaction>, ApiError> {
    // get reaction data
    let raw: Vec<ReactionData> = ids.iter()
        .fold(redis::pipe().atomic(), |pipe, id|
            pipe
                .cmd("hgetall").arg(&ReactionKeys::data_str(group, id, shared))
                .cmd("smembers").arg(&ReactionKeys::jobs_str(group, id, shared))
                .cmd("smembers").arg(&ReactionKeys::generators_str(group, id, shared)))
        .query_async(conn!(shared)).await?;
    // cast to reaction structs
    let reactions = cast!(raw, Reaction::try_from);
    Ok(reactions)
}

/// Finds the cost of executing the current stages and the stages after it
///
/// The cost will be retuned as a tuple containing the costs of the next
/// stages to execute and the total cost of all stages after that.
///
/// # Arguments
///
/// * `group` - The group this reaction is in
/// * `stages` - The images left to execute
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
async fn cost(
    group: &str,
    stages: &[Vec<String>],
    shared: &Shared
) -> Result<(Vec<f64>, f64), ApiError> {
    // get the cost of the stages to execute   
    let mut costs: Vec<f64> = stages.iter()
        .flatten()
        .fold(redis::pipe().atomic(), |pipe, stage|
            pipe.cmd("hget").arg(&ImageKeys::data(group, stage, shared)).arg("runtime"))
        .query_async(conn!(shared)).await?;

    // short circuit if there are no stages after this
    if costs.len() == stages[0].len() {
        Ok((costs, 0.0))
    } else {
        // get the cost of each of the stages to execute next
        let next = costs.drain(..stages[0].len()).collect();
        // get cost of all the other stages left to execute
        let rest = costs.iter()
            .fold(0.0, |accum, cost| accum + cost);
        Ok((next, rest))
    }
}

/// Adds command to increment a parent reaction sub reaction counter
///
/// # Arguments
///
/// * `reaction` - The reaction to check for a parent reaction
/// * `pipe` - The Redis [`redis::Pipeline`] to build on top of
/// * `shared` - Shared Thorium objects
fn incr_parent<'a>(reaction: &Reaction, pipe: &'a mut redis::Pipeline, shared: &Shared) {
    // check if we have a parent reaction
    if let Some(parent) = reaction.parent.as_ref() {
        // build key to our parent reactions data
        let parent_data = ReactionKeys::data(&reaction.group, parent, shared);
        // build key to the our sub reaction status sets
        // the old status is always Started because we cannot incr a parent from any other status
        let old_status =
            ReactionKeys::sub_status_set(&reaction.group, parent, &ReactionStatus::Started, shared);
        let new_status =
            ReactionKeys::sub_status_set(&reaction.group, parent, &reaction.status, shared);
        // move from old status list to new status list
        pipe.cmd("srem")
            .arg(old_status)
            .arg(&reaction.id.to_string())
            .cmd("sadd")
            .arg(new_status)
            .arg(&reaction.id.to_string())
            // increment our parents completed reactions by one
            .cmd("hincrby")
            .arg(&parent_data)
            .arg("completed_sub_reactions")
            .arg(1)
            .cmd("hget")
            .arg(&parent_data)
            .arg("sub_reactions");
    }
}

/// Adds an expire command to a redis pipeline
macro_rules! add_expire {
    ($pipe:expr, $expire:expr, $cmd:expr, $key:expr, $id:expr, $shared:expr) => {
        $pipe
            .cmd("zadd")
            .arg(StreamKeys::system_global("expire", $shared))
            .arg($expire)
            .arg(force_serialize!(&ReactionExpire::new($cmd, $key, $id)))
    };
}

/// Adds expire commands to a redis pipeline and sets TTL values in scylla
///
/// # Arguments
///
/// * `pipe` - The redis [`redis::Pipeline`] to build commands on top of
/// * `reaction` - The [`Reaction`] to create jobs for
/// * `keys` - The keys to this reactions dat
/// * `dest` - The destination group status set this is being moved to
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
fn build_expire<'a>(
    pipe: &'a mut redis::Pipeline,
    reaction: &Reaction,
    keys: &ReactionKeys,
    dest: &str,
    shared: &Shared,
) -> &'a mut redis::Pipeline {
    // get time when we should expire things out of reaction status list
    let expiration =
        chrono::Utc::now() + chrono::Duration::seconds(shared.config.thorium.retention.data as i64);
    let expiration = expiration.timestamp();
    // add command to expire out of the destination set
    add_expire!(pipe, expiration, "srem", dest, &reaction.id, shared);
    // build key to reaction set for this group/pipeline
    let set_key = ReactionKeys::set(&reaction.group, &reaction.pipeline, shared);
    // add command to expire out of the pipeline set
    add_expire!(pipe, expiration, "srem", &set_key, &reaction.id, shared);
    // build key to reaction set for this group
    let group_key = ReactionKeys::group_set(&reaction.group, &reaction.status, shared);
    // add command to expire out of the group status set
    add_expire!(pipe, expiration, "zrem", &group_key, &reaction.id, shared);
    // build key to sub reaction lists
    let sub_reacts = SubReactionLists::new(reaction, shared);
    // add command to expire all tags
    let pipe = reaction.tags.iter().fold(pipe, |pipe, tag| {
        add_expire!(
            pipe,
            expiration,
            "srem",
            ReactionKeys::tag(&reaction.group, tag, shared),
            &reaction.id,
            shared
        )
    });
    // push expire objects for all lists
    pipe.cmd("expire").arg(&keys.data).arg(shared.config.thorium.retention.data)
        .cmd("expire").arg(&keys.jobs).arg(shared.config.thorium.retention.data)
        .cmd("expire").arg(&keys.logs).arg(shared.config.thorium.retention.data)
        .cmd("expire")
        .arg(&keys.sub)
        .arg(shared.config.thorium.retention.data)
        // expire all sub reaction status lists
        .cmd("expire").arg(&sub_reacts.created).arg(shared.config.thorium.retention.data)
        .cmd("expire").arg(&sub_reacts.started).arg(shared.config.thorium.retention.data)
        .cmd("expire").arg(&sub_reacts.completed).arg(shared.config.thorium.retention.data)
        .cmd("expire").arg(&sub_reacts.failed).arg(shared.config.thorium.retention.data)
}

/// Completes a [`Reaction`]
///
/// This will set all jobs and reaction data to expire based on the
/// configured retention time.
///
/// # Arguments
///
/// * `pipe` - The redis [`redis::Pipeline`] to build commands on top of
/// * `reaction` - The [`Reaction`] to create jobs for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn complete<'a>(
    pipe: &'a mut redis::Pipeline,
    mut reaction: Reaction,
    shared: &Shared
) -> Result<Reaction, ApiError> {  
    // build keys to this reactions data
    let keys = ReactionKeys::new(&reaction, shared);
    // build key to pipeline reaction src status set
    let src = ReactionKeys::status(&reaction.group, &reaction.pipeline, &reaction.status, shared);
    // update status
    reaction.status = ReactionStatus::Completed;
    // build status log for completing this reaction
    let update = status_complete!(&reaction);
    // build key to pipeline reaction dest status set
    let dest = ReactionKeys::status(&reaction.group, &reaction.pipeline, &reaction.status, shared);
    // push in our expire orders
    let pipe = build_expire(pipe, &reaction, &keys, &dest, shared);
    // get the timestamp for this reactions sla
    let timestamp = reaction.sla.timestamp();
    // update reaction
    pipe.cmd("hset").arg(&keys.data).arg("status").arg(&force_serialize!(&reaction.status))
        .cmd("hset").arg(&keys.data).arg("current_stage").arg(reaction.current_stage)
        .cmd("rpush").arg(logs::queue_name(&update, shared)).arg(force_serialize!(&update))
        .cmd("srem").arg(src).arg(&reaction.id.to_string())
        .cmd("sadd").arg(dest).arg(&reaction.id.to_string())
        // move from started group set to completed group set
        .cmd("zrem").arg(&ReactionKeys::group_set(&reaction.group, &ReactionStatus::Started, shared))
            .arg(&reaction.id.to_string())
        .cmd("zadd").arg(&ReactionKeys::group_set(&reaction.group, &reaction.status, shared))
            .arg(timestamp).arg(&reaction.id.to_string()); 
    // crawl over the jobs for this reaction and expire all of their data
    let mut cursor = 0;
    loop {
        // 200 jobs to expire at at time
        let jobs = list_jobs(&reaction, cursor, 200, shared).await?;
        // expire all the data for these jobs
        for id in jobs.names.iter(){
            let key = JobKeys::data(id, shared);
            pipe.cmd("expire").arg(key).arg(shared.config.thorium.retention.data);
        }
        // check if we have expired all jobs
        if jobs.cursor.is_none() {
            break;
        }
        // if we haven't expired all jobs then set new cursor
        cursor = jobs.cursor.unwrap();
    }
    // handle parent reaction incrementing if we have a parent
    incr_parent(&reaction, pipe, shared);
    // try to delete any ephemeral files
    delete_ephemeral(&reaction, shared).await?;
    Ok(reaction)
}

/// Handles the creation of jobs for the current stage of the reaction
///
/// If this is the last stage in a reaction it will complete it.
///
/// # Arguments
///
/// * `pipe` - The redis [`redis::Pipeline`] to build commands on top of
/// * `pipeline` - The [`Pipeline`] this [`Reaction`] is built around
/// * `reaction` - The [`Reaction`] to create jobs for
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[rustfmt::skip]
#[instrument(name = "db::reactions::react", skip_all, err(Debug))]
pub async fn react<'a>(
    pipe: &'a mut redis::Pipeline,
    pipeline: &Pipeline,
    mut reaction: Reaction,
    shared: &Shared,
) -> Result<(Reaction, JobHandleStatus), ApiError> {
    // set status to complete if reaction has completed its final stage
    if reaction.current_stage as usize > pipeline.order.len() - 1 {
        // complete reaction and set the expire time on its data
        let reaction = complete(pipe, reaction, shared).await?;
        return Ok((reaction, JobHandleStatus::Completed));
    }

    // get stages to launch
    let stages = &pipeline.order[reaction.current_stage as usize];
    reaction.current_stage_length = stages.len() as u64;
    reaction.current_stage_progress = 0;
    // get cost of stages to left to execute and the next stages to execute
    let (next, rest) = cost(&pipeline.group, &pipeline.order[reaction.current_stage as usize..], shared).await?;

    // get the image info on all required images
    let info = images::job_info(&pipeline.group, stages, shared).await?;
    // launch all sub stages
    for (index, sub) in stages.iter().enumerate() {
        // calculate cost to execute this job
        let cost = *next.get(index).unwrap_or(&600.0);
        // get the timestamp we need to start this job by in order to meet the SLA
        let deadline = reaction.sla - chrono::Duration::seconds((cost + rest).ceil() as i64);
        // build a raw job object for this stage
        let cast: RawJob = RawJob::build(&reaction, sub, deadline, &info).await?;
        // add job build command onto our redis pipeline
        jobs::build(pipe, &cast, shared).await?;
    }

    // update reaction data
    let key = ReactionKeys::data(&reaction.group, &reaction.id, shared);
    pipe.cmd("hset").arg(&key).arg("current_stage").arg(reaction.current_stage)
        .cmd("hset").arg(&key).arg("current_stage_length").arg(reaction.current_stage_length)
        .cmd("hset").arg(&key).arg("current_stage_progress").arg(reaction.current_stage_progress);
    // build return message
    Ok((reaction, JobHandleStatus::Proceeding))
}

/// Checks if a reaction has a set status and returns conflict error if it does
macro_rules! status_guard {
    ($react:expr, $status:expr) => {
        if ($react.status == $status) {
            return conflict!(format!("reaction {} is already {}", &$react.id, $status));
        }
    };
}

/// Proceeds with the parent reaction if all sub reactions have been completed
///
/// # Arguments
///
/// * `reaction` - The sub [`Reaction`] to get the parent for
/// * `progress` - The progress returns from the react function
/// * `shared` - Shared Thorium objects
#[async_recursion]
#[instrument(name = "db::reactions::parent_proceed", skip_all, err(Debug))]
async fn parent_proceed(
    reaction: &Reaction,
    progress: Vec<u64>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // check if we have parent reaction
    if let Some(parent) = reaction.parent {
        // log this parent id
        event!(Level::INFO, parent = parent.to_string());
        // check if our parent reaction has completed all of its sub reactions
        // progress should contain a list of u64's where the last two are
        // completed_sub_reactions and sub_reactions
        let len = progress.len();
        if progress[len - 2] == progress[len - 1] {
            // get our parent reaction
            let parent = get(&reaction.group, &parent, shared).await?;
            // check if we should proceed with our parent or not
            if parent.current_stage_progress == parent.current_stage_length
                && parent.status == ReactionStatus::Started
            {
                proceed(parent, shared).await?;
            }
        }
    }
    Ok(())
}

/// Increments a [`Reaction`]s progress and creates jobs for the next stage
///
/// # Arguments
///
/// * `reaction` - The [`Reaction`] to proceed with
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::reactions::proceed", skip_all, err(Debug))]
pub async fn proceed(mut reaction: Reaction, shared: &Shared) -> Result<JobHandleStatus, ApiError> {
    // check if reaction is already complete
    status_guard!(reaction, ReactionStatus::Completed);
    // if we have pending sub reactions then short circuit and wait
    if reaction.sub_reactions > reaction.completed_sub_reactions {
        return Ok(JobHandleStatus::Waiting);
    }

    // check if we any active generators
    if reaction.generators.is_empty() {
        // either all generators completed or we never had any
        // so increment current stage
        reaction.current_stage += 1;
        // get pipeline data
        let pipeline = pipelines::get(&reaction.group, &reaction.pipeline, shared).await?;
        // create the current stages jobs
        let mut pipe = redis::pipe();
        let (reaction, status) = react(&mut pipe, &pipeline, reaction, shared).await?;
        // execute pipeline
        let progress: Vec<u64> = pipe.atomic().query_async(conn!(shared)).await?;
        // we only try to proceed parent reactions if we are completing our current reaction
        if status == JobHandleStatus::Completed {
            // if we have a parent reaction then proceed it
            parent_proceed(&reaction, progress, shared).await?;
        }
        Ok(status)
    } else {
        // get the scalers for all our jobs
        let scaler_map = jobs::get_scalers(reaction.generators.clone(), shared).await?;
        // reset each scalers jobs
        for (scaler, jobs) in scaler_map {
            // we have active generators for this scaler so reset them
            let resets = JobResets {
                scaler,
                requestor: JobResetRequestor::Component(SystemComponents::Api),
                reason: "Generator Reset".to_owned(),
                jobs,
            };
            // reset the jobs for our active generators
            jobs::bulk_reset(resets, shared).await?;
        }
        // respond that we are waiting since we reset our generators
        Ok(JobHandleStatus::Waiting)
    }
}

/// Deletes any ephemeral files in s3 tied to a specific reaction
///
/// # Arguments
///
/// * `reaction` - The reaction to delete ephemeral files from
/// * `shared` - Shared Thorium objects
async fn delete_ephemeral(reaction: &Reaction, shared: &Shared) -> Result<(), ApiError> {
    // crawl over any ephemeral files and delete them
    for name in reaction.ephemeral.iter() {
        // build the path to this file
        let path = format!("{}/{}", reaction.id, name);
        // delete this ephemeral file
        shared.s3.ephemeral.delete(&path).await?;
    }
    Ok(())
}

/// Fails a reaction
///
/// # Arguments
///
/// * `reaction` - The reaction to fail
/// * `shared` - Shared Thorium objects
/// * `span` - The span to trace logs under
#[rustfmt::skip]
#[instrument(name = "db::reactions::fail", skip_all, err(Debug))]
pub async fn fail(
    mut reaction: Reaction,
    shared: &Shared,
) -> Result<ReactionStatus, ApiError> {
    // check if reaction is already failed
    status_guard!(reaction, ReactionStatus::Failed);
    // build key to pipeline reaction src status set
    let src = ReactionKeys::status(
        &reaction.group,
        &reaction.pipeline,
        &reaction.status,
        shared,
    );
    // set status to failed
    reaction.status = ReactionStatus::Failed;
    // build key to pipeline reaction failed status set
    let dest = ReactionKeys::status(
        &reaction.group,
        &reaction.pipeline,
        &reaction.status,
        shared,
    );
    // build reaction data keys
    let keys = ReactionKeys::new(&reaction, shared);
    // start build redis pipeline for failing this reaction
    let mut pipe = redis::pipe();
    // add expire commands for this failed reaction 
    let pipe = build_expire(&mut pipe, &reaction, &keys, &dest, shared);
    // get the timestamp for this reactions sla
    let timestamp = reaction.sla.timestamp();
    pipe.cmd("hset").arg(&keys.data).arg("status").arg(serialize!(&reaction.status))
        // move from src status to failed status set
        .cmd("srem").arg(src).arg(&reaction.id.to_string())
        .cmd("sadd").arg(dest).arg(&reaction.id.to_string())
        // move from started group set to failed group set
       .cmd("zrem").arg(&ReactionKeys::group_set(&reaction.group, &ReactionStatus::Started, shared))
            .arg(&reaction.id.to_string())
       .cmd("zadd").arg(&ReactionKeys::group_set(&reaction.group, &reaction.status, shared))
            .arg(timestamp).arg(&reaction.id.to_string());

    // create status log for failing out this reaction
    let update_cast = StatusUpdate::new(
        StatusRequest::from_reaction(&reaction, ReactionActions::Failed),
        None,
    );
    super::logs::build(pipe, &[update_cast], shared)?;
    // handle parent reaction incrementing if we have a parent
    incr_parent(&reaction, pipe, shared);
    // execute redis pipeline
    let progress: Vec<u64> = pipe.atomic().query_async(conn!(shared)).await?;
    // proceed with our parent reactions
    parent_proceed(&reaction, progress, shared).await?;
    // try to delete any ephemeral files
    delete_ephemeral(&reaction, shared).await?;
    Ok(ReactionStatus::Failed)
}

/// Saves stage logs into scylla
///
/// # Arguments
///
/// * `reaction` - The reaction to save stage logs for
/// * `stage` - The stage to save logs for
/// * `logs` - The logs to save into scylla
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::reactions::add_stage_logs", skip_all, err(Debug))]
pub async fn add_stage_logs(
    reaction: &Uuid,
    stage: &str,
    logs: StageLogsAdd,
    shared: &Shared,
) -> Result<(), ApiError> {
    // log some stats on the stage logs we are saving
    event!(
        Level::INFO,
        index = logs.index,
        lines = logs.logs.len(),
        return_code = logs.return_code
    );
    // crawl over logs and insert them into scylla 10 at a time
    stream::iter(logs.logs)
        .map(|line| {
            // determine the bucket for this log line
            let bucket: i32 = (line.index / 2500) as i32;
            // send this log line to scylla
            shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.logs.insert,
                (reaction, stage, bucket, line.index as i64, line.line),
            )
        })
        .buffer_unordered(10)
        .collect::<Vec<Result<_, _>>>()
        .await
        .into_iter()
        .for_each(|res| {
            log_scylla_err!(res);
        });
    Ok(())
}

#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
struct LogLine {
    line: String,
}

/// Gets stage logs for a stage in a reaction from Redis
///
/// # Arguments
///
/// * `reaction` - The reaction to get a stages logs for
/// * `cursor` - The number of log lines to skip
/// * `limit` - The max number of log lines to return (strongly enforced)
/// * `stage` - The stage to get logs for
/// * `shared` - Shared Thorium objects
pub async fn stage_logs(
    reaction: &Reaction,
    stage: &str,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<StageLogs, ApiError> {
    // convert our cursor to an i64
    let cursor: i64 = cursor.try_into()?;
    // if we want to crawl more than 250,000 things then return an error
    if limit > 250_000 {
        return bad!("Limit can be no more than 250,000 lines".to_owned());
    }
    // determine our current bucket
    let current_bucket: i32 = (cursor / 2500).try_into()?;
    // determine the number of buckets we need to search in
    let max_buckets: i32 = std::cmp::max((limit / 2500).try_into()?, 1);
    // build a vec to store our buckets
    let mut buckets = Vec::with_capacity((max_buckets + 1).try_into()?);
    // add our current bucket
    buckets.push(current_bucket);
    // add the remaining buckets to search
    buckets.extend((0..max_buckets).map(|i| current_bucket + i));
    // get our log lines from scylla
    let query = shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.logs.get,
            (&reaction.id, &stage, &buckets, cursor, limit as i32),
        )
        .await?;
    // assume we will pull the max number of logs we want
    let mut logs = Vec::with_capacity(limit);
    // enable rows on this query response
    let query_rows = query.into_rows_result()?;
    // crawl over logs and convert them into Strings
    for row in query_rows.rows::<LogLine>()? {
        // try to deserialize this line
        let line = row?;
        // add this line to our logs
        logs.push(line.line);
    }
    Ok(StageLogs { logs })
}

/// Gets status logs from redis
///
/// These are reaction status logs not stage logs.
///
/// # Arguments
///
/// * `reaction` - The reaction to get status logs for
/// * `cursor` - The number of status logs to skip
/// * `limit` - The max number of status logs to return (weakly enforced)
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
pub async fn logs(
    reaction: &Reaction,
    cursor: usize,
    limit: usize,
    shared: &Shared,
    span: &Span,
) -> Result<Vec<StatusUpdate>, ApiError> {
    // Start our get reaction logs span
    span!(
        parent: span,
        Level::INFO,
        "Get Reaction Logs From Redis",
        cursor = cursor,
        limit = limit
    );
    // build reaction data keys
    let keys = ReactionKeys::new(reaction, shared);
    // get end range based on cursor
    // subtract 1 because our range is inclusive
    let end = cursor + limit.saturating_sub(1);
    // get all log objects
    let raw_logs: Vec<String> =
        query!(cmd("lrange").arg(keys.logs).arg(cursor).arg(end), shared).await?;
    let logs = raw_logs
        .iter()
        .map(|raw| StatusUpdate::deserialize(raw))
        .filter_map(Result::ok)
        .collect();
    Ok(logs)
}

/// Lists jobs within a reaction
///
/// # Arguments
///
/// * `reaction` - The reaction to list jobs from
/// * `cursor` - The cursor to use when paging through jobs in this reaction
/// * `limit` - The number of jobs to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list_jobs(
    reaction: &Reaction,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<JobList, ApiError> {
    // get key to jobs in reaction
    let key = ReactionKeys::jobs(&reaction.group, &reaction.id, shared);
    // get list of jobs in this reaction
    let (new_cursor, names): (usize, Vec<String>) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // convert our job list to uuids
    let names = names
        .iter()
        .map(|raw| Uuid::parse_str(raw))
        .filter_map(|res| res.ok())
        .collect();
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(JobList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(JobList::new(Some(new_cursor), names))
    }
}

/// Deletes a reaction from redis
///
/// This will also delete all jobs for this reaction.
///
/// # Arguments
///
/// * `reaction` - The reaction to delete
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::reactions::delete", skip_all, err(Debug))]
pub async fn delete(reaction: &Reaction, shared: &Shared) -> Result<(), ApiError> {
    // loop until jobs in this stage are deleted
    let mut cursor = 0;
    loop {
        // list this reactions jobs
        let ids = reaction.list_jobs(cursor, 1000, shared).await?;
        let jobs = RawJob::list_details(ids, shared).await?;
        // build our redis pipe
        let mut pipe = redis::pipe();
        // add the delete job data commands
        jobs.details.iter()
            .fold(&mut pipe, |pipe, job|
                pipe.cmd("del").arg(&JobKeys::data(&job.id, shared))
                    .cmd("zrem").arg(&JobKeys::status_queue(&reaction.group, 
                            &reaction.pipeline, &job.stage, &reaction.creator, &job.status, shared))
                         .arg(&job.id.to_string())
                    .cmd("zrem").arg(&StreamKeys::system_scaler(job.scaler, "deadlines", shared))
                         .arg(job.stream_data())
                    .cmd("del").arg(&ReactionKeys::stage_logs(&reaction.id, &job.stage, shared)));
        // filter out any jobs that don't have a running worker
        jobs.details.iter().filter(|job| job.worker.is_some()).fold(&mut pipe, |pipe, job|
            // remove this job from the running queue
            pipe.cmd("zrem")
                .arg(&StreamKeys::system_scaler(job.scaler, "running", shared))
                .arg(force_serialize!(&serde_json::json!({"job_id": job.id, "worker": &job.worker.as_ref().unwrap()})))
        );
        // execute our redis pipeline
        let _: () = pipe.atomic().query_async(conn!(shared)).await?;
        // check if our cursor has been exhausted
        if jobs.cursor.is_none() {
            break
        }
        // update cursor
        cursor = jobs.cursor.unwrap();
    }
    let keys = ReactionKeys::new(reaction, shared);
    // build pipeline to delete this reactions data
    let mut pipe = redis::pipe();
    // add commands to clear remove this reaction from the tag lists
    reaction.tags.iter()
        .fold(&mut pipe, |pipe, tag| {
            pipe.cmd("srem").arg(ReactionKeys::tag(&reaction.group, tag, shared))
                    .arg(&reaction.id.to_string())
        });

    // execute pipeline to delete our reaction specific data
    let _: () = pipe.atomic()
        .cmd("del").arg(&keys.data)
        .cmd("del").arg(&keys.logs)
        .cmd("zrem").arg(&ReactionKeys::group_set(&reaction.group, &reaction.status, shared))
            .arg(&reaction.id.to_string())
        .cmd("srem").arg(&keys.set).arg(reaction.id.to_string())
        .cmd("srem").arg(&ReactionKeys::status(&reaction.group, &reaction.pipeline, &reaction.status, shared))
            .arg(&reaction.id.to_string())
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Deletes all reactions in a pipeline
///
/// This will also delete all jobs for this pipeline
///
/// # Arguments
///
/// * `user` - The user that is deleting these reactions
/// * `group` - The group to delete reactions from
/// * `pipeline` - The pipeline to delete reactions from
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn delete_all(
    user: &User,
    group: &Group,
    pipeline: &Pipeline,
    shared: &Shared,
) -> Result<(), ApiError> {
    // loop until jobs in this stage are deleted
    let mut cursor = 0;
    loop {
        // get list of reactions
        let reactions = Reaction::list(pipeline, cursor, 10, shared).await?
            .details(&group.name, shared).await?;
        let delete_futs = reactions.details.iter()
            .map(|item| item.delete(user, group, shared));
        try_join_all(delete_futs).await?;

        // check if our cursor has been exhausted
        if reactions.cursor.is_none() {
            break;
        }
        // update cursor
        cursor = reactions.cursor.unwrap();
    }
    Ok(())
}

/// Expire reactions out of any final status lists
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::reactions::expire_lists", skip_all, err(Debug))]
pub async fn expire_lists(shared: &Shared) -> Result<(), ApiError> {
    // get the timestamp to expire to
    let now = Utc::now().timestamp();
    // crawl over expire list and expire them 100k at a time
    // do at most 100 loops to keep response time low
    let mut count = 0;
    loop {
        // read up to 100k expire items
        let raw = streams::read_no_scores("system", "global", "expire", 0, now, 100_000, shared).await?;
        // deserialize into reaction expirations
        let expires = raw.into_iter()
            .map(|item| ReactionExpire::cast(&item))
            .filter_map(|res| log_err!(res))
            .collect::<Vec<ReactionExpire>>();

        // execute redis pipeline to remove these reactions from the status list
        let expire_stream = StreamKeys::system_global("expire", shared);
        // remove any expired data
        let _: () = expires.iter()
            .fold(redis::pipe().atomic(), |pipe, exp|
                pipe.cmd(&exp.cmd).arg(&exp.list).arg(&exp.id)
                    .cmd("zrem").arg(&expire_stream).arg(force_serialize!(&exp)))
            .query_async(conn!(shared)).await?;

        // check if we have run out of things to expire
        if expires.is_empty() || count > 100 {
            break;
        }
        // increment our counter
        count += 1;
    }
    Ok(())
}

/// Updates the remaining stages of this reaction
    ///
/// # Arguments
///
/// * `reaction` - The updated reaction data
/// * `new_tags` - The new tags for this reaction
/// * `old_tags` - The old tags to remove
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn update(
    reaction: &Reaction,
    new_tags: &[String],
    old_tags: &[String],
    shared: &Shared
) -> Result<(), ApiError> {
    // build reaction data keys
    let keys = ReactionKeys::new(reaction, shared);
    // build pipeline to update our reactions data
    let mut pipe = redis::pipe();
    pipe.cmd("hset").arg(&keys.data).arg("args").arg(serialize!(&reaction.args))
        .cmd("hset").arg(&keys.data).arg("sla").arg(serialize!(&reaction.sla))
        .cmd("hsetnx").arg(&keys.data).arg("ephemeral").arg(serialize!(&reaction.ephemeral));
    // remove the old tags
    old_tags.iter()
        .fold(&mut pipe, |pipe, tag|
            pipe.cmd("srem").arg(ReactionKeys::tag(&reaction.group, tag, shared))
                .arg(&reaction.id.to_string()));
    // add new tags
    new_tags.iter()
        .fold(&mut pipe, |pipe, tag|
            pipe.cmd("sadd").arg(ReactionKeys::tag(&reaction.group, tag, shared))
                .arg(&reaction.id.to_string()));
    // execute pipeline updating this reactions data
    let _: () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}
