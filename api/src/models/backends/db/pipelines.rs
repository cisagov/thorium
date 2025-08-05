use bb8_redis::redis::cmd;
use std::collections::HashMap;
use tracing::instrument;

use super::helpers;
use super::keys::{EventKeys, GroupKeys, ImageKeys, JobKeys, PipelineKeys};
use crate::models::backends::NotificationSupport;
use crate::models::{
    Group, JobStatus, Pipeline, PipelineKey, PipelineList, PipelineRequest, PipelineStats,
    Reaction, StageStats, User,
};
use crate::utils::{ApiError, Shared};
use crate::{
    cast, conn, hset_del_opt_serialize, hsetnx_opt_serialize, not_found, query, serialize,
};

/// Builds a pipeline creation pipeline for Redis
///
/// # Arguments
///
/// * `pipe` - The redis pipeline to add onto
/// * `cast` - The pipeline to create in redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub fn build(
    pipe: &mut redis::Pipeline,
    cast: &Pipeline,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build keys
    let keys = PipelineKeys::new(cast, shared);
    // build the key to our event cache status flags
    let cache_status = EventKeys::cache(shared);
    // add commands to save this pipeline
    pipe.cmd("hsetnx").arg(&keys.data).arg("group").arg(&cast.group)
        .cmd("hsetnx").arg(&keys.data).arg("name").arg(&cast.name)
        .cmd("hsetnx").arg(&keys.data).arg("creator").arg(&cast.creator)
        .cmd("hsetnx").arg(&keys.data).arg("order").arg(serialize!(&cast.order))
        .cmd("hsetnx").arg(&keys.data).arg("sla").arg(cast.sla)
        .cmd("hsetnx").arg(&keys.data).arg("triggers").arg(serialize!(&cast.triggers))
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        .cmd("sadd").arg(&keys.set).arg(&cast.name);
    // add option value if set
    hsetnx_opt_serialize!(pipe, &keys.data, "description", &cast.description);
    // add this pipeline to our images used_by lists
    cast.order.iter().flatten()
        .fold(pipe, |pipe, image| {
            pipe.cmd("sadd").arg(ImageKeys::used_by(&cast.group, image, shared))
                .arg(&cast.name)
        });
    Ok(())
}

/// Creates a pipeline in the redis backend
///
/// # Arguments
///
/// * `user` - The user creating this pipeline
/// * `group` - The group this pipeline is in
/// * `request` - The PipelineRequest to create in the backend
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::pipelines::create", skip_all, err(Debug))]
pub async fn create(
    user: &User,
    group: &Group,
    request: PipelineRequest,
    shared: &Shared,
) -> Result<Pipeline, ApiError> {
    // get pipeline object that will be returned
    let cast = request.cast(user, group, shared).await?;
    // build redis pipeline to save pipeline data
    let mut pipe = redis::pipe();
    // add commands to build this pipeline
    build(&mut pipe, &cast, shared)?;
    // create our pipeline
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(cast)
}

/// Lists all pipelines in the redis backend
///
/// # Arguments
///
/// * `group` - The group to list pipelines for
/// * `cursor` - The cursor to use when paging through pipelines
/// * `limit` - The number of pipelines to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::pipelines::list", skip(shared), err(Debug))]
pub async fn list(
    group: &str,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<PipelineList, ApiError> {
    // get group image set key
    let key = PipelineKeys::set(group, shared);
    // get list of created groups
    let (new_cursor, names) = query!(cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit), shared).await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(PipelineList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(PipelineList::new(Some(new_cursor), names))
    }
}

/// Lists pipelines and their details in the redis backend
///
/// # Arguments
///
/// * `group` - The group to list pipelines for
/// * `names` - The pipeline names to list details for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::pipelines::list_details", skip(shared), err(Debug))]
pub async fn list_details(
    group: &str,
    names: &[String],
    shared: &Shared,
) -> Result<Vec<Pipeline>, ApiError> {
    // get pipeline data
    let raw: Vec<HashMap<String, String>> = names.iter()
        .fold(redis::pipe().atomic(), |pipe, name|
            pipe.cmd("hgetall").arg(&PipelineKeys::data(group, name, shared)))
        .query_async(conn!(shared)).await?;
    // cast to pipeline structs
    let pipelines = cast!(raw, Pipeline::try_from);
    Ok(pipelines)
}

/// Lists all pipelines in redis for a backup
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::pipelines::backup", skip_all, err(Debug))]
pub async fn backup(shared: &Shared) -> Result<Vec<Pipeline>, ApiError> {
    // build key to group set
    let key = GroupKeys::set(shared);
    // get all group names
    let groups: Vec<String> = query!(cmd("smembers").arg(key), shared).await?;
    // build pipeline to retrieve all pipeline data
    let mut pipe = redis::pipe();
    for group in &groups {
        // get all pipeline names
        let image_key = PipelineKeys::set(group, shared);
        let names: Vec<String> = query!(cmd("smembers").arg(image_key), shared).await?;
        names.iter().fold(&mut pipe, |pipe, name| {
            pipe.cmd("hgetall")
                .arg(PipelineKeys::data(group, name, shared))
        });
    }
    // execute pipeline to get all pipeline data
    let raw: Vec<HashMap<String, String>> = pipe.query_async(conn!(shared)).await?;
    // cast to a vector of pipelines
    let pipelines = cast!(raw, Pipeline::try_from);
    Ok(pipelines)
}

/// Restore pipeline data
///
/// # Arguments
///
/// * `pipelines` - The list of pipeline to restore
/// * `shared` - Shared Thorium objects
pub async fn restore(pipelines: &[Pipeline], shared: &Shared) -> Result<(), ApiError> {
    // build our redis pipeline
    let mut pipe = redis::pipe();
    // crawl over pipeline and build the pipeline to restore each one
    pipelines
        .iter()
        .map(|pipeline| build(&mut pipe, pipeline, shared))
        .collect::<Result<Vec<()>, ApiError>>()?;
    // try to save pipelines into redis
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Gets a pipelines data if it exists
///
/// # Arguments
///
/// * `group` - The group containing the pipeline
/// * `name` - The name of the pipeline to check
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::pipelines::get", skip(shared), err(Debug))]
pub async fn get(group: &str, name: &str, shared: &Shared) -> Result<Pipeline, ApiError> {
    // build key to pipeline data
    let key = PipelineKeys::data(group, name, shared);
    // get pipeline data
    let raw: HashMap<String, String> = query!(cmd("hgetall").arg(&key), shared).await?;
    // check if any data was retrieved
    if raw.is_empty() {
        not_found!(format!("Pipeline {}:{} does not exist", group, name))
    } else {
        // cast to pipeline
        Pipeline::try_from(raw)
    }
}

/// Updates a pipeline in Redis
///
/// # Arguments
///
/// * `pipeline` - The pipeline to update in Redis
/// * `add` - The images that have been added to this pipeline
/// * `remove` - The images that have been removed from this pipeline
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
#[instrument(name = "db::pipelines::update", skip_all, fields(pipeline = &pipeline.name), err(Debug))]
pub async fn update(pipeline: &Pipeline, add: &[String], remove: &[String], shared: &Shared) -> Result<(), ApiError> {
    // build image keys
    let keys = PipelineKeys::new(pipeline, shared);
    // get our event handler cache key
    let cache_status = EventKeys::cache(shared);
    // save pipeline to backend
    let mut pipe = redis::pipe();
    pipe.cmd("hset").arg(&keys.data).arg("order").arg(serialize!(&pipeline.order))
        .cmd("hset").arg(&keys.data).arg("sla").arg(pipeline.sla)
        .cmd("hset").arg(&keys.data).arg("bans").arg(serialize!(&pipeline.bans));
    // add this pipeline to our images used_by lists
    add.iter()
        .fold(&mut pipe, |pipe, image| {
            pipe.cmd("sadd").arg(ImageKeys::used_by(&pipeline.group, image, shared))
                .arg(&pipeline.name)
        });
    // remove this pipeline from the images no longer in this pipeline
    remove.iter()
        .fold(&mut pipe, |pipe, image| {
            pipe.cmd("srem").arg(ImageKeys::used_by(&pipeline.group, image, shared))
                .arg(&pipeline.name)
        });
    // if any triggers are set then update them
    if !pipeline.triggers.is_empty() {
        // save our updated triggers
        pipe.cmd("hset").arg(&keys.data).arg("triggers").arg(serialize!(&pipeline.triggers))
            .cmd("hset").arg(cache_status).arg("status").arg(true);
    }
    // update optional values if set
    hset_del_opt_serialize!(pipe, &keys.data, "description", &pipeline.description);
    // execute this query
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Delete a pipeline and all its jobs
///
/// This will also delete all reactions and jobs for this pipeline.
///
/// # Arguments
///
/// * `user` - The user that is deleting this pipeline
/// * `group` - The group to delete a pipeline from
/// * `pipeline` - The pipeline to delete
/// * `skip_check` - Skip the perms check for deleting reactions
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
#[instrument(
    name = "db::pipelines::delete",
    skip(user, group, shared),
    fields(group = &group.name, pipeline = &pipeline.name),
    err(Debug)
)]
pub async fn delete(
    user: &User,
    group: &Group,
    pipeline: Pipeline,
    skip_check: bool,
    shared: &Shared,
) -> Result<(), ApiError> {
    // delete this pipelines reactions
    Reaction::delete_all(user, group, &pipeline, skip_check, shared).await?;
    // delete the pipeline's notifications
    let key = PipelineKey::from(&pipeline);
    pipeline.delete_all_notifications(&key, shared).await?;
    // delete pipeline data
    let mut pipe = redis::pipe();
    pipe.cmd("del").arg(&PipelineKeys::data(&pipeline.group, &pipeline.name, shared))
        .cmd("srem").arg(&PipelineKeys::set(&pipeline.group, shared))
            .arg(&pipeline.name);
    // remove this pipeline from its images used_by lists
    pipeline.order.iter().flatten()
        .fold(&mut pipe, |pipe, image| {
            pipe.cmd("srem").arg(ImageKeys::used_by(&group.name, image, shared))
                .arg(&pipeline.name)
        });
    // execute this pipeline
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Deletes all pipelines from a group
///
/// This will also delete all reactions and jobs in this group.
///
/// # Arguments
///
/// * `user` - The user deleting all pipelines in a group
/// * `group` - The user group to delete all pipelines from
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
#[instrument(
    name = "db::pipelines::delete_all",
    skip_all,
    fields(group = &group.name), 
    err(Debug)
)]
pub async fn delete_all(
    user: &User,
    group: &Group,
    shared: &Shared,
) -> Result<(), ApiError> {
    // crawl pipelines and delete their jobs
    let mut cursor = 0;
    loop {
        // get names on 200 pipelines at a time
        let names = Pipeline::list(group, cursor, 200, shared).await?;
        // get the details for these pipelines
        let pipelines = names.details(group, shared).await?;
        // delete objects owned by these pipelines
        for pipeline in pipelines.details.iter() {
            // delete this pipelines reactions
            Reaction::delete_all(user, group, pipeline, true, shared).await?;
            // delete the pipeline's notifications
            pipeline.delete_all_notifications(&PipelineKey::from(pipeline), shared).await?;
        }
        // delete all pipeline data
        pipelines.details.iter()
            .fold(redis::pipe().atomic(), |pipe, data| {
                pipe.cmd("del")
                        .arg(&PipelineKeys::data(&data.group, &data.name, shared))
                    .cmd("srem").arg(&PipelineKeys::set(&data.group, shared))
                        .arg(&data.name)
            })
            .exec_async(conn!(shared))
            .await?;

        // check if we have iterated over all pipelines
        if pipelines.cursor.is_none() {
            break;
        }
        // update cursor
        cursor = pipelines.cursor.unwrap();
    }
    Ok(())
}

/// Checks if a pipeline exists in the Redis backend after authentication
///
/// Requiring a reference to a Group object obtained after authorization
/// decreases the likelihood of prematurely checking for the existence of
/// the pipeline and leaking information to an unauthorized user
///
/// # Arguments
///
/// * `name` - The names of the pipelines to check the existence of
/// * `group` - The pipeline's group
/// * `shared` - Shared thorium objects
#[instrument(
    name = "db::pipelines::exists_authenticated",
    skip(group, shared),
    err(Debug)
)]
pub async fn exists_authenticated(
    name: &str,
    group: &Group,
    shared: &Shared,
) -> Result<bool, ApiError> {
    let set_key = PipelineKeys::set(&group.name, shared);
    helpers::exists(name, &set_key, shared).await
}

/// builds a key to a status queue for a specific pipeline
macro_rules! status {
    ($pipeline:expr, $stage:expr, $user:expr, $status:expr, $shared:expr) => {
        JobKeys::status_queue(
            &$pipeline.group,
            &$pipeline.name,
            $stage,
            $user,
            &$status,
            $shared,
        )
    };
}

/// Get status queue counts for all stages in this pipeline for each user
///
/// # Arguments
///
/// * `pipeline` - The pipeline to get status summaries for
/// * `users` - The users to check the status queues for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::pipelines::status", skip_all, err(Debug))]
pub async fn status(
    pipeline: &Pipeline,
    users: &[&String],
    shared: &Shared,
) -> Result<PipelineStats, ApiError> {
    // build a redis pipeline to get the status updates for each stage and user
    let mut pipe = redis::pipe();
    // crawl over the stages of this pipeline
    for stage in pipeline.order.iter().flatten() {
        // crawl over the users who could have jobs in this pipeline
        for user in users.iter() {
            pipe.cmd("zcount").arg(status!(pipeline, stage, user, JobStatus::Created, shared)).arg(0).arg(i64::MAX)
                .cmd("zcount").arg(status!(pipeline, stage, user, JobStatus::Running, shared)).arg(0).arg(i64::MAX)
                .cmd("zcount").arg(status!(pipeline, stage, user, JobStatus::Completed, shared)).arg(0).arg(i64::MAX)
                .cmd("zcount").arg(status!(pipeline, stage, user, JobStatus::Failed, shared)).arg(0).arg(i64::MAX)
                .cmd("zcount").arg(status!(pipeline, stage, user, JobStatus::Sleeping, shared)).arg(0).arg(i64::MAX);
        }
    }
    // execute the built pipeline
    let raw: Vec<(u64, u64, u64, u64, u64)> = pipe.query_async(conn!(shared)).await?;
    // create an empty pipeline status to fill
    let mut status = PipelineStats {
        stages: HashMap::default(),
    };
    // use a counter variable to determine what status tuple to use
    let mut i = 0;
    // crawl over our stages and insert their status counts
    for stage in pipeline.order.iter().flatten() {
        // build a map to store this stages status counts
        let mut stage_map: HashMap<String, StageStats> = HashMap::with_capacity(users.len());
        // crawl over the users and insert any with total counts over 0
        for user in users.iter() {
            // get the total count
            let total = raw[i].0 + raw[i].1 + raw[i].2 + raw[i].3 + raw[i].4;
            // if total is greater than 0 then insert this stage/user stats
            if total > 0 {
                // build this users stage status object
                let stage_status = StageStats {
                    created: raw[i].0,
                    running: raw[i].1,
                    completed: raw[i].2,
                    failed: raw[i].3,
                    sleeping: raw[i].4,
                    total,
                };
                stage_map.insert(user.to_string(), stage_status);
            }
            i += 1;
        }
        status.stages.insert(stage.clone(), stage_map);
    }
    Ok(status)
}
