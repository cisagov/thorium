use bb8_redis::redis::cmd;
use chrono::prelude::*;
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, HashSet};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::keys::{self, StreamKeys, SystemKeys, UserKeys};
use super::{helpers, SimpleScyllaCursor};
use crate::models::system::{
    WorkerStatus, BARE_METAL_CACHE_KEY, DEFAULT_IFF, EXTERNAL_CACHE_KEY, K8S_CACHE_KEY,
    KVM_CACHE_KEY, WINDOWS_CACHE_KEY,
};
use crate::models::{
    ApiCursor, GroupStats, ImageScaler, Node, NodeGetParams, NodeHealth, NodeListLine,
    NodeListParams, NodeRegistration, NodeRow, NodeUpdate, ScalerStats, SystemInfo, SystemSettings,
    SystemStats, User, Worker, WorkerDeleteMap, WorkerRegistrationList, WorkerUpdate,
};
use crate::utils::{ApiError, Shared};
use crate::{
    conn, deserialize, exec_query, internal_err, log_scylla_err, not_found, query, serialize,
    unauthorized,
};

/// Check if Thorium is healthy
///
/// Currently this just checks redis but we should add scylla too.
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn health(shared: &Shared) -> Result<bool, ApiError> {
    // try to ping redis
    let status: String = redis::cmd("PING").query_async(conn!(shared)).await?;
    // if our status is pong then return true
    if status == "PONG" {
        return Ok(true);
    }
    // default to an unhealthy status
    Ok(false)
}

/// Get this Thorium instances IFF string
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn iff(shared: &Shared) -> Result<String, ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    let iff = query!(cmd("hget").arg(&keys.data).arg("iff"), shared).await?;
    Ok(iff)
}

/// Initialize [`SystemInfo`], [`SystemSettings`], and the IFF string in Redis with default values
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn init(shared: &Shared) -> Result<(), ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    let default_info = SystemInfo::default();
    let default_settings = SystemSettings::default();
    let _: () = redis::pipe()
        .atomic()
        .cmd("hsetnx").arg(&keys.data).arg(K8S_CACHE_KEY).arg(default_info.k8s_cache)
        .cmd("hsetnx").arg(&keys.data).arg(BARE_METAL_CACHE_KEY).arg(default_info.bare_metal_cache)
        .cmd("hsetnx").arg(&keys.data).arg(WINDOWS_CACHE_KEY).arg(default_info.windows_cache)
        .cmd("hsetnx").arg(&keys.data).arg(KVM_CACHE_KEY).arg(default_info.kvm_cache)
        .cmd("hsetnx").arg(&keys.data).arg(EXTERNAL_CACHE_KEY).arg(default_info.external_cache)
        .cmd("hsetnx").arg(&keys.settings).arg("reserved_cpu").arg(default_settings.reserved_cpu)
        .cmd("hsetnx").arg(&keys.settings).arg("reserved_memory").arg(default_settings.reserved_memory)
        .cmd("hsetnx").arg(&keys.settings).arg("reserved_storage").arg(default_settings.reserved_storage)
        .cmd("hsetnx").arg(&keys.settings).arg("fairshare_cpu").arg(default_settings.fairshare_cpu)
        .cmd("hsetnx").arg(&keys.settings).arg("fairshare_memory").arg(default_settings.fairshare_memory)
        .cmd("hsetnx").arg(&keys.settings).arg("fairshare_storage").arg(default_settings.fairshare_storage)
        .cmd("hsetnx").arg(&keys.settings).arg("host_path_whitelist").arg(serialize!(&default_settings.host_path_whitelist))
        .cmd("hsetnx").arg(&keys.settings).arg("allow_unrestricted_host_paths").arg(serialize!(&default_settings.allow_unrestricted_host_paths))
        .cmd("hsetnx").arg(&keys.data).arg("iff").arg(DEFAULT_IFF)
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// Cast a [`HashMap`] into a [`SystemInfo`] object
///
/// # Arguments
///
/// * `raw` - The hashmap to cast
fn cast(mut raw: HashMap<String, String>) -> Result<SystemInfo, ApiError> {
    // return 404 if hashmap is empty
    if raw.is_empty() {
        return not_found!("SystemInfo not found".to_owned());
    }

    let info = SystemInfo {
        k8s_cache: helpers::extract_bool(&mut raw, K8S_CACHE_KEY)?,
        bare_metal_cache: helpers::extract_bool(&mut raw, BARE_METAL_CACHE_KEY)?,
        windows_cache: helpers::extract_bool(&mut raw, WINDOWS_CACHE_KEY)?,
        kvm_cache: helpers::extract_bool(&mut raw, KVM_CACHE_KEY)?,
        external_cache: helpers::extract_bool(&mut raw, EXTERNAL_CACHE_KEY)?,
    };
    Ok(info)
}

/// Gets [`SystemInfo`] from redis
///
/// # Arguments
///
/// * `reset` - Whether to also reset certain values when getting them
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn get_info(reset: Option<ImageScaler>, shared: &Shared) -> Result<SystemInfo, ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    // if reset is set get and clear system info otherwise just get it
    if let Some(scaler) = reset {
        // reset and get system info
        let (raw, _): (HashMap<String, String>, i64) = redis::pipe()
            .atomic()
            .cmd("hgetall").arg(&keys.data)
            .cmd("hset").arg(&keys.data).arg(scaler.cache_key()).arg(false)
            .query_async(conn!(shared))
            .await?;
        // cast raw data to system info
        cast(raw)
    } else {
        // just get system info
        let raw = query!(cmd("hgetall").arg(&keys.data), shared).await?;
        // cast raw data to system info
        cast(raw)
    }
}

/// Initialize the [`SystemInfo`] in redis with default values and sets the IFF string
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn reset_cache(shared: &Shared) -> Result<(), ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    // sets the scaler cache to reset
    () = redis::pipe()
        .cmd("hset").arg(&keys.data).arg(K8S_CACHE_KEY).arg(true)
        .cmd("hset").arg(&keys.data).arg(BARE_METAL_CACHE_KEY).arg(true)
        .cmd("hset").arg(&keys.data).arg(WINDOWS_CACHE_KEY).arg(true)
        .cmd("hset").arg(&keys.data).arg(KVM_CACHE_KEY).arg(true)
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// Gets [`SystemStats`] from redis
///
/// # Arguments
///
/// * `groups` - The groups to get stats for
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn get_stats(
    groups: HashMap<String, GroupStats>,
    shared: &Shared
) -> Result<SystemStats, ApiError> {
    // get system statistics
    let counts: Vec<i64> = redis::pipe()
    // count the number of deadlines for each scaler type
        .cmd("zcard").arg(StreamKeys::system_scaler(ImageScaler::K8s, "deadlines", shared))
        .cmd("zcard").arg(StreamKeys::system_scaler(ImageScaler::BareMetal, "deadlines", shared))
        .cmd("zcard").arg(StreamKeys::system_scaler(ImageScaler::External, "deadlines", shared))
        // count the number of currently running jobs for each scaler type
        .cmd("zcard").arg(StreamKeys::system_scaler(ImageScaler::K8s, "running", shared))
        .cmd("zcard").arg(StreamKeys::system_scaler(ImageScaler::BareMetal, "running", shared))
        .cmd("zcard").arg(StreamKeys::system_scaler(ImageScaler::External, "running", shared))
        // count number of users
        .cmd("scard").arg(UserKeys::global(shared))
        .query_async(conn!(shared)).await?;
    // build the stats for the specific scalers
    let k8s = ScalerStats::new(counts[0], counts[3]);
    let baremetal = ScalerStats::new(counts[1], counts[4]);
    let external = ScalerStats::new(counts[2], counts[5]);
    // cast raw data to system auth keys
    Ok(SystemStats {
        deadlines: counts[0] + counts[1] + counts[2],
        running: counts[3] + counts[4] + counts[5],
        users: counts[6],
        k8s,
        baremetal,
        external,
        groups,
    })
}

/// Resets the [`SystemSettings`] in redis
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn reset_settings(shared: &Shared) -> Result<(), ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    // build default settings
    let default = SystemSettings::default();
    // reset system settings
    let _: () = redis::pipe()
        // reset reserved resources
        .cmd("hset").arg(&keys.settings).arg("reserved_cpu").arg(default.reserved_cpu)
        .cmd("hset").arg(&keys.settings).arg("reserved_memory").arg(default.reserved_memory)
        .cmd("hset").arg(&keys.settings).arg("reserved_storage").arg(default.reserved_storage)
        .cmd("hset").arg(&keys.settings).arg("fairshare_cpu").arg(default.fairshare_cpu)
        .cmd("hset").arg(&keys.settings).arg("fairshare_memory").arg(default.fairshare_memory)
        .cmd("hset").arg(&keys.settings).arg("fairshare_storage").arg(default.fairshare_storage)
        .cmd("hset").arg(&keys.settings).arg("host_path_whitelist").arg(serialize!(&default.host_path_whitelist))
        .cmd("hset").arg(&keys.settings).arg("allow_unrestricted_host_paths").arg(serialize!(&default.allow_unrestricted_host_paths))
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// Cast a [`HashMap`] into a [`SystemSettings`] object
///
/// # Arguments
///
/// * `raw` - The hashmap to cast
#[rustfmt::skip]
fn cast_settings(mut raw: HashMap<String, String>) -> Result<SystemSettings, ApiError> {
    // return 404 if hashmap is empty
    if raw.is_empty() {
        return not_found!("SystemSettings not found".to_owned());
    }
    // deserialize into a SystemSettings object
    let settings = SystemSettings {
        reserved_cpu: deserialize!(&helpers::extract(&mut raw, "reserved_cpu")?),
        reserved_memory: deserialize!(&helpers::extract(&mut raw, "reserved_memory")?),
        reserved_storage: deserialize!(&helpers::extract(&mut raw, "reserved_storage")?),
        fairshare_cpu: deserialize!(&helpers::extract(&mut raw, "fairshare_cpu")?),
        fairshare_memory: deserialize!(&helpers::extract(&mut raw, "fairshare_memory")?),
        fairshare_storage: deserialize!(&helpers::extract(&mut raw, "fairshare_storage")?),
        host_path_whitelist: deserialize!(&helpers::extract(&mut raw, "host_path_whitelist")?),
        allow_unrestricted_host_paths: deserialize!(&helpers::extract(&mut raw, "allow_unrestricted_host_paths")?),
    };
    Ok(settings)
}

/// Gets [`SystemSettings`] from redis
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn get_settings(shared: &Shared) -> Result<SystemSettings, ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    // get system auth keys
    let raw = query!(cmd("hgetall").arg(&keys.settings), shared).await?;
    // cast to a SystemSettings object
    cast_settings(raw)
}

/// Updates the [`SystemSettings`] in redis
///
/// # Arguments
///
/// * `update` - The updates to apply
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn update_settings(settings: &SystemSettings, shared: &Shared) -> Result<(), ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    // reset system settings
    () = redis::pipe().atomic()
        // update reserved settings
        .cmd("hset").arg(&keys.settings).arg("reserved_cpu").arg(&settings.reserved_cpu)
        .cmd("hset").arg(&keys.settings).arg("reserved_memory").arg(&settings.reserved_memory)
        .cmd("hset").arg(&keys.settings).arg("reserved_storage").arg(&settings.reserved_storage)
        // update fair share settings
        .cmd("hset").arg(&keys.settings).arg("fairshare_cpu").arg(&settings.fairshare_cpu)
        .cmd("hset").arg(&keys.settings).arg("fairshare_memory").arg(&settings.fairshare_memory)
        .cmd("hset").arg(&keys.settings).arg("fairshare_storage").arg(&settings.fairshare_storage)
        // update host path settings
        .cmd("hset").arg(&keys.settings).arg("host_path_whitelist").arg(serialize!(&settings.host_path_whitelist))
        .cmd("hset").arg(&keys.settings).arg("allow_unrestricted_host_paths").arg(serialize!(&settings.allow_unrestricted_host_paths))
        .query_async(conn!(shared))
        .await?;
    Ok(())
}

/// Restores the [`SystemSettings`] from a backup
///
/// # Arguments
///
/// * `settings` - The settings to restore
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn restore_settings(settings: &SystemSettings, shared: &Shared) -> Result<(), ApiError> {
    // build system keys
    let keys = SystemKeys::new(shared);
    // build a redis pipeline to restore the system settings
    let _: () = redis::pipe()
        .atomic()
        .cmd("hsetnx").arg(&keys.settings).arg("reserved_cpu").arg(settings.reserved_cpu)
        .cmd("hsetnx").arg(&keys.settings).arg("reserved_memory").arg(settings.reserved_memory)
        .cmd("hsetnx").arg(&keys.settings).arg("reserved_storage").arg(settings.reserved_storage)
        .cmd("hsetnx").arg(&keys.settings).arg("fairshare_cpu").arg(settings.fairshare_cpu)
        .cmd("hsetnx").arg(&keys.settings).arg("fairshare_memory").arg(settings.fairshare_memory)
        .cmd("hsetnx").arg(&keys.settings).arg("fairshare_storage").arg(settings.fairshare_storage)
        .cmd("hsetnx").arg(&keys.settings).arg("host_path_whitelist").arg(serialize!(&settings.host_path_whitelist))
        .cmd("hsetnx").arg(&keys.settings).arg("allow_unrestricted_host_paths").arg(serialize!(&settings.allow_unrestricted_host_paths))
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Wipes all Thorium controlled databases
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
pub async fn wipe(shared: &Shared) -> Result<(), ApiError> {
    // wipe the currently selected redis db
    let _: () = exec_query!(cmd("flushdb"), shared).await?;
    Ok(())
}

/// Save a node to scylla
///
/// # Arguments
///
/// * `node` - This nodes registration info
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::register", skip_all, fields(cluster = &node.cluster, node = &node.name), err(Debug))]
pub async fn register(node: &NodeRegistration, shared: &Shared) -> Result<(), ApiError> {
    // save this nodes info in scylla
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.nodes.insert,
            (
                &node.cluster,
                &node.name,
                serialize!(&NodeHealth::Registered),
                serialize!(&node.resources),
            ),
        )
        .await?;
    Ok(())
}

/// Gets the workers for a specific node
///
/// # Arguments
///
/// * `row` - The node row to get workers for and cast
/// * `scalers` - The different scalers to get workers for
/// * `shared` - Shared Thorium objects
#[instrument(
    name = "db::system::get_workers_for_node",
    skip(row, shared),
    fields(cluster = &row.cluster, node = &row.node),
    err(Debug)
)]
pub async fn get_workers_for_node(
    row: NodeRow,
    scalers: &Vec<ImageScaler>,
    shared: &Shared,
) -> Result<Node, ApiError> {
    // cast this row to a node
    let mut node = Node::try_from(row)?;
    // get the workers on this node for each scaler
    for scaler in scalers {
        // build the key to this nodes worker set and scaler
        let worker_set = keys::system::worker_set(&node.cluster, &node.name, *scaler, shared);
        // get the workers on this node
        let names: Vec<String> = query!(cmd("smembers").arg(worker_set), shared).await?;
        // build a redis pipeline to get these workers info
        let mut pipe = redis::pipe();
        for name in &names {
            // build the key to this workers info
            let data_key = keys::system::worker_data(name, shared);
            // add the command to get this workers info
            pipe.cmd("hgetall").arg(data_key);
        }
        // get our workers info
        let info: Vec<HashMap<String, String>> = pipe.query_async(conn!(shared)).await?;
        // cast our worker info maps to workers
        for map in info {
            // convert this map to a worker
            let worker = Worker::try_from(map)?;
            // add this worker to a node
            node.workers.insert(worker.name.clone(), worker);
        }
    }
    Ok(node)
}

/// Gets a nodes info from scylla
///
/// # Arguments
///
/// * `cluster` - The cluster this node is in
/// * `node` - The name of the node to get info on
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::get_node", skip(params, shared), err(Debug))]
pub async fn get_node(
    cluster: &str,
    node: &str,
    params: &NodeGetParams,
    shared: &Shared,
) -> Result<Node, ApiError> {
    // get this nodes info from scylla
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.nodes.get, (cluster, node))
        .await?;
    // enable casting to types for this query
    let query_rows = query.into_rows_result()?;
    // try to get the first row
    if let Some(row) = query_rows.maybe_first_row::<NodeRow>()? {
        // get this nodes workers if any exist
        let node = get_workers_for_node(row, &params.scalers, shared).await?;
        return Ok(node);
    }
    not_found!(format!("Node {}:{} doesn't exist!", cluster, node))
}

/// Gets a nodes info from scylla
///
/// # Arguments
///
/// * `clusters` - The clusters this node is in
/// * `nodes` - The names of the nodes to get info on
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::get_node_rows", skip_all, err(Debug))]
pub async fn get_node_rows(
    clusters: &Vec<String>,
    nodes: &Vec<String>,
    shared: &Shared,
) -> Result<Vec<NodeRow>, ApiError> {
    // build a list to store our queries in
    let mut queries = Vec::with_capacity(1);
    // check if this query can be done in one request or not
    if clusters.len() * nodes.len() < 100 {
        // execute this query
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.nodes.get_many, (clusters, nodes))
            .await?;
        // add our query
        queries.push(query);
    } else {
        // get each clusters node rows one at a time since many clusters is rare
        for cluster in clusters {
            // break out nodes into chunks of one
            for chunk in nodes.chunks(100) {
                // cast our node chunk to a vec
                let chunk_vec = chunk.to_vec();
                // execute this query
                let query = shared
                    .scylla
                    .session
                    .execute_unpaged(&shared.scylla.prep.nodes.get_many, ((cluster), chunk_vec))
                    .await?;
                // add our query
                queries.push(query);
            }
        }
    }
    // build a list of our node rows
    let mut node_rows = Vec::with_capacity(nodes.len());
    // cast each query into node rows
    for query in queries {
        // enable casting to types for this query
        let query_rows = query.into_rows_result()?;
        // set the type to cast this iter too
        let typed_iter = query_rows
            // set the type
            .rows::<NodeRow>()?
            // log any rows we fail to deserialize
            .filter_map(|row| log_scylla_err!(row));
        // extend our vec with this queries node rows
        node_rows.extend(typed_iter);
    }
    Ok(node_rows)
}

/// Update a nodes info
///
/// # Arguments
///
/// * `node` - The node to apply this update too
/// * `update` - The update to apply to this node
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::update_node", skip_all, err(Debug))]
pub async fn update_node(
    node: &Node,
    update: &NodeUpdate,
    shared: &Shared,
) -> Result<(), ApiError> {
    // determine if heart beat should be updated or not
    if update.heart_beat {
        // this update should update the heart beat timestamp
        let heart_beat = Utc::now();
        // update this nodes info in scylla
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.nodes.update_heart_beat,
                (
                    serialize!(&update.health),
                    serialize!(&update.resources),
                    heart_beat,
                    &node.cluster,
                    &node.name,
                ),
            )
            .await?;
    } else {
        // this update should not update the heart beat timestamp
        // update this nodes info in scylla
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.nodes.update,
                (
                    serialize!(&update.health),
                    serialize!(&update.resources),
                    &node.cluster,
                    &node.name,
                ),
            )
            .await?;
    }
    Ok(())
}

/// List node names for specific clusters
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::list_nodes", skip(shared), err(Debug))]
pub async fn list_nodes(
    params: NodeListParams,
    shared: &Shared,
) -> Result<SimpleScyllaCursor<NodeListLine>, ApiError> {
    // if a cursor id was set then get it otherwise make a new cursor
    let mut cursor = match params.cursor {
        Some(cursor_id) => SimpleScyllaCursor::get(cursor_id, params.page_size, shared).await?,
        None => SimpleScyllaCursor::new(params.clusters, params.page_size),
    };
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save or delete this cursor based on if its exhausted or not
    cursor.save(shared).await?;
    Ok(cursor)
}

/// List node details for specific clusters
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::list_node_details", skip(shared), err(Debug))]
pub async fn list_node_details(
    params: NodeListParams,
    shared: &Shared,
) -> Result<ApiCursor<Node>, ApiError> {
    // if a cursor id was set then get it otherwise make a new cursor
    let mut cursor: SimpleScyllaCursor<NodeRow> = match params.cursor {
        Some(cursor_id) => SimpleScyllaCursor::get(cursor_id, params.page_size, shared).await?,
        None => SimpleScyllaCursor::new(params.clusters, params.page_size),
    };
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    // create an empty user facing cursor to store this cursors data
    let mut api_cursor = ApiCursor::empty(cursor.data.len());
    // get all of our nodes workers 25 at a time
    let nodes = stream::iter(cursor.data)
        .map(|row| get_workers_for_node(row, &params.scalers, shared))
        .buffered(25)
        .collect::<Vec<Result<Node, ApiError>>>()
        .await
        .into_iter()
        .filter_map(|res| log_scylla_err!(res));
    // move our nodes to our user facing cursor
    api_cursor.data.extend(nodes);
    Ok(api_cursor)
}

/// Adds new worker to the workers tables in Scylla
///
/// # Arguments
///
/// * `scaler` - The scaler this worker is under
/// * `creates` - The workers to register
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::system::register_workers", skip_all, err(Debug))]
pub async fn register_workers(
    scaler: ImageScaler,
    creates: &WorkerRegistrationList,
    shared: &Shared,
) -> Result<(), ApiError> {
    //  get the current timestamp
    let now = Utc::now();
    // this worker doesn't have a heart beat yet because it just registered
    let heart_beat: Option<DateTime<Utc>> = None;
    // get a redis pipeline
    let mut pipe = redis::pipe();
    // set this pipeline to be atomic
    pipe.atomic();
    // save these workers info in scylla
    for worker in &creates.workers {
        // get the key for this workers data
        let data = keys::system::worker_data(&worker.name, shared);
        // get the key for this workers cluster/node/scaler worker set
        let set = keys::system::worker_set(&worker.cluster, &worker.node, scaler, shared);
        pipe
            // add this workers info
            .cmd("hsetnx").arg(&data).arg("cluster").arg(&worker.cluster)
            .cmd("hsetnx").arg(&data).arg("node").arg(&worker.node)
            .cmd("hsetnx").arg(&data).arg("scaler").arg(serialize!(&scaler))
            .cmd("hsetnx").arg(&data).arg("name").arg(&worker.name)
            .cmd("hsetnx").arg(&data).arg("user").arg(&worker.user)
            .cmd("hsetnx").arg(&data).arg("pipeline").arg(&worker.pipeline)
            .cmd("hsetnx").arg(&data).arg("group").arg(&worker.group)
            .cmd("hsetnx").arg(&data).arg("stage").arg(&worker.stage)
            .cmd("hsetnx").arg(&data).arg("status").arg(serialize!(&WorkerStatus::Spawning))
            .cmd("hsetnx").arg(&data).arg("spawned").arg(serialize!(&now))
            .cmd("hsetnx").arg(&data).arg("heart_beat").arg(serialize!(&heart_beat))
            .cmd("hsetnx").arg(&data).arg("resources").arg(serialize!(&worker.resources))
            .cmd("hsetnx").arg(&data).arg("pool").arg(serialize!(&worker.pool))
            // add this workers to its set
            .cmd("sadd").arg(&set).arg(&worker.name);
    }
    // execute the redis pipeline we built
    let _: () = pipe.query_async(conn!(shared)).await?;
    Ok(())
}

/// Gets info on a specific worker
///
/// # Arguments
///
/// * `name` - The name of this worker
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::system::get_worker", skip(shared), err(Debug))]
pub async fn get_worker(name: &str, shared: &Shared) -> Result<Worker, ApiError> {
    // get the key for this workers info
    let data_key = keys::system::worker_data(name, shared);
    // get this workers data
    let raw: HashMap<String, String> = query!(cmd("hgetall").arg(data_key), shared).await?;
    // if we got any data then we should have gotten a worker
    if raw.is_empty() {
        not_found!(format!("Worker {} does not exist", name))
    } else {
        Worker::try_from(raw)
    }
}

/// Updates a workers current status
///
/// # Arguments
///
/// * `worker` - The worker to register
/// * `update` - The update to apply to this worker
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::system::update_worker", skip_all, err(Debug))]
pub async fn update_worker(
    worker: &Worker,
    update: &WorkerUpdate,
    shared: &Shared,
) -> Result<(), ApiError> {
    // get the key for this workers data
    let data = keys::system::worker_data(&worker.name, shared);
    //  get the current timestamp
    let heart_beat = Utc::now();
    // get a redis pipeline
    let mut pipe = redis::pipe();
    // set this pipeline to be atomic
    pipe.atomic();
    // update this workers status
    let _: () = pipe.cmd("hset").arg(&data).arg("status").arg(serialize!(&update.status))
        .cmd("hset").arg(&data).arg("heart_beat").arg(serialize!(&heart_beat))
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Updates a workers current reaction and job
///
/// # Arguments
///
/// * `worker` - The name of the worker to set a job for
/// * `reaction` - The new reaction to set
/// * `job` - The new job to set
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[rustfmt::skip]
#[instrument(
    name = "db::system::update_worker_job",
    skip(worker, shared),
    err(Debug)
)]
pub async fn update_worker_job(
    worker: &Worker,
    reaction: &Uuid,
    job: &Uuid,
    shared: &Shared,
) -> Result<(), ApiError> {
    // get the key for this workers data
    let data = keys::system::worker_data(&worker.name, shared);
    //  get the current timestamp
    let heart_beat = Utc::now();
    // get a redis pipeline
    let mut pipe = redis::pipe();
    // set this pipeline to be atomic
    pipe.atomic();
    // update this workers status
    let _: () = pipe.cmd("hset").arg(&data).arg("status").arg(serialize!(&WorkerStatus::Running))
        .cmd("hset").arg(&data).arg("heart_beat").arg(serialize!(&heart_beat))
        .cmd("hset").arg(&data).arg("reaction").arg(reaction.to_string())
        .cmd("hset").arg(&data).arg("job").arg(job.to_string())
        .query_async(conn!(shared)).await?;
    Ok(())
}

/// Get the owners for a map of workers we want to delete
///
/// # Arguments
///
/// * `user` - The user that is trying to delete workers
/// * `scaler` - The scaler to check if we can delete workers form
/// * `deletes` - The workers we want to delete
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
#[instrument(name = "db::system::can_delete_workers", skip_all, err(Debug))]
pub async fn can_delete_workers(
    user: &User,
    deletes: &WorkerDeleteMap,
    shared: &Shared,
) -> Result<(), ApiError> {
    // only check if we can delete this if we aren't an admin
    if !user.is_admin() {
        // get the owners for all workers we are going to delete 100 at a time
        let chunked = deletes.workers.as_slice().chunks(100);
        // check the workers in each chunk
        for chunk in chunked {
            // build the redis pipe for checking this chunk
            let mut pipe = redis::pipe();
            // add each worker in this chunk to our redis pipeline
            for worker in chunk {
                // get the key for this workers data
                let data = keys::system::worker_data(&worker, shared);
                // get this workers owner
                pipe.cmd("hget").arg(data).arg("user");
            }
            // get all of these workers users
            let users_set: HashSet<String> = pipe.query_async(conn!(shared)).await?;
            // if we have multiple users or one user that isn't us then reject this request
            if users_set.len() != 1 || !users_set.contains(&user.username) {
                return unauthorized!();
            }
        }
    }
    Ok(())
}

/// Deletes workers from scylla
///
/// # Arguments
///
/// * `scaler` - The scaler to delete workers form
/// * `deletes` - The workers to delete
/// * `shared` -Shared Thorium objects
/// * `span` - The span to log traces under
#[rustfmt::skip]
#[instrument(name = "db::system::can_delete_workers", skip_all, err(Debug))]
pub async fn delete_workers(
    scaler: ImageScaler,
    deletes: WorkerDeleteMap,
    shared: &Shared,
) -> Result<(), ApiError> {
    // track the worker data keys to delete
    let mut data_keys = Vec::with_capacity(std::cmp::min(25, deletes.workers.len()));
    // get these workers info 25 at a time
    let chunked = deletes.workers.as_slice().chunks(25);
    // get the info for this chunk of workers
    for chunk in chunked {
        // build the redis pipe for checking this chunk
        let mut pipe = redis::pipe();
        // crawl over the workers in this chunk
        for worker in chunk {
            // get the key for this workers data
            let data = keys::system::worker_data(&worker, shared);
            // get this workers cluster and node
            pipe.cmd("hget").arg(&data).arg("cluster")
                .cmd("hget").arg(&data).arg("node");
            // keep our data key to delete later
            data_keys.push(data);
        }
        // get the cluster and node for each worker
        let maybe_info: Vec<Option<String>> = pipe.query_async(conn!(shared)).await?;
        // convert our info list to an iterator
        let mut info_iter = maybe_info.iter();
        // build the pipeline for deleting workers
        let mut pipe = redis::pipe();
        // combine our worker and cluster/node info
        for worker in chunk {
            // get this workers cluster/node
            let (cluster, node) = match (info_iter.next(), info_iter.next()) {
                (Some(Some(cluster)), Some(Some(node))) => (cluster, node),
                (Some(None), Some(None)) => return not_found!(format!("Worker {} not found", worker)),
                _ => {
                    // we got partial info on this worker
                    // log something went wrong
                    event!(Level::ERROR, worker = worker, partial_info=true);
                    return internal_err!();
                }
            };
            // build the key to this workers cluster/node worker set
            let set_key = keys::system::worker_set(cluster, node, scaler, shared);
            // delete this worker from its cluster/node worker set
            pipe.cmd("srem").arg(set_key).arg(worker);
        }
        // delete this workers data as well
        data_keys.iter().for_each(|data_key| { pipe.cmd("del").arg(data_key); });
        // execute the redis pipeline for this chunk of workers
        let _: () = pipe.query_async(conn!(shared)).await?;
    }
    Ok(())
}
