//! Logic for interacting with network policies in the database

use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use std::collections::{BTreeMap, HashMap};
use tracing::{event, instrument, Level};
use uuid::Uuid;

use super::keys::{NetworkPolicyKeys, SystemKeys};
use super::GroupedScyllaCursor;
use crate::models::system::K8S_CACHE_KEY;
use crate::models::{
    Group, NetworkPolicy, NetworkPolicyListLine, NetworkPolicyListParams, NetworkPolicyListRow,
    NetworkPolicyRequest, NetworkPolicyRow, NetworkPolicyUpdate,
};
use crate::utils::{helpers, ApiError, Shared};
use crate::{bad, conn, log_scylla_err, serialize};

/// Create a `NetworkPolicy` in Scylla
///
/// # Arguments
///
/// * `req` - The request to create a network policy
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::create", skip(shared), err(Debug))]
pub async fn create(req: NetworkPolicyRequest, shared: &Shared) -> Result<(), ApiError> {
    // cast the request to a network policy
    let cast = req.cast()?;
    // concurrently insert the network policy for each group
    stream::iter(cast.groups.iter())
        .map(Ok::<&String, ApiError>)
        .try_for_each_concurrent(100, |group| {
            let cast = &cast;
            async move {
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.network_policies.insert,
                        (
                            group,
                            &cast.name,
                            cast.id,
                            &cast.k8s_name,
                            &cast.created,
                            serialize!(&cast.ingress),
                            serialize!(&cast.egress),
                            cast.forced_policy,
                            cast.default_policy,
                        ),
                    )
                    .await?;
                Ok(())
            }
        })
        .await?;
    // Invalidate the K8's cache
    redis::cmd("hset")
        .arg(&SystemKeys::new(shared).data)
        .arg(K8S_CACHE_KEY)
        .arg(true)
        .exec_async(conn!(shared))
        .await?;
    Ok(())
}

/// List network policies for specific groups
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `dedupe` - Whether to dedupe when listing samples or not
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::list", skip(shared), err(Debug))]
pub async fn list(
    params: NetworkPolicyListParams,
    dedupe: bool,
    shared: &Shared,
) -> Result<GroupedScyllaCursor<NetworkPolicyListLine>, ApiError> {
    // get our cursor
    let mut cursor = GroupedScyllaCursor::from_params(params, dedupe, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    Ok(cursor)
}

/// Gets details on specific network policies in the given groups
///
/// # Arguments
///
/// * `groups` - The groups to search in
/// * `policy_names` - The policy names we want to get details for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::list_details", skip(shared), err(Debug))]
pub async fn list_details(
    groups: &Vec<String>,
    policy_names: Vec<String>,
    shared: &Shared,
) -> Result<Vec<NetworkPolicy>, ApiError> {
    // build a btreemap to preserve network policies order by name
    let mut sorted: BTreeMap<String, NetworkPolicy> = BTreeMap::default();
    // split our groups and policy names into chunks of 50
    for (names_chunk, groups_chunk) in policy_names.chunks(50).cartesian_product(groups.chunks(50))
    {
        // send a query to get this chunks data
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.network_policies.get_many,
                (names_chunk, groups_chunk),
            )
            .await?;
        // enable rows on this query response
        let query_rows = query.into_rows_result()?;
        // crawl the rows returned by this query
        for row in query_rows.rows::<NetworkPolicyRow>()? {
            // propagate any casting errors
            let row = row?;
            // see if we have this policy already
            match sorted.get_mut(&row.name) {
                Some(policy) => {
                    // if we do, just add this row's group
                    policy.groups.push(row.group);
                }
                // if we don't, insert a new policy
                None => {
                    sorted.insert(row.name.clone(), NetworkPolicy::try_from(row)?);
                }
            }
        }
    }
    // get the network policies' used_by info from Redis
    // and add them to the network policies
    stream::iter(sorted.values_mut())
        .map(Ok::<_, ApiError>)
        .try_for_each_concurrent(25, |netpol| async {
            netpol.used_by = used_by(&netpol.groups, &netpol.name, shared).await?;
            Ok(())
        })
        .await?;
    // return the sorted policies
    Ok(sorted.into_values().collect())
}

/// Get a `NetworkPolicy` from Scylla
///
/// # Arguments
///
/// * `groups` - The list of groups to check for the network policy
/// * `name` - The name of the network policy
/// * `id` - An optional network policy ID required if one or more distinct network policies
///          share the same name
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::get", skip(shared), err(Debug))]
pub async fn get(
    groups: &[String],
    policy_name: &str,
    id: Option<Uuid>,
    shared: &Shared,
) -> Result<Option<NetworkPolicy>, ApiError> {
    if groups.is_empty() {
        return bad!("Groups cannot be empty when getting a network policy".to_string());
    }
    // save the groups containing the policy
    let mut policy_groups: Vec<String> = Vec::new();
    // save at least one row to use its information later
    let mut policy_row: Option<NetworkPolicyRow> = None;
    // break our groups into chunks of 100
    for chunk in groups.chunks(100) {
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.network_policies.get,
                (policy_name, chunk),
            )
            .await?;
        // enable rows on this query response
        let query_rows = query.into_rows_result()?;
        // set the type for our rows
        let typed_iter = query_rows.rows::<NetworkPolicyRow>()?;
        // check if we have an id set
        if let Some(id) = &id {
            // if we specified an ID, filter out all rows that don't correspond to that ID
            typed_iter
                .filter_map(|res| {
                    // try casting the row
                    let cast = log_scylla_err!(res);
                    // make sure this row is the policy we're looking for
                    cast.and_then(|cast| (&cast.id == id).then_some(cast))
                })
                .for_each(|cast| {
                    policy_groups.push(cast.group.clone());
                    policy_row = Some(cast);
                });
        } else {
            // cast rows to network policy rows and save the info
            for cast in typed_iter.filter_map(|res| log_scylla_err!(res)) {
                // we didn't specify an ID, so return an error if any of the rows have a different ID
                if policy_row.as_ref().is_some_and(|row| row.id != cast.id) {
                    // make sure this is the same policy, not a policy with the same name in a different group
                    return bad!(format!(
                        "More than one distinct policy with the name '{policy_name}' exists! \
                            Please specify the network policy's ID."
                    ));
                }
                policy_groups.push(cast.group.clone());
                policy_row = Some(cast);
            }
        }
    }
    // map the saved row to a network policy; if we didn't get at least one row,
    // the network policy wasn't found in those groups
    let mut network_policy: Option<NetworkPolicy> =
        policy_row.map(TryInto::try_into).transpose()?;
    if let Some(policy) = network_policy.as_mut() {
        // set the network policy's groups to all the network policy's groups that we found
        policy.groups = policy_groups;
        // get used_by data from redis at the policy's groups
        policy.used_by = used_by(&policy.groups, policy_name, shared).await?;
    }
    Ok(network_policy)
}

/// Get all of the default network policies in a given group
///
/// # Arguments
///
/// * `group` - The group to get default network policies from
/// * `shared` - Shared Thorium objects
#[instrument(
    name = "db::network_policies::get_all_default",
    skip(shared),
    err(Debug)
)]
pub async fn get_all_default(
    group: &str,
    shared: &Shared,
) -> Result<Vec<NetworkPolicyListLine>, ApiError> {
    // perform our query and cast to rows then list lines
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.network_policies.get_default, (group,))
        // propagate query errors
        .await?;
    // enable rows in this query response
    let query_rows = query.into_rows_result()?;
    // set the type for our returned rows
    let netpol_list = query_rows
        .rows::<NetworkPolicyListRow>()?
        // propagate any into row errors
        .collect::<Result<Vec<NetworkPolicyListRow>, _>>()?
        .into_iter()
        // cast to list lines
        .map(Into::into)
        .collect();
    Ok(netpol_list)
}

/// Check if a network policy exists in multiple groups
///
/// Returns the groups containing the network policy
///
/// # Arguments
///
/// * `groups` - The groups to check
/// * `name` - The name of the network policy
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::exists", skip(shared), err(Debug))]
pub async fn exists(
    groups: &[String],
    name: &str,
    shared: &Shared,
) -> Result<Vec<String>, ApiError> {
    // TODO: kinda a goofy workaround, but it works; without asserting,
    // we get a weird compiler error "FnOnce isn't general enough";
    // see description for `assert_send_stream`
    let exists_groups = helpers::assert_send_stream(
        stream::iter(groups.iter())
            .map(|group| async move {
                // check if the policy exists in this group
                let query = shared
                    .scylla
                    .session
                    .execute_unpaged(&shared.scylla.prep.network_policies.exists, (name, group))
                    .await?;
                // try to enable rows on this query response
                let query_rows = query.into_rows_result()?;
                // check if didn't retrieve any rows
                if query_rows.rows_num() == 0 {
                    // no rows were returned, so the policy wasn't found
                    Ok(None)
                } else {
                    // get the first row returned
                    match query_rows.maybe_first_row::<(String, String)>()? {
                        Some((_, group)) => Ok(Some(group)),
                        // no rows were returned, so the policy wasn't found
                        None => Ok(None),
                    }
                }
            })
            .buffer_unordered(100),
    )
    .collect::<Vec<Result<Option<String>, ApiError>>>()
    .await
    .into_iter()
    // recollect into a single result, propagating the first error we found
    .collect::<Result<Vec<Option<String>>, ApiError>>()?
    .into_iter()
    // flatten into just the groups that were found
    .flatten()
    .collect();
    Ok(exists_groups)
}

/// Generate a pipeline to rename a network policy's data in Redis by finding
/// which groups it was already in, updating its keys and deleting its old name
/// and adding its new name to any sets it was in
///
/// Returns the pipe and its old list of groups
macro_rules! rename_redis {
    ($pipe:expr, $old_name:expr, $network_policy:expr, $update:expr, $shared:expr) => {
        async {
            // get all of the network policy's groups except for the ones we
            // just added in the update
            let old_groups: Vec<String> = $network_policy
                .groups
                .iter()
                .cloned()
                .filter(|group| !$update.add_groups.contains(group))
                .collect();
            let mut exists_pipe = redis::pipe();
            // first iteration checks for policy in each Redis netpols group key
            old_groups.iter().fold(&mut exists_pipe, |pipe, group| {
                // is it in this Redis group?
                pipe.cmd("sismember")
                    .arg(NetworkPolicyKeys::netpols(group, $shared))
                    .arg($old_name)
            });
            // second iteration checks for policy's used_by key
            old_groups.iter().fold(&mut exists_pipe, |pipe, group| {
                // does the used_by key exist for this group?
                pipe.cmd("exists")
                    .arg(NetworkPolicyKeys::used_by(group, $old_name, $shared))
            });
            let exists: Vec<bool> = exists_pipe
                .atomic()
                .query_async(conn!($shared))
                .await
                .map_err(ApiError::from)?;
            let mut exists_iter = exists.into_iter();
            let mut pipe = $pipe;
            // rename the network policy in the group sets
            old_groups
                .iter()
                // give zip() a mutable reference to advance the exists iterator
                // forward without moving it
                .zip(exists_iter.by_ref())
                // only rename the network policy in groups it was found in
                .filter_map(|(group, exists)| exists.then_some(group))
                .fold(&mut pipe, |pipe, group| {
                    // remove the old name
                    pipe.cmd("srem")
                        .arg(NetworkPolicyKeys::netpols(group, $shared))
                        .arg($old_name)
                        // add the new name
                        .cmd("sadd")
                        .arg(NetworkPolicyKeys::netpols(group, $shared))
                        .arg(&$network_policy.name)
                });
            // rename the network policy's used_by keys
            old_groups
                .iter()
                // iterate exists the rest of the way
                .zip(exists_iter)
                // only rename used_by keys if they exist
                .filter_map(|(group, exists)| exists.then_some(group))
                .fold(&mut pipe, |pipe, group| {
                    // rename any of the old keys to our new name
                    pipe.cmd("renamenx")
                        .arg(NetworkPolicyKeys::used_by(group, $old_name, $shared))
                        .arg(NetworkPolicyKeys::used_by(
                            group,
                            &$network_policy.name,
                            $shared,
                        ))
                });
            Ok::<_, ApiError>((pipe, old_groups))
        }
    };
}

/// Generate a pipe to delete network policy data in each group from Redis
macro_rules! delete_redis {
    ($pipe:expr, $del_groups:expr, $name:expr, $shared:expr) => {
        {
            let mut pipe = $pipe;
            $del_groups.iter().fold(&mut pipe, |pipe, group| {
                // delete the used_by data for this network policy in each group
                pipe.cmd("del")
                    .arg(NetworkPolicyKeys::used_by(group, $name, $shared))
                    // remove the network policy from the group's list of network policies with Redis data
                    .cmd("srem")
                    .arg(NetworkPolicyKeys::netpols(group, $shared))
                    .arg($name)
            });
            pipe
        }
    };
}

/// Update a network policy in the db's
///
/// # Arguments
///
/// * `network_policy` - The network policy to update
/// * `update` - The update to apply to the network policy
/// * `old_name` - The policy's old name if we updated it
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::update", skip_all, err(Debug))]
pub async fn update(
    network_policy: &NetworkPolicy,
    update: &NetworkPolicyUpdate,
    old_name: &Option<String>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // concurrently insert the updated network policy for each group
    stream::iter(network_policy.groups.iter())
        .map(Ok::<&String, ApiError>)
        .try_for_each_concurrent(100, |group| async move {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.network_policies.insert,
                    (
                        group,
                        &network_policy.name,
                        network_policy.id,
                        &network_policy.k8s_name,
                        &network_policy.created,
                        serialize!(&network_policy.ingress),
                        serialize!(&network_policy.egress),
                        network_policy.forced_policy,
                        network_policy.default_policy,
                    ),
                )
                .await?;
            Ok(())
        })
        .await?;
    // define a Redis pipeline to run, and the groups/policy name to delete from Scylla based
    // on whether we changed the policy's name and/or have removed groups
    let (mut pipe, scylla_delete_groups, scylla_delete_name) =
        match (old_name, update.remove_groups.is_empty()) {
            // if we didn't change the name and remove groups is empty, there's nothing to delete/update further
            (None, true) => return Ok(()),
            // we have removed groups so just delete those
            (None, false) => {
                // delete the network policy from those groups to remove
                let pipe = delete_redis!(
                    redis::pipe(),
                    update.remove_groups,
                    &network_policy.name,
                    shared
                );
                // delete from the groups we removed and delete
                // using our current name because we didn't update it
                (pipe, update.remove_groups.clone(), &network_policy.name)
            }
            // we have removed groups AND changed the name, so delete using the old name
            // in ALL groups, not just the ones to remove
            (Some(old_name), false) => {
                // make a pipeline to delete any of the removed groups
                let pipe = delete_redis!(redis::pipe(), update.remove_groups, old_name, shared);
                // add commands to rename any groups that are left
                let (pipe, old_groups) =
                    rename_redis!(pipe, old_name, network_policy, update, shared).await?;
                (
                    pipe,
                    // delete from the old groups the network policy is still in AND the
                    // groups we removed to get rid of the old name
                    old_groups
                        .into_iter()
                        .chain(update.remove_groups.iter().cloned())
                        .collect(),
                    // delete the old name from Scylla
                    old_name,
                )
            }
            // we changed the name, so delete the data at the old name
            (Some(old_name), true) => {
                // make a pipeline to rename the policy's data in Redis
                let (pipe, old_groups) =
                    rename_redis!(redis::pipe(), old_name, network_policy, update, shared).await?;
                // delete from all of its old groups in Scylla using the old name
                (pipe, old_groups, old_name)
            }
        };
    // add a command to invalidate the K8's cache
    pipe.cmd("hset")
        .arg(&SystemKeys::new(shared).data)
        .arg(K8S_CACHE_KEY)
        .arg(true);
    // update/delete data in Scylla and Redis concurrently
    tokio::try_join!(
        async {
            pipe.atomic()
                .exec_async(conn!(shared))
                .await
                .map_err(ApiError::from)
        },
        async {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.network_policies.delete,
                    (&scylla_delete_groups, scylla_delete_name),
                )
                .await
                .map_err(ApiError::from)
        }
    )?;
    Ok(())
}

/// Delete a network policy from the db's
///
/// # Arguments
///
/// * `network_policy` - The network policy to delete
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::delete", skip_all, err(Debug))]
pub async fn delete(network_policy: NetworkPolicy, shared: &Shared) -> Result<(), ApiError> {
    event!(
        Level::INFO,
        network_policy = network_policy.name,
        msg = "Deleting network policy"
    );
    // generate a pipe to delete a network policy's data from Redis
    let mut pipe = delete_redis!(
        redis::pipe(),
        network_policy.groups,
        &network_policy.name,
        shared
    );
    // add a command to invalidate the K8's cache
    pipe.cmd("hset")
        .arg(&SystemKeys::new(shared).data)
        .arg(K8S_CACHE_KEY)
        .arg(true);
    // delete from Redis and Scylla concurrently
    tokio::try_join!(
        // delete all the network policy's data in each group in Redis
        async {
            pipe.atomic()
                .exec_async(conn!(shared))
                .await
                .map_err(ApiError::from)
        },
        // delete the network policy's rows in each group in Scylla
        async {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.network_policies.delete,
                    (&network_policy.groups, &network_policy.name),
                )
                .await
                .map_err(ApiError::from)
        }
    )?;
    Ok(())
}

/// Delete all network policy data for a group
///
/// Does not remove network policies from images, because the images were
/// previously deleted with the group
///
/// # Arguments
///
/// * `group` - The group to delete policies from
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::delete_all_group", skip_all, err(Debug))]
pub async fn delete_all_group(group: &Group, shared: &Shared) -> Result<(), ApiError> {
    // log that we're deleting these policies
    event!(
        Level::INFO,
        group = group.name,
        msg = "Deleting all network policies"
    );
    // delete the group's network policy redis data
    let policy_names: Vec<String> = redis::cmd("smembers")
        .arg(NetworkPolicyKeys::netpols(&group.name, shared))
        .query_async(conn!(shared))
        .await?;
    // fold requests to delete all network policy data in this group from Redis into a Redis pipeline
    let mut pipe = redis::pipe();
    policy_names.iter().fold(&mut pipe, |pipe, policy_name| {
        pipe.cmd("del")
            .arg(NetworkPolicyKeys::used_by(&group.name, policy_name, shared))
    });
    // delete the policy list
    pipe.cmd("del")
        .arg(NetworkPolicyKeys::netpols(&group.name, shared));
    // delete from Redis and Scylla concurrently
    tokio::try_join!(
        // delete all of the group's network policy data in Redis
        async {
            pipe.atomic()
                .exec_async(conn!(shared))
                .await
                .map_err(ApiError::from)
        },
        // delete all of a group's network policy rows in Scylla
        async {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.network_policies.delete_all_in_group,
                    (&group.name,),
                )
                .await
                .map_err(ApiError::from)
        }
    )?;
    Ok(())
}

/// Get a list of images in the given groups used by the given network policy
///
/// Returns a list of tuples composed of the group and the images in that group
/// using the network policy
///
/// # Arguments
///
/// * `groups` - The groups to check for images that use the network policy
/// * `policy_name` - The name of the network policy
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::network_policies::used_by", skip(shared), err(Debug))]
pub async fn used_by(
    groups: &[String],
    policy_name: &str,
    shared: &Shared,
) -> Result<HashMap<String, Vec<String>>, ApiError> {
    // fold requests to get used_by sets in each group for this network policy into a Redis pipeline
    let mut pipe = redis::pipe();
    groups.iter().fold(&mut pipe, |pipe, group| {
        pipe.cmd("smembers")
            .arg(NetworkPolicyKeys::used_by(group, policy_name, shared))
    });
    // get the list of image lists by group
    let images: Vec<Vec<String>> = pipe.atomic().query_async(conn!(shared)).await?;
    // add groups to the list
    let images_with_groups = groups
        .iter()
        .zip(images.into_iter())
        .filter_map(|(group, image_list)| {
            // only add if the list isn't empty
            (!image_list.is_empty()).then_some((group.to_owned(), image_list))
        })
        .collect();
    Ok(images_with_groups)
}

/// Add/remove an image that one or more network policies `used_by` set
///
/// # Arguments
///
/// * `group` - One of the network policy's groups
/// * `policies_added` - The network policies added to this image
/// * `policies_removed` - The network policies removed from this image
/// * `image` - The image that's using the network policy
/// * `shared` - Shared Thorium objects
#[instrument(
    name = "db::network_policies::set_used_by",
    skip(policies_added, policies_removed, shared),
    err(Debug)
)]
pub async fn set_used_by<I, J, T, S>(
    group: &str,
    policies_added: I,
    policies_removed: J,
    image: &str,
    shared: &Shared,
) -> Result<(), ApiError>
where
    I: Iterator<Item = T>,
    J: Iterator<Item = S>,
    T: AsRef<str>,
    S: AsRef<str>,
{
    // fold requests to add the image to used_by set for each network policy into a Redis
    // pipeline, as well as add the network policies to the group to keep track of which
    // policies are added later
    let mut pipe = redis::pipe();
    policies_added.fold(&mut pipe, |pipe, policy_name| {
        pipe.cmd("sadd")
            .arg(NetworkPolicyKeys::netpols(group, shared))
            .arg(policy_name.as_ref())
            .cmd("sadd")
            .arg(NetworkPolicyKeys::used_by(
                group,
                policy_name.as_ref(),
                shared,
            ))
            .arg(image)
    });
    policies_removed.fold(&mut pipe, |pipe, policy_name| {
        pipe.cmd("srem")
            .arg(NetworkPolicyKeys::used_by(
                group,
                policy_name.as_ref(),
                shared,
            ))
            .arg(image)
    });
    // run pipeline to add this image to the network policies' used by sets
    let () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}
