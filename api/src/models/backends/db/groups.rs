use bb8_redis::redis::cmd;
use std::collections::{HashMap, HashSet};
use tracing::{instrument, span, Level, Span};

use super::helpers;
use super::keys::{EventKeys, GroupKeys, UserKeys};
use crate::models::{Group, GroupList, GroupRequest, Image, NetworkPolicy, Pipeline, User};
use crate::utils::{ApiError, Shared};
use crate::{
    conn, hset_del_opt_serialize, hsetnx_opt_serialize, log_err, not_found,
    query, serialize,
};

/// Adds the commands to modify users groups to a redis pipeline
///
/// # Arguments
///
/// * `pipe` - The Redis pipeline we are adding to
/// * `users` - The usernames to add for this role
/// * `cmd` - The redis command to execute
/// * `group` - the name of the group we are adding to
/// * `shared` - The Thorium shared object
macro_rules! modify_users {
    ($pipe:expr, $users:expr, $cmd:expr, $group:expr, $shared:expr) => {
        for name in $users {
            $pipe
                .cmd($cmd)
                .arg(UserKeys::groups(&name, $shared))
                .arg($group);
        }
    };
}

/// get the user lists and ldap sync groups for this group
///
/// it will be returned in a Vec<Vec<String>> format like
/// [[<owners>], [<managers>], [<users>], [<monitors>]]
#[rustfmt::skip]
macro_rules! get_members {
    ($pipe:expr, $group:expr, $shared:expr) => {
        $pipe.cmd("smembers").arg(GroupKeys::combined($group, "owners", $shared))
             .cmd("smembers").arg(GroupKeys::direct($group, "owners", $shared))
             .cmd("smembers").arg(GroupKeys::metagroups($group, "owners", $shared))
             .cmd("smembers").arg(GroupKeys::combined($group, "managers", $shared))
             .cmd("smembers").arg(GroupKeys::direct($group, "managers", $shared))
             .cmd("smembers").arg(GroupKeys::metagroups($group, "managers", $shared))
             .cmd("smembers").arg(GroupKeys::combined($group, "users", $shared))
             .cmd("smembers").arg(GroupKeys::direct($group, "users", $shared))
             .cmd("smembers").arg(GroupKeys::metagroups($group, "users", $shared))
             .cmd("smembers").arg(GroupKeys::combined($group, "monitors", $shared))
             .cmd("smembers").arg(GroupKeys::direct($group, "monitors", $shared))
             .cmd("smembers").arg(GroupKeys::metagroups($group, "monitors", $shared))
    }
}

/// Creates a group in the redis backend
///
/// # Arguments
///
/// * `user` - The user who is creating this group
/// * `req` - The group request to create in the backend
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::groups::create", skip_all, err(Debug))]
pub async fn create(
    user: &User,
    req: GroupRequest,
    shared: &Shared,
) -> Result<Group, ApiError> {
    // cast to a group object
    let cast = req.cast(user, shared).await?;
    // build group keys
    let keys = GroupKeys::new(&cast.name, shared);
    // build the key to our event cache status flags
    let cache_status = EventKeys::cache(shared);
    // build pipeline to modify user accounts groups and add this group
    let mut pipe = redis::pipe();
    // add command to insert this group into the group set
    pipe.cmd("sadd").arg(&keys.set).arg(&cast.name)
        // invalidate our cache status
        .cmd("hset").arg(cache_status).arg("status").arg(true)
        // set our group allowed settings
        .cmd("hset").arg(&keys.data).arg("allowed").arg(serialize!(&cast.allowed));
    // update user accounts
    modify_users!(pipe, &cast.owners.combined, "sadd", &cast.name, shared);
    modify_users!(pipe, &cast.managers.combined, "sadd", &cast.name, shared);
    modify_users!(pipe, &cast.users.combined, "sadd", &cast.name, shared);
    modify_users!(pipe, &cast.monitors.combined, "sadd", &cast.name, shared);
    // set combined user roles
    update_role(&mut pipe, &cast.owners.combined, &keys.combined_owners);
    update_role(&mut pipe, &cast.managers.combined, &keys.combined_managers);
    update_role(&mut pipe, &cast.users.combined, &keys.combined_users);
    update_role(&mut pipe, &cast.monitors.combined, &keys.combined_monitors);
    // set our direct user roles
    update_role(&mut pipe, &cast.owners.direct, &keys.direct_owners);
    update_role(&mut pipe, &cast.managers.direct, &keys.direct_managers);
    update_role(&mut pipe, &cast.users.direct, &keys.direct_users);
    update_role(&mut pipe, &cast.monitors.direct, &keys.direct_monitors);
    // set our metagroups user roles
    update_role(&mut pipe, &cast.owners.metagroups, &keys.metagroups_owners);
    update_role(&mut pipe, &cast.managers.metagroups, &keys.metagroups_managers);
    update_role(&mut pipe, &cast.users.metagroups, &keys.metagroups_users);
    update_role(&mut pipe, &cast.monitors.metagroups, &keys.metagroups_monitors);
    // add description
    hsetnx_opt_serialize!(pipe, &keys.data, "description", &cast.description);
    // execute pipeline and create our group
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(cast)
}

/// Gets a group from the backend
///
/// # Arguments
///
/// * `group` - The name of the group to get
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn get(group: &str, shared: &Shared) -> Result<Group, ApiError> {
    // get the user lists for this group
    let raw_members: MembersLists = get_members!(redis::pipe(), group, shared)
        .query_async(conn!(shared))
        .await?;
    // must get data in second connection, as pipelines are limited to 12 commands
    // build a pipeline to get the rest of this groups data
    let mut pipe = redis::pipe();
    // build the keys to this groups extra data
    let data_key = GroupKeys::data(group, shared);
    // build the key to all analysts in Thorium
    let analyst_key = UserKeys::analysts(shared);
    // get this groups description/allowed data
    let (raw_data, analysts): OtherData = pipe.cmd("hgetall").arg(data_key)
        .cmd("smembers").arg(analyst_key)
        .query_async(conn!(shared)).await?;
    // if there are no owners or ldap_owners then this group doesn't exist
    if raw_members.0.is_empty() && raw_members.3.is_empty() {
        not_found!(format!("group {} does not exist", group))
    } else {
        let group_data: RawGroupData = (group.to_owned(), raw_members, raw_data, analysts);
        Group::try_from(group_data)
    }
}

/// Lists all groups in the redis backend
///
/// # Arguments
///
/// * `cursor` - The cursor to use when paging through images
/// * `limit` - The number of objects to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list(cursor: usize, limit: usize, shared: &Shared) -> Result<GroupList, ApiError> {
    // key to group set
    let key = GroupKeys::set(shared);
    // get list of created groups
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(GroupList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(GroupList::new(Some(new_cursor), names))
    }
}

/// Raw lists of members returned from database
///
/// Owners > Managers > Users > Monitors | description
pub type MembersLists = (
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
);

/// The remaining data to retrieve for this group
pub type OtherData = (HashMap<String, String>, HashSet<String>);

/// Raw group data returned from database
///
/// Contains name, MembersLists, and data
pub type RawGroupData = (
    String,
    MembersLists,
    HashMap<String, String>,
    HashSet<String>,
);

/// Lists all groups in the redis backend with their details
///
/// # Arguments
///
/// * `groups` - The names of the groups to get details for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::groups::list_details", skip_all, err(Debug))]
pub async fn list_details<'a, T>(groups: T, shared: &Shared) -> Result<Vec<Group>, ApiError>
where
    T: Iterator<Item = &'a String> + Clone,
{
    // get members lists of all requested groups
    let raw_members: Vec<MembersLists> = groups
        .clone()
        .fold(redis::pipe().atomic(), |pipe, name| {
            get_members!(pipe, name, shared)
        })
        .query_async(conn!(shared))
        .await?;
    // get data of all requested groups
    let raw_data: Vec<HashMap<String, String>> = {
        let mut raw_data: Vec<HashMap<String, String>> = Vec::new();
        for name in groups.clone() {
            raw_data.push(
                redis::cmd("hgetall")
                    .arg(GroupKeys::data(name, shared))
                    .query_async(conn!(shared))
                    .await?,
            );
        }
        raw_data
    };
    // build the key to all analysts in Thorium
    let analyst_key = UserKeys::analysts(shared);
    // get the list of analysts in Thorium
    let analysts: HashSet<String> = query!(cmd("smembers").arg(analyst_key), shared).await?;
    // build an iterator that zips 3 names, members lists, and data into one
    let zip_iter = itertools::izip!(groups.cloned(), raw_members, raw_data);
    // cast the zipped iter to a vector of groups
    let groups = zip_iter
        .into_iter()
        .map(|(name, members, data)| Group::try_from((name, members, data, &analysts)))
        .filter_map(|res| log_err!(res))
        .collect();
    Ok(groups)
}

/// gets details on all groups in redis for a backup
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::groups::backup", skip_all, err(Debug))]
pub async fn backup(shared: &Shared) -> Result<Vec<Group>, ApiError> {
    // build key to group set
    let key = GroupKeys::set(shared);
    // get all group names
    let names: Vec<String> = query!(cmd("smembers").arg(key), shared).await?;
    // build pipeline to retrieve all group data
    let mut pipe = redis::pipe();
    names.iter().fold(&mut pipe, |pipe, name| get_members!(pipe, name, shared));
    let raw_members: Vec<MembersLists> = pipe.query_async(conn!(shared)).await?;
    // get data of all requested groups
    let raw_data: Vec<HashMap<String, String>> = {
        let mut raw_data: Vec<HashMap<String, String>> = Vec::new();
        for name in names.iter() {
            raw_data.push(
                redis::cmd("hgetall")
                    .arg(GroupKeys::data(name, shared))
                    .query_async(conn!(shared))
                    .await?
            );
        }
        raw_data
    };
    // build the key to all analysts in Thorium
    let analyst_key = UserKeys::analysts(shared);
    // get the list of analysts in Thorium
    let analysts: HashSet<String> = query!(cmd("smembers").arg(analyst_key), shared).await?;
    // combine the data retrieved with the list of names and cast it to group objects
    let zip_iter = itertools::izip!(names, raw_members, raw_data);
    // cast the zipped iter to a vector of groups
    //let groups = cast!(zip_iter, Group::try_from);
    // cast the zipped iter to a vector of groups
    let groups = zip_iter
        .into_iter()
        .map(|(name, members, data)| Group::try_from((name, members, data, &analysts)))
        .filter_map(|res| log_err!(res))
        .collect();
    Ok(groups)
}

/// Restore group data
///
/// # Arguments
///
/// * `groups` - The list of groups to restore
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
pub async fn restore(groups: &[Group], shared: &Shared, span: &Span) -> Result<(), ApiError> {
    // start our restore groups to redis span
    span!(parent: span, Level::INFO, "Restore Groups To Redis");
    // build the our redis pipeline
    let mut pipe = redis::pipe();
    // crawl over groups and build the pipeline to restore each one
    for group in groups.iter() {
        // build group keys
        let keys = GroupKeys::new(&group.name, shared);
        // add command to insert this group into the group set
        pipe.cmd("sadd").arg(&keys.set).arg(&group.name);
        // set combined user roles
        update_role(&mut pipe, &group.owners.combined, &keys.combined_owners);
        update_role(&mut pipe, &group.managers.combined, &keys.combined_managers);
        update_role(&mut pipe, &group.users.combined, &keys.combined_users);
        update_role(&mut pipe, &group.monitors.combined, &keys.combined_monitors);
        // set our direct user roles
        update_role(&mut pipe, &group.owners.direct, &keys.direct_owners);
        update_role(&mut pipe, &group.managers.direct, &keys.direct_managers);
        update_role(&mut pipe, &group.users.direct, &keys.direct_users);
        update_role(&mut pipe, &group.monitors.direct, &keys.direct_monitors);
        // set our metagroups user roles
        update_role(&mut pipe, &group.owners.metagroups, &keys.metagroups_owners);
        update_role(
            &mut pipe,
            &group.managers.metagroups,
            &keys.metagroups_managers,
        );
        update_role(&mut pipe, &group.users.metagroups, &keys.metagroups_users);
        update_role(
            &mut pipe,
            &group.monitors.metagroups,
            &keys.metagroups_monitors,
        );
        // add command to update description
        hset_del_opt_serialize!(pipe, &keys.data, "description", &group.description);
    }
    // restore this group to redis
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Checks if groups exist in the Redis backend
///
/// # Arguments
///
/// * `names` - The names of the groups to check the existence of
/// * `shared` - Shared Thorium objects
pub async fn exists<'a>(names: &[String], shared: &Shared) -> Result<bool, ApiError> {
    helpers::exists_all(names, &GroupKeys::set(shared), shared).await
}

/// update the users for a role in a group
fn update_role<'a>(pipe: &'a mut redis::Pipeline, users: &HashSet<String>, key: &str) {
    // clear this key first
    pipe.cmd("del").arg(key);
    // add each of our users to this set
    for user in users.iter() {
        pipe.cmd("sadd").arg(key).arg(user);
    }
}

/// Updates a group in backend
///
/// # Arguments
///
/// * `group` - The name of the group to update in the backend
/// * `update` - The updates to apply to this group
/// * `added` - The users to add this group to
/// * `removed` - The users to remove this group from
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
#[instrument(name = "db::groups::update", skip_all, fields(group = &group.name), err(Debug))]
pub async fn update(
    group: &Group,
    added: &HashSet<String>,
    removed: &HashSet<String>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build image keys
    let keys = GroupKeys::new(&group.name, shared);
    // build the key to our event cache status flags
    let cache_status = EventKeys::cache(shared);
    // build pipeline to modify user accounts groups and add this group
    let mut pipe = redis::pipe();
    // make sure that we only add/remove users whose group memberships are changing
    // and not just are changing roles
    // update user accounts
    modify_users!(pipe, added.difference(removed), "sadd", &group.name, shared);
    modify_users!(pipe, removed.difference(added), "srem", &group.name, shared);
    // set combined user roles
    update_role(&mut pipe, &group.owners.combined, &keys.combined_owners);
    update_role(&mut pipe, &group.managers.combined, &keys.combined_managers);
    update_role(&mut pipe, &group.users.combined, &keys.combined_users);
    update_role(&mut pipe, &group.monitors.combined, &keys.combined_monitors);
    // set our direct user roles
    update_role(&mut pipe, &group.owners.direct, &keys.direct_owners);
    update_role(&mut pipe, &group.managers.direct, &keys.direct_managers);
    update_role(&mut pipe, &group.users.direct, &keys.direct_users);
    update_role(&mut pipe, &group.monitors.direct, &keys.direct_monitors);
    // set our metagroups user roles
    update_role(&mut pipe, &group.owners.metagroups, &keys.metagroups_owners);
    update_role(&mut pipe, &group.managers.metagroups, &keys.metagroups_managers);
    update_role(&mut pipe, &group.users.metagroups, &keys.metagroups_users);
    update_role(&mut pipe, &group.monitors.metagroups, &keys.metagroups_monitors);
    // update description
    hset_del_opt_serialize!(pipe, &keys.data, "description", &group.description);
    // invalidate our event cache
    pipe.cmd("hset").arg(cache_status).arg("status").arg(true);
    // set our group allowed settings
    pipe.cmd("hset").arg(&keys.data).arg("allowed").arg(serialize!(&group.allowed));
    // execute pipeline and check if it failed
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// get all groups in Thorium
pub async fn list_all(user: &User, shared: &Shared) -> Result<Vec<Group>, ApiError> {
    // crawl over all groups and build a list of them
    let mut cursor = 0;
    let mut groups = Vec::with_capacity(20);
    loop {
        // get 100 groups at a time
        let mut chunk = Group::list_details(user, cursor, 100, shared).await?;
        // add this chunk of groups to our list
        groups.append(&mut chunk.details);
        // check if our cursor has been exhausted
        if chunk.cursor.is_none() {
            break;
        }
        // update cursor
        cursor = chunk.cursor.unwrap();
    }
    Ok(groups)
}

/// Deletes a group from the redis backend
///
/// # Arguments
///
/// * `user` - The user that is deleting this group
/// * `group` - The group object to remove from the backend
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
#[instrument(name = "db::groups::delete", skip_all, fields(group = &group.name), err(Debug))]
pub async fn delete(
    user: &User,
    group: &Group,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build group keys
    let keys = GroupKeys::new(&group.name, shared);
    // delete all pipelines and their jobs in this group
    Pipeline::delete_all(user, group, shared).await?;
    // delete all images in this group
    Image::delete_all(user, group, shared).await?;
    // delete network policies from this group
    NetworkPolicy::delete_all_group(user, group, shared).await?;
    // build pipeline to modify user accounts groups and add this group
    let mut pipe = redis::pipe();
    // remove this group from its user accounts
    modify_users!(pipe, &group.owners.combined, "srem", &group.name, shared);
    modify_users!(pipe, &group.managers.combined, "srem", &group.name, shared);
    modify_users!(pipe, &group.users.combined, "srem", &group.name, shared);
    modify_users!(pipe, &group.monitors.combined, "srem", &group.name, shared);
    // delete this groups user sets
    pipe.cmd("del").arg(&keys.combined_owners)
        .cmd("del").arg(&keys.combined_managers)
        .cmd("del").arg(&keys.combined_users)
        .cmd("del").arg(&keys.combined_monitors)
        .cmd("del").arg(&keys.direct_owners)
        .cmd("del").arg(&keys.direct_managers)
        .cmd("del").arg(&keys.direct_users)
        .cmd("del").arg(&keys.direct_monitors)
        .cmd("del").arg(&keys.metagroups_owners)
        .cmd("del").arg(&keys.metagroups_managers)
        .cmd("del").arg(&keys.metagroups_users)
        .cmd("del").arg(&keys.metagroups_monitors)
        // delete data (e.g. description)
        .cmd("del").arg(&keys.data)
        // remove this group from the global group set
        .cmd("srem").arg(&keys.set).arg(&group.name);
    // attempt to delete group from redis backend
    // we can't confirm the delete because if a role has no users then it will return false
    () = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}
