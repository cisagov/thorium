//! Logic for interacting with entities in the databases

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Datelike, Utc};
use futures::stream::{self, StreamExt, TryStreamExt};
use itertools::Itertools;
use tracing::instrument;
use uuid::Uuid;

use crate::models::backends::TagSupport;
use crate::models::backends::db::ScyllaCursor;
use crate::models::{
    AssociationListOpts, AssociationListParams, AssociationTargetColumn, Entity, EntityForm,
    EntityListLine, EntityListParams, EntityListSupplementRow, EntityMetadata, EntityRow,
    KeySupport, TagDeleteRequest, TagRequest, User,
};
use crate::utils::{ApiError, Shared, helpers};
use crate::{bad, not_found, serialize};

/// Create a `Entity` in Scylla
///
/// # Arguments
///
/// * `req` - The request to create an entity
/// * `entity_id` - The ID to set for this entity
/// * `user` - The user creating the entity
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::entities::create", skip(shared), err(Debug))]
pub async fn create(
    user: &User,
    mut form: EntityForm,
    id: Uuid,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build an association req for this entity
    let assoc_req = form.build_association_req(user, id, shared).await?;
    // take our form tags since we need those to setup tags and this entity object
    // gets abandoned in this function anyways
    let tags = std::mem::take(&mut form.tags);
    // cast our form to an entity
    let entity = form.cast(user, id, shared).await?;
    // serialize our metadata
    let serialized_meta = serialize!(&entity.metadata);
    // get the current timestamp for when this entity was created
    let now = Utc::now();
    let year = now.year();
    // get the partition size for repos
    let chunk_size = shared.config.thorium.entities.partition_size;
    // get the partition to write this repo off too
    let bucket = helpers::partition(now, year, chunk_size);
    // concurrently insert the entity in each group
    stream::iter(entity.groups.iter())
        .map(Ok::<&String, ApiError>)
        .try_for_each_concurrent(100, |group| {
            let local_entity = &entity;
            let local_meta = &serialized_meta;
            async move {
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.entities.insert,
                        (
                            entity.kind,
                            group,
                            &year,
                            &bucket,
                            &now,
                            &entity.id,
                            &local_entity.name,
                            &user.username,
                            local_meta,
                            &local_entity.description,
                            &local_entity.image,
                        ),
                    )
                    .await?;
                Ok(())
            }
        })
        .await?;
    // make a vec to store our keys
    let mut keys = Vec::with_capacity(entity.groups.len());
    // calculate the grouping for this form
    let grouping = bucket / 10_000;
    // build the keys for this items census cache
    super::keys::entities::census_keys(&mut keys, &entity.groups, year, bucket, grouping, shared);
    // update this samples census cache info
    super::census::incr_cache(keys, shared).await?;
    // add tags if we have any
    if !tags.is_empty() {
        // build a tag request
        let mut req = TagRequest::<Entity>::default().groups(entity.groups.clone());
        // move our tags to the request
        req.tags = tags;
        // create the earliest map; in this case it's just the created timestamp for all groups
        let earliest = entity
            .groups
            .iter()
            .map(|group| (group.clone(), now))
            .collect();
        // save the tags in scylla
        super::tags::create_owned(
            user,
            Entity::build_key(entity.id.to_string(), &()),
            req,
            &earliest,
            shared,
        )
        .await?;
    }
    // create any associations if requested
    if let Some(assoc_req) = assoc_req {
        // create the associations for this entity
        assoc_req.apply(user, shared).await?;
    }
    Ok(())
}

/// Get an entity from Scylla
///
/// # Arguments
///
/// * `groups` - The list of groups to check for the entity
/// * `name` - The name of the entity
/// * `id` - An optional network policy ID required if one or more distinct entities
///          share the same name
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::entities::get", skip(shared), err(Debug))]
pub async fn get(groups: &[String], id: Uuid, shared: &Shared) -> Result<Entity, ApiError> {
    // make sure we don't have empty groups
    if groups.is_empty() {
        return bad!("Groups cannot be empty when getting an entity".to_string());
    }
    let mut entity: Option<Entity> = None;
    // break our groups into chunks of 100
    for groups_chunk in groups.chunks(100) {
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.entities.get, (&id, groups_chunk))
            .await?;
        // enable rows on this query response
        let query_rows = query.into_rows_result()?;
        // set the type for our rows
        let typed_iter = query_rows.rows::<EntityRow>()?;
        // cast rows to entity rows and save the info
        for row in typed_iter {
            // make sure the row is valid
            let row = row?;
            // get the entity we are building or cast this row to one
            match entity.as_mut() {
                Some(handle) => handle.groups.push(row.group),
                None => {
                    entity = Some(Entity::try_from(row)?);
                }
            }
        }
    }
    // if we didn't build an entity then return a 404
    match entity {
        Some(mut entity) => {
            // get the tags for this entity
            entity.get_tags(groups, shared).await?;
            // return the entity and its tags
            Ok(entity)
        }
        None => not_found!(format!("Entity {id} not found")),
    }
}

/// List entities in specific groups and with tags
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `dedupe` - Whether to dedupe submissions for the same sha256
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::entities::list", skip(shared), err(Debug))]
pub async fn list(
    params: EntityListParams,
    dedupe: bool,
    shared: &Shared,
) -> Result<ScyllaCursor<EntityListLine>, ApiError> {
    // get our cursor
    let mut cursor = ScyllaCursor::from_params(params, dedupe, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // if we are searching on tags, we need to supplement the data
    // with names+kinds because tag rows don't have them
    if cursor.retain.tags_retain.is_some() {
        supplement_tag_lines(&mut cursor.data, shared).await?;
    }
    // save this cursor
    cursor.save(shared).await?;
    Ok(cursor)
}

/// Get all details for many entities
///
/// # Arguments
///
/// * `groups` - The groups to get from
/// * `names` - The names of the entities to get details on
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::entities::get_many", skip(shared), err(Debug))]
pub async fn get_many(
    groups: &[String],
    ids: &[Uuid],
    shared: &Shared,
) -> Result<Vec<Entity>, ApiError> {
    // Create a btreemap to keep the entities sorted by creation timestamp.
    // Make the value a list of entity ids in case they were uploaded at the exact same time.
    // We're okay to have a list with possibly repeated ids because we'll take from a the data
    // map to guarantee we don't return repeats. This also maintains insert order.
    let mut sorted: BTreeMap<DateTime<Utc>, Vec<Uuid>> = BTreeMap::new();
    // make a map of the entity's id to its actual data
    let mut data_map: HashMap<Uuid, Entity> = HashMap::new();
    for (ids_chunk, groups_chunk) in ids.chunks(50).cartesian_product(groups.chunks(50)) {
        // query for this chunk combo's data
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.entities.get_many,
                (ids_chunk, groups_chunk),
            )
            .await?;
        // make sure we got rows
        let query_rows = query.into_rows_result()?;
        // try to cast rows to entiry rows
        for cast in query_rows.rows::<EntityRow>()? {
            // make sure the row is valid
            let row = cast?;
            // get an entry to the timestamp map
            let timestamp_list = sorted.entry(row.created).or_default();
            // add our entity's id to it
            timestamp_list.push(row.id);
            // set this entity's details
            match data_map.entry(row.id) {
                Entry::Vacant(empty) => {
                    // make a new entity from the row
                    let entity = Entity::try_from(row)?;
                    // insert it into the entry
                    empty.insert(entity);
                }
                // insert the row's group if we already have this entity
                Entry::Occupied(occupied) => occupied.into_mut().add_row(row),
            }
        }
    }
    // get tags for all of the entities concurrently
    stream::iter(data_map.values_mut())
        .map(Ok::<_, ApiError>)
        .try_for_each_concurrent(10, |entity| async {
            entity.get_tags(groups, shared).await?;
            Ok(())
        })
        .await?;
    // compile our details into a final list
    let mut details = Vec::with_capacity(data_map.len());
    // keep creation order by using the sorted map
    for (_, ids) in sorted.into_iter().rev() {
        for id in ids {
            // try to get the entity's details if we haven't already
            if let Some(entity) = data_map.remove(&id) {
                details.push(entity);
            }
        }
    }
    Ok(details)
}

/// Supplement entity list lines from tag rows with the names and kinds of
/// their entities; tag rows only have the entities' ids
///
/// # Arguments
///
/// * `lines` - The lines to supplement
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::entities::supplement_tag_lines", skip_all, err(Debug))]
async fn supplement_tag_lines(
    lines: &mut [EntityListLine],
    shared: &Shared,
) -> Result<(), ApiError> {
    // make a map of ids to their index in the mutable vec
    let line_map = lines
        .iter()
        .enumerate()
        .map(|(index, line)| (line.id, index))
        .collect::<HashMap<_, _>>();
    // collect all our ids to a list to chunk them
    let ids = lines.iter().map(|line| &line.id).collect::<Vec<_>>();
    // get all of the supplement rows concurrently
    let supplement_rows = helpers::assert_send_stream(
        stream::iter(ids.chunks(100))
            .map(|ids_chunk| async move {
                // get the names/kinds from these ids
                let query = shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.entities.get_names_kinds_by_ids,
                        (ids_chunk,),
                    )
                    .await?;
                // try to enable rows on this query response
                let query_rows = query.into_rows_result()?;
                let mut rows = Vec::new();
                for row in query_rows.rows::<EntityListSupplementRow>()? {
                    rows.push(row?);
                }
                Ok(rows)
            })
            .buffer_unordered(10),
    )
    .collect::<Vec<Result<_, ApiError>>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    // add the supplemental row info to our lines;
    // we'll have duplicate rows due to groups, but that's okay
    for row in supplement_rows.into_iter().flatten() {
        let index = line_map.get(&row.id).copied().unwrap_or_default();
        let line = &mut lines[index];
        line.name = row.name;
        line.kind = row.kind;
    }
    Ok(())
}

/// Update an entity with its new data, removing rows for removed groups
/// and old names if needed
///
/// # Arguments
///
/// * `entity` - The entity that's being updated
/// * `add_groups` - The groups that were added to the entity
/// * `remove_groups` - The groups that were removed from the entity
/// * `shared` - Shared Thorium objects
pub async fn update(
    mut entity: Entity,
    add_groups: &[String],
    remove_groups: &[String],
    shared: &Shared,
) -> Result<(), ApiError> {
    // drop any association specific data
    entity.drop_associated_data();
    // serialize our metadata
    let serialized_meta = serialize!(&entity.metadata);
    // ge the year this entity was created
    let year = entity.created.year();
    // get the partition size for repos
    let chunk_size = shared.config.thorium.entities.partition_size;
    // get the partition to write this repo off too
    let bucket = helpers::partition(entity.created, year, chunk_size);
    // concurrently update the entity in each group
    stream::iter(entity.groups.iter())
        .map(Ok::<&String, ApiError>)
        .try_for_each_concurrent(100, |group| {
            let local_entity = &entity;
            let local_meta = &serialized_meta;
            async move {
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.entities.insert,
                        (
                            entity.kind,
                            group,
                            &year,
                            &bucket,
                            &entity.created,
                            &entity.id,
                            &local_entity.name,
                            &local_entity.submitter,
                            local_meta,
                            &local_entity.description,
                            &local_entity.image,
                        ),
                    )
                    .await?;
                Ok(())
            }
        })
        .await?;
    // break up the groups delete into chunks to avoid cartesian product errors
    let delete_groups_chunks = remove_groups.chunks(100);
    // concurrently delete the entity's old data for each group chunk
    stream::iter(delete_groups_chunks)
        .map(Ok::<_, ApiError>)
        .try_for_each_concurrent(100, |groups_chunk| async move {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.entities.delete,
                    (
                        entity.kind,
                        groups_chunk,
                        &year,
                        &bucket,
                        &entity.created,
                        &entity.id,
                    ),
                )
                .await?;
            Ok(())
        })
        .await?;
    // update census info for any new groups
    if !add_groups.is_empty() {
        // increment the census where we've added groups
        let mut keys = Vec::with_capacity(entity.groups.len());
        // get the bucket grouping for this census info
        let grouping = bucket / 10_000;
        super::keys::entities::census_keys(&mut keys, add_groups, year, bucket, grouping, shared);
        super::census::incr_cache(keys, shared).await?;
    }
    // update census info for any removed groups
    if !remove_groups.is_empty() {
        // decrement the census where we've removed groups
        let mut keys = Vec::with_capacity(entity.groups.len());
        // get the bucket grouping for this census info
        let grouping = bucket / 10_000;
        super::keys::entities::census_keys(
            &mut keys,
            remove_groups,
            year,
            bucket,
            grouping,
            shared,
        );
        super::census::decr_cache(keys, shared).await?;
    }
    // delete tag rows for the deleted groups
    let mut tag_delete_req = TagDeleteRequest::<Entity>::default().groups(remove_groups);
    // delete all tags from these groups
    tag_delete_req.tags = entity
        .tags
        .iter()
        .map(|(key, values_with_groups)| {
            (key.clone(), values_with_groups.keys().cloned().collect())
        })
        .collect();
    super::tags::delete(
        &Entity::build_key(entity.id.to_string(), &()),
        &tag_delete_req,
        shared,
    )
    .await?;
    Ok(())
}

/// Delete all tags from an entity
///
/// # Arguments
///
/// * `user` - The user that is pruning tags from an entity
/// * `entity` - The entity that is having its tags pruned
/// * `shared` - Shared Thorium objects
async fn prune_tags(user: &User, entity: &Entity, shared: &Shared) -> Result<(), ApiError> {
    // cast this entities tags into a tag deletion request
    let mut tag_del = TagDeleteRequest::<Entity>::default();
    // build a set of groups to delete tags from
    let mut group_set = HashSet::with_capacity(entity.groups.len());
    // iterate and all all of our entities tags
    for (key, value_map) in &entity.tags {
        // add each tag value and its grousp to our tag delete request
        for (value, groups) in value_map {
            // add this group to the set of groups to delete tags from
            group_set.extend(groups.iter().map(ToOwned::to_owned));
            // add this value to our tags to delete
            tag_del.add_ref(key, value);
        }
    }
    // set the groups to delete tags form
    tag_del.groups = group_set.into_iter().collect();
    // delete these tags if our tag delete request isn't empty
    if !tag_del.tags.is_empty() {
        // delete the requested tags
        entity.delete_tags(user, tag_del, shared).await?;
    }
    Ok(())
}

/// Delete an entity completely from all of its groups
///
/// # Arguments
///
/// * `user` - The user that is deleting this entity
/// * `entity` - The entity that's being updated
/// * `shared` - Shared Thorium objects
pub async fn delete(user: &User, entity: &Entity, shared: &Shared) -> Result<(), ApiError> {
    // we delete tags and associations this first so if any failures occur we don't leave
    // dangling references
    // prune this entities tags first
    prune_tags(user, entity, shared).await?;
    // get the year this entity was created
    let year = entity.created.year();
    // get the partition size for entities
    let chunk_size = shared.config.thorium.entities.partition_size;
    // get the bucket for this entity
    let bucket = helpers::partition(entity.created, year, chunk_size);
    // concurrently delete the entity's rows
    stream::iter(entity.groups.chunks(100))
        .map(Ok::<_, ApiError>)
        .try_for_each_concurrent(100, |groups_chunk| async move {
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.entities.delete,
                    (
                        entity.kind,
                        groups_chunk,
                        &year,
                        &bucket,
                        &entity.created,
                        &entity.id,
                    ),
                )
                .await?;
            Ok(())
        })
        .await?;
    // make a vec to store our census keys
    // we're deleting so there will only ever be one
    let mut keys = Vec::with_capacity(entity.groups.len());
    // calculate the grouping for this form
    let grouping = bucket / 10_000;
    // build the keys for this items census cache
    super::keys::entities::census_keys(&mut keys, &entity.groups, year, bucket, grouping, shared);
    // update this samples census cache info
    super::census::decr_cache(keys, shared).await?;
    Ok(())
}
