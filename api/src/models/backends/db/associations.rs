//! Save/Load associations into the DB

use chrono::prelude::*;
use futures::stream::{self, StreamExt};
use scylla::errors::ExecutionError;
use tracing::instrument;

use super::{ScyllaCursor, keys};
use crate::models::{
    AssociationKind, AssociationListParams, AssociationTarget, AssociationTargetColumn,
    Directionality, ListableAssociation, User,
};
use crate::serialize;
use crate::utils::{ApiError, Shared, helpers};

#[instrument(name = "db::associations::create_helper", skip_all, err(Debug))]
async fn create_helper(
    user: &User,
    group: &String,
    year: i32,
    bucket: i32,
    now: DateTime<Utc>,
    kind: AssociationKind,
    source_str: &String,
    target_str: String,
    extra_source: Option<String>,
    extra_target: Option<String>,
    direction: Directionality,
    shared: &Shared,
) -> Result<(), ExecutionError> {
    // This row is in the source -> target direction
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.associations.insert,
            (
                group,
                year,
                bucket,
                now,
                direction,
                kind,
                &source_str,
                &target_str,
                &user.username,
                &extra_source,
                &extra_target,
            ),
        )
        .await?;
    // get our opposite direction
    let opposite_dir = direction.opposite();
    // This row is in the target  -> source direction
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.associations.insert,
            (
                group,
                year,
                bucket,
                now,
                opposite_dir,
                kind,
                target_str,
                &source_str,
                &user.username,
                extra_target,
                extra_source,
            ),
        )
        .await?;
    Ok(())
}

#[instrument(
    name = "db::associations::create",
    skip(user, targets, shared),
    err(Debug)
)]
pub async fn create(
    user: &User,
    size_hint: usize,
    kind: AssociationKind,
    source: AssociationTarget,
    targets: &Vec<(AssociationTarget, Vec<String>)>,
    direction: Directionality,
    shared: &Shared,
) -> Result<(), ApiError> {
    // get the current time for when we are inserting these rows
    let now = Utc::now();
    // get the current year
    let year = now.year();
    // get the partition size to use for files and tags
    let chunk = shared.config.thorium.associations.partition_size;
    // calculate the bucket for our timestamp
    let bucket = helpers::partition(now, year, chunk);
    // convert our source
    let (converted_src, extra_src) = source.to_column()?;
    // serialize our source info
    let source_str = serialize!(&converted_src);
    // keep a set of keys of census info to update
    let mut keys = Vec::with_capacity(size_hint + 10);
    // setup a list to store all of the futures were about to build
    let mut futs = Vec::with_capacity(size_hint);
    // build a set of groups that our associations_from table is getting rows for
    let mut groups = Vec::with_capacity(10);
    // iterate over all of our targets and build futures to add them
    for (target, target_groups) in targets {
        // clone and cast our target to a target column
        let (converted_targ, extra_targ) = target.clone().to_column()?;
        // serialize our source info
        let target_str = serialize!(&converted_targ);
        // build the keys for our associations_to census cache
        keys::associations::census_keys(
            &mut keys,
            &target_groups,
            year,
            bucket,
            &source_str,
            &target_str,
            shared,
        );
        // create futures for each group for this target
        for group in target_groups {
            // if this group isn't already in our from_groups then add it
            if groups.contains(&group) {
                // add this group to our from groups list
                groups.push(group);
            }
            // create a future to insert our associations
            let fut = create_helper(
                user,
                group,
                year,
                bucket,
                now,
                kind,
                &source_str,
                target_str.clone(),
                extra_src.clone(),
                extra_targ.clone(),
                direction,
                shared,
            );
            // add this to our futures
            futs.push(fut);
        }
    }
    // execute our to/from futures
    stream::iter(futs)
        .buffer_unordered(25)
        .collect::<Vec<Result<(), ExecutionError>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<()>, ExecutionError>>()?;
    // update this samples census cache info
    super::census::incr_cache(keys, shared).await?;
    Ok(())
}

/// Help delete both the regular and inverted rows for this association
async fn delete_helper(
    group_chunk: &[String],
    year: i32,
    bucket: i32,
    source_serialized: &String,
    association: &ListableAssociation,
    shared: &Shared,
) -> Result<(), ExecutionError> {
    // build the future to delete this association
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.associations.delete,
            (
                group_chunk,
                year,
                bucket,
                &source_serialized,
                association.created,
                &association.other,
                association.direction,
            ),
        )
        .await?;
    // get our opposite direction
    let opposite_dir = association.direction.opposite();
    // build the future to delete this association
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.associations.delete,
            (
                group_chunk,
                year,
                bucket,
                &association.other,
                association.created,
                &source_serialized,
                opposite_dir,
            ),
        )
        .await?;
    Ok(())
}

/// Delete a list of associations from multiple groups
pub async fn delete_many(
    source: &AssociationTargetColumn,
    associations: &Vec<ListableAssociation>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // serialize our source
    let source_serialized = serialize!(&source);
    // instance a list of futures for our deletes
    let mut futs = Vec::with_capacity(associations.len());
    // create the futures for each association to delete
    for assoc in associations {
        // get the year and bucket for this association
        let year = assoc.created.year();
        // get the partition size to use for files and tags
        let chunk = shared.config.thorium.associations.partition_size;
        // calculate the bucket for our timestamp
        let bucket = helpers::partition(assoc.created, year, chunk);
        // chunk this associations groups into chunks of 99
        for group_chunk in assoc.groups.chunks(99) {
            // build the future to delete this association
            let fut = delete_helper(group_chunk, year, bucket, &source_serialized, assoc, shared);
            // add this future to our future set
            futs.push(fut);
        }
    }
    // execute all of our deletes 10 at a time
    stream::iter(futs)
        .buffer_unordered(10)
        .collect::<Vec<Result<(), ExecutionError>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<()>, ExecutionError>>()?;
    Ok(())
}

/// List associations for a specific entity/file/repo
///
/// # Arguments
///
/// * `params` - The query params to use when listing files
/// * `dedupe` - Whether to dedupe submissions for the same sha256
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::associations::list", skip(shared), err(Debug))]
pub async fn list<P: Into<AssociationListParams> + std::fmt::Debug>(
    opts: P,
    source: &AssociationTargetColumn,
    shared: &Shared,
) -> Result<ScyllaCursor<ListableAssociation>, ApiError> {
    // convert our params
    let params = opts.into();
    // serialize our association target column before list things related to it
    let source_str = serialize!(&source);
    // get our cursor
    let mut cursor = ScyllaCursor::from_params_extra(params, source_str, false, shared).await?;
    // get the next page of data for this cursor
    cursor.next(shared).await?;
    // save this cursor
    cursor.save(shared).await?;
    Ok(cursor)
}
