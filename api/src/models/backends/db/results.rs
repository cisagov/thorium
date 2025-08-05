//! Saves results into the backend

use chrono::prelude::*;
use itertools::Itertools;
use std::collections::{BTreeMap, HashMap};
use tracing::{event, instrument, span, Level, Span};
use uuid::Uuid;

use crate::models::backends::OutputSupport;
use crate::models::{
    Output, OutputDisplayType, OutputForm, OutputId, OutputIdRow, OutputKind, OutputMap, OutputRow,
    ResultSearchEvent,
};
use crate::utils::{helpers, ApiError, Shared};
use crate::{internal_err, log_scylla_err, unauthorized};

/// Saves a files result into the backend
///
/// # Arguments
///
/// * `key` - The key to use when saving our results.
/// * `form` - The results to save to scylla
/// * `shared` - Shared Thorium objects
pub async fn create<O: OutputSupport>(
    key: &str,
    form: &OutputForm<O>,
    shared: &Shared,
    span: &Span,
) -> Result<(), ApiError> {
    // get the type of tag we are creating
    let kind = O::output_kind();
    // start our create results span
    let span = span!(parent: span, Level::INFO, "Save Results To Scylla");
    // wrap our tool in a vec
    let tools = vec![form.tool.clone()];
    // get our previous results
    let mut past = get(kind, &form.groups, key, &tools, true, shared).await?;
    // downselect to just this tools results
    let past = past.results.remove(&form.tool).unwrap_or_default();
    // get the current year and number of hours so far this year
    let now = Utc::now();
    let year = now.year();
    // get the chunk size for results
    let chunk_size = shared.config.thorium.results.partition_size;
    // determine what bucket to add data too
    let bucket = helpers::partition(now, year, chunk_size);
    // get the current timestamp
    let now = Utc::now();
    // save the result object
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.results.insert,
            (
                &form.id,
                now,
                &form.tool,
                &form.tool_version,
                &form.cmd,
                &form.result,
                &form.files,
                form.display_type,
            ),
        )
        .await?;
    // save the stream rows for this result into scylla
    for group in &form.groups {
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.results.insert_stream,
                (
                    kind,
                    &group,
                    year,
                    bucket,
                    key,
                    &form.tool,
                    &form.tool_version,
                    form.display_type,
                    now,
                    &form.cmd,
                    &form.id,
                ),
            )
            .await?;
    }
    // if we have more than our max results stored then delete any past that
    if past.len() >= shared.config.thorium.retention.results {
        // prune any results in groups with more than 3 values
        prune(kind, &form.groups, key, &past, shared, &span).await?;
    }
    // create an event since we've modified results
    let event = ResultSearchEvent::modified::<O>(key.to_string(), form.groups.clone());
    if let Err(err) = super::search::events::create(event, shared).await {
        return internal_err!(format!(
            "Failed to create result search event! {}",
            err.msg
                .unwrap_or_else(|| "An unknown error occurred".to_string())
        ));
    }
    Ok(())
}

/// Authorize a user has access to a specific result_id
///
/// # Arguments
///
/// * `groups` - The groups a user is in
/// * `sha256` - The sha256 this result is for
/// * `tool` - The tool this result is from
/// * `result_id` - The result ID to authorize
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::results::authorize", skip(kind, shared), err(Debug))]
pub async fn authorize(
    kind: OutputKind,
    groups: &Vec<String>,
    key: &str,
    tool: &str,
    result_id: &Uuid,
    shared: &Shared,
) -> Result<(), ApiError> {
    // if we have more than 100 groups then break this into chunks of 100
    if groups.len() > 100 {
        // we have more than 100 groups so break this into chunks of 100
        for chunk in groups.chunks(100) {
            // cast our chunk array into a vec
            let chunk_vec = chunk.to_vec();
            // check if any of our groups have access to this sample
            let query = shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.results.get_id,
                    (key, &kind, chunk_vec, tool),
                )
                .await?;
            // cast this query to a rows query
            let query_rows = query.into_rows_result()?;
            // crawl over our rows and return early if we find our result ID
            for row in query_rows.rows::<(Uuid,)>()? {
                // try to cast our id into a uuid
                if let Some(cast) = log_scylla_err!(row) {
                    // check if this matches our id
                    if cast.0 == *result_id {
                        // the id matches so return early
                        return Ok(());
                    }
                }
            }
        }
    } else {
        // we have less than 100 groups so just query them all at once
        // check if any of our groups have access to this sample
        let query = shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.results.get_id,
                (key, kind, groups, tool),
            )
            .await?;
        // cast this query to a rows query
        let query_rows = query.into_rows_result()?;
        // crawl over our rows and return early if we find our result ID
        for row in query_rows.rows::<(Uuid,)>()? {
            // try to cast our id into a uuid
            if let Some(cast) = log_scylla_err!(row) {
                // check if this matches our id
                if cast.0 == *result_id {
                    // the id matches so return early
                    return Ok(());
                }
            }
        }
    }
    // no matching result IDs were found so bounce this request
    unauthorized!()
}

/// Gets result IDs from scylla
///
/// If no tools are passed then all tool results will be returned.
///
/// # Arguments
///
/// * `kind` - The kind of results we are getting IDs for
/// * `groups` - The groups to get result IDs for
/// * `key` - The key to get result IDs for
/// * `tools` - The tools to optionally restrict results retrieved to
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::results::get_ids", skip_all, err(Debug))]
async fn get_ids(
    kind: OutputKind,
    groups: &Vec<String>,
    key: &str,
    tools: &Vec<String>,
    hidden: bool,
    shared: &Shared,
) -> Result<Vec<OutputId>, ApiError> {
    // build a list of queries
    let mut queries = Vec::with_capacity(groups.len() * tools.len() / 50);
    // get the number of tools or default to 10
    let tools_len = if tools.len() > 0 { tools.len() } else { 1 };
    event!(Level::INFO, groups = groups.len(), tools = tools.len());
    // check if can do this all in one request or not
    if groups.len() * tools_len < 100 {
        // get our result IDs from scylla
        let query = if tools.is_empty() {
            // get our get result IDs query
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.results.get_with_key,
                    (kind, groups, key),
                )
                .await?
        } else {
            // get our get result IDs restricted by tools query
            shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.results.get_with_key_and_tool,
                    (kind, groups, key, tools),
                )
                .await?
        };
        // add our query
        queries.push(query);
    } else if tools.len() > 0 {
        // our combined cartesian product is too high so we need to break up this query
        // crawl our groups 50 at a time and our tools 50 at a time
        for (groups_chunk, tools_chunk) in groups[..]
            .chunks(10)
            .cartesian_product(tools[..].chunks(10))
        {
            // turn our chunks into vecs
            let groups_vec = groups_chunk.to_vec();
            let tools_vec = tools_chunk.to_vec();
            // send get our get result IDs restricted by tools query
            let query = shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.results.get_with_key_and_tool,
                    (kind, groups_vec, key, tools_vec),
                )
                .await?;
            // add our query
            queries.push(query);
        }
    } else {
        // we have no tools so just crawl our groups
        for chunk in groups.chunks(100) {
            // turn our chunks into vecs
            let groups_vec = chunk.to_vec();
            // send our query
            let query = shared
                .scylla
                .session
                .execute_unpaged(
                    &shared.scylla.prep.results.get_with_key,
                    (kind, groups_vec, key),
                )
                .await?;
            // add our query
            queries.push(query);
        }
    }
    // build a vec to store our queries in
    let mut deduped = Vec::with_capacity(tools_len * 3);
    // build a map of timestamps and uuids
    let mut map: HashMap<Uuid, OutputId> = HashMap::with_capacity(10);
    // crawl our queries
    for query in queries {
        // enable rows in this query
        let query_rows = query.into_rows_result()?;
        // turn our query results into a typed iter
        let typed_iter = query_rows.rows::<OutputIdRow>()?;
        // duplicate our rows based on id and build the correct order to return them in
        let mut order: BTreeMap<DateTime<Utc>, Uuid> = BTreeMap::default();
        // crawl over our output id rows
        for row in typed_iter {
            // error on any invalid casts
            let row = row?;
            // skip any hidden rows unless we should return them
            if !hidden && row.display_type == OutputDisplayType::Hidden {
                // we are not showing hidden results so skip this result
                continue;
            }
            // if this id has already been seen then just add a new group
            if let Some(item) = map.get_mut(&row.id) {
                item.groups.push(row.group);
            } else {
                // build and insert our base deduplicated id
                let output_id = OutputId {
                    id: row.id,
                    tool: row.tool,
                    cmd: row.cmd,
                    groups: vec![row.group],
                    uploaded: row.uploaded,
                };
                map.insert(row.id, output_id);
                // add this row to our sorted btree map
                order.insert(row.uploaded, row.id);
            }
        }
        // build the deduplicated iter of IDs
        let ids_iter = order.iter().rev().filter_map(|(_, id)| map.remove(id));
        // extend our deduped vec
        deduped.extend(ids_iter);
        // clear our map
        map.clear();
    }
    Ok(deduped)
}

/// Gets results from scylla
///
/// If no tools are passed then all tool results will be returned.
///
/// # Arguments
///
/// * `kind` - The kind of results to get
/// * `groups` - The groups to get results for
/// * `key` - The key to get results with
/// * `tools` - The tools to optionally restrict results retrieved to
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::results::get", skip_all, err(Debug))]
pub async fn get(
    kind: OutputKind,
    groups: &Vec<String>,
    key: &str,
    tools: &Vec<String>,
    hidden: bool,
    shared: &Shared,
) -> Result<OutputMap, ApiError> {
    // get the IDs we want to pull for our results
    let ids = get_ids(kind, groups, key, tools, hidden, shared).await?;
    // if we found no IDs then short circuit with empty results
    if ids.is_empty() {
        return Ok(OutputMap::default());
    }
    // build a list of just IDs
    let id_list = ids.iter().map(|row| &row.id).collect::<Vec<&Uuid>>();
    // instance an empty result map to insert any retrieved rows into
    let mut temp = HashMap::with_capacity(id_list.len());
    // if we have more than 100 IDs then chunk it into bathes of 100  otherwise just get our info
    if id_list.len() > 100 {
        // break our IDs into chunks of 100
        for chunk in id_list.chunks(100) {
            // turn our id chunk into a vec
            let chunk_vec = chunk.to_vec();
            // get this chunks data
            let query = shared
                .scylla
                .session
                .execute_unpaged(&shared.scylla.prep.results.get, (chunk_vec,))
                .await?;
            // cast this query to a rows query
            let query_rows = query.into_rows_result()?;
            // cast our rows to an output and insert into our map
            query_rows
                .rows::<OutputRow>()?
                .filter_map(|res| log_scylla_err!(res))
                .for_each(|row| {
                    temp.insert(row.id, row);
                });
        }
    } else {
        // get this chunks data
        let query = shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.results.get, (&id_list,))
            .await?;
        // cast this query to a rows query
        let query_rows = query.into_rows_result()?;
        // cast our rows to an output and insert into our map
        query_rows
            .rows::<OutputRow>()?
            .filter_map(|res| log_scylla_err!(res))
            .for_each(|row| {
                temp.insert(row.id, row);
            });
    }
    // build the ordered output map
    let mut outputs = OutputMap::default();
    // crawl our results and add them into our output map
    for id in ids {
        // skip any results that aren't in our result map
        if let Some(output) = temp.remove(&id.id) {
            // add this result to our output map
            outputs.add(output, id.groups);
        } else {
            // we are missing this result so log the error
            event!(
                Level::ERROR,
                result_id = id.id.to_string(),
                msg = "Missing output"
            );
        }
    }
    Ok(outputs)
}

/// Help the result pruner prune a specific result if its no longer reachable
///
/// # Arguments
///
/// * `kind` - The kind of results we are pruning
/// * `key` - The key to prune results from
/// * `result_id` - The id of the result to possibly prune
/// * `files` - The files in s3 to prune if necessary
/// * `shared` - Shared Thorium objects
async fn prune_helper(
    kind: OutputKind,
    key: &str,
    result_id: &Uuid,
    files: &[String],
    shared: &Shared,
) -> Result<(), ApiError> {
    // query scylla to see if this result is reachable
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.results.count, (kind, key, result_id))
        .await?;
    // cast this query to a rows query
    let query_rows = query.into_rows_result()?;
    // delete any backing objects if its  no longer reachable
    if query_rows.rows_num() == 0 {
        // delete this result from scylla
        shared
            .scylla
            .session
            .execute_unpaged(&shared.scylla.prep.results.delete, (result_id,))
            .await?;
        // try to delete all result files from s3
        // if any fail this will orphan things but orphaned result files are better
        // then dangling ones
        for name in files {
            // build our result ID path
            let s3_path = format!("{}/{}", &result_id, name);
            shared.s3.results.delete(&s3_path).await?;
        }
    }
    Ok(())
}

/// Prune any older results and their files
///
/// # Arguments
///
/// * `kind` - The kind of results we are pruning
/// * `groups` - The groups to prune results from
/// * `key` - The key to prune results from
/// * `results` - The results we are deciding whether to prune or not
/// * `shared` - Shared Thorium objects
/// * `span` - The span to log traces under
pub async fn prune(
    kind: OutputKind,
    groups: &[String],
    key: &str,
    results: &[Output],
    shared: &Shared,
    span: &Span,
) -> Result<(), ApiError> {
    // create our span
    span!(parent: span, Level::INFO, "Pruning Results");
    // track how many results for each group we have
    let mut stock: HashMap<&str, usize> = HashMap::default();
    // track the results we prune
    let mut pruned = Vec::default();
    // increment the groups for the results we just added
    for group in groups.iter() {
        stock.insert(group, 1);
    }
    // crawl these results and prune any that are not longer needed
    for result in results.iter() {
        // get the year and hour this result was uploaded
        let year = result.uploaded.year();
        // get the partition size for results
        let chunk_size = shared.config.thorium.results.partition_size;
        // get the partition for this result
        let bucket = helpers::partition(result.uploaded, year, chunk_size);
        // track if we pruned this result for a group
        let mut prune_flag = false;
        for group in result.groups.iter() {
            // get this group if it wasn' already added
            let entry = stock.entry(group).or_insert(0);
            // if we are already at our retention limit then prune this result
            if *entry >= shared.config.thorium.retention.results {
                // get the year and hour for this result
                shared
                    .scylla
                    .session
                    .execute_unpaged(
                        &shared.scylla.prep.results.delete_stream,
                        (kind, group, year, bucket, result.uploaded, result.id),
                    )
                    .await?;
                // track that we pruned a result
                prune_flag = true;
            } else {
                // increment the number of results for this group
                *entry += 1;
            }
        }
        // if a result was pruned then add this to our list of pruned IDs
        if prune_flag {
            pruned.push((&result.id, &result.files));
        }
    }
    // crawl over the result files that might need to be pruned
    for (result_id, files) in pruned.iter() {
        // prune this result if its no longer needed
        prune_helper(kind, key, result_id, files, shared).await?;
    }
    Ok(())
}
