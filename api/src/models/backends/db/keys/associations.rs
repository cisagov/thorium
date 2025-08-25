//! The keys for associations in Redis

use crate::models::CensusKeys;
use crate::utils::Shared;

/// Build the count key for this partition
///
/// # Arguments
///
/// * `group` - The group to look for census info for
/// * `year` - The year this sample is in
/// * `grouping` - The grouping for this bucket
/// * `source` - The source entity/object to build a census key for
/// * `shared` - Shared Thorium objects
pub fn census_count<T: std::fmt::Display>(
    group: &T,
    year: i32,
    grouping: i32,
    source: &str,
    shared: &Shared,
) -> String {
    // build the key for this row
    format!(
        "{namespace}:census:associations:counts:{group}:{source}:{year}:{grouping}",
        namespace = shared.config.thorium.namespace,
        group = group,
        year = year,
        grouping = grouping,
        source = source,
    )
}

/// Build the sorted set key for this census operation
///
/// # Arguments
///
/// * `group` - The group to look for census info for
/// * `year` - The year this sample is in
/// * `source` - The source/target entity/object to build a census key for
/// * `shared` - Shared Thorium objects
pub fn census_stream<T: std::fmt::Display>(
    group: &T,
    year: i32,
    source: &str,
    shared: &Shared,
) -> String {
    format!(
        "{namespace}:census:associations:stream:{group}:{source}:{year}",
        namespace = shared.config.thorium.namespace,
        group = group,
        year = year,
        source = source,
    )
}

/// Build the keys for the assocations_from cursor/census caches
///
/// # Arguments
///
/// * `keys` - The list of keys to add our keys too
/// * `groups` - The groups these samples submissions are in
/// * `year` - The year this census info is for
/// * `bucket` - This objects bucket
/// * `source` - The source entity/object to build a census key for
/// * `target` - The target entity/object ot build a census key for
/// * `shared` - Shared Thorium objects
pub fn census_keys(
    keys: &mut Vec<CensusKeys>,
    groups: &Vec<String>,
    year: i32,
    bucket: i32,
    source: &str,
    target: &str,
    shared: &Shared,
) {
    // calculate the grouping for this form
    let grouping = bucket / 10_000;
    // for each group build our key
    for group in groups {
        // build the count key for this rows source
        let count = census_count(group, year, grouping, source, shared);
        // build the stream key for this rows source
        let stream = census_stream(group, year, source, shared);
        // build our census key object
        let key = CensusKeys {
            count,
            stream,
            bucket,
        };
        // add our key
        keys.push(key);
        // build the count key for this rows target
        let count = census_count(group, year, grouping, target, shared);
        // build the stream key for this rows target
        let stream = census_stream(group, year, target, shared);
        // build our census key object
        let key = CensusKeys {
            count,
            stream,
            bucket,
        };
        // add our key
        keys.push(key);
    }
}

/// Build the keys for the assocations_from cursor/census caches
///
/// # Arguments
///
/// * `keys` - The list of keys to add our keys too
/// * `groups` - The groups these samples submissions are in
/// * `year` - The year this census info is for
/// * `bucket` - This objects bucket
/// * `source` - The source entity/object to build a census key for
/// * `target` - The target entity/object ot build a census key for
/// * `shared` - Shared Thorium objects
pub fn census_keys_ref(
    keys: &mut Vec<CensusKeys>,
    groups: &Vec<&String>,
    year: i32,
    bucket: i32,
    source: &str,
    target: &str,
    shared: &Shared,
) {
    // calculate the grouping for this form
    let grouping = bucket / 10_000;
    // for each group build our key
    for group in groups {
        // build the count key for this row
        let count = census_count(*group, year, grouping, source, shared);
        // build the stream key for this row
        let stream = census_stream(*group, year, source, shared);
        // build our census key object
        let key = CensusKeys {
            count,
            stream,
            bucket,
        };
        // add our key
        keys.push(key);
        // build the count key for this rows target
        let count = census_count(group, year, grouping, target, shared);
        // build the stream key for this rows target
        let stream = census_stream(group, year, target, shared);
        // build our census key object
        let key = CensusKeys {
            count,
            stream,
            bucket,
        };
        // add our key
        keys.push(key);
    }
}

///// Build the count key for this partition
/////
///// # Arguments
/////
///// * `group` - The group to look for census info for
///// * `year` - The year this sample is in
///// * `grouping` - The grouping for this bucket
///// * `target` - The target entity/object to build a census key for
///// * `shared` - Shared Thorium objects
//pub fn to_census_count<T: std::fmt::Display>(
//    group: &T,
//    year: i32,
//    grouping: i32,
//    target: &str,
//    shared: &Shared,
//) -> String {
//    // build the key for this row
//    format!(
//        "{namespace}:census:associations_to:counts:{group}:{target}:{year}:{grouping}",
//        namespace = shared.config.thorium.namespace,
//        group = group,
//        year = year,
//        grouping = grouping,
//        target = target,
//    )
//}
//
///// Build the sorted set key for this census operation
/////
///// # Arguments
/////
///// * `group` - The group to look for census info for
///// * `year` - The year this sample is in
///// * `target` - The target entity/object to build a census key for
///// * `shared` - Shared Thorium objects
//pub fn to_census_stream<T: std::fmt::Display>(
//    group: &T,
//    year: i32,
//    target: &str,
//    shared: &Shared,
//) -> String {
//    format!(
//        "{namespace}:census:associations_to:stream:{group}:{target}:{year}",
//        namespace = shared.config.thorium.namespace,
//        group = group,
//        year = year,
//        target = target,
//    )
//}
//
///// Build the keys for the assocations_to cursor/census caches
/////
///// # Arguments
/////
///// * `keys` - The list of keys to add our keys too
///// * `groups` - The groups these samples submissions are in
///// * `year` - The year this census info is for
///// * `bucket` - This objects bucket
///// * `target` - The target entity/object to build a census key for
///// * `shared` - Shared Thorium objects
//pub fn to_census_keys(
//    keys: &mut Vec<CensusKeys>,
//    groups: &Vec<String>,
//    year: i32,
//    bucket: i32,
//    target: &str,
//    shared: &Shared,
//) {
//    // calculate the grouping for this form
//    let grouping = bucket / 10_000;
//    // for each group build our key
//    for group in groups {
//        // build the count key for this row
//        let count = to_census_count(group, year, grouping, target, shared);
//        // build the stream key for this row
//        let stream = to_census_stream(group, year, target, shared);
//        // build our census key object
//        let key = CensusKeys {
//            count,
//            stream,
//            bucket,
//        };
//        // add our key
//        keys.push(key);
//    }
//}
