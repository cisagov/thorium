//! The keys for repo data in redis

use crate::models::CensusKeys;
use crate::utils::Shared;

/// Build the count key for this partition
///
/// # Arguments
///
/// * `group` - The group to look for census info for
/// * `year` - The year this sample is in
/// * `grouping` - This repos bucket grouping
/// * `shared` - Shared Thorium objects
pub fn census_count<T: std::fmt::Display>(
    group: &T,
    year: i32,
    grouping: i32,
    shared: &Shared,
) -> String {
    // build the key for this row
    format!(
        "{namespace}:census:repos:counts:{group}:{year}:{grouping}",
        namespace = shared.config.thorium.namespace,
        group = group,
        year = year,
        grouping = grouping,
    )
}

/// Build the sorted set key for this census operation
///
/// # Arguments
///
/// * `group` - The group to look for census info for
/// * `year` - The year this repo is in
/// * `shared` - Shared Thorium objects
pub fn census_stream<T: std::fmt::Display>(group: &T, year: i32, shared: &Shared) -> String {
    format!(
        "{namespace}:census:repos:stream:{group}:{year}",
        namespace = shared.config.thorium.namespace,
        group = group,
        year = year,
    )
}

/// Build the keys for this items cursor/census caches
///
/// # Arguments
///
/// * `keys` - The vec to add our built keys too
/// * `groups` - The groups these repo submissions are in
/// * `year` - The year this census info is for
/// * `bucket` - This objects bucket
/// * `shared` - Shared Thorium objects
pub fn census_keys(
    keys: &mut Vec<CensusKeys>,
    groups: &Vec<String>,
    year: i32,
    bucket: i32,
    shared: &Shared,
) {
    // calculate our census grouping
    let grouping = bucket / 10_000;
    // for each group build our key
    for group in groups {
        // build the count key for this row
        let count = census_count(group, year, grouping, shared);
        // build the stream key for this row
        let stream = census_stream(group, year, shared);
        // build our census key object
        let key = CensusKeys {
            count,
            stream,
            bucket,
        };
        // add this key
        keys.push(key);
    }
}
