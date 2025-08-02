//! The keys for commitish info in redis

use crate::models::CensusKeys;
use crate::models::CommitishKinds;
use crate::utils::Shared;

/// Build the count key for this partition
///
/// # Arguments
///
/// * `group` - The group to look for census info for
/// * `year` - The year this sample is in
/// * `grouping` - This commitishes bucket grouping
/// * `shared` - Shared Thorium objects
pub fn census_count<T: std::fmt::Display>(
    kind: CommitishKinds,
    group: &T,
    repo: &str,
    year: i32,
    grouping: i32,
    shared: &Shared,
) -> String {
    format!(
        "{namespace}:census:commitish:counts:{kind}:{group}:{repo}:{year}:{grouping}",
        namespace = shared.config.thorium.namespace,
        kind = kind,
        group = group,
        repo = repo,
        year = year,
        grouping = grouping,
    )
}

/// Build the sorted set key for this census operation
///
/// # Arguments
///
/// * `group` - The group to look for census info for
/// * `year` - The year this commitish is in
/// * `shared` - Shared Thorium objects
pub fn census_stream<T: std::fmt::Display>(
    kind: CommitishKinds,
    group: &T,
    repo: &str,
    year: i32,
    shared: &Shared,
) -> String {
    format!(
        "{namespace}:census:commitish:stream:{kind}:{group}:{repo}:{year}",
        namespace = shared.config.thorium.namespace,
        kind = kind,
        group = group,
        repo = repo,
        year = year,
    )
}

/// Build the keys for this items cursor/census caches
///
/// # Arguments
///
/// * `keys` - The vec of keys to add too
/// * `repo` - The url for the repo we are building commitish icensus keys for
/// * `kind` - The kind of commitish to build a key for
/// * `year` - The year this census info is for
/// * `bucket` - This objects bucket
/// * `shared` - Shared Thorium objects
pub fn census_keys(
    keys: &mut Vec<CensusKeys>,
    repo: &str,
    kind: CommitishKinds,
    groups: &Vec<String>,
    year: i32,
    bucket: i32,
    shared: &crate::utils::Shared,
) {
    // calculate our census grouping
    let grouping = bucket / 10_000;
    // for each group build our key
    for group in groups {
        // build the count key for this row
        let count = census_count(kind, group, repo, year, grouping, shared);
        // build the stream key for this row
        let stream = census_stream(kind, group, repo, year, shared);
        // build our census key object
        let key = crate::models::CensusKeys {
            count,
            stream,
            bucket,
        };
        // add our key
        keys.push(key);
    }
}
