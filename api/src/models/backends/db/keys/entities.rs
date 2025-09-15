//! Databse keys for entities

use crate::models::CensusKeys;
use crate::utils::Shared;

pub mod vendors;

pub use vendors::VendorKeys;

/// Build the keys for this items cursor/census caches
///
/// # Arguments
///
/// * `keys` - The vec to add our built keys too
/// * `groups` - The groups these samples submissions are in
/// * `year` - The year this census info is for
/// * `bucket` - This objects bucket
/// * `shared` - Shared Thorium objects
pub fn census_keys(
    keys: &mut Vec<CensusKeys>,
    groups: &[String],
    year: i32,
    bucket: i32,
    grouping: i32,
    shared: &Shared,
) {
    // for each group build our key
    for group in groups {
        // build the count key for this row
        let count = format!(
            "{namespace}:census:entities:counts:{group}:{year}:{grouping}",
            namespace = shared.config.thorium.namespace,
            group = group,
            year = year,
            grouping = grouping,
        );
        // build the stream key for this row
        let stream = format!(
            "{namespace}:census:entities:stream:{group}:{year}",
            namespace = shared.config.thorium.namespace,
            group = group,
            year = year,
        );
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
