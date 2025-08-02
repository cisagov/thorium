//! The keys used for tag data in redis

use crate::models::TagType;
use crate::utils::Shared;

/// Build the count key for this partition
///
/// # Arguments
///
/// * `kind` - The kind of tag we are getting/setting census info for
/// * `group` - The group to look for census info for
/// * `key` - The tag key to use
/// * `value` - The tag value to use
/// * `year` - The year this tag is in
/// * `bucket` - This tags bucket
/// * `shared` - Shared Thorium objects
pub fn census_count<T: std::fmt::Display>(
    kind: TagType,
    group: &T,
    key: &str,
    value: &str,
    year: i32,
    bucket: i32,
    shared: &Shared,
) -> String {
    // calculate our census grouping
    let grouping = bucket / 10_000;
    // build the key for this row
    format!(
        "{namespace}:census:tags:counts:{kind}:{group}:{key}:{value}:{year}:{grouping}",
        namespace = shared.config.thorium.namespace,
        kind = kind,
        group = group,
        key = key,
        value = value,
        year = year,
        grouping = grouping,
    )
}

/// Build the sorted set key for this census operation
///
/// # Arguments
///
/// * `kind` - The kind of tag we are getting/setting census info for
/// * `group` - The group to look for census info for
/// * `key` - The tag key to use
/// * `value` - The tag value to use
/// * `year` - The year this tag is in
/// * `shared` - Shared Thorium objects
pub fn census_stream<T: std::fmt::Display>(
    kind: TagType,
    group: &T,
    key: &str,
    value: &str,
    year: i32,
    shared: &Shared,
) -> String {
    format!(
        "{namespace}:census:tags:stream:{kind}:{group}:{key}:{value}:{year}",
        namespace = shared.config.thorium.namespace,
        group = group,
        kind = kind,
        key = key,
        value = value,
        year = year,
    )
}

/// Build the count key for this partition
///
/// # Arguments
///
/// * `kind` - The kind of tag we are getting/setting census info for
/// * `group` - The group to look for census info for
/// * `key` - The tag key to use
/// * `value` - The tag value to use
/// * `year` - The year this tag is in
/// * `bucket` - This tags bucket
/// * `shared` - Shared Thorium objects
pub fn census_count_case_insensitive<T: std::fmt::Display>(
    kind: TagType,
    group: &T,
    key: &str,
    value: &str,
    year: i32,
    bucket: i32,
    shared: &Shared,
) -> String {
    // calculate our census grouping
    let grouping = bucket / 10_000;
    // lowercase our key and value
    let lower_key = key.to_lowercase();
    let lower_value = value.to_lowercase();
    // build the key for this row
    format!(
        "{namespace}:census:tags_case_insensitive:counts:{kind}:{group}:{lower_key}:{lower_value}:{year}:{grouping}",
        namespace = shared.config.thorium.namespace,
        kind = kind,
        group = group,
        lower_key = lower_key,
        lower_value = lower_value,
        year = year,
        grouping = grouping,
    )
}

/// Build the sorted set key for this census operation
///
/// # Arguments
///
/// * `kind` - The kind of tag we are getting/setting census info for
/// * `group` - The group to look for census info for
/// * `key` - The tag key to use
/// * `value` - The tag value to use
/// * `year` - The year this tag is in
/// * `shared` - Shared Thorium objects
pub fn census_stream_case_insensitive<T: std::fmt::Display>(
    kind: TagType,
    group: &T,
    key: &str,
    value: &str,
    year: i32,
    shared: &Shared,
) -> String {
    // lowercase our key and value
    let lower_key = key.to_lowercase();
    let lower_value = value.to_lowercase();
    format!(
        "{namespace}:census:tags_case_insensitive:stream:{kind}:{group}:{lower_key}:{lower_value}:{year}",
        namespace = shared.config.thorium.namespace,
        group = group,
        kind = kind,
        lower_key = lower_key,
        lower_value = lower_value,
        year = year,
    )
}
