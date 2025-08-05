//! Structures for tagging objects in Thorium
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::str::FromStr;

use super::InvalidEnum;
use super::backends::TagSupport;

/// The different types of tags
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Hash, Eq)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum TagType {
    /// This operation is working on files/samples tags
    Files,
    /// This operation is working on repo tags
    Repos,
}

impl TagType {
    /// Cast our tag type to a str
    pub fn as_str(&self) -> &str {
        match self {
            TagType::Files => "Files",
            TagType::Repos => "Repos",
        }
    }
}

impl FromStr for TagType {
    type Err = InvalidEnum;

    /// Convert this str to an [`EventType`]
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "Files" => Ok(TagType::Files),
            "Repos" => Ok(TagType::Repos),
            _ => Err(InvalidEnum(format!("Unknown TagType: {raw}"))),
        }
    }
}

impl std::fmt::Display for TagType {
    /// Allow tag kinds to be displayed
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            &Self::Files => write!(f, "Files"),
            &Self::Repos => write!(f, "Repos"),
        }
    }
}

/// A request to add new tags to a sample or repo
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TagRequest<T: TagSupport> {
    /// The groups these tags should be visible to
    #[serde(default)]
    pub groups: Vec<String>,
    /// The tags to add
    pub tags: HashMap<String, HashSet<String>>,
    /// The trigger depth for this request
    #[serde(default)]
    pub trigger_depth: u8,
    /// The type we are implementing a tag request for
    #[serde(default)]
    #[cfg_attr(feature = "api", schema(ignore, value_type = String))]
    phantom: PhantomData<T>,
}

// manually implement Clone to avoid requiring TagSupport to implement Clone,
// as we just have PhantomData here, not an actual instance of TagSupport
impl<T: TagSupport> Clone for TagRequest<T> {
    fn clone(&self) -> Self {
        Self {
            groups: self.groups.clone(),
            tags: self.tags.clone(),
            trigger_depth: self.trigger_depth,
            phantom: PhantomData,
        }
    }
}

impl<T: TagSupport> TagRequest<T> {
    /// Adds a single group to this tag request
    ///
    /// # Arguments
    ///
    /// * `group` - The group this tag should be visible to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagRequest, Sample};
    ///
    /// TagRequest::<Sample>::default().group("corn");
    /// ```
    pub fn group<G: Into<String>>(mut self, group: G) -> Self {
        // convert this group to a string and set it
        self.groups.push(group.into());
        self
    }

    /// Adds multiple groups to this tag request
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups this tag should be visible to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagRequest, Sample};
    ///
    /// TagRequest::<Sample>::default().groups(vec!("corn", "tacos"));
    /// ```
    pub fn groups<G: Into<String>>(mut self, groups: Vec<G>) -> Self {
        // convert these groups  to strings and add them
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// Adds a new tag to this tag request
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The value of this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagRequest, Sample};
    ///
    /// // build our tag req
    /// let mut req = TagRequest::<Sample>::default();
    /// // add new tags
    /// req.add_ref("plant", "corn");
    /// ```
    pub fn add_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        // get an entry for this key or insert a default vec
        let entry = self.tags.entry(key.into()).or_default();
        // add our value to this keys list
        entry.insert(value.into());
    }

    /// Adds multiple new values for a tag to this tag request
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The values of this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagRequest, Sample};
    ///
    /// // build our tag req
    /// let mut req = TagRequest::<Sample>::default();
    /// // add new tags
    /// req.add_values_ref("plant", vec!("corn", "oranges"));
    /// ```
    pub fn add_values_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, values: Vec<V>) {
        // get an entry for this key or insert a default vec
        let entry = self.tags.entry(key.into()).or_default();
        // add our value to this keys list
        entry.extend(values.into_iter().map(|value| value.into()));
    }

    /// Adds a new tag to this tag request in a builder pattern
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The value of this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagRequest, Sample};
    ///
    /// TagRequest::<Sample>::default().add("plant", "corn");
    /// ```
    pub fn add<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.add_ref(key, value);
        self
    }

    /// Adds multiple new values for a tag to this tag request in a builder pattern
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The values of this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagRequest, Sample};
    ///
    /// TagRequest::<Sample>::default().add_values("plant", vec!("corn", "soybeans", "apples"));
    /// ```
    pub fn add_values<K: Into<String>, V: Into<String>>(mut self, key: K, values: Vec<V>) -> Self {
        self.add_values_ref(key, values);
        self
    }

    /// Set the trigger depth for this tag request
    ///
    /// # Arguments
    ///
    /// * `trigger_depth` - The trigger depth to set
    pub fn trigger_depth(mut self, trigger_depth: u8) -> Self {
        // update our trigger depth
        self.trigger_depth = trigger_depth;
        self
    }
}

impl<T: TagSupport> Default for TagRequest<T> {
    // build a default tag request
    fn default() -> Self {
        TagRequest {
            groups: Vec::default(),
            tags: HashMap::with_capacity(1),
            trigger_depth: 0,
            phantom: PhantomData::default(),
        }
    }
}

/// A request to delete tags from a sample or repo
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TagDeleteRequest<T: TagSupport> {
    /// The groups these tags should be deleted from
    #[serde(default)]
    pub groups: Vec<String>,
    /// The tags to delete
    pub tags: HashMap<String, Vec<String>>,
    /// The type we are implementing a tag delete request for
    #[serde(default)]
    #[cfg_attr(feature = "api", schema(ignore, value_type = String))]
    phantom: PhantomData<T>,
}

impl<T: TagSupport> TagDeleteRequest<T> {
    /// Adds a single group to this tag delete request
    ///
    /// # Arguments
    ///
    /// * `group` - The group this tag should be deleted from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDeleteRequest, Sample};
    ///
    /// TagDeleteRequest::<Sample>::default().group("corn");
    /// ```
    pub fn group<G: Into<String>>(mut self, group: G) -> Self {
        // convert this group to a string and set it
        self.groups.push(group.into());
        self
    }

    /// Adds multiple groups to this tag delete request
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups this tag should be deleted from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDeleteRequest, Sample};
    ///
    /// TagDeleteRequest::<Sample>::default().groups(vec!("corn", "tacos"));
    /// ```
    pub fn groups<I, G>(mut self, groups: I) -> Self
    where
        G: Into<String>,
        I: IntoIterator<Item = G>,
    {
        // convert these groups to strings and add them
        self.groups
            .extend(groups.into_iter().map(|group| group.into()));
        self
    }

    /// Adds a new tag to this tag request
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The value of this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDeleteRequest, Sample};
    ///
    /// // build our tag delete request
    /// let mut req = TagDeleteRequest::<Sample>::default();
    /// // add tags to be deleted
    /// req.add_ref("plant", "corn");
    /// ```
    pub fn add_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        // get an entry for this key or insert a default vec
        let entry = self.tags.entry(key.into()).or_default();
        // add our value to this keys list
        entry.push(value.into());
    }

    /// Adds multiple new values for a tag to this tag request
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies the tag to be deleted
    /// * `value` - The values of the tag to delete
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDeleteRequest, Sample};
    ///
    /// // build our tag delete request
    /// let mut req = TagDeleteRequest::<Sample>::default();
    /// // add tags to be deleted
    /// req.add_values_ref("plant", vec!("corn", "oranges"));
    /// ```
    pub fn add_values_ref<I, K, V>(&mut self, key: K, values: I)
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = V>,
    {
        // get an entry for this key or insert a default vec
        let entry = self.tags.entry(key.into()).or_default();
        // add our value to this keys list
        entry.extend(values.into_iter().map(|value| value.into()));
    }

    /// Adds a new tag to be deleted to this tag request in a builder pattern
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies the tag to be deleted
    /// * `value` - The value of the tag to delete
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDeleteRequest, Sample};
    ///
    /// TagDeleteRequest::<Sample>::default().add("plant", "corn");
    /// ```
    pub fn add<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.add_ref(key, value);
        self
    }

    /// Adds multiple new values to be deleted from a tag
    /// to this tag request in a builder pattern
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The values of this tag
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDeleteRequest, Sample};
    ///
    /// TagDeleteRequest::<Sample>::default().add_values("plant", vec!("corn", "soybeans", "apples"));
    /// ```
    pub fn add_values<K: Into<String>, V: Into<String>>(mut self, key: K, values: Vec<V>) -> Self {
        self.add_values_ref(key, values);
        self
    }
}

impl<T: TagSupport> Default for TagDeleteRequest<T> {
    // build a default tag request
    fn default() -> Self {
        TagDeleteRequest {
            groups: Vec::default(),
            tags: HashMap::with_capacity(1),
            phantom: PhantomData::default(),
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "scylla-utils", derive(scylla::DeserializeRow))]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
pub struct TagCensusRow {
    pub kind: TagType,
    pub group: String,
    pub year: i32,
    pub bucket: i32,
    pub key: String,
    pub value: String,
    pub count: i64,
}

#[cfg(feature = "scylla-utils")]
impl<T: TagSupport + 'static + Send> super::CensusSupport for TagRequest<T> {
    /// The type returned by our prepared statement
    type Row = TagCensusRow;

    /// Build the prepared statement for getting partition count info
    async fn scan_prepared_statement(
        scylla: &scylla::client::session::Session,
        ns: &str,
    ) -> Result<scylla::statement::prepared::PreparedStatement, scylla::errors::PrepareError> {
        // build tags get partition count prepared statement
        scylla
        .prepare(format!(
            "SELECT type, group, year, bucket, key, value, count(*) \
            FROM {}.tags \
            WHERE token(type, group, year, bucket, key, value) >= ? AND token(type, group, year, bucket, key, value) <= ? \
            GROUP BY type, group, year, bucket, key, value",
            ns,
        ))
        .await
    }

    /// Get the count for this partition
    fn get_count(row: &TagCensusRow) -> i64 {
        row.count
    }

    /// Get the bucket for this partition
    fn get_bucket(row: &TagCensusRow) -> i32 {
        row.bucket
    }

    /// Build the count key for this partition
    fn count_key_from_row(namespace: &str, row: &Self::Row, grouping: i32) -> String {
        // build the key for this row
        format!(
            "{namespace}:census:tags:counts:{kind}:{group}:{key}:{value}:{year}:{grouping}",
            namespace = namespace,
            kind = row.kind,
            group = row.group,
            key = row.key,
            value = row.value,
            year = row.year,
            grouping = grouping,
        )
    }

    /// Build the sorted set key for this census operation
    fn stream_key_from_row(namespace: &str, row: &Self::Row) -> String {
        format!(
            "{namespace}:census:tags:stream:{kind}:{group}:{key}:{value}:{year}",
            namespace = namespace,
            kind = row.kind,
            group = row.group,
            key = row.key,
            value = row.value,
            year = row.year,
        )
    }
}

/// A placeholder struct for implementing census support for case-insensitive
/// tags
#[cfg(feature = "scylla-utils")]
pub struct TagCensusCaseInsensitive<T: TagSupport> {
    phantom: PhantomData<T>,
}

#[derive(Debug)]
#[cfg_attr(feature = "scylla-utils", derive(scylla::DeserializeRow))]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
pub struct TagCensusCaseInsensitiveRow {
    pub kind: TagType,
    pub group: String,
    pub year: i32,
    pub bucket: i32,
    pub key_lower: String,
    pub value_lower: String,
    pub count: i64,
}

#[cfg(feature = "scylla-utils")]
impl<T: TagSupport + 'static + Send> super::CensusSupport for TagCensusCaseInsensitive<T> {
    /// The type returned by our prepared statement
    type Row = TagCensusCaseInsensitiveRow;

    /// Build the prepared statement for getting partition count info
    async fn scan_prepared_statement(
        scylla: &scylla::client::session::Session,
        ns: &str,
    ) -> Result<scylla::statement::prepared::PreparedStatement, scylla::errors::PrepareError> {
        // build tags case insensitive get partition count prepared statement
        scylla
        .prepare(format!(
            "SELECT type, group, year, bucket, key_lower, value_lower, count(*) \
            FROM {ns}.tags_case_insensitive \
            WHERE token(type, group, year, bucket, key_lower, value_lower) >= ? AND token(type, group, year, bucket, key_lower, value_lower) <= ? \
            GROUP BY type, group, year, bucket, key_lower, value_lower"
        ))
        .await
    }

    /// Get the count for this partition
    fn get_count(row: &TagCensusCaseInsensitiveRow) -> i64 {
        row.count
    }

    /// Get the bucket for this partition
    fn get_bucket(row: &TagCensusCaseInsensitiveRow) -> i32 {
        row.bucket
    }

    /// Build the count key for this partition
    fn count_key_from_row(namespace: &str, row: &Self::Row, grouping: i32) -> String {
        // build the key for this row
        format!(
            "{namespace}:census:tags_case_insensitive:counts:{kind}:{group}:{key}:{value}:{year}:{grouping}",
            namespace = namespace,
            kind = row.kind,
            group = row.group,
            key = row.key_lower,
            value = row.value_lower,
            year = row.year,
            grouping = grouping,
        )
    }

    /// Build the sorted set key for this census operation
    fn stream_key_from_row(namespace: &str, row: &Self::Row) -> String {
        format!(
            "{namespace}:census:tags_case_insensitive:stream:{kind}:{group}:{key}:{value}:{year}",
            namespace = namespace,
            kind = row.kind,
            group = row.group,
            key = row.key_lower,
            value = row.value_lower,
            year = row.year,
        )
    }
}
