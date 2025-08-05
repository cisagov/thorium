//! Automatically extracts tags from results and submits them

use crossbeam::channel::Sender;
use itertools::Itertools;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use thorium::models::backends::TagSupport;
use thorium::models::{
    AutoTag, AutoTagLogic, GenericJob, OutputCollection, Repo, Sample, TagRequest,
};
use thorium::{Error, Thorium};
use tracing::instrument;

use crate::{fail, log};

use super::results::RawResults;

/// A bundle of different tag types from this job
#[derive(Debug, Default)]
pub struct TagBundle {
    /// The tags to apply to some samples
    samples: Option<TagRequest<Sample>>,
    /// The tags to apply to some repos
    repos: Option<TagRequest<Repo>>,
}

/// determine if the target output value exists at the root or not
///
/// # Arguments
///
/// * `src` - The map of results to pull tags from
/// * `key` - The key to get a tag value for
/// * `logs` - The logs to send to the API
#[instrument(name = "tags::exists", skip(src, logs), err(Debug))]
fn exists(
    src: &mut Map<String, Value>,
    key: &String,
    logs: &mut Sender<String>,
) -> Result<Option<Vec<String>>, Error> {
    // if this key exists then remove its value
    if let Some(value) = src.remove(key) {
        // if this is string representable then cast it to a string
        match value {
            serde_json::Value::Bool(bool) => Ok(Some(vec![bool.to_string()])),
            serde_json::Value::Number(num) => Ok(Some(vec![num.to_string()])),
            serde_json::Value::String(string) => Ok(Some(vec![string])),
            serde_json::Value::Array(array) => {
                // crawl and build our array of values
                let mut casts = Vec::with_capacity(array.len());
                for val in array {
                    // make sure each value is a sting
                    if let Some(str) = val.as_str() {
                        casts.push(str.to_owned());
                    } else {
                        // build and add error message
                        let err = format!(
                            "Error: {key} - Exists auto Tagging only supports bools, Numbers, or Strings"
                        );
                        fail!(logs, err);
                    }
                }
                Ok(Some(casts))
            }
            // ignore anything not easily string representable
            _ => {
                // build and add error message
                let err = format!(
                    "Error: {key} - Exists auto Tagging only supports bools, Numbers, or Strings"
                );
                fail!(logs, err);
            }
        }
    } else {
        // this tag was no present in the results
        Ok(None)
    }
}

macro_rules! equal {
    ($left:expr, $right:expr) => {
        if $left == *$right {
            Ok(Some(vec![$left.to_string()]))
        } else {
            Ok(None)
        }
    };
}

/// Create a tag if and only if the value matches some other value
///
/// # Arguments
///
/// * `src` - The map of results to pull tags from
/// * `key` - The key to get a tag value for
/// * `right` - The value on the right side of this equality check
/// * `logs` - The logs to send to the API
#[instrument(name = "tags::equal", skip(src, right, logs), err(Debug))]
fn equal(
    src: &mut Map<String, Value>,
    key: &String,
    right: &Value,
    logs: &mut Sender<String>,
) -> Result<Option<Vec<String>>, Error> {
    // if this key exists then remove its value
    if let Some(value) = src.remove(key) {
        // if this is string representable then cast it to a string
        match (value, right) {
            (Value::Bool(bool), Value::Bool(right)) => equal!(bool, right),
            (Value::Number(num), Value::Number(right)) => equal!(num, right),
            (Value::String(string), Value::String(right)) => equal!(string, right),
            // ignore anything not easily string representable
            _ => {
                // build and add error message
                let err = format!(
                    "Error: {key} - Equals auto Tagging only supports bools, Numbers, or Strings"
                );
                fail!(logs, err);
            }
        }
    } else {
        // this tag was no present in the results
        Ok(None)
    }
}

/// Evaluate our results for tags to create
///
/// # Arguments
///
/// * `settings` - The auto tag settings to use
/// * `src` - The results to pull tags from
/// * `key` - The key to create a tag for
/// * `logs` - The logs to send to the API
#[instrument(name = "tags::eval", skip(settings, src, logs), err(Debug))]
pub fn eval(
    settings: &AutoTag,
    src: &mut Map<String, Value>,
    key: &String,
    logs: &mut Sender<String>,
) -> Result<Option<Vec<String>>, Error> {
    match &settings.logic {
        AutoTagLogic::Exists => exists(src, key, logs),
        AutoTagLogic::Equal(right) => equal(src, key, right, logs),
        AutoTagLogic::Not(_) => fail!(logs, "AutoTagging does yet support this"),
        AutoTagLogic::Greater(_) => fail!(logs, "AutoTagging does yet support this"),
        AutoTagLogic::GreaterOrEqual(_) => fail!(logs, "AutoTagging does yet support this"),
        AutoTagLogic::LesserOrEqual(_) => fail!(logs, "AutoTagging does yet support this"),
        AutoTagLogic::Lesser(_) => fail!(logs, "AutoTagging does yet support this"),
        AutoTagLogic::In(_) => fail!(logs, "AutoTagging does yet support this"),
        AutoTagLogic::NotIn(_) => fail!(logs, "AutoTagging does yet support this"),
    }
}

/// The raw tag key/values pairs the agent is building
#[derive(Debug, Default)]
struct RawTags {
    /// The raw tag values for this request
    tags: HashMap<String, HashSet<String>>,
}

impl RawTags {
    /// Adds a new tag to this raw tag request
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The value of this tag
    pub fn add_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        // get an entry for this key or insert a default vec
        let entry = self.tags.entry(key.into()).or_default();
        // add our value to this keys list
        entry.insert(value.into());
    }

    /// Adds multiple new values for a tag to this raw tag request
    ///
    /// # Arguments
    ///
    /// * `key` - The key that identifies this tag
    /// * `value` - The values of this tag
    pub fn add_values_ref<K: Into<String>, V: Into<String>>(&mut self, key: K, values: Vec<V>) {
        // get an entry for this key or insert a default vec
        let entry = self.tags.entry(key.into()).or_default();
        // add our value to this keys list
        entry.extend(values.into_iter().map(std::convert::Into::into));
    }

    /// Cast our raw tags to a specific tag request type
    pub fn to_req<T: TagSupport>(&self, depth: u8) -> TagRequest<T> {
        // get a default tag request with the correct depth
        let mut req = TagRequest::<T>::default().trigger_depth(depth);
        // add our tags
        req.tags = self.tags.clone();
        req
    }
}

/// Tries to extract tags from a result
#[derive(Debug, Default)]
pub struct Extractor {
    /// Whether we have found a valid target to extract tags from or not
    found_valid: bool,
    /// The tags we have extracted
    tags: RawTags,
}

impl Extractor {
    /// Recursively attempt to extract tags from [`Value`]'s
    ///
    /// # Arguments
    ///
    #[instrument(name = "tags::extract_helper", skip_all, err(Debug))]
    fn extract_helper(
        &mut self,
        json_value: Value,
        settings: &OutputCollection,
        logs: &mut Sender<String>,
    ) -> Result<(), Error> {
        // Try to extract these tags based on the values type
        match json_value {
            Value::Array(array) => {
                // crawl over this values and recursively try and extract tags
                for inner_value in array {
                    self.extract_helper(inner_value, settings, logs)?
                }
            }
            Value::Object(mut map) => {
                // set that we found a valid value to attempt extraction
                self.found_valid = true;
                // try to extract tags from this map
                for (key, logic) in &settings.auto_tag {
                    // try to extract this tag
                    if let Some(values) = eval(logic, &mut map, key, logs)? {
                        // if a key rename was set then use that
                        match &logic.key {
                            Some(rename) => self.tags.add_values_ref(rename, values),
                            None => self.tags.add_values_ref(key, values),
                        }
                    }
                }
            }
            Value::Null => log!(logs, "Skipping unsupported null value"),
            Value::Bool(_) => log!(logs, "Skipping unsupported bool value"),
            Value::Number(_) => log!(logs, "Skipping unsupported number value"),
            Value::String(_) => log!(logs, "Skipping unsupported string value"),
        }
        Ok(())
    }

    /// Extract tags from a result and try to submit them
    ///
    /// # Arguments
    ///
    /// * `output` - The output to extract tags from
    /// * `settings` - The auto tag settings to use
    /// * `logs` - The logs to send to the API
    #[instrument(name = "tags::extract", skip_all, err(Debug))]
    fn extract(
        &mut self,
        output: &String,
        settings: &OutputCollection,
        logs: &mut Sender<String>,
    ) -> Result<RawTags, Error> {
        // only try to extract tags if we have any to extract
        if !settings.auto_tag.is_empty() {
            // try to deserialize our results
            let json_value: Value = match serde_json::from_str(output) {
                Ok(value) => value,
                Err(error) => {
                    fail!(
                        logs,
                        format!(
                            "ERROR: Only json dictionary results support auto tags: {:#?}",
                            error
                        )
                    )
                }
            };
            // recursively extract tags from this result
            self.extract_helper(json_value, settings, logs)?;
        }
        // if we never found a valid extraction target then raise an error
        if !self.found_valid {
            fail!(logs, "ERROR: A dictionary must be present for auto tagging")
        }
        Ok(std::mem::take(&mut self.tags))
    }
}

/// Read in the tags file if it exists and overlay it on any result tags
///
/// # Arguments
///
/// * `tags` - The tag request to overlay tags on
/// * `path` - The path to read our user created tags file from
/// * `logs` - The logs to send to the API
#[instrument(name = "tags::overlay", skip(tags, logs), err(Debug))]
fn overlay(mut tags: RawTags, path: &Path, logs: &mut Sender<String>) -> Result<RawTags, Error> {
    // read in out tags file
    if let Ok(tags_str) = std::fs::read_to_string(path) {
        // try to convert our tags_str to a HashMap
        let map: HashMap<String, Value> = match serde_json::from_str(&tags_str) {
            Ok(map) => map,
            Err(err) => return Err(Error::from(err)),
        };
        // crawl through all tags and try to add them to our tag map
        for (key, value) in map {
            // try to extract the values based on type
            match value {
                // The value for this is just a string so just add it
                Value::String(value) => tags.add_ref(key, value),
                // The value for this is just a number so just add it
                Value::Number(value) => tags.add_ref(key, value.to_string()),
                // The value for this is just a bool so just add it
                Value::Bool(value) => tags.add_ref(key, value.to_string()),
                // the value for this is an array so crawl the values in the array
                Value::Array(values) => {
                    for value in values {
                        // try to add each individual value and error on objects or nested vectors
                        match value {
                            // The value for this is just a string so just add it
                            Value::String(value) => tags.add_ref(&key, value),
                            // The value for this is just a number so just add it
                            Value::Number(value) => tags.add_ref(&key, value.to_string()),
                            // The value for this is just a bool so just add it
                            Value::Bool(value) => tags.add_ref(&key, value.to_string()),
                            // error out on unsupported types
                            _ => fail!(logs, "ERROR: Only strings, numbers, or bools are allowed in nested tag arrays"),
                        }
                    }
                }
                // error out on unsupported types
                _ => fail!(
                    logs,
                    "ERROR: Only strings, numbers, bools, or arrays are allowed in the tags file"
                ),
            }
        }
    }
    Ok(tags)
}

/// Gather all tags for a specific job
///
/// # Arguments
///
/// * `job` - The job we are executing
/// * `output`: The serialized output to look for tags in
/// * `settings` - The settings to use for output collection
/// * `path` - The path to pull tags from
/// * `logs` - The logs to send to the API
#[instrument(name = "tags::collect", skip_all, fields(path = path.as_ref().to_string_lossy().into_owned()), err(Debug))]
pub async fn collect<P: AsRef<Path>>(
    job: &GenericJob,
    output: &RawResults,
    settings: &OutputCollection,
    path: P,
    logs: &mut Sender<String>,
) -> Result<TagBundle, Error> {
    // skip extracting tags if we didn't get any results
    let raw = if output.scan && !settings.auto_tag.is_empty() {
        // get our results
        let results = output.results.get_results();
        // build an extractor
        let mut extractor = Extractor::default();
        // extract any tags from any dumped results
        extractor.extract(results, settings, logs)?
    } else {
        RawTags::default()
    };
    // read in any tags from our tags file and overlay them on our tags object
    let raw = overlay(raw, path.as_ref(), logs)?;
    // log any tags we discovered
    for (key, values) in &raw.tags {
        log!(logs, "Found tags {}={}", key, values.iter().join(", "));
    }
    // get our trigger depth
    let depth = job.trigger_depth.unwrap_or(0);
    // start with an empty tag bundle
    let mut bundle = TagBundle::default();
    // add in the tags if we have a corresponding input
    if !job.repos.is_empty() {
        // get the repo tag req and add it
        bundle.repos = Some(raw.to_req(depth));
    }
    // do samples last as its likely to be the most common
    // so we can just take instead of cloning
    if !job.samples.is_empty() {
        // add this to our bundle
        bundle.samples = Some(raw.to_req(depth));
    }
    Ok(bundle)
}

/// Submit any collected tags to Thorium
///
/// # Arguments
///
/// * `thorium` - A client for the Thorium API
/// * `bundle` - The tag bundles to submit
/// * `job` - The job we are executing
/// * `logs` - The logs to send to the API
#[instrument(name = "tags::submit", skip_all, err(Debug))]
pub async fn submit(
    thorium: &Thorium,
    bundle: TagBundle,
    job: &GenericJob,
    logs: &mut Sender<String>,
) -> Result<(), Error> {
    // if any tags were found then create them for our samples
    if let Some(req) = bundle.samples {
        // try to add these tags to our input samples
        for sha256 in &job.samples {
            // Add new tags to these files
            log!(logs, "Adding tags to {}", sha256);
            thorium.files.tag(sha256, &req).await?;
        }
    }
    // if any tags were found then create them for our repos
    if let Some(req) = bundle.repos {
        // try to add these tags to our input repos
        for repo in &job.repos {
            // Add new tags to these files
            log!(logs, "Adding tags to {}", repo.url);
            thorium.repos.tag(&repo.url, &req).await?;
        }
    }
    Ok(())
}
