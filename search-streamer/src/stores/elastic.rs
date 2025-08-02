//! Support streaming data into elastic

use elasticsearch::auth::Credentials;
use elasticsearch::cert::CertificateValidation;
use elasticsearch::http::request::JsonBody;
use elasticsearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use elasticsearch::indices::{IndicesCreateParts, IndicesDeleteParts, IndicesExistsParts};
use elasticsearch::{BulkParts, Elasticsearch};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use thorium::models::ElasticIndex;
use thorium::{Conf, Error};
use tracing::{event, instrument, Level};
use url::Url;

use super::SearchStore;

#[derive(Clone)]
pub struct Elastic {
    /// The elastic client to use when streaming data
    elastic: Elasticsearch,
    /// The elastic config set in the Thorium config
    elastic_conf: thorium::conf::Elastic,
}

impl Elastic {
    /// Create a new Elastic streamer
    ///
    /// # Arguments
    ///
    /// * `conf` - A Thorium config
    /// * `index` - The index to send docs too
    pub fn new(conf: &Conf) -> Result<Self, Error> {
        // Until https://github.com/elastic/elasticsearch-rs/pull/189 is merged
        // we can only support a single node connection pool
        // try to cast our node to a url
        let url = Url::parse(&conf.elastic.node)?;
        // build our connection pool
        let pool = SingleNodeConnectionPool::new(url);
        // get our username and password
        let username = conf.elastic.username.clone();
        let password = conf.elastic.password.clone();
        // build our transport object for elastic
        let transport = TransportBuilder::new(pool)
            .auth(Credentials::Basic(username, password))
            .cert_validation(CertificateValidation::None)
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        // build our elastic client
        let elastic = Elasticsearch::new(transport);
        // create our elastic struct
        Ok(Elastic {
            elastic,
            elastic_conf: conf.elastic.clone(),
        })
    }

    /// Return the index create body for the given elastic index in a JSON Value
    ///
    /// # Arguments
    ///
    /// * `index` - The elastic index to get mappings for
    ///
    /// # Panics
    ///
    /// Panics if the body is not a JSON object, as values are inserted into
    /// the body object after it's initially created
    fn index_create_body(&self, index: &ElasticIndex) -> Value {
        // first set the mappings based on the elastic index
        let mut body = match index {
            ElasticIndex::SampleResults => serde_json::json!({
                "mappings": {
                    "properties": {
                        "group": { "type": "keyword" },
                        "sha256": { "type": "keyword" },
                        "streamed": { "type": "date" },
                        "results": { "type": "text" },
                        "files": { "type": "text" },
                        "children": { "type": "text" }
                    }
                }
            }),
            ElasticIndex::SampleTags => serde_json::json!({
                "mappings": {
                    "properties": {
                        "group": { "type": "keyword" },
                        "sha256": { "type": "keyword" },
                        "streamed": { "type": "date" },
                        "tags": { "type": "text" }
                    }
                }
            }),
            ElasticIndex::RepoResults => serde_json::json!({
                "mappings": {
                    "properties": {
                        "group": { "type": "keyword" },
                        "url": { "type": "keyword" },
                        "streamed": { "type": "date" },
                        "results": { "type": "text" },
                        "files": { "type": "text" },
                        "children": { "type": "text" }
                    }
                }
            }),
            ElasticIndex::RepoTags => serde_json::json!({
                "mappings": {
                    "properties": {
                        "group": { "type": "keyword" },
                        "url": { "type": "keyword" },
                        "streamed": { "type": "date" },
                        "tags": { "type": "text" }
                    }
                }
            }),
        };
        // insert the rest of the body
        let body_mut = body
            .as_object_mut()
            .expect("Elastic index create body is not a valid JSON object!");
        body_mut.insert(
            "settings".to_string(),
            json!({
                "index": {
                    "highlight": {
                        "max_analyzed_offset": self.elastic_conf.max_analyzed_offset
                    }
                }
            }),
        );
        body
    }
}

#[async_trait::async_trait]
impl SearchStore for Elastic {
    /// The name of this search store
    const STORE_NAME: &'static str = "Elastic";

    /// The index to use in the search store
    type Index = ElasticIndex;

    /// Create a new search store client
    ///
    /// # Arguments
    ///
    /// * `conf` - A Thorium config
    /// * `index` - The index to send docs too
    fn new(conf: &Conf) -> Result<Self, Error> {
        Elastic::new(conf)
    }

    /// Initiate the search store in case it hasn't been already
    ///
    /// # Arguments
    ///
    /// * `indexes` - The indexes to initiate
    /// * `reindex` - Whether we should force a reindex, whether or not
    ///               indexes already exist
    ///
    /// # Returns
    ///
    /// Returns true if the store did not already exist and was initiated
    /// in this function. If [`reindex`] is true, this will always return true.
    ///
    /// # Caveat
    ///
    /// Because a search store may reference multiple indexes, `init` will return
    /// if *any* of the indexes it's responsible for are unintialized, and all of
    /// the indexes with then be initialized by the `search-streamer`. No data will
    /// be lost if one of the indexes already existed, but data may be overridden
    /// to the current state of data in Scylla.
    #[instrument(name = "SearchStore<Elastic>::init", skip_all, err(Debug))]
    async fn init(&self, indexes: &[ElasticIndex], reindex: bool) -> Result<bool, Error> {
        // track whether we initiated any indexes
        let mut init = false;
        // get the list of indexes we need to initiate
        for index in indexes {
            let index_full_name = index.full_name(&self.elastic_conf);
            // check if the index already exists
            let exists_response = self
                .elastic
                .indices()
                .exists(IndicesExistsParts::Index(&[index_full_name]))
                .send()
                .await
                .map_err(|err| {
                    Error::new(format!(
                        "Failed to check if index '{index_full_name}' exists: {err}",
                    ))
                })?;
            // if the index exists and we're forcing a reindex, we need to delete and recreate the index
            let create = match (exists_response.status_code().is_success(), reindex) {
                // index exists, but we want to reindex so delete it first
                (true, true) => {
                    event!(
                        Level::INFO,
                        msg = "Index already exists! Recreating to reindex...",
                        index = index_full_name
                    );
                    let response = self
                        .elastic
                        .indices()
                        .delete(IndicesDeleteParts::Index(&[index_full_name]))
                        .send()
                        .await?;
                    if !response.status_code().is_success() {
                        // return an error if the index did not delete successfully
                        let response_body = response.json::<serde_json::Value>().await?;
                        return Err(Error::new(format!(
                            "Failed to delete index '{index_full_name}': {response_body}",
                        )));
                    }
                    true
                }
                // index exists and we're not reindexing so no creation necessary
                (true, false) => false,
                // index does not exist, so we need to index whether or not we're reindexing
                (false, _) => {
                    event!(
                        Level::INFO,
                        msg = "Index does not exist! Creating...",
                        index = index_full_name
                    );
                    true
                }
            };
            if create {
                // generate the body based on the type of index we're creating
                let body = self.index_create_body(index);
                // create the index in elastic
                let response = self
                    .elastic
                    .indices()
                    .create(IndicesCreateParts::Index(index_full_name))
                    .body(body)
                    .send()
                    .await?;
                if response.status_code().is_success() {
                    event!(
                        Level::INFO,
                        msg = "Index created successfully",
                        index = index_full_name
                    );
                    init = true;
                } else {
                    // return an error if the index did not create successfully
                    let response_body = response.json::<serde_json::Value>().await?;
                    return Err(Error::new(format!(
                        "Failed to create index '{index_full_name}': {response_body}",
                    )));
                }
            }
        }
        Ok(init)
    }

    /// Create documents in elastic to be indexed
    ///
    /// All of the values must be `create` requests or else the search-streamer
    /// will be confused to get anything other than `create` responses back
    ///
    /// # Arguments
    ///
    /// * `index` - The index to send the values to
    /// * `values` - The values to send
    #[instrument(name = "SearchStore<Elastic>::create", skip_all, fields(index = index.to_string(), values = values.len()), err(Debug))]
    async fn create(&self, index: ElasticIndex, values: Vec<Value>) -> Result<(), Error> {
        // ensure there are actually documents to send, otherwise just return
        if values.is_empty() {
            return Ok(());
        }
        // chunk the docs into request bodies of reasonable size
        let chunks = chunk_docs(values)?;
        for chunk in chunks {
            // convert our values to json bodies
            let body = chunk
                .into_iter()
                .map(JsonBody::from)
                .collect::<Vec<JsonBody<Value>>>();
            // send these documents
            let resp = self
                .elastic
                .bulk(BulkParts::Index(index.full_name(&self.elastic_conf)))
                .body(body)
                // set a 2 minute timeout; data can be very large and may take awhile
                .request_timeout(Duration::from_secs(120))
                // filter to get only errors in the response
                .filter_path(&["items.*.error"])
                .send()
                .await?;
            // check if we ran into an error or not
            if resp.status_code().is_success() {
                // check if any errors occurred
                let resp: ElasticBulkFilteredResponse = resp.json().await?;
                // TODO: make a let chain when upgrading to 2024
                if let Some(errors) = &resp.errors {
                    if !errors.is_empty() {
                        // return an error if we got any back
                        return Err(Error::new(format!(
                            "Failed to create documents in elastic: {}",
                            serde_json::to_string(&resp).unwrap()
                        )));
                    }
                }
            } else {
                // get the error message to return
                let status_code = resp.status_code();
                let msg = resp.text().await?;
                return Err(Error::new(format!(
                    "Failed to create documents in elastic: {msg} ({status_code})",
                )));
            }
        }
        Ok(())
    }

    /// Delete documents from elastic
    ///
    /// All of the values must be `delete` requests or else the search-streamer
    /// will be confused to get anything other than `delete` responses back
    ///
    /// # Arguments
    ///
    /// * `index` - The index to send the values to
    /// * `values` - The values to send
    #[instrument(
        name = "SearchStore<Elastic>::delete",
        skip(self, store_ids),
        err(Debug)
    )]
    async fn delete(&self, index: Self::Index, store_ids: &[String]) -> Result<(), Error> {
        // ensure there are actually documents to delete, otherwise just return
        if store_ids.is_empty() {
            return Ok(());
        }
        // Create the bulk delete actions
        let query = "delete";
        let body = store_ids
            .iter()
            .map(|id| serde_json::json!({ query: { "_id": id } }).into())
            .collect::<Vec<JsonBody<Value>>>();
        // Perform the bulk delete operation
        let resp = self
            .elastic
            .bulk(BulkParts::Index(index.full_name(&self.elastic_conf)))
            .body(body)
            // filter to get only errors in the response
            .filter_path(&["items.*.error"])
            .send()
            .await?;
        // check if we ran into an error or not
        if resp.status_code().is_success() {
            // check if any errors occurred
            let resp: ElasticBulkFilteredResponse = resp.json().await?;
            if resp.errors.as_ref().is_none_or(Vec::is_empty) {
                Ok(())
            } else {
                // get the id's to log them
                let failed = resp.get_ids(query)?;
                Err(Error::new(format!(
                    "Failed to delete documents from elastic: {failed:?}"
                )))
            }
        } else {
            // get the error message to return
            let msg = resp.text().await?;
            Err(Error::new(msg))
        }
    }
}

/// A response from elastic from a bulk submission, filtered to only get errors
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ElasticBulkFilteredResponse {
    /// Any errors that occurred
    #[serde(rename = "items")]
    errors: Option<Vec<Value>>,
}

impl ElasticBulkFilteredResponse {
    /// Attempt to get a list of id's that errored from a response
    ///
    /// Returns an error if the response does not adhere to an expected format based on
    /// `ElasticSearch`'s API: <https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-bulk>
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to match on
    fn get_ids(&self, query: &str) -> Result<Vec<&str>, Error> {
        self.errors
            .iter()
            .flatten()
            // get the response for each item
            .map(|item| item.get(query))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::new(format!(
                    "malformed elastic bulk {query} response: one or more items missing '{query}' field",
                ))
            })?
            .into_iter()
            // make sure all the delete responses are valid
            .map(|item| {
                item.get("_id").and_then(|id| id.as_str())
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else( || Error::new(format!("malformed elastic bulk {query} response")))
    }
}

/// Chunk documents into groups where each group's estimated size is less
/// than the defined maximum
///
/// # Arguments
///
/// * `values` - The values to chunk
#[instrument(name = "elastic::chunk_docs", skip_all, err(Debug))]
fn chunk_docs(values: Vec<Value>) -> Result<Vec<Vec<Value>>, Error> {
    // define the maximum size of the request body in bytes
    // (1020 MB, leaving 4 MB in case of overhead)
    // TODO: maybe make this configurable?
    const MAX_BODY_SIZE: usize = 1024 * 1024 * 1000;
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_chunk_size = 0;
    // TODO: replace with 'array_chunks' when that is stabilized
    for mut chunk in values
        .into_iter()
        .chunks(2)
        .into_iter()
        .map(Iterator::collect)
        .collect::<Vec<Vec<_>>>()
    {
        let val = chunk.pop().ok_or(Error::new("Missing create document!"))?;
        let index_val = chunk.pop().ok_or(Error::new("Missing index document!"))?;
        // estimate the size of these values
        let size = sizeof_val(&val) + sizeof_val(&index_val);
        // make sure the size of this pair isn't bigger than our maximum by itself
        if size > MAX_BODY_SIZE {
            // TODO: if we hit this error, we either need to increase MAX_BODY_SIZE (maximum of 2GB because
            // elastic's doc size maximum is 2GB) or truncate the data
            return Err(Error::new(format!(
                "Document larger than the maximum request size of {MAX_BODY_SIZE} bytes!"
            )));
        }
        // check if adding this value would exceed the maximum size
        if current_chunk_size + size > MAX_BODY_SIZE {
            // Push the current chunk to the list of chunks
            chunks.push(current_chunk);
            // start a new chunk
            current_chunk = Vec::new();
            current_chunk_size = 0;
        }
        // Add the value to the current chunk
        current_chunk.push(index_val);
        current_chunk.push(val);
        current_chunk_size += size;
    }
    // push the remaining chunk if it has any values in it
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    Ok(chunks)
}

/// Estimate the size in bytes of the JSON value
///
/// Copied from <https://crates.io/crates/json_size/0.1.1>
///
/// # Arguments
///
/// * `val` - The JSON value to estimate the size of
pub fn sizeof_val(v: &serde_json::Value) -> usize {
    std::mem::size_of::<serde_json::Value>()
        + match v {
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                0
            }
            serde_json::Value::String(s) => s.len(),
            serde_json::Value::Array(a) => a.iter().map(sizeof_val).sum(),
            serde_json::Value::Object(o) => o
                .iter()
                .map(|(k, v)| {
                    std::mem::size_of::<String>()
                        + k.len()
                        + sizeof_val(v)
                        + std::mem::size_of::<usize>() * 3 // crude approximation of overhead
                })
                .sum(),
        }
}
