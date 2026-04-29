//! Handles saving results into the backend

use aws_sdk_s3::primitives::ByteStream;
use axum::extract::multipart::Field;
use axum::extract::{FromRequestParts, Multipart};
use axum::http::StatusCode;
use axum::http::request::Parts;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::instrument;
use uuid::Uuid;

use super::db::{self};
use crate::models::backends::OutputSupport;
use crate::models::{
    AutoTag, AutoTagUpdate, EntityKinds, ImageVersion, Output, OutputChunk, OutputCollection,
    OutputCollectionUpdate, OutputDisplayType, OutputForm, OutputFormBuilder, OutputKind,
    OutputMap, OutputRow, Repo, ResultGetParams, Sample, User,
};
use crate::utils::{ApiError, Shared, bounder};
use crate::{bad, bad_internal, deserialize, update, update_clear, update_opt};

/// The different kinds of files to upload for results
enum ResultFileUpload<'a> {
    /// Some entities need to be uploaded
    Entities { field: Field<'a>, kind: EntityKinds },
    /// A result file needs to be uploaded
    ResultFile(Field<'a>),
}

impl<O: OutputSupport> OutputFormBuilder<O> {
    /// Upload a entities file or entity file to s3
    ///
    /// # Arguments
    ///
    /// * `result_id` - The id of the result we are uploading an entities file for
    /// * `kind` - The kind of entity we are uploading
    /// * `field` - The field to upload an entity file from
    /// * `shared` - Shared Thorium objects
    async fn upload_entities_file<'a>(
        &mut self,
        kind: EntityKinds,
        field: Field<'a>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // throw an error if the correct content type is not used
        if field.content_type().is_none() {
            return bad!("A content type must be set for the entities form entry!".to_owned());
        }
        // we don't need to validate this path because we completely control it
        // build the path to save this attachment at in s3
        let s3_path = format!("{}/__thorium_entities/{}.json", self.id, kind);
        // cart and stream this file into s3
        shared.s3.results.stream(&s3_path, field).await?;
        // add this entity kind to our form
        self.entities.entry(kind).or_default();
        Ok(())
    }

    /// Upload a result file or entity file to s3
    ///
    /// # Arguments
    ///
    /// * `result_id` - The id of the result we are uploading a result file for
    /// * `field` - The field to upload a result file from
    /// * `shared` - Shared Thorium objects
    async fn upload_result_file<'a>(
        &mut self,
        field: Field<'a>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // throw an error if the correct content type is not used
        if field.content_type().is_none() {
            return bad!("A content type must be set for the result file form entry!".to_owned());
        }
        // validate our file name for this field if we have one
        // if we don't then just use a random uuid
        let file_name = bounder::multipart_path(&field, "Result File")?;
        // result files cannot start with __thorium_ as those are protected names
        if file_name.starts_with("__thorium_") {
            // raise an error since this result file uses a protected prefix
            return bad!(format!(
                "Result files cannot start with __thorium_: '{file_name}'"
            ));
        }
        // build the path to save this attachment at in s3
        let s3_path = format!("{}/{}", self.id, file_name);
        // cart and stream this file into s3
        shared.s3.results.stream(&s3_path, field).await?;
        // add this file name to our form
        self.files.push(file_name);
        Ok(())
    }

    /// Adds a multipart field to our sample form
    ///
    /// # Arguments
    ///
    /// * `field` - The field to try to add
    async fn add<'a>(
        &mut self,
        field: Field<'a>,
    ) -> Result<Option<ResultFileUpload<'a>>, ApiError> {
        // get the name of this field
        if let Some(name) = field.name() {
            // add this fields value to our form
            // iterate over the segments ('<NAME>[<KEY1>][<KEY2>]') in the field name
            let name_segments = super::helpers::parse_bracket_segments(&name)?;
            let mut name_segments_iter = name_segments.into_iter();
            // add this fields value to our form
            match name_segments_iter
                .next()
                .ok_or(bad_internal!("Multipart field name is empty".to_string()))?
            {
                "groups" => self.groups.push(field.text().await?),
                "tool" => self.tool = Some(field.text().await?),
                "tool_version" => {
                    self.tool_version = Some(ImageVersion::from(&field.text().await?));
                }
                "cmd" => self.cmd = Some(field.text().await?),
                "result" => self.result = Some(field.text().await?),
                "display_type" => {
                    self.display_type =
                        Some(OutputDisplayType::from_str(&field.text().await?[..])?);
                }
                "extra" => self.extra = Some(deserialize!(&field.text().await?)),
                "files" => return Ok(Some(ResultFileUpload::ResultFile(field))),
                "entities_count" => {
                    // the next segment contains the kind of entity this buffer contains
                    let kind = match name_segments_iter.next() {
                        Some(kind_str) => EntityKinds::try_from(kind_str)?,
                        None => {
                            return bad!(
                                "entities must contain a kind (ex. entities[WindowsProcess])"
                                    .to_string()
                            );
                        }
                    };
                    // get the number of entities for this kind
                    let count: i64 = field.text().await?.parse()?;
                    // update our count for this entity kind
                    self.entities.insert(kind, count);
                }
                "entities" => {
                    // the next segment contains the kind of entity this buffer contains
                    let kind = match name_segments_iter.next() {
                        Some(kind_str) => EntityKinds::try_from(kind_str)?,
                        None => {
                            return bad!(
                                "entities must contain a kind (ex. entities[WindowsProcess])"
                                    .to_string()
                            );
                        }
                    };
                    // return this entity field so we can stream
                    return Ok(Some(ResultFileUpload::Entities { field, kind }));
                }
                _ => return bad!(format!("'{name}' is not a valid form name")),
            }
            // we found and consumed a valid form entry
            return Ok(None);
        }
        bad!(format!("All form entries must have a name!"))
    }

    ///  Validate and convert this [`OutputFormBuilder`] to an [`OutputForm`]
    ///
    /// This takes a mutable ref and takes most of the values in the form builder
    /// but leaves files so that we can safely clean them up in case of errors.
    fn build(&mut self) -> Result<OutputForm<O>, ApiError> {
        // make sure that all of our required options are set
        if self.tool.is_none() || self.display_type.is_none() || !O::validate_extra(&self.extra) {
            // reject this invalid request
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                Some("OutputRequest is missing fields!".to_owned()),
            ));
        }
        // Raise an error if any of our entities have a count of 0
        if let Some((kind, _)) = self.entities.iter().find(|(_, count)| **count == 0) {
            // some entity kind was uploaded but the count was set to 0 or never set
            return bad!(format!("{kind} entities did not have a count set"));
        }
        // build our output request
        let valid = OutputForm {
            id: self.id,
            groups: std::mem::take(&mut self.groups),
            tool: self.tool.take().unwrap(),
            tool_version: std::mem::take(&mut self.tool_version),
            cmd: self.cmd.take(),
            result: self.result.take().unwrap(),
            display_type: self.display_type.take().unwrap(),
            files: self.files.clone(),
            entities: self.entities.clone(),
            extra: O::extract_extra(self.extra.take()),
        };
        Ok(valid)
    }

    /// Save a result to the backend for specific samples
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is adding new results
    /// * `upload` - The mutlipart form containing our results
    /// * `form` - The results form to add our multipart entries too
    /// * `shared` - Shared objects in Thorium
    #[instrument(
        name = "OutputForm::create_results_helper",
        skip(self, user, object, upload, shared),
        err(Debug)
    )]
    async fn create_results_helper(
        &mut self,
        user: &User,
        key: O::Key,
        object: &O,
        mut upload: Multipart,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // begin crawling over our multipart form upload
        while let Some(field) = upload.next_field().await? {
            // try to consume this field
            match self.add(field).await? {
                // stream this entity file to s3
                Some(ResultFileUpload::Entities { field, kind }) => {
                    self.upload_entities_file(kind, field, shared).await?;
                }
                // stream this result file to s3
                Some(ResultFileUpload::ResultFile(field)) => {
                    self.upload_result_file(field, shared).await?;
                }
                None => (),
            }
        }
        // validate and cast our results
        let mut form = self.build()?;
        // make sure these groups are valid for this result
        object
            .validate_groups_editable(user, &mut form.groups, shared)
            .await?;
        // build the key to save results and tags too
        let key = O::build_key(key.clone(), &form.extra);
        // save these results to the backend
        db::results::create(&key, &form, shared).await?;
        // build the tag request for this results tags
        let tag_req = O::tag_req()
            .groups(form.groups.clone())
            .add("Results", &form.tool);
        // get the earliest each group has seen this object
        let earliest = object.earliest();
        // add the tags for this result
        db::tags::create(user, key, tag_req, &earliest, shared).await?;
        Ok(())
    }

    /// Save a result to the backend for a specific kind of data
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is adding new results
    /// * `kind` - The kind of data we are saving results for
    /// * `key` - The key for the data we are saving results for
    /// * `upload` - The mutlipart form containing our results
    /// * `shared` - Shared objects in Thorium
    #[instrument(
        name = "OutputForm::create_results",
        skip(self, user, object, upload, shared),
        err(Debug)
    )]
    pub async fn create_results(
        mut self,
        user: &User,
        key: O::Key,
        object: &O,
        upload: Multipart,
        shared: &Shared,
    ) -> Result<Uuid, ApiError> {
        // try to save this result to the backend
        match self
            .create_results_helper(user, key, object, upload, shared)
            .await
        {
            Ok(()) => Ok(self.id),
            Err(err) => {
                // delete all our dangling result files
                for name in self.files {
                    // build the path to delete this attachment at in s3
                    let s3_path = format!("{}/{}", self.id, name);
                    // delete this result file from s3
                    shared.s3.results.delete(&s3_path).await?;
                }
                // delete all of our dangling entity files
                for kind in self.entities.keys() {
                    // build the path to save this attachment at in s3
                    let s3_path = format!("{}/__thorium_entities/{}.json", self.id, kind);
                    // delete this result file from s3
                    shared.s3.results.delete(&s3_path).await?;
                }
                Err(err)
            }
        }
    }
}

impl OutputMap {
    /// Get results for a specific object
    ///
    /// # Arguments
    ///
    /// * `key` - The full key to get our results at
    /// * `item` - The object we are getting results for
    /// * `user` - The user that is getting results
    /// * `params` - The query params for getting results
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "OutputMap::get", skip_all, err(Debug))]
    pub async fn get<T: OutputSupport>(
        key: &str,
        item: &T,
        user: &User,
        mut params: ResultGetParams,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // authorize this user can get results from the requested groups
        item.validate_groups_viewable(user, &mut params.groups, shared)
            .await?;
        // get our results
        db::results::get(
            T::output_kind(),
            &params.groups,
            key,
            &params.tools,
            params.hidden,
            shared,
        )
        .await
    }
}

impl OutputMap {
    /// Add an output row to this map
    ///
    /// # Arguments
    ///
    /// * `row` - The output to add to this map
    /// * `groups` - The groups this result is from
    pub(super) fn add(&mut self, row: OutputRow, groups: Vec<String>) {
        // get an entry to this tools command map
        let results = self.results.entry(row.tool.clone()).or_default();
        // try to deserialize our string as a json Value
        let (result, deserialization_error) = match serde_json::from_str(&row.result) {
            Ok(value) => (value, None),
            Err(e) => (serde_json::Value::String(row.result), Some(e.to_string())),
        };
        // convert our entity counts into unsigned ints
        let entities = match row.entities {
            Some(entities) => entities
                .into_iter()
                .map(|(kind, count)| (kind, count as usize))
                .collect(),
            None => HashMap::default(),
        };
        // build our output object for this row
        let output = Output {
            id: row.id,
            groups,
            tool_version: row.tool_version,
            cmd: row.cmd,
            uploaded: row.uploaded,
            deserialization_error,
            result,
            files: row.files.unwrap_or_default(),
            entities,
            display_type: row.display_type,
            children: row.children.unwrap_or_default(),
        };
        // push our results
        results.push(output);
    }

    /// limit our output map to at most N results for each tool
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of results to keep for each tool
    pub fn limit(&mut self, limit: usize) {
        // limit all of our results to at most N
        for results in self.results.values_mut() {
            results.truncate(limit);
        }
    }
}

impl Output {
    /// Downloads a result file
    ///
    /// # Arguments
    ///
    /// * `user` - The user submitting these results
    /// * `sha256` - The sha256 we are trying to download results from
    /// * `tool` - The name of the tool these results are from
    /// * `result_id` - The ID for the result to download files from
    /// * `name` - The name of the file to download
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Output::download", skip(kind, user, shared), err(Debug))]
    pub async fn download(
        kind: OutputKind,
        user: &User,
        key: &str,
        tool: &str,
        result_id: &Uuid,
        file_path: PathBuf,
        shared: &Shared,
    ) -> Result<ByteStream, ApiError> {
        // make sure that this user has access to this repo or sample
        kind.authorize(user, key, shared).await?;
        // authorize this user has access to this result id if we are not an admin
        if !user.is_admin() {
            // we are not an admin so make sure we can see this result
            db::results::authorize(kind, &user.groups, key, tool, result_id, shared).await?;
        }
        // build the path to this file in s3
        let path = format!("{}/{}", result_id, file_path.to_string_lossy());
        // download this result file
        shared.s3.results.download(&path).await
    }
}

impl AutoTag {
    /// Update this auto tag settings object
    ///
    /// # Arguments
    ///
    /// * `update` - The updates to apply
    pub fn update(&mut self, mut update: AutoTagUpdate) {
        // update these auto tag settings
        update!(self.logic, update.logic);
        update_opt!(self.key, update.key);
        update_clear!(self.key, update.clear_key);
    }
}

impl OutputCollection {
    /// Update this output collection settings object
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply
    pub fn update(&mut self, update: OutputCollectionUpdate) {
        update!(self.handler, update.handler);
        update!(self.files.results, update.files.results);
        update!(self.files.result_files, update.files.result_files);
        update!(self.files.tags, update.files.tags);
        update!(self.children, update.children);
        update!(self.as_filesystem, update.as_filesystem);
        // update the names in the files handler
        self.files
            .names
            .retain(|name| !update.files.remove_names.contains(name));
        self.files.names.extend(update.files.add_names);
        // clear names if requested
        if update.files.clear_names {
            self.files.names = Vec::default();
        }
        // update the groups in the groups restrictions if they were specified
        if !update.groups.is_empty() {
            self.groups = update.groups;
        }
        // clear group restrictions if thats requested
        if update.clear_groups {
            self.groups = Vec::default();
        }
        // crawl over all auto tag updates
        for (key, update) in update.auto_tag {
            // if this auto tag is set to be deleted then delete it and skip to the next update
            if update.delete {
                self.auto_tag.remove(&key);
                continue;
            }
            // if this auto tag setting doesn't exist then create it
            let entry = self.auto_tag.entry(key).or_default();
            // determine if this auto tag setting should be deleted or updated
            entry.update(update);
        }
    }
}

impl From<OutputRow> for OutputChunk {
    /// Convert a [`OutputRow`] to a [`OutputChunk`]
    ///
    /// # Arguments
    ///
    /// * `row` - The row to convert
    fn from(row: OutputRow) -> Self {
        // try to deserialize our string as a json Value
        let (result, deserialization_error) = match serde_json::from_str(&row.result) {
            Ok(value) => (value, None),
            Err(e) => (serde_json::Value::String(row.result), Some(e.to_string())),
        };
        OutputChunk {
            id: row.id,
            cmd: row.cmd,
            tool_version: row.tool_version,
            uploaded: row.uploaded,
            deserialization_error,
            result,
            files: row.files.unwrap_or_default(),
            children: row.children.unwrap_or_default(),
        }
    }
}

impl<S> FromRequestParts<S> for ResultGetParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new()
                .max_depth(5)
                .deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl OutputKind {
    /// Authorize access to a result
    ///
    /// # Arguments
    ///
    /// * `user` - The user that we are authorizing
    /// * `key` - The key to determine what we are authorizing access too
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ResultKind::authorize", skip(user, shared), err(Debug))]
    pub async fn authorize(&self, user: &User, key: &str, shared: &Shared) -> Result<(), ApiError> {
        // if we are an admin then short circuit and authorize access
        if user.is_admin() {
            return Ok(());
        }
        // check if this user has access to this file
        match self {
            // authorize access to this file
            OutputKind::Files => Sample::authorize(user, &vec![key.to_owned()], shared).await,
            // authorize access to this repo
            OutputKind::Repos => Repo::authorize(user, &vec![key.to_owned()], shared).await,
        }
    }
}

/// The query params for downloading result files
#[derive(Deserialize, Debug)]
pub struct ResultFileDownloadParams {
    /// The path to the result file to download
    pub result_file: PathBuf,
}

impl<S> FromRequestParts<S> for ResultFileDownloadParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new()
                .max_depth(5)
                .deserialize_str(query)?)
        } else {
            bad!("result file query paramter required but was not given".to_string())
        }
    }
}
