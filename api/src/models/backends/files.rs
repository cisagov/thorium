//! Handles saving and retrieving files objects from the backend

use aws_sdk_s3::primitives::ByteStream;
use axum::extract::multipart::Field;
use axum::extract::{FromRequestParts, Multipart};
use axum::http::request::Parts;
use chrono::prelude::*;
use futures_util::stream::{self, StreamExt};
use futures_util::{Future, TryStreamExt};
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use tracing::instrument;
use uuid::Uuid;

use super::db::{self, CursorCore, ScyllaCursorSupport};
use super::CommentSupport;
use crate::models::{
    ApiCursor, CarvedOrigin, CarvedOriginTypes, Comment, CommentForm, CommentResponse, CommentRow,
    DeleteCommentParams, DeleteSampleParams, FileListParams, Group, GroupAllowAction, Origin,
    OriginForm, OriginRequest, OriginTypes, S3Objects, Sample, SampleCheck, SampleCheckResponse,
    SampleForm, SampleListLine, SampleSubmissionResponse, Submission, SubmissionChunk,
    SubmissionListRow, SubmissionRow, SubmissionUpdate, TagListRow, TagType, User,
    ZipDownloadParams,
};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, can_create_all, can_modify, deserialize, disjoint, for_groups, not_found, serialize,
    unauthorized, update_opt,
};

impl FromStr for OriginTypes {
    type Err = ApiError;

    /// Try to convert a str into an origin type
    ///
    /// # Arguments
    ///
    /// * `raw` - The str to try to convert
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "Downloaded" => Ok(OriginTypes::Downloaded),
            "Unpacked" => Ok(OriginTypes::Unpacked),
            "Transformed" => Ok(OriginTypes::Transformed),
            "Wire" => Ok(OriginTypes::Wire),
            "Incident" => Ok(OriginTypes::Incident),
            "MemoryDump" => Ok(OriginTypes::MemoryDump),
            "Source" => Ok(OriginTypes::Source),
            "CarvedUnknown" => Ok(OriginTypes::Carved(CarvedOriginTypes::Unknown)),
            "CarvedPcap" => Ok(OriginTypes::Carved(CarvedOriginTypes::Pcap)),
            "None" => Ok(OriginTypes::None),
            _ => bad!(format!("{} is not a valid origin type", raw)),
        }
    }
}

impl TryFrom<String> for OriginTypes {
    type Error = ApiError;

    /// Try to convert a string into an origin type
    ///
    /// # Arguments
    ///
    /// * `raw` - The string to try to convert
    fn try_from(raw: String) -> Result<Self, Self::Error> {
        OriginTypes::from_str(&raw)
    }
}

impl SampleForm {
    /// Adds a multipart field to our sample form
    ///
    /// # Arguments
    ///
    /// * `field` - The field to try to add
    pub async fn add<'a>(&'a mut self, field: Field<'a>) -> Result<Option<Field<'a>>, ApiError> {
        // get the name of this field
        if let Some(name) = field.name() {
            // add this fields value to our form
            match name {
                "groups" => self.groups.push(field.text().await?),
                "description" => self.description = Some(field.text().await?),
                "origin[origin_type]" => {
                    self.origin.origin_type = OriginTypes::try_from(field.text().await?)?;
                }
                "origin[result_ids]" => self
                    .origin
                    .result_ids
                    .push(Uuid::from_str(&field.text().await?)?),
                "origin[url]" => self.origin.url = Some(field.text().await?),
                "origin[name]" => self.origin.name = Some(field.text().await?),
                "origin[tool]" => self.origin.tool = Some(field.text().await?),
                "origin[parent]" => self.origin.parent = Some(field.text().await?),
                "origin[flags]" => self.origin.flags.push(field.text().await?),
                "origin[cmd]" => self.origin.cmd = Some(field.text().await?),
                "origin[sniffer]" => self.origin.sniffer = Some(field.text().await?),
                "origin[source]" => self.origin.source = Some(field.text().await?),
                "origin[destination]" => self.origin.destination = Some(field.text().await?),
                "origin[incident]" => self.origin.incident = Some(field.text().await?),
                "origin[cover_term]" => self.origin.cover_term = Some(field.text().await?),
                "origin[mission_team]" => self.origin.mission_team = Some(field.text().await?),
                "origin[network]" => self.origin.network = Some(field.text().await?),
                "origin[machine]" => self.origin.machine = Some(field.text().await?),
                "origin[location]" => self.origin.location = Some(field.text().await?),
                "origin[memory_type]" => self.origin.memory_type = Some(field.text().await?),
                "origin[reconstructed]" => self.origin.reconstructed.push(field.text().await?),
                "origin[base_addr]" => self.origin.base_addr = Some(field.text().await?),
                "origin[repo]" => self.origin.repo = Some(field.text().await?),
                "origin[commit]" => self.origin.commit = Some(field.text().await?),
                "origin[system]" => self.origin.system = Some(field.text().await?),
                "origin[supporting]" => self.origin.supporting = Some(field.text().await?.parse()?),
                "origin[src_ip]" => self.origin.src_ip = Some(field.text().await?.parse()?),
                "origin[dest_ip]" => self.origin.dest_ip = Some(field.text().await?.parse()?),
                "origin[src_port]" => self.origin.src_port = Some(field.text().await?.parse()?),
                "origin[dest_port]" => self.origin.dest_port = Some(field.text().await?.parse()?),
                "origin[proto]" => self.origin.proto = Some(field.text().await?.parse()?),
                "trigger_depth" => self.trigger_depth = field.text().await?.parse()?,
                // this is the data so return it so we can stream it to s3
                "data" => return Ok(Some(field)),
                _ => {
                    // check if this is a tags key
                    if name.starts_with("tags[") {
                        // this is a tag to get the key substring
                        let key = &name[5..name.len() - 1];
                        // get an entry to this tags value vec
                        let entry = self.tags.entry(key.to_owned()).or_default();
                        // add our value
                        entry.insert(field.text().await?);
                        return Ok(None);
                    }
                    return bad!(format!("{} is not a valid form name", name));
                }
            }
            // we found and consumed a valid form entry
            return Ok(None);
        }
        bad!(format!("All form entries must have a name!"))
    }
}

impl CommentForm {
    /// Adds a multipart field to our sample form
    ///
    /// # Arguments
    ///
    /// * `field` - The field to try to add
    pub async fn add<'a>(&'a mut self, field: Field<'a>) -> Result<Option<Field<'a>>, ApiError> {
        // get the name of this field
        if let Some(name) = field.name() {
            match name {
                "comment" => self.comment = field.text().await?,
                // this is an attachment  so return it so we can stream it to s3
                "files" => return Ok(Some(field)),
                _ => return bad!(format!("{} is not a valid form name", name)),
            }
            // we found and consumed a valid form entry
            return Ok(None);
        }
        bad!(format!("All form entries must have a name!"))
    }
}

impl Sample {
    /// Helps the public create method save a sample to the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The User trying to save this sample
    /// * `s3_id` - The id to save this file with in s3
    /// * `upload` - The multipart form containing the sample being uploaded
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "Sample::create_helper", skip(user, upload, shared), err(Debug))]
    async fn create_helper(
        user: &User,
        s3_id: &Uuid,
        mut upload: Multipart,
        shared: &Shared,
    ) -> Result<SampleSubmissionResponse, ApiError> {
        // build a sample form to populate
        let mut form = SampleForm::default();
        // build an option to store our hashes and file_name
        let mut hashes_opt = None;
        let mut file_opt = None;
        // begin crawling over our multipart form upload
        while let Some(field) = upload.next_field().await? {
            // try to consume our fields
            if let Some(data_field) = form.add(field).await? {
                // ignore any new data fields once our hashes have been set
                if hashes_opt.is_some() {
                    continue;
                }
                // throw an error if the correct content type is not used
                if data_field.content_type().is_none() {
                    return bad!("A content type must be set for the data form entry!".to_owned());
                }
                // try to get the name for this file
                file_opt = data_field.file_name().map(|name| name.to_owned());
                // cart and stream this file into s3
                let hashes = shared
                    .s3
                    .files
                    .hash_cart_and_stream(s3_id, data_field)
                    .await?;
                // store our hashes in our hashes option
                hashes_opt = Some(hashes);
            }
        }
        // return an error if we didn't get any data to hash
        let Some(hashes) = hashes_opt else {
            return bad!(format!("Data entry must be set!"));
        };
        // make sure we actually have groups
        if form.groups.is_empty() {
            return bad!(format!(
                "No groups provided! Sample must be uploaded to at least one group."
            ));
        }
        // make sure we actually have access to all requested groups
        let groups =
            Group::authorize_check_allow_all(user, &form.groups, GroupAllowAction::Files, shared)
                .await?;
        // make sure we have the roles to upload samples in all of these groups
        can_create_all!(groups, user, shared);
        // set our file name if one was found
        form.file_name = file_opt;
        // determine if this file already exists in s3
        let exists = db::s3::object_exists(S3Objects::File, &hashes.sha256, shared).await?;
        // add this samples metadata to scylla
        match db::files::create(user, form, hashes, shared).await {
            Ok(resp) => {
                // add our new object if it doesn't already exist
                if !exists {
                    // this is a new object so add this id
                    db::s3::insert_s3_id(S3Objects::File, s3_id, &resp.sha256, shared).await?;
                } else {
                    shared.s3.files.delete(&s3_id.to_string()).await?;
                }
                Ok(resp)
            }
            Err(err) => Err(err),
        }
    }

    /// Tries to save a sample to the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The User trying to save this sample
    /// * `upload` - The multipart form containing the sample being uploaded
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::create", skip_all, err(Debug))]
    pub async fn create(
        user: &User,
        upload: Multipart,
        shared: &Shared,
    ) -> Result<SampleSubmissionResponse, ApiError> {
        // try to generate a random uuid for this sample
        let s3_id = db::s3::generate_id(S3Objects::File, shared).await?;
        // try to save this file
        match Self::create_helper(user, &s3_id, upload, shared).await {
            Ok(resp) => Ok(resp),
            Err(err) => {
                // determine if this file already exists in s3
                if db::s3::s3_id_exists(S3Objects::File, &s3_id, shared).await? {
                    // delete our multipart upload since this failed
                    shared.s3.files.delete(&s3_id.to_string()).await?;
                }
                Err(err)
            }
        }
    }

    /// Check if a submission has already been created
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is checking if this submission already exists
    /// * `check` - The submission info to check for
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::exists", skip_all, err(Debug))]
    pub async fn exists(
        user: &User,
        check: &SampleCheck,
        shared: &Shared,
    ) -> Result<SampleCheckResponse, ApiError> {
        // check if this submission exists in a group that the user can access
        db::files::exists(user, check, shared).await
    }

    /// Get a sample object for a specific sha256
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting this sample
    /// * `sha256` - The sha256 of the sample to get
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::get", skip(user, shared), err(Debug))]
    pub async fn get(user: &User, sha256: &str, shared: &Shared) -> Result<Sample, ApiError> {
        // for users we can search their groups but for admins we need to get all groups
        // try to get this sample if it exists
        match for_groups!(db::files::get, user, shared, user, sha256)? {
            // this sample exists return it
            Some(sample) => Ok(sample),
            // this sample does not exist return a 404
            None => not_found!(format!("sample {} not found", sha256)),
        }
    }

    /// Authorize that a user has access to a list of samples
    ///
    /// # Arguments
    ///
    /// * `user` - The user we are authorizing
    /// * `sha256s` - The sha256s we are authorizing a user for
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::authorize", skip(user, shared), err(Debug))]
    pub async fn authorize(
        user: &User,
        sha256s: &Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // check if this user has access to these sha256s
        // for users we can search their groups but for admins we need to get all groups
        for_groups!(db::files::authorize, user, shared, sha256s)
    }

    /// Download an object by sha256
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting this sample
    /// * `sha256` - The sha256 of the sample to get
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::download", skip(user, shared), err(Debug))]
    pub async fn download(
        user: &User,
        sha256: String,
        shared: &Shared,
    ) -> Result<ByteStream, ApiError> {
        Sample::authorize(user, &vec![sha256.clone()], shared).await?;
        // get the s3 id for this object
        let s3_id = db::s3::get_s3_id(S3Objects::File, &sha256, shared).await?;
        // this sample exists and we have access to it so download it
        shared.s3.files.download(&s3_id.to_string()).await
    }

    /// Download an object by sha256 as an encrypted zip
    ///
    /// This is not near as efficient as using CaRT and should not be used for large files.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting this sample
    /// * `sha256` - The sha256 of the sample to get
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::download_as_zip", skip(user, shared), err(Debug))]
    pub async fn download_as_zip(
        user: &User,
        sha256: String,
        params: ZipDownloadParams,
        shared: &Shared,
    ) -> Result<Vec<u8>, ApiError> {
        Sample::authorize(user, &vec![sha256.clone()], shared).await?;
        // get the s3 id for this object
        let s3_id = db::s3::get_s3_id(S3Objects::File, &sha256, shared).await?;
        // this sample exists and we have access to it so download it
        shared
            .s3
            .files
            .download_as_zip(&s3_id.to_string(), &sha256, params, shared)
            .await
    }

    /// Updates a submission for a sample
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is listing samples
    /// * `update` - The update to apply to this submission object
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::update", skip_all, err(Debug))]
    pub async fn update(
        mut self,
        user: &User,
        mut update: SubmissionUpdate,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // get the submission to update if it exists
        if let Some(sub) = self.submissions.iter_mut().find(|sub| update.id == sub.id) {
            // make sure this user can modify this submission
            can_modify!(sub.submitter, user);
            // overlay any updates
            update_opt!(sub.name, update.name);
            update_opt!(sub.description, update.description);
            // make sure we aren't adding any groups multiple times
            disjoint!([&sub.groups, &update.add_groups]);
            // validate this user can upload samples to all new groups
            Group::authorize_check_allow_all(
                user,
                &update.add_groups,
                GroupAllowAction::Files,
                shared,
            )
            .await?;
            // this user is apart of all new groups so add them to this submission
            sub.groups.append(&mut update.add_groups);
            // remove any requested groups
            sub.groups
                .retain(|group| !update.remove_groups.contains(group));
            // ensure this user did not patch away all groups as that must be an explicit delete
            if sub.groups.is_empty() {
                return bad!("You cannot remove all groups when patching a submission".to_owned());
            }
            // if an origin was set then serialize it and update our submission
            if let Some(origin) = update.origin.take() {
                let origin = Origin::try_from(origin)?;
                // update our origin
                sub.origin = origin;
            }
            // update this submission object in scylla
            db::files::update(self, &update, shared).await
        } else {
            not_found!(format!(
                "Submission {}:{} not found",
                self.sha256, update.id
            ))
        }
    }

    /// Delete a submission from this file
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting this submission
    /// * `sub_id` - The id of the submission to delete
    /// * `groups` - The groups to delete this submission from
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::delete", skip(self, user, shared), err(Debug))]
    pub async fn delete(
        &self,
        user: &User,
        sub_id: &Uuid,
        groups: &Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // try to get our submission object
        let Some(submission) = self.submissions.iter().find(|sub| &sub.id == sub_id) else {
            return not_found!(format!("Submission {} not found", sub_id));
        };
        // filter out any groups that already don't have access to the submission
        let filtered_groups: &Vec<String> = &groups
            .iter()
            .filter(|group| submission.groups.contains(group))
            .cloned()
            .collect();
        // if no groups were given, assume it's our submission's groups
        let groups = if groups.is_empty() {
            // make sure we have access to all submission groups (or are the owner)
            submission
                .can_modify(user, &submission.groups, shared)
                .await?;
            &submission.groups
        } else {
            // make sure we have access to all given groups (even those filtered out)
            submission.can_modify(user, groups, shared).await?;
            // if no groups are left after filtering, none of the given groups
            // have access to the submission, so return not found error
            if filtered_groups.is_empty() {
                return not_found!(
                    "None of the specified groups have access to the submission!".to_owned()
                );
            }
            filtered_groups
        };
        // delete this submissions rows from scylla
        db::files::delete_submission(self, submission, groups, shared).await?;
        Ok(())
    }

    /// Ensures any user requested groups are valid for this sample
    ///
    /// If no groups are specified then all groups we can see this sample in
    /// will added to the mutable groups list
    ///
    /// # Arguments
    ///
    /// * `user` - The use that is validating this samples is in some groups
    /// * `groups` - The user specified groups to check against
    /// * `editable` - Make sure these groups are editable not just viewable
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::validate_groups", skip(self, user, shared), err(Debug))]
    pub async fn validate_groups(
        &self,
        user: &User,
        groups: &mut Vec<String>,
        editable: bool,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the groups this sample is in that we can see
        let sample_groups = self.groups();
        if groups.is_empty() {
            // this user specified no groups so default to the ones we can edit
            // cast our sample groups to a vec
            let sample_groups = sample_groups
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<Vec<String>>();
            // make sure we actually have access to all requested groups
            let info = Group::authorize_all(user, &sample_groups, shared).await?;
            // check if we should filter down to just the editable groups
            if editable {
                // only add ones that we can make changes too
                let iter = info
                    .into_iter()
                    .filter(|group| group.editable(user).is_ok())
                    .map(|group| group.name);
                // add our editable group names
                groups.extend(iter);
            } else {
                // only add ones that we can make changes too
                let iter = info.into_iter().map(|group| group.name);
                // add all viewable group names
                groups.extend(iter);
            }
        } else {
            // validate our sample is in the specified groups
            if !groups
                .iter()
                .all(|group| sample_groups.contains(group.as_str()))
            {
                return unauthorized!(format!("{} is not in all specified groups", self.sha256));
            }
            // make sure we actually have access to all requested groups
            let info = Group::authorize_all(user, &groups, shared).await?;
            // make sure we have modification privileges in these groups if we are editing data
            if editable {
                can_create_all!(info, user, shared);
            }
        }
        // all groups are valid
        Ok(())
    }

    /// Ensures any user requested groups are valid for this sample
    ///
    /// If no groups are specified then all groups we can see this sample in
    /// will added to the mutable groups list
    ///
    /// # Arguments
    ///
    /// * `user` - The use that is validating this samples is in some groups
    /// * `groups` - The user specified groups to check against
    /// * `editable` - Whether to check if data in this groups is editable as well
    /// * `shared` - Shared objects in Thorium
    #[instrument(
        name = "Sample::validate_check_allow_groups",
        skip(self, user, shared),
        err(Debug)
    )]
    pub async fn validate_check_allow_groups(
        &self,
        user: &User,
        groups: &mut Vec<String>,
        action: GroupAllowAction,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the groups this sample is in that we can see
        let sample_groups = self.groups();
        if groups.is_empty() {
            // this user specified no groups so default to the ones we can edit
            // cast our sample groups to a vec
            let sample_groups = sample_groups
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<Vec<String>>();
            // make sure we actually have access to all requested groups
            let info = Group::authorize_all(user, &sample_groups, shared).await?;
            // only add ones that we can make changes too
            // if this user is trying to create new data or edit data then check for edit perms
            let iter = info
                .into_iter()
                // filter down to groups we can edit
                .filter(|group| group.editable(user).is_ok())
                // filter down to groups that accept this action
                .filter(|group| group.allowable(action).is_ok())
                // get just the group names
                .map(|group| group.name);
            // add the names of all valid groups
            groups.extend(iter);
        } else {
            // validate our sample is in the specified groups
            if !groups
                .iter()
                .all(|group| sample_groups.contains(group.as_str()))
            {
                return unauthorized!(format!("{} is not in all specified groups", self.sha256));
            }
            // make sure we actually have access to all requested groups
            let info = Group::authorize_check_allow_all(user, &groups, action, shared).await?;
            // make sure we have modification privileges in these groups
            can_create_all!(info, user, shared);
        }
        // make sure at least some groups valid
        if groups.is_empty() {
            return unauthorized!(format!("No groups allow {} to be created!", action));
        }
        // all groups are valid
        Ok(())
    }

    /// List all samples sorted by date
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is listing samples
    /// * `params` - The params to use when listing samples
    /// * `dedupe` - Whether to dedupe when listing samples or not
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Sample::list", skip(user, shared), err(Debug))]
    pub async fn list<P: Into<FileListParams> + std::fmt::Debug>(
        user: &User,
        params: P,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ApiCursor<SampleListLine>, ApiError> {
        // convert our params if needed
        let mut params = params.into();
        // authorize the groups to list files from
        user.authorize_groups(&mut params.groups, shared).await?;
        // get a chunk of the files list
        let scylla_cursor = db::files::list(params, dedupe, shared).await?;
        // convert our scylla cursor to a user facing cursor
        Ok(ApiCursor::from(scylla_cursor))
    }

    /// Adds a submission onto a sample object
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups this user can see
    /// * `sub` - The submission to add to this sample
    /// * `shared` - Shared Thorium objects
    pub(super) async fn add(
        &mut self,
        groups: &[String],
        sub: Submission,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // deserialize our origin if one was set
        let origin = match &sub.origin {
            Some(raw) => Origin::deserialize(groups, raw, shared).await?,
            None => Origin::None,
        };
        // downselect to just the fields for a submission chunk
        let chunk = SubmissionChunk {
            id: sub.id,
            name: sub.name,
            description: sub.description,
            groups: sub.groups,
            submitter: sub.submitter,
            uploaded: sub.uploaded,
            origin,
        };
        // add it to our sample object
        self.submissions.push(chunk);
        Ok(())
    }

    /// Adds a submission onto a sample object
    ///
    /// # Arguments
    ///
    /// * `sub` - The submission to add to this sample
    pub(super) fn add_row(&mut self, row: SubmissionRow) -> Result<(), ApiError> {
        // check if we already have a submission object for this submission
        if let Some(sub) = self.submissions.iter_mut().find(|sub| sub.id == row.id) {
            // we have an existing submission so just add our group
            sub.groups.push(row.group);
        } else {
            // we do not have a matching submission so create and add this one
            // deserialize our origin if one was set
            let origin = match &row.origin {
                Some(raw_origin) => deserialize!(raw_origin),
                None => Origin::None,
            };
            // downselect to just the fields for a submission chunk
            let chunk = SubmissionChunk {
                id: row.id,
                name: row.name,
                description: row.description,
                groups: vec![row.group],
                submitter: row.submitter,
                uploaded: row.uploaded,
                origin,
            };
            // add it to our sample object
            self.submissions.push(chunk);
        }
        Ok(())
    }

    /// Try to convert a submission to a sample
    ///
    /// # Arguments
    ///
    /// * `submission` - The submission to convert
    pub(crate) async fn try_from_submission(
        groups: &[String],
        sub: Submission,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // build sample with just current submission
        let mut sample = Sample {
            sha256: sub.sha256.clone(),
            sha1: sub.sha1.clone(),
            md5: sub.md5.clone(),
            tags: HashMap::with_capacity(1),
            submissions: Vec::with_capacity(1),
            comments: Vec::default(),
        };
        // add current submission as submission chunk
        sample.add(groups, sub, shared).await?;
        Ok(sample)
    }
}

impl CommentSupport for Sample {
    /// Creates a new comment
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is adding a comment
    /// * `req` - The multipart form containing the new comment
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn create_comment(
        &self,
        user: &User,
        req: Multipart,
        shared: &Shared,
    ) -> Result<CommentResponse, ApiError> {
        // build our comment form
        let mut form = CommentForm::default();
        // get the groups we can see this sample is in
        let groups = self.groups();
        // try to save this comment to the backend
        match super::comments::create_comment_helper(
            user,
            &self.sha256,
            &groups,
            req,
            &mut form,
            shared,
        )
        .await
        {
            Ok(()) => Ok(CommentResponse { id: form.id }),
            Err(err) => {
                // delete all our dangling comment attachments
                for (_, s3_id) in form.attachments {
                    // build the path to delete this attachment at in s3
                    let s3_path = format!("{}/{}/{}", &self.sha256, form.id, s3_id);
                    // delete this attachment from s3
                    shared.s3.attachments.delete(&s3_path).await?;
                }
                Err(err)
            }
        }
    }

    /// Deletes a comment
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting the comment
    /// * `groups` - The groups to delete the comment from
    /// * `id` - The id of the comment to delete
    /// * `shared` - Shared Thorium objects
    async fn delete_comment(
        &self,
        user: &User,
        groups: &[String],
        id: &Uuid,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the comment at the given id
        let Some(comment) = self.comments.iter().find(|comment| &comment.id == id) else {
            return not_found!("Error deleting comment: comment not found".to_string());
        };
        // return an unauthorized error if the user is not the comment author or isn't an admin
        if user.username != comment.author && !user.is_admin() {
            return unauthorized!(
                "Error deleting comment: comments can only be deleted by their authors".to_string()
            );
        }
        // delete from all comment groups of none were given,
        // otherwise check the comment is in all of the given groups
        let groups = if groups.is_empty() {
            &comment.groups
        } else {
            if !groups.iter().all(|group| comment.groups.contains(group)) {
                return bad!(
                    "Error deleting comment: comment is not in all of the given groups".to_string()
                );
            }
            groups
        };
        db::files::delete_comment(&self.sha256, groups, comment, shared).await?;
        db::files::prune_comment_attachments(&[comment.to_owned()], &self.sha256, shared).await
    }

    /// Downloads an attachment from a specific comment
    ///
    /// # Arguments
    ///
    /// * `comment` - The id of the comment to download
    /// * `attachment` - The id of the attachment to download
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn download_attachment(
        &self,
        comment: &Uuid,
        attachment: &Uuid,
        shared: &Shared,
    ) -> Result<ByteStream, ApiError> {
        // make sure this is a valid comment for this sample
        if let Some(comment) = self.comments.iter().find(|com| &com.id == comment) {
            // make sure this attachment is from this comment
            if comment.attachments.iter().any(|(_, id)| attachment == id) {
                // build the path to this attachment
                let path = format!("{}/{}/{}", self.sha256, comment.id, attachment);
                // download and return this attachment
                return shared.s3.attachments.download(&path).await;
            }
        }
        not_found!(format!(
            "Attachment {} for comment {} not found",
            attachment, comment
        ))
    }
}

impl SubmissionChunk {
    /// Check if a user can modify this submission
    #[instrument(name = "SubmissionChunk::can_modify", skip(user, shared), err(Debug))]
    pub async fn can_modify(
        &self,
        user: &User,
        groups: &Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // if we are the owner of this submission then we can delete it
        if self.submitter == user.username || user.is_admin() {
            // ensure that all groups exist concurrently
            stream::iter(groups)
                .map(Ok)
                .try_for_each_concurrent(None, |name| async {
                    Group::get(user, name, shared).await.map(|_| ())
                })
                .await?;
        } else {
            // we are not the owner of this submission so first check if that all
            // groups exist, then make sure we are a manager or owner in each group
            stream::iter(groups)
                .map(Ok)
                .try_for_each_concurrent(None, |name| async {
                    Group::get(user, name, shared)
                        .await?
                        .modifiable(user)
                        .map(|()| ())
                })
                .await?;
        }
        Ok(())
    }
}

impl From<SubmissionRow> for Submission {
    /// convert a [`SubmissionRow`] to a [`Submission`]
    ///
    /// # Arguments
    ///
    /// * `row` - The submission row to convert
    fn from(row: SubmissionRow) -> Self {
        Submission {
            sha256: row.sha256,
            sha1: row.sha1,
            md5: row.md5,
            id: row.id,
            name: row.name,
            description: row.description,
            groups: vec![row.group],
            submitter: row.submitter,
            origin: row.origin,
            uploaded: row.uploaded,
        }
    }
}

impl TryFrom<SubmissionRow> for Sample {
    type Error = ApiError;

    fn try_from(row: SubmissionRow) -> Result<Self, Self::Error> {
        // deserialize our origin if one was set
        let origin = match &row.origin {
            Some(raw_origin) => deserialize!(raw_origin),
            None => Origin::None,
        };
        // downselect to just the fields for a submission chunk
        let sub = SubmissionChunk {
            id: row.id,
            name: row.name,
            description: row.description,
            groups: vec![row.group],
            submitter: row.submitter,
            uploaded: row.uploaded,
            origin,
        };
        // build sample with just current submission
        let sample = Sample {
            sha256: row.sha256,
            sha1: row.sha1,
            md5: row.md5,
            tags: HashMap::with_capacity(1),
            submissions: vec![sub],
            comments: Vec::default(),
        };
        Ok(sample)
    }
}

/// If a value exists get it as a String otherwise throw an error
macro_rules! get {
    ($value:expr, $name:expr) => {
        match $value {
            Some(value) => value.to_owned(),
            None => {
                return bad!(format!(
                    "origin value {} must be set with this origin type",
                    $name
                ))
            }
        }
    };
}

impl OriginForm {
    /// Convert an [`OriginForm`] to an [`Origin`], including its associated result ID's
    ///
    /// # Arguments
    ///
    /// * `req` - The origin request to validate
    pub fn to_origin(self) -> Result<(Origin, Vec<Uuid>), ApiError> {
        // if an origin request was set then validate and serialize it
        let origin = match self.origin_type {
            OriginTypes::None => return Ok((Origin::None, Vec::default())),
            OriginTypes::Downloaded => Origin::Downloaded {
                url: get!(self.url, "url"),
                name: self.name,
            },
            OriginTypes::Unpacked => Origin::Unpacked {
                tool: self.tool,
                parent: get!(self.parent, "parent"),
                dangling: false,
            },
            OriginTypes::Transformed => Origin::Transformed {
                tool: self.tool,
                parent: get!(self.parent, "parent"),
                dangling: false,
                flags: self.flags,
                cmd: self.cmd,
            },
            OriginTypes::Wire => Origin::Wire {
                sniffer: get!(self.sniffer, "sniffer"),
                source: self.source,
                destination: self.destination,
            },
            OriginTypes::Incident => Origin::Incident {
                incident: get!(self.incident, "incident"),
                cover_term: self.cover_term,
                mission_team: self.mission_team,
                network: self.network,
                machine: self.machine,
                location: self.location,
            },
            OriginTypes::MemoryDump => Origin::MemoryDump {
                parent: get!(self.parent, "parent"),
                dangling: false,
                reconstructed: self.reconstructed,
                base_addr: self.base_addr,
            },
            OriginTypes::Source => Origin::Source {
                repo: get!(self.repo, "repo"),
                commitish: self.commitish,
                commit: get!(self.commit, "commit"),
                flags: self.flags,
                system: get!(self.system, "system"),
                supporting: get!(self.supporting, "supporting"),
            },
            OriginTypes::Carved(carved_type) => Origin::Carved {
                parent: get!(self.parent, "parent"),
                tool: self.tool,
                dangling: false,
                carved_origin: match carved_type {
                    CarvedOriginTypes::Pcap => CarvedOrigin::Pcap {
                        src_ip: self.src_ip,
                        dest_ip: self.dest_ip,
                        src_port: self.src_port,
                        dest_port: self.dest_port,
                        proto: self.proto,
                        url: self.url,
                    },
                    CarvedOriginTypes::Unknown => CarvedOrigin::Unknown,
                },
            },
        };
        Ok((origin, self.result_ids))
    }
}

impl TryFrom<OriginRequest> for Origin {
    type Error = ApiError;
    /// converts a [`OriginRequest`] into an [`Origin`]
    ///
    /// # Arguments
    ///
    /// * `req` - The origin request
    fn try_from(req: OriginRequest) -> Result<Self, Self::Error> {
        // if an origin update was set then validate and serialize it
        let origin = match req.origin_type.as_ref() {
            "Downloaded" => Origin::Downloaded {
                url: get!(req.url, "url"),
                name: req.name,
            },
            "Unpacked" => Origin::Unpacked {
                tool: req.tool,
                parent: get!(req.parent, "parent"),
                dangling: false,
            },
            "Transformed" => Origin::Transformed {
                tool: req.tool,
                parent: get!(req.parent, "parent"),
                dangling: false,
                flags: req.flags,
                cmd: req.cmd,
            },
            "Wire" => Origin::Wire {
                sniffer: get!(req.sniffer, "sniffer"),
                source: req.source,
                destination: req.destination,
            },
            "Incident" => Origin::Incident {
                incident: get!(req.incident, "incident"),
                cover_term: req.cover_term,
                mission_team: req.mission_team,
                network: req.network,
                machine: req.machine,
                location: req.location,
            },
            "MemoryDump" => Origin::MemoryDump {
                parent: get!(req.parent, "parent"),
                dangling: false,
                reconstructed: req.reconstructed,
                base_addr: req.base_addr,
            },
            "Carved" => Origin::Carved {
                parent: get!(req.parent, "parent"),
                tool: req.tool,
                dangling: false,
                carved_origin: CarvedOrigin::Unknown,
            },
            "CarvedPcap" => Origin::Carved {
                parent: get!(req.parent, "parent"),
                tool: req.tool,
                dangling: false,
                carved_origin: CarvedOrigin::Pcap {
                    src_ip: req.src_ip,
                    dest_ip: req.dest_ip,
                    src_port: req.src_port,
                    dest_port: req.dest_port,
                    proto: req.proto,
                    url: req.url,
                },
            },
            _ => {
                return bad!(
                    "unknown origin type (does it start with a capital letter?)".to_string()
                )
            }
        };
        Ok(origin)
    }
}

impl Origin {
    /// Serializes this origin if its set or return None
    pub fn serialize(&self) -> Result<Option<String>, ApiError> {
        // serialize origin if its not None
        match self {
            Origin::None => Ok(None),
            _ => Ok(Some(serialize!(self))),
        }
    }

    /// Deserialize our origin and check if any parents are dangling
    ///
    /// # Arguments
    ///
    /// * `serial` - This origin serialized as a string
    /// * `shared` - Shared Thorium objects
    pub async fn deserialize(
        groups: &[String],
        serial: &str,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // try to deserialize our origin
        let mut origin: Self = deserialize!(serial);
        // validate this origins parent if one exists and update our parent dangling bool
        match &mut origin {
            // get whether this parent sample exists or not
            Origin::Unpacked {
                parent, dangling, ..
            }
            | Origin::Transformed {
                parent, dangling, ..
            }
            | Origin::MemoryDump {
                parent, dangling, ..
            }
            | Origin::Carved {
                parent, dangling, ..
            } => *dangling = !db::files::sha256_exists(groups, &*parent, shared).await?,
            _ => (),
        }
        Ok(origin)
    }
}

impl From<SubmissionListRow> for SampleListLine {
    /// Convert a [`SubmissionRow`] to a [`SubmissionListLine`]
    ///
    /// # Arguments
    ///
    /// * `row` - The submission row to convert
    fn from(row: SubmissionListRow) -> Self {
        // build our initial group set
        let mut groups = HashSet::with_capacity(1);
        // ad this group
        groups.insert(row.group);
        // build this sample list line
        SampleListLine {
            groups,
            sha256: row.sha256,
            submission: Some(row.submission),
            uploaded: row.uploaded,
        }
    }
}

impl From<TagListRow> for SampleListLine {
    /// Convert a [`TagListRow`] to a [`SampleListLine`]
    ///
    /// # Arguments
    ///
    /// * `row` - The tag row to convert
    fn from(row: TagListRow) -> Self {
        // build our initial group set
        let mut groups = HashSet::with_capacity(1);
        // ad this group
        groups.insert(row.group);
        // build this sample list line
        SampleListLine {
            groups,
            sha256: row.item,
            submission: None,
            uploaded: row.uploaded,
        }
    }
}

impl TryFrom<CommentRow> for Comment {
    type Error = ApiError;
    /// Try to convert a [`CommentRow`] to a [`Comment`]
    ///
    /// # Arguments
    ///
    /// * `row` - The comment to convert
    fn try_from(row: CommentRow) -> Result<Self, Self::Error> {
        let comment = Comment {
            groups: vec![row.group],
            id: row.id,
            author: row.author,
            uploaded: row.uploaded,
            comment: row.comment,
            attachments: deserialize!(&row.files),
        };
        Ok(comment)
    }
}

// implement cursor for our files list
impl CursorCore for SampleListLine {
    /// The params to build this cursor from
    type Params = FileListParams;

    /// The extra info to filter with
    type ExtraFilters = ();

    /// The type of data to group our rows by
    type GroupBy = String;

    /// The data structure to store tie info in
    type Ties = HashMap<String, String>;

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &mut Self::Params) -> Option<Uuid> {
        params.cursor.take()
    }

    /// Get our start and end timestamps
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_start_end(
        params: &Self::Params,
        shared: &Shared,
    ) -> Result<(DateTime<Utc>, DateTime<Utc>), ApiError> {
        // get our end timestamp
        let end = params.end(shared)?;
        Ok((params.start, end))
    }

    /// Get any group restrictions from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_group_by(params: &mut Self::Params) -> Vec<Self::GroupBy> {
        std::mem::take(&mut params.groups)
    }

    /// Get our extra filters from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_extra_filters(_params: &mut Self::Params) -> Self::ExtraFilters {}

    /// Get our tag filters from our params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_tag_filters(
        params: &mut Self::Params,
    ) -> Option<(TagType, HashMap<String, Vec<String>>)> {
        // Only return tags if some were set
        if params.tags.is_empty() {
            None
        } else {
            Some((TagType::Files, params.tags.clone()))
        }
    }

    /// Get whether the matching on tags should be case-insensitive
    ///
    /// # Arguments
    ///
    /// `params` - The params to use to build this cursor
    fn get_tags_case_insensitive(params: &Self::Params) -> bool {
        params.tags_case_insensitive
    }

    /// Get our the max number of rows to return
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_limit(params: &Self::Params) -> usize {
        params.limit
    }

    /// Get the partition size for this cursor
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    fn partition_size(shared: &Shared) -> u16 {
        // get our partition size
        shared.config.thorium.files.partition_size
    }

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    /// * `tags` - Whether this is a tag tie or not
    fn add_tie(&self, ties: &mut Self::Ties) {
        // if its not already in the tie map then add each of its groups to our map
        for group in &self.groups {
            // get an entry to this group tie
            let entry = ties.entry(group.clone());
            // if this tie has a submission then use that otherwise use the repo url
            match &self.submission {
                Some(submission) => entry.or_insert_with(|| submission.to_string()),
                None => entry.or_insert_with(|| self.sha256.clone()),
            };
        }
    }

    /// Determines if a new item is a duplicate or not
    ///
    /// # Arguments
    ///
    /// * `set` - The current set of deduped data
    fn dedupe_item(&self, dedupe_set: &mut HashSet<String>) -> bool {
        // if this is already in our dedupe set then skip it
        if dedupe_set.contains(&self.sha256) {
            // we already have this sample so skip it
            false
        } else {
            // add this new sample to our dedupe set
            dedupe_set.insert(self.sha256.clone());
            // keep this new sample
            true
        }
    }

    /// Get the tag clustering key for a row without the timestamp
    fn get_tag_clustering_key(&self) -> &String {
        &self.sha256
    }
}

// implement cursor for our files list
#[async_trait::async_trait]
impl ScyllaCursorSupport for SampleListLine {
    /// The intermediate list row to use
    type IntermediateRow = SubmissionListRow;

    /// The unique key for this cursors row
    type UniqueType<'a> = Option<Uuid>;

    /// Add an item to our tag tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tag_tie(&self, ties: &mut HashMap<String, String>) {
        // if its not already in the tie map then add each of its groups to our map
        for group in &self.groups {
            // insert this groups tie
            ties.insert(group.clone(), self.sha256.clone());
        }
    }

    /// Get the timestamp from this items intermediate row
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a timestamp for
    fn get_intermediate_timestamp(intermediate: &Self::IntermediateRow) -> DateTime<Utc> {
        intermediate.uploaded
    }

    /// Get the timestamp for this item
    ///
    /// # Arguments
    ///
    /// * `item` - The item to get a timestamp for
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.uploaded
    }

    /// Get the unique key for this intermediate row if it exists
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a unique key for
    fn get_intermediate_unique_key<'a>(
        intermediate: &'a Self::IntermediateRow,
    ) -> Self::UniqueType<'a> {
        Some(intermediate.submission)
    }

    /// Get the unique key for this row if it exists
    fn get_unique_key<'a>(&'a self) -> Self::UniqueType<'a> {
        self.submission
    }

    /// Add a group to a specific returned line
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add to this line
    fn add_group_to_line(&mut self, group: String) {
        // add this group
        self.groups.insert(group);
    }

    /// Add a group to a specific returned line
    fn add_intermediate_to_line(&mut self, intermediate: Self::IntermediateRow) {
        // add this intermediate rows group
        self.groups.insert(intermediate.group);
    }

    /// Convert a tag list row into our list line
    fn from_tag_row(row: TagListRow) -> Self {
        SampleListLine::from(row)
    }

    /// Build all of the keys needs to retrieve census data
    ///
    /// # Arguments
    ///
    /// * `group_by` - The values to group our the rows by
    /// * `extra` - The extra values required to list this data
    /// * `year` - The year we are looking for census data for
    /// * `keys` - The vec to add our census stream keys too
    /// * `shared` - Shared Thorium objects
    fn census_keys<'a>(
        group_by: &'a Vec<Self::GroupBy>,
        _extra: &Self::ExtraFilters,
        year: i32,
        bucket: u32,
        keys: &mut Vec<(&'a Self::GroupBy, String, i32)>,
        shared: &Shared,
    ) {
        // build the keys for each census stream we are going to crawl
        for group in group_by {
            // build the key for this commitish kinds census stream
            let key = db::keys::samples::census_stream(group, year, shared);
            // add this key to our keys
            keys.push((group, key, bucket as i32));
        }
    }

    /// builds the query string for getting data from ties in the last query
    ///
    /// # Arguments
    ///
    /// * `group` - The group that this query is for
    /// * `_filters` - Any filters to apply to this query
    /// * `year` - The year to get data for
    /// * `bucket` - The bucket to get data for
    /// * `uploaded` - The timestamp to get the remaining tied values for
    /// * `breaker` - The value to use as a tie breaker
    /// * `limit` - The max number of rows to return
    /// * `shared` - Shared Thorium objects
    #[allow(clippy::too_many_arguments)]
    fn ties_query(
        ties: &mut Self::Ties,
        _extra: &Self::ExtraFilters,
        year: i32,
        bucket: i32,
        uploaded: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<Vec<impl Future<Output = Result<QueryResult, ExecutionError>>>, ApiError> {
        // allocate space for 300 futures
        let mut futures = Vec::with_capacity(ties.len());
        // if any ties were found then get the rest of them and add them to data
        for (group, id) in ties.drain() {
            // cast our id to a uuid
            let id = Uuid::parse_str(&id)?;
            // build our query future
            let future = shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.samples.list_ties,
                (group, year, bucket, uploaded, id, limit),
            );
            // add this future to our set
            futures.push(future);
        }
        Ok(futures)
    }

    /// builds the query string for getting the next page of values
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `_filters` - Any filters to apply to this query
    /// * `year` - The year to get data for
    /// * `bucket` - The bucket to get data for
    /// * `start` - The earliest timestamp to get data from
    /// * `end` - The oldest timestamp to get data from
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    #[allow(clippy::too_many_arguments)]
    async fn pull(
        group: &Self::GroupBy,
        _extra: &Self::ExtraFilters,
        year: i32,
        bucket: Vec<i32>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError> {
        // query the samples list table
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.samples.list_pull,
                (group, year, bucket, start, end, limit),
            )
            .await
    }
}

impl ApiCursor<SampleListLine> {
    /// Turns a [`SampleList`] into a [`SampleDetailsList`]
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting the details for this list
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "ApiCursor<SampleListLine>::details",
        skip_all
        err(Debug)
    )]
    pub(crate) async fn details(
        self,
        user: &User,
        shared: &Shared,
    ) -> Result<ApiCursor<Sample>, ApiError> {
        // build a string of the sha256s we want to retrieve
        let sha256s = self
            .data
            .into_iter()
            .map(|line| line.sha256)
            .collect::<Vec<String>>();
        // use correct backend to list sample details
        let data = for_groups!(db::files::list_details, user, shared, sha256s)?;
        // build our new cursor object
        Ok(ApiCursor {
            cursor: self.cursor,
            data,
        })
    }
}

impl<S> FromRequestParts<S> for FileListParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl<S> FromRequestParts<S> for DeleteCommentParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl<S> FromRequestParts<S> for DeleteSampleParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl ZipDownloadParams {
    /// Use the user specified password or the password in our config
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub fn get_password<'a>(&'a self, shared: &'a Shared) -> &'a String {
        // if the user set a password then use that otherwise use our configs password
        match &self.password {
            Some(password) => password,
            None => &shared.config.thorium.files.password,
        }
    }
}

impl<S> FromRequestParts<S> for ZipDownloadParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}
