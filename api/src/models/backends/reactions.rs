//! Wrappers for interacting with reactions within Thorium with different backends
//! Currently only Redis is supported

use aws_sdk_s3::primitives::ByteStream;
use chrono::prelude::*;
use std::collections::{HashMap, HashSet};
use tracing::{event, instrument, span, Level, Span};
use uuid::Uuid;

use super::db;
use crate::models::{
    BulkReactionResponse, GenericJobArgs, Group, GroupAllowAction, JobList, Pipeline, Reaction,
    ReactionDetailsList, ReactionExpire, ReactionList, ReactionRequest, ReactionStatus,
    ReactionUpdate, Repo, RepoDependency, Sample, StageLogs, StageLogsAdd, StatusUpdate, User,
};
use crate::utils::{bounder, ApiError, Shared};
use crate::{
    bad, can_delete, can_modify, deserialize, deserialize_ext, deserialize_opt, extract, is_admin,
    not_found, unauthorized,
};

impl ReactionRequest {
    /// Validate we are allowed to overwrite args for any images we try to
    ///
    /// # Arguments
    ///
    /// * `user` - The user trying to override args
    /// * `group` - The group this reaction is in
    /// * `shared` - Shared Thorium objects
    pub async fn can_override(
        &self,
        user: &User,
        group: &Group,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the images we are overriding
        let overrides = self
            .args
            .iter()
            .filter(|(_, args)| {
                args.opts.override_positionals
                    || args.opts.override_kwargs
                    || args.opts.override_cmd.is_some()
            })
            .map(|(name, _)| name)
            .collect::<Vec<&String>>();
        // only validate reactions if we found some overrides
        if !overrides.is_empty() {
            // get these images scalers
            let scalers = db::images::get_scalers(&self.group, &overrides, shared).await?;
            // make sure we have the right developer roles
            if group.developer_many(user, &scalers).is_err() {
                return unauthorized!();
            }
        }
        Ok(())
    }

    /// Uploads any ephemeral files required for this sample to execute
    ///
    /// # Arguments
    ///
    /// * `reactions` - The reaction to upload ephemeral files for
    /// * `files` - The files to save to s3
    /// * `shared` - Shared Thorium objects
    pub async fn upload_files<'v>(
        reaction: &Uuid,
        files: HashMap<String, String>,
        shared: &Shared,
    ) -> Result<Vec<String>, ApiError> {
        // validate all file names are safe
        for (name, _) in files.iter() {
            bounder::file_name(name, "ephemeral file names", 1, 32)?;
        }
        // build the list of file paths
        let mut s3_paths = Vec::with_capacity(files.len());
        // try saving these files to s3
        for (name, encoded) in files.into_iter() {
            // build the path to save this file to in s3
            let path = format!("{}/{}", reaction, name);
            // write this file to s3
            shared.s3.ephemeral.upload_base64(&path, &encoded).await?;
            s3_paths.push(name);
        }
        Ok(s3_paths)
    }

    /// Casts a reaction request to a bounds checked Reaction
    ///
    /// # Arguments
    ///
    /// * `user` - The user casting this reaction request
    /// * `pipeline` - The pipeline this reaction is for
    /// * `parent_ephemeral` - Any ephemeral files from any parent reactions
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "ReactionRequest::cast", skip_all, err(Debug))]
    pub async fn cast<'a>(
        mut self,
        user: &User,
        pipeline: &'a Pipeline,
        parent_ephemeral: HashMap<String, Uuid>,
        shared: &Shared,
    ) -> Result<(Reaction, &'a Pipeline), ApiError> {
        // ensure that all args defined are contained in the pipeline
        for image in self.args.keys() {
            if !pipeline.order.iter().any(|stage| stage.contains(image)) {
                return bad!(format!("image {} is not in this pipeline", image));
            }
        }
        // if no sla is given use the pipelines default
        let sla_seconds = self.sla.unwrap_or(pipeline.sla);
        // bounds check sla
        bounder::number(sla_seconds as i64, "sla", 1, 3.154e+9 as i64)?;
        // build the repo dependency objects
        let mut repos = Vec::with_capacity(self.repos.len());
        for req in self.repos {
            // try to get this repo to make sure this user actually has access
            let repo = Repo::get(user, &req.url, shared).await?;
            // get the commit we are going to be building against
            let commitish = match req.commitish {
                // the user specified a commit so just use that
                Some(commitish) => Some(commitish),
                // A commit wasn't specified so use the default checkout
                None => repo
                    .default_checkout
                    .map(|commitish| commitish.value().to_owned()),
            };
            // build a repo dependency object and insert it
            repos.push(RepoDependency {
                url: req.url,
                commitish,
                kind: req.kind,
            });
        }
        // generate a uuid for this reaction
        let id = Uuid::new_v4();
        // upload our extra files
        let ephemeral = Self::upload_files(&id, self.buffers, shared).await?;
        // automatically add the sha256 and submitter tags
        self.tags.append(&mut self.samples.clone());
        self.tags.push(user.username.clone());
        // create job instance
        let cast = Reaction {
            id,
            group: self.group,
            pipeline: self.pipeline,
            creator: user.username.clone(),
            status: ReactionStatus::Created,
            current_stage: 0,
            current_stage_progress: 0,
            current_stage_length: pipeline.stage_length(0)? as u64,
            args: self.args,
            sla: Utc::now() + chrono::Duration::seconds(sla_seconds as i64),
            jobs: Vec::default(),
            tags: self.tags,
            parent: self.parent,
            sub_reactions: 0,
            completed_sub_reactions: 0,
            generators: Vec::default(),
            samples: self.samples,
            ephemeral,
            parent_ephemeral,
            repos,
            trigger_depth: self.trigger_depth,
        };
        Ok((cast, pipeline))
    }
}

impl ReactionList {
    /// Creates new reaction list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used to page to the next list of reactions
    /// * `names` - A list of reaction names
    pub(super) fn new(cursor: Option<usize>, names: Vec<String>) -> Self {
        ReactionList { cursor, names }
    }

    /// Turns a [`ReactionList`] into a [`ReactionDetailsList`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group these reactions are from
    /// * `shared` - Shared Thorium objects
    pub(crate) async fn details(
        self,
        group: &str,
        shared: &Shared,
    ) -> Result<ReactionDetailsList, ApiError> {
        // use correct backend to list reaction details
        let details = db::reactions::list_details(group, &self.names, shared).await?;
        // cast to reaction details list
        let details_list = ReactionDetailsList::new(self.cursor, details);
        Ok(details_list)
    }
}

impl ReactionDetailsList {
    /// Creates a new reaction details list object
    ///
    /// # Arguments
    ///
    /// * `cursor` - A cursor used to page to the next list of reactions
    /// * `details` - A list of reaction details
    pub(super) fn new(cursor: Option<usize>, details: Vec<Reaction>) -> Self {
        ReactionDetailsList { cursor, details }
    }
}

impl Reaction {
    /// Creates a new reaction
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating this reaction
    /// * `group` - The group this reaction is in
    /// * `pipeline` - The pipeline this reaction is apart of
    /// * `request` - The ReactionRequest to build a reaction on
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Reactions::create", skip_all, err(Debug))]
    pub async fn create(
        user: &User,
        group: &Group,
        pipeline: &Pipeline,
        request: ReactionRequest,
        shared: &Shared,
    ) -> Result<Reaction, ApiError> {
        // log the group and pipeline we are creating a reaction for
        event!(Level::INFO, group = &group.name, pipeline = &pipeline.name);
        // make sure we can create reactions in this group
        group.allowable(GroupAllowAction::Reactions)?;
        // make sure we can create reactions in this group
        group.editable(user)?;
        // make sure we have access to any samples we are trying to create reactions for
        if !request.samples.is_empty() {
            // authorize this user has access to all the samples to pass in to this reaction
            Sample::authorize(user, &request.samples, shared).await?;
        }
        // make sure we are allowed to override any args we try too
        request.can_override(user, group, shared).await?;
        // add reaction to backend
        db::reactions::create(user, request, pipeline, shared).await
    }

    /// Creates a new reactions in bulk
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating this reaction
    /// * `requests` - The ReactionRequests to build reactions on
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Reactions::create_bulk", skip_all, err(Debug))]
    pub async fn create_bulk(
        user: &User,
        requests: Vec<ReactionRequest>,
        shared: &Shared,
    ) -> Result<BulkReactionResponse, ApiError> {
        // build cache of all different pipelines and groups we are creating reactions for
        let mut pipe_cache = HashMap::with_capacity(1);
        let mut group_cache = HashMap::with_capacity(1);
        // build the map of images we are overriding args for
        let mut override_cache: HashMap<&String, HashSet<&String>> = HashMap::default();
        for req in &requests {
            // build combined group + pipeline key
            let key = format!("{}:{}", &req.group, &req.pipeline);
            // get pipeline if we don't have it in the cache
            if let std::collections::hash_map::Entry::Vacant(e) = pipe_cache.entry(key) {
                // get this group and pipeline
                let (group, pipeline) =
                    Pipeline::get(user, &req.group, &req.pipeline, shared).await?;
                // insert into cache and set
                e.insert(pipeline);
                // check if this group is also not in the cache
                group_cache.entry(&req.group).or_insert(group);
                // get the images we are overriding args for
                let overrides = req
                    .args
                    .iter()
                    .filter(|(_, args)| {
                        args.opts.override_positionals
                            || args.opts.override_kwargs
                            || args.opts.override_cmd.is_some()
                    })
                    .map(|(name, _)| name)
                    .collect::<Vec<&String>>();
                // if we are overriding args then add them to our map
                if !overrides.is_empty() {
                    // get an entry to this groups override set
                    let entry = override_cache
                        .entry(&req.group)
                        .or_insert_with(|| HashSet::with_capacity(overrides.len()));
                    // add our images
                    entry.extend(overrides.into_iter());
                }
            }
        }
        // make sure none of these pipelines are banned
        let banned_pipelines = pipe_cache
            .iter()
            .filter_map(|(key, pipe)| (!pipe.bans.is_empty()).then_some(key))
            .collect::<Vec<&String>>();
        if !banned_pipelines.is_empty() {
            return bad!(format!(
                "Unable to create reaction(s)! The following pipelines have \
                one or more bans: '{banned_pipelines:?}'. See their notifications for details."
            ));
        }
        // make sure we can create reactions in all of these groups
        for group in group_cache.values() {
            // make sure we can create reactions in this group
            group.allowable(GroupAllowAction::Reactions)?;
            // make sure this group is editable
            group.editable(user)?;
        }
        // make sure we can override args in each of the images we try too
        for (group, images) in override_cache {
            // get our group info from the group cache
            let group = match group_cache.get(group) {
                Some(group) => group,
                None => return unauthorized!(),
            };
            // cast our set of images to a vec
            let overrides = images.into_iter().collect::<Vec<&String>>();
            // get these images scalers
            let scalers = db::images::get_scalers(&group.name, &overrides, shared).await?;
            // make sure we have the right developer roles
            if group.developer_many(user, &scalers).is_err() {
                return unauthorized!();
            }
        }
        // add reaction to backend
        db::reactions::create_bulk(user, requests, &pipe_cache, shared).await
    }

    /// Creates a new reactions in bulk for different users
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is creating these reactions for others
    /// * `requests` - The ReactionRequests to build reactions on
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Reactions::create_bulk_by_user", skip_all, err(Debug))]
    pub async fn create_bulk_by_user(
        user: &User,
        requests: HashMap<String, Vec<ReactionRequest>>,
        shared: &Shared,
    ) -> Result<HashMap<String, BulkReactionResponse>, ApiError> {
        // only admins can create reactions for other users
        is_admin!(user);
        // build a map of reaction creation responses by user
        let mut resp = HashMap::with_capacity(requests.len());
        // create each users reactions
        for (username, reqs) in requests {
            // get this users info
            let other_user = User::force_get(&username, shared).await?;
            // create this users reactions
            let user_resp = Self::create_bulk(&other_user, reqs, shared).await?;
            // add the responses for creating this users reactions
            resp.insert(username, user_resp);
        }
        Ok(resp)
    }

    /// Gets a reaction object from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting this reaction
    /// * `group` - The group this reaction is in
    /// * `id` - The id of the reaction to retrieve
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    pub async fn get(
        user: &User,
        group: &str,
        id: &Uuid,
        shared: &Shared,
        span: &Span,
    ) -> Result<(Group, Self), ApiError> {
        // start our get reaction span
        span!(
            parent: span,
            Level::INFO,
            "Get Reaction",
            group = group,
            id = id.to_string()
        );
        // make sure we are a member of this group and it exists
        let group = Group::authorize(user, group, shared).await?;
        let reaction = db::reactions::get(&group.name, id, shared).await?;
        Ok((group, reaction))
    }

    /// Gets the status logs for a reaction
    ///
    /// # Arguments
    ///
    /// * `cursor` - The number of status logs to skip in the backend
    /// * `limit` - The max number of status logs to retrieve (strongly enforced)
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    pub async fn logs(
        &self,
        cursor: usize,
        limit: usize,
        shared: &Shared,
        span: &Span,
    ) -> Result<Vec<StatusUpdate>, ApiError> {
        // use correct backend to get reaction logs
        db::reactions::logs(self, cursor, limit, shared, span).await
    }

    /// Adds logs for a specific stage within a pipeline
    ///
    /// This is for stage logs not status logs for an entire reaction.
    ///
    /// # Arguments
    ///
    /// * `stage` - The stage these logs are from
    /// * `logs` - The logs to save into the backend
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "Reactions::add_stage_logs", skip_all, err(Debug))]
    pub async fn add_stage_logs(
        &self,
        stage: &str,
        logs: StageLogsAdd,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        event!(Level::INFO, reaction = self.id.to_string());
        // use correct backend to get reaction logs
        db::reactions::add_stage_logs(&self.id, stage, logs, shared).await
    }

    /// Gets the stdout/stderr output from a specific stage with a cursor
    ///
    /// # Arguments
    ///
    /// * `stage` - The stage to retrieve logs from
    /// * `cursor` - The number of logs to skip in the backend
    /// * `limit` - The max number of logs to retrieve (strongly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn stage_logs(
        &self,
        stage: &str,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<StageLogs, ApiError> {
        // use correct backend to get reaction logs
        db::reactions::stage_logs(self, stage, cursor, limit, shared).await
    }

    /// Lists reactions for a pipeline
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The pipeline to list reactions from
    /// * `cursor` - The page of reactions to retrieve
    /// * `limit` - The max number of reactions to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list(
        pipeline: &Pipeline,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ReactionList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list(&pipeline.group, &pipeline.name, cursor, limit, shared).await
    }

    /// Lists reactions by a status
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `pipeline` - The pipeline to list reactions from
    /// * `status` - The status of reactions to list
    /// * `cursor` - The page of reactions to retrieve
    /// * `limit` - The max number of reactions to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_status(
        group: &Group,
        pipeline: &Pipeline,
        status: &ReactionStatus,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ReactionList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list_status(&group.name, &pipeline.name, status, cursor, limit, shared).await
    }

    /// Lists reactions by a tag
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `tag` - The tag of reactions to list
    /// * `cursor` - The page of reactions to retrieve
    /// * `limit` - The max number of reactions to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_tag(
        group: &Group,
        tag: &str,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ReactionList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list_tag(&group.name, tag, cursor, limit, shared).await
    }

    /// Lists reactions for an entire group with a set status
    ///
    /// # Arguments
    ///
    /// * `group` - The group to list reactions from
    /// * `status` - The status for this reaction
    /// * `cursor` - The page of reactions to retrieve
    /// * `limit` - The max number of reactions to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_group_set(
        group: &Group,
        status: &ReactionStatus,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ReactionList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list_group_set(&group.name, status, cursor, limit, shared).await
    }

    /// Lists sub reactions for a reaction
    ///
    /// # Arguments
    ///
    /// * `reaction` - The reaction to list subreactions for
    /// * `cursor` - The page of reactions to retrieve
    /// * `limit` - The max number of reactions to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_sub(
        reaction: &Reaction,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ReactionList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list_sub(&reaction.group, &reaction.id, cursor, limit, shared).await
    }

    /// Lists sub reactions for a reaction by status
    ///
    /// # Arguments
    ///
    /// * `reaction` - The reaction to list subreactions for
    /// * `status` - The status of subreactions to list
    /// * `cursor` - The page of reactions to retrieve
    /// * `limit` - The max number of reactions to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_sub_status(
        reaction: &Reaction,
        status: &ReactionStatus,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<ReactionList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list_sub_status(&reaction.group, &reaction.id, status, cursor, limit, shared)
            .await
    }

    /// lists all jobs in a reaction
    ///
    /// # Arguments
    ///
    /// * `cursor` - The page of jobs to retrieve
    /// * `limit` - The max number of jobs to retrieve (weakly enforced)
    /// * `shared` - Shared objects in Thorium
    pub async fn list_jobs(
        &self,
        cursor: usize,
        limit: usize,
        shared: &Shared,
    ) -> Result<JobList, ApiError> {
        // use correct backend to list reaction names
        db::reactions::list_jobs(self, cursor, limit, shared).await
    }

    /// Proceeds with the reaction
    ///
    /// This increments the current stage and either completes the reaction or creates
    /// jobs for that stage.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is proceeding with the reaction
    /// * `group` - The group this reaction is in
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "Reaction::proceed", skip_all, err(Debug))]
    pub async fn proceed(
        self,
        user: &User,
        group: &Group,
        shared: &Shared,
    ) -> Result<ReactionStatus, ApiError> {
        // make sure we can modify reactions in this group
        can_modify!(self.creator, group, user);
        // use correct backend for proceeding with this reaction
        let status = db::reactions::proceed(self, shared).await?;
        // cast this JobHandleStatus to a reaction status
        Ok(ReactionStatus::from(status))
    }

    /// ApiErrors a reaction out
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is erroring out this reaction
    /// * `group` - The group this reaction is in
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "Reaction::fail", skip_all, err(Debug))]
    pub async fn fail(
        self,
        user: &User,
        group: &Group,
        shared: &Shared,
    ) -> Result<ReactionStatus, ApiError> {
        // make sure we can modify reactions in this group
        can_modify!(self.creator, group, user);
        // use correct backend for failing this reaction
        db::reactions::fail(self, shared).await
    }

    /// Deletes a reaction from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting this reaction
    /// * `group` - The group this reaction is in
    /// * `shared` - Shared objects in Thorium
    pub async fn delete(
        &self,
        user: &User,
        group: &Group,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // Create our span
        span!(Level::INFO, "Deleting Reaction");
        // make sure we can modify reactions in this group
        can_delete!(self, group, user);
        // use correct backend for deleting this reaction
        db::reactions::delete(self, shared).await
    }

    /// Deletes all reactions in a pipeline from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting all reactions from a pipeline
    /// * `group` - The group the pipeline to delete reactions from is in
    /// * `pipeline` - The pipeline to delete reactions for
    /// * `skip_check` - skip checking if this user can delete these reactions
    /// * `shared` - Shared objects in Thorium
    pub async fn delete_all(
        user: &User,
        group: &Group,
        pipeline: &Pipeline,
        skip_check: bool,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // Create our span
        span!(Level::INFO, "Deleting All Reactions For Pipeline");
        // skip the perms check if we are sure this user is already able to delete this pipelines reactions
        if !skip_check {
            // make sure we are an owner of this group
            group.modifiable(user)?;
        }
        // use correct backend for deleting all reactions
        db::reactions::delete_all(user, group, pipeline, shared).await
    }

    /// Cleans up expired reactions in the reaction status list
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting all reactions from a pipeline
    /// * `shared` - Shared objects in Thorium
    pub async fn expire_lists(user: &User, shared: &Shared) -> Result<(), ApiError> {
        // only admins can clean up reaction lists
        is_admin!(user);
        // use correct backend for proceeding with reaction
        db::reactions::expire_lists(shared).await
    }

    /// Updates the arguments for later stages of this [`Reaction`]
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is updating this reaction
    /// * `group` - The group this reaction is in
    /// * `update` - The update to apply to this reaction
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Reaction::update", skip_all, err(Debug))]
    pub async fn update(
        mut self,
        user: &User,
        group: &Group,
        mut update: ReactionUpdate,
        shared: &Shared,
    ) -> Result<Reaction, ApiError> {
        // make sure we can edit/create reactions in this group
        group.allowable(GroupAllowAction::Reactions)?;
        // make sure this user can modify this reaction
        can_modify!(self.creator, group, user);
        // apply all updates to the args in this reaction
        for (stage, mut args) in update.args.drain() {
            // get args entry or insert a default one
            let entry = self
                .args
                .entry(stage)
                .or_insert_with(GenericJobArgs::default);
            // apply our updates
            // if any positionals were set overwrite the original ones
            if !args.positionals.is_empty() {
                entry.positionals = args.positionals.drain(..).collect();
            }
            // remove any kwargs requested to be removed
            entry
                .kwargs
                .retain(|key, _| !args.remove_kwargs.contains(key));
            // overlay any new kwargs on top of the originals
            for (key, value) in args.kwargs.drain() {
                entry.kwargs.insert(key, value);
            }
            // remove any requested switches
            entry
                .switches
                .retain(|val| !args.remove_switches.contains(val));
            // inject new switches
            entry.switches.extend(args.add_switches);
            // overwrite our options if any have been set
            if let Some(opts) = args.opts.take() {
                entry.opts = opts;
            }
        }
        // update our reaction
        self.tags.retain(|tag| !update.remove_tags.contains(tag));
        // add new tags
        self.tags.extend(update.add_tags.clone());
        // upload any new ephemeral files and add them to our reaction
        let mut paths = ReactionRequest::upload_files(&self.id, update.ephemeral, shared).await?;
        self.ephemeral.append(&mut paths);
        // apply our updates to the reaction data in the backend
        db::reactions::update(&self, &update.add_tags, &update.remove_tags, shared).await?;
        Ok(self)
    }

    /// Downloads an ephemeral file tied to a reaction
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the ephemeral file to download
    /// * `shared` - Shared Thorium objects
    pub async fn download_ephemeral(
        &self,
        name: &str,
        shared: &Shared,
    ) -> Result<ByteStream, ApiError> {
        // make sure this is a valid ephemeral file for this reaction
        if self.ephemeral.iter().any(|value| value == name) {
            // build the path to save this file to in s3
            let path = format!("{}/{}", self.id, name);
            // download this attachment
            return shared.s3.ephemeral.download(&path).await;
        }
        not_found!(format!(
            "Ephemeral file {} not found for reaction {}",
            name, self.id
        ))
    }
}

/// This should probably a TryFrom but I am unsure how to enforce that ApiError implements Deserialize
impl ReactionExpire {
    pub(super) fn cast(raw: &str) -> Result<Self, ApiError> {
        // Ok wrap this to get around error not be deserializable
        let expire = deserialize!(raw);
        Ok(expire)
    }
}

impl TryFrom<(HashMap<String, String>, Vec<String>, Vec<String>)> for Reaction {
    type Error = ApiError;

    /// Try to cast a HashMap of strings and a vector of strings into a Reaction
    ///
    /// # Arguments
    ///
    /// * `raw` - The HashMap and strings to cast into a Reaction
    fn try_from(
        raw: (HashMap<String, String>, Vec<String>, Vec<String>),
    ) -> Result<Self, Self::Error> {
        // unwrap into hashmap and vector of job IDs
        let (mut map, raw_jobs, raw_generators) = raw;
        // return 404 if hashmap is empty
        if map.is_empty() {
            return not_found!("reaction not found".to_string());
        }

        // cast jobs to uuids
        let jobs = raw_jobs
            .iter()
            .map(|id| Uuid::parse_str(id))
            .filter_map(Result::ok)
            .collect();

        // cast generators to uuids
        let generators = raw_generators
            .iter()
            .map(|id| Uuid::parse_str(id))
            .filter_map(Result::ok)
            .collect();

        // cast to a Reaction
        let reaction = Reaction {
            id: Uuid::parse_str(&extract!(map, "id"))?,
            group: extract!(map, "group"),
            pipeline: extract!(map, "pipeline"),
            creator: extract!(map, "creator"),
            status: deserialize_ext!(map, "status"),
            current_stage: extract!(map, "current_stage").parse::<u64>()?,
            current_stage_progress: extract!(map, "current_stage_progress").parse::<u64>()?,
            current_stage_length: extract!(map, "current_stage_length").parse::<u64>()?,
            args: deserialize_ext!(map, "args"),
            sla: deserialize_ext!(map, "sla"),
            tags: deserialize_ext!(map, "tags"),
            jobs,
            parent: deserialize_opt!(map, "parent"),
            sub_reactions: extract!(map, "sub_reactions", "0".to_owned()).parse::<u64>()?,
            completed_sub_reactions: extract!(map, "completed_sub_reactions").parse::<u64>()?,
            generators,
            samples: deserialize_ext!(map, "samples", Vec::default()),
            ephemeral: deserialize_ext!(map, "ephemeral", Vec::default()),
            parent_ephemeral: deserialize_ext!(map, "parent_ephemeral", HashMap::default()),
            repos: deserialize_ext!(map, "repos", Vec::default()),
            trigger_depth: deserialize_opt!(map, "trigger_depth"),
        };
        Ok(reaction)
    }
}
