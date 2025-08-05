//! Handles saving and retrieving repos from the backend

use aws_sdk_s3::primitives::ByteStream;
use axum::extract::multipart::Field;
use axum::extract::{FromRequestParts, Multipart};
use axum::http::request::Parts;
use chrono::prelude::*;
use futures_util::Future;
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;
use std::collections::{HashMap, HashSet};
use tracing::instrument;
use url::Url;
use uuid::Uuid;

use super::db::{self, CursorCore, ScyllaCursorSupport};
use crate::models::{
    ApiCursor, Branch, Commit, Commitish, CommitishDetails, CommitishKinds, CommitishListParams,
    CommitishListRow, CommitishMapRequest, GitTag, Group, GroupAllowAction, Repo, RepoDataForm,
    RepoDownloadOpts, RepoListLine, RepoListParams, RepoListRow, RepoRequest, RepoRow, RepoScheme,
    RepoSubmission, RepoSubmissionChunk, RepoUrlComponents, S3Objects, TagListRow, TagMap, TagType,
    User, UserRole,
};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, can_create_all, deserialize, deserialize_opt, for_groups, not_found, unauthorized,
};

/// Check if an option contains a non-empty value and cast it to a String or error
macro_rules! get_string {
    ($opt:expr, $msg:expr) => {
        match $opt {
            Some(value) => {
                if value.is_empty() {
                    return bad!($msg.to_owned());
                }
                value.to_string()
            }
            None => return bad!($msg.to_owned()),
        }
    };
}

impl RepoUrlComponents {
    /// Attempt to parse a repo URL into its components
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to parse
    pub fn parse(url: &str) -> Result<Self, ApiError> {
        const ERROR_PREFIX: &str = "Invalid repo URL";
        // make sure this url contains a base and then parse it
        let url = match (url.starts_with("https://"), url.starts_with("http://")) {
            (false, false) => Url::parse(&format!("https://{url}"))?,
            _ => Url::parse(url)?,
        };
        // try to extract the provider from this url
        let provider = get_string!(
            url.host_str(),
            format!("{ERROR_PREFIX}: \"{}\" does not contain a host!", url)
        );
        // try to extract the user and repo name from the url
        if let Some(mut segments) = url.path_segments() {
            // get the user who created this repo
            let user = get_string!(
                segments.next(),
                format!("{ERROR_PREFIX}: '{url}' does not contain a username!",)
            );
            // because nested repos exist the name of this repo will be the rest of the
            // path segments separated by '/'
            let name = itertools::Itertools::intersperse(segments, "/")
                .collect::<String>()
                .trim_end_matches('/')
                .trim_end_matches(".git")
                .to_string();
            // make sure there are no empty paths;
            // we can't just check for empty segments because the last segment can be empty
            // e.g. "github.com", "github.com/", or "github.com/user/project/"
            // (see url::Url::path_segments)
            if name.contains("//") {
                return bad!(format!(
                    "{ERROR_PREFIX}: '{url}' contains empty path components!",
                ));
            }
            if name.is_empty() {
                return bad!(format!(
                    "{ERROR_PREFIX}: '{url}' does not contain a project name!",
                ));
            }
            // try to extract the scheme
            let scheme = RepoScheme::try_from(&url)?;
            Ok(RepoUrlComponents {
                provider,
                user,
                name,
                scheme,
            })
        } else {
            bad!(format!(
                "Invalid repo URL: \"{}\" must contain a username and project name!",
                url
            ))
        }
    }
}

impl RepoDataForm {
    /// Adds a multipart field to our repo data form
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
                // this is the data so return it so we can stream it to s3
                "data" => return Ok(Some(field)),
                _ => return bad!(format!("{} is not a valid form name", name)),
            }
            // we found and consumed a valid form entry
            return Ok(None);
        }
        bad!(format!("All form entries must have a name!"))
    }
}

impl Repo {
    /// Tries to save a repo to the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user trying to save this repo
    /// * `req` - The repo request to save
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Repo::create", skip(user, shared), err(Debug))]
    pub async fn create(
        user: &User,
        req: RepoRequest,
        shared: &Shared,
    ) -> Result<String, ApiError> {
        // require at least some groups to be set
        if req.groups.is_empty() {
            return bad!("At least one group must be specified!".to_owned());
        }
        // authorize the user is apart of these groups
        let groups =
            Group::authorize_check_allow_all(user, &req.groups, GroupAllowAction::Repos, shared)
                .await?;
        // make sure we have the roles to upload samples in all of these groups
        can_create_all!(groups, user, shared);
        // add this repo to scylla
        db::repos::create(user, req, shared).await
    }

    /// Gets info about a repo from the backend
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting this repo
    /// * `repo` - The repo to get info on
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Repo::get", skip(user, shared), err(Debug))]
    pub async fn get(user: &User, repo: &str, shared: &Shared) -> Result<Repo, ApiError> {
        // for users we can search their groups but for admins we need to get all groups
        // try to get this sample if it exists
        for_groups!(db::repos::get, user, shared, user, repo)
    }

    /// Authorize that a user has access to a list of repos
    ///
    /// # Arguments
    ///
    /// * `user` - The user we are authorizing
    /// * `repos` - The repos we are authorizing a user for
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "Repo::authorize", skip(user, shared), err(Debug))]
    pub async fn authorize(
        user: &User,
        repos: &Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // check if this user has access to these repos
        // for users we can search their groups but for admins we need to get all groups
        for_groups!(db::repos::authorize, user, shared, repos)
    }

    /// Ensures any user requested groups are valid for this repo
    ///
    /// If no groups are specified then all groups we can see this repo in will be returned.
    ///
    /// # Arguments
    ///
    /// * `user` - The user we are validating groups for
    /// * `groups` - The user specified groups to check against
    /// * `editable` - Make sure these groups are editable not just viewable
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Repo::validate_groups", skip(self, user, shared), err(Debug))]
    pub async fn validate_groups<'a>(
        &'a self,
        user: &User,
        groups: &mut Vec<String>,
        editable: bool,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the groups this repo is in that we can see
        let repo_groups = self.groups();
        // if groups were specified then validate this repo is in them
        if !groups.is_empty() {
            // validate our repo is in the specified groups
            if !groups.iter().all(|group| repo_groups.contains(group)) {
                return unauthorized!(format!("{} is not in all specified groups", self.url));
            }
            // make sure we actually have access to all requested groups
            let info = Group::authorize_all(user, &groups, shared).await?;
            // if we want to edit these repos then check for edit permissions
            if editable {
                // make sure we have modification privileges in these groups
                can_create_all!(info, user, shared);
            }
        } else {
            // this user specified no groups so default to the ones we can edit
            // cast our repo groups to a vec
            let repo_groups = repo_groups
                .into_iter()
                .map(|group| group.to_owned())
                .collect::<Vec<String>>();
            // make sure we actually have access to all requested groups
            let info = Group::authorize_all(user, &repo_groups, shared).await?;
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
        }
        // all groups are valid
        Ok(())
    }

    /// Ensures any user requested groups are valid for this repo
    ///
    /// If no groups are specified then all groups we can see this repo in will be returned.
    ///
    /// # Arguments
    ///
    /// * `user` - The user we are validating groups for
    /// * `groups` - The user specified groups to check against
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(
        name = "Repo::validate_groups_action",
        skip(self, user, shared),
        err(Debug)
    )]
    pub async fn validate_check_allow_groups<'a>(
        &'a self,
        user: &User,
        groups: &mut Vec<String>,
        action: GroupAllowAction,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the groups this repo is in that we can see
        let repo_groups = self.groups();
        // if groups were specified then validate this repo is in them
        if !groups.is_empty() {
            // validate our repo is in the specified groups
            if !groups.iter().all(|group| repo_groups.contains(group)) {
                return unauthorized!(format!("{} is not in all specified groups", self.url));
            }
            // make sure we actually have access to all requested groups
            let info = Group::authorize_check_allow_all(user, &groups, action, shared).await?;
            // make sure we have modification privileges in these groups
            can_create_all!(info, user, shared);
        } else {
            // this user specified no groups so default to the ones we can edit
            // cast our repo groups to a vec
            let repo_groups = repo_groups
                .into_iter()
                .map(|group| group.to_owned())
                .collect::<Vec<String>>();
            // make sure we actually have access to all requested groups
            let info = Group::authorize_all(user, &repo_groups, shared).await?;
            // only add ones that we can make changes too
            let iter = info
                .into_iter()
                .filter(|group| group.editable(user).is_ok())
                .filter(|group| group.allowable(action).is_ok())
                .map(|group| group.name);
            groups.extend(iter);
        }
        // make sure at least some groups valid
        if groups.is_empty() {
            return unauthorized!(format!("No groups allow {} to be created!", action));
        }
        // all groups are valid
        Ok(())
    }

    /// Helps the public upload method save new data for this repository
    ///
    ///  If a user does not tie any commits to this data it will be pruned the next time commits are added.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is uploading this repo data
    /// * `s3_id` - The id to use when saving this s3 object
    /// * `upload` - The repo data to upload
    /// * `shared` - Shared objects in Thorium
    #[instrument(
        name = "Repo::upload_helper",
        skip(self, user, upload, shared),
        err(Debug)
    )]
    async fn upload_helper(
        &self,
        user: &User,
        s3_id: &Uuid,
        mut upload: Multipart,
        shared: &Shared,
    ) -> Result<String, ApiError> {
        // build a repo upload form to populate
        let mut form = RepoDataForm::default();
        // build an option to store our hashes
        let mut sha256_opt = None;
        // begin crawling our multipart form
        while let Some(field) = upload.next_field().await? {
            // try to consume our fields
            if let Some(data_field) = form.add(field).await? {
                // ignore any new data fields once our hashes have been set
                if sha256_opt.is_some() {
                    continue;
                }
                // throw an error if the correct content type is not used
                if data_field.content_type().is_none() {
                    return bad!("A content type must be set for the data form entry!".to_owned());
                }
                // cart and stream this file into s3
                let hashes = shared
                    .s3
                    .repos
                    .sha256_cart_and_stream(s3_id, data_field)
                    .await?;
                // store our hashes in our hashes option
                sha256_opt = Some(hashes);
            }
        }
        // return an error if we didn't get any data to hash
        let sha256 = match sha256_opt {
            Some(sha256) => sha256,
            None => return bad!(format!("Data entry must be set!")),
        };
        // make sure we actually have access to all requested groups
        let groups =
            Group::authorize_check_allow_all(user, &form.groups, GroupAllowAction::Repos, shared)
                .await?;
        // make sure we have the roles to upload repos in all of these groups
        can_create_all!(groups, user, shared);
        // build the path to uniquely identify this repos data
        let path = format!("{}/{}", self.url, sha256);
        // determine if this file already exists in s3
        let exists = db::s3::object_exists(S3Objects::Repo, &path, shared).await?;
        // add this samples metadata to scylla
        match db::repos::upload(&self.url, &sha256, shared).await {
            Ok(()) => {
                // add our new object if it doesn't already exist
                if !exists {
                    // this is a new object so add this id
                    db::s3::insert_s3_id(S3Objects::Repo, s3_id, &path, shared).await?;
                } else {
                    shared.s3.repos.delete(&s3_id.to_string()).await?;
                }
                Ok(sha256)
            }
            Err(err) => Err(err),
        }
    }

    /// Upload new data for this repository
    ///
    ///  If a user does not tie any commits to this data it will be pruned the next time commits are added.
    ///
    /// # Arguments
    ///
    /// * `user` - The use that is uploading new repo data
    /// * `upload` - The repo data that is being uploaded
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Repo::upload", skip_all, err(Debug))]
    pub async fn upload(
        &self,
        user: &User,
        upload: Multipart,
        shared: &Shared,
    ) -> Result<String, ApiError> {
        // try to generate a random uuid for this repo
        let s3_id = db::s3::generate_id(S3Objects::Repo, shared).await?;
        // try to save this repos data
        match self.upload_helper(user, &s3_id, upload, shared).await {
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

    /// Add or update commitishes in a specific repo
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is saving these commitishes
    /// * `data` - The sha256 of the data we are saving commitishes from
    /// * `req` - The commitishes to save
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Repo::add_commitishes", skip_all, err(Debug))]
    pub async fn add_commitishes(
        &mut self,
        user: &User,
        data: &str,
        mut req: CommitishMapRequest,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // build the path to the repo data we are saving commits for
        let path = format!("{}/{}", self.url, data);
        // make sure the backing data actually exists
        if !db::s3::object_exists(S3Objects::Repo, &path, shared).await? {
            return not_found!(format!("Data {} not found for {}", data, self.url));
        }
        // validate any user specified groups or get defaults
        self.validate_check_allow_groups(user, &mut req.groups, GroupAllowAction::Repos, shared)
            .await?;
        // save these commitishes to the backend
        db::repos::add_commitishes(self, data, req, shared).await?;
        Ok(())
    }

    /// List the commitishes for a specific repo
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is listing a repos commits
    /// * `params` - The params to use when listing commits
    /// * `dedupe` - Whether to dedupe any data that is returned
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "Repo::commits", skip(self, user, shared), err(Debug))]
    pub async fn commitishes(
        &self,
        user: &User,
        mut params: CommitishListParams,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ApiCursor<Commitish>, ApiError> {
        // authorize the groups to list files from
        user.authorize_groups(&mut params.groups, shared).await?;
        // get a chunk of the files list
        let scylla_cursor = db::repos::commitishes(&self, params, dedupe, shared).await?;
        // convert our scylla cursor to a user facing cursor
        Ok(ApiCursor::from(scylla_cursor))
    }

    /// Make sure a commit exists for a repo
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of commitish to check
    /// * `commit` - The commit to check the existence of
    /// * `shared` - Shared Thorium objects
    pub async fn commitish_exists(
        &self,
        kind: CommitishKinds,
        commit: &str,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // get the groups we can see this repo is in
        let groups = self.groups();
        // check if this commit exists for this repo
        db::repos::commitish_exists(&groups, &self.url, kind, commit, shared).await
    }

    /// Get the latest commit for this repo
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "Repo::latest_commit", skip_all, err(Debug))]
    pub async fn latest_commit(&self, shared: &Shared) -> Result<String, ApiError> {
        // get the groups we can see this repo is in
        let groups = self.groups();
        // get the latest commit for this repo
        db::repos::latest_commit(&groups, &self, shared).await
    }

    /// Get the correct repo data hash for a commit hash
    ///
    /// # Arguments
    ///
    /// * `kinds` - The kinds of commitish to download with if specified
    /// * `commit` - The commit to get the repo data hash for
    /// * `shared` - Shared Thorium objects
    pub async fn repo_data_hash(
        &self,
        kinds: &Vec<CommitishKinds>,
        commit: &str,
        shared: &Shared,
    ) -> Result<String, ApiError> {
        // get the groups we can see this repo is in
        let groups = self.groups();
        // get the correct repo data hash for this commit
        db::repos::repo_data_hash(&groups, &self.url, kinds, commit, shared).await
    }

    /// Download a tarred repo from s3
    ///
    /// If no commitish is specified then the default checkout will be used and if that doesn't exist
    /// the latest commit will be used.
    ///
    /// # Arguments
    ///
    /// * `kinds` - The kinds of commitishes to download with if specified
    /// * `commit` - The commit to download if one is specified
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(name = "Repo::download", skip(self, shared), err(Debug))]
    pub async fn download(
        &self,
        kinds: &Vec<CommitishKinds>,
        commitish: Option<String>,
        shared: &Shared,
    ) -> Result<ByteStream, ApiError> {
        // get the groups this repo is in
        let groups = self.groups();
        // if no commit was specified then get the latest commit
        let commitish = match commitish {
            Some(commitish) => commitish,
            // no commitish was specified so use our default checkout commitish
            None => match &self.default_checkout {
                Some(commitish) => commitish.value().to_owned(),
                // we don't have a default checkout so try to get the latest commit
                None => db::repos::latest_commit(&groups, &self, shared).await?,
            },
        };
        // get the repo data hash for this repo + commit combo
        let hash = db::repos::repo_data_hash(&groups, &self.url, kinds, &commitish, shared).await?;
        // build the target path for this repo object
        let path = format!("{}/{}", self.url, hash);
        // get the s3 id for the target object
        let s3_id = db::s3::get_s3_id(S3Objects::Repo, &path, shared).await?;
        // download this repo from s3
        shared.s3.repos.download(&s3_id.to_string()).await
    }

    /// List repos sorted by data
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is listing repos
    /// * `params` - The params to use when listing repos
    /// * `shared` - Shared objects in Thorium
    /// * `span` - The span to log traces under
    #[instrument(name = "Repo::list", skip(user, shared), err(Debug))]
    pub async fn list(
        user: &User,
        mut params: RepoListParams,
        shared: &Shared,
    ) -> Result<ApiCursor<RepoListLine>, ApiError> {
        // authorize the groups to list files from
        user.authorize_groups(&mut params.groups, shared).await?;
        // get a chunk of the repos list
        let scylla_cursor = db::repos::list(params, shared).await?;
        // convert our scylla cursor to a user facing cursor
        Ok(ApiCursor::from(scylla_cursor))
    }

    /// Adds a submission onto a repo object
    ///
    /// # Arguments
    ///
    /// * `sub` - The submission to add to this repo
    /// * `user` - The user that is adding a repo submission to this repo
    pub(super) fn add_row(&mut self, row: RepoRow, user: &User) -> Result<(), ApiError> {
        // check if we already have a submission object for this submission
        if let Some(sub) = self.submissions.iter_mut().find(|sub| sub.id == row.id) {
            // we have an existing submission so just add our group
            sub.groups.push(row.group);
        } else {
            // we do not have a matching submission so create and add this one
            // build our repo submission
            let chunk = RepoSubmissionChunk::try_from((row, user))?;
            // add it to our repo object
            self.submissions.push(chunk);
        }
        Ok(())
    }
}

impl TryFrom<&Url> for RepoScheme {
    type Error = ApiError;

    /// Cast a url to a RepoScheme
    ///
    /// # Arguments
    ///
    /// * `url` - The url to cast to a RepoScheme
    fn try_from(url: &Url) -> Result<Self, Self::Error> {
        // get the scheme for this repo
        let scheme = match (url.scheme(), url.password()) {
            ("https", None) => RepoScheme::Https,
            ("http", None) => RepoScheme::Http,
            ("https", Some(password)) => RepoScheme::HttpsAuthed {
                username: url.username().to_owned(),
                password: password.to_owned(),
            },
            ("http", Some(password)) => RepoScheme::HttpAuthed {
                username: url.username().to_owned(),
                password: password.to_owned(),
            },
            (_, _) => return bad!("Invalid url scheme/base".to_owned()),
        };
        Ok(scheme)
    }
}

impl TryFrom<(RepoRow, &User)> for RepoSubmission {
    type Error = ApiError;

    /// Cast a RepoRow to a ReposSubmission
    ///
    /// # Arguments
    ///
    /// * `row` - The repo row to cast
    fn try_from(input: (RepoRow, &User)) -> Result<Self, Self::Error> {
        // unpack row and user
        let (row, user) = input;
        // try to deserialize the scheme used
        let scheme = deserialize!(&row.scheme);
        // try to deserialize our default checkout behavior if its set
        let default_checkout = deserialize_opt!(&row.default_checkout);
        // if this user is an admin or the owner of this submission keep auth info
        let scheme = if user.role == UserRole::Admin || user.username == row.user {
            // keep auth info if it exists
            scheme
        } else {
            // purge auth info since this user is not an admin or the owner of this submission
            match scheme {
                RepoScheme::HttpsAuthed { .. } => RepoScheme::ScrubbedAuth,
                RepoScheme::HttpAuthed { .. } => RepoScheme::ScrubbedAuth,
                _ => scheme,
            }
        };
        // convert to a repo submission object
        let sub = RepoSubmission {
            groups: vec![row.group],
            provider: row.provider,
            user: row.user,
            name: row.name,
            url: row.url,
            id: row.id,
            creator: row.creator,
            uploaded: row.uploaded,
            scheme,
            default_checkout,
            earliest: row.earliest,
        };
        Ok(sub)
    }
}

impl TryFrom<(RepoRow, &User)> for RepoSubmissionChunk {
    type Error = ApiError;

    /// Cast a RepoRow to a ReposSubmissionChunk
    ///
    /// # Arguments
    ///
    /// * `row` - The repo row to cast
    fn try_from(input: (RepoRow, &User)) -> Result<Self, Self::Error> {
        // unpack row and user
        let (row, user) = input;
        // try to deserialize the scheme used
        let scheme = deserialize!(&row.scheme);
        // if this user is an admin or the owner of this submission keep auth info
        let scheme = if user.role == UserRole::Admin || user.username == row.user {
            // keep auth info if it exists
            scheme
        } else {
            // purge auth info since this user is not an admin or the owner of this submission
            match scheme {
                RepoScheme::HttpsAuthed { .. } => RepoScheme::ScrubbedAuth,
                RepoScheme::HttpAuthed { .. } => RepoScheme::ScrubbedAuth,
                _ => scheme,
            }
        };
        // convert to a repo submission object
        let sub = RepoSubmissionChunk {
            groups: vec![row.group],
            id: row.id,
            creator: row.creator,
            uploaded: row.uploaded,
            scheme,
            earliest: row.earliest,
        };
        Ok(sub)
    }
}

impl TryFrom<(RepoRow, &User)> for Repo {
    type Error = ApiError;

    /// Cast a RepoRow to a ReposSubmission
    ///
    /// # Arguments
    ///
    /// * `row` - The repo row to cast
    fn try_from(input: (RepoRow, &User)) -> Result<Self, Self::Error> {
        // unpack row and user
        let (row, user) = input;
        // try to deserialize the scheme used
        let scheme = deserialize!(&row.scheme);
        // try to deserialize our default checkout behavior if its set
        let default_checkout = deserialize_opt!(&row.default_checkout);
        // if this user is an admin or the owner of this submission keep auth info
        let scheme = if user.role == UserRole::Admin || user.username == row.user {
            // keep auth info if it exists
            scheme
        } else {
            // purge auth info since this user is not an admin or the owner of this submission
            match scheme {
                RepoScheme::HttpsAuthed { .. } => RepoScheme::ScrubbedAuth,
                RepoScheme::HttpAuthed { .. } => RepoScheme::ScrubbedAuth,
                _ => scheme,
            }
        };
        // build the groups this repo is in
        let groups = vec![row.group];
        // build our repo submission chunk
        let submission = RepoSubmissionChunk {
            groups,
            id: row.id,
            creator: row.creator,
            uploaded: row.uploaded,
            scheme,
            earliest: row.earliest,
        };
        // build our repo object
        let repo = Repo {
            provider: row.provider,
            user: row.user,
            name: row.name,
            url: row.url,
            tags: TagMap::default(),
            default_checkout,
            submissions: vec![submission],
            earliest: row.earliest,
        };
        Ok(repo)
    }
}

impl From<CommitishListRow> for Commitish {
    /// Convert a [`CommitListRow`] to a [`Commitish`]
    ///
    /// # Arguments
    ///
    /// * `row` - The commit row to convert
    fn from(row: CommitishListRow) -> Self {
        match row.kind {
            CommitishKinds::Commit => Commitish::Commit(Commit {
                hash: row.key,
                groups: vec![row.group],
                timestamp: row.timestamp,
            }),
            CommitishKinds::Branch => Commitish::Branch(Branch {
                name: row.key,
                groups: vec![row.group],
                timestamp: row.timestamp,
            }),
            CommitishKinds::Tag => Commitish::Tag(GitTag {
                name: row.key,
                groups: vec![row.group],
                timestamp: row.timestamp,
            }),
        }
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl CursorCore for Commitish {
    /// The params to build this cursor from
    type Params = CommitishListParams;

    /// The extra info to filter with
    type ExtraFilters = (Vec<CommitishKinds>, String);

    /// The type of data to group our rows by
    type GroupBy = String;

    /// The data structure to store tie info in
    type Ties = HashMap<String, String>;

    /// The number of buckets to crawl at once for non tag queries
    ///
    /// # Arguments
    ///
    /// * `extra_filters` - The extra filters for this query
    fn bucket_limit(extra_filters: &Self::ExtraFilters) -> u32 {
        // keep our cartesian product under 98 by dividing the number of kinds
        // we are searching against
        (98 / extra_filters.0.len()) as u32
    }

    /// Get our cursor id from params
    ///
    /// # Arguments
    ///
    /// * `params` - The params to use to build this cursor
    fn get_id(params: &mut Self::Params) -> Option<Uuid> {
        params.cursor.take()
    }

    // Get our start and end timestamps
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
    fn get_extra_filters(_params: &mut Self::Params) -> Self::ExtraFilters {
        unimplemented!("USE FROM PARAMS EXTRA INSTEAD!")
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
        shared.config.thorium.repos.partition_size
    }

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tie(&self, ties: &mut Self::Ties) {
        // if its not already in the tie map then add each of its groups to our map
        for group in self.groups() {
            // if this group doesn't already have a tie entry then add it
            ties.entry(group.clone())
                .or_insert_with(|| self.key().to_owned());
        }
    }

    /// Determines if a new item is a duplicate or not
    ///
    /// # Arguments
    ///
    /// * `set` - The current set of deduped data
    fn dedupe_item(&self, dedupe_set: &mut HashSet<String>) -> bool {
        // get our key
        let key = self.key();
        // if this is already in our dedupe set then skip it
        if dedupe_set.contains(key) {
            // we already have this commit so skip it
            false
        } else {
            // add this new commit to our dedupe set
            dedupe_set.insert(key.clone());
            // keep this new commit
            true
        }
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl ScyllaCursorSupport for Commitish {
    /// The intermediate list row to use
    type IntermediateRow = CommitishListRow;

    /// The unique key for this cursors row
    type UniqueType<'a> = (CommitishKinds, &'a String);

    /// Get the timestamp from this items intermediate row
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a timestamp for
    fn get_intermediate_timestamp(intermediate: &Self::IntermediateRow) -> DateTime<Utc> {
        intermediate.timestamp
    }

    /// Get the timestamp for this item
    ///
    /// # Arguments
    ///
    /// * `item` - The item to get a timestamp for
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.timestamp()
    }

    /// Get the unique key for this intermediate row if it exists
    ///
    /// # Arguments
    ///
    /// * `intermediate` - The intermediate row to get a unique key for
    fn get_intermediate_unique_key<'a>(
        intermediate: &'a Self::IntermediateRow,
    ) -> Self::UniqueType<'a> {
        (intermediate.kind, &intermediate.key)
    }

    /// Get the unique key for this row if it exists
    fn get_unique_key<'a>(&'a self) -> Self::UniqueType<'a> {
        (self.kind(), self.key())
    }

    /// Add a group to a specific returned line
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add to this line
    fn add_group_to_line(&mut self, group: String) {
        // add this group
        self.add_group(group);
    }

    /// Add a group to a specific returned line
    fn add_intermediate_to_line(&mut self, intermediate: Self::IntermediateRow) {
        // add this intermediate rows group
        self.add_group(intermediate.group);
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
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: u32,
        keys: &mut Vec<(&'a Self::GroupBy, String, i32)>,
        shared: &Shared,
    ) {
        // build the keys for each census stream we are going to crawl
        for group in group_by {
            // build a key for each kind we are searching
            for kind in &extra.0 {
                // build the key for this commitish kinds census stream
                let key = super::db::keys::repos::census_stream(group, year, shared);
                // add this key to our keys
                keys.push((group, key, bucket as i32));
            }
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
    fn ties_query(
        ties: &mut Self::Ties,
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: i32,
        uploaded: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<Vec<impl Future<Output = Result<QueryResult, ExecutionError>>>, ApiError> {
        // allocate space for 300 futures
        let mut futures = Vec::with_capacity(ties.len());
        // if any ties were found then get the rest of them and add them to data
        for (group, hash) in ties.drain() {
            // execute our query
            let future = shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.commitishes.list_ties,
                (
                    &extra.0, group, year, bucket, &extra.1, uploaded, hash, limit,
                ),
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
        extra: &Self::ExtraFilters,
        year: i32,
        bucket: Vec<i32>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError> {
        // execute our query
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.commitishes.list_pull,
                (&extra.0, group, year, bucket, &extra.1, start, end, limit),
            )
            .await
    }
}

impl ApiCursor<Commitish> {
    /// Turns a [`CommitList`] into a [`CommitDetailsList`]
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting commit details
    /// * `repo` - The repo we have a commit list from
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    #[instrument(
        name = "ApiCursor<Commitish>::details",
        skip(self, user, shared),
        err(Debug)
    )]
    pub(crate) async fn details(
        self,
        user: &User,
        repo: &str,
        shared: &Shared,
    ) -> Result<ApiCursor<CommitishDetails>, ApiError> {
        // get the details for these commit hashes
        let data = for_groups!(db::repos::commitish_details, user, shared, repo, self.data)?;
        // build our new cursor object
        Ok(ApiCursor {
            cursor: self.cursor,
            data,
        })
    }
}

impl From<RepoListRow> for RepoListLine {
    /// Convert a repo list row to a repo list line
    fn from(row: RepoListRow) -> RepoListLine {
        // build our initial group set
        let mut groups = HashSet::with_capacity(1);
        // ad this group
        groups.insert(row.group);
        // build our repo list line
        RepoListLine {
            groups,
            url: row.url,
            submission: Some(row.submission),
            uploaded: row.uploaded,
        }
    }
}

impl From<TagListRow> for RepoListLine {
    /// Convert a tag list row to a repo list line
    fn from(row: TagListRow) -> RepoListLine {
        // build our initial group set
        let mut groups = HashSet::with_capacity(1);
        // ad this group
        groups.insert(row.group);
        // build our repo list line
        RepoListLine {
            groups,
            url: row.item,
            submission: None,
            uploaded: row.uploaded,
        }
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl CursorCore for RepoListLine {
    /// The params to build this cursor from
    type Params = RepoListParams;

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

    // Get our start and end timestamps
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
            Some((TagType::Repos, params.tags.clone()))
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
        shared.config.thorium.repos.partition_size
    }

    /// Add an item to our tie breaker map
    ///
    /// # Arguments
    ///
    /// * `ties` - Our current ties
    fn add_tie(&self, ties: &mut Self::Ties) {
        // if its not already in the tie map then add each of its groups to our map
        for group in &self.groups {
            // get an entry to this group tie
            let entry = ties.entry(group.clone());
            // if this tie has a submission then use that otherwise use the repo url
            match &self.submission {
                Some(submission) => entry.or_insert_with(|| submission.to_string()),
                None => entry.or_insert_with(|| self.url.clone()),
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
        if dedupe_set.contains(&self.url) {
            // we already have this repo so skip it
            false
        } else {
            // add this new repo to our dedupe set
            dedupe_set.insert(self.url.clone());
            // keep this new repo
            true
        }
    }

    /// Get the tag clustering key for a row without the timestamp
    fn get_tag_clustering_key(&self) -> &String {
        &self.url
    }
}

// implement cursor for our results stream
#[async_trait::async_trait]
impl ScyllaCursorSupport for RepoListLine {
    /// The intermediate list row to use
    type IntermediateRow = RepoListRow;

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
            ties.insert(group.clone(), self.url.clone());
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
        RepoListLine::from(row)
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
            let key = super::db::keys::repos::census_stream(group, year, shared);
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
            // execute our query
            let future = shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.repos.list_ties,
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
        // execute our query
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.repos.list_pull,
                (group, year, bucket, start, end, limit),
            )
            .await
    }
}

impl ApiCursor<RepoListLine> {
    /// Gets the details for the repos in a cursor
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting the details for this list
    /// * `shared` - Shared Thorium objects
    /// * `span` - The span to log traces under
    /// * `span` - The span to log traces under
    #[instrument(name = "ApiCursor<RepoListLine>::details", skip_all, err(Debug))]
    pub(crate) async fn details(
        self,
        user: &User,
        shared: &Shared,
    ) -> Result<ApiCursor<Repo>, ApiError> {
        // build a string of the repo urls we want to retrieve
        let repos = self
            .data
            .into_iter()
            .map(|line| line.url)
            .collect::<Vec<String>>();
        // use correct backend to list repo details
        let data = for_groups!(db::repos::list_details, user, shared, repos, user)?;
        // build our new cursor object
        Ok(ApiCursor {
            cursor: self.cursor,
            data,
        })
    }
}

impl<S> FromRequestParts<S> for RepoListParams
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

impl<S> FromRequestParts<S> for CommitishListParams
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

impl<S> FromRequestParts<S> for RepoDownloadOpts
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
            // build a default RepoDownloadOpts but with all the commitish kinds set
            let default = RepoDownloadOpts {
                kinds: CommitishKinds::all(),
                commitish: None,
                progress: None,
            };
            Ok(default)
        }
    }
}
