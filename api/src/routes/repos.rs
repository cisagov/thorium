//! The repos related routes for Thorium

use axum::Router;
use axum::extract::{Json, Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum_extra::body::AsyncReadBody;
use tracing::instrument;
use utoipa::OpenApi;

/* TODO_UTOIPA: many routes in this file depend on path wildcards (e.g.
   /repos/data/\*repo_path), but Utoipa (and maybe OpenAPI?) does not
   handle this format. If the utoipa annotations for those functions
   are left in, the code will not compile because rust marks them as
   "unreachable".

   One potential fix for this is to convert all the wildcards currently
   handled as Vec<String> into a single String argument that can be
   split on '/' later.

   All functions affected by this are marked with 'TODO_UTOIPA: WILDCARD'
   and have their utoipa annotations commented out. The annotations
   should be at least somewhat functional if they were able to be used
   but have not been tested due to the aforementioned compilation
   issue.
*/

use super::OpenApiSecurity;
use crate::models::backends::TagSupport;
use crate::models::{
    ApiCursor, Branch, BranchDetails, BranchRequest, Commit, CommitDetails, CommitRequest,
    Commitish, CommitishDetails, CommitishKinds, CommitishListParams, CommitishMapRequest,
    CommitishRequest, GitTag, GitTagDetails, GitTagRequest, Output, OutputFormBuilder, OutputKind,
    OutputMap, OutputResponse, Repo, RepoCheckout, RepoCreateResponse, RepoDataUploadResponse,
    RepoDownloadOpts, RepoListLine, RepoListParams, RepoRequest, RepoScheme, RepoSubmissionChunk,
    ResultFileDownloadParams, ResultGetParams, TagDeleteRequest, TagRequest, User,
};
use crate::utils::{ApiError, AppState, bounder};

/// Allow users to add a repo to Thorium
///
/// # Arguments
///
/// * `user` - The user that is uploading sample
/// * `shared` - Shared Thorium objects
/// * `req` - The repo that is being added
#[utoipa::path(
    post,
    path = "/api/repos/",
    params(
        ("req" = RepoRequest, description = "The repo that is being added"),
    ),
    responses(
        (status = 200, description = "Repo created", body = RepoCreateResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::repos::create", skip_all, err(Debug))]
async fn create(
    user: User,
    State(state): State<AppState>,
    Json(req): Json<RepoRequest>,
) -> Result<Json<RepoCreateResponse>, ApiError> {
    // save this repo into the backend
    let url = Repo::create(&user, req, &state.shared).await?;
    Ok(Json(RepoCreateResponse { url }))
}

/// Get info about a specific repo
///
/// # Arguments
///
/// * `user` - The user that is uploading sample
/// * `path` - The path of the repo to get info about
/// * `shared` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     get,
//     path = "/api/repos/data/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path of the repo to get info about"),
//     ),
//     responses(
//         (status = 200, description = "Repo created", body = Repo),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::get_repo", skip_all, err(Debug))]
async fn get_repo(
    user: User,
    Path(repo): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Repo>, ApiError> {
    // get this repos info
    let info = Repo::get(&user, &repo, &state.shared).await?;
    Ok(Json(info))
}

/// Save new data to a repo
///
/// # Arguments
///
/// * `user` - The user that is updating this repos commit data
/// * `path` - The path of the repo to add commits too
/// * `shared` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     post,
//     path = "/api/repos/data/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path of the repo to add commits too"),
//     ),
//     responses(
//         (status = 200, description = "Repo updated", body = RepoDataUploadResponse),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::upload", skip_all, err(Debug))]
async fn upload(
    user: User,
    Path(repo_path): Path<String>,
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<RepoDataUploadResponse>, ApiError> {
    // get this repos info
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // save the data for this repo into the backend
    let sha256 = repo.upload(&user, multipart, &state.shared).await?;
    Ok(Json(RepoDataUploadResponse { sha256 }))
}

/// Add commitishes to a repo
///
/// # Arguments
///
/// * `user` - The user that is updating this repos commit data
/// * `path` - The path containing this urls args
/// * `shared` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     post,
//     path = "/api/repos/commitishes/:data/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path containing this urls args"),
//         ("req" = CommitishMapRequest, description = "Commits to update"),
//     ),
//     responses(
//         (status = 204, description = "Commitishes added"),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::update_commitishes", skip_all, err(Debug))]
async fn update_commitishes(
    user: User,
    Path((data, repo_path)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(req): Json<CommitishMapRequest>,
) -> Result<StatusCode, ApiError> {
    // get this repos info
    let mut repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // update the commits for this repo
    repo.add_commitishes(&user, &data, req, &state.shared)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Adds new tags to a repo
///
/// # Arguments
///
/// * `user` - The user that is checking whether a sample exists or not
/// * `sha256` - The sample to add a tag too
/// * `state` - Shared Thorium objects
/// * `tags` - The new tags to apply
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     post,
//     path = "/api/repos/tags/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path containing this urls args"),
//         ("tags" = TagRequest<Repo>, description = "The new tags to apply"),
//     ),
//     responses(
//         (status = 204, description = "Tags added"),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::tag", skip_all, err(Debug))]
async fn tag(
    user: User,
    Path(repo_path): Path<String>,
    State(state): State<AppState>,
    Json(tags): Json<TagRequest<Repo>>,
) -> Result<StatusCode, ApiError> {
    // get the repo we are adding tags too
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // try to add the new tags for this sample
    repo.tag(&user, tags, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes a tag from a specific repo
///
/// # Arguments
///
/// * `user` - The user that is deleting a tag
/// * `sha256` - The repo to delete a tag from
/// * `key` - The key of the tag to delete
/// * `value` - The value of the tag to delete
/// * `params` - The url query params to use
/// * `state` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     delete,
//     path = "/api/repos/tags/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path containing this urls args"),
//         ("tags" = TagDeleteRequest<Repo>, description = "The tags to delete"),
//     ),
//     responses(
//         (status = 204, description = "Tags deleted"),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::delete_tags", skip_all, err(Debug))]
async fn delete_tags(
    user: User,
    Path(repo_path): Path<String>,
    State(state): State<AppState>,
    Json(tags_del): Json<TagDeleteRequest<Repo>>,
) -> Result<StatusCode, ApiError> {
    // get the repo we are deleting tags from
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // delete the tags from this repo
    repo.delete_tags(&user, tags_del, &state.shared).await?;
    // delete the tags from this
    Ok(StatusCode::NO_CONTENT)
}

/// List the commitishes for a repo
///
/// # Arguments
///
/// * `user` - The user that is listing the commits for a repo
/// * `params` - The params to use when listing commits for this repo
/// * `repo` - The repo to list commits from
/// * `shared` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     get,
//     path = "/api/repos/commitishes/:data/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path containing this urls args"),
//         ("params" = CommitishListParams, description = "The params to use when listing commits for this repo"),
//     ),
//     responses(
//         (status = 200, description = "Commits for repo", body = ApiCursor<Commitish>),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::commitishes", skip_all, err(Debug))]
async fn commitishes(
    user: User,
    params: CommitishListParams,
    Path(path): Path<Vec<String>>,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<Commitish>>, ApiError> {
    // build our repo path, joining the `:data`
    // param with the `*repo_path` wildcard
    let repo_path = path.join("/");
    // get this repos info
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // list the commits for this repo
    let cursor = repo
        .commitishes(&user, params, false, &state.shared)
        .await?;
    Ok(Json(cursor))
}

/// List the commit details for a repo
///
/// # Arguments
///
/// * `user` - The user that is listing the commit details for a repo
/// * `params` - The params to use when listing commit details for this repo
/// * `repo` - The repo to list commit details from
/// * `shared` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     get,
//     path = "/api/repos/commitish-details/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path containing this urls args"),
//         ("params" = CommitishListParams, description = "The params to use when listing commit details for this repo"),
//     ),
//     responses(
//         (status = 200, description = "Commit details for repo", body = ApiCursor<CommitishDetails>),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::commitish_details", skip_all, err(Debug))]
async fn commitish_details(
    user: User,
    params: CommitishListParams,
    Path(repo_path): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<CommitishDetails>>, ApiError> {
    // get this repos info
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // list the next commits for this repo in our cursor
    let cursor = repo.commitishes(&user, params, true, &state.shared).await?;
    // get the details for each of the commits on this page of the cursor
    let cursor = cursor.details(&user, &repo.url, &state.shared).await?;
    Ok(Json(cursor))
}

/// Downloads a repo
///
/// # Arguments
///
/// * `user` - The user that is uploading sample
/// * `repo` - The repo to get info about
/// * `shared` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     get,
//     path = "/api/repos/download/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The path containing this urls args"),
//         ("params" = RepoDownloadOpts, description = "The query params to use with this request"),
//     ),
//     responses(
//         (status = 200, description = "Bytestream for repo download", body = Vec<u8>),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::download", skip_all, err(Debug))]
async fn download(
    user: User,
    Path(repo_path): Path<String>,
    params: RepoDownloadOpts,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // get this repos info
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // download this repos data
    let stream = repo
        .download(&params.kinds, params.commitish, &state.shared)
        .await?;
    // convert our byte stream to a streamable body
    let body = AsyncReadBody::new(stream.into_async_read());
    Ok(body)
}

/// Lists repos by submission date
///
/// # Arguments
///
/// * `user` - The user that is listing repos
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/repos/",
    params(
        ("params" = RepoListParams, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "List of repos by submission date", body = ApiCursor<RepoListLine>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[axum_macros::debug_handler]
#[instrument(name = "routes::repos::list", skip_all, err(Debug))]
async fn list(
    user: User,
    params: RepoListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<RepoListLine>>, ApiError> {
    // get a list of all samples in these groups
    let cursor = Repo::list(&user, params, &state.shared).await?;
    Ok(Json(cursor))
}

/// Lists repo data by submission date with details
///
/// # Arguments
///
/// * `user` - The user that is listing repos with details
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/repos/details/",
    params(
        ("params" = RepoListParams, description = "The query params to use for this request"),
    ),
    responses(
        (status = 200, description = "List of repo data by submission date with details", body = ApiCursor<Repo>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::repos::list_details", skip_all, err(Debug))]
async fn list_details(
    user: User,
    params: RepoListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<Repo>>, ApiError> {
    // get a list of all repos in these groups
    let list = Repo::list(&user, params, &state.shared).await?;
    // convert our list to a details list
    let cursor = list.details(&user, &state.shared).await?;
    Ok(Json(cursor))
}

/// Allow users to upload results for repos to Thorium
///
/// # Arguments
///
/// * `user` - The user submitting these results
/// * `path` - The repo path derived from the URL path
/// * `state` - Shared Thorium objects
/// * `upload` - The results being submitted
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     post,
//     path = "/api/repos/results/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The repo path derived from the URL path"),
//         ("upload" = Multipart, description = "The query params to use with this request"),
//     ),
//     responses(
//         (status = 200, description = "Results uploaded", body = OutputResponse),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::upload_results", skip_all, err(Debug))]
async fn upload_results(
    user: User,
    Path(repo_path): Path<String>,
    State(state): State<AppState>,
    upload: Multipart,
) -> Result<Json<OutputResponse>, ApiError> {
    // get our repo
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // build an empty form to stream results metadata into
    let form = OutputFormBuilder::<Repo>::default();
    // save these new results
    let result_id = form
        .create_results(&user, repo_path, &repo, upload, &state.shared)
        .await?;
    Ok(Json(OutputResponse { id: result_id }))
}

/// Get results for a specific repo
///
/// # Arguments
///
/// * `user` - The user getting these results
/// * `path` - The repo path derived from the URL path
/// * `params` - The query params to use with this request
/// * `state` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     get,
//     path = "/api/repos/results/*repo_path",
//     params(
//         ("path" = Vec<String>, Path, description = "The repo path derived from the URL path"),
//         ("params" = ResultGetParams, description = "The query params to use with this request"),
//     ),
//     responses(
//         (status = 200, description = "Repo results", body = OutputMap),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::get_results", skip_all, err(Debug))]
async fn get_results(
    user: User,
    Path(repo_path): Path<String>,
    params: ResultGetParams,
    State(state): State<AppState>,
) -> Result<Json<OutputMap>, ApiError> {
    // get our repo
    let repo = Repo::get(&user, &repo_path, &state.shared).await?;
    // get the results for this repo
    let mut outputs = OutputMap::get(&repo_path, &repo, &user, params, &state.shared).await?;
    // trim our results to the max retained amount
    outputs.limit(state.shared.config.thorium.retention.results);
    Ok(Json(outputs))
}

/// Downloads a files results file from s3
///
/// # Arguments
///
/// * `user` - The user submitting these results
/// * `path_params` - All params in this url path
/// * `state` - Shared Thorium objects
// TODO_UTOIPA: WILDCARD
// #[utoipa::path(
//     get,
//     path = "/api/repos/result-files/:tool/:result_id/*repo_path",
//     params(
//         ("path_params" = Vec<String>, Path, description = "All params in this url path"),
//     ),
//     responses(
//         (status = 200, description = "Result file bytestream", body = Vec<u8>),
//         (status = 401, description = "This user is not authorized to access this route"),
//         (status = 404, description = "Result file now found"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::repos::download_result_file", skip_all, err(Debug))]
async fn download_result_file(
    user: User,
    Path(path_params): Path<String>,
    params: ResultFileDownloadParams,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // split the path on '/'
    let mut path_split: Vec<&str> = path_params.split('/').collect();
    // if we have less than 3 path params then return a 404
    if path_split.len() < 3 {
        return Err(ApiError::new(StatusCode::NOT_FOUND, None));
    }
    // pop the required params
    if let Some(raw_uuid) = path_split.pop() {
        let result_id = bounder::uuid(raw_uuid, "result ID")?;
        if let Some(tool) = path_split.pop() {
            // build our repo path from what's left
            let repo_path = itertools::join(path_split.iter(), "/");
            // start streaming a results file from s3
            let stream = Output::download(
                OutputKind::Repos,
                &user,
                &repo_path,
                tool,
                &result_id,
                params.result_file,
                &state.shared,
            )
            .await?;
            // convert our byte stream to a streamable body
            let body = AsyncReadBody::new(stream.into_async_read());
            return Ok(body);
        }
    }
    Err(ApiError::new(StatusCode::NOT_FOUND, None))
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    // TODO_UTOIPA: WILDCARD add these back in once all the wildcard issues are resolved
    // paths(list, create, list_details, get_repo, upload, commitishes, update_commitishes, commitish_details, download, tag, delete_tags, get_results, upload_results, download_result_file, bundle_results),
    paths(list, create, list_details),
    components(schemas(ApiCursor<Repo>, ApiCursor<RepoListLine>, Branch, BranchDetails, BranchRequest, Commit, CommitDetails, Commitish, CommitishDetails, CommitishKinds, CommitishMapRequest, CommitishRequest, CommitRequest, GitTag, GitTagDetails, GitTagRequest, OutputMap, OutputResponse, Repo, RepoCheckout, RepoCreateResponse, RepoDownloadOpts, RepoListParams, RepoDataUploadResponse, RepoRequest, RepoScheme, RepoSubmissionChunk, ResultGetParams, TagDeleteRequest<Repo>, TagRequest<Repo>)),
    modifiers(&OpenApiSecurity),
)]
pub struct RepoApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(RepoApiDocs::openapi())
}

/// Add the file routes to our router
///
/// # Arguments
///
/// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/repos/", get(list).post(create))
        .route("/api/repos/details/", get(list_details))
        .route("/api/repos/data/{*repo_path}", get(get_repo).post(upload))
        .route(
            "/api/repos/commitishes/{data}/{*repo_path}",
            get(commitishes).post(update_commitishes),
        )
        .route(
            "/api/repos/commitish-details/{*repo_path}",
            get(commitish_details),
        )
        .route("/api/repos/download/{*repo_path}", get(download))
        .route(
            "/api/repos/tags/{*repo_path}",
            post(tag).delete(delete_tags),
        )
        .route(
            "/api/repos/results/{*repo_path}",
            get(get_results).post(upload_results),
        )
        .route(
            "/api/repos/result-files/{*repo_path}",
            get(download_result_file),
        )
}
