//! The files related routes for Thorium

use axum::extract::{Json, Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, patch, post};
use axum::Router;
use axum_extra::body::AsyncReadBody;
use tracing::instrument;
use utoipa::OpenApi;
use uuid::Uuid;

use super::OpenApiSecurity;
use crate::models::backends::{CommentSupport, TagSupport};
use crate::models::{
    ApiCursor, CarvedOrigin, Comment, CommentResponse, DeleteCommentParams, DeleteSampleParams,
    FileListParams, ImageVersion, Origin, OriginRequest, Output, OutputDisplayType,
    OutputFormBuilder, OutputHandler, OutputKind, OutputMap, OutputResponse, PcapNetworkProtocol,
    ResultFileDownloadParams, ResultGetParams, Sample, SampleCheck, SampleCheckResponse,
    SampleListLine, SampleSubmissionResponse, SubmissionChunk, SubmissionUpdate, TagDeleteRequest,
    TagRequest, User, ZipDownloadParams,
};
use crate::utils::{ApiError, AppState};

/* TODO_UTOIPA: the '/files/download_result_file/:sha256/:tool/:result_id/\*path'
   route is implemented with a wildcard on the path variable, but
   Utoipa (and maybe OpenAPI?) does not handle this format. If the
   utoipa annotation for this functions is left in, the code will not
   compile because rust marks that function as "unreachable".

   One potential fix for this is to convert all the wildcards currently
   handled as Vec<String> into a single String argument that can be
   split on '/' later.

   The affected function is marked with 'TODO_UTOIPA: WILDCARD'
   and the utoipa annotation is commented out. The annotation
   should be at least somewhat functional if it were able to be used
   but has not been tested due to the aforementioned compilation
   issue.
*/

/// Allow users to upload a file to Thorium
///
/// # Arguments
///
/// * `user` - The user that is uploading sample
/// * `state` - Shared Thorium objects
/// * `multipart` - The multipart form containing the file upload
#[utoipa::path(
    get,
    path = "/api/files/",
    params(
        ("multipart", description = "The multipart form containing the file upload")
    ),
    responses(
        (status = 200, description = "File uploaded to Thorium", body = SampleSubmissionResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::upload", skip_all, err(Debug))]
async fn upload(
    user: User,
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<SampleSubmissionResponse>, ApiError> {
    // save this file into the backend
    let resp: SampleSubmissionResponse = Sample::create(&user, multipart, &state.shared).await?;
    Ok(Json(resp))
}

/// Get info on a specific sample by sha256
///
/// # Arguments
///
/// * `user` - The user that is getting info for a specific sha256
/// * `sha256` - The sha256 to get info about
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/sample/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 of the sample to get info on")
    ),
    responses(
        (status = 200, description = "Return a sample in Thorium", body = Sample),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::get_sample", skip_all, err(Debug))]
async fn get_sample(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Sample>, ApiError> {
    // try to get info on the sample
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    Ok(Json(sample))
}

/// Checks if a sample already exists with this submission info
///
/// # Arguments
///
/// * `user` - The user that is checking whether a sample exists or not
/// * `check` - The info to use when checking this file exists
/// * `state` - Shared Thorium objects
#[utoipa::path(
    post,
    path = "/api/files/exists",
    params(
        ("check" = SampleCheck, description = "Sha256 and group of the sample for which to check existence, optionally include sample name and origin information")
    ),
    responses(
        (status = 200, description = "Check if a submission has already been created", body = SampleCheckResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::exists", skip_all, err(Debug))]
async fn exists(
    user: User,
    State(state): State<AppState>,
    Json(check): Json<SampleCheck>,
) -> Result<Json<SampleCheckResponse>, ApiError> {
    // try to get info on the sample
    let resp = Sample::exists(&user, &check, &state.shared).await?;
    Ok(Json(resp))
}

/// Download a file by sha256
///
/// # Arguments
///
/// * `user` - The user that is downloading this file
/// * `sha256` - The sha256 to download
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/sample/:sha256/download",
    params(
        ("sha256" = String, Path, description = "Sha256 of file to download")
    ),
    responses(
        (status = 200, description = "Download a file by sha256", body = Vec<u8>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::download", skip_all, err(Debug))]
async fn download(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // check if we have access to this sample and download it if we do
    let stream = Sample::download(&user, sha256, &state.shared).await?;
    // convert our byte stream to a streamable body
    let body = AsyncReadBody::new(stream.into_async_read());
    Ok(body)
}

/// Download a file by sha2566 as an encrypted zip
///
/// # Arguments
///
/// * `user` - The user that is downloading this file
/// * `sha256` - The sha256 to download
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/sample/:sha256/download/zip",
    params(
        ("sha256" = String, Path, description = "Sha256 of file to download"),
        ("params" = ZipDownloadParams, description = "Optional password to encrypt the ZIP download")
    ),
    responses(
        (status = 200, description = "Download a file by sha2566 as an encrypted zip", body = Vec<u8>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::download_as_zip", skip_all, err(Debug))]
async fn download_as_zip(
    user: User,
    params: ZipDownloadParams,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // check if we have access to this sample and download it if we do
    Sample::download_as_zip(&user, sha256, params, &state.shared).await
}

/// Updates a submission for a specific sample
///
/// # Arguments
///
/// * `user` - The user that is updating this file
/// * `sha256` - The sha256 to update
/// * `state` - Shared Thorium objects
/// * `update` - The update to apply to this submission
#[utoipa::path(
    patch,
    path = "/api/files/sample/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 of sample to update"),
        ("update" = SubmissionUpdate, description = "JSON-formatted update to apply to this submission")
    ),
    responses(
        (status = 204, description = "Sample updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::update", skip_all, err(Debug))]
async fn update(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
    Json(update): Json<SubmissionUpdate>,
) -> Result<StatusCode, ApiError> {
    // try to get info on the sample
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // update this samples info
    sample.update(&user, update, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes a sample submission
///
/// # Arguments
///
/// * `user` - The user that is getting info for a specific sha256
/// * `sha256` - The sha256 to delete a submission from
/// * `submission` - The submission to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/files/sample/:sha256/:submission",
    params(
        ("sha256" = String, Path, description = "Sha256 to delete a submission from"),
        ("submission" = Uuid, Path, description = "Uuid of the sample submission to delete"),
        ("DeleteSampleParams" = DeleteSampleParams, description = "Group of the sample to delete")
    ),
    responses(
        (status = 204, description = "Sample deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::delete_sample", skip_all, err(Debug))]
async fn delete_sample(
    user: User,
    params: DeleteSampleParams,
    Path((sha256, submission)): Path<(String, Uuid)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // try to get info on the sample
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // delete the target submission
    sample
        .delete(&user, &submission, &params.groups, &state.shared)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Adds new tags to a sample
///
/// # Arguments
///
/// * `user` - The user that is adding new tags
/// * `sha256` - The sample to add a tag too
/// * `state` - Shared Thorium objects
/// * `tags` - The new tags to apply
#[utoipa::path(
    post,
    path = "/api/files/tags/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 for which to add tags"),
        ("tags" = TagRequest<Sample>, description = "JSON-formatted tags to apply to sample")
    ),
    responses(
        (status = 204, description = "Sample tags updated"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::tag", skip_all, err(Debug))]
async fn tag(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
    Json(tags): Json<TagRequest<Sample>>,
) -> Result<StatusCode, ApiError> {
    // get the sample we are adding tags too
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // try to add the new tags for this sample
    sample.tag(&user, tags, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Deletes tags from a sample
///
/// # Arguments
///
/// * `user` - The user that is deleting tags
/// * `sha256` - The sample to delete tags from
/// * `state` - Shared Thorium objects
/// * `tags_del` - The tags to delete and the groups to delete them from
#[utoipa::path(
    delete,
    path = "/api/files/tags/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 for which to delete tags"),
        ("tags_del" = TagDeleteRequest<Sample>, description = "JSON-formatted tags to delete")
    ),
    responses(
        (status = 204, description = "Sample tags deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::delete_tags", skip_all, err(Debug))]
async fn delete_tags(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
    Json(tags_del): Json<TagDeleteRequest<Sample>>,
) -> Result<StatusCode, ApiError> {
    // get the sample we are deleting tags from
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // try to delete the tags for this sample
    sample.delete_tags(&user, tags_del, &state.shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Allow users to comment on a file in Thorium
///
/// # Arguments
///
/// * `user` - The user that is commenting on this sample
/// * `state` - Shared Thorium objects
/// * `multipart` - The multipart form to parse
#[utoipa::path(
    post,
    path = "/api/files/comment/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 for which to create comment"),
        ("multipart", description = "The multipart form to parse to create comment"),
    ),
    responses(
        (status = 200, description = "Comment creation response", body = CommentResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::create_comment", skip_all, err(Debug))]
async fn create_comment(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<CommentResponse>, ApiError> {
    // get the sample we are commenting on
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // save this file into the backend
    let resp: CommentResponse = sample
        .create_comment(&user, multipart, &state.shared)
        .await?;
    Ok(Json(resp))
}

/// Allow users to delete a file's comment
///
/// # Arguments
///
/// * `user` - The user that is deleting the comment
/// * `params` - The url query params to use
/// * `sha256` - The sha256 of the sample to delete a comment from
/// * `id` - The id of the comment to delete
/// * `state` - Shared Thorium objects
#[utoipa::path(
    delete,
    path = "/api/files/comment/:sha256/:id",
    params(
        ("sha256" = String, Path, description = "Sha256 for which to delete comment"),
        ("id" = Uuid, Path, description = "Uuid of the comment to delete"),
        ("params" = DeleteCommentParams, description = "Groups the comment to be deleted is part of")
    ),
    responses(
        (status = 204, description = "Sample comment deleted"),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::delete_comment", skip_all, err(Debug))]
async fn delete_comment(
    user: User,
    params: DeleteCommentParams,
    Path((sha256, id)): Path<(String, Uuid)>,
    State(state): State<AppState>,
) -> Result<StatusCode, ApiError> {
    // get the sample we are deleting the comment from
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // save this file into the backend
    sample
        .delete_comment(&user, &params.groups, &id, &state.shared)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Allow users to download comment attachments
///
/// # Arguments
///
/// * `user` - The user that is downloading this comment attachment
/// * `sha256` - The sample this comment attachment is tied to
/// * `comment` - The comment ID we are downloading an attachment from
/// * `attachment` - The id of the attachment to download
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/comment/download/:sha256/:comment/:name",
    params(
        ("sha256" = String, Path, description = "Sha256 of sample the comment attachment is tied to"),
        ("comment" = Uuid, Path, description = "Uuid of of the comment to download an attachment from"),
        ("attachment" = Uuid, Path, description = "Uuid of the attachment to download")
    ),
    responses(
        (status = 200, description = "Download a file by sha256", body = Vec<u8>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::download_attachment", skip_all, err(Debug))]
async fn download_attachment(
    user: User,
    Path((sha256, comment, attachment)): Path<(String, Uuid, Uuid)>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // get the sample we are downloading an attachment from
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // download this attachment
    let stream = sample
        .download_attachment(&comment, &attachment, &state.shared)
        .await?;
    // convert our byte stream to a streamable body
    let body = AsyncReadBody::new(stream.into_async_read());
    Ok(body)
}

/// Lists sha256's by submission date
///
/// # Arguments
///
/// * `user` - The user that is listing submissions
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/",
    params(
        ("params" = FileListParams, description = "Query params to use for this file list request"),
    ),
    responses(
        (status = 200, description = "JSON-formatted cursor response containing the sha256 of a sample, the samples group information, submission uuid, and upload timestamp", body = ApiCursor<SampleListLine>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::list", skip_all, err(Debug))]
async fn list(
    user: User,
    params: FileListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<SampleListLine>>, ApiError> {
    // get a list of all samples in these groups
    let cursor = Sample::list(&user, params, false, &state.shared).await?;
    Ok(Json(cursor))
}

/// Lists sha256's by submission date with details
///
/// # Arguments
///
/// * `user` - The user that is listing submissions with details
/// * `params` - The query params to use for this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/details",
    params(
        ("params" = FileListParams, description = "Query params to use for this file list request"),
    ),
    responses(
        (status = 200, description = "JSON-formatted cursor response containing the sha256, sha1, and md5 hashes of a sample, plus tags amd submission information", body = ApiCursor<Sample>),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::list_details", skip_all, err(Debug))]
#[axum_macros::debug_handler]
async fn list_details(
    user: User,
    params: FileListParams,
    State(state): State<AppState>,
) -> Result<Json<ApiCursor<Sample>>, ApiError> {
    // get a list of all samples in these groups
    let list = Sample::list(&user, params, true, &state.shared).await?;
    // convert our list to a details list
    let cursor = list.details(&user, &state.shared).await?;
    Ok(Json(cursor))
}

/// Allow users to upload results for files to Thorium
///
/// # Arguments
///
/// * `user` - The user submitting these results
/// * `sha256` - The sha256 to save results for
/// * `state` - Shared Thorium objects
/// * `upload` - The results being submitted
#[utoipa::path(
    post,
    path = "/api/files/results/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 of sample for which to save results"),
        ("upload", description = "The multipart form containing the results upload")    ),
    responses(
        (status = 200, description = "JSON-formatted response containing the uuid of the new result", body = OutputResponse),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::upload_results", skip_all, err(Debug))]
async fn upload_results(
    user: User,
    Path(sha256): Path<String>,
    State(state): State<AppState>,
    upload: Multipart,
) -> Result<Json<OutputResponse>, ApiError> {
    // get this sample from the db
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // build an empty form to stream results metadata into
    let form = OutputFormBuilder::<Sample>::default();
    // save these new results
    let result_id = form
        .create_results(&user, sha256, &sample, upload, &state.shared)
        .await?;
    Ok(Json(OutputResponse { id: result_id }))
}

/// Get results for a specific hash
///
/// # Arguments
///
/// * `user` - The user getting these results
/// * `sha256` - The sample to get results for
/// * `params` - The query params to use with this request
/// * `state` - Shared Thorium objects
#[utoipa::path(
    get,
    path = "/api/files/results/:sha256",
    params(
        ("sha256" = String, Path, description = "Sha256 of sample for which to get results"),
        ("params" = ResultGetParams, description = "Query params to use for this results get request"),
    ),
    responses(
        (status = 200, description = "JSON-formatted response containing tool results", body = OutputMap),
        (status = 401, description = "This user is not authorized to access this route"),
    ),
    security(
        ("basic" = []),
    )
)]
#[instrument(name = "routes::files::get_results", skip_all, err(Debug))]
async fn get_results(
    user: User,
    Path(sha256): Path<String>,
    params: ResultGetParams,
    State(state): State<AppState>,
) -> Result<Json<OutputMap>, ApiError> {
    // get the sample we are getting results for
    let sample = Sample::get(&user, &sha256, &state.shared).await?;
    // get the results for this sample
    let mut outputs = OutputMap::get(&sha256, &sample, &user, params, &state.shared).await?;
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
//     path = "/api/files/results/:sha256/:tool/:result_id/*path",
//     params(
//         ("path_params" = Vec<String>, Path, description = "3 path-formatted parameters containing the sample sha256, tool name, and result uuid")
//     ),
//     responses(
//         (status = 200, description = "Response containing body of requested result file", body = Vec<u8>),
//         (status = 401, description = "This user is not authorized to access this route"),
//     ),
//     security(
//         ("basic" = []),
//     )
// )]
#[instrument(name = "routes::files::download_result_file", skip_all, err(Debug))]
async fn download_result_file(
    user: User,
    Path((sha256, tool, result_id)): Path<(String, String, Uuid)>,
    params: ResultFileDownloadParams,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    // start streaming a results file from s3
    let stream = Output::download(
        OutputKind::Files,
        &user,
        &sha256,
        &tool,
        &result_id,
        params.result_file,
        &state.shared,
    )
    .await?;
    // convert our byte stream to a streamable body
    let body = AsyncReadBody::new(stream.into_async_read());
    Ok(body)
}

/// The struct containing our openapi docs
#[derive(OpenApi)]
#[openapi(
    paths(list, upload, list_details, get_sample, delete_sample, exists, download, download_as_zip, /*download_result_file,*/ update, tag, delete_tags, create_comment, delete_comment, download_attachment, get_results, upload_results),
    components(schemas(ApiCursor<Sample>, ApiCursor<SampleListLine>, CarvedOrigin, Comment, CommentResponse, DeleteCommentParams, DeleteSampleParams,FileListParams, ImageVersion, Origin, OriginRequest, Output, OutputDisplayType, OutputHandler, OutputMap, OutputResponse, PcapNetworkProtocol, ResultGetParams, Sample, SampleCheck, SampleCheckResponse, SampleListLine, SampleSubmissionResponse, SubmissionChunk, SubmissionUpdate, TagDeleteRequest<Sample>, TagRequest<Sample>, ZipDownloadParams)),
    modifiers(&OpenApiSecurity),
)]
pub struct FileApiDocs;

/// Return the openapi docs for these routes
#[allow(dead_code)]
async fn openapi() -> Json<utoipa::openapi::OpenApi> {
    Json(FileApiDocs::openapi())
}

/// Add the file routes to our router
///
/// # Arguments
///
/// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/api/files/", get(list).post(upload))
        .route("/api/files/details/", get(list_details))
        .route("/api/files/sample/{sha256}", get(get_sample))
        .route(
            "/api/files/sample/{sha256}/{submission}",
            delete(delete_sample),
        )
        .route("/api/files/exists", post(exists))
        .route("/api/files/sample/{sha256}/download", get(download))
        .route(
            "/api/files/sample/{sha256}/download/zip",
            get(download_as_zip),
        )
        .route("/api/files/sample/{sha256}", patch(update))
        .route("/api/files/tags/{sha256}", post(tag).delete(delete_tags))
        .route("/api/files/comment/{sha256}", post(create_comment))
        .route("/api/files/comment/{sha256}/{id}", delete(delete_comment))
        .route(
            "/api/files/comment/download/{sha256}/{comment}/{name}",
            get(download_attachment),
        )
        .route(
            "/api/files/results/{sha256}",
            get(get_results).post(upload_results),
        )
        .route(
            "/api/files/result-files/{sha256}/{tool}/{result_id}",
            get(download_result_file),
        )
}
