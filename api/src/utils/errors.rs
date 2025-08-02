//! The error class for the Thorium API

use aws_sdk_s3::error::SdkError;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::fmt;
use tracing::{event, span, Level};
use utoipa::ToSchema;

use crate::models::conversions::ConversionError;
use crate::models::InvalidEnum;
use crate::utils::trace;

/// Builds an error http response
#[derive(Debug, ToSchema, Serialize)]
pub struct ApiError {
    /// The status code to return
    #[serde(skip)]
    pub code: StatusCode,
    /// The error message to return
    pub msg: Option<String>,
}

impl ApiError {
    /// creates a new error object
    ///
    /// # Arguments
    ///
    /// * `code` - status of error response
    /// * `msg` - message to put in the response
    #[must_use]
    pub fn new(code: StatusCode, msg: Option<String>) -> ApiError {
        ApiError { code, msg }
    }
}

impl IntoResponse for ApiError {
    /// Allow Axum to build a response from error messages
    fn into_response(self) -> Response {
        // get our trace id
        let trace = trace::get_trace();
        // check if we have an error message or not
        match self.msg {
            // we have a message so build our error response
            Some(msg) => {
                // log this error msg
                let span = span!(Level::ERROR, "Error Message");
                event!(parent: &span, Level::ERROR, msg = &msg,);
                // wrap our message in a json object with a trace id if we have one
                let err_json = match trace {
                    Some(trace) => Json(serde_json::json!({ "error": msg, "trace": &trace })),
                    None => Json(serde_json::json!({ "error": msg })),
                };
                (self.code, err_json).into_response()
            }
            // we do not have an error message so just return the trace
            None => match trace {
                // we have a trace so return that trace id
                Some(trace) => {
                    let body = Json(serde_json::json!({ "trace": &trace }));
                    (self.code, body).into_response()
                }
                // we do not have a trace so just return an empty body
                None => self.code.into_response(),
            },
        }
    }
}

/// 400 bad request
#[macro_export]
macro_rules! bad {
    ($($msg:tt)+) => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::BAD_REQUEST, Some($($msg)+)))}
}

/// 409 conflict
#[macro_export]
macro_rules! conflict {
    ($($msg:tt)+) => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::CONFLICT, Some($($msg)+)))}
}

/// 404 not found
#[macro_export]
macro_rules! not_found {
    ($($msg:tt)+) => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::NOT_FOUND, Some($($msg)+)))}
}

/// 204 no content
#[macro_export]
macro_rules! no_content {
    () => {
        Err($crate::utils::ApiError::new(
            axum::http::status::StatusCode::NoContent,
            None,
        ))
    };
}

/// 304 not modified
#[macro_export]
macro_rules! not_modified {
    ($($msg:tt)+) => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::NOT_MODIFIED, Some($($msg)+)))}
}

/// 500 internal server error
#[macro_export]
macro_rules! internal_err {
    () => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::INTERNAL_SERVER_ERROR, None))};
    ($($msg:tt)+) => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::INTERNAL_SERVER_ERROR, Some($($msg)+)))}
}

/// 500 internal server error without the Err wrap
#[macro_export]
macro_rules! internal_err_unwrapped {
    () => {$crate::utils::ApiError::new(axum::http::status::StatusCode::INTERNAL_SERVER_ERROR, None)};
    ($($msg:tt)+) => {$crate::utils::ApiError::new(axum::http::status::StatusCode::INTERNAL_SERVER_ERROR, Some($($msg)+))}
}

/// 503 service unavailable
#[macro_export]
macro_rules! unavailable {
    ($($msg:tt)+) => {Err($crate::utils::ApiError::new(axum::http::status::StatusCode::SERVICE_UNAVAILABLE, Some($($msg)+)))}
}

/// 401 unauthorized
#[macro_export]
macro_rules! unauthorized {
    () => {
        Err($crate::utils::ApiError::new(
            axum::http::status::StatusCode::UNAUTHORIZED,
            None,
        ))
    };
    ($msg:expr) => {
        Err($crate::utils::ApiError::new(
            axum::http::status::StatusCode::UNAUTHORIZED,
            Some($msg),
        ))
    };
}

/// 400 bad request without the Err wrap
#[macro_export]
macro_rules! bad_internal {
    ($($msg:tt)+) => {$crate::utils::ApiError::new(axum::http::status::StatusCode::BAD_REQUEST, Some($($msg)+))}
}

impl fmt::Display for ApiError {
    /// Cast this error to either a string based on the message or the code
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter that is being used
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.msg {
            Some(msg) => write!(f, "{msg}"),
            // if we have a status code then return that and the reason if one exists
            None => write!(f, "code {} - {}", self.code.as_u16(), self.code),
        }
    }
}

impl From<InvalidEnum> for ApiError {
    fn from(error: InvalidEnum) -> Self {
        bad_internal!(error.inner())
    }
}

impl From<ConversionError> for ApiError {
    fn from(error: ConversionError) -> Self {
        bad_internal!(format!("Unable to convert value {:#?}", error))
    }
}

impl From<uuid::Error> for ApiError {
    fn from(error: uuid::Error) -> Self {
        bad_internal!(format!("Failed cast to Uuid {:#?}", error))
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(error: serde_json::Error) -> Self {
        bad_internal!(format!("Failed cast JsonValue to String {:#?}", error))
    }
}

impl From<std::num::ParseIntError> for ApiError {
    fn from(error: std::num::ParseIntError) -> Self {
        bad_internal!(format!("Failed cast to int {:#?}", error))
    }
}

impl From<std::num::ParseFloatError> for ApiError {
    fn from(error: std::num::ParseFloatError) -> Self {
        bad_internal!(format!("Failed cast to float {:#?}", error))
    }
}

impl From<std::str::ParseBoolError> for ApiError {
    fn from(error: std::str::ParseBoolError) -> Self {
        bad_internal!(format!("Failed cast to bool {:#?}", error))
    }
}

impl From<std::io::Error> for ApiError {
    fn from(error: std::io::Error) -> Self {
        bad_internal!(format!("IO Error {:#?}", error))
    }
}

impl From<chrono::format::ParseError> for ApiError {
    fn from(error: chrono::format::ParseError) -> Self {
        bad_internal!(format!("Failed to parse timestamp {:#?}", error))
    }
}

impl From<base64::DecodeError> for ApiError {
    fn from(error: base64::DecodeError) -> Self {
        bad_internal!(format!("Failed to base64 decode string {:#?}", error))
    }
}

impl From<std::str::Utf8Error> for ApiError {
    fn from(error: std::str::Utf8Error) -> Self {
        bad_internal!(format!("Failed to cast str to Utf8 {:#?}", error))
    }
}

impl From<argon2::Error> for ApiError {
    fn from(error: argon2::Error) -> Self {
        bad_internal!(format!("Argon2 error {:#?}", error))
    }
}

impl From<argon2::password_hash::Error> for ApiError {
    fn from(error: argon2::password_hash::Error) -> Self {
        bad_internal!(format!("Argon2 password error {:#?}", error))
    }
}

impl From<ldap3::result::LdapError> for ApiError {
    fn from(error: ldap3::result::LdapError) -> Self {
        bad_internal!(format!("ldap error {:#?}", error))
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        bad_internal!(format!("s3 error {:#?}", error))
    }
}

impl From<cart_rs::Error> for ApiError {
    fn from(error: cart_rs::Error) -> Self {
        bad_internal!(format!("Cart error {:#?}", error))
    }
}

impl From<url::ParseError> for ApiError {
    fn from(error: url::ParseError) -> Self {
        bad_internal!(format!("URL parse error {:#?}", error))
    }
}

impl From<std::num::TryFromIntError> for ApiError {
    fn from(error: std::num::TryFromIntError) -> Self {
        bad_internal!(format!("Int casting error {:#?}", error))
    }
}

impl From<elasticsearch::Error> for ApiError {
    fn from(error: elasticsearch::Error) -> Self {
        bad_internal!(format!("Elasticsearch error {:#?}", error))
    }
}

impl From<elasticsearch::http::response::Error> for ApiError {
    fn from(error: elasticsearch::http::response::Error) -> Self {
        bad_internal!(format!("Elasticsearch error {:#?}", error))
    }
}

impl From<scylla::response::query_result::IntoRowsResultError> for ApiError {
    fn from(error: scylla::response::query_result::IntoRowsResultError) -> Self {
        bad_internal!(format!("Scylla into rows error {error:#?}"))
    }
}

impl From<scylla::response::query_result::RowsError> for ApiError {
    fn from(error: scylla::response::query_result::RowsError) -> Self {
        bad_internal!(format!("Scylla rows error {error:#?}"))
    }
}

impl From<scylla::deserialize::DeserializationError> for ApiError {
    fn from(error: scylla::deserialize::DeserializationError) -> Self {
        bad_internal!(format!("Scylla deserialization error {error:#?}"))
    }
}

impl From<scylla::response::query_result::MaybeFirstRowError> for ApiError {
    fn from(error: scylla::response::query_result::MaybeFirstRowError) -> Self {
        bad_internal!(format!("Scylla maybe first row error {error:#?}"))
    }
}

impl From<scylla::client::pager::NextRowError> for ApiError {
    fn from(error: scylla::client::pager::NextRowError) -> Self {
        bad_internal!(format!("Scylla next row error {:#?}", error))
    }
}

impl From<scylla::errors::PagerExecutionError> for ApiError {
    fn from(error: scylla::errors::PagerExecutionError) -> Self {
        bad_internal!(format!("Scylla pager execution error {:#?}", error))
    }
}

impl From<scylla::deserialize::TypeCheckError> for ApiError {
    fn from(error: scylla::deserialize::TypeCheckError) -> Self {
        bad_internal!(format!("Scylla type check error {:#?}", error))
    }
}

impl From<axum::extract::multipart::MultipartError> for ApiError {
    fn from(error: axum::extract::multipart::MultipartError) -> Self {
        bad_internal!(format!("Failed to extract multipart form {:#?}", error))
    }
}

impl From<serde_qs::Error> for ApiError {
    fn from(error: serde_qs::Error) -> Self {
        bad_internal!(format!("Failed to deserialize query params {:#?}", error))
    }
}

impl From<semver::Error> for ApiError {
    fn from(error: semver::Error) -> Self {
        bad_internal!(format!("Invalid semver version: {:#?}", error))
    }
}

impl From<aws_sdk_s3::operation::head_object::HeadObjectError> for ApiError {
    fn from(error: aws_sdk_s3::operation::head_object::HeadObjectError) -> Self {
        bad_internal!(format!("Failed to check if an object exists {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::head_object::HeadObjectError>> for ApiError {
    fn from(error: SdkError<aws_sdk_s3::operation::head_object::HeadObjectError>) -> Self {
        bad_internal!(format!("Failed to check if an object exists {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::create_multipart_upload::CreateMultipartUploadError>>
    for ApiError
{
    fn from(
        error: SdkError<aws_sdk_s3::operation::create_multipart_upload::CreateMultipartUploadError>,
    ) -> Self {
        bad_internal!(format!("Failed to create multipart upload {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::upload_part::UploadPartError>> for ApiError {
    fn from(error: SdkError<aws_sdk_s3::operation::upload_part::UploadPartError>) -> Self {
        bad_internal!(format!("Failed to upload multipart chunk {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::complete_multipart_upload::CompleteMultipartUploadError>>
    for ApiError
{
    fn from(
        error: SdkError<
            aws_sdk_s3::operation::complete_multipart_upload::CompleteMultipartUploadError,
        >,
    ) -> Self {
        bad_internal!(format!("Failed to complete multipart upload {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::abort_multipart_upload::AbortMultipartUploadError>>
    for ApiError
{
    fn from(
        error: SdkError<aws_sdk_s3::operation::abort_multipart_upload::AbortMultipartUploadError>,
    ) -> Self {
        bad_internal!(format!("Failed to abort multipart upload {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::put_object::PutObjectError>> for ApiError {
    fn from(error: SdkError<aws_sdk_s3::operation::put_object::PutObjectError>) -> Self {
        bad_internal!(format!("Failed to upload object to s3 {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::get_object::GetObjectError>> for ApiError {
    fn from(error: SdkError<aws_sdk_s3::operation::get_object::GetObjectError>) -> Self {
        bad_internal!(format!("Failed to get object from s3 {:#?}", error))
    }
}

impl From<SdkError<aws_sdk_s3::operation::delete_object::DeleteObjectError>> for ApiError {
    fn from(error: SdkError<aws_sdk_s3::operation::delete_object::DeleteObjectError>) -> Self {
        bad_internal!(format!("Failed to delete object from s3 {:#?}", error))
    }
}

impl From<tokio::task::JoinError> for ApiError {
    fn from(error: tokio::task::JoinError) -> Self {
        bad_internal!(format!("Tokio task failed to join: {:#?}", error))
    }
}

impl From<zip::result::ZipError> for ApiError {
    fn from(error: zip::result::ZipError) -> Self {
        bad_internal!(format!("Failed to zip file: {:#?}", error))
    }
}

impl From<std::net::AddrParseError> for ApiError {
    fn from(error: std::net::AddrParseError) -> Self {
        bad_internal!(format!("Error parsing IP address: {error}"))
    }
}
