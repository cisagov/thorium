//! The errors that may occur during backup or restore

use bytecheck::{SliceCheckError, StructCheckError};
use rkyv::validation::{
    owned::OwnedPointerError, validators::DefaultValidatorError, CheckArchiveError,
};
use std::convert::Infallible;

#[derive(Debug)]
pub enum Error {
    /// A generic backup error
    Generic(String),
    /// A Thorium API error
    Thorium(thorium::Error),
    /// A Scylla new session error occurred
    ScyllaNewSession(scylla::errors::NewSessionError),
    /// An error preparing a scylla statement
    ScyllaPrepareSatement(scylla::errors::PrepareError),
    /// A Scylla query error occurred
    ScyllaQuery(scylla::errors::ExecutionError),
    /// A Scylla paged query error occurred
    ScyllaPagedQuery(scylla::errors::PagerExecutionError),
    /// A Scylla next row error occurred
    ScyllaNextRow(scylla::client::pager::NextRowError),
    /// A Redis error
    Redis(redis::RedisError),
    /// A tokio join error
    TokioJoin(tokio::task::JoinError),
    /// A kanal send error
    KanalSend(kanal::SendError),
    /// A kanal close error
    KanalClose(kanal::CloseError),
    /// An IO Error
    IO(std::io::Error),
    /// A config error
    Config(config::ConfigError),
    /// A deserialization/conversion error
    Conversion(std::convert::Infallible),
    /// An error from converting a type to a Uuid
    Uuid(uuid::Error),
    /// An error from converting a value with `serde_json`
    SerdeJson(serde_json::Error),
    /// An error from converting a value with `serde_yaml`
    SerdeYaml(serde_yaml::Error),
    /// An s3 error
    S3 {
        code: Option<String>,
        message: Option<String>,
    },
    /// An s3 bytestream error
    S3ByteStream(aws_sdk_s3::primitives::ByteStreamError),
    /// Failed to receive data from a kanal channel
    KanalRecv(kanal::ReceiveError),
    /// An error with deserializing rkyv data
    RkyvDeserialize(String),
    /// An error from stripping a prefix from a path
    StripPrefix(std::path::StripPrefixError),
    // An error from dialoguer
    Dialoguer(dialoguer::Error),
}

impl Error {
    /// Create a new generic error
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to use
    pub fn new<T: Into<String>>(msg: T) -> Self {
        Error::Generic(msg.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Generic(err) => write!(f, "{err}"),
            Error::Thorium(err) => write!(f, "Thorium Client Error: {err}"),
            Error::ScyllaNewSession(err) => write!(f, "ScyllaNewSession Error: {err}"),
            Error::ScyllaPrepareSatement(err) => write!(f, "ScyllaPreparedSatement Error: {err}"),
            Error::ScyllaQuery(err) => write!(f, "ScyllaQuery Error: {err}"),
            Error::ScyllaPagedQuery(err) => write!(f, "ScyllaPagedQuery Error: {err}"),
            Error::ScyllaNextRow(err) => write!(f, "ScyllaNextRow Error: {err}"),
            Error::Redis(err) => write!(f, "Redis Error: {err}"),
            Error::TokioJoin(err) => write!(f, "TokioJoin Error: {err}"),
            Error::KanalSend(err) => write!(f, "KanalSend Error: {err}"),
            Error::KanalClose(err) => write!(f, "KanalClose Error: {err}"),
            Error::IO(err) => write!(f, "IO Error: {err}"),
            Error::Config(err) => write!(f, "Config Error: {err}"),
            Error::Conversion(err) => write!(f, "Conversion Error: {err}"),
            Error::Uuid(err) => write!(f, "Uuid Error: {err}"),
            Error::SerdeJson(err) => write!(f, "SerdeJson Error: {err}"),
            Error::SerdeYaml(err) => write!(f, "SerdeYaml Error: {err}"),
            Error::S3 { code, message } => {
                write!(
                    f,
                    "S3 Error {}: {}",
                    code.clone().unwrap_or_default(),
                    message.clone().unwrap_or_default()
                )
            }
            Error::S3ByteStream(err) => write!(f, "S3ByteStream Error: {err}"),
            Error::KanalRecv(err) => write!(f, "KanalRecv Error: {err}"),
            Error::RkyvDeserialize(err) => write!(f, "RkyvDeserialize Error: {err}"),
            Error::StripPrefix(err) => write!(f, "StripPrefix Error: {err}"),
            Error::Dialoguer(err) => write!(f, "Dialoguer: {err}"),
        }
    }
}

impl From<thorium::Error> for Error {
    fn from(error: thorium::Error) -> Self {
        Error::Thorium(error)
    }
}

impl From<scylla::errors::NewSessionError> for Error {
    fn from(error: scylla::errors::NewSessionError) -> Self {
        Error::ScyllaNewSession(error)
    }
}

impl From<scylla::errors::PrepareError> for Error {
    fn from(error: scylla::errors::PrepareError) -> Self {
        Error::ScyllaPrepareSatement(error)
    }
}

impl From<scylla::errors::ExecutionError> for Error {
    fn from(error: scylla::errors::ExecutionError) -> Self {
        Error::ScyllaQuery(error)
    }
}

impl From<scylla::errors::PagerExecutionError> for Error {
    fn from(error: scylla::errors::PagerExecutionError) -> Self {
        Error::ScyllaPagedQuery(error)
    }
}

impl From<scylla::client::pager::NextRowError> for Error {
    fn from(error: scylla::client::pager::NextRowError) -> Self {
        Error::ScyllaNextRow(error)
    }
}

impl From<redis::RedisError> for Error {
    fn from(error: redis::RedisError) -> Self {
        Error::Redis(error)
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(error: tokio::task::JoinError) -> Self {
        Error::TokioJoin(error)
    }
}

impl From<kanal::SendError> for Error {
    fn from(error: kanal::SendError) -> Self {
        Error::KanalSend(error)
    }
}

impl From<kanal::CloseError> for Error {
    fn from(error: kanal::CloseError) -> Self {
        Error::KanalClose(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IO(error)
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        Error::Config(error)
    }
}

impl From<std::convert::Infallible> for Error {
    fn from(error: std::convert::Infallible) -> Self {
        Error::Conversion(error)
    }
}

impl From<uuid::Error> for Error {
    fn from(error: uuid::Error) -> Self {
        Error::Uuid(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::SerdeJson(error)
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(error: serde_yaml::Error) -> Self {
        Error::SerdeYaml(error)
    }
}

impl
    From<
        aws_sdk_s3::error::SdkError<
            aws_sdk_s3::operation::create_multipart_upload::CreateMultipartUploadError,
        >,
    > for Error
{
    fn from(
        error: aws_sdk_s3::error::SdkError<
            aws_sdk_s3::operation::create_multipart_upload::CreateMultipartUploadError,
        >,
    ) -> Self {
        // cast this error into a service error
        let service_error = error.into_service_error();
        // get this errors metadata
        let meta = service_error.meta();
        Error::S3 {
            code: meta.code().map(ToOwned::to_owned),
            message: meta.message().map(ToOwned::to_owned),
        }
    }
}

impl From<aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::upload_part::UploadPartError>>
    for Error
{
    fn from(
        error: aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::upload_part::UploadPartError>,
    ) -> Self {
        // cast this error into a service error
        let service_error = error.into_service_error();
        // get this errors metadata
        let meta = service_error.meta();
        Error::S3 {
            code: meta.code().map(ToOwned::to_owned),
            message: meta.message().map(ToOwned::to_owned),
        }
    }
}

impl
    From<
        aws_sdk_s3::error::SdkError<
            aws_sdk_s3::operation::complete_multipart_upload::CompleteMultipartUploadError,
        >,
    > for Error
{
    fn from(
        error: aws_sdk_s3::error::SdkError<
            aws_sdk_s3::operation::complete_multipart_upload::CompleteMultipartUploadError,
        >,
    ) -> Self {
        // cast this error into a service error
        let service_error = error.into_service_error();
        // get this errors metadata
        let meta = service_error.meta();
        Error::S3 {
            code: meta.code().map(ToOwned::to_owned),
            message: meta.message().map(ToOwned::to_owned),
        }
    }
}

impl
    From<
        aws_sdk_s3::error::SdkError<
            aws_sdk_s3::operation::abort_multipart_upload::AbortMultipartUploadError,
        >,
    > for Error
{
    fn from(
        error: aws_sdk_s3::error::SdkError<
            aws_sdk_s3::operation::abort_multipart_upload::AbortMultipartUploadError,
        >,
    ) -> Self {
        // cast this error into a service error
        let service_error = error.into_service_error();
        // get this errors metadata
        let meta = service_error.meta();
        Error::S3 {
            code: meta.code().map(ToOwned::to_owned),
            message: meta.message().map(ToOwned::to_owned),
        }
    }
}

impl From<aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>>
    for Error
{
    fn from(
        error: aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>,
    ) -> Self {
        // cast this error into a service error
        let service_error = error.into_service_error();
        // get this errors metadata
        let meta = service_error.meta();
        Error::S3 {
            code: meta.code().map(ToOwned::to_owned),
            message: meta.message().map(ToOwned::to_owned),
        }
    }
}

impl From<aws_sdk_s3::primitives::ByteStreamError> for Error {
    fn from(error: aws_sdk_s3::primitives::ByteStreamError) -> Self {
        Error::S3ByteStream(error)
    }
}

impl From<kanal::ReceiveError> for Error {
    fn from(error: kanal::ReceiveError) -> Self {
        Error::KanalRecv(error)
    }
}

impl
    From<
        CheckArchiveError<
            OwnedPointerError<Infallible, SliceCheckError<StructCheckError>, DefaultValidatorError>,
            DefaultValidatorError,
        >,
    > for Error
{
    fn from(
        error: CheckArchiveError<
            OwnedPointerError<Infallible, SliceCheckError<StructCheckError>, DefaultValidatorError>,
            DefaultValidatorError,
        >,
    ) -> Self {
        Error::RkyvDeserialize(error.to_string())
    }
}

impl From<CheckArchiveError<StructCheckError, DefaultValidatorError>> for Error {
    fn from(error: CheckArchiveError<StructCheckError, DefaultValidatorError>) -> Self {
        Error::RkyvDeserialize(error.to_string())
    }
}

impl From<std::path::StripPrefixError> for Error {
    fn from(error: std::path::StripPrefixError) -> Self {
        Error::StripPrefix(error)
    }
}

impl From<dialoguer::Error> for Error {
    fn from(error: dialoguer::Error) -> Self {
        Error::Dialoguer(error)
    }
}
