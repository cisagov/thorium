//! An error from the Thorium client
use crate::models::conversions::ConversionError;
use futures::executor::block_on;
use reqwest::StatusCode;

/// An error from the Thorium client
#[derive(Debug)]
pub enum Error {
    /// A Thorium error
    Thorium {
        code: StatusCode,
        msg: Option<String>,
    },
    /// A generic error with a message
    Generic(String),
    /// An error from sending or receiving a request
    Reqwest(reqwest::Error),
    /// An IO Error
    IO(std::io::Error),
    /// An error from parsing a timestamp/date
    ChronoParse(chrono::ParseError),
    /// An error from stripping the prefix from a path
    PrefixStrip(std::path::StripPrefixError),
    /// An error from interacting with a git repo
    Git(git2::Error),
    /// An error from opening a repo with gix
    GitOpen(gix::open::Error),
    /// An error finding a git reference
    GitFindReference(gix::reference::find::existing::Error),
    /// An error from a gix reference iter
    GitReferenceIter(gix::reference::iter::Error),
    /// An error from initing a git reference iter
    GitReferenceIterInit(gix::reference::iter::init::Error),
    /// An error from peeling a git reference
    GitReferencePeel(gix::reference::peel::Error),
    /// An error from finding an existing object in git
    GitFindObject(gix::object::find::existing::Error),
    /// An error from decoding an object in git
    GitDecodeObject(gix::worktree::object::decode::Error),
    /// An error from retrieving info from a git commit
    GitCommit(gix::object::commit::Error),
    /// An error from casting a git object to a target type
    GitObjectTryInto(gix::object::try_into::Error),
    /// An error from parsing a git date
    GitDateParse(gix_date::parse::Error),
    /// An error from converting a type to a Uuid
    Uuid(uuid::Error),
    /// An error from loading a config
    Config(config::ConfigError),
    /// An error from building an elastic client
    BuildElastic(elasticsearch::http::transport::BuildError),
    /// An error from an elastic client
    Elastic(elasticsearch::Error),
    /// An error from converting a value with serde
    Serde(serde_json::Error),
    /// An error from converting a value with serde to YAML
    SerdeYaml(serde_yaml::Error),
    /// An error from expanding values in a shell
    ShellExpand(String),
    /// An error from using a regex
    Regex(regex::Error),
    /// An error from uncarting a sample
    Cart(cart_rs::Error),
    /// An error from parsing a URL
    UrlParse(url::ParseError),
    /// An error from parsing an IP CIDR
    CidrParse(cidr::errors::NetworkParseError),
    /// An error from parsing an int
    ParseInt(std::num::ParseIntError),
    /// An error from joining a tokio task
    JoinError(tokio::task::JoinError),
    /// An error from converting values
    ConversionError(ConversionError),
    /// An error from parsing a semver version
    Semver(semver::Error),
    /// An error casting bytes to a utf8 formatted string
    StringFromUtf8(std::string::FromUtf8Error),
    /// An error from rustix
    #[cfg(feature = "rustix")]
    Rustix(rustix::io::Errno),
    /// An error from the k8s client
    #[cfg(feature = "k8s")]
    K8s(kube::Error),
    /// An error from getting a k8s config
    #[cfg(feature = "k8s")]
    K8sConfig(kube::config::KubeconfigError),
    /// An error from cgroups
    #[cfg(feature = "cgroups")]
    Cgroups(cgroups_rs::error::Error),
    /// An error from sending a kanal message
    #[cfg(feature = "kanal-err")]
    KanalSend(kanal::SendError),
    /// An error from receiving a kanal message
    #[cfg(feature = "kanal-err")]
    KanalRecv(kanal::ReceiveError),
    // An error from sending a crossbeam message
    #[cfg(feature = "crossbeam-err")]
    CrossbeamSend(crossbeam::channel::SendError<String>),
    #[cfg(feature = "scylla-utils")]
    ScyllaType(scylla::deserialize::TypeCheckError),
    #[cfg(feature = "scylla-utils")]
    ScyllaNextRow(scylla::errors::NextRowError),
    /// An error from dialoguer
    #[cfg(feature = "dialoguer-err")]
    Dialoguer(dialoguer::Error),
}

impl Error {
    /// Create a new generic error
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to set
    pub fn new<T: Into<String>>(msg: T) -> Self {
        Error::Generic(msg.into())
    }

    /// Get the status code from this error if one exists
    pub fn status(&self) -> Option<StatusCode> {
        // get the status code from any error types that support it
        match self {
            Error::Thorium { code, .. } => Some(code.to_owned()),
            Error::Reqwest(err) => err.status(),
            Error::Elastic(err) => err.status_code(),
            #[cfg(feature = "k8s")]
            Error::K8s(err) => match err {
                kube::Error::Api(resp) => StatusCode::from_u16(resp.code).ok(),
                _ => None,
            },
            _ => None,
        }
    }

    /// Get the error message for this error if one exists
    pub fn msg(&self) -> Option<String> {
        // get the msg from any error types that support it
        match self {
            Error::Thorium { msg, .. } => msg.clone(),
            Error::Generic(msg) => Some(msg.clone()),
            Error::Reqwest(err) => Some(err.to_string()),
            Error::IO(err) => Some(err.to_string()),
            Error::ChronoParse(err) => Some(err.to_string()),
            Error::PrefixStrip(err) => Some(err.to_string()),
            Error::Git(err) => Some(err.to_string()),
            Error::GitOpen(err) => Some(err.to_string()),
            Error::GitFindReference(err) => Some(err.to_string()),
            Error::GitReferenceIter(err) => Some(err.to_string()),
            Error::GitReferenceIterInit(err) => Some(err.to_string()),
            Error::GitReferencePeel(err) => Some(err.to_string()),
            Error::GitFindObject(err) => Some(err.to_string()),
            Error::GitDecodeObject(err) => Some(err.to_string()),
            Error::GitCommit(err) => Some(err.to_string()),
            Error::GitObjectTryInto(err) => Some(err.to_string()),
            Error::GitDateParse(err) => Some(err.to_string()),
            Error::Uuid(err) => Some(err.to_string()),
            Error::Config(err) => Some(err.to_string()),
            Error::Elastic(err) => Some(err.to_string()),
            Error::Serde(err) => Some(err.to_string()),
            Error::SerdeYaml(err) => Some(err.to_string()),
            Error::ShellExpand(err) => Some(err.clone()),
            Error::Regex(err) => Some(err.to_string()),
            Error::Cart(err) => Some(err.to_string()),
            Error::BuildElastic(err) => Some(err.to_string()),
            Error::UrlParse(err) => Some(err.to_string()),
            Error::CidrParse(err) => Some(err.to_string()),
            Error::ParseInt(err) => Some(err.to_string()),
            Error::JoinError(err) => Some(err.to_string()),
            Error::ConversionError(err) => Some(err.msg.to_owned()),
            Error::Semver(err) => Some(err.to_string()),
            Error::StringFromUtf8(err) => Some(err.to_string()),
            #[cfg(feature = "rustix")]
            Error::Rustix(err) => Some(err.to_string()),
            #[cfg(feature = "k8s")]
            Error::K8s(err) => Some(err.to_string()),
            #[cfg(feature = "k8s")]
            Error::K8sConfig(err) => Some(err.to_string()),
            #[cfg(feature = "cgroups")]
            Error::Cgroups(err) => Some(err.to_string()),
            #[cfg(feature = "crossbeam-err")]
            Error::CrossbeamSend(err) => Some(err.to_string()),
            #[cfg(feature = "kanal-err")]
            Error::KanalSend(err) => Some(err.to_string()),
            #[cfg(feature = "kanal-err")]
            Error::KanalRecv(err) => Some(err.to_string()),
            #[cfg(feature = "scylla-utils")]
            Error::ScyllaType(err) => Some(err.to_string()),
            #[cfg(feature = "scylla-utils")]
            Error::ScyllaNextRow(err) => Some(err.to_string()),
            #[cfg(feature = "dialoguer-err")]
            Error::Dialoguer(err) => Some(err.to_string()),
        }
    }

    /// get the kind of error as a str
    pub fn kind(&self) -> &'static str {
        // get the msg from any error types that support it
        match self {
            Error::Thorium { .. } => "Thorium",
            Error::Generic(_) => "Generic",
            Error::Reqwest(_) => "Reqwest",
            Error::IO(_) => "IO",
            Error::ChronoParse(_) => "ChronoParse",
            Error::PrefixStrip(_) => "PrefixStrip",
            Error::Git(_) => "Git",
            Error::GitFindReference(_) => "GitFindReference",
            Error::GitReferenceIter(_) => "GitReferenceIter",
            Error::GitReferenceIterInit(_) => "GitReferenceIterInit",
            Error::GitReferencePeel(_) => "GitReferencePeel",
            Error::GitFindObject(_) => "GitFindObject",
            Error::GitDecodeObject(_) => "GitDecodeObject",
            Error::GitCommit(_) => "GitCommit",
            Error::GitObjectTryInto(_) => "GitObjectTryInto",
            Error::GitDateParse(_) => "GitDateParse",
            Error::GitOpen(_) => "GitOpen",
            Error::Uuid(_) => "Uuid",
            Error::Config(_) => "Config",
            Error::Elastic(_) => "Elastic",
            Error::Serde(_) => "Serde",
            Error::SerdeYaml(_) => "SerdeYaml",
            Error::ShellExpand(_) => "ShellExpand",
            Error::Regex(_) => "Regex",
            Error::Cart(_) => "Cart",
            Error::BuildElastic(_) => "BuildElastic",
            Error::UrlParse(_) => "UrlParse",
            Error::CidrParse(_) => "CidrParse",
            Error::ParseInt(_) => "ParseInt",
            Error::JoinError(_) => "JoinError",
            Error::ConversionError(_) => "ConversionError",
            Error::Semver(_) => "Semver",
            Error::StringFromUtf8(_) => "StringFromUtf8",
            #[cfg(feature = "rustix")]
            Error::Rustix(_) => "rustix",
            #[cfg(feature = "k8s")]
            Error::K8s(_) => "K8s",
            #[cfg(feature = "k8s")]
            Error::K8sConfig(_) => "K8sConf",
            #[cfg(feature = "cgroups")]
            Error::Cgroups(_) => "Cgroups",
            #[cfg(feature = "kanal-err")]
            Error::KanalSend(_) => "KanalSend",
            #[cfg(feature = "kanal-err")]
            Error::KanalRecv(_) => "KanalRecv",
            #[cfg(feature = "crossbeam-err")]
            Error::CrossbeamSend(_) => "Crossbeam",
            #[cfg(feature = "scylla-utils")]
            Error::ScyllaType(_) => "ScyllaType",
            #[cfg(feature = "scylla-utils")]
            Error::ScyllaNextRow(_) => "ScyllaNextRow",
            #[cfg(feature = "dialoguer-err")]
            Error::Dialoguer(_) => "Dialoguer",
        }
    }
}

impl std::fmt::Display for Error {
    /// display this error in a easy readable format
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match (self.status(), self.msg()) {
            (Some(code), Some(msg)) => write!(f, "Code: {} Error: {}", code, msg),
            (None, Some(msg)) => write!(f, "Error: {}", msg),
            (Some(code), None) => write!(f, "Code: {}", code),
            (None, None) => write!(f, "Kind: {}", self.kind()),
        }
    }
}

// mark that this is an error struct
impl std::error::Error for Error {}

impl From<reqwest::Response> for Error {
    fn from(resp: reqwest::Response) -> Self {
        Error::Thorium {
            code: resp.status(),
            msg: block_on(resp.text()).ok().filter(|s| !s.is_empty()),
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Error::Reqwest(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IO(error)
    }
}

impl From<chrono::ParseError> for Error {
    fn from(error: chrono::ParseError) -> Self {
        Error::ChronoParse(error)
    }
}

impl From<std::path::StripPrefixError> for Error {
    fn from(error: std::path::StripPrefixError) -> Self {
        Error::PrefixStrip(error)
    }
}

impl From<git2::Error> for Error {
    fn from(error: git2::Error) -> Self {
        Error::Git(error)
    }
}

impl From<gix::open::Error> for Error {
    fn from(error: gix::open::Error) -> Self {
        Error::GitOpen(error)
    }
}

impl From<gix::reference::find::existing::Error> for Error {
    fn from(error: gix::reference::find::existing::Error) -> Self {
        Error::GitFindReference(error)
    }
}

impl From<gix::reference::iter::Error> for Error {
    fn from(error: gix::reference::iter::Error) -> Self {
        Error::GitReferenceIter(error)
    }
}

impl From<gix::reference::iter::init::Error> for Error {
    fn from(error: gix::reference::iter::init::Error) -> Self {
        Error::GitReferenceIterInit(error)
    }
}

impl From<gix::reference::peel::Error> for Error {
    fn from(error: gix::reference::peel::Error) -> Self {
        Error::GitReferencePeel(error)
    }
}

impl From<gix::object::find::existing::Error> for Error {
    fn from(error: gix::object::find::existing::Error) -> Self {
        Error::GitFindObject(error)
    }
}

impl From<gix::worktree::object::decode::Error> for Error {
    fn from(error: gix::worktree::object::decode::Error) -> Self {
        Error::GitDecodeObject(error)
    }
}

impl From<gix::object::commit::Error> for Error {
    fn from(error: gix::object::commit::Error) -> Self {
        Error::GitCommit(error)
    }
}

impl From<gix::object::try_into::Error> for Error {
    fn from(error: gix::object::try_into::Error) -> Self {
        Error::GitObjectTryInto(error)
    }
}

impl From<gix_date::parse::Error> for Error {
    fn from(error: gix_date::parse::Error) -> Self {
        Error::GitDateParse(error)
    }
}

impl From<uuid::Error> for Error {
    fn from(error: uuid::Error) -> Self {
        Error::Uuid(error)
    }
}

impl From<config::ConfigError> for Error {
    fn from(error: config::ConfigError) -> Self {
        Error::Config(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Serde(error)
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(error: serde_yaml::Error) -> Self {
        Error::SerdeYaml(error)
    }
}

impl<E: std::fmt::Display> From<shellexpand::LookupError<E>> for Error {
    fn from(error: shellexpand::LookupError<E>) -> Self {
        Error::ShellExpand(format!("{}", error))
    }
}

impl From<regex::Error> for Error {
    fn from(error: regex::Error) -> Self {
        Error::Regex(error)
    }
}

impl From<cart_rs::Error> for Error {
    fn from(error: cart_rs::Error) -> Self {
        Error::Cart(error)
    }
}

impl From<elasticsearch::Error> for Error {
    fn from(error: elasticsearch::Error) -> Self {
        Error::Elastic(error)
    }
}

impl From<elasticsearch::http::transport::BuildError> for Error {
    fn from(error: elasticsearch::http::transport::BuildError) -> Self {
        Error::BuildElastic(error)
    }
}

impl From<url::ParseError> for Error {
    fn from(error: url::ParseError) -> Self {
        Error::UrlParse(error)
    }
}

impl From<cidr::errors::NetworkParseError> for Error {
    fn from(error: cidr::errors::NetworkParseError) -> Self {
        Error::CidrParse(error)
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(error: std::num::ParseIntError) -> Self {
        Error::new(error.to_string())
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(error: tokio::task::JoinError) -> Self {
        Error::JoinError(error)
    }
}

impl From<ConversionError> for Error {
    fn from(error: ConversionError) -> Self {
        Error::ConversionError(error)
    }
}

impl From<semver::Error> for Error {
    fn from(error: semver::Error) -> Self {
        Error::Semver(error)
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Error::StringFromUtf8(error)
    }
}

#[cfg(feature = "rustix")]
impl From<rustix::io::Errno> for Error {
    fn from(error: rustix::io::Errno) -> Self {
        Error::Rustix(error)
    }
}

#[cfg(feature = "k8s")]
impl From<kube::Error> for Error {
    fn from(error: kube::Error) -> Self {
        Error::K8s(error)
    }
}

#[cfg(feature = "k8s")]
impl From<kube::config::KubeconfigError> for Error {
    fn from(error: kube::config::KubeconfigError) -> Self {
        Error::K8sConfig(error)
    }
}

#[cfg(feature = "cgroups")]
impl From<cgroups_rs::error::Error> for Error {
    fn from(error: cgroups_rs::error::Error) -> Self {
        Error::Cgroups(error)
    }
}

#[cfg(feature = "crossbeam-err")]
impl From<crossbeam::channel::SendError<std::string::String>> for Error {
    fn from(error: crossbeam::channel::SendError<std::string::String>) -> Self {
        Error::CrossbeamSend(error)
    }
}

#[cfg(feature = "kanal-err")]
impl From<kanal::SendError> for Error {
    fn from(error: kanal::SendError) -> Self {
        Error::KanalSend(error)
    }
}

#[cfg(feature = "kanal-err")]
impl From<kanal::ReceiveError> for Error {
    fn from(error: kanal::ReceiveError) -> Self {
        Error::KanalRecv(error)
    }
}

#[cfg(feature = "scylla-utils")]
impl From<scylla::deserialize::TypeCheckError> for Error {
    fn from(error: scylla::deserialize::TypeCheckError) -> Self {
        Error::ScyllaType(error)
    }
}

#[cfg(feature = "scylla-utils")]
impl From<scylla::errors::NextRowError> for Error {
    fn from(error: scylla::errors::NextRowError) -> Self {
        Error::ScyllaNextRow(error)
    }
}

#[cfg(feature = "dialoguer-err")]
impl From<dialoguer::Error> for Error {
    fn from(error: dialoguer::Error) -> Self {
        Error::Dialoguer(error)
    }
}
