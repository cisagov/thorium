//! Possible errors in the Thorium reactor

/// An error in the reactor
#[derive(Debug, thiserror::Error, strum::AsRefStr)]
pub enum Error {
    /// A generic error with a message
    #[error("{0}")]
    Generic(String),
    /// A Thorium error
    #[error(transparent)]
    Thorium(#[from] Box<thorium::Error>),
    /// An IO error
    #[error(transparent)]
    IO(#[from] std::io::Error),
    /// An error from parsing a semver version
    #[error(transparent)]
    Semver(#[from] semver::Error),
    /// A rustix error
    #[error(transparent)]
    Rustix(#[from] rustix::io::Errno),
    /// A `serde_json` Error
    #[error("JSON serialization/deserialization error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    /// A libvirt error
    #[cfg(feature = "kvm")]
    #[error(transparent)]
    Virt(#[from] virt::error::Error),
    /// An error during guest OS detection (bad image format, unrecognized layout, etc.)
    #[cfg(feature = "kvm")]
    #[error("OS detection failed: {0}")]
    OsDetect(String),
}

impl Error {
    /// Create a new generic error
    pub fn new<T: Into<String>>(msg: T) -> Self {
        Self::Generic(msg.into())
    }

    /// Returns the error message
    #[must_use]
    pub fn msg(&self) -> String {
        self.to_string()
    }

    /// Returns the kind of error this is
    #[must_use]
    pub fn kind(&self) -> &str {
        self.as_ref()
    }

    /// Create a generic error with a message and context from another error
    pub fn with_context<T: Into<String>, E: std::error::Error>(msg: T, err: E) -> Self {
        Self::Generic(format!("{}: {:?}", msg.into(), err))
    }
}

// Manual impl so `?` boxes automatically
impl From<thorium::Error> for Error {
    fn from(err: thorium::Error) -> Self {
        Self::Thorium(Box::new(err))
    }
}
