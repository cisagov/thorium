//! A unified error class for Cart-rs

use std::{collections::TryReserveError, num::TryFromIntError};

/// a unified error class for Cart-rs
#[derive(Debug)]
pub enum Error {
    /// a Generic error
    Generic(String),
    /// An error from performing some IO
    IO(std::io::Error),
    /// An error from deserializing binary data
    BincodeDecode(bincode::error::DecodeError),
    /// An error from serializing binary data
    BincodeEncode(bincode::error::EncodeError),
    /// An error from converting an integer
    TryFromInt(TryFromIntError),
    /// An error from reserving more space for data
    TryReserve(TryReserveError),
    /// Finish was called before any data was specified
    FinishBeforeData,
}

impl Error {
    /// Creates a new error instance
    ///
    /// # Arguments
    ///
    /// * `msg` - The error message to return
    #[must_use]
    pub fn new<T: Into<String>>(msg: T) -> Self {
        Error::Generic(msg.into())
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Generic(msg) => write!(fmt, "Generic: {msg}"),
            Error::IO(err) => write!(fmt, "IO: {err}"),
            Error::BincodeDecode(err) => write!(fmt, "BincodeDecode: {err}"),
            Error::BincodeEncode(err) => write!(fmt, "BincodeEncode: {err}"),
            Error::TryFromInt(err) => write!(fmt, "TryFromInt: {err}"),
            Error::TryReserve(err) => write!(fmt, "TryReserve: {err}"),
            Error::FinishBeforeData => write!(fmt, "FinishBeforeData"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IO(error)
    }
}

impl From<bincode::error::DecodeError> for Error {
    fn from(error: bincode::error::DecodeError) -> Self {
        Error::BincodeDecode(error)
    }
}

impl From<bincode::error::EncodeError> for Error {
    fn from(error: bincode::error::EncodeError) -> Self {
        Error::BincodeEncode(error)
    }
}

impl From<TryReserveError> for Error {
    fn from(error: TryReserveError) -> Self {
        Error::TryReserve(error)
    }
}

impl From<TryFromIntError> for Error {
    fn from(error: TryFromIntError) -> Self {
        Error::TryFromInt(error)
    }
}
