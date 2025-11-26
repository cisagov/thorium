//! A synchronous Python client for Thorium, derived from the Rust client
//! using [PyO3](https://docs.rs/pyo3/latest/pyo3/)
//!
//! This is a helper crate designed to separate the any requirements for
//! building the Python client from the rest of the API library code (e.g.
//! building cdylib, Python version requirements, etc.). This crate simply defines
//! the API crate and its Python-related code as a dependency, then
//! re-exports that code as a Python module.

use pyo3::pymodule;

/// A synchronous Python client for Thorium, derived from the Rust client
/// using [PyO3](https://docs.rs/pyo3/latest/pyo3/)
#[pymodule]
pub mod thorium {
    // export all types relevant to the Python client
    //
    // any exported type must be a `pyo3::pyclass`
    #[pymodule_export]
    pub use thorium::client::conf::ClientSettings;
    #[pymodule_export]
    pub use thorium::client::{BasicBlocking, JobsBlocking, ReactionsBlocking, ThoriumBlocking};
    #[pymodule_export]
    pub use thorium::models::{
        BulkReactionResponse, CommitishKinds, GenericJob, GenericJobArgs, GenericJobOpts,
        HandleJobResponse, JobHandleStatus, Reaction, ReactionCreation, ReactionRequest,
        ReactionStatus, RepoDependency, RepoDependencyRequest,
    };
}
