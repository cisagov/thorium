//! The scylla specific features that can be exposed outside of the api
//!
//! This is largely for use in Thoradm

#[cfg(feature = "scylla-utils")]
#[path = "scylla_utils"]
mod scylla_utils_reexport {
    pub mod errors;
    pub mod events;
    pub mod files;
    pub mod network_policies;
    pub mod repos;
    pub mod results;
    pub mod s3;
    pub mod system;
    pub mod tags;
}

#[cfg(feature = "scylla-utils")]
pub use scylla_utils_reexport::*;

// export any modules the client might need
#[cfg(any(feature = "scylla-utils", feature = "client"))]
#[path = "scylla_utils"]
mod scylla_utils_client_reexport {
    pub mod keys;
}

#[cfg(any(feature = "scylla-utils", feature = "client"))]
pub use scylla_utils_client_reexport::*;
