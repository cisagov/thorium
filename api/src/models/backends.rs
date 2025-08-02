//! Wrappers for all objects within Thorium with different backends
//!
//! Currently only Redis and Scylla are supported

#[cfg(feature = "api")]
#[path = "backends"]
mod backends_reexport {
    pub mod comments;
    pub mod db;
    pub mod deadlines;
    pub mod elastic;
    pub mod events;
    pub mod files;
    pub mod groups;
    pub mod helpers;
    pub mod images;
    pub mod jobs;
    pub mod logs;
    pub mod network_policies;
    pub mod pipelines;
    pub mod reactions;
    pub mod repos;
    pub mod results;
    pub mod s3;
    pub mod search;
    pub mod setup;
    pub mod streams;
    pub mod system;
    pub mod trees;
    pub mod users;
    pub mod version;
    pub mod volumes;

    pub use comments::CommentSupport;
}

#[cfg(feature = "api")]
pub use backends_reexport::*;

// Dependencies required for client functionality
#[cfg(any(feature = "api", feature = "client"))]
#[path = "backends"]
mod backends_reexport_client {
    mod support;

    pub use support::NotificationSupport;
    pub use support::OutputSupport;
    pub use support::TagSupport;
}

#[cfg(any(feature = "api", feature = "client"))]
pub use backends_reexport_client::*;
