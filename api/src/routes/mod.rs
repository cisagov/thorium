#[cfg(feature = "api")]
#[path = ""]
mod routes_reexport {
    pub mod basic;
    pub mod binaries;
    pub mod docs;
    pub mod events;
    pub mod files;
    pub mod groups;
    pub mod images;
    pub mod jobs;
    pub mod network_policies;
    pub mod pipelines;
    pub mod reactions;
    pub mod repos;
    pub mod search;
    mod shared;
    pub mod streams;
    pub mod system;
    pub mod trees;
    pub mod ui;
    pub mod users;

    use basic::BasicApiDocs;
    use docs::OpenApiSecurity;
}

#[cfg(feature = "api")]
pub use routes_reexport::*;
