//! Utilties for testing the Thorium API

mod api;
pub mod generators;
mod helpers;
mod impls;

pub use api::{CONF, admin_client};

// expose a blocking admin client for sync tests
#[cfg(all(feature = "sync", not(feature = "python")))]
pub use api::admin_client_blocking;
