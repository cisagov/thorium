use axum::Router;
use tower_http::services::ServeDir;

use crate::{Conf, utils::AppState};

/// Serve our binaries
///
///  # Arguments
///
/// * `conf` - The Thorium config
fn user(conf: &Conf) -> ServeDir {
    // build the full path to our target file
    let full = &conf.thorium.assets.binaries.as_path();
    // build the router for our user docs
    ServeDir::new(full)
}

/// Add the binaries routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>, conf: &Conf) -> Router<AppState> {
    router.nest_service("/binaries", user(conf))
}
