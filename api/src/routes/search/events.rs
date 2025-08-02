//! The routes for search events

use axum::Router;

use crate::utils::AppState;

mod results;
mod tags;

pub use results::ResultSearchEventApiDocs;
pub use tags::TagSearchEventApiDocs;

/// Mount search event routes to the router
///
/// # Arguments
///
/// * `router` - The router to add routes to
#[allow(clippy::let_and_return)]
pub(super) fn mount(router: Router<AppState>) -> Router<AppState> {
    // mount tag search events routes
    let router = results::mount(router);
    let router = tags::mount(router);
    router
}
