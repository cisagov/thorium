use axum::Router;
use axum::routing::get_service;
use tower_http::services::{ServeDir, ServeFile};

use crate::utils::AppState;

/// Add the UI route to our router
///
/// This will setup a fallback service that will server the index.html file
/// to anything that doesn't match a route thats not:
/// - A route in the API router
/// - A route in /assets/*
/// - A ui route
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>) -> Router<AppState> {
    // create a new router for ui routes
    let ui_router = Router::new()
        .nest_service("/assets", get_service(ServeDir::new("./ui/assets")))
        .nest_service(
            "/thorium.ico",
            get_service(ServeFile::new("./ui/thorium.ico")),
        )
        .nest_service(
            "/ferris-scientist.png",
            get_service(ServeDir::new("./ui/ferris-scientist.png")),
        )
        .nest_service(
            "/manifest.json",
            get_service(ServeFile::new("./ui/manifest.json")),
        )
        // always fallback to the ui index bundle for non api queries
        .fallback(get_service(ServeFile::new("./ui/index.html")));
    // merge our ui router into our global router
    router.merge(ui_router)
}
