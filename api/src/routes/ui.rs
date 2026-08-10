use axum::Router;
use http::{HeaderValue, header};
use tower::ServiceBuilder;
use tower_http::{
    services::{ServeDir, ServeFile},
    set_header::SetResponseHeaderLayer,
};

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
        .nest_service(
            "/assets",
            ServiceBuilder::new()
                .layer(SetResponseHeaderLayer::overriding(
                    header::CACHE_CONTROL,
                    //30 day cache limit. Can increase to a year (31536000)
                    //everything in assets/ is fingerprinted; new versions will
                    //have different fingerprinted bundles
                    HeaderValue::from_static("public, max-age=2592000, immutable"),
                ))
                .service(ServeDir::new("./ui/assets")),
        )
        .nest_service(
            "/thorium.ico",
            ServiceBuilder::new()
                .layer(SetResponseHeaderLayer::overriding(
                    header::CACHE_CONTROL,
                    //7 day cache limit
                    HeaderValue::from_static("public, max-age=604800"),
                ))
                .service(ServeFile::new("./ui/thorium.ico")),
        )
        .nest_service(
            "/ferris-scientist.png",
            ServiceBuilder::new()
                .layer(SetResponseHeaderLayer::overriding(
                    header::CACHE_CONTROL,
                    //7 day cache limit
                    HeaderValue::from_static("public, max-age=604800"),
                ))
                .service(ServeFile::new("./ui/ferris-scientist.png")),
        )
        .nest_service(
            "/manifest.json",
            ServiceBuilder::new()
                .layer(SetResponseHeaderLayer::overriding(
                    header::CACHE_CONTROL,
                    HeaderValue::from_static("no-cache"),
                ))
                .service(ServeFile::new("./ui/manifest.json")),
        )
        // always fallback to the ui index bundle for non api queries
        .fallback_service(
            ServiceBuilder::new()
                .layer(SetResponseHeaderLayer::overriding(
                    header::CACHE_CONTROL,
                    //important: index.html is not fingerprinted. Always want
                    //to fetch newest version of index.html, which will call
                    //newest versions of fingerprinted bundles in assets/
                    HeaderValue::from_static("no-cache"),
                ))
                .service(ServeFile::new("./ui/index.html")),
        );
    // merge our ui router into our global router
    router.merge(ui_router)
}
