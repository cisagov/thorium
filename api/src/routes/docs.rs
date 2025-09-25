use axum::Router;
use axum::routing::{MethodRouter, get_service};
use tower_http::services::{ServeDir, ServeFile};
use utoipa::openapi::security::{Http, HttpAuthScheme, SecurityScheme};
use utoipa::{Modify, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

use super::BasicApiDocs;
use super::events::EventApiDocs;
use super::files::FileApiDocs;
use super::groups::GroupApiDocs;
use super::images::ImageApiDocs;
use super::jobs::JobApiDocs;
use super::network_policies::NetworkPolicyDocs;
use super::pipelines::PipelineApiDocs;
use super::reactions::ReactionApiDocs;
use super::repos::RepoApiDocs;
use super::search::SearchApiDocs;
use super::search::events::{ResultSearchEventApiDocs, TagSearchEventApiDocs};
use super::streams::StreamApiDocs;
use super::system::SystemApiDocs;
use super::users::UserApiDocs;

use crate::models::{ResultSearchEvent, SearchEvent, TagSearchEvent};
use crate::{Conf, utils::AppState};

/// The struct containing our OpenAPI security info
pub struct OpenApiSecurity;

impl Modify for OpenApiSecurity {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        // get our components
        let components = openapi.components.as_mut().unwrap();
        components.add_security_scheme(
            "basic",
            SecurityScheme::Http(Http::new(HttpAuthScheme::Basic)),
        );
    }
}

/// Serve our docs
///
///  # Arguments
///
/// * `conf` - The Thorium config
fn user(conf: &Conf) -> MethodRouter {
    // build the full path to our target file
    let full = &conf.thorium.assets.user_docs.as_path();
    // build the router for our user docs
    get_service(
        ServeDir::new(full).not_found_service(ServeFile::new(&conf.thorium.assets.not_found)),
    )
}

/// Serve our developer docs
///
///  # Arguments
///
/// * `conf` - The Thorium config
fn dev(conf: &Conf) -> MethodRouter {
    // build the full path to our target file
    let full = &conf.thorium.assets.dev_docs.as_path();
    // build the router for our user docs
    get_service(
        ServeDir::new(full).not_found_service(ServeFile::new(&conf.thorium.assets.not_found)),
    )
}

/// Add the docs routes to our router
///
/// # Arguments
///
// * `router` - The router to add routes too
pub fn mount(router: Router<AppState>, conf: &Conf) -> Router<AppState> {
    router
        .nest_service("/docs/user", user(conf))
        .nest_service("/docs/dev", dev(conf))
        .merge(
            SwaggerUi::new("/docs/swagger-ui")
                .url("/openapi.json", BasicApiDocs::openapi())
                .url("/events/openapi.json", EventApiDocs::openapi())
                .url("/files/openapi.json", FileApiDocs::openapi())
                .url("/groups/openapi.json", GroupApiDocs::openapi())
                .url("/images/openapi.json", ImageApiDocs::openapi())
                .url("/jobs/openapi.json", JobApiDocs::openapi())
                .url(
                    "/networkpolicies/openapi.json",
                    NetworkPolicyDocs::openapi(),
                )
                .url("/pipelines/openapi.json", PipelineApiDocs::openapi())
                .url("/reactions/openapi.json", ReactionApiDocs::openapi())
                .url("/repos/openapi.json", RepoApiDocs::openapi())
                .url("/search/openapi.json", SearchApiDocs::openapi())
                .url(
                    format!("/search/events/{}", ResultSearchEvent::url()),
                    ResultSearchEventApiDocs::openapi(),
                )
                .url(
                    format!("/search/events/{}", TagSearchEvent::url()),
                    TagSearchEventApiDocs::openapi(),
                )
                .url("/stream/openapi.json", StreamApiDocs::openapi())
                .url("/system/openapi.json", SystemApiDocs::openapi())
                .url("/users/openapi.json", UserApiDocs::openapi()),
        )
}
