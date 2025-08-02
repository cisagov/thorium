use axum::routing::{get_service, MethodRouter};
use axum::Router;
use tower_http::services::{ServeDir, ServeFile};
use utoipa::openapi::security::{Http, HttpAuthScheme, SecurityScheme};
use utoipa::{Modify, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

use super::events::EventApiDocs;
use super::files::FileApiDocs;
use super::groups::GroupApiDocs;
use super::images::ImageApiDocs;
use super::jobs::JobApiDocs;
use super::network_policies::NetworkPolicyDocs;
use super::pipelines::PipelineApiDocs;
use super::reactions::ReactionApiDocs;
use super::repos::RepoApiDocs;
use super::search::events::{ResultSearchEventApiDocs, TagSearchEventApiDocs};
use super::search::SearchApiDocs;
use super::streams::StreamApiDocs;
use super::system::SystemApiDocs;
use super::users::UserApiDocs;
use super::BasicApiDocs;

use crate::models::{ResultSearchEvent, SearchEvent, TagSearchEvent};
use crate::{utils::AppState, Conf};

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
        .nest_service("/api/docs/user", user(conf))
        .nest_service("/api/docs/dev", dev(conf))
        .merge(
            SwaggerUi::new("/api/docs/swagger-ui")
                .url("/api/openapi.json", BasicApiDocs::openapi())
                .url("/api/events/openapi.json", EventApiDocs::openapi())
                .url("/api/files/openapi.json", FileApiDocs::openapi())
                .url("/api/groups/openapi.json", GroupApiDocs::openapi())
                .url("/api/images/openapi.json", ImageApiDocs::openapi())
                .url("/api/jobs/openapi.json", JobApiDocs::openapi())
                .url(
                    "/api/networkpolicies/openapi.json",
                    NetworkPolicyDocs::openapi(),
                )
                .url("/api/pipelines/openapi.json", PipelineApiDocs::openapi())
                .url("/api/reactions/openapi.json", ReactionApiDocs::openapi())
                .url("/api/repos/openapi.json", RepoApiDocs::openapi())
                .url("/api/search/openapi.json", SearchApiDocs::openapi())
                .url(
                    format!("/api/search/events/{}", ResultSearchEvent::url()),
                    ResultSearchEventApiDocs::openapi(),
                )
                .url(
                    format!("/api/search/events/{}", TagSearchEvent::url()),
                    TagSearchEventApiDocs::openapi(),
                )
                .url("/api/stream/openapi.json", StreamApiDocs::openapi())
                .url("/api/system/openapi.json", SystemApiDocs::openapi())
                .url("/api/users/openapi.json", UserApiDocs::openapi()),
        )
}
