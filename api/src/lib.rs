//! The Thorium API, client, and objects

#![feature(proc_macro_hygiene, decl_macro, io_error_more, round_char_boundary)]

#[macro_use]
extern crate serde_derive;

// import any API only structures
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        extern crate serde_json;

        use std::net::{SocketAddr, IpAddr};
        use tower_http::cors::{CorsLayer};
        use axum::http::Method;
    }
}

mod args;
pub mod conf;
pub mod models;
mod routes;
pub mod utils;

// expose test utilities if that feature is enabled
#[cfg(feature = "test-utilities")]
pub mod test_utilities;

// expose the clients if that feature is enabled
#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "client")]
pub use client::{CtlConf, Cursor, Error, Keys, SearchDate, Thorium};

// if the sync client is enabled then also re-export that
#[cfg(feature = "sync")]
pub use client::ThoriumBlocking;

pub use conf::Conf;

/// Run an initial consistency scan using current
/// system settings before starting the API
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
/// * `log_level` - The log level configured in the Thorium config
#[cfg(feature = "api")]
async fn initial_settings_consistency_scan(
    shared: std::sync::Arc<utils::Shared>,
    log_level: conf::LogLevel,
) -> Result<(), utils::ApiError> {
    // try to get the system settings from the backend
    let settings = match crate::models::backends::db::system::get_settings(&shared).await {
        Ok(settings) => settings,
        Err(err) => {
            if err.code == axum::http::StatusCode::NOT_FOUND {
                // we got a 404, so assume this is first
                // run and just use default settings
                let default_settings = crate::models::SystemSettings::default();
                setup!(
                    log_level,
                    format!(
                        "No existing system settings found! \
                        This is likely Thorium's first run. \
                        Using default system settings: {default_settings:?}"
                    )
                );
                // set default settings in the db
                crate::models::backends::db::system::reset_settings(&shared)
                    .await
                    .map_err(|err| {
                        utils::ApiError::new(
                            err.code,
                            Some(format!(
                                "Failed to set default system settings: {}",
                                err.msg.unwrap_or("An unknown error occurred".to_string())
                            )),
                        )
                    })?;
                default_settings
            } else {
                return Err(utils::ApiError::new(
                    err.code,
                    Some(format!(
                        "An error occurred retrieving system settings: {}",
                        err.msg.unwrap_or("An unknown error occurred".to_string())
                    )),
                ));
            }
        }
    };
    // if this is first start, no admin user will exist, so
    // make a fake admin User to force a consistency scan
    let fake_admin = crate::models::User {
        username: String::default(),
        password: Option::default(),
        email: String::default(),
        // make this user an admin
        role: crate::models::UserRole::Admin,
        groups: Vec::default(),
        token: String::default(),
        token_expiration: chrono::Utc::now(),
        unix: Option::default(),
        settings: crate::models::UserSettings::default(),
        verified: bool::default(),
        verification_token: None,
    };
    // do a scan for consistency according to current settings
    settings.consistency_scan(&fake_admin, &shared).await?;
    Ok(())
}

#[cfg(feature = "api")]
/// Build the axum app
fn build_app(
    state: utils::AppState,
    conf: &Conf,
) -> (
    axum::Router,
    Option<opentelemetry_sdk::trace::SdkTracerProvider>,
) {
    use axum::extract::DefaultBodyLimit;
    use axum::http::header::{HeaderName, HeaderValue};
    use axum::{http::Request, response::Response};
    use routes::{
        basic, binaries, docs, events, files, groups, images, jobs, network_policies, pipelines,
        reactions, repos, search, streams, system, trees, ui, users,
    };
    use std::time::Duration;
    use tower_http::set_header::SetResponseHeaderLayer;
    use tower_http::trace::{DefaultMakeSpan, TraceLayer};
    use tracing::{event, Level, Span};

    use crate::utils::trace;

    // build an axum router
    let mut app = axum::Router::new();
    app = basic::mount(app);
    app = binaries::mount(app, conf);
    app = docs::mount(app, conf);
    app = events::mount(app);
    app = files::mount(app);
    app = groups::mount(app);
    app = images::mount(app);
    app = jobs::mount(app);
    app = pipelines::mount(app);
    app = network_policies::mount(app);
    app = reactions::mount(app);
    app = repos::mount(app);
    app = search::mount(app);
    app = streams::mount(app);
    app = system::mount(app);
    app = ui::mount(app);
    app = users::mount(app);
    app = trees::mount(app);
    // setup our tracing
    let trace_provider = trace::setup("ThoriumAPI", &conf.thorium.tracing);
    // build cors middleware for our app
    let cors = if conf.thorium.cors.insecure {
        CorsLayer::permissive()
    } else {
        // start building our cors settings and allow all methods we use
        let cors = CorsLayer::new().allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::DELETE,
        ]);
        // cast the domains we want to add to the correct type
        let origins = conf
            .thorium
            .cors
            .domains
            .iter()
            .map(|domain| domain.parse())
            .collect::<Result<Vec<HeaderValue>, _>>()
            .expect("Failed to parse CORS domains");
        cors.allow_origin(origins)
    };
    // add middleware to our app
    app = app
        .layer(DefaultBodyLimit::disable())
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_request(|req: &Request<_>, span: &Span| {
                    // get our uri as a str
                    let url_and_query = match req.uri().path_and_query() {
                        Some(path_and_query) => path_and_query.as_str(),
                        None => req.uri().path(),
                    };
                    // get our base url as a str
                    let url = req.uri().path();
                    event!(
                        parent: span,
                        Level::INFO,
                        url = url,
                        uri = url_and_query,
                        msg = "Starting Request"
                    );
                })
                .on_response(|response: &Response, latency: Duration, span: &Span| {
                    // get our status code
                    let code = response.status();
                    // build our response event
                    event!(
                        parent: span,
                        Level::INFO,
                        code = code.as_u16(),
                        status = code.as_str(),
                        latency = latency.as_millis(),
                        msg = "Responding to Request"
                    );
                }),
        )
        .layer(cors)
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("thorium-version"),
            HeaderValue::from_str(env!("CARGO_PKG_VERSION"))
                .expect("Thorium version is not a valid header value"),
        ));
    (app.with_state(state), trace_provider)
}

#[cfg(feature = "api")]
/// Launches the Thorium api using axum
///
/// # Panics
///
/// Will panic if we cannot connect to any databases or execute db setup commands.
pub async fn axum(config: Conf) {
    // setup shared object
    let shared = Box::pin(utils::Shared::new(config.clone())).await;
    // get our log level
    let log_level = shared.config.thorium.tracing.local.level;
    // log interface/port we are binding to
    info!(
        log_level,
        format!(
            "binding to {}:{}",
            &config.thorium.interface, &config.thorium.port
        ),
    );
    // build our app state
    let state = utils::AppState::new(shared);
    // run a scan on our data based on the current system settings
    let scan_handle = tokio::spawn(initial_settings_consistency_scan(
        state.shared.clone(),
        log_level,
    ));
    // build our app
    let (app, trace_provider) = build_app(state, &config);
    // parse our interface addr
    let bind_addr: IpAddr = config
        .thorium
        .interface
        .parse()
        .expect("Failed to parse interface addr");
    // get the address and port to bind too
    let addr = SocketAddr::new(bind_addr, config.thorium.port);
    // make sure our scan completed successfully before we start
    let scan_result = scan_handle
        .await
        .unwrap_or_else(|err| panic!("Tokio join error when running consistency scan: {err}"));
    if let Err(err) = scan_result {
        // our scan failed, so don't start the API
        panic!("Error running initial consistency scan: {err}");
    }
    // track how many bind attempts we have tried
    let mut attempts = 0;
    // bind and start handling requests
    loop {
        // try to bind the listener for our server
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|_| panic!("Failed to bind to {addr}"));
        // start handling requests
        match axum::serve(listener, app.clone()).await {
            Ok(()) => break,
            Err(error) => {
                error!(log_level, format!("Failed to bind server: {:#?}", error));
            }
        }
        // increment our attempt count
        attempts += 1;
        // check if we reached our attempt limit
        if attempts <= 10 {
            // we have tried and failed 10 times now so abort
            break;
        }
        // sleep for 3 seconds between attempts
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }
    // log that we failed to start
    error!(log_level, "Failed to bind server in 10 attempts".to_owned());
    // shutdown our trace provider if we ever exit
    crate::utils::trace::shutdown(trace_provider);
}
