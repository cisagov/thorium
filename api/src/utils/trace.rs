//! Sets up tracing for Thorium using either jaeger or stdout/stderr

use opentelemetry::trace::TraceContextExt;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::path::Path;
use tracing::Span;
use tracing_core::LevelFilter;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{filter::Filtered, fmt::Layer, layer::Layered, prelude::*, Registry};

use crate::conf::{LogLevel, Tracing, TracingLocal, TracingServices};

/// Log a message at the info level
#[macro_export]
macro_rules! info {
    ($level:expr, $($msg:tt)+) => {
        if $level ==  crate::conf::LogLevel::Info
        || $level ==  crate::conf::LogLevel::Debug
        || $level ==  crate::conf::LogLevel::Trace  {
            println!("{}", serde_json::json!({"timestamp": chrono::Utc::now(), "level": "INFO", "msg": $($msg)+}));
        }
    }
}

/// Log a message at the info level
#[macro_export]
macro_rules! setup {
    ($level:expr, $($msg:tt)+) => {
        if $level ==  crate::conf::LogLevel::Setup
        || $level ==  crate::conf::LogLevel::Info
        || $level ==  crate::conf::LogLevel::Debug
        || $level ==  crate::conf::LogLevel::Trace  {
            println!("{}", serde_json::json!({"timestamp": chrono::Utc::now(), "level": "SETP", "msg": $($msg)+}));
        }
    }
}

/// Log a message at the error level
#[macro_export]
macro_rules! error {
    ($level:expr, $($msg:tt)+) => {
        if $level !=  crate::conf::LogLevel::Off {
            println!("{}", serde_json::json!({"timestamp": chrono::Utc::now(), "level": "ERRO", "msg": $($msg)+}));
        }
    }
}

/// Get the current traces id
pub fn get_trace() -> Option<String> {
    // get our current context and span
    let context = Span::current().context();
    let span = context.span();
    // get this spans context
    let span_context = span.span_context();
    // try to extract our trace id
    span_context
        .is_valid()
        .then(|| span_context.trace_id().to_string())
}

/// Setup our grpc tracer.
///
/// # Arguments
///
/// * `name` - The name of the service to trace
/// * `endpoint` - The gRPC endpoint to send traces too
/// * `level` - The log level to set
/// * `registry` - The registry to add our tracers too
fn setup_grpc(
    name: &str,
    endpoint: &str,
    level: LogLevel,
    registry: Layered<Filtered<Layer<Registry>, LevelFilter, Registry>, Registry>,
) -> SdkTracerProvider {
    // setup an exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .expect("Failed to setup tracing grpc exporter");
    // setup our tracer provider
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        //.with_simple_exporter(exporter)
        .build();
    // build a tracer
    let tracer = provider.tracer(name.to_owned());
    // define our layer filter
    let filtered = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(level.to_filter());
    // init our tracing registry
    registry
        .with(filtered)
        .try_init()
        .expect("Failed to register opentelemetry tracers/subscribers");
    info!(
        level,
        format!(
            "Sending {} traces for {} to gRPC trace sink at {}",
            level, name, endpoint
        )
    );
    provider
}

/// Setup our local tracer
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
fn setup_local(
    name: &str,
    conf: &TracingLocal,
) -> Filtered<Layer<Registry>, LevelFilter, Registry> {
    // log that local tracing is enabled
    info!(
        conf.level,
        format!("Sending {} for {name} to stdout", conf.level)
    );
    tracing_subscriber::fmt::layer().with_filter(conf.level.to_filter())
}

/// Set up the correct tracer
///
/// # Arguments
///
/// * `name` - The name of the application we are tracing
/// * `trace_conf` - The Thorium tracing config to use
#[must_use = "Tracing provider must be manually shutdown"]
pub fn setup(name: &str, trace_conf: &Tracing) -> Option<SdkTracerProvider> {
    // build our local tracer/subscriber
    let local = setup_local(name, &trace_conf.local);
    // Add our local tracer to our registry
    let registry = tracing_subscriber::registry().with(local);
    // get out external tracing settings
    if let Some(external) = &trace_conf.external {
        // send traces to an external application and get a provider
        let provider = match external {
            // set up the correct external tracer
            TracingServices::Grpc { endpoint, level } => {
                setup_grpc(name, endpoint, *level, registry)
            }
        };
        // return our newly setup provider
        Some(provider)
    } else {
        registry
            .try_init()
            .expect("Failed to register stdout registry");
        // local traces have no provider
        None
    }
}

/// Set up the correct tracer from a stand alone config file
///
/// # Arguments
///
/// * `name` - The name of the application we are tracing
/// * `trace_conf` - The Thorium tracing config to use
#[must_use = "Tracing provider must be manually shutdown"]
pub fn from_file(name: &str, path: &str) -> Option<SdkTracerProvider> {
    // Check if our tracing file exists or not
    let trace_config = if Path::new(path).exists() {
        // load our tracing config from a file
        Tracing::from_file(path).expect("Failed to load tracing config")
    } else {
        // use a default config
        Tracing::default()
    };
    // setup our tracers/subscribers
    setup(name, &trace_config)
}

/// Shutdown this tracer
///
/// # Arguments
///
/// * `provider` - The tracing provider to shutdown
pub fn shutdown(provider: Option<SdkTracerProvider>) {
    // if we have a provider shut it down
    if let Some(provider) = provider {
        // shutdown this provider
        provider
            .shutdown()
            .expect("Failed to shutdown tracing provider");
    }
}
