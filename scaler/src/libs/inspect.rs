use chrono::prelude::*;
use serde::{Deserialize, Deserializer};
use std::process::Command;
use thorium::conf::Crane;
use thorium::Error;
use tracing::{event, Level, Span};

fn def_working_dir() -> String {
    "/".to_string()
}

/// Deserialize null values to their default
fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: Deserializer<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// The docker config from skopeo
#[derive(Debug, Deserialize, Clone)]
pub struct DockerConfig {
    /// All environment variables set in the container
    #[allow(dead_code)]
    #[serde(rename = "Env", default)]
    pub env: Option<Vec<String>>,
    /// The entrypoint of the container
    #[serde(
        rename = "Entrypoint",
        deserialize_with = "deserialize_null_default",
        default
    )]
    pub entrypoint: Vec<String>,
    /// The command to execute in the container
    #[serde(rename = "Cmd", deserialize_with = "deserialize_null_default", default)]
    pub cmd: Vec<String>,
    /// The working directory of the container
    #[allow(dead_code)]
    #[serde(rename = "WorkingDir", default = "def_working_dir")]
    pub working_dir: String,
}

fn default_as_timestamp() -> DateTime<Utc> {
    Utc::now() + chrono::Duration::minutes(10)
}

/// Docker info from skopeo
#[derive(Debug, Deserialize, Clone)]
pub struct DockerInfo {
    /// The architecture of the container
    #[allow(dead_code)]
    pub architecture: String,
    /// The os this container is built on
    #[allow(dead_code)]
    pub os: String,
    /// The config for this docker image
    pub config: DockerConfig,
    /// When this docker info should expire at
    #[allow(dead_code)]
    #[serde(default = "default_as_timestamp")]
    pub expires: DateTime<Utc>,
}

impl DockerInfo {
    /// Inspects a image stored in a registry
    ///
    /// # Arguments
    ///
    /// * `image` - The url/path on the registry to inspect
    /// * `span` - The span to log traces under
    pub fn inspect(crane: &Crane, image: &str, span: &Span) -> Result<Self, Error> {
        // build the args for getting this images config
        let args = if crane.insecure {
            vec!["--insecure", "config", &image]
        } else {
            vec!["config", &image]
        };
        // Inspect docker image using Skopeo
        let output = Command::new(&crane.path).args(args).output()?;
        event!(parent: span, Level::INFO, image = image);
        // cast both stdout and stderr to strings
        let raw = String::from_utf8_lossy(&output.stdout);
        let err = String::from_utf8_lossy(&output.stderr);
        // Throw an error if stdout has text
        if !err.is_empty() {
            return Err(Error::new(format!("Error while inspecting image {}", err)));
        }
        DockerInfo::deserialize(&raw)
    }

    /// Deserializes a string into DockerInfo
    ///
    /// # Arguments
    ///
    /// * `data` - The raw docker info as a string
    pub fn deserialize(data: &str) -> Result<Self, Error> {
        match serde_json::from_str(data) {
            Ok(info) => Ok(info),
            Err(e) => Err(Error::new(format!(
                "Failed to deserialize docker info: {:#?}",
                e
            ))),
        }
    }
}
