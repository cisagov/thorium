//! Key file for authenticating to Thorium

use std::path::{Path, PathBuf};

/// Auth keys for Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Keys {
    /// The ip to access the api at (must begin with http:// or https://)
    pub api: String,
    /// The username to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    /// The password to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    /// The token to use in place of basic auth
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

impl Keys {
    /// Create a new auth Keys object
    ///
    /// # Arguments
    ///
    /// * `path` - The path to use when reading in the Thorium auth keys
    pub fn new(path: &str) -> Result<Self, config::ConfigError> {
        Self::from_path(PathBuf::from(path))
    }

    /// Create a new auth Keys object from a path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to use when reading in the Thorium auth keys
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        config::Config::builder()
            // load from a file first
            .add_source(config::File::from(path.as_ref()).format(config::FileFormat::Yaml))
            // then overlay any environment args on top
            .add_source(config::Environment::with_prefix("THORIUM_KEYS").separator("__"))
            .build()?
            .try_deserialize()
    }

    /// Create a new auth keys object from an API URL and a token
    ///
    /// # Arguments
    ///
    /// * `api` - The url of the Thorium api to talk too
    /// * `token` - The token to use to authenticate to Thorium
    pub fn new_token<A: Into<String>, T: Into<String>>(api: A, token: T) -> Self {
        Keys {
            api: api.into(),
            username: None,
            password: None,
            token: Some(token.into()),
        }
    }
}
