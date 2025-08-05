use std::path::{Path, PathBuf};

use super::Keys;
use serde_derive::Deserialize;

use crate::Error;

/// The settings to use when cloning repos
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GitSettings {
    /// The path to the SSH keys to use
    pub ssh_keys: PathBuf,
}

impl GitSettings {
    /// Create a new [`GitSettings`]
    ///
    /// # Arguments
    ///
    /// * `ssh_keys` - The path to SSH keys to set
    #[must_use]
    pub fn new(ssh_keys: impl Into<PathBuf>) -> Self {
        Self {
            ssh_keys: ssh_keys.into(),
        }
    }
}

/// Help serde default our timeout to 600 seconds
fn default_client_timeout() -> u64 {
    600
}

/// The config options for our [`reqwest::Client`]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClientSettings {
    /// Ignore invalid certificates
    #[serde(default)]
    pub invalid_certs: bool,
    /// Ignore invalid hostnames when verifying certificates
    #[serde(default)]
    pub invalid_hostnames: bool,
    /// The certificate authorities to trust
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub certificate_authorities: Vec<PathBuf>,
    /// The number of seconds to wait before timing out
    #[serde(default = "default_client_timeout")]
    pub timeout: u64,
}

impl Default for ClientSettings {
    /// Default client settings to a sane default
    fn default() -> Self {
        ClientSettings {
            invalid_certs: false,
            invalid_hostnames: false,
            certificate_authorities: Vec::default(),
            timeout: default_client_timeout(),
        }
    }
}

/// Provide a default editor for serde
#[must_use]
pub fn default_default_editor() -> String {
    "vi".to_string()
}

/// A config for running Thorctl in user mode
///
/// This will not give the user the ability to deploy clusters/agents
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CtlConf {
    /// The settings tied to talking to the API
    pub keys: Keys,
    /// The git settings to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git: Option<GitSettings>,
    /// The settings for thorctls client
    #[serde(default)]
    pub client: ClientSettings,
    /// Skip the warning about possibly insecure connections
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_insecure_warning: Option<bool>,
    /// Skip automatic check for Thorctl updates with the API
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_update: Option<bool>,
    /// The default editor Thorctl will use
    #[serde(default = "default_default_editor")]
    pub default_editor: String,
}

impl CtlConf {
    /// Create a new [`CtlConf`] from the given [`Keys`]
    ///
    /// # Arguments
    ///
    /// * `keys` - The Thorium keys to set
    #[must_use]
    pub fn new(keys: Keys) -> Self {
        Self {
            keys,
            git: None,
            skip_update: None,
            client: ClientSettings::default(),
            skip_insecure_warning: None,
            default_editor: default_default_editor(),
        }
    }

    /// Check if our api url ends in '/api' and update it if needed
    ///
    /// # Arguments
    ///
    /// * `path` - The path to write our fixed config to
    fn fix_api_url(&mut self, path: &Path) -> Result<(), Error> {
        // check if our api url ends in api
        if self.keys.api.ends_with("/api") {
            // remove the /api from our config
            let trimmed = self.keys.api.trim_end_matches("/api");
            // update the api url in our config
            self.keys.api = trimmed.to_owned();
            // write the new configuration file
            let conf_file = std::fs::File::create(path)?;
            serde_yaml::to_writer(conf_file, &self)?;
        }
        Ok(())
    }

    /// Loads a [`CtlConf`] from the given path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load this config from
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        // get the path to our config
        let path = path.as_ref();
        // build the Thorctl config
        let mut config: CtlConf = config::Config::builder()
            // load from a file first
            .add_source(config::File::from(path).format(config::FileFormat::Yaml))
            // then overlay any environment args on top
            .add_source(config::Environment::with_prefix("THORCTL").separator("__"))
            .build()?
            .try_deserialize()?;
        // fix our config if needed
        config.fix_api_url(path)?;
        Ok(config)
    }
}
