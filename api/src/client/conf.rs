use std::path::{Path, PathBuf};
use url::Url;

use super::Keys;
use crate::Error;

#[cfg(feature = "python")]
use pyo3::pyclass;

/// Normalize a URL into the host string reqwest compares against when matching `NO_PROXY`
///
/// reqwest's `NO_PROXY` matching tests the request's host (no scheme, port, or path), so a
/// `no_proxy` entry meant to bypass a given URL has to be reduced to just that host. This parses
/// `url` with the same library reqwest uses, so the result matches the host the client actually
/// connects to. Returns `None` if `url` has no host (e.g. it isn't a valid URL).
///
/// This is only meant for inputs that are URLs; raw `NO_PROXY` entries (bare domains, leading-dot
/// domains, IPs, CIDRs) are already in the form reqwest expects and should be used as-is.
///
/// # Arguments
///
/// * `url` - The URL whose host should be used as a `no_proxy` entry
#[must_use]
pub fn no_proxy_host(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_owned))
}

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
pub fn default_client_timeout() -> u64 {
    600
}

/// The config options for our [`reqwest::Client`]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "python", thorium_derive::pyclass(get_all))]
pub struct ClientSettings {
    /// Ignore invalid certificates
    #[serde(default)]
    pub invalid_certs: bool,
    /// Ignore invalid hostnames when verifing certificates
    #[serde(default)]
    pub invalid_hostnames: bool,
    /// The certificate authorities to trust
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub certificate_authorities: Vec<PathBuf>,
    /// The number of seconds to wait before timing out
    #[serde(default = "default_client_timeout")]
    pub timeout: u64,
    /// Disable all proxies and connect to hosts directly
    ///
    /// When false (the default), proxy settings are read from the environment
    /// (`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`, honoring `NO_PROXY`). Setting this to true
    /// ignores those and connects directly, which is useful for in-cluster traffic where
    /// every host is reachable without a proxy.
    #[serde(default)]
    pub disable_proxy: bool,
    /// An explicit proxy URL to route all (http + https) traffic through
    ///
    /// This takes precedence over environment proxy settings but is ignored when
    /// `disable_proxy` is true. The URL may embed credentials (e.g.
    /// `http://user:pass@host:port`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// A comma-separated list of hosts/domains/CIDRs that bypass the explicit `proxy`
    ///
    /// This mirrors the `NO_PROXY` environment variable and only applies when `proxy` is
    /// set; environment-based proxies already honor `NO_PROXY` on their own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_proxy: Option<String>,
}

impl Default for ClientSettings {
    /// Default client settings to a sane default
    fn default() -> Self {
        ClientSettings {
            invalid_certs: false,
            invalid_hostnames: false,
            certificate_authorities: Vec::default(),
            timeout: default_client_timeout(),
            disable_proxy: false,
            proxy: None,
            no_proxy: None,
        }
    }
}

/// Provide a default default editor for serde
#[must_use]
pub fn default_default_editor() -> String {
    "vi".to_string()
}

/// The settings to use when using AI
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AISettings {
    /// The endpoint to talk to an AI model at
    pub endpoint: String,
    /// The API key to use when talking to our AI model
    pub api_key: String,
    /// The model to use
    pub model: String,
}

/// The container CLI tool thorctl shells out to for image operations
///
/// podman is CLI-compatible with docker for the verbs thorctl uses (pull, save,
/// load, tag, push, build), so the runtime only changes which binary is invoked.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ContainerRuntime {
    /// The docker CLI (`docker`)
    #[default]
    Docker,
    /// The podman CLI (`podman`)
    Podman,
}

impl ContainerRuntime {
    /// The executable name to invoke for this runtime
    #[must_use]
    pub fn binary(self) -> &'static str {
        // map each runtime to the CLI binary thorctl spawns
        match self {
            ContainerRuntime::Docker => "docker",
            ContainerRuntime::Podman => "podman",
        }
    }
}

impl std::fmt::Display for ContainerRuntime {
    /// Format the runtime as its binary name (used in user-facing messages)
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter to write the binary name into
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.binary())
    }
}

/// An activated scoped token for Thorctl to authenticate with
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActiveScopedToken {
    /// The name of the activated scoped token
    pub name: String,
    /// The value of the activated scoped token
    pub token: String,
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
    /// The container CLI thorctl uses for image operations (auto-detected when unset)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_runtime: Option<ContainerRuntime>,
    /// The settings to use when using AI
    pub ai: Option<AISettings>,
    /// The currently activated scoped token if one is active
    ///
    /// When set Thorctl will authenticate with this scoped token instead of
    /// the users primary credentials for all commands except scoped token
    /// management.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoped_token: Option<ActiveScopedToken>,
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
            container_runtime: None,
            ai: None,
            scoped_token: None,
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
            serde_norway::to_writer(conf_file, &self)?;
        }
        Ok(())
    }

    /// Assemble a [`CtlConf`] from a config file body overlaid with `THORCTL_*` environment values
    ///
    /// The file is the trusted, persistent configuration; the environment overlay is a convenience
    /// for ad-hoc overrides. Security-relevant toggles must NOT be silently flippable by the
    /// environment: `invalid_certs`/`invalid_hostnames` disable TLS validation and
    /// `skip_insecure_warning` suppresses the only interactive guardrail, so an attacker who can
    /// influence the environment (a hostile `.envrc`, CI job, or sourced script) without write
    /// access to the config file could otherwise disable certificate validation with no warning
    /// and no persistent record. Centralizing the overlay here lets the fix filter those
    /// keys out of the environment source in one place.
    ///
    /// # Arguments
    ///
    /// * `file_body` - The raw YAML body of the config file
    /// * `env` - The `THORCTL_*` environment values to overlay, already stripped of the prefix
    fn assemble_config(
        file_body: &str,
        env: config::Environment,
    ) -> Result<Self, Error> {
        // build the Thorctl config from the file body first, then the environment overlay
        let mut config: CtlConf = config::Config::builder()
            // load from the file body first
            .add_source(config::File::from_str(file_body, config::FileFormat::Yaml))
            // overlay the environment for ad-hoc, non-security-relevant overrides
            .add_source(env)
            .build()?
            .try_deserialize()?;
        // the `config` crate's environment source has no per-key filter, so instead parse the
        // trusted file body on its own and force the security-relevant toggles back to their file
        // values. This guarantees `client.invalid_certs`, `client.invalid_hostnames`, and
        // `skip_insecure_warning` can only be set in the persistent config file — the environment can
        // never silently disable TLS validation or suppress the insecure-connection warning.
        let file_only: CtlConf = config::Config::builder()
            .add_source(config::File::from_str(file_body, config::FileFormat::Yaml))
            .build()?
            .try_deserialize()?;
        // restore the TLS validation toggles from the trusted file
        config.client.invalid_certs = file_only.client.invalid_certs;
        config.client.invalid_hostnames = file_only.client.invalid_hostnames;
        // restore the insecure-warning suppressor from the trusted file
        config.skip_insecure_warning = file_only.skip_insecure_warning;
        Ok(config)
    }

    /// Loads a [`CtlConf`] from the given path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to load this config from
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        // get the path to our config
        let path = path.as_ref();
        // read the raw config file body from disk
        let file_body = std::fs::read_to_string(path)?;
        // build the environment overlay for THORCTL_* variables
        let env = config::Environment::with_prefix("THORCTL").separator("__");
        // assemble the config from the file body overlaid with the environment
        let mut config = Self::assemble_config(&file_body, env)?;
        // fix our config if needed
        config.fix_api_url(path)?;
        Ok(config)
    }
}

#[cfg(test)]
mod conf_tests {
    use super::CtlConf;

    /// Build a `THORCTL_*` environment overlay from an explicit key/value map
    ///
    /// This drives `assemble_config` deterministically without mutating the real process
    /// environment, matching how `from_path` builds its overlay (prefix `THORCTL`, `__` separator).
    ///
    /// # Arguments
    ///
    /// * `pairs` - The already-prefixed environment variable name/value pairs to overlay
    fn env_overlay(pairs: &[(&str, &str)]) -> config::Environment {
        // collect the pairs into the map the config crate expects
        let map = pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        // build a THORCTL-prefixed overlay sourced from our explicit map
        config::Environment::with_prefix("THORCTL")
            .separator("__")
            .source(Some(map))
    }

    /// the `THORCTL_*` environment overlay must NOT be able to disable TLS certificate
    /// validation. A config file with validation ON must stay ON even when the environment tries to
    /// flip `client.invalid_certs`.
    #[test]
    fn env_cannot_disable_tls_validation() {
        // a minimal config file that keeps TLS validation enabled
        let file_body = "keys:\n  api: https://thorium.example\n  username: analyst\n  token: t\n";
        // an attacker-controlled environment that tries to accept invalid certificates. The overlay
        // uses the `__` prefix separator `from_path` configures, so the key is `THORCTL__CLIENT__...`
        let env = env_overlay(&[("THORCTL__CLIENT__INVALID_CERTS", "true")]);
        // assemble the config exactly as `from_path` would
        let conf = CtlConf::assemble_config(file_body, env)
            .expect("config with a valid file body should assemble");
        // the environment must not have been able to disable certificate validation
        assert!(
            !conf.client.invalid_certs,
            "THORCTL_CLIENT__INVALID_CERTS disabled TLS validation via the environment"
        );
    }

    /// the `THORCTL_*` environment overlay must NOT be able to suppress the interactive
    /// insecure-connection warning, which is the only guardrail left when validation is off.
    #[test]
    fn env_cannot_suppress_insecure_warning() {
        // a minimal config file that leaves the insecure warning enabled
        let file_body = "keys:\n  api: https://thorium.example\n  username: analyst\n  token: t\n";
        // an attacker-controlled environment that tries to suppress the insecure warning
        let env = env_overlay(&[("THORCTL__SKIP_INSECURE_WARNING", "true")]);
        // assemble the config exactly as `from_path` would
        let conf = CtlConf::assemble_config(file_body, env)
            .expect("config with a valid file body should assemble");
        // the environment must not have been able to suppress the insecure warning
        assert!(
            !conf.skip_insecure_warning.unwrap_or(false),
            "THORCTL_SKIP_INSECURE_WARNING suppressed the warning via the environment"
        );
    }
}
