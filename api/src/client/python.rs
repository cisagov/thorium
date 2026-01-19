//! Synchronous Python client based on Rust
//!
//! The actual Python module is exported/built in the `thorpy` crate which has
//! this crate as a dependency.

mod files;

use base64::Engine;
use pyo3::{pymethods, types::PyType, Bound};
use std::path::PathBuf;

use crate::{
    client::{
        conf::default_client_timeout, helpers, BasicBlocking, ClientSettings, FilesBlocking,
        JobsBlocking, ReactionsBlocking,
    },
    Error, ThoriumBlocking,
};

#[pymethods]
impl ThoriumBlocking {
    /// Create a new Thorium blocking client
    ///
    /// You must provide either a token or a username/password combination to authenticate
    #[new]
    #[pyo3(signature =
        (
            host,
            token=None,
            username=None,
            password=None,
            settings = ClientSettings::default()
        )
    )]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        host: &str,
        token: Option<String>,
        username: Option<String>,
        password: Option<String>,
        settings: ClientSettings,
    ) -> Result<Self, Error> {
        // build a client
        let client = helpers::build_blocking_reqwest_client(&settings)?;
        // authenticate if needed
        let (token, expires) = match (token, username, password) {
            (None, Some(username), Some(password)) => {
                ThoriumBlocking::basic_auth(host, &username, &password, &client)?
            }
            (Some(token), _, _) => (token, None),
            _ => return Err(Error::new("Either username/password or token must be set")),
        };
        // convert our buffer into a Vec<u8> and base64 it
        let encoded = base64::engine::general_purpose::STANDARD.encode(token.as_bytes());
        // build token auth string
        let auth_str = format!("token {encoded}");
        let basic = BasicBlocking::new(host, &client);
        let jobs = JobsBlocking::new(host, &auth_str, &client);
        let reactions = ReactionsBlocking::new(host, &auth_str, &client);
        let files = FilesBlocking::new(host, &auth_str, &client);
        Ok(Self {
            basic,
            jobs,
            reactions,
            files,
            host: host.to_string(),
            _auth_str: auth_str,
            expires,
            _client: client,
        })
    }

    /// Create a Thorium client from a path to a key file on disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to read [`Keys`] from
    #[classmethod]
    #[pyo3(name = "from_key_file")]
    pub fn from_key_file_py(_cls: &Bound<'_, PyType>, path: &str) -> Result<Self, Error> {
        Self::from_key_file(path)
    }

    /// Create a Thorium client from a path to a Thorctl config on disk
    ///
    /// # Arguments
    ///
    /// * `path` - The path to read [`CtlConf`] from
    #[classmethod]
    #[pyo3(name = "from_ctl_conf_file")]
    pub fn from_ctl_conf_file_py(_cls: &Bound<'_, PyType>, path: &str) -> Result<Self, Error> {
        Self::from_ctl_conf_file(path)
    }
}

#[pymethods]
impl ClientSettings {
    /// Create new client settings
    #[new]
    #[pyo3(signature =
        (
            invalid_certs=false,
            invalid_hostnames=false,
            certificate_authorities=Vec::new(),
            timeout=default_client_timeout()
        )
    )]
    fn new_py(
        invalid_certs: bool,
        invalid_hostnames: bool,
        certificate_authorities: Vec<PathBuf>,
        timeout: u64,
    ) -> Self {
        Self {
            invalid_certs,
            invalid_hostnames,
            certificate_authorities,
            timeout,
        }
    }
}
