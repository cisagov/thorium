//! Determines if this binary needs an update and updates it if required

use futures::TryStreamExt;
use http::StatusCode;
use std::path::Path;
use tokio::fs::{File, OpenOptions};
use tokio_util::io::StreamReader;

use super::Error;
use crate::models::{Arch, Component, Os, Version};
use crate::send_build;

/// A handler for updates in Thorium
#[derive(Clone)]
pub struct Updates {
    /// The host/url that Thorium can be reached at
    host: String,
    /// token to use for auth
    token: String,
    /// A reqwest client for reqwests
    client: reqwest::Client,
}

impl Updates {
    /// Creates a new updates handler
    ///
    /// Instead of directly creating this handler you likely want to simply create a
    /// `thorium::Thorium` and use the handler within that instead.
    ///
    /// # Arguments
    ///
    /// * `host` - url/ip of the Thorium api
    /// * `token` - The token used for authentication
    /// * `client` - The reqwest client to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::Updates;
    ///
    /// let client = reqwest::Client::new();
    /// let updates = Updates::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Updates {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

impl Updates {
    /// Get the current versions for Thorium
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Get the current versions in Thorium
    /// thorium.updates.get_version().await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn get_version(&self) -> Result<Version, Error> {
        // build url for claiming a job
        let url = format!("{base}/api/version", base = self.host);
        // build request
        let req = self.client.get(&url).header("authorization", &self.token);
        // send this request
        send_build!(self.client, req, Version)
    }

    /// Downloads the latest binary
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{Os, Arch, Component};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Get the current versions in Thorium
    /// thorium.updates.download(Os::Linux, Arch::X86_64, Component::Agent, "/tmp/thorium-agent").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn download<P: AsRef<Path>>(
        &self,
        os: Os,
        arch: Arch,
        component: Component,
        path: P,
    ) -> Result<File, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/binaries/{os}/{arch}/{component}",
            base = self.host,
            os = os,
            arch = arch,
            component = component.to_file_name(os)
        );
        // build and send the request
        let resp = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .send()
            .await?;
        // make sure we got a 200
        match resp.status() {
            StatusCode::OK => {
                // get our response as a stream of bytes
                let stream = resp
                    .bytes_stream()
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()));
                // convert our async read to a buf reader
                let mut reader = StreamReader::new(stream);
                // make a file to save the response stream too
                let mut file = open_file(path).await?;
                // write our uncarted stream to disk
                tokio::io::copy(&mut reader, &mut file).await?;
                Ok(file)
            }
            // the response had an error status
            _ => Err(Error::from(resp)),
        }
    }

    /// Download and update the current binary to the latest version
    ///
    /// # Arguments
    ///
    /// * `component` - The component to update
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::Component;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Get the current versions in Thorium
    /// thorium.updates.update(Component::Agent).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update(&self, component: Component) -> Result<(), Error> {
        // get the path of the current exe
        let exe_path = match std::env::current_exe() {
            Ok(exe_path) => exe_path,
            Err(error) => {
                return Err(Error::new(format!(
                    "Failed to get current exe path: {:#?}",
                    error
                )))
            }
        };
        // get the current os, arch, and component
        let os = Os::default();
        let arch = Arch::default();
        // download and update our file
        self.update_specific(os, arch, component, exe_path).await
    }

    /// Download and update another binary to the latest version
    ///
    /// # Arguments
    ///
    /// * `component` - The component to update
    /// * `path` - The path to the binary to update
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::Component;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Get the current versions in Thorium
    /// thorium.updates.update_other(Component::Agent, "/opt/thorium/thorium-agent").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update_other<P: AsRef<Path>>(
        &self,
        component: Component,
        path: P,
    ) -> Result<(), Error> {
        // get the current os, arch, and component
        let os = Os::default();
        let arch = Arch::default();
        // download and update our file
        self.update_specific(os, arch, component, path).await
    }

    /// Download and update the current binary to the latest version
    ///
    /// # Arguments
    ///
    /// * `os` - The os to download a binary for
    /// * `arch` - The arch to download a binary for
    /// * `component` - The component to download a binary for
    /// * `path` - The path to update the final binary at
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{Os, Arch, Component};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // Get the current versions in Thorium
    /// thorium.updates.update_specific(Os::Linux, Arch::X86_64, Component::Agent, "/opt/thorium/thorium-agent").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    pub async fn update_specific<P: AsRef<Path>>(
        &self,
        os: Os,
        arch: Arch,
        component: Component,
        path: P,
    ) -> Result<(), Error> {
        // get our current exes file name
        let name = match path.as_ref().file_name() {
            Some(name) => name,
            None => {
                return Err(Error::new(format!(
                    "Failed to get exe name from {:#?}",
                    path.as_ref()
                )))
            }
        };
        // build a path to temp hidden update file
        if let Some(parent_path) = path.as_ref().parent() {
            // add the hidden file name
            let temp_path = parent_path.join(format!(".{}", name.to_string_lossy()));
            // download our new file to the temp path
            self.download(os, arch, component, &temp_path).await?;
            // build the backup file path
            let bak_path = parent_path.join(format!("{}.bak", name.to_string_lossy()));
            // rename the old file to a backup
            match tokio::fs::rename(&path, bak_path).await {
                // rename the new agent file after renaming old
                Ok(_) => tokio::fs::rename(temp_path, &path).await?,
                Err(e) => match e.kind() {
                    // rename the new agent even if old agent file wasn't found
                    std::io::ErrorKind::NotFound => tokio::fs::rename(temp_path, &path).await?,
                    _ => {
                        return Err(Error::new(format!(
                            "Failed to rename old agent file: {:#?}",
                            e
                        )));
                    }
                },
            }
        }
        Ok(())
    }
}

/// Open a file with the execute permissions
///
/// # Arguments
///
/// * `path` - The path to open
#[cfg(unix)]
async fn open_file<P: AsRef<Path>>(path: P) -> Result<File, std::io::Error> {
    // make a file to save the response stream too
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o775)
        .truncate(true)
        .open(&path)
        .await
}

/// Open a file with the execute permissions
///
/// # Arguments
///
/// * `path` - The path to open
#[cfg(target_os = "windows")]
async fn open_file<P: AsRef<Path>>(path: P) -> Result<File, std::io::Error> {
    // make a file to save the response stream too
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .await
}
