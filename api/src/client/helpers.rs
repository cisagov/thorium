use reqwest::{Certificate, ClientBuilder, NoProxy, Proxy};

use super::{ClientSettings, Error};

/// The per-request timeout (in seconds) to use for large file transfers
///
/// Large bodies (samples, repos, result files, and cache files) can take far longer to
/// upload or download than the default client timeout allows. These endpoints override
/// the per-request timeout with this generous value (24 hours) so that a slow-but-healthy
/// transfer is never aborted by the client's default deadline.
pub(crate) const LARGE_UPLOAD_TIMEOUT_SECS: u64 = 86_400;

/// Apply our proxy settings to a [`reqwest::ClientBuilder`]
///
/// The precedence is:
/// 1. If `disable_proxy` is set, disable all proxies and connect directly.
/// 2. Otherwise if an explicit `proxy` is set, route all traffic through it. Explicit proxies
///    do not automatically honor `NO_PROXY`, so we attach a bypass list from the `no_proxy`
///    setting if given, falling back to the `NO_PROXY` environment variable.
/// 3. Otherwise leave the builder untouched so reqwest reads proxy settings from the
///    environment and honors `NO_PROXY` automatically.
///
/// # Arguments
///
/// * `builder` - The client builder to apply proxy settings to
/// * `settings` - The settings for building a client
fn apply_proxy_settings(
    builder: ClientBuilder,
    settings: &ClientSettings,
) -> Result<ClientBuilder, Error> {
    // disabling proxies takes precedence over any other proxy setting
    if settings.disable_proxy {
        return Ok(builder.no_proxy());
    }
    // route all traffic through an explicit proxy if one was set
    if let Some(proxy_url) = &settings.proxy {
        // build a bypass list from our settings or fall back to the environment
        let bypass = settings
            .no_proxy
            .as_deref()
            .and_then(NoProxy::from_string)
            .or_else(NoProxy::from_env);
        return Ok(builder.proxy(Proxy::all(proxy_url)?.no_proxy(bypass)));
    }
    // otherwise let reqwest read proxy settings from the environment
    Ok(builder)
}

/// Build a reqwest client for thorctl
///
/// # Arguments
///
/// * `settings` - The settings for building a client
pub(super) async fn build_reqwest_client(
    settings: &ClientSettings,
) -> Result<reqwest::Client, Error> {
    // start building our client
    let mut builder = reqwest::Client::builder()
        .danger_accept_invalid_certs(settings.invalid_certs)
        .danger_accept_invalid_hostnames(settings.invalid_hostnames)
        .timeout(std::time::Duration::from_secs(settings.timeout));
    // apply our proxy settings
    builder = apply_proxy_settings(builder, settings)?;
    // crawl over any custom CAs and add them to our trust store
    for ca_path in &settings.certificate_authorities {
        // try to load this CA from disk
        let ca_bytes = tokio::fs::read(ca_path).await.map_err(|err| {
            Error::new(format!(
                "Unable to read certificate file '{}': {}.",
                ca_path.to_string_lossy(),
                err
            ))
        })?;
        // based on the type of certificate cast try to cast this to a cert
        let cert = match ca_path.extension().and_then(|ext| ext.to_str()) {
            Some("der") => Certificate::from_der(&ca_bytes)?,
            Some("pem") => Certificate::from_pem(&ca_bytes)?,
            // this is a bundle of certificates instead of just one
            Some("crt") => {
                for cert in Certificate::from_pem_bundle(&ca_bytes)? {
                    // add this cert to our clients trust store
                    builder = builder.add_root_certificate(cert);
                }
                // continue to the next cert
                continue;
            }
            _ => continue,
        };
        // add this cert to our clients trust store
        builder = builder.add_root_certificate(cert);
    }
    // build our client
    Ok(builder.build()?)
}

/// Build a reqwest client for thorctl
///
/// # Arguments
///
/// * `settings` - The settings for building a client
#[cfg(feature = "sync")]
pub(super) fn build_blocking_reqwest_client(
    settings: &ClientSettings,
) -> Result<reqwest::Client, Error> {
    // start building our client
    let mut builder = reqwest::Client::builder()
        .danger_accept_invalid_certs(settings.invalid_certs)
        .danger_accept_invalid_hostnames(settings.invalid_hostnames)
        .timeout(std::time::Duration::from_secs(settings.timeout));
    // apply our proxy settings
    builder = apply_proxy_settings(builder, settings)?;
    // crawl over any custom CAs and add them to our trust store
    for ca_path in &settings.certificate_authorities {
        // try to load this CA from disk
        let ca_bytes = std::fs::read(ca_path)?;
        // based on the type of certificate cast try to cast this to a cert
        let cert = match ca_path.extension().and_then(|ext| ext.to_str()) {
            Some("der") => Certificate::from_der(&ca_bytes)?,
            Some("pem") => Certificate::from_pem(&ca_bytes)?,
            // this is a bundle of certificates instead of just one
            Some("crt") => {
                for cert in Certificate::from_pem_bundle(&ca_bytes)? {
                    // add this cert to our clients trust store
                    builder = builder.add_root_certificate(cert);
                }
                // continue to the next cert
                continue;
            }
            _ => continue,
        };
        // add this cert to our clients trust store
        builder = builder.add_root_certificate(cert);
    }
    // build our client
    Ok(builder.build()?)
}

#[doc(hidden)]
#[macro_export]
macro_rules! send {
    ($client:expr, $req:expr) => {
        // attempt to send request
        match $client.execute($req.build()?).await {
            // response was received
            Ok(resp) => {
                // check if a response has an error status or not
                if resp.status().is_success() {
                    // response is successful return it
                    Ok(resp)
                } else {
                    // the response had an error status
                    Err(Error::from(resp))
                }
            }
            Err(e) => Err(Error::from(e)),
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! send_build {
    ($client:expr, $req:expr, $build:ty) => {
        // attempt to send request
        match $client.execute($req.build()?).await {
            // response was received
            Ok(resp) => {
                // check if a response has an error status or not
                if resp.status().is_success() {
                    // attempt to build this response or return an error
                    match resp.json::<$build>().await {
                        // successfully built object
                        Ok(val) => Ok(val),
                        // failed to build object create error
                        Err(e) => Err(Error::from(e)),
                    }
                } else {
                    // the response had an error status
                    Err(Error::from(resp))
                }
            }
            Err(e) => Err(Error::from(e)),
        }
    };
}

/// Send a request and if its successful get the body as bytes
#[doc(hidden)]
#[macro_export]
macro_rules! send_bytes {
    ($client:expr, $req:expr) => {
        // attempt to send request
        match $client.execute($req.build()?).await {
            // response was received
            Ok(resp) => {
                // check if a response has an error status or not
                if resp.status().is_success() {
                    // attempt to get this response as bytes or return an error
                    match resp.bytes().await {
                        // successfully got bytes
                        Ok(val) => Ok(val),
                        // failed to build object create error
                        Err(e) => Err(Error::from(e)),
                    }
                } else {
                    // the response had an error status
                    Err(Error::from(resp))
                }
            }
            Err(e) => Err(Error::from(e)),
        }
    };
}

/// Adds a query param if it not None
#[doc(hidden)]
#[macro_export]
macro_rules! add_query {
    ($vec:expr, $key:expr, $value:expr) => {
        if let Some(value) = &$value {
            $vec.push(($key, value.to_string()));
        }
    };
}

/// Adds a vector of the same query param
#[doc(hidden)]
#[macro_export]
macro_rules! add_query_list {
    ($vec:expr, $key:expr, $value:expr) => {
        for value in $value.iter() {
            $vec.push(($key, value.to_string()));
        }
    };
}

/// Adds a vector of the same query param
#[doc(hidden)]
#[macro_export]
macro_rules! add_query_list_clone {
    ($vec:expr, $key:expr, $value:expr) => {
        for value in $value.iter() {
            $vec.push(($key.clone(), value.to_string()));
        }
    };
}

/// Adds a query param if its true
#[doc(hidden)]
#[macro_export]
macro_rules! add_query_bool {
    ($vec:expr, $key:expr, $value:expr) => {
        if $value {
            $vec.push(($key, "true".to_string()));
        }
    };
}

/// adds a text field to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! add_text {
    ($form:expr, $key:expr, $value:expr) => {
        match $value.take() {
            Some(value) => $form.text($key, value),
            None => $form,
        }
    };
}

/// adds a list of fields to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! add_list {
    ($form:expr, $key:expr, $value:expr) => {
        $value
            .drain(..)
            .fold($form, |form, value| form.text($key, value))
    };
}
/// reads in a file from a path and adds it to a form
#[doc(hidden)]
#[macro_export]
macro_rules! add_file {
    ($form:expr, $key:expr, $path:expr) => {{
        // a path was set so read it an and set it to be upload
        let mut fp = tokio::fs::File::open(&$path).await?;
        let mut buff = vec![];
        fp.read_to_end(&mut buff).await?;
        // build the form part that contains this file
        let file_part = reqwest::multipart::Part::bytes(buff).mime_str("multipart/form-data")?;
        // try to set this files name
        let file_part = match $path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
        {
            Some(name) => file_part.file_name(name),
            None => file_part,
        };
        // add the file to upload
        $form.part($key, file_part)
    }};
}

/// Adds a timestamp as a query param in rfc3339 formate
#[doc(hidden)]
#[macro_export]
macro_rules! add_date {
    ($vec:expr, $key:expr, $value:expr) => {
        if let Some(value) = &$value {
            $vec.push(($key, value.to_rfc3339()));
        }
    };
}

/// adds a text field to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_text {
    ($form:expr, $key:expr, $value:expr) => {
        match $value.take() {
            Some(value) => $form.text($key, value),
            None => $form,
        }
    };
}

/// converts a text field to a string and adds it to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_text_to_string {
    ($form:expr, $key:expr, $value:expr) => {
        match $value.map(|v| v.to_string()) {
            Some(value) => $form.text($key, value),
            None => $form,
        }
    };
}

/// adds a list of fields to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_list {
    ($form:expr, $key:expr, $value:expr) => {
        $value
            .drain(..)
            .fold($form, |form, value| form.text($key.to_owned(), value))
    };
}

/// adds a list of fields to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_set {
    ($form:expr, $key:expr, $value:expr) => {
        $value
            .drain()
            .fold($form, |form, value| form.text($key.to_owned(), value))
    };
}

/// adds a list of non strings to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_list_conv {
    ($form:expr, $key:expr, $value:expr) => {
        $value.drain(..).fold($form, |form, value| {
            form.text($key.to_owned(), value.to_string())
        })
    };
}

/// adds a list of non strings to a multipart form by serializing them to json
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_list_serialize {
    ($form:expr, $key:expr, $value:expr) => {
        for value in &$value {
            $form = $form.text($key.to_owned(), serde_json::to_string(value)?);
        }
    };
}

/// converts a date to a string and adds it to a multipart form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_date {
    ($form:expr, $key:expr, $value:expr) => {
        match $value.map(|v| v.to_rfc3339()) {
            Some(value) => $form.text($key, value),
            None => $form,
        }
    };
}

/// reads in a file from a path and adds it to a form
#[doc(hidden)]
#[macro_export]
macro_rules! multipart_file {
    ($form:expr, $key:expr, $path:expr) => {{
        // a path was set so read it an and set it to be upload
        let file = tokio::fs::File::open(&$path).await?;
        // get the length of this file so we can size our buffer correctly
        let len = file.metadata().await?.len();
        // get the name this file is going to be uploaded under
        let file_name = $path.to_string_lossy().to_string();
        // convert our file into a framed read stream
        let codec = tokio_util::codec::BytesCodec::new();
        let stream = tokio_util::codec::FramedRead::new(file, codec);
        // convert our stream to a body to pass to reqwest
        let body = reqwest::Body::wrap_stream(stream);
        // build the form part that contains this file
        let file_part = reqwest::multipart::Part::stream_with_length(body, len)
            .mime_str("multipart/form-data")?
            // set this files name
            .file_name(file_name);
        // add the file to upload
        let form = $form.part($key, file_part);
        form
    }};
    ($form:expr, $key:expr, $path:expr, $prefix:expr) => {{
        // a path was set so read it an and set it to be upload
        let file = tokio::fs::File::open(&$path).await?;
        // get the length of this file so we can size our buffer correctly
        let len = file.metadata().await?.len();
        // get the name this file is going to be uploaded under trimming any prefix that was set
        let file_name = match $prefix {
            Some(prefix) => $path.strip_prefix(prefix)?.to_string_lossy().to_string(),
            None => $path.to_string_lossy().to_string(),
        };
        // convert our file into a framed read stream
        let codec = tokio_util::codec::BytesCodec::new();
        let stream = tokio_util::codec::FramedRead::new(file, codec);
        // convert our stream to a body to pass to reqwest
        let body = reqwest::Body::wrap_stream(stream);
        // build the form part that contains this file
        let file_part = reqwest::multipart::Part::stream_with_length(body, len)
            .mime_str("multipart/form-data")?
            // set this files name
            .file_name(file_name);
        // add the file to upload
        let form = $form.part($key, file_part);
        form
    }};
}
