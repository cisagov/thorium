//! A module containing various utility functions for use in multiple handlers

use colored::Colorize;
use data_encoding::HEXLOWER;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use thorium::Keys;
use thorium::{Error, Thorium};
use tokio::io::{AsyncReadExt, BufReader};

use crate::{Args, CtlConf};

pub mod fs;
pub mod groups;
pub mod images;
pub mod notifications;
pub mod pipelines;
pub mod reactions;
pub mod repos;

/// Get a Thorium client or setup keys
pub async fn get_client(args: &Args) -> Result<(CtlConf, Thorium), Error> {
    let (config, thorium) = match &args.keys {
        Some(keys_path) => {
            // parse the keys from the file
            let keys = Keys::from_path(keys_path)?;
            // build our Thorium client based on our config
            let thorium = Thorium::from_keys(keys.clone()).await?;
            // build a base ctl conf containing the keys
            let config = CtlConf::new(keys);
            (config, thorium)
        }
        None => {
            // load ctl conf
            let config = CtlConf::from_path(&args.config)?;
            // build our Thorium client based on our config
            let thorium = Thorium::from_ctl_conf(config.clone()).await?;
            (config, thorium)
        }
    };
    Ok((config, thorium))
}

/// Get the sha256 for a file
pub async fn sha256<P: AsRef<Path>>(path: P) -> Result<String, Error> {
    // get buffered reader for this file
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file);
    // read this file into a local buffer and hash it
    let mut sha256 = Sha256::new();
    let mut buff = [0; 2048];
    loop {
        // read in 2048 bytes and count how many are read
        let count = reader.read(&mut buff[..]).await?;
        // if we read in no bytes then we have read our entire file
        if count == 0 {
            break;
        }
        // update our hashers with our newly read data
        sha256.update(&buff[..count]);
    }
    // build a digest for this
    let sha256 = HEXLOWER.encode(&sha256.finalize());
    Ok(sha256)
}

/// Log any errors and return
#[doc(hidden)]
#[macro_export]
macro_rules! check {
    ($self:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                // log this error
                $self.bar.error(error.to_string());
                // return early
                return;
            }
        }
    };
    ($self:expr, $result:expr, $path:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                // log this error
                $self.bar.error(error.to_string());
                // check if our path exists
                if let Ok(true) = tokio::fs::try_exists($path).await {
                    // clean up this repos dir
                    if let Err(error) = tokio::fs::remove_dir_all($path).await {
                        // log this io error
                        $self.bar.error(error.to_string());
                    }
                }
                // return early
                return;
            }
        }
    };
}

/// Print a warning message that Thorctl is possibly insecure as well as a command to run
/// to disable the setting that caused the warning message
macro_rules! print_warning {
    ($msg:expr, $api:expr, $cmd:expr) => {
        println!(
            "{}: Thorctl is currently set to {} when connecting \
            to Thorium. Only continue if you 100% trust the instance at '{}'.\n\
            \n    \
            Note: You can avoid this error message in the future by either running `{}` \
            or by disabling this warning message altogether with `{}`\n",
            "WARNING".bright_yellow(),
            $msg.bright_red(),
            $api.blue(),
            $cmd.green(),
            "thorctl config --skip-insecure-warning=true".green()
        );
    };
}

/// Print a warning message if any of the insecure settings are set
///
/// # Arguments
///
/// * `api` - The API we're connecting to
/// * `invalid_certs` - Accept invalid certs
/// * `invalid_hostnames` - Accept invalid hostnames
/// * `certificate_authorities` - A list of certificate authorities to implicitly trust
pub fn warn_insecure(
    api: &str,
    invalid_certs: bool,
    invalid_hostnames: bool,
    certificate_authorities: &[PathBuf],
) -> Result<(), Error> {
    // check possibly insecure settings in order of most to least severe
    if invalid_certs {
        print_warning!(
            "skip all certificate validation",
            api,
            "thorctl config --invalid-certs=false"
        );
    } else if invalid_hostnames {
        print_warning!(
            "skip hostname validation",
            api,
            "thorctl config --invalid-hostnames=false"
        );
    } else if !certificate_authorities.is_empty() {
        print_warning!(
            format!(
                "implicitly trust certificate authorities '{:?}'",
                certificate_authorities
            ),
            api,
            "thorctl config --clear-certificate-authorities"
        );
    } else {
        // return immediately if none of the insecure options are set
        return Ok(());
    }
    // ask the user for permission to update Thorctl
    let response = dialoguer::Confirm::new()
        .with_prompt("Continue?:")
        .interact()?;
    if !response {
        // inform the user Thorctl will exit then exit
        println!("Exiting...");
        std::process::exit(0);
    }
    Ok(())
}

/// Print an insecure warning message if a [`CtlConf`] is configured for
/// insecure connections
///
/// # Arguments
///
/// * `conf` - The [`CtlConf`] set when running the command
pub fn warn_insecure_conf(conf: &CtlConf) -> Result<(), Error> {
    warn_insecure(
        &conf.keys.api,
        conf.client.invalid_certs,
        conf.client.invalid_hostnames,
        &conf.client.certificate_authorities,
    )
}

/// Converts a size to bytes
///
/// # Arguments
///
/// * `raw` - A raw storage value
pub fn convert_size_to_bytes(raw: &str) -> Result<u64, Error> {
    // try to cast this directly to a u64
    // This is because we assume that any u64 value is # of bytes
    if let Ok(bytes) = raw.parse::<u64>() {
        // if parse was successful then convert to mebibytes
        return Ok(bytes);
    }
    // u64 failed parse check lets find first occurrence of a any valid char
    let unit_regex = Regex::new(r"[KMGTPE]").unwrap();
    // find index where unit starts
    let reg = match unit_regex.find(&raw) {
        Some(reg) => reg,
        None => return Err(Error::new(format!("Failed to parse unit: {raw}"))),
    };
    // split raw based on where unit was found
    let (amt, unit) = raw.split_at(reg.start());
    // cast amt to u64
    let amt = match amt.parse::<u64>() {
        Ok(amt) => amt,
        Err(error) => {
            return Err(Error::new(format!(
                "Failed to parse amount: {raw} - {error:#?}"
            )))
        }
    };
    // convert to bytes
    let mebibytes = match unit {
        "K" => amt * 1000,
        "M" => amt * 1_000_000,
        "G" => amt * 1_000_000_000,
        "T" => amt * 1_000_000_000_000,
        "P" => amt * 1_000_000_000_000_000,
        "E" => amt * 1_000_000_000_000_000_000,
        "Ki" => amt * 1024,
        "Mi" => amt * 1_048_576,
        "Gi" => amt * 1_073_741_824,
        "Ti" => amt * 1_099_511_627_776,
        "Pi" => amt * 1_125_899_906_842_624,
        "Ei" => amt * 1_152_921_504_606_846_976,
        _ => return Err(Error::new(format!("Failed to parse storage value: {raw}"))),
    };
    Ok(mebibytes)
}

/// Extract a hostname from a url
///
/// # Arguments
///
/// * `url` - The url to get a base from
pub fn get_hostname(url: &str) -> Result<&str, Error> {
    // if this url has a base then strip it out
    let without_base = if url.contains("://") {
        // split on "://" to skip the base
        match url.split("://").nth(1) {
            Some(without_base) => without_base,
            None => return Err(Error::new(format!("Failed to get hostname for {url}"))),
        }
    } else {
        url
    };
    // get just the hostname from our trimmed url
    match without_base.split('/').next() {
        Some(hostname) => Ok(hostname),
        None => Err(Error::new(format!("Failed to get hostname for {url}"))),
    }
}

/// Return a descriptive error that the function requires admin access if we get
/// a 401; otherwise, just return the error
#[macro_export]
macro_rules! err_not_admin {
    ($func:expr) => {
        if let Err(err) = $func {
            if err
                .status()
                .is_some_and(|status| status == http::StatusCode::UNAUTHORIZED)
            {
                return Err(Error::new("You must be an admin to perform this function!"));
            }
            // just return the error if not a 401
            return Err(err);
        }
    };
    ($func:expr, $msg:expr) => {
        if let Err(err) = $func {
            if err
                .status()
                .is_some_and(|status| status == http::StatusCode::UNAUTHORIZED)
            {
                return Err(Error::new(format!("You must be an admin to {}!", $msg)));
            }
            // just return the error if not a 401
            return Err(err);
        }
    };
}
