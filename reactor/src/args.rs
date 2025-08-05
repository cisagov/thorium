//! The arguments to pass to the Thorium node reactor daemon

use clap::Parser;
use std::path::PathBuf;
use thorium::{models::ImageScaler, Error};

/// Command line args
#[derive(Parser, Debug, Clone)]
#[clap(version, author)]
pub struct Args {
    /// The path to the keys to use for this node daemon
    #[clap(short, long, default_value = "keys.yml")]
    pub keys: String,
    /// The path to use for the tracing config for this daemon
    #[clap(short, long, default_value = "/opt/thorium/tracing.yml")]
    pub trace: String,
    /// The scaler this reactor should spawn jobs for
    #[clap(short, long)]
    pub scaler: ImageScaler,
    /// The name of the cluster this node is in
    #[clap(short, long)]
    pub cluster: String,
    /// The name of the cluster this node is in
    #[clap(short, long)]
    pub name: Option<String>,
    /// the different scaler types to spawn jobs for
    #[clap(subcommand)]
    pub launchers: Launchers,
}

impl Args {
    /// Get this nodes hostname
    pub fn node(&self) -> Result<String, Error> {
        match &self.name {
            Some(name) => Ok(name.clone()),
            None => match gethostname::gethostname().into_string() {
                Ok(hostname) => Ok(hostname),
                Err(err) => {
                    return Err(Error::new(format!(
                        "Failed to get hostname with {:#?}",
                        err
                    )))
                }
            },
        }
    }
}

/// The different scaler types to spawn jobs for
#[derive(Parser, Debug, Clone)]
pub enum Launchers {
    /// Spawn jobs the current bare metal node
    #[cfg(target_os = "linux")]
    #[clap(version, author)]
    BareMetal,
    /// Spawn Windows containers on the current node
    #[cfg(target_os = "windows")]
    #[clap(version, author)]
    Windows,
    #[cfg(feature = "kvm")]
    #[cfg(target_os = "linux")]
    Kvm(Kvm),
}

impl Launchers {
    /// Get the scaler type for our launcher
    pub fn scaler(&self) -> ImageScaler {
        match self {
            #[cfg(target_os = "linux")]
            Launchers::BareMetal => ImageScaler::BareMetal,
            #[cfg(target_os = "windows")]
            Launchers::Windows => ImageScaler::Windows,
            #[cfg(feature = "kvm")]
            #[cfg(target_os = "linux")]
            Launchers::Kvm(_) => ImageScaler::Kvm,
        }
    }
}

/// Spawn KVM based vms on the current node
#[derive(Parser, Debug, Clone)]
#[clap(version, author)]
pub struct Kvm {
    /// The socket to connect to our libvirt/kvm daemon at
    #[clap(short, long, default_value = "qemu:///system")]
    pub socket: String,
    /// Where to write our temp qcow2 images and isos too
    #[clap(short, long, default_value = "/tmp/qcow2")]
    pub temp: PathBuf,
}
