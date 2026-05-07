//! The arguments to pass to the Thorium node reactor daemon

use clap::Parser;
use std::{
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
};
use thorium::{Error, models::ImageScaler};

/// Command line args
#[derive(Parser, Debug, Clone)]
#[clap(version, author)]
pub struct Args {
    /// The path to the keys to use for this node daemon
    #[clap(short, long, default_value = "keys.yml")]
    pub keys: String,
    /// The path to use for the tracing config for this deamon
    #[clap(short, long, default_value = "/opt/thorium/tracing.yml")]
    pub trace: String,
    /// The directory where the Thorium agents binaries are located in
    ///
    /// The agents for Linux and Windows should be named "thorium-agent" and
    /// "thorium-agent.exe", respectively.
    #[clap(short, long, default_value = "/opt/thorium")]
    pub agents_dir: String,
    /// The scaler this reactor should spawn jobs for
    #[clap(short, long)]
    pub scaler: ImageScaler,
    /// The name of the cluster this node is in
    #[clap(short, long)]
    pub cluster: String,
    /// The name of the cluster this node is in
    #[clap(short, long)]
    pub name: Option<String>,
    /// The amount of time in seconds to wait between reactor loops
    #[clap(short, long, default_value_t = NonZeroU64::new(2).unwrap())]
    pub dwell_time: NonZeroU64,
    /// Skip the update check for the reactor and agents
    ///
    /// This is particularly helpful when developing/testing. Note that agents will not run
    /// if their versions do not match the API they are talking to
    #[clap(long)]
    pub skip_update: bool,
    /// Skip the update check for just the reactor
    ///
    /// This is particularly helpful when developing/testing. Agent binaries will be updated
    /// automatically, but the reactor will not
    #[clap(long)]
    pub skip_self_update: bool,
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
                Err(bad_hostname) => Err(Error::new(format!(
                    "Error getting node's hostname: hostname is not valid UTF-8: '{}'",
                    bad_hostname.display()
                ))),
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
    Kvm(KvmArgs),
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
pub struct KvmArgs {
    /// Base directory for KVM storage
    #[clap(short, long)]
    base_dir: PathBuf,
    /// Base directory for VM's containing golden qcow2 disk files and XML definitions for VMs
    /// organized by group/image (e.g. `<GOLDEN_DIR>/<GROUP>/<IMAGE>/<IMAGE>.qcow2`)
    ///
    /// By default, the reactor will look in `<BASE_DIR>/golden`. It's recommended that the temp
    /// directory and the base directory from which golden images are retrieved are on the same
    /// file system.
    #[clap(short, long)]
    golden_dir: Option<PathBuf>,
    /// Where to write temporary qcow2 files for running VM's
    ///
    /// By default, the files are written to `<BASE_DIR>/tmp/`. It's recommended that the temp
    /// directory and the base directory from which golden images are retrieved are on the same
    /// file system.
    #[clap(short, long)]
    temp_dir: Option<PathBuf>,
    /// The socket to connect to our libvirt/kvm daemon at
    #[clap(short, long, default_value = "qemu:///system")]
    socket: String,
    /// The number of threads to allocate to the libvirt threadpool
    ///
    /// The libvirt threadpool is used for blocking calls to libvirt to avoid blocking the
    /// async executor.
    #[clap(long, default_value_t = NonZeroUsize::new(10).unwrap())]
    virt_threads: NonZeroUsize,
    /// The size of the cache for detecting the OS VM disk images
    ///
    /// The cache size should be about 2x the number of KVM images you have;
    /// cache entries are very small, so memory should not be a concern.
    #[clap(long, default_value_t = NonZeroUsize::new(1000).unwrap())]
    os_detector_cache_size: NonZeroUsize,
}

impl From<KvmArgs> for Kvm {
    fn from(args: KvmArgs) -> Self {
        let golden_dir = args
            .golden_dir
            .unwrap_or_else(|| args.base_dir.join("golden"));
        let temp_dir = args.temp_dir.unwrap_or_else(|| args.base_dir.join("tmp"));
        Kvm {
            base_dir: args.base_dir,
            golden_dir,
            temp_dir,
            socket: args.socket,
            virt_threads: args.virt_threads,
            os_detector_cache_size: args.os_detector_cache_size,
        }
    }
}

/// The resolved KVM args with computed paths to avoid unwrapping options
/// over and over
pub struct Kvm {
    /// Base directory for KVM storage
    pub base_dir: PathBuf,
    /// Base directory for VM's containing golden qcow2 disk files and XML definitions for VMs
    /// organized by group/image (e.g. `<GOLDEN_DIR>/<GROUP>/<IMAGE>/<IMAGE>.qcow2`)
    pub golden_dir: PathBuf,
    /// Where to write temporary qcow2 files for running VM's
    pub temp_dir: PathBuf,
    /// The socket to connect to our libvirt/kvm daemon at
    pub socket: String,
    /// The number of threads to allocate to the libvirt threadpool
    ///
    /// The libvirt threadpool is used for blocking calls to libvirt to avoid blocking the
    /// async executor.
    pub virt_threads: NonZeroUsize,
    /// The size of the cache for detecting the OS VM disk images
    pub os_detector_cache_size: NonZeroUsize,
}
