//! Structs for volumes within kubernetes to be bound to pods created by Thorium

use std::fmt;

/// Different types of [`HostPath`] volumes in k8s
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum HostPathTypes {
    DirectoryOrCreate,
    Directory,
    FileOrCreate,
    File,
    Socket,
    CharDevice,
    BlockDevice,
}

impl fmt::Display for HostPathTypes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            HostPathTypes::DirectoryOrCreate => write!(f, "DirectoryOrCreate"),
            HostPathTypes::Directory => write!(f, "Directory"),
            HostPathTypes::FileOrCreate => write!(f, "FileOrCreate"),
            HostPathTypes::File => write!(f, "File"),
            HostPathTypes::Socket => write!(f, "Socket"),
            HostPathTypes::CharDevice => write!(f, "CharDevice"),
            HostPathTypes::BlockDevice => write!(f, "BlockDevice"),
        }
    }
}

/// Specific arguments for a host path volume in k8s
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct HostPath {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_type: Option<HostPathTypes>,
}

/// Specific Arguments for a config map in k8s
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ConfigMap {
    /// The mode bits to set on files in this volume
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_mode: Option<i32>,
    /// Whether this configmap is optional or not
    pub optional: Option<bool>,
}

/// Specific Arguments for a secret in k8s
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Secret {
    /// The mode bits to set on files in this volume
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_mode: Option<i32>,
    /// Whether this secret is optional or not
    pub optional: Option<bool>,
}

/// Specific Arguments for a NFS mount in k8s
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NFS {
    /// The path that is exported by the NFS server
    pub path: String,
    /// The host/ip:port of the NFS server
    pub server: String,
}

/// Helps default a serde value to false
// TODO: remove this when https://github.com/serde-rs/serde/issues/368 is resolved
fn default_as_false() -> bool {
    false
}

/// Different types of volumes supported by Thorium
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum VolumeTypes {
    /// HostPath volumes
    HostPath,
    /// ConfigMap volumes
    ConfigMap,
    /// Secret volumes
    Secret,
    /// NFS volumes
    NFS,
}

impl fmt::Display for VolumeTypes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VolumeTypes::HostPath => write!(f, "host_path"),
            VolumeTypes::ConfigMap => write!(f, "config_map"),
            VolumeTypes::Secret => write!(f, "secret"),
            VolumeTypes::NFS => write!(f, "nfs"),
        }
    }
}

/// A volume to bind in of arbitrary type
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Volume {
    /// The name of the volume in k8s
    pub name: String,
    /// The type of volume this is
    pub archetype: VolumeTypes,
    /// Where this should be mounted at in the pod
    pub mount_path: String,
    /// A sub path for mounting specific files
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_path: Option<String>,
    /// Whether this volume should be readonly or not
    #[serde(default = "default_as_false")]
    pub read_only: bool,
    /// whether to use the most recent config created by kustomize
    #[serde(default = "default_as_false")]
    pub kustomize: bool,
    // Specific options for all the different types of volumes
    /// Host path settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_path: Option<HostPath>,
    // Config map settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_map: Option<ConfigMap>,
    /// Secret settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret: Option<Secret>,
    /// NFS settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nfs: Option<NFS>,
}

impl Volume {
    /// Creates a new barebones volume without any archetype specific settings
    ///
    /// The name of the volume should match the name of the volume already created in k8s if a
    /// volume must already exist in k8s in order for this volume to be bound. When kustomize
    /// support is enabled then it must match the prefix of your kustomize created thing.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the volume
    /// * `mount_path` - The path this volume should be bound at in the pod
    /// * `archetype` - What type of volume this is
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::ConfigMap);
    /// ```
    pub fn new<T, S>(name: T, mount_path: S, archetype: VolumeTypes) -> Self
    where
        T: Into<String>,
        S: Into<String>,
    {
        let vol = Volume {
            name: name.into(),
            mount_path: mount_path.into(),
            sub_path: None,
            archetype: archetype.clone(),
            read_only: false,
            kustomize: false,
            host_path: None,
            config_map: None,
            secret: None,
            nfs: None,
        };
        // based on our archetype use the correct defaults
        match archetype {
            // host paths assume the mount/bind paths are the same by default
            VolumeTypes::HostPath => {
                let mount_path = vol.mount_path.clone();
                vol.host_path(mount_path, None)
            }
            _ => vol,
        }
    }

    /// Sets the sub path that should be used when binding in a volume
    ///
    /// This how you can bind in specific files from a volume.
    ///
    /// # Arguments
    ///
    /// * `sub_path` - The path to the file to bind in from the volume
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::ConfigMap)
    ///     .sub_path("conf.yml");
    /// ```
    #[must_use]
    pub fn sub_path<T: Into<String>>(mut self, sub_path: T) -> Self {
        self.sub_path = Some(sub_path.into());
        self
    }

    /// Whether this volume should be read only or not
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::ConfigMap)
    ///     .read_only();
    /// ```
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Whether this volume binds in a volume created by Kustomize
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::ConfigMap)
    ///     .kustomize();
    /// ```
    #[must_use]
    pub fn kustomize(mut self) -> Self {
        self.kustomize = true;
        self
    }

    /// Set [`HostPath`] specific settings for this [`Volume`]
    ///
    /// # Arguments
    ///
    /// * `path` - The path on the host to mount
    /// * `type_` - The type of the host path volume to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes, HostPathTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::HostPath)
    ///     .host_path("/mnt/share", Some(HostPathTypes::FileOrCreate));
    /// ```
    #[must_use]
    pub fn host_path<T: Into<String>>(mut self, path: T, type_: Option<HostPathTypes>) -> Self {
        // setup ConfigMap specific settings
        self.host_path = Some(HostPath {
            path: path.into(),
            path_type: type_,
        });
        self
    }

    /// Set [`ConfigMap`] specific settings for this [`Volume`]
    ///
    /// # Arguments
    ///
    /// * `default_mode` - The permissions of the files in this volume (must be an octal)
    /// * `optional` - Whether this volume is required to spawn images with this volume
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::ConfigMap)
    ///     .config_map(Some(0400), Some(true));
    /// ```
    #[must_use]
    pub fn config_map(mut self, default_mode: Option<i32>, optional: Option<bool>) -> Self {
        // setup ConfigMap specific settings
        self.config_map = Some(ConfigMap {
            default_mode,
            optional,
        });
        self
    }

    /// Set [`Secret`] specific settings for this [`Volume`]
    ///
    /// # Arguments
    ///
    /// * `default_mode` - The permissions of the files in this volume (must be an octal)
    /// * `optional` - Whether this volume is required to spawn images with this volume
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::ConfigMap)
    ///     .secret(Some(0400), Some(true));
    /// ```
    #[must_use]
    pub fn secret(mut self, default_mode: Option<i32>, optional: Option<bool>) -> Self {
        // setup secret specific settings
        self.secret = Some(Secret {
            default_mode,
            optional,
        });
        self
    }

    /// Set [`NFS`] specific settings for this [`Volume`]
    ///
    /// # Arguments
    ///
    /// * `server` - The hostname or ip for the server that is hosting the NFS share
    /// * `path` - The path to use in order to locate the correct share on the server
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Volume, VolumeTypes};
    ///
    /// Volume::new("conf", "/path", VolumeTypes::NFS)
    ///     .nfs("nfs-server", "storage");
    /// ```
    #[must_use]
    pub fn nfs<T: Into<String>>(mut self, server: T, path: T) -> Self {
        // setup nfs specific settings
        let nfs = NFS {
            server: server.into(),
            path: path.into(),
        };
        self.nfs = Some(nfs);
        self
    }
}
