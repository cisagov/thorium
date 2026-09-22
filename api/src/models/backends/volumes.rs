use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::bytes::Regex;

use crate::bad;
use crate::models::{HostPath, SystemSettings, User, Volume, VolumeTypes};
use crate::utils::ApiError;

/// Attempts to unwrap opt or returns an error if it's is none
macro_rules! unwrap_opt {
    ($field:expr, $name:expr, $val:expr) => {
        match $field.as_ref() {
            Some(val) => val,
            None => {
                return bad!(format!(
                    "{} volumes require {} options to be set",
                    $name, $val
                ));
            }
        }
    };
}

/// A [`regex::bytes::Regex`] matching any relative traversal in a path (one or more ".'s")
static RELATIVE_DIR_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\A\.+\z").unwrap());

impl HostPath {
    /// Returns true if the given path is a valid host path
    ///
    /// # Arguments
    ///
    /// * `path` - The path to validate
    pub fn is_valid<T: AsRef<Path>>(path: T) -> bool {
        let path: &Path = path.as_ref();
        // the host path is invalid if it's relative or has any relative-like components (".", "..", "...", etc.)
        if path.is_relative()
            || path
                .components()
                .any(|comp| RELATIVE_DIR_REGEX.is_match(comp.as_os_str().as_bytes()))
        {
            return false;
        }
        true
    }
}

impl Volume {
    /// validate a volume contains the information needed
    ///
    /// This does not ensure that it wont fail later on (for instance if a NFS server is wrong).
    ///
    /// # Arguments
    ///
    /// * `user` - The user validating this volume
    /// * `settings` - The Thorium [`SystemSettings`]
    pub fn validate(&self, user: &User, settings: &SystemSettings) -> Result<(), ApiError> {
        // in order to prevent Thorium-created volumes from being leaked ban any volumes that start with "thorium"
        if self.name.starts_with("thorium") {
            return bad!("Volume names cannot start with 'thorium'".to_owned());
        }
        // validate specific options
        match self.archetype {
            VolumeTypes::HostPath => {
                // make sure the host path field is set
                let host_path = unwrap_opt!(self.host_path, "HostPath", "host_path");
                // check if unrestricted host paths is set or the host path is on the whitelist
                if !settings.allow_unrestricted_host_paths
                    && !settings.is_whitelisted_host_path(&host_path.path)
                {
                    // determine what the error message should advise depending on if the user is an admin
                    let admin_error_msg = if user.is_admin() {
                        "Add it to the host path whitelist"
                    } else {
                        "Ask an admin to add it"
                    };
                    return bad!(format!(
                        "The host path '{}' in volume '{}' is not in the list of allowed host paths. \
                        {} or choose a different path.",
                        host_path.path,
                        self.name,
                        admin_error_msg
                    ));
                }
                // make sure the host path is valid
                if !HostPath::is_valid(&host_path.path) {
                    return bad!(format!(
                        "The host path '{}' in volume '{}' is invalid! Host paths must be absolute \
                        and must not contain relative traversal ('.', '..', etc.)",
                        host_path.path,
                        self.name
                    ));
                }
                Ok(())
            }
            VolumeTypes::ConfigMap => Ok(()),
            VolumeTypes::Secret => Ok(()),
            VolumeTypes::NFS => {
                // make sure nfs exists
                let nfs = unwrap_opt!(self.nfs, "NFS", "nfs");
                // an NFS mount lets the pod reach an arbitrary file server, so gate it the
                // same way as host paths — only an admin (or an explicitly unrestricted cluster) may
                // mount an operator-chosen server/path, otherwise any developer could mount internal
                // filers into their container.
                if !settings.allow_unrestricted_host_paths && !user.is_admin() {
                    // determine what the error message should advise depending on if the user is an admin
                    let admin_error_msg = if user.is_admin() {
                        "Enable unrestricted host paths"
                    } else {
                        "Ask an admin to create it"
                    };
                    return bad!(format!(
                        "NFS volume '{}' points at server '{}' path '{}', which only an admin may \
                        mount. {}.",
                        self.name, nfs.server, nfs.path, admin_error_msg
                    ));
                }
                // make sure the export path is an absolute path with no relative traversal
                if !HostPath::is_valid(&nfs.path) {
                    return bad!(format!(
                        "The NFS export path '{}' in volume '{}' is invalid! Export paths must be \
                        absolute and must not contain relative traversal ('.', '..', etc.)",
                        nfs.path, self.name
                    ));
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;
    use crate::models::{NFS, UserRole, UserSettings};

    /// Build a minimal non-admin developer user for volume validation tests
    ///
    /// `Volume::validate` only consults the user to phrase error messages, so this fills the
    /// remaining fields with inert defaults and gives the user the non-admin Developer role.
    fn dev_user() -> User {
        User {
            username: "dev".to_owned(),
            password: Some("hash".to_owned()),
            email: "dev@dev.gov".to_owned(),
            role: UserRole::Developer {
                k8s: true,
                bare_metal: false,
                windows: false,
                external: false,
                kvm: false,
            },
            groups: vec!["dev-group".to_owned()],
            actual_groups: Vec::default(),
            token: "token".to_owned(),
            token_expiration: chrono::Utc::now(),
            unix: None,
            settings: UserSettings::default(),
            verified: true,
            verification_token: None,
            verification_sent: None,
            aliases: HashMap::default(),
            image: None,
        }
    }

    #[test]
    fn nfs_volume_with_arbitrary_server_is_rejected() {
        // build a non-admin developer that a group membership would let create images
        let user = dev_user();
        // use default settings, which whitelist no host paths and allow no unrestricted mounts
        let settings = SystemSettings::default();
        // craft an NFS volume pointing at an arbitrary internal filer with write access
        let mut volume = Volume::new("data", "/mnt/loot", VolumeTypes::NFS);
        // attach the attacker-chosen NFS server and export path
        volume.nfs = Some(NFS {
            server: "10.0.5.20".to_owned(),
            path: "/exports/backups".to_owned(),
        });
        // an unvalidated NFS mount from a non-admin developer must be rejected, mirroring
        // the host-path whitelist rather than accepting any server/path the user supplies
        assert!(
            volume.validate(&user, &settings).is_err(),
            "NFS volume with an arbitrary server/path was accepted for a non-admin"
        );
    }

    #[test]
    fn test_host_path_validate() {
        let mut path = PathBuf::from("relative/path");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from("..");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from(".");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from("");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from("/absolute/../but/traverses/upward");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from("/absolute/......../arbitrary/number/dots");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from("/😎/../");
        assert!(!HostPath::is_valid(path));
        path = PathBuf::from("/😎/valid/");
        assert!(HostPath::is_valid(path));
        path = PathBuf::from("/.valid/path");
        assert!(HostPath::is_valid(path));
        path = PathBuf::from("/...........valid/path");
        assert!(HostPath::is_valid(path));
    }
}
