use k8s_openapi::api::core::v1::{
    ConfigMapVolumeSource, EmptyDirVolumeSource, HostPathVolumeSource, NFSVolumeSource,
    SecretVolumeSource, Volume, VolumeMount,
};
use serde_json::json;
use thorium::models::{Image, ScrubbedUser, Volume as ThoriumVolume, VolumeTypes};
use thorium::{Conf, Error};

use super::{ConfigMaps, Secrets};

/// extract settings for a volume type or return an error
macro_rules! extract {
    ($field:expr, $name:expr) => {
        match $field {
            Some(val) => val,
            None => {
                return Err(Error::new(format!(
                    "Volume settings for {} must not be none",
                    $name
                )))
            }
        }
    };
}

/// extracts the name of an object or returns an error
macro_rules! name {
    ($object:expr, $vol:expr) => {
        match $object {
            Some(obj) => Ok(obj.metadata.name.unwrap()),
            None => {
                return Err(Error::new(format!(
                    "Could not find kustomize volume prefixed with {}",
                    $vol.name
                )))
            }
        }
    };
}

pub struct Volumes {
    /// API for listing secrets
    secrets: Secrets,
    /// API for listing config_maps
    config_maps: ConfigMaps,
}

impl Volumes {
    /// Creates a new [`Volumes`] wrapper
    ///
    /// # Arguments
    ///
    /// * `client` - A Kubernetes client
    /// * `conf` - A Thorium config
    /// * `context_name` - The name of this context
    pub fn new(client: &kube::Client, conf: &Conf, context_name: &str) -> Self {
        // build Secrets API
        let secrets = Secrets::new(client, conf, context_name);
        // build ConfigMaps API
        let config_maps = ConfigMaps::new(client);
        Volumes {
            secrets,
            config_maps,
        }
    }

    /// Generate all volumes needed for a container
    ///
    /// # Arguments
    ///
    /// * `image` - The image to generate volumes for
    /// * `user` - The user whose volumes we are binding in
    pub async fn generate(&self, image: &Image, user: &ScrubbedUser) -> Result<Vec<Volume>, Error> {
        // start with default Thorium volumes
        let mut volumes = Vec::default();
        volumes.push(Self::thorium()?);
        volumes.push(Self::home());
        volumes.push(Self::scratch());
        volumes.push(Self::keys(&user.username)?);
        // if this user has unix info then bind in a valid etc/passwd file
        if user.unix.is_some() {
            volumes.push(Self::passwd(&user.username)?);
        }
        // inject any user requested volumes
        for vol in image.volumes.iter() {
            volumes.push(self.build(vol, &image.group).await?);
        }
        Ok(volumes)
    }

    /// Build the base volume object
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this volume
    fn base(name: &str) -> Volume {
        // creat base volume
        Volume {
            name: name.to_owned(),
            ..Default::default()
        }
    }

    /// Setup the shared Thorium volume
    fn thorium() -> Result<Volume, serde_json::Error> {
        // build base volume
        let mut vol = Self::base("thorium");
        // inject host path in
        vol.host_path = Some(HostPathVolumeSource {
            path: "/opt/thorium".to_owned(),
            type_: None,
        });
        Ok(vol)
    }

    /// Setup an empty home dir
    fn home() -> Volume {
        // build base volume
        let mut vol = Self::base("thorium-home");
        // inject host path in
        vol.empty_dir = Some(EmptyDirVolumeSource::default());
        vol
    }

    /// Setup an empty home dir
    fn scratch() -> Volume {
        // build base volume
        let mut vol = Self::base("thorium-scratch");
        // inject host path in
        vol.empty_dir = Some(EmptyDirVolumeSource::default());
        vol
    }

    /// Setup the Thorium keys volume
    ///
    /// # Arguments
    ///
    /// * `user` - The username of the user whose keys we are binding in
    fn keys(username: &str) -> Result<Volume, serde_json::Error> {
        // build name of the secret
        let name = format!("thorium-{}-keys", username);
        // build base volume
        let mut vol = Self::base(&name);
        // create secret source
        let secret = SecretVolumeSource {
            secret_name: Some(name),
            // set the rest of the fields to default
            ..Default::default()
        };
        // inject source into volume
        vol.secret = Some(secret);
        Ok(vol)
    }

    /// Setup the Thorium passwd volume
    ///
    /// # Arguments
    ///
    /// * `user` - The username of the user whose keys we are binding in
    fn passwd(username: &str) -> Result<Volume, serde_json::Error> {
        // build name of the secret
        let name = format!("thorium-{}-passwd", username);
        // build base volume
        let mut vol = Self::base(&name);
        // create config map source
        let config = ConfigMapVolumeSource {
            name: name,
            // set the rest of the fields to default
            ..Default::default()
        };
        // inject source into volume
        vol.config_map = Some(config);
        Ok(vol)
    }

    /// Supports kustomize generated Secrets/ConfigMaps by using the must recent version
    ///
    /// This is a very naive approach and could benefit from some form of caching.
    ///
    /// # Arguments
    ///
    /// * `vol` - The Thorium volume to try to support
    async fn kustomize_support(&self, vol: &ThoriumVolume, group: &str) -> Result<String, Error> {
        // if kustomize support is disabled just return the volume name
        if !vol.kustomize {
            return Ok(vol.name.clone());
        }

        // try and find the latest secret or config map
        match vol.archetype {
            VolumeTypes::ConfigMap => name!(self.config_maps.latest(&vol.name, group).await?, vol),
            VolumeTypes::Secret => name!(self.secrets.latest(&vol.name, group).await?, vol),
            _ => Err(Error::new(format!(
                "Volume {} is not supported with kustomize",
                vol.name
            ))),
        }
    }

    /// Builds an host path [`Volume`] based on a [`thorium::models::Volume`]
    ///
    /// # Arguments
    ///
    /// * `built` - The volume to inject host path settings into
    /// * `vol` - The requested settings for this volume from Thorium
    fn host_path(mut built: Volume, vol: &ThoriumVolume) -> Result<Volume, Error> {
        // get the path type if host_path settings exist
        let path_type = match &vol.host_path {
            // if settings exist try to unwrap and cast path_type to a string
            Some(settings) => settings.path_type.as_ref().map(ToString::to_string),
            None => None,
        };

        // inject nfs settings in
        built.host_path = Some(HostPathVolumeSource {
            path: extract!(&vol.host_path, "host_path").path.clone(),
            type_: path_type,
        });
        Ok(built)
    }

    /// Builds an ConfigMap [`Volume`] based on a [`thorium::models::Volume`]
    ///
    /// # Arguments
    ///
    /// * `built` - The volume to inject ConfigMap settings into
    /// * `vol` - The requested settings for this volume from Thorium
    /// * `ns` - The namespace this volume will be in
    async fn config_map(
        &self,
        mut built: Volume,
        vol: &ThoriumVolume,
        ns: &str,
    ) -> Result<Volume, Error> {
        // if read_only is set then force default_mode to that otherwise use default mode
        let default_mode = match vol.read_only {
            // if read only is set then force the mode to the octal 0444
            true => Some(0o444),
            // otherwise use default mode in specific settings
            _ => match &vol.config_map {
                Some(settings) => settings.default_mode,
                None => None,
            },
        };

        // build required config_map settings in
        let mut cmap = ConfigMapVolumeSource {
            default_mode,
            name: self.kustomize_support(vol, ns).await?,
            items: None,
            optional: None,
        };
        // insert anything from the optional specific settings
        if let Some(specific) = vol.config_map.as_ref() {
            // set if this volume is optional
            cmap.optional = specific.optional
        }
        // inject volume
        built.config_map = Some(cmap);
        Ok(built)
    }

    /// Builds an Secret [`Volume`] based on a [`thorium::models::Volume`]
    ///
    /// # Arguments
    ///
    /// * `built` - The volume to inject Secret settings into
    /// * `vol` - The requested settings for this volume from Thorium
    /// * `ns` - The namespace this volume will be in
    async fn secret(
        &self,
        mut built: Volume,
        vol: &ThoriumVolume,
        ns: &str,
    ) -> Result<Volume, Error> {
        // if read_only is set then force default_mode to that otherwise use default mode
        let default_mode = match vol.read_only {
            // if read only is set then force the mode to the octal 0444
            true => Some(0o444),
            _ => match &vol.secret {
                Some(settings) => settings.default_mode,
                None => None,
            },
        };
        // build required config_map settings in
        let mut secret = SecretVolumeSource {
            default_mode,
            secret_name: Some(self.kustomize_support(vol, ns).await?),
            items: None,
            optional: None,
        };
        // copy over the optional flag from the volumes specific settings
        Self::apply_secret_optional(&mut secret, vol);
        // inject volume
        built.secret = Some(secret);
        Ok(built)
    }

    /// Copies the `optional` flag from a Thorium volumes settings onto a k8s secret source
    ///
    /// This is the inline flag copy in [`Volumes::secret`] (it never needs `self`) extracted so
    /// its source-of-truth logic can be unit tested without a live kube client. The flag is read
    /// from the volumes secret settings.
    ///
    /// # Arguments
    ///
    /// * `secret` - The k8s secret volume source to set the optional flag on
    /// * `vol` - The requested settings for this volume from Thorium
    fn apply_secret_optional(secret: &mut SecretVolumeSource, vol: &ThoriumVolume) {
        // insert anything from the secret specific settings
        if let Some(specific) = vol.secret.as_ref() {
            // set if this volume is optional
            secret.optional = specific.optional
        }
    }

    /// Builds an NFS [`Volume`] based on a [`thorium::models::Volume`]
    ///
    /// # Arguments
    ///
    /// * `built` - The volume to inject NFS settings into
    /// * `vol` - The requested settings for this volume from Thorium
    fn nfs(mut built: Volume, vol: &ThoriumVolume) -> Result<Volume, Error> {
        // inject nfs settings in
        built.nfs = Some(NFSVolumeSource {
            path: extract!(&vol.nfs, "nfs").path.clone(),
            read_only: Some(vol.read_only),
            server: extract!(&vol.nfs, "nfs").server.clone(),
        });
        Ok(built)
    }

    /// builds a [`Volume`] based on a [`thorium::models::Volume`]
    ///
    /// # Arguments
    ///
    /// * `vol` - A volume request from Thorium
    /// * `ns` - The namespace this volume will be in
    pub async fn build(&self, vol: &ThoriumVolume, ns: &str) -> Result<Volume, Error> {
        // build base volume
        let built = Self::base(&vol.name);
        // inject the correct volume settings
        match &vol.archetype {
            VolumeTypes::HostPath => Self::host_path(built, vol),
            VolumeTypes::ConfigMap => self.config_map(built, vol, ns).await,
            VolumeTypes::Secret => self.secret(built, vol, ns).await,
            VolumeTypes::NFS => Self::nfs(built, vol),
        }
    }
}

/// Generate volume mounts for Thorium
pub struct MountGen;

impl MountGen {
    /// Generate all volume binds for a container
    ///
    /// Currently this only generates default volume binds
    ///
    /// # Arguments
    ///
    /// * `image` - The image we are creating mounts for
    /// * `user` - The username of the user whose keys we are binding in
    pub fn generate(
        image: &Image,
        user: &ScrubbedUser,
    ) -> Result<Vec<VolumeMount>, serde_json::Error> {
        // start with default thorium_volume
        let mut mounts = Vec::default();
        mounts.push(Self::thorium()?);
        mounts.push(Self::home(&user.username)?);
        mounts.push(Self::scratch()?);
        mounts.push(Self::keys(&user.username)?);
        // if this user has unix info then bind in a valid etc/passwd file
        if user.unix.is_some() {
            mounts.push(Self::passwd(&user.username)?);
        }
        // build all user specified mounts
        image
            .volumes
            .iter()
            .for_each(|vol| mounts.push(Self::build(vol)));
        Ok(mounts)
    }

    /// Setup Thorium shared volume bind
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user whose keys we are binding in
    fn thorium() -> Result<VolumeMount, serde_json::Error> {
        // use the camelCase keys k8s expects so the read-only flag is not silently dropped
        // on deserialization (snake_case `read_only` deserializes to None, leaving the mount
        // writable and letting a tool trojan the shared agent binary)
        serde_json::from_value(json!({
            "name": "thorium",
            "mountPath": "/opt/thorium",
            "readOnly": true,
            "recursiveReadOnly": "Enabled"

        }))
    }

    /// Setup home dir volume bind
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user whose keys we are binding in
    fn home(user: &str) -> Result<VolumeMount, serde_json::Error> {
        serde_json::from_value(json!({
            "name": "thorium-home",
            "mountPath": format!("/home/{}", user),
        }))
    }

    /// Setup Thorium shared volume bind
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user whose keys we are binding in
    fn scratch() -> Result<VolumeMount, serde_json::Error> {
        serde_json::from_value(json!({
            "name": "thorium-scratch",
            "mountPath": "/tmp",
        }))
    }

    /// Setup Thorium keys volume bind
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user whose keys we are binding in
    fn keys(username: &str) -> Result<VolumeMount, serde_json::Error> {
        serde_json::from_value(json!({
            "name": format!("thorium-{}-keys", username),
            "mountPath": "/opt/thorium-keys"
        }))
    }

    /// Setup /etc/passwd volume bind
    ///
    /// # Arguments
    ///
    /// * `username` - The username of the user whose /etc/passwd we are binding in
    fn passwd(username: &str) -> Result<VolumeMount, serde_json::Error> {
        serde_json::from_value(json!({
            "name": format!("thorium-{}-passwd", username),
            "mountPath": "/etc/passwd",
            "subPath": "passwd"
        }))
    }

    /// Build mounts for a user specified volume
    ///
    /// # Arguments
    ///
    /// * `vol` - A volume to add to a pod in k8s
    fn build(vol: &ThoriumVolume) -> VolumeMount {
        // set recursive read only correctly based on our volume settings
        let recursive_read_only = if vol.read_only {
            // this volume is supposed to be read only so lets enforce this
            // recusively if possible
            Some("IfPossible".to_owned())
        } else {
            // this volume is not read only
            None
        };
        VolumeMount {
            name: vol.name.clone(),
            mount_path: vol.mount_path.clone(),
            read_only: Some(vol.read_only),
            sub_path: vol.sub_path.clone(),
            // Thorium doesn't support these two current just ignore them
            sub_path_expr: None,
            mount_propagation: None,
            recursive_read_only,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MountGen, Volumes};
    use k8s_openapi::api::core::v1::SecretVolumeSource;
    use thorium::models::{Secret, Volume as ThoriumVolume, VolumeTypes};

    /// the shared `/opt/thorium` volume (which holds the `thorium-agent` binary) must be
    /// mounted read-only in every tool pod. Otherwise a root-in-container tool can overwrite the
    /// node's agent binary and trojan every subsequent tenant's workload on that node.
    #[test]
    fn thorium_mount_is_read_only() {
        // build the shared thorium volume mount exactly as the scheduler injects it
        let mount = MountGen::thorium().expect("thorium mount should deserialize");
        // the mount must actually be read-only after deserialization
        assert_eq!(
            mount.read_only,
            Some(true),
            "the /opt/thorium mount is not read-only, so a tool can trojan the agent binary"
        );
        // the recursive read-only flag must also survive deserialization under its camelCase key
        assert_eq!(
            mount.recursive_read_only.as_deref(),
            Some("Enabled"),
            "the /opt/thorium mount is not recursively read-only"
        );
    }

    /// The built `SecretVolumeSource.optional` reflects `vol.secret.optional`, not the
    /// `vol.config_map` settings (which are `None` for a Secret archetype volume), so a user who
    /// marks a Secret volume `optional=true` gets an optional secret rather than a required one.
    #[test]
    fn secret_optional_is_read_from_secret_settings() {
        // build a Secret archetype volume whose secret settings mark it optional
        let mut vol = ThoriumVolume::new("creds", "/creds", VolumeTypes::Secret);
        // the user marked the secret optional so a missing secret does not block the pod
        vol.secret = Some(Secret {
            default_mode: None,
            optional: Some(true),
        });
        // there are no config_map settings on a Secret archetype volume
        vol.config_map = None;
        // build a bare secret source the way Volumes::secret does before copying the flag
        let mut secret = SecretVolumeSource {
            default_mode: None,
            secret_name: Some("creds".to_owned()),
            items: None,
            optional: None,
        };
        // apply the optional flag using the same helper Volumes::secret uses
        Volumes::apply_secret_optional(&mut secret, &vol);
        // the built secret must honor the users optional=true setting
        assert_eq!(
            secret.optional,
            Some(true),
            "a Secret volume marked optional=true must produce optional=Some(true), not be dropped"
        );
    }
}
