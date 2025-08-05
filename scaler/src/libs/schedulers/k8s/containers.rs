use k8s_openapi::api::core::v1::{Container, EnvVar, SecurityContext};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use serde_json::json;
use std::collections::BTreeMap;
use thorium::models::{Image, Resources, ScrubbedUser};
use thorium::Error;

use super::MountGen;
use crate::libs::schedulers::Spawned;
use crate::libs::Cache;
use crate::serialize;

// used when casting to a quantity
macro_rules! quantity {
    ($($raw:tt)+) => {serde_json::from_value(json!($($raw)+))}
}

/// K8s API wrappers for containers
pub struct Containers {
    /// The name of the cluster this container will be spawned on
    pub cluster_name: String,
}

impl Containers {
    /// Create a new containers handler
    ///
    /// # Arguments
    ///
    /// * `cluster_name` - The name of this cluster
    pub fn new<T: Into<String>>(cluster_name: T) -> Self {
        Containers {
            cluster_name: cluster_name.into(),
        }
    }
    /// converts a resource request to a BTreeMap
    ///
    /// This will ignore any value that is None
    ///
    /// # Arguments
    ///
    /// * `raw` - The resource request to convert
    fn request_conv(raw: &Resources) -> Result<BTreeMap<String, Quantity>, Error> {
        // creat btreemap of requests
        let mut btree = BTreeMap::default();
        // build the resource request map
        btree.insert("cpu".to_owned(), quantity!(format!("{}m", raw.cpu))?);
        btree.insert("memory".to_owned(), quantity!(format!("{}Mi", raw.memory))?);
        if raw.ephemeral_storage > 0 {
            btree.insert(
                "ephemeral-storage".to_owned(),
                quantity!(format!("{}Mi", raw.ephemeral_storage))?,
            );
        }
        Ok(btree)
    }

    /// converts a resource limit request to a BTreeMap
    ///
    /// This will ignore any value that is None
    ///
    /// # Arguments
    ///
    /// * `raw` - The resource request to convert
    fn limit_conv(raw: &Resources) -> Result<BTreeMap<String, Quantity>, Error> {
        // creat btreemap of limits
        let mut btree = BTreeMap::default();
        // build the resource memory map
        btree.insert("cpu".to_owned(), quantity!(format!("{}m", raw.cpu))?);
        btree.insert("memory".to_owned(), quantity!(format!("{}Mi", raw.memory))?);
        // inject ephemeral storage if its greater than 0
        if raw.ephemeral_storage > 0 {
            btree.insert(
                "ephemeral-storage".to_owned(),
                quantity!(format!("{}Mi", raw.ephemeral_storage))?,
            );
        }
        // inject nvidia gpu if its greater than 0
        if raw.nvidia_gpu > 0 {
            btree.insert(
                "nvidia/gpu".to_owned(),
                quantity!(raw.nvidia_gpu.to_string())?,
            );
        }
        // inject amd gpu if its greater than 0
        if raw.amd_gpu > 0 {
            btree.insert("amd/gpu".to_owned(), quantity!(raw.amd_gpu.to_string())?);
        }
        Ok(btree)
    }

    /// Builds a K8s environment variable
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the environment variable
    /// * `value` - The value to set for this environment variable
    fn build_env_var<T: Into<String>>(name: T, value: &Option<String>) -> EnvVar {
        // build environment variables
        EnvVar {
            name: name.into(),
            value: value.clone(),
            ..Default::default()
        }
    }

    /// Builds a container specific security context
    ///
    /// # Arguments
    ///
    /// * `image` - The details for this container image in Thorium
    fn build_security_context(image: &Image) -> SecurityContext {
        // build this containers security context
        SecurityContext {
            allow_privilege_escalation: Some(image.security_context.allow_privilege_escalation),
            ..Default::default()
        }
    }

    /// Generate the container struct
    ///
    /// # Arguments
    ///
    /// * `cache` - The Thorium scalers cache
    /// * `req` - A requisition for a specific image type
    /// * `user` - The user this containers are being spawned for
    pub fn generate(
        &self,
        cache: &Cache,
        spawn: &Spawned,
        user: &ScrubbedUser,
    ) -> Result<Vec<Container>, Error> {
        // grab our docker info
        let docker = &cache.docker[&spawn.req.group][&spawn.req.stage];
        // grab our image info
        let image = &cache.images[&spawn.req.group][&spawn.req.stage];
        // serialize our docker cmd/entrypoint
        let entrypoint = match &image.args.entrypoint {
            Some(entrypoint) => serialize!(entrypoint),
            None => serialize!(&docker.config.entrypoint),
        };
        let cmd = match &image.args.command {
            Some(cmd) => serialize!(cmd),
            None => serialize!(&docker.config.cmd),
        };
        // build our environment vars
        let mut env: Vec<EnvVar> = image
            .env
            .iter()
            .map(|(name, val)| Self::build_env_var(name, val))
            .collect();
        // only add user specific vars if we aren't overriding the user
        if image.security_context.user.is_none() {
            // add our default environment vars
            env.push(Self::build_env_var("USER", &Some(spawn.req.user.clone())));
            env.push(Self::build_env_var(
                "HOME",
                &Some(format!("/home/{}", &spawn.req.user)),
            ));
        }
        // build container json
        let raw = json!({
            "name": &spawn.req.stage,
            "image": &image.image,
            "command": ["/opt/thorium/thorium-agent"],
            // force pulling this image if there any new layers
            "imagePullPolicy": "Always",
            "env": env,
            "args": [
                "--cluster",
                &self.cluster_name,
                "--group",
                &spawn.req.group,
                "--pipeline",
                &spawn.req.pipeline,
                "--stage",
                &spawn.req.stage,
                "--node",
                &spawn.node,
                "--name",
                &spawn.name,
                "--keys",
                "/opt/thorium-keys/keys.yml",
                "k8s",
                "--entrypoint",
                entrypoint,
                "--cmd",
                cmd
            ],
            "resources": {
                "requests": Self::request_conv(&image.resources)?,
                "limits": Self::limit_conv(&image.resources)?
            },
            "security_context": Self::build_security_context(image),
        });
        // cast to container strcut
        let mut container: Container = serde_json::from_value(raw)?;
        // inject volume mounts
        container.volume_mounts = Some(MountGen::generate(&image, &user)?);
        Ok(vec![container])
    }
}
