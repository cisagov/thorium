use k8s_openapi::api::core::v1::{Container, EnvVar, SecurityContext};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use serde_json::json;
use std::collections::BTreeMap;
use thorium::models::{Image, Resources, ScrubbedUser};
use thorium::{Conf, Error};

use super::MountGen;
use crate::libs::Cache;
use crate::libs::schedulers::Spawned;
use crate::serialize;

// used when casting to a quantity
macro_rules! quantity {
    ($($raw:tt)+) => {serde_json::from_value(json!($($raw)+))}
}

/// K8s API wrappers for containers
pub struct Containers {
    /// The name of the cluster this contianer will be spawned on
    pub cluster_name: String,
    /// The context name for this cluster, used to resolve cluster specific config
    pub context_name: String,
}

impl Containers {
    /// Create a new containers handler
    ///
    /// # Arguments
    ///
    /// * `cluster_name` - The name of this cluster
    /// * `context_name` - The context name for this cluster
    pub fn new<T: Into<String>>(cluster_name: T, context_name: &str) -> Self {
        Containers {
            cluster_name: cluster_name.into(),
            context_name: context_name.to_owned(),
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
        // if this image has burstable resources then add those
        let cpu_burst = raw.cpu.saturating_add(raw.burstable.cpu);
        let memory_burst = raw.memory.saturating_add(raw.burstable.memory);
        // build the resource memory map
        btree.insert("cpu".to_owned(), quantity!(format!("{}m", cpu_burst))?);
        btree.insert(
            "memory".to_owned(),
            quantity!(format!("{}Mi", memory_burst))?,
        );
        // inject ephemeral storage if its greater then 0
        if raw.ephemeral_storage > 0 {
            btree.insert(
                "ephemeral-storage".to_owned(),
                quantity!(format!("{}Mi", raw.ephemeral_storage))?,
            );
        }
        // inject nvidia gpu if its greater then 0
        if raw.nvidia_gpu > 0 {
            btree.insert(
                "nvidia/gpu".to_owned(),
                quantity!(raw.nvidia_gpu.to_string())?,
            );
        }
        // inject amd gpu if its greater then 0
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

    /// Inject the agent's `NO_PROXY` bypass entries into its container environment
    ///
    /// The agent respects the standard proxy environment variables, so to make it bypass any
    /// image/environment proxy for certain hosts we extend its `NO_PROXY`. The Thorium API host is
    /// added when `agent_auto_bypass_proxy_for_api` is set so the agent reaches the API directly
    /// even if the image's proxy can't route to it, alongside any admin configured `agent_no_proxy`
    /// entries. Any `NO_PROXY`/`no_proxy` the image already set is extended in place rather than
    /// overwritten, since reqwest reads the existing variable and a fresh one would shadow it.
    ///
    /// # Arguments
    ///
    /// * `env` - The container environment variables to inject the bypass list into
    /// * `conf` - The Thorium config holding the agent proxy bypass settings and namespace
    /// * `context` - The context name of the cluster the agent is being spawned in
    /// * `docker_env` - The Docker image's baked-in `ENV` (`KEY=VALUE` entries), if known
    fn apply_agent_no_proxy(
        env: &mut Vec<EnvVar>,
        conf: &Conf,
        context: &str,
        docker_env: Option<&[String]>,
    ) {
        let scaler = &conf.thorium.scaler;
        // start with the admin configured raw NO_PROXY entries
        let mut entries = scaler.agent_no_proxy.clone();
        // add the API host so the agent bypasses the proxy for the API when enabled, avoiding
        // duplicating a host the configured entries already cover exactly
        if scaler.agent_auto_bypass_proxy_for_api {
            let api_url = scaler
                .k8s
                .resolved_api_url(context, &conf.thorium.namespace);
            if let Some(host) = thorium::client::conf::no_proxy_host(&api_url)
                && !entries.iter().any(|entry| entry == &host)
            {
                entries.push(host);
            }
        }
        Self::merge_no_proxy(env, docker_env, &entries);
    }

    /// Merge `NO_PROXY` bypass entries into a container's environment variables
    ///
    /// Extends each `NO_PROXY`/`no_proxy` the image already defines (in either casing) with
    /// `entries`, or adds a `NO_PROXY` if it defines none. The image can define these in its
    /// Thorium env config or bake them into the Docker image's `ENV` (`docker_env`); because a k8s
    /// env var overrides the image's `ENV` by name, we fold the image's existing value into the var
    /// we set so we extend it instead of clobbering it. Empty `entries` is a no-op.
    ///
    /// # Arguments
    ///
    /// * `env` - The container environment variables to merge into
    /// * `docker_env` - The Docker image's baked-in `ENV` (`KEY=VALUE` entries), if known
    /// * `entries` - The `NO_PROXY` bypass entries to add
    fn merge_no_proxy(env: &mut Vec<EnvVar>, docker_env: Option<&[String]>, entries: &[String]) {
        // nothing to bypass, so leave the environment as-is
        if entries.is_empty() {
            return;
        }
        let additions = entries.join(",");
        // collect the no_proxy value the image already defines, keyed by its exact variable name so
        // each casing is handled on its own
        let mut image_no_proxy: BTreeMap<String, Option<String>> = BTreeMap::new();
        // lowest precedence: the Docker image's baked-in ENV ("KEY=VALUE" entries)
        for entry in docker_env.unwrap_or_default() {
            if let Some((name, value)) = entry.split_once('=')
                && name.eq_ignore_ascii_case("no_proxy")
            {
                image_no_proxy.insert(name.to_owned(), Some(value.to_owned()));
            }
        }
        // highest precedence: the Thorium env config overrides any Docker ENV value of the same name
        for env_var in env
            .iter()
            .filter(|env_var| env_var.name.eq_ignore_ascii_case("no_proxy"))
        {
            image_no_proxy.insert(env_var.name.clone(), env_var.value.clone());
        }
        // the image defines no no_proxy, so add our own NO_PROXY
        if image_no_proxy.is_empty() {
            env.push(Self::build_env_var("NO_PROXY", &Some(additions)));
            return;
        }
        // extend each no_proxy the image defines with our additions
        for (name, current) in image_no_proxy {
            let merged = match current {
                Some(current) if !current.is_empty() => format!("{current},{additions}"),
                _ => additions.clone(),
            };
            match env.iter_mut().find(|env_var| env_var.name == name) {
                Some(env_var) => env_var.value = Some(merged),
                None => env.push(Self::build_env_var(&name, &Some(merged))),
            }
        }
    }

    /// Builds a container soecific security context
    ///
    /// # Arguments
    ///
    /// * `iamge` - The details for this container image in Thorium
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
    /// * `req` - A requistion for a specific image type
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
        // build our environemnt vars
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
        // inject the agent's NO_PROXY bypass so it can reach the API past any image/env proxy,
        // folding in any NO_PROXY baked into the Docker image so we extend it instead of clobbering
        Self::apply_agent_no_proxy(
            &mut env,
            &cache.conf,
            &self.context_name,
            docker.config.env.as_deref(),
        );
        // get our limbo as a string
        let limbo = cache.conf.thorium.scaler.k8s.limbo.to_string();
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
                "--limbo",
                limbo,
                "k8s",
                "--entrypoint",
                entrypoint,
                "--cmd",
                cmd,
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

#[cfg(test)]
mod tests {
    use super::Containers;
    use k8s_openapi::api::core::v1::EnvVar;

    /// Build an [`EnvVar`] for a test environment
    fn env_var(name: &str, value: Option<&str>) -> EnvVar {
        EnvVar {
            name: name.to_owned(),
            value: value.map(str::to_owned),
            ..Default::default()
        }
    }

    /// Get the value of the named env var, panicking if it is missing
    fn value_of(env: &[EnvVar], name: &str) -> String {
        env.iter()
            .find(|var| var.name == name)
            .unwrap_or_else(|| panic!("missing env var {name}"))
            .value
            .clone()
            .unwrap_or_else(|| panic!("env var {name} has no value"))
    }

    #[test]
    fn merge_no_proxy_no_entries_is_noop() {
        // with no bypass entries the environment is left exactly as-is
        let mut env = vec![env_var("FOO", Some("bar"))];
        Containers::merge_no_proxy(&mut env, None, &[]);
        assert_eq!(env.len(), 1);
        assert_eq!(value_of(&env, "FOO"), "bar");
    }

    #[test]
    fn merge_no_proxy_adds_var_when_absent() {
        // a NO_PROXY var is added when the image set none
        let mut env = vec![env_var("FOO", Some("bar"))];
        Containers::merge_no_proxy(
            &mut env,
            None,
            &["api.thorium".to_owned(), ".local".to_owned()],
        );
        assert_eq!(value_of(&env, "NO_PROXY"), "api.thorium,.local");
    }

    #[test]
    fn merge_no_proxy_extends_existing_uppercase() {
        // an existing NO_PROXY in the Thorium env config is extended in place rather than shadowed
        let mut env = vec![env_var("NO_PROXY", Some("corp.example.com"))];
        Containers::merge_no_proxy(&mut env, None, &["api.thorium".to_owned()]);
        assert_eq!(env.len(), 1);
        assert_eq!(value_of(&env, "NO_PROXY"), "corp.example.com,api.thorium");
    }

    #[test]
    fn merge_no_proxy_extends_existing_lowercase() {
        // a lowercase no_proxy is matched case-insensitively and extended in place
        let mut env = vec![env_var("no_proxy", Some("corp.example.com"))];
        Containers::merge_no_proxy(&mut env, None, &["api.thorium".to_owned()]);
        assert_eq!(env.len(), 1);
        assert_eq!(value_of(&env, "no_proxy"), "corp.example.com,api.thorium");
    }

    #[test]
    fn merge_no_proxy_extends_both_casings() {
        // if the image set both casings each is extended so the bypass applies either way
        let mut env = vec![
            env_var("NO_PROXY", Some("upper.example.com")),
            env_var("no_proxy", Some("lower.example.com")),
        ];
        Containers::merge_no_proxy(&mut env, None, &["api.thorium".to_owned()]);
        assert_eq!(value_of(&env, "NO_PROXY"), "upper.example.com,api.thorium");
        assert_eq!(value_of(&env, "no_proxy"), "lower.example.com,api.thorium");
    }

    #[test]
    fn merge_no_proxy_replaces_empty_value() {
        // an existing but empty bypass var gets the entries without a leading comma
        let mut env = vec![env_var("NO_PROXY", Some(""))];
        Containers::merge_no_proxy(&mut env, None, &["api.thorium".to_owned()]);
        assert_eq!(value_of(&env, "NO_PROXY"), "api.thorium");
    }

    #[test]
    fn merge_no_proxy_folds_in_docker_image_env() {
        // a NO_PROXY/no_proxy baked into the Docker image ENV must be preserved, since the k8s env
        // var we set overrides the image's by name and would otherwise clobber it
        let mut env = vec![env_var("FOO", Some("bar"))];
        let docker_env = [
            "HTTPS_PROXY=http://proxy.example:80/".to_owned(),
            "no_proxy=localhost,sandia.gov,127.0.0.1".to_owned(),
            "NO_PROXY=localhost,sandia.gov,127.0.0.1".to_owned(),
        ];
        Containers::merge_no_proxy(&mut env, Some(&docker_env), &["api.thorium".to_owned()]);
        assert_eq!(
            value_of(&env, "NO_PROXY"),
            "localhost,sandia.gov,127.0.0.1,api.thorium"
        );
        assert_eq!(
            value_of(&env, "no_proxy"),
            "localhost,sandia.gov,127.0.0.1,api.thorium"
        );
    }

    #[test]
    fn merge_no_proxy_thorium_env_wins_over_docker_env() {
        // the Thorium env config overrides the Docker image ENV for the same name, so we extend the
        // Thorium value and don't fold in the (overridden) Docker value
        let mut env = vec![env_var("NO_PROXY", Some("thorium.example"))];
        let docker_env = ["NO_PROXY=docker.example".to_owned()];
        Containers::merge_no_proxy(&mut env, Some(&docker_env), &["api.thorium".to_owned()]);
        assert_eq!(env.len(), 1);
        assert_eq!(value_of(&env, "NO_PROXY"), "thorium.example,api.thorium");
    }
}
