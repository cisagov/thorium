use futures::stream::{self, StreamExt};
use hashbrown::HashMap;
use k8s_openapi::api::core::v1::{Pod, PodSecurityContext, PodSpec};
use kube::api::{Api, DeleteParams, ListParams, ObjectList, PostParams};
use reqwest::StatusCode;
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use thorium::conf::K8sHostAliases;
use thorium::models::Image;
use thorium::{Conf, Error};
use tracing::{event, instrument, Level};

use super::{Containers, Secrets, Volumes};
use crate::libs::scaler::ErrorOutKinds;
use crate::libs::schedulers::{Spawned, WorkerDeletion};
use crate::libs::Cache;
use crate::{raw_entry_map_extend, raw_entry_map_insert};

/// Pods api wrapper for kubernetes
pub struct Pods {
    /// Client to use for creating namespaced clients
    client: kube::Client,
    /// Secrets wrapper
    secrets: Secrets,
    /// volume wrappers
    volumes: Volumes,
    /// The wrapper for container operations with k8s
    containers: Containers,
    /// Pod API client for all namespaces
    api: Api<Pod>,
    /// The host aliases to apply to pods in this cluster
    host_aliases: Vec<K8sHostAliases>,
    /// The deletes that are still pending
    pub pending_deletes: HashMap<String, HashMap<String, Spawned>>,
    /// An intermediary vec to store pending deletes when checking them
    temp_deletes: Vec<(String, Spawned)>,
}

impl Pods {
    /// Creates new pods wrapper
    ///
    /// # Arguments
    ///
    /// * `client` - Kubernetes client
    /// * `conf` - Thorium Config
    /// * `cluster_name` - The name of this cluster
    /// * `context_name` - The name of this context
    pub fn new<T: Into<String>>(
        client: &kube::Client,
        conf: &Conf,
        cluster_name: T,
        context_name: &str,
    ) -> Self {
        // build pods api client
        let api: Api<Pod> = Api::all(client.clone());
        // build secrets wrapper
        let secrets = Secrets::new(client, conf, context_name);
        // build volumes wrapper
        let volumes = Volumes::new(client, conf, context_name);
        // build the containers wrapper
        let containers = Containers::new(cluster_name);
        // get our host aliases
        let host_aliases = conf.thorium.scaler.k8s.host_aliases(context_name);
        // clone our host aliases
        let host_aliases = host_aliases
            .map(|aliases| aliases.clone())
            .unwrap_or_default();
        // get client for creating namespaced clients with
        let client = client.clone();
        Pods {
            client,
            secrets,
            volumes,
            containers,
            api,
            host_aliases,
            pending_deletes: HashMap::default(),
            temp_deletes: Vec::default(),
        }
    }

    /// List all pods in a namespace
    ///
    /// # Arguments
    ///
    /// * `ns` - Namespace to list pods from
    /// * `node` - Optional node to list pods from
    #[instrument(name = "k8s::Pods::list", skip(self))]
    pub async fn list(&self, ns: &str, node: Option<&str>) -> Result<ObjectList<Pod>, Error> {
        // build list params
        let params = match node {
            Some(node) => ListParams::default()
                .fields(&format!("metadata.namespace=={ns},spec.nodeName=={node}")),
            None => ListParams::default().fields(&format!("metadata.namespace=={ns}")),
        };
        // list all pods in namespace
        let pods = self.api.list(&params).await?;
        Ok(pods)
    }

    /// List all pods across all namespaces
    ///
    /// # Arguments
    ///
    /// * `node` - Optional node to list pods from
    #[instrument(name = "k8s::Pods::list_all", skip(self))]
    pub async fn list_all(&self, node: Option<String>) -> Result<ObjectList<Pod>, Error> {
        // build list params
        let params = match node {
            Some(node) => ListParams::default().fields(&format!("spec.nodeName=={node}")),
            None => ListParams::default(),
        };
        // list all pods
        let pods = self.api.list(&params).await?;
        Ok(pods)
    }

    /// Checks if a pod is owned by Thorium through labels
    ///
    /// # Arguments
    ///
    /// * `pod` - The pod to check for ownership by Thorium
    pub fn thorium_owned(pod: &Pod) -> bool {
        // if labels don't exist then this is not owned by Thorium
        let labels = match &pod.metadata.labels {
            Some(labels) => labels,
            None => return false,
        };

        // all of these labels must exist for this to be a valid Thorium pod
        let required = ["user", "group", "pipeline", "stage", "thorium"];
        // check each label exists and return false if one doesn't
        for require in required.iter() {
            if !labels.contains_key(require.to_owned()) {
                return false;
            }
        }
        true
    }

    /// Sort any terminal pods into either Failed or Succeeded buckets
    ///
    /// # Arguments
    ///
    /// * `pod` - The pod to check if its running
    /// * `failed` - The pods that have failed due to an error
    /// * `succeeded` - The pods that are in a succeeded state
    /// * `error_out` - The pods whose workers we should fail out instead of just resetting
    fn filter_terminal(
        &self,
        pod: &Pod,
        failed: &mut Vec<String>,
        succeeded: &mut Vec<String>,
        error_out: &mut Vec<ErrorOutKinds>,
    ) -> Result<(), Error> {
        // get this pods name
        let pod_name = match &pod.metadata.name {
            Some(name) => name,
            None => {
                return Err(Error::new("Pod has no name".to_owned()));
            }
        };
        // error on pods without a status block
        if pod.status.is_none() {
            let err = Error::new(format!("Pod {pod_name} has no status"));
            return Err(err);
        }
        // get our pods status
        if let Some(status) = &pod.status {
            // check this pods status
            match &status.phase.as_deref() {
                Some("Succeeded") => succeeded.push(pod_name.to_owned()),
                Some("Failed") => {
                    // check if this pod failed due to OOM in container state
                    if let Some(container_statuses) = pod
                        .status
                        .as_ref()
                        .and_then(|status| status.container_statuses.as_ref())
                    {
                        // check if any of our container statuses are an oom error
                        for container_status in container_statuses {
                            // check if we have a container state
                            if let Some(state) = &container_status.state {
                                // check if we have a terminated state
                                if let Some(terminated) = &state.terminated {
                                    // check if this pod OOMd
                                    if terminated.reason.as_deref() == Some("OOMKilled") {
                                        // log that we found an oomed worker
                                        event!(
                                            Level::INFO,
                                            worker = pod_name,
                                            reason = "OOMKilled"
                                        );
                                        // build the error out reason for this worker
                                        let reason = ErrorOutKinds::oom(pod_name);
                                        // set this pods job to be failed out instead of reset
                                        error_out.push(reason);
                                    }
                                }
                            }
                        }
                    }
                    failed.push(pod_name.to_owned());
                }
                _ => (),
            }
        }
        Ok(())
    }

    /// Deletes a pod by name within a namespace
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace of the pod to delete
    /// * `name` - The name of the pod to delete
    #[instrument(name = "k8s::Pods::delete", skip(self))]
    pub async fn delete(&self, ns: &str, name: &str) -> Result<(), Error> {
        // pod api instance
        let pod_api: Api<Pod> = Api::namespaced(self.client.clone(), ns);
        // build delete params
        let params = DeleteParams::default().grace_period(0);
        // delete target pod
        if let Err(error) = pod_api.delete(name, &params).await {
            // if we got an error while deleting this pod then check if its a 404
            // if it was ignore it since the pod is already gone
            // match on the error enum
            match &error {
                // this is an API error check the return code
                kube::Error::Api(api_err) => {
                    if api_err.code != 404 {
                        return Err(Error::from(error));
                    }
                }
                // not an API error
                _ => return Err(Error::from(error)),
            }
        }
        Ok(())
    }

    /// Deletes pods 10 at a time in an unordered buffer
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to delete pods from
    /// * `spawns` - The spawned objects to delete
    /// * `results` - The results of these and any other pending deletes
    #[instrument(name = "k8s::Pods::delete_many", skip(self, spawns, results))]
    pub async fn delete_many<'a>(
        &mut self,
        ns: &str,
        spawns: &Vec<Spawned>,
        results: &mut Vec<WorkerDeletion>,
    ) {
        // pod api instance
        let api: Api<Pod> = Api::namespaced(self.client.clone(), ns);
        // build delete params
        let params = DeleteParams::default().grace_period(0);
        // build a list to store our cloned names
        let mut cloned_names = Vec::with_capacity(spawns.len());
        // clone our names to work around lifetime issues
        for spawn in spawns {
            // log the pod we are deleting
            event!(Level::INFO, pod = spawn.name);
            // add this pods name
            cloned_names.push(spawn.name.clone());
        }
        // delete pods 10 at a time
        let deletes = stream::iter(cloned_names)
            .map(|name| {
                // get references to minimize cloning
                let api_ref = &api;
                let params_ref = &params;
                async move { api_ref.delete(&name, params_ref).await }
            })
            .buffered(5)
            .collect::<Vec<Result<_, _>>>()
            .await;
        // crawl our deletes and determine if any failures occurred
        for (spawn, delete) in spawns.iter().zip(deletes.into_iter()) {
            // check if our delete failed
            if let Err(error) = delete {
                // cast this k8s error to a Thorium error
                let error = Error::from(error);
                // if this is an error because the pod no longer exists then ignore it
                if error.status() != Some(StatusCode::NOT_FOUND) {
                    // this error will get logged by the scaler later
                    // build our error object
                    let err = WorkerDeletion::Error {
                        delete: spawn.clone(),
                        error,
                    };
                    // this attempt failed so track this error
                    results.push(err);
                }
            }
        }
    }

    /// Validates that pods were deleted or not
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to delete pods from
    /// * `names` - The names of the pods to delete
    /// * `results` - The results of these and any other pending deletes
    #[instrument(name = "k8s::Pods::validate_deletes", skip(self, spawns, results))]
    pub async fn validate_deletes(
        &mut self,
        ns: &str,
        spawns: Vec<Spawned>,
        results: &mut Vec<WorkerDeletion>,
    ) {
        // get a list of all pods in our target namespace
        match self.list(ns, None).await {
            Ok(pods) => {
                // condense our list to just the pod names
                let condensed = pods
                    .iter()
                    .filter_map(|pod| pod.metadata.name.as_ref())
                    .collect::<HashSet<&String>>();
                // check if any of our prior pending deletions have completed yet
                if let Some(mut ns_map) = self.pending_deletes.remove(ns) {
                    // its probably best to assume it takes less than a scale loop to delete pods
                    // so drain and store our still pending deletes in a temp vec
                    // prune any of our no longer pending deletes
                    for (name, delete) in ns_map.drain() {
                        // check if this pod has been deleted
                        if condensed.contains(&name) {
                            // this pod has been deleted
                            results.push(WorkerDeletion::Deleted(delete));
                        } else {
                            self.temp_deletes.push((name, delete));
                        }
                    }
                    // readd any still pending deletes to our pending deletes map
                    for (name, delete) in self.temp_deletes.drain(..) {
                        ns_map.insert(name, delete);
                    }
                    // if we still have pending pods then readd this namespaces list
                    if !ns_map.is_empty() {
                        self.pending_deletes.insert(ns.to_owned(), ns_map);
                    }
                }
                // add any pods still in our pods list to our results list as pending
                for spawn in spawns {
                    // if this pod is still in our list then mark it as pending
                    if condensed.contains(&spawn.name) {
                        // track that our pending deleted worker hasn't been deleted yet
                        raw_entry_map_insert!(self.pending_deletes, ns, spawn.name.clone(), spawn);
                    } else {
                        // this pod no longer exists so mark it as deleted
                        results.push(WorkerDeletion::Deleted(spawn));
                    }
                }
            }
            Err(err) => {
                // TODO: log this error with TRACING
                println!("ERROR: {err:#?}");
                // we failed to list our pods so mark all of their deletes as pending
                raw_entry_map_extend!(
                    self.pending_deletes,
                    ns,
                    spawns.into_iter().map(|spawn| (spawn.name.clone(), spawn))
                );
            }
        }
    }

    /// Deletes pods 10 at a time in an unordered buffer
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to delete pods from
    /// * `names` - The names of the pods to delete
    #[instrument(name = "k8s::Pods::delete_many_owned", skip(self), err(Debug))]
    pub async fn delete_many_owned(&self, ns: &str, names: Vec<String>) -> Result<(), Error> {
        // delete our pods 10 at a time
        stream::iter(names)
            .map(|name| async move { self.delete(ns, &name).await })
            .buffer_unordered(10)
            .collect::<Vec<Result<(), _>>>()
            .await
            // check for any errors and propagate the first one upwards
            .into_iter()
            .collect::<Result<Vec<()>, _>>()?;
        Ok(())
    }

    /// Deletes pods 10 at a time in an unordered buffer in a best effort manner
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace to delete pods from
    /// * `names` - The names of the pods to delete
    #[instrument(name = "k8s::Pods::delete_many_owned", skip(self))]
    pub async fn try_delete_many(&self, ns: &str, names: Vec<&String>) {
        // assert this stream is a send stream to prevent FnOnce errors
        //let asserted = thorium::utils::helpers::assert_send_stream(names);
        // delete our pods 10 at a time
        let stream_iter = stream::iter(&names)
            .map(|name| async move { self.delete(ns, &name).await })
            .buffered(10);
        // assert this stream is send to avoid FnOnce errors
        let attempts = thorium::utils::helpers::assert_send_stream(stream_iter)
            .collect::<Vec<Result<(), Error>>>()
            .await;
        // log any errors
        for (pod_name, attempt) in names.iter().zip(attempts) {
            // check if this delete failed
            if let Err(error) = attempt {
                // log our failed delete
                event!(
                    Level::ERROR,
                    msg = "Failed to delete worker",
                    worker = pod_name,
                    error = error.to_string()
                );
            }
        }
    }

    /// Builds a security context for this pod
    ///
    /// * `cache` - A cache of info from Thorium
    /// * `user` - The user this pod will be executing jobs as
    /// * `image` - The image this pod is using
    fn build_security_ctx(cache: &Cache, user: &String, image: &Image) -> PodSecurityContext {
        // if this user has any unix info then inject that
        if let Some(unix) = &cache.users[user].unix {
            // if we have a user set in the security context override then use that
            let user = image.security_context.user.unwrap_or(unix.user as i64);
            // if we have a group set in the security context override then use that
            let group = image.security_context.group.unwrap_or(unix.group as i64);
            // if we are allowed
            // build our pod security context
            PodSecurityContext {
                run_as_user: Some(user),
                run_as_group: Some(group),
                ..Default::default()
            }
        } else {
            // get a reference to our image specs security_context settings
            let ctx = &image.security_context;
            // build this pods security context
            PodSecurityContext {
                run_as_user: ctx.user,
                run_as_group: ctx.group,
                ..Default::default()
            }
        }
    }

    /// Generate the pod spec to deploy into k8s
    ///
    /// All pods generated by this will by default have a termination grace period
    /// of 0 and a restart policy of never.
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium
    /// * `spawn` - The spawn to generate a pod for
    pub async fn generate(&self, cache: &Cache, spawn: &Spawned) -> Result<Pod, Error> {
        // get our user info
        let user = match cache.users.get(&spawn.req.user) {
            Some(user) => user,
            None => {
                // build our error message
                let msg = format!("User {} not in cache", spawn.req.user);
                return Err(Error::new(msg));
            }
        };
        // build base pod spec without a name
        let raw = json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "namespace": &spawn.req.group,
                "name": &spawn.name,
                "labels": {
                    "user": &spawn.req.user,
                    "group": &spawn.req.group,
                    "pipeline": &spawn.req.pipeline,
                    "stage": &spawn.req.stage,
                    "pool": spawn.pool.as_str(),
                    "thorium": "true",
                    "name": &spawn.name
                }
            },
            "spec": {
                "containers": self.containers.generate(cache, spawn, user)?,
                "nodeSelector": {"thorium": "enabled"},
                "nodeName": spawn.node,
                "hostAliases": &self.host_aliases,
            }
        });
        // grab our image info
        let image = cache
            .images
            .get(&spawn.req.group)
            .ok_or(Error::new(format!(
                "Unable to retrieve group '{}' from cache",
                spawn.req.group
            )))?
            .get(&spawn.req.stage)
            .ok_or(Error::new(format!(
                "Unable to retrieve stage '{}/{}' from cache. Is the image banned?",
                spawn.req.group, spawn.req.stage
            )))?;
        // cast this json into a barebones pod
        let mut pod: Pod = serde_json::from_value(raw)?;
        // get this pod's labels (should never provide default because we always provide labels in raw)
        let pod_labels = pod.metadata.labels.get_or_insert(BTreeMap::default());
        // add the base network policy labels
        for policy in cache.conf_base_network_policies() {
            pod_labels.insert(policy.name.clone(), "base".to_string());
        }
        // add any forced network policies in this group
        if let Some(forced_policies) = cache.forced_network_policies(&image.group)? {
            for policy in forced_policies {
                pod_labels.insert(policy.k8s_name.clone(), "true".to_string());
            }
        }
        // add any user-defined network policies
        for policy_name in &image.network_policies {
            let policy_k8s_name = &cache
                .get_network_policy(&image.group, policy_name)?
                .k8s_name;
            pod_labels.insert(policy_k8s_name.clone(), "true".to_string());
        }
        // get this pods specs or build defaults
        let pod_spec = pod.spec.get_or_insert(PodSpec::default());
        // insert our image specs into this pod
        pod_spec.volumes = Some(self.volumes.generate(image, user).await?);
        pod_spec.image_pull_secrets = Some(self.secrets.registry_token());
        pod_spec.termination_grace_period_seconds = Some(1);
        pod_spec.restart_policy = Some("Never".to_owned());
        pod_spec.security_context = Some(Self::build_security_ctx(cache, &spawn.req.user, image));
        Ok(pod)
    }

    /// Deploys pods within a specific namespace
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of info from Thorium
    /// * `req` - The Requisition to spawn
    /// * `scale` - The number of pods to scale up or down
    #[instrument(name = "k8s::Pods::deploy", skip(self, pods, errors))]
    pub async fn deploy<'a>(&self, ns: &str, pods: Vec<Pod>, errors: &mut HashMap<String, Error>) {
        // get a namespaced client for our pods target namespace
        let api: Api<Pod> = Api::namespaced(self.client.clone(), ns);
        // set the default params
        let params = PostParams::default();
        // get the pods names in case of any errors
        let names = pods
            .iter()
            // use map instead of filter map so we don't lose Nones and mess up the order
            .map(|pod| pod.metadata.name.clone())
            .collect::<Vec<Option<String>>>();
        // deploy any new pods 5 at a time
        let deploys = stream::iter(pods)
            .map(|pod| {
                // get references to minimize cloning
                let api_ref = &api;
                let params_ref = &params;
                async move { api_ref.create(params_ref, &pod).await }
            })
            .buffered(5)
            .collect::<Vec<Result<_, _>>>()
            .await;
        // crawl our deployments and determine if any failures occurred
        for (name, deploy) in names.into_iter().zip(deploys.into_iter()) {
            // check if our deployment failed
            if let Err(error) = deploy {
                // get the name of this pod
                if let Some(name) = name {
                    // this attempt failed so track this error
                    errors.insert(name.clone(), Error::from(error));
                }
            }
        }
    }

    /// Clear out any failing or completed pods
    ///
    /// # Arguments
    ///
    /// * `namespaces` - The namespaces to check for failing pods in
    /// * `failed` - The names of the failed pods
    /// * `terminal` - The names of the terminal pods
    /// * `error_out` - The pods whose workers we should fail out instead of just resetting
    #[instrument(name = "k8s::Pods::clear_failing", skip_all)]
    pub async fn clear_failing(
        &self,
        namespaces: &HashSet<String>,
        failed: &mut HashSet<String>,
        terminal: &mut HashSet<String>,
        error_out: &mut HashSet<ErrorOutKinds>,
    ) -> Result<(), Error> {
        // iterate over groups
        for namespace in namespaces {
            // get pods in this group
            let pods = self.list(namespace, None).await?;
            // skip any pods not owned by Thorium
            let pods_iter = pods
                .into_iter()
                // filter any pods not owned by Thorium
                .filter(Self::thorium_owned)
                // skip any pods without names
                .filter(|pod| pod.metadata.name.is_some());
            // track all succeeded pods in this namespace
            let mut succeeded = Vec::default();
            // build a list of failed and errored out pods in this namespace
            let mut failed_ns = Vec::default();
            let mut error_out_ns = Vec::default();
            // crawl over these pods and determine if they need to be deleted
            for pod in pods_iter {
                // try to check if this pod is failing
                if let Err(error) =
                    self.filter_terminal(&pod, &mut failed_ns, &mut succeeded, &mut error_out_ns)
                {
                    // log this error and ignore the pod for now
                    event!(
                        Level::ERROR,
                        pod = pod.metadata.name,
                        error = error.to_string()
                    );
                }
            }
            // only delete failed pods if there exists some to delete
            if !failed_ns.is_empty() {
                // log how many failed pods to delete
                event!(
                    Level::INFO,
                    failing = failed_ns.len(),
                    namespace = namespace
                );
                // extend our failed pod list
                failed.extend(failed_ns.iter().cloned());
                // delete this groups failed pods 10 at a time
                self.delete_many_owned(namespace, failed_ns).await?;
            }
            // only delete error out pods if there exists some to delete
            if !error_out_ns.is_empty() {
                // log how many failed pods to delete
                event!(
                    Level::INFO,
                    error_out = error_out_ns.len(),
                    namespace = namespace
                );
                // build a list of the pods we are deleting
                // we clone the strings to work around some FnOnce is not general enough errors
                let pod_names = error_out_ns
                    .iter()
                    .map(|kind| kind.worker())
                    .collect::<Vec<&String>>();
                // try delete this groups failed pods 10 at a time
                // if we fail thats fine we can try again later
                self.try_delete_many(namespace, pod_names).await;
                // extend our failed pod list
                error_out.extend(error_out_ns);
            }

            // extend our terminal pod list
            terminal.extend(succeeded.iter().cloned());
            // delete any succeeded pods if we have them
            if !succeeded.is_empty() {
                // delete any succeeded pods
                self.delete_many_owned(namespace, succeeded).await?;
            }
        }
        Ok(())
    }
}
