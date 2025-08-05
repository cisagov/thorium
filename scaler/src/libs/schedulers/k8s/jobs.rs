use futures::stream::{self, StreamExt};
use k8s_openapi::api::batch::v1::Job;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use kube::api::{Api, DeleteParams, ListParams, ObjectList, PostParams};
use serde_json::json;
use std::collections::HashSet;
use std::convert::TryInto;
use thorium::{Conf, Error};

use super::{ImageSpec, Pods, Scaled};
use crate::libs::{helpers, Requisition};

/// Checks if an objects metadata for pipeline/stage matches
///
/// # Arguments
///
/// * `metadata` - The metadata tied to this k8s object
/// * `pipeline` - The pipeline to filter on
/// * `stage` - The stage to filter on
fn stage_filter(metadata: &ObjectMeta, pipeline: &str, stage: &str) -> bool {
    // if we have labels then check them
    if let Some(labels) = &metadata.labels {
        if labels.get("pipeline") == Some(&pipeline.to_owned())
            && labels.get("stage") == Some(&stage.to_owned())
        {
            // this is object is owned by our pipeline/stage
            return true;
        }
    }
    // this object is not owned by our pipeline/stage
    false
}

/// Get the number of active pods or pods left to be spawned for a job
///
/// # Arguments
///
/// * `job` - The k8s job to inspect
fn pods_left(job: &Job) -> i32 {
    let left = job.spec.as_ref().unwrap().completions.unwrap_or(1)
        - job.status.as_ref().unwrap().succeeded.unwrap_or(0);
    left
}

/// Pods api wrapper for kubernetes
pub struct Jobs {
    /// Client to use for creating namespaced clients
    client: kube::Client,
    /// Wrapper for Pod routes in kubernetes
    pods: Pods,
    /// Pod API client for all namespaces
    api: Api<Job>,
    /// A config for Thorium
    conf: Conf,
}

impl Jobs {
    /// Creates new jobs wrapper
    ///
    /// # Arguments
    ///
    /// * `client` - Kubernetes client
    /// * `conf` - Thorium Config
    pub fn new(client: &kube::Client, conf: &Conf) -> Self {
        // build pods api client
        let api: Api<Job> = Api::all(client.clone());
        // build pods wrapper
        let pods = Pods::new(client, conf);
        // get client for creating namespaced clients with
        let client = client.clone();
        Jobs {
            client,
            pods,
            api,
            conf: conf.clone(),
        }
    }

    /// List all jobs in a namespace
    ///
    /// # Arguments
    ///
    /// * `ns` - Namespace to list jobs from
    /// * `node` - Optional node to list jobs from
    pub async fn list(&self, ns: &str, node: Option<String>) -> Result<ObjectList<Job>, Error> {
        // build list params
        let params = match node {
            Some(node) => ListParams::default().fields(&format!(
                "metadata.namespace=={},spec.nodeName=={}",
                ns, node
            )),
            None => ListParams::default().fields(&format!("metadata.namespace=={}", ns)),
        };
        // list all jobs in namespace
        let pods = self.api.list(&params).await?;
        Ok(pods)
    }

    /// Checks if a job is owned by Thorium through labels
    ///
    /// # Arguments
    ///
    /// * `job` - The job to check for ownership by Thorium
    pub fn thorium_owned(job: &Job) -> bool {
        // if labels don't exist then this is not owned by Thorium
        let labels = match &job.metadata.labels {
            Some(labels) => labels,
            None => return false,
        };

        // all of these labels must exist for this to be a valid Thorium pod
        let required = ["user", "group", "pipeline", "stage", "thorium", "name"];
        // check each label exists and return false if one doesn't
        for require in required.iter() {
            if !labels.contains_key(require.to_owned()) {
                return false;
            }
        }
        true
    }

    /// Get a count of all the uncompleted pods for jobs
    ///
    /// This will only count running jobs
    ///
    /// # Arguments
    ///
    /// * `ns` - Namespace to count pod types in
    /// * `pipeline` - Name of the pipeline stage is in
    /// * `stage` - Name of the stage this image is for
    pub async fn count(&self, ns: &str, pipeline: &str, stage: &str) -> Result<u64, Error> {
        // list all pods in a namespace
        let jobs = self.list(ns, None).await?;
        // count all pods of a particular type
        let count = jobs
            .into_iter()
            // filter out jobs not owned by Thorium
            .filter(Self::thorium_owned)
            // filter out jobs not owned by this pipeline/stage
            .filter(|job| stage_filter(&job.metadata, pipeline, stage))
            // filter out jobs without specs or statuses
            .filter(|job| job.spec.is_some() || job.status.is_some())
            // count the number of pods that are active or are being spawned
            .fold(0, |count, job| count + pods_left(&job));
        Ok(count as u64)
    }

    /// Gets the expected change in pod counts for our [`Vec<ImageSpec>`]
    ///
    /// # Arguments
    ///
    /// * `specs` - The image specs to be spawned
    pub async fn get_sways(
        &self,
        mut specs: Vec<ImageSpec>,
    ) -> Result<(Vec<ImageSpec>, Vec<i64>), Error> {
        // crawl over the image specs and get their sways
        let counts = stream::iter(&specs)
            .map(|spec| (&spec.image.group, &spec.pipeline, &spec.image.name))
            .map(|(ns, pipeline, stage)| async move { self.count(ns, pipeline, stage).await })
            .buffer_unordered(10)
            .collect::<Vec<Result<u64, _>>>()
            .await
            // check for any errors and propagate the first one upwards
            .into_iter()
            .collect::<Result<Vec<u64>, _>>()?;
        // get expected change in pod scales
        let sways = counts
            .iter()
            .enumerate()
            .map(|(i, count)| &specs[i].scale - *count as i64)
            .collect();
        // update the current pod counts in the image specs
        counts
            .into_iter()
            .enumerate()
            .for_each(|(i, count)| specs[i].current = count);
        Ok((specs, sways))
    }

    /// Deletes a Job by name within a namespace if it exists
    ///
    /// # Arguments
    ///
    /// * `ns` - The namespace of the job to delete
    /// * `name` - The name of the job to delete
    pub async fn delete(&self, ns: &str, name: String) -> Result<(), Error> {
        // job api instance
        let job_api: Api<Job> = Api::namespaced(self.client.clone(), &ns);
        // build delete params
        let params = DeleteParams {
            // delete jobs immediately
            grace_period_seconds: Some(0),
            // delete pods in the foreground to prevent build up of pods
            propagation_policy: Some(kube::api::PropagationPolicy::Foreground),
            ..Default::default()
        };
        // delete target job
        if let Err(error) = job_api.delete(&name, &params).await {
            // if we got an error while deleting this job then check if its a 404
            // if it was ignore it since the job is already gone
            // match on the error enum
            match error {
                // this is an API error check the return code
                kube::Error::Api(error) => {
                    if error.code != 404 {
                        return Err(Error::from(error));
                    }
                }
                // not an API error
                _ => return Err(Error::from(error)),
            }
        }
        Ok(())
    }

    /// Scale down jobs
    ///
    /// currently this will just delete the first N jobs listed
    ///
    /// # Arguments
    ///
    /// * `spec` - The image space to delete jobs for
    /// * `quota` - The number of pods to delete
    async fn downscale(&self, spec: &ImageSpec, mut quota: u64) -> Result<(), Error> {
        let jobs = self.list(&spec.image.group, None).await?;
        let mut jobs = jobs.items;
        // sort our jobs by age older -> younger
        // this clones data unnecessarily but I am unsure how to extract the timestamp without it
        jobs.sort_unstable_by_key(|job| job.metadata.creation_timestamp.as_ref().unwrap().clone());
        // only downscale the specified number of active pods
        // go in reverse order so we start by culling younger jobs
        let mut delete_futs = Vec::default();
        for job in jobs.into_iter().rev() {
            // get number of pods remaining for this job and remove it from our quota
            quota = quota.saturating_sub(pods_left(&job) as u64);
            delete_futs.push(self.delete(&spec.image.group, job.metadata.name.unwrap()));

            // if we have filled out quota then break
            if quota == 0 {
                break;
            }
        }
        // execute deletes in batches of 10
        stream::iter(delete_futs)
            .map(|fut| async move { fut.await })
            .buffer_unordered(10)
            .collect::<Vec<Result<(), _>>>()
            .await
            // check for any errors and propagate the first one upwards
            .into_iter()
            .collect::<Result<Vec<()>, _>>()?;
        Ok(())
    }

    /// Generate the pod spec to deploy into k8s
    ///
    /// All pods generated by this will by default have a termination grace period
    /// of 0 and a restart policy of never.
    ///
    /// # Arguments
    ///
    /// * `spec` - The spec of the image to base this pod template on
    /// * `quantity` - The number of pods to generate
    pub async fn generate(&self, spec: &ImageSpec, quantity: u64) -> Result<Vec<Job>, Error> {
        // build base job spec without a name
        let raw = json!({
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "namespace": &spec.image.group,
                "labels": {
                    "user": &spec.user,
                    "group": &spec.image.group,
                    "pipeline": &spec.pipeline,
                    "stage": &spec.image.name,
                    "thorium": "true",
                }
            },
            "spec": {
                "active_deadline_seconds": 0,
                "ttl_seconds_after_finished": 0,
                "backoff_limit": 10,
                "template": self.pods.template(spec).await?,
            }
        });
        // cast this json into a barebones job
        let mut job: Job = serde_json::from_value(raw)?;
        // guess our capacity based on the desired quantity
        let mut jobs = Vec::with_capacity(quantity as usize);
        // create the required number of job objects
        for _ in 0..quantity {
            // generate a random name
            let append = helpers::gen_string(8);
            let name = format!("{}-{}-{}", &spec.pipeline, &spec.image.name, append);
            // update the name and labels in this pods metadata
            job.metadata.name = Some(name.clone());
            // labels are wrapped in an opt so we have to get a mutable reference to it
            if let Some(labels) = job.metadata.labels.as_mut() {
                labels.insert("name".into(), name);
            }
            // clone and push this job spec into our jobs vector
            jobs.push(job.clone());
        }
        Ok(jobs)
    }

    /// Deploys a pod to kubernetes
    ///
    /// # Arguments
    ///
    /// * `spec` - The image specification to deploy
    /// * `scale` - The number of pods to create
    pub async fn deploy(&self, spec: &ImageSpec, scale: u64) -> Result<(), Error> {
        // create the spec files for all of our deployments at once
        let specs = self.generate(&spec, scale).await?;

        // create pod with a namespaced api client
        // kube-rs seems to disallow creating it without global api client?
        let api: Api<Job> = Api::namespaced(self.client.clone(), &spec.image.group);
        let params = PostParams::default();
        // create our pods in bulk with 10 being create at once at most
        stream::iter(&specs)
            .map(|spec| {
                let api_ref = &api;
                let param_ref = &params;
                async move { api_ref.create(param_ref, spec).await }
            })
            .buffer_unordered(10)
            .collect::<Vec<Result<Job, _>>>()
            .await
            // check for any errors and propagate the first one upwards
            .into_iter()
            .collect::<Result<Vec<Job>, _>>()?;
        Ok(())
    }

    /// Scales a pod to new desired count
    ///
    /// # Arguments
    ///
    /// * `spec` - The image spec to scale up or down
    pub async fn scale(&self, spec: ImageSpec) -> Result<Scaled, Error> {
        // call correct handler depending on if we want to scale up or down
        if spec.scale > 0 {
            // try and convert sway into a u64 if possible
            let increase: u64 = spec.scale.try_into()?;
            // scale up
            self.deploy(&spec, increase).await?;
        } else if spec.scale < 0 {
            // get number of pods to delete
            let decrease: u64 = spec.scale.abs().try_into()?;
            // scale down
            self.downscale(&spec, decrease).await?;
        }
        let total = spec.scale + spec.current as i64;
        Ok(Scaled { total, spec })
    }

    /// Scales any reqs not found in a supplied list to 0
    ///
    /// # Arguments
    ///
    /// * `scaled` - The reqs that were scaled
    /// * `groups` - The groups or namespaces to scale down
    pub async fn scaledown(
        &self,
        scaled: HashSet<Requisition>,
        groups: &[String],
    ) -> Result<(), Error> {
        // iterate over groups
        for group in groups {
            // get pods in this group
            let pods = self.list(group, None).await?;
            // find all the pods that we need to delete
            let names = pods
                .into_iter()
                // filter any pods not owned by Thorium
                .filter(|pod| Self::thorium_owned(pod))
                // downselect to labels
                .filter_map(|pod| pod.metadata.labels)
                // filter any pods that were scaled this round
                .filter(|labels| !scaled.contains(&Requisition::from_labels(labels).unwrap()))
                // get the name of this pod to delete
                .map(|labels| labels.get("name").unwrap().to_owned())
                .collect::<Vec<String>>();

            // delete our pods 10 at a time
            stream::iter(names)
                .map(|name| async move { self.delete(group, name).await })
                .buffer_unordered(10)
                .collect::<Vec<Result<(), _>>>()
                .await
                // check for any errors and propagate the first one upwards
                .into_iter()
                .collect::<Result<Vec<()>, _>>()?;
        }
        Ok(())
    }
}
