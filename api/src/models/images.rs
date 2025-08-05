//! Wrappers for interacting with images within Thorium with different backends
//! Currently only Redis is supported

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Add, AddAssign, SubAssign};
use std::path::PathBuf;
use std::str::FromStr;
use uuid::Uuid;

use super::bans::Ban;
use super::{
    conversions, GenericJob, OutputCollection, OutputCollectionUpdate, OutputDisplayType, Volume,
};
use crate::{
    matches_adds, matches_adds_iter, matches_adds_map, matches_clear, matches_clear_opt,
    matches_removes, matches_removes_map, matches_update, matches_update_opt, matches_vec, same,
};

/// Helps serde default worker slots consumed by this image to 1
fn default_worker_slots() -> u64 {
    1
}

/// The resources available on a node or required for an image
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, Eq, PartialEq)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreJson))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Resources {
    /// The total amount of cpu in millicpu
    pub cpu: u64,
    /// The total amount of ram in mebibytes
    pub memory: u64,
    /// The total amount of ephemeral storage in mebibytes
    #[serde(default)]
    pub ephemeral_storage: u64,
    /// The number of available worker slots if its applicable
    #[serde(default = "default_worker_slots")]
    pub worker_slots: u64,
    /// The total number of Nvidia GPUs
    #[serde(default)]
    pub nvidia_gpu: u64,
    /// The total number of AMD GPUs
    #[serde(default)]
    pub amd_gpu: u64,
}

impl Resources {
    /// Create a new resources struct
    ///
    /// # Arguments
    ///
    /// * `cpu` - The amount of cpu cores this node has in millicpu
    /// * `memory` - The amount of ram this node has in mebibytes
    /// * `ephemeral_storage` - The amount of ephemeral storage this node has in mebibytes
    /// * `worker_slots` - The total number of jobs this node can run at once
    #[must_use]
    pub fn new(cpu: u64, memory: u64, ephemeral_storage: u64, worker_slots: u64) -> Self {
        Resources {
            cpu,
            memory,
            ephemeral_storage,
            worker_slots,
            nvidia_gpu: 0,
            amd_gpu: 0,
        }
    }

    /// Check if we have enough resources to spawn one of these images
    ///
    /// # Arguments
    ///
    /// * `resources` - The resources we need to spawn a worker
    #[must_use]
    pub fn enough(&self, resources: &Resources) -> bool {
        // check if we have enough cpu to spawn this worker
        if self.cpu < resources.cpu {
            return false;
        }
        // check if we have enough memory to spawn this worker
        if self.memory < resources.memory {
            return false;
        }
        // check if we have enough storage to spawn this worker
        if self.ephemeral_storage < resources.ephemeral_storage {
            return false;
        }
        // check if we have enough open Nvidia gpus to spawn this worker
        if self.nvidia_gpu < resources.nvidia_gpu {
            return false;
        }
        // check if we have enough open AMD gpus to spawn this worker
        if self.amd_gpu < resources.amd_gpu {
            return false;
        }
        true
    }

    /// Remove the resources to spawn an image of this type
    ///
    /// # Arguments
    ///
    /// * `resources` - The resources to consume
    /// * `count` - The number of images to consume resources for
    pub fn consume(&mut self, resources: &Resources, count: u64) {
        // subtract this pods resources from our available pool
        self.cpu = self.cpu.saturating_sub(resources.cpu * count);
        self.memory = self.memory.saturating_sub(resources.memory * count);
        self.ephemeral_storage = self
            .ephemeral_storage
            .saturating_sub(resources.ephemeral_storage * count);
        self.worker_slots = self.worker_slots.saturating_sub(count);
        self.nvidia_gpu = self.nvidia_gpu.saturating_sub(resources.nvidia_gpu * count);
        self.amd_gpu = self.amd_gpu.saturating_sub(resources.amd_gpu * count);
    }

    /// Makes sure this resources is not empty of all resources
    #[must_use]
    pub fn some(&self) -> bool {
        self.cpu > 0 && self.memory > 0
    }
}

impl AddAssign for Resources {
    fn add_assign(&mut self, other: Self) {
        // add our resource counts to their respective values
        self.cpu += other.cpu;
        self.memory += other.memory;
        self.ephemeral_storage += other.ephemeral_storage;
        self.worker_slots += self.worker_slots;
    }
}

impl SubAssign for Resources {
    fn sub_assign(&mut self, other: Self) {
        // add our resource counts to their respective values
        self.cpu = self.cpu.saturating_sub(other.cpu);
        self.memory = self.memory.saturating_sub(other.memory);
        self.ephemeral_storage = self
            .ephemeral_storage
            .saturating_sub(other.ephemeral_storage);
        self.worker_slots = self.worker_slots.saturating_sub(other.worker_slots);
    }
}

impl Add for Resources {
    type Output = Self;

    /// Add a `Resources` to another `Resources`
    fn add(self, other: Self) -> Self {
        Resources {
            cpu: self.cpu + other.cpu,
            memory: self.memory + other.memory,
            ephemeral_storage: self.ephemeral_storage + other.ephemeral_storage,
            worker_slots: self.worker_slots + other.worker_slots,
            nvidia_gpu: self.nvidia_gpu + other.nvidia_gpu,
            amd_gpu: self.amd_gpu + other.amd_gpu,
        }
    }
}

impl fmt::Display for Resources {
    /// Implement display for Resources
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "cpu: {}m, memory: {}Mi, storage: {}Mi, worker_slots: {}",
            self.cpu, self.memory, self.ephemeral_storage, self.worker_slots
        )
    }
}

impl PartialEq<ResourcesRequest> for Resources {
    /// Check if a [`Resources`] contains all the same values as a [`ResourcesRequest`]
    ///
    /// # Arguments
    ///
    /// * `req` - The request to compare against
    fn eq(&self, req: &ResourcesRequest) -> bool {
        // make sure all resources were set correctly
        same!(self.cpu, &req.cpu, conversions::cpu);
        same!(self.memory, &req.memory, conversions::storage);
        matches_update!(
            self.ephemeral_storage,
            req.ephemeral_storage,
            conversions::storage
        );
        same!(self.nvidia_gpu, req.nvidia_gpu);
        same!(self.amd_gpu, req.amd_gpu);
        true
    }
}

impl PartialEq<ResourcesUpdate> for Resources {
    /// Check if a [`Resources`] contains all the updates from a [`ResourcesUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The update to compare against
    fn eq(&self, update: &ResourcesUpdate) -> bool {
        // make sure all resources were set correctly
        matches_update!(self.cpu, &update.cpu, conversions::cpu);
        matches_update!(self.memory, &update.memory, conversions::storage);
        matches_update!(
            self.ephemeral_storage,
            update.ephemeral_storage,
            conversions::storage
        );
        matches_update!(self.nvidia_gpu, update.nvidia_gpu);
        matches_update!(self.amd_gpu, update.amd_gpu);
        true
    }
}

/// The requested resources to spawn the container with
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ResourcesRequest {
    /// Cpu cores in millicpus
    pub cpu: String,
    /// Ram in mebibytes
    pub memory: String,
    /// Ephemeral storage in mebibytes
    pub ephemeral_storage: Option<String>,
    /// The total number of Nvidia GPUs
    #[serde(default)]
    pub nvidia_gpu: u64,
    /// The total number of AMD GPUs
    #[serde(default)]
    pub amd_gpu: u64,
}

impl ResourcesRequest {
    /// Create a new [`ResourceRequest`]
    ///
    /// # Arguments
    ///
    /// * `cpu` - The amount of cpu to require
    /// * `memory` - The amount of memory to require
    #[must_use]
    pub fn new(cpu: String, memory: String) -> Self {
        ResourcesRequest {
            cpu,
            memory,
            ephemeral_storage: None,
            nvidia_gpu: 0,
            amd_gpu: 0,
        }
    }

    /// Sets the cpu value to request in millicpu
    ///
    /// # Arguments
    ///
    /// * `cpu` - The CPU value in millicpu to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesRequest;
    ///
    /// ResourcesRequest::default().millicpu(2500);
    /// ```
    #[must_use]
    pub fn millicpu(mut self, cpu: u64) -> Self {
        self.cpu = format!("{cpu}m");
        self
    }

    /// Sets the cpu value to request in cores
    ///
    /// # Arguments
    ///
    /// * `cores` - The CPU value in cores
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesRequest;
    ///
    /// ResourcesRequest::default().cores(2.5);
    /// ```
    #[must_use]
    pub fn cores(mut self, cores: f64) -> Self {
        self.cpu = format!("{}m", (cores * 1000.0) as u64);
        self
    }

    /// Sets the memory to request
    ///
    /// # Arguments
    ///
    /// * `memory` - The amount of memory to request in any format
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesRequest;
    ///
    /// ResourcesRequest::default().memory("2Gi");
    /// ```
    #[must_use]
    pub fn memory<T: Into<String>>(mut self, memory: T) -> Self {
        self.memory = memory.into();
        self
    }

    /// Sets the ephemeral storage to request
    ///
    /// # Arguments
    ///
    /// * `storage` - The amount of storage to request in any format
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesRequest;
    ///
    /// ResourcesRequest::default().storage("32Gi");
    /// ```
    #[must_use]
    pub fn storage<T: Into<String>>(mut self, storage: T) -> Self {
        self.ephemeral_storage = Some(storage.into());
        self
    }

    /// Sets the number of Nvidia GPUs to require to spawn this pod
    ///
    /// # Arguments
    ///
    /// * `gpu` - The amount of Nvidia GPUs to require
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesRequest;
    ///
    /// ResourcesRequest::default().nvidia_gpu(3);
    /// ```
    #[must_use]
    pub fn nvidia_gpu(mut self, gpu: u64) -> Self {
        self.nvidia_gpu = gpu;
        self
    }

    /// Sets the number of AMD GPUs to require to spawn this pod
    ///
    /// # Arguments
    ///
    /// * `gpu` - The amount of AMD GPUs to require
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesRequest;
    ///
    /// ResourcesRequest::default().amd_gpu(6);
    /// ```
    #[must_use]
    pub fn amd_gpu(mut self, gpu: u64) -> Self {
        self.amd_gpu = gpu;
        self
    }
}

impl Default for ResourcesRequest {
    /// Create a default resource request with 250m cpu and 256Mi of ram
    fn default() -> Self {
        ResourcesRequest {
            cpu: "250m".to_owned(),
            memory: "256Mi".to_owned(),
            ephemeral_storage: None,
            nvidia_gpu: 0,
            amd_gpu: 0,
        }
    }
}

impl PartialEq<Resources> for ResourcesRequest {
    /// Check if a [`ResourcesRequest`] contains all the same values as a [`Resources`]
    ///
    /// # Arguments
    ///
    /// * `res` - The resources to compare against
    fn eq(&self, res: &Resources) -> bool {
        // make sure all resources were set correctly
        same!(res.cpu, &self.cpu, conversions::cpu);
        same!(res.memory, &self.memory, conversions::storage);
        matches_update!(
            res.ephemeral_storage,
            self.ephemeral_storage,
            conversions::storage
        );
        same!(res.nvidia_gpu, self.nvidia_gpu);
        same!(res.amd_gpu, self.amd_gpu);
        true
    }
}

impl From<Resources> for ResourcesRequest {
    /// Convert a resources object to a resources request object
    fn from(resources: Resources) -> Self {
        ResourcesRequest {
            cpu: format!("{}m", resources.cpu),
            memory: format!("{}Mi", resources.memory),
            ephemeral_storage: Some(format!("{}Mi", resources.ephemeral_storage)),
            nvidia_gpu: resources.nvidia_gpu,
            amd_gpu: resources.amd_gpu,
        }
    }
}

/// The requested resources to spawn the container with
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ResourcesUpdate {
    /// Cpu cores in millicpus
    pub cpu: Option<String>,
    /// Ram in mebibytes
    pub memory: Option<String>,
    /// Ephemeral storage in mebibytes
    pub ephemeral_storage: Option<String>,
    /// The total number of Nvidia GPUs
    pub nvidia_gpu: Option<u64>,
    /// The total number of AMD GPUs
    pub amd_gpu: Option<u64>,
}

impl ResourcesUpdate {
    /// Sets the cpu value to request in millicpu
    ///
    /// # Arguments
    ///
    /// * `cpu` - The CPU value in millicpu to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesUpdate;
    ///
    /// ResourcesUpdate::default().millicpu(2500);
    /// ```
    #[must_use]
    pub fn millicpu(mut self, cpu: u64) -> Self {
        self.cpu = Some(format!("{cpu}m"));
        self
    }

    /// Sets the cpu value to request in cores
    ///
    /// # Arguments
    ///
    /// * `cores` - The CPU value in cores
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesUpdate;
    ///
    /// ResourcesUpdate::default().cores(2.5);
    /// ```
    #[must_use]
    pub fn cores(mut self, cores: f64) -> Self {
        self.cpu = Some(format!("{}m", (cores * 1000.0) as u64));
        self
    }

    /// Sets the memory to request
    ///
    /// # Arguments
    ///
    /// * `memory` - The amount of memory to request in any format
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesUpdate;
    ///
    /// ResourcesUpdate::default().memory("2Gi");
    /// ```
    #[must_use]
    pub fn memory<T: Into<String>>(mut self, memory: T) -> Self {
        self.memory = Some(memory.into());
        self
    }

    /// Sets the ephemeral storage to request
    ///
    /// # Arguments
    ///
    /// * `storage` - The amount of storage to request in any format
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesUpdate;
    ///
    /// ResourcesUpdate::default().storage("32Gi");
    /// ```
    #[must_use]
    pub fn storage<T: Into<String>>(mut self, storage: T) -> Self {
        self.ephemeral_storage = Some(storage.into());
        self
    }

    /// Sets the number of Nvidia GPUs to require to spawn this pod
    ///
    /// # Arguments
    ///
    /// * `gpu` - The amount of Nvidia GPUs to require
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesUpdate;
    ///
    /// ResourcesUpdate::default().nvidia_gpu(3);
    /// ```
    #[must_use]
    pub fn nvidia_gpu(mut self, gpu: u64) -> Self {
        self.nvidia_gpu = Some(gpu);
        self
    }

    /// Sets the number of AMD GPUs to require to spawn this pod
    ///
    /// # Arguments
    ///
    /// * `gpu` - The amount of AMD GPUs to require
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResourcesUpdate;
    ///
    /// ResourcesUpdate::default().amd_gpu(6);
    /// ```
    #[must_use]
    pub fn amd_gpu(mut self, gpu: u64) -> Self {
        self.amd_gpu = Some(gpu);
        self
    }
}

impl PartialEq<Resources> for ResourcesUpdate {
    /// Check if a [`ResourcesUpdate`] contains the same values as a [`Resources`]
    ///
    /// # Arguments
    ///
    /// * `res` - The resources to compare against
    fn eq(&self, res: &Resources) -> bool {
        // make sure all resources were set correctly
        matches_update!(res.cpu, &self.cpu, conversions::cpu);
        matches_update!(res.memory, &self.memory, conversions::storage);
        matches_update!(
            res.ephemeral_storage,
            self.ephemeral_storage,
            conversions::storage
        );
        matches_update!(res.nvidia_gpu, self.nvidia_gpu);
        matches_update!(res.amd_gpu, self.amd_gpu);
        true
    }
}

/// Limit the number of workers for this image can spawned across all clusters controlled by a single scaler
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum SpawnLimits {
    /// Limit the amount of spawned workers for this image using a basic limit
    Basic(u64),
    /// This does not a have limit on the number of workers that cn be spawned
    Unlimited,
}

impl Default for SpawnLimits {
    /// Create a default unlimited spawn limit
    fn default() -> Self {
        SpawnLimits::Unlimited
    }
}

/// The possible ways to pass these args in
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum ArgStrategy {
    /// Do not pass this arg in
    None,
    /// Append this arg to an existing positionals
    Append,
    /// Pass this arg in as a kwarg
    Kwarg(String),
}

impl Default for ArgStrategy {
    /// Default the arg strategy to None
    fn default() -> Self {
        ArgStrategy::None
    }
}

/// Deserialize an optional vec of strings to None if its empty
fn from_opt_vec<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    // deserialize this into an optional vec
    let opt_vec: Option<Vec<String>> = Deserialize::deserialize(deserializer)?;
    // if this contains a vec then check its length
    match &opt_vec {
        Some(list) => {
            if list.is_empty() {
                return Ok(None);
            }
            Ok(opt_vec)
        }
        None => Ok(None),
    }
}

/// The args to pass to all jobs for an image
///
/// The args will be appended in order if they are all set to be appended
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageArgs {
    /// The entrypoint to force all jobs to use
    #[serde(default)]
    #[serde(deserialize_with = "from_opt_vec")]
    pub entrypoint: Option<Vec<String>>,
    /// The command to force all jobs to use
    #[serde(default)]
    #[serde(deserialize_with = "from_opt_vec")]
    pub command: Option<Vec<String>>,
    /// What kwarg to pass the current reaction id in with
    pub reaction: Option<String>,
    /// What kwarg to pass the repo url in as
    pub repo: Option<String>,
    /// What kwarg to pass the repo commit in with
    pub commit: Option<String>,
    /// What kwarg pass the output location as
    #[serde(default)]
    pub output: ArgStrategy,
}

/// The args to pass to all jobs for an image
///
/// The args will be appended in order if they are all set to be appended
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageArgsUpdate {
    /// The entrypoint to force all jobs to use
    pub entrypoint: Option<Vec<String>>,
    /// Clear this images entrypoint
    #[serde(default)]
    pub clear_entrypoint: bool,
    /// The command to force all jobs to use
    pub command: Option<Vec<String>>,
    /// Clear this images command
    #[serde(default)]
    pub clear_command: bool,
    /// What kwarg to pass the current reaction id in with
    pub reaction: Option<String>,
    /// Clear the reaction kwarg
    #[serde(default)]
    pub clear_reaction: bool,
    /// What kwarg to pass the repo url in as
    pub repo: Option<String>,
    /// Clear the reaction kwarg
    #[serde(default)]
    pub clear_repo: bool,
    /// What kwarg to pass the repo commit in with
    pub commit: Option<String>,
    /// Clear the reaction kwarg
    #[serde(default)]
    pub clear_commit: bool,
    /// What kwarg pass the output location as
    pub output: Option<ArgStrategy>,
}

impl ImageArgsUpdate {
    /// Set the updated entrypoint
    ///
    /// # Arguments
    ///
    /// * `entrypoint` - The entrypoint to set
    #[must_use]
    pub fn entrypoint<T: Into<String>>(mut self, entrypoint: Vec<T>) -> Self {
        self.entrypoint = Some(entrypoint.into_iter().map(Into::into).collect());
        self
    }

    /// Clear the entrypoint for this image
    #[must_use]
    pub fn clear_entrypoint(mut self) -> Self {
        self.clear_entrypoint = true;
        self
    }

    /// Set the updated command
    ///
    /// # Arguments
    ///
    /// * `command` - The command to set
    #[must_use]
    pub fn command<T: Into<String>>(mut self, command: Vec<T>) -> Self {
        self.command = Some(command.into_iter().map(Into::into).collect());
        self
    }

    /// Clear the command for this image
    #[must_use]
    pub fn clear_command(mut self) -> Self {
        self.clear_command = true;
        self
    }

    /// Set a new kwarg to pass reaction IDs in with
    ///
    /// # Arguments
    ///
    /// * `reaction` - The kwarg to pass reaction IDs in with
    #[must_use]
    pub fn reaction<T: Into<String>>(mut self, reaction: T) -> Self {
        self.reaction = Some(reaction.into());
        self
    }

    /// Clear the reaction for this image
    #[must_use]
    pub fn clear_reaction(mut self) -> Self {
        self.clear_reaction = true;
        self
    }

    /// Set a new kwarg to pass repo urls in with
    ///
    /// # Arguments
    ///
    /// * `repo` - The kwarg to pass repo urls in with
    #[must_use]
    pub fn repo<T: Into<String>>(mut self, repo: T) -> Self {
        self.repo = Some(repo.into());
        self
    }

    /// Clear the repo for this image
    #[must_use]
    pub fn clear_repo(mut self) -> Self {
        self.clear_repo = true;
        self
    }

    /// Set a new kwarg to pass commit hashes in with
    ///
    /// # Arguments
    ///
    /// * `commit` - The kwarg to pass commit hashes in with
    #[must_use]
    pub fn commit<T: Into<String>>(mut self, commit: T) -> Self {
        self.commit = Some(commit.into());
        self
    }

    /// Clear the commit for this image
    #[must_use]
    pub fn clear_commit(mut self) -> Self {
        self.clear_commit = true;
        self
    }

    /// Set the kwarg for the output path
    #[must_use]
    pub fn output(mut self, output: ArgStrategy) -> Self {
        self.output = Some(output);
        self
    }
}

/// List of image names with a cursor
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageList {
    /// A cursor used to page through image names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// List of image names
    pub names: Vec<String>,
}

/// List of image details with a cursor
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageDetailsList {
    /// A cursor used to page through image details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of image details
    pub details: Vec<Image>,
}

/// This allows users to specify the lifetime of their pods
/// Allowing a user to have a pod terminate after n jobs or n time
/// Time will only be checked in between jobs and so is not strongly enforced
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageLifetime {
    /// The counter used to determine lifetime (jobs || time are valid)
    pub counter: String,
    /// The number of jobs to execute or the number of seconds a pod should live
    pub amount: u64,
}

impl ImageLifetime {
    /// Create an image lifetime based on number of jobs executed
    ///
    /// # Arguments
    ///
    /// * `amount` - The number of jobs to execute
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageLifetime;
    ///
    /// ImageLifetime::jobs(32);
    /// ```
    #[must_use]
    pub fn jobs(amount: u64) -> Self {
        ImageLifetime {
            counter: "jobs".into(),
            amount,
        }
    }

    /// Create an image lifetime based on the pods lifetime
    ///
    /// # Arguments
    ///
    /// * `amount` - The number of seconds the pod can live for
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageLifetime;
    ///
    /// ImageLifetime::time(300);
    /// ```
    #[must_use]
    pub fn time(amount: u64) -> Self {
        ImageLifetime {
            counter: "time".into(),
            amount,
        }
    }
}

/// Helps default a serde value to false
// TODO: remove this when https://github.com/serde-rs/serde/issues/368 is resolved
fn default_as_false() -> bool {
    false
}

/// Helps default a serde value to false
// TODO: remove this when https://github.com/serde-rs/serde/issues/368 is resolved
fn default_as_true() -> bool {
    true
}

/// The security context to enforce when running this image
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SecurityContext {
    /// The user to run as
    pub user: Option<i64>,
    /// The group to use
    pub group: Option<i64>,
    /// Allow users to escalate their privileges
    #[serde(default = "default_as_false")]
    pub allow_privilege_escalation: bool,
}

impl SecurityContext {
    /// Sets the user id for this images security context
    ///
    /// # Arguments
    ///
    /// * `user` - The user id to run as
    #[must_use]
    pub fn user(mut self, user: i64) -> Self {
        self.user = Some(user);
        self
    }

    /// Sets the group id for this security context
    ///
    /// # Arguments
    ///
    /// * `user` - The group id to run as
    #[must_use]
    pub fn group(mut self, group: i64) -> Self {
        self.group = Some(group);
        self
    }

    /// Allows privilege escalation
    #[must_use]
    pub fn allow_escalation(mut self) -> Self {
        self.allow_privilege_escalation = true;
        self
    }

    /// Disables privilege escalation
    #[must_use]
    pub fn disallow_escalation(mut self) -> Self {
        self.allow_privilege_escalation = false;
        self
    }
}

impl PartialEq<SecurityContextUpdate> for SecurityContext {
    /// Check if a [`SecurityContext`] contains all the updates from a [`SecurityContextUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The security context update  to compare against
    fn eq(&self, update: &SecurityContextUpdate) -> bool {
        // make sure any updates were propagated
        matches_update_opt!(self.user, update.user);
        matches_update_opt!(self.group, update.group);
        matches_update!(
            self.allow_privilege_escalation,
            update.allow_privilege_escalation
        );
        true
    }
}

/// The security context to enforce when running this image
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SecurityContextUpdate {
    /// The user to run as
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<i64>,
    /// The group to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<i64>,
    /// Allow users to escalate their privileges
    pub allow_privilege_escalation: Option<bool>,
    /// Clear the user id field
    #[serde(default)]
    pub clear_user: bool,
    /// Clear the group id field
    #[serde(default)]
    pub clear_group: bool,
}

impl SecurityContextUpdate {
    /// Sets the user id for this images security context
    ///
    /// # Arguments
    ///
    /// * `user` - The user id to run as
    #[must_use]
    pub fn user(mut self, user: i64) -> Self {
        self.user = Some(user);
        self
    }

    /// Sets the group id for this security context
    ///
    /// # Arguments
    ///
    /// * `user` - The group id to run as
    #[must_use]
    pub fn group(mut self, group: i64) -> Self {
        self.group = Some(group);
        self
    }

    /// Allows privilege escalation
    #[must_use]
    pub fn allow_escalation(mut self) -> Self {
        self.allow_privilege_escalation = Some(true);
        self
    }

    /// Disables privilege escalation
    #[must_use]
    pub fn disallow_escalation(mut self) -> Self {
        self.allow_privilege_escalation = Some(false);
        self
    }

    /// Clear the user id field
    #[must_use]
    pub fn clear_user(mut self) -> Self {
        self.clear_user = true;
        self
    }

    /// Clear the group id field
    #[must_use]
    pub fn clear_group(mut self) -> Self {
        self.clear_group = true;
        self
    }
}

impl PartialEq<SecurityContext> for SecurityContextUpdate {
    /// Check if a [`SecurityContextUpdate`] was applied to a [`SecurityContext`]
    ///
    /// # Arguments
    ///
    /// * `security_context` - The `SecurityContext` to compare against
    fn eq(&self, security_context: &SecurityContext) -> bool {
        // make sure any updates were propagated
        matches_update_opt!(security_context.user, self.user);
        matches_update_opt!(security_context.group, self.group);
        matches_update!(
            security_context.allow_privilege_escalation,
            self.allow_privilege_escalation
        );
        true
    }
}

/// The different strategies used to passed downloaded dependencies to tools
///
/// The default strategy will be by path.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum DependencyPassStrategy {
    /// Tell the agent to pass all dependencies as paths
    #[default]
    Paths,
    /// Tell the agent to pass all dependencies by sha256
    Names,
    /// Tell the agent to pass the path to the dependencies directory
    Directory,
    /// Tell the agent to not pass in dependencies
    Disabled,
}

/// The default location the agent should download samples too
fn default_samples_location() -> String {
    "/tmp/thorium/samples".to_owned()
}

/// The settings for the agent downloading samples for jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SampleDependencySettings {
    /// Where the agent should store downloaded files
    #[serde(default = "default_samples_location")]
    pub location: String,
    /// The kwarg to pass these samples in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// The strategy the agent should use when passing samples downloaded to jobs
    pub strategy: DependencyPassStrategy,
}

impl Default for SampleDependencySettings {
    /// Create a default samples dependency settings
    fn default() -> Self {
        SampleDependencySettings {
            location: default_samples_location(),
            kwarg: None,
            strategy: DependencyPassStrategy::default(),
        }
    }
}

impl SampleDependencySettings {
    /// Create a new sample dependency settings object
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{SampleDependencySettings, DependencyPassStrategy};
    ///
    /// SampleDependencySettings::new("/data/dependencies", DependencyPassStrategy::Names);
    /// ```
    pub fn new<T: Into<String>>(location: T, strategy: DependencyPassStrategy) -> Self {
        SampleDependencySettings {
            location: location.into(),
            kwarg: None,
            strategy,
        }
    }

    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleDependencySettings;
    ///
    ///SampleDependencySettings::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = location.into();
        self
    }

    /// Set the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::SampleDependencySettings;
    ///
    ///SampleDependencySettings::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{SampleDependencySettings, DependencyPassStrategy};
    ///
    ///SampleDependencySettings::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = strategy;
        self
    }
}

/// The default location the agent should download repos too
fn default_repos_location() -> String {
    "/tmp/thorium/repos".to_owned()
}

/// The settings for the agent downloading repos for jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct RepoDependencySettings {
    /// Where the agent should store downloaded repos
    #[serde(default = "default_repos_location")]
    pub location: String,
    /// The kwarg to pass these repos in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// The strategy the agent should use when passing repos downloaded to jobs
    pub strategy: DependencyPassStrategy,
}

impl Default for RepoDependencySettings {
    /// Create a default repos dependency settings
    fn default() -> Self {
        RepoDependencySettings {
            location: default_repos_location(),
            kwarg: None,
            strategy: DependencyPassStrategy::default(),
        }
    }
}

impl RepoDependencySettings {
    /// Create a new dependency settings object
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{RepoDependencySettings, DependencyPassStrategy};
    ///
    /// RepoDependencySettings::new("/data/dependencies", DependencyPassStrategy::Names);
    /// ```
    pub fn new<T: Into<String>>(location: T, strategy: DependencyPassStrategy) -> Self {
        RepoDependencySettings {
            location: location.into(),
            kwarg: None,
            strategy,
        }
    }

    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::RepoDependencySettings;
    ///
    ///RepoDependencySettings::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = location.into();
        self
    }

    /// Set the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::RepoDependencySettings;
    ///
    ///RepoDependencySettings::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{RepoDependencySettings, DependencyPassStrategy};
    ///
    ///RepoDependencySettings::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = strategy;
        self
    }
}

/// The default location the agent should download repos too
fn default_tags_location() -> String {
    "/tmp/thorium/prior-tags".to_owned()
}

/// The default tags dependency pass strategy to set
fn default_tags_strategy() -> DependencyPassStrategy {
    DependencyPassStrategy::default()
}

/// The settings the agent should use when passing tags to tools
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TagDependencySettings {
    /// Whether this job wants tags to be downloaded or not
    #[serde(default)]
    pub enabled: bool,
    /// Where the agent should store downloaded tags
    #[serde(default = "default_tags_location")]
    pub location: String,
    /// The kwarg to pass these tags in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    #[serde()]
    /// The strategy the agent should use when passing tags downloaded to jobs
    #[serde(default = "default_tags_strategy")]
    pub strategy: DependencyPassStrategy,
}

impl Default for TagDependencySettings {
    /// Create a default tags dependency settings
    fn default() -> Self {
        TagDependencySettings {
            enabled: false,
            location: default_tags_location(),
            kwarg: None,
            strategy: DependencyPassStrategy::default(),
        }
    }
}

impl TagDependencySettings {
    /// Create a new dependency settings object
    ///
    /// Tag dependencies will start in an enabled state.
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDependencySettings, DependencyPassStrategy};
    ///
    /// TagDependencySettings::new("/data/dependencies", DependencyPassStrategy::Names);
    /// ```
    pub fn new<T: Into<String>>(location: T, strategy: DependencyPassStrategy) -> Self {
        TagDependencySettings {
            enabled: true,
            location: location.into(),
            kwarg: None,
            strategy,
        }
    }

    /// Enable tag dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDependencySettings, DependencyPassStrategy};
    ///
    /// TagDependencySettings::new("/data/dependencies", DependencyPassStrategy::Names)
    ///   .enable();
    /// ```
    #[must_use]
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable tag dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDependencySettings, DependencyPassStrategy};
    ///
    /// TagDependencySettings::new("/data/dependencies", DependencyPassStrategy::Names)
    ///   .disable();
    /// ```
    #[must_use]
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TagDependencySettings;
    ///
    ///TagDependencySettings::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = location.into();
        self
    }

    /// Set the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TagDependencySettings;
    ///
    ///TagDependencySettings::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{TagDependencySettings, DependencyPassStrategy};
    ///
    ///TagDependencySettings::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = strategy;
        self
    }
}

/// The settings for the agent downloading samples for jobs
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct TagDependencySettingsUpdate {
    //// Whether tag dependencies should be enabled or not
    pub enabled: Option<bool>,
    /// Where the agent should store downloaded dependencies
    pub location: Option<String>,
    /// The kwarg to pass these samples in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// Whether to clear the kwarg setting or not
    #[serde(default)]
    pub clear_kwarg: bool,
    /// The strategy the agent should use when passing downloaded dependencies to jobs
    pub strategy: Option<DependencyPassStrategy>,
}

impl PartialEq<TagDependencySettingsUpdate> for TagDependencySettings {
    /// Check if a [`TagDependencySettings`] contains all the updates from a [`TagDependencySettingsUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `TagDependencySettingsUpdate` to compare against
    fn eq(&self, update: &TagDependencySettingsUpdate) -> bool {
        // make sure any updates were propagated
        matches_update!(self.enabled, update.enabled);
        matches_update!(self.location, update.location);
        matches_update_opt!(self.kwarg, update.kwarg);
        matches_clear!(self.kwarg, update.clear_kwarg);
        matches_update!(self.strategy, update.strategy);
        true
    }
}

impl TagDependencySettingsUpdate {
    /// Enable tag dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TagDependencySettingsUpdate;
    ///
    /// TagDependencySettingsUpdate::default().location("/data/dependencies")
    ///   .enable();
    /// ```
    #[must_use]
    pub fn enable(mut self) -> Self {
        self.enabled = Some(true);
        self
    }

    /// Disable tag dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::TagDependencySettingsUpdate;
    ///
    /// TagDependencySettingsUpdate::default().location("/data/dependencies")
    ///   .disable();
    /// ```
    #[must_use]
    pub fn disable(mut self) -> Self {
        self.enabled = Some(false);
        self
    }

    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = Some(location.into());
        self
    }

    /// Updates the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Clears the kwarg arg value
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().clear_kwarg();
    /// ```
    #[must_use]
    pub fn clear_kwarg(mut self) -> Self {
        // set the clear kwarg flag to true
        self.clear_kwarg = true;
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependencySettingsUpdate, DependencyPassStrategy};
    ///
    ///DependencySettingsUpdate::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = Some(strategy);
        self
    }
}

/// The default location the agent should download repos too
fn default_children_location() -> String {
    "/tmp/thorium/prior-children".to_owned()
}

/// The default children dependency pass strategy to set
fn default_children_strategy() -> DependencyPassStrategy {
    DependencyPassStrategy::default()
}

/// The settings the agent should use when passing children to tools
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ChildrenDependencySettings {
    //// Whether children dependencies should be enabled or not
    pub enabled: bool,
    /// The prior images to restrict children collection too
    #[serde(default)]
    pub images: Vec<String>,
    /// Where the agent should store downloaded children
    #[serde(default = "default_children_location")]
    pub location: String,
    /// The kwarg to pass these children in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    #[serde()]
    /// The strategy the agent should use when passing children downloaded to jobs
    #[serde(default = "default_children_strategy")]
    pub strategy: DependencyPassStrategy,
}

impl Default for ChildrenDependencySettings {
    /// Create a default children dependency settings
    fn default() -> Self {
        ChildrenDependencySettings {
            enabled: false,
            images: Vec::default(),
            location: default_children_location(),
            kwarg: None,
            strategy: default_children_strategy(),
        }
    }
}

impl ChildrenDependencySettings {
    /// Create a new [`ChildrenDepdendencySettings`]
    ///
    /// Specifying an empty list of images will mean any children from all tools get downloaded.
    ///
    /// # Arguments
    ///
    /// * `images` - The images we depend on children from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ChildrenDependencySettings, KwargDependency, DependencyPassStrategy};
    ///
    /// ChildrenDependencySettings::new(vec!("plant", "water"))
    ///    .image("harvest")
    ///    .location("/tmp/thorium/prior-results")
    ///    .kwarg("--children")
    ///    .strategy(DependencyPassStrategy::Names);
    /// ```
    pub fn new<T, I>(images: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        ChildrenDependencySettings {
            enabled: true,
            images: images.into_iter().map(Into::into).collect(),
            location: default_prior_results(),
            kwarg: None,
            strategy: DependencyPassStrategy::default(),
        }
    }

    /// Enable children dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ChildrenDependencySettings, DependencyPassStrategy};
    ///
    /// ChildrenDependencySettings::new(["plant", "water"])
    ///   .enable();
    /// ```
    #[must_use]
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Disable children dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ChildrenDependencySettings, DependencyPassStrategy};
    ///
    /// ChildrenDependencySettings::new(["plant", "water"])
    ///   .disable();
    /// ```
    #[must_use]
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Add new image to this children dependency settings object
    ///
    /// # Arguments
    ///
    /// * `image` - The image to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettings;
    ///
    /// ChildrenDependencySettings::new(vec!("plant", "water"))
    ///    .image("harvest");
    /// ```
    #[must_use]
    pub fn image<T: Into<String>>(mut self, image: T) -> Self {
        self.images.push(image.into());
        self
    }

    /// Add new images to this children dependency settings object
    ///
    /// # Arguments
    ///
    /// * `images` - The images to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettings;
    ///
    /// ChildrenDependencySettings::new(vec!("plant", "water"))
    ///    .images(vec!("harvest", "sell"));
    /// ```
    #[must_use]
    pub fn images<T: Into<String>>(mut self, images: Vec<T>) -> Self {
        self.images.extend(images.into_iter().map(Into::into));
        self
    }

    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettings;
    ///
    ///ChildrenDependencySettings::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = location.into();
        self
    }

    /// Set the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettings;
    ///
    ///ChildrenDependencySettings::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ChildrenDependencySettings, DependencyPassStrategy};
    ///
    ///ChildrenDependencySettings::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = strategy;
        self
    }
}

/// The settings for the agent downloading samples for jobs
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ChildrenDependencySettingsUpdate {
    //// Whether tag dependencies should be enabled or not
    pub enabled: Option<bool>,
    /// The prior images to pass results from
    #[serde(default)]
    pub add_images: Vec<String>,
    /// The images to stop passing results from
    #[serde(default)]
    pub remove_images: Vec<String>,
    /// Where the agent should store downloaded dependencies
    pub location: Option<String>,
    /// The kwarg to pass these samples in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// Whether to clear the kwarg setting or not
    #[serde(default)]
    pub clear_kwarg: bool,
    /// The strategy the agent should use when passing downloaded dependencies to jobs
    pub strategy: Option<DependencyPassStrategy>,
}

impl PartialEq<ChildrenDependencySettingsUpdate> for ChildrenDependencySettings {
    /// Check if a [`DependencySettings`] contains all the updates from a [`ChildrenDependencySettingsUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `ChildrenDependencySettingsUpdate` to compare against
    fn eq(&self, update: &ChildrenDependencySettingsUpdate) -> bool {
        // make sure any updates were propagated
        matches_update!(self.enabled, update.enabled);
        matches_adds!(self.images, update.add_images);
        matches_removes!(self.images, update.remove_images);
        matches_update!(self.location, update.location);
        matches_update_opt!(self.kwarg, update.kwarg);
        matches_clear!(self.kwarg, update.clear_kwarg);
        matches_update!(self.strategy, update.strategy);
        true
    }
}

impl ChildrenDependencySettingsUpdate {
    /// Enable children dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettingsUpdate;
    ///
    /// ChildrenDependencySettingsUpdate::default().location("/data/dependencies")
    ///   .enable();
    /// ```
    #[must_use]
    pub fn enable(mut self) -> Self {
        self.enabled = Some(true);
        self
    }

    /// Disable children dependencies
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettingsUpdate;
    ///
    /// ChildrenDependencySettingsUpdate::default().location("/data/dependencies")
    ///   .disable();
    /// ```
    #[must_use]
    pub fn disable(mut self) -> Self {
        self.enabled = Some(false);
        self
    }

    /// Add new image to this children dependency settings object
    ///
    /// # Arguments
    ///
    /// * `image` - The image to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettingsUpdate;
    ///
    /// ChildrenDependencySettingsUpdate::default()
    ///    .image("harvest");
    /// ```
    #[must_use]
    pub fn image<T: Into<String>>(mut self, image: T) -> Self {
        self.add_images.push(image.into());
        self
    }

    /// Add new images to this children dependency settings object
    ///
    /// # Arguments
    ///
    /// * `images` - The images to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettingsUpdate;
    ///
    /// ChildrenDependencySettingsUpdate::default()
    ///    .images(vec!("harvest", "sell"));
    /// ```
    #[must_use]
    pub fn images<T: Into<String>>(mut self, images: Vec<T>) -> Self {
        self.add_images.extend(images.into_iter().map(Into::into));
        self
    }

    /// Remove an image from this children dependency settings object
    ///
    /// # Arguments
    ///
    /// * `image` - The image to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettingsUpdate;
    ///
    /// ChildrenDependencySettingsUpdate::default()
    ///    .remove_image("harvest");
    /// ```
    #[must_use]
    pub fn remove_image<T: Into<String>>(mut self, image: T) -> Self {
        self.remove_images.push(image.into());
        self
    }

    /// Removes images from this children dependency settings object
    ///
    /// # Arguments
    ///
    /// * `images` - The images to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ChildrenDependencySettingsUpdate;
    ///
    /// ChildrenDependencySettingsUpdate::default()
    ///    .remove_images(vec!("harvest", "sell"));
    /// ```
    #[must_use]
    pub fn remove_images<T: Into<String>>(mut self, images: Vec<T>) -> Self {
        self.remove_images
            .extend(images.into_iter().map(Into::into));
        self
    }

    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = Some(location.into());
        self
    }

    /// Updates the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Clears the kwarg arg value
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().clear_kwarg();
    /// ```
    #[must_use]
    pub fn clear_kwarg(mut self) -> Self {
        // set the clear kwarg flag to true
        self.clear_kwarg = true;
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependencySettingsUpdate, DependencyPassStrategy};
    ///
    ///DependencySettingsUpdate::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = Some(strategy);
        self
    }
}

/// The settings for the agent downloading samples for jobs
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DependencySettingsUpdate {
    /// Where the agent should store downloaded dependencies
    pub location: Option<String>,
    /// The kwarg to pass these samples in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// Whether to clear the kwarg setting or not
    #[serde(default)]
    pub clear_kwarg: bool,
    /// The strategy the agent should use when passing downloaded dependencies to jobs
    pub strategy: Option<DependencyPassStrategy>,
}

impl PartialEq<DependencySettingsUpdate> for SampleDependencySettings {
    /// Check if a [`SampleDependencySettings`] contains all the updates from a [`DependencySettingsUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `DependencySettingsUpdate` to compare against
    fn eq(&self, update: &DependencySettingsUpdate) -> bool {
        // make sure any updates were propagated
        matches_update!(self.location, update.location);
        matches_update_opt!(self.kwarg, update.kwarg);
        matches_clear!(self.kwarg, update.clear_kwarg);
        matches_update!(self.strategy, update.strategy);
        true
    }
}

impl PartialEq<DependencySettingsUpdate> for RepoDependencySettings {
    /// Check if a [`RepoDependencySettings`] contains all the updates from a [`DependencySettingsUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `DependencySettingsUpdate` to compare against
    fn eq(&self, update: &DependencySettingsUpdate) -> bool {
        // make sure any updates were propagated
        matches_update!(self.location, update.location);
        matches_update_opt!(self.kwarg, update.kwarg);
        matches_clear!(self.kwarg, update.clear_kwarg);
        matches_update!(self.strategy, update.strategy);
        true
    }
}

impl DependencySettingsUpdate {
    /// Change the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().location("/data/dependencies");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = Some(location.into());
        self
    }

    /// Updates the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Clears the kwarg arg value
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass dependencies in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::DependencySettingsUpdate;
    ///
    ///DependencySettingsUpdate::default().clear_kwarg();
    /// ```
    #[must_use]
    pub fn clear_kwarg(mut self) -> Self {
        // set the clear kwarg flag to true
        self.clear_kwarg = true;
        self
    }

    /// Change the strategy used to pass dependencies into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependencySettingsUpdate, DependencyPassStrategy};
    ///
    ///DependencySettingsUpdate::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = Some(strategy);
        self
    }
}

/// Set the default ephemeral files download location
fn default_ephemeral_location() -> String {
    "/tmp/thorium/ephemeral".to_owned()
}

/// The settings for the agent downloading dependencies for jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EphemeralDependencySettings {
    /// Where the agent should store downloaded ephemeral files
    #[serde(default = "default_ephemeral_location")]
    pub location: String,
    /// The kwarg to pass these files in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// The strategy the agent should use when passing dependencies downloaded to jobs
    #[serde(default)]
    pub strategy: DependencyPassStrategy,
    /// Any files to limit this image to downloading
    #[serde(default)]
    pub names: Vec<String>,
}

impl Default for EphemeralDependencySettings {
    fn default() -> Self {
        EphemeralDependencySettings {
            location: default_ephemeral_location(),
            kwarg: None,
            strategy: DependencyPassStrategy::default(),
            names: Vec::default(),
        }
    }
}

impl EphemeralDependencySettings {
    /// Create a new ephemeral dependency settings object
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    /// * `strategy` - The strategy to use when passing dependency files to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{EphemeralDependencySettings, DependencyPassStrategy};
    ///
    /// EphemeralDependencySettings::new("/data/ephemeral", DependencyPassStrategy::Names);
    /// ```
    pub fn new<T: Into<String>>(location: T, strategy: DependencyPassStrategy) -> Self {
        EphemeralDependencySettings {
            location: location.into(),
            kwarg: None,
            strategy,
            names: Vec::default(),
        }
    }

    /// Set the location to save dependencies to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded dependencies to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettings;
    ///
    /// EphemeralDependencySettings::default().location("/data/ephemeral");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = location.into();
        self
    }

    /// Set the kwarg to pass these dependencies in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass ephemeral in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettings;
    ///
    /// EphemeralDependencySettings::default().kwarg("--ephemeral");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Set the strategy used to pass ephemeral files in
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing dependencies to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{EphemeralDependencySettings, DependencyPassStrategy};
    ///
    /// EphemeralDependencySettings::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = strategy;
        self
    }

    /// Add a file name to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to restrict this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettings;
    ///
    /// EphemeralDependencySettings::default().name("file.txt");
    /// ```
    #[must_use]
    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.names.push(name.into());
        self
    }

    /// Add multiple file names to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `names` - The names of the files to restrict this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettings;
    ///
    /// EphemeralDependencySettings::default().names(vec!("file.txt", "other.txt"));
    /// ```
    #[must_use]
    pub fn names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        // convert our names to a string and set it
        self.names.extend(names.into_iter().map(Into::into));
        self
    }
}

/// The settings for the agent downloading samples for jobs
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct EphemeralDependencySettingsUpdate {
    /// Where the agent should store downloaded files
    pub location: Option<String>,
    /// The kwarg to pass these samples in with if one is set (otherwise use positional args)
    pub kwarg: Option<String>,
    /// Whether to clear the kwarg setting or not
    #[serde(default)]
    pub clear_kwarg: bool,
    /// The strategy the agent should use when passing samples downloaded to jobs
    pub strategy: Option<DependencyPassStrategy>,
    /// Any names to add to the list of dependencies to restrict this image too
    #[serde(default)]
    pub add_names: Vec<String>,
    /// The names to remove from the list of dependencies to restrict this image too
    #[serde(default)]
    pub remove_names: Vec<String>,
}

impl PartialEq<EphemeralDependencySettingsUpdate> for EphemeralDependencySettings {
    /// Check if a [`SampleDependencySettings`] contains all the updates from a [`EphemeralDependencySettingsUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `EphemeralDependencySettingsUpdate` to compare against
    fn eq(&self, update: &EphemeralDependencySettingsUpdate) -> bool {
        // make sure any updates were propagated
        matches_update!(self.location, update.location);
        matches_update_opt!(self.kwarg, update.kwarg);
        matches_clear!(self.kwarg, update.clear_kwarg);
        matches_update!(self.strategy, update.strategy);
        matches_adds!(self.names, update.add_names);
        matches_removes!(self.names, update.remove_names);
        true
    }
}

impl EphemeralDependencySettingsUpdate {
    /// Change the location to save ephemeral files to
    ///
    /// # Arguments
    ///
    /// * `location` - The location to save downloaded ephemeral files to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettingsUpdate;
    ///
    ///EphemeralDependencySettingsUpdate::default().location("/data/ephemeral");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        // convert our location to a string and set it
        self.location = Some(location.into());
        self
    }

    /// Updates the kwarg to pass these ephemeral files in with if one exists
    ///
    /// This should include the '--' characters.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - The kwarg arg to pass ephemeral files in with
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettingsUpdate;
    ///
    ///EphemeralDependencySettingsUpdate::default().kwarg("--inputs");
    /// ```
    #[must_use]
    pub fn kwarg<T: Into<String>>(mut self, kwarg: T) -> Self {
        // convert our kwarg to a string and set it
        self.kwarg = Some(kwarg.into());
        self
    }

    /// Clears the kwarg arg value
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettingsUpdate;
    ///
    ///EphemeralDependencySettingsUpdate::default().clear_kwarg();
    /// ```
    #[must_use]
    pub fn clear_kwarg(mut self) -> Self {
        // set the clear kwarg flag to true
        self.clear_kwarg = true;
        self
    }

    /// Change the strategy used to pass ephemeral files into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing ephemeral files to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{EphemeralDependencySettingsUpdate, DependencyPassStrategy};
    ///
    ///EphemeralDependencySettingsUpdate::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = Some(strategy);
        self
    }

    /// Add a new name to the list of dependencies to download
    ///
    /// # Arguments
    ///
    /// * `name` - The name to add our list of dependencies to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettingsUpdate;
    ///
    ///EphemeralDependencySettingsUpdate::default().add_name("file.txt");
    /// ```
    #[must_use]
    pub fn add_name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.add_names.push(name.into());
        self
    }

    /// Removes a name from the list of dependencies to download
    ///
    /// # Arguments
    ///
    /// * `name` - The name to remove from our list of dependencies to download
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::EphemeralDependencySettingsUpdate;
    ///
    ///EphemeralDependencySettingsUpdate::default().remove_name("file.txt");
    /// ```
    #[must_use]
    pub fn remove_name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.remove_names.push(name.into());
        self
    }
}

/// How prior result dependencies should be be passed in by kwargs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum KwargDependency {
    /// Pass in all results with a single kwarg key
    List(String),
    /// Pass in all results with unique kwarg keys (image name, key)
    Map(HashMap<String, String>),
    /// Pass in all results with positional args
    None,
}

impl Default for KwargDependency {
    /// Create a default [`KwargDependency`]
    fn default() -> Self {
        KwargDependency::None
    }
}

/// Helps serde default the prior results path
fn default_prior_results() -> String {
    "/tmp/thorium/prior-results".to_owned()
}

/// The settings for the agent downloading prior results for jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ResultDependencySettings {
    /// The prior images to collect results from
    #[serde(default)]
    pub images: Vec<String>,
    /// Where the agent should store downloaded prior result files
    #[serde(default = "default_prior_results")]
    pub location: String,
    /// The kwarg to pass these files in with if one is set (otherwise use positional args)
    #[serde(default)]
    pub kwarg: KwargDependency,
    /// The strategy the agent should use when passing dependencies downloaded to jobs
    #[serde(default)]
    pub strategy: DependencyPassStrategy,
    /// Any files to limit this image to downloading
    #[serde(default)]
    pub names: Vec<String>,
}

impl Default for ResultDependencySettings {
    /// Create a default [`ResultDependencySettings`]
    fn default() -> Self {
        ResultDependencySettings {
            images: Vec::default(),
            location: default_prior_results(),
            kwarg: KwargDependency::default(),
            strategy: DependencyPassStrategy::default(),
            names: Vec::default(),
        }
    }
}

impl ResultDependencySettings {
    /// Create a new [`ResultDepdendencySettings`]
    ///
    /// # Arguments
    ///
    /// * `images` - The images we depend on results from
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ResultDependencySettings, KwargDependency, DependencyPassStrategy};
    ///
    /// ResultDependencySettings::new(vec!("plant", "water"))
    ///    .image("harvest")
    ///    .location("/tmp/thorium/prior-results")
    ///    .kwarg(KwargDependency::List("--crops".to_owned()))
    ///    .strategy(DependencyPassStrategy::Names)
    ///    .name("corn.json");
    /// ```
    #[must_use]
    pub fn new<T: Into<String>>(images: Vec<T>) -> Self {
        ResultDependencySettings {
            images: images.into_iter().map(Into::into).collect(),
            location: default_prior_results(),
            kwarg: KwargDependency::default(),
            strategy: DependencyPassStrategy::default(),
            names: Vec::default(),
        }
    }

    /// Add new image to this result dependency settings object
    ///
    /// # Arguments
    ///
    /// * `image` - The image to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettings;
    ///
    /// ResultDependencySettings::new(vec!("plant", "water"))
    ///    .image("harvest");
    /// ```
    #[must_use]
    pub fn image<T: Into<String>>(mut self, image: T) -> Self {
        self.images.push(image.into());
        self
    }

    /// Add new images to this result dependency settings object
    ///
    /// # Arguments
    ///
    /// * `images` - The images to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettings;
    ///
    /// ResultDependencySettings::new(vec!("plant", "water"))
    ///    .images(vec!("harvest", "sell"));
    /// ```
    #[must_use]
    pub fn images<T: Into<String>>(mut self, images: Vec<T>) -> Self {
        self.images.extend(images.into_iter().map(Into::into));
        self
    }

    /// The directory to save prior results to
    ///
    /// # Arguments
    ///
    /// * `location` - The directory to save prior results to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettings;
    ///
    /// ResultDependencySettings::new(vec!("plant", "water"))
    ///    .location("/tmp/thorium/prior-results");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        self.location = location.into();
        self
    }

    /// How to pass prior results to the tool as kwargs
    ///
    /// If [`KwargDependency::None`] is used then the default position logic will be used.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - How to pass in prior results as kwargs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ResultDependencySettings, KwargDependency};
    ///
    /// ResultDependencySettings::new(vec!("plant", "water"))
    ///    .kwarg(KwargDependency::List("--crops".to_owned()));
    /// ```
    #[must_use]
    pub fn kwarg(mut self, kwarg: KwargDependency) -> Self {
        self.kwarg = kwarg;
        self
    }

    /// Change the strategy used to pass prior results into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing prior results to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ResultDependencySettings, DependencyPassStrategy};
    ///
    /// ResultDependencySettings::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = strategy;
        self
    }

    /// Add a file name to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to restrict this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettings;
    ///
    /// ResultDependencySettings::default().name("file.txt");
    /// ```
    #[must_use]
    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.names.push(name.into());
        self
    }

    /// Add multiple file names to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `names` - The names of the files to restrict this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettings;
    ///
    /// ResultDependencySettings::default().names(vec!("file.txt", "other.txt"));
    /// ```
    #[must_use]
    pub fn names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        // convert our names to a string and set it
        self.names.extend(names.into_iter().map(Into::into));
        self
    }
}

/// The updated settings for the agent downloading prior results for jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ResultDependencySettingsUpdate {
    /// The prior images to pass results from
    #[serde(default)]
    pub add_images: Vec<String>,
    /// The images to stop passing results from
    #[serde(default)]
    pub remove_images: Vec<String>,
    /// Where the agent should store downloaded prior result files
    pub location: Option<String>,
    /// The kwarg to pass these files in with if one is set (otherwise use positional args)
    pub kwarg: Option<KwargDependency>,
    /// The strategy the agent should use when passing dependencies downloaded to jobs
    pub strategy: Option<DependencyPassStrategy>,
    /// Any files to limit this image to downloading
    #[serde(default)]
    pub add_names: Vec<String>,
    /// The file names to remove form our download list
    #[serde(default)]
    pub remove_names: Vec<String>,
}

impl ResultDependencySettingsUpdate {
    /// Add new image to this result dependency settings object
    ///
    /// # Arguments
    ///
    /// * `image` - The image to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default()
    ///    .image("harvest");
    /// ```
    #[must_use]
    pub fn image<T: Into<String>>(mut self, image: T) -> Self {
        self.add_images.push(image.into());
        self
    }

    /// Add new images to this result dependency settings object
    ///
    /// # Arguments
    ///
    /// * `images` - The images to add
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default()
    ///    .images(vec!("harvest", "sell"));
    /// ```
    #[must_use]
    pub fn images<T: Into<String>>(mut self, images: Vec<T>) -> Self {
        self.add_images.extend(images.into_iter().map(Into::into));
        self
    }

    /// Remove an image from this result dependency settings object
    ///
    /// # Arguments
    ///
    /// * `image` - The image to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default()
    ///    .remove_image("harvest");
    /// ```
    #[must_use]
    pub fn remove_image<T: Into<String>>(mut self, image: T) -> Self {
        self.remove_images.push(image.into());
        self
    }

    /// Removes images from this result dependency settings object
    ///
    /// # Arguments
    ///
    /// * `images` - The images to remove
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default()
    ///    .remove_images(vec!("harvest", "sell"));
    /// ```
    #[must_use]
    pub fn remove_images<T: Into<String>>(mut self, images: Vec<T>) -> Self {
        self.remove_images
            .extend(images.into_iter().map(Into::into));
        self
    }

    /// The directory to save prior results to
    ///
    /// # Arguments
    ///
    /// * `location` - The directory to save prior results to
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default()
    ///    .location("/tmp/thorium/prior-results");
    /// ```
    #[must_use]
    pub fn location<T: Into<String>>(mut self, location: T) -> Self {
        self.location = Some(location.into());
        self
    }

    /// How to pass prior results to the tool as kwargs
    ///
    /// If [`KwargDependency::None`] is used then the default position logic will be used.
    ///
    /// # Arguments
    ///
    /// * `kwarg` - How to pass in prior results as kwargs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ResultDependencySettingsUpdate, KwargDependency};
    ///
    /// ResultDependencySettingsUpdate::default()
    ///    .kwarg(KwargDependency::List("--crops".to_owned()));
    /// ```
    #[must_use]
    pub fn kwarg(mut self, kwarg: KwargDependency) -> Self {
        self.kwarg = Some(kwarg);
        self
    }

    /// Change the strategy used to pass prior results into jobs
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use when passing prior results to jobs
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ResultDependencySettingsUpdate, DependencyPassStrategy};
    ///
    /// ResultDependencySettingsUpdate::default().strategy(DependencyPassStrategy::Names);
    /// ```
    #[must_use]
    pub fn strategy(mut self, strategy: DependencyPassStrategy) -> Self {
        // update our dependency passing strategy
        self.strategy = Some(strategy);
        self
    }

    /// Add a file name to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to restrict this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default().name("file.txt");
    /// ```
    #[must_use]
    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.add_names.push(name.into());
        self
    }

    /// Add multiple file names to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `names` - The names of the files to restrict this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default().names(vec!("file.txt", "other.txt"));
    /// ```
    #[must_use]
    pub fn names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        // convert our names to a string and set it
        self.add_names.extend(names.into_iter().map(Into::into));
        self
    }

    /// Removes a file name to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the file to stop restricting this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default().remove_name("file.txt");
    /// ```
    #[must_use]
    pub fn remove_name<T: Into<String>>(mut self, name: T) -> Self {
        // convert our name to a string and set it
        self.remove_names.push(name.into());
        self
    }

    /// Remove multiple file names to restrict dependencies too
    ///
    /// # Arguments
    ///
    /// * `names` - The names of the files to stop restricting this image too
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ResultDependencySettingsUpdate;
    ///
    /// ResultDependencySettingsUpdate::default().remove_names(vec!("file.txt", "other.txt"));
    /// ```
    #[must_use]
    pub fn remove_names<T: Into<String>>(mut self, names: Vec<T>) -> Self {
        // convert our names to a string and set it
        self.remove_names.extend(names.into_iter().map(Into::into));
        self
    }
}

impl PartialEq<ResultDependencySettingsUpdate> for ResultDependencySettings {
    /// Check if a [`ResultDependencySettings`] contains all the updates from a [`ResultDependencySettingsUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `ResultDependencySettingsUpdate` to compare against
    fn eq(&self, update: &ResultDependencySettingsUpdate) -> bool {
        // make sure any updates were propagated
        matches_adds!(self.images, update.add_images);
        matches_removes!(self.images, update.remove_images);
        matches_update!(self.location, update.location);
        matches_update!(self.kwarg, update.kwarg);
        matches_update!(self.strategy, update.strategy);
        matches_adds!(self.names, update.add_names);
        matches_removes!(self.names, update.remove_names);
        true
    }
}

/// How this image should handle dependencies it needs for jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Dependencies {
    /// The settings  the agent should use when passing downloaded samples to tools
    #[serde(default)]
    pub samples: SampleDependencySettings,
    /// The settings the agent should use when passing downloaded ephemeral files to tools
    #[serde(default)]
    pub ephemeral: EphemeralDependencySettings,
    /// The settings the agent should use when passing prior results to tools
    #[serde(default)]
    pub results: ResultDependencySettings,
    /// The settings the agent should use when passing prior repos to tools
    #[serde(default)]
    pub repos: RepoDependencySettings,
    /// The settings the agent should use when passing tags to tools
    #[serde(default)]
    pub tags: TagDependencySettings,
    /// The settings the agent should use when passing children files from past tools
    #[serde(default)]
    pub children: ChildrenDependencySettings,
}

impl Dependencies {
    /// Sets the sample settings
    ///
    /// # Arguments
    ///
    /// * `samples` - The settings to use for sample dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Dependencies, SampleDependencySettings, DependencyPassStrategy};
    ///
    /// Dependencies::default()
    ///     .samples(SampleDependencySettings::new("/data/samples", DependencyPassStrategy::Names));
    /// ```
    #[must_use]
    pub fn samples(mut self, samples: SampleDependencySettings) -> Self {
        self.samples = samples;
        self
    }

    /// Sets the ephemeral settings
    ///
    /// # Arguments
    ///
    /// * `ephemeral` - The settings to use for ephemeral dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Dependencies, EphemeralDependencySettings, DependencyPassStrategy};
    ///
    /// Dependencies::default()
    ///     .ephemeral(EphemeralDependencySettings::new("/data/ephemeral", DependencyPassStrategy::Names));
    /// ```
    #[must_use]
    pub fn ephemeral(mut self, ephemeral: EphemeralDependencySettings) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    /// Sets the results settings
    ///
    /// # Arguments
    ///
    /// * `results` - The settings to use for results dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Dependencies, ResultDependencySettings, DependencyPassStrategy};
    ///
    /// Dependencies::default().results(ResultDependencySettings::default());
    /// ```
    #[must_use]
    pub fn results(mut self, results: ResultDependencySettings) -> Self {
        self.results = results;
        self
    }

    /// Sets the repos settings
    ///
    /// # Arguments
    ///
    /// * `repos` - The settings to use for repos dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Dependencies, RepoDependencySettings, DependencyPassStrategy};
    ///
    /// Dependencies::default()
    ///     .repos(RepoDependencySettings::new("/data/repos", DependencyPassStrategy::Names));
    /// ```
    #[must_use]
    pub fn repos(mut self, repos: RepoDependencySettings) -> Self {
        self.repos = repos;
        self
    }
}

impl PartialEq<DependenciesUpdate> for Dependencies {
    /// Check if a [`Dependencies`] contains all the updates from a [`DependenciesUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `DependenciesUpdate` to compare against
    fn eq(&self, update: &DependenciesUpdate) -> bool {
        // make sure any updates were propagated
        same!(self.samples, update.samples);
        same!(self.ephemeral, update.ephemeral);
        same!(self.results, update.results);
        same!(self.repos, update.repos);
        true
    }
}

/// Updates how this image should handle dependencies it needs for jobs
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DependenciesUpdate {
    /// The strategy the agent should use when passing downloaded samples to tools
    #[serde(default)]
    pub samples: DependencySettingsUpdate,
    /// The strategy the agent should use when passing downloaded ephemeral files to tools
    #[serde(default)]
    pub ephemeral: EphemeralDependencySettingsUpdate,
    /// The strategy the agent should use when passing in prior results
    #[serde(default)]
    pub results: ResultDependencySettingsUpdate,
    /// The strategy the agent should use when passing downloaded repos to tools
    #[serde(default)]
    pub repos: DependencySettingsUpdate,
    /// The strategy the agent should use when passing downloaded tags to tools
    #[serde(default)]
    pub tags: TagDependencySettingsUpdate,
    /// The settings the agent should use when passing children files from past tools
    #[serde(default)]
    pub children: ChildrenDependencySettingsUpdate,
}

impl DependenciesUpdate {
    /// Sets the sample settings that should be updated
    ///
    /// # Arguments
    ///
    /// * `samples` - The settings to update in this images sample dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependenciesUpdate, DependencySettingsUpdate, DependencyPassStrategy};
    ///
    /// DependenciesUpdate::default()
    ///     .samples(DependencySettingsUpdate::default()
    ///         .location("/data/samples")
    ///         .strategy(DependencyPassStrategy::Names));
    /// ```
    #[must_use]
    pub fn samples(mut self, samples: DependencySettingsUpdate) -> Self {
        self.samples = samples;
        self
    }

    /// Sets the ephemeral settings that should be updated
    ///
    /// # Arguments
    ///
    /// * `ephemeral` - The settings to update in this images sample dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependenciesUpdate, EphemeralDependencySettingsUpdate, DependencyPassStrategy};
    ///
    /// DependenciesUpdate::default()
    ///     .ephemeral(EphemeralDependencySettingsUpdate::default()
    ///         .location("/data/ephemeral")
    ///         .strategy(DependencyPassStrategy::Names)
    ///         .add_name("updated.txt")
    ///         .remove_name("file.txt"));
    /// ```
    #[must_use]
    pub fn ephemeral(mut self, ephemeral: EphemeralDependencySettingsUpdate) -> Self {
        self.ephemeral = ephemeral;
        self
    }

    /// Sets the results settings that should be updated
    ///
    /// # Arguments
    ///
    /// * `results` - The settings to update in this images repo dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependenciesUpdate, ResultDependencySettingsUpdate, DependencyPassStrategy};
    ///
    /// DependenciesUpdate::default()
    ///     .results(ResultDependencySettingsUpdate::default()
    ///         .image("harvest")
    ///         .location("/data/results")
    ///         .strategy(DependencyPassStrategy::Names)
    ///         .name("field.txt"));
    /// ```
    #[must_use]
    pub fn results(mut self, results: ResultDependencySettingsUpdate) -> Self {
        self.results = results;
        self
    }

    /// Sets the repos settings that should be updated
    ///
    /// # Arguments
    ///
    /// * `repos` - The settings to update in this images repo dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependenciesUpdate, DependencySettingsUpdate, DependencyPassStrategy};
    ///
    /// DependenciesUpdate::default()
    ///     .repos(DependencySettingsUpdate::default()
    ///         .location("/data/repos")
    ///         .strategy(DependencyPassStrategy::Names));
    /// ```
    #[must_use]
    pub fn repos(mut self, repos: DependencySettingsUpdate) -> Self {
        self.repos = repos;
        self
    }

    /// Sets the tags settings that should be updated
    ///
    /// # Arguments
    ///
    /// * `tags` - The settings to update in this images repo dependencies
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{DependenciesUpdate, TagDependencySettingsUpdate, DependencyPassStrategy};
    ///
    /// DependenciesUpdate::default()
    ///     .tags(TagDependencySettingsUpdate::default()
    ///         .enable()
    ///         .location("/data/tags")
    ///         .strategy(DependencyPassStrategy::Names));
    /// ```
    #[must_use]
    pub fn tags(mut self, tags: TagDependencySettingsUpdate) -> Self {
        self.tags = tags;
        self
    }
}

/// Regex filters to apply to children before submission
///
/// By default, only children that match any of the given filters will
/// be submitted. If  If `submit_non_matches`is set, only children that
/// do *not* match *any* of the given filters will be submitted. If
/// no filters are given, all children will be submitted.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ChildFilters {
    /// Any filters to apply to the MIME type
    pub mime: HashSet<String>,
    /// Any filters to apply to the file name (including the extension)
    pub file_name: HashSet<String>,
    /// Any filters to apply to the file extension, not including the dot
    /// (e.g. "txt", "so", "exe", etc.)
    pub file_extension: HashSet<String>,
    /// Submit children that do *not* match any of the given filters rather
    /// than ones that do match
    pub submit_non_matches: bool,
}

impl ChildFilters {
    /// Returns true if `self` contains no child filters
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mime.is_empty() && self.file_name.is_empty() && self.file_extension.is_empty()
    }

    /// Add a mime child filter regular expression
    ///
    /// The regular expression must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add
    #[must_use]
    pub fn mime<T: Into<String>>(mut self, filter: T) -> Self {
        self.mime.insert(filter.into());
        self
    }

    /// Add multiple mime child filter regular expressions
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add
    #[must_use]
    pub fn mimes<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.mime.extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a file name child filter regular expression
    ///
    /// The regular expression must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add
    #[must_use]
    pub fn file_name<T: Into<String>>(mut self, filter: T) -> Self {
        self.file_name.insert(filter.into());
        self
    }

    /// Add multiple file name child filter regular expressions
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add
    #[must_use]
    pub fn file_names<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.file_name.extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a file extension child filter regular expression
    ///
    /// The regular expression must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add
    #[must_use]
    pub fn file_extension<T: Into<String>>(mut self, filter: T) -> Self {
        self.file_extension.insert(filter.into());
        self
    }

    /// Add multiple file extension child filter regular expressions
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add
    #[must_use]
    pub fn file_extensions<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.file_extension
            .extend(filters.into_iter().map(Into::into));
        self
    }

    /// Only submit children that *don't* match any the child filters
    /// rather than those that do
    #[must_use]
    pub fn submit_non_matches(mut self) -> Self {
        self.submit_non_matches = true;
        self
    }
}

/// An update to an image's child filters
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ChildFiltersUpdate {
    /// The mime filters to add
    #[serde(default)]
    pub add_mime: HashSet<String>,
    /// The mime filters to remove
    #[serde(default)]
    pub remove_mime: HashSet<String>,
    /// The file name filters to add
    #[serde(default)]
    pub add_file_name: HashSet<String>,
    /// The file name filters to remove
    #[serde(default)]
    pub remove_file_name: HashSet<String>,
    /// The file extension filters to add
    #[serde(default)]
    pub add_file_extension: HashSet<String>,
    /// The file extension filters to remove
    #[serde(default)]
    pub remove_file_extension: HashSet<String>,
    #[serde(default)]
    pub submit_non_matches: Option<bool>,
}

impl ChildFiltersUpdate {
    /// Add a mime child filter regular expression to add
    ///
    /// The regular expression must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add
    #[must_use]
    pub fn add_mime<T: Into<String>>(mut self, filter: T) -> Self {
        self.add_mime.insert(filter.into());
        self
    }

    /// Add multiple mime child filter regular expressions to add
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add
    #[must_use]
    pub fn add_mimes<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.add_mime.extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a mime child filter regular expression to remove
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add for removal
    #[must_use]
    pub fn remove_mime<T: Into<String>>(mut self, filter: T) -> Self {
        self.remove_mime.insert(filter.into());
        self
    }

    /// Add multiple mime child filter regular expressions to remove
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add for removal
    #[must_use]
    pub fn remove_mimes<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.remove_mime.extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a file name child filter regular expression to add
    ///
    /// The regular expression must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add
    #[must_use]
    pub fn add_file_name<T: Into<String>>(mut self, filter: T) -> Self {
        self.add_file_name.insert(filter.into());
        self
    }

    /// Add multiple file name child filter regular expressions to add
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add
    #[must_use]
    pub fn add_file_names<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.add_file_name
            .extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a file name filter regular expression to remove
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add for removal
    #[must_use]
    pub fn remove_file_name<T: Into<String>>(mut self, filter: T) -> Self {
        self.remove_file_name.insert(filter.into());
        self
    }

    /// Add multiple file name filter regular expressions to remove
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add for removal
    #[must_use]
    pub fn remove_file_names<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.remove_file_name
            .extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a file extension child filter regular expression to add
    ///
    /// The regular expression must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add
    #[must_use]
    pub fn add_file_extension<T: Into<String>>(mut self, filter: T) -> Self {
        self.add_file_extension.insert(filter.into());
        self
    }

    /// Add multiple file extension child filter regular expressions to add
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add
    #[must_use]
    pub fn add_file_extensions<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.add_file_extension
            .extend(filters.into_iter().map(Into::into));
        self
    }

    /// Add a file extension filter regular expression to remove
    ///
    /// # Arguments
    ///
    /// * `filter` - The filter to add for removal
    #[must_use]
    pub fn remove_file_extension<T: Into<String>>(mut self, filter: T) -> Self {
        self.remove_file_extension.insert(filter.into());
        self
    }

    /// Add multiple file extension filter regular expressions to remove
    ///
    /// # Arguments
    ///
    /// * `filters` - The filters to add for removal
    #[must_use]
    pub fn remove_file_extensions<T, I>(mut self, filters: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.remove_file_extension
            .extend(filters.into_iter().map(Into::into));
        self
    }

    /// Set whether to only submit children that match or *don't* match
    /// any the child filters, `false` to submit those that match and `true`
    /// to submit those that don't match
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set for this setting
    #[must_use]
    pub fn submit_non_matches(mut self, value: bool) -> Self {
        self.submit_non_matches = Some(value);
        self
    }
}

impl PartialEq<ChildFilters> for ChildFiltersUpdate {
    /// Check that a [`ChildFiltersUpdate`] was properly applied
    /// to the given `ChildFilters`
    fn eq(&self, filters: &ChildFilters) -> bool {
        // remove any mime filters to add that would be removed
        let mut mime_added = self.add_mime.difference(&self.remove_mime);
        matches_adds_iter!(filters.mime.iter(), mime_added);
        matches_removes!(filters.mime, self.remove_mime);
        // remove any file name filters to add that would be removed
        let mut file_name_added = self.add_file_name.difference(&self.remove_file_name);
        matches_adds_iter!(filters.file_name.iter(), file_name_added);
        matches_removes!(filters.file_name, self.remove_file_name);
        // remove any file extension filters to add that would be removed
        let mut file_extension_added = self
            .add_file_extension
            .difference(&self.remove_file_extension);
        matches_adds_iter!(filters.file_extension.iter(), file_extension_added);
        matches_removes!(filters.file_extension, self.remove_file_extension);
        // check that `submit_non_matches` was set properly
        matches_update!(filters.submit_non_matches, self.submit_non_matches);
        true
    }
}

/// The scaler that is responsible for scaling an image
#[derive(
    Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy, clap::ValueEnum, Default, Hash,
)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum ImageScaler {
    /// This image will be scheduled in k8s
    #[default]
    K8s,
    /// This image will be scheduled on baremetal
    BareMetal,
    /// This image will be spawned on Windows
    Windows,
    /// This image will be spawned on KVM vms
    Kvm,
    /// This image will be scheduled by something outside of Thorium
    External,
}

impl std::fmt::Display for ImageScaler {
    /// write our scaler to this formatter
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ImageScaler {
    type Err = &'static str;
    /// Cast a str to an `ImageScaler`
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "k8s" | "K8s" => Ok(ImageScaler::K8s),
            "baremetal" | "BareMetal" => Ok(ImageScaler::BareMetal),
            "windows" | "Windows" => Ok(ImageScaler::Windows),
            "kvm" | "Kvm" => Ok(ImageScaler::Kvm),
            "external" | "External" => Ok(ImageScaler::External),
            _ => Err("expected `K8s` or `BareMetal` or `Windows` or 'Kvm' or `External`"),
        }
    }
}

impl ImageScaler {
    /// Cast an [`ImageScaler`] to a str
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            ImageScaler::K8s => "K8s",
            ImageScaler::BareMetal => "BareMetal",
            ImageScaler::Windows => "Windows",
            ImageScaler::Kvm => "Kvm",
            ImageScaler::External => "External",
        }
    }
}

/// Adds an arg based on its arg strategy
macro_rules! add_arg {
    ($setting:expr, $value:expr, $cmd:expr) => {
        match &$setting {
            ArgStrategy::None => (),
            ArgStrategy::Append => $cmd.push($value),
            ArgStrategy::Kwarg(kwarg) => {
                $cmd.push(kwarg.clone());
                $cmd.push($value)
            }
        }
    };
}

/// The settings for cleaning up an images cancelled jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Cleanup {
    /// How to pass in the id of the cancelled job
    pub job_id: ArgStrategy,
    /// how to pass in this images result path
    pub results: ArgStrategy,
    /// How to pass in the output dir for this tools result files
    pub result_files_dir: ArgStrategy,
    /// The clean up script to call
    pub script: String,
}

impl Cleanup {
    /// Create a new clean up script config
    ///
    /// # Arguments
    ///
    /// * `script` - The path to the clean up script to call
    pub fn new<S: Into<String>>(script: S) -> Self {
        Cleanup {
            job_id: ArgStrategy::None,
            results: ArgStrategy::None,
            result_files_dir: ArgStrategy::None,
            script: script.into(),
        }
    }

    /// Set the strategy to use for passing in the job id
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Cleanup, ArgStrategy};
    ///
    /// // create a clean up script config
    /// Cleanup::new("/scripts/cleanup.py")
    ///   .job_id(ArgStrategy::Kwarg("--job_id".to_string()));
    /// ```
    #[must_use]
    pub fn job_id(mut self, strategy: ArgStrategy) -> Self {
        // set the strategy for passing in our job id
        self.job_id = strategy;
        self
    }

    /// Set the strategy to use for passing in the results path
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Cleanup, ArgStrategy};
    ///
    /// // create a clean up script config
    /// Cleanup::new("/scripts/cleanup.py")
    ///   .results(ArgStrategy::Kwarg("--results".to_string()));
    /// ```
    #[must_use]
    pub fn results(mut self, strategy: ArgStrategy) -> Self {
        // set the strategy for passing in our results path
        self.results = strategy;
        self
    }

    /// Set the strategy to use for passing in the result files dir path
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{Cleanup, ArgStrategy};
    ///
    /// // create a clean up script config
    /// Cleanup::new("/scripts/cleanup.py")
    ///   .result_files_dir(ArgStrategy::Kwarg("--result_files_dir".to_string()));
    /// ```
    #[must_use]
    pub fn result_files_dir(mut self, strategy: ArgStrategy) -> Self {
        // set the strategy for passing in our result files dir path
        self.result_files_dir = strategy;
        self
    }

    /// Build the command used to call our cleanup script
    ///
    /// # Arguments
    ///
    /// * `job` - The job we are cancelling
    /// * `results` - The path to the results for the job we are cancelling
    /// * `result_files_dir` - The path to the result files for the job we are cancelling
    #[must_use]
    pub fn build(
        &self,
        job: &GenericJob,
        results: String,
        result_files_dir: String,
    ) -> Vec<String> {
        // build the command to call our clean up script with
        let mut cmd = vec![self.script.clone()];
        // add our job id if its configured
        add_arg!(self.job_id, job.id.to_string(), cmd);
        add_arg!(self.results, results, cmd);
        add_arg!(self.result_files_dir, result_files_dir, cmd);
        cmd
    }
}

/// The update to apply to settings for cleaning up an images cancelled jobs
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct CleanupUpdate {
    /// How to pass in the id of the cancelled job
    pub job_id: Option<ArgStrategy>,
    /// how to pass in this images result file path
    pub results: Option<ArgStrategy>,
    /// How to pass in the output dir for this tools result files
    pub result_files_dir: Option<ArgStrategy>,
    /// The clean up script to call
    pub script: Option<String>,
    /// Whether to clear our clean up settings
    pub clear: bool,
}

impl CleanupUpdate {
    /// Set the strategy to use for passing in the job id
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CleanupUpdate, ArgStrategy};
    ///
    /// // create a clean up script config
    /// CleanupUpdate::default()
    ///   .job_id(ArgStrategy::Kwarg("--job_id".to_string()));
    /// ```
    #[must_use]
    pub fn job_id(mut self, strategy: ArgStrategy) -> Self {
        // set the strategy for passing in our job id
        self.job_id = Some(strategy);
        self
    }

    /// Set the strategy to use for passing in the results path
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CleanupUpdate, ArgStrategy};
    ///
    /// // create a clean up script config
    /// CleanupUpdate::default()
    ///   .results(ArgStrategy::Kwarg("--results".to_string()));
    /// ```
    #[must_use]
    pub fn results(mut self, strategy: ArgStrategy) -> Self {
        // set the strategy for passing in our results path
        self.results = Some(strategy);
        self
    }

    /// Set the strategy to use for passing in the result files dir path
    ///
    /// # Arguments
    ///
    /// * `strategy` - The strategy to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{CleanupUpdate, ArgStrategy};
    ///
    /// // create a clean up script config
    /// CleanupUpdate::default()
    ///   .result_files_dir(ArgStrategy::Kwarg("--result_files_dir".to_string()));
    /// ```
    #[must_use]
    pub fn result_files_dir(mut self, strategy: ArgStrategy) -> Self {
        // set the strategy for passing in our result files dir path
        self.result_files_dir = Some(strategy);
        self
    }

    /// Set the clean up script to use
    ///
    /// # Arguments
    ///
    /// * `script` - The clean up script to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::CleanupUpdate;
    ///
    /// // create a clean up script config
    /// CleanupUpdate::default()
    ///   .script("/updated/cleanup.py");
    /// ```
    #[must_use]
    pub fn script<S: Into<String>>(mut self, script: S) -> Self {
        // set our clean up script path
        self.script = Some(script.into());
        self
    }

    /// Set this images clean up settings to be cleared
    #[must_use]
    pub fn clear(mut self) -> Self {
        self.clear = true;
        self
    }
}

/// A version of an image, formatted according to various standards
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreJson))]
pub enum ImageVersion {
    #[cfg_attr(feature = "api", schema(value_type = String))]
    SemVer(semver::Version),
    Custom(String),
}

impl From<&str> for ImageVersion {
    fn from(s: &str) -> Self {
        // check if the string can be parsed to semver
        if let Ok(semver) = semver::Version::parse(s) {
            ImageVersion::SemVer(semver)
        // otherwise convert to a custom version
        } else {
            ImageVersion::Custom(s.to_string())
        }
    }
}

impl From<&String> for ImageVersion {
    fn from(s: &String) -> Self {
        ImageVersion::from(s.as_str())
    }
}

/// This is a request for an image to be added to Thorium
///
/// None of the values in this have been bounds checked in any way yet
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageRequest {
    /// The group this image is in
    pub group: String,
    /// The name of this image
    pub name: String,
    /// The version of this image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<ImageVersion>,
    /// What scaler is responsible for scaling this image
    #[serde(default)]
    pub scaler: ImageScaler,
    /// The image to use (url or tag)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// The lifetime of a pod
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifetime: Option<ImageLifetime>,
    /// The timeout for individual jobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    /// The resources to request the container to have
    #[serde(default)]
    pub resources: ResourcesRequest,
    /// The limit to use for how many workers of this image type can be spawned
    #[serde(default)]
    pub spawn_limit: SpawnLimits,
    /// Any volumes to bind into this container
    #[serde(default)]
    pub volumes: Vec<Volume>,
    /// The environment args to set
    #[serde(default)]
    pub env: HashMap<String, Option<String>>,
    /// The arguments to add to this images jobs
    #[serde(default)]
    pub args: ImageArgs,
    /// The path to the modifier folders for this image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifiers: Option<String>,
    /// The image description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The security context for this image
    #[serde(default)]
    pub security_context: Option<SecurityContext>,
    /// Whether the agent should stream stdout/stderr back to Thorium
    #[serde(default = "default_as_true")]
    pub collect_logs: bool,
    /// Whether this is a generator or not
    #[serde(default = "default_as_false")]
    pub generator: bool,
    /// How to handle dependencies for this image
    #[serde(default)]
    pub dependencies: Dependencies,
    /// The type of display class to use in the UI for this images output
    #[serde(default)]
    pub display_type: OutputDisplayType,
    /// The settings for collecting results from this image
    #[serde(default)]
    pub output_collection: OutputCollection,
    /// Any regex filters to match on when uploading children
    ///
    /// If no filters are given, all children will be uploaded. For now, this is
    /// only being used to match on MIME headers in the agent, but it may have
    /// other uses in the future. Regular expressions must conform to standards
    /// according to the [regex crate](https://docs.rs/regex/latest/regex/) or an
    /// error will be returned.
    #[serde(default)]
    pub child_filters: ChildFilters,
    /// The settings to use when cleaning up canceled jobs
    pub clean_up: Option<Cleanup>,
    /// The settings to use for Kvm jobs
    pub kvm: Option<Kvm>,
    /// The set of network policies to apply to the image once it's been spawned
    ///
    /// This currently only applies to images scaled by K8's
    #[serde(default)]
    pub network_policies: HashSet<String>,
}

impl ImageRequest {
    /// Create a new basic [`ImageRequest`]
    ///
    /// # Arguments
    ///
    /// * `group` - The group this image is in
    /// * `name` - The name of this image
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{
    ///     ImageRequest, ImageLifetime, ResourcesRequest, Volume, VolumeTypes, Dependencies,
    ///     SampleDependencySettings, SecurityContext, Cleanup, ArgStrategy, ChildFilters
    /// };
    ///
    /// // create an image request for an image in the Corn group called harvester
    /// // This image is built on top of the Thorium:CornHarvester docker image
    /// // It has a lifetime of 1 job
    /// // A timeout of 300 seconds
    /// // The pod consumes 2 cores and 1 gibibyte of memory
    /// // it runs as the uid 1234 in group 3000 and disables privilege escalation
    /// // it is also a generator of sub reactions and will be recreated until exhausted
    /// let request = ImageRequest::new("Corn", "harvester")
    ///     .image("Thorium:CornHarvester")
    ///     .lifetime(ImageLifetime::jobs(1))
    ///     .timeout(300)
    ///     .resources(ResourcesRequest::default()
    ///         .millicpu(2000)
    ///         .memory("1Gi"))
    ///     .env("Field", "1")
    ///     .unset_env("Soybeans")
    ///     .volume(Volume::new("corn-vol", "/files", VolumeTypes::ConfigMap))
    ///     .security_context(SecurityContext::default().user(1234))
    ///     .generator()
    ///     .dependencies(Dependencies::default()
    ///         .samples(SampleDependencySettings::default()
    ///             .location("/tmp/downloads")))
    ///     .clean_up(Cleanup::new("/script/cleanup.sh")
    ///         .job_id(ArgStrategy::Append))
    ///     .child_filters(ChildFilters::default()
    ///         .mime("image"));
    /// ```
    pub fn new<T: Into<String>>(group: T, name: T) -> Self {
        ImageRequest {
            group: group.into(),
            name: name.into(),
            version: None,
            scaler: ImageScaler::default(),
            image: None,
            lifetime: None,
            timeout: None,
            resources: ResourcesRequest::default(),
            spawn_limit: SpawnLimits::Unlimited,
            volumes: Vec::default(),
            env: HashMap::default(),
            args: ImageArgs::default(),
            modifiers: None,
            description: None,
            security_context: None,
            collect_logs: true,
            generator: false,
            dependencies: Dependencies::default(),
            display_type: OutputDisplayType::default(),
            output_collection: OutputCollection::default(),
            child_filters: ChildFilters::default(),
            clean_up: None,
            kvm: None,
            network_policies: HashSet::default(),
        }
    }

    /// Sets the version of this image
    ///
    /// # Arguments
    ///
    /// * `version` - The semver image version
    #[must_use]
    pub fn version(mut self, version: ImageVersion) -> Self {
        self.version = Some(version);
        self
    }

    /// Set the scaler type this image should use
    ///
    /// # Arguments
    ///
    /// * `scaler` - The scaler type to set
    #[must_use]
    pub fn scaler(mut self, scaler: ImageScaler) -> Self {
        // update our scaler type
        self.scaler = scaler;
        self
    }

    /// Set the docker image this [`ImageRequest`] is built on
    ///
    /// # Arguments
    ///
    /// * `image` - The url/name of the image to set
    #[must_use]
    pub fn image<T: Into<String>>(mut self, image: T) -> Self {
        self.image = Some(image.into());
        self
    }

    /// Set the lifetime this [`ImageRequest`] should have
    ///
    /// Image lifetime is how long an image should live not how long a job being executed in this
    /// image should live. Currently you can base image lifetime on number of jobs executed or
    /// time. This is checked in between claiming jobs and so is not strongly enforced.
    ///
    /// # Arguments
    ///
    /// * `lifetime` - How an image/pod should determine its lifetime
    #[must_use]
    pub fn lifetime(mut self, lifetime: ImageLifetime) -> Self {
        // build and set lifetime
        self.lifetime = Some(lifetime);
        self
    }

    /// Set the timeout this [`ImageRequest`] should have
    ///
    /// Image timeout is the max time any particular job being executed by the agent in this image
    /// should be executed before being aborted for.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The max number of seconds a job for this image can execute for
    #[must_use]
    pub fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the resources to require when spawning this image
    ///
    /// # Arguments
    ///
    /// * `requests` - The Resources to require for this image
    #[must_use]
    pub fn resources(mut self, resources: ResourcesRequest) -> Self {
        self.resources = resources;
        self
    }

    /// Sets the limit to use for workers spawned for this image
    ///
    /// This max is across all clusters for a specific scaler.
    ///
    /// # Arguments
    ///
    /// * `max` - The max number of this worker to spawn
    #[must_use]
    pub fn spawn_limit(mut self, limit: SpawnLimits) -> Self {
        self.spawn_limit = limit;
        self
    }

    /// Adds an environment variable to set inside this image
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the environment arg to set
    /// * `value` - The value to set
    #[must_use]
    pub fn env<T: Into<String>>(mut self, name: T, value: T) -> Self {
        // set this environment variable
        self.env.insert(name.into(), Some(value.into()));
        self
    }

    /// Adds an environment variable to unset inside this image
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the environment arg to unset
    #[must_use]
    pub fn unset_env<T: Into<String>>(mut self, name: T) -> Self {
        // set this environment variable
        self.env.insert(name.into(), None);
        self
    }

    /// Adds a volume to be bound inside this image
    ///
    /// # Arguments
    ///
    /// * `volume` - The volume to bind in
    #[must_use]
    pub fn volume(mut self, vol: Volume) -> Self {
        // add this volume to our volume list
        self.volumes.push(vol);
        self
    }

    /// Sets the modifiers path in this image request
    ///
    /// # Arguments
    ///
    /// * `path` - The path the Thorium agent should look for modifiers at
    #[must_use]
    pub fn modifiers<T: Into<String>>(mut self, path: T) -> Self {
        // set the modifier path
        self.modifiers = Some(path.into());
        self
    }

    /// Set the image description
    ///
    /// # Arguments
    ///
    /// * `description` - The image description to set
    #[must_use]
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the security context settings for this image
    ///
    /// Only admins can change these settings.
    ///
    /// # Arguments
    ///
    /// * `user` - The user id to run as
    #[must_use]
    pub fn security_context(mut self, security_context: SecurityContext) -> Self {
        self.security_context = Some(security_context);
        self
    }

    /// Disables log collection
    #[must_use]
    pub fn disable_logs(mut self) -> Self {
        self.collect_logs = false;
        self
    }

    /// Tells Thorium this image is a generator of sub reactions
    ///
    /// This means that Thorium should loop this image until it has told Thorium
    /// it no longer needs to spawn more sub reactions.
    #[must_use]
    pub fn generator(mut self) -> Self {
        self.generator = true;
        self
    }

    /// The dependency settings to use for this image
    ///
    /// # Arguments
    ///
    /// * `dependencies` - The dependency settings to set
    #[must_use]
    pub fn dependencies(mut self, dependencies: Dependencies) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// The display class to use when displaying this images output in the UI
    ///
    /// # Arguments
    ///
    /// * `display_type` - The display type to set
    #[must_use]
    pub fn display_type(mut self, display_type: OutputDisplayType) -> Self {
        self.display_type = display_type;
        self
    }

    /// Set the output collection settings
    ///
    /// # Arguments
    ///
    /// * `collection` - The output collection settings to set
    #[must_use]
    pub fn output_collection(mut self, output_collection: OutputCollection) -> Self {
        self.output_collection = output_collection;
        self
    }

    /// Add child filter regular expression
    ///
    /// The regular expressions must conform to the standards set in the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or a 400 error will
    /// be returned by the API.
    ///
    /// # Arguments
    ///
    /// * `child_filters` - The child filters to set
    #[must_use]
    pub fn child_filters(mut self, child_filters: ChildFilters) -> Self
where {
        self.child_filters = child_filters;
        self
    }

    /// Set the clean up script settings
    ///
    /// # Arguments
    ///
    /// * `collection` - The clean up settings to set
    #[must_use]
    pub fn clean_up(mut self, clean_up: Cleanup) -> Self {
        self.clean_up = Some(clean_up);
        self
    }

    /// Set the kvm settings
    ///
    /// # Arguments
    ///
    /// * `kvm` - The kvm settings to set
    #[must_use]
    pub fn kvm(mut self, kvm: Kvm) -> Self {
        self.kvm = Some(kvm);
        self
    }

    /// Add the name of a network policy to apply to the image when it's spawned
    ///
    /// This currently only applies when the image is spawned with K8's
    ///
    /// # Arguments
    ///
    /// * `network_policy` - The name of the network policy to add
    #[must_use]
    pub fn network_policy<T: Into<String>>(mut self, network_policy: T) -> Self {
        self.network_policies.insert(network_policy.into());
        self
    }

    /// Add names of a network policy to apply to the image when it's spawned
    ///
    /// This currently only applies when the image is spawned with K8's
    ///
    /// # Arguments
    ///
    /// * `network_policies` - The names of the network policy to add
    #[must_use]
    pub fn network_policies<I, T>(mut self, network_policies: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.network_policies
            .extend(network_policies.into_iter().map(Into::into));
        self
    }
}

impl From<Image> for ImageRequest {
    fn from(image: Image) -> Self {
        // convert our resources to a resources request
        let resources = ResourcesRequest::from(image.resources);
        // convert our image into an image request
        ImageRequest {
            group: image.group,
            name: image.name,
            version: image.version,
            scaler: image.scaler,
            image: image.image,
            lifetime: image.lifetime,
            timeout: image.timeout,
            resources,
            spawn_limit: image.spawn_limit,
            volumes: image.volumes,
            env: image.env,
            args: image.args,
            modifiers: image.modifiers,
            description: image.description,
            security_context: Some(image.security_context),
            collect_logs: image.collect_logs,
            generator: image.generator,
            dependencies: image.dependencies,
            display_type: image.display_type,
            output_collection: image.output_collection,
            child_filters: image.child_filters,
            clean_up: image.clean_up,
            kvm: image.kvm,
            network_policies: image.network_policies,
        }
    }
}

/// The settings for kvm jobs
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct KvmUpdate {
    /// The path to the golden XML file to use
    pub xml: Option<String>,
    /// The path to the golden qcow2 image to use
    pub qcow2: Option<String>,
}

/// An update to the image ban list containing bans to be added or removed
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageBanUpdate {
    /// The list of bans to be added
    pub bans_added: Vec<ImageBan>,
    /// The list of bans to be removed
    pub bans_removed: Vec<Uuid>,
}

impl ImageBanUpdate {
    /// Returns true if no bans are set to be added or removed
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bans_added.is_empty() && self.bans_removed.is_empty()
    }

    /// Add a ban to be added to the image ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `ban` - The ban to be added
    #[must_use]
    pub fn add_ban(mut self, ban: ImageBan) -> Self {
        self.bans_added.push(ban);
        self
    }

    /// Add multiple bans to be added to the image ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `bans` - The bans to be added
    #[must_use]
    pub fn add_bans(mut self, mut bans: Vec<ImageBan>) -> Self {
        self.bans_added.append(&mut bans);
        self
    }

    /// Add a ban to be removed from the image ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the ban to be removed
    #[must_use]
    pub fn remove_ban(mut self, id: Uuid) -> Self {
        self.bans_removed.push(id);
        self
    }

    /// Add multiple bans to be removed from the image ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `ids` - The id's of the bans to be removed
    #[must_use]
    pub fn remove_bans(mut self, mut ids: Vec<Uuid>) -> Self {
        self.bans_removed.append(&mut ids);
        self
    }
}

/// An update to the network policy list containing network policies
/// to be added or removed
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageNetworkPolicyUpdate {
    /// The list of policies to be added
    pub policies_added: HashSet<String>,
    /// The list of policies to be removed
    pub policies_removed: HashSet<String>,
}

impl ImageNetworkPolicyUpdate {
    /// Returns true if no policies are set to be added or removed
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.policies_added.is_empty() && self.policies_removed.is_empty()
    }

    /// Add a network policy to be added to the network policy list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `policy` - The network policy to be added
    #[must_use]
    pub fn add_policy<T: Into<String>>(mut self, policy: T) -> Self {
        self.policies_added.insert(policy.into());
        self
    }

    /// Add multiple policies to be added to the network policy list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `policies` - The policies to be added
    #[must_use]
    pub fn add_policies<I, T>(mut self, policies: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.policies_added
            .extend(policies.into_iter().map(Into::into));
        self
    }

    /// Add a network policy to be removed from the network policy list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `policy` - The network policy to be removed
    #[must_use]
    pub fn remove_policy<T: Into<String>>(mut self, policy: T) -> Self {
        self.policies_removed.insert(policy.into());
        self
    }

    /// Add multiple policies to be removed from the network policy list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `policies` - The policies to be removed
    #[must_use]
    pub fn remove_policies<I, T>(mut self, policies: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.policies_removed
            .extend(policies.into_iter().map(Into::into));
        self
    }
}

/// An update for an image in Thorium
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageUpdate {
    /// The image version to update
    pub version: Option<ImageVersion>,
    /// Whether the scaler should spawn containers for this image
    pub external: Option<bool>,
    /// The image to use (url or tag)
    pub image: Option<String>,
    /// What scaler is responsible for scaling this image
    pub scaler: Option<ImageScaler>,
    /// The lifetime of a pod
    pub lifetime: Option<ImageLifetime>,
    /// The timeout for individual jobs
    pub timeout: Option<u64>,
    /// The resources to require for this image
    pub resources: Option<ResourcesUpdate>,
    /// The limit to use for how many workers of this image type can be spawned
    pub spawn_limit: Option<SpawnLimits>,
    /// The volumes to add
    #[serde(default)]
    pub add_volumes: Vec<Volume>,
    /// The names of the volumes to remove
    #[serde(default)]
    pub remove_volumes: Vec<String>,
    /// Environment args to add
    #[serde(default)]
    pub add_env: HashMap<String, Option<String>>,
    /// Environment args to remove
    #[serde(default)]
    pub remove_env: Vec<String>,
    /// Whether to clear the version or not
    #[serde(default = "default_as_false")]
    pub clear_version: bool,
    /// Whether to clear the image or not
    #[serde(default = "default_as_false")]
    pub clear_image: bool,
    /// Whether to clear the lifetime or not
    #[serde(default = "default_as_false")]
    pub clear_lifetime: bool,
    /// Whether to clear the description or not
    #[serde(default = "default_as_false")]
    pub clear_description: bool,
    /// The arguments to add to this images jobs
    pub args: Option<ImageArgsUpdate>,
    /// The path to the modifier folders for this image
    pub modifiers: Option<String>,
    /// The image description
    pub description: Option<String>,
    /// The updates to the security context for this image
    pub security_context: Option<SecurityContextUpdate>,
    /// Whether the agent should stream stdout/stderr back to Thorium
    pub collect_logs: Option<bool>,
    /// Whether this is a generator or not
    pub generator: Option<bool>,
    /// Updates the dependency settings for this image
    #[serde(default)]
    pub dependencies: DependenciesUpdate,
    /// The type of display class to use in the UI for this images output
    pub display_type: Option<OutputDisplayType>,
    /// The settings for collecting results from this image
    #[serde(default)]
    pub output_collection: Option<OutputCollectionUpdate>,
    /// An update to the image's child filters
    #[serde(default)]
    pub child_filters: Option<ChildFiltersUpdate>,
    /// The settings to use when cleaning up canceled jobs
    #[serde(default)]
    pub clean_up: CleanupUpdate,
    /// The settings to use for Kvm jobs
    #[serde(default)]
    pub kvm: KvmUpdate,
    /// An update to the ban list containing a list of bans to add or remove
    #[serde(default)]
    pub bans: ImageBanUpdate,
    /// An update to the network policies to apply to the image
    #[serde(default)]
    pub network_policies: ImageNetworkPolicyUpdate,
}

impl ImageUpdate {
    /// Sets the external flag to true
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().external();
    /// ```
    #[must_use]
    pub fn external(mut self) -> Self {
        self.external = Some(true);
        self
    }

    /// Sets the image string in a [`ImageUpdate`]
    ///
    /// # Arguments
    ///
    /// * `image` - The new image url/name to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().image("Rust:1.48.0");
    /// ```
    #[must_use]
    pub fn image<T: Into<String>>(mut self, image: T) -> Self {
        self.image = Some(image.into());
        self
    }

    /// Sets the image version in a [`ImageUpdate`]
    ///
    /// # Arguments
    ///
    /// * `version` - The semver version (see [semver specifications](https://semver.org))
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ImageUpdate, ImageVersion};
    /// use semver::Version;
    ///
    /// let version = ImageVersion::SemVer(Version::parse("1.1.0").unwrap());
    /// ImageUpdate::default().version(version);
    /// ```
    #[must_use]
    pub fn version(mut self, version: ImageVersion) -> Self {
        self.version = Some(version);
        self
    }

    /// Sets the scaler an image should use in a [`ImageUpdate`]
    ///
    /// # Arguments
    ///
    /// * `scaler` - The new scaler to spawn under
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ImageUpdate, ImageScaler};
    ///
    /// ImageUpdate::default().scaler(ImageScaler::K8s);
    /// ```
    #[must_use]
    pub fn scaler(mut self, scaler: ImageScaler) -> Self {
        self.scaler = Some(scaler);
        self
    }

    /// Sets [`ImageLifetime`] to update an [`Image`] with
    ///
    /// # Arguments
    ///
    /// * `lifetime` - The new lifetime to enforce
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ImageUpdate, ImageLifetime};
    ///
    /// ImageUpdate::default().lifetime(ImageLifetime::jobs(1));
    /// ```
    #[must_use]
    pub fn lifetime(mut self, lifetime: ImageLifetime) -> Self {
        self.lifetime = Some(lifetime);
        self
    }

    /// Sets the timeout in seconds to weakly enforce on jobs executed in this image.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The new job timeout to weakly enforce
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().timeout(300);
    /// ```
    #[must_use]
    pub fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the [`ResourceRequest`] this image should require
    ///
    /// # Arguments
    ///
    /// * `requests` - The resources this image requires to be spawned
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::{ImageUpdate, ResourcesUpdate};
    ///
    /// // build resource requests with 2 cores, 4Gi of ram and 128Gi of storage
    /// let resources = ResourcesUpdate::default()
    ///     .millicpu(2000)
    ///     .memory("4Gi")
    ///     .storage("128Gi");
    /// ImageUpdate::default().resources(resources);
    /// ```
    #[must_use]
    pub fn resources(mut self, resources: ResourcesUpdate) -> Self {
        self.resources = Some(resources);
        self
    }

    /// Sets the limit to use for workers spawned for this image
    ///
    /// This max is across all clusters for a specific scaler.
    ///
    /// # Arguments
    ///
    /// * `max` - The max number of this worker to spawn
    #[must_use]
    pub fn spawn_limit(mut self, limit: SpawnLimits) -> Self {
        self.spawn_limit = Some(limit);
        self
    }

    /// Adds a new [`Volume`] to add to the [`Image`] in this update
    ///
    /// # Arguments
    ///
    /// * `volume` - A volume to add
    #[must_use]
    pub fn add_volume(mut self, volume: Volume) -> Self {
        self.add_volumes.push(volume);
        self
    }

    /// Adds a [`Volume`] to be removed from this [`Image`] in this update
    ///
    /// # Arguments
    ///
    /// * `volume` - A volume to remove
    #[must_use]
    pub fn remove_volume<T: Into<String>>(mut self, volume: T) -> Self {
        self.remove_volumes.push(volume.into());
        self
    }

    /// Adds a list of [`Volume`]s to be removed from this [`Image`] in this update
    ///
    /// # Arguments
    ///
    /// * `volume` - A volume to remove
    #[must_use]
    pub fn remove_volumes<T: Into<String>>(mut self, volumes: Vec<T>) -> Self {
        // collect into a vector of strings
        let volumes = volumes.into_iter().map(Into::into).collect::<Vec<String>>();
        self.remove_volumes.extend(volumes);
        self
    }

    /// Adds or modifies an environment variable to an [`Image`]
    ///
    /// # Arguments
    ///
    /// * `key` - The name of the environment variable to add
    /// * `val` - The value of the environment variable to set
    #[must_use]
    pub fn add_env<T: Into<String>>(mut self, key: T, val: Option<T>) -> Self {
        // convert our value to a String if one was passed
        match val {
            Some(val) => self.add_env.insert(key.into(), Some(val.into())),
            None => self.add_env.insert(key.into(), None),
        };
        self
    }

    /// Removes an environment variable to an [`Image`]
    ///
    /// # Arguments
    ///
    /// * `key` - The name of the environment variable to remove
    #[must_use]
    pub fn remove_env<T: Into<String>>(mut self, key: T) -> Self {
        // insert into the list of env vars to remove
        self.remove_env.push(key.into());
        self
    }

    /// Sets the clear version flag to true
    ///
    /// This will clear the image's current version number and set it to None.
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().clear_version();
    /// ```
    #[must_use]
    pub fn clear_version(mut self) -> Self {
        self.clear_version = true;
        self
    }

    /// Sets the clear image flag to true
    ///
    /// This will clear the images current image url/name and set it to None.
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().clear_image();
    /// ```
    #[must_use]
    pub fn clear_image(mut self) -> Self {
        self.clear_image = true;
        self
    }

    /// Sets the clear lifetime flag to true
    ///
    /// This will clear the images current lifetime and set it to None.
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().clear_lifetime();
    /// ```
    #[must_use]
    pub fn clear_lifetime(mut self) -> Self {
        self.clear_lifetime = true;
        self
    }

    /// Sets the clear description flag to true
    ///
    /// This will clear the images current description and set it to None.
    ///
    /// ```
    /// use thorium::models::ImageUpdate;
    ///
    /// ImageUpdate::default().clear_description();
    /// ```
    #[must_use]
    pub fn clear_description(mut self) -> Self {
        self.clear_description = true;
        self
    }

    /// Sets the modifiers path in this image update
    ///
    /// # Arguments
    ///
    /// * `path` - The path the Thorium agent should look for modifiers at
    #[must_use]
    pub fn modifiers<T: Into<String>>(mut self, path: T) -> Self {
        // set the modifier path
        self.modifiers = Some(path.into());
        self
    }

    /// Set the image description
    ///
    /// # Arguments
    ///
    /// * `description` - The image description to set
    #[must_use]
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Disables log collection
    #[must_use]
    pub fn enable_logs(mut self) -> Self {
        self.collect_logs = Some(true);
        self
    }

    /// Disables log collection
    #[must_use]
    pub fn disable_logs(mut self) -> Self {
        self.collect_logs = Some(false);
        self
    }

    /// Tells Thorium this image is a generator of sub reactions
    ///
    /// This means that Thorium should loop this image until it has told Thorium
    /// it no longer needs to spawn more sub reactions.
    #[must_use]
    pub fn enable_generator(mut self) -> Self {
        self.generator = Some(true);
        self
    }

    /// Tells Thorium this image is not a generator of sub reactions
    #[must_use]
    pub fn disable_generator(mut self) -> Self {
        self.generator = Some(true);
        self
    }

    /// The updated dependency settings to use for this image
    ///
    /// # Arguments
    ///
    /// * `dependencies` - The dependency settings to update
    #[must_use]
    pub fn dependencies(mut self, dependencies: DependenciesUpdate) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// The display class to use when displaying this images output in the UI
    ///
    /// # Arguments
    ///
    /// * `display_type` - The display type to set
    #[must_use]
    pub fn display_type(mut self, display_type: OutputDisplayType) -> Self {
        self.display_type = Some(display_type);
        self
    }

    /// The output collection settings to use for this image
    ///
    /// # Arguments
    ///
    /// * `collection` - The output collection settings to use
    #[must_use]
    pub fn output_collection(mut self, collection: OutputCollectionUpdate) -> Self {
        self.output_collection = Some(collection);
        self
    }

    /// The child filter update to apply to this image
    ///
    /// # Arguments
    ///
    /// * `child_filters` - The child filter update
    #[must_use]
    pub fn child_filters(mut self, child_filters: ChildFiltersUpdate) -> Self {
        self.child_filters = Some(child_filters);
        self
    }

    /// Set the clean up script settings
    ///
    /// # Arguments
    ///
    /// * `collection` - The clean up settings to set
    #[must_use]
    pub fn clean_up(mut self, clean_up: CleanupUpdate) -> Self {
        self.clean_up = clean_up;
        self
    }

    /// Set the image bans to add/remove
    ///
    /// # Arguments
    ///
    /// * `bans` - The bans to add/remove
    #[must_use]
    pub fn bans(mut self, bans: ImageBanUpdate) -> Self {
        self.bans = bans;
        self
    }

    /// Set the network policies to add/remove
    ///
    /// # Arguments
    ///
    /// * `network_policies` - The network policies to add/remove
    #[must_use]
    pub fn network_policies(mut self, network_policies: ImageNetworkPolicyUpdate) -> Self {
        self.network_policies = network_policies;
        self
    }
}

/// The settings for kvm jobs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Kvm {
    /// The path to the golden XML file to use
    pub xml: String,
    /// The path to the golden qcow2 image to use
    pub qcow2: String,
}

/// The various kinds of bans an image can have
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum ImageBanKind {
    /// A generic ban manually set by an admin
    Generic(GenericBan),
    /// Created when the image URL is not reachable or invalid
    InvalidImageUrl(InvalidUrlBan),
    /// Created when the given host path is not valid or not on the whitelist (see [`super::SystemSettings`])
    InvalidHostPath(InvalidHostPathBan),
}

impl ImageBanKind {
    /// Create a new [`ImageBanKind::Generic`] ban type
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to set for the ban
    pub fn generic<T: Into<String>>(msg: T) -> Self {
        Self::Generic(GenericBan { msg: msg.into() })
    }

    /// Create a new [`ImageBanKind::InvalidImageUrl`] ban type
    ///
    /// # Arguments
    ///
    /// * `url` - The URL that caused the ban
    pub fn image_url<T: Into<String>>(url: T) -> Self {
        Self::InvalidImageUrl(InvalidUrlBan { url: url.into() })
    }

    /// Create a new [`ImageBanKind::InvalidHostPath`] ban type
    ///
    /// # Arguments
    ///
    /// * `volume_name` - The name of the volume that caused the ban
    /// * `host_path` - The host path that caused the ban
    pub fn host_path<T, P>(volume_name: T, host_path: P) -> Self
    where
        T: Into<String>,
        P: Into<PathBuf>,
    {
        Self::InvalidHostPath(InvalidHostPathBan {
            volume_name: volume_name.into(),
            host_path: host_path.into(),
        })
    }
}

/// Contains data related to a [`ImageBanKind::Generic`]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GenericBan {
    /// A message containing the reason this image was banned
    pub msg: String,
}

/// Contains data related to a [`ImageBanKind::InvalidImageUrl`]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct InvalidUrlBan {
    /// The url that resulted in the ban
    pub url: String,
}

/// Contains data related to a [`ImageBanKind::InvalidHostPath`]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct InvalidHostPathBan {
    /// The name of the volume that caused the ban
    pub volume_name: String,
    /// The host path that caused the ban
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub host_path: PathBuf,
}

/// A particular reason an image has been banned
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageBan {
    /// The unique id for this ban
    pub id: Uuid,
    /// The time in UTC that the ban was made
    pub time_banned: DateTime<Utc>,
    /// The kind of ban this is
    pub ban_kind: ImageBanKind,
}

impl ImageBan {
    /// Create a new `ImageBan`
    ///
    /// # Arguments
    ///
    /// * `ban_type` - The kind of ban we're creating
    #[must_use]
    pub fn new(ban_kind: ImageBanKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            time_banned: Utc::now(),
            ban_kind,
        }
    }
}

impl Ban<Image> for ImageBan {
    fn id(&self) -> &Uuid {
        &self.id
    }

    fn msg(&self) -> String {
        // create a message based on the kind of ban
        match &self.ban_kind {
            ImageBanKind::Generic(ban) => ban.msg.clone(),
            ImageBanKind::InvalidImageUrl(ban) => format!(
                "The image URL '{}' cannot be reached or is invalid!",
                ban.url
            ),
            ImageBanKind::InvalidHostPath(ban) => format!(
                "The image volume '{}' has a host path of '{}' that is \
            not on the list of allowed host paths! Ask an admin to add it to the allowed list or \
            pick an allowed host path.",
                ban.volume_name,
                ban.host_path.to_string_lossy()
            ),
        }
    }

    fn time_banned(&self) -> &DateTime<Utc> {
        &self.time_banned
    }
}

/// Image that can be used in a pipeline
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Image {
    /// The group this image is in
    pub group: String,
    /// The name of this image
    pub name: String,
    /// The creator of this image
    pub creator: String,
    /// The version of this image
    pub version: Option<ImageVersion>,
    /// What scaler is responsible for scaling this image
    #[serde(default)]
    pub scaler: ImageScaler,
    /// The image to use (url or tag)
    pub image: Option<String>,
    /// The lifetime of a pod
    pub lifetime: Option<ImageLifetime>,
    /// The timeout for individual jobs
    pub timeout: Option<u64>,
    /// The resources to required to spawn this image
    pub resources: Resources,
    /// The limit to use for how many workers of this image type can be spawned
    pub spawn_limit: SpawnLimits,
    /// The environment variables to set
    #[serde(default)]
    pub env: HashMap<String, Option<String>>,
    /// How long this image takes to execute on average in seconds (defaults to
    /// 10 minutes on image creation).
    pub runtime: f64,
    /// Any volumes to bind into this container
    pub volumes: Vec<Volume>,
    /// The arguments to add to this images jobs
    #[serde(default)]
    pub args: ImageArgs,
    /// The path to the modifier folders for this image
    pub modifiers: Option<String>,
    /// The image description
    pub description: Option<String>,
    /// The security context for this image
    pub security_context: SecurityContext,
    /// The pipelines that are using this image
    pub used_by: Vec<String>,
    /// Whether the agent should stream stdout/stderr back to Thorium
    pub collect_logs: bool,
    /// Whether this is a generator or not
    pub generator: bool,
    /// How to handle dependencies for this image
    #[serde(default)]
    pub dependencies: Dependencies,
    /// The type of display class to use in the UI for this images output
    #[serde(default)]
    pub display_type: OutputDisplayType,
    /// The settings for collecting results from this image
    #[serde(default)]
    pub output_collection: OutputCollection,
    /// Any regex filters to match on when uploading children
    ///
    /// If no filters are given, all children will be uploaded. Regular expressions
    /// must conform to standards according to the
    /// [regex crate](https://docs.rs/regex/latest/regex/) or an error will be
    /// returned on image creation/update.
    #[serde(default)]
    pub child_filters: ChildFilters,
    /// The settings to use when cleaning up canceled jobs
    pub clean_up: Option<Cleanup>,
    /// The settings to use for Kvm jobs
    pub kvm: Option<Kvm>,
    /// A list of reasons an image is banned mapped by ban UUID;
    /// if the list has any bans, the image cannot be spawned
    pub bans: HashMap<Uuid, ImageBan>,
    /// A set of the names of network policies to apply to the image when it's spawned
    ///
    /// Only applies when scaled with K8's currently
    #[serde(default)]
    pub network_policies: HashSet<String>,
}

impl PartialEq<ImageRequest> for Image {
    /// Check if a [`ImageRequest`] and a [`Image`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The `ImageRequest` to compare against
    fn eq(&self, request: &ImageRequest) -> bool {
        // make sure all fields are the same
        same!(self.name, request.name);
        same!(self.group, request.group);
        same!(&self.version, &request.version);
        same!(self.scaler, request.scaler);
        same!(self.image, request.image);
        same!(&self.lifetime, &request.lifetime);
        same!(self.timeout, request.timeout);
        same!(self.resources, request.resources);
        same!(self.spawn_limit, request.spawn_limit);
        same!(self.env, request.env);
        matches_vec!(&self.volumes, &request.volumes);
        same!(self.description, request.description);
        matches_update!(self.security_context, request.security_context);
        same!(self.collect_logs, request.collect_logs);
        same!(self.generator, request.generator);
        same!(self.dependencies, request.dependencies);
        same!(self.display_type, request.display_type);
        same!(self.output_collection, request.output_collection);
        same!(self.child_filters, request.child_filters);
        same!(self.network_policies, request.network_policies);
        true
    }
}

impl PartialEq<ImageUpdate> for Image {
    /// Check if a [`Image`] contains all the updates from a [`ImageUpdate`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `ImageUpdate` to compare against
    #[rustfmt::skip]
    fn eq(&self, update: &ImageUpdate) -> bool {
        // make sure any updates were propagated
        matches_update_opt!(self.image, update.image);
        matches_clear_opt!(self.lifetime, update.lifetime, update.clear_lifetime);
        matches_update!(self.scaler, update.scaler);
        matches_update_opt!(self.timeout, update.timeout);
        matches_update!(self.resources, update.resources);
        matches_update!(self.spawn_limit, update.spawn_limit);
        matches_clear_opt!(self.image, update.image, update.clear_image);
        matches_clear_opt!(self.version, update.version, update.clear_version);
        matches_adds!(self.volumes, update.add_volumes);
        matches_clear_opt!(self.description, update.description, update.clear_description);
        // build list of volume names
        let volume_names: Vec<String> = self.volumes.iter().map(|vol| vol.name.clone()).collect();
        // make sure we have removed any volumes requested for removal
        matches_removes!(volume_names, update.remove_volumes);
        // make sure the security context was correctly updated
        matches_update!(self.security_context, update.security_context);
        matches_update!(self.collect_logs, update.collect_logs);
        matches_update!(self.generator, update.generator);
        // make sure any dependency settings were updated
        same!(self.dependencies, update.dependencies);
        // make sure display type is updated
        matches_update!(self.display_type, update.display_type);
        matches_update!(self.output_collection, update.output_collection);
        matches_update!(self.child_filters, update.child_filters);
        // filter out any bans from the adds list that would have been
        // removed by the removes list
        let mut bans_added = update.bans.bans_added.iter().filter_map(|ban| {
            if update.bans.bans_removed.contains(&ban.id) {
                None
            } else {
                Some((&ban.id, ban))
            }
        });
        matches_adds_map!(self.bans, bans_added);
        matches_removes_map!(self.bans, update.bans.bans_removed);
        true
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        use crate::models::backends::NotificationSupport;
        use crate::models::{ImageKey, KeySupport, NotificationType};

        impl NotificationSupport for Image {
            /// Provide the image notification type
            fn notification_type() -> NotificationType {
                NotificationType::Images
            }
        }

        impl KeySupport for Image {
            /// The image's unique key to access its data in scylla
            type Key = ImageKey;

            /// Images have no extra optional components for their keys
            type ExtraKey = ();

            /// Build the key for this image if we need the key as one field
            ///
            /// # Arguments
            ///
            /// * `key` - The key to build from
            fn build_key(key: Self::Key, _extra: &Self::ExtraKey) -> String {
                serde_json::to_string(&key).expect("Failed to serialize image key!")
            }

            /// Build a URL component composed of the key to access the resource
            ///
            /// # Arguments
            ///
            /// * `key` - The root part of this key
            /// * `extra` - Any extra info required to build this key
            fn key_url(key: &Self::Key, _extra: Option<&Self::ExtraKey>) -> String {
                // make a URL component made up of the group and image
                format!("{}/{}", key.group, key.image)
            }
        }
    }
}

/// The needed from a image to create a raw job
pub struct ImageJobInfo {
    /// Whether this job should be a generator or not
    pub generator: bool,
    /// What scaler is responsible for scaling this image
    pub scaler: ImageScaler,
}

/// Helps serde default the image list limit to 50
fn default_list_limit() -> usize {
    50
}

/// The parameters for a image list request
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ImageListParams {
    /// The cursor id to user if one exists
    #[serde(default)]
    pub cursor: usize,
    /// The max amount of images to return in on request
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

impl Default for ImageListParams {
    /// Build the defaults for image list params
    fn default() -> Self {
        ImageListParams {
            cursor: 0,
            limit: 50,
        }
    }
}
