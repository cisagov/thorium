//! The `KvmWorker` state machine, describing the lifecycle of a worker in KVM
//!
//! Logic for individual states can be found within the state's module
//! (e.g. logic for [`KvmWorker`] in the state [`state::Init`] can be found in [`state::init`])

use std::fmt::Debug;

use thorium::models::{Image, Worker};
use tracing::instrument;
use virt::domain::Domain;

use super::{LaunchError, LibvirtClient};
use crate::libs::launchers::kvm::{AgentPaths, OsDetector};
use crate::{Error, args::Kvm};

mod state;

pub use state::{AgentState, Created, Launched};
use state::{Defined, Init, KvmWorkerState, Overlayed};

/// A KVM worker associated with a single Domain and Thorium worker running in KVM
#[derive(Debug, Clone)]
pub struct KvmWorker<T: KvmWorkerState> {
    /// The state of this worker (and any data associated to that state)
    state: T,
}

impl<T: KvmWorkerState> KvmWorker<T> {
    /// Proceed to the next state
    ///
    /// # Arguments
    ///
    /// * `next_state` - The next state and its data to proceed to
    fn next_state(next_state: T::NextState) -> KvmWorker<T::NextState> {
        KvmWorker { state: next_state }
    }

    /// Destroy this worker and all of its related data
    ///
    /// The destruction process is performed at the current state and cascades back the
    /// state machine until the first [`Init`] state
    ///
    /// # Arguments
    ///
    /// * `client` - A libvirt client
    #[instrument(name = "KvmWorker::destroy", skip_all, fields(worker = self.state.get_worker()), err(Debug))]
    pub async fn destroy(self, client: &LibvirtClient) -> Result<(), Error> {
        self.state.destroy(client).await
    }
}

impl KvmWorker<Init> {
    /// Initiate a new [`KvmWorker`], assigning it a Thorium worker
    ///
    /// # Arguments
    ///
    /// * `worker` - The name of the Thorium [`Worker`] this [`KvmWorker`] is responsible for
    /// * `os_detector` - A detector that attempts to determine the OS of guest VMs
    pub fn init(worker: String) -> Self {
        Self {
            state: Init { worker },
        }
    }

    /// Create a qcow2 overlay for this worker based on a golden image, saving on disk space and
    /// I/O by leveraging copy-on-write
    ///
    /// # Arguments
    ///
    /// * `args` - The KVM-specific args passed to the reactor
    /// * `image` - The image configuration for this worker
    /// * `os_detector` - A detector that attempts to determine the OS of the guest VM
    #[instrument(name = "KvmWorker<Init>::overlay", skip_all, fields(worker = self.state.worker), err(Debug))]
    pub async fn overlay(
        self,
        args: &Kvm,
        image: &Image,
        os_detector: &OsDetector,
    ) -> Result<KvmWorker<Overlayed>, Error> {
        let overlayed = self.state.overlay(args, image, os_detector).await?;
        Ok(Self::next_state(overlayed))
    }

    /// Proceed to the next state as if we actually overlayed the worker
    ///
    /// # Arguments
    ///
    /// * `args` - The KVM-specific args passed to the reactor
    pub fn mock_overlay(self, args: &Kvm) -> KvmWorker<Overlayed> {
        // presume the overlay path is defined correctly
        let mock_overlayed = self.state.mock_overlay(args);
        Self::next_state(mock_overlayed)
    }
}

impl KvmWorker<Overlayed> {
    /// Define the worker, defining its domain and returning a handle to it
    ///
    /// # Arguments
    ///
    /// * `args` - The KVM-specific args passed to the reactor
    /// * `client` - A libvirt client
    /// * `worker` - The worker to define a domain for
    ///
    /// # Resource Allocation
    /// - CPU: Always rounds UP from mCPU to vCPU (minimum 1)
    /// - Memory: Uses MiB directly without conversion
    #[instrument(name = "KvmWorker<Overlayed>::define", skip_all, fields(worker = self.state.worker), err(Debug))]
    pub async fn define(
        self,
        client: &LibvirtClient,
        args: &Kvm,
        worker: &Worker,
        image: &Image,
    ) -> Result<(KvmWorker<Defined>, Domain), Error> {
        let (defined, domain) = self.state.define(client, args, worker, image).await?;
        Ok((Self::next_state(defined), domain))
    }

    /// Proceed to the next state as if we actually defined the worker
    pub fn mock_define(self) -> KvmWorker<Defined> {
        // presume the domain name was defined correctly
        let mock_defined = self.state.mock_define();
        Self::next_state(mock_defined)
    }
}

impl KvmWorker<Defined> {
    /// Create the worker, spawning its domain in libvirt
    ///
    /// # Arguments
    ///
    /// * `client` - A libvirt client
    /// * `domain` - A handle to this worker's domain to create
    #[instrument(name = "KvmWorker<Defined>::create", skip_all, fields(worker = self.state.worker), err(Debug))]
    pub async fn create(
        self,
        client: &LibvirtClient,
        domain: &Domain,
    ) -> Result<KvmWorker<Created>, Error> {
        // try to create the domain
        let created = self.state.create(client, domain).await?;
        Ok(Self::next_state(created))
    }

    /// Proceed to the next state as if we actually created the worker
    pub fn mock_create(self) -> KvmWorker<Created> {
        let mock_created = self.state.mock_create();
        Self::next_state(mock_created)
    }
}

impl KvmWorker<Created> {
    /// Launch the worker by launching the Thorium agent on the target VM using QEMU guest agent
    ///
    /// On error, the worker in its created state is returned with its launch attempts incremented,
    /// along with the error that occurred so we can maybe try to launch it again next time
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - The domain to launch the agent in
    /// * `worker` - The worker we are launching an agent for
    /// * `agent_paths` - The path to the Thorium agent binaries on disk
    #[instrument(name = "KvmWorker<Created>::launch>", skip_all, fields(worker = self.state.worker), err(Debug))]
    pub async fn launch(
        self,
        client: &LibvirtClient,
        domain: &Domain,
        worker: &Worker,
        agent_paths: &AgentPaths,
    ) -> Result<KvmWorker<Launched>, LaunchError> {
        match self.state.launch(client, domain, worker, agent_paths).await {
            Ok(launched) => Ok(Self::next_state(launched)),
            Err((failed_state, err)) => Err(LaunchError(
                Self {
                    state: failed_state,
                },
                err,
            )),
        }
    }

    /// Returns the domain name for this worker
    pub fn domain_name(&self) -> &str {
        &self.state.domain_name
    }

    /// Proceed to the next state as if we actually launched the worker
    pub fn mock_launch(self) -> KvmWorker<Launched> {
        let mock_launched = self.state.mock_launch();
        Self::next_state(mock_launched)
    }

    /// Returns whether this worker has attempted to launch the maximum
    /// number of times (or more)
    pub fn has_reached_max_attempts(&self) -> bool {
        self.state.has_reached_max_attempts()
    }

    /// Returns true if the domain is ready to attempt launch
    pub fn ready_to_launch(&self) -> bool {
        self.state.ready_to_launch()
    }
}

impl KvmWorker<Launched> {
    /// Get the state of the Thorium agent in this worker
    ///
    /// Update the heartbeat of the worker if we see the agent running
    ///
    /// # Arguments
    ///
    /// * `client` - A libvirt client
    #[instrument(name = "KvmWorker<Launched>::agent_state", skip_all, err(Debug))]
    pub async fn agent_state(&mut self, client: &LibvirtClient) -> Result<AgentState, Error> {
        self.state.agent_state(client).await
    }

    /// Returns whether this worker is due for a health check
    pub fn is_health_check_due(&self) -> bool {
        self.state.is_health_check_due()
    }

    /// Schedule this worker for its next health check
    pub fn schedule_next_health_check(&mut self) {
        self.state.schedule_next_health_check();
    }

    /// Returns whether this worker has timed out completely
    pub fn timed_out(&self) -> bool {
        self.state.timed_out()
    }

    /// Returns whether this unresponsive worker (missing agent) has timed out
    pub fn timed_out_unresponsive(&self) -> bool {
        self.state.timed_out_unresponsive()
    }
}

/// A [`KvmWorker`] that is running (active)
pub enum RunningKvmWorker {
    /// A worker that is active, but the Thorium agent hasn't been launched
    Created(KvmWorker<Created>),
    /// A worker that is launched and the Thorium agent should be running
    Launched(KvmWorker<Launched>),
}

impl RunningKvmWorker {
    /// Destroy this worker
    pub async fn destroy(self, client: &LibvirtClient) -> Result<(), Error> {
        match self {
            RunningKvmWorker::Created(kvm_worker) => kvm_worker.destroy(client).await,
            RunningKvmWorker::Launched(kvm_worker) => kvm_worker.destroy(client).await,
        }
    }
}
