//! The possible states of a [`super::KvmWorker`]

use std::fmt::Debug;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use tracing::{Level, event, instrument};
use virt::domain::Domain;

use crate::libs::Error;
use crate::libs::launchers::kvm::os_detector::GuestOs;
use crate::libs::launchers::kvm::virt_async::LibvirtClient;

// modules for state-specific logic
#[rustfmt::skip]
mod init;
#[rustfmt::skip]
mod overlayed;
#[rustfmt::skip]
mod defined;
#[rustfmt::skip]
mod created;
#[rustfmt::skip]
mod launched;

pub use launched::AgentState;

/// A state a [`super::KvmWorker`] can be in, including any data associated with the
/// state
pub trait KvmWorkerState: Debug + Clone {
    /// The next state
    type NextState: KvmWorkerState;

    /// The previous state
    type PreviousState: KvmWorkerState;

    /// The new data to include in the next state
    type NextStateData;

    /// Proceed to the next state
    ///
    /// # Arguments
    ///
    /// * `next_state_data` - Any data relevant to the next state
    fn next_state(self, next_state_data: Self::NextStateData) -> Self::NextState;

    /// Revert to the previous state
    fn previous_state(self) -> Self::PreviousState;

    /// Destroy this worker and all of its associated data
    ///
    /// # Arguments
    ///
    /// * `client` - A libvirt client to use to destroy things
    fn destroy(
        self,
        client: &LibvirtClient,
    ) -> impl std::future::Future<Output = Result<(), Error>> + Send;

    /// Get the Thorium worker this KVM worker is associated with
    fn get_worker(&self) -> &str;
}

/// This worker has only been initiated and assigned a Thorium worker
#[derive(Debug, Clone)]
pub struct Init {
    /// The name of the worker
    pub worker: String,
}

/// This worker has an overlay disk image created and assigned
#[derive(Debug, Clone)]
pub struct Overlayed {
    /// The name of the worker
    pub worker: String,
    /// The detected OS of the guest
    pub guest_os: GuestOs,
    /// The path to the worker's overlay file
    pub overlay_path: PathBuf,
}

/// This worker has a defined domain and is ready to be launched
#[derive(Debug, Clone)]
pub struct Defined {
    /// The name of the worker
    pub worker: String,
    /// The detected OS of the guest
    pub guest_os: GuestOs,
    /// The path to the worker's overlay file
    pub overlay_path: PathBuf,
    /// The name of the worker's domain
    pub domain_name: String,
}

/// This worker is created in that its domain is defined *and* running
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct Created {
    /// The name of the worker
    pub worker: String,
    /// The detected OS of the guest
    pub guest_os: GuestOs,
    /// The path to the worker's overlay file
    pub overlay_path: PathBuf,
    /// The name of the worker's domain
    pub domain_name: String,
    /// The time this domain was created
    pub created: DateTime<Utc>,
    /// The time this domain attempted a launch
    pub last_launch_attempt: Option<DateTime<Utc>>,
    /// The number of times we've attempted to launch this domain
    pub launch_attempts: u8,
}

/// This worker has been launched and the Thorium agent was started,
/// though it may have crashed or exited
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct Launched {
    /// The name of the worker
    pub worker: String,
    /// The detected OS of the guest
    pub guest_os: GuestOs,
    /// The path to the worker's overlay file
    pub overlay_path: PathBuf,
    /// The name of the worker's domain
    pub domain_name: String,
    /// The PID of the Thorium agent running in the domain
    pub agent_pid: Option<u64>,
    /// The time this worker was launched
    pub launched: DateTime<Utc>,
    /// The time we should check the status of this worker next
    pub next_check: DateTime<Utc>,
    /// The last time we've heard from this worker's Thorium agent
    pub heartbeat: DateTime<Utc>,
}

impl KvmWorkerState for () {
    type NextState = ();

    type PreviousState = ();

    type NextStateData = ();

    async fn destroy(self, _client: &LibvirtClient) -> Result<(), Error> {
        Ok(())
    }

    fn next_state(self, _next_state_data: Self::NextStateData) -> Self::NextState {}

    fn previous_state(self) -> Self::PreviousState {}

    fn get_worker(&self) -> &'static str {
        // provide an empty placeholder since this is an empty state
        ""
    }
}

impl KvmWorkerState for Init {
    type NextState = Overlayed;

    type PreviousState = ();

    /// The data for the next state is the detected guest os and the path to the overlay
    type NextStateData = (GuestOs, PathBuf);

    fn next_state(self, next_state_data: Self::NextStateData) -> Self::NextState {
        let (guest_os, overlay_path) = next_state_data;
        Overlayed {
            worker: self.worker,
            guest_os,
            overlay_path,
        }
    }

    fn previous_state(self) -> Self::PreviousState {}

    async fn destroy(self, _client: &LibvirtClient) -> Result<(), Error> {
        // there's nothing left to destroy
        Ok(())
    }

    fn get_worker(&self) -> &str {
        &self.worker
    }
}

impl KvmWorkerState for Overlayed {
    type NextState = Defined;

    type PreviousState = Init;

    /// The data for the next state is the name of the its defined domain
    type NextStateData = String;

    fn next_state(self, next_state_data: Self::NextStateData) -> Self::NextState {
        Defined {
            worker: self.worker,
            guest_os: self.guest_os,
            overlay_path: self.overlay_path,
            domain_name: next_state_data,
        }
    }

    fn previous_state(self) -> Self::PreviousState {
        Init {
            worker: self.worker,
        }
    }

    #[instrument(name = "Overlayed::destroy", skip_all)]
    async fn destroy(self, client: &LibvirtClient) -> Result<(), Error> {
        // delete the overlay image for this worker
        if let Err(err) = tokio::fs::remove_file(&self.overlay_path).await {
            match err.kind() {
                // ignore "not found" errors
                std::io::ErrorKind::NotFound => (),
                // propagate any other error
                _ => {
                    // log any errors, but don't error out so we don't leave data dangling
                    event!(
                        Level::ERROR,
                        msg = "Error deleting worker overlay file",
                        overlay = self.overlay_path.to_string_lossy().to_string(),
                        worker = self.worker,
                        err = err.to_string()
                    );
                }
            }
        }
        self.previous_state().destroy(client).await
    }

    fn get_worker(&self) -> &str {
        &self.worker
    }
}

impl KvmWorkerState for Defined {
    type NextState = Created;

    type PreviousState = Overlayed;

    // No new data is needed for the next state
    type NextStateData = ();

    fn next_state(self, _next_state_data: Self::NextStateData) -> Self::NextState {
        Created {
            worker: self.worker,
            guest_os: self.guest_os,
            overlay_path: self.overlay_path,
            domain_name: self.domain_name,
            created: Utc::now(),
            last_launch_attempt: None,
            launch_attempts: 0,
        }
    }

    fn previous_state(self) -> Self::PreviousState {
        Overlayed {
            worker: self.worker,
            guest_os: self.guest_os,
            overlay_path: self.overlay_path,
        }
    }

    #[instrument(name = "Defined::destroy", skip_all)]
    async fn destroy(self, client: &LibvirtClient) -> Result<(), Error> {
        let domain_name = self.domain_name.clone();
        if let Err(err) = {
            match client
                .with_conn(move |conn| Domain::lookup_by_name(conn, &domain_name))
                .await
            {
                Ok(domain) => {
                    // this worker is just defined and should not be active,
                    // but we'll make doubly sure to avoid errors trying to undefine an active domain
                    client
                        .with_domain(&domain, move |d| {
                            let name = d.get_name()?;
                            if d.is_active()? {
                                // destroy the domain first if it's active
                                d.destroy()?;
                            }
                            // undefine the domain and everything related to it
                            let undefine_flags = virt::sys::VIR_DOMAIN_UNDEFINE_CHECKPOINTS_METADATA
                                | virt::sys::VIR_DOMAIN_UNDEFINE_NVRAM
                                | virt::sys::VIR_DOMAIN_UNDEFINE_SNAPSHOTS_METADATA
                                | virt::sys::VIR_DOMAIN_UNDEFINE_MANAGED_SAVE;
                            d.undefine_flags(undefine_flags)?;
                            event!(Level::DEBUG, "Undefine -> {name}");
                            Ok::<_, virt::error::Error>(())
                        })
                        .await?;
                    Ok(())
                }
                Err(err) => {
                    // if the error is just that the domain doesn't exist, no shutdown is needed;
                    // otherwise propagate this error
                    if err.code() == virt::error::ErrorNumber::NoDomain {
                        Ok(())
                    } else {
                        Err(Error::from(err))
                    }
                }
            }
        } {
            // log any errors, but don't error out so we don't leave data dangling
            event!(
                Level::ERROR,
                msg = "Error undefining KVM worker",
                worker = self.worker,
                err = err.to_string()
            );
        }
        // revert to overlayed state and continue destruction
        self.previous_state().destroy(client).await
    }

    fn get_worker(&self) -> &str {
        &self.worker
    }
}

impl KvmWorkerState for Created {
    type NextState = Launched;

    type PreviousState = Defined;

    // The data for the next state is the PID of the Thorium agent
    type NextStateData = Option<u64>;

    fn next_state(self, next_state_data: Self::NextStateData) -> Self::NextState {
        let now = Utc::now();
        Launched {
            worker: self.worker,
            guest_os: self.guest_os,
            overlay_path: self.overlay_path,
            domain_name: self.domain_name,
            agent_pid: next_state_data,
            launched: now,
            next_check: Launched::next_health_check(),
            heartbeat: now,
        }
    }

    fn previous_state(self) -> Self::PreviousState {
        Defined {
            worker: self.worker,
            guest_os: self.guest_os,
            overlay_path: self.overlay_path,
            domain_name: self.domain_name,
        }
    }

    #[instrument(name = "Created::destroy", skip_all)]
    async fn destroy(self, client: &LibvirtClient) -> Result<(), Error> {
        // there's nothing specific to the 'Created' that needs to be handled;
        // revert to defined state and continue destruction
        self.previous_state().destroy(client).await
    }

    fn get_worker(&self) -> &str {
        &self.worker
    }
}

impl KvmWorkerState for Launched {
    // There is no state after launched
    type NextState = ();

    type PreviousState = Created;

    type NextStateData = ();

    fn next_state(self, _next_state_data: Self::NextStateData) -> Self::NextState {}

    fn previous_state(self) -> Self::PreviousState {
        Created {
            worker: self.worker,
            guest_os: self.guest_os,
            overlay_path: self.overlay_path,
            domain_name: self.domain_name,
            created: Utc::now(),
            last_launch_attempt: None,
            launch_attempts: 0,
        }
    }

    #[instrument(name = "Launched::destroy", skip_all)]
    async fn destroy(self, client: &LibvirtClient) -> Result<(), Error> {
        // there's nothing specific to the 'Launched' that needs to be handled;
        // revert to Created state and continue destruction
        self.previous_state().destroy(client).await
    }

    fn get_worker(&self) -> &str {
        &self.worker
    }
}
