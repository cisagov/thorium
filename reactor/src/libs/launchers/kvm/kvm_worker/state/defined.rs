//! The logic for a `KvmWorker` in the [`Defined`] state

use tracing::instrument;
use virt::domain::Domain;

use super::{Created, Defined, KvmWorkerState};
use crate::libs::{Error, launchers::kvm::virt_async::LibvirtClient};

impl Defined {
    /// Create the worker, spawning its domain in libvirt
    ///
    /// # Arguments
    ///
    /// * `client` - A libvirt client
    /// * `domain` - A handle to this worker's domain to create
    #[instrument(skip_all, fields(worker = self.worker))]
    pub async fn create(self, client: &LibvirtClient, domain: &Domain) -> Result<Created, Error> {
        // try to create the domain
        client
            .with_domain(domain, virt::domain::Domain::create)
            .await?;
        Ok(self.next_state(()))
    }

    /// Proceed to the next state as if we actually created the worker
    pub fn mock_create(self) -> Created {
        self.next_state(())
    }
}
