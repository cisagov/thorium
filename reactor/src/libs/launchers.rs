//! Handles the launching of jobs for specific reactor types
//!
//! Currently only windows is supported;

use std::collections::{HashMap, HashSet};

use thorium::Thorium;
use thorium::models::{Node, Worker};

use crate::Error;

#[cfg(target_os = "linux")]
mod bare_metal;
//#[cfg(feature = "kvm")]
#[cfg(target_os = "linux")]
#[cfg(feature = "kvm")]
pub mod kvm;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
pub use bare_metal::BareMetal;
#[cfg(target_os = "windows")]
pub use windows::Windows;

#[async_trait::async_trait]
pub trait Launcher: Send + Sync + 'static {
    /// The data returned for an individual launched worker
    type LaunchedWorkerData: Send;

    /// Launch a worker in a new task, returning the worker that was launched and
    /// any data to resolve in the launcher's state afterward
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    async fn launch(
        &self,
        thorium: &Thorium,
        worker: Worker,
    ) -> Result<(Worker, Self::LaunchedWorkerData), Error>;

    /// Resolve all of the data from this batch of launches, modifying any relevant data
    /// in the launcher's state
    ///
    /// # Arguments
    ///
    /// * `launches` - The batch of workers and their data to resolve
    fn resolve_launches(&mut self, launches: Vec<(&Worker, Self::LaunchedWorkerData)>);

    /// Reconcile the current worker state with what is reported in the API,
    /// handling worker failures, reclaiming workers, etc.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    async fn reconcile(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error>;

    /// Shutdown a list of workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The workers to shutdown
    async fn shutdown(
        &mut self,
        thorium: &Thorium,
        mut workers: HashSet<String>,
    ) -> Result<(), Error>;
}
