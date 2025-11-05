use thorium::models::{Image, Pools, ScrubbedUser, WorkerDeleteMap, WorkerStatus, WorkerUpdate};
use thorium::{Error, Thorium};
use tracing::instrument;

use crate::args::Args;

/// The Target stage to claim and run a job for
pub struct Target {
    /// The name of this worker
    pub name: String,
    /// The group this stage is in
    pub group: String,
    /// The pipeline this stage is in
    pub pipeline: String,
    /// The name of the stage to run
    pub stage: String,
    /// Details about our image from Thorium
    pub image: Image,
    /// The user this target is for
    pub user: ScrubbedUser,
    /// The client to use for this image
    pub thorium: Thorium,
    /// What pool of resources this worker was spawned under
    pub pool: Pools,
}

impl Target {
    /// Tell Thorium this target is now running jobs
    ///
    /// # Arguments
    ///
    /// * `args` - The args used to spawn this agent
    /// * `status` - The status to set
    #[instrument(name = "Target::update_worker", skip(self))]
    pub async fn update_worker(&self, status: WorkerStatus) -> Result<(), Error> {
        // build the update for this worker
        let update = WorkerUpdate::new(status);
        // tell Thorium we are now running
        self.thorium
            .system
            .update_worker(&self.name, &update)
            .await?;
        Ok(())
    }

    /// Removes a no longer active worker
    ///
    /// # Arguments
    ///
    /// * `args` - The args used to spawn this agent
    pub async fn remove_worker(&self, args: &Args) -> Result<(), Error> {
        // get the scaler this worker is running under
        let scaler = args.scaler();
        // build the worker delete map
        let deletes = WorkerDeleteMap::default().add(&self.name);
        // tell Thorium we are now running
        self.thorium.system.delete_workers(scaler, &deletes).await?;
        Ok(())
    }
}

impl std::fmt::Display for Target {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        //write this target to the formatter
        write!(
            f,
            "{}:{}:{} - {}",
            self.group, self.pipeline, self.stage, self.name
        )
    }
}
