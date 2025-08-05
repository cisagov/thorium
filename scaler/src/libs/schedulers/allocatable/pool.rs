//! A pool of resources to schedule

use thorium::models::{Image, Pools, Resources, SystemSettings};

/// Tracks what resources are freed from what pool
#[derive(Debug, Default)]
pub struct PoolFrees {
    /// The resources freed from our fair share pool
    pub fairshare: Resources,
    /// The resources freed from our deadline pool
    pub deadline: Resources,
}

impl PoolFrees {
    /// Add resources to the correct free pool
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of pool to add resources too
    /// * `resources` - The resources to add
    pub fn add(&mut self, kind: Pools, resources: Resources) {
        match kind {
            Pools::FairShare => self.fairshare += resources,
            Pools::Deadline => self.deadline += resources,
        }
    }
}

/// A pool of resources to schedule based solely on deadlines
#[derive(Debug, Default)]
pub struct Pool {
    /// The currently available resources in this specific pool
    pub resources: Resources,
    /// The total amount of resources in this pool
    total: Resources,
}

impl Pool {
    /// Allocate some resources for a pending spawn
    ///
    /// # Arguments
    ///
    /// * `image` - The image this deadline is based on
    pub fn enough(&self, image: &Image) -> bool {
        // check if this pool has enough resources to spawn this image
        self.resources.enough(&image.resources)
    }

    /// Consume the resources for a spawned worker
    pub fn consume(&mut self, image: &Image) {
        self.resources.consume(&image.resources, 1);
    }

    /// Add resources back to our pool
    ///
    /// # Arguments
    ///
    /// * `free` - The resources to release
    pub fn release(&mut self, free: Resources) {
        self.resources += free;
    }

    /// Setup this pool for fair share based on system settings
    pub fn setup_fairshare(settings: &SystemSettings) -> Self {
        // build the total amount of resources in this pool
        let total = Resources::new(
            settings.fairshare_cpu,
            settings.fairshare_memory,
            settings.reserved_storage,
            100,
        );
        // build our pool
        Pool {
            resources: total.clone(),
            total,
        }
    }
}
