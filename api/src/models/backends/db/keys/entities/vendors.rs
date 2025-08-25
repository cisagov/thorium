//! Keys for accessing vendor data

use uuid::Uuid;

use crate::utils::Shared;

/// Keys to use to access vendor data
pub struct VendorKeys {
    /// The key to store devices associated with this vendor in
    pub devices: String,
}

impl VendorKeys {
    /// Builds the keys to store devices in redis
    ///
    /// # Arguments
    ///
    /// * `id` - The vendor's id
    /// * `shared` - Shared Thorium objects
    pub fn new(id: Uuid, shared: &Shared) -> Self {
        // build key to store devices at
        let devices = Self::devices(id, shared);
        // build key object
        Self { devices }
    }

    /// Builds the devices key for a vendor
    ///
    /// # Arguments
    ///
    /// * `id` - The vendor's id
    /// * `shared` - Shared Thorium objects
    pub fn devices(id: Uuid, shared: &Shared) -> String {
        format!(
            "{ns}:entities:vendors:{id}:devices",
            ns = shared.config.thorium.namespace
        )
    }
}
