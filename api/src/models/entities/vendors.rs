//! Contains models for vendor entities

use std::collections::BTreeSet;

use crate::models::{Country, CriticalSector};

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use crate::models::EntityMetadataForm;
        use crate::{multipart_list, multipart_set};
    }
}

/// A vendor for software, hardware, or services
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct VendorEntity {
    /// The country this vendor operates in or is headquatered in
    pub countries: BTreeSet<Country>,
    /// The critical sectors this device is in or associated with
    #[serde(default)]
    pub critical_sectors: BTreeSet<CriticalSector>,
}

impl VendorEntity {
    /// Create a vendor entity from the info in a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to take the info from
    #[cfg(feature = "api")]
    #[must_use]
    pub fn from_form(form: EntityMetadataForm) -> Self {
        Self {
            countries: form.countries,
            critical_sectors: form.critical_sectors,
        }
    }
}

/// A request for creating a new vendor entity in Thorium
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct VendorEntityRequest {
    /// The country this vendor operates in or is headquatered in
    pub countries: BTreeSet<String>,
    /// The critical sectors this device is in or associated with
    #[serde(default)]
    pub critical_sectors: BTreeSet<CriticalSector>,
}

impl VendorEntityRequest {
    /// Add this device entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let mut form = form.text("kind", super::EntityKinds::Vendor.as_str());
        // add our device metadata
        for country in self.countries {
            form = form.text("metadata[countries][]", country);
        }
        // add the critical sectors this device is a part of
        for sector in self.critical_sectors {
            form = form.text("metadata[critical_sectors][]", sector.to_string());
        }
        Ok(form)
    }
}
