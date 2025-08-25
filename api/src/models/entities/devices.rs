//! Contains models for device entities

use std::collections::BTreeSet;
use uuid::Uuid;

use crate::models::{CriticalSector, Entity, VendorEntity};

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use crate::{bad, internal_err};
        use crate::utils::{ApiError, Shared};
        use crate::models::backends::db;
        use crate::models::EntityMetadataForm;
    }
}

#[cfg(feature = "client")]
use crate::{multipart_list, multipart_list_conv, multipart_text, multipart_text_to_string};

/// A device entity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DeviceEntity {
    /// Any URL's associated with this device
    #[serde(default)]
    pub urls: Vec<String>,
    /// The vendor entity associated with this device
    pub vendors: Vec<Entity>,
    pub critical_system: Option<bool>,
    pub sensitive_location: Option<bool>,
    /// The critical sectors this device is in or associated with
    #[serde(default)]
    pub critical_sectors: BTreeSet<CriticalSector>,
}

impl DeviceEntity {
    /// Create a new device entity with the info in the form
    ///
    /// # Errors
    ///
    /// * The vendor in the form does not exist in the given groups
    /// * An error occurred checking if the vendor in the form exists
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    /// * `groups` - The groups the entity is in (used to check for vendor)
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    pub async fn from_form(
        form: EntityMetadataForm,
        groups: &[String],
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        //// get all of the entities that were specified as vendors
        //let entities = db::entities::get_many(groups, &form.vendors, shared).await?;
        //// validate that all of these entities are vendors
        //if let Some(bad_id) = entities
        //    .iter()
        //    .find(|entity| entity.kind != crate::models::EntityKinds::Vendor)
        //{
        //    // we found an entity that is not a vendor so reject this request
        //    return bad!(format!("Entity {} is not a vendor!", bad.id));
        //}
        // build our device entity
        Ok(DeviceEntity {
            urls: form.urls,
            vendors: vec![],
            critical_system: form.critical_system,
            sensitive_location: form.sensitive_location,
            critical_sectors: form.critical_sectors.into_iter().collect(),
        })
    }
}

/// A device entity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct DeviceEntityRequest {
    /// Any URL's associated with this device
    #[serde(default)]
    pub urls: Vec<String>,
    /// The vendor entity associated with this device
    pub vendors: Vec<Uuid>,
    /// Whether this device is a critical system
    pub critical_system: Option<bool>,
    /// Whether this device is in a sensitive location
    pub sensitive_location: Option<bool>,
    /// The critical sectors this device is in or associated with
    #[serde(default)]
    pub critical_sectors: BTreeSet<CriticalSector>,
}

impl DeviceEntityRequest {
    /// Add this device entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        mut self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form.text("kind", super::EntityKinds::Device.as_str());
        // add our device metadata
        let form = multipart_list!(form, "metadata[urls][]", self.urls);
        // add our vendor ids
        let form = multipart_list_conv!(form, "metadata[vendors][]", self.vendors);
        // add whether this device is in a critical system or not
        let form =
            multipart_text_to_string!(form, "metadata[critical_system]", self.critical_system);
        // add whether this device is in a sensitive location or not
        let mut form = multipart_text_to_string!(
            form,
            "metadata[sensitive_location]",
            self.sensitive_location
        );
        // add the critical sectors this device is a part of
        for sector in self.critical_sectors {
            form = form.text("metadata[critical_sectors][]", sector.to_string());
        }
        Ok(form)
    }
}
