use uuid::Uuid;

use crate::models::backends::db;
use crate::models::{DeviceEntity, EntityMetadataUpdateForm};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, internal_err, update_add_rem_ordered_owned, update_add_rem_set_owned, update_clear_opt,
    update_opt,
};

impl DeviceEntity {
    /// Update the device entity with the info in the form
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    /// * `id` - The device's entity id
    /// * `groups` - The groups the entity is in (used to check for vendor)
    /// * `shared` - Shared Thorium objects
    pub async fn update(
        &mut self,
        mut form: EntityMetadataUpdateForm,
        id: Uuid,
        groups: &[String],
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // update urls
        update_add_rem_ordered_owned!(
            self.urls,
            form.add_urls,
            form.remove_urls,
            "Device",
            "URL(s)"
        )?;
        // update fields
        update_opt!(self.critical_system, form.critical_system);
        update_opt!(self.sensitive_location, form.sensitive_location);
        // clear fields
        // TODO update vendors field
        //update_clear_opt!(self.vendor, form.clear_vendor);
        update_clear_opt!(self.critical_system, form.clear_critical_system);
        update_clear_opt!(self.sensitive_location, form.clear_sensitive_location);
        // update critical sectors
        update_add_rem_set_owned!(
            self.critical_sectors,
            form.add_critical_sectors,
            form.remove_critical_sectors,
            "Device",
            "critical sector(s)"
        )?;
        //// first make sure the new vendor exists if one is given
        //if let Some(new_vendor) = form.new_vendor {
        //    match db::entities::exists_groups(groups, new_vendor, shared).await {
        //        Ok(groups) => {
        //            if groups.is_empty() {
        //                return bad!(format!("Vendor with id '{new_vendor}' not found"));
        //            }
        //        }
        //        Err(err) => {
        //            return internal_err!(format!(
        //                "Error checking if vendor with id '{new_vendor}' exists: {err}",
        //            ));
        //        }
        //    }
        //}
        //// update the vendor
        //match (
        //    &self.vendor,
        //    form.clear_vendor.is_some_and(|clear| clear),
        //    form.new_vendor,
        //) {
        //    // adding a new vendor with no previous one
        //    (None, false, Some(new_vendor)) => {
        //        // add the device to the vendor's set
        //        db::entities::vendors::add_device(new_vendor, id, shared).await?;
        //        // set the vendor
        //        self.vendor = Some(new_vendor);
        //    }
        //    // we're clearing a previous vendor; ignore the new one because we're set to clear
        //    (Some(old_vendor), true, _) => {
        //        // remove the device from the vendor's set
        //        db::entities::vendors::remove_device(*old_vendor, id, shared).await?;
        //    }
        //    // replace a vendor with a new one
        //    (Some(old_vendor), false, Some(new_vendor)) => {
        //        // remove the device from the old vendor
        //        db::entities::vendors::remove_device(*old_vendor, id, shared).await?;
        //        // add it to the new one
        //        db::entities::vendors::add_device(new_vendor, id, shared).await?;
        //        // set the vendor
        //        self.vendor = Some(new_vendor);
        //    }
        //    // everything else is a noop
        //    _ => (),
        //}
        Ok(())
    }
}
