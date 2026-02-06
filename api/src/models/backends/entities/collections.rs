//! Backend logic for collection entities

use tracing::instrument;

use crate::models::{CollectionEntity, EntityMetadataForm, EntityMetadataUpdateForm};
use crate::utils::ApiError;
use crate::{bad, bad_internal, update, update_clear_opt, update_opt};

impl CollectionEntity {
    /// Update a collection entity based on the data in the form
    ///
    /// May remove or modify any metadata related to collections in the
    /// form
    ///
    /// # Arguments
    ///
    /// * `form` - The form whose data to use to update
    #[instrument(name = "CollectionEntity::update", skip_all, err(Debug))]
    pub fn update(&mut self, form: &mut EntityMetadataUpdateForm) -> Result<(), ApiError> {
        // add requested collection tags
        for (key, add_values) in form.add_collection_tags.drain() {
            self.collection_tags
                .entry(key)
                .or_default()
                .extend(add_values.into_iter());
        }
        // delete requested collection tags
        for (key, delete_values) in form.delete_collection_tags.drain() {
            if let Some(values) = self.collection_tags.get_mut(&key) {
                values.retain(|v| !delete_values.contains(v));
                if values.is_empty() {
                    // all values have been removed; delete the entire entry
                    self.collection_tags.remove(&key);
                }
            }
        }
        update!(
            self.tags_case_insensitive,
            form.collection_tags_case_insensitive
        );
        update!(self.ignore_groups, form.collection_ignore_groups);
        // ensure end is older than start;
        // prioritize comparing the update values, otherwise use existing ones
        if let (Some(start), Some(end)) = (
            form.collection_start.as_ref().or(self.start.as_ref()),
            form.collection_end.as_ref().or(self.end.as_ref()),
        ) {
            // make sure that the update form start/end were actually set;
            // ensures that users can update and fix the values if there were bad ones somehow
            if (form.collection_start.is_some() || form.collection_end.is_some()) && start < end {
                return bad!(format!(
                    "Start must be more recent than end: Start '{start}' < End '{end}'"
                ));
            }
        }
        update_opt!(self.start, form.collection_start);
        update_opt!(self.end, form.collection_end);
        update_clear_opt!(self.start, form.clear_collection_start);
        update_clear_opt!(self.end, form.clear_collection_end);
        Ok(())
    }

    /// Create a collection entity from the info in a form
    ///
    /// # Errors
    ///
    /// * The collection kind was not set in the form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to take the info from
    #[instrument(name = "CollectionEntity::from_form", skip_all, err(Debug))]
    pub fn from_form(form: EntityMetadataForm) -> Result<Self, ApiError> {
        let collection_kind = form.collection_kind.ok_or(bad_internal!(
            "'collection_kind' must be explicitly set to create a collection".to_string()
        ))?;
        Ok(Self {
            collection_kind,
            collection_tags: form.collection_tags,
            tags_case_insensitive: form.collection_tags_case_insensitive.unwrap_or_default(),
            ignore_groups: form.collection_ignore_groups.unwrap_or_default(),
            start: form.collection_start,
            end: form.collection_end,
        })
    }
}
