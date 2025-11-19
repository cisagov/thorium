//! Backend related logic for entities

use axum::extract::multipart::Field;
use axum::extract::{FromRequestParts, Multipart};
use axum::http::request::Parts;
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use scylla::errors::ExecutionError;
use scylla::response::query_result::QueryResult;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::{event, instrument};
use uuid::Uuid;

use super::db;
use crate::models::backends::GraphicSupport;
use crate::models::backends::db::{CursorCore, ScyllaCursor, ScyllaCursorSupport};
use crate::models::entities::{EntityMetadata, EntityMetadataForm};
use crate::models::{
    ApiCursor, AssociationKind, AssociationListOpts, AssociationRequest, AssociationTarget,
    AssociationTargetColumn, Country, CriticalSector, DeviceEntity, Entity, EntityForm,
    EntityKinds, EntityListLine, EntityListParams, EntityListRow, EntityMetadataUpdateForm,
    EntityResponse, EntityRow, EntityUpdateForm, Group, GroupAllowAction, ListableAssociation,
    TagListRow, TagMap, TagType, TreeSupport, User, VendorEntity,
};
use crate::utils::{ApiError, Shared};
use crate::{
    bad, deserialize, for_groups, internal_err, not_found, serialize, unauthorized, update,
    update_add_rem, update_clear_opt, update_opt,
};

mod devices;

impl Entity {
    /// A helper function for creating an entity by taking a form, validating
    /// it, and submitting it to the database
    ///
    /// Returns the ID of the created entity
    ///
    /// # Arguments
    ///
    /// * `form` - The multipart form that was submitted
    /// * `s3_path` - The s3 path to set if an image is uploaded
    /// * `user` - The user creating the entity
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Entity::create_helper", skip_all, err(Debug))]
    async fn create_helper(
        mut form: Multipart,
        s3_path: &mut Option<String>,
        user: &User,
        shared: &Shared,
    ) -> Result<Uuid, ApiError> {
        // generate a UUID for this entity
        let entity_id = Uuid::new_v4();
        // build an entity form to populate
        let mut entity_form = EntityForm::default();
        // crawl the multipart form
        while let Some(field) = form.next_field().await? {
            // try to consume the field
            if let Some(image_field) = entity_form.add(field).await? {
                // get the base path for this entity
                let base_path = Self::build_graphic_base_path(&entity_id);
                // upload the graphic to S3
                let path = Self::upload_graphic(base_path, image_field, None, shared).await?;
                // set the s3 path
                s3_path.replace(path.clone());
                // mark that this entity has an image
                entity_form.image = Some(path);
            }
        }
        // first, make sure we actually have edit access in all requested groups
        let _ = Group::authorize_check_allow_all(
            user,
            &entity_form.groups,
            Group::editable,
            "edit",
            Some(GroupAllowAction::Entities),
            shared,
        )
        .await?;
        // create the entity
        db::entities::create(user, entity_form, entity_id, shared).await?;
        // the entity was created successfully
        Ok(entity_id)
    }

    /// Create an `Entity` in the db
    ///
    /// # Arguments
    ///
    /// * `form` - The multipart form that was uploaded
    /// * `user` - The user creating the entity
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Entity::create", skip_all, err(Debug))]
    pub async fn create(
        user: &User,
        form: Multipart,
        shared: &Shared,
    ) -> Result<EntityResponse, ApiError> {
        // make a mutable option for the S3 path that's set if an image is uploaded to S3
        let mut s3_path: Option<String> = None;
        // try creating the entity
        match Self::create_helper(form, &mut s3_path, user, shared).await {
            Ok(entity_id) => {
                // entity was successfully created, so return a response
                let resp = EntityResponse::new(entity_id);
                Ok(resp)
            }
            // we got an error creating the entity, so delete the image from S3 in case
            // we uploaded it and propagate the error
            Err(err) => {
                // delete from S3 if our path was set to avoid a dangling image
                if let Some(s3_path) = &s3_path {
                    // we have the path in S3, so just delete it with the client directly
                    if let Err(s3_err) = shared.s3.graphics.delete(s3_path).await {
                        return internal_err!(format!(
                            "Error cleaning up image after entity create error: {s3_err}. Original error: {err}"
                        ));
                    }
                }
                // propogate the error
                Err(err)
            }
        }
    }

    /// Populate some entities associations
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is populating assocations for an entity
    /// * `shared` - Shared thorium objects
    async fn populate_associations(
        mut self,
        user: &User,
        shared: &Shared,
    ) -> Result<Self, ApiError> {
        // build the source for this entity
        if let Some(source) = self.build_association_target_column() {
            // build the options for listing this entities associations
            let opts = AssociationListOpts::default().groups(user.groups.clone());
            // list associations for this entity
            let mut cursor = db::associations::list(opts, &source, shared).await?;
            // based on our kind populate any association data
            match &mut self.metadata {
                EntityMetadata::Device(metadata) => {
                    // build a set of entities associated with our entity
                    let mut ids = Vec::with_capacity(3);
                    // get only the vendor associations
                    loop {
                        // filter only to developed by associations
                        for association in cursor.data.drain(..) {
                            // skip any associations that are not developed by
                            if association.kind == AssociationKind::DevelopedBy {
                                // parse our other value for this association
                                let other: AssociationTargetColumn =
                                    deserialize!(&association.other);
                                // add any entity ids we find to our list
                                if let AssociationTargetColumn::Entity(id) = other {
                                    ids.push(id);
                                }
                            }
                        }
                        // if our cursor is exhausted then stop looping
                        if cursor.exhausted() {
                            break;
                        }
                        // get the next page of associations
                        cursor.next(shared).await?;
                    }
                    // get all of the entities we found
                    metadata.vendors = db::entities::get_many(&user.groups, &ids, shared).await?;
                }
                // vendor/other has no data that we need to retrieve
                EntityMetadata::Vendor(_) | EntityMetadata::Other => (),
            }
        }
        Ok(self)
    }

    /// Get an `Entity` from the db
    ///
    /// # Arguments
    ///
    /// * `id` - The entity's id
    /// * `user` - The user getting the entity
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Entity::get", skip_all, err(Debug))]
    pub async fn get(user: &User, id: Uuid, shared: &Shared) -> Result<Entity, ApiError> {
        // for users we can search their groups but for admins we need to get all groups
        // try to get this entity if it exists
        let entity = for_groups!(db::entities::get, user, shared, id)?;
        // populate any info based on associations for this entity
        entity.populate_associations(user, shared).await
    }

    /// List entities names, ids, and kinds according to the given params
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is listing entities
    /// * `params` - The params to use when listing entities
    /// * `dedupe` - Whether to dedupe when listing entities or not
    /// * `shared` - Shared objects in Thorium
    #[instrument(name = "Entity::list", skip(user, shared), err(Debug))]
    pub async fn list(
        user: &User,
        mut params: EntityListParams,
        dedupe: bool,
        shared: &Shared,
    ) -> Result<ApiCursor<EntityListLine>, ApiError> {
        // authorize the groups to list entities from
        user.authorize_groups(&mut params.groups, shared).await?;
        // get or create a cursor over entities
        let scylla_cursor = db::entities::list(params, dedupe, shared).await?;
        // convert our scylla cursor to a user facing cursor
        Ok(ApiCursor::from(scylla_cursor))
    }

    /// Get a listable association cursor for this entity
    #[instrument(name = "Entity::list_associations", skip_all, err(Debug))]
    pub(crate) async fn list_associations(
        &self,
        shared: &Shared,
    ) -> Result<Option<ScyllaCursor<ListableAssociation>>, ApiError> {
        // get our source target column if possible
        match self.build_association_target_column() {
            Some(source) => {
                // list associations for this
                let opts = AssociationListOpts::default().groups(self.groups.clone());
                // list associations for this entity
                let cursor = db::associations::list(opts, &source, shared).await?;
                Ok(Some(cursor))
            }
            None => Ok(None),
        }
    }

    /// Update an entity's kind specific metadata with the data in the form
    ///
    /// # Arguments
    ///
    /// * `form` - The form whose data to use to update
    #[instrument(name = "Entity::update_meta", skip_all, err(Debug))]
    async fn update_meta(
        &mut self,
        user: &User,
        mut form: EntityMetadataUpdateForm,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // update the fields for our kind
        match &mut self.metadata {
            // update our device info
            EntityMetadata::Device(device) => {
                // add any new urls
                device.urls.append(&mut form.add_urls);
                // remove any requested urls
                device.urls.retain(|url| !form.remove_urls.contains(url));
                // update our critical system flag if requested
                match (form.clear_critical_system, form.critical_system) {
                    (Some(true), _) => device.critical_system = None,
                    (_, Some(critical_system)) => device.critical_system = Some(critical_system),
                    (_, None) => (),
                }
                // update our sensitive location field
                update_opt!(device.sensitive_location, form.sensitive_location);
                // clear our sensistive location if needed
                update_clear_opt!(device.sensitive_location, form.clear_sensitive_location);
                // add any new critical sectors
                device
                    .critical_sectors
                    .extend(form.add_critical_sectors.drain(..));
                // remove any old critical sectors
                device
                    .critical_sectors
                    .retain(|sector| !form.remove_critical_sectors.contains(sector));
            }
            // update our vendor info
            EntityMetadata::Vendor(vendor) => {
                // Update the countries for this vendor
                vendor.countries.extend(form.add_countries.drain(..));
                // remove any countries from this vendor
                vendor
                    .countries
                    .retain(|country| !form.remove_countries.contains(country));
                // add any new critical sectors
                vendor
                    .critical_sectors
                    .extend(form.add_critical_sectors.drain(..));
                // remove any old critical sectors
                vendor
                    .critical_sectors
                    .retain(|sector| !form.remove_critical_sectors.contains(sector));
            }
            // other has no metadata to upadte
            EntityMetadata::Other => (),
        }
        // update any associations based on this update
        form.update_associations(user, self, shared).await
    }

    /// A helper function for updating an entity by taking an update form, validating
    /// it, and submitting the changes to the database
    ///
    /// # Arguments
    ///
    /// * `form` - The multipart form that was submitted
    /// * `s3_path` - The s3 path to set if an image is uploaded
    /// * `user` - The user creating the entity
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Entity::update_helper", skip_all, err(Debug))]
    async fn update_helper(
        mut self,
        user: &User,
        mut form: Multipart,
        s3_path: &mut Option<String>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // build an entity update form to populate
        let mut update_form = EntityUpdateForm::default();
        // track any images we need to delete
        let mut deletes = vec![];
        // crawl the multipart form
        while let Some(field) = form.next_field().await? {
            // try to consume the field
            if let Some(image_field) = update_form.add(field).await? {
                // get the base path for this entity
                let base_path = Self::build_graphic_base_path_from_self(&self);
                // upload the graphic to S3
                let path = Self::upload_graphic(base_path, image_field, None, shared).await?;
                // set our new image
                if let Some(old_path) = self.image.replace(path.clone()) {
                    // add our old path to the list of images to delete
                    deletes.push(old_path);
                }
                // keep track of this path to ensure that we handle deleting the new image
                // if we fail to update our image
                s3_path.replace(path);
            }
        }
        // update the entity's groups
        update_form.update_groups(&mut self, shared).await?;
        // make sure the new name is valid
        if update_form.name.as_ref().is_some_and(String::is_empty) {
            return bad!("Entity cannot have an empty name!".to_string());
        }
        // update the entity's name
        update!(self.name, update_form.name);
        // update the entity's other data
        update_opt!(self.description, update_form.description);
        update_clear_opt!(self.description, update_form.clear_description);
        // update our entity metadata
        self.update_meta(user, update_form.metadata, shared).await?;
        // clear our entities image if clear image is set
        if update_form.clear_image == Some(true) {
            // add our current image to our deletes by taking it from the entity
            if let Some(image_path) = self.image.take() {
                deletes.push(image_path);
            }
        }
        // update this entity
        db::entities::update(
            self,
            &update_form.add_groups,
            &update_form.remove_groups,
            shared,
        )
        .await?;
        // delete any graphics that are no longer needed
        stream::iter(deletes)
            .map(|key| async move { Self::delete_graphic(&key, shared).await })
            .buffer_unordered(10)
            .collect::<Vec<Result<(), ApiError>>>()
            .await
            .into_iter()
            .collect::<Result<(), ApiError>>()?;
        Ok(())
    }

    /// Update an `Entity`
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply
    /// * `user` - The user updating the entity
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "Entity::update", skip_all, err(Debug))]
    pub async fn update(
        self,
        user: &User,
        form: Multipart,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // validate that this user can edit this entity in all requested groups
        let _ = Group::authorize_check_allow_all(
            user,
            &self.groups,
            Group::editable,
            "edit",
            Some(GroupAllowAction::Entities),
            shared,
        )
        .await?;
        // track any newly added image paths so we can delete them in needed
        // this will only track when an image is added to an entity that did
        // not previously have an image.
        let mut s3_path: Option<String> = None;
        // try to update
        match self.update_helper(user, form, &mut s3_path, shared).await {
            Ok(()) => Ok(()),
            Err(error) => {
                // delete from S3 if our path was set to avoid a dangling image
                if let Some(s3_path) = &s3_path {
                    // we have the path in S3, so just delete it with the client directly
                    if let Err(s3_error) = Self::delete_graphic(s3_path, shared).await {
                        return internal_err!(format!(
                            "Error cleaning up image after entity create error: {s3_error}. Original error: {error}"
                        ));
                    }
                }
                // propogate the error
                Err(error)
            }
        }
    }

    /// Delete all associations for this
    ///
    /// # Arguments
    ///
    /// * `shared` - Shared Thorium objects
    pub(crate) async fn delete_associations(&self, shared: &Shared) -> Result<(), ApiError> {
        // build the associations list opts for this entity
        let opts = AssociationListOpts::default()
            .groups(self.groups.clone())
            .limit(500);
        // build the source target for this entity if we have one
        if let Some(source) = self.build_association_target_column() {
            // list all associations for this entity
            let mut cursor = db::associations::list(opts, &source, shared).await?;
            // step over our associations and delete them
            loop {
                // delete this page of associations
                db::associations::delete_many(&source, &cursor.data, shared).await?;
                // check if this cursor has been exhausted
                if cursor.exhausted() {
                    break;
                }
                // clear our current cursor
                cursor.data.clear();
                // get the next page of data
                cursor.next(shared).await?;
            }
        }
        Ok(())
    }

    /// Delete an `Entity`
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply
    /// * `user` - The user updating the entity
    /// * `shared` - Shared Thorium objects
    pub async fn delete(self, user: &User, shared: &Shared) -> Result<(), ApiError> {
        // if we are the owner of this entity then we can delete it from all groups
        if self.submitter != user.username && !user.is_admin() {
            // we are not the owner so we can only delete this from groups we are a manager for
            // get the group info for all the groups we want to delete this from
            let groups = db::groups::list_details(self.groups.iter(), shared).await?;
            // make sure we can delete data in all of these groups
            for group in groups {
                // check if we are a manager or owner in this group
                if !group.is_manager_or_owner(&user.username) {
                    // we cannot delete this entity so raise an error
                    return unauthorized!(format!("Cannot delete data from group {}", group.name));
                }
            }
        };
        // remove any associations for this entity
        self.delete_associations(shared).await?;
        // delete the entity
        db::entities::delete(user, &self, shared).await?;
        // delete this entities image if one exists
        if let Some(s3_path) = &self.image {
            // delete our image graphic
            Self::delete_graphic(s3_path, shared).await?;
        }
        Ok(())
    }

    /// Ensure that user has group privileges up to the given `role_check` and
    /// that all the groups allow the given `action`
    ///
    /// # Arguments
    ///
    /// * `user` - The user requesting the action
    /// * `groups` - The entity groups for which the action was requested
    /// * `role_check` - The function used to check for the user's role/privileges in the groups
    /// * `role_check_name` - The name of the role check to use in logging (i.e. "view"/"edit")
    /// * `action` - The action to check in each group if one is given
    /// * `shared` - Shared Thorium objects
    pub async fn validate_check_allow_groups<F>(
        &self,
        user: &User,
        groups: &mut Vec<String>,
        role_check: F,
        role_check_name: &str,
        action: Option<GroupAllowAction>,
        shared: &Shared,
    ) -> Result<(), ApiError>
    where
        F: Fn(&Group, &User) -> Result<(), ApiError>,
    {
        if groups.is_empty() {
            // the user specified no groups, so default to ones
            // that pass the privilege check function and allow the given action;
            // first check that we have access to all of the groups we have
            let group_objs = Group::authorize_all(user, &self.groups, shared).await?;
            if let Some(action) = action {
                groups.extend(
                    group_objs
                        .into_iter()
                        .filter(|group| group.allowable(action).is_ok())
                        .filter(|group| role_check(group, user).is_ok())
                        .map(|group| group.name),
                );
            } else {
                groups.extend(
                    group_objs
                        .into_iter()
                        .filter(|group| role_check(group, user).is_ok())
                        .map(|group| group.name),
                );
            }
        } else {
            // make sure the entity is in all the given groups
            if !groups.iter().all(|group| self.groups.contains(group)) {
                return unauthorized!(format!(
                    "Entity '{}' is not in all specified groups",
                    self.name
                ));
            }
            // make sure we have access in all requested groups
            let _ = Group::authorize_check_allow_all(
                user,
                groups,
                role_check,
                role_check_name,
                action,
                shared,
            )
            .await?;
        }
        // make sure we got at least some groups
        if groups.is_empty() {
            return unauthorized!(format!(
                "The user does not have permissions to {role_check_name} \
                    entities in any of the given groups!"
            ));
        }
        // all groups valid
        Ok(())
    }

    /// Add an [`EntityRow`] to an existing entity
    ///
    /// # Arguments
    ///
    /// * `row` - The row to add
    pub(super) fn add_row(&mut self, row: EntityRow) {
        // add this row's group to the list
        self.groups.push(row.group);
    }

    /// Drop any data that is built based on associations instead of statically
    ///
    /// We do this because we don't want to write this to scylla as it is built dynamically at retrieval time
    pub(super) fn drop_associated_data(&mut self) {
        match &mut self.metadata {
            // vendors in devices is built by association
            EntityMetadata::Device(device) => device.vendors.clear(),
            // vendor/other has no association specific data
            EntityMetadata::Vendor(_) | EntityMetadata::Other => (),
        }
    }
}

// Add graphic support for entities
impl GraphicSupport for Entity {
    /// A unique, immutable key to use to reference the implementing object
    #[cfg(feature = "api")]
    type GraphicKey<'a> = &'a Uuid;

    /// Build the base path for this graphic
    fn build_graphic_base_path<'a>(key: Self::GraphicKey<'a>) -> PathBuf {
        // entities just use their id as their key
        PathBuf::from(key.to_string())
    }

    /// Build the base path for this graphic
    fn build_graphic_base_path_from_self(&self) -> PathBuf {
        // call our base path builder
        Self::build_graphic_base_path(&self.id)
    }
}

impl EntityMetadata {
    /// Split an entity kind into its name and its serialized data if it has any
    pub fn split(&self) -> Result<(EntityKinds, Option<String>), ApiError> {
        let data = match self {
            EntityMetadata::Device(device_entity) => Some(serialize!(device_entity)),
            EntityMetadata::Vendor(vendor_entity) => Some(serialize!(vendor_entity)),
            EntityMetadata::Other => None,
        };
        Ok((self.into(), data))
    }
}

impl EntityForm {
    /// Adds a multipart field to our entity form
    ///
    /// # Returns
    ///
    /// Returns the field again if it's an image, otherwise attempts to consume it and
    /// returns None on success
    ///
    /// # Errors
    ///
    /// Returns an error if the field is invalid
    ///
    /// # Arguments
    ///
    /// * `field` - The field to try to add
    pub async fn add<'a>(&'a mut self, field: Field<'a>) -> Result<Option<Field<'a>>, ApiError> {
        // get the name of this field
        if let Some(name) = field.name() {
            // add this fields value to our form
            match name {
                "name" => self.name = Some(field.text().await?),
                "groups" => self.groups.push(field.text().await?),
                "description" => self.description = Some(field.text().await?),
                // this is image data so return it so we can stream it to s3
                "image" => return Ok(Some(field)),
                // kind fields
                "kind" => {
                    // try to cast our kind to the correct kind
                    let cast = EntityKinds::from_str(&field.text().await?)?;
                    // set our kind
                    self.kind = Some(cast);
                }
                "metadata[urls]" => {
                    self.metadata.urls.push(field.text().await?);
                }
                "metadata[vendor]" => {
                    // try to parse this vendor id
                    let vendor_id = field.text().await?.parse::<Uuid>()?;
                    // add this id to our vendor list
                    self.metadata.vendors.push(vendor_id);
                }
                "metadata[critical_system]" => {
                    self.metadata.critical_system = Some(field.text().await?.parse()?);
                }
                "metadata[critical_sectors]" => {
                    // try to convert this field to a critical sector
                    let sector = CriticalSector::from_str(&field.text().await?)?;
                    // add this critical sector
                    self.metadata.critical_sectors.insert(sector);
                }
                "metadata[sensitive_location]" => {
                    self.metadata.sensitive_location = Some(field.text().await?.parse()?);
                }
                "metadata[countries]" => {
                    // validate and parse this country
                    let country = Country::new(&field.text().await?)?;
                    // add this country to our metadata form
                    self.metadata.countries.insert(country);
                }
                _ => {
                    // check if this is a tags key
                    if name.starts_with("tags[") {
                        // this is a tag so get the key substring
                        let key = &name[5..name.len() - 1];
                        // get an entry to this tags value vec
                        let entry = self.tags.entry(key.to_owned()).or_default();
                        // add our value
                        entry.insert(field.text().await?);
                        return Ok(None);
                    }
                    return bad!(format!("'{}' is not a valid form name", name));
                }
            }
            // we found and consumed a valid form entry
            return Ok(None);
        }
        bad!(format!("All entity form entries must have a name!"))
    }

    /// Build an association request for anything in this entity form
    pub(super) async fn build_association_req(
        &mut self,
        user: &User,
        id: Uuid,
        shared: &Shared,
    ) -> Result<Option<AssociationRequest>, ApiError> {
        // theres some validation duplication here
        // get our kind or raise an error
        let kind = match self.kind {
            Some(kind) => kind,
            None => return bad!("No kind set for entity?".to_owned()),
        };
        // get our name or raise an error
        let name = match &self.name {
            Some(name) => name,
            None => return bad!("No name set for entity?".to_owned()),
        };
        // build the association request for each of our different entities
        match kind {
            EntityKinds::Device => {
                // devices to vendor relationships is always developed by
                let assoc_kind = AssociationKind::DevelopedBy;
                // get the numer of associations to make
                let vendor_len = self.metadata.vendors.len();
                // build our source target
                let source = AssociationTarget::Entity {
                    id,
                    name: name.to_owned(),
                };
                // start with an empty req
                let mut req = AssociationRequest::with_capacity(assoc_kind, source, vendor_len);
                // step over the vendors we are associating with this device
                for vendor_id in self.metadata.vendors.drain(..) {
                    // get the entity for this vendor if it exists
                    let entity = Entity::get(user, vendor_id, shared).await?;
                    // make sure this is a vendor entity
                    if entity.kind != EntityKinds::Vendor {
                        return bad!(format!("Entity {vendor_id} is not a vendor!"));
                    }
                    // build the source object for this entity
                    let other = AssociationTarget::Entity {
                        id: vendor_id,
                        name: entity.name,
                    };
                    // add this link to the other
                    req.targets.push(other);
                }
                Ok(Some(req))
            }
            _ => Ok(None),
        }
    }

    #[instrument(name = "EntityForm::cast", skip_all, fields(name = self.name), err(Debug))]
    pub async fn cast(
        mut self,
        user: &User,
        id: Uuid,
        shared: &Shared,
    ) -> Result<Entity, ApiError> {
        // make sure we have a kind set
        let kind = match self.kind.take() {
            Some(kind) => kind,
            None => return bad!("A kind must be set for all entities".to_owned()),
        };
        // make sure we got a name
        let name = match self.name.take() {
            Some(name) => name,
            None => return bad!("Entity must have a name!".to_string()),
        };
        // make sure the name isn't empty
        if name.is_empty() {
            return bad!("Entity's name cannot be empty!".to_string());
        }
        // make sure we got groups
        if self.groups.is_empty() {
            return bad!("Entity must be in at least 1 group!".to_string());
        }
        // make sure none of the groups have empty names
        if self.groups.iter().any(String::is_empty) {
            return bad!(format!(
                "Entity cannot have any groups with empty names: {:?}",
                self.groups
            ));
        }
        // convert our metadata to an actual entity
        let metadata = self.metadata.cast(kind, &self.groups, shared).await?;
        // build an association request for this
        // build our casted entity
        let cast = Entity {
            id,
            name,
            kind,
            metadata,
            description: self.description,
            submitter: user.username.clone(),
            groups: self.groups,
            created: Utc::now(),
            tags: HashMap::default(),
            image: self.image,
        };
        Ok(cast)
    }
}

impl EntityMetadataForm {
    /// Attempt to cast the entity kind form to an [`EntityKind`], verifying the
    /// form is valid
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups the entity will be added to
    /// * `shared` - Shared Thorium objects
    pub async fn cast(
        self,
        kind: EntityKinds,
        groups: &[String],
        shared: &Shared,
    ) -> Result<EntityMetadata, ApiError> {
        // cast the kind form depending on the kind name
        match kind {
            EntityKinds::Device => Ok(EntityMetadata::Device(
                DeviceEntity::from_form(self, groups, shared).await?,
            )),
            EntityKinds::Vendor => Ok(EntityMetadata::Vendor(VendorEntity::from_form(self))),
            EntityKinds::Other => Ok(EntityMetadata::Other),
        }
    }
}

impl EntityUpdateForm {
    /// Adds a multipart field to our entity update form
    ///
    /// # Returns
    ///
    /// Returns the field again if it's an image, otherwise attempts to consume it and
    /// returns None on success
    ///
    /// # Errors
    ///
    /// Returns an error if the field is invalid
    ///
    /// # Arguments
    ///
    /// * `field` - The field to try to add
    pub async fn add<'a>(&'a mut self, field: Field<'a>) -> Result<Option<Field<'a>>, ApiError> {
        // get the name of this field
        if let Some(name) = field.name() {
            // add this fields value to our form
            match name {
                "name" => self.name = Some(field.text().await?),
                "add_groups" => self.add_groups.push(field.text().await?),
                "remove_groups" => self.remove_groups.push(field.text().await?),
                "clear_image" => self.clear_image = Some(field.text().await?.parse()?),
                "description" => self.description = Some(field.text().await?),
                "clear_description" => self.clear_description = Some(field.text().await?.parse()?),
                // this is image data so return it so we can stream it to s3
                "image" => return Ok(Some(field)),
                _ => {
                    // check if this is a kind field
                    if let Some(kind_name) =
                        EntityMetadataUpdateForm::parse_metadata_field_name(name)
                    {
                        // get an owned value before we move the field
                        let kind_name = kind_name.to_owned();
                        // try to add the kind field
                        self.metadata.add(field, &kind_name).await?;
                    } else {
                        // this is an invalid form field
                        return bad!(format!("'{name}' is not a valid entity update form name"));
                    }
                }
            }
            // we found and consumed a valid form entry
            return Ok(None);
        }
        bad!(format!("All entity update form entries must have a name!"))
    }

    /// Update an entity's groups
    ///
    /// Does not consume the form's groups because we need them later to calculate
    /// what should be modified in the db
    ///
    /// # Arguments
    ///
    /// * `entity` - The entity whose groups to update
    /// * `shared` - Shared Thorium objects
    #[instrument(name = "EntityUpdate::update_groups", skip_all, fields(entity = entity.name), err(Debug))]
    async fn update_groups(&self, entity: &mut Entity, shared: &Shared) -> Result<(), ApiError> {
        // make sure the groups added/removed aren't empty
        if self.add_groups.iter().any(String::is_empty) {
            return bad!("One or more of the groups to add has an empty name!".to_string());
        }
        if self.remove_groups.iter().any(String::is_empty) {
            return bad!("One or more of the groups to remove has an empty name!".to_string());
        }
        // make sure all of the added groups exist
        if !db::groups::exists(&self.add_groups, shared)
            .await
            .map_err(|err| {
                ApiError::new(
                    err.code,
                    Some(format!("Unable to verify that added groups exist: {err}")),
                )
            })?
        {
            return not_found!(format!(
                "One or more of the specified groups to add doesn't exist!"
            ));
        }
        update_add_rem!(
            entity.groups,
            self.add_groups,
            self.remove_groups,
            "Entity",
            "groups"
        )?;
        // make sure we still have some groups left
        if entity.groups.is_empty() {
            return bad!(
                "You cannot delete all of an entity's groups with an update request!".to_string()
            );
        }
        Ok(())
    }
}

impl EntityMetadataUpdateForm {
    /// Try to parse the kind field name from the field if it's a kind field
    /// (i.e. `kind[<NAME>]` → `<NAME>`)
    ///
    /// # Arguments
    ///
    /// * `field_name` - The name of the field
    fn parse_metadata_field_name(field_name: &str) -> Option<&str> {
        if field_name.starts_with("metadata[") && field_name.ends_with(']') {
            Some(&field_name[9..field_name.len() - 1])
        } else {
            None
        }
    }

    /// Adds a multipart field to our entity kind update form
    ///
    /// # Errors
    ///
    /// Returns an error if the field is invalid
    ///
    /// # Arguments
    ///
    /// * `field` - The field to try to add
    async fn add<'a>(&'a mut self, field: Field<'a>, kind_name: &str) -> Result<(), ApiError> {
        match kind_name {
            "add_urls" => self.add_urls.push(field.text().await?),
            "remove_urls" => self.remove_urls.push(field.text().await?),
            "add_vendors" => self.add_vendors.push(field.text().await?.parse()?),
            "remove_vendors" => self.remove_vendors.push(field.text().await?.parse()?),
            "critical_system" => self.critical_system = Some(field.text().await?.parse()?),
            "clear_critical_system" => {
                self.clear_critical_system = Some(field.text().await?.parse()?);
            }
            "sensitive_location" => self.sensitive_location = Some(field.text().await?.parse()?),
            "clear_sensitive_location" => {
                self.clear_sensitive_location = Some(field.text().await?.parse()?);
            }
            "add_critical_sectors" => {
                self.add_critical_sectors.push(field.text().await?.parse()?);
            }
            "remove_critical_sectors" => {
                self.remove_critical_sectors
                    .push(field.text().await?.parse()?);
            }
            "add_countries" => {
                // validate and parse this country
                let country = Country::new(&field.text().await?)?;
                // add this country to our metadata form
                self.add_countries.push(country);
            }
            "remove_countries" => {
                // validate and parse this country
                let country = Country::new(&field.text().await?)?;
                // add this country to our metadata form
                self.remove_countries.push(country);
            }
            _ => {
                return bad!(format!(
                    "'{kind_name}' is not a valid entity kind update form name"
                ));
            }
        }
        Ok(())
    }

    /// Build an association request for anything in this entity form
    pub(super) async fn update_associations(
        &mut self,
        user: &User,
        entity: &mut Entity,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // build the association request for each of our different entities
        match &entity.metadata {
            EntityMetadata::Device(_) => {
                // we only have to update vendors for device updates
                if !self.add_vendors.is_empty() {
                    // devices to vendor relationships is always developed by
                    let assoc_kind = AssociationKind::DevelopedBy;
                    // get the numer of associations to make
                    let vendor_len = self.add_vendors.len();
                    // build our source target
                    let source = AssociationTarget::Entity {
                        id: entity.id,
                        name: entity.name.clone(),
                    };
                    // start with an empty req
                    let mut req = AssociationRequest::with_capacity(assoc_kind, source, vendor_len);
                    // step over the vendors we are associating with this device
                    for vendor_id in self.add_vendors.drain(..) {
                        // get the entity for this vendor if it exists
                        let other_entity = Entity::get(user, vendor_id, shared).await?;
                        // make sure this is a vendor entity
                        if other_entity.kind != EntityKinds::Vendor {
                            return bad!(format!("Entity {vendor_id} is not a vendor!"));
                        }
                        // build the source object for this entity
                        let other = AssociationTarget::Entity {
                            id: vendor_id,
                            name: other_entity.name,
                        };
                        // add this link to the other
                        req.targets.push(other);
                    }
                    // create these associations
                    req.apply(user, shared).await?;
                }
                // if we have any associations to delete then do that
                if !self.remove_vendors.is_empty() {
                    // convert out list of vendors to a list of serialized target columns
                    let serialized = self
                        .remove_vendors
                        .iter()
                        .filter_map(|id| {
                            // this should never actually fail
                            serde_json::to_string(&AssociationTargetColumn::Entity(*id)).ok()
                        })
                        .collect::<Vec<String>>();
                    // build our association target column
                    let source = match entity.build_association_target_column() {
                        Some(source) => source,
                        // this should be impossible and never occur
                        None => {
                            return internal_err!(format!(
                                "Failed to build association target column for {entity:#?}"
                            ));
                        }
                    };
                    // build the opts for listing associations
                    let opts = AssociationListOpts::default()
                        .groups(user.groups.clone())
                        .limit(100);
                    // list associations for this entity
                    let mut cursor = db::associations::list(opts, &source, shared).await?;
                    // build a list of associations to remove
                    let mut remove_assoc = Vec::with_capacity(self.remove_vendors.len());
                    // step over all of our associations and find the ones to delete
                    loop {
                        // filter down to just the vendors that we want to remove
                        for assoc in cursor.data.drain(..) {
                            // determine if this is a vendor association
                            if assoc.kind == AssociationKind::DevelopedBy
                                && serialized.contains(&assoc.other)
                            {
                                // add this association that we want to remove
                                remove_assoc.push(assoc);
                            }
                        }
                        // check if this cursor is exhausted
                        if cursor.exhausted() {
                            break;
                        }
                        // get the next page of data in this cusor
                        cursor.next(shared).await?;
                    }
                    // get our source target column
                    if let Some(source) = entity.build_association_target_column() {
                        // delete all of the requested associations
                        db::associations::delete_many(&source, &remove_assoc, shared).await?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl TryFrom<EntityRow> for Entity {
    type Error = ApiError;

    fn try_from(row: EntityRow) -> Result<Self, Self::Error> {
        // deserialize our metadata
        let metadata = deserialize!(&row.metadata);
        // return the entity with just the single group from this row and no tags
        Ok(Self {
            id: row.id,
            name: row.name,
            kind: row.kind,
            metadata,
            groups: vec![row.group],
            created: row.created,
            submitter: row.submitter,
            description: row.description,
            tags: TagMap::with_capacity(1),
            image: row.image,
        })
    }
}

// Implement cursor support for entities
impl CursorCore for EntityListLine {
    /// The params to build this cursor from
    type Params = EntityListParams;

    /// Filter by entity kind
    type ExtraFilters = Vec<EntityKinds>;

    /// The type of data to group our rows by
    type GroupBy = String;

    /// The data structure to store tie info in
    ///
    /// For entities this is a mapping of group to a tuple of
    /// name and ID
    type Ties = HashMap<String, Uuid>;

    fn bucket_limit(extra_filters: &Self::ExtraFilters) -> u32 {
        // keep our cartesian product under 98 by dividing the number of kinds
        // we are searching against
        (98 / extra_filters.len()) as u32
    }

    fn partition_size(shared: &Shared) -> u16 {
        shared.config.thorium.entities.partition_size
    }

    fn get_id(params: &Self::Params) -> Option<Uuid> {
        params.cursor.clone()
    }

    fn get_start_end(
        params: &Self::Params,
        shared: &Shared,
    ) -> Result<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>), ApiError> {
        // get our end timestmap
        let end = params.end(shared)?;
        Ok((params.start, end))
    }

    fn get_group_by(params: &mut Self::Params) -> Vec<Self::GroupBy> {
        std::mem::take(&mut params.groups)
    }

    fn get_extra_filters(params: &mut Self::Params) -> Self::ExtraFilters {
        std::mem::take(&mut params.kinds)
    }

    fn get_tag_filters(
        params: &mut Self::Params,
    ) -> Option<(TagType, HashMap<String, Vec<String>>)> {
        // Only return tags if some were set
        if params.tags.is_empty() {
            None
        } else {
            Some((TagType::Entities, params.tags.clone()))
        }
    }

    fn get_limit(params: &Self::Params) -> usize {
        params.limit
    }

    fn add_tie(&self, ties: &mut Self::Ties) {
        // if its not already in the tie map then add each of its groups to our map
        for group in &self.groups {
            // get an entry to this group tie
            let entry = ties.entry(group.clone());
            // insert our entity id
            entry.or_insert_with(|| self.id);
        }
    }

    fn dedupe_item(&self, dedupe_set: &mut HashSet<String>) -> bool {
        let id = self.id.to_string();
        // if this is already in our dedupe set then skip it
        if dedupe_set.contains(&id) {
            // we already have this sample so skip it
            false
        } else {
            // add this new sample to our dedupe set
            dedupe_set.insert(id);
            // keep this new sample
            true
        }
    }
}

#[async_trait::async_trait]
impl ScyllaCursorSupport for EntityListLine {
    type IntermediateRow = EntityListRow;

    type UniqueType<'a> = Uuid;

    fn add_tag_tie(&self, ties: &mut HashMap<String, String>) {
        // if its not already in the tie map then add each of its groups to our map
        for group in &self.groups {
            // insert this groups tie
            ties.insert(group.clone(), self.id.to_string());
        }
    }

    fn get_intermediate_timestamp(intermediate: &Self::IntermediateRow) -> DateTime<Utc> {
        // return the created time as the timestamp
        intermediate.created
    }

    fn get_timestamp(&self) -> DateTime<Utc> {
        // return the created time as the timestamp
        self.created
    }

    fn get_intermediate_unique_key(intermediate: &Self::IntermediateRow) -> Self::UniqueType<'_> {
        // the entity's id is a unique key
        intermediate.id
    }

    fn get_unique_key(&self) -> Self::UniqueType<'_> {
        // the entity's id is a unique key
        self.id
    }

    fn add_group_to_line(&mut self, group: String) {
        self.groups.insert(group);
    }

    fn add_intermediate_to_line(&mut self, intermediate: Self::IntermediateRow) {
        self.groups.insert(intermediate.group);
    }

    fn from_tag_row(row: TagListRow) -> Self {
        Self::from(row)
    }

    fn census_keys<'a>(
        group_by: &'a Vec<Self::GroupBy>,
        _extra: &Self::ExtraFilters,
        year: i32,
        bucket: u32,
        keys: &mut Vec<(&'a Self::GroupBy, String, i32)>,
        shared: &Shared,
    ) {
        // build the keys for each census stream we are going to crawl
        for group in group_by {
            // build the key for this entities census stream
            let key = format!(
                "{namespace}:census:entities:stream:{group}:{year}",
                namespace = shared.config.thorium.namespace,
                group = group,
                year = year,
            );
            // add this key to our keys
            keys.push((group, key, bucket as i32));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ties_query(
        ties: &mut Self::Ties,
        kinds: &Self::ExtraFilters,
        year: i32,
        bucket: i32,
        uploaded: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<Vec<impl Future<Output = Result<QueryResult, ExecutionError>>>, ApiError> {
        // allocate space for 300 futures
        let mut futures = Vec::with_capacity(ties.len());
        // if any ties were found then get the rest of them and add them to data
        for (group, id) in ties.drain() {
            // execute our query
            let future = shared.scylla.session.execute_unpaged(
                &shared.scylla.prep.entities.list_ties,
                (kinds, group, year, bucket, uploaded, id, limit),
            );
            // add this future to our set
            futures.push(future);
        }
        Ok(futures)
    }

    /// Builds the query string for getting the next page of entity rows
    ///
    /// # Arguments
    ///
    /// * `group` - The group to restrict our query too
    /// * `kinds` - The entity kinds to filter on
    /// * `year` - The year to get data for
    /// * `bucket` - The bucket to get data for
    /// * `start` - The earliest timestamp to get data from
    /// * `end` - The oldest timestamp to get data from
    /// * `limit` - The max amount of data to get from this query
    /// * `shared` - Shared Thorium objects
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity, clippy::type_repetition_in_bounds)]
    async fn pull(
        group: &Self::GroupBy,
        kinds: &Self::ExtraFilters,
        year: i32,
        buckets: Vec<i32>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: i32,
        shared: &Shared,
    ) -> Result<QueryResult, ExecutionError> {
        // execute our query
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.entities.list_pull,
                (kinds, group, year, buckets, start, end, limit),
            )
            .await
    }
}

impl From<EntityListRow> for EntityListLine {
    fn from(row: EntityListRow) -> Self {
        Self {
            groups: HashSet::from([row.group]),
            name: row.name,
            id: row.id,
            kind: row.kind,
            created: row.created,
        }
    }
}

impl From<TagListRow> for EntityListLine {
    /// Convert a tag list row to an entity list line
    ///
    /// # Panics
    ///
    /// Panics if the tag row's item is not the entity's UUID as it should be
    #[allow(clippy::expect_used)]
    fn from(row: TagListRow) -> Self {
        // build our initial group set
        let mut groups = HashSet::with_capacity(1);
        // add this group
        groups.insert(row.group);
        // parse the id from the row item column
        let id = Uuid::parse_str(&row.item).expect("Failed to parse UUID from tag row item");
        // build our repo list line
        Self {
            groups,
            id,
            // set defaults for name/kind because the tag rows don't have them;
            // we'll get them from the entities materialized view after we've
            // finished listing this round
            name: String::default(),
            kind: EntityKinds::default(),
            created: row.uploaded,
        }
    }
}

impl ApiCursor<EntityListLine> {
    /// Turns a cursor of [`EntityListLine`] into a cursor of [`Entity`]
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is getting the details for this list
    /// * `shared` - Shared Thorium objects
    #[instrument(
        name = "ApiCursor<EntityListLine>::details",
        skip_all
        err(Debug)
    )]
    pub(crate) async fn details(
        self,
        user: &User,
        shared: &Shared,
    ) -> Result<ApiCursor<Entity>, ApiError> {
        // build a list of entity names we need to get details on
        let ids = self
            .data
            .into_iter()
            .map(|line| line.id)
            .collect::<Vec<_>>();
        // use correct backend to list sample details
        let data = for_groups!(db::entities::get_many, user, shared, &ids)?;
        // get any required association data for the entities we found
        let mut populated = Vec::with_capacity(data.len());
        // create a stream of entities to populate with any association data
        let mut populator_stream = stream::iter(data)
            .map(|entity| entity.populate_associations(user, shared))
            .buffer_unordered(20);
        // get populated items from our stream until we hit a problem or no more exist
        while let Some(entity) = populator_stream.next().await {
            // raise an error if we ran into a problem
            populated.push(entity?);
        }
        // build our new cursor object
        Ok(ApiCursor {
            cursor: self.cursor,
            data: populated,
        })
    }
}

impl<S> FromRequestParts<S> for EntityListParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new()
                .max_depth(5)
                .deserialize_str(query)?)
        } else {
            // provide default params if none were given
            Ok(Self::default())
        }
    }
}
