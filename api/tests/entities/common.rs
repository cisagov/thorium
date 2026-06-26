//! Shared helpers for the entity CRUD integration tests
//!
//! Each entity kind has its own test binary under `tests/entities/`. Every one of
//! those binaries pulls this module in with `mod common;` to reuse the setup and
//! assertion logic below, so some helpers may be unused in any single binary.
#![allow(dead_code)]

use thorium::models::{Entity, EntityListOpts, EntityMetadataUpdate, EntityRequest, EntityUpdate};
use thorium::test_utilities::{self, generators};
use thorium::{Error, Thorium, fail, is, is_in};

/// Get an admin client and create a single group to hold test entities
///
/// Returns the client and the name of the created group.
pub async fn setup() -> Result<(Thorium, String), Error> {
    // get an admin client
    let client = test_utilities::admin_client().await?;
    // create a group for our entities to live in
    let group = generators::groups(1, &client).await?.remove(0).name;
    Ok((client, group))
}

/// List all entity details in a group, paging through the entire cursor
///
/// # Arguments
///
/// * `client` - The client to list entities with
/// * `group` - The group to list entities from
pub async fn list_all(client: &Thorium, group: &str) -> Result<Vec<Entity>, Error> {
    // build the options to list all of the entities in our group
    let opts = EntityListOpts::default()
        .groups(vec![group.to_owned()])
        .limit(10_000);
    // get the first page of entity details
    let mut cursor = client.entities.list_details(&opts).await?;
    // start collecting the entities from the first page
    let mut all = std::mem::take(&mut cursor.data);
    // page through the rest of the cursor until it is exhausted
    while !cursor.exhausted() {
        cursor.refill().await?;
        all.append(&mut cursor.data);
    }
    Ok(all)
}

/// Create an entity from a request and verify it matches and is listed
///
/// # Arguments
///
/// * `client` - The client to create the entity with
/// * `group` - The group the entity should be created in
/// * `req` - The entity request to create
pub async fn check_create(client: &Thorium, group: &str, req: EntityRequest) -> Result<(), Error> {
    // create the entity and fetch its full details back
    let (req, entity) = generators::entity(req, client).await?;
    // make sure the created entity matches the request
    is!(entity, req);
    // make sure the entity shows up in a detailed listing for its group
    let listed = list_all(client, group).await?;
    is_in!(listed, req);
    Ok(())
}

/// Create an entity from a request, update it, and verify the update was applied
///
/// # Arguments
///
/// * `client` - The client to create and update the entity with
/// * `req` - The entity request to create then update
pub async fn check_update(client: &Thorium, req: EntityRequest) -> Result<(), Error> {
    // create the entity we are going to update
    let (_, entity) = generators::entity(req, client).await?;
    // create a second group to move the entity into
    let new_group = generators::groups(1, client).await?.remove(0).name;
    // build an update that touches the name, description, and groups
    let update = EntityUpdate::default()
        .name(generators::gen_string(16))
        .description(generators::gen_string(32))
        .group(new_group);
    // apply the update to our entity
    client.entities.update(entity.id, update.clone()).await?;
    // get the entity back and make sure the update was applied
    let updated = client.entities.get(entity.id).await?;
    is!(updated, update);
    Ok(())
}

/// Create an entity, update its generic and kind-specific fields, and verify
///
/// Builds a kind-specific metadata update from the created entity (so it can
/// exercise remove-semantics against existing values), applies it alongside the
/// generic name/description/group changes, and verifies every field was applied.
///
/// # Arguments
///
/// * `client` - The client to create and update the entity with
/// * `req` - The entity request to create then update
/// * `gen_meta` - Builds the kind-specific metadata update from the created entity
pub async fn check_update_meta(
    client: &Thorium,
    req: EntityRequest,
    gen_meta: impl FnOnce(&Entity) -> EntityMetadataUpdate,
) -> Result<(), Error> {
    // create the entity we are going to update
    let (_, entity) = generators::entity(req, client).await?;
    // create a second group to move the entity into
    let new_group = generators::groups(1, client).await?.remove(0).name;
    // build an update touching the name, description, groups, and kind metadata
    let update = EntityUpdate::default()
        .name(generators::gen_string(16))
        .description(generators::gen_string(32))
        .group(new_group)
        .metadata(gen_meta(&entity));
    // apply the update to our entity
    client.entities.update(entity.id, update.clone()).await?;
    // get the entity back and make sure the update was applied
    let updated = client.entities.get(entity.id).await?;
    is!(updated, update);
    Ok(())
}

/// Create an entity from a request, delete it, and verify it can't be fetched
///
/// # Arguments
///
/// * `client` - The client to create and delete the entity with
/// * `req` - The entity request to create then delete
pub async fn check_delete(client: &Thorium, req: EntityRequest) -> Result<(), Error> {
    // create the entity we are going to delete
    let (_, entity) = generators::entity(req, client).await?;
    // delete the entity
    client.entities.delete(entity.id).await?;
    // make sure the entity can no longer be fetched
    let resp = client.entities.get(entity.id).await;
    fail!(resp, 404);
    Ok(())
}
