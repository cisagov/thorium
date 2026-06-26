//! Integration tests for `Other` entity CRUD

mod common;

use test_utilities::generators;
use thorium::test_utilities;

/// Verify a other entity can be created and matches its request
#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build an other entity request and verify creation
    let req = generators::gen_entity(&group, generators::gen_other_meta());
    common::check_create(&client, &group, req).await
}

/// Verify a other entity is updated as requested
#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build an other entity request and verify updates apply
    let req = generators::gen_entity(&group, generators::gen_other_meta());
    common::check_update(&client, req).await
}

/// Verify a other entity can be deleted
#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build an other entity request and verify deletion
    let req = generators::gen_entity(&group, generators::gen_other_meta());
    common::check_delete(&client, req).await
}

// Sync tests

#[cfg(all(feature = "sync", not(feature = "python")))]
/// Verify other entity create and delete via the blocking client
#[test]
fn create_delete_blocking() -> Result<(), thorium::Error> {
    // get a blocking admin client
    let client = test_utilities::admin_client_blocking()?;
    // create a group for our entity
    let group = generators::groups_blocking(1, &client)?.remove(0).name;
    // build and create an other entity
    let req = generators::gen_entity(&group, generators::gen_other_meta());
    let resp = client.entities.create(req)?;
    // delete the entity we just created
    client.entities.delete(resp.id)?;
    Ok(())
}
