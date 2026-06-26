//! Integration tests for `Vendor` entity CRUD

mod common;

use test_utilities::generators;
use thorium::test_utilities;

/// Verify a vendor entity can be created and matches its request
#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a vendor entity request and verify creation
    let req = generators::gen_entity(&group, generators::gen_vendor_meta());
    common::check_create(&client, &group, req).await
}

/// Verify a vendor entity is updated as requested
#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a vendor entity request and verify updates apply
    let req = generators::gen_entity(&group, generators::gen_vendor_meta());
    common::check_update_meta(&client, req, generators::gen_vendor_update).await
}

/// Verify a vendor entity can be deleted
#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a vendor entity request and verify deletion
    let req = generators::gen_entity(&group, generators::gen_vendor_meta());
    common::check_delete(&client, req).await
}
