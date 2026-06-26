//! Integration tests for `Device` entity CRUD
//!
//! Device entities reference vendor entities by id, so each test creates a vendor
//! entity first to satisfy that dependency.

mod common;

use test_utilities::generators;
use thorium::models::EntityRequest;
use thorium::test_utilities;
use thorium::{Error, Thorium};

/// Build a device entity request backed by a freshly created vendor
///
/// # Arguments
///
/// * `client` - The client to create the vendor dependency with
/// * `group` - The group the vendor and device should be in
async fn request(client: &Thorium, group: &str) -> Result<EntityRequest, Error> {
    // create a vendor entity to associate with this device
    let vendor = generators::vendor_entity(group, client).await?;
    // build a device entity request that references our vendor
    Ok(generators::gen_entity(
        group,
        generators::gen_device_meta(vec![vendor.id]),
    ))
}

/// Verify a device entity can be created and matches its request
#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a device entity request and verify creation
    let req = request(&client, &group).await?;
    common::check_create(&client, &group, req).await
}

/// Verify a device entity is updated as requested
#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a device entity request and verify updates apply
    let req = request(&client, &group).await?;
    common::check_update_meta(&client, req, generators::gen_device_update).await
}

/// Verify a device entity can be deleted
#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a device entity request and verify deletion
    let req = request(&client, &group).await?;
    common::check_delete(&client, req).await
}
