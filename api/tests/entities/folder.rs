//! Integration tests for `Folder` entity CRUD
//!
//! Folder entities reference a filesystem entity by id, so each test creates a
//! filesystem entity first to satisfy that dependency.

mod common;

use test_utilities::generators;
use thorium::models::EntityRequest;
use thorium::test_utilities;
use thorium::{Error, Thorium};

/// Build a folder entity request backed by a freshly created filesystem
///
/// # Arguments
///
/// * `client` - The client to create the filesystem dependency with
/// * `group` - The group the filesystem and folder should be in
async fn request(client: &Thorium, group: &str) -> Result<EntityRequest, Error> {
    // create a filesystem entity for this folder to belong to
    let filesystem = generators::filesystem_entity(group, client).await?;
    // build a folder entity request that references our filesystem
    Ok(generators::gen_entity(
        group,
        generators::gen_folder_meta(filesystem.id),
    ))
}

/// Verify a folder entity can be created and matches its request
#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a folder entity request and verify creation
    let req = request(&client, &group).await?;
    common::check_create(&client, &group, req).await
}

/// Verify a folder entity is updated as requested
#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a folder entity request and verify updates apply
    let req = request(&client, &group).await?;
    common::check_update(&client, req).await
}

/// Verify a folder entity can be deleted
#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a folder entity request and verify deletion
    let req = request(&client, &group).await?;
    common::check_delete(&client, req).await
}
