//! Integration tests for `WindowsProcessTree` entity CRUD

mod common;

use test_utilities::generators;
use thorium::test_utilities;

/// Verify a windows process tree entity can be created and matches its request
#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a windows process tree entity request and verify creation
    let req = generators::gen_entity(&group, generators::gen_windows_process_tree_meta());
    common::check_create(&client, &group, req).await
}

/// Verify a windows process tree entity is updated as requested
#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a windows process tree entity request and verify updates apply
    let req = generators::gen_entity(&group, generators::gen_windows_process_tree_meta());
    common::check_update_meta(&client, req, generators::gen_windows_process_tree_update).await
}

/// Verify a windows process tree entity can be deleted
#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // set up an admin client and a group
    let (client, group) = common::setup().await?;
    // build a windows process tree entity request and verify deletion
    let req = generators::gen_entity(&group, generators::gen_windows_process_tree_meta());
    common::check_delete(&client, req).await
}
