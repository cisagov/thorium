//! The tree related database methods

use redis::cmd;
use uuid::Uuid;

use super::cursors::CursorKind;
use super::keys::cursors;
use crate::models::{Tree, User};
use crate::utils::{ApiError, Shared};
use crate::{deserialize, query, serialize};

/// Save this tree to redis
///
/// User should be eventually used to "namespace" cursors to specific users
///
/// # Arguments
///
/// * `user` - The  user that is saving this cursor
/// * `tree` - The tree to save
/// * `shared` - Shared Thorium objects
pub async fn save(_user: &User, tree: &mut Tree, shared: &Shared) -> Result<(), ApiError> {
    // swap our growable items with an empty vec
    let growable = std::mem::take(&mut tree.growable);
    // serialize this tree
    let serialized = serialize!(&tree);
    // build the key to save this cursor data too
    let key = cursors::data(CursorKind::Tree, &tree.id, shared);
    // save this cursors data to redis
    let _: () = query!(
        cmd("set").arg(key).arg(serialized).arg("EX").arg(2_628_000),
        shared
    )
    .await?;
    // place our growable items back
    tree.growable = growable;
    Ok(())
}

/// load this tree from redis
///
/// User should be eventually used to "namespace" cursors to specific users
///
/// # Arguments
///
/// * `user` - The  user that is saving this cursor
/// * `id` - The id of the tree to load
/// * `shared` - Shared Thorium objects
pub async fn load(_user: &User, id: &Uuid, shared: &Shared) -> Result<Tree, ApiError> {
    // build the key to save this cursor data too
    let key = cursors::data(CursorKind::Tree, &id, shared);
    // data this cursors data to redis
    let serialized: String = query!(cmd("get").arg(key), shared).await?;
    // try to deserialize this tree
    let tree = deserialize!(&serialized);
    Ok(tree)
}
