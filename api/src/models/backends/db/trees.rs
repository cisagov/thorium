//! The tree related database methods

use redis::cmd;
use uuid::Uuid;

use super::cursors::CursorKind;
use super::keys::cursors;
use crate::models::{Tree, User};
use crate::utils::{ApiError, Shared};
use crate::{deserialize, not_found, query, serialize};

/// Save this tree to redis
///
/// The tree is scoped to the user that created it by nesting its id under the
/// user's username in redis
///
/// # Arguments
///
/// * `user` - The  user that is saving this cursor
/// * `tree` - The tree to save
/// * `shared` - Shared Thorium objects
pub async fn save(user: &User, tree: &mut Tree, shared: &Shared) -> Result<(), ApiError> {
    // swap our growable items with an empty vec
    let growable = std::mem::take(&mut tree.growable);
    // serialize this tree
    let serialized = serialize!(&tree);
    // build the key to save this cursor data too scoped to the creating user
    let key = cursors::data(CursorKind::Tree, &user.username, &tree.id, shared);
    // save this cursors data to redis and expire it after 3 days
    let _: () = query!(
        cmd("set").arg(key).arg(serialized).arg("EX").arg(259_200),
        shared
    )
    .await?;
    // place our growable items back
    tree.growable = growable;
    Ok(())
}

/// load this tree from redis
///
/// The tree is scoped to the user that created it by nesting its id under the
/// user's username in redis
///
/// # Arguments
///
/// * `user` - The  user that is saving this cursor
/// * `id` - The id of the tree to load
/// * `shared` - Shared Thorium objects
pub async fn load(user: &User, id: Uuid, shared: &Shared) -> Result<Tree, ApiError> {
    // build the key to load this cursor data from scoped to the creating user
    let key = cursors::data(CursorKind::Tree, &user.username, &id, shared);
    // get this cursors data from redis
    let serialized: Option<String> = query!(cmd("get").arg(key), shared).await?;
    // check if we got any tree data for this user
    match serialized {
        // try to deserialize the tree we found
        Some(serialized) => Ok(deserialize!(&serialized)),
        // no tree exists for this user so return a not found instead of leaking that
        // this id is a valid tree owned by another user
        None => not_found!(format!("Tree {} doesn't exist", id)),
    }
}
