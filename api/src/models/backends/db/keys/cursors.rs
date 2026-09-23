//! Build the keys to cursors in redis

use uuid::Uuid;

use crate::models::backends::db::cursors::CursorKind;
use crate::utils::Shared;

/// Build the key to this cursors data
///
/// The key nests the cursor id under the creating user's username so that cursors
/// are scoped to the user that created them
///
/// # Arguments
///
/// * `kind` - The kind of cursor this key is for
/// * `user` - The username of the user this cursor is scoped to
/// * `id` - The id of the cursor this key is for
/// * `shared` - Shared Thorium objects
pub fn data(kind: CursorKind, user: &str, id: &Uuid, shared: &Shared) -> String {
    // scope this cursor to its creating user by nesting the id under the username
    format!(
        "{ns}:cursors:{kind}:{user}:{id}",
        kind = kind.as_str(),
        ns = &shared.config.thorium.namespace
    )
}
