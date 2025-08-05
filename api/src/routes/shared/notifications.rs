//! Contains generic functions for interacting with notifications from a route

use axum::http::StatusCode;
use axum::Json;
use tracing::instrument;
use uuid::Uuid;

use crate::models::backends::NotificationSupport;
use crate::models::{Notification, NotificationParams, NotificationRequest};
use crate::not_found;
use crate::utils::{ApiError, Shared};

/// Create a notification for an entity in scylla
///
/// # Arguments
///
/// * `entity` - The entity whose notification we're deleting
/// * `key` - The key to the entity
/// * `req` - The request to create the notification
/// * `params` - The params to use when creating the notification
/// * `req` - The notification request that was sent
#[instrument(
    name = "routes::shared::notifications::create_notification",
    skip_all,
    err(Debug)
)]
pub async fn create_notification<N: NotificationSupport>(
    entity: N,
    key: N::Key,
    req: NotificationRequest<N>,
    params: NotificationParams,
    shared: &Shared,
) -> Result<StatusCode, ApiError> {
    // create the notification
    entity.create_notification(key, req, params, shared).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get the all of the image's notifications
///
/// # Arguments
///
/// * `entity` - The entity whose notification we're deleting
/// * `key` - The key to the entity
/// * `shared` - Shared Thorium objects
#[instrument(
    name = "routes::shared::notifications::get_notifications",
    skip_all,
    err(Debug)
)]
pub async fn get_notifications<N: NotificationSupport>(
    entity: N,
    key: N::Key,
    shared: &Shared,
) -> Result<Json<Vec<Notification<N>>>, ApiError> {
    // get all of the entity's notifications
    let notifications = entity.get_notifications(&key, shared).await?;
    Ok(Json(notifications))
}

/// Delete a specific notification from an entity
///
/// # Arguments
///
/// * `entity` - The entity whose notification we're deleting
/// * `key` - The key to the entity
/// * `extra` - Any extra info we might need to display the key
/// * `id` - The id of the notification we're deleting
/// * `shared` - Shared Thorium objects
#[instrument(
    name = "routes::shared::notifications::delete_notification",
    skip_all,
    err(Debug)
)]
pub async fn delete_notification<N: NotificationSupport>(
    entity: N,
    key: N::Key,
    extra: Option<N::ExtraKey>,
    id: Uuid,
    shared: &Shared,
) -> Result<StatusCode, ApiError> {
    // get all of the entity's notifications
    let notifications = entity.get_notifications(&key, shared).await?;
    // find the notification we want to delete
    let target = notifications.iter().find(|n| n.id == id);
    // delete the notification if it exists
    match target {
        Some(notification) => {
            entity.delete_notification(notification, shared).await?;
        }
        None => {
            // return a 404 if the log does not exist
            return not_found!(format!(
                "A notification with id '{}' does not exist for '{}'",
                id,
                N::key_url(&key, extra.as_ref())
            ));
        }
    }
    Ok(StatusCode::NO_CONTENT)
}
