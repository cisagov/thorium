//! Support for interacting with notifications
use crate::models::KeySupport;
use crate::models::NotificationType;

// dependencies required for api
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use axum::extract::FromRequestParts;
        use axum::http::request::Parts;
        use tracing::instrument;
        use futures::stream;
        use futures::{StreamExt, TryStreamExt};
        use uuid::Uuid;

        use crate::internal_err;
        use crate::models::backends::db;
        use crate::models::bans::Ban;
        use crate::utils::{ApiError, Shared};
        use crate::models::{Notification, NotificationParams, NotificationRequest};
    }
}

/// Describe an entity that supports creating, deleting, and retrieving notifications
/// relevant to that entity
#[allow(async_fn_in_trait)]
pub trait NotificationSupport: KeySupport + Sized {
    /// Returns the implementor's [`NotificationType`]
    fn notification_type() -> NotificationType;

    /// Create's the notification in Thorium
    ///
    /// # Arguments
    ///
    /// * `key` - The key to the underlying entity the notification is referencing
    /// * `req` - The notification request that was sent
    /// * `params` - The notification params that were sent
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[instrument(
        name = "NotificationSupport::create_notification",
        skip(self, req, params, shared),
        err(Debug)
    )]
    async fn create_notification(
        &self,
        key: Self::Key,
        req: NotificationRequest<Self>,
        params: NotificationParams,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // cast the request to a notification
        let notification: Notification<Self> = Notification::new(key, req.msg, req.level);
        // create the notification
        db::notifications::create(notification, params.expire, shared).await?;
        Ok(())
    }

    /// Retrieves all of an entity's notifications from Thorium
    ///
    /// # Arguments
    ///
    /// * `key` - The entity's unique key to retrieve the notifications
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[instrument(
        name = "NotificationSupport:get_notifications",
        skip(self, shared),
        err(Debug)
    )]
    async fn get_notifications(
        &self,
        key: &Self::Key,
        shared: &Shared,
    ) -> Result<Vec<Notification<Self>>, ApiError> {
        // get all of the entity's notifications
        let notifications = db::notifications::get_all(key, shared).await?;
        Ok(notifications)
    }

    /// Deletes a notification in Thorium
    ///
    /// # Arguments
    ///
    /// * `notification` - The notification to delete
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[instrument(name = "NotificationSupport:delete_notification", skip_all, err(Debug))]
    async fn delete_notification(
        &self,
        notification: &Notification<Self>,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // delete the notification in Thorium
        db::notifications::delete(notification, shared).await?;
        Ok(())
    }

    /// Deletes all of an entity's notifications in Thorium
    ///
    /// # Arguments
    ///
    /// * `key` - The entity's unique key to delete its notifications
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[instrument(
        name = "NotificationSupport:delete_all_notifications",
        skip_all,
        err(Debug)
    )]
    async fn delete_all_notifications(
        &self,
        key: &Self::Key,
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // delete the notifications in Thorium
        db::notifications::delete_all::<Self>(key, shared).await?;
        Ok(())
    }

    /// Update the entity's notifications based on the bans added/removed
    ///
    /// # Arguments
    ///
    /// * `key` - The key to the underlying entity to retrieve notifications
    /// * `bans_added` - The bans that were added
    /// * `bans_removed` - The bans that were removed
    /// * `shared` - Shared Thorium objects
    #[cfg(feature = "api")]
    #[instrument(
        name = "NotificationSupport:update_ban_notifications",
        skip(self, bans_added, bans_removed, shared),
        err(Debug)
    )]
    async fn update_ban_notifications<B: Ban<Self>>(
        &self,
        key: &Self::Key,
        bans_added: &[B],
        bans_removed: &[Uuid],
        shared: &Shared,
    ) -> Result<(), ApiError> {
        // first get all of the entity's notifications
        let notifications: Vec<Notification<Self>> =
            match db::notifications::get_all(key, shared).await {
                Ok(notifications) => notifications,
                Err(err) => {
                    return internal_err!(format!(
                        "Error while updating notifications: {}",
                        err.msg.unwrap_or_else(|| {
                            "an unknown error occurred retrieving notifications".to_string()
                        })
                    ))
                }
            };
        // create a notification for each added ban
        let new_notifications = bans_added
            .iter()
            .map(|ban| Notification::new_ban(ban, key.clone()));
        // save each notification to scylla
        stream::iter(new_notifications)
            .map(Ok)
            .try_for_each_concurrent(None, |notification| {
                db::notifications::create(notification, None, shared)
            })
            .await?;
        // determine which notifications need to be removed
        let remove_notifications = notifications.iter().filter(|notification| {
            notification
                .ban_id
                .as_ref()
                .is_some_and(|id| bans_removed.contains(id))
        });
        // delete notifications from bans in scylla
        stream::iter(remove_notifications)
            .map(Ok)
            .try_for_each_concurrent(None, |notification| {
                db::notifications::delete(notification, shared)
            })
            .await?;
        Ok(())
    }
}

#[cfg(feature = "api")]
impl<S> FromRequestParts<S> for NotificationParams
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // try to extract our query
        if let Some(query) = parts.uri.query() {
            // try to deserialize our query string
            Ok(serde_qs::Config::new(5, false).deserialize_str(query)?)
        } else {
            Ok(Self::default())
        }
    }
}
