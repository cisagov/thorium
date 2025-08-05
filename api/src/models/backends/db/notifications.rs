//! Logic for interacting with notifications in the database

use chrono::prelude::*;
use tracing::instrument;
use uuid::Uuid;

use crate::{
    models::backends::NotificationSupport,
    models::{Notification, NotificationLevel},
    utils::{ApiError, Shared},
};

/// Save a notification to scylla
///
/// # Arguments
///
/// * `notification` - The notification to save
/// * `expire` - Whether or not this notification should automatically expire
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::notifications::create", skip_all, err(Debug))]
pub async fn create<N: NotificationSupport>(
    notification: Notification<N>,
    expire: Option<bool>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // determine whether or not this notification should automatically expire
    // if no explicit setting was given, the notification should only expire if it's not an error
    let expire = expire.unwrap_or_else(|| notification.level != NotificationLevel::Error);
    if expire {
        // save the notification to scylla
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.notifications.insert,
                (
                    N::notification_type(),
                    notification.key,
                    notification.created,
                    notification.id,
                    notification.msg,
                    notification.level,
                    notification.ban_id,
                ),
            )
            .await?;
    } else {
        // save the notification to scylla with no expiration
        shared
            .scylla
            .session
            .execute_unpaged(
                &shared.scylla.prep.notifications.insert_no_expire,
                (
                    N::notification_type(),
                    notification.key,
                    notification.created,
                    notification.id,
                    notification.msg,
                    notification.level,
                    notification.ban_id,
                ),
            )
            .await?;
    }
    Ok(())
}

/// Get all notifications for an entity at the given key
///
/// # Arguments
///
/// * `key` - The entity's unique key
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::notifications::get_all", skip_all, err(Debug))]
pub async fn get_all<N: NotificationSupport>(
    key: &N::Key,
    shared: &Shared,
) -> Result<Vec<Notification<N>>, ApiError> {
    // query for the notifications
    let query = shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.notifications.get,
            (N::notification_type(), key),
        )
        .await?;
    // enable rows on this query response
    let query_rows = query.into_rows_result()?;
    // cast the rows to notifications
    let rows = query_rows.rows::<(
        N::Key,
        DateTime<Utc>,
        Uuid,
        String,
        NotificationLevel,
        Option<Uuid>,
    )>()?;
    // instance a list of notification with the right size
    let mut notifs = Vec::with_capacity(query_rows.rows_num());
    // build our notifications
    for row in rows {
        // try to deserialize this row
        let (key, created, id, msg, level, ban_id) = row?;
        // build this notification and add it to our list
        notifs.push(Notification {
            key,
            created,
            id,
            msg,
            level,
            ban_id,
        });
    }
    Ok(notifs)
}

/// Delete a specific notification
///
/// # Arguments
///
/// * `notification` - The notification to delete
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::notifications::delete", skip_all, err(Debug))]
pub async fn delete<N: NotificationSupport>(
    notification: &Notification<N>,
    shared: &Shared,
) -> Result<(), ApiError> {
    // delete the notification in scylla
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.notifications.delete,
            (
                N::notification_type(),
                &notification.key,
                &notification.created,
                &notification.id,
            ),
        )
        .await?;
    Ok(())
}

/// Deletes all notifications for a given entity
///
/// # Arguments
///
/// * `key` - The entity's unique key
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::notifications::delete_all", skip_all, err(Debug))]
pub async fn delete_all<N: NotificationSupport>(
    key: &N::Key,
    shared: &Shared,
) -> Result<(), ApiError> {
    // delete all of the entity's notifications in scylla
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.notifications.delete_all,
            (N::notification_type(), key),
        )
        .await?;
    Ok(())
}
