//! Handles image notification related commands

use thorium::{
    Thorium,
    models::{NotificationParams, NotificationRequest},
};

use crate::Error;
use crate::utils;
use crate::{
    args::images::{
        CreateImageNotification, DeleteImageNotification, GetImageNotifications, ImageNotifications,
    },
    err_not_admin,
};

/// Get notifications for a specific image
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The describe image command to execute
async fn get_notifications(thorium: Thorium, cmd: &GetImageNotifications) -> Result<(), Error> {
    // get the image's notifications
    let notifications = thorium
        .images
        .get_notifications(&cmd.group, &cmd.image)
        .await?;
    // print the notifications
    utils::notifications::print_notifications(&notifications, cmd.opts.ids);
    Ok(())
}

/// Add an image notification
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The add notification command that was run
async fn add_notification(thorium: Thorium, cmd: &CreateImageNotification) -> Result<(), Error> {
    // create a notification request
    let req =
        NotificationRequest::new(cmd.notification.msg.clone(), cmd.notification.level.clone());
    // create the notification params
    let params = NotificationParams {
        expire: cmd.notification.expire,
    };
    // create the notification
    err_not_admin!(
        thorium
            .images
            .create_notification(&cmd.group, &cmd.image, &req, &params)
            .await,
        "create image notifications"
    );
    Ok(())
}

/// Delete an image notification
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The delete notification command that was run
async fn delete_notification(thorium: Thorium, cmd: &DeleteImageNotification) -> Result<(), Error> {
    // delete the notification
    err_not_admin!(
        thorium
            .images
            .delete_notification(&cmd.group, &cmd.image, &cmd.id)
            .await,
        "delete image notifications"
    );
    Ok(())
}

/// Handle the image notification sub command
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The subcommand that was run
pub async fn handle(thorium: Thorium, cmd: &ImageNotifications) -> Result<(), Error> {
    match cmd {
        ImageNotifications::Get(cmd) => get_notifications(thorium, cmd).await,
        ImageNotifications::Create(cmd) => add_notification(thorium, cmd).await,
        ImageNotifications::Delete(cmd) => delete_notification(thorium, cmd).await,
    }
}
