//! Handles pipeline notification related commands

use thorium::{
    Thorium,
    models::{NotificationParams, NotificationRequest},
};

use crate::Error;
use crate::utils;
use crate::{
    args::pipelines::{
        CreatePipelineNotification, DeletePipelineNotification, GetPipelineNotifications,
        PipelineNotifications,
    },
    err_not_admin,
};

/// Get notifications for a specific pipeline
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The describe pipeline command to execute
async fn get_notifications(thorium: Thorium, cmd: &GetPipelineNotifications) -> Result<(), Error> {
    // get the pipeline's notifications
    let notifications = thorium
        .pipelines
        .get_notifications(&cmd.group, &cmd.pipeline)
        .await?;
    // print the notifications
    utils::notifications::print_notifications(&notifications, cmd.opts.ids);
    Ok(())
}

/// Add a pipeline notification
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The add notification command that was run
async fn add_notification(thorium: Thorium, cmd: &CreatePipelineNotification) -> Result<(), Error> {
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
            .pipelines
            .create_notification(&cmd.group, &cmd.pipeline, &req, &params)
            .await,
        "add pipeline notifications"
    );
    Ok(())
}

/// Delete a pipeline notification
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The delete notification command that was run
async fn delete_notification(
    thorium: Thorium,
    cmd: &DeletePipelineNotification,
) -> Result<(), Error> {
    // delete the notification
    err_not_admin!(
        thorium
            .pipelines
            .delete_notification(&cmd.group, &cmd.pipeline, &cmd.id)
            .await,
        "delete pipeline notifications"
    );
    Ok(())
}

/// Handle the pipeline notification sub command
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The subcommand that was run
pub async fn handle(thorium: Thorium, cmd: &PipelineNotifications) -> Result<(), Error> {
    match cmd {
        PipelineNotifications::Get(cmd) => get_notifications(thorium, cmd).await,
        PipelineNotifications::Create(cmd) => add_notification(thorium, cmd).await,
        PipelineNotifications::Delete(cmd) => delete_notification(thorium, cmd).await,
    }
}
