//! Handle image ban related commands

use thorium::{
    Thorium,
    models::{ImageBan, ImageBanKind, ImageBanUpdate, ImageUpdate},
};

use crate::args::images::{CreateImageBan, DeleteImageBan, ImageBans};
use crate::{Error, err_not_admin};

/// Add a ban to an image in Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The add image ban command that was run
async fn add_ban(thorium: Thorium, cmd: &CreateImageBan) -> Result<(), Error> {
    // create a new ban
    let ban = ImageBan::new(ImageBanKind::generic(&cmd.msg));
    // add the ban to an image update
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().add_ban(ban));
    // send the update
    err_not_admin!(
        thorium.images.update(&cmd.group, &cmd.image, &update).await,
        "ban images"
    );
    Ok(())
}

/// Remove a ban from an image in Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The remove image ban command that was run
async fn remove_ban(thorium: Thorium, cmd: &DeleteImageBan) -> Result<(), Error> {
    // add the ban to remove to an image update
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().remove_ban(cmd.id));
    // send the update
    err_not_admin!(
        thorium.images.update(&cmd.group, &cmd.image, &update).await,
        "remove image bans"
    );
    Ok(())
}

/// Handle image ban commands
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The image bans sub command that was run
pub async fn handle(thorium: Thorium, cmd: &ImageBans) -> Result<(), Error> {
    match cmd {
        ImageBans::Create(cmd) => add_ban(thorium, cmd).await,
        ImageBans::Delete(cmd) => remove_ban(thorium, cmd).await,
    }
}
