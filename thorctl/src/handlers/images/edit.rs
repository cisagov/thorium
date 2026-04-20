//! Handles the image edit command

use thorium::{Error, Thorium};

use crate::CtlConf;
use crate::args::images::EditImage;
use crate::handlers::edit::{self, ImageEditable};

/// Edit an image using a text editor, detect the updates, then update the image
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl conf
/// * `cmd` - The edit image command that was run
pub async fn edit(thorium: Thorium, conf: &CtlConf, cmd: &EditImage) -> Result<(), Error> {
    edit::edit_entity::<ImageEditable>(
        &thorium,
        conf,
        &cmd.image,
        cmd.group.as_deref(),
        cmd.editor.as_deref(),
    )
    .await
}
