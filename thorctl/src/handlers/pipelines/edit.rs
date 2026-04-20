//! Handles the pipeline edit command

use thorium::{Error, Thorium};

use crate::CtlConf;
use crate::args::pipelines::EditPipeline;
use crate::handlers::edit::{self, PipelineEditable};

/// Edit a pipeline using a text editor, detect the updates, then update the pipeline
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl conf
/// * `cmd` - The edit pipeline command that was run
pub async fn edit(thorium: Thorium, conf: &CtlConf, cmd: &EditPipeline) -> Result<(), Error> {
    edit::edit_entity::<PipelineEditable>(
        &thorium,
        conf,
        &cmd.pipeline,
        cmd.group.as_deref(),
        cmd.editor.as_deref(),
    )
    .await
}
