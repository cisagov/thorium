//! Structs and code related to updating pipelines

use thorium::models::{Pipeline, PipelineBanUpdate, PipelineRequest, PipelineUpdate};

use crate::{calc_remove_add_map, set_clear, set_modified_new_opt, set_modified_opt};

/// A pipeline that needs to be updated and how that update should
/// be performed
#[derive(Debug)]
pub struct ToolboxPipelineUpdate {
    /// The name of the pipeline
    pub pipeline_name: String,
    /// The name of the pipeline's version
    pub pipeline_version: String,
    /// The group the pipeline is in
    pub group: String,
    /// The operation that needs to be performed to update the
    /// pipeline
    pub op: ToolboxPipelineUpdateOp,
}

impl ToolboxPipelineUpdate {
    pub fn new<N, V, G>(
        pipeline_name: N,
        pipeline_version: V,
        group: G,
        op: ToolboxPipelineUpdateOp,
    ) -> Self
    where
        N: Into<String>,
        V: Into<String>,
        G: Into<String>,
    {
        Self {
            pipeline_name: pipeline_name.into(),
            pipeline_version: pipeline_version.into(),
            group: group.into(),
            op,
        }
    }
}

/// An operation that needs to be performed to update a pipeline
#[derive(Debug)]
pub enum ToolboxPipelineUpdateOp {
    /// This is a brand new pipeline that needs to be created
    Create(Box<PipelineRequest>),
    /// This is an existing pipeline that needs to be updated
    Update(Box<PipelineUpdate>),
    /// Nothing needs to be done for this pipeline
    Unchanged,
}

/// Calculate what updates need to be made to a pipeline (if any) based
/// on the pipeline's current state and the request from the toolbox manifest
///
/// # Arguments
///
/// * `pipeline` - The current state of the pipeline in Thorium
/// * `req` - The pipeline request from the toolbox manifest
#[allow(clippy::needless_pass_by_value)]
pub fn calculate_pipeline_update(
    pipeline: Pipeline,
    mut req: PipelineRequest,
) -> Option<PipelineUpdate> {
    if pipeline == req {
        // no update is necessary if the pipeline and the request are the same
        return None;
    }
    let (remove_triggers, triggers) = calc_remove_add_map!(pipeline.triggers, req.triggers);
    Some(PipelineUpdate {
        // set order if the orders are different
        order: (!req.compare_order(&pipeline.order)).then_some(req.order),
        sla: set_modified_new_opt!(pipeline.sla, req.sla),
        triggers,
        remove_triggers,
        clear_description: set_clear!(pipeline.description, req.description),
        description: set_modified_opt!(pipeline.description, req.description),
        // bans aren't in a manifest, so we can just set default here
        bans: PipelineBanUpdate::default(),
    })
}
