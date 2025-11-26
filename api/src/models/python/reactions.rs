use pyo3::pymethods;
use std::collections::HashMap;
use uuid::Uuid;

use crate::models::{ReactionArgs, ReactionRequest, RepoDependencyRequest};

#[pymethods]
impl ReactionRequest {
    /// Creates a [`ReactionRequest`] for a new reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group this reaction should be in
    /// * `pipeline` - The pipeline this reaction should be based on
    #[new]
    #[pyo3(signature =
        (
            group,
            pipeline,
            args: "dict[str, GenericJobArgs]" = ReactionArgs::default(),
            sla = None,
            tags: "list[str]" = Vec::new(),
            parent: "UUID | None" = None,
            samples: "list[str]" = Vec::new(),
            buffers: "dict[str, str]" = HashMap::new(),
            repos: "list[RepoDependencyRequest]" = Vec::new(),
            trigger_depth = None,
        ) -> "ReactionRequest"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new_py(
        group: String,
        pipeline: String,
        args: ReactionArgs,
        sla: Option<u64>,
        tags: Vec<String>,
        parent: Option<Uuid>,
        samples: Vec<String>,
        buffers: HashMap<String, String>,
        repos: Vec<RepoDependencyRequest>,
        trigger_depth: Option<u8>,
    ) -> Self {
        Self {
            group,
            pipeline,
            args,
            sla,
            tags,
            parent,
            samples,
            buffers,
            repos,
            trigger_depth,
        }
    }
}
