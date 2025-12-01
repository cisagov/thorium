use pyo3::pymethods;
use std::collections::HashMap;
use uuid::Uuid;

use crate::models::{ReactionArgs, ReactionCache, ReactionRequest, RepoDependencyRequest};

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
            cache: "ReactionCache" = ReactionCache::default(),
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
        cache: ReactionCache,
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
            cache,
        }
    }
}

#[pymethods]
impl ReactionCache {
    /// Defines a [`ReactionCache`] of information for a reaction
    ///
    /// # Arguments
    ///
    /// * `generic` - A generic key/value cache of info across this reaction
    /// * `files` - Files in this reaction cache
    #[new]
    #[pyo3(signature =
        (
            generic: "dict[str, str]" = HashMap::new(),
            files: "list[str]" = Vec::new(),
        ) -> "ReactionCache"
    )]
    fn new_py(generic: HashMap<String, String>, files: Vec<String>) -> Self {
        Self { generic, files }
    }
}
