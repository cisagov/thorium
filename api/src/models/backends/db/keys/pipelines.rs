use crate::models::pipelines::Pipeline;
use crate::utils::Shared;

/// The keys to store/retrieve pipeline data/sets/lists
pub struct PipelineKeys {
    /// The pipeline data key
    pub data: String,
    /// The group pipeline set key
    pub set: String,
}

impl PipelineKeys {
    /// Builds the keys to access pipeline data/sets in redis
    ///
    /// # Arguments
    ///
    /// * `pipeline` - Pipeline object to build keys for
    /// * `shared` - Shared Thorium objects
    pub fn new(pipeline: &Pipeline, shared: &Shared) -> Self {
        // build keys to use in redis
        // pipelines data key
        let data = Self::data(&pipeline.group, &pipeline.name, shared);
        // group pipeline set key
        let set = Self::set(&pipeline.group, shared);

        // build pipeline Keys
        PipelineKeys { data, set }
    }

    /// Builds key to group pipeline set
    ///
    /// # Arguments
    ///
    /// * `group` - The group containing the pipeline
    /// * `shared` - Shared Thorium objects
    pub fn set(group: &str, shared: &Shared) -> String {
        format!(
            "{ns}:pipelines:{group}",
            ns = shared.config.thorium.namespace,
            group = group
        )
    }

    /// Builds key to pipeline data
    ///
    /// # Arguments
    ///
    /// * `group` - The group containing the pipeline
    /// * `name` - The name of the pipeline
    /// * `shared` - Shared Thorium objects
    pub fn data(group: &str, name: &str, shared: &Shared) -> String {
        format!(
            "{ns}:pipeline_data:{group}:{name}",
            ns = shared.config.thorium.namespace,
            group = group,
            name = name
        )
    }
}
