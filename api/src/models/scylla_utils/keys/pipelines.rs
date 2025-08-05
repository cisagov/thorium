//! The scylla utils for pipelines
use crate::models::Pipeline;

#[cfg(feature = "scylla-utils")]
use thorium_derive::ScyllaStoreJson;

/// The components forming a unique key to access a pipeline's data in Scylla;
/// these components may make up only part of a partition key
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(ScyllaStoreJson))]
pub struct PipelineKey {
    /// The group containing the pipeline
    pub group: String,
    /// The pipeline of the image
    pub pipeline: String,
}

impl AsRef<PipelineKey> for PipelineKey {
    fn as_ref(&self) -> &PipelineKey {
        self
    }
}

impl PipelineKey {
    /// Create a new `PipelineKey`
    ///
    /// # Arguments
    ///
    /// * `group` - The group the image is in
    /// * `pipeline` - The name of the image
    pub fn new<S, T>(group: S, pipeline: T) -> Self
    where
        S: Into<String>,
        T: Into<String>,
    {
        Self {
            group: group.into(),
            pipeline: pipeline.into(),
        }
    }
}

/// Produce a pipeline key from a pipeline
impl From<&Pipeline> for PipelineKey {
    fn from(pipeline: &Pipeline) -> Self {
        PipelineKey {
            group: pipeline.group.clone(),
            pipeline: pipeline.name.clone(),
        }
    }
}
