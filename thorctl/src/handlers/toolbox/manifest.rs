//! The toolbox manifest structure

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thorium::Error;
use thorium::models::{ImageRequest, PipelineRequest};

/// A toolbox manifest – a description of pipelines and images that
/// can be imported into Thorium
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolboxManifest {
    /// The name of this toolbox
    pub name: String,
    /// The registry the images can be found at
    pub registry: String,
    /// A map of pipeline names to their details
    pub pipelines: HashMap<String, PipelineManifest>,
    /// A map of image names to their details
    pub images: HashMap<String, ImageManifest>,
}

impl ToolboxManifest {
    /// Validate the toolbox manifest
    pub fn validate(&self) -> Result<(), Error> {
        // validate the manifest's pipelines
        self.validate_pipelines()?;
        Ok(())
    }

    /// Validate the manifest's pipelines
    fn validate_pipelines(&self) -> Result<(), Error> {
        // make sure all pipelines' requested images are in the manifest
        let mut pipelines_missing_images: HashMap<String, Vec<String>> = HashMap::new();
        for (pipeline_name, pipeline_manifest) in &self.pipelines {
            for pipeline_version in pipeline_manifest.versions.values() {
                for (image_name, image_version) in &pipeline_version.images {
                    match self.images.get(image_name) {
                        Some(image) => {
                            if !image.versions.contains_key(&image_version.version) {
                                // the pipeline requires a specific image version that isn't in the manifest
                                pipelines_missing_images
                                    .entry(pipeline_name.clone())
                                    .or_default()
                                    .push(format!("{}:{}", image_name, image_version.version));
                            }
                        }
                        None => {
                            // the pipeline requires an image that isn't in the manifest
                            pipelines_missing_images
                                .entry(pipeline_name.clone())
                                .or_default()
                                .push(image_name.clone());
                        }
                    }
                }
            }
        }
        if !pipelines_missing_images.is_empty() {
            return Err(Error::new(format!(
                "One or more pipelines require images not found in the manifest: {pipelines_missing_images:?}"
            )));
        }
        Ok(())
    }

    /// Returns all of the groups the manifest expects to exist
    pub fn groups(&self) -> HashSet<String> {
        self.pipelines
            .values()
            .flat_map(|pipeline_manifest| pipeline_manifest.versions.values())
            .map(|pipeline_version| &pipeline_version.config.group)
            .chain(
                self.images
                    .values()
                    .flat_map(|image_manifest| image_manifest.versions.values())
                    .map(|image_version| &image_version.config.group),
            )
            .cloned()
            .collect()
    }

    /// Force all images and pipelines to be imported to the given group by
    /// setting the group for each item in the manifest, returning the updated
    /// manifest
    ///
    /// # Arguments
    ///
    /// * `group` - The group to force items to be imported to
    pub fn override_group(mut self, group: &str) -> Self {
        let group = group.to_string();
        // get references to all groups
        self.pipelines
            .values_mut()
            .flat_map(|pipeline_manifest| pipeline_manifest.versions.values_mut())
            .map(|pipeline_version| &mut pipeline_version.config.group)
            .chain(
                self.images
                    .values_mut()
                    .flat_map(|image_manifest| image_manifest.versions.values_mut())
                    .map(|image_version| &mut image_version.config.group),
            )
            // set each group reference to the given group
            .for_each(|group_ref| group_ref.clone_from(&group));
        self
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineManifest {
    /// A map of pipeline versions to their details
    #[serde(flatten)]
    pub versions: HashMap<String, PipelineVersion>,
}

/// Details for a specific pipeline version
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineVersion {
    /// A description of the pipeline for the purpose of the toolbox, not for
    /// Thorium itself
    pub description: String,
    /// A map of image names to their info for the pipeline
    pub images: HashMap<String, PipelineImage>,
    /// The pipeline's Thorium configuration
    pub config: PipelineRequest,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineImage {
    /// The version of the image this pipeline expects
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageManifest {
    /// A map of image versions to their details
    #[serde(flatten)]
    pub versions: HashMap<String, ImageVersion>,
}

/// Details for a specific image version
#[derive(Debug, Serialize, Deserialize)]
pub struct ImageVersion {
    /// The image's build path relative to the toolbox manifest's location
    pub build_path: String,
    /// The image's Thorium config
    pub config: ImageRequest,
}
