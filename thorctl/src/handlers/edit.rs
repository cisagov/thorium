//! A shared editor-based "edit an entity" flow for images and pipelines.
//!
//! Both `images edit` and `pipelines edit` fetch the resource, open its editable
//! [`MergeableImage`]/[`MergeablePipeline`] view in the user's editor, then diff
//! the result into an update. The differences between the two are captured by the
//! [`EditableEntity`] trait; [`edit_entity`] is the shared driver.

use colored::Colorize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use thorium::models::{Image, ImageUpdate, Pipeline, PipelineUpdate};
use thorium::{CtlConf, Error, Thorium};

use crate::handlers::imports::editor::{editor_loop, resolve_editor};
use crate::handlers::imports::merge::{
    IMAGE_FIELD_ORDER, MergeableImage, MergeablePipeline, PIPELINE_FIELD_ORDER,
};
use crate::handlers::imports::update;
use crate::utils;

/// An entity that can be edited in a text editor and updated in Thorium
pub trait EditableEntity {
    /// The entity as stored in Thorium
    type Data: Clone;
    /// The serializable editing view (static fields marked, server defaults omitted)
    type View: Serialize + DeserializeOwned + From<Self::Data>;
    /// The update computed by diffing the edited view against the current entity
    type Update;
    /// The human-readable kind, used in log messages (e.g. "Image")
    const KIND: &'static str;
    /// The curated top-level key order for the editor view (see [`utils::curated_yaml`])
    const FIELD_ORDER: &'static [&'static str];

    /// Resolve the entity's group, finding it when one isn't supplied
    async fn resolve_group(
        thorium: &Thorium,
        name: &str,
        group: Option<&str>,
    ) -> Result<String, Error>;

    /// Fetch the current entity from Thorium
    async fn fetch(thorium: &Thorium, group: &str, name: &str) -> Result<Self::Data, Error>;

    /// Compute the update from the current entity and the editor-resolved view
    fn calculate_update(data: Self::Data, view: Self::View) -> Result<Option<Self::Update>, Error>;

    /// Apply the update in Thorium
    async fn send_update(
        thorium: &Thorium,
        group: &str,
        name: &str,
        update: &Self::Update,
    ) -> Result<(), Error>;
}

/// Edit an entity: resolve its group, fetch it, open its editing view in the
/// user's editor, then compute and apply any update.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config (for the default editor)
/// * `name` - The name of the entity to edit
/// * `group` - The group the entity is in, if known
/// * `editor_override` - An optional `--editor` override
pub async fn edit_entity<E: EditableEntity>(
    thorium: &Thorium,
    conf: &CtlConf,
    name: &str,
    group: Option<&str>,
    editor_override: Option<&str>,
) -> Result<(), Error> {
    let group = E::resolve_group(thorium, name, group).await?;
    let data = E::fetch(thorium, &group, name).await?;
    let view = E::View::from(data.clone());
    let yaml = utils::curated_yaml(&view, E::FIELD_ORDER)
        .map_err(|err| Error::new(format!("Failed to serialize {} to YAML: {err}", E::KIND)))?;
    let editor = resolve_editor(editor_override, conf);
    let label = format!("{group}-{name}");
    let resolved: E::View = match editor_loop(&yaml, &label, editor).await? {
        Some(resolved) => resolved,
        None => {
            println!("Cancelled.");
            return Ok(());
        }
    };
    match E::calculate_update(data, resolved)? {
        Some(update) => {
            E::send_update(thorium, &group, name, &update).await?;
            println!(
                "{} {} {}",
                E::KIND.bright_green(),
                format!("'{group}:{name}'").yellow(),
                "updated successfully!".bright_green()
            );
        }
        None => println!("No changes detected! Exiting..."),
    }
    Ok(())
}

/// [`EditableEntity`] for Thorium images
pub struct ImageEditable;

impl EditableEntity for ImageEditable {
    type Data = Image;
    type View = MergeableImage;
    type Update = ImageUpdate;
    const KIND: &'static str = "Image";
    const FIELD_ORDER: &'static [&'static str] = IMAGE_FIELD_ORDER;

    /// Resolve the image's group, searching for it when one isn't supplied
    async fn resolve_group(
        thorium: &Thorium,
        name: &str,
        group: Option<&str>,
    ) -> Result<String, Error> {
        match group {
            Some(group) => Ok(group.to_string()),
            None => utils::images::find_image_group(thorium, name).await,
        }
    }

    /// Fetch the current image from Thorium
    async fn fetch(thorium: &Thorium, group: &str, name: &str) -> Result<Image, Error> {
        thorium.images.get(group, name).await
    }

    /// Compute the image update from the current image and the edited view
    fn calculate_update(data: Image, view: MergeableImage) -> Result<Option<ImageUpdate>, Error> {
        update::calculate_image_update_from_mergeable(data, view)
    }

    /// Apply the image update in Thorium
    async fn send_update(
        thorium: &Thorium,
        group: &str,
        name: &str,
        update: &ImageUpdate,
    ) -> Result<(), Error> {
        thorium.images.update(group, name, update).await?;
        Ok(())
    }
}

/// [`EditableEntity`] for Thorium pipelines
pub struct PipelineEditable;

impl EditableEntity for PipelineEditable {
    type Data = Pipeline;
    type View = MergeablePipeline;
    type Update = PipelineUpdate;
    const KIND: &'static str = "Pipeline";
    const FIELD_ORDER: &'static [&'static str] = PIPELINE_FIELD_ORDER;

    /// Resolve the pipeline's group, searching for it when one isn't supplied
    async fn resolve_group(
        thorium: &Thorium,
        name: &str,
        group: Option<&str>,
    ) -> Result<String, Error> {
        match group {
            Some(group) => Ok(group.to_string()),
            None => utils::pipelines::find_pipeline_group(thorium, &name.to_string()).await,
        }
    }

    /// Fetch the current pipeline from Thorium
    async fn fetch(thorium: &Thorium, group: &str, name: &str) -> Result<Pipeline, Error> {
        thorium.pipelines.get(group, name).await
    }

    /// Compute the pipeline update from the current pipeline and the edited view
    fn calculate_update(
        data: Pipeline,
        view: MergeablePipeline,
    ) -> Result<Option<PipelineUpdate>, Error> {
        update::calculate_pipeline_update_from_mergeable(data, view)
    }

    /// Apply the pipeline update in Thorium
    async fn send_update(
        thorium: &Thorium,
        group: &str,
        name: &str,
        update: &PipelineUpdate,
    ) -> Result<(), Error> {
        thorium.pipelines.update(group, name, update).await?;
        Ok(())
    }
}
