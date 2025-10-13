//! Handlers for updating a toolbox already imported to Thorium

use colored::Colorize;
use futures::{StreamExt, TryStreamExt, stream};
use http::StatusCode;
use std::collections::HashSet;
use std::fmt;
use thorium::models::ScrubbedUser;
use thorium::{CtlConf, Error, Thorium};

use crate::args::toolbox::UpdateToolbox;
use crate::handlers::progress::{Bar, BarKind};
use crate::handlers::toolbox::manifest::{ImageVersion, PipelineVersion, ToolboxManifest};
use crate::handlers::toolbox::shared;

mod images;
mod pipelines;

use images::{ToolboxImageUpdate, ToolboxImageUpdateOp};
use pipelines::{ToolboxPipelineUpdate, ToolboxPipelineUpdateOp};

/// Concurrently calculate updates for items in a manifest using the given
/// helper function
macro_rules! calculate_updates {
    ($manifest_items:expr, $helper_fn:path, $thorium:expr) => {
        async {
            stream::iter($manifest_items.into_iter().flat_map(|(name, manifest)| {
                manifest
                    .versions
                    .into_iter()
                    .map(move |(version_name, version)| {
                        (name.clone(), version_name.clone(), version)
                    })
            }))
            .map(|(name, version_name, version)| $helper_fn($thorium, name, version_name, version))
            .buffer_unordered(10)
            .collect::<Vec<Result<_, _>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
        }
    };
}

/// All the updates that need to be performed in Thorium
/// to update a toolbox
#[derive(Debug)]
struct ToolboxUpdate {
    /// The image updates that need to be done
    images: Vec<ToolboxImageUpdate>,
    /// The pipeline updates that need to be done
    pipelines: Vec<ToolboxPipelineUpdate>,
    /// The groups that need to be created
    groups: Vec<String>,
}

impl ToolboxUpdate {
    /// Based on a image in the manifest and the state in Thorium,
    /// calculate what needs to be done to update the image in Thorium
    /// to match the manifest
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `image_name` - The name of the image in the manifest
    /// * `image_version_name` - The name of the image's version in the manifest
    /// * `image_version` - The manifest for the image version
    async fn calculate_image_update(
        thorium: &Thorium,
        image_name: String,
        image_version_name: String,
        image_version: ImageVersion,
    ) -> Result<ToolboxImageUpdate, Error> {
        match thorium
            .images
            .get(&image_version.config.group, &image_version.config.name)
            .await
        {
            Ok(image) => {
                let group = image_version.config.group.clone();
                match images::calculate_image_update(image, image_version.config).map_err(
                    |err| {
                        Error::new(format!(
                            "Error diffing image '{}:{}': {}",
                            image_name,
                            image_version_name,
                            err.msg().unwrap_or_default()
                        ))
                    },
                )? {
                    Some(image_update) => Ok(ToolboxImageUpdate::new(
                        image_name,
                        image_version_name,
                        group,
                        ToolboxImageUpdateOp::Update(Box::new(image_update)),
                    )),
                    None => Ok(ToolboxImageUpdate::new(
                        image_name,
                        image_version_name,
                        group,
                        ToolboxImageUpdateOp::Unchanged,
                    )),
                }
            }
            Err(err) => {
                if err
                    .status()
                    .is_some_and(|status| status == StatusCode::NOT_FOUND)
                {
                    Ok(ToolboxImageUpdate::new(
                        image_name,
                        image_version_name,
                        image_version.config.group.clone(),
                        ToolboxImageUpdateOp::Create(Box::new(image_version.config)),
                    ))
                } else {
                    Err(Error::new(format!(
                        "Error diffing image '{image_name}:{image_version_name}': {err}"
                    )))
                }
            }
        }
    }

    /// Based on a pipeline in the manifest and the state in Thorium,
    /// calculate what needs to be done to update the pipeline in Thorium
    /// to match the manifest
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `pipeline_name` - The name of the pipeline in the manifest
    /// * `pipeline_version_name` - The name of the pipeline's version in the manifest
    /// * `pipeline_version` - The manifest for the pipeline version
    async fn calculate_pipeline_update(
        thorium: &Thorium,
        pipeline_name: String,
        pipeline_version_name: String,
        pipeline_version: PipelineVersion,
    ) -> Result<ToolboxPipelineUpdate, Error> {
        match thorium
            .pipelines
            .get(
                &pipeline_version.config.group,
                &pipeline_version.config.name,
            )
            .await
        {
            Ok(pipeline) => {
                let group = pipeline_version.config.group.clone();
                match pipelines::calculate_pipeline_update(pipeline, pipeline_version.config) {
                    Some(pipeline_update) => Ok(ToolboxPipelineUpdate::new(
                        pipeline_name,
                        pipeline_version_name,
                        group,
                        ToolboxPipelineUpdateOp::Update(Box::new(pipeline_update)),
                    )),
                    None => Ok(ToolboxPipelineUpdate::new(
                        pipeline_name,
                        pipeline_version_name,
                        group,
                        ToolboxPipelineUpdateOp::Unchanged,
                    )),
                }
            }
            Err(err) => {
                if err
                    .status()
                    .is_some_and(|status| status == StatusCode::NOT_FOUND)
                {
                    Ok(ToolboxPipelineUpdate::new(
                        pipeline_name,
                        pipeline_version_name,
                        pipeline_version.config.group.clone(),
                        ToolboxPipelineUpdateOp::Create(Box::new(pipeline_version.config)),
                    ))
                } else {
                    Err(Error::new(format!(
                        "Error diffing pipeline '{pipeline_name}:{pipeline_version_name}': {err}"
                    )))
                }
            }
        }
    }

    /// Based on a new toolbox manifest and the state in Thorium,
    /// calculate what needs to be done to update Thorium
    /// to match the manifest
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `manifest` - The toolbox manifest we're updating with
    /// * `manifest_groups` - All the groups the manifest expects, previously calculated
    async fn calculate(
        thorium: &Thorium,
        manifest: ToolboxManifest,
        manifest_groups: HashSet<String>,
    ) -> Result<Self, Error> {
        // calculate updates for images
        let images =
            calculate_updates!(manifest.images, Self::calculate_image_update, thorium).await?;
        // calculate updates for pipelines
        let pipelines =
            calculate_updates!(manifest.pipelines, Self::calculate_pipeline_update, thorium)
                .await?;
        // see which groups are missing in Thorium
        let groups = shared::get_missing_groups(thorium, manifest_groups)
            .await
            .map_err(|err| {
                Error::new(format!("Error retrieving missing groups in Thorium: {err}"))
            })?;
        Ok(Self {
            images,
            pipelines,
            groups,
        })
    }

    /// Returns true if the toolbox update has no updates
    fn is_unchanged(&self) -> bool {
        self.images
            .iter()
            .all(|update_image| matches!(update_image.op, ToolboxImageUpdateOp::Unchanged))
            && self.pipelines.iter().all(|update_pipeline| {
                matches!(update_pipeline.op, ToolboxPipelineUpdateOp::Unchanged)
            })
            && self.groups.is_empty()
    }

    /// Have the user confirm the update
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorctl conf
    /// * `current_user` - The user updating the toolbox
    fn confirm(&self, conf: &CtlConf, current_user: &ScrubbedUser) -> Result<bool, Error> {
        // print out the update to stdout
        println!("{self}\n");
        // confirm with the user that they want to import
        let response = dialoguer::Confirm::new()
            .with_prompt(format!(
                "Perform the above updates to Thorium instance at '{}' as user '{}'?",
                conf.keys.api.bright_green(),
                current_user.username.bright_green()
            ))
            .interact()?;
        Ok(response)
    }

    /// Apply image updates
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `progress` - The progress bar
    async fn apply_images(&mut self, thorium: &Thorium, progress: &Bar) -> Result<(), Error> {
        // remove all the noops from the image updates
        self.images
            .retain(|update_image| !matches!(update_image.op, ToolboxImageUpdateOp::Unchanged));
        // if we still have updates, proceed
        if !self.images.is_empty() {
            progress.refresh("Updating images", BarKind::Bound(self.images.len() as u64));
            stream::iter(self.images.drain(..))
                .map(Ok::<_, Error>)
                .try_for_each_concurrent(10, |update_image| async move {
                    match update_image.op {
                        ToolboxImageUpdateOp::Create(image_request) => {
                            thorium.images.create(&image_request).await.map_err(|err| {
                                Error::new(format!(
                                    "Error creating image '{}:{}': {}",
                                    update_image.image_name, update_image.image_version, err
                                ))
                            })?;
                        }
                        ToolboxImageUpdateOp::Update(image_update) => {
                            thorium
                                .images
                                .update(
                                    &update_image.group,
                                    &update_image.image_name,
                                    &image_update,
                                )
                                .await
                                .map_err(|err| {
                                    Error::new(format!(
                                        "Error updating image '{}:{}': {}",
                                        update_image.image_name, update_image.image_version, err
                                    ))
                                })?;
                        }
                        ToolboxImageUpdateOp::Unchanged => (),
                    }
                    Ok(())
                })
                .await?;
        }
        Ok(())
    }

    /// Apply pipeline updates
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `progress` - The progress bar
    async fn apply_pipelines(&mut self, thorium: &Thorium, progress: &Bar) -> Result<(), Error> {
        // remove all the noops from the pipeline updates
        self.pipelines.retain(|update_pipeline| {
            !matches!(update_pipeline.op, ToolboxPipelineUpdateOp::Unchanged)
        });
        // if we still have updates, proceed
        if !self.pipelines.is_empty() {
            progress.refresh(
                "Updating pipelines",
                BarKind::Bound(self.pipelines.len() as u64),
            );
            stream::iter(self.pipelines.drain(..))
                .map(Ok::<_, Error>)
                .try_for_each_concurrent(10, |update_pipeline| async move {
                    match update_pipeline.op {
                        ToolboxPipelineUpdateOp::Create(pipeline_request) => {
                            thorium
                                .pipelines
                                .create(&pipeline_request)
                                .await
                                .map_err(|err| {
                                    Error::new(format!(
                                        "Error creating pipeline '{}:{}': {}",
                                        update_pipeline.pipeline_name,
                                        update_pipeline.pipeline_version,
                                        err
                                    ))
                                })?;
                        }
                        ToolboxPipelineUpdateOp::Update(pipeline_update) => {
                            thorium
                                .pipelines
                                .update(
                                    &update_pipeline.group,
                                    &update_pipeline.pipeline_name,
                                    &pipeline_update,
                                )
                                .await
                                .map_err(|err| {
                                    Error::new(format!(
                                        "Error updating pipeline '{}:{}': {}",
                                        update_pipeline.pipeline_name,
                                        update_pipeline.pipeline_version,
                                        err
                                    ))
                                })?;
                        }
                        ToolboxPipelineUpdateOp::Unchanged => (),
                    }
                    Ok(())
                })
                .await?;
        }
        Ok(())
    }

    /// Apply updates to Thorium
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    /// * `progress` - The progress bar
    async fn apply(mut self, thorium: &Thorium, progress: &Bar) -> Result<(), Error> {
        // first create groups
        if !self.groups.is_empty() {
            progress.refresh("Creating groups", BarKind::Bound(self.groups.len() as u64));
            let groups = std::mem::take(&mut self.groups);
            shared::create_groups(thorium, groups, progress).await?;
        }
        // apply image updates
        self.apply_images(thorium, progress).await?;
        // apply pipeline updates
        self.apply_pipelines(thorium, progress).await?;
        Ok(())
    }
}

impl fmt::Display for ToolboxUpdate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // separate image updates into creates and updates
        let (new_images, updated_images): (Vec<&ToolboxImageUpdate>, Vec<&ToolboxImageUpdate>) =
            self.images
                .iter()
                // filter out noops
                .filter(|update| !matches!(update.op, ToolboxImageUpdateOp::Unchanged))
                .partition(|update| matches!(update.op, ToolboxImageUpdateOp::Create(_)));
        // separate pipeline updates into creates and updates
        let (new_pipelines, updated_pipelines): (
            Vec<&ToolboxPipelineUpdate>,
            Vec<&ToolboxPipelineUpdate>,
        ) = self
            .pipelines
            .iter()
            .filter(|update| !matches!(update.op, ToolboxPipelineUpdateOp::Unchanged))
            .partition(|update| matches!(update.op, ToolboxPipelineUpdateOp::Create(_)));
        // display images
        if !new_images.is_empty() || !updated_images.is_empty() {
            writeln!(f, "{}", "Images:".bright_blue().bold())?;
        }
        if !new_images.is_empty() {
            writeln!(f, "  {}", "New Images:".bright_green())?;
            for image in new_images {
                writeln!(
                    f,
                    "    {}:{}",
                    image.image_name.bright_green(),
                    image.image_version.bright_green()
                )?;
            }
        }
        if !updated_images.is_empty() {
            writeln!(f, "  {}", "Updated Images:".bright_yellow())?;
            for image in updated_images {
                writeln!(
                    f,
                    "    {}:{}",
                    image.image_name.bright_yellow(),
                    image.image_version.bright_yellow()
                )?;
            }
        }
        // display pipelines
        if !new_pipelines.is_empty() || !updated_pipelines.is_empty() {
            writeln!(f, "{}", "Pipelines:".bright_blue().bold())?;
        }
        if !new_pipelines.is_empty() {
            writeln!(f, "  {}", "New Pipelines:".bright_green())?;
            for pipeline in new_pipelines {
                writeln!(
                    f,
                    "    {}:{}",
                    pipeline.pipeline_name.bright_green(),
                    pipeline.pipeline_version.bright_green()
                )?;
            }
        }
        if !updated_pipelines.is_empty() {
            writeln!(f, "  {}", "Updated Pipelines:".bright_yellow())?;
            for pipeline in updated_pipelines {
                writeln!(
                    f,
                    "    {}:{}",
                    pipeline.pipeline_name.bright_yellow(),
                    pipeline.pipeline_version.bright_yellow()
                )?;
            }
        }
        // display groups
        if !self.groups.is_empty() {
            writeln!(f, "{}", "Groups:".bright_blue().bold())?;
            writeln!(f, "  {}", "New Groups:".bright_green())?;
            for group in &self.groups {
                writeln!(f, "    {}", group.bright_green())?;
            }
        }
        Ok(())
    }
}

/// Update a toolbox already imported into Thorium by updating the tool/pipeline configurations
/// based on the newer manifest
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `cmd` - The toolbox update command that was run
pub async fn update(thorium: Thorium, conf: CtlConf, cmd: &UpdateToolbox) -> Result<(), Error> {
    // get the new manifest at the given location as well as a progress bar
    let (mut manifest, progress) = shared::get_manifest(&cmd.manifest).await?;
    // validate the manifest
    if let Err(err) = manifest.validate() {
        return Err(Error::new(format!(
            "Invalid toolbox manifest: {}",
            err.msg().unwrap_or_else(|| "Unknown error".to_string())
        )));
    }
    // get all the groups the manifest expects to exist, overriding them if we're set to
    let manifest_groups = if let Some(group_override) = &cmd.group_override {
        progress.info_anonymous(format!(
            "Overriding all image/pipeline update groups to '{}'",
            group_override.bright_yellow()
        ));
        // replace all groups in the manifest with the override and get the modified manifest
        manifest = manifest.override_group(group_override);
        // validate the manifest again after overriding groups
        if let Err(err) = manifest.validate() {
            return Err(Error::new(format!(
                "Invalid toolbox manifest after group override '{}': {}",
                group_override.bright_yellow(),
                err.msg().unwrap_or_else(|| "Unknown error".to_string())
            )));
        }
        // return a set with just our group override since we replaced it
        HashSet::from([group_override.to_string()])
    } else {
        // get all of the groups the manifest refers to
        manifest.groups()
    };
    // calculate what needs to be done to update based on Thorium's current state
    progress.refresh("Calculating updates", BarKind::Timer);
    let update = ToolboxUpdate::calculate(&thorium, manifest, manifest_groups).await?;
    if update.is_unchanged() {
        // exit early if the update has nothing to do
        progress.finish_with_message("No update needed!");
        return Ok(());
    }
    if !cmd.skip_confirm {
        // get info on the current user
        let current_user = thorium
            .users
            .info()
            .await
            .map_err(|err| Error::new(format!("Error getting current user info: {err}")))?;
        // confirm with the user to proceed and suspend the progress bar while we do so
        let confirm = progress.suspend(|| update.confirm(&conf, &current_user))?;
        if !confirm {
            // exit early if the user didn't confirm
            return Ok(());
        }
    }
    // apply the update
    update.apply(&thorium, &progress).await?;
    // inform the user the update is complete
    progress.refresh("Update complete!", BarKind::Timer);
    progress.finish();
    Ok(())
}
