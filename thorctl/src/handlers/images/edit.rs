//! Handles the image edit command

use colored::Colorize;
use owo_colors::OwoColorize;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thorium::models::{
    ChildFilters, Cleanup, Dependencies, Image, ImageArgs, ImageBan, ImageBanUpdate, ImageLifetime,
    ImageScaler, ImageUpdate, ImageVersion, Kvm, OutputCollection, OutputDisplayType,
    ResourcesUpdate, SecurityContext, SpawnLimits, Volume,
};
use thorium::{Error, Thorium};
use uuid::Uuid;

use crate::args::images::EditImage;
use crate::utils::diff;
use crate::{CtlConf, set_clear, set_modified, set_modified_opt, utils};

/// A Thorium [`Image`] modified and serialized such that it's easily editable
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditableImage {
    /// The group this image is in
    #[serde(rename = "*group*")]
    pub group: String,
    /// The name of this image
    #[serde(rename = "*name*")]
    pub name: String,
    /// The creator of this image
    #[serde(rename = "*creator*")]
    pub creator: String,
    /// The version of this image
    pub version: Option<ImageVersion>,
    /// What scaler is responsible for scaling this image
    pub scaler: ImageScaler,
    /// The image to use (url or tag)
    pub image: Option<String>,
    /// The lifetime of a pod
    pub lifetime: Option<ImageLifetime>,
    /// The timeout for individual jobs
    pub timeout: Option<u64>,
    /// The resources to required to spawn this image
    pub resources: ResourcesUpdate,
    /// The limit to use for how many workers of this image type can be spawned
    pub spawn_limit: SpawnLimits,
    /// The environment variables to set
    pub env: HashSet<String>,
    /// How long this image takes to execute on average in seconds (defaults to
    /// 10 minutes on image creation).
    pub runtime: f64,
    /// Any volumes to bind in to this container
    pub volumes: Vec<Volume>,
    /// The arguments to add to this images jobs
    pub args: ImageArgs,
    /// The path to the modifier folders for this image
    pub modifiers: Option<String>,
    /// The image description
    pub description: Option<String>,
    /// The security context for this image
    pub security_context: SecurityContext,
    /// Whether the agent should stream stdout/stderr back to Thorium
    pub collect_logs: bool,
    /// Whether this is a generator or not
    pub generator: bool,
    /// How to handle dependencies for this image
    pub dependencies: Dependencies,
    /// The type of display class to use in the UI for this images output
    pub display_type: OutputDisplayType,
    /// The settings for collecting results from this image
    pub output_collection: OutputCollection,
    /// Any regex filters to match on when uploading children
    pub child_filters: ChildFilters,
    /// The settings to use when cleaning up canceled jobs
    pub clean_up: Option<Cleanup>,
    /// The settings to use for Kvm jobs
    pub kvm: Option<Kvm>,
    /// A list of reasons an image is banned mapped by ban UUID;
    /// if the list has any bans, the image cannot be spawned
    pub bans: HashMap<Uuid, ImageBan>,
    /// A set of the names of network policies to apply to the image when it's spawned
    ///
    /// Only applies when scaled with K8's currently
    pub network_policies: HashSet<String>,
}

// implement PartialEq by hand to ignore uneditable fields (group, name, creator)
impl PartialEq for EditableImage {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.scaler == other.scaler
            && self.image == other.image
            && self.lifetime == other.lifetime
            && self.timeout == other.timeout
            && self.resources == other.resources
            && self.spawn_limit == other.spawn_limit
            && self.env == other.env
            && self.runtime == other.runtime
            && self.volumes == other.volumes
            && self.args == other.args
            && self.modifiers == other.modifiers
            && self.description == other.description
            && self.security_context == other.security_context
            && self.collect_logs == other.collect_logs
            && self.generator == other.generator
            && self.dependencies == other.dependencies
            && self.display_type == other.display_type
            && self.output_collection == other.output_collection
            && self.child_filters == other.child_filters
            && self.clean_up == other.clean_up
            && self.kvm == other.kvm
            && self.bans == other.bans
            && self.network_policies == other.network_policies
    }
}

impl From<Image> for EditableImage {
    fn from(image: Image) -> Self {
        EditableImage {
            group: image.group,
            name: image.name,
            creator: image.creator,
            version: image.version,
            scaler: image.scaler,
            image: image.image,
            lifetime: image.lifetime,
            timeout: image.timeout,
            resources: ResourcesUpdate {
                cpu: Some(format!("{}m", image.resources.cpu)),
                memory: Some(format!("{}Mi", image.resources.memory)),
                ephemeral_storage: Some(format!("{}Mi", image.resources.ephemeral_storage)),
                nvidia_gpu: Some(image.resources.nvidia_gpu),
                amd_gpu: Some(image.resources.amd_gpu),
            },
            spawn_limit: image.spawn_limit,
            env: image
                .env
                .into_iter()
                .map(|(key, value)| format!("{}={}", key, value.unwrap_or_default()))
                .collect(),
            runtime: image.runtime,
            volumes: image.volumes,
            args: image.args,
            modifiers: image.modifiers,
            description: image.description,
            security_context: image.security_context,
            collect_logs: image.collect_logs,
            generator: image.generator,
            dependencies: image.dependencies,
            display_type: image.display_type,
            output_collection: image.output_collection,
            child_filters: image.child_filters,
            clean_up: image.clean_up,
            kvm: image.kvm,
            bans: image.bans,
            network_policies: image.network_policies,
        }
    }
}

/// A single environment variable key/value pair
type Env = (String, Option<String>);

/// Try to parse a raw environment variable formatted `<KEY>=<VALUE>` to a
/// (String, Option<String>) that the Thorium API wants
///
/// # Arguments
///
/// * `raw_env` - The raw environment variable to parse
fn parse_env(raw_env: &str) -> Result<Env, Error> {
    let mut split = raw_env.split('=');
    let key = split.next();
    let value = split.next();
    match (key, value, split.next()) {
        (Some(key), None, None) => Ok((key.to_string(), None)),
        (Some(key), Some(value), None) => Ok((key.to_string(), Some(value.to_string()))),
        _ => Err(Error::new(format!(
            "Invalid environment variable '{raw_env}'! Environment variables must be formatted `<KEY>=<VALUE>`."
        ))),
    }
}

/// Calculate a bans update by diffing old and new bans
///
/// # Arguments
///
/// * `old_bans` - The map of old bans
/// * `new_bans` - The map of new bans
fn calculate_bans_update(
    mut old_bans: HashMap<Uuid, ImageBan>,
    mut new_bans: HashMap<Uuid, ImageBan>,
) -> Result<ImageBanUpdate, Error> {
    if old_bans == new_bans {
        // if nothing has changed, return a noop
        Ok(ImageBanUpdate::default())
    } else {
        // the bans removed are bans that are in the old but not in the new;
        // we're okay to just mutate 'old' because bans that we still have in
        // 'new' won't be removed, so we don't need to worry about "re-adding"
        let bans_removed = old_bans
            .extract_if(|key, _| !new_bans.contains_key(key))
            .map(|(key, _)| key)
            .collect();
        // the bans added are bans that are in the new but not in the old
        let bans_added = new_bans
            .extract_if(|key, _| !old_bans.contains_key(key))
            .map(|(_, value)| value)
            .collect();
        // if we have any bans left over, make sure they're all the same; otherwise return an error;
        // bans cannot be updated; they can only be added/removed
        if old_bans == new_bans {
            Ok(ImageBanUpdate {
                bans_added,
                bans_removed,
            })
        } else {
            Err(Error::new(
                "Invalid bans update! Bans cannot be updated, only added or removed. \
                If you want to modify an existing ban, create a new ban and remove the old one.",
            ))
        }
    }
}

/// Calculate an image update by diffing an image before and after
/// it's edited
///
/// # Arguments
///
/// * `image` - The original image
/// * `edited_image` - The image post-editing
fn calculate_update(
    image: EditableImage,
    edited_image: EditableImage,
) -> Result<ImageUpdate, Error> {
    let (add_volumes, remove_volumes) = if image.volumes == edited_image.volumes {
        (vec![], vec![])
    } else {
        let remove_volumes: Vec<String> = image
            .volumes
            .iter()
            .filter_map(|old_vol| {
                edited_image
                    .volumes
                    .iter()
                    .all(|new_vol| old_vol != new_vol)
                    .then_some(old_vol.name.clone())
            })
            .collect();
        let add_volumes = edited_image
            .volumes
            .into_iter()
            .filter(|new_vol| image.volumes.iter().all(|old_vol| new_vol != old_vol));
        (add_volumes.collect(), remove_volumes)
    };
    let (add_env, remove_env) = if image.env == edited_image.env {
        (HashMap::default(), vec![])
    } else {
        // calculate the environment variables to remove
        let remove_env = image
            .env
            // remove variables that are not in the new env
            .difference(&edited_image.env)
            .map(|old_env| parse_env(old_env))
            .collect::<Result<Vec<Env>, Error>>()?
            .into_iter()
            .map(|(key, _)| key);
        // calculate the environment variables to add
        let add_env = edited_image
            .env
            // add variables that are in the new env but not in the old one
            .difference(&image.env)
            .map(|new_env| parse_env(new_env))
            .collect::<Result<HashMap<String, Option<String>>, Error>>()?;
        (add_env, remove_env.collect())
    };
    Ok(ImageUpdate {
        // seems to be unused?
        external: None,
        // needs template
        scaler: set_modified!(image.scaler, edited_image.scaler),
        timeout: set_modified_opt!(image.timeout, edited_image.timeout),
        // needs template for millicpu and storage
        resources: set_modified!(image.resources, edited_image.resources),
        // needs template
        spawn_limit: set_modified!(image.spawn_limit, edited_image.spawn_limit),
        add_volumes,
        remove_volumes,
        // needs template
        add_env,
        remove_env,
        clear_version: set_clear!(image.version, edited_image.version),
        // needs template for semver
        version: set_modified_opt!(image.version, edited_image.version),
        clear_image: set_clear!(image.image, edited_image.image),
        image: set_modified_opt!(image.image, edited_image.image),
        // needs template
        clear_lifetime: set_clear!(image.lifetime, edited_image.lifetime),
        lifetime: set_modified_opt!(image.lifetime, edited_image.lifetime),
        clear_description: set_clear!(image.description, edited_image.description),
        args: diff::images::calculate_image_args_update(image.args, edited_image.args),
        modifiers: set_modified_opt!(image.modifiers, edited_image.modifiers),
        description: set_modified_opt!(image.description, edited_image.description),
        security_context: diff::images::calculate_security_context_update(
            image.security_context,
            edited_image.security_context,
        ),
        collect_logs: set_modified!(image.collect_logs, edited_image.collect_logs),
        generator: set_modified!(image.generator, edited_image.generator),
        // needs template
        dependencies: diff::images::calculate_dependencies_update(
            image.dependencies,
            edited_image.dependencies,
        ),
        // needs template
        display_type: set_modified!(image.display_type, edited_image.display_type),
        output_collection: diff::images::calculate_output_collection_update(
            image.output_collection,
            edited_image.output_collection,
        ),
        child_filters: diff::images::calculate_child_filters_update(
            image.child_filters,
            edited_image.child_filters,
        ),
        clean_up: diff::images::calculate_clean_up_update(image.clean_up, edited_image.clean_up),
        kvm: diff::images::calculate_kvm_update(image.kvm, edited_image.kvm),
        bans: calculate_bans_update(image.bans, edited_image.bans)?,
        network_policies: diff::images::calculate_network_policies_update(
            image.network_policies,
            edited_image.network_policies,
        ),
    })
}

/// Delete the temporary file before returning an error
macro_rules! err_del_temp {
    ($func:expr, $temp_path:expr) => {
        $func.map_err(|err| {
            // wrap error in our error type
            let err = Error::from(err);
            if let Err(del_err) = del_temp!($temp_path) {
                // we failed to delete the temp file so log the main error and
                // bubble up the delete error so the errors are printed in the
                // correct order
                eprintln!("{err}");
                del_err
            } else {
                err
            }
        })
    };
}

/// Try to delete the temporary file or return an error if we couldn't
macro_rules! del_temp {
    ($temp_path:expr) => {
        std::fs::remove_file($temp_path)
            .map_err(|err| Error::new(format!("Failed to remove temporary image file: {err}")))
    };
}

/// Edit an image using a text editor, detect the updates, then update the image
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl conf
/// * `cmd` - The edit image command that was run
pub async fn edit(thorium: Thorium, conf: &CtlConf, cmd: &EditImage) -> Result<(), Error> {
    let group = if let Some(group) = &cmd.group {
        group.clone()
    } else {
        // find the image's group if we weren't given one
        utils::images::find_image_group(&thorium, &cmd.image).await?
    };
    // get the image we want to edit
    let image = thorium.images.get(&group, &cmd.image).await?;
    // convert the image to something easier to edit
    let image = EditableImage::from(image);
    // create a temp directory
    let temp_dir = std::env::temp_dir().join("thorium");
    std::fs::create_dir_all(&temp_dir).map_err(|err| {
        Error::new(format!(
            "Failed to create temporary directory '{}': {}",
            temp_dir.to_string_lossy(),
            err
        ))
    })?;
    // serialize the image's data to a temporary file
    let temp_path = temp_dir.join(format!("image-{}.yml", Uuid::new_v4()));
    let mut temp_file = std::fs::File::create(&temp_path).map_err(|err| {
        Error::new(format!(
            "Failed to create temporary image file to edit at '{}': {}",
            temp_path.to_string_lossy(),
            err
        ))
    })?;
    err_del_temp!(serde_yaml::to_writer(&mut temp_file, &image), &temp_path)?;
    // drop the file descriptor
    drop(temp_file);
    // open the file to edit it
    let editor = cmd.editor.as_ref().unwrap_or(&conf.default_editor);
    let status = err_del_temp!(
        std::process::Command::new(editor)
            .arg(&temp_path)
            .status()
            .map_err(|err| Error::new(format!("Unable to open editor '{editor}': {err}"))),
        &temp_path
    )?;
    if !status.success() {
        match status.code() {
            Some(code) => {
                return err_del_temp!(
                    Err(Error::new(format!(
                        "Editor '{editor}' exited with error code: {code}"
                    ))),
                    &temp_path
                );
            }
            None => {
                return err_del_temp!(
                    Err(Error::new(format!("Editor '{editor}' exited with error!"))),
                    &temp_path
                );
            }
        }
    }
    // deserialize the file to the now edited image
    let edited_image_file = err_del_temp!(std::fs::File::open(&temp_path), &temp_path)?;
    let edited_image: EditableImage =
        err_del_temp!(serde_yaml::from_reader(&edited_image_file), &temp_path)?;
    // check if there were no changes
    if edited_image == image {
        // if no changes were found, delete the file and exit early
        println!("No changes detected! Exiting...");
        del_temp!(&temp_path)?;
        return Ok(());
    }
    let image_update = err_del_temp!(calculate_update(image, edited_image), &temp_path)?;
    err_del_temp!(
        thorium
            .images
            .update(&group, &cmd.image, &image_update)
            .await,
        &temp_path
    )?;
    println!(
        "{} {} {}",
        "Image".bright_green(),
        format!("'{}:{}'", group, cmd.image).yellow(),
        "updated successfully! ✅".bright_green()
    );
    // remove the temporary file
    del_temp!(&temp_path)?;
    Ok(())
}
