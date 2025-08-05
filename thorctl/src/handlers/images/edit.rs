//! Handles the image edit command

use colored::Colorize;
use owo_colors::OwoColorize;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thorium::models::{
    AutoTag, AutoTagUpdate, ChildFilters, ChildFiltersUpdate, ChildrenDependencySettings,
    ChildrenDependencySettingsUpdate, Cleanup, CleanupUpdate, Dependencies, DependenciesUpdate,
    DependencySettingsUpdate, EphemeralDependencySettings, EphemeralDependencySettingsUpdate,
    FilesHandler, FilesHandlerUpdate, Image, ImageArgs, ImageArgsUpdate, ImageBan, ImageBanUpdate,
    ImageLifetime, ImageNetworkPolicyUpdate, ImageScaler, ImageUpdate, ImageVersion, Kvm,
    KvmUpdate, OutputCollection, OutputCollectionUpdate, OutputDisplayType, RepoDependencySettings,
    ResourcesUpdate, ResultDependencySettings, ResultDependencySettingsUpdate,
    SampleDependencySettings, SecurityContext, SecurityContextUpdate, SpawnLimits,
    TagDependencySettings, TagDependencySettingsUpdate, Volume,
};
use thorium::{Error, Thorium};
use uuid::Uuid;

use crate::args::images::EditImage;
use crate::{utils, CtlConf};

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
    /// Any volumes to bind into this container
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

/// Set this field in the update if it was modified
macro_rules! set_modified {
    ($image_field:expr, $edited_image_field:expr) => {
        ($image_field != $edited_image_field).then_some($edited_image_field)
    };
}

/// Set this optional field in the update if it was modified
macro_rules! set_modified_opt {
    ($image_field:expr, $edited_image_field:expr) => {
        ($image_field != $edited_image_field)
            .then_some($edited_image_field)
            .flatten()
    };
}

/// Clear this field if it was some in the image and was set to none
macro_rules! set_clear {
    ($image_field:expr, $edited_image_field:expr) => {
        $image_field.is_some() && $edited_image_field.is_none()
    };
}

/// Clear this field if it was not empty in the image but it was set to empty
macro_rules! set_clear_vec {
    ($image_field:expr, $edited_image_field:expr) => {
        !$image_field.is_empty() && $edited_image_field.is_empty()
    };
}

/// Calculate the values to remove/add based on what's in the
/// old vec and the new vec
///
/// Returns collections of values to remove/add in a tuple: `(remove, add)`
///
/// We remove everything that's in the old but not in the new and
/// add everything that's in the new but not in the old. We're okay
/// to mutate `old` with `extract_if` before calculating what we add because
/// `new` will contain all the values we want to keep; nothing we
/// extract from `old` in the first step will affect the calculation of
/// what we want to add.
///
/// # Limitations
///
/// - Cannot add/remove duplicate values
macro_rules! calc_remove_add_vec {
    ($old:expr, $new:expr) => {
        (
            $old.extract_if(.., |old| !$new.contains(old)).collect(),
            $new.extract_if(.., |new| !$old.contains(new)).collect(),
        )
    };
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
        _ => Err(Error::new(format!("Invalid environment variable '{raw_env}'! Environment variables must be formatted `<KEY>=<VALUE>`.")))
    }
}

/// Calculate an image args update by diffing old and
/// new image args settings
///
/// # Arguments
///
/// * `old_args` - The old args settings
/// * `new_args` - The new args settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_image_args_update(
    old_args: ImageArgs,
    new_args: ImageArgs,
) -> Option<ImageArgsUpdate> {
    if old_args == new_args {
        None
    } else {
        Some(ImageArgsUpdate {
            clear_entrypoint: set_clear!(old_args.entrypoint, new_args.entrypoint),
            entrypoint: set_modified_opt!(old_args.entrypoint, new_args.entrypoint),
            clear_command: set_clear!(old_args.command, new_args.command),
            command: set_modified_opt!(old_args.command, new_args.command),
            clear_reaction: set_clear!(old_args.reaction, new_args.reaction),
            reaction: set_modified_opt!(old_args.reaction, new_args.reaction),
            clear_repo: set_clear!(old_args.repo, new_args.repo),
            repo: set_modified_opt!(old_args.repo, new_args.repo),
            clear_commit: set_clear!(old_args.commit, new_args.commit),
            commit: set_modified_opt!(old_args.commit, new_args.commit),
            // TODO: template
            output: set_modified!(old_args.output, new_args.output),
        })
    }
}

/// Calculate a security context update by diffing old and
/// new security context settings
///
/// # Arguments
///
/// * `old_security_context` - The old security context settings
/// * `new_security_context` - The new security context settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_security_context_update(
    old_security_context: SecurityContext,
    new_security_context: SecurityContext,
) -> Option<SecurityContextUpdate> {
    if old_security_context == new_security_context {
        None
    } else {
        Some(SecurityContextUpdate {
            clear_user: set_clear!(old_security_context.user, new_security_context.user),
            user: set_modified_opt!(old_security_context.user, new_security_context.user),
            clear_group: set_clear!(old_security_context.group, new_security_context.group),
            group: set_modified_opt!(old_security_context.group, new_security_context.group),
            allow_privilege_escalation: set_modified!(
                old_security_context.allow_privilege_escalation,
                new_security_context.allow_privilege_escalation
            ),
        })
    }
}

/// Calculate a sample dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_sample_dependencies_update(
    old_dependencies: SampleDependencySettings,
    new_dependencies: SampleDependencySettings,
) -> DependencySettingsUpdate {
    DependencySettingsUpdate {
        location: set_modified!(old_dependencies.location, new_dependencies.location),
        clear_kwarg: set_clear!(old_dependencies.kwarg, new_dependencies.kwarg),
        kwarg: set_modified_opt!(old_dependencies.kwarg, new_dependencies.kwarg),
        strategy: set_modified!(old_dependencies.strategy, new_dependencies.strategy),
    }
}

/// Calculate a ephemeral dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_ephemeral_dependencies_update(
    mut old_dependencies: EphemeralDependencySettings,
    mut new_dependencies: EphemeralDependencySettings,
) -> EphemeralDependencySettingsUpdate {
    // calculate which names to remove/add
    let (remove_names, add_names) =
        calc_remove_add_vec!(old_dependencies.names, new_dependencies.names);
    EphemeralDependencySettingsUpdate {
        location: set_modified!(old_dependencies.location, new_dependencies.location),
        clear_kwarg: set_clear!(old_dependencies.kwarg, new_dependencies.kwarg),
        kwarg: set_modified_opt!(old_dependencies.kwarg, new_dependencies.kwarg),
        strategy: set_modified!(old_dependencies.strategy, new_dependencies.strategy),
        remove_names,
        add_names,
    }
}

/// Calculate a results dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_results_dependencies_update(
    mut old_dependencies: ResultDependencySettings,
    mut new_dependencies: ResultDependencySettings,
) -> ResultDependencySettingsUpdate {
    // calculate which images to remove/add
    let (remove_images, add_images) =
        calc_remove_add_vec!(old_dependencies.images, new_dependencies.images);
    // calculate which names to remove/add
    let (remove_names, add_names) =
        calc_remove_add_vec!(old_dependencies.names, new_dependencies.names);
    ResultDependencySettingsUpdate {
        // remove images that are in the old but not in the new
        remove_images,
        // add images that are in the new but not in the old
        add_images,
        location: set_modified!(old_dependencies.location, new_dependencies.location),
        // TODO: template
        kwarg: set_modified!(old_dependencies.kwarg, new_dependencies.kwarg),
        strategy: set_modified!(old_dependencies.strategy, new_dependencies.strategy),
        remove_names,
        add_names,
    }
}

/// Calculate a repo dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_repo_dependencies_update(
    old_dependencies: RepoDependencySettings,
    new_dependencies: RepoDependencySettings,
) -> DependencySettingsUpdate {
    DependencySettingsUpdate {
        location: set_modified!(old_dependencies.location, new_dependencies.location),
        clear_kwarg: set_clear!(old_dependencies.kwarg, new_dependencies.kwarg),
        kwarg: set_modified_opt!(old_dependencies.kwarg, new_dependencies.kwarg),
        strategy: set_modified!(old_dependencies.strategy, new_dependencies.strategy),
    }
}

/// Calculate a tags dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_tags_dependencies_update(
    old_dependencies: TagDependencySettings,
    new_dependencies: TagDependencySettings,
) -> TagDependencySettingsUpdate {
    TagDependencySettingsUpdate {
        enabled: set_modified!(old_dependencies.enabled, new_dependencies.enabled),
        location: set_modified!(old_dependencies.location, new_dependencies.location),
        clear_kwarg: set_clear!(old_dependencies.kwarg, new_dependencies.kwarg),
        kwarg: set_modified_opt!(old_dependencies.kwarg, new_dependencies.kwarg),
        strategy: set_modified!(old_dependencies.strategy, new_dependencies.strategy),
    }
}

/// Calculate a children dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_child_dependencies_update(
    mut old_dependencies: ChildrenDependencySettings,
    mut new_dependencies: ChildrenDependencySettings,
) -> ChildrenDependencySettingsUpdate {
    // calculate which images to remove/add
    let (remove_images, add_images) =
        calc_remove_add_vec!(old_dependencies.images, new_dependencies.images);
    ChildrenDependencySettingsUpdate {
        enabled: set_modified!(old_dependencies.enabled, new_dependencies.enabled),
        remove_images,
        add_images,
        location: set_modified!(old_dependencies.location, new_dependencies.location),
        clear_kwarg: set_clear!(old_dependencies.kwarg, new_dependencies.kwarg),
        kwarg: set_modified_opt!(old_dependencies.kwarg, new_dependencies.kwarg),
        strategy: set_modified!(old_dependencies.strategy, new_dependencies.strategy),
    }
}

/// Calculate a dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old_dependencies` - The old dependencies settings
/// * `new_dependencies` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_dependencies_update(
    old_dependencies: Dependencies,
    new_dependencies: Dependencies,
) -> DependenciesUpdate {
    DependenciesUpdate {
        samples: calculate_sample_dependencies_update(
            old_dependencies.samples,
            new_dependencies.samples,
        ),
        ephemeral: calculate_ephemeral_dependencies_update(
            old_dependencies.ephemeral,
            new_dependencies.ephemeral,
        ),
        results: calculate_results_dependencies_update(
            old_dependencies.results,
            new_dependencies.results,
        ),
        repos: calculate_repo_dependencies_update(old_dependencies.repos, new_dependencies.repos),
        tags: calculate_tags_dependencies_update(old_dependencies.tags, new_dependencies.tags),
        children: calculate_child_dependencies_update(
            old_dependencies.children,
            new_dependencies.children,
        ),
    }
}

/// Calculate a clean up update by diffing old and
/// new clean up settings
///
/// # Arguments
///
/// * `old_clean_up` - The old clean up settings
/// * `new_clean_up` - The new clean up settings
fn calculate_clean_up_update(
    old_clean_up: Option<Cleanup>,
    new_clean_up: Option<Cleanup>,
) -> CleanupUpdate {
    match (old_clean_up, new_clean_up) {
        // nothing changed, so we return a noop
        (None, None) => CleanupUpdate::default(),
        // we made a whole new cleanup so update all fields
        (None, Some(new_clean_up)) => CleanupUpdate {
            job_id: Some(new_clean_up.job_id),
            results: Some(new_clean_up.results),
            result_files_dir: Some(new_clean_up.result_files_dir),
            script: Some(new_clean_up.script),
            clear: false,
        },
        // we had some and now we have none, so clear it
        (Some(_), None) => CleanupUpdate::default().clear(),
        // both are some, so we need to compare each field and update as needed
        (Some(old_clean_up), Some(new_clean_up)) => CleanupUpdate {
            job_id: set_modified!(old_clean_up.job_id, new_clean_up.job_id),
            results: set_modified!(old_clean_up.results, new_clean_up.results),
            result_files_dir: set_modified!(
                old_clean_up.result_files_dir,
                new_clean_up.result_files_dir
            ),
            script: set_modified!(old_clean_up.script, new_clean_up.script),
            clear: false,
        },
    }
}

/// Calculate a files handler update by diffing old and
/// new files handler settings
///
/// # Arguments
///
/// * `old_files_handler` - The old files handler settings
/// * `new_files_handler` - The new files handler settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_files_handler_update(
    mut old_files_handler: FilesHandler,
    mut new_files_handler: FilesHandler,
) -> FilesHandlerUpdate {
    if old_files_handler == new_files_handler {
        // if there were no changes, return a noop
        FilesHandlerUpdate::default()
    } else {
        let (remove_names, add_names) =
            calc_remove_add_vec!(old_files_handler.names, new_files_handler.names);
        FilesHandlerUpdate {
            results: set_modified!(old_files_handler.results, new_files_handler.results),
            result_files: set_modified!(
                old_files_handler.result_files,
                new_files_handler.result_files
            ),
            tags: set_modified!(old_files_handler.tags, new_files_handler.tags),
            clear_names: set_clear_vec!(old_files_handler.names, new_files_handler.names),
            remove_names,
            add_names,
        }
    }
}

/// Calculate an auto tag update by diffing old and
/// new auto tag settings
///
/// # Arguments
///
/// * `old_auto_tag` - The old auto tag settings
/// * `new_auto_tag` - The new auto tag settings
fn calculate_auto_tag_updates(
    mut old_auto_tag: HashMap<String, AutoTag>,
    mut new_auto_tag: HashMap<String, AutoTag>,
) -> HashMap<String, AutoTagUpdate> {
    if old_auto_tag == new_auto_tag {
        // the auto tag settings were unchanged so return a noop
        HashMap::default()
    } else {
        let mut update = HashMap::new();
        // iterate over auto tag settings that were removed and set to delete them
        for (removed_key, _) in old_auto_tag.extract_if(|key, _| !new_auto_tag.contains_key(key)) {
            update.insert(removed_key, AutoTagUpdate::default().delete());
        }
        // iterate over new auto tags and add them
        for (added_key, auto_tag) in
            new_auto_tag.extract_if(|key, _| !old_auto_tag.contains_key(key))
        {
            update.insert(
                added_key.clone(),
                AutoTagUpdate {
                    logic: Some(auto_tag.logic),
                    key: Some(added_key),
                    clear_key: false,
                    delete: false,
                },
            );
        }
        // now we're just left with keys/values that are in both, so compare them and update as needed
        for (key, new_value) in new_auto_tag {
            // we can be certain the old auto tags has this key, but wrap in an if statement anyway
            if let Some(old_value) = old_auto_tag.remove(&key) {
                if new_value == old_value {
                    // the auto tags are the same so skip this key
                    continue;
                }
                // insert an update for this key
                update.insert(
                    key,
                    AutoTagUpdate {
                        logic: set_modified!(old_value.logic, new_value.logic),
                        clear_key: set_clear!(old_value.key, new_value.key),
                        key: set_modified_opt!(old_value.key, new_value.key),
                        delete: false,
                    },
                );
            }
        }
        update
    }
}

/// Calculate an output collection update by diffing old and
/// new output collection settings
///
/// # Arguments
///
/// * `old_collection` - The old output collection settings
/// * `new_collection` - The new output collection settings
fn calculate_output_collection_update(
    old_collection: OutputCollection,
    new_collection: OutputCollection,
) -> Option<OutputCollectionUpdate> {
    if old_collection == new_collection {
        None
    } else {
        Some(OutputCollectionUpdate {
            handler: set_modified!(old_collection.handler, new_collection.handler),
            // TODO: not sure how to deal with this field
            clear_files: false,
            files: calculate_files_handler_update(old_collection.files, new_collection.files),
            auto_tag: calculate_auto_tag_updates(old_collection.auto_tag, new_collection.auto_tag),
            children: set_modified!(old_collection.children, new_collection.children),
            clear_groups: set_clear_vec!(old_collection.groups, new_collection.groups),
            groups: new_collection.groups,
        })
    }
}

/// Calculate a child filters update by diffing old and
/// new child filters settings
///
/// # Arguments
///
/// * `old_filters` - The old child filters settings
/// * `new_filters` - The new child filters settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_child_filters_update(
    old_filters: ChildFilters,
    new_filters: ChildFilters,
) -> Option<ChildFiltersUpdate> {
    if old_filters == new_filters {
        None
    } else {
        Some(ChildFiltersUpdate {
            // add ones in the new but not in the old
            add_mime: new_filters
                .mime
                .difference(&old_filters.mime)
                .cloned()
                .collect(),
            // remove ones in the old but not in the new
            remove_mime: old_filters
                .mime
                .difference(&new_filters.mime)
                .cloned()
                .collect(),
            add_file_name: new_filters
                .file_name
                .difference(&old_filters.file_name)
                .cloned()
                .collect(),
            remove_file_name: old_filters
                .file_name
                .difference(&new_filters.file_name)
                .cloned()
                .collect(),
            add_file_extension: new_filters
                .file_extension
                .difference(&old_filters.file_extension)
                .cloned()
                .collect(),
            remove_file_extension: old_filters
                .file_extension
                .difference(&new_filters.file_extension)
                .cloned()
                .collect(),
            submit_non_matches: set_modified!(
                old_filters.submit_non_matches,
                new_filters.submit_non_matches
            ),
        })
    }
}

/// Calculate a kvm update by diffing old and new kvm settings
///
/// # Arguments
///
/// * `old_kvm` - The old kvm settings
/// * `new_kvm` - The new kvm settings
fn calculate_kvm_update(old_kvm: Option<Kvm>, new_kvm: Option<Kvm>) -> KvmUpdate {
    match (old_kvm, new_kvm) {
        // none in both cases, so return a noop
        (None, None) => KvmUpdate::default(),
        // we added kvm settings, so set the update to whatever the new one is
        (None, Some(new_kvm)) => KvmUpdate {
            xml: Some(new_kvm.xml),
            qcow2: Some(new_kvm.qcow2),
        },
        // TODO: we set it from some to none, so we ought to clear it, but there's currently no mechanism for that
        (Some(_), None) => KvmUpdate::default(),
        (Some(old_kvm), Some(new_kvm)) => {
            if old_kvm == new_kvm {
                KvmUpdate::default()
            } else {
                KvmUpdate {
                    xml: set_modified!(old_kvm.xml, new_kvm.xml),
                    qcow2: set_modified!(old_kvm.qcow2, new_kvm.qcow2),
                }
            }
        }
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

/// Calculate a network policy update by diffing old and new policies
///
/// # Arguments
///
/// * `old_policies` - The set of old policies
/// * `new_policies` - The set of new policies
#[allow(clippy::needless_pass_by_value)]
fn calculate_network_policies_update(
    old_policies: HashSet<String>,
    new_policies: HashSet<String>,
) -> ImageNetworkPolicyUpdate {
    ImageNetworkPolicyUpdate {
        // policies added are ones in the new but not in the old
        policies_added: new_policies.difference(&old_policies).cloned().collect(),
        // policies removed are ones in the old but not in the new
        policies_removed: old_policies.difference(&new_policies).cloned().collect(),
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
        // TODO: seems to be unused?
        external: None,
        // TODO: template
        scaler: set_modified!(image.scaler, edited_image.scaler),
        timeout: set_modified_opt!(image.timeout, edited_image.timeout),
        // TODO: template millicpu and storage
        // TODO: deal with letter conversions...
        resources: set_modified!(image.resources, edited_image.resources),
        // TODO: template
        spawn_limit: set_modified!(image.spawn_limit, edited_image.spawn_limit),
        add_volumes,
        remove_volumes,
        // TODO: template
        add_env,
        remove_env,
        clear_version: set_clear!(image.version, edited_image.version),
        // TODO: template or use raw??
        version: set_modified_opt!(image.version, edited_image.version),
        clear_image: set_clear!(image.image, edited_image.image),
        image: set_modified_opt!(image.image, edited_image.image),
        // TODO: lifetime
        clear_lifetime: set_clear!(image.lifetime, edited_image.lifetime),
        lifetime: set_modified_opt!(image.lifetime, edited_image.lifetime),
        clear_description: set_clear!(image.description, edited_image.description),
        args: calculate_image_args_update(image.args, edited_image.args),
        modifiers: set_modified_opt!(image.modifiers, edited_image.modifiers),
        description: set_modified_opt!(image.description, edited_image.description),
        security_context: calculate_security_context_update(
            image.security_context,
            edited_image.security_context,
        ),
        collect_logs: set_modified!(image.collect_logs, edited_image.collect_logs),
        generator: set_modified!(image.generator, edited_image.generator),
        // TODO: template
        dependencies: calculate_dependencies_update(image.dependencies, edited_image.dependencies),
        // TODO: template
        display_type: set_modified!(image.display_type, edited_image.display_type),
        output_collection: calculate_output_collection_update(
            image.output_collection,
            edited_image.output_collection,
        ),
        child_filters: calculate_child_filters_update(
            image.child_filters,
            edited_image.child_filters,
        ),
        clean_up: calculate_clean_up_update(image.clean_up, edited_image.clean_up),
        kvm: calculate_kvm_update(image.kvm, edited_image.kvm),
        bans: calculate_bans_update(image.bans, edited_image.bans)?,
        network_policies: calculate_network_policies_update(
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
