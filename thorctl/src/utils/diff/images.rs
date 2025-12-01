//! Code for calculating diffs between images and image-related structs

use std::collections::{HashMap, HashSet};
use thorium::models::{
    AutoTag, AutoTagUpdate, CacheDependencySettings, CacheDependencySettingsUpdate, ChildFilters,
    ChildFiltersUpdate, ChildrenDependencySettings, ChildrenDependencySettingsUpdate, Cleanup,
    CleanupUpdate, Dependencies, DependenciesUpdate, DependencySettingsUpdate,
    EphemeralDependencySettings, EphemeralDependencySettingsUpdate, FilesHandler,
    FilesHandlerUpdate, GenericCacheDependencySettingsUpdate, ImageArgs, ImageArgsUpdate,
    ImageNetworkPolicyUpdate, Kvm, KvmUpdate, OutputCollection, OutputCollectionUpdate,
    RepoDependencySettings, ResultDependencySettings, ResultDependencySettingsUpdate,
    SampleDependencySettings, SecurityContext, SecurityContextUpdate, TagDependencySettings,
    TagDependencySettingsUpdate,
};

use crate::{calc_remove_add_vec, set_clear, set_clear_vec, set_modified, set_modified_opt};

/// Calculate an image args update by diffing old and
/// new image args settings
///
/// # Arguments
///
/// * `old_args` - The old args settings
/// * `new_args` - The new args settings
#[allow(clippy::needless_pass_by_value)]
pub fn calculate_image_args_update(
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
            // needs template
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
pub fn calculate_security_context_update(
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
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_sample_dependencies_update(
    old: SampleDependencySettings,
    new: SampleDependencySettings,
) -> DependencySettingsUpdate {
    DependencySettingsUpdate {
        location: set_modified!(old.location, new.location),
        clear_kwarg: set_clear!(old.kwarg, new.kwarg),
        kwarg: set_modified_opt!(old.kwarg, new.kwarg),
        strategy: set_modified!(old.strategy, new.strategy),
    }
}

/// Calculate a ephemeral dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_ephemeral_dependencies_update(
    mut old: EphemeralDependencySettings,
    mut new: EphemeralDependencySettings,
) -> EphemeralDependencySettingsUpdate {
    // calculate which names to remove/add
    let (remove_names, add_names) = calc_remove_add_vec!(old.names, new.names);
    EphemeralDependencySettingsUpdate {
        location: set_modified!(old.location, new.location),
        clear_kwarg: set_clear!(old.kwarg, new.kwarg),
        kwarg: set_modified_opt!(old.kwarg, new.kwarg),
        strategy: set_modified!(old.strategy, new.strategy),
        remove_names,
        add_names,
    }
}

/// Calculate a results dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_results_dependencies_update(
    mut old: ResultDependencySettings,
    mut new: ResultDependencySettings,
) -> ResultDependencySettingsUpdate {
    // calculate which images to remove/add
    let (remove_images, add_images) = calc_remove_add_vec!(old.images, new.images);
    // calculate which names to remove/add
    let (remove_names, add_names) = calc_remove_add_vec!(old.names, new.names);
    ResultDependencySettingsUpdate {
        // remove images that are in the old but not in the new
        remove_images,
        // add images that are in the new but not in the old
        add_images,
        location: set_modified!(old.location, new.location),
        // needs template
        kwarg: set_modified!(old.kwarg, new.kwarg),
        strategy: set_modified!(old.strategy, new.strategy),
        remove_names,
        add_names,
    }
}

/// Calculate a repo dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_repo_dependencies_update(
    old: RepoDependencySettings,
    new: RepoDependencySettings,
) -> DependencySettingsUpdate {
    DependencySettingsUpdate {
        location: set_modified!(old.location, new.location),
        clear_kwarg: set_clear!(old.kwarg, new.kwarg),
        kwarg: set_modified_opt!(old.kwarg, new.kwarg),
        strategy: set_modified!(old.strategy, new.strategy),
    }
}

/// Calculate a tags dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_tags_dependencies_update(
    old: TagDependencySettings,
    new: TagDependencySettings,
) -> TagDependencySettingsUpdate {
    TagDependencySettingsUpdate {
        enabled: set_modified!(old.enabled, new.enabled),
        location: set_modified!(old.location, new.location),
        clear_kwarg: set_clear!(old.kwarg, new.kwarg),
        kwarg: set_modified_opt!(old.kwarg, new.kwarg),
        strategy: set_modified!(old.strategy, new.strategy),
    }
}

/// Calculate a children dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_childen_dependencies_update(
    mut old: ChildrenDependencySettings,
    mut new: ChildrenDependencySettings,
) -> ChildrenDependencySettingsUpdate {
    // calculate which images to remove/add
    let (remove_images, add_images) = calc_remove_add_vec!(old.images, new.images);
    ChildrenDependencySettingsUpdate {
        enabled: set_modified!(old.enabled, new.enabled),
        remove_images,
        add_images,
        location: set_modified!(old.location, new.location),
        clear_kwarg: set_clear!(old.kwarg, new.kwarg),
        kwarg: set_modified_opt!(old.kwarg, new.kwarg),
        strategy: set_modified!(old.strategy, new.strategy),
    }
}

/// Calculate a cache dependencies update by diffing old and
/// new cache dependencies settings
///
/// # Arguments
///
/// * `old` - The old cache dependencies settings
/// * `new` - The new cache dependencies settings
#[allow(clippy::needless_pass_by_value)]
fn calculate_cache_dependencies_update(
    old: CacheDependencySettings,
    new: CacheDependencySettings,
) -> CacheDependencySettingsUpdate {
    // calculate our generic dependency settings update first
    let generic = GenericCacheDependencySettingsUpdate {
        clear_kwarg: set_clear!(old.generic.kwarg, new.generic.kwarg),
        kwarg: set_modified_opt!(old.generic.kwarg, new.generic.kwarg),
        strategy: set_modified!(old.generic.strategy, new.generic.strategy),
    };
    // build the update for our cache dependency settings
    CacheDependencySettingsUpdate {
        location: set_modified!(old.location, new.location),
        generic,
        use_parent_cache: set_modified!(old.use_parent_cache, new.use_parent_cache),
        enabled: set_modified!(old.enabled, new.enabled),
    }
}

/// Calculate a dependencies update by diffing old and
/// new dependencies settings
///
/// # Arguments
///
/// * `old` - The old dependencies settings
/// * `new` - The new dependencies settings
#[allow(clippy::needless_pass_by_value)]
pub fn calculate_dependencies_update(old: Dependencies, new: Dependencies) -> DependenciesUpdate {
    // calculate the updates for our dependency settings
    let samples = calculate_sample_dependencies_update(old.samples, new.samples);
    let ephemeral = calculate_ephemeral_dependencies_update(old.ephemeral, new.ephemeral);
    let results = calculate_results_dependencies_update(old.results, new.results);
    let repos = calculate_repo_dependencies_update(old.repos, new.repos);
    let tags = calculate_tags_dependencies_update(old.tags, new.tags);
    let children = calculate_childen_dependencies_update(old.children, new.children);
    let cache = calculate_cache_dependencies_update(old.cache, new.cache);
    // build our dependencies update
    DependenciesUpdate {
        samples,
        ephemeral,
        results,
        repos,
        tags,
        children,
        cache,
    }
}

/// Calculate a clean up update by diffing old and
/// new clean up settings
///
/// # Arguments
///
/// * `old_clean_up` - The old clean up settings
/// * `new_clean_up` - The new clean up settings
pub fn calculate_clean_up_update(
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
pub fn calculate_files_handler_update(
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
pub fn calculate_auto_tag_updates(
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
pub fn calculate_output_collection_update(
    old_collection: OutputCollection,
    new_collection: OutputCollection,
) -> Option<OutputCollectionUpdate> {
    if old_collection == new_collection {
        None
    } else {
        Some(OutputCollectionUpdate {
            handler: set_modified!(old_collection.handler, new_collection.handler),
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
pub fn calculate_child_filters_update(
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
pub fn calculate_kvm_update(old_kvm: Option<Kvm>, new_kvm: Option<Kvm>) -> KvmUpdate {
    match (old_kvm, new_kvm) {
        // none in both cases, so return a noop
        (None, None) => KvmUpdate::default(),
        // we added kvm settings, so set the update to whatever the new one is
        (None, Some(new_kvm)) => KvmUpdate {
            xml: Some(new_kvm.xml),
            qcow2: Some(new_kvm.qcow2),
        },
        // set from Some to None, but there's no way to clear Kvm settings currently
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

/// Calculate a network policy update by diffing old and new policies
///
/// # Arguments
///
/// * `old_policies` - The set of old policies
/// * `new_policies` - The set of new policies
#[allow(clippy::needless_pass_by_value)]
pub fn calculate_network_policies_update(
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
