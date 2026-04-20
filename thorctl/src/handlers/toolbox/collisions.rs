//! Resolution of `(group, name)` collisions in a toolbox manifest
//!
//! A toolbox import collapses every resource into one group when
//! `--group-override` is used (and a hand-authored manifest may already
//! double-define a name). When two image or pipeline entries resolve to the same
//! Thorium `(group, name)` identity, importing both would silently overwrite one
//! with the other. This module detects those collisions and resolves them before
//! any writes happen:
//!
//! - pure duplicates (byte-identical configs) are de-duped silently
//! - in an interactive session, the user chooses per collision to **Rename** the
//!   extras (dependent pipelines are repointed automatically) or **Skip** them
//! - otherwise the colliding resources, and any pipelines that depend on them,
//!   are skipped with a warning so the rest of the toolbox still imports

use colored::Colorize;
use std::collections::{HashMap, HashSet};
use thorium::Error;

use super::manifest::{Collision, CollisionMember, SourceGroups, ToolboxManifest};
use super::prompt;
use crate::handlers::progress::Bar;

/// The user-chosen resolution for a real (non-duplicate) collision
enum CollisionAction {
    /// Keep every colliding resource, renaming all but the one the user elects to
    /// retain the original name
    Rename,
    /// Drop the colliding identity entirely; for images this also drops any
    /// pipelines that depended on it
    Skip,
}

/// Detect and resolve every image and pipeline collision in the manifest
///
/// Returns a map of each renamed image's new manifest key to its original key.
/// Bundled image tarballs are saved on disk under the original key, so callers that
/// push bundled images need this to find an archive after its image was renamed.
///
/// # Arguments
///
/// * `manifest` - The manifest to mutate in place (rename/skip resolutions)
/// * `sources` - Pre-override groups captured before [`ToolboxManifest::override_group`]
/// * `can_prompt` - Whether the session can ask the user (TTY, not `-y`)
/// * `progress` - The progress bar, suspended while prompting
pub fn resolve_collisions(
    manifest: &mut ToolboxManifest,
    sources: &SourceGroups,
    can_prompt: bool,
    progress: &Bar,
) -> Result<HashMap<String, String>, Error> {
    // collect image renames as new-key -> original-key so bundled tarballs (named by
    // the original on-disk key) stay resolvable after a rename
    let image_renames = resolve_kind::<ImageCollisions>(manifest, sources, can_prompt, progress)?;
    // pipelines have no bundled tarballs, so their renames need no remap and are dropped
    resolve_kind::<PipelineCollisions>(manifest, sources, can_prompt, progress)?;
    Ok(image_renames.into_iter().collect())
}

/// The per-kind operations the generic resolver dispatches through
///
/// Image and pipeline resolution share one control flow (de-dupe → prompt →
/// rename-or-skip); only these manifest operations differ — notably the
/// image-only cascade of dropping pipelines that depended on a skipped image.
trait CollisionKind {
    /// The resource noun used in prompts and messages ("image"/"pipeline")
    const NOUN: &'static str;
    /// Detect this kind's `(group, name)` collisions across the manifest
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to scan for colliding identities
    /// * `sources` - Pre-override groups recorded so each member knows its origin
    fn detect(manifest: &ToolboxManifest, sources: &SourceGroups) -> Result<Vec<Collision>, Error>;
    /// Collapse a byte-identical collision down to a single surviving entry
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The duplicate collision whose extra copies are removed
    fn dedupe(manifest: &mut ToolboxManifest, collision: &Collision);
    /// Suggest a not-yet-used name to rename a colliding member to
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest used to check which names are already taken
    /// * `collision` - The collision the member belongs to (supplies the base name)
    /// * `member` - The member a rename is being suggested for
    fn suggested_rename(
        manifest: &ToolboxManifest,
        collision: &Collision,
        member: &CollisionMember,
    ) -> String;
    /// The names already used in `group`, which a rename must avoid to not create a
    /// fresh collision
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to read existing names from
    /// * `group` - The target group whose used names are returned
    fn names_in_group(manifest: &ToolboxManifest, group: &str) -> HashSet<String>;
    /// Rename a member and repoint everything that referenced it (images repoint
    /// dependent pipelines; pipelines have nothing to repoint)
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The collision being resolved (supplies the original name)
    /// * `member` - The member to rename
    /// * `new_name` - The new name to give the member
    /// * `sources` - Pre-override groups used to pick which dependents to repoint
    fn rename_member(
        manifest: &mut ToolboxManifest,
        collision: &Collision,
        member: &CollisionMember,
        new_name: &str,
        sources: &SourceGroups,
    );
    /// Drop the colliding identity and warn; images also drop dependent pipelines
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The collision identity being dropped
    /// * `progress` - The progress bar the skip warnings are routed through
    fn skip(manifest: &mut ToolboxManifest, collision: &Collision, progress: &Bar);
}

/// Image collision operations
struct ImageCollisions;
/// Pipeline collision operations
struct PipelineCollisions;

impl CollisionKind for ImageCollisions {
    const NOUN: &'static str = "image";
    /// Detect colliding image identities across source groups
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to scan for colliding image identities
    /// * `sources` - Pre-override groups recorded on each detected member
    fn detect(manifest: &ToolboxManifest, sources: &SourceGroups) -> Result<Vec<Collision>, Error> {
        manifest.detect_image_collisions(sources)
    }
    /// Collapse a byte-identical image collision down to a single entry
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The duplicate image collision to de-dupe
    fn dedupe(manifest: &mut ToolboxManifest, collision: &Collision) {
        manifest.dedupe_image_collision(collision);
    }
    /// Suggest a not-yet-used name to rename a colliding image member to
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest used to find an unused name
    /// * `collision` - The collision supplying the base name
    /// * `member` - The image member a rename is suggested for
    fn suggested_rename(
        manifest: &ToolboxManifest,
        collision: &Collision,
        member: &CollisionMember,
    ) -> String {
        manifest.suggested_image_rename(collision, member)
    }
    /// The image names already used in `group`
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to read existing image names from
    /// * `group` - The target group whose used image names are returned
    fn names_in_group(manifest: &ToolboxManifest, group: &str) -> HashSet<String> {
        manifest.image_names_in_group(group)
    }
    /// Rename an image member and repoint the pipelines that referenced it
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The collision being resolved (supplies the original name)
    /// * `member` - The image member to rename
    /// * `new_name` - The new name to give the image
    /// * `sources` - Pre-override groups used to pick which dependents to repoint
    fn rename_member(
        manifest: &mut ToolboxManifest,
        collision: &Collision,
        member: &CollisionMember,
        new_name: &str,
        sources: &SourceGroups,
    ) {
        // repointing covers both each dependent pipeline's image map and its order,
        // and uses sources to pick the pipelines that wanted this exact variant
        manifest.rename_image_member(collision, member, new_name, sources);
    }
    /// Drop the colliding image identity (and its dependent pipelines) and warn
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The image collision identity being dropped
    /// * `progress` - The progress bar the skip warnings are routed through
    fn skip(manifest: &mut ToolboxManifest, collision: &Collision, progress: &Bar) {
        // dropping the image identity also returns the pipelines that depended on it
        // so the cascade can be reported, since a pipeline missing its image is invalid
        let dropped =
            manifest.remove_image_identity_and_dependents(&collision.group, &collision.name);
        warn_skipped_image(progress, collision, &dropped);
    }
}

impl CollisionKind for PipelineCollisions {
    const NOUN: &'static str = "pipeline";
    /// Detect colliding pipeline identities across source groups
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to scan for colliding pipeline identities
    /// * `sources` - Pre-override groups recorded on each detected member
    fn detect(manifest: &ToolboxManifest, sources: &SourceGroups) -> Result<Vec<Collision>, Error> {
        manifest.detect_pipeline_collisions(sources)
    }
    /// Collapse a byte-identical pipeline collision down to a single entry
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The duplicate pipeline collision to de-dupe
    fn dedupe(manifest: &mut ToolboxManifest, collision: &Collision) {
        manifest.dedupe_pipeline_collision(collision);
    }
    /// Suggest a not-yet-used name to rename a colliding pipeline member to
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest used to find an unused name
    /// * `collision` - The collision supplying the base name
    /// * `member` - The pipeline member a rename is suggested for
    fn suggested_rename(
        manifest: &ToolboxManifest,
        collision: &Collision,
        member: &CollisionMember,
    ) -> String {
        manifest.suggested_pipeline_rename(collision, member)
    }
    /// The pipeline names already used in `group`
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to read existing pipeline names from
    /// * `group` - The target group whose used pipeline names are returned
    fn names_in_group(manifest: &ToolboxManifest, group: &str) -> HashSet<String> {
        manifest.pipeline_names_in_group(group)
    }
    /// Rename a pipeline member (no cascade — nothing references a pipeline by name)
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `_collision` - Unused; pipelines have no dependents to disambiguate
    /// * `member` - The pipeline member to rename
    /// * `new_name` - The new name to give the pipeline
    /// * `_sources` - Unused; there are no dependents to repoint
    fn rename_member(
        manifest: &mut ToolboxManifest,
        _collision: &Collision,
        member: &CollisionMember,
        new_name: &str,
        _sources: &SourceGroups,
    ) {
        // unlike images, nothing references a pipeline by name, so renaming one needs
        // no cascade and ignores the collision/sources args
        manifest.rename_pipeline_member(member, new_name);
    }
    /// Drop the colliding pipeline identity and warn (no dependents to cascade)
    ///
    /// # Arguments
    ///
    /// * `manifest` - The manifest to mutate in place
    /// * `collision` - The pipeline collision identity being dropped
    /// * `progress` - The progress bar the skip warning is routed through
    fn skip(manifest: &mut ToolboxManifest, collision: &Collision, progress: &Bar) {
        // remove the colliding pipeline identity; nothing depends on a pipeline so
        // there is no cascade to report, unlike the image skip path
        manifest.remove_pipeline_identity(&collision.group, &collision.name);
        progress.warning(format!(
            "Skipping colliding pipeline '{}:{}' ({} conflicting definitions: {})",
            collision.group.bright_yellow(),
            collision.name.bright_yellow(),
            collision.members.len(),
            describe_members("pipeline", collision),
        ));
    }
}

/// Resolve every collision of one kind: de-dupe duplicates, rename or skip real
/// conflicts
///
/// Distinct collision identities don't affect each other, so resolving against the
/// snapshot taken up front is safe. Returns the renames performed as `(new manifest
/// key, original manifest key)` pairs so callers can remap on-disk artifacts named
/// by the original key.
///
/// # Arguments
///
/// * `manifest` - The manifest to mutate in place (rename/skip resolutions)
/// * `sources` - Pre-override groups used to disambiguate which variant a pipeline wanted
/// * `can_prompt` - Whether the session can ask the user (TTY, not `-y`)
/// * `progress` - The progress bar, suspended while prompting
fn resolve_kind<K: CollisionKind>(
    manifest: &mut ToolboxManifest,
    sources: &SourceGroups,
    can_prompt: bool,
    progress: &Bar,
) -> Result<Vec<(String, String)>, Error> {
    // record each rename so the caller can remap artifacts keyed by the original name
    let mut renames = Vec::new();
    // detect collisions once up front; distinct identities never affect each other, so
    // iterating the snapshot while mutating the manifest is safe
    for collision in K::detect(manifest, sources)? {
        // byte-identical copies aren't a real conflict, so silently collapse them to one
        // and move on without bothering the user
        if collision.identical {
            K::dedupe(manifest, &collision);
            progress.info_anonymous(format!(
                "De-duplicated identical {} '{}:{}' ({} copies)",
                K::NOUN,
                collision.group.bright_yellow(),
                collision.name.bright_yellow(),
                collision.members.len(),
            ));
            continue;
        }
        // a real conflict: ask the user when possible, otherwise default to skipping so
        // an unattended run still imports the rest of the toolbox instead of aborting
        let action = if can_prompt {
            progress.suspend(|| prompt_collision_action(K::NOUN, &collision))?
        } else {
            CollisionAction::Skip
        };
        match action {
            CollisionAction::Rename => {
                // let the user pick which entry keeps the original name; rename the rest
                let keep = progress.suspend(|| prompt_keep_member(K::NOUN, &collision))?;
                for (index, member) in collision.members.iter().enumerate() {
                    // skip the one entry the user chose to keep under the original name
                    if index == keep {
                        continue;
                    }
                    // prefill the prompt with a name known to be free
                    let suggested = K::suggested_rename(manifest, &collision, member);
                    // names already used in the target group are off-limits, so a
                    // rename can't introduce a fresh collision; recomputed each iteration
                    // because the prior rename may have added a new name to the group
                    let taken = K::names_in_group(manifest, &collision.group);
                    let new_name = progress.suspend(|| {
                        prompt_new_name(K::NOUN, &collision.name, member, &suggested, &taken)
                    })?;
                    // apply the rename and cascade it to any dependents (image map + order)
                    K::rename_member(manifest, &collision, member, &new_name, sources);
                    // map the new key back to the original on-disk key (the member's
                    // manifest key) so a bundled tarball saved under it is still found
                    renames.push((new_name.clone(), member.manifest_key.clone()));
                }
                progress.info_anonymous(format!(
                    "Renamed colliding {} '{}:{}' into {} distinct {}s",
                    K::NOUN,
                    collision.group.bright_yellow(),
                    collision.name.bright_yellow(),
                    collision.members.len(),
                    K::NOUN,
                ));
            }
            CollisionAction::Skip => K::skip(manifest, &collision, progress),
        }
    }
    Ok(renames)
}

/// Prompt the user for how to resolve a collision
///
/// # Arguments
///
/// * `kind` - The resource noun ("image" or "pipeline") shown in the prompt
/// * `collision` - The collision whose members are listed for the user
fn prompt_collision_action(kind: &str, collision: &Collision) -> Result<CollisionAction, Error> {
    // print a header explaining that these copies would overwrite each other
    println!(
        "\n{} {} '{}:{}' is defined {} times; the copies would overwrite each other:",
        "Collision:".bright_yellow(),
        kind,
        collision.group.bright_blue(),
        collision.name.bright_blue(),
        collision.members.len(),
    );
    // list every conflicting member so the user can see what is in tension
    for member in &collision.members {
        println!(
            "  - manifest entry '{}' version '{}' (from group '{}')",
            member.manifest_key, member.version, member.source_group,
        );
    }
    // offer the two resolution choices, rename first as the default
    let items = &[
        "Rename - keep all of them; rename the extras (dependent pipelines are repointed)",
        "Skip   - skip these and any pipelines that depend on them",
    ];
    // read the user's selection
    let selection = dialoguer::Select::new()
        .items(items)
        .default(0)
        .interact()
        .map_err(|err| Error::new(format!("Failed to read user input: {err}")))?;
    // map the selected index to its action
    Ok(match selection {
        0 => CollisionAction::Rename,
        _ => CollisionAction::Skip,
    })
}

/// Describe a collision member by its manifest key and full config identity
/// (`<source group>/<config name>:<version>`), so the distinguishing fields are
/// all visible when choosing which entry to keep or rename
///
/// # Arguments
///
/// * `noun` - The resource kind ("image" or "pipeline") for the message
/// * `name` - The shared config name of the collision (same for every member)
/// * `member` - The member to describe
fn describe_member(noun: &str, name: &str, member: &CollisionMember) -> String {
    format!(
        "manifest '{}' — {noun} '{}/{}:{}'",
        member.manifest_key, member.source_group, name, member.version,
    )
}

/// Prompt for which colliding member should keep the original name; the rest are renamed
///
/// # Arguments
///
/// * `noun` - The resource kind ("image" or "pipeline") for each member's description
/// * `collision` - The collision whose members are offered as choices
fn prompt_keep_member(noun: &str, collision: &Collision) -> Result<usize, Error> {
    // explain that the chosen entry keeps the name and the rest are renamed
    println!(
        "\nWhich entry should keep the name '{}'? The others will be renamed.",
        collision.name.bright_blue(),
    );
    // render each member as a selectable, fully-qualified description
    let items: Vec<String> = collision
        .members
        .iter()
        .map(|member| describe_member(noun, &collision.name, member))
        .collect();
    // read the index of the member the user wants to keep
    dialoguer::Select::new()
        .items(&items)
        .default(0)
        .interact()
        .map_err(|err| Error::new(format!("Failed to read user input: {err}")))
}

/// Prompt for a new name for a renamed collision member, prefilled with `suggested`
/// and rejecting any name already used in the target group (`taken`)
///
/// # Arguments
///
/// * `noun` - The resource kind ("image" or "pipeline") for the member's description
/// * `old_name` - The original colliding name shown in the member's description
/// * `member` - The member being renamed
/// * `suggested` - The default name prefilled into the prompt
/// * `taken` - The names already used in the target group, rejected by validation
fn prompt_new_name(
    noun: &str,
    old_name: &str,
    member: &CollisionMember,
    suggested: &str,
    taken: &HashSet<String>,
) -> Result<String, Error> {
    // prompt for a new name, prefilled with the suggestion and rejecting taken names
    dialoguer::Input::new()
        .with_prompt(format!("New name for {}", describe_member(noun, old_name, member)))
        .default(suggested.to_string())
        .validate_with(|value: &String| {
            // enforce the same name rules as any other image/pipeline name
            prompt::validate_name(value, prompt::RESOURCE_NAME_MAX)?;
            // reject a name already used by another resource in the target group
            if taken.contains(value) {
                return Err(format!(
                    "'{value}' is already used by another resource in this group; pick a different name"
                ));
            }
            Ok(())
        })
        .interact_text()
        .map_err(|err| Error::new(format!("Failed to read new name: {err}")))
}

/// Warn that a colliding image (and its dependent pipelines) were skipped
///
/// # Arguments
///
/// * `progress` - The progress bar to route the warnings through
/// * `collision` - The skipped image collision being described
/// * `dropped_pipelines` - The pipelines dropped because they depended on the image
fn warn_skipped_image(progress: &Bar, collision: &Collision, dropped_pipelines: &[String]) {
    // warn that the colliding image identity was skipped, listing its members
    progress.warning(format!(
        "Skipping colliding image '{}:{}' ({} conflicting definitions: {})",
        collision.group.bright_yellow(),
        collision.name.bright_yellow(),
        collision.members.len(),
        describe_members("image", collision),
    ));
    // warn about each pipeline that was dropped along with the image
    for pipeline in dropped_pipelines {
        progress.warning(format!(
            "  └ also skipping dependent pipeline '{}'",
            pipeline.bright_yellow()
        ));
    }
}

/// A comma-separated description of every member of a collision
///
/// # Arguments
///
/// * `noun` - The resource kind ("image" or "pipeline") for each member's description
/// * `collision` - The collision whose members are described
fn describe_members(noun: &str, collision: &Collision) -> String {
    collision
        .members
        .iter()
        .map(|member| describe_member(noun, &collision.name, member))
        .collect::<Vec<_>>()
        .join(", ")
}
