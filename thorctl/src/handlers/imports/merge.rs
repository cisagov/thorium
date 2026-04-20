//! Merge-conflict resolution for toolbox imports
//!
//! When a toolbox import encounters images or pipelines that already exist,
//! this module handles the interactive merge workflow: prompting the user for
//! an action, generating YAML with conflict markers, opening the editor,
//! and calculating the appropriate update to apply.

use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thorium::models::{
    ChildFilters, Cleanup, Dependencies, Image, ImageArgs, ImageBan, ImageLifetime, ImageRequest,
    ImageScaler, ImageUpdate, ImageVersion, Kvm, OutputCollection, OutputDisplayType, Pipeline,
    PipelineRequest, PipelineUpdate, ResourcesRequest, SecurityContext, SpawnLimits, Volume,
};
use thorium::{CtlConf, Error, Thorium};
use uuid::Uuid;

use super::ImportOutcome;
use super::categorize::Categorized;
use super::editor;
use super::kind::ImportKind;
use super::rollback::Journal;
use super::update;
use crate::handlers::progress::Bar;

// ─── Curated Editor Key Order ────────────────────────────────────────────────

/// Curated top-level key order for the image editor view, mirroring the UI image
/// form (`ui/src/pages/images/ImageCreate.jsx` + `components/pages/images/Fields.jsx`).
///
/// Non-editable fields (`name`/`group`/`creator`/`bans`, and `runtime`) lead/sit with
/// the static-marked fields; commonly edited basics follow the form order; sub-sections
/// follow `ImageCreate.jsx`. Covers every `MergeableImage` field and every
/// `build_image_config` key. Keys not listed still appear (sorted) at the end — see
/// [`crate::utils::curated_yaml`]. Plain names; the helper also matches the `*name*`
/// static-marked forms.
pub const IMAGE_FIELD_ORDER: &[&str] = &[
    // non-editable: identity + server/managed (shown for context, marked *...*, ignored on save)
    "name", "group", "creator", "bans",
    // commonly edited basics (form order; runtime — non-editable — kept in its UI display slot)
    "description", "version", "scaler", "image", "timeout", "lifetime", "runtime",
    "display_type", "spawn_limit", "collect_logs", "generator",
    // sub-sections (ImageCreate.jsx order; child_filters with output_collection;
    // clean_up/kvm/modifiers just before security_context)
    "resources", "args", "output_collection", "child_filters", "dependencies", "env", "volumes",
    "network_policies", "clean_up", "kvm", "modifiers", "security_context",
];

/// Curated top-level key order for the pipeline editor view. `name`/`group` only
/// appear in the init config shape (not the edit view); harmless when absent.
pub const PIPELINE_FIELD_ORDER: &[&str] =
    &["name", "group", "order", "description", "sla", "triggers"];

// ─── Mergeable Structs ───────────────────────────────────────────────────────

/// Normalize a description into its canonical comparison form
///
/// `toolbox build` injects the description from `description.md`, trimming trailing
/// whitespace (build.rs `apply_description_md`), while the live image/pipeline in
/// Thorium keeps whatever was stored (often with a trailing newline). Without this
/// a freshly exported toolbox would diff dirty on the description alone. Trimming
/// both sides — and collapsing an empty result to `None` — makes the comparison
/// insensitive to that round-trip.
///
/// # Arguments
///
/// * `description` - The raw description to normalize
fn normalize_description(description: Option<String>) -> Option<String> {
    description
        .map(|text| text.trim_end().to_string())
        .filter(|text| !text.is_empty())
}

/// An image converted to a common format for editing, merging, and diffing.
///
/// Used by both standalone `images edit` and `toolbox import` merge workflows.
/// Identity fields (group, name, creator) and server-managed fields (runtime,
/// bans) are `Option` — populated when editing an existing image via
/// `From<Image>`, left as `None` when converting from an `ImageRequest`
/// (toolbox merge). Fields set to `None` are omitted from serialized YAML.
#[derive(Debug, Serialize, Deserialize)]
pub struct MergeableImage {
    /// The group the image belongs to (identity, non-editable; `None` from a request)
    #[serde(rename = "*group*", skip_serializing_if = "Option::is_none", default)]
    pub group: Option<String>,
    /// The image's name (identity, non-editable; `None` from a request)
    #[serde(rename = "*name*", skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    /// The user that created the image (server-managed, non-editable; `None` from a request)
    #[serde(rename = "*creator*", skip_serializing_if = "Option::is_none", default)]
    pub creator: Option<String>,
    /// The image's version, or `None` for a versionless image
    pub version: Option<ImageVersion>,
    /// The scaler that schedules this image
    pub scaler: ImageScaler,
    /// The container image URL, or `None` when not set
    pub image: Option<String>,
    /// How long the image lives before it is reaped, or `None` for no limit
    pub lifetime: Option<ImageLifetime>,
    /// The per-job timeout in seconds, or `None` for the default
    pub timeout: Option<u64>,
    /// The compute resources requested for each job
    pub resources: ResourcesRequest,
    /// The limit on how many copies of this image can run at once
    pub spawn_limit: SpawnLimits,
    /// The environment variables to set, where `None` unsets the variable
    pub env: HashMap<String, Option<String>>,
    /// The image's average runtime (server-managed, non-editable; `None` from a request)
    #[serde(rename = "*runtime*", skip_serializing_if = "Option::is_none", default)]
    pub runtime: Option<f64>,
    /// The volumes mounted into each job
    pub volumes: Vec<Volume>,
    /// The command-line argument layout passed to the tool
    pub args: ImageArgs,
    /// Free-form scaler modifiers, or `None` when not set
    pub modifiers: Option<String>,
    /// The image's description (normalized via [`normalize_description`])
    pub description: Option<String>,
    /// The container security context, or `None` for the server default
    pub security_context: Option<SecurityContext>,
    /// Whether the agent collects job stdout/stderr as logs
    pub collect_logs: bool,
    /// Whether this image generates children (and reruns itself until done)
    pub generator: bool,
    /// The other images this image depends on for inputs
    pub dependencies: Dependencies,
    /// How this image's results are displayed in the UI
    pub display_type: OutputDisplayType,
    /// How and where this image's output is collected
    pub output_collection: OutputCollection,
    /// The filters controlling which children are submitted back to Thorium
    pub child_filters: ChildFilters,
    /// The cleanup tool to run after each job, or `None` for none
    pub clean_up: Option<Cleanup>,
    /// The KVM/VM configuration for this image, or `None` for a non-VM image
    pub kvm: Option<Kvm>,
    /// The bans on this image keyed by ban id (server-managed, non-editable; `None` from a request)
    #[serde(rename = "*bans*", skip_serializing_if = "Option::is_none", default)]
    pub bans: Option<HashMap<Uuid, ImageBan>>,
    /// The network policies applied to this image's jobs
    pub network_policies: HashSet<String>,
}

impl From<Image> for MergeableImage {
    /// Build a mergeable image from an existing Thorium image, populating the
    /// identity and server-managed fields the request form lacks
    ///
    /// # Arguments
    ///
    /// * `image` - The existing Thorium image to convert
    fn from(image: Image) -> Self {
        Self {
            // identity + server-managed fields are present on a live image; carry them
            // so the editor view shows them (marked static) and mirroring has values
            group: Some(image.group),
            name: Some(image.name),
            creator: Some(image.creator),
            version: image.version,
            scaler: image.scaler,
            image: image.image,
            lifetime: image.lifetime,
            timeout: image.timeout,
            // the live image carries full `Resources`; convert to the request shape
            // the editor edits in
            resources: image.resources.into(),
            spawn_limit: image.spawn_limit,
            env: image.env,
            runtime: Some(image.runtime),
            volumes: image.volumes,
            args: image.args,
            modifiers: image.modifiers,
            // normalize so a description.md round-trip doesn't read as drift
            description: normalize_description(image.description),
            // a live image always has a concrete security context; wrap it to match
            // the request-sourced side, which defaults an omitted context
            security_context: Some(image.security_context),
            collect_logs: image.collect_logs,
            generator: image.generator,
            dependencies: image.dependencies,
            display_type: image.display_type,
            output_collection: image.output_collection,
            child_filters: image.child_filters,
            clean_up: image.clean_up,
            kvm: image.kvm,
            bans: Some(image.bans),
            network_policies: image.network_policies,
        }
    }
}

impl From<ImageRequest> for MergeableImage {
    /// Build a mergeable image from an incoming manifest request, leaving identity
    /// and server-managed fields unset since a request never carries them
    ///
    /// # Arguments
    ///
    /// * `req` - The incoming image request from the manifest
    fn from(req: ImageRequest) -> Self {
        Self {
            // a request never carries identity/server-managed fields; leave them unset
            // so they're omitted from the serialized editor view
            group: None,
            name: None,
            creator: None,
            version: req.version,
            scaler: req.scaler,
            image: req.image,
            lifetime: req.lifetime,
            timeout: req.timeout,
            resources: req.resources,
            spawn_limit: req.spawn_limit,
            env: req.env,
            // server-managed, never present on a request
            runtime: None,
            volumes: req.volumes,
            args: req.args,
            modifiers: req.modifiers,
            // normalize so a description.md round-trip doesn't read as drift
            description: normalize_description(req.description),
            // a request that omits the security context (None) is equivalent to the
            // server default, which is what `From<Image>` always carries. Default it
            // here so an omitted context doesn't read as drift against an existing
            // image (the update calc already treats default-vs-default as no change).
            security_context: Some(req.security_context.unwrap_or_default()),
            collect_logs: req.collect_logs,
            generator: req.generator,
            dependencies: req.dependencies,
            display_type: req.display_type,
            output_collection: req.output_collection,
            child_filters: req.child_filters,
            clean_up: req.clean_up,
            kvm: req.kvm,
            // server-managed, never present on a request
            bans: None,
            network_policies: req.network_policies,
        }
    }
}

/// A pipeline converted to a common format for merge comparison and YAML editing.
/// Only contains editable fields — group, name, creator, and bans are excluded.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeablePipeline {
    /// The stage ordering, as groups of image names that run together in sequence
    pub order: Vec<Vec<String>>,
    /// The pipeline's SLA in seconds
    pub sla: u64,
    /// The pipeline's triggers keyed by name, kept as raw JSON for the editor view
    pub triggers: HashMap<String, serde_json::Value>,
    /// The pipeline's description (normalized via [`normalize_description`])
    pub description: Option<String>,
}

impl From<Pipeline> for MergeablePipeline {
    /// Build a mergeable pipeline from an existing Thorium pipeline
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The existing Thorium pipeline to convert
    fn from(pipeline: Pipeline) -> Self {
        Self {
            order: pipeline.order,
            sla: pipeline.sla,
            // EventTrigger -> Value is infallible in practice; a Null fallback would
            // only affect this comparison view. The authoritative validation is the
            // write-back parse in `calculate_pipeline_update_from_mergeable`, which
            // errors on a bad trigger rather than dropping it.
            triggers: pipeline
                .triggers
                .into_iter()
                .map(|(k, v)| (k, serde_json::to_value(v).unwrap_or_default()))
                .collect(),
            // normalize so a description.md round-trip doesn't read as drift
            description: normalize_description(pipeline.description),
        }
    }
}

impl From<PipelineRequest> for MergeablePipeline {
    /// Build a mergeable pipeline from an incoming manifest request, applying the
    /// API's default SLA when the request omits one
    ///
    /// # Arguments
    ///
    /// * `req` - The incoming pipeline request from the manifest
    fn from(req: PipelineRequest) -> Self {
        // deserialize the order from the flexible Value format to Vec<Vec<String>>
        let order: Vec<Vec<String>> = req
            .deserialize_image_order()
            .unwrap_or_default()
            .into_iter()
            .map(|inner| inner.into_iter().map(String::from).collect())
            .collect();
        Self {
            order,
            // a request without an explicit SLA falls back to one week (604,800s),
            // matching the API's default so the diff doesn't show phantom drift
            sla: req.sla.unwrap_or(604_800),
            // convert each trigger to raw JSON for the editor view; the authoritative
            // validation is the write-back parse in calculate_pipeline_update_from_mergeable
            triggers: req
                .triggers
                .into_iter()
                .map(|(k, v)| (k, serde_json::to_value(v).unwrap_or_default()))
                .collect(),
            // normalize so a description.md round-trip doesn't read as drift
            description: normalize_description(req.description),
        }
    }
}

// ─── Per-Resource Prompt ─────────────────────────────────────────────────────

/// The action the user wants to take for a changed resource
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeAction {
    /// Open the editor to review and resolve conflicts
    Edit,
    /// Keep the existing configuration unchanged
    Skip,
    /// Accept all incoming changes from the manifest
    Apply,
    /// Stop processing remaining resources
    Quit,
}

/// Prompt the user for what action to take on a changed resource
///
/// # Arguments
///
/// * `resource_type` - Either "Image" or "Pipeline"
/// * `group` - The group the resource is in
/// * `name` - The name of the resource
fn prompt_merge_action(resource_type: &str, group: &str, name: &str) -> Result<MergeAction, Error> {
    println!(
        "\n{} '{}:{}' has changes:",
        resource_type.bright_yellow(),
        group.bright_blue(),
        name.bright_blue(),
    );
    let items = &[
        "Edit   - Open editor to review and resolve conflicts",
        "Skip   - Keep the existing configuration unchanged",
        "Apply  - Accept all incoming changes from the manifest",
        "Quit   - Stop processing remaining resources",
    ];
    let selection = dialoguer::Select::new()
        .items(items)
        .default(0)
        .interact()
        .map_err(|err| Error::new(format!("Failed to read user input: {err}")))?;
    Ok(match selection {
        0 => MergeAction::Edit,
        1 => MergeAction::Skip,
        2 => MergeAction::Apply,
        3 => MergeAction::Quit,
        _ => MergeAction::Skip,
    })
}

/// Copy identity and server-managed fields from `current` onto `incoming`
///
/// A toolbox manifest never carries an image's identity fields (group, name,
/// creator) or its server-managed fields (runtime, bans), and this merge can't
/// edit them — so the incoming (manifest) side always omits them. Left alone, the
/// line-based conflict diff would flag every such field as a conflict (a value on
/// the current side, nothing on the incoming side). Mirroring them makes both
/// sides match so they render as shared context instead. `calculate_image_update*`
/// ignores these fields when computing what to apply, so this is display-only.
fn mirror_non_editable_fields(incoming: &mut MergeableImage, current: &MergeableImage) {
    incoming.group.clone_from(&current.group);
    incoming.name.clone_from(&current.name);
    incoming.creator.clone_from(&current.creator);
    incoming.runtime = current.runtime;
    incoming.bans.clone_from(&current.bans);
}

// ─── Single-Resource Interactive Merge ───────────────────────────────────────

/// Resolve an image merge conflict via the editor and return the resulting
/// image update, or None if the user's edits result in no changes or they
/// cancelled
///
/// # Arguments
///
/// * `image` - The current image in Thorium
/// * `req` - The incoming image request from the manifest
/// * `conf` - The Thorctl config
/// * `editor_override` - Optional editor override from the CLI
pub(crate) async fn merge_image_interactive(
    image: &Image,
    req: &ImageRequest,
    conf: &CtlConf,
    editor_override: Option<&str>,
) -> Result<Option<ImageUpdate>, Error> {
    let current = MergeableImage::from(image.clone());
    let mut incoming = MergeableImage::from(req.clone());
    // a manifest never carries identity/server-managed fields and this merge
    // can't edit them, so mirror them from the current image to avoid spurious
    // conflicts (see [`mirror_non_editable_fields`])
    mirror_non_editable_fields(&mut incoming, &current);
    // serialize both to canonical (sorted-key) YAML so reordered maps (env,
    // triggers, …) don't surface as spurious conflicts
    let current_yaml = crate::utils::curated_yaml(&current, IMAGE_FIELD_ORDER)
        .map_err(|err| Error::new(format!("Failed to serialize current image to YAML: {err}")))?;
    let incoming_yaml = crate::utils::curated_yaml(&incoming, IMAGE_FIELD_ORDER)
        .map_err(|err| Error::new(format!("Failed to serialize incoming image to YAML: {err}")))?;
    // generate the conflict YAML
    let conflict_yaml = editor::generate_conflict_view(
        &current_yaml,
        &incoming_yaml,
        "Current (Thorium)",
        "Incoming (Manifest)",
    );
    // open the editor
    let editor_cmd = editor_override.unwrap_or(&conf.default_editor);
    let label = format!("{}-{}", image.group, image.name);
    let resolved: MergeableImage =
        match editor::editor_loop(&conflict_yaml, &label, editor_cmd).await? {
            Some(resolved) => resolved,
            None => return Ok(None),
        };
    // calculate update from the current image to the resolved state
    update::calculate_image_update_from_mergeable(image.clone(), resolved)
}

/// Resolve a pipeline merge conflict via the editor and return the resulting
/// pipeline update, or None if the user's edits result in no changes or they
/// cancelled
///
/// # Arguments
///
/// * `pipeline` - The current pipeline in Thorium
/// * `req` - The incoming pipeline request from the manifest
/// * `conf` - The Thorctl config
/// * `editor_override` - Optional editor override from the CLI
pub(crate) async fn merge_pipeline_interactive(
    pipeline: &Pipeline,
    req: &PipelineRequest,
    conf: &CtlConf,
    editor_override: Option<&str>,
) -> Result<Option<PipelineUpdate>, Error> {
    let current = MergeablePipeline::from(pipeline.clone());
    let incoming = MergeablePipeline::from(req.clone());
    // serialize both to canonical (sorted-key) YAML so reordered maps (triggers)
    // don't surface as spurious conflicts
    let current_yaml = crate::utils::curated_yaml(&current, PIPELINE_FIELD_ORDER).map_err(|err| {
        Error::new(format!(
            "Failed to serialize current pipeline to YAML: {err}"
        ))
    })?;
    let incoming_yaml =
        crate::utils::curated_yaml(&incoming, PIPELINE_FIELD_ORDER).map_err(|err| {
            Error::new(format!(
                "Failed to serialize incoming pipeline to YAML: {err}"
            ))
        })?;
    // generate the conflict YAML
    let conflict_yaml = editor::generate_conflict_view(
        &current_yaml,
        &incoming_yaml,
        "Current (Thorium)",
        "Incoming (Manifest)",
    );
    // open the editor
    let editor_cmd = editor_override.unwrap_or(&conf.default_editor);
    let label = format!("{}-{}", pipeline.group, pipeline.name);
    let resolved: MergeablePipeline =
        match editor::editor_loop(&conflict_yaml, &label, editor_cmd).await? {
            Some(resolved) => resolved,
            None => return Ok(None),
        };
    // calculate update from the current pipeline to the resolved state
    update::calculate_pipeline_update_from_mergeable(pipeline.clone(), resolved)
}

// ─── Batch Interactive Merge ─────────────────────────────────────────────────

/// Interactively handle existing resources that have changes, prompting the user
/// for each one to Edit (merge editor), Skip, Apply (accept incoming), or Quit
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to apply updates
/// * `existing` - Resources from the manifest that already exist in Thorium
/// * `conf` - The Thorctl config (used for the default editor)
/// * `editor_override` - Optional editor command that overrides `conf.default_editor`
/// * `progress` - The progress bar (suspended during interactive prompts)
/// * `journal` - The journal to snapshot pre-update state in for rollback
pub async fn interactive_merge<K: ImportKind>(
    thorium: &Thorium,
    existing: Vec<&Categorized<K>>,
    conf: &CtlConf,
    editor_override: Option<&str>,
    progress: &Bar,
    journal: &Journal,
) -> Result<ImportOutcome, Error> {
    // filter to only resources with actual changes
    let changed: Vec<_> = existing
        .into_iter()
        .filter(|item| {
            item.existing
                .as_ref()
                .is_some_and(|current| K::changed(current, &item.request))
        })
        .collect();
    if changed.is_empty() {
        return Ok(ImportOutcome::Completed);
    }
    for item in changed {
        // the filter above guarantees Some, but bind defensively rather than unwrap
        let Some(current) = item.existing.as_ref() else {
            continue;
        };
        let group = K::group(&item.request);
        let name = K::name(&item.request);
        // suspend the progress bar for interactive prompts
        let action = progress.suspend(|| prompt_merge_action(K::TITLE, group, name))?;
        match action {
            MergeAction::Edit => {
                // open the editor to resolve this resource's conflicts
                let update = progress
                    .suspend_async(K::merge_interactive(
                        current,
                        &item.request,
                        conf,
                        editor_override,
                    ))
                    .await?;
                if let Some(update) = update {
                    apply_merge_update::<K>(thorium, item, current, &update, journal).await?;
                } else {
                    println!(
                        "{} No changes detected for {} '{group}:{name}'",
                        "Skipped:".bright_blue(),
                        K::NOUN,
                    );
                }
            }
            MergeAction::Skip => {
                progress.info_anonymous(format!(
                    "Skipping {} '{}:{}'",
                    K::NOUN,
                    item.name.bright_yellow(),
                    item.version.bright_yellow()
                ));
            }
            MergeAction::Apply => {
                if let Some(update) = K::calculate_update(current.clone(), item.request.clone()) {
                    apply_merge_update::<K>(thorium, item, current, &update, journal).await?;
                }
            }
            MergeAction::Quit => {
                println!("Stopping further resource processing.");
                return Ok(ImportOutcome::Quit);
            }
        }
    }
    Ok(ImportOutcome::Completed)
}

/// Apply a resolved update to Thorium, snapshot the prior state for rollback, and
/// print the success line
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to apply the update
/// * `item` - The categorized resource being updated (source of group/name)
/// * `current` - The pre-update resource state to snapshot for rollback
/// * `update` - The resolved update payload to apply
/// * `journal` - The journal to record the pre-update snapshot in
async fn apply_merge_update<K: ImportKind>(
    thorium: &Thorium,
    item: &Categorized<K>,
    current: &K::Existing,
    update: &K::Update,
    journal: &Journal,
) -> Result<(), Error> {
    // resolve the resource's group and name for the update call and messages
    let group = K::group(&item.request);
    let name = K::name(&item.request);
    K::update(thorium, group, name, update).await.map_err(|err| {
        Error::new(format!(
            "Error updating {} '{}:{}': {}",
            K::NOUN,
            item.name,
            item.version,
            err
        ))
    })?;
    // snapshot the pre-update state so the update can be reverted
    K::record_updated(journal, current.clone());
    println!(
        "{} {} {}",
        K::TITLE.bright_green(),
        format!("'{group}:{name}'").yellow(),
        "updated successfully!".bright_green()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::imports::editor::generate_conflict_view;
    use std::collections::HashMap;
    use thorium::models::ImageRequest;

    /// Build a MergeableImage with the non-editable fields populated
    fn populated_mergeable_image() -> MergeableImage {
        let mut image = MergeableImage::from(ImageRequest::new("g", "n"));
        image.group = Some("g".to_string());
        image.name = Some("n".to_string());
        image.creator = Some("c".to_string());
        image.runtime = Some(1.0);
        image.bans = Some(HashMap::new());
        image
    }

    /// Every non-editable image field renders with the `*<field>*` static marker so
    /// users can't mistake it for editable
    #[test]
    fn non_editable_image_fields_are_static_marked() {
        let yaml = serde_norway::to_string(&populated_mergeable_image()).unwrap();
        for marked in ["*group*", "*name*", "*creator*", "*runtime*", "*bans*"] {
            assert!(yaml.contains(marked), "expected static marker {marked} in:\n{yaml}");
        }
    }

    /// Every serialized top-level field of a MergeableImage is listed in
    /// IMAGE_FIELD_ORDER, so the curated order stays complete as fields are added
    /// (the `curated_yaml` fallback still emits unlisted keys, but they'd lose their
    /// curated position — this guards against that drift)
    #[test]
    fn image_field_order_covers_serialized_fields() {
        let value = serde_json::to_value(populated_mergeable_image()).unwrap();
        for key in value.as_object().unwrap().keys() {
            // strip the *...* static marker before matching the plain-name list
            let plain = key.trim_matches('*');
            assert!(
                IMAGE_FIELD_ORDER.contains(&plain),
                "MergeableImage field '{key}' missing from IMAGE_FIELD_ORDER"
            );
        }
    }

    /// A description.md round-trip trims trailing whitespace while the live value
    /// keeps it; both must normalize to the same form so a fresh export diffs clean
    #[test]
    fn description_normalization_ignores_trailing_whitespace() {
        assert_eq!(
            normalize_description(Some("hello\n".to_string())),
            Some("hello".to_string())
        );
        assert_eq!(
            normalize_description(Some("hello".to_string())),
            Some("hello".to_string())
        );
        // whitespace-only and empty descriptions collapse to None on both sides
        assert_eq!(normalize_description(Some("  \n".to_string())), None);
        assert_eq!(normalize_description(Some(String::new())), None);
        assert_eq!(normalize_description(None), None);
    }

    /// Identity and server-managed fields present only on the current (Thorium)
    /// side must not surface as conflicts once mirrored onto the incoming side
    #[test]
    fn mirroring_removes_spurious_server_field_conflicts() {
        // the current image (from Thorium) carries identity + server-managed fields
        let mut current = MergeableImage::from(ImageRequest::new("static", "exiftool"));
        current.group = Some("static".to_string());
        current.name = Some("exiftool".to_string());
        current.creator = Some("test".to_string());
        current.runtime = Some(1.5);
        current.bans = Some(HashMap::new());
        // a manifest-sourced request omits all of those (editable fields match)
        let mut incoming = MergeableImage::from(ImageRequest::new("static", "exiftool"));

        // without mirroring, those fields diff and produce a conflict block
        let before = generate_conflict_view(
            &serde_norway::to_string(&current).unwrap(),
            &serde_norway::to_string(&incoming).unwrap(),
            "Current (Thorium)",
            "Incoming (Manifest)",
        );
        assert!(before.contains("<<<<<<<"), "expected a conflict before mirroring");

        // after mirroring, the two sides match and there is no conflict
        mirror_non_editable_fields(&mut incoming, &current);
        let after = generate_conflict_view(
            &serde_norway::to_string(&current).unwrap(),
            &serde_norway::to_string(&incoming).unwrap(),
            "Current (Thorium)",
            "Incoming (Manifest)",
        );
        assert!(
            !after.contains("<<<<<<<"),
            "unexpected conflict after mirroring:\n{after}"
        );
    }

    /// A request that omits the security context must surface the server default
    /// (an object), not `null`, so it matches an existing image and doesn't read as
    /// drift in `toolbox diff` or the merge view
    #[test]
    fn omitted_security_context_defaults_to_object() {
        let mergeable = MergeableImage::from(ImageRequest::new("static", "exiftool"));
        let value = serde_json::to_value(&mergeable).unwrap();
        assert!(
            value["security_context"].is_object(),
            "expected the default security context object, got {:?}",
            value["security_context"]
        );
    }

    /// Canonical YAML (used to build the conflict view) must still deserialize
    /// back into the mergeable types, since the editor parses the resolved text
    #[test]
    fn canonical_yaml_round_trips_mergeables() {
        let image = MergeableImage::from(ImageRequest::new("static", "exiftool"));
        let image_yaml = crate::utils::canonical_yaml(&image).unwrap();
        serde_norway::from_str::<MergeableImage>(&image_yaml)
            .expect("canonical image YAML must deserialize");

        let pipeline = MergeablePipeline::from(PipelineRequest::new(
            "static",
            "p",
            serde_json::json!([["a", "b"]]),
        ));
        let pipeline_yaml = crate::utils::canonical_yaml(&pipeline).unwrap();
        serde_norway::from_str::<MergeablePipeline>(&pipeline_yaml)
            .expect("canonical pipeline YAML must deserialize");
    }
}
