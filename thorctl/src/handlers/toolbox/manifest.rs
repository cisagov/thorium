//! The toolbox manifest structure

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thorium::Error;
use thorium::models::{ImageRequest, NetworkPolicyRequest, PipelineRequest};

/// A toolbox manifest – a description of pipelines and images that
/// can be imported into Thorium
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolboxManifest {
    /// The name of this toolbox
    pub name: String,
    /// The registry the images can be found at, if the toolbox declares a central one
    ///
    /// Optional: a toolbox documenting tools whose images live in various external
    /// registries leaves this unset and relies on each image config's own `image` url.
    /// Informational on import (the per-image `image` urls are authoritative).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<String>,
    /// A map of pipeline names to their details
    pub pipelines: HashMap<String, PipelineManifest>,
    /// A map of image names to their details
    pub images: HashMap<String, ImageManifest>,
    /// Whether this toolbox bundles container image tarballs alongside its configs
    ///
    /// When true, an import must load each `images/<name>/<name>.tar.gz`, push it to a
    /// target registry, and rewrite the image's url before creating it in Thorium.
    #[serde(default)]
    pub bundled_images: bool,
    /// The default registry base path that bundled images are pushed under on import
    ///
    /// Used as the fallback when `--image-path-prefix` is not given. Bundled images are
    /// pushed to `<image_path_prefix>/<group>/<name>:<tag>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_path_prefix: Option<String>,
}

/// A report of the image and pipeline versions dropped during lenient
/// validation, so the caller can warn the user about what was skipped
#[derive(Debug, Default)]
pub struct DroppedItems {
    /// `("name:version", reason)` for each removed image version
    pub images: Vec<(String, String)>,
    /// `("name:version", reasons)` for each removed pipeline version
    pub pipelines: Vec<(String, Vec<String>)>,
}

impl ToolboxManifest {
    /// Validate the manifest's intrinsic, group-independent structure, removing
    /// invalid image or pipeline versions instead of aborting
    ///
    /// Run this BEFORE any group override. Checks that every image and pipeline
    /// version has a resolved config, that every image a pipeline names (in its
    /// `images` map or its order) is present somewhere in the manifest, and that
    /// the image order parses. Group membership is checked separately, after the
    /// override, by [`Self::validate_group_coherence`].
    ///
    /// Must be called after configs have been resolved (via
    /// `shared::resolve_manifest_configs`). Returns a [`DroppedItems`] report
    /// describing every version removed and why.
    pub fn validate_structural(&mut self) -> DroppedItems {
        // the running tally of everything removed, returned for the caller to warn on
        let mut dropped = DroppedItems::default();
        // drop images without a resolved config first; this must precede the pipeline
        // pass because a pipeline that depended on a just-removed image then fails the
        // manifest-membership check below and is dropped too
        self.drop_unconfigured_images(&mut dropped);
        // drop pipelines whose config/order/image references are unusable
        self.drop_structurally_invalid_pipelines(&mut dropped);
        dropped
    }

    /// Validate that each pipeline's order references images present in the
    /// pipeline's group, dropping any that don't
    ///
    /// Run this AFTER the group override so it checks the final group layout.
    /// Returns a [`DroppedItems`] report describing every version removed.
    pub fn validate_group_coherence(&mut self) -> DroppedItems {
        // the running tally of everything removed, returned for the caller to warn on
        let mut dropped = DroppedItems::default();
        // drop pipelines whose order names an image not present in the pipeline's group
        self.drop_incoherent_pipelines(&mut dropped);
        dropped
    }

    /// Remove image versions that have no resolved config, recording each in `dropped`
    ///
    /// # Arguments
    ///
    /// * `dropped` - The report to record each removed image version in
    fn drop_unconfigured_images(&mut self, dropped: &mut DroppedItems) {
        // walk every image, pruning unconfigured versions; an empty image is then removed
        self.images.retain(|name, manifest| {
            // keep only versions that resolved a config; record the rest with the
            // config_from they failed to resolve so the user can see what was unreachable
            manifest.versions.retain(|version, v| {
                if v.config.is_none() {
                    dropped.images.push((
                        format!("{name}:{version}"),
                        format!("config not resolved (config_from: {:?})", v.config_from),
                    ));
                    false
                } else {
                    true
                }
            });
            // drop the image entry entirely once all of its versions are gone
            !manifest.versions.is_empty()
        });
    }

    /// Remove pipeline versions that lack a resolved config, have a malformed
    /// image order, or reference (in their `images` map or order) an image absent
    /// from the manifest, recording each in `dropped`
    ///
    /// # Arguments
    ///
    /// * `dropped` - The report to record each removed pipeline version in
    fn drop_structurally_invalid_pipelines(&mut self, dropped: &mut DroppedItems) {
        // build the lookup indexes once so the per-pipeline checks below are O(1)
        // instead of re-scanning every image version for each order entry/image ref.
        // an image reference may be a top-level manifest key (a tool name, e.g.
        // "sqlitediff") OR an image's config name (e.g. "sqldiff"), so both are indexed.
        let config_names: HashSet<&str> = self
            .images
            .values()
            .flat_map(|image_manifest| image_manifest.versions.values())
            .filter_map(|v| v.config.as_ref())
            .map(|config| config.name.as_str())
            .collect();
        let ref_names: HashSet<&str> = self
            .images
            .keys()
            .map(String::as_str)
            .chain(config_names.iter().copied())
            .collect();
        // index every (reference-name, version) pair an image is reachable under, again
        // accepting both the manifest key and the config name so a pipeline pinning either
        // form resolves; used to validate the pipeline's per-image version pins
        let mut ref_versions: HashSet<(&str, &str)> = HashSet::new();
        for (key, image_manifest) in &self.images {
            for (version, v) in &image_manifest.versions {
                ref_versions.insert((key.as_str(), version.as_str()));
                if let Some(config) = &v.config {
                    ref_versions.insert((config.name.as_str(), version.as_str()));
                }
            }
        }
        // gather the reasons each pipeline version is invalid into a side map keyed
        // "pipeline:version", since we only hold a shared borrow of self while iterating
        // and can't mutate the pipelines until the scan finishes
        let mut invalid: HashMap<String, Vec<String>> = HashMap::new();
        for (pipeline, pipeline_manifest) in &self.pipelines {
            for (version_name, version) in &pipeline_manifest.versions {
                let mut reasons = Vec::new();
                // every image the pipeline pins in its map must exist at the pinned version;
                // distinguish "image absent entirely" from "image present, wrong version"
                for (image_name, image_version) in &version.images {
                    if !ref_versions
                        .contains(&(image_name.as_str(), image_version.version.as_str()))
                    {
                        if ref_names.contains(image_name.as_str()) {
                            reasons.push(format!(
                                "requires image '{image_name}:{}' not in manifest",
                                image_version.version
                            ));
                        } else {
                            reasons.push(format!("requires image '{image_name}' not in manifest"));
                        }
                    }
                }
                // the config must have resolved and its order must parse before we can
                // check the order entries; a missing or malformed config is itself fatal
                match &version.config {
                    None => reasons.push("config not resolved".to_string()),
                    Some(config) => match config.deserialize_image_order() {
                        Err(err) => reasons.push(format!("malformed image order: {err}")),
                        Ok(order) => {
                            // order entries always use the config name, so check them against
                            // config_names only (not the manifest-key alias set)
                            let mut missing: Vec<&str> = Vec::new();
                            for image in order.iter().flat_map(|sub_order| sub_order.iter()) {
                                if !config_names.contains(image) {
                                    missing.push(image);
                                }
                            }
                            if !missing.is_empty() {
                                reasons.push(format!(
                                    "order references image(s) not in manifest: {missing:?}"
                                ));
                            }
                        }
                    },
                }
                // a version with any accumulated reason is marked invalid for removal
                if !reasons.is_empty() {
                    invalid.insert(format!("{pipeline}:{version_name}"), reasons);
                }
            }
        }
        // apply the removals now that the borrow of self.pipelines has been released
        Self::remove_invalid_pipeline_versions(&mut self.pipelines, &invalid, dropped);
    }

    /// Remove pipeline versions whose order references images not present in the
    /// pipeline's group, recording each in `dropped`
    ///
    /// # Arguments
    ///
    /// * `dropped` - The report to record each removed pipeline version in
    fn drop_incoherent_pipelines(&mut self, dropped: &mut DroppedItems) {
        // map of group -> set of image names present in the (surviving) manifest,
        // so the per-order-entry membership check below is O(1)
        let group_images = self
            .images
            .values()
            .flat_map(|image_manifest| image_manifest.versions.values())
            .filter_map(|v| v.config.as_ref())
            .fold(HashMap::<&str, HashSet<&str>>::new(), |mut map, config| {
                map.entry(&config.group).or_default().insert(&config.name);
                map
            });
        let mut invalid: HashMap<String, Vec<String>> = HashMap::new();
        for (pipeline, pipeline_manifest) in &self.pipelines {
            for (version_name, version) in &pipeline_manifest.versions {
                // structural validation already dropped versions without a config
                // or with a malformed order, so skip anything that won't parse
                let Some(config) = &version.config else {
                    continue;
                };
                let Ok(order) = config.deserialize_image_order() else {
                    continue;
                };
                // the images actually present in this pipeline's (post-override) group;
                // None when the group has no images at all, which makes every entry missing
                let images_in_group = group_images.get(config.group.as_str());
                // an order entry is incoherent when its image isn't in the pipeline's group,
                // even if that image exists elsewhere in the manifest under another group
                let mut missing: Vec<&str> = Vec::new();
                for image in order.iter().flat_map(|sub_order| sub_order.iter()) {
                    if !images_in_group.is_some_and(|names| names.contains(image)) {
                        missing.push(image);
                    }
                }
                if !missing.is_empty() {
                    invalid.insert(
                        format!("{pipeline}:{version_name}"),
                        vec![format!(
                            "order references image(s) not in group '{}': {missing:?}",
                            config.group
                        )],
                    );
                }
            }
        }
        // apply the removals now that the borrow of self.pipelines has been released
        Self::remove_invalid_pipeline_versions(&mut self.pipelines, &invalid, dropped);
    }

    /// Remove the pipeline versions named in `invalid` (keyed "pipeline:version"),
    /// dropping any pipeline left with no versions, and record them in `dropped`
    ///
    /// # Arguments
    ///
    /// * `pipelines` - The manifest's pipelines to remove the invalid versions from
    /// * `invalid` - Map of "pipeline:version" labels to the reasons each is invalid
    /// * `dropped` - The report to record each removed pipeline version in
    fn remove_invalid_pipeline_versions(
        pipelines: &mut HashMap<String, PipelineManifest>,
        invalid: &HashMap<String, Vec<String>>,
        dropped: &mut DroppedItems,
    ) {
        // nothing flagged means no mutation and no report entries
        if invalid.is_empty() {
            return;
        }
        // drop each flagged version, then drop any pipeline left with no versions
        pipelines.retain(|pipeline, pipeline_manifest| {
            pipeline_manifest
                .versions
                .retain(|version, _| !invalid.contains_key(&format!("{pipeline}:{version}")));
            !pipeline_manifest.versions.is_empty()
        });
        // record every removal (with its reasons) so the caller can warn the user
        for (key, reasons) in invalid {
            dropped.pipelines.push((key.clone(), reasons.clone()));
        }
    }

    /// Returns all of the groups the manifest expects to exist
    ///
    /// Collected from every resolved image and pipeline config; unconfigured
    /// versions contribute no group. The caller uses this to create any group
    /// missing from the target instance before importing.
    pub fn groups(&self) -> HashSet<String> {
        // union the group of every resolved pipeline config with that of every
        // resolved image config; the HashSet collapses duplicates
        self.pipelines
            .values()
            .flat_map(|pipeline_manifest| pipeline_manifest.versions.values())
            .filter_map(|v| v.config.as_ref())
            .map(|config| &config.group)
            .chain(
                self.images
                    .values()
                    .flat_map(|image_manifest| image_manifest.versions.values())
                    .filter_map(|v| v.config.as_ref())
                    .map(|config| &config.group),
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
        // own the target group string once so each assignment below can clone from it
        let group = group.to_string();
        // chain a mutable reference to every resolved pipeline and image config's group,
        // then point each at the target; unconfigured versions have no group to set
        self.pipelines
            .values_mut()
            .flat_map(|pipeline_manifest| pipeline_manifest.versions.values_mut())
            .filter_map(|v| v.config.as_mut())
            .map(|config| &mut config.group)
            .chain(
                self.images
                    .values_mut()
                    .flat_map(|image_manifest| image_manifest.versions.values_mut())
                    .filter_map(|v| v.config.as_mut())
                    .map(|config| &mut config.group),
            )
            // set each group reference to the given group
            .for_each(|group_ref| group_ref.clone_from(&group));
        self
    }

    // ─── Collision detection & resolution ────────────────────────────────────

    /// Snapshot each image/pipeline version's current group, keyed by
    /// `(manifest_key, version)`, so collision resolution can later disambiguate
    /// which pipeline wanted which image variant by its original group
    ///
    /// Call this BEFORE [`Self::override_group`] collapses the groups.
    pub fn capture_source_groups(&self) -> SourceGroups {
        // record each resolved image version's group keyed by (manifest_key, version);
        // unconfigured versions are skipped because they have no group to remember
        let images = self
            .images
            .iter()
            .flat_map(|(key, manifest)| {
                manifest.versions.iter().filter_map(move |(version, v)| {
                    v.config
                        .as_ref()
                        .map(|c| ((key.clone(), version.clone()), c.group.clone()))
                })
            })
            .collect();
        // same snapshot for pipeline versions, so a renamed image's dependents can be
        // matched back to the variant they originally wanted by their own source group
        let pipelines = self
            .pipelines
            .iter()
            .flat_map(|(key, manifest)| {
                manifest.versions.iter().filter_map(move |(version, v)| {
                    v.config
                        .as_ref()
                        .map(|c| ((key.clone(), version.clone()), c.group.clone()))
                })
            })
            .collect();
        SourceGroups { images, pipelines }
    }

    /// Find sets of image versions that resolve to the same `(group, name)`
    /// Thorium identity and would therefore overwrite each other on import
    ///
    /// # Arguments
    ///
    /// * `sources` - The pre-override group snapshot used to tag each colliding member
    pub fn detect_image_collisions(&self, sources: &SourceGroups) -> Result<Vec<Collision>, Error> {
        // group members by their post-override (group, name) identity; every bucket with
        // more than one member is an overwrite collision
        let mut buckets: HashMap<(String, String), Vec<(CollisionMember, serde_json::Value)>> =
            HashMap::new();
        for (key, manifest) in &self.images {
            for (version, v) in &manifest.versions {
                // unconfigured versions have no identity to collide on
                let Some(config) = &v.config else {
                    continue;
                };
                // recover the pre-override group from the snapshot; fall back to the
                // current group when this version wasn't captured (e.g. no override ran)
                let source_group = sources
                    .images
                    .get(&(key.clone(), version.clone()))
                    .cloned()
                    .unwrap_or_else(|| config.group.clone());
                let member = CollisionMember {
                    manifest_key: key.clone(),
                    version: version.clone(),
                    source_group,
                };
                // a serialization failure must not be coerced to a shared `Null`,
                // or two distinct configs would compare "identical" and be wrongly
                // auto-deduped — surface it instead
                let json = serde_json::to_value(config).map_err(|err| {
                    Error::new(format!(
                        "Failed to serialize image '{}' for collision check: {err}",
                        config.name
                    ))
                })?;
                // file the member under its identity alongside its serialized config so
                // buckets_to_collisions can tell a pure duplicate from a real conflict
                buckets
                    .entry((config.group.clone(), config.name.clone()))
                    .or_default()
                    .push((member, json));
            }
        }
        // collapse the buckets into the multi-member collisions only
        Ok(Self::buckets_to_collisions(buckets))
    }

    /// Find sets of pipeline versions that resolve to the same `(group, name)`
    /// Thorium identity and would therefore overwrite each other on import
    ///
    /// # Arguments
    ///
    /// * `sources` - The pre-override group snapshot used to tag each colliding member
    pub fn detect_pipeline_collisions(
        &self,
        sources: &SourceGroups,
    ) -> Result<Vec<Collision>, Error> {
        // group members by their post-override (group, name) identity; every bucket with
        // more than one member is an overwrite collision
        let mut buckets: HashMap<(String, String), Vec<(CollisionMember, serde_json::Value)>> =
            HashMap::new();
        for (key, manifest) in &self.pipelines {
            for (version, v) in &manifest.versions {
                // unconfigured versions have no identity to collide on
                let Some(config) = &v.config else {
                    continue;
                };
                // recover the pre-override group from the snapshot; fall back to the
                // current group when this version wasn't captured (e.g. no override ran)
                let source_group = sources
                    .pipelines
                    .get(&(key.clone(), version.clone()))
                    .cloned()
                    .unwrap_or_else(|| config.group.clone());
                let member = CollisionMember {
                    manifest_key: key.clone(),
                    version: version.clone(),
                    source_group,
                };
                // see detect_image_collisions: never coerce a serialization failure
                // into a shared `Null` that would falsely read as a duplicate
                let json = serde_json::to_value(config).map_err(|err| {
                    Error::new(format!(
                        "Failed to serialize pipeline '{}' for collision check: {err}",
                        config.name
                    ))
                })?;
                // file the member under its identity alongside its serialized config so
                // buckets_to_collisions can tell a pure duplicate from a real conflict
                buckets
                    .entry((config.group.clone(), config.name.clone()))
                    .or_default()
                    .push((member, json));
            }
        }
        // collapse the buckets into the multi-member collisions only
        Ok(Self::buckets_to_collisions(buckets))
    }

    /// Turn `(group, name)` buckets into deterministic [`Collision`]s, keeping
    /// only buckets with more than one member
    ///
    /// # Arguments
    ///
    /// * `buckets` - Map of `(group, name)` identity to its members and their serialized configs
    fn buckets_to_collisions(
        buckets: HashMap<(String, String), Vec<(CollisionMember, serde_json::Value)>>,
    ) -> Vec<Collision> {
        let mut collisions = Vec::new();
        for ((group, name), members) in buckets {
            // a single occupant of an identity isn't a collision
            if members.len() < 2 {
                continue;
            }
            // a bucket where every member's config is byte-identical is a pure
            // duplicate, safe to de-dupe without asking the user to pick
            let first = &members[0].1;
            let identical = members.iter().all(|(_, json)| json == first);
            // the serialized configs were only needed for the identical check; drop them
            let mut members: Vec<CollisionMember> = members.into_iter().map(|(m, _)| m).collect();
            // sort by (manifest_key, version) so the first member is a stable canonical
            // choice and the rendered collision is reproducible across runs
            members
                .sort_by(|a, b| (&a.manifest_key, &a.version).cmp(&(&b.manifest_key, &b.version)));
            collisions.push(Collision {
                group,
                name,
                members,
                identical,
            });
        }
        // sort the collisions themselves so the import/diff output order is deterministic
        collisions.sort_by(|a, b| (&a.group, &a.name).cmp(&(&b.group, &b.name)));
        collisions
    }

    /// Suggest a unique new name for a colliding image member: `<name>-<version>`,
    /// with a numeric suffix appended if that is already taken in the group
    ///
    /// # Arguments
    ///
    /// * `collision` - The collision the member belongs to (supplies the group and base name)
    /// * `member` - The colliding member to suggest a new name for (supplies the version)
    pub fn suggested_image_rename(
        &self,
        collision: &Collision,
        member: &CollisionMember,
    ) -> String {
        // base the suggestion on "<name>-<version>" then ensure it's free in the group
        self.unique_image_name(
            &collision.group,
            &format!("{}-{}", collision.name, member.version),
        )
    }

    /// Suggest a unique new name for a colliding pipeline member: `<name>-<version>`,
    /// with a numeric suffix appended if that is already taken in the group
    ///
    /// # Arguments
    ///
    /// * `collision` - The collision the member belongs to (supplies the group and base name)
    /// * `member` - The colliding member to suggest a new name for (supplies the version)
    pub fn suggested_pipeline_rename(
        &self,
        collision: &Collision,
        member: &CollisionMember,
    ) -> String {
        // base the suggestion on "<name>-<version>" then ensure it's free in the group
        self.unique_pipeline_name(
            &collision.group,
            &format!("{}-{}", collision.name, member.version),
        )
    }

    /// Every image config's `(group, name)` identity across all versions
    fn image_identities(&self) -> impl Iterator<Item = (&str, &str)> {
        // flatten every resolved image version down to its Thorium identity; unconfigured
        // versions are skipped because they have no identity yet
        self.images
            .values()
            .flat_map(|manifest| manifest.versions.values())
            .filter_map(|v| v.config.as_ref())
            .map(|config| (config.group.as_str(), config.name.as_str()))
    }

    /// Every pipeline config's `(group, name)` identity across all versions
    fn pipeline_identities(&self) -> impl Iterator<Item = (&str, &str)> {
        // flatten every resolved pipeline version down to its Thorium identity; unconfigured
        // versions are skipped because they have no identity yet
        self.pipelines
            .values()
            .flat_map(|manifest| manifest.versions.values())
            .filter_map(|v| v.config.as_ref())
            .map(|config| (config.group.as_str(), config.name.as_str()))
    }

    /// A name `taken` reports as free, appending `-2`, `-3`, … to `base` until one
    /// is found
    ///
    /// `taken` is the caller's freshness test (it decides what "already used"
    /// means); the `(2..)` range is unbounded, so a free name is always found.
    ///
    /// # Arguments
    ///
    /// * `base` - The preferred name to return unchanged when it is free
    /// * `taken` - Predicate reporting whether a candidate name is already used
    fn unique_name(base: &str, mut taken: impl FnMut(&str) -> bool) -> String {
        // prefer the unsuffixed name when it is already free
        if !taken(base) {
            return base.to_string();
        }
        // otherwise try "base-2", "base-3", … the unbounded range guarantees a hit so the
        // expect can never fire, but find returns Option and must be unwrapped
        (2..)
            .map(|n| format!("{base}-{n}"))
            .find(|candidate| !taken(candidate))
            .expect("unbounded range always yields a free name")
    }

    /// A name not already used by any image in `group`
    ///
    /// # Arguments
    ///
    /// * `group` - The group to check existing image names against
    /// * `base` - The preferred name, suffixed until it is free in the group
    fn unique_image_name(&self, group: &str, base: &str) -> String {
        // "taken" means some image already owns this name in the same group; cross-group
        // names never conflict because Thorium identity is (group, name)
        Self::unique_name(base, |candidate| {
            self.image_identities()
                .any(|(g, n)| g == group && n == candidate)
        })
    }

    /// A name not already used by any pipeline in `group`
    ///
    /// # Arguments
    ///
    /// * `group` - The group to check existing pipeline names against
    /// * `base` - The preferred name, suffixed until it is free in the group
    fn unique_pipeline_name(&self, group: &str, base: &str) -> String {
        // "taken" means some pipeline already owns this name in the same group; cross-group
        // names never conflict because Thorium identity is (group, name)
        Self::unique_name(base, |candidate| {
            self.pipeline_identities()
                .any(|(g, n)| g == group && n == candidate)
        })
    }

    /// The set of image names currently used in `group`
    ///
    /// Used to validate a user-supplied rename so it can't introduce a fresh
    /// collision with another image in the same group.
    ///
    /// # Arguments
    ///
    /// * `group` - The group whose image names to collect
    pub fn image_names_in_group(&self, group: &str) -> HashSet<String> {
        // keep only the names whose identity is in the requested group
        self.image_identities()
            .filter(|(g, _)| *g == group)
            .map(|(_, n)| n.to_string())
            .collect()
    }

    /// The set of pipeline names currently used in `group`
    ///
    /// Used to validate a user-supplied rename so it can't introduce a fresh
    /// collision with another pipeline in the same group.
    ///
    /// # Arguments
    ///
    /// * `group` - The group whose pipeline names to collect
    pub fn pipeline_names_in_group(&self, group: &str) -> HashSet<String> {
        // keep only the names whose identity is in the requested group
        self.pipeline_identities()
            .filter(|(g, _)| *g == group)
            .map(|(_, n)| n.to_string())
            .collect()
    }

    /// De-duplicate a pure-duplicate image collision by keeping the first member
    /// and removing the rest (their configs are identical, so dependents are
    /// unaffected)
    ///
    /// # Arguments
    ///
    /// * `collision` - The identical-config collision whose extra members to remove
    pub fn dedupe_image_collision(&mut self, collision: &Collision) {
        // members[0] is the sorted canonical keeper; remove every other version, which is
        // safe only because the caller guarantees this collision is byte-identical
        for member in collision.members.iter().skip(1) {
            self.remove_image_version(&member.manifest_key, &member.version);
        }
    }

    /// De-duplicate a pure-duplicate pipeline collision by keeping the first member
    ///
    /// # Arguments
    ///
    /// * `collision` - The identical-config collision whose extra members to remove
    pub fn dedupe_pipeline_collision(&mut self, collision: &Collision) {
        // members[0] is the sorted canonical keeper; remove every other version, which is
        // safe only because the caller guarantees this collision is byte-identical
        for member in collision.members.iter().skip(1) {
            self.remove_pipeline_version(&member.manifest_key, &member.version);
        }
    }

    /// Remove a single image version, dropping the image entry if it becomes empty
    ///
    /// # Arguments
    ///
    /// * `manifest_key` - The top-level image key the version lives under
    /// * `version` - The version label to remove
    fn remove_image_version(&mut self, manifest_key: &str, version: &str) {
        // a missing key is a no-op; only touch the entry when it exists
        if let Some(manifest) = self.images.get_mut(manifest_key) {
            // drop the one version, then drop the whole entry if that emptied it so no
            // version-less image lingers in the manifest
            manifest.versions.remove(version);
            if manifest.versions.is_empty() {
                self.images.remove(manifest_key);
            }
        }
    }

    /// Remove a single pipeline version, dropping the pipeline entry if it becomes empty
    ///
    /// # Arguments
    ///
    /// * `manifest_key` - The top-level pipeline key the version lives under
    /// * `version` - The version label to remove
    fn remove_pipeline_version(&mut self, manifest_key: &str, version: &str) {
        // a missing key is a no-op; only touch the entry when it exists
        if let Some(manifest) = self.pipelines.get_mut(manifest_key) {
            // drop the one version, then drop the whole entry if that emptied it so no
            // version-less pipeline lingers in the manifest
            manifest.versions.remove(version);
            if manifest.versions.is_empty() {
                self.pipelines.remove(manifest_key);
            }
        }
    }

    /// Remove every image version matching `(group, name)` along with every
    /// pipeline that references that image, returning the dropped pipeline labels
    ///
    /// Used for the non-interactive (or user-chosen) skip resolution: the whole
    /// colliding identity is dropped, so any pipeline that needed it is dropped too.
    ///
    /// # Arguments
    ///
    /// * `group` - The group of the image identity to remove
    /// * `name` - The name of the image identity to remove
    pub fn remove_image_identity_and_dependents(&mut self, group: &str, name: &str) -> Vec<String> {
        // first drop every image version matching the (group, name) identity; an
        // unconfigured version (no group/name) can never match, so it is kept
        self.images.retain(|_key, manifest| {
            manifest.versions.retain(|_version, v| {
                v.config
                    .as_ref()
                    .is_none_or(|c| !(c.group == group && c.name == name))
            });
            !manifest.versions.is_empty()
        });
        // then drop every pipeline version that referenced the now-removed image, since
        // it can no longer run; collect their labels to report which dependents fell
        let mut dropped_pipelines = Vec::new();
        self.pipelines.retain(|pipeline_key, manifest| {
            manifest.versions.retain(|version_name, version| {
                // image references are bare names resolved within the pipeline's own
                // group, so only a pipeline in the removed image's group could have
                // referenced it; a same-named image in another group is unrelated and
                // its dependents must not be dropped
                let in_same_group = version
                    .config
                    .as_ref()
                    .is_some_and(|config| config.group == group);
                if in_same_group && pipeline_references_image(version, name) {
                    dropped_pipelines.push(format!("{pipeline_key}:{version_name}"));
                    false
                } else {
                    true
                }
            });
            !manifest.versions.is_empty()
        });
        // sort so the dropped-dependents list is deterministic for the caller's warning
        dropped_pipelines.sort();
        dropped_pipelines
    }

    /// Remove every pipeline version matching `(group, name)`
    ///
    /// # Arguments
    ///
    /// * `group` - The group of the pipeline identity to remove
    /// * `name` - The name of the pipeline identity to remove
    pub fn remove_pipeline_identity(&mut self, group: &str, name: &str) {
        // drop every pipeline version matching the (group, name) identity; an unconfigured
        // version (no group/name) can never match, so it is kept. no cascade is needed
        // because nothing else in the manifest references a pipeline by name
        self.pipelines.retain(|_key, manifest| {
            manifest.versions.retain(|_version, v| {
                v.config
                    .as_ref()
                    .is_none_or(|c| !(c.group == group && c.name == name))
            });
            !manifest.versions.is_empty()
        });
    }

    /// Rename one colliding image member to `new_name` and repoint every pipeline
    /// that wanted *that* member's variant
    ///
    /// The version is split into its own top-level entry under `new_name` (with
    /// `config.name` updated). Pipelines are disambiguated by pinned image version
    /// when the colliding members have distinct versions, otherwise by their
    /// original (pre-override) group.
    ///
    /// # Arguments
    ///
    /// * `collision` - The collision being resolved (supplies the original name)
    /// * `member` - The colliding member to rename
    /// * `new_name` - The new name to give the member
    /// * `sources` - The pre-override group snapshot used to disambiguate dependents
    pub fn rename_image_member(
        &mut self,
        collision: &Collision,
        member: &CollisionMember,
        new_name: &str,
        sources: &SourceGroups,
    ) {
        // the identity name every dependent pipeline currently points at
        let old_name = collision.name.clone();
        // pull just the colliding version out of its current entry and re-file it under
        // new_name, so the other versions of that entry keep the original name
        if let Some(manifest) = self.images.get_mut(&member.manifest_key)
            && let Some(mut version) = manifest.versions.remove(&member.version)
        {
            // rewrite the config name so the imported image carries the new identity
            if let Some(config) = &mut version.config {
                config.name = new_name.to_string();
            }
            // remove the source entry if pulling that version emptied it
            if manifest.versions.is_empty() {
                self.images.remove(&member.manifest_key);
            }
            // file the moved version under the new top-level key, creating it if needed
            self.images
                .entry(new_name.to_string())
                .or_insert_with(|| ImageManifest {
                    versions: HashMap::new(),
                })
                .versions
                .insert(member.version.clone(), version);
        }
        // do the colliding members carry distinct version labels? if so, a pipeline's
        // pinned image version tells us which variant it wanted; if not, fall back to
        // matching the pipeline's original group to the member's source group
        // HashSet::insert returns false on a repeat, so `all` is true exactly when every
        // member carries a distinct version label; that decides the disambiguation strategy
        let mut seen = HashSet::new();
        let versions_distinct = collision.members.iter().all(|m| seen.insert(&m.version));
        // repoint only the dependents that wanted *this* renamed variant; the others keep
        // pointing at old_name (which now belongs to a different surviving member)
        for (pipeline_key, manifest) in self.pipelines.iter_mut() {
            for (version_name, version) in manifest.versions.iter_mut() {
                // skip pipelines that never referenced the colliding image at all
                if !pipeline_references_image(version, &old_name) {
                    continue;
                }
                // distinct versions: the pipeline's pinned image version identifies the
                // variant. same version: fall back to matching the pipeline's pre-override
                // source group to the renamed member's source group
                let wants = if versions_distinct {
                    version
                        .images
                        .get(&old_name)
                        .is_some_and(|pi| pi.version == member.version)
                } else {
                    sources
                        .pipelines
                        .get(&(pipeline_key.clone(), version_name.clone()))
                        == Some(&member.source_group)
                };
                // rewrite both the images map and the order so the pipeline tracks the move
                if wants {
                    rewrite_pipeline_image_ref(version, &old_name, new_name);
                }
            }
        }
    }

    /// Rename one colliding pipeline member to `new_name` (pipelines aren't
    /// referenced by name elsewhere in the manifest, so there is no cascade)
    ///
    /// # Arguments
    ///
    /// * `member` - The colliding pipeline member to rename
    /// * `new_name` - The new name to give the member
    pub fn rename_pipeline_member(&mut self, member: &CollisionMember, new_name: &str) {
        // pull just the colliding version out of its current entry and re-file it under
        // new_name, leaving the entry's other versions under the original name
        if let Some(manifest) = self.pipelines.get_mut(&member.manifest_key)
            && let Some(mut version) = manifest.versions.remove(&member.version)
        {
            // rewrite the config name so the imported pipeline carries the new identity
            if let Some(config) = &mut version.config {
                config.name = new_name.to_string();
            }
            // remove the source entry if pulling that version emptied it
            if manifest.versions.is_empty() {
                self.pipelines.remove(&member.manifest_key);
            }
            // file the moved version under the new top-level key, creating it if needed
            self.pipelines
                .entry(new_name.to_string())
                .or_insert_with(|| PipelineManifest {
                    versions: HashMap::new(),
                })
                .versions
                .insert(member.version.clone(), version);
        }
    }
}

/// A snapshot of every image/pipeline version's group before a group override,
/// used to disambiguate collisions (see [`ToolboxManifest::capture_source_groups`])
#[derive(Debug, Default)]
pub struct SourceGroups {
    /// `(manifest_key, version)` -> pre-override group for images
    images: HashMap<(String, String), String>,
    /// `(manifest_key, version)` -> pre-override group for pipelines
    pipelines: HashMap<(String, String), String>,
}

/// A set of image (or pipeline) versions that resolve to the same `(group, name)`
/// Thorium identity and would overwrite each other on import
#[derive(Debug)]
pub struct Collision {
    /// The shared (post-override) group
    pub group: String,
    /// The shared Thorium name
    pub name: String,
    /// The colliding versions, sorted deterministically
    pub members: Vec<CollisionMember>,
    /// Whether every member's config is byte-identical (a pure duplicate)
    pub identical: bool,
}

/// One member of a [`Collision`]
#[derive(Debug, Clone)]
pub struct CollisionMember {
    /// The top-level manifest key this version lives under
    pub manifest_key: String,
    /// The version label
    pub version: String,
    /// The version's group before any override, for disambiguation
    pub source_group: String,
}

/// Whether a pipeline version references the given image name in its `images`
/// map or its order
///
/// # Arguments
///
/// * `version` - The pipeline version to inspect
/// * `name` - The image name to look for
fn pipeline_references_image(version: &PipelineVersion, name: &str) -> bool {
    // a hit in the images map is conclusive without parsing the order
    if version.images.contains_key(name) {
        return true;
    }
    // otherwise the order may still name it; a missing or unparsable order counts as no
    // reference (is_some_and short-circuits both the None and the Err to false)
    version
        .config
        .as_ref()
        .and_then(|config| config.deserialize_image_order().ok())
        .is_some_and(|order| order.iter().flatten().any(|image| *image == name))
}

/// Rewrite a pipeline version's references to `old_name` to point at `new_name`,
/// in both the `images` map and the image order
///
/// # Arguments
///
/// * `version` - The pipeline version to rewrite in place
/// * `old_name` - The image name currently referenced
/// * `new_name` - The image name to replace it with
fn rewrite_pipeline_image_ref(version: &mut PipelineVersion, old_name: &str, new_name: &str) {
    // move the images-map pin (if any) from the old key to the new one, preserving its
    // pinned version; the map may legitimately lack the key if only the order names it
    if let Some(pipeline_image) = version.images.remove(old_name) {
        version.images.insert(new_name.to_string(), pipeline_image);
    }
    if let Some(config) = &mut version.config {
        // deserialize the order into owned strings so the borrow of config is released
        // before we reassign config.order below; an unparsable order yields None and is
        // left untouched
        let rewritten: Option<Vec<Vec<String>>> =
            config.deserialize_image_order().ok().map(|order| {
                order
                    .into_iter()
                    .map(|stage| {
                        stage
                            .into_iter()
                            // swap the renamed image, leave every other entry as-is
                            .map(|image| {
                                if image == old_name {
                                    new_name.to_string()
                                } else {
                                    image.to_string()
                                }
                            })
                            .collect()
                    })
                    .collect()
            });
        // write the rewritten order back as JSON; a serialization failure leaves the
        // original order in place rather than corrupting it
        if let Some(new_order) = rewritten
            && let Ok(value) = serde_json::to_value(new_order)
        {
            config.order = value;
        }
    }
}

/// A pipeline entry in a toolbox manifest: its versions keyed by version label
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineManifest {
    /// A map of pipeline versions to their details
    #[serde(flatten)]
    pub versions: HashMap<String, PipelineVersion>,
}

/// Details for a specific pipeline version
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineVersion {
    /// The tool directory (where this pipeline's `manifest.toml` lives), relative to the
    /// `toolbox.json`'s location. Lets `export` find where a pipeline already lives and update it in
    /// place; empty for older toolboxes that predate the field (the default layout is used then).
    #[serde(default)]
    pub dir: String,
    /// A description of the pipeline for the purpose of the toolbox, not for
    /// Thorium itself
    pub description: String,
    /// A map of image names to their info for the pipeline
    pub images: HashMap<String, PipelineImage>,
    /// URL to fetch the pipeline config from (alternative to inline config)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_from: Option<String>,
    /// The pipeline's Thorium configuration (inline or resolved from config_from)
    #[serde(default)]
    pub config: Option<PipelineRequest>,
}

/// A pipeline's reference to an image version it runs
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineImage {
    /// The version of the image this pipeline expects
    pub version: String,
}

/// An image entry in a toolbox manifest: its versions keyed by version label
#[derive(Debug, Serialize, Deserialize)]
pub struct ImageManifest {
    /// A map of image versions to their details
    #[serde(flatten)]
    pub versions: HashMap<String, ImageVersion>,
}

/// Details for a specific image version
#[derive(Debug, Serialize, Deserialize)]
pub struct ImageVersion {
    /// The tool directory (where this image's `manifest.toml` and any bundled tarball live),
    /// relative to the `toolbox.json`'s location. Used to find a bundled image's tarball; empty
    /// for older toolboxes that predate the field, in which case the bundled-image lookup falls
    /// back to the default `images/<name>` layout.
    #[serde(default)]
    pub dir: String,
    /// The image's build path relative to the toolbox manifest's location
    pub build_path: String,
    /// URL to fetch the image config from (alternative to inline config)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_from: Option<String>,
    /// The image's Thorium configuration (inline or resolved from config_from)
    #[serde(default)]
    pub config: Option<ImageRequest>,
    /// URLs to fetch network policy definitions from (resolved into
    /// `network_policies` before import, like `config_from`)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub network_policies_from: Vec<String>,
    /// The network policies this image references, bundled so an import can
    /// create them in the target instance when missing
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub network_policies: Vec<NetworkPolicyRequest>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use thorium::models::{ImageRequest, PipelineRequest};

    /// An image version whose config carries a distinguishing `image` url, so two
    /// versions of the same `(group, name)` aren't byte-identical
    ///
    /// # Arguments
    ///
    /// * `group` - The group for the image config
    /// * `name` - The name for the image config
    /// * `distinct` - The image url that makes this config distinguishable
    fn image_version(group: &str, name: &str, distinct: &str) -> ImageVersion {
        // build the base request, then stamp a distinguishing image url so two versions of
        // the same (group, name) serialize differently and aren't treated as duplicates
        let mut config = ImageRequest::new(group, name);
        config.image = Some(distinct.to_string());
        ImageVersion {
            dir: String::new(),
            build_path: String::new(),
            config_from: None,
            config: Some(config),
            network_policies_from: Vec::new(),
            network_policies: Vec::new(),
        }
    }

    /// Build an image manifest with a single resolved `latest` version in `group`
    ///
    /// # Arguments
    ///
    /// * `group` - The group for the image config
    /// * `name` - The name for the image config
    fn image(group: &str, name: &str) -> ImageManifest {
        ImageManifest {
            versions: HashMap::from([("latest".to_string(), image_version(group, name, name))]),
        }
    }

    /// Build an image manifest with one entry per version label (each config distinct)
    ///
    /// # Arguments
    ///
    /// * `group` - The group for each image config
    /// * `name` - The name for each image config
    /// * `versions` - The version labels to build an entry for
    fn versioned_image(group: &str, name: &str, versions: &[&str]) -> ImageManifest {
        ImageManifest {
            // give each version a "name:version" image url so the variants are distinct
            versions: versions
                .iter()
                .map(|v| {
                    (
                        (*v).to_string(),
                        image_version(group, name, &format!("{name}:{v}")),
                    )
                })
                .collect(),
        }
    }

    /// Build an image manifest whose single `latest` version never resolved a config
    fn unresolved_image() -> ImageManifest {
        ImageManifest {
            versions: HashMap::from([(
                "latest".to_string(),
                ImageVersion {
                    dir: String::new(),
                    build_path: String::new(),
                    config_from: Some("https://example/cfg.json".to_string()),
                    config: None,
                    network_policies_from: Vec::new(),
                    network_policies: Vec::new(),
                },
            )]),
        }
    }

    /// Build a pipeline manifest with a single resolved `latest` version, whose
    /// `images` map pins each named image to version `latest`
    ///
    /// # Arguments
    ///
    /// * `group` - The group for the pipeline config
    /// * `name` - The name for the pipeline config
    /// * `order` - The pipeline's image order as a JSON value
    /// * `images` - The image names to pin to version `latest`
    fn pipeline(
        group: &str,
        name: &str,
        order: serde_json::Value,
        images: &[&str],
    ) -> PipelineManifest {
        // pin every named image to "latest" and defer to the pinned builder
        let pins: Vec<(&str, &str)> = images.iter().map(|i| (*i, "latest")).collect();
        pipeline_pinned(group, name, order, &pins)
    }

    /// Build a pipeline manifest pinning each named image to a specific version
    ///
    /// # Arguments
    ///
    /// * `group` - The group for the pipeline config
    /// * `name` - The name for the pipeline config
    /// * `order` - The pipeline's image order as a JSON value
    /// * `images` - The `(image, version)` pins for the pipeline's images map
    fn pipeline_pinned(
        group: &str,
        name: &str,
        order: serde_json::Value,
        images: &[(&str, &str)],
    ) -> PipelineManifest {
        // turn the (image, version) pairs into the pipeline's images map
        let image_map = images
            .iter()
            .map(|(image, version)| {
                (
                    (*image).to_string(),
                    PipelineImage {
                        version: (*version).to_string(),
                    },
                )
            })
            .collect();
        PipelineManifest {
            versions: HashMap::from([(
                "latest".to_string(),
                PipelineVersion {
                    dir: String::new(),
                    description: String::new(),
                    images: image_map,
                    config_from: None,
                    config: Some(PipelineRequest::new(group, name, order)),
                },
            )]),
        }
    }

    /// Assemble a toolbox manifest from the given images and pipelines
    ///
    /// # Arguments
    ///
    /// * `images` - The `(key, manifest)` image entries to include
    /// * `pipelines` - The `(key, manifest)` pipeline entries to include
    fn manifest(
        images: Vec<(&str, ImageManifest)>,
        pipelines: Vec<(&str, PipelineManifest)>,
    ) -> ToolboxManifest {
        // key each entry by its given name; the remaining fields are inert test defaults
        ToolboxManifest {
            name: "t".to_string(),
            registry: Some("r".to_string()),
            images: images
                .into_iter()
                .map(|(n, m)| (n.to_string(), m))
                .collect(),
            pipelines: pipelines
                .into_iter()
                .map(|(n, m)| (n.to_string(), m))
                .collect(),
            bundled_images: false,
            image_path_prefix: None,
        }
    }

    /// The deserialized image order of a pipeline's `latest` version, as owned strings
    ///
    /// # Arguments
    ///
    /// * `m` - The manifest to read the pipeline from
    /// * `pipeline_key` - The top-level key of the pipeline to read
    fn order_of(m: &ToolboxManifest, pipeline_key: &str) -> Vec<Vec<String>> {
        // reach into the "latest" version's resolved config, parse its order, and own the
        // borrowed entries so callers can compare against literal Vec<Vec<String>>
        m.pipelines[pipeline_key].versions["latest"]
            .config
            .as_ref()
            .unwrap()
            .deserialize_image_order()
            .unwrap()
            .into_iter()
            .map(|stage| stage.into_iter().map(String::from).collect())
            .collect()
    }

    // ─── structural / coherence validation ───────────────────────────────────

    /// A manifest whose pipelines reference present, correctly-grouped images
    /// survives both validation passes untouched
    #[test]
    fn keeps_valid_manifest() {
        let mut m = manifest(
            vec![("a", image("g", "a")), ("b", image("g", "b"))],
            vec![("p", pipeline("g", "p", json!(["a", "b"]), &["a", "b"]))],
        );
        assert!(m.validate_structural().pipelines.is_empty());
        assert!(m.validate_group_coherence().pipelines.is_empty());
        assert!(m.pipelines.contains_key("p"));
        assert_eq!(m.images.len(), 2);
    }

    /// Structural validation drops a pipeline that references an image absent
    /// from the manifest
    #[test]
    fn structural_drops_pipeline_referencing_missing_image() {
        let mut m = manifest(
            vec![("a", image("g", "a"))],
            vec![("p", pipeline("g", "p", json!(["a"]), &["a", "missing"]))],
        );
        let dropped = m.validate_structural();
        assert!(!m.pipelines.contains_key("p"));
        assert_eq!(dropped.pipelines.len(), 1);
        assert_eq!(dropped.pipelines[0].0, "p:latest");
        assert!(dropped.pipelines[0].1.iter().any(|r| r.contains("missing")));
    }

    /// Group-coherence validation drops a pipeline whose order references an
    /// image that exists but isn't in the pipeline's group
    #[test]
    fn coherence_drops_misgrouped_order_image() {
        // image 'a' lives in group 'other', but the pipeline imports into 'g'
        let mut m = manifest(
            vec![("a", image("other", "a"))],
            vec![("p", pipeline("g", "p", json!(["a"]), &["a"]))],
        );
        // structurally fine (the image exists in the manifest)...
        assert!(m.validate_structural().pipelines.is_empty());
        // ...but incoherent: 'a' isn't in the pipeline's group
        let dropped = m.validate_group_coherence();
        assert!(!m.pipelines.contains_key("p"));
        assert!(
            dropped.pipelines[0]
                .1
                .iter()
                .any(|r| r.contains("not in group 'g'"))
        );
    }

    /// A pipeline validates when its images map keys by the tool name while its
    /// order uses the image's config name
    #[test]
    fn keeps_pipeline_when_image_map_uses_tool_name() {
        // the image's manifest key (tool name) is 'sqlitediff' but its config name
        // is 'sqldiff'; the pipeline's images map keys by the tool name while its
        // order uses the config name — both must validate
        let mut m = manifest(
            vec![(
                "sqlitediff",
                ImageManifest {
                    versions: HashMap::from([(
                        "latest".to_string(),
                        image_version("g", "sqldiff", "url"),
                    )]),
                },
            )],
            vec![(
                "sqlitediff",
                pipeline_pinned(
                    "g",
                    "sqlitediff",
                    json!(["sqldiff"]),
                    &[("sqlitediff", "latest")],
                ),
            )],
        );
        assert!(m.validate_structural().pipelines.is_empty());
        assert!(m.validate_group_coherence().pipelines.is_empty());
        assert!(m.pipelines.contains_key("sqlitediff"));
    }

    /// A group override collapses a misgrouped image into the target group so the
    /// dependent pipeline then validates
    #[test]
    fn group_override_resolves_misgrouped_image() {
        let m = manifest(
            vec![("a", image("other", "a"))],
            vec![("p", pipeline("g", "p", json!(["a"]), &["a"]))],
        );
        let mut m = m.override_group("static");
        assert!(m.validate_structural().pipelines.is_empty());
        assert!(m.validate_group_coherence().pipelines.is_empty());
        assert!(m.pipelines.contains_key("p"));
    }

    /// Structural validation drops an unresolved image and any pipeline that
    /// depended on it
    #[test]
    fn structural_drops_unresolved_image_and_dependent_pipeline() {
        let mut m = manifest(
            vec![("a", unresolved_image())],
            vec![("p", pipeline("g", "p", json!(["a"]), &["a"]))],
        );
        let dropped = m.validate_structural();
        assert!(m.images.is_empty());
        assert_eq!(dropped.images.len(), 1);
        assert_eq!(dropped.images[0].0, "a:latest");
        assert!(!m.pipelines.contains_key("p"));
    }

    // ─── collision detection & resolution ────────────────────────────────────

    /// One image with two distinct version configs is detected as a single
    /// non-identical collision
    #[test]
    fn detects_multi_version_collision() {
        // one image with two distinct version configs collides on (g, exiftool)
        let m = manifest(
            vec![(
                "exiftool",
                versioned_image("g", "exiftool", &["latest", "1.2"]),
            )],
            vec![],
        );
        let sources = m.capture_source_groups();
        let collisions = m.detect_image_collisions(&sources).unwrap();
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].name, "exiftool");
        assert_eq!(collisions[0].members.len(), 2);
        assert!(!collisions[0].identical);
    }

    /// Renaming one version variant repoints only the pipeline pinned to that
    /// version, leaving the other variant under the original name
    #[test]
    fn rename_multi_version_cascades_by_pinned_version() {
        // pipeline pins exiftool@latest; renaming the latest variant repoints it
        let m = manifest(
            vec![(
                "exiftool",
                versioned_image("g", "exiftool", &["latest", "1.2"]),
            )],
            vec![(
                "p",
                pipeline_pinned("g", "p", json!(["exiftool"]), &[("exiftool", "latest")]),
            )],
        );
        let sources = m.capture_source_groups();
        let mut m = m;
        let collision = m.detect_image_collisions(&sources).unwrap().remove(0);
        // members sort to [1.2, latest]; keep the first, rename the second (latest)
        let latest = collision
            .members
            .iter()
            .find(|mem| mem.version == "latest")
            .unwrap()
            .clone();
        m.rename_image_member(&collision, &latest, "exiftool-latest", &sources);
        // both images now coexist
        assert!(m.images.contains_key("exiftool")); // the 1.2 variant kept the name
        assert!(m.images.contains_key("exiftool-latest"));
        // the pipeline that wanted latest was repointed
        assert_eq!(order_of(&m, "p"), vec![vec!["exiftool-latest".to_string()]]);
        assert!(
            m.pipelines["p"].versions["latest"]
                .images
                .contains_key("exiftool-latest")
        );
    }

    /// Renaming one of two same-named images merged from different source groups
    /// repoints only the pipeline from that source group
    #[test]
    fn rename_same_name_diff_group_cascades_by_source_group() {
        // two genuinely-different images named 'exiftool' in different groups,
        // merged by the override (distinct `image` urls so they aren't deduped)
        let m = manifest(
            vec![
                (
                    "exiftool",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("static", "exiftool", "url-static"),
                        )]),
                    },
                ),
                (
                    "exiftool-uur",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("dynamic", "exiftool", "url-uur"),
                        )]),
                    },
                ),
            ],
            vec![
                (
                    "p-static",
                    pipeline("static", "p-static", json!(["exiftool"]), &["exiftool"]),
                ),
                (
                    "p-uur",
                    pipeline("dynamic", "p-uur", json!(["exiftool"]), &["exiftool"]),
                ),
            ],
        );
        // capture BEFORE the override, then collapse groups
        let sources = m.capture_source_groups();
        let mut m = m.override_group("static");
        let collision = m.detect_image_collisions(&sources).unwrap().remove(0);
        assert_eq!(collision.members.len(), 2);
        assert!(!collision.identical);
        // canonical = first member by key ('exiftool', source static); rename the uur one
        let uur = collision
            .members
            .iter()
            .find(|mem| mem.source_group == "dynamic")
            .unwrap()
            .clone();
        let new_name = m.suggested_image_rename(&collision, &uur);
        m.rename_image_member(&collision, &uur, &new_name, &sources);
        // the static pipeline keeps 'exiftool'; the uur pipeline is repointed
        assert_eq!(order_of(&m, "p-static"), vec![vec!["exiftool".to_string()]]);
        assert_eq!(order_of(&m, "p-uur"), vec![vec![new_name.clone()]]);
    }

    /// Two entries with distinct manifest keys but the same config name + group
    /// are still detected as a collision
    #[test]
    fn detects_collision_by_config_name_not_manifest_key() {
        // distinct manifest keys, same config.name + group -> still a collision
        let m = manifest(
            vec![
                (
                    "a",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-a"),
                        )]),
                    },
                ),
                (
                    "b",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-b"),
                        )]),
                    },
                ),
            ],
            vec![],
        );
        let sources = m.capture_source_groups();
        let collisions = m.detect_image_collisions(&sources).unwrap();
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].name, "x");
        assert!(!collisions[0].identical);
    }

    /// Two byte-identical entries are detected as an identical collision and
    /// de-duped down to the first member
    #[test]
    fn dedupes_identical_duplicate() {
        // two manifest entries with identical configs -> pure duplicate
        let m = manifest(vec![("a", image("g", "x")), ("b", image("g", "x"))], vec![]);
        let sources = m.capture_source_groups();
        let mut m = m;
        let collisions = m.detect_image_collisions(&sources).unwrap();
        assert_eq!(collisions.len(), 1);
        assert!(collisions[0].identical);
        m.dedupe_image_collision(&collisions[0]);
        // only the first member (key 'a') survives
        assert_eq!(m.images.len(), 1);
        assert!(m.images.contains_key("a"));
    }

    /// Skipping a collision removes every entry of that identity and the pipelines
    /// depending on it, while leaving unrelated pipelines intact
    #[test]
    fn skip_removes_identity_and_dependents() {
        let m = manifest(
            vec![
                (
                    "a",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-a"),
                        )]),
                    },
                ),
                (
                    "b",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-b"),
                        )]),
                    },
                ),
            ],
            vec![
                ("dep", pipeline("g", "dep", json!(["x"]), &["x"])),
                ("free", pipeline("g", "free", json!(["other"]), &["other"])),
            ],
        );
        let mut m = m;
        let dropped = m.remove_image_identity_and_dependents("g", "x");
        // both colliding image entries are gone...
        assert!(!m.image_names_in_group("g").contains("x"));
        // ...the pipeline that used it is dropped, the one that didn't is kept
        assert_eq!(dropped, vec!["dep:latest".to_string()]);
        assert!(!m.pipelines.contains_key("dep"));
        assert!(m.pipelines.contains_key("free"));
    }

    /// Removing one `(group, name)` image identity must be scoped to that group: a
    /// same-named image in a different group, and a pipeline in that other group that
    /// depends on it, must both survive (image references are bare names resolved within
    /// the pipeline's own group, so a cross-group same-named image is unrelated)
    #[test]
    fn skip_removal_is_scoped_to_the_image_group() {
        let mut m = manifest(
            vec![("x-g1", image("g1", "x")), ("x-g2", image("g2", "x"))],
            // a pipeline in g2 depending on the g2 copy of x
            vec![("p2", pipeline("g2", "p2", json!(["x"]), &["x"]))],
        );
        let dropped = m.remove_image_identity_and_dependents("g1", "x");
        // the g1 identity is removed...
        assert!(!m.image_names_in_group("g1").contains("x"));
        // ...while the g2 copy and its dependent pipeline are left untouched
        assert!(m.image_names_in_group("g2").contains("x"));
        assert!(dropped.is_empty());
        assert!(m.pipelines.contains_key("p2"));
    }

    /// When the natural rename candidates are exhausted, a numeric suffix is
    /// appended to keep the suggested name unique
    #[test]
    fn suggested_rename_appends_numeric_suffix() {
        // three same-version variants force a fallback suffix on the third name
        let m = manifest(
            vec![
                (
                    "a",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-a"),
                        )]),
                    },
                ),
                (
                    "b",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-b"),
                        )]),
                    },
                ),
                (
                    "c",
                    ImageManifest {
                        versions: HashMap::from([(
                            "latest".to_string(),
                            image_version("g", "x", "url-c"),
                        )]),
                    },
                ),
            ],
            vec![],
        );
        let sources = m.capture_source_groups();
        let mut m = m;
        let collision = m.detect_image_collisions(&sources).unwrap().remove(0);
        // rename the 2nd and 3rd members (keep the 1st canonical)
        let m1 = collision.members[1].clone();
        let n1 = m.suggested_image_rename(&collision, &m1);
        assert_eq!(n1, "x-latest");
        m.rename_image_member(&collision, &m1, &n1, &sources);
        let m2 = collision.members[2].clone();
        let n2 = m.suggested_image_rename(&collision, &m2);
        assert_eq!(n2, "x-latest-2");
    }
}
