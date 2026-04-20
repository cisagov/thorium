//! Network policy transport for toolboxes
//!
//! Images reference network policies by name, so a toolbox that doesn't carry
//! the policy definitions imports broken on instances that lack them. This
//! module collects the policies bundled in a manifest, dedupes them, checks
//! them against the target instance, and creates the missing ones before any
//! images are imported.
//!
//! By default policies are not updated, even under `--overwrite`: they are
//! cluster security state owned by instance admins, so an existing-but-different
//! policy only produces a warning. Passing `--update-network-policy` opts into
//! pushing the toolbox's definition: rule/flag drift overwrites the existing
//! policy and a groups-only difference adds the toolbox's groups to it (group
//! coverage is only ever added, never removed).

use colored::Colorize;
use std::collections::HashMap;
use thorium::models::{
    NetworkPolicy, NetworkPolicyListOpts, NetworkPolicyRequest, NetworkPolicyRule,
    NetworkPolicyRuleRaw, NetworkPolicyUpdate,
};
use thorium::{Error, Thorium};
use uuid::Uuid;

use super::manifest::ToolboxManifest;
use crate::handlers::imports::rollback::Journal;
use crate::handlers::progress::{Bar, BarKind};

/// The bundled policies categorized against the target instance
pub struct PolicyPlan {
    /// Policies missing from the target that will be created
    pub new: Vec<NetworkPolicyRequest>,
    /// Policies that exist in the target but differ from the bundled definition
    /// (rule/flag drift and/or partial group coverage); left in place and only
    /// warned about unless `--update-network-policy` is set
    pub mismatched: Vec<PolicyMismatch>,
    /// Existing Thorium policies to update toward the toolbox's definition,
    /// populated only when `--update-network-policy` is set
    pub updates: Vec<PolicyUpdatePlan>,
    /// How many bundled policies already match the target exactly
    pub unchanged: usize,
}

/// One bundled policy that differs from what already exists in Thorium
///
/// Carries enough context to attribute the difference (rule/flag drift vs missing
/// group coverage) and, when updating is requested, to build the delta against the
/// distinct existing policies the toolbox definition overlaps.
pub struct PolicyMismatch {
    /// The policy name
    pub name: String,
    /// The rule/flag fields that differ against a matched existing policy
    /// (group membership excluded); empty when the only difference is coverage
    pub drift: Vec<String>,
    /// Target groups the policy is missing from in Thorium
    pub missing_groups: Vec<String>,
    /// The toolbox's chosen definition for this policy
    pub chosen: NetworkPolicyRequest,
    /// The distinct existing Thorium policies (by id) this overlaps
    pub existing: Vec<NetworkPolicy>,
}

/// A single existing Thorium policy to update toward the toolbox's definition
///
/// One bundled policy name can map to several distinct existing policies (the same
/// name in different groups is a different policy), so each gets its own plan.
pub struct PolicyUpdatePlan {
    /// The policy name
    pub name: String,
    /// The id of the existing policy this update targets
    pub id: Uuid,
    /// A representative group the targeted policy lives in, for display
    pub group: String,
    /// The rule/flag fields being overwritten (empty for a groups-only add)
    pub drift: Vec<String>,
    /// The groups being added to the policy (additive only)
    pub add_groups: Vec<String>,
    /// The toolbox's chosen definition, used to rebuild the delta against the
    /// freshly-fetched policy state at apply time
    pub chosen: NetworkPolicyRequest,
}

/// Normalize a policy request so semantically equal policies compare equal
///
/// Group order carries no meaning, so it is sorted; rule order is preserved
/// since rules are applied as written.
///
/// # Arguments
///
/// * `policy` - The policy request to normalize
fn normalize(mut policy: NetworkPolicyRequest) -> NetworkPolicyRequest {
    policy.groups.sort_unstable();
    policy
}

/// Whether two policy requests are semantically equal
///
/// `NetworkPolicyRequest` doesn't implement `PartialEq`, so equality is
/// checked on the serialized form (callers normalize first).
///
/// # Arguments
///
/// * `ours` - The bundled policy from the toolbox
/// * `theirs` - The policy to compare against
fn requests_equal(ours: &NetworkPolicyRequest, theirs: &NetworkPolicyRequest) -> bool {
    match (serde_json::to_value(ours), serde_json::to_value(theirs)) {
        (Ok(ours), Ok(theirs)) => ours == theirs,
        // treat serialization failures as different so they surface as warnings
        _ => false,
    }
}

/// List the top-level fields that differ between two policy requests
///
/// # Arguments
///
/// * `ours` - The bundled policy from the toolbox
/// * `theirs` - The existing policy in the target instance
fn diff_fields(ours: &NetworkPolicyRequest, theirs: &NetworkPolicyRequest) -> Vec<String> {
    diff_fields_excluding(ours, theirs, &[])
}

/// List the top-level fields that differ between two policy requests, ignoring the
/// named fields
///
/// Used when comparing a bundled policy against an existing instance policy to ignore
/// group membership: the toolbox targets specific groups while the instance policy may
/// legitimately span more, and policies are never updated anyway — so only rule/flag
/// drift should surface.
///
/// # Arguments
///
/// * `ours` - The bundled policy from the toolbox
/// * `theirs` - The existing policy in the target instance
/// * `exclude` - Top-level field names to drop from both sides before comparing
fn diff_fields_excluding(
    ours: &NetworkPolicyRequest,
    theirs: &NetworkPolicyRequest,
    exclude: &[&str],
) -> Vec<String> {
    // compare serialized forms field by field so new model fields are
    // covered automatically
    let (Ok(ours), Ok(theirs)) = (serde_json::to_value(ours), serde_json::to_value(theirs)) else {
        return vec!["<unserializable>".to_string()];
    };
    let (serde_json::Value::Object(mut ours), serde_json::Value::Object(mut theirs)) =
        (ours, theirs)
    else {
        return vec!["<unserializable>".to_string()];
    };
    // drop the excluded fields from both sides so they don't count as differences
    for field in exclude {
        ours.remove(*field);
        theirs.remove(*field);
    }
    let mut fields: Vec<String> = ours
        .iter()
        .filter(|(key, value)| theirs.get(key.as_str()) != Some(*value))
        .map(|(key, _)| key.clone())
        .collect();
    fields.sort_unstable();
    fields
}

/// Collect and dedupe the network policies bundled across a manifest's images
///
/// Identical duplicates (after normalization) collapse into one entry. The same name
/// bundled with differing definitions can't be reconciled, but it must not fail the
/// whole import: the first definition (in sorted order) is kept, the rest are ignored,
/// and a warning is emitted per conflicting name.
///
/// # Arguments
///
/// * `manifest` - The toolbox manifest holding bundled policies
/// * `group_override` - The group override applied to the rest of the import
/// * `progress` - The progress bar to log conflict warnings through
pub fn collect_policies(
    manifest: &ToolboxManifest,
    group_override: Option<&str>,
    progress: &Bar,
) -> Vec<NetworkPolicyRequest> {
    // dedupe purely, then surface any conflicts as warnings so a conflicting bundle
    // degrades to "first definition wins" instead of aborting the import
    let (policies, warnings) = dedupe_policies(manifest, group_override);
    for warning in warnings {
        progress.warning(warning);
    }
    policies
}

/// Dedupe bundled policies by name, returning the kept policies and a warning message
/// for every name that was bundled with conflicting definitions
///
/// Images and their versions are visited in sorted (key, then version) order so the
/// definition kept on a conflict is deterministic rather than hash-map dependent. Pure
/// (no I/O) so the dedup/conflict behavior can be unit tested.
///
/// # Arguments
///
/// * `manifest` - The toolbox manifest holding bundled policies
/// * `group_override` - The group override applied to the rest of the import
fn dedupe_policies(
    manifest: &ToolboxManifest,
    group_override: Option<&str>,
) -> (Vec<NetworkPolicyRequest>, Vec<String>) {
    // track each kept policy alongside the image key it was first seen in so a genuine
    // conflict can name both sides
    let mut by_name: HashMap<String, (NetworkPolicyRequest, String)> = HashMap::new();
    let mut warnings = Vec::new();
    // visit images in sorted key order so "first definition wins" is deterministic
    let mut images: Vec<_> = manifest.images.iter().collect();
    images.sort_by(|a, b| a.0.cmp(b.0));
    for (image_key, image_manifest) in images {
        // visit versions in sorted order for the same determinism
        let mut versions: Vec<_> = image_manifest.versions.iter().collect();
        versions.sort_by(|a, b| a.0.cmp(b.0));
        for (_version, version) in versions {
            for policy in &version.network_policies {
                let mut policy = normalize(policy.clone());
                // policies are group-scoped, so the override applies to them
                // exactly like it does to images and pipelines
                if let Some(group) = group_override {
                    policy.groups = vec![group.to_string()];
                }
                match by_name.get_mut(&policy.name) {
                    // first sighting of this policy name
                    None => {
                        by_name.insert(policy.name.clone(), (policy, image_key.clone()));
                    }
                    // identical duplicates across images are expected and fine
                    Some((existing, _)) if requests_equal(existing, &policy) => {}
                    // a groups-only difference isn't a conflict: the two copies describe the
                    // same policy in different groups, so union their groups and keep going
                    Some((existing, _)) if diff_fields(existing, &policy) == ["groups"] => {
                        existing.groups.extend(policy.groups);
                        existing.groups.sort_unstable();
                        existing.groups.dedup();
                    }
                    // the same name with genuinely different rules/flags can't be reconciled;
                    // keep the first definition and warn rather than failing the import
                    Some((existing, first_key)) => {
                        warnings.push(format!(
                            "Network policy '{}' is defined differently by two images in the \
                             toolbox ('{first_key}' vs '{image_key}') (differs: [{}]); keeping \
                             '{first_key}'s definition and ignoring '{image_key}'s",
                            policy.name,
                            diff_fields(existing, &policy).join(", ")
                        ));
                    }
                }
            }
        }
    }
    // sort for deterministic create order and confirmation display, dropping the
    // first-seen image key now that conflicts have been resolved
    let mut policies: Vec<_> = by_name.into_values().map(|(policy, _)| policy).collect();
    policies.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    (policies, warnings)
}

/// The distinct groups targeted across a set of bundled policies, in first-seen order
///
/// # Arguments
///
/// * `policies` - The bundled policies to collect target groups from
fn unique_groups(policies: &[NetworkPolicyRequest]) -> Vec<String> {
    // the set tracks membership for O(1) dedup while the vec preserves first-seen order,
    // which keeps the resulting group list stable for callers (e.g. error messages)
    let mut seen = std::collections::HashSet::new();
    let mut groups = Vec::new();
    for policy in policies {
        for group in &policy.groups {
            // push only the first sighting of each group; insert returns false on repeats
            if seen.insert(group.clone()) {
                groups.push(group.clone());
            }
        }
    }
    groups
}

/// Fetch the network policies that already exist in the given groups, indexed by
/// `(group, name)`
///
/// Scoping the lookup to specific groups is what resolves cross-group name ambiguity:
/// within a single group a name maps to at most one policy, so this avoids the global
/// get-by-name that 400s when several distinct policies share a name across groups.
/// Mirrors how `network-policies describe --group` lists by group.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `groups` - The groups to scope the lookup to
pub(super) async fn fetch_existing_in_groups(
    thorium: &Thorium,
    groups: &[String],
) -> Result<HashMap<(String, String), NetworkPolicy>, Error> {
    let mut existing = HashMap::new();
    // nothing to look up when no groups are targeted
    if groups.is_empty() {
        return Ok(existing);
    }
    // list (with full details) only the policies in the groups we care about
    let opts = NetworkPolicyListOpts::default().groups(groups.to_vec());
    let mut cursor = thorium
        .network_policies
        .list_details(&opts)
        .await
        .map_err(|err| {
            Error::new(format!(
                "Error listing network policies for groups {groups:?}: {err}"
            ))
        })?;
    loop {
        for policy in cursor.data.drain(..) {
            // index the policy under each queried group it belongs to; a name is unique
            // within a group, so (group, name) identifies exactly one policy
            for group in &policy.groups {
                if groups.iter().any(|wanted| wanted == group) {
                    existing.insert((group.clone(), policy.name.clone()), policy.clone());
                }
            }
        }
        if cursor.exhausted() {
            break;
        }
        cursor
            .refill()
            .await
            .map_err(|err| Error::new(format!("Error listing network policies: {err}")))?;
    }
    Ok(existing)
}

/// The outcome of categorizing one bundled policy against the existing instance state
///
/// `Mismatched` can't derive `PartialEq`/`Eq` because it carries
/// [`NetworkPolicy`] (no `Eq`) and the bundled request, so tests match on it
/// rather than comparing it.
#[derive(Debug)]
enum PolicyClassification {
    /// The policy exists in no target group and will be created
    New,
    /// The policy already exists in every target group and matches
    Unchanged,
    /// The policy exists but differs (rule/flag drift) and/or is missing from some
    /// target groups; carries the drift fields, the missing groups, and the distinct
    /// existing policies it overlaps for the warning/update path
    Mismatched {
        /// The rule/flag fields that differ against a matched existing policy
        drift: Vec<String>,
        /// Target groups the policy is missing from in Thorium
        missing_groups: Vec<String>,
        /// The distinct existing policies (by id) this overlaps
        existing: Vec<NetworkPolicy>,
    },
}

/// Classify one bundled policy against the existing `(group, name)` index
///
/// A policy absent from every target group is new (safe to create). A policy present in
/// any target group is never created — create rejects a name already in any requested
/// group — so its rules/flags are compared (ignoring group membership) and any drift,
/// or partial coverage across the target groups, is reported as a mismatch.
///
/// # Arguments
///
/// * `policy` - The bundled policy (its `groups` are the target groups)
/// * `existing` - The existing policies indexed by `(group, name)`
fn classify_policy(
    policy: &NetworkPolicyRequest,
    existing: &HashMap<(String, String), NetworkPolicy>,
) -> PolicyClassification {
    // split the target groups into those the policy already exists in and those it
    // doesn't, so we can tell "create" from "exists" from "partially exists"
    let mut present = Vec::new();
    let mut missing = Vec::new();
    for group in &policy.groups {
        if existing.contains_key(&(group.clone(), policy.name.clone())) {
            present.push(group.clone());
        } else {
            missing.push(group.clone());
        }
    }
    // absent everywhere it would be created → new
    if present.is_empty() {
        return PolicyClassification::New;
    }
    // exists in at least one target group → never created; gather the distinct existing
    // policies it overlaps (by id) and the rule/flag drift against them
    let mut overlapped: Vec<NetworkPolicy> = Vec::new();
    let mut drift: Vec<String> = Vec::new();
    for group in &present {
        if let Some(existing_policy) = existing.get(&(group.clone(), policy.name.clone())) {
            // record each distinct existing policy once, since the same name across groups
            // can resolve to several independent policies
            if !overlapped.iter().any(|seen| seen.id == existing_policy.id) {
                overlapped.push(existing_policy.clone());
            }
            // surface the first rule/flag drift we find (group membership ignored)
            if drift.is_empty() {
                let diff =
                    diff_fields_excluding(policy, &NetworkPolicyRequest::from(existing_policy), &["groups"]);
                if !diff.is_empty() {
                    drift = diff;
                }
            }
        }
    }
    // partial coverage: targets multiple groups but only exists in some of them
    missing.sort_unstable();
    // matches everywhere it exists and covers every target group → nothing to do
    if drift.is_empty() && missing.is_empty() {
        return PolicyClassification::Unchanged;
    }
    PolicyClassification::Mismatched {
        drift,
        missing_groups: missing,
        existing: overlapped,
    }
}

/// Categorize bundled policies against the target instance, scoped by each policy's
/// target groups
///
/// The existence check is scoped to the groups each bundled policy targets (a single
/// group after `--group-override`), so a name that exists in several groups across the
/// instance still resolves uniquely instead of erroring with a 400. See
/// [`classify_policy`] for the per-policy new/unchanged/mismatched decision.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `policies` - The deduped bundled policies
/// * `update` - Whether `--update-network-policy` is set, turning mismatches into
///   planned updates instead of passive warnings
/// * `progress` - The progress bar
pub async fn categorize_policies(
    thorium: &Thorium,
    policies: Vec<NetworkPolicyRequest>,
    update: bool,
    progress: &Bar,
) -> Result<PolicyPlan, Error> {
    let mut plan = PolicyPlan {
        new: Vec::new(),
        mismatched: Vec::new(),
        updates: Vec::new(),
        unchanged: 0,
    };
    if policies.is_empty() {
        return Ok(plan);
    }
    progress.refresh(
        "Checking network policies",
        BarKind::Bound(policies.len() as u64),
    );
    // fetch existing policies scoped to just the groups the bundle targets; the full
    // policy is retained (not reduced to a request) so the update path can target it by
    // id and build a delta from its rule UUIDs
    let groups = unique_groups(&policies);
    let existing: HashMap<(String, String), NetworkPolicy> =
        fetch_existing_in_groups(thorium, &groups).await?;
    for policy in policies {
        match classify_policy(&policy, &existing) {
            PolicyClassification::New => plan.new.push(policy),
            PolicyClassification::Unchanged => plan.unchanged += 1,
            PolicyClassification::Mismatched {
                drift,
                missing_groups,
                existing,
            } => {
                // assemble the mismatch with everything needed to warn about or update it
                let mismatch = PolicyMismatch {
                    name: policy.name.clone(),
                    drift,
                    missing_groups,
                    chosen: policy,
                    existing,
                };
                // with the flag, plan concrete updates against each overlapped policy;
                // without it, keep the mismatch for a source-attributed warning
                if update {
                    plan.updates.extend(build_updates(&mismatch));
                } else {
                    plan.mismatched.push(mismatch);
                }
            }
        }
        progress.inc(1);
    }
    Ok(plan)
}

/// Build the per-policy update plans for one mismatch
///
/// A bundled name can overlap several distinct existing policies (the same name in
/// different groups), so one plan is produced per existing policy. Rule/flag drift
/// overwrites that policy; coverage gaps (target groups with no existing policy at all)
/// are folded into the first existing policy so the toolbox's coverage is satisfied
/// without removing anything.
///
/// # Arguments
///
/// * `mismatch` - The classified mismatch to turn into update plans
fn build_updates(mismatch: &PolicyMismatch) -> Vec<PolicyUpdatePlan> {
    // sort the overlapped policies by id so the choice of "first" (which absorbs the
    // coverage gaps) is deterministic
    let mut existing = mismatch.existing.clone();
    existing.sort_by_key(|policy| policy.id);
    let mut plans = Vec::new();
    for (idx, policy) in existing.iter().enumerate() {
        // recompute drift against this specific policy: distinct policies sharing a name
        // can differ from the toolbox in different ways
        let drift =
            diff_fields_excluding(&mismatch.chosen, &NetworkPolicyRequest::from(policy), &["groups"]);
        // only the truly-missing groups (no existing policy anywhere) need adding, and
        // they go to the first policy; groups already covered by a sibling policy are
        // left alone so we never create a duplicate name within a group
        let add_groups = if idx == 0 {
            mismatch.missing_groups.clone()
        } else {
            Vec::new()
        };
        // this policy already matches and needs no new coverage → nothing to do
        if drift.is_empty() && add_groups.is_empty() {
            continue;
        }
        plans.push(PolicyUpdatePlan {
            name: mismatch.name.clone(),
            id: policy.id,
            group: policy.groups.first().cloned().unwrap_or_default(),
            drift,
            add_groups,
            chosen: mismatch.chosen.clone(),
        });
    }
    plans
}

/// Build the delta to make an existing policy match the toolbox's chosen definition
///
/// `add_groups` is applied as-is (additive only). Rule/flag fields are touched only when
/// they appear in `drift`, so a groups-only add leaves rules untouched. Ingress/egress
/// follow the three-case semantics in [`rule_direction_delta`].
///
/// # Arguments
///
/// * `chosen` - The toolbox's chosen definition
/// * `drift` - The rule/flag fields that differ (from [`diff_fields_excluding`])
/// * `add_groups` - The groups to add to the policy
/// * `existing` - The current state of the policy being updated
fn build_delta(
    chosen: &NetworkPolicyRequest,
    drift: &[String],
    add_groups: &[String],
    existing: &NetworkPolicy,
) -> NetworkPolicyUpdate {
    let mut update = NetworkPolicyUpdate {
        add_groups: add_groups.to_vec(),
        ..NetworkPolicyUpdate::default()
    };
    // overwrite ingress only when it drifted, sourcing the removed rule UUIDs from this
    // exact policy's current state
    if drift.iter().any(|field| field == "ingress") {
        let (clear, deny_all, remove, add) =
            rule_direction_delta(chosen.ingress.as_ref(), existing.ingress.as_ref());
        update.clear_ingress = clear;
        update.deny_all_ingress = deny_all;
        update.remove_ingress = remove;
        update.add_ingress = add;
    }
    // egress is symmetric
    if drift.iter().any(|field| field == "egress") {
        let (clear, deny_all, remove, add) =
            rule_direction_delta(chosen.egress.as_ref(), existing.egress.as_ref());
        update.clear_egress = clear;
        update.deny_all_egress = deny_all;
        update.remove_egress = remove;
        update.add_egress = add;
    }
    // flags only when they differ
    if drift.iter().any(|field| field == "forced_policy") {
        update.forced_policy = Some(chosen.forced_policy);
    }
    if drift.iter().any(|field| field == "default_policy") {
        update.default_policy = Some(chosen.default_policy);
    }
    update
}

/// Build the ingress/egress portion of a delta to make one direction's rules match a
/// target set of rules
///
/// Returns `(clear, deny_all, remove, add)` matching the `*_ingress`/`*_egress` fields of
/// [`NetworkPolicyUpdate`]. The three cases mirror the model's rule semantics: a `None`
/// target imposes no rules (allow-all) so the direction is cleared; an empty `Some`
/// denies all traffic, which must use the `deny_all` flag rather than a bare remove (a
/// bare remove would collapse the direction back to allow-all); a populated `Some` drops
/// the existing rules by id and adds the target rules.
///
/// # Arguments
///
/// * `target` - The rules the direction should end up with (the toolbox or prior state)
/// * `existing` - The direction's current rules, whose ids are removed
pub(crate) fn rule_direction_delta(
    target: Option<&Vec<NetworkPolicyRuleRaw>>,
    existing: Option<&Vec<NetworkPolicyRule>>,
) -> (bool, bool, Vec<Uuid>, Vec<NetworkPolicyRuleRaw>) {
    match target {
        // no rules → allow all → clear the direction
        None => (true, false, Vec::new(), Vec::new()),
        // empty rule set → deny all; use the flag, not an empty remove
        Some(rules) if rules.is_empty() => (false, true, Vec::new(), Vec::new()),
        // explicit rules → drop the existing rules by id and add the target rules
        Some(rules) => {
            let remove = existing
                .map(|rules| rules.iter().map(|rule| rule.id).collect())
                .unwrap_or_default();
            (false, false, remove, rules.clone())
        }
    }
}

/// Print the policy sections of the import confirmation screen
///
/// # Arguments
///
/// * `plan` - The categorized policy plan
pub fn print_plan(plan: &PolicyPlan) {
    if !plan.new.is_empty() {
        println!("{}", "New Network Policies:".bright_green());
        for policy in &plan.new {
            println!("  {} (groups: {})", policy.name, policy.groups.join(", "));
        }
    }
    if !plan.mismatched.is_empty() {
        println!(
            "{}",
            "Existing Network Policies that differ (left unchanged; re-run with \
             --update-network-policy to apply):"
                .bright_yellow()
        );
        for mismatch in &plan.mismatched {
            // rule/flag drift and coverage gaps are distinct problems, so describe each
            if !mismatch.drift.is_empty() {
                println!(
                    "  {} (differs: [{}])",
                    mismatch.name,
                    mismatch.drift.join(", ")
                );
            }
            if !mismatch.missing_groups.is_empty() {
                println!(
                    "  {} (missing from group(s): {:?})",
                    mismatch.name, mismatch.missing_groups
                );
            }
        }
    }
    if !plan.updates.is_empty() {
        println!("{}", "Network Policies to update:".bright_yellow());
        for update in &plan.updates {
            // each line names the exact policy (group + id) being changed, since one
            // bundled name can map to several distinct existing policies
            if !update.drift.is_empty() {
                println!(
                    "  {} (group: {}, id: {}) — overwrite (differs: [{}])",
                    update.name,
                    update.group,
                    update.id,
                    update.drift.join(", ")
                );
            }
            if !update.add_groups.is_empty() {
                println!(
                    "  {} (id: {}) — add group(s): {}",
                    update.name,
                    update.id,
                    update.add_groups.join(", ")
                );
            }
        }
    }
}

/// Warn for every bundled policy that exists in the target but differs, attributing the
/// problem to Thorium's existing policy and pointing at the update flag
///
/// # Arguments
///
/// * `plan` - The categorized policy plan
/// * `progress` - The progress bar to log warnings through
pub fn warn_mismatched(plan: &PolicyPlan, progress: &Bar) {
    for mismatch in &plan.mismatched {
        // rule/flag drift: Thorium's copy differs and would be overwritten under the flag
        if !mismatch.drift.is_empty() {
            progress.warning(format!(
                "Network policy '{}' already exists in Thorium with a different definition \
                 (differs: [{}]); leaving Thorium's policy unchanged — re-run with \
                 --update-network-policy to overwrite it with the toolbox's definition",
                mismatch.name,
                mismatch.drift.join(", ")
            ));
        }
        // coverage gap: the policy exists but not in every target group
        if !mismatch.missing_groups.is_empty() {
            progress.warning(format!(
                "Network policy '{}' exists in Thorium but not in group(s) {:?}; not creating \
                 it there — re-run with --update-network-policy to add those group(s) to the \
                 existing policy",
                mismatch.name, mismatch.missing_groups
            ));
        }
    }
}

/// Apply the planned updates to existing Thorium policies
///
/// Each policy is fetched immediately before it is patched so the rollback snapshot is its
/// true pre-update state (not a stale categorization view) and the delta's removed rule
/// UUIDs come from the current state. The snapshot is journaled before the patch lands so
/// a stop after this point can still revert it.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `updates` - The planned per-policy updates
/// * `progress` - The progress bar
/// * `journal` - The journal to record pre-update snapshots in for rollback
pub async fn update_policies(
    thorium: &Thorium,
    updates: &[PolicyUpdatePlan],
    progress: &Bar,
    journal: &Journal,
) -> Result<(), Error> {
    if updates.is_empty() {
        return Ok(());
    }
    progress.refresh(
        "Updating network policies",
        BarKind::Bound(updates.len() as u64),
    );
    for plan in updates {
        // fetch the policy's current state right before mutating it
        let current = thorium
            .network_policies
            .get(&plan.name, Some(plan.id))
            .await
            .map_err(|err| {
                Error::new(format!(
                    "Error fetching network policy '{}' (id {}) before update: {err}",
                    plan.name, plan.id
                ))
            })?;
        // record the pre-update snapshot before the patch so rollback can restore it
        journal.updated_network_policy(current.clone());
        // build the delta from the freshly-fetched state so removed rule UUIDs are current
        let delta = build_delta(&plan.chosen, &plan.drift, &plan.add_groups, &current);
        // log the action being taken, splitting overwrite from a groups-only add
        if !plan.drift.is_empty() {
            progress.info_anonymous(format!(
                "Updating network policy '{}' (group '{}', id {}) in Thorium to match the \
                 toolbox (differs: [{}])",
                plan.name,
                plan.group,
                plan.id,
                plan.drift.join(", ")
            ));
        }
        if !plan.add_groups.is_empty() {
            progress.info_anonymous(format!(
                "Adding group(s) [{}] to existing network policy '{}' (id {})",
                plan.add_groups.join(", "),
                plan.name,
                plan.id
            ));
        }
        // apply the patch targeting this exact policy by id
        thorium
            .network_policies
            .update(&plan.name, Some(plan.id), &delta)
            .await
            .map_err(|err| {
                Error::new(format!(
                    "Error updating network policy '{}' (id {}): {err}",
                    plan.name, plan.id
                ))
            })?;
        progress.inc(1);
    }
    Ok(())
}

/// Create the missing network policies in the target instance
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `policies` - The policies missing from the target
/// * `progress` - The progress bar
/// * `journal` - The journal to record created policies in for rollback
pub async fn create_policies(
    thorium: &Thorium,
    policies: &[NetworkPolicyRequest],
    progress: &Bar,
    journal: &Journal,
) -> Result<(), Error> {
    if policies.is_empty() {
        return Ok(());
    }
    progress.refresh(
        "Creating network policies",
        BarKind::Bound(policies.len() as u64),
    );
    for policy in policies {
        // create the policy in the instance; the request already carries its target groups
        thorium
            .network_policies
            .create(policy.clone())
            .await
            .map_err(|err| {
                Error::new(format!(
                    "Error creating network policy '{}': {err}",
                    policy.name
                ))
            })?;
        // journal the creation only after it succeeds so rollback never tries to delete a
        // policy that was never actually created
        journal.created_network_policy(&policy.name);
        progress.inc(1);
    }
    Ok(())
}

/// Unit tests for the pure (no-I/O) policy logic: dedupe/conflict resolution,
/// classification against existing state, delta construction, and update planning
#[cfg(test)]
mod tests {
    use super::super::manifest::{ImageManifest, ImageVersion, ToolboxManifest};
    use super::*;

    /// A minimal policy with the given name, groups, and `default_policy` flag (the flag
    /// is just a cheap way to make two policies differ)
    fn policy(name: &str, groups: &[&str], default_policy: bool) -> NetworkPolicyRequest {
        NetworkPolicyRequest {
            name: name.to_string(),
            groups: groups.iter().map(|group| (*group).to_string()).collect(),
            ingress: None,
            egress: None,
            forced_policy: false,
            default_policy,
        }
    }

    /// Build a manifest where each `(image_key, policies)` entry is one image with a
    /// single `latest` version bundling those policies
    fn manifest(images: Vec<(&str, Vec<NetworkPolicyRequest>)>) -> ToolboxManifest {
        let images = images
            .into_iter()
            .map(|(key, policies)| {
                let version = ImageVersion {
                    dir: String::new(),
                    build_path: String::new(),
                    config_from: None,
                    config: None,
                    network_policies_from: Vec::new(),
                    network_policies: policies,
                };
                (
                    key.to_string(),
                    ImageManifest {
                        versions: HashMap::from([("latest".to_string(), version)]),
                    },
                )
            })
            .collect();
        ToolboxManifest {
            name: "t".to_string(),
            registry: None,
            pipelines: HashMap::new(),
            images,
            bundled_images: false,
            image_path_prefix: None,
        }
    }

    /// The same policy bundled identically across images collapses to one, no warning
    #[test]
    fn dedupes_identical_policy_without_warning() {
        let m = manifest(vec![
            ("a", vec![policy("p", &["g"], false)]),
            ("b", vec![policy("p", &["g"], false)]),
        ]);
        let (policies, warnings) = dedupe_policies(&m, None);
        assert_eq!(policies.len(), 1);
        assert!(warnings.is_empty());
    }

    /// A name bundled with conflicting definitions warns instead of erroring, keeping
    /// the first definition in sorted-key order
    #[test]
    fn conflicting_policy_warns_and_keeps_first() {
        // images visited in sorted key order, so "a" (default_policy = false) wins over
        // "b" (default_policy = true) regardless of insertion order here
        let m = manifest(vec![
            ("b", vec![policy("p", &["g"], true)]),
            ("a", vec![policy("p", &["g"], false)]),
        ]);
        let (policies, warnings) = dedupe_policies(&m, None);
        assert_eq!(policies.len(), 1);
        assert!(!policies[0].default_policy, "the 'a' definition should be kept");
        assert_eq!(warnings.len(), 1);
        // the warning names both images involved and the kept side
        assert!(warnings[0].contains("defined differently by two images"));
        assert!(warnings[0].contains("'a'") && warnings[0].contains("'b'"));
    }

    /// A groups-only difference is no longer a conflict: the two copies merge into one
    /// policy spanning both groups; an override collapses them to the override group
    #[test]
    fn group_override_resolves_groups_only_conflict() {
        let m = manifest(vec![
            ("a", vec![policy("p", &["x"], false)]),
            ("b", vec![policy("p", &["y"], false)]),
        ]);
        // without an override the differing groups merge into one policy, no warning
        let (policies, warnings) = dedupe_policies(&m, None);
        assert!(warnings.is_empty());
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].groups, vec!["x".to_string(), "y".to_string()]);
        // with an override both collapse to ["g"], a single deduped policy, no warning
        let (policies, warnings) = dedupe_policies(&m, Some("g"));
        assert!(warnings.is_empty());
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].groups, vec!["g".to_string()]);
    }

    /// Two images bundling the same policy in different groups merge into one entry whose
    /// groups are the (sorted, deduped) union, with no warning
    #[test]
    fn groups_only_conflict_merges_union() {
        let m = manifest(vec![
            ("a", vec![policy("p", &["a"], false)]),
            ("b", vec![policy("p", &["b"], false)]),
        ]);
        let (policies, warnings) = dedupe_policies(&m, None);
        assert!(warnings.is_empty());
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].groups, vec!["a".to_string(), "b".to_string()]);
    }

    /// Build a stored `NetworkPolicy` for the existing-index and delta tests
    fn existing_policy(
        name: &str,
        groups: &[&str],
        default_policy: bool,
        id: Uuid,
    ) -> NetworkPolicy {
        NetworkPolicy {
            name: name.to_string(),
            id,
            k8s_name: format!("{name}-{id}"),
            groups: groups.iter().map(|group| (*group).to_string()).collect(),
            created: chrono::DateTime::from_timestamp(0, 0).expect("valid epoch"),
            ingress: None,
            egress: None,
            forced_policy: false,
            default_policy,
            used_by: HashMap::new(),
        }
    }

    /// Build an existing-policy index keyed by `(group, name)` from `(group, policy)` pairs
    fn index(entries: Vec<(&str, NetworkPolicy)>) -> HashMap<(String, String), NetworkPolicy> {
        entries
            .into_iter()
            .map(|(group, policy)| ((group.to_string(), policy.name.clone()), policy))
            .collect()
    }

    /// A policy absent from its target group is classified new (safe to create)
    #[test]
    fn classify_new_when_absent() {
        let existing = index(vec![]);
        let bundled = policy("allow-all", &["b"], false);
        assert!(matches!(
            classify_policy(&bundled, &existing),
            PolicyClassification::New
        ));
    }

    /// A policy with the same name in a *different* group is not a match — scoping by
    /// group is what disambiguates the otherwise-ambiguous name
    #[test]
    fn classify_new_when_same_name_other_group() {
        let existing = index(vec![(
            "a",
            existing_policy("allow-all", &["a"], false, Uuid::from_u128(1)),
        )]);
        let bundled = policy("allow-all", &["b"], false);
        assert!(matches!(
            classify_policy(&bundled, &existing),
            PolicyClassification::New
        ));
    }

    /// A policy present in its target group with matching rules is unchanged
    #[test]
    fn classify_unchanged_when_rules_match() {
        let existing = index(vec![(
            "a",
            existing_policy("allow-all", &["a"], false, Uuid::from_u128(1)),
        )]);
        let bundled = policy("allow-all", &["a"], false);
        assert!(matches!(
            classify_policy(&bundled, &existing),
            PolicyClassification::Unchanged
        ));
    }

    /// An instance policy that spans more groups than the bundle targets is still
    /// unchanged when the rules match (group membership is ignored in the comparison)
    #[test]
    fn classify_unchanged_ignores_group_superset() {
        let existing = index(vec![(
            "a",
            existing_policy("allow-all", &["a", "z"], false, Uuid::from_u128(1)),
        )]);
        let bundled = policy("allow-all", &["a"], false);
        assert!(matches!(
            classify_policy(&bundled, &existing),
            PolicyClassification::Unchanged
        ));
    }

    /// Rule/flag drift against an existing policy is a mismatch carrying the drift fields
    /// and no missing groups
    #[test]
    fn classify_mismatched_on_rule_drift() {
        // differ only on default_policy
        let existing = index(vec![(
            "a",
            existing_policy("allow-all", &["a"], true, Uuid::from_u128(1)),
        )]);
        let bundled = policy("allow-all", &["a"], false);
        match classify_policy(&bundled, &existing) {
            PolicyClassification::Mismatched {
                drift,
                missing_groups,
                existing,
            } => {
                assert!(drift.iter().any(|field| field == "default_policy"));
                assert!(missing_groups.is_empty());
                assert_eq!(existing.len(), 1);
            }
            other => panic!("expected mismatched, got {other:?}"),
        }
    }

    /// A policy that targets multiple groups but exists in only some is a mismatch that
    /// names the missing groups, with no rule/flag drift
    #[test]
    fn classify_mismatched_on_partial_coverage() {
        let existing = index(vec![(
            "a",
            existing_policy("allow-all", &["a"], false, Uuid::from_u128(1)),
        )]);
        let bundled = policy("allow-all", &["a", "b"], false);
        match classify_policy(&bundled, &existing) {
            PolicyClassification::Mismatched {
                drift,
                missing_groups,
                ..
            } => {
                assert!(drift.is_empty());
                assert_eq!(missing_groups, vec!["b".to_string()]);
            }
            other => panic!("expected mismatched, got {other:?}"),
        }
    }

    /// Rule drift and a coverage gap can occur together; both are reported
    #[test]
    fn classify_mismatched_drift_and_coverage() {
        let existing = index(vec![(
            "a",
            existing_policy("allow-all", &["a"], true, Uuid::from_u128(1)),
        )]);
        let bundled = policy("allow-all", &["a", "b"], false);
        match classify_policy(&bundled, &existing) {
            PolicyClassification::Mismatched {
                drift,
                missing_groups,
                ..
            } => {
                assert!(drift.iter().any(|field| field == "default_policy"));
                assert_eq!(missing_groups, vec!["b".to_string()]);
            }
            other => panic!("expected mismatched, got {other:?}"),
        }
    }

    /// `None` rules mean "allow all", so the direction is cleared
    #[test]
    fn rule_direction_delta_none_clears() {
        let (clear, deny_all, remove, add) = rule_direction_delta(None, None);
        assert!(clear);
        assert!(!deny_all);
        assert!(remove.is_empty());
        assert!(add.is_empty());
    }

    /// An empty rule set means "deny all", which must use the flag, not a bare remove
    #[test]
    fn rule_direction_delta_empty_denies_all() {
        let target: Vec<NetworkPolicyRuleRaw> = Vec::new();
        let (clear, deny_all, remove, add) = rule_direction_delta(Some(&target), None);
        assert!(!clear);
        assert!(deny_all);
        assert!(remove.is_empty());
        assert!(add.is_empty());
    }

    /// Explicit rules drop the existing rules by id and add the target rules
    #[test]
    fn rule_direction_delta_rules_remove_and_add() {
        let rule_id = Uuid::from_u128(7);
        let existing_rules = vec![NetworkPolicyRule {
            id: rule_id,
            ..NetworkPolicyRule::default()
        }];
        let target = vec![NetworkPolicyRuleRaw {
            allowed_local: true,
            ..NetworkPolicyRuleRaw::default()
        }];
        let (clear, deny_all, remove, add) =
            rule_direction_delta(Some(&target), Some(&existing_rules));
        assert!(!clear);
        assert!(!deny_all);
        assert_eq!(remove, vec![rule_id]);
        assert_eq!(add.len(), 1);
        assert!(add[0].allowed_local);
    }

    /// A groups-only update adds the groups and never touches rules or flags, and never
    /// removes groups
    #[test]
    fn build_delta_groups_only() {
        let chosen = policy("p", &["a", "b"], false);
        let existing = existing_policy("p", &["a"], false, Uuid::from_u128(1));
        let delta = build_delta(&chosen, &[], &["b".to_string()], &existing);
        assert_eq!(delta.add_groups, vec!["b".to_string()]);
        assert!(delta.remove_groups.is_empty());
        assert!(delta.remove_ingress.is_empty());
        assert!(delta.add_ingress.is_empty());
        assert!(!delta.clear_ingress);
        assert!(!delta.deny_all_ingress);
        assert!(delta.forced_policy.is_none());
        assert!(delta.default_policy.is_none());
    }

    /// A rule-drift update removes the existing ingress by id, adds the toolbox's ingress,
    /// and toggles only the flags that drifted, never removing groups
    #[test]
    fn build_delta_rule_drift() {
        let rule_id = Uuid::from_u128(9);
        let mut existing = existing_policy("p", &["a"], false, Uuid::from_u128(1));
        existing.ingress = Some(vec![NetworkPolicyRule {
            id: rule_id,
            ..NetworkPolicyRule::default()
        }]);
        let mut chosen = policy("p", &["a"], true);
        chosen.ingress = Some(vec![NetworkPolicyRuleRaw {
            allowed_local: true,
            ..NetworkPolicyRuleRaw::default()
        }]);
        let drift = vec!["ingress".to_string(), "default_policy".to_string()];
        let delta = build_delta(&chosen, &drift, &[], &existing);
        assert_eq!(delta.remove_ingress, vec![rule_id]);
        assert_eq!(delta.add_ingress.len(), 1);
        assert!(delta.add_ingress[0].allowed_local);
        assert_eq!(delta.default_policy, Some(true));
        assert!(delta.forced_policy.is_none());
        assert!(delta.remove_groups.is_empty());
    }

    /// `None` toolbox ingress clears the direction (allow-all)
    #[test]
    fn build_delta_clears_ingress_when_none() {
        let mut existing = existing_policy("p", &["a"], false, Uuid::from_u128(1));
        existing.ingress = Some(vec![NetworkPolicyRule {
            id: Uuid::from_u128(9),
            ..NetworkPolicyRule::default()
        }]);
        // chosen.ingress is None (allow all)
        let chosen = policy("p", &["a"], false);
        let delta = build_delta(&chosen, &["ingress".to_string()], &[], &existing);
        assert!(delta.clear_ingress);
        assert!(!delta.deny_all_ingress);
        assert!(delta.add_ingress.is_empty());
    }

    /// An empty-`Some` toolbox ingress denies all via the flag, not an empty remove
    #[test]
    fn build_delta_deny_all_when_empty_some() {
        let existing = existing_policy("p", &["a"], false, Uuid::from_u128(1));
        let mut chosen = policy("p", &["a"], false);
        chosen.ingress = Some(Vec::new());
        let delta = build_delta(&chosen, &["ingress".to_string()], &[], &existing);
        assert!(delta.deny_all_ingress);
        assert!(!delta.clear_ingress);
        assert!(delta.add_ingress.is_empty());
    }

    /// One bundled name overlapping two distinct existing policies updates only the one
    /// that drifts, leaving the matching one untouched
    #[test]
    fn build_updates_only_targets_drifting_policy() {
        // policy "p" matches in group a (id 1) but differs in group b (id 2)
        let matching = existing_policy("p", &["a"], false, Uuid::from_u128(1));
        let drifting = existing_policy("p", &["b"], true, Uuid::from_u128(2));
        let mismatch = PolicyMismatch {
            name: "p".to_string(),
            drift: vec!["default_policy".to_string()],
            missing_groups: Vec::new(),
            chosen: policy("p", &["a", "b"], false),
            existing: vec![matching, drifting],
        };
        let updates = build_updates(&mismatch);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].id, Uuid::from_u128(2));
        assert!(updates[0].drift.iter().any(|field| field == "default_policy"));
    }

    /// A coverage gap is folded into the first existing policy as an additive group add
    #[test]
    fn build_updates_adds_missing_coverage() {
        let existing = existing_policy("p", &["a"], false, Uuid::from_u128(1));
        let mismatch = PolicyMismatch {
            name: "p".to_string(),
            drift: Vec::new(),
            missing_groups: vec!["b".to_string()],
            chosen: policy("p", &["a", "b"], false),
            existing: vec![existing],
        };
        let updates = build_updates(&mismatch);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].add_groups, vec!["b".to_string()]);
        assert!(updates[0].drift.is_empty());
    }
}
