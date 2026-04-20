//! A journal of applied import actions that can be rolled back
//!
//! Multi-resource imports (a pipeline plus all of its images, or a toolbox)
//! can stop partway through — the user quits the merge editor, or an apply
//! step fails. The journal records every change as it lands so the user can
//! choose to undo the partial import instead of being left with half the
//! resources created.
//!
//! Undo strategy per action:
//! - created resources/groups are deleted
//! - updated resources are reverted by diffing the live state back to the
//!   original snapshot taken before the update (this also correctly reverts
//!   partial edits made through the merge editor)
//!
//! Registry writes (docker pushes) are intentionally NOT journaled — deleting
//! tags from arbitrary registries is registry-specific and often unauthorized.

use colored::Colorize;
use std::sync::Mutex;
use thorium::models::{
    Image, ImageRequest, NetworkPolicy, NetworkPolicyRequest, NetworkPolicyUpdate, Pipeline,
    PipelineRequest,
};
use thorium::{Error, Thorium};

use super::update;
use crate::handlers::progress::{Bar, BarKind};
use crate::handlers::toolbox::policies;

/// A single applied change and how to undo it
enum AppliedAction {
    /// A group was created; undo by deleting it
    CreatedGroup { name: String },
    /// An image was created; undo by deleting it
    CreatedImage { group: String, name: String },
    /// A pipeline was created; undo by deleting it
    CreatedPipeline { group: String, name: String },
    /// A network policy was created; undo by deleting it
    CreatedNetworkPolicy { name: String },
    /// A network policy was updated; undo by reverting to the pre-update snapshot
    UpdatedNetworkPolicy { original: Box<NetworkPolicy> },
    /// An image was updated; undo by reverting to the pre-update snapshot
    UpdatedImage { original: Box<Image> },
    /// A pipeline was updated; undo by reverting to the pre-update snapshot
    UpdatedPipeline { original: Box<Pipeline> },
}

impl AppliedAction {
    /// A one-line human-readable description of this applied change
    fn describe(&self) -> String {
        match self {
            Self::CreatedGroup { name } => format!("created group '{name}'"),
            Self::CreatedImage { group, name } => format!("created image '{group}:{name}'"),
            Self::CreatedPipeline { group, name } => format!("created pipeline '{group}:{name}'"),
            Self::CreatedNetworkPolicy { name } => format!("created network policy '{name}'"),
            Self::UpdatedNetworkPolicy { original } => {
                format!("updated network policy '{}' (id {})", original.name, original.id)
            }
            Self::UpdatedImage { original } => {
                format!("updated image '{}:{}'", original.group, original.name)
            }
            Self::UpdatedPipeline { original } => {
                format!("updated pipeline '{}:{}'", original.group, original.name)
            }
        }
    }
}

/// The journal of changes applied so far by an import run
///
/// Records happen behind a mutex so concurrent create/update tasks can append
/// while still capturing everything that landed before a sibling task failed.
#[derive(Default)]
pub struct Journal {
    /// The applied changes in the order they landed
    actions: Mutex<Vec<AppliedAction>>,
}

impl Journal {
    /// Create an empty journal
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an applied action onto the journal
    fn push(&self, action: AppliedAction) {
        // a poisoned mutex means another import task panicked; the journal is
        // best-effort book-keeping, so keep recording with whatever we have
        let mut actions = self.actions.lock().unwrap_or_else(|err| err.into_inner());
        actions.push(action);
    }

    /// Whether any changes have been applied yet
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of applied changes
    pub fn len(&self) -> usize {
        self.actions
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .len()
    }

    /// Describe every applied change in apply order
    pub fn describe(&self) -> Vec<String> {
        self.actions
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .iter()
            .map(AppliedAction::describe)
            .collect()
    }

    /// Record a created group
    pub fn created_group<T: Into<String>>(&self, name: T) {
        self.push(AppliedAction::CreatedGroup { name: name.into() });
    }

    /// Record a created image
    pub fn created_image<G: Into<String>, N: Into<String>>(&self, group: G, name: N) {
        self.push(AppliedAction::CreatedImage {
            group: group.into(),
            name: name.into(),
        });
    }

    /// Record a created pipeline
    pub fn created_pipeline<G: Into<String>, N: Into<String>>(&self, group: G, name: N) {
        self.push(AppliedAction::CreatedPipeline {
            group: group.into(),
            name: name.into(),
        });
    }

    /// Record a created network policy
    pub fn created_network_policy<T: Into<String>>(&self, name: T) {
        self.push(AppliedAction::CreatedNetworkPolicy { name: name.into() });
    }

    /// Record an updated network policy, snapshotting its pre-update state
    pub fn updated_network_policy(&self, original: NetworkPolicy) {
        self.push(AppliedAction::UpdatedNetworkPolicy {
            original: Box::new(original),
        });
    }

    /// Record an updated image, snapshotting its pre-update state
    pub fn updated_image(&self, original: Image) {
        self.push(AppliedAction::UpdatedImage {
            original: Box::new(original),
        });
    }

    /// Record an updated pipeline, snapshotting its pre-update state
    pub fn updated_pipeline(&self, original: Pipeline) {
        self.push(AppliedAction::UpdatedPipeline {
            original: Box::new(original),
        });
    }

    /// Undo every applied change in reverse order, best-effort
    ///
    /// Rolls back pipelines before the images they reference and resources
    /// before the groups that hold them simply by replaying in reverse, since
    /// imports always create groups, then images, then pipelines. Undo failures
    /// are collected rather than fatal so one un-undoable action does not strand
    /// the rest of the partial import; if any action failed, a combined error
    /// listing them is returned after every action has been attempted.
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client used to undo changes
    /// * `progress` - The progress bar to update as changes are undone
    pub async fn rollback(self, thorium: &Thorium, progress: &Bar) -> Result<(), Error> {
        let actions = self
            .actions
            .into_inner()
            .unwrap_or_else(|err| err.into_inner());
        progress.refresh("Rolling back", BarKind::Bound(actions.len() as u64));
        // collect per-action failures instead of stopping at the first one: rollback
        // is best-effort restoration, so a single un-undoable action must not strand
        // the rest of the partial import in place
        let mut errors: Vec<String> = Vec::new();
        for action in actions.into_iter().rev() {
            let label = action.describe();
            // undo this single action, returning the raw API error (if any) so the
            // loop can record it and keep going rather than short-circuiting
            let outcome: Result<(), Error> = async {
                match action {
                    AppliedAction::CreatedGroup { name } => {
                        thorium.groups.delete(&name).await?;
                    }
                    AppliedAction::CreatedImage { group, name } => {
                        thorium.images.delete(&group, &name).await?;
                    }
                    AppliedAction::CreatedPipeline { group, name } => {
                        thorium.pipelines.delete(&group, &name).await?;
                    }
                    AppliedAction::CreatedNetworkPolicy { name } => {
                        thorium.network_policies.delete(&name, None).await?;
                    }
                    AppliedAction::UpdatedNetworkPolicy { original } => {
                        // unbox the owned snapshot, keeping the identity needed to target
                        // the exact policy after it is borrowed below
                        let original = *original;
                        let (name, id) = (original.name.clone(), original.id);
                        // diff the live state back to the snapshot so the rule/flag/group
                        // changes the import made are reverted exactly (rollback may remove
                        // groups, unlike the additive-only import path)
                        let live = thorium.network_policies.get(&name, Some(id)).await?;
                        let revert = network_policy_restore(&original, &live);
                        thorium.network_policies.update(&name, Some(id), &revert).await?;
                    }
                    AppliedAction::UpdatedImage { original } => {
                        // unbox the owned snapshot; clone only the identity we need after
                        // it is moved into the request below, not the whole Image
                        let original = *original;
                        let (group, name) = (original.group.clone(), original.name.clone());
                        // diff the live state back to the snapshot so even partial
                        // editor merges revert cleanly
                        let live = thorium.images.get(&group, &name).await?;
                        if let Some(revert) =
                            update::calculate_image_update(live, ImageRequest::from(original))
                        {
                            thorium.images.update(&group, &name, &revert).await?;
                        }
                    }
                    AppliedAction::UpdatedPipeline { original } => {
                        // unbox the owned snapshot; clone only the identity we need after
                        // it is moved into the request below, not the whole Pipeline
                        let original = *original;
                        let (group, name) = (original.group.clone(), original.name.clone());
                        let live = thorium.pipelines.get(&group, &name).await?;
                        if let Some(revert) =
                            update::calculate_pipeline_update(live, PipelineRequest::from(original))
                        {
                            thorium.pipelines.update(&group, &name, &revert).await?;
                        }
                    }
                }
                Ok(())
            }
            .await;
            // record a failure and surface it, or report the successful undo
            match outcome {
                Ok(()) => progress.info_anonymous(format!("Rolled back: {label}")),
                Err(err) => {
                    progress.warning(format!("Failed to roll back {label}: {err}"));
                    errors.push(format!("{label}: {err}"));
                }
            }
            progress.inc(1);
        }
        // if any action failed to undo, report them together so the caller knows the
        // restore was incomplete (the original import error is still propagated above)
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::new(format!(
                "Rollback completed with {} error(s):\n  {}",
                errors.len(),
                errors.join("\n  ")
            )))
        }
    }
}

/// Build the delta that restores a live network policy back to a prior snapshot
///
/// Unlike the additive-only import update, a rollback must restore the prior state
/// exactly, so groups are synced in both directions and every rule/flag is reset to the
/// snapshot. Ingress/egress reuse the toolbox path's three-case rule delta, treating the
/// prior rules (converted to their raw request form) as the target.
///
/// # Arguments
///
/// * `prior` - The pre-update snapshot to restore to
/// * `live` - The policy's current state to diff back from
fn network_policy_restore(prior: &NetworkPolicy, live: &NetworkPolicy) -> NetworkPolicyUpdate {
    // the prior rules in raw request form so they can be re-added
    let prior_request = NetworkPolicyRequest::from(prior);
    // groups added during the import are removed and any the import removed are restored
    let add_groups: Vec<String> = prior
        .groups
        .iter()
        .filter(|group| !live.groups.contains(*group))
        .cloned()
        .collect();
    let remove_groups: Vec<String> = live
        .groups
        .iter()
        .filter(|group| !prior.groups.contains(*group))
        .cloned()
        .collect();
    // restore ingress to exactly the prior rules
    let (clear_ingress, deny_all_ingress, remove_ingress, add_ingress) =
        policies::rule_direction_delta(prior_request.ingress.as_ref(), live.ingress.as_ref());
    // restore egress symmetrically
    let (clear_egress, deny_all_egress, remove_egress, add_egress) =
        policies::rule_direction_delta(prior_request.egress.as_ref(), live.egress.as_ref());
    NetworkPolicyUpdate {
        new_name: None,
        add_groups,
        remove_groups,
        add_ingress,
        remove_ingress,
        clear_ingress,
        deny_all_ingress,
        add_egress,
        remove_egress,
        clear_egress,
        deny_all_egress,
        // flags are always reset to the snapshot's values
        forced_policy: Some(prior.forced_policy),
        default_policy: Some(prior.default_policy),
    }
}

/// Ask the user whether a partial import should be rolled back
///
/// # Arguments
///
/// * `journal` - The journal of changes applied before the import stopped
pub fn confirm_rollback(journal: &Journal) -> Result<bool, Error> {
    println!(
        "\n{} changes were applied before the import stopped:",
        journal.len().to_string().bright_yellow()
    );
    for line in journal.describe() {
        println!("  {line}");
    }
    dialoguer::Confirm::new()
        .with_prompt("Roll back the changes listed above?")
        .default(false)
        .interact()
        .map_err(|err| Error::new(format!("Failed to read rollback choice: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The journal describes applied actions in apply order
    #[test]
    fn describe_preserves_apply_order() {
        let journal = Journal::new();
        journal.created_group("grp");
        journal.created_image("grp", "img");
        journal.created_pipeline("grp", "pipe");
        let described = journal.describe();
        assert_eq!(
            described,
            vec![
                "created group 'grp'".to_string(),
                "created image 'grp:img'".to_string(),
                "created pipeline 'grp:pipe'".to_string(),
            ]
        );
    }

    /// Empty journals report empty so callers can skip the rollback prompt
    #[test]
    fn empty_journal_is_empty() {
        let journal = Journal::new();
        assert!(journal.is_empty());
        assert_eq!(journal.len(), 0);
    }

    /// Build a stored `NetworkPolicy` for the restore tests
    fn np(
        groups: &[&str],
        default_policy: bool,
        ingress: Option<Vec<thorium::models::NetworkPolicyRule>>,
    ) -> NetworkPolicy {
        let id = uuid::Uuid::from_u128(1);
        NetworkPolicy {
            name: "p".to_string(),
            id,
            k8s_name: format!("p-{id}"),
            groups: groups.iter().map(|group| (*group).to_string()).collect(),
            created: chrono::DateTime::from_timestamp(0, 0).expect("valid epoch"),
            ingress,
            egress: None,
            forced_policy: false,
            default_policy,
            used_by: std::collections::HashMap::new(),
        }
    }

    /// Restoring reverts the groups, rules, and flags an import changed, removing the
    /// groups the import added and re-adding the prior rules
    #[test]
    fn network_policy_restore_reverts_groups_rules_flags() {
        let prior_rule_id = uuid::Uuid::from_u128(10);
        let live_rule_id = uuid::Uuid::from_u128(11);
        // before the import: one group, a rule, default off
        let prior = np(
            &["a"],
            false,
            Some(vec![thorium::models::NetworkPolicyRule {
                id: prior_rule_id,
                allowed_local: true,
                ..thorium::models::NetworkPolicyRule::default()
            }]),
        );
        // after the import overwrote it: extra group, different rule, default on
        let live = np(
            &["a", "b"],
            true,
            Some(vec![thorium::models::NetworkPolicyRule {
                id: live_rule_id,
                ..thorium::models::NetworkPolicyRule::default()
            }]),
        );
        let restore = network_policy_restore(&prior, &live);
        // the import-added group is removed and nothing extra is added
        assert!(restore.add_groups.is_empty());
        assert_eq!(restore.remove_groups, vec!["b".to_string()]);
        // the live rule is dropped and the prior rule re-added
        assert_eq!(restore.remove_ingress, vec![live_rule_id]);
        assert_eq!(restore.add_ingress.len(), 1);
        assert!(restore.add_ingress[0].allowed_local);
        // flags reset to the prior values
        assert_eq!(restore.default_policy, Some(false));
        assert_eq!(restore.forced_policy, Some(false));
    }

    /// A groups-only import is inverted by removing exactly the added group
    #[test]
    fn network_policy_restore_groups_only() {
        let prior = np(&["a"], false, None);
        let live = np(&["a", "b"], false, None);
        let restore = network_policy_restore(&prior, &live);
        assert_eq!(restore.remove_groups, vec!["b".to_string()]);
        assert!(restore.add_groups.is_empty());
    }

    /// Restoring a deny-all (empty `Some`) prior uses the deny-all flag, not an empty
    /// remove that would collapse to allow-all
    #[test]
    fn network_policy_restore_deny_all_prior() {
        // prior denied all ingress; the import cleared it to allow-all (None)
        let prior = np(&["a"], false, Some(Vec::new()));
        let live = np(&["a"], false, None);
        let restore = network_policy_restore(&prior, &live);
        assert!(restore.deny_all_ingress);
        assert!(!restore.clear_ingress);
        assert!(restore.add_ingress.is_empty());
    }
}
