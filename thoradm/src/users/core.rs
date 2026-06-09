//! The pure logic backing the user email dedupe commands
//!
//! These functions perform no Redis or API calls so they can be unit tested in
//! isolation.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use thorium::models::{ScrubbedUser, UserRole};

use crate::Error;

/// Build the Redis key to a users data hash
///
/// This must match the canonical key built by `UserKeys::data` in
/// `api/src/models/backends/db/keys/users.rs` (the `db` backend is gated behind
/// the API crate's heavy `api` feature so we mirror the format here instead).
///
/// # Arguments
///
/// * `ns` - The Thorium namespace these keys live under
/// * `username` - The username to build the data key for
pub fn user_data_key(ns: &str, username: &str) -> String {
    format!("{ns}:user_data:{username}")
}

/// Build the Redis key to the email->username map hash
///
/// This must match the canonical key built by `UserKeys::by_email` in
/// `api/src/models/backends/db/keys/users.rs` (the `db` backend is gated behind
/// the API crate's heavy `api` feature so we mirror the format here instead).
///
/// # Arguments
///
/// * `ns` - The Thorium namespace these keys live under
pub fn email_map_key(ns: &str) -> String {
    format!("{ns}:users_email_map")
}

/// A group of user accounts that share the same email address
pub struct DuplicateGroup {
    /// The normalized email shared by these accounts
    pub email: String,
    /// The accounts that share this email
    pub members: Vec<ScrubbedUser>,
}

/// A single email change to apply to an account
pub struct EmailChange {
    /// The username of the account to update
    pub username: String,
    /// The new, unique email to assign to this account
    pub new_email: String,
}

/// The full set of changes required to dedupe emails
pub struct DedupePlan {
    /// The email changes to apply to user accounts
    pub changes: Vec<EmailChange>,
    /// The emails to remove from the email->username map (no account retains them)
    pub map_removals: Vec<String>,
    /// The email->username mappings to (re)assert so every non-system user is mapped
    pub map_additions: Vec<(String, String)>,
}

/// Normalize an email for comparison
///
/// Emails are compared case-insensitively with surrounding whitespace trimmed.
///
/// # Arguments
///
/// * `email` - The email to normalize
pub fn normalize_email(email: &str) -> String {
    // trim surrounding whitespace and lowercase for case-insensitive comparison
    email.trim().to_lowercase()
}

/// A short human readable label for a users role
///
/// # Arguments
///
/// * `role` - The role to build a label for
pub fn role_label(role: &UserRole) -> &'static str {
    // map each role to a concise label for reporting
    match role {
        UserRole::Admin => "admin",
        UserRole::Analyst => "analyst",
        UserRole::Developer { .. } => "developer",
        UserRole::User => "user",
    }
}

/// Find groups of non-system accounts that share an email address
///
/// Accounts whose email matches the exempt system email are skipped since system
/// accounts are allowed to share a single email. Only emails used by two or more
/// accounts are returned.
///
/// # Arguments
///
/// * `users` - The users to scan for duplicate emails
/// * `system_email` - The email used by system accounts which is allowed to be duplicated
pub fn find_duplicate_groups(users: &[ScrubbedUser], system_email: &str) -> Vec<DuplicateGroup> {
    // normalize the exempt system email once for comparison
    let system = normalize_email(system_email);
    // group users by their normalized email (BTreeMap for deterministic ordering)
    let mut by_email: BTreeMap<String, Vec<ScrubbedUser>> = BTreeMap::new();
    // crawl over each user and bucket them by their normalized email
    for user in users {
        // normalize this users email
        let norm = normalize_email(&user.email);
        // skip any accounts using the exempt system email
        if norm == system {
            continue;
        }
        // add this user to the group for their email
        by_email.entry(norm).or_default().push(user.clone());
    }
    // keep only the emails shared by two or more accounts
    by_email
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .map(|(email, members)| DuplicateGroup { email, members })
        .collect()
}

/// Validate that applying a set of changes leaves every account with a unique email
///
/// This projects the final email of every account (current email overlaid with any
/// change) and ensures no two non-system accounts share a normalized email and that
/// no change targets the reserved system email. It is a safety net over the checks
/// performed during interactive collection.
///
/// # Arguments
///
/// * `users` - The current set of users
/// * `changes` - The email changes that will be applied
/// * `system_email` - The email used by system accounts which is allowed to be duplicated
pub fn validate_plan(
    users: &[ScrubbedUser],
    changes: &[EmailChange],
    system_email: &str,
) -> Result<(), Error> {
    // normalize the exempt system email once for comparison
    let system = normalize_email(system_email);
    // build the projected final normalized email for every account
    let mut finals: BTreeMap<String, String> = users
        .iter()
        .map(|user| (user.username.clone(), normalize_email(&user.email)))
        .collect();
    // overlay each change onto the projected final state
    for change in changes {
        // reject assigning the reserved system email to a normal account
        if normalize_email(&change.new_email) == system {
            return Err(Error::new(format!(
                "Cannot assign the reserved system email '{}' to {}",
                system_email, change.username
            )));
        }
        // update this accounts projected email
        finals.insert(change.username.clone(), normalize_email(&change.new_email));
    }
    // track which account claimed each final email to detect collisions
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    // crawl over the projected emails and ensure each is unique
    for (username, email) in &finals {
        // the system email is exempt from the uniqueness requirement
        if email == &system {
            continue;
        }
        // error if this email was already claimed by a different account
        if let Some(other) = seen.insert(email.clone(), username.clone()) {
            return Err(Error::new(format!(
                "Email '{email}' is still shared by '{other}' and '{username}'"
            )));
        }
    }
    Ok(())
}

/// Compute the email->username map removals and additions for a dedupe plan
///
/// The email map is keyed by the literal (exact) email string. This returns:
///
/// * `removals` - shared duplicate emails that no account retains after the changes
///   and that currently exist in the map, so their stale entries are deleted.
/// * `additions` - the `(email, username)` pairs needed so that *every* non-system
///   account is mapped to its username. This both adds users missing from the map
///   (the map was only ever populated with `hsetnx` at create time) and corrects
///   entries pointing at the wrong username. System accounts (those using the exempt
///   email) are left untouched since they share a single non-unique email.
///
/// # Arguments
///
/// * `users` - The current set of users
/// * `groups` - The duplicate groups being resolved
/// * `changes` - The email changes that will be applied
/// * `current_map` - The current email->username map read from Redis
/// * `system_email` - The email used by system accounts which is allowed to be duplicated
pub fn compute_map_plan(
    users: &[ScrubbedUser],
    groups: &[DuplicateGroup],
    changes: &[EmailChange],
    current_map: &HashMap<String, String>,
    system_email: &str,
) -> (Vec<String>, Vec<(String, String)>) {
    // normalize the exempt system email once for comparison
    let system = normalize_email(system_email);
    // build the final exact email for every account starting from current state
    let mut final_email: BTreeMap<String, String> = users
        .iter()
        .map(|user| (user.username.clone(), user.email.clone()))
        .collect();
    // overlay each change onto the final exact email state
    for change in changes {
        final_email.insert(change.username.clone(), change.new_email.clone());
    }
    // collect the exact email string of every account in a duplicate group
    let mut group_emails: BTreeSet<String> = BTreeSet::new();
    for group in groups {
        for member in &group.members {
            group_emails.insert(member.email.clone());
        }
    }
    // remove any freed duplicate email that still has a stale entry in the map
    let mut removals = Vec::new();
    // crawl over the duplicate emails and decide which are now orphaned
    for email in group_emails {
        // count how many accounts still end up owning this exact email
        let owned = final_email.values().any(|current| current == &email);
        // remove the entry if it is now orphaned but still present in the map
        if !owned && current_map.contains_key(&email) {
            removals.push(email);
        }
    }
    // (re)assert a mapping for every non-system account that is missing or wrong
    let mut additions = Vec::new();
    // crawl over every accounts final email (sorted by username for determinism)
    for (username, email) in &final_email {
        // skip system accounts since they share the exempt email
        if normalize_email(email) == system {
            continue;
        }
        // add or correct the mapping if it does not already point at this user
        if current_map.get(email) != Some(username) {
            additions.push((email.clone(), username.clone()));
        }
    }
    (removals, additions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;
    use thorium::models::{ScrubbedUser, UserRole, UserSettings};

    /// Build a minimal user fixture with a username and email
    fn user(username: &str, email: &str) -> ScrubbedUser {
        ScrubbedUser {
            username: username.to_string(),
            role: UserRole::User,
            email: email.to_string(),
            groups: Vec::new(),
            token: String::new(),
            token_expiration: Utc::now(),
            unix: None,
            settings: UserSettings::default(),
            local: false,
            verified: true,
        }
    }

    #[test]
    fn key_builders_match_canonical_format() {
        // the data and email map keys must match the api crates UserKeys format
        assert_eq!(
            user_data_key("thorium", "mcarson"),
            "thorium:user_data:mcarson"
        );
        assert_eq!(email_map_key("thorium"), "thorium:users_email_map");
    }

    #[test]
    fn normalize_trims_and_lowercases() {
        // surrounding whitespace is trimmed and the email is lowercased
        assert_eq!(normalize_email("  Foo@Bar.COM "), "foo@bar.com");
    }

    #[test]
    fn grouping_excludes_system_and_singletons() {
        // build users with a shared email, the system email, and unique emails
        let users = vec![
            user("a", "dup@x.com"),
            user("b", "dup@x.com"),
            user("sys1", "thorium"),
            user("sys2", "thorium"),
            user("c", "unique@x.com"),
        ];
        // find the duplicate groups exempting the system email
        let groups = find_duplicate_groups(&users, "thorium");
        // only the shared non-system email should be returned
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].email, "dup@x.com");
        assert_eq!(groups[0].members.len(), 2);
    }

    #[test]
    fn grouping_is_case_insensitive() {
        // build two users whose emails only differ in case
        let users = vec![user("a", "Dup@X.com"), user("b", "dup@x.COM")];
        // they should be detected as a single duplicate group
        let groups = find_duplicate_groups(&users, "thorium");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 2);
    }

    #[test]
    fn validate_rejects_unchanged_duplicates() {
        // two users share an email and no changes are made
        let users = vec![user("a", "dup@x.com"), user("b", "dup@x.com")];
        // validation should fail since the duplicate remains
        assert!(validate_plan(&users, &[], "thorium").is_err());
    }

    #[test]
    fn validate_rejects_same_normalized_change() {
        // two users share an email and one is "changed" to the same email
        let users = vec![user("a", "dup@x.com"), user("b", "dup@x.com")];
        let changes = vec![EmailChange {
            username: "b".to_string(),
            new_email: "DUP@x.com".to_string(),
        }];
        // the change does not actually make the emails unique
        assert!(validate_plan(&users, &changes, "thorium").is_err());
    }

    #[test]
    fn validate_rejects_reserved_system_email() {
        // a user is changed to the reserved system email
        let users = vec![user("a", "dup@x.com"), user("b", "dup@x.com")];
        let changes = vec![EmailChange {
            username: "b".to_string(),
            new_email: "thorium".to_string(),
        }];
        // assigning the reserved system email is not allowed
        assert!(validate_plan(&users, &changes, "thorium").is_err());
    }

    #[test]
    fn validate_rejects_inter_group_collision() {
        // two separate duplicate groups exist
        let users = vec![
            user("a", "x@x.com"),
            user("b", "x@x.com"),
            user("c", "z@z.com"),
            user("d", "z@z.com"),
        ];
        // resolving group x by stealing group z's email keeps a collision
        let changes = vec![EmailChange {
            username: "a".to_string(),
            new_email: "z@z.com".to_string(),
        }];
        assert!(validate_plan(&users, &changes, "thorium").is_err());
    }

    #[test]
    fn validate_accepts_unique_plan() {
        // two users share an email and one is given a unique new email
        let users = vec![user("a", "dup@x.com"), user("b", "dup@x.com")];
        let changes = vec![EmailChange {
            username: "b".to_string(),
            new_email: "new@x.com".to_string(),
        }];
        // validation should succeed since all emails are now unique
        assert!(validate_plan(&users, &changes, "thorium").is_ok());
    }

    #[test]
    fn map_plan_adds_all_non_system_when_map_empty() {
        // two users share an email and one keeps it while the other changes
        let users = vec![user("a", "dup@x.com"), user("b", "dup@x.com")];
        let groups = find_duplicate_groups(&users, "thorium");
        let changes = vec![EmailChange {
            username: "b".to_string(),
            new_email: "new@x.com".to_string(),
        }];
        // the email map starts out empty so every account needs adding
        let current_map = HashMap::new();
        // compute the resulting map plan
        let (removals, additions) =
            compute_map_plan(&users, &groups, &changes, &current_map, "thorium");
        // nothing is fully freed so there are no removals
        assert!(removals.is_empty());
        // the retained email maps to a and the new email maps to b
        assert!(additions.contains(&("dup@x.com".to_string(), "a".to_string())));
        assert!(additions.contains(&("new@x.com".to_string(), "b".to_string())));
    }

    #[test]
    fn map_plan_removes_fully_freed_email() {
        // two users share an email and both move to new emails
        let users = vec![user("a", "dup@x.com"), user("b", "dup@x.com")];
        let groups = find_duplicate_groups(&users, "thorium");
        let changes = vec![
            EmailChange {
                username: "a".to_string(),
                new_email: "a@x.com".to_string(),
            },
            EmailChange {
                username: "b".to_string(),
                new_email: "b@x.com".to_string(),
            },
        ];
        // the shared email currently maps to the first registrant
        let current_map = HashMap::from([("dup@x.com".to_string(), "a".to_string())]);
        // compute the resulting map plan
        let (removals, additions) =
            compute_map_plan(&users, &groups, &changes, &current_map, "thorium");
        // the shared email is now owned by nobody so it is removed
        assert_eq!(removals, vec!["dup@x.com".to_string()]);
        // both new emails are asserted into the map
        assert!(additions.contains(&("a@x.com".to_string(), "a".to_string())));
        assert!(additions.contains(&("b@x.com".to_string(), "b".to_string())));
    }

    #[test]
    fn map_plan_handles_case_variant_stale_key() {
        // two users share an email by normalization but with different exact case
        let users = vec![user("a", "Dup@X.com"), user("b", "dup@x.com")];
        let groups = find_duplicate_groups(&users, "thorium");
        // a keeps their exact email and b moves to a new email
        let changes = vec![EmailChange {
            username: "b".to_string(),
            new_email: "new@x.com".to_string(),
        }];
        // the lowercase exact email currently maps to b
        let current_map = HashMap::from([("dup@x.com".to_string(), "b".to_string())]);
        // compute the resulting map plan
        let (removals, additions) =
            compute_map_plan(&users, &groups, &changes, &current_map, "thorium");
        // b's old lowercase exact email is now owned by nobody so it is removed
        assert_eq!(removals, vec!["dup@x.com".to_string()]);
        // a's exact email and b's new email are asserted into the map
        assert!(additions.contains(&("Dup@X.com".to_string(), "a".to_string())));
        assert!(additions.contains(&("new@x.com".to_string(), "b".to_string())));
    }

    #[test]
    fn map_plan_skips_already_correct_entries() {
        // two users with unique emails that are already correctly mapped
        let users = vec![user("a", "a@x.com"), user("b", "b@x.com")];
        // the map already points each email at the right user
        let current_map = HashMap::from([
            ("a@x.com".to_string(), "a".to_string()),
            ("b@x.com".to_string(), "b".to_string()),
        ]);
        // compute the resulting map plan with no duplicates or changes
        let (removals, additions) = compute_map_plan(&users, &[], &[], &current_map, "thorium");
        // there is nothing to remove or add
        assert!(removals.is_empty());
        assert!(additions.is_empty());
    }

    #[test]
    fn map_plan_corrects_wrong_username() {
        // a user whose email maps to a stale username in the map
        let users = vec![user("a", "a@x.com")];
        let current_map = HashMap::from([("a@x.com".to_string(), "old".to_string())]);
        // compute the resulting map plan with no duplicates or changes
        let (removals, additions) = compute_map_plan(&users, &[], &[], &current_map, "thorium");
        // the entry is corrected to point at the real owner
        assert!(removals.is_empty());
        assert_eq!(additions, vec![("a@x.com".to_string(), "a".to_string())]);
    }

    #[test]
    fn map_plan_excludes_system_accounts() {
        // a normal user alongside a system account using the exempt email
        let users = vec![user("a", "a@x.com"), user("sys", "thorium")];
        // the map starts out empty
        let current_map = HashMap::new();
        // compute the resulting map plan with no duplicates or changes
        let (removals, additions) = compute_map_plan(&users, &[], &[], &current_map, "thorium");
        // only the non-system user is added and the system account is left alone
        assert!(removals.is_empty());
        assert_eq!(additions, vec![("a@x.com".to_string(), "a".to_string())]);
    }
}
