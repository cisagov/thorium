//! The interactive duplicate email dedupe command for Thoradm
//!
//! This lists users through the normal Thorium API but writes the resolved emails
//! directly to Redis since the API has no path for updating a users email.

use std::collections::{HashMap, HashSet};
use thorium::models::ScrubbedUser;
use thorium::{Conf, Thorium};

use super::core;
use crate::Error;
use crate::args::{Args, DedupeEmails};

/// Interactively resolve non-system accounts that share an email address
///
/// All users are listed through the normal Thorium API and any duplicate emails
/// are resolved by prompting the admin for new unique emails. The resolved emails
/// are written directly to Redis and the changed accounts are marked unverified.
///
/// # Arguments
///
/// * `opts` - The dedupe command options
/// * `args` - The Thoradm args
pub async fn dedupe(opts: &DedupeEmails, args: &Args) -> Result<(), Error> {
    // build a Thorium client for listing users via the normal API
    let thorium = Thorium::from_ctl_conf_file(&args.ctl_conf).await?;
    // load the cluster config for the namespace and redis connection
    let config = Conf::new(&args.cluster_conf)?;
    // list all users with details via the normal API
    let users = thorium.users.list_details().await?;
    // find groups of accounts that share an email
    let groups = core::find_duplicate_groups(&users, &opts.system_email);
    // interactively resolve any duplicates into a set of email changes
    let changes = if groups.is_empty() {
        // nothing to resolve interactively but we still repair the email map below
        println!(
            "No duplicate emails found ({} accounts scanned).",
            users.len()
        );
        Vec::new()
    } else {
        // print the duplicates we found
        print_groups(&groups);
        // interactively collect new unique emails to resolve every duplicate
        let changes = collect_changes(&users, &groups, &opts.system_email)?;
        // double check the resulting plan leaves every account with a unique email
        core::validate_plan(&users, &changes, &opts.system_email)?;
        changes
    };
    // read the current email->username map directly from redis
    let current_map = read_email_map(&config).await?;
    // compute the map removals and additions so every non-system user is mapped
    let (map_removals, map_additions) =
        core::compute_map_plan(&users, &groups, &changes, &current_map, &opts.system_email);
    // assemble the final plan
    let plan = core::DedupePlan {
        changes,
        map_removals,
        map_additions,
    };
    // there is nothing to do if the plan is completely empty
    if plan.changes.is_empty() && plan.map_removals.is_empty() && plan.map_additions.is_empty() {
        println!("The email map is already consistent, nothing to do.");
        return Ok(());
    }
    // show the plan before applying it
    print_plan(&plan);
    // stop here if this is a dry run
    if opts.dry_run {
        println!("\nDry run - no changes applied.");
        return Ok(());
    }
    // confirm before applying unless the admin opted to skip the prompt
    if !opts.assume_yes {
        // ask the admin to confirm the changes
        let confirmed = dialoguer::Confirm::new()
            .with_prompt("Apply these changes to Redis?")
            .default(false)
            .interact()?;
        // abort if the admin did not confirm
        if !confirmed {
            println!("Aborted - no changes applied.");
            return Ok(());
        }
    }
    // apply the plan directly to redis
    apply_plan(&config, &plan).await?;
    // report success and remind the admin about re-verification
    println!(
        "Applied {} email change(s) and updated {} email map entr{}. Changed accounts are now unverified and must re-verify before logging in.",
        plan.changes.len(),
        plan.map_additions.len(),
        if plan.map_additions.len() == 1 {
            "y"
        } else {
            "ies"
        }
    );
    Ok(())
}

/// Print the duplicate groups that were found
///
/// # Arguments
///
/// * `groups` - The duplicate groups to print
fn print_groups(groups: &[core::DuplicateGroup]) {
    // count how many accounts are involved in a duplicate
    let affected: usize = groups.iter().map(|group| group.members.len()).sum();
    // print a summary header
    println!(
        "Found {} duplicate email(s) across {affected} accounts:",
        groups.len()
    );
    // print each duplicate group and its members
    for group in groups {
        // print the shared email and how many accounts use it
        println!("\n  {} ({} accounts):", group.email, group.members.len());
        // print each account that shares this email
        for member in &group.members {
            println!(
                "    - {} [role: {}, verified: {}]",
                member.username,
                core::role_label(&member.role),
                member.verified
            );
        }
    }
}

/// Interactively collect a new unique email for each duplicate account
///
/// Each group is resolved in turn and a group is re-prompted until all of its
/// members have unique, valid emails that do not collide with any other account.
/// An empty response keeps the accounts current email.
///
/// # Arguments
///
/// * `users` - The current set of users
/// * `groups` - The duplicate groups to resolve
/// * `system_email` - The email used by system accounts which is allowed to be duplicated
fn collect_changes(
    users: &[ScrubbedUser],
    groups: &[core::DuplicateGroup],
    system_email: &str,
) -> Result<Vec<core::EmailChange>, Error> {
    // normalize the exempt system email once for comparison
    let system = core::normalize_email(system_email);
    // build the set of usernames that are part of a duplicate group
    let in_group: HashSet<&str> = groups
        .iter()
        .flat_map(|group| group.members.iter())
        .map(|member| member.username.as_str())
        .collect();
    // seed the claimed email set with the emails of all unaffected accounts
    let mut claimed: HashSet<String> = users
        .iter()
        .filter(|user| !in_group.contains(user.username.as_str()))
        .map(|user| core::normalize_email(&user.email))
        .collect();
    // the email changes we collect across all groups
    let mut changes = Vec::new();
    // resolve each duplicate group interactively
    for group in groups {
        // print the group we are about to resolve
        println!("\nResolving duplicate email '{}':", group.email);
        // loop until this group's members all have unique, valid emails
        loop {
            // the tentative final email for each member of this group
            let mut group_finals: Vec<(&ScrubbedUser, String)> = Vec::new();
            // the normalized emails claimed within this group this round
            let mut group_norms: HashSet<String> = HashSet::new();
            // whether this round resolved the group without conflicts
            let mut ok = true;
            // prompt for a new email for each member of the group
            for member in &group.members {
                // prompt for the new email keeping the current one on empty input
                let input: String = dialoguer::Input::new()
                    .with_prompt(format!(
                        "  New email for '{}' (currently '{}', blank to keep)",
                        member.username, member.email
                    ))
                    .allow_empty(true)
                    .interact_text()?;
                // trim the input and treat an empty value as keeping the current email
                let trimmed = input.trim();
                let final_email = if trimmed.is_empty() {
                    member.email.clone()
                } else {
                    trimmed.to_string()
                };
                // normalize the final email for conflict checks
                let norm = core::normalize_email(&final_email);
                // reject assigning the reserved system email
                if norm == system {
                    println!(
                        "    '{system_email}' is reserved for system accounts, please choose another email"
                    );
                    ok = false;
                    break;
                }
                // reject an email already in use by an unaffected or resolved account
                if claimed.contains(&norm) {
                    println!(
                        "    '{final_email}' is already in use by another account, please choose another email"
                    );
                    ok = false;
                    break;
                }
                // reject an email already claimed by another member of this group
                if group_norms.contains(&norm) {
                    println!(
                        "    '{final_email}' was already used in this group, please choose another email"
                    );
                    ok = false;
                    break;
                }
                // record this members tentative final email
                group_norms.insert(norm);
                group_finals.push((member, final_email));
            }
            // retry the whole group if we hit a conflict
            if !ok {
                println!("  Let's try this group of accounts again.");
                continue;
            }
            // commit this group's resolutions now that they are conflict free
            for (member, final_email) in group_finals {
                // mark this email as claimed so later groups cannot reuse it
                claimed.insert(core::normalize_email(&final_email));
                // only record a change if the email actually changed
                if core::normalize_email(&final_email) != core::normalize_email(&member.email) {
                    changes.push(core::EmailChange {
                        username: member.username.clone(),
                        new_email: final_email,
                    });
                }
            }
            // this group is resolved so move on to the next one
            break;
        }
    }
    Ok(changes)
}

/// Print the changes that will be applied by a dedupe plan
///
/// # Arguments
///
/// * `plan` - The dedupe plan to print
fn print_plan(plan: &core::DedupePlan) {
    // print the email changes that will be applied
    println!("\nPlanned changes:");
    // note if there is nothing to change
    if plan.changes.is_empty() {
        println!("  (no email changes)");
    }
    // print each email change and that the account will be marked unverified
    for change in &plan.changes {
        println!(
            "  - {} -> {} (will be marked unverified)",
            change.username, change.new_email
        );
    }
    // report how many accounts will be added or corrected in the email map
    if !plan.map_additions.is_empty() {
        println!(
            "  {} user(s) will be added or corrected in the email map.",
            plan.map_additions.len()
        );
    }
    // report how many stale email map entries will be removed
    if !plan.map_removals.is_empty() {
        println!(
            "  {} stale email map entr{} will be removed.",
            plan.map_removals.len(),
            if plan.map_removals.len() == 1 {
                "y"
            } else {
                "ies"
            }
        );
    }
}

/// Apply a dedupe plan directly to Redis in a single atomic pipeline
///
/// # Arguments
///
/// * `config` - The Thorium cluster config providing the namespace and redis settings
/// * `plan` - The dedupe plan to apply
#[rustfmt::skip]
async fn apply_plan(config: &Conf, plan: &core::DedupePlan) -> Result<(), Error> {
    // the namespace these user keys live under
    let ns = &config.thorium.namespace;
    // build the email->username map key
    let email_map_key = core::email_map_key(ns);
    // get a redis connection pool
    let pool = crate::shared::redis::get_client(config).await?;
    // get a connection from the pool
    let mut conn = pool
        .get()
        .await
        .map_err(|err| Error::new(format!("Failed to get redis connection: {err}")))?;
    // build an atomic pipeline for all of our updates
    let mut pipe = redis::pipe();
    // remove any stale email->username map entries first
    for email in &plan.map_removals {
        pipe.cmd("hdel").arg(&email_map_key).arg(email);
    }
    // update each changed accounts email and mark it unverified
    for change in &plan.changes {
        // build the user data key for this account
        let data_key = core::user_data_key(ns, &change.username);
        // set the new email and clear verification (matches the api create/save format)
        pipe.cmd("hset").arg(&data_key).arg("email").arg(&change.new_email)
            .cmd("hset").arg(&data_key).arg("verified").arg(false);
    }
    // add or correct the email->username map so every non-system user is mapped
    for (email, username) in &plan.map_additions {
        pipe.cmd("hset").arg(&email_map_key).arg(email).arg(username);
    }
    // execute everything atomically
    let _: () = pipe.atomic().query_async(&mut *conn).await?;
    Ok(())
}

/// Read the current email->username map directly from Redis
///
/// # Arguments
///
/// * `config` - The Thorium cluster config providing the namespace and redis settings
async fn read_email_map(config: &Conf) -> Result<HashMap<String, String>, Error> {
    // the namespace this map lives under
    let ns = &config.thorium.namespace;
    // build the email->username map key
    let email_map_key = core::email_map_key(ns);
    // get a redis connection pool
    let pool = crate::shared::redis::get_client(config).await?;
    // get a connection from the pool
    let mut conn = pool
        .get()
        .await
        .map_err(|err| Error::new(format!("Failed to get redis connection: {err}")))?;
    // read the entire email->username map
    let map: HashMap<String, String> = redis::cmd("hgetall")
        .arg(&email_map_key)
        .query_async(&mut *conn)
        .await?;
    Ok(map)
}
