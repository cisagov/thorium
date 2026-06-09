//! The read-only duplicate email scan command for Thoradm

use thorium::Thorium;

use super::core;
use crate::args::{Args, ScanEmails};
use crate::Error;

/// Scan Thorium for non-system accounts that share an email address
///
/// This lists all users through the normal Thorium API and reports any groups of
/// accounts that share an email. It is read-only and makes no changes.
///
/// # Arguments
///
/// * `opts` - The scan command options
/// * `args` - The Thoradm args
pub async fn scan(opts: &ScanEmails, args: &Args) -> Result<(), Error> {
    // build a Thorium client from the thorctl config
    let thorium = Thorium::from_ctl_conf_file(&args.ctl_conf).await?;
    // list all users with details via the normal API
    let users = thorium.users.list_details().await?;
    // find groups of accounts that share an email
    let groups = core::find_duplicate_groups(&users, &opts.system_email);
    // report that everything is unique if we found no duplicates
    if groups.is_empty() {
        println!("No duplicate emails found ({} accounts scanned).", users.len());
        return Ok(());
    }
    // count how many accounts are involved in a duplicate
    let affected: usize = groups.iter().map(|group| group.members.len()).sum();
    // print a summary header
    println!(
        "Found {} duplicate email(s) across {affected} accounts:",
        groups.len()
    );
    // print each duplicate group and its members
    for group in &groups {
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
    // point the admin at the dedupe command to resolve the duplicates
    println!("\nRun `thoradm users dedupe` to resolve these duplicates.");
    Ok(())
}
