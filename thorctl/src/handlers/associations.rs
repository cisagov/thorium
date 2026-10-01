//! Handle association related Thorctl commands

use futures::stream::{self, StreamExt, TryStreamExt};
use owo_colors::OwoColorize;
use std::io::Write;
use thorium::models::{
    Association, AssociationDeleteRequest, AssociationListOpts, AssociationRequest,
    AssociationTarget, Cursor, Directionality,
};
use thorium::{Error, Thorium};

use crate::args::Args;
use crate::args::associations::{
    AssocObject, Associations, CreateAssociations, DeleteAssociations, ListAssociations,
};
use crate::handlers::update;
use crate::utils;

/// Make sure an object exists and convert it to an association target
///
/// Repos are resolved to their canonical url and entities are resolved to get their name.
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `object` - The object to resolve
async fn resolve(thorium: &Thorium, object: AssocObject) -> Result<AssociationTarget, Error> {
    // build a nicer error if this object can't be found
    let not_found = |error: Error| Error::new(format!("Failed to get {object}: {error}"));
    match &object {
        AssocObject::File(sha256) => {
            // make sure this file exists
            thorium.files.get(sha256).await.map_err(not_found)?;
            Ok(AssociationTarget::File(sha256.clone()))
        }
        AssocObject::Repo(url) => {
            // get this repo so we use its canonical url
            let repo = thorium.repos.get(url).await.map_err(not_found)?;
            Ok(AssociationTarget::Repo(repo.url))
        }
        AssocObject::Entity(id) => {
            // get this entity so we know its name
            let entity = thorium.entities.get(*id).await.map_err(not_found)?;
            Ok(AssociationTarget::Entity {
                id: entity.id,
                name: entity.name,
            })
        }
    }
}

/// Convert an object to an association target without checking that it exists
///
/// The name of entities is not used when deleting associations so it is left empty.
///
/// # Arguments
///
/// * `object` - The object to convert
fn to_target(object: AssocObject) -> AssociationTarget {
    match object {
        AssocObject::File(sha256) => AssociationTarget::File(sha256),
        AssocObject::Repo(url) => AssociationTarget::Repo(url),
        AssocObject::Entity(id) => AssociationTarget::Entity {
            id,
            name: String::new(),
        },
    }
}

/// Create associations in Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The create associations command to execute
async fn create(thorium: &Thorium, args: &Args, cmd: &CreateAssociations) -> Result<(), Error> {
    // resolve our source
    let source = resolve(thorium, cmd.source.object()).await?;
    // resolve all of our targets concurrently
    let targets = stream::iter(cmd.targets.objects())
        .map(|object| resolve(thorium, object))
        .buffered(args.workers.max(1))
        .try_collect::<Vec<AssociationTarget>>()
        .await?;
    // build our association request
    let mut req = AssociationRequest::with_capacity(cmd.kind, source, targets.len())
        .groups(cmd.groups.iter().cloned());
    for target in targets {
        req = req.target(target);
    }
    if cmd.bidirectional {
        req = req.biderectional();
    }
    // get the number of associations we are creating
    let cnt = req.targets.len();
    // create our associations
    thorium.associations.create(&req).await?;
    println!(
        "{} {cnt} {} association(s) from {}",
        "Created".bright_green(),
        cmd.kind,
        cmd.source.object()
    );
    Ok(())
}

/// Delete associations in Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The delete associations command to execute
async fn delete(thorium: &Thorium, cmd: &DeleteAssociations) -> Result<(), Error> {
    // get our source and targets
    let source = cmd.source.object();
    let targets = cmd.targets.objects();
    // confirm the deletion unless the user opted to skip the prompt
    if !cmd.skip_confirm {
        // make sure we can actually prompt the user
        utils::require_confirm_terminal("--skip-confirm (-y)")?;
        // show what we are about to delete
        println!(
            "Deleting {} associations from {}:",
            cmd.kind.bright_red(),
            source.bright_red()
        );
        for target in &targets {
            println!("  {}", target.bright_red());
        }
        // ask the user to confirm this destructive action
        let confirmed = dialoguer::Confirm::new()
            .with_prompt("Delete these associations?")
            .default(false)
            .interact()?;
        // abort if the user declined
        if !confirmed {
            println!("Aborted");
            return Ok(());
        }
    }
    // build our delete request
    let mut req = AssociationDeleteRequest::new(cmd.kind, to_target(source.clone()))
        .targets(targets.into_iter().map(to_target))
        .groups(cmd.groups.iter().cloned());
    for direction in &cmd.direction {
        req = req.direction(Directionality::from(*direction));
    }
    // delete these associations
    match thorium.associations.delete(&req).await {
        Ok(_) => {
            println!(
                "{} {} associations from {source}",
                "Deleted".bright_green(),
                cmd.kind
            );
            Ok(())
        }
        Err(error) => match error.status().map(|status| status.as_u16()) {
            Some(404) => Err(Error::new(format!(
                "No matching {} associations found from {source}: {error}",
                cmd.kind
            ))),
            Some(401) => Err(Error::new(format!(
                "{error}\nUse --groups to only delete associations from groups you manage"
            ))),
            _ => Err(error),
        },
    }
}

/// Get a printable name for the other side of an association
///
/// # Arguments
///
/// * `target` - The target to get a name for
fn target_name(target: &AssociationTarget) -> (&'static str, String) {
    match target {
        AssociationTarget::File(sha256) => ("File", sha256.clone()),
        AssociationTarget::Repo(url) => ("Repo", url.clone()),
        AssociationTarget::Entity { id, name } => ("Entity", format!("{name} ({id})")),
    }
}

/// A single line in an association list table
struct AssociationLine;

impl AssociationLine {
    /// Print the header for an association table
    fn header() {
        println!(
            "{:<20} | {:<13} | {:<6} | {:<72} | {:<20} | {:<30} | CREATED",
            "KIND", "DIRECTION", "TYPE", "OTHER", "SUBMITTER", "GROUPS"
        );
        println!(
            "{:-<21}+{:-<15}+{:-<8}+{:-<74}+{:-<22}+{:-<32}+{:-<26}",
            "", "", "", "", "", "", ""
        );
    }

    /// Print a single association
    ///
    /// # Arguments
    ///
    /// * `assoc` - The association to print
    fn print(assoc: &Association) {
        // get the type and name of the other side of this association
        let (kind, name) = target_name(&assoc.other);
        println!(
            "{:<20} | {:<13} | {:<6} | {:<72} | {:<20} | {:<30} | {}",
            assoc.kind.as_str(),
            assoc.direction.as_str(),
            kind,
            name,
            assoc.submitter,
            assoc.groups.join(","),
            assoc.created.to_rfc3339()
        );
    }
}

/// List the associations for some data in Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The list associations command to execute
async fn list(thorium: &Thorium, cmd: &ListAssociations) -> Result<(), Error> {
    // build our list options
    let mut opts = AssociationListOpts::default()
        .groups(cmd.groups.clone())
        .page_size(cmd.page_size);
    if !cmd.no_limit {
        opts = opts.limit(cmd.limit);
    }
    // list associations for the right kind of object
    let mut cursor: Cursor<Association> = match cmd.source.object() {
        AssocObject::File(sha256) => thorium.associations.list_file(&sha256, &opts).await?,
        AssocObject::Repo(url) => thorium.associations.list_repo(&url, &opts).await?,
        AssocObject::Entity(id) => thorium.associations.list_entity(id, &opts).await?,
    };
    // collect all of our associations
    let mut assocs = Vec::default();
    loop {
        // filter this page down to the kinds we want
        assocs.extend(
            cursor
                .data
                .drain(..)
                .filter(|assoc| cmd.kind.is_empty() || cmd.kind.contains(&assoc.kind)),
        );
        // check if this cursor is exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of associations
        cursor.refill().await?;
    }
    // print our associations as JSON or a table
    match (&cmd.output, cmd.condensed, cmd.json) {
        (None, false, false) => {
            AssociationLine::header();
            for assoc in &assocs {
                AssociationLine::print(assoc);
            }
        }
        (None, true, _) => {
            serde_json::to_writer(std::io::stdout(), &assocs)?;
            println!();
        }
        (None, false, true) => {
            serde_json::to_writer_pretty(std::io::stdout(), &assocs)?;
            println!();
        }
        (Some(output_path), true, _) => {
            let mut output_file = std::fs::File::create(output_path)?;
            serde_json::to_writer(&mut output_file, &assocs)?;
            writeln!(&mut output_file)?;
        }
        (Some(output_path), false, _) => {
            let mut output_file = std::fs::File::create(output_path)?;
            serde_json::to_writer_pretty(&mut output_file, &assocs)?;
            writeln!(&mut output_file)?;
        }
    }
    Ok(())
}

/// Handle all association commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The associations command to execute
pub async fn handle(args: &Args, cmd: &Associations) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        update::ask_update(&thorium).await?;
    }
    // call the right associations handler
    match cmd {
        Associations::Create(cmd) => create(&thorium, args, cmd).await,
        Associations::Delete(cmd) => delete(&thorium, cmd).await,
        Associations::List(cmd) => list(&thorium, cmd).await,
    }
}
