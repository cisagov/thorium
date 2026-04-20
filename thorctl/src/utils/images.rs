//! Utility functions relating to images

use std::io::Write;
use std::sync::{Arc, Mutex};

use futures::{TryStreamExt, stream};
use owo_colors::OwoColorize;
use thorium::{
    Cursor, Error, Thorium,
    models::{Image, ImageBanKind, ImageVersion, SpawnLimits},
};

use super::render::{field, header, label, render_markdown};

/// Write a single line to the output, mapping any IO error
///
/// # Arguments
///
/// * `out` - The writer to write to
/// * `line` - The line to write
fn write_line(out: &mut dyn Write, line: &str) -> Result<(), Error> {
    writeln!(out, "{line}").map_err(|err| Error::new(format!("Error writing output: {err}")))
}

/// Print an image's details in a human-readable format
///
/// # Arguments
///
/// * `image` - The image to print
/// * `out` - The writer to print to
/// * `ansi` - Whether to style the output with ANSI escape codes
pub fn print_image_details(image: &Image, out: &mut dyn Write, ansi: bool) -> Result<(), Error> {
    // header and the always-shown scalar fields
    write_line(out, &header(&image.name, &image.group, ansi))?;
    write_line(out, &field("Creator", &image.creator, ansi))?;
    if let Some(version) = &image.version {
        let version = match version {
            ImageVersion::SemVer(version) => version.to_string(),
            ImageVersion::Custom(version) => version.clone(),
        };
        write_line(out, &field("Version", &version, ansi))?;
    }
    write_line(out, &field("Scaler", image.scaler.as_str(), ansi))?;
    if let Some(url) = &image.image {
        write_line(out, &field("Image", url, ansi))?;
    }
    write_line(out, &field("Generator", &image.generator.to_string(), ansi))?;
    if let Some(timeout) = image.timeout {
        write_line(out, &field("Timeout", &format!("{timeout}s"), ansi))?;
    }
    write_line(out, &field("Avg runtime", &format!("{}s", image.runtime), ansi))?;
    let resources = format!(
        "{} mCPU, {} MiB memory, {} MiB storage",
        image.resources.cpu, image.resources.memory, image.resources.ephemeral_storage
    );
    write_line(out, &field("Resources", &resources, ansi))?;
    let spawn_limit = match &image.spawn_limit {
        SpawnLimits::Basic(limit) => limit.to_string(),
        SpawnLimits::Unlimited => "unlimited".to_string(),
    };
    write_line(out, &field("Spawn limit", &spawn_limit, ansi))?;
    if !image.used_by.is_empty() {
        write_line(out, &field("Used by", &image.used_by.join(", "), ansi))?;
    }
    // bans are important, so call them out in red
    if !image.bans.is_empty() {
        write_line(out, &field("Bans", &image.bans.len().to_string(), ansi))?;
        for ban in image.bans.values() {
            // build a message from the ban kind (the `Ban` trait is api-private)
            let message = match &ban.ban_kind {
                ImageBanKind::Generic(ban) => ban.msg.clone(),
                ImageBanKind::InvalidImageUrl(ban) => {
                    format!("invalid or unreachable image URL '{}'", ban.url)
                }
                ImageBanKind::InvalidHostPath(ban) => format!(
                    "host path '{}' for volume '{}' is not allowed",
                    ban.host_path.to_string_lossy(),
                    ban.volume_name
                ),
            };
            let line = format!("  - {message}");
            write_line(out, &if ansi { line.bright_red().to_string() } else { line })?;
        }
    }
    if !image.network_policies.is_empty() {
        let policies = image.network_policies.iter().cloned().collect::<Vec<_>>().join(", ");
        write_line(out, &field("Network policies", &policies, ansi))?;
    }
    // finally render the full description markdown
    if let Some(description) = &image.description {
        write_line(out, "")?;
        write_line(out, &label("Description", ansi))?;
        write_line(out, &render_markdown(description, ansi))?;
    }
    Ok(())
}

/// List every image in a group with full details, draining the list cursor
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `group` - The group to list images from
pub async fn list_all_images(thorium: &Thorium, group: &str) -> Result<Vec<Image>, Error> {
    let mut images = Vec::new();
    // a single group can't realistically exceed this limit; the cursor still
    // pages underneath it
    let mut cursor = thorium.images.list(group).limit(super::LIST_ALL_LIMIT).details();
    loop {
        cursor
            .next()
            .await
            .map_err(|e| Error::new(format!("Failed to list images in group '{group}': {e}")))?;
        images.append(&mut cursor.details);
        if cursor.exhausted {
            break;
        }
    }
    Ok(images)
}

/// Search an image cursor for a given image
///
/// # Arguments
///
/// * `cursor` - The image cursor to search
/// * `group` - The group the image cursor is crawling
/// * `image_name` - The name of the image we are searching for
/// * `matching_groups` - A list of groups containing the matching image
async fn search_image_cursor(
    mut cursor: Cursor<Image>,
    group: String,
    image_name: &str,
    matching_groups: Arc<Mutex<Vec<String>>>,
) -> Result<(), Error> {
    while !cursor.exhausted {
        cursor.next().await?;
        if cursor.names.iter().any(|name| name == image_name) {
            // add the matching group to the list; recover from a poisoned lock
            // (a sibling task panicking shouldn't abort the whole search)
            matching_groups
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(group.clone());
            // stop searching because the image can only appear once within a group
            return Ok(());
        }
    }
    Ok(())
}

/// Find the group that a given image belongs to among the current user's groups
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `image_name` - The name of the image
pub async fn find_image_group(thorium: &Thorium, image_name: &str) -> Result<String, Error> {
    // get all groups for the current user
    let groups = super::groups::get_all_groups(thorium).await?;
    // create a list to contain groups that have a image of the given name
    let matching_groups: Vec<String> = Vec::new();
    // wrap in an Arc<Mutex<>> to add to the list concurrently
    let matching_groups = Arc::new(Mutex::new(matching_groups));
    // create image cursors for each group
    stream::iter(
        groups
            .into_iter()
            .map(|group| Ok((thorium.images.list(&group).limit(super::LIST_ALL_LIMIT), group))),
    )
    // concurrently search for the image in each group and add matching groups to the list
    .try_for_each_concurrent(None, |(cursor, group)| {
        search_image_cursor(cursor, group, image_name, matching_groups.clone())
    })
    .await?;
    // unwrap the matching groups from the Arc and Mutex
    let matching_groups = Arc::into_inner(matching_groups)
        .ok_or(Error::new("Concurrency error retrieving image"))?
        .into_inner()
        .map_err(|_| Error::new("Poison mutex error retrieving image"))?;
    // ensure that only a single matching group was found
    match matching_groups.len() {
        len if len < 1 => Err(Error::new("Image not found")),
        len if len > 1 => Err(Error::new(format!(
            "Images with the given name exist in more than one group: {matching_groups:?}. Please specify a group"
        ))),
        _ => matching_groups
            .into_iter()
            .next()
            .ok_or(Error::new("Unable to retrieve image")),
    }
}
