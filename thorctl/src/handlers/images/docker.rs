//! Shared helpers for driving the `docker` CLI during image import/export

use std::process::Output;
use thorium::Error;
use thorium::models::ImageRequest;

use crate::handlers::progress::Bar;

/// Ensure a `docker` command succeeded, surfacing its stderr if it didn't
///
/// `docker` reports failures via a non-zero exit code with the detail on
/// stderr, so we log those lines to the user before erroring out; otherwise the
/// failure would be invisible behind the progress bar.
pub fn check_command(bar: &Bar, output: &Output, msg: &str) -> Result<(), Error> {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        for line in stderr.lines().filter(|line| !line.is_empty()) {
            bar.error(line);
        }
        return Err(Error::new(msg));
    }
    Ok(())
}

/// Build a new image URL pointing at an override registry, if one is set
///
/// Only the registry domain is swapped; the original image path is preserved.
/// Returns `None` when there's no override or the image has no URL to rewrite,
/// so callers can use the `Some` case as proof that `image.image` is set.
pub fn override_registry(image: &ImageRequest, registry_override: Option<&str>) -> Option<String> {
    match (registry_override, &image.image) {
        (Some(registry), Some(old_url)) => {
            // keep everything after the first '/' (the image path), replacing
            // only the leading registry domain
            let url_path = old_url
                .split_once('/')
                .map_or(old_url.as_str(), |(_, path)| path);
            Some(format!("{registry}/{url_path}"))
        }
        _ => None,
    }
}
