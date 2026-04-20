//! Resource categorization for imports
//!
//! Checks which incoming images and pipelines already exist in Thorium,
//! categorizing them for downstream handling (create vs update). The inputs
//! are plain request lists so every import flow (toolbox manifests, on-disk
//! export directories) can share this step.

use futures::{StreamExt, TryStreamExt, stream};
use http::StatusCode;
use std::path::Path;
use thorium::models::{ImageRequest, PipelineRequest};
use thorium::{Error, Thorium};

use super::kind::{ImageKind, ImportKind, PipelineKind};
use crate::handlers::progress::{Bar, BarKind};

/// Load a request from `<import_dir>/<noun>s/<name>.json` and point it at `group`
///
/// Shared by the `images import` and `pipelines import` on-disk loaders.
///
/// # Arguments
///
/// * `import_dir` - The export directory root
/// * `group` - The group the import targets
/// * `name` - The resource name whose `<name>.json` config to read
pub async fn load_request<K: ImportKind>(
    import_dir: &Path,
    group: &str,
    name: &str,
) -> Result<K::Request, Error> {
    let file_path = import_dir
        .join(format!("{}s", K::NOUN))
        .join(format!("{name}.json"));
    let data = tokio::fs::read_to_string(&file_path).await.map_err(|err| {
        Error::new(format!(
            "Failed to read {} data at {file_path:?}: {err}",
            K::NOUN
        ))
    })?;
    let mut req: K::Request = serde_json::from_str(&data).map_err(|err| {
        Error::new(format!("Failed to parse {} data for {name}: {err}", K::NOUN))
    })?;
    // imports always land in the group given on the command line
    K::set_group(&mut req, group);
    Ok(req)
}

/// An incoming resource categorized by whether it already exists in Thorium
pub struct Categorized<K: ImportKind> {
    /// The display name for this resource (the manifest name for toolboxes)
    pub name: String,
    /// The display version for this resource ("latest" when versionless)
    pub version: String,
    /// The incoming request
    pub request: K::Request,
    /// The existing resource in Thorium, if any
    pub existing: Option<K::Existing>,
}

/// An incoming image categorized by whether it already exists in Thorium
pub type CategorizedImage = Categorized<ImageKind>;
/// An incoming pipeline categorized by whether it already exists in Thorium
pub type CategorizedPipeline = Categorized<PipelineKind>;

/// Categorize incoming resources by checking which ones already exist in Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client used to look up existing resources
/// * `items` - The incoming resources as (name, version, request) tuples
/// * `progress` - The progress bar to increment as each resource is checked
pub async fn categorize<K: ImportKind>(
    thorium: &Thorium,
    items: Vec<(String, String, K::Request)>,
    progress: &Bar,
) -> Result<Vec<Categorized<K>>, Error> {
    progress.refresh(
        format!("Checking existing {}s", K::NOUN),
        BarKind::Bound(items.len() as u64),
    );
    stream::iter(items)
        .map(|(name, version, request)| async move {
            let existing = match K::get(thorium, K::group(&request), K::name(&request)).await {
                Ok(resource) => Some(resource),
                // a not-found resource is simply new, not an error
                Err(err)
                    if err
                        .status()
                        .is_some_and(|status| status == StatusCode::NOT_FOUND) =>
                {
                    None
                }
                Err(err) => {
                    return Err(Error::new(format!(
                        "Error checking {} '{name}:{version}': {err}",
                        K::NOUN
                    )));
                }
            };
            progress.inc(1);
            Ok(Categorized {
                name,
                version,
                request,
                existing,
            })
        })
        // bounded concurrency; try_collect short-circuits on the first error
        // without materializing an intermediate Vec of Results. Bounded (not wired to
        // --workers) on purpose: this read-only existence check is shared by import,
        // diff, and remove, and --workers is meant for the apply (write) phase.
        .buffer_unordered(10)
        .try_collect()
        .await
}

/// Categorize incoming images by checking which ones already exist in Thorium
pub async fn categorize_images(
    thorium: &Thorium,
    items: Vec<(String, String, ImageRequest)>,
    progress: &Bar,
) -> Result<Vec<CategorizedImage>, Error> {
    categorize::<ImageKind>(thorium, items, progress).await
}

/// Categorize incoming pipelines by checking which ones already exist in Thorium
pub async fn categorize_pipelines(
    thorium: &Thorium,
    items: Vec<(String, String, PipelineRequest)>,
    progress: &Bar,
) -> Result<Vec<CategorizedPipeline>, Error> {
    categorize::<PipelineKind>(thorium, items, progress).await
}
