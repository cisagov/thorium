//! Generic abstraction over the image and pipeline resource kinds
//!
//! The image and pipeline import paths are mirror images of each other. This
//! trait captures the handful of operations that actually differ between them —
//! the client calls, the update calculation, and the journal hooks — so the
//! categorize, create, force-update, interactive-merge, and warn-skipped passes
//! can each be written once and instantiated for both kinds rather than copied.

use thorium::models::{
    Image, ImageRequest, ImageUpdate, Pipeline, PipelineRequest, PipelineUpdate,
};
use thorium::{CtlConf, Error, Thorium};

use super::rollback::Journal;
use super::{merge, update};

/// A resource kind (image or pipeline) the import engine can operate on
///
/// Implementors are zero-sized marker types ([`ImageKind`]/[`PipelineKind`]); the
/// trait is the seam every generic import pass dispatches through.
pub trait ImportKind {
    /// The incoming request type (`ImageRequest`/`PipelineRequest`)
    ///
    /// `DeserializeOwned` so requests can be loaded from on-disk export configs.
    type Request: Clone + serde::de::DeserializeOwned;
    /// The existing resource as stored in Thorium (`Image`/`Pipeline`)
    type Existing: Clone;
    /// The update payload applied to an existing resource
    ///
    /// `Serialize` so a skipped resource's changed fields can be rendered for the
    /// `--skip-conflicts` warning (see `summary::render_changed_fields`).
    type Update: serde::Serialize;
    /// The lower-case noun used in user-facing messages ("image"/"pipeline")
    const NOUN: &'static str;
    /// The capitalized title used in user-facing messages ("Image"/"Pipeline")
    const TITLE: &'static str;

    /// The group a request targets
    fn group(req: &Self::Request) -> &str;
    /// The name a request targets
    fn name(req: &Self::Request) -> &str;
    /// Point a request at a target group (imports always land in the CLI group)
    fn set_group(req: &mut Self::Request, group: &str);

    /// Fetch the existing resource from Thorium
    async fn get(thorium: &Thorium, group: &str, name: &str) -> Result<Self::Existing, Error>;
    /// Create a new resource in Thorium
    async fn create(thorium: &Thorium, req: &Self::Request) -> Result<(), Error>;
    /// Apply an update to an existing resource in Thorium
    async fn update(
        thorium: &Thorium,
        group: &str,
        name: &str,
        update: &Self::Update,
    ) -> Result<(), Error>;

    /// Compute the update needed to bring `existing` in line with `req`, or
    /// `None` when nothing changed
    fn calculate_update(existing: Self::Existing, req: Self::Request) -> Option<Self::Update>;
    /// Resolve a merge conflict interactively via the editor, returning the
    /// resulting update (or `None` if the edit was a no-op or was cancelled)
    async fn merge_interactive(
        existing: &Self::Existing,
        req: &Self::Request,
        conf: &CtlConf,
        editor_override: Option<&str>,
    ) -> Result<Option<Self::Update>, Error>;
    /// Whether the incoming request differs from the existing resource
    fn changed(existing: &Self::Existing, req: &Self::Request) -> bool;

    /// Record a creation in the rollback journal
    fn record_created(journal: &Journal, group: &str, name: &str);
    /// Snapshot the pre-update state in the rollback journal
    fn record_updated(journal: &Journal, existing: Self::Existing);
}

/// The image resource kind
pub struct ImageKind;
/// The pipeline resource kind
pub struct PipelineKind;

impl ImportKind for ImageKind {
    type Request = ImageRequest;
    type Existing = Image;
    type Update = ImageUpdate;
    const NOUN: &'static str = "image";
    const TITLE: &'static str = "Image";

    /// The group an image request targets
    fn group(req: &ImageRequest) -> &str {
        &req.group
    }
    /// The name an image request targets
    fn name(req: &ImageRequest) -> &str {
        &req.name
    }
    /// Point an image request at a target group
    fn set_group(req: &mut ImageRequest, group: &str) {
        req.group = group.to_string();
    }
    /// Fetch the existing image from Thorium
    async fn get(thorium: &Thorium, group: &str, name: &str) -> Result<Image, Error> {
        thorium.images.get(group, name).await
    }
    /// Create a new image in Thorium
    async fn create(thorium: &Thorium, req: &ImageRequest) -> Result<(), Error> {
        thorium.images.create(req).await.map(|_| ())
    }
    /// Apply an update to an existing image in Thorium
    async fn update(
        thorium: &Thorium,
        group: &str,
        name: &str,
        update: &ImageUpdate,
    ) -> Result<(), Error> {
        thorium.images.update(group, name, update).await.map(|_| ())
    }
    /// Compute the update needed to bring an existing image in line with a request
    fn calculate_update(existing: Image, req: ImageRequest) -> Option<ImageUpdate> {
        update::calculate_image_update(existing, req)
    }
    /// Resolve an image merge conflict interactively via the editor
    async fn merge_interactive(
        existing: &Image,
        req: &ImageRequest,
        conf: &CtlConf,
        editor_override: Option<&str>,
    ) -> Result<Option<ImageUpdate>, Error> {
        merge::merge_image_interactive(existing, req, conf, editor_override).await
    }
    /// Whether the incoming image request differs from the existing image
    fn changed(existing: &Image, req: &ImageRequest) -> bool {
        existing != req
    }
    /// Record an image creation in the rollback journal
    fn record_created(journal: &Journal, group: &str, name: &str) {
        journal.created_image(group, name);
    }
    /// Snapshot the pre-update image state in the rollback journal
    fn record_updated(journal: &Journal, existing: Image) {
        journal.updated_image(existing);
    }
}

impl ImportKind for PipelineKind {
    type Request = PipelineRequest;
    type Existing = Pipeline;
    type Update = PipelineUpdate;
    const NOUN: &'static str = "pipeline";
    const TITLE: &'static str = "Pipeline";

    /// The group a pipeline request targets
    fn group(req: &PipelineRequest) -> &str {
        &req.group
    }
    /// The name a pipeline request targets
    fn name(req: &PipelineRequest) -> &str {
        &req.name
    }
    /// Point a pipeline request at a target group
    fn set_group(req: &mut PipelineRequest, group: &str) {
        req.group = group.to_string();
    }
    /// Fetch the existing pipeline from Thorium
    async fn get(thorium: &Thorium, group: &str, name: &str) -> Result<Pipeline, Error> {
        thorium.pipelines.get(group, name).await
    }
    /// Create a new pipeline in Thorium
    async fn create(thorium: &Thorium, req: &PipelineRequest) -> Result<(), Error> {
        thorium.pipelines.create(req).await.map(|_| ())
    }
    /// Apply an update to an existing pipeline in Thorium
    async fn update(
        thorium: &Thorium,
        group: &str,
        name: &str,
        update: &PipelineUpdate,
    ) -> Result<(), Error> {
        thorium.pipelines.update(group, name, update).await.map(|_| ())
    }
    /// Compute the update needed to bring an existing pipeline in line with a request
    fn calculate_update(existing: Pipeline, req: PipelineRequest) -> Option<PipelineUpdate> {
        update::calculate_pipeline_update(existing, req)
    }
    /// Resolve a pipeline merge conflict interactively via the editor
    async fn merge_interactive(
        existing: &Pipeline,
        req: &PipelineRequest,
        conf: &CtlConf,
        editor_override: Option<&str>,
    ) -> Result<Option<PipelineUpdate>, Error> {
        merge::merge_pipeline_interactive(existing, req, conf, editor_override).await
    }
    /// Whether the incoming pipeline request differs from the existing pipeline
    fn changed(existing: &Pipeline, req: &PipelineRequest) -> bool {
        existing != req
    }
    /// Record a pipeline creation in the rollback journal
    fn record_created(journal: &Journal, group: &str, name: &str) {
        journal.created_pipeline(group, name);
    }
    /// Snapshot the pre-update pipeline state in the rollback journal
    fn record_updated(journal: &Journal, existing: Pipeline) {
        journal.updated_pipeline(existing);
    }
}
