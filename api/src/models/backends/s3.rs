//! Handle operations for s3 id mapping in Scylla

use uuid::Uuid;

use super::db;
use crate::models::S3Objects;
use crate::utils::{ApiError, Shared};

/// Generate a unique id for this s3 object
pub async fn generate_id(objects: S3Objects, shared: &Shared) -> Result<Uuid, ApiError> {
    db::s3::generate_id(objects, shared).await
}
