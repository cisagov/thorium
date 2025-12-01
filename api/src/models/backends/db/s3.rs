use tracing::instrument;
/// Maps s3 data to sha256s using scylla
use uuid::Uuid;

use crate::models::S3Objects;
use crate::utils::{ApiError, Shared};
use crate::{not_found, unavailable};

/// Check if an s3 object id already exists in scylla
///
/// # Arguments
///
/// * `s3_id` - The s3 id to check for
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::s3::s3_id_exists", skip(s3_id, shared), err(Debug))]
pub async fn s3_id_exists(
    objects: S3Objects,
    s3_id: &Uuid,
    shared: &Shared,
) -> Result<bool, ApiError> {
    // execute our query
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.s3.id_exists, (objects, &s3_id))
        .await?;
    // cast this query to a rows query
    let query_rows = query.into_rows_result()?;
    // make sure we got at least one row from scylla
    Ok(query_rows.rows_num() > 0)
}

/// Checks if we have an existing s3 object for a target sha256/path already
///
/// # Arguments
///
/// * `object` - The type of objects to perform an existence check
/// * `path` - The path to check against
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::s3::object_exists", skip(shared), err(Debug))]
pub async fn object_exists(
    object: S3Objects,
    path: &str,
    shared: &Shared,
) -> Result<bool, ApiError> {
    // execute our query
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.s3.object_exists, (object, path))
        .await?;
    // cast this query to a rows query
    let query_rows = query.into_rows_result()?;
    // make sure we got at least one row from scylla
    Ok(query_rows.rows_num() > 0)
}

/// Insert a new s3 object id
///
/// # Arguments
///
/// * `s3_id` - The s3 id to insert
/// * `sha256` - The sha256 of the file this s3 id contains
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::s3::insert_s3_id", skip(s3_id, shared), err(Debug))]
pub async fn insert_s3_id(
    objects: S3Objects,
    s3_id: &Uuid,
    sha256: &str,
    shared: &Shared,
) -> Result<(), ApiError> {
    shared
        .scylla
        .session
        .execute_unpaged(
            &shared.scylla.prep.s3.insert,
            (objects.as_str(), &s3_id, sha256),
        )
        .await?;
    Ok(())
}

/// Generate a new uuid that is not yet in use
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::s3::generate_id", skip(shared), err(Debug))]
pub async fn generate_id(objects: S3Objects, shared: &Shared) -> Result<Uuid, ApiError> {
    // try 10 times to generate an unused uuid before failing
    for _ in 0..10 {
        // generate a random uuid
        let id = Uuid::new_v4();
        if !s3_id_exists(objects, &id, shared).await? {
            return Ok(id);
        }
    }
    unavailable!("Unable to generate a unique sample id".to_owned())
}

/// Gets an s3 object id for a specific sha256
///
/// # Arguments
///
/// * `s3_id` - The s3 id to check for
/// * `key` - The key of the s3 id to get
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::s3::get_s3_id", skip(shared), err(Debug))]
pub async fn get_s3_id(objects: S3Objects, key: &str, shared: &Shared) -> Result<Uuid, ApiError> {
    // execute our query
    let query = shared
        .scylla
        .session
        .execute_unpaged(&shared.scylla.prep.s3.get, (objects, key))
        .await?;
    // cast this query to a rows query
    let query_rows = query.into_rows_result()?;
    // try to get the first row
    if let Some((s3_id,)) = query_rows.maybe_first_row::<(Uuid,)>()? {
        Ok(s3_id)
    } else {
        // we didn't get any rows so this id does not exist
        not_found!(format!(
            "Failed to find an s3 object for {}:{}",
            objects, key
        ))
    }
}
