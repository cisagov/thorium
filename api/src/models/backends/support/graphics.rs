//! Support for graphics tied to objects in Thorium

use aws_sdk_s3::operation::get_object::GetObjectOutput;
use axum::extract::multipart::Field;
use mime::Mime;
use std::path::PathBuf;

use crate::utils::s3::S3Client;
use crate::utils::{ApiError, Shared};
use crate::{bad, internal_err_unwrapped};

/// Support for graphics for objects in Thorium
pub(crate) trait GraphicSupport {
    /// A unique, immutable key to use to reference the implementing object
    type GraphicKey<'a>: std::fmt::Debug + Send;

    /// Build the base path for this graphic
    fn build_graphic_base_path<'a>(key: Self::GraphicKey<'a>) -> PathBuf;

    /// Build the base path for this graphic
    fn build_graphic_base_path_from_self(&self) -> PathBuf;

    /// Upload a graphic associated with the given key
    ///
    /// # Arguments
    ///
    /// * `s3_path` - The base S3 path (the object's graphic key) to upload this graphic under
    /// * `field` - The multipart form field containing the graphic data
    /// * `name_override` - The name to override this fields original name with without the extension
    /// * `shared` - Shared Thorium objects
    async fn upload_graphic(
        mut s3_path: PathBuf,
        field: Field<'_>,
        name_override: Option<String>,
        shared: &Shared,
    ) -> Result<String, ApiError> {
        // add the name for this path
        match (&name_override, field.file_name()) {
            (Some(name_override), _) => s3_path.push(name_override),
            (None, Some(filename)) => s3_path.push(filename),
            // just use a random uuid
            (None, None) => return bad!("Graphics must have a name!".to_owned()),
        }
        // make sure our extension is correct
        let content_type = match field.content_type() {
            Some(unparsed) => {
                // parse the field's declared content type once; a malformed value is a client error (400),
                // never a panic — any authenticated user can hit this route
                let ctype: Mime = match unparsed.parse() {
                    Ok(ctype) => ctype,
                    Err(_) => return bad!(format!("Invalid content type: {unparsed}")),
                };
                // set our extension from the parsed content type; the explicit allowlist is the only
                // gate (animated GIF and short video clips are valid graphics alongside static images),
                // so anything not matched here is rejected as an unsupported type
                match ctype.essence_str() {
                    "image/bmp" => s3_path.set_extension("bmp"),
                    "image/gif" => s3_path.set_extension("gif"),
                    "image/jpeg" => s3_path.set_extension("jpeg"),
                    "image/png" => s3_path.set_extension("png"),
                    "image/svg+xml" => s3_path.set_extension("svg"),
                    "video/mp4" => s3_path.set_extension("mp4"),
                    "video/webm" => s3_path.set_extension("webm"),
                    // don't allow arbitrary content types
                    _ => {
                        return bad!(
                            "Only BMP, GIF, JPEG, PNG, SVG, MP4, and WebM graphics are supported"
                                .to_owned()
                        );
                    }
                };
                // return our content type as a string
                unparsed.to_owned()
            }
            None => return bad!("A content type must be set!".to_owned()),
        };
        // convert our path a string and return an error if its not castable
        let s3_path_str = match s3_path.into_os_string().into_string() {
            Ok(s3_path_str) => s3_path_str,
            Err(_) => return bad!("Graphics name must be valid utf-8".to_owned()),
        };
        // upload the graphic to S3
        shared
            .s3
            .graphics
            .stream_with_content_type(&s3_path_str, field, &content_type)
            .await
            .map_err(|err| {
                internal_err_unwrapped!(format!("Error streaming image to S3: {err}"))
            })?;
        // return the path the graphic was uploaded to
        Ok(s3_path_str)
    }

    /// Download the graphic associated with the given key
    ///
    /// Returns the object from S3 and the path it was retrieved from
    ///
    /// # Arguments
    ///
    /// * `s3_path` - The path to the graphic to download in s3
    /// * `shared` - Shared Thorium objects
    async fn download_graphic(
        &self,
        s3_path: &str,
        shared: &Shared,
    ) -> Result<GetObjectOutput, ApiError> {
        // download from S3 with the path
        shared.s3.graphics.download_with_metadata(s3_path).await
    }

    /// Delete the graphic at the given S3 key
    ///
    /// # Arguments
    ///
    /// * `key` - The S3 key of the graphic to delete
    /// * `shared` - Shared Thorium objects
    async fn delete_graphic(key: &str, shared: &Shared) -> Result<(), ApiError> {
        // delete our object
        shared.s3.graphics.delete(key).await
    }
}
