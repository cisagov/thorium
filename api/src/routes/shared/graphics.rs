//! Contains generic functions for interacting with graphics from a route

use aws_sdk_s3::operation::get_object::GetObjectOutput;
use axum::http::{HeaderMap, HeaderValue, header};
use tracing::{Level, event, instrument};

/// Function to sanitize a UTF-8 string into an ASCII-only string
fn sanitize_to_ascii(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii() {
                c // Keep ASCII characters as-is
            } else {
                match c {
                    'ü' => 'u',
                    'ñ' => 'n',
                    'î' => 'i',
                    'ç' => 'c',
                    'ø' => 'o',
                    _ => '_', // Replace unsupported characters with '_'
                }
            }
        })
        .collect()
}

/// Set headers according to the data in the output object
///
/// # Arguments
///
/// * `output` - The get object output from S3
/// * `path` - The path the object was retrieved from
#[instrument(name = "routes::shared::graphics::get_headers", skip(output))]
pub fn get_headers(output: &GetObjectOutput, path: &str) -> HeaderMap {
    // create headers for the response
    let mut headers = HeaderMap::new();
    // set the Content-Type header if we have one
    if let Some(content_type) = &output.content_type {
        match HeaderValue::from_str(content_type) {
            Ok(content_type) => {
                headers.insert(header::CONTENT_TYPE, content_type);
            }
            Err(err) => {
                event!(
                    Level::WARN,
                    "Invalid entity image content type '{content_type}': {err}",
                );
            }
        }
    }
    // get the filename from our path
    if let Some(file_name) = std::path::Path::new(path).file_name() {
        // convert our file_name to a utf-8 string
        let file_name_utf8 = file_name.to_string_lossy();
        let file_name_str = file_name_utf8.to_string();
        // set the CONTENT_DISPOSITION header to "inline" with a file name if we have one
        // encode the UTF-8 file name
        let encoded = percent_encoding::utf8_percent_encode(
            &file_name_str,
            percent_encoding::NON_ALPHANUMERIC,
        );
        // provide an ASCII version for compatibility
        let sanitized = sanitize_to_ascii(&file_name_str);
        // attempt to create a valid header value with the filename
        if let Ok(content_disposition_header) = HeaderValue::from_str(&format!(
            "inline; filename*=UTF-8''{encoded}; filename=\"{sanitized}\""
        )) {
            headers.insert(header::CONTENT_DISPOSITION, content_disposition_header);
        }
    }
    // set the content length if we have it
    if let Some(content_length) = output.content_length {
        if let Ok(content_length_header) = HeaderValue::from_str(&content_length.to_string()) {
            headers.insert(header::CONTENT_LENGTH, content_length_header);
        }
    }
    headers
}
