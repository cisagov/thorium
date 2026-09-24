//! Utilties for testing the Thorium API

mod api;
pub mod generators;
mod helpers;
mod impls;

use crate::client::Error;
pub use api::{CONF, admin_client};

// expose a blocking admin client for sync tests
#[cfg(all(feature = "sync", not(feature = "python")))]
pub use api::admin_client_blocking;

/// Assert that a result is an error carrying the expected HTTP status code
///
/// The `fail!` test macro re-evaluates its argument (which moves owned inputs) and requires the
/// `Ok` type to implement `Debug`; several client calls here return non-`Debug` values
/// (`DownloadedFile`, `Cursor<_>`) or consume owned blueprints, so this helper checks an
/// already-evaluated result without either constraint.
///
/// # Arguments
///
/// * `result` - The already-evaluated result to check
/// * `code` - The HTTP status code the error must carry
/// * `context` - A short description of what was expected, used in failure messages
pub fn assert_status<T>(result: Result<T, Error>, code: u16, context: &str) -> Result<(), Error> {
    // inspect the result, discarding any success value so `T` need not be `Debug`
    match result {
        // a success means the secure behavior is not enforced yet, so fail the test
        Ok(_) => Err(Error::new(format!(
            "{context}: expected status {code} but the request succeeded"
        ))),
        // an error must carry exactly the expected status code
        Err(error) => match error.status() {
            // the status matched so the secure behavior held
            Some(status) if status.as_u16() == code => Ok(()),
            // a different status is still a failure to enforce the expected boundary
            Some(status) => Err(Error::new(format!(
                "{context}: expected status {code} but got {}",
                status.as_u16()
            ))),
            // no status at all means the error was not the authorization rejection we expected
            None => Err(Error::new(format!(
                "{context}: expected status {code} but the error carried no status: {error}"
            ))),
        },
    }
}
