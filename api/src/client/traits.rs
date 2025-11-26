//! Traits defining shared behavior between elements of the Thorium client
use reqwest::Client;

mod notifications;
mod progress;
mod results;

pub(super) use notifications::NotificationsClient;
pub use results::ResultsClient;
pub(super) use results::ResultsClientHelper;

#[cfg(feature = "sync")]
pub use results::ResultsClientBlocking;

/// A helper trait that returns the necessary information to send/receive for a
/// particular client
pub(super) trait GenericClient {
    /// Get a base URL to the respective route in the API from the implementor
    fn base_url(&self) -> String;

    /// Get a configured client from the implementor for this route in the API
    fn client(&self) -> &Client;

    /// Get an auth token from the implementor
    fn token(&self) -> &str;
}

/// Update a download or uploads progress bar
pub trait TransferProgress {
    /// Update a progress bar to reflect a slice of bytes was transferred
    ///
    /// # Arguments
    ///
    /// * `transferred` - The bytes that were transferred
    fn update_progress_bytes(&self, transferred: &[u8]);
}
