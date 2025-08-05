//! Structs for tracking the export of data from Thorium to another service

use chrono::prelude::*;
use uuid::Uuid;

/// A request to start exporting data from Thorium to another system
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ExportRequest {
    /// The name of this export operation
    pub name: String,
    /// The starting timestamp to stream from (earliest)
    pub start: Option<DateTime<Utc>>,
    /// The ending timestamp to stream to (latest)
    pub end: DateTime<Utc>,
}

impl ExportRequest {
    /// Create a new export request
    ///
    /// # Arguments
    ///
    /// * `name` - The name of this new export request
    /// * `end` - The date to stop exporting at
    /// * `groups` - The groups to export
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::ExportRequest;
    /// use chrono::prelude::*;
    ///
    /// // Export the last day of objects
    /// let start = Utc::now();
    /// let end = start - chrono::Duration::days(1);
    /// ExportRequest::new("last 24 hours", end)
    ///   .start(start);
    /// ````
    pub fn new<T: Into<String>>(name: T, end: DateTime<Utc>) -> Self {
        ExportRequest {
            name: name.into(),
            start: None,
            end,
        }
    }

    /// Set the start date for the most recent item to export
    ///
    /// # Arguments
    ///
    /// * `start` - The most recent timestamp to export items at
    pub fn start(mut self, start: DateTime<Utc>) -> Self {
        self.start = Some(start);
        self
    }
}

/// A request to add a new cursor to an export operation
///
/// Its important to note that Thorium peers backwards into the past when
/// listing data. This means start > current > end.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ExportCursorRequest {
    /// The ID for this cursor
    pub id: Uuid,
    /// The date this cursor should start crawling at (earliest)
    pub start: DateTime<Utc>,
    /// The timestamp for the latest object that was exported
    pub current: DateTime<Utc>,
    /// the date this cursor should stop crawling at (latest)
    pub end: DateTime<Utc>,
    /// The export error this cursor is retrying if its tied to an error
    pub error: Option<Uuid>,
}

impl ExportCursorRequest {
    /// Create a new [`ExportCursorRequest`]
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the cursor to add
    /// * `start` - The newest timestamp of data this cursor will export
    /// * `current` - The current timestamp this cursor has exported
    /// * `end` - The oldest timestamp of data this cursor will export
    pub fn new(id: Uuid, start: DateTime<Utc>, current: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        ExportCursorRequest {
            id,
            start,
            current,
            end,
            error: None,
        }
    }

    /// Set the error flag to denote this cursor is related to retrying an error
    ///
    /// # Arguments
    ///
    /// * `error` - The export error we are crawling
    pub fn error(mut self, error: Uuid) -> Self {
        // set the error flag
        self.error = Some(error);
        self
    }

    /// Set the error flag to denote this cursor is related to retrying an error using a mutable reference
    ///
    /// # Arguments
    ///
    /// * `error` - The export error we are crawling
    pub fn error_mut(&mut self, error: Uuid) {
        // set the error flag
        self.error = Some(error);
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        /// Default the Result list limit to 50
        fn default_list_limit() -> usize {
            50
        }

        /// Export all results kinds by default
        #[must_use]
        pub fn default_export_kinds() -> Vec<super::ExportOps> {
            vec![super::ExportOps::Results]
        }

        /// The query params for searching results
        #[derive(Deserialize, Debug)]
        #[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
        pub struct ExportListParams {
            /// The kinds of results to export
            #[serde(default = "default_export_kinds")]
            pub kinds: Vec<super::ExportOps>,
            /// When to start listing data at
            #[serde(default = "Utc::now")]
            pub start: DateTime<Utc>,
            /// When to stop listing data at
            pub end: Option<DateTime<Utc>>,
            /// The cursor id to use if one exists
            pub cursor: Option<Uuid>,
            /// The max number of items to return in this response
            #[serde(default = "default_list_limit")]
            pub limit: usize,
        }

        impl ExportListParams {
            /// Get the end timestamp or get a sane default for results
            #[cfg(feature = "api")]
            pub fn end(
                &self,
                shared: &crate::utils::Shared,
            ) -> Result<DateTime<Utc>, crate::utils::ApiError> {
                match self.end {
                    Some(end) => Ok(end),
                    None => match Utc.timestamp_opt(shared.config.thorium.results.earliest, 0) {
                        chrono::LocalResult::Single(default_end) => Ok(default_end),
                        _ => crate::internal_err!(format!(
                            "default earliest results timestamp is invalid or ambiguous - {}",
                            shared.config.thorium.results.earliest
                        )),
                    },
                }
            }
        }

        impl Default for ExportListParams {
            /// Create a default export list params
            fn default() -> Self {
                ExportListParams {
                    kinds: Vec::default(),
                    start: Utc::now(),
                    end: None,
                    cursor: None,
                    limit: default_list_limit(),
                }
            }
        }
    }
}

/// A request to update an export operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ExportUpdate {
    /// The timestamp for the latest object that was exported
    pub current: DateTime<Utc>,
}

impl ExportUpdate {
    /// Create a new [`ExportUpdate`]
    ///
    /// # Arguments
    ///
    /// * `id` - The export cursor to update
    /// * `current` - The oldest timestamp of data this export operation has exported
    pub fn new(current: DateTime<Utc>) -> Self {
        ExportUpdate { current }
    }
}

/// An export of data from Thorium to another service
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "scylla-utils", derive(scylla::DeserializeRow))]
#[cfg_attr(
    feature = "scylla-utils",
    scylla(flavor = "enforce_order", skip_name_checks)
)]
/// Add FromRow support for scylla loading if API mode is enabled
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Export {
    /// The name of this export operation
    pub name: String,
    /// The user that owns this export operation
    pub user: String,
    /// The starting timestamp to stream from (earliest)
    pub start: Option<DateTime<Utc>>,
    /// The current spot to start exporting new data from in export
    pub current: DateTime<Utc>,
    /// The ending timestamp to stream to (latest)
    pub end: DateTime<Utc>,
}

/// An error from streaming a chunk of an export stream
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ExportError {
    /// The ID for this error
    pub id: Uuid,
    /// The start of the chunk of data that we failed to export
    pub start: DateTime<Utc>,
    /// The end of the chunk of data that we failed to export
    pub end: DateTime<Utc>,
    /// The error number/code that occurred
    pub code: Option<u16>,
    /// A message explaining the error that occurred
    pub msg: String,
}

/// A request to save an error for a specific export operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ExportErrorRequest {
    /// The start of the chunk of data that we failed to export
    pub start: DateTime<Utc>,
    /// The end of the chunk of data that we failed to export
    pub end: DateTime<Utc>,
    /// The error number/code that occurred
    pub code: Option<u16>,
    /// A message explaining the error that occurred
    pub msg: String,
}

impl ExportErrorRequest {
    /// Creates a new export error request
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the chunk of data that we failed to export
    /// * `end` - The end of the chunk of data that we failed to export
    /// * `msg` - A message explaining that error that occurred
    pub fn new<M: Into<String>>(start: DateTime<Utc>, end: DateTime<Utc>, msg: M) -> Self {
        ExportErrorRequest {
            start,
            end,
            code: None,
            msg: msg.into(),
        }
    }

    /// Set the code for the error that occurred
    ///
    /// # Arguments
    ///
    /// * `code` - The error code that occurred
    pub fn code(mut self, code: u16) -> Self {
        self.code = Some(code);
        self
    }

    /// set the code for the error that occurred with a mutable reference
    ///
    /// # Arguments
    ///
    /// * `code` - The error code that occurred
    pub fn code_mut(&mut self, code: u16) {
        self.code = Some(code);
    }
}

/// A response from saving an export error
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct ExportErrorResponse {
    /// The uuid of the saved error
    pub id: Uuid,
}
