//! Synchronous Python files client based on Rust
//!
//! Specific logic/functions that are difficult/awkward to implement
//! with our proc-macros are manually written here.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use pyo3::pymethods;

use crate::Error;
use crate::client::FilesBlocking;
use crate::models::FileListOpts;
use crate::models::python::{SampleCursor, SampleListLineCursor, TagCountsCursor};

#[pymethods]
impl FileListOpts {
    /// Creates a new [`FileListOpts`] for listing files
    #[new]
    #[pyo3(signature =
        (
            cursor: "UUID | None" = None,
            start = None,
            end = None,
            page_size = 50,
            limit = None,
            groups = Vec::new(),
            tags = HashMap::new(),
            tags_case_insensitive = false,
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn new_py(
        cursor: Option<uuid::Uuid>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        page_size: usize,
        limit: Option<usize>,
        groups: Vec<String>,
        tags: HashMap<String, Vec<String>>,
        tags_case_insensitive: bool,
    ) -> Self {
        Self {
            cursor,
            start,
            end,
            page_size,
            limit,
            groups,
            tags,
            tags_case_insensitive,
        }
    }
}

#[pymethods]
impl FilesBlocking {
    #[pyo3(name = "list")]
    pub fn list_py(&self, opts: &FileListOpts) -> Result<SampleListLineCursor, Error> {
        let cursor = self.list(opts)?;
        Ok(SampleListLineCursor::from(cursor))
    }

    #[pyo3(name = "list_details")]
    pub fn list_details_py(&self, opts: &FileListOpts) -> Result<SampleCursor, Error> {
        let cursor = self.list_details(opts)?;
        Ok(SampleCursor::from(cursor))
    }

    #[pyo3(name = "count")]
    pub fn count_py(&self, opts: &FileListOpts) -> Result<TagCountsCursor, Error> {
        let cursor = self.count(opts)?;
        Ok(cursor.into())
    }
}
