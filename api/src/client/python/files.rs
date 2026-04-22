//! Synchronous Python files client based on Rust
//!
//! Specific logic/functions that are difficult/awkward to implement
//! with our proc-macros are manually written here.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use pyo3::pymethods;
use uuid::Uuid;

use crate::Error;
use crate::client::{FilesBlocking, ResultsClientBlocking};
use crate::models::python::{SampleCursor, SampleListLineCursor, TagCountsCursor};
use crate::models::{Attachment, FileListOpts, OutputMap, ResultGetParams};

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

    #[pyo3(name = "get_results")]
    fn get_results_py(&self, sha256: &str, params: &ResultGetParams) -> Result<OutputMap, Error> {
        self.get_results(sha256, params)
    }

    #[pyo3(
        name = "download_result_file",
        signature = (
            sha256: "str",
            tool: "str",
            result_id: "UUID",
            path: "Path"
        ) -> "Attachment"
    )]
    fn download_result_file_py(
        &self,
        sha256: &str,
        tool: &str,
        result_id: Uuid,
        path: PathBuf,
    ) -> Result<Attachment, Error> {
        self.download_result_file(sha256, tool, &result_id, path)
    }
}
