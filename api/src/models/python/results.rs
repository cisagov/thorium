//! Python-specific things for results

use pyo3::pymethods;

use crate::models::{ImageVersion, Output, ResultGetParams};

#[pymethods]
impl ResultGetParams {
    #[new]
    #[pyo3(signature =
        (
            hidden = false,
            tools = Vec::new(),
            groups = Vec::new()
        ),
    )]
    fn new(hidden: bool, tools: Vec<String>, groups: Vec<String>) -> Self {
        Self {
            hidden,
            tools,
            groups,
        }
    }
}

#[pymethods]
impl Output {
    #[pyo3(name = "result")]
    #[getter]
    fn result_py(&self) -> String {
        self.result.to_string()
    }

    #[pyo3(name = "tool_version")]
    #[getter]
    fn tool_version_py(&self) -> Option<String> {
        self.tool_version.as_ref().map(|v| match v {
            ImageVersion::SemVer(version) => version.to_string(),
            ImageVersion::Custom(version_raw) => version_raw.clone(),
        })
    }
}
