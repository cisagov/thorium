use pyo3::pymethods;

use crate::models::{CommitishKinds, RepoDependencyRequest};

#[pymethods]
impl RepoDependencyRequest {
    #[new]
    #[pyo3(signature =
        (
            url,
            commitish = None,
            kind = None
        ) -> "RepoDependencyRequest"
    )]
    fn new_py(url: String, commitish: Option<String>, kind: Option<CommitishKinds>) -> Self {
        Self {
            url,
            commitish,
            kind,
        }
    }
}
