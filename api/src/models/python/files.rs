//! Python-specific things for files

use pyo3::{Bound, Python, pymethods, types::PyBytes};

use crate::models::Attachment;

#[pymethods]
impl Attachment {
    #[pyo3(name = "bytes")]
    #[getter]
    fn bytes_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }
}
