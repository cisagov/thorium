use pyo3::pymethods;

use crate::models::{GenericJobArgs, GenericJobKwargs, GenericJobOpts};

#[pymethods]
impl GenericJobArgs {
    #[new]
    #[pyo3(signature =
        (
            positionals: "list[str]" = Vec::new(),
            kwargs: "dict[str, list[str]]" = GenericJobKwargs::default(),
            switches: "list[str]" = Vec::new(),
            opts: "GenericJobOpts" = GenericJobOpts::default()
        ) -> "GenericJobArgs",
    )]
    fn new(
        positionals: Vec<String>,
        kwargs: GenericJobKwargs,
        switches: Vec<String>,
        opts: GenericJobOpts,
    ) -> Self {
        Self {
            positionals,
            kwargs,
            switches,
            opts,
        }
    }
}

#[pymethods]
impl GenericJobOpts {
    #[new]
    #[pyo3(signature =
        (
            override_positionals=false,
            override_kwargs=false,
            override_cmd: "list[str] | None" = None
        ) -> "GenericJobOpts"
    )]
    fn new_py(
        override_positionals: bool,
        override_kwargs: bool,
        override_cmd: Option<Vec<String>>,
    ) -> Self {
        Self {
            override_positionals,
            override_kwargs,
            override_cmd,
        }
    }
}
