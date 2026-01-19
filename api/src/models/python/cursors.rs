//! Definitions of non-generic cursors compatible with `PyO3`

use pyo3::{pyclass, pymethods};

use crate::Error;
use crate::models::{
    CountCursorBlocking, CursorBlocking, Sample, SampleListLine, SamplePy, TagCounts,
};

/// Create a non-generic blocking cursor for the given type
macro_rules! non_generic_cursor {
    ($name: ident, $type: ident, $pydata: literal) => {
        #[pyclass(skip_from_py_object)]
        pub struct $name {
            inner: CursorBlocking<$type>,
        }

        impl From<CursorBlocking<$type>> for $name {
            /// Construct the non-generic blocking cursor from the generic one
            fn from(cursor: CursorBlocking<$type>) -> Self {
                Self { inner: cursor }
            }
        }

        #[pymethods]
        impl $name {
            /// Returns true if the cursor is exhausted
            pub fn exhausted(&self) -> bool {
                self.inner.exhausted()
            }

            /// Retrieve the data in the cursor
            pub fn data(&self) -> Vec<$type> {
                self.inner.data().clone()
            }

            /// Attempt to retrieve more data
            pub fn refill(&mut self) -> Result<(), Error> {
                self.inner.refill()
            }
        }
    };
    // with PyO3-compatible mapping type
    ($name: ident, $type: ident, $map_type: ident, $pydata: literal) => {
        #[pyclass(skip_from_py_object)]
        pub struct $name {
            inner: CursorBlocking<$type>,
        }

        impl From<CursorBlocking<$type>> for $name {
            /// Construct the non-generic blocking cursor from the generic one
            fn from(cursor: CursorBlocking<$type>) -> Self {
                Self { inner: cursor }
            }
        }

        #[pymethods]
        impl $name {
            /// Returns true if the cursor is exhausted
            pub fn exhausted(&self) -> bool {
                self.inner.exhausted()
            }

            /// Retrieve the data in the cursor
            pub fn data(&self) -> Vec<$map_type> {
                self.inner
                    .data()
                    .into_iter()
                    .cloned()
                    .map(std::convert::Into::into)
                    .collect()
            }

            /// Attempt to retrieve more data
            pub fn refill(&mut self) -> Result<(), Error> {
                self.inner.refill()
            }
        }
    };
}

/// Create a non-generic blocking count cursor for the given type
macro_rules! non_generic_count_cursor {
    ($name: ident, $type: ident, $pydata: literal) => {
        #[pyclass(skip_from_py_object)]
        pub struct $name {
            inner: CountCursorBlocking<$type>,
        }

        impl From<CountCursorBlocking<$type>> for $name {
            /// Construct the non-generic blocking cursor from the generic one
            fn from(cursor: CountCursorBlocking<$type>) -> Self {
                Self { inner: cursor }
            }
        }

        #[pymethods]
        impl $name {
            /// Returns true if the cursor is exhausted
            pub fn exhausted(&self) -> bool {
                self.inner.exhausted()
            }

            /// Retrieve the data in the cursor
            pub fn data(&self) -> <$type as crate::models::CountCursorSupport>::Data {
                self.inner.data().clone()
            }

            /// Attempt to retrieve more data
            pub fn refill(&mut self) -> Result<(), Error> {
                self.inner.refill()
            }
        }
    };
}

non_generic_cursor!(SampleListLineCursor, SampleListLine, "list[SampleListLine]");
non_generic_cursor!(SampleCursor, Sample, SamplePy, "list[Sample]");
non_generic_count_cursor!(TagCountsCursor, TagCounts, "dict[str, TagKeyCounts]");
