//! PE static-analysis entities extracted from executables
//!
//! These mirror the per-file "PE Sections" and "Imports" tables found in CISA
//! Malware Analysis Reports (MARs). Each entity attaches to its file/sample via
//! an association rather than nesting under a parent executable entity.

use std::hash::{Hash, Hasher};

/// A single section within a PE/binary (e.g. `.text`, `.rsrc`, `UPX1`)
///
/// The section's name is carried by the parent entity's `name` field, so this
/// metadata only holds the per-section details from a MAR's "PE Sections" table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PeSectionEntity {
    /// The MD5 of this section's raw data
    pub md5: Option<String>,
    /// The raw (on disk) size of this section in bytes
    pub raw_size: Option<u64>,
    /// The virtual (in memory) size of this section in bytes
    pub virtual_size: Option<u64>,
    /// The Shannon entropy of this section's data
    pub entropy: Option<f64>,
}

impl Hash for PeSectionEntity {
    /// Hash this section's identifying data
    ///
    /// `entropy` is intentionally excluded: `f64` has no sound `Hash` (e.g.
    /// `+0.0`/`-0.0` differ in bits and `NaN` never equals itself), and entropy
    /// is not part of a section's identity since its `md5` already is. `Hash`
    /// only requires that equal values hash equally, so omitting a field is safe.
    ///
    /// # Arguments
    ///
    /// * `state` - The hasher to write our identifying data to
    fn hash<H: Hasher>(&self, state: &mut H) {
        // hash this section's content hash and sizes, skipping the entropy float
        self.md5.hash(state);
        self.raw_size.hash(state);
        self.virtual_size.hash(state);
    }
}

impl PeSectionEntity {
    /// Create a new empty [`PeSectionEntity`]
    ///
    /// This will not save this entity to Thorium.
    #[must_use]
    pub fn new() -> Self {
        // start with all section details unset
        PeSectionEntity {
            md5: None,
            raw_size: None,
            virtual_size: None,
            entropy: None,
        }
    }

    /// Set the MD5 of this section's raw data
    ///
    /// # Arguments
    ///
    /// * `md5` - The MD5 of this section's raw data
    #[must_use]
    pub fn md5(mut self, md5: impl Into<String>) -> Self {
        self.md5 = Some(md5.into());
        self
    }

    /// Set the raw (on disk) size of this section
    ///
    /// # Arguments
    ///
    /// * `raw_size` - The raw size of this section in bytes
    #[must_use]
    pub fn raw_size(mut self, raw_size: u64) -> Self {
        self.raw_size = Some(raw_size);
        self
    }

    /// Set the virtual (in memory) size of this section
    ///
    /// # Arguments
    ///
    /// * `virtual_size` - The virtual size of this section in bytes
    #[must_use]
    pub fn virtual_size(mut self, virtual_size: u64) -> Self {
        self.virtual_size = Some(virtual_size);
        self
    }

    /// Set the Shannon entropy of this section's data
    ///
    /// # Arguments
    ///
    /// * `entropy` - The Shannon entropy of this section's data
    #[must_use]
    pub fn entropy(mut self, entropy: f64) -> Self {
        self.entropy = Some(entropy);
        self
    }

    /// Create a new [`PeSectionEntity`] with the info in the form
    ///
    /// # Arguments
    ///
    /// * `form` - The metadata form to build this section from
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // all section details are optional so just map them over
        Ok(PeSectionEntity {
            md5: form.md5,
            raw_size: form.raw_size,
            virtual_size: form.virtual_size,
            entropy: form.entropy,
        })
    }

    /// Add this [`PeSectionEntity`]'s metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form.text("kind", super::EntityKinds::PeSection.as_str());
        // set the optional metadata fields if they exist
        let form = crate::multipart_text_to_string!(form, "metadata[md5]", self.md5);
        let form = crate::multipart_text_to_string!(form, "metadata[raw_size]", self.raw_size);
        let form =
            crate::multipart_text_to_string!(form, "metadata[virtual_size]", self.virtual_size);
        let form = crate::multipart_text_to_string!(form, "metadata[entropy]", self.entropy);
        Ok(form)
    }
}

/// An imported library and the functions imported from it
///
/// The DLL/library name is carried by the parent entity's `name` field, so this
/// metadata only holds the functions imported from that library.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PeImportEntity {
    /// The functions imported from this library
    pub functions: Vec<String>,
}

impl PeImportEntity {
    /// Create a new empty [`PeImportEntity`]
    ///
    /// This will not save this entity to Thorium.
    #[must_use]
    pub fn new() -> Self {
        // start with no imported functions
        PeImportEntity {
            functions: Vec::default(),
        }
    }

    /// Add a single imported function to this library
    ///
    /// # Arguments
    ///
    /// * `function` - The name of the imported function to add
    #[must_use]
    pub fn function(mut self, function: impl Into<String>) -> Self {
        // add this function to our list of imported functions
        self.functions.push(function.into());
        self
    }

    /// Set the full list of functions imported from this library
    ///
    /// # Arguments
    ///
    /// * `functions` - The imported functions to set
    #[must_use]
    pub fn functions<I, T>(mut self, functions: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        // convert and set our list of imported functions
        self.functions = functions.into_iter().map(Into::into).collect();
        self
    }

    /// Create a new [`PeImportEntity`] with the info in the form
    ///
    /// # Arguments
    ///
    /// * `form` - The metadata form to build this import from
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // build our import entity from the form's functions list
        Ok(PeImportEntity {
            functions: form.functions,
        })
    }

    /// Add this [`PeImportEntity`]'s metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let mut form = form.text("kind", super::EntityKinds::PeImport.as_str());
        // add each imported function to our form
        for function in self.functions {
            form = form.text("metadata[functions][]", function);
        }
        Ok(form)
    }
}
