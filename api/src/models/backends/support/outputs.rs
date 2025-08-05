//! Support for interacting with outputs (results)
use super::TagSupport;
use crate::models::KeySupport;
use crate::models::{OutputKind, TagRequest};

// dependencies required for api
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        use crate::models::User;
        use crate::utils::{ApiError, Shared};
    }
}

// dependencies required for the client
cfg_if::cfg_if! {
    if #[cfg(feature = "client")] {
        use crate::client::Error;
        use reqwest::multipart::Form;
    }
}

/// The trait for results support in Thorium
pub trait OutputSupport: TagSupport + KeySupport {
    /// Get the tag kind to write to the DB
    fn output_kind() -> OutputKind;

    /// Build a tag request for this output kind
    fn tag_req() -> TagRequest<Self>;

    /// Extend our form with any special data if needed
    ///
    /// # Arguments
    ///
    /// * `form` - The form to extend
    #[cfg(feature = "client")]
    fn extend_form(&mut self, form: Form) -> Result<Form, Error> {
        Ok(form)
    }

    /// Validate our extra field is set if needed
    ///
    /// # Arguments
    ///
    /// * `field` - The extra field to validate
    fn validate_extra(_field: &Option<Self::ExtraKey>) -> bool {
        true
    }

    /// get our extra info
    ///
    /// # Arguments
    ///
    /// `extra` - The extra field to extract
    fn extract_extra(extra: Option<Self::ExtraKey>) -> Self::ExtraKey;

    /// Ensures any user requested groups are valid for this result.
    ///
    /// If no groups are specified then all groups we can see this object in will be returned.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is validating this object is in some groups
    /// * `groups` - The user specified groups to check against
    /// * `shared` - Shared objects in Thorium
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn validate_groups_viewable(
        &self,
        user: &User,
        groups: &mut Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError>;

    /// Ensures any user requested groups are valid and editable for this result.
    ///
    /// If no groups are specified then all groups we can see/edit this object in will be returned.
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is validating this object is in some groups
    /// * `groups` - The user specified groups to check against
    /// * `shared` - Shared objects in Thorium
    #[cfg(feature = "api")]
    #[allow(async_fn_in_trait)]
    async fn validate_groups_editable(
        &self,
        user: &User,
        groups: &mut Vec<String>,
        shared: &Shared,
    ) -> Result<(), ApiError>;
}
