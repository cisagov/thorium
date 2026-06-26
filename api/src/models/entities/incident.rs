//! An incident or engagement

#[cfg(feature = "client")]
use crate::{multipart_list, multipart_text};

/// An incident entity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Incident {
    /// The cover term used for this incident
    pub cover_term: Option<String>,
    /// The mission teams involved in this incident
    pub mission_teams: Vec<String>,
    /// The networks this incident was on
    pub networks: Vec<String>,
    /// The machines this incident was found on
    pub machines: Vec<String>,
    /// The locations this incident is from
    pub locations: Vec<String>,
}

impl Incident {
    /// Create a new incident entity with the info in the form
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Self {
        // build incident entity
        Incident {
            cover_term: form.cover_term,
            mission_teams: form.mission_teams,
            networks: form.networks,
            machines: form.machines,
            locations: form.locations,
        }
    }
}

/// A request to create an incident entity
#[derive(Debug, Clone, Default, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct IncidentRequest {
    /// The cover term used for this incident
    pub cover_term: Option<String>,
    /// The mission teams involved in this incident
    pub mission_teams: Vec<String>,
    /// The networks this incident was on
    pub networks: Vec<String>,
    /// The machines this incident was found on
    pub machines: Vec<String>,
    /// The locations this incident is from
    pub locations: Vec<String>,
}

impl IncidentRequest {
    /// Add this device entity metadata to a form
    ///
    /// # Arguments
    ///
    /// * `form` - The form to add too
    #[cfg(feature = "client")]
    pub fn add_to_form(
        mut self,
        form: reqwest::multipart::Form,
    ) -> Result<reqwest::multipart::Form, crate::Error> {
        // always set our entity kind
        let form = form.text("kind", super::EntityKinds::Incident.as_str());
        // add our incident metadata
        let form = multipart_text!(form, "metadata[cover_term]", self.cover_term);
        let form = multipart_list!(form, "metadata[mission_teams][]", self.mission_teams);
        let form = multipart_list!(form, "metadata[networks][]", self.networks);
        let form = multipart_list!(form, "metadata[machines][]", self.machines);
        let form = multipart_list!(form, "metadata[locations][]", self.locations);
        Ok(form)
    }
}
