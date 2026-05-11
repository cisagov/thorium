//! An entity denoting something interesting, odd, or suspicious about something in Thorium

/// How confident we are in this flag
#[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::Display, strum::EnumString, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum Confidence {
    /// This is known to be a fact
    Fact,
    /// This is more then likely true
    Likely,
    /// This may or may not be true (50/50 odds)
    Unsure,
    /// This is unlikely to be true and should be validated
    Untrusted,
}

/// A flag is a reason that something is interesting, odd, or suspicious
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Flag {
    /// How suspicious this flag is where higher numbers are more suspicious
    pub suspicion: i64,
    /// How confident/reliable this flag is
    pub confidence: Confidence,
    /// The interesting, odd, or suspicious characteristic
    pub content: Option<String>,
    /// The reason for this Flag
    pub reasoning: String,
}

impl Flag {
    /// Create a new [`Flag`] with the info in the form
    ///
    /// # Errors
    ///
    /// * A suspicion, confidence, or reasoning was not found in the form.
    ///
    /// # Arguments
    ///
    /// * `form` -  The update form
    #[cfg(feature = "api")]
    pub fn from_form(form: super::EntityMetadataForm) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the source field then return an error
        let suspicion = match form.suspicion {
            Some(suspicion) => suspicion,
            None => {
                return crate::bad!("Flag entities must have a suspicion!".to_owned());
            }
        };
        // if we don't have the confidence field then return an error
        let confidence = match form.confidence {
            Some(confidence) => confidence,
            None => {
                return crate::bad!("Flag entities must have a confidence!".to_owned());
            }
        };
        // if we don't have the reasoning field then return an error
        let reasoning = match form.reasoning {
            Some(reasoning) => reasoning,
            None => {
                return crate::bad!("Flag entities must have a reasoning!".to_owned());
            }
        };
        // build our flag entity
        Ok(Flag {
            suspicion,
            confidence,
            content: form.content,
            reasoning,
        })
    }

    /// Add this flags entity metadata to a form
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
        let form = form
            .text("kind", super::EntityKinds::Flag.as_str())
            // always set our required fields
            .text("metadata[suspicion]", self.suspicion.to_string())
            .text("metadata[confidence]", self.confidence.to_string())
            .text("metadata[reasoning]", self.reasoning);
        // set the metadata fields for this entity if htey exist
        let form = crate::multipart_text!(form, "metadata[content]", self.content.clone());
        Ok(form)
    }
}
