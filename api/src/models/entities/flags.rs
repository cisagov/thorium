//! An entity denoting something interesting, odd, or suspicious about something in Thorium

/// How confident we are in this flag
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Flag {
    /// How suspicious this flag is where higher numbers are more suspicious
    suspicion: isize,
    /// How confident/reliable this flag is
    confidence: Confidence,
    /// The interesting, odd, or suspicious characteristic
    content: String,
    /// The reason for this Flag
    reasoning: String,
}
