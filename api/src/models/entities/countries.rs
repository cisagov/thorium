//! The country entity in Thorium

use isocountry::{CountryCode, CountryCodeParseErr};

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Country {
    /// This countries code
    #[cfg_attr(feature = "api", schema(value_type = String))]
    code: CountryCode,
    /// The name of the country in question
    name: String,
}

impl Country {
    /// Create a new country object from a country code
    ///
    /// # Arguments
    ///
    /// * `code_str` - The code for this country as a str
    pub fn new(code_str: &String) -> Result<Self, CountryCodeParseErr> {
        // get this countries code
        let code = CountryCode::for_alpha2(code_str)?;
        // get our countries name
        let name = code.name().to_owned();
        // build our country object
        let country = Country { code, name };
        Ok(country)
    }
}

impl Ord for Country {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name.cmp(&other.name)
    }
}

impl PartialOrd for Country {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
