//! A Sigma rule that can be applied to logs or log like data
use sigma_rust::rule::Level;
use std::str::FromStr;

use crate::models::InvalidEnum;

/// The different kinds of data this rule should be run on
#[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::Display, PartialEq, Eq)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum SigmaRuleAppliesTo {
    /// Apply this rule to windows processes
    WindowsProcesses,
    /// Apply this rule to network connections
    NetworkConnections,
}

impl SigmaRuleAppliesTo {
    /// Convert this [`RuleAppliesTo`] into a str
    pub fn as_str(&self) -> &'static str {
        match self {
            SigmaRuleAppliesTo::WindowsProcesses => "WindowsProcesses",
            SigmaRuleAppliesTo::NetworkConnections => "NetworkConnections",
        }
    }
}

impl FromStr for SigmaRuleAppliesTo {
    type Err = InvalidEnum;

    /// Cast a str to an [`RuleAppliesTo`]
    ///
    /// # Arguments
    ///
    /// * `val` - The str to cast
    fn from_str(val: &str) -> Result<Self, Self::Err> {
        match val {
            "WindowsProcesses" => Ok(SigmaRuleAppliesTo::WindowsProcesses),
            "NetworkConnections" => Ok(SigmaRuleAppliesTo::NetworkConnections),
            _ => Err(InvalidEnum(format!("Unknown enum variant: {val}"))),
        }
    }
}

/// Automatically promote this sigma rule hit to a flag
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SigmaAutoFlag {
    /// The interesting, odd, or suspicious characteristic
    pub content: Option<String>,
    /// The reason for this Flag
    pub reasoning: String,
}

/// The action to take when a sigma rule hits
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreJson))]
pub enum SigmaActionToTake {
    /// Automatically promote hit to a flag
    Flag(SigmaAutoFlag),
}

/// A Sigma rule that can be applied to logs or log like data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct SigmaRule {
    /// The original unparsed rule
    pub rule: String,
    /// What types of data this sigma rule applies too
    pub applies_to: Vec<SigmaRuleAppliesTo>,
    /// The score to apply when this rule hits
    pub score: i64,
    /// The actions to take when this sigma rule hits
    #[serde(default)]
    pub actions: Vec<SigmaActionToTake>,
}

impl SigmaRule {
    /// Create a new [`SigmaRule`]
    ///
    /// # Arguments
    ///
    /// * `rule` - The sigma rule to create
    /// * `applies_to` - What this sigma rule applies too
    pub fn new(
        rule: impl Into<String>,
        applies_to: SigmaRuleAppliesTo,
    ) -> Result<Self, serde_norway::Error> {
        // convert this rule to a string
        let rule = rule.into();
        // parse our rule to make sure its valid
        let score = Self::validate_and_get_score(&rule)?;
        // build our validated sigma rule
        let validated = SigmaRule {
            rule,
            applies_to: vec![applies_to],
            score,
            actions: Vec::default(),
        };
        Ok(validated)
    }

    /// Update this sigma rules score
    ///
    /// # Arguments
    ///
    /// * `score` - The new score to set
    pub fn score(mut self, score: i64) -> Self {
        self.score = score;
        self
    }

    /// Update the action this sigma rule should take
    ///
    /// # Arguments
    ///
    /// * `action` - The action to take when this sigma rule hits
    pub fn action(mut self, action: SigmaActionToTake) -> Self {
        self.actions.push(action);
        self
    }

    /// Validate a sigma rule is valid and get its score
    ///
    /// # Arguments
    ///
    /// * `rule_str` - The sigma rule to parse and validate
    fn validate_and_get_score(rule_str: &str) -> Result<i64, serde_norway::Error> {
        // parse and validate this sigma rule
        let parsed = sigma_rust::rule_from_yaml(rule_str)?;
        // use a different score for each level
        match parsed.level {
            None | Some(Level::Informational) => Ok(0),
            Some(Level::Low) => Ok(1),
            Some(Level::Medium) => Ok(5),
            Some(Level::High) => Ok(20),
            Some(Level::Critical) => Ok(50),
        }
    }

    /// Create a new [`SigmaRule`] with the info in the form
    ///
    /// # Errors
    ///
    /// * A sigma rule was not found in the form
    ///
    /// # Arguments
    ///
    /// * `form` -  The metadata form
    #[cfg(feature = "api")]
    pub fn from_form(
        form: crate::models::entities::EntityMetadataForm,
    ) -> Result<Self, crate::utils::ApiError> {
        // if we don't have the rule field then return an error
        let rule = match form.sigma_rule {
            Some(rule) => rule,
            None => {
                return crate::bad!("Sigma rules must have a rule!".to_owned());
            }
        };
        // parse our rule to make sure its valid
        let level_score = Self::validate_and_get_score(&rule)?;
        // if we don't have a user supplied score then add our own based on the level for this rule
        let score = match form.score {
            // use the user supplied score
            Some(score) => score,
            None => level_score,
        };
        // build our windows process entity
        Ok(SigmaRule {
            rule,
            applies_to: form.sigma_applies_to,
            score,
            actions: form.sigma_actions,
        })
    }

    /// Add this [`SigmaRule`]'s metadata to a form
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
        let form = form
            .text("kind", crate::models::EntityKinds::SigmaRule.as_str())
            .text("metadata[rule]", self.rule)
            .text("metadata[score]", self.score.to_string());
        // add what data this sigma rule applies to
        let mut form =
            crate::multipart_list_conv!(form, "metadata[sigma_applies_to]", self.applies_to);
        // add what actions to take when this sigma rule hits
        crate::multipart_list_serialize!(form, "metadata[sigma_actions]", self.actions);
        Ok(form)
    }
}
