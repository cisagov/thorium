//! Helpers for toolbox init
//!
//! `toolbox init` fills in image/pipeline configs via an editor (see
//! `review_config_in_editor`), so this module only provides resource-name
//! validation, the small toolbox-level config prompt (name/registry), and the
//! default-config "answer" seeds that build the starting configs.

use colored::Colorize;
use regex::Regex;
use std::sync::LazyLock;
use thorium::Error;

// ─── Validation ─────────────────────────────────────────────────────────────

/// The maximum length the Thorium API allows for an image or pipeline name (`bounder::string_lower`)
pub(super) const RESOURCE_NAME_MAX: usize = 25;

/// The maximum length the Thorium API allows for a group name (`bounder::string_lower`)
pub(super) const GROUP_NAME_MAX: usize = 50;

// Guards a name to exactly the character set the Thorium API accepts (`bounder::string_lower`):
// lowercase ASCII letters, digits, and '-'. Validating client-side with the same rule means a name
// `init` accepts is one the API will accept too, so authoring fails fast instead of being rejected
// later at `build`/`import`. The leading-character restriction (no leading '-') is stricter than the
// API but keeps every name a valid k8s name and a bare TOML key. Both the interactive and
// `--non-interactive` init paths must run it.
static NAME_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9-]*$").unwrap());

/// Validates a resource name against the Thorium name rule (character set + length)
///
/// # Arguments
///
/// * `value` - The name to validate
/// * `max` - The maximum allowed length (e.g. [`RESOURCE_NAME_MAX`] or [`GROUP_NAME_MAX`])
pub(super) fn validate_name(value: &str, max: usize) -> Result<(), String> {
    // reject an empty name up front: the regex would also reject it, but a dedicated
    // message tells the user the field is required rather than malformed
    if value.is_empty() {
        return Err("Name cannot be empty".into());
    }
    // bound the length to match the API's own cap so an over-long name fails here, not at import.
    // names are ASCII (enforced by the regex below) so byte length equals character count
    if value.len() > max {
        return Err(format!("Name must be at most {max} characters"));
    }
    // enforce the API's character set so the name round-trips to Thorium and stays a safe TOML
    // manifest key / k8s name
    if !NAME_RE.is_match(value) {
        return Err(
            "Must start with a lowercase letter or digit and contain only lowercase letters, \
             digits, or '-'"
                .into(),
        );
    }
    Ok(())
}

// ─── Prompt Helpers ──────────────────────────────────────────────────────────

/// Prompts for a free-form value with a default, returning the user's entry
///
/// # Arguments
///
/// * `label` - The prompt label
/// * `default` - The value used when the user enters nothing
fn prompt_input(label: &str, default: &str) -> Result<String, Error> {
    // build a free-form text prompt whose empty-entry fallback is the supplied default,
    // then surface any dialoguer failure (e.g. a closed/non-interactive stdin) as an Error
    dialoguer::Input::new()
        .with_prompt(label)
        .default(default.to_string())
        .interact_text()
        .map_err(|e| Error::new(format!("Failed to read input: {e}")))
}

/// Prompts for a required name, re-prompting until it passes [`validate_name`]
///
/// # Arguments
///
/// * `label` - The prompt label
/// * `max` - The maximum allowed name length, passed through to [`validate_name`]
fn prompt_name_required(label: &str, max: usize) -> Result<String, Error> {
    // attach `validate_name` as the per-keystroke validator so dialoguer re-prompts in place
    // until the entry is a valid Thorium name, then map a read failure to an Error
    dialoguer::Input::<String>::new()
        .with_prompt(label)
        .validate_with(move |value: &String| validate_name(value, max))
        .interact_text()
        .map_err(|e| Error::new(format!("Failed to read input: {e}")))
}

/// Prompts for a Thorium group name (a required, validated identifier)
///
/// # Arguments
///
/// * `label` - The prompt label
pub fn prompt_group_name(label: &str) -> Result<String, Error> {
    prompt_name_required(label, GROUP_NAME_MAX)
}

// ─── Answer Structs ──────────────────────────────────────────────────────────

/// The default-config seed used to scaffold an image's config + manifest
pub struct ImageConfigAnswers {
    /// The Thorium group the image is created in
    pub group: String,
    /// The image name (also the on-disk JSON/dir name)
    pub name: String,
    /// The manifest `image_name`: the registry tag path leaf used at build time
    /// (`<registry>/<image_name>:<version>`); defaults to the build directory name/path
    pub image_name: String,
    /// The container image tag; empty means "fill in later"
    pub image_tag: String,
    /// The per-job timeout in seconds
    pub timeout: u64,
    /// The CPU request, in a unit Thorium accepts (e.g. "1000m", "2")
    pub cpu: String,
    /// The memory request, in a unit Thorium accepts (e.g. "1024Mi")
    pub memory: String,
    /// The scaler the image runs under (K8s, BareMetal, …)
    pub scaler: String,
    /// How the tool's results are displayed in the UI
    pub display_type: String,
    /// Whether this image generates child jobs
    pub generator: bool,
    /// Skip building this image in CI/CD; maps to `build = false` in the manifest
    pub no_build: bool,
    /// An optional description, seeded into the config and description.md
    pub description: Option<String>,
}

impl ImageConfigAnswers {
    /// Builds the default image config answers for a scaffolded image
    ///
    /// # Arguments
    ///
    /// * `name` - The image name
    /// * `group` - The Thorium group the image is created in
    /// * `no_build` - Whether to mark the image as not built by CI
    /// * `image_name` - The manifest `image_name` (registry tag path leaf)
    pub fn defaults(name: &str, group: &str, no_build: bool, image_name: &str) -> Self {
        // seed every field with the documented scaffold defaults; `image_tag` is left blank
        // (filled in later by the author/build) and `description` is None so no stub text
        // is forced into the config
        Self {
            group: group.to_string(),
            name: name.to_string(),
            image_name: image_name.to_string(),
            image_tag: String::new(),
            timeout: 300,
            cpu: "1000m".to_string(),
            memory: "1024Mi".to_string(),
            scaler: "K8s".to_string(),
            display_type: "Json".to_string(),
            generator: false,
            no_build,
            description: None,
        }
    }
}

/// The default-config seed used to scaffold a pipeline's config + manifest
pub struct PipelineConfigAnswers {
    /// The Thorium group the pipeline is created in
    pub group: String,
    /// The pipeline name (also the on-disk JSON/dir name)
    pub name: String,
    /// The execution order as parallel stages; defaults to all images in one stage
    pub order: Vec<Vec<String>>,
    /// The SLA in seconds (defaults to one week, matching the API default)
    pub sla: u64,
    /// An optional description, seeded into the config and description.md
    pub description: Option<String>,
}

impl PipelineConfigAnswers {
    /// Builds the default pipeline config answers for a scaffolded pipeline
    ///
    /// # Arguments
    ///
    /// * `name` - The pipeline name
    /// * `group` - The Thorium group the pipeline is created in
    /// * `images` - The images to run, placed in a single default stage
    pub fn defaults(name: &str, group: &str, images: &[String]) -> Self {
        // wrap all images in a single inner vec so the default order is one parallel stage
        // (every image runs concurrently); the author edits this to introduce sequencing
        Self {
            group: group.to_string(),
            name: name.to_string(),
            order: vec![images.to_vec()],
            // one week in seconds, chosen to match the API's own default SLA
            sla: 604_800,
            description: None,
        }
    }
}

/// The resolved answers used to write a toolbox's `config.toml`
pub struct ToolboxConfigAnswers {
    /// The human-readable toolbox name
    pub name: String,
    /// The primary container registry images are tagged/pushed under, or `None` when
    /// the user left it blank (the toolbox declares no central registry)
    pub registry: Option<String>,
}

// ─── Toolbox Wizard ──────────────────────────────────────────────────────────

/// Prompts for the toolbox-level config (name and primary registry)
///
/// The registry is optional: a blank entry resolves to `None`, leaving the toolbox
/// without a central registry so each image's own `image` url is used instead.
///
/// # Arguments
///
/// * `default_name` - The pre-filled toolbox name
/// * `default_registry` - The pre-filled container registry, if any
pub fn prompt_toolbox_config(
    default_name: &str,
    default_registry: Option<&str>,
) -> Result<ToolboxConfigAnswers, Error> {
    // print a colored section header so the toolbox-level questions stand apart from the
    // surrounding init output
    println!(
        "\n{}\n{}",
        "Toolbox Configuration".bright_green().bold(),
        "─".repeat(30).bright_green(),
    );
    // ask for the toolbox name, falling back to the pre-filled default on an empty entry
    let name = prompt_input("Toolbox name", default_name)?;
    // ask for the registry, defaulting the prompt to the supplied registry or "" when none
    let registry = prompt_input(
        "Container registry (optional)",
        default_registry.unwrap_or(""),
    )?;
    // collapse an empty registry (default or entered) to None so the toolbox declares no
    // central registry and each image's own `image` url is used instead
    let registry = if registry.is_empty() {
        None
    } else {
        Some(registry)
    };
    Ok(ToolboxConfigAnswers { name, registry })
}

#[cfg(test)]
mod tests {
    use super::{GROUP_NAME_MAX, RESOURCE_NAME_MAX, validate_name};

    /// A name made of lowercase letters, digits, and '-' (starting with a letter or digit) is valid
    #[test]
    fn validate_name_accepts_thorium_charset() {
        for name in ["clamav", "detect-it-easy", "av1", "0day"] {
            assert!(
                validate_name(name, RESOURCE_NAME_MAX).is_ok(),
                "'{name}' should be valid"
            );
        }
    }

    /// Uppercase, dots, underscores, a leading '-', and empty all violate the API name rule
    #[test]
    fn validate_name_rejects_non_thorium_charset() {
        for name in ["ClamAV", "clam.av", "under_score", "-lead", ""] {
            assert!(
                validate_name(name, RESOURCE_NAME_MAX).is_err(),
                "'{name}' should be rejected"
            );
        }
    }

    /// The length cap is enforced and is per-kind: 25 for resources, 50 for groups
    #[test]
    fn validate_name_enforces_length_cap() {
        // a 26-char name exceeds the resource cap but fits the group cap
        let name = "a".repeat(26);
        assert!(validate_name(&name, RESOURCE_NAME_MAX).is_err());
        assert!(validate_name(&name, GROUP_NAME_MAX).is_ok());
        // exactly at the resource cap is allowed; one past the group cap is not
        assert!(validate_name(&"a".repeat(RESOURCE_NAME_MAX), RESOURCE_NAME_MAX).is_ok());
        assert!(validate_name(&"a".repeat(GROUP_NAME_MAX + 1), GROUP_NAME_MAX).is_err());
    }
}
