//! Renders a computed update as a short list of changed field names
//!
//! Used by `--skip-conflicts` (and skip warnings in general) so a skipped
//! resource tells the user exactly *what* would have changed, without dumping
//! the full config diff into the log.

use serde::Serialize;
use thorium::Error;

/// List the top-level fields of an update struct that carry a real change
///
/// Works by serializing the update to JSON and keeping the keys whose values
/// are "meaningful": update structs encode "no change" as `null`, `false`
/// (clear flags), empty arrays (add/remove lists), or objects containing only
/// those. This stays correct as fields are added to the update models without
/// this module needing to know about them.
///
/// # Arguments
///
/// * `update` - The computed update struct (e.g. `ImageUpdate`, `PipelineUpdate`)
pub fn changed_fields<T: Serialize>(update: &T) -> Result<Vec<String>, Error> {
    let value = serde_json::to_value(update)
        .map_err(|err| Error::new(format!("Failed to serialize update for summary: {err}")))?;
    let serde_json::Value::Object(map) = value else {
        return Err(Error::new("Update did not serialize to a JSON object"));
    };
    let mut fields: Vec<String> = map
        .into_iter()
        .filter(|(_, value)| is_meaningful(value))
        .map(|(key, _)| key)
        .collect();
    // sort so the output is stable regardless of serialization order
    fields.sort_unstable();
    Ok(fields)
}

/// Render the changed fields of an update as a single bracketed list
///
/// # Arguments
///
/// * `update` - The computed update struct to summarize
pub fn render_changed_fields<T: Serialize>(update: &T) -> String {
    match changed_fields(update) {
        Ok(fields) if fields.is_empty() => "[no field-level changes detected]".to_string(),
        Ok(fields) => format!("[{}]", fields.join(", ")),
        // a summary failure shouldn't break the import; degrade to a generic note
        Err(_) => "[unable to summarize changes]".to_string(),
    }
}

/// Whether a serialized update value represents an actual change
///
/// `null`, `false`, empty arrays/objects, and objects/arrays whose members are
/// all non-meaningful encode "no change" in the update models.
///
/// # Arguments
///
/// * `value` - The serialized update field value to test for meaning
fn is_meaningful(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(set) => *set,
        serde_json::Value::Number(_) | serde_json::Value::String(_) => true,
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(map) => map.values().any(is_meaningful),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thorium::models::{ImageUpdate, PipelineUpdate};

    /// A default update carries no changes at all
    #[test]
    fn empty_updates_have_no_changed_fields() {
        let image_update = ImageUpdate::default();
        assert!(changed_fields(&image_update).unwrap().is_empty());
        let pipeline_update = PipelineUpdate::default();
        assert!(changed_fields(&pipeline_update).unwrap().is_empty());
    }

    /// Simple scalar/option fields show up by name
    #[test]
    fn scalar_changes_are_listed() {
        let update = ImageUpdate::default().image("registry.local/new:latest");
        let fields = changed_fields(&update).unwrap();
        assert!(fields.contains(&"image".to_string()), "got {fields:?}");
    }

    /// Clear flags count as changes; unset clear flags do not
    #[test]
    fn clear_flags_are_meaningful() {
        let update = ImageUpdate {
            clear_description: true,
            ..Default::default()
        };
        let fields = changed_fields(&update).unwrap();
        assert!(
            fields.contains(&"clear_description".to_string()),
            "got {fields:?}"
        );
    }

    /// Sub-structs that contain only defaults are filtered out, while
    /// sub-structs carrying a change surface their top-level field name
    #[test]
    fn pipeline_order_changes_are_listed() {
        let update = PipelineUpdate {
            order: Some(serde_json::json!([["harvester"]])),
            ..Default::default()
        };
        let fields = changed_fields(&update).unwrap();
        assert!(fields.contains(&"order".to_string()), "got {fields:?}");
        // default ban update inside the struct must not leak in
        assert!(!fields.contains(&"bans".to_string()), "got {fields:?}");
    }

    /// Rendering produces the bracketed form used in skip warnings
    #[test]
    fn render_formats_brackets() {
        let update = ImageUpdate::default().image("x");
        let rendered = render_changed_fields(&update);
        assert!(rendered.starts_with('['), "got {rendered}");
        assert!(rendered.contains("image"), "got {rendered}");
    }
}
