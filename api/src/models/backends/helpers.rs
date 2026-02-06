//! Helpers for the backends level of the Thorium API

use uuid::Uuid;

use crate::bad;
use crate::utils::ApiError;

/// The max length of most names in K8's
const K8_NAME_MAX_CHARS: usize = 63;

/// Convert a UTF-8 encoded Rust str with possible special characters to a valid
/// name in K8's (<253 characters, lowercase alphanumeric or "-", starts and ends
/// with alphanumeric); see <https://kubernetes.io/docs/concepts/overview/working-with-objects/names/>
///
/// Appends the given UUID to ensure the output name is unique even if another
/// UTF-8 encoding str has the same pattern of valid/invalid characters
/// (i.e. `my-name😄` and `my-name😎` both yield `my-name-`, so a UUID is required
/// to distinguish them)
///
/// # Errors
///
/// Returns an Error if the str value is empty
///
/// # Arguments
///
/// * `value` - The str value to convert to a K8's name
/// * `id` - The UUID to append to the K8's name
pub fn to_k8s_name<T: AsRef<str>>(value: T, id: Uuid) -> Result<String, ApiError> {
    // cast the value to a str
    let value = value.as_ref();
    // check that the value isn't empty
    if value.is_empty() {
        return bad!("Name cannot be empty!".to_string());
    }
    let id = id.to_string();
    // truncate to the maximum number of K8's name characters
    // with space left over for the UUID and a hyphen separator
    let truncated = value
        .chars()
        .take(K8_NAME_MAX_CHARS - id.len() - 1)
        .collect::<String>();
    // convert to lowercase
    let lower = truncated.to_lowercase();
    // replace non-ascii and non-alphanumeric/non-"-" to "-"
    let replaced = lower.replace(
        |c: char| !c.is_ascii() || (!c.is_alphanumeric() && c != '-'),
        "-",
    );
    let mut chars = replaced.chars();
    let k8s_name = if let Some(first_char) = chars.next() {
        if first_char == '-' {
            // replace the first "-" with a "z"; k8's names must start with alphanumeric
            format!("z{}-{}", chars.collect::<String>(), id)
        } else {
            format!("{replaced}-{id}")
        }
    } else {
        format!("{replaced}-{id}")
    };
    Ok(k8s_name)
}

/// Checks to make sure that the next segment in the bracket segment list
/// is empty, otherwise returns an error.
///
/// This check is important for list fields that should end with `[]`
#[macro_export]
macro_rules! ensure_empty_segment {
    ($segments_iter:expr, $name:expr) => {{
        if !$segments_iter.next().is_some_and(str::is_empty) {
            bad!(format!(
                "Malformed multipart field name '{}': missing final '[]' on list",
                $name
            ))
        } else {
            Ok(())
        }
    }};
}

/// Checks to make sure that there are no more segments left in the given
/// iterator, otherwise returns an error
#[macro_export]
macro_rules! ensure_segments_complete {
    ($segments_iter:expr, $name:expr) => {{
        if $segments_iter.next().is_some() {
            bad!(format!(
                "Multipart field name '{}' has invalid extra components",
                $name
            ))
        } else {
            Ok(())
        }
    }};
}

/// Parses a string like "metadata[key][value]" into a `Vec` of `&str` segments.
///
/// Particularly useful when parsing multipart form field names
///
/// The first segment is everything before the first `[`.
/// Subsequent segments are the contents of each `[...]` bracket pair.
/// An empty input returns an empty `Vec`.
///
/// # Errors
///
/// - No matching closing bracket `]` after open bracket `[`
/// - Extra input after closing bracket `]`
pub fn parse_bracket_segments(input: &str) -> Result<Vec<&str>, ApiError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut segments = Vec::new();
    // split off the first segment (before the first `[`)
    let (first, rest) = match input.find('[') {
        Some(pos) => (&input[..pos], &input[pos..]),
        // no brackets at all, return the whole string
        None => return Ok(vec![input]),
    };
    segments.push(first);
    // parse each [...] bracket group from the remainder
    let mut remaining = rest;
    while !remaining.is_empty() {
        match remaining.find('[') {
            // no more opening brackets
            None => {
                return bad!(format!(
                    "malformed input '{input}': extra characters after brackets"
                ));
            }
            Some(open) => {
                match remaining[open..].find(']') {
                    // malformed: no closing bracket
                    None => {
                        return bad!(format!(
                            "malformed input '{input}': missing closing bracket"
                        ));
                    }
                    Some(close_offset) => {
                        // close_offset is relative to remaining[open..]
                        let close = open + close_offset;
                        let content = &remaining[open + 1..close];
                        segments.push(content);
                        remaining = &remaining[close + 1..];
                    }
                }
            }
        }
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    // k8s name tests

    fn is_valid_k8s_name(name: &str) -> bool {
        !name.is_empty()
            && name.is_ascii()
            && name.len() <= K8_NAME_MAX_CHARS
            && name.to_ascii_lowercase() == name
            && name.chars().next().unwrap().is_alphanumeric()
    }

    #[test]
    fn test_k8s_name_already_valid() {
        let name = "already-valid-name123";
        let id = Uuid::new_v4();
        let k8s_name = to_k8s_name(name, id).unwrap();
        assert!(k8s_name.starts_with("already-valid-name123"));
        assert!(is_valid_k8s_name(&k8s_name));
        assert!(k8s_name.ends_with(&format!("-{id}")));
    }

    #[test]
    fn test_k8s_name_too_long() {
        let long_name = "a".repeat(300);
        let k8s_name = to_k8s_name(long_name, Uuid::new_v4()).unwrap();
        assert_eq!(k8s_name.len(), K8_NAME_MAX_CHARS);
        assert!(is_valid_k8s_name(&k8s_name));
    }

    #[test]
    fn test_k8s_name_invalid_chars() {
        let name = "invalid!@#name";
        let k8s_name = to_k8s_name(name, Uuid::new_v4()).unwrap();
        assert!(k8s_name.starts_with("invalid---name"));
        assert!(is_valid_k8s_name(&k8s_name));
    }

    #[test]
    fn test_k8s_name_start_end_chars() {
        let name = "-start-and-end-";
        let k8s_name = to_k8s_name(name, Uuid::new_v4()).unwrap();
        assert!(k8s_name.starts_with("zstart-and-end"));
        assert!(is_valid_k8s_name(&k8s_name));
    }

    #[test]
    fn test_k8s_name_empty() {
        let name = "";
        assert!(to_k8s_name(name, Uuid::new_v4()).is_err());
    }

    #[test]
    fn test_k8s_name_utf8() {
        let name = "naмe-with-нон-ascii";
        let k8s_name = to_k8s_name(name, Uuid::new_v4()).unwrap();
        assert!(k8s_name.starts_with("na-e-with-----ascii"));
        assert!(is_valid_k8s_name(&k8s_name));
    }

    #[test]
    fn test_k8s_name_cap() {
        let name = "name-wItH-CAPS";
        let k8s_name = to_k8s_name(name, Uuid::new_v4()).unwrap();
        assert!(k8s_name.starts_with("name-with-caps"));
        assert!(is_valid_k8s_name(&k8s_name));
    }

    // bracket segments tests

    #[test]
    fn test_parse_brackets_empty() {
        assert_eq!(parse_bracket_segments("").unwrap(), Vec::<&str>::new());
    }

    #[test]
    fn test_parse_brackets_no_brackets() {
        assert_eq!(
            parse_bracket_segments("metadata").unwrap(),
            vec!["metadata"]
        );
    }

    #[test]
    fn test_parse_brackets_single_pair() {
        assert_eq!(
            parse_bracket_segments("metadata[key]").unwrap(),
            vec!["metadata", "key"]
        );
    }

    #[test]
    fn test_parse_brackets_two_pairs() {
        assert_eq!(
            parse_bracket_segments("metadata[key][value]").unwrap(),
            vec!["metadata", "key", "value"]
        );
    }

    #[test]
    fn test_parse_brackets_empty_pairs() {
        assert_eq!(
            parse_bracket_segments("metadata[][key][][value]").unwrap(),
            vec!["metadata", "", "key", "", "value"]
        );
    }

    #[test]
    fn test_parse_brackets_all_empty() {
        assert_eq!(
            parse_bracket_segments("field[][]").unwrap(),
            vec!["field", "", ""]
        );
    }

    #[test]
    fn test_parse_brackets_empty_first() {
        // Edge case: input starts with `[`
        assert_eq!(
            parse_bracket_segments("[key][value]").unwrap(),
            vec!["", "key", "value"]
        );
    }

    #[test]
    fn test_parse_brackets_single_empty() {
        assert_eq!(
            parse_bracket_segments("field[]").unwrap(),
            vec!["field", ""]
        );
    }

    #[test]
    fn test_parse_brackets_many() {
        assert_eq!(
            parse_bracket_segments("a[b][c][d][e]").unwrap(),
            vec!["a", "b", "c", "d", "e"]
        );
    }

    #[test]
    fn test_parse_brackets_characters_after_brackets() {
        let res = parse_bracket_segments("metadata[key]extra_bad_stuff");
        let err = res.expect_err("Should be an error");
        assert!(
            err.msg
                .is_some_and(|msg| msg.contains("extra characters after brackets"))
        );
    }

    #[test]
    fn test_parse_brackets_missing_closing_bracket() {
        let res = parse_bracket_segments("metadata[key][value");
        let err = res.expect_err("Should be an error");
        assert!(
            err.msg
                .is_some_and(|msg| msg.contains("missing closing bracket"))
        );
    }
}
