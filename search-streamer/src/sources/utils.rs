//! Utility functions shared between data sources

use bytesize::ByteSize;
use tracing::{Level, event, instrument};
use unicode_segmentation::UnicodeSegmentation;

/// The string to add at the end of a truncated string
const TRUNCATE_SUFFIX: &str = "...";

/// Calculate the number of bytes in a slice of Strings by summing
/// up their lengths
///
/// # Arguments
///
/// * `slice` - The slice of strings to calculate the size of
fn num_bytes_in_string_slice(slice: &[String]) -> u64 {
    slice.iter().map(String::len).sum::<usize>() as u64
}

/// Truncate strings in a vec proportionally based on the amount
/// of space each item takes up
///
/// # Arguments
///
/// * `items` - The items to truncate
/// * `max_size` - The maximum size of the entire vec to truncate to
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[instrument(name = "utils::proportional_tuncate", skip_all)]
pub fn proportional_truncate(items: &mut [String], max_len: u64) {
    // calculate the total size in bytes
    let total_size = num_bytes_in_string_slice(items);
    // determine if we need to truncate these results
    if total_size > max_len {
        // calculate the amount we need to truncate by in total
        let target_decrease = total_size - max_len;
        // calculate proportional truncation for each string
        for item in items.iter_mut() {
            let size = item.len() as u64;
            // calculate the proportion this result is taking up
            let proportion_used = size as f64 / total_size as f64;
            // get the amount to truncate by based on the proportion, rounded up so we're
            // sure we'll truncate enough + 3 bytes for an ellipsis
            let mut truncation_amt = (target_decrease as f64 * proportion_used).ceil() as u64;
            // add the truncate suffix if we're truncating to at least its length
            let truncated = if truncation_amt > TRUNCATE_SUFFIX.len() as u64 {
                truncation_amt += TRUNCATE_SUFFIX.len() as u64;
                let new_len = size - truncation_amt;
                // ensure truncation respects grapheme cluster boundaries
                let mut truncated = truncate_graphemes(item, new_len);
                // add the ellipsis
                truncated.push_str(TRUNCATE_SUFFIX);
                truncated
            } else {
                let new_len = size - truncation_amt;
                // ensure truncation respects grapheme cluster boundaries
                truncate_graphemes(item, new_len)
            };
            *item = truncated;
        }
        // calculate the new size
        let new_size = num_bytes_in_string_slice(items);
        // log that we truncated results
        let old_size = ByteSize::b(total_size);
        let new_size = ByteSize::b(new_size);
        event!(
            Level::WARN,
            msg = "Truncated search items",
            original_size = old_size.to_string(),
            new_size = new_size.to_string()
        );
    }
}

/// Take only the graphemes from the string that get to `new_len`
/// bytes or fewer
///
/// # Arguments
///
/// * `s` - The string to truncate
/// * `new_len` - The new length to truncate to or less
#[allow(clippy::needless_pass_by_value)]
fn truncate_graphemes(s: &str, new_len: u64) -> String {
    let mut truncated = String::new();
    let mut accumulated_bytes = 0;
    for grapheme in s.graphemes(true) {
        let grapheme_bytes = grapheme.len() as u64;
        if accumulated_bytes + grapheme_bytes > new_len {
            // return our truncated string
            return truncated;
        }
        truncated.push_str(grapheme);
        accumulated_bytes += grapheme_bytes;
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, distr::Alphanumeric};

    #[test]
    fn test_no_truncation_needed() {
        let mut items = vec![
            String::from("short"),
            String::from("medium"),
            String::from("longer"),
        ];
        // no truncation needed
        let max_len = items.iter().map(String::len).sum::<usize>() as u64;
        proportional_truncate(&mut items, max_len);
        assert_eq!(items, vec!["short", "medium", "longer"]);
    }

    #[test]
    fn test_truncation_needed() {
        let mut items = vec![
            String::from("short"),
            String::from("medium"),
            String::from("longer"),
        ];
        // total length must be truncated to 10
        let max_len = 10;
        proportional_truncate(&mut items, max_len);
        assert!(num_bytes_in_string_slice(&items) <= max_len);
    }

    #[test]
    fn test_single_item_truncation() {
        let mut items = vec![String::from("verylongstring")];
        // total length must be truncated to 8
        let max_len = 8;
        proportional_truncate(&mut items, max_len);
        assert!(num_bytes_in_string_slice(&items) <= max_len);
    }

    #[test]
    fn test_empty_vector() {
        let mut items: Vec<String> = vec![];
        let max_len = 10;
        proportional_truncate(&mut items, max_len);
        assert!(items.is_empty());
    }

    #[test]
    fn test_unicode_grapheme_clusters() {
        let mut items = vec![String::from("😊😊😊😊😊"), String::from("hello")];
        // total length must be truncated to 10
        let max_len = 10;
        proportional_truncate(&mut items, max_len);
        assert!(num_bytes_in_string_slice(&items) <= max_len);
    }

    #[test]
    fn test_large_random_string() {
        // generate a random string with at least 100 bytes
        let random_string: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(100)
            .map(char::from)
            .collect();
        let mut items = vec![random_string.clone(), random_string.clone()];
        // total length must be truncated to 50
        let max_len = 50;
        proportional_truncate(&mut items, max_len);
        // ensure the strings are truncated and ends with "..."
        assert!(num_bytes_in_string_slice(&items) <= max_len);
        assert!(items[0].ends_with(TRUNCATE_SUFFIX));
        assert!(items[1].ends_with(TRUNCATE_SUFFIX));
        // ensure the truncated strings are shorter than the original
        assert!(items[0].len() < random_string.len());
        assert!(items[1].len() < random_string.len());
    }
}
