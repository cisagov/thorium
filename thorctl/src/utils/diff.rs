//! Shared code for calculating diffs for updating models in Thorium
//!
//! Used in the `edit` and `toolbox update` commands

pub mod images;

/// Return the new field if it was modified
#[macro_export]
macro_rules! set_modified {
    ($old_field:expr, $new_field:expr) => {
        ($old_field != $new_field).then_some($new_field)
    };
}

/// Set the optional field if it was modified
#[macro_export]
macro_rules! set_modified_opt {
    ($old_field:expr, $new_field:expr) => {
        ($old_field != $new_field).then_some($new_field).flatten()
    };
}

/// Set the optional field if the new one is Some and modified compared to
/// the old, non-optional field
#[macro_export]
macro_rules! set_modified_new_opt {
    ($old_field:expr, $new_field:expr) => {
        $new_field.filter(|new| new != &$old_field)
    };
}

/// Clear the field if it was some in the before and is set to none now
#[macro_export]
macro_rules! set_clear {
    ($old_field:expr, $new_field:expr) => {
        $old_field.is_some() && $new_field.is_none()
    };
}

/// Clear this field if it was not empty before but it is set to empty now
#[macro_export]
macro_rules! set_clear_vec {
    ($old_field:expr, $new_field:expr) => {
        !$old_field.is_empty() && $new_field.is_empty()
    };
}

/// Calculate the values to remove/add based on what's in the
/// old vec and the new vec
///
/// Returns collections of values to remove/add in a tuple: `(remove, add)`
///
/// We remove everything that's in the old but not in the new and
/// add everything that's in the new but not in the old. We're okay
/// to mutate `old` with `extract_if` before calculating what we add because
/// `new` will contain all the values we want to keep; nothing we
/// extract from `old` in the first step will affect the calculation of
/// what we want to add.
///
/// # Limitations
///
/// - Cannot add/remove duplicate values
#[macro_export]
macro_rules! calc_remove_add_vec {
    ($old:expr, $new:expr) => {
        (
            $old.extract_if(.., |old| !$new.contains(old)).collect(),
            $new.extract_if(.., |new| !$old.contains(new)).collect(),
        )
    };
    // do the same thing, but add map functions to return the type we need
    ($old:expr, $remove_map_fn:expr, $new:expr, $add_map_fn:expr) => {
        (
            $old.extract_if(.., |old| !$new.contains(old))
                .map($remove_map_fn)
                .collect(),
            $new.extract_if(.., |new| !$old.contains(new))
                .map($add_map_fn)
                .collect(),
        )
    };
}

/// Calculate the values to remove/add based on what's in the
/// old map and the new map
///
/// Returns a vec of values to remove and a map of values to add
/// in a tuple: `(remove, add)`
///
/// We remove everything that's in the old but not in the new and
/// add everything that's in the new but not in the old. We're okay
/// to mutate `old` with `extract_if` before calculating what we add because
/// `new` will contain all the values we want to keep; nothing we
/// extract from `old` in the first step will affect the calculation of
/// what we want to add.
///
/// # Limitations
///
/// - Cannot add/remove duplicate values
#[macro_export]
macro_rules! calc_remove_add_map {
    ($old:expr, $new:expr) => {
        (
            // calculate the list of keys to remove
            $old.iter()
                .filter_map(|(key, value)| {
                    // check if either the value was modified or the key is missing
                    match $new.get(key) {
                        Some(new_value) => value != new_value,
                        None => true,
                    }
                    // return the key to remove it
                    .then_some(key)
                })
                .cloned()
                .collect(),
            // calculate the map to add
            $new
                // check if either the value was modified or the key is missing
                .extract_if(|key, value| match $old.get(key) {
                    Some(old_value) => value != old_value,
                    None => true,
                })
                .collect(),
        )
    };
}
