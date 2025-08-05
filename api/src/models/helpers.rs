//! Helper macros/functions used in the Thorium API

/// return false if left and right are not equal
#[doc(hidden)]
#[macro_export]
macro_rules! same {
    ($left:expr, $right:expr) => {
        if $left != $right {
            return false;
        }
    };
    ($left:expr, $right:expr, $translator:expr) => {
        // try to convert this value first
        if let Ok(converted) = $translator($right) {
            if $left != converted {
                return false;
            }
        } else {
            return false;
        }
    };
}

/// return false if left and right are not equal
#[doc(hidden)]
#[macro_export]
macro_rules! matches_opt {
    ($left:expr, $right:expr) => {
        match ($left, $right) {
            (&Some(ref left), &Some(ref right)) if left != right => return false,
            _ => (),
        }
    };
}

/// return false if left and right sets are not equal
#[doc(hidden)]
#[macro_export]
macro_rules! matches_set {
    ($left:expr, $right:expr) => {
        if !$left.len() == $right.len() && !$left.is_superset(&$right) {
            return false;
        }
    };
}

/// check if two vectors match
pub fn matches_vecs_helper<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    // make sure a and b have the same length
    if a.len() != b.len() {
        return false;
    }

    // make sure that B contains all elements in A
    if !a.iter().all(|x| b.contains(x)) || !b.iter().all(|x| a.contains(x)) {
        return false;
    }
    true
}

/// return false if two vectors do no match
#[doc(hidden)]
#[macro_export]
macro_rules! matches_vec {
    ($left:expr, $right:expr) => {
        if !$crate::models::helpers::matches_vecs_helper(&$left, &$right) {
            println!("FAILED CHECK -> {:#?} == {:#?}", $left, $right);
            return false;
        }
    };
}

/// return false if the right is set and the left hand side does not match
#[doc(hidden)]
#[macro_export]
macro_rules! matches_update {
    ($left:expr, $right:expr) => {
        if let Some(right) = $right.as_ref() {
            if right != &$left {
                return false;
            }
        }
    };
    ($left:expr, $right:expr, $translator:expr) => {
        // only check if the right side was set
        if let Some(right) = $right.as_ref() {
            // try to convert this value first
            if let Ok(converted) = $translator(right) {
                if $left != converted {
                    return false;
                }
            } else {
                return false;
            }
        }
    };
}

/// return false if the update is set and the value doesn't match or is None
#[doc(hidden)]
#[macro_export]
macro_rules! matches_update_opt {
    ($val:expr, $update:expr) => {
        match (&$val, &$update) {
            (Some(val), Some(update)) => {
                if val != update {
                    return false;
                }
            }
            (None, Some(_)) => return false,
            _ => (),
        }
    };
}

/// Checks if a field was cleared as requested in an update
#[doc(hidden)]
#[macro_export]
macro_rules! matches_clear {
    ($val:expr, $clear:expr) => {
        if $clear && $val.is_some() {
            return false;
        }
    };
}

/// If clear flag is true, checks that the field is set to
/// Some empty vec
#[doc(hidden)]
#[macro_export]
macro_rules! matches_clear_vec_opt {
    ($val:expr, $clear:expr) => {
        match (&$val, $clear) {
            (Some(vec), true) => {
                // if set to clear, make sure the vec is empty
                if !vec.is_empty() {
                    return false;
                }
            }
            (None, true) => {
                // if set to clear and the vec is None, return false
                return false;
            }
            _ => (),
        }
    };
}

/// If clear flag is true, checks that the field is cleared,
/// otherwise checks that the fields match
#[doc(hidden)]
#[macro_export]
macro_rules! matches_clear_opt {
    ($val:expr, $update:expr, $clear:expr) => {
        matches_clear!($val, $clear);
        if !$clear {
            matches_update_opt!($val, $update);
        }
    };
}

/// returns false if the values in the right were not added to the left
#[doc(hidden)]
#[macro_export]
macro_rules! matches_adds {
    ($left:expr, $right:expr) => {
        if !$right.iter().all(|val| $left.contains(val)) {
            return false;
        }
    };
}

/// returns false if the values in the right iterator were not added to the left iterator
#[doc(hidden)]
#[macro_export]
macro_rules! matches_adds_iter {
    ($left:expr, $right:expr) => {
        if !$right.all(|right_val| $left.any(|left_val| left_val == right_val)) {
            return false;
        }
    };
}

/// returns false if the values in the right were not removed from the left
#[doc(hidden)]
#[macro_export]
macro_rules! matches_removes {
    ($left:expr, $right:expr) => {
        if $right.iter().any(|val| $left.contains(val)) {
            return false;
        }
    };
}

/// returns false if the values in the right iterator were not removed from the left iterator
#[doc(hidden)]
#[macro_export]
macro_rules! matches_removes_iter {
    ($left:expr, $right:expr) => {
        if $right.any(|right_val| $left.any(|left_val| left_val == right_val)) {
            return false;
        }
    };
}

/// returns false if the keys in the right were not removed from the left map
#[doc(hidden)]
#[macro_export]
macro_rules! matches_removes_map {
    ($map:expr, $removes:expr) => {
        if $removes.iter().any(|key| $map.contains_key(key)) {
            return false;
        }
    };
}

/// returns false if the (keys, values) in the right iterator were not added to the left map
#[doc(hidden)]
#[macro_export]
macro_rules! matches_adds_map {
    ($map:expr, $keys_values:expr) => {
        if $keys_values.any(|(key, value)| {
            if let Some(map_val) = $map.get(key) {
                map_val != value
            } else {
                false
            }
        }) {
            return false;
        }
    };
}

/// returns false if the values in the optional right were not added to the left
#[doc(hidden)]
#[macro_export]
macro_rules! matches_adds_opt {
    ($left:expr, $right:expr) => {
        if let Some(right) = $right.as_ref() {
            if !right.iter().all(|val| $left.contains(val)) {
                return false;
            }
        }
    };
}

/// returns false if the values in the optional right were not added to the left
#[doc(hidden)]
#[macro_export]
macro_rules! matches_removes_opt {
    ($left:expr, $right:expr) => {
        if let Some(right) = $right.as_ref() {
            if right.iter().any(|val| $left.contains(val)) {
                return false;
            }
        }
    };
}

/// return false if a cpu conversion fails
#[doc(hidden)]
#[macro_export]
macro_rules! cpu {
    ($raw:expr) => {
        match $crate::models::conversions::cpu($raw) {
            Ok(converted) => converted,
            Err(_) => return false,
        }
    };
}

/// return false if a cpu conversion fails
#[doc(hidden)]
#[macro_export]
macro_rules! cpu_opt {
    ($raw:expr) => {
        if let Some(raw) = $raw {
            match $crate::models::conversions::cpu(raw) {
                Ok(converted) => Some(converted),
                Err(_) => return false,
            }
        } else {
            None
        }
    };
}

/// return false if a optional storage conversion fails
#[doc(hidden)]
#[macro_export]
macro_rules! storage {
    ($raw:expr) => {
        match $crate::models::conversions::storage($raw) {
            Ok(converted) => converted,
            Err(_) => return false,
        }
    };
}

/// return false if a optional storage conversion fails
#[doc(hidden)]
#[macro_export]
macro_rules! storage_opt {
    ($raw:expr) => {
        if let Some(raw) = $raw {
            match $crate::models::conversions::storage(raw) {
                Ok(converted) => Some(converted),
                Err(_) => return false,
            }
        } else {
            None
        }
    };
}

/// Create a HashSet with values
#[doc(hidden)]
#[macro_export]
macro_rules! set {
    ( $( $x:expr ),* ) => {
        {
            // create empty mutable set
            let mut set = std::collections::HashSet::new();
            // insert all passed in values into set
            $(
                set.insert($x);
            )*
            set
        }
    };
}

/// Returns the right value if its greater than the left
#[doc(hidden)]
#[macro_export]
macro_rules! at_least {
    ($left:expr, $right:expr) => {
        if ($left < $right) {
            $right
        } else {
            $left
        }
    };
}
