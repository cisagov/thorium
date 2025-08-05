#[macro_export]
macro_rules! is {
    ($left:expr, $right:expr) => {
        if $left != $right {
            return Err(thorium::Error::new(format!(
                "Failed == check because the value {:#?} != {:#?}",
                $left, $right
            )));
        }
    };
    ($left:expr, $right:expr, $msg:expr) => {
        if $left != $right {
            return Err(thorium::Error::new(format!(
                "Failed == check because {:#?} != {:#?}: Check '{}'",
                $left, $right, $msg
            )));
        }
    };
}

#[macro_export]
macro_rules! is_not {
    ($left:expr, $right:expr) => {
        if $left == $right {
            return Err(thorium::Error::new(format!(
                "Failed != check {:#?} == {:#?}",
                $left, $right
            )));
        }
    };
}

#[macro_export]
macro_rules! is_in {
    ($list:expr, $item:expr) => {
        if !$list.iter().any(|x| *x == $item) {
            return Err(thorium::Error::new(format!(
                "Failed is_in check because {:#?} is not in {:#?}",
                $item, $list
            )));
        }
    };
}

#[macro_export]
macro_rules! contains {
    ($list:expr, $item:expr) => {
        if !$list.contains($item) {
            return Err(thorium::Error::new(format!(
                "Failed contains check because {:#?} is not in {:#?}",
                $item, $list
            )));
        }
    };
}

#[macro_export]
macro_rules! is_not_in {
    ($list:expr, $item:expr) => {
        if $list.contains(&$item) {
            return Err(thorium::Error::new(format!(
                "Failed \"does not contain\" check because {:#?} is in {:#?}",
                $item, $list
            )));
        }
    };
}

#[macro_export]
macro_rules! is_empty {
    ($val:expr) => {
        if !$val.is_empty() {
            return Err(thorium::Error::new(format!(
                "Failed is_empty check because it contains: {:#?}",
                $val
            )));
        }
    };
}

#[macro_export]
macro_rules! contains_key {
    ($list:expr, $item:expr) => {
        if !$list.contains_key($item) {
            return Err(thorium::Error::new(format!(
                "Failed contains check because {:#?} is not in {:#?}",
                $item, $list
            )));
        }
    };
}

#[macro_export]
macro_rules! starts_with {
    ($val:expr, $pattern:expr) => {
        if !$val.starts_with($pattern) {
            return Err(thorium::Error::new(format!(
                "Failed starts_with check because {:#?} does not start with {:#?}",
                $val, $pattern
            )));
        }
    };
}

/// Checks that all the values in the left iter are in the left iter;
/// fundamentally, the left and right iters may or may not be "equal"; they
/// may be sorted differently *and/or* the right iter may have elements the right
/// one does not
#[macro_export]
macro_rules! iter_in_iter {
    ($left:expr, $right:expr) => {
        if !$left.all(|l_item| $right.any(|r_item| l_item == r_item)) {
            return Err(thorium::Error::new(format!(
                "Failed == check {:#?} == {:#?}",
                $left, $right
            )));
        }
    };
    // print a helpful marker message to show where the error occurred
    ($left:expr, $right:expr, $msg:expr) => {
        if !$left.all(|l_item| $right.any(|r_item| l_item == r_item)) {
            return Err(thorium::Error::new(format!(
                "Failed == check {:#?} == {:#?}: Check '{}'",
                $left, $right, $msg
            )));
        }
    };
}

/// Checks that all the values in the left `Vec` are in the right `Vec`;
/// fundamentally, the left and right `Vec`s may or may not be "equal"; they
/// may be sorted differently *and/or* the right `Vec` may have elements the left
/// one does not
#[macro_export]
macro_rules! vec_in_vec {
    ($left:expr, $right:expr) => {
        if !$left
            .iter()
            .all(|l_item| $right.iter().any(|r_item| l_item == r_item))
        {
            return Err(thorium::Error::new(format!(
                "Failed == check; {:#?} not all in {:#?}",
                $left, $right
            )));
        }
    };
    // print a helpful marker message to show where the error occurred
    ($left:expr, $right:expr, $msg:expr) => {
        if !$left
            .iter()
            .all(|l_item| $right.iter().any(|r_item| l_item == r_item))
        {
            return Err(thorium::Error::new(format!(
                "Failed == check {:#?} not all in {:#?}: Check '{}'",
                $left, $right, $msg
            )));
        }
    };
}

/// makes sure a list of values is in descending order
#[macro_export]
macro_rules! is_desc {
    ($values:expr) => {{
        // get the first value of our list
        if let Some(last) = $values.get(0) {
            // get the timestamp for this value
            let mut last_ts = last.uploaded;
            for value in $values.iter() {
                // if our lst timestamp came after this one then error
                if value.uploaded > last_ts {
                    return Err(thorium::Error::new(format!(
                        "Failed Descending order check {:#?}",
                        $values
                    )));
                }
                // update our last timestamp to the new value
                last_ts = value.uploaded;
            }
        }
    }};
}

/// check if a tag exists
#[macro_export]
macro_rules! has_tag {
    // check if the given tags contain a value
    ($tags:expr, $key:expr, $val:expr) => {
        if let Some(val_map) = $tags.get($key) {
            if !val_map.contains_key($val) {
                return Err(thorium::Error::new(format!(
                    "Failed has_tag check because the tags for \
                        key {:#?} are missing the value {:#?}",
                    $key, $val
                )));
            }
        } else {
            return Err(thorium::Error::new(format!(
                "Failed has_tag check because tags are missing the key {:#?}",
                $key
            )));
        }
    };
    // check if the given tags contain a value and a group at that value
    ($tags:expr, $key:expr, $val:expr, $group:expr) => {
        if let Some(val_map) = $tags.get($key) {
            if let Some(groups) = val_map.get($val) {
                if !groups.contains($group) {
                    return Err(thorium::Error::new(format!(
                        "Failed has_tag check because the tags for \
                            key {:#?} and value {:#?} are missing the group {:#?}",
                        $key, $val, $group
                    )));
                }
            } else {
                return Err(thorium::Error::new(format!(
                    "Failed has_tag check because the tags for \
                        key {:#?} are missing the value {:#?}",
                    $key, $val
                )));
            }
        } else {
            return Err(thorium::Error::new(format!(
                "Failed has_tag check because tags are missing the key {:#?}",
                $key
            )));
        }
    };
}

/// check that a tag does not exist
#[macro_export]
macro_rules! no_tag {
    // check that the given tag does not exist at all
    ($tags:expr, $key:expr) => {
        if $tags.contains_key($key) {
            return Err(thorium::Error::new(format!(
                "Failed no_tag check because the tags for \
                    key {:#?} contain values",
                $key,
            )));
        }
    };
    // check that the given tags do not contain a value
    ($tags:expr, $key:expr, $val:expr) => {
        if let Some(val_map) = $tags.get($key) {
            if val_map.contains_key($val) {
                return Err(thorium::Error::new(format!(
                    "Failed no_tag check because the tags for \
                        key {:#?} contain the value {:#?}",
                    $key, $val
                )));
            }
        } else {
            return Err(thorium::Error::new(format!(
                "Failed no_tag check because tags are missing the key {:#?}",
                $key
            )));
        }
    };
    // check that the given tags do not contain a certain group at a value
    // will return an error if the value does not exist at all
    ($tags:expr, $key:expr, $val:expr, $group:expr) => {
        if let Some(val_map) = $tags.get($key) {
            if let Some(groups) = val_map.get($val) {
                if groups.contains($group) {
                    return Err(thorium::Error::new(format!(
                        "Failed no_tag check because the tags for \
                            key {:#?} and value {:#?} contain the group {:#?}",
                        $key, $val, $group
                    )));
                }
            } else {
                return Err(thorium::Error::new(format!(
                    "Failed no_tag check for group {:#?} because the tags for \
                        key {:#?} are missing the value {:#?}, so no groups exist",
                    $group, $key, $val
                )));
            }
        } else {
            return Err(thorium::Error::new(format!(
                "Failed no_tag check because tags are missing the key {:#?}",
                $key
            )));
        }
    };
}

#[macro_export]
macro_rules! fail {
    ($check:expr, $code:expr) => {
        match &$check {
            Ok(_) => {
                return Err(thorium::Error::new(format!(
                    "Failed failure check with Ok status {:#?}",
                    $check
                )));
            }
            Err(e) => {
                // make sure a status code was returned
                match e.status() {
                    Some(status) => {
                        let status = status.as_u16();
                        if status != $code {
                            return Err(thorium::Error::new(format!(
                                "Failed failure status code check {:#?} != {:#?}. Full error: {}",
                                status, $code, e
                            )));
                        }
                    }
                    None => {
                        return Err(thorium::Error::new(format!(
                            "Failed failure check with no status returned. Full response: {:#?}",
                            $check
                        )))
                    }
                }
            }
        }
    };
    // check if the error has a specific code AND message
    ($check:expr, $code:expr, $msg:expr) => {
        match &$check {
            Ok(_) => {
                return Err(thorium::Error::new(format!(
                    "Failed error message check with Ok status: {:#?}",
                    $check
                )));
            }
            Err(e) => {
                // make sure a status code was returned
                match e.status() {
                    Some(status) => {
                        let status = status.as_u16();
                        if status != $code {
                            return Err(thorium::Error::new(format!(
                                "Failed failure status code check {:#?} != {:#?}. Full error: {}",
                                status, $code, e
                            )));
                        }
                    }
                    None => {
                        return Err(thorium::Error::new(format!(
                            "Failed failure check with no status returned. Full response: {:#?}",
                            $check
                        )))
                    }
                }
                match e.msg() {
                    Some(check_msg) => {
                        if !&check_msg.contains($msg) {
                            return Err(thorium::Error::new(format!(
                                "Failed error message check: {} does not contain '{}'.",
                                &check_msg, $msg
                            )));
                        }
                    }
                    None => {
                        return Err(thorium::Error::new(
                            "Failed error message check with no message returned".to_owned(),
                        ))
                    }
                }
            }
        }
    };
}

/// unwraps a variant from an enum or returns an error if the enum is not of that variant
#[macro_export]
macro_rules! unwrap_variant {
    ($check:expr, $variant:path) => {
        if let $variant(val) = $check {
            // unwrap the enum variant
            val
        } else {
            // return an error if the enum is not of that variant
            return Err(thorium::Error::new(format!(
                "Failed to unwrap variant '{}' from enum. Enum is actually '{:?}''",
                stringify!($variant),
                $check
            )));
        }
    };
}
