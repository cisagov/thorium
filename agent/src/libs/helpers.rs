//! Helper functions/macros for the agent

use walkdir::DirEntry;

/// Serializes data to a string
#[doc(hidden)]
#[macro_export]
macro_rules! serialize {
    ($data:expr) => {
        match serde_json::to_string($data) {
            Ok(serial) => Ok(serial),
            Err(e) => Err(thorium::Error::new(&format!(
                "Failed to serialize data with error {}",
                e
            ))),
        }
    };
}

/// Deserializes data from a string
#[doc(hidden)]
#[macro_export]
macro_rules! deserialize {
    ($data:expr) => {
        match serde_json::from_str($data) {
            Ok(serial) => serial,
            Err(e) => {
                return Err(thorium::Error::new(format!(
                    "Failed to deserialize data {} with error {}",
                    $data, e
                )))
            }
        }
    };
}

/// Try to complete an action for a job in a best effort fashion
#[doc(hidden)]
#[macro_export]
macro_rules! best_effort {
    ($info:expr, $id:expr, $future:expr, $attempts:expr) => {
        for i in 0..$attempts {
            match $future.await {
                Ok(_) => break,
                Err(err) => {
                    // build logs to send for this error
                    let mut err_logs = thorium::models::StageLogsAdd::default();
                    err_logs.add_logs(vec![format!("THORIUM_ERROR: {:#?}", err)]);
                    // try to send error logs for this action
                    if let Err(log_err) = $info
                        .client
                        .reactions
                        .add_stage_logs(&$info.target.group, &$id, &$info.target.stage, &err_logs)
                        .await
                    {
                        println!("FAILED TO SEND ERROR logs -> {:#?}", log_err);
                    };
                    // if this is the last attempt then bubble this error up
                    if i == $attempts {
                        return Err(err)?;
                    }
                }
            }
        }
    };
    ($info:expr, $id:expr, $action:expr) => {
        best_effort!($info, $id, $action, 3)
    };
}

/// Try to complete an action for a job in a best effort fashion
#[doc(hidden)]
#[macro_export]
macro_rules! best_effort_panic {
    ($info:expr, $id:expr, $action:expr, $attempts:expr) => {
        for i in 0..$attempts {
            match $action.await {
                Ok(_) => break,
                Err(err) => {
                    // build logs to send for this error
                    let mut err_logs = thorium::models::StageLogsAdd::default();
                    err_logs.add_logs(vec![format!("THORIUM_ERROR: {:#?}", err)]);
                    // try to send error logs for this action
                    if let Err(log_err) = $info
                        .client
                        .reactions
                        .add_stage_logs(&$info.target.group, &$id, &$info.target.stage, &err_logs)
                        .await
                    {
                        println!("FAILED TO SEND ERROR logs -> {:#?}", log_err);
                    };
                    // if this is the last attempt then bubble this error up
                    if i == $attempts {
                        panic!("FATAL_ERROR: {:#?}", err)
                    }
                }
            }
        }
    };
    ($info:expr, $id:expr, $action:expr) => {
        best_effort_panic!($info, $id, $action, 3)
    };
}

/// add this log
#[doc(hidden)]
#[macro_export]
macro_rules! log {
    ($logs:expr, $msg:expr) => {
        crate::check!($logs.send($msg.to_string()))
    };
    ($logs:expr, $format:expr, $($args:expr),+) => {
        crate::check!($logs.send(format!($format, $($args),+)))
    };
}

/// Add logs and then return an error
#[doc(hidden)]
#[macro_export]
macro_rules! fail {
    ($logs:expr, $msg:expr) => {{
        crate::check!($logs.send($msg.to_string()));
        return Err(thorium::Error::new($msg));
    }};
}

/// log a fail if it one was detected
#[doc(hidden)]
#[macro_export]
macro_rules! log_fail {
    ($logs:expr, $format:expr, $result:expr) => {{
        match $result {
            Ok(val) => Ok(val),
            Err(err) => {
                $logs.add(format!($format, err));
                Err(err)
            }
        }
    }};
}

/// Checks if an entry is a hidden file or not
///
/// # Arguments
///
/// * `entry` - The entry to check
pub fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with("."))
        .unwrap_or(false)
}

/// Checks if an entry is a file or not
///
/// # Arguments
///
/// * `entry` - The entry to check
pub fn is_file(entry: &DirEntry) -> bool {
    if let Ok(metadata) = entry.metadata() {
        metadata.is_file() && metadata.len() > 0
    } else {
        false
    }
}

/// purge a directory if its a file or directory
#[doc(hidden)]
#[macro_export]
macro_rules! purge {
    ($target:expr) => {
        // build the path to remove if it exists
        let path = std::path::Path::new(&$target);
        // check if this path exists
        if path.exists() {
            // check if this is a file so we can delete it
            if path.is_file() {
                std::fs::remove_file(path)?;
            } else if path.is_dir() {
                std::fs::remove_dir_all(path)?;
            }
        }
    };
}

/// purge a directory if its a file or directory
#[doc(hidden)]
#[macro_export]
macro_rules! purge_parent {
    ($target:expr) => {
        // build the path to remove if it exists
        let path = std::path::Path::new(&$target);
        let parent = path.parent().unwrap();
        // if our parent is just /tmp/thorium then remove the full target path instead
        let target = if parent == Path::new("/tmp/thorium") {
            path
        } else {
            parent
        };
        // check if this path exists
        if target.exists() {
            // check if this is a file so we can delete it
            if target.is_file() {
                println!("PURGING PAR FILE -> {:#?}", target);
                std::fs::remove_file(target)?;
            } else if target.is_dir() {
                println!("PURGING PAR DIR -> {:#?}", target);
                std::fs::remove_dir_all(target)?;
            }
        }
    };
}
