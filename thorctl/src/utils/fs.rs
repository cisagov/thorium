//! Utility functions relating to file system interactions

use async_walkdir::{DirEntry, Filtering, WalkDir};
use futures::stream::StreamExt;
use regex::{Regex, RegexSet};
use std::{
    collections::HashSet,
    io,
    path::{Path, PathBuf},
    sync::LazyLock,
};
use thorium::Error;

/// Processes files in each of the given targets with the given process function,
/// filtering out targets whose file names don't match given filters (or *do* match
/// given skip filters).
///
/// Target directories are walked recursively and processed concurrently. The
/// targets themselves are not processed concurrently because we can't be sure if
/// a target is a child of another target, and their processing could affect each
/// other's directory walks.
///
/// If an IO error occurs while traversing the targets, the given error function
/// is called to log the error.
///
/// # Arguments
///
/// * `targets_iter` - An iterator of targets to process
/// * `process_fn` - The asynchronous function used to process each path in a target
/// * `err_fn` - The error function to call if an error occurs
/// * `filter` - The regex filters set by the user to determine which files to include
/// * `skip` - The regex skip filters set by the user to determine which files to skip
/// * `include_hidden` - Whether we should include hidden files/directories in the walk
/// * `concurrency` - The number of paths to process concurrently at maximum
pub async fn process_async_walk<I, F, Fut, E>(
    targets_iter: I,
    procces_fn: F,
    err_fn: E,
    filter: &RegexSet,
    skip: &RegexSet,
    include_hidden: bool,
    concurrency: usize,
) where
    I: Iterator<Item = PathBuf>,
    F: Fn(PathBuf) -> Fut + Send,
    Fut: Future<Output = ()> + Send,
    E: for<'a, 'b> Fn(&'a Path, &'b Error),
{
    for target in targets_iter {
        if target.is_file() {
            // the target is a file, so no directory walk is necessary;
            // just process the target if it passes our filter
            if let Some(target_str) = target.to_str()
                && super::filter_str(target_str, filter, skip)
            {
                procces_fn(target).await;
            }
        } else {
            // the target is a directory, so we need to walk it recursively;
            // get an async walkdir to walk the directory recursively
            let walkdir = get_async_walk(&target, include_hidden);
            // attempt to process each entry and log any errors
            walkdir
                .map(|entry_result| async {
                    match entry_result {
                        Ok(entry) => {
                            let path = entry.path();
                            // if this path passes the filter, process it
                            if let Some(path_str) = path.to_str()
                                && super::filter_str(path_str, filter, skip)
                            {
                                procces_fn(path).await;
                            }
                        }
                        Err(walkdir_err) => err_fn(
                            &walkdir_err
                                .path()
                                .map(Path::to_path_buf)
                                .unwrap_or_default(),
                            // cast async_walkdir::Error to thorium::Error
                            &Error::from(io::Error::from(walkdir_err)),
                        ),
                    }
                })
                .buffer_unordered(concurrency)
                .collect::<Vec<()>>()
                .await;
        }
    }
}

/// Recursively walks through the target and returns all file
/// entries filtered based on user preference
///
/// # Arguments
///
/// * `target` - The path to the target file/directory
/// * `include_hidden` - When set, hidden files/folders will not be filtered out
fn get_async_walk(target: &Path, mut include_hidden: bool) -> WalkDir {
    // if the target itself is hidden, assume we want to include hidden files/directories
    if is_hidden(target) {
        include_hidden = true;
    }
    // return a WalkDir that returns only files but recursively walks directories
    WalkDir::new(target)
        .filter(move |entry| async move { filter_entry(&entry, include_hidden).await })
}

/// Regex used to pattern match hidden files/directories
static HIDDEN_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\.+[^\.]+\b").unwrap());

/// Checks if a target file/directory is hidden
///
/// # Arguments
///
/// * `target` - The target file/directory path
fn is_hidden<P: AsRef<Path>>(target: P) -> bool {
    // get the file name (final component) of the path
    match target
        .as_ref()
        .file_name()
        .map(|file_name| file_name.to_string_lossy())
    {
        // check if the final component is hidden
        Some(file_name) => HIDDEN_REGEX.is_match(&file_name),
        // the target has no file name (so it's empty); it can't be a hidden file/directory
        None => false,
    }
}

/// Returns filter settings for an async directory walk
///
/// The filter walks directories recursively but will only return the
/// files in those directories. It will also refrain from traversing hidden
/// directories if the `include_hidden` flag is not set.
///
/// # Arguments
///
/// * `entry` - The file/directory to check
/// * `include_hidden` - When set, hidden files/folders will not be filtered
async fn filter_entry(entry: &DirEntry, include_hidden: bool) -> Filtering {
    // get the entry's path
    let entry_path = entry.path();
    if !include_hidden && is_hidden(&entry_path) {
        // ignore hidden files and don't traverse hidden directories
        Filtering::IgnoreDir
    } else if entry
        .file_type()
        .await
        .unwrap_or_else(|err| {
            panic!(
                "Failed to get file type for entry '{}': {}",
                entry_path.display(),
                err
            )
        })
        .is_dir()
    {
        // don't include directories in the list, but traverse them recursively
        Filtering::Ignore
    } else {
        // add this file to the list
        Filtering::Continue
    }
}

/// Retrieve a set of lines from a file as Strings
///
/// # Arguments
///
/// * `path` - The path to the file to read from
pub async fn lines_set_from_file(path: &Path) -> Result<HashSet<String>, Error> {
    // read the file into a raw String
    let raw = match tokio::fs::read_to_string(path).await {
        Ok(raw) => raw,
        Err(err) => {
            return Err(Error::new(format!(
                "Unable to read file \"{}\": {}",
                path.to_string_lossy(),
                err
            )));
        }
    };
    // separate the file by lines, filter out all empty lines, and collect to a set
    Ok(raw
        .lines()
        .filter(|&line| !line.is_empty())
        .map(str::to_string)
        .collect())
}

/// Prepend "./" or ".\" to relative paths to make it clearer that the
/// output is a path
///
/// # Arguments
///
/// * `output` - The output path to save file details in
pub fn prepend_current_dir(output: &Path) -> String {
    if output.is_relative() {
        // set patterns for Unix-style operating systems
        #[cfg(unix)]
        const CURRENT_DIR_PATTERN: &str = "./";
        #[cfg(unix)]
        const PARENT_DIR_PATTERN: &str = "../";
        // set patterns for Windows
        #[cfg(target_os = "windows")]
        const CURRENT_DIR_PATTERN: &str = ".\\";
        #[cfg(target_os = "windows")]
        const PARENT_DIR_PATTERN: &str = "..\\";
        if !output.starts_with(CURRENT_DIR_PATTERN) && !output.starts_with(PARENT_DIR_PATTERN) {
            return PathBuf::from(".")
                .join(output)
                .to_string_lossy()
                .to_string();
        }
    }
    output.to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {

    use super::is_hidden;

    #[test]
    fn test_is_hidden() {
        // Paths that should be considered hidden
        let matching = vec![
            ".hidden",
            ".config",
            "..hidden",
            ".hidden.txt",
            ".a",
            ".gitignore",
            ".my-hidden-file.txt",
            "folder/.hidden", // full path contains a hidden component and should match
        ];
        for s in matching {
            assert!(is_hidden(s), "expected hidden regex to match '{s}'",);
        }
        // Paths that should NOT be considered hidden
        let non_matching = vec![
            "visible",
            "file.txt",
            "test.",
            ".",
            "..",
            "",
            ".hidden/not-hidden", // first component is hidden, but the file name is not
        ];
        for s in non_matching {
            assert!(!is_hidden(s), "expected hidden regex NOT to match '{s}'",);
        }
    }
}
