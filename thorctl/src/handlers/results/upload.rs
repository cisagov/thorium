use futures::TryStreamExt;
use http::StatusCode;
use owo_colors::OwoColorize;
use std::ffi::OsStr;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use thorium::client::ResultsClient;
use thorium::models::{OnDiskFile, OutputRequest, Sample};
use thorium::{Error, Thorium};

use crate::args::results::UploadResults;

/// prints out a single downloaded result line
macro_rules! upload_print {
    ($code:expr, $sample:expr, $tool:expr, $msg:expr) => {
        println!(
            "{:<4} | {:<64} | {:<20} | {:<32} ",
            $code, $sample, $tool, $msg
        )
    };
}

struct UploadLine;

impl UploadLine {
    /// Print this log lines header
    #[rustfmt::skip]
    #[allow(clippy::print_literal)]
    pub fn header() {
        println!(
            "{} | {:<64} | {:<20} | {:<32}",
            "CODE", "SAMPLE", "TOOL", "MESSAGE"
        );
        println!("{:-<5}+{:-<66}+{:-<22}+{:-<34}", "", "", "", "");
    }

    /// Print a log line for an uploaded result
    pub fn uploaded<S: Display, T: Display>(sample: S, tool: T) {
        // log this line
        upload_print!(200.bright_green(), sample, tool, "-");
    }

    /// Print a log line for an uploaded result
    pub fn uploaded_dry_run<S: Display, T: Display>(sample: S, tool: T) {
        // log this line
        upload_print!("-".bright_green(), sample, tool, "-");
    }

    /// Print an error log line for a result that could not be uploaded
    pub fn error<S: Display, T: Display>(sample: S, tool: T, err: &Error) {
        // get the error message for this line
        let msg = err.msg().unwrap_or_else(|| "-".to_owned());
        // get the error status
        let status = err.status();
        // get a default "-" if no status, otherwise map to a str
        let status_str = status.as_ref().map_or("-", StatusCode::as_str);
        // log this line
        upload_print!(status_str.bright_red(), sample, tool, msg);
    }
}

/// Attempt to upload a result for a file, outputting a log line on success or
/// error, or just output a successful line if we're in dry run mode
macro_rules! upload {
    ($thorium:expr, $req:expr, $sha256:expr, $tool:expr, $cmd:expr) => {
        async {
            if $cmd.dry_run {
                // just log a line if we're in dry run mode
                UploadLine::uploaded_dry_run($sha256, $tool);
            } else {
                match $thorium.files.create_result($req).await {
                    Ok(_) => UploadLine::uploaded($sha256, $tool),
                    Err(err) => UploadLine::error($sha256, $tool, &err),
                }
            }
        }
    };
}

/// Returns true if the string is a valid SHA256 (is 64 ASCII hex digits)
///
/// # Arguments
///
/// * `s` - The string to check if it's probably a SHA256
fn is_sha256<T: AsRef<str>>(s: T) -> bool {
    let s = s.as_ref();
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Uploads results for a file based on tool sub-directories
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The upload results command
/// * `sha256` - The SHA256 of the file to upload results to
/// * `tool_subdirs` - The list of tool sub-directories to upload
async fn upload_tool_subdirs(
    thorium: &Thorium,
    cmd: &UploadResults,
    sha256: &str,
    tool_subdirs: Vec<PathBuf>,
) -> Result<(), Error> {
    // walk each tool subdirectory recursively and upload results for that tool
    for tool_subdir in tool_subdirs {
        // get the tool name (the name of the sub-directory)
        let tool = match tool_subdir.file_name() {
            Some(file_name) => file_name.to_string_lossy().to_string(),
            // there is no file name; just proceed to the next tool
            None => continue,
        };
        // construct the path to the main results file
        let results_path = tool_subdir.join(&cmd.results);
        // check if the main results exist
        if !tokio::fs::try_exists(&results_path).await.map_err(|err| {
            Error::new(format!(
                "Error checking that results exist at '{}': {}",
                results_path.display(),
                err
            ))
        })? {
            // there are no main results, so log an error for each tool and move on to the next tool
            UploadLine::error(
                sha256,
                tool,
                &Error::new(format!(
                    "No results file found at '{}'",
                    results_path.display()
                )),
            );
            continue;
        }
        // try to read the results file to a string
        let results_string = match tokio::fs::read_to_string(&results_path).await {
            Ok(results_string) => results_string,
            Err(err) => {
                // log an error that we couldn't read this results file for each tool and move on
                UploadLine::error(
                    sha256,
                    tool,
                    &Error::new(format!(
                        "Error reading results file '{}': {}",
                        results_path.display(),
                        err
                    )),
                );
                return Ok(());
            }
        };
        // walk the directory recursively
        let walkdir = async_walkdir::WalkDir::new(&tool_subdir);
        let result_files = walkdir
            .try_fold(Vec::new(), |mut result_files, entry| {
                let results_path_ref = &results_path;
                let tool_subdir_ref = &tool_subdir;
                async move {
                    let path = entry.path();
                    // only add this path if it's not the main result *and* it's a file
                    if &path != results_path_ref && path.is_file() {
                        // trim the tool subdir so we only include the nested part
                        let on_disk = OnDiskFile::new(path).trim_prefix(tool_subdir_ref);
                        result_files.push(on_disk);
                    }
                    Ok(result_files)
                }
            })
            .await
            .map_err(|err| {
                Error::new(format!(
                    "Error reading tool sub-directory '{}': {}",
                    tool_subdir.display(),
                    err
                ))
            })?;
        // upload the results for this tool
        let results_req = OutputRequest::<Sample>::new(
            sha256.to_string(),
            tool.clone(),
            results_string,
            cmd.display_type,
        )
        .files(result_files);
        upload!(thorium, results_req, sha256, tool, cmd).await;
    }
    Ok(())
}

/// Uploads result to all tools given by flags, not organized in tool
/// sub directories
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The upload results command
/// * `sha256` - The SHA256 of the file to upload results to
/// * `path` - The path to the results
/// * `unnested_result_files` - The result files found in the path collected previously
async fn upload_tool_flags(
    thorium: &Thorium,
    cmd: &UploadResults,
    sha256: &str,
    path: &Path,
    unnested_result_files: Vec<OnDiskFile>,
) -> Result<(), Error> {
    // build a path to the main results file as defined by the user
    let results = path.join(&cmd.results);
    // check if we even have an unnested results file
    if !tokio::fs::try_exists(&results).await.map_err(|err| {
        Error::new(format!(
            "Error checking that results exist at '{}': {}",
            results.display(),
            err
        ))
    })? {
        // there is no unnested results, so log an error for each tool and exit
        for tool in &cmd.tools {
            UploadLine::error(
                sha256,
                tool,
                &Error::new(format!("No results file found at '{}'", results.display())),
            );
        }
        return Ok(());
    }
    // try to read the results file to a string
    let results_string = match tokio::fs::read_to_string(&results).await {
        Ok(results_string) => results_string,
        Err(err) => {
            // log an error that we couldn't read this results file for each tool and move on
            for tool in &cmd.tools {
                UploadLine::error(
                    sha256,
                    tool,
                    &Error::new(format!(
                        "Error reading results file '{}': {}",
                        results.display(),
                        err
                    )),
                );
            }
            return Ok(());
        }
    };
    // upload unnested files for each tool given in the command
    for tool in &cmd.tools {
        let results_req = OutputRequest::<Sample>::new(
            sha256.to_string(),
            tool.clone(),
            results_string.clone(),
            cmd.display_type,
        )
        .files(unnested_result_files.clone());
        upload!(thorium, results_req, sha256, tool, cmd).await;
    }
    Ok(())
}

/// Uploads results for a single file to Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The upload results command
/// * `sha256` - The SHA256 of the file to upload results to
/// * `path` - The path to the results
async fn upload_helper(
    thorium: &Thorium,
    cmd: &UploadResults,
    sha256: &str,
    path: &Path,
) -> Result<(), Error> {
    // check if this is a file or a directory
    if path.is_file() {
        // this is a sha256 file, so just upload it as a result for each given tool
        if cmd.tools.is_empty() {
            // log an error and move on to the next target if no tools were given
            UploadLine::error(
                sha256,
                "-",
                &Error::new("Results file found but no tools were supplied with flag '--tools'"),
            );
            return Ok(());
        }
        for tool in &cmd.tools {
            // try to read in this sha256 file as the results file
            let results_string = tokio::fs::read_to_string(path).await.map_err(|err| {
                Error::new(format!(
                    "Error reading results file '{}': {}",
                    path.display(),
                    err
                ))
            })?;
            let results_req = OutputRequest::<Sample>::new(
                sha256.to_string(),
                tool,
                results_string,
                cmd.display_type,
            );
            upload!(thorium, results_req, sha256, tool, cmd).await;
        }
    } else {
        let mut unnested_result_files = Vec::new();
        let mut tool_subdirs = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&path).await.map_err(|err| {
            Error::new(format!(
                "Error reading directory '{}': {}",
                path.display(),
                err
            ))
        })?;
        while let Some(entry) = read_dir.next_entry().await.map_err(|err| {
            Error::new(format!(
                "Error reading directory '{}': {}",
                path.display(),
                err
            ))
        })? {
            let inner_path = entry.path();
            // check if the entry is a file
            if entry
                .file_type()
                .await
                .map_err(|err| {
                    Error::new(format!(
                        "Error reading metadata for file '{}': {}",
                        inner_path.display(),
                        err
                    ))
                })?
                .is_file()
            {
                // add this file to the list of files not nested in tool directories
                let on_disk = OnDiskFile::new(inner_path).trim_prefix(path);
                unnested_result_files.push(on_disk);
            } else {
                // this is a directory, so assume it's a tool and add it to the list
                tool_subdirs.push(inner_path);
            }
        }
        // upload results by tool sub-directory
        upload_tool_subdirs(thorium, cmd, sha256, tool_subdirs).await?;
        if !unnested_result_files.is_empty() && cmd.tools.is_empty() {
            // log an error if there are unnested result files but no --tools were given
            UploadLine::error(
                sha256,
                "-",
                &Error::new(
                    "Found result files not in tool sub-directories but no tools were given with flag '--tools'",
                ),
            );
        } else if !cmd.tools.is_empty() {
            // remove the main results file so we don't upload it twice (as main results and as attachment)
            if let Some(index) = unnested_result_files.iter().position(|file| {
                file.path
                    .file_name()
                    // this is the main results file if its name matches the results set in the command
                    .is_some_and(|file_name| cmd.results.as_str() == file_name)
            }) {
                unnested_result_files.swap_remove(index);
            }
            // upload unnested files to every tool given by the cmd flags
            upload_tool_flags(thorium, cmd, sha256, path, unnested_result_files).await?;
        }
    }
    Ok(())
}

/// Crawl target directories and upload their results to Thorium
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `cmd` - The full result upload command/args
pub async fn upload(thorium: &Thorium, cmd: &UploadResults) -> Result<(), Error> {
    // print the header
    UploadLine::header();
    // crawl over each path and upload them if they are new
    for target in &cmd.targets {
        // check if this file name in the path is already a sha256
        let path = Path::new(target);
        if let Some(file_name) = path.file_name().map(OsStr::to_string_lossy)
            && is_sha256(&file_name)
        {
            // upload this sha256
            upload_helper(thorium, cmd, &file_name, path).await?;
        } else {
            // not a sha256, so look for sha256 sub-directories/files
            let mut read_dir = tokio::fs::read_dir(target)
                .await
                .map_err(|err| Error::new(format!("Error reading target '{target}': {err}")))?;
            while let Some(entry) = read_dir
                .next_entry()
                .await
                .map_err(|err| Error::new(format!("Error reading target '{target}': {err}")))?
            {
                let path = entry.path();
                // only process files/directories that are sha256's
                if let Some(file_name) = path.file_name().map(OsStr::to_string_lossy)
                    && is_sha256(&file_name)
                {
                    upload_helper(thorium, cmd, &file_name, &path).await?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_sha256;

    #[test]
    fn test_is_sha256_valid() {
        // 64 hex digits
        let s = "0123456789abcdefABCDEF0123456789abcdefABCDEF0123456789abcdefABCD";
        // make sure there are actully 64 characters above
        assert_eq!(s.len(), 64);
        // test the function itself
        assert!(is_sha256(s));
    }

    #[test]
    fn test_is_sha256_invalid_nonhex() {
        // 64 hex digits (except an invalid 'G' at the end)
        let s = "0123456789abcdefABCDEF0123456789abcdefABCDEF0123456789abcdefABCG";
        // make sure there are actully 64 characters above
        assert_eq!(s.len(), 64);
        // test the function itself
        assert!(!is_sha256(s));
    }

    #[test]
    fn test_is_sha256_invalid_length() {
        let short = "abcd";
        assert!(!is_sha256(short));
        let long = "a".repeat(65);
        assert!(!is_sha256(&long));
    }

    #[test]
    fn test_is_sha256_invalid_characters() {
        let invalid = "!".repeat(64);
        assert!(!is_sha256(&invalid));
    }

    #[test]
    fn test_is_sha256_mixed_invalid() {
        let mut mixed = "a".repeat(63);
        mixed.push('!');
        assert!(!is_sha256(&mixed));
    }

    #[test]
    fn test_is_sha256_invalid_spaces() {
        let mut mixed = "a".repeat(63);
        mixed.push(' ');
        assert!(!is_sha256(&mixed));
    }
}
