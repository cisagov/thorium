//! Arguments for uncart-related Thorctl commands

use std::path::PathBuf;

use clap::Parser;

/// Provide a default output directory
fn default_output_path() -> PathBuf {
    PathBuf::from(".").join("uncarted")
}

/// Provide a default temp directory
fn default_temp_uncart_path() -> PathBuf {
    #[cfg(unix)]
    return PathBuf::from("/tmp/uncarted");
    #[cfg(target_os = "windows")]
    return std::env::temp_dir().join("uncarted");
}

/// A command to uncart a file
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct Uncart {
    /// The files to uncart
    pub targets: Vec<PathBuf>,
    /// The output directory to save the uncarted file(s) to
    #[clap(short, long, default_value = default_output_path().into_os_string(), conflicts_with = "in_place")]
    pub output: PathBuf,
    /// Preserve the file structure of the targets rather than placing output files together in one directory.
    /// For example, a target of "/my/dir" and an output directory of "./output" will create an output in
    /// "./output/my/dir", including all subdirectories within "/my/dir".
    ///     Note: This flag is necessary if the targets contain files with identical names in order to
    ///           prevent files from being overwritten.
    #[clap(short = 'D', long, conflicts_with = "in_place", verbatim_doc_comment)]
    pub preserve_dir_structure: bool,
    /// Append the postfix "_uncarted" to the output files
    #[clap(long, conflicts_with = "in_place")]
    pub postfix: bool,
    /// Uncart the files in place, overwriting the carts and replacing them with the uncarted output
    #[clap(short, long, conflicts_with = "output")]
    pub in_place: bool,
    /// The directory to store temporary uncarted data when using `--in-place`
    #[clap(long, default_value = default_temp_uncart_path().into_os_string())]
    pub temp_dir: PathBuf,
    /// Any regular expressions to use to determine which files to uncart
    ///
    /// Regular expressions follow the syntax for the Rust regex crate:
    /// <https://docs.rs/regex/latest/regex/#syntax>
    #[clap(short, long)]
    pub filter: Vec<String>,
    /// Any regular expressions to use to determine which files to skip
    ///
    /// Regular expressions follow the syntax for the Rust regex crate:
    /// <https://docs.rs/regex/latest/regex/#syntax>
    #[clap(short, long, default_value = ".*_uncarted")]
    pub skip: Vec<String>,
    /// Apply include/skip filters to directories as well as files
    #[clap(short = 'F', long)]
    pub filter_dirs: bool,
    /// Include hidden directories/files
    #[clap(long)]
    pub include_hidden: bool,
}
