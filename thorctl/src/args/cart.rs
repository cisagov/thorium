//! Arguments for cart-related Thorctl commands

use std::path::PathBuf;

use clap::Parser;

/// Provide a default output directory
fn default_output_path() -> PathBuf {
    PathBuf::from(".").join("carted")
}

/// Provide a default temp directory
fn default_temp_cart_path() -> PathBuf {
    #[cfg(unix)]
    return PathBuf::from("/tmp/carted");
    #[cfg(target_os = "windows")]
    return std::env::temp_dir().join("carted");
}

/// A command to cart a file
#[derive(Parser, Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cart {
    /// The files/directories to cart
    pub targets: Vec<PathBuf>,
    /// The password to use for encryption (no longer than 16 characters)
    ///     Note: encryption is only used to bypass malware scans;
    ///           the password is stored in plaintext in the cart header
    #[clap(short, long, default_value = "SecretCornIsBest", verbatim_doc_comment)]
    pub password: String,
    /// The output directory to save the carted file(s) to
    #[clap(short, long, default_value = default_output_path().into_os_string(), conflicts_with = "in_place")]
    pub output: PathBuf,
    /// Preserve the file structure of the targets rather than placing output files together in one directory.
    /// For example, a target of "/my/dir" and an output directory of "./output" will create an output in
    /// "./output/my/dir", including all subdirectories within "/my/dir".
    ///     Note: This flag is necessary if the targets contain files with identical names in order to
    ///           prevent files from being overwritten.
    #[clap(short = 'D', long, conflicts_with = "in_place", verbatim_doc_comment)]
    pub preserve_dir_structure: bool,
    /// Cart the files in place, overwriting the input files and replacing them with the carted output
    ///     Note: The ".cart" extension will be added to the file unless the "--no-extension" flag is on
    #[clap(
        short,
        long,
        default_value = "false",
        conflicts_with = "output",
        verbatim_doc_comment
    )]
    pub in_place: bool,
    /// The directory to store temporary carted data when using `--in-place`
    #[clap(long, default_value = default_temp_cart_path().into_os_string())]
    pub temp_dir: PathBuf,
    /// Refrain from adding the ".cart" extension to the output files
    #[clap(long)]
    pub no_extension: bool,
    /// Any regular expressions to use to determine which files to cart
    ///
    /// Regular expressions follow the syntax for the Rust regex crate:
    /// <https://docs.rs/regex/latest/regex/#syntax>
    #[clap(short, long)]
    pub filter: Vec<String>,
    /// Any regular expressions to use to determine which files to skip
    ///
    /// Regular expressions follow the syntax for the Rust regex crate:
    /// <https://docs.rs/regex/latest/regex/#syntax>
    #[clap(short, long)]
    pub skip: Vec<String>,
    /// Include hidden directories/files
    ///
    /// Note that if a given target is itself hidden, hidden files/directories within that
    /// target will be included
    #[clap(long)]
    pub include_hidden: bool,
}
