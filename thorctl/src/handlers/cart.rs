use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use cart_rs::CartStream;

use futures::{stream, StreamExt};
use generic_array::{typenum::U16, GenericArray};
use regex::RegexSet;
use thorium::Error;
use tokio::{
    fs::{File, OpenOptions},
    io::BufStream,
    task::JoinError,
};
use uuid::Uuid;
use walkdir::DirEntry;

use crate::args::{cart::Cart, Args};
use crate::utils;

/// Handle the cart command
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The cart command to execute
pub async fn handle(args: &Args, cmd: &Cart) -> Result<(), Error> {
    // build the set of regexs to determine which files to include or skip
    let filter = RegexSet::new(&cmd.filter)?;
    let skip = RegexSet::new(&cmd.skip)?;
    let password_array: GenericArray<u8, U16> =
        GenericArray::clone_from_slice(&pad_zeroes(cmd.password.as_bytes())?);
    // construct base output path
    let base_out_path = if cmd.in_place {
        &cmd.temp_dir
    } else {
        &cmd.output
    };
    tokio::fs::create_dir_all(base_out_path).await?;
    // print headers
    CartLine::header();
    // iterate over targets and uncart them
    for target in &cmd.targets {
        cart_target(
            target,
            base_out_path,
            cmd,
            &filter,
            &skip,
            &password_array,
            args.workers,
        )
        .await;
    }
    // remove temporary directory after in-place conversion
    if cmd.in_place {
        tokio::fs::remove_dir_all(base_out_path).await?;
    }
    Ok(())
}

/// Cart the file(s) at the given target
///
/// # Arguments
///
/// * `target` - The path to the target file or directory
/// * `base_out_path` - The base output path
/// * `cmd` - The cart command including user options
/// * `filter` - Regex set used to determine which files to uncart
/// * `skip` - Regex set used to determine which files to skip when uncarting
/// * `password` - The password used to encrypt the cart file
/// * `workers` - The number of workers that will cart the files
async fn cart_target(
    target: &String,
    base_out_path: &Path,
    cmd: &Cart,
    filter: &RegexSet,
    skip: &RegexSet,
    password: &GenericArray<u8, U16>,
    workers: usize,
) {
    // create Arcs to share references between threads
    let target_path = Arc::new(PathBuf::from(target));
    let base_out_path = Arc::new(PathBuf::from(base_out_path));
    let cmd = Arc::new(cmd.clone());
    let password = Arc::new(*password);
    // cart all file entries at the given target path, filtered based on settings
    stream::iter(
        // filter the entries based on user settings
        utils::fs::get_filtered_entries(target, filter, skip, cmd.include_hidden, cmd.filter_dirs)
            .into_iter()
            // map each filtered entry to a future that will spawn a new thread
            // that carts its given entry. The future awaits the carting thread,
            // then prints out the result
            .map(|entry| async {
                // clone the entry path to use for printing the results
                let entry_path = PathBuf::from(entry.path());
                // clone the Arcs for this thread
                let target_path = target_path.clone();
                let base_out_path = base_out_path.clone();
                let cmd = cmd.clone();
                let password = password.clone();
                // cart the entry in a new thread
                let cart_result: Result<Result<PathBuf, Error>, JoinError> =
                    tokio::spawn(cart_entry(entry, target_path, base_out_path, cmd, password))
                        .await;
                // print out the result form the thread
                match cart_result {
                    Ok(Ok(out_path)) => CartLine::success(entry_path, out_path),
                    Ok(Err(err)) => CartLine::error(entry_path, &err),
                    Err(err) => CartLine::error(entry_path, &Error::from(err)),
                }
            }),
    )
    // await the mapped futures, limiting the maximum running at any given time by the number of workers
    .for_each_concurrent(workers, |future| future)
    .await;
}

/// Cart the given [`DirEntry`]
///
/// Returns the path to the carted file or an error on failure
///
/// # Arguments
///
/// * `entry` - The [`DirEntry`] to cart
/// * `target_path` - The path to the target
/// * `base_out_path` - The base output path
/// * `cmd` - The cart command including user options
/// * `password` - The password used to encrypt the cart file
async fn cart_entry(
    entry: DirEntry,
    target_path: Arc<PathBuf>,
    base_out_path: Arc<PathBuf>,
    cmd: Arc<Cart>,
    password: Arc<GenericArray<u8, U16>>,
) -> Result<PathBuf, Error> {
    // read input file
    let input: File = OpenOptions::new()
        .read(true)
        .write(false)
        .open(&entry.path())
        .await?;
    // generate output path and create necessary directories
    let mut out_path = construct_out_path(
        &base_out_path,
        &target_path,
        entry.path(),
        cmd.preserve_dir_structure,
        cmd.no_extension,
        cmd.in_place,
    )?;
    if let Some(out_path_parent) = out_path.parent() {
        tokio::fs::create_dir_all(&out_path_parent).await?;
    }
    // open output carted file
    let mut output_cart = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&out_path)
        .await?;
    // create a stream to cart the file and copy the stream's contents to the output path
    let mut cart_stream = CartStream::new(BufStream::new(input), &password)?;
    if let Err(err) = tokio::io::copy(&mut cart_stream, &mut output_cart).await {
        // if an error occurred while carting, delete the output file and return the error
        drop(output_cart);
        tokio::fs::remove_file(&out_path).await?;
        return Err(Error::from(err));
    }
    // if conversion is in-place, replace the input file with the output cart
    if cmd.in_place {
        tokio::fs::rename(&out_path, entry.path()).await?;
        if cmd.no_extension {
            out_path = PathBuf::from(entry.path());
        } else {
            out_path = PathBuf::from(entry.path());
            out_path.as_mut_os_string().push(".cart");
            tokio::fs::rename(entry.path(), &out_path).await?;
        }
    }
    Ok(out_path)
}

/// Pad the given byte array with 0's to create a 16-byte array
///
/// # Arguments
///
/// * `arr` - The input byte array
fn pad_zeroes(arr: &[u8]) -> Result<[u8; 16], Error> {
    match arr.len() {
        len if len > 16 => Err(Error::new("Password is greater than 16 characters!")),
        16 => match arr.try_into() {
            Ok(arr) => Ok(arr),
            _ => Err(Error::new(
                "Unable to statically size 16-byte array to 16 bytes",
            )),
        },
        _ => {
            let mut padded: [u8; 16] = [0; 16];
            padded[..arr.len()].copy_from_slice(arr);
            Ok(padded)
        }
    }
}

/// Construct the output path for the uncarted file
///
/// # Arguments
///
/// * `base_out_path` - The base output path
/// * `target_path` - The path to the target given by the user
/// * `entry_path` - The path to the entry or input cart file
/// * `preserve_dir_structure` - User flag to preserve the directory structure of the target
/// * `no_extension` - User flag to refrain from adding the ".cart" extension to the file
/// * `in_place` - User flag to convert in-place
fn construct_out_path(
    base_out_path: &Path,
    target_path: &Path,
    entry_path: &Path,
    preserve_dir_structure: bool,
    no_extension: bool,
    in_place: bool,
) -> Result<PathBuf, Error> {
    let mut out_path = PathBuf::from(base_out_path);
    // preserve the target structure if flag is set
    if preserve_dir_structure {
        // append target path
        if target_path.is_absolute() {
            out_path.as_mut_os_string().push(target_path);
        } else {
            out_path.push(target_path);
        }
        let entry_path_buf = PathBuf::from(entry_path);
        // remove superfluous target path from entry path
        match entry_path_buf.strip_prefix(target_path) {
            Ok(stripped_entry_path) => {
                // append the rest of the entry to the output path;
                // stripped entry path is blank if target is the path to the file,
                // so only append if stripped entry path has components
                if stripped_entry_path.components().next().is_some() {
                    out_path.push(stripped_entry_path);
                }
            }
            Err(error) => return Err(Error::from(error)),
        }
    } else {
        // if directory structure is not preserved, just add the filename;
        // unchecked because every entry given to this function is a file
        let file_name = entry_path.file_name().ok_or(Error::Generic(String::from(
            "Error creating output path! The given entry is not a file",
        )))?;
        out_path.push(file_name);
    }
    if in_place {
        out_path.set_file_name(format!(".temp-{}.cart", Uuid::new_v4()));
    } else if !no_extension {
        out_path.as_mut_os_string().push(".cart");
    }
    Ok(out_path)
}

/// A single line for a file uncart log
struct CartLine;

impl CartLine {
    /// Print this log lines header
    pub fn header() {
        println!("{:<48} | {:<48} | {:<48}", "INPUT", "OUTPUT", "MESSAGE",);
        println!("{:-<49}+{:-<50}+{:-<48}", "", "", "");
    }

    /// Log successful carting
    ///
    /// # Arguments
    ///
    /// * `input_path` - The path to the input cart file
    /// * `output_path` - The path to the output uncarted file
    pub fn success<P: AsRef<Path>>(input_path: P, output_path: P) {
        Self::print(input_path.as_ref(), output_path.as_ref(), "");
    }

    /// Log an error in carting
    ///
    /// # Arguments
    ///
    /// * `input_path` - The path to the input cart file
    /// * `output_path` - The path to the output uncarted file
    pub fn error<P: AsRef<Path>>(input_path: P, err: &Error) {
        Self::print(
            input_path.as_ref(),
            Path::new("."),
            err.msg()
                .unwrap_or(String::from("Unknown error carting file")),
        );
    }

    /// Print a line
    ///
    /// # Arguments
    ///
    /// * `input_path` - The path to the input file
    /// * `output_path` - The path to the output file
    /// * `msg` - The message to print
    fn print<T: AsRef<str>>(input_path: &Path, output_path: &Path, msg: T) {
        if let (Some(cart_path_str), Some(output_path_str)) =
            (input_path.to_str(), output_path.to_str())
        {
            println!(
                "{:<48} | {:<48} | {}",
                cart_path_str,
                output_path_str,
                msg.as_ref()
            );
        } else {
            println!(
                "{:<48} | {:<48} | Paths contain non-ASCII characters",
                "-", "-"
            );
        }
    }
}
