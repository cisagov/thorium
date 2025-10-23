use cart_rs::UncartStream;
use regex::RegexSet;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use thorium::Error;
use tokio::{
    fs::{File, OpenOptions},
    io::BufStream,
    task::JoinError,
};
use uuid::Uuid;

use crate::args::{Args, uncart::Uncart};
use crate::utils;

/// The postfix appended to uncarted files
const POSTFIX: &str = "_uncarted";

/// Handle the uncart command
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The uncart command to execute
pub async fn handle(args: &Args, cmd: &Uncart) -> Result<(), Error> {
    // build the set of regexs to determine which files to include or skip
    let filter = RegexSet::new(&cmd.filter)?;
    let skip = RegexSet::new(&cmd.skip)?;
    // prepare data to save between tasks
    let cmd = Arc::new(cmd.clone());
    // construct base output path
    let base_out_path = if cmd.in_place {
        &cmd.temp_dir
    } else {
        &cmd.output
    };
    tokio::fs::create_dir_all(base_out_path).await?;
    // print headers
    UncartLine::header();
    // iterate over targets and uncart them
    utils::fs::process_async_walk(
        cmd.targets.clone().into_iter(),
        |path| {
            // clone the entry path to use for printing results
            let path_clone = path.clone();
            // clone data for this task
            let cmd = cmd.clone();
            let base_out_path = base_out_path.clone();
            async move {
                // uncart the entry in a new task
                let uncart_result: Result<Result<PathBuf, Error>, JoinError> =
                    tokio::spawn(uncart_path(path, base_out_path, cmd)).await;
                // print the results of the uncarting
                match uncart_result {
                    Ok(Ok(out_path)) => UncartLine::success(&path_clone, &out_path),
                    Ok(Err(err)) => UncartLine::error(&path_clone, &err),
                    Err(join_err) => UncartLine::error(&path_clone, &Error::from(join_err)),
                }
            }
        },
        UncartLine::error,
        &filter,
        &skip,
        cmd.include_hidden,
        // run process concurrently based on the number of workers set
        args.workers,
    )
    .await;
    // remove temporary drectory after in-place conversion
    if cmd.in_place {
        tokio::fs::remove_dir_all(base_out_path).await?;
    }
    Ok(())
}

/// Uncart the given [`DirEntry`]
///
/// Returns the path to the uncarted file or an error on failure
///
/// # Arguments
///
/// * `entry` - The [`DirEntry`] to uncart
/// * `target_path` - The path to the target
/// * `base_out_path` - The base output path
/// * `cmd` - The uncart command including user options
async fn uncart_path(
    path: PathBuf,
    base_out_path: PathBuf,
    cmd: Arc<Uncart>,
) -> Result<PathBuf, Error> {
    // read cart file
    let cart: File = OpenOptions::new()
        .read(true)
        .write(false)
        .open(&path)
        .await?;
    let mut out_path = construct_out_path(
        &base_out_path,
        &path,
        cmd.preserve_dir_structure,
        cmd.postfix,
        cmd.in_place,
    )?;
    if let Some(out_path_parent) = out_path.parent() {
        tokio::fs::create_dir_all(&out_path_parent).await?;
    }
    // create a stream to uncart the file and copy the stream's contents to the output path
    let mut uncart_stream = UncartStream::new(BufStream::new(cart));
    let mut uncarted = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&out_path)
        .await?;
    if let Err(err) = tokio::io::copy(&mut uncart_stream, &mut uncarted).await {
        // if an error occurred while uncarting, delete the output file and return the error
        drop(uncarted);
        tokio::fs::remove_file(&out_path).await?;
        return Err(Error::from(err));
    }
    // if conversion is in-place, replace the carted file with the uncarted output
    if cmd.in_place {
        tokio::fs::rename(&out_path, &path).await?;
        // if the original file had a ".cart" extension, remove the extension and rename the file
        if path.extension().is_some_and(|ext| ext == "cart") {
            out_path = path.with_extension("");
            tokio::fs::rename(path, &out_path).await?;
        } else {
            out_path.clone_from(&path);
        }
    }
    Ok(out_path)
}

/// Construct the output path for the uncarted file
///
/// # Arguments
///
/// * `base_out_path` - The base output path
/// * `path` - The path to the input cart file
/// * `preserve_dir_structure` - User flag to preserve the directory structure of the target
/// * `postfix` - User flag to append postfix
/// * `in_place` - User flag to convert in-place
fn construct_out_path(
    base_out_path: &Path,
    path: &Path,
    preserve_dir_structure: bool,
    postfix: bool,
    in_place: bool,
) -> Result<PathBuf, Error> {
    let mut out_path = PathBuf::from(base_out_path);
    // preserve the target structure if flag is set
    if preserve_dir_structure {
        // append target path
        if path.is_absolute() {
            // leading '/', so push as a raw os str
            out_path.as_mut_os_string().push(path);
        } else {
            // no leading '/', so push as a path
            out_path.push(path);
        }
    } else {
        // if directory structure is not preserved, just add the filename;
        // unchecked because every entry given to this function is a file
        let file_name = path.file_name().ok_or(Error::Generic(String::from(
            "Error creating output path! The given entry is not a file",
        )))?;
        out_path.push(file_name);
    }
    // remove ".cart" file extension if present
    if out_path.extension().is_some_and(|ext| ext == "cart") {
        out_path.set_extension("");
    }
    if postfix {
        out_path.as_mut_os_string().push(POSTFIX);
    }
    if in_place {
        out_path.set_file_name(format!(".temp-{}{}", Uuid::new_v4(), POSTFIX));
    }
    Ok(out_path)
}

/// A single line for a file uncart log
struct UncartLine;

impl UncartLine {
    /// Print this log lines header
    pub fn header() {
        println!("{:<48} | {:<48} | {:<48}", "INPUT", "OUTPUT", "MESSAGE",);
        println!("{:-<49}+{:-<50}+{:-<48}", "", "", "");
    }

    /// Log successful uncarting
    ///
    /// # Arguments
    ///
    /// * `input_path` - The path to the input cart file
    /// * `output_path` - The path to the output uncarted file
    pub fn success<P: AsRef<Path>>(input_path: P, output_path: P) {
        Self::print(input_path.as_ref(), output_path.as_ref(), "");
    }

    /// Log an error in uncarting
    ///
    /// # Arguments
    ///
    /// * `input_path` - The path to the input cart file
    /// * `output_path` - The path to the output uncarted file
    pub fn error(input_path: &Path, err: &Error) {
        Self::print(
            input_path,
            Path::new("."),
            err.msg()
                .unwrap_or(String::from("Unknown error uncarting file")),
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
                "-", "-",
            );
        }
    }
}
