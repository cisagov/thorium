//! Generates Python type-hints/stubs for the Thorium Python module
//! using `PyO3`'s experimental introspection feature.

use anyhow::{Context, Error};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

/// The name of the Python module
const MAIN_MODULE_NAME: &str = "thorium";

/// Imports we need for the stubs
const IMPORTS: &str = r"

from pathlib import Path
from uuid import UUID
";

/// Try searching for the module library file we want to introspect in the given directory
/// recursively
fn find_lib_file(lib_dir: &Path) -> Result<Option<PathBuf>, io::Error> {
    let walkdir = WalkDir::new(lib_dir);
    // walk the directory recursively
    for entry in walkdir {
        let entry = entry?;
        let path = entry.path();
        // check whether this is probably the library file
        if entry.file_type().is_file()
            && let Some(file_name) = entry.file_name().to_str()
            && file_name.starts_with(MAIN_MODULE_NAME)
            && path.extension().is_some_and(|ext| ext == "so")
        {
            return Ok(Some(path.to_path_buf()));
        }
    }
    Ok(None)
}

/// Write the stubs data to a file
fn write_stubs(writer: &mut BufWriter<File>, stubs: &str) -> Result<(), io::Error> {
    // write the generated stubs
    writer.write_all(stubs.as_bytes())?;
    // write the imports
    writer.write_all(IMPORTS.as_bytes())?;
    // flush
    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Error> {
    // get metadata for the workspace using cargo
    let metadata = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .context("Error retrieving cargo metadata")?;
    // based on the workspace root, define the root to the thorpy package
    let thorpy_root = metadata
        .workspace_root
        .as_std_path()
        .join("python")
        .join("thorpy");

    println!("Generating stub files");
    // search for a dynamic library matching our module in the lib directory
    let lib_dir = thorpy_root.join(".venv").join("lib");
    let thorium_lib_path = find_lib_file(&lib_dir)
        .with_context(|| {
            format!(
                "Error searching for {} cdylib in directory '{}'",
                MAIN_MODULE_NAME,
                lib_dir.display()
            )
        })?
        .ok_or_else(|| {
            Error::msg(format!(
                "No {} library found in directory '{}'",
                MAIN_MODULE_NAME,
                lib_dir.display()
            ))
        })?;

    // introspect the library
    let python_module = pyo3_introspection::introspect_cdylib(thorium_lib_path, MAIN_MODULE_NAME)
        .with_context(|| format!("Failed introspection of {MAIN_MODULE_NAME}"))?;
    let type_stubs = pyo3_introspection::module_stub_files(&python_module);

    // retrieve the stubs data
    let stubs_string = type_stubs
        .get(&PathBuf::from("__init__.pyi"))
        .context("Failed to get __init__.pyi")?;
    // write the stubs file
    let stubs_path = thorpy_root.join(format!("{MAIN_MODULE_NAME}.pyi"));
    let stubs_file = File::create(&stubs_path)
        .with_context(|| format!("Failed to create stubs file '{}'", stubs_path.display()))?;
    let mut writer = BufWriter::new(stubs_file);
    write_stubs(&mut writer, stubs_string)
        .with_context(|| format!("Failed to write stubs file '{}'", stubs_path.display()))?;

    println!("Generated stubs: {}", stubs_path.display());
    Ok(())
}
