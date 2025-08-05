//! Handles user keys operations

use thorium::Error;

use std::path::PathBuf;

/// Get the base path to store keys at
#[cfg(target_os = "linux")]
pub fn base_path() -> PathBuf {
    PathBuf::from("/opt/thorium-keys")
}

/// Get the base path to store keys at
#[cfg(target_os = "windows")]
pub fn base_path() -> PathBuf {
    PathBuf::from("C:\\thorium\\keys")
}

/// Build the path to a user's keys
///
/// # Arguments
///
/// * `username` - The user whose keys we want to write/get
pub fn path(username: &str) -> PathBuf {
    // start with the bas path for our keys
    let mut path = base_path();
    // append our username
    path.push(username);
    path.push("keys.yml");
    path
}

/// Check if a target users key file exists and if it is correct
///
/// # Arguments
///
/// `path` - The path to check against
/// `token` - The token to look for
pub async fn exists(path: &PathBuf, token: &str) -> Result<bool, Error> {
    // determine if this users keys exist on disk or not
    if path.exists() {
        // this users key exists so make sure its correct
        // read in this users current key file
        let data = tokio::fs::read_to_string(path).await?;
        // check this users token is in our key file
        if data.contains(token) {
            return Ok(true);
        } else {
            // a keys file exists but its wrong so delete it
            tokio::fs::remove_file(path).await?;
        }
    }
    // this file doesn't currently exist
    Ok(false)
}
