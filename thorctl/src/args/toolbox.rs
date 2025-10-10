//! Arguments for toolbox-related Thorctl commands

use clap::Parser;
use std::path::PathBuf;
use url::Url;

/// A command to interact with Thorium toolboxes
#[derive(Parser, Debug)]
pub enum Toolbox {
    /// Import a toolbox into Thorium
    ///
    /// A Thorium toolbox is an external collection of tools and pipelines pre-configured
    /// and ready to run in Thorium
    #[clap(version, author)]
    Import(ImportToolbox),
}

/// The location of the toolbox manifest, either by URL or by file path
#[derive(Debug, Clone)]
pub enum ManifestLocation {
    /// The manifest is at this URL
    Url(Url),
    /// The manifest is at this file path
    Path(PathBuf),
}

impl std::str::FromStr for ManifestLocation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // try parsing as a URL first
        if let Ok(url) = Url::parse(s) {
            return Ok(Self::Url(url));
        }
        // if URL parsing fails, treat it as a file path
        Ok(Self::Path(PathBuf::from(s)))
    }
}

/// Download a toolbox manifest and import it into Thorium
#[derive(Parser, Debug)]
pub struct ImportToolbox {
    /// The URL or file path on the system where the toolbox manifest is found
    pub manifest: ManifestLocation,
    /// Skip the confirmation dialog
    #[clap(short = 'y', long)]
    pub skip_confirm: bool,
    /// Force the images and pipelines to be uploaded to a specific group
    ///
    /// The group will be created if it doesn't already exist
    #[clap(long)]
    pub group_override: Option<String>,
}
