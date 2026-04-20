//! Arguments for image-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use clap::Parser;
use clap::builder::NonEmptyStringValueParser;
use std::path::PathBuf;
use thorium::client::conf;
use thorium::models::ImageScaler;
use uuid::Uuid;

use crate::utils;

use super::traits::describe::{DescribeCommand, DescribeFormat, DescribeSealed};
use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};
use super::{CreateNotification, GetNotificationOpts};

/// The commands to send to the images task handler
#[derive(Parser, Debug)]
pub enum Images {
    /// Get available images and their details
    #[clap(version, author)]
    Get(GetImages),
    /// Describe specific images in a human-readable format (use `--format json`
    /// for the full raw config)
    #[clap(version, author)]
    Describe(DescribeImages),
    /// Edit/update an image
    ///
    /// Static/uneditable fields are marked '*<field>*'
    #[clap(version, author)]
    Edit(EditImage),
    /// Manage/list image notifications
    #[clap(subcommand)]
    Notifications(ImageNotifications),
    /// Manage/list image bans
    #[clap(subcommand)]
    Bans(ImageBans),
    /// Delete images
    #[clap(version, author)]
    Delete(DeleteImages),
    /// Import images
    #[clap(version, author)]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    Import(ImportImages),
    /// Export images
    #[clap(version, author)]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    Export(ExportImages),
}

/// A command to get info on some images
#[derive(Parser, Debug)]
pub struct GetImages {
    /// Any groups to filter by when searching for images
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// Filter by a specific scaler
    #[clap(short, long, ignore_case = true)]
    pub scaler: Option<ImageScaler>,
    /// The max number of images to list
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving images
    ///     Note: This can lead to retrieving info for many thousands of images
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The page size to use in retrieving the images
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// Print the images in alphabetical order rather than by group, then creation date
    ///     Note: Sorting can require many system resources for large amounts of images
    #[clap(short, long, verbatim_doc_comment)]
    pub alpha: bool,
}

/// A command to describe images in full
#[derive(Parser, Debug)]
pub struct DescribeImages {
    /// Any specific images to describe, optionally with a specific group delimited
    /// with a colon in case other groups have an image with the same name
    /// (e.g. '<IMAGE>:<OPTIONAL-GROUP>')
    pub images: Vec<String>,
    /// The path to a file containing a list of images to describe separated by newlines;
    /// optionally, each image can have a specific group delimited with a colon in case
    /// other groups have an image with the same name
    /// (e.g. '<IMAGE>:<OPTIONAL-GROUP>')
    #[clap(short = 'L', long = "list")]
    pub list: Option<PathBuf>,
    /// The path to the file to write output to; if not provided, details will be output to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// The output format for the description (human-readable by default)
    #[clap(long, value_enum, default_value_t, ignore_case = true)]
    pub format: DescribeFormat,
    /// Output details as condensed JSON (no formatting/whitespace); implies `--format json`
    #[clap(long)]
    pub condensed: bool,
    /// Any specific groups to filter by when describing images
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// Describe all images to which you have access (still within the limit given in `--limit`)
    #[clap(long)]
    pub all: bool,
    /// The maximum number of images to retrieve per group
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Describe images with no limit
    #[clap(long)]
    pub no_limit: bool,
    /// The number of images to retrieve per request
    #[clap(short, long, default_value_t = 50)]
    pub page_size: usize,
}

impl SearchSealed for DescribeImages {
    fn get_search_params(&self) -> SearchParams<'_> {
        SearchParams {
            groups: &self.groups,
            tags: &[],
            tags_case_insensitive: false,
            delimiter: '=',
            start: &None,
            end: &None,
            date_fmt: "",
            cursor: None,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

impl SearchParameterized for DescribeImages {
    fn has_targets(&self) -> bool {
        !self.images.is_empty() || self.list.is_some()
    }

    fn apply_to_all(&self) -> bool {
        self.all
    }
}

/// A specific image target containing an optional group in case
/// more than one group has an image with the same name
pub struct ImageTarget {
    /// The name of the image
    image: String,
    /// The optional group that the image belongs to
    group: Option<String>,
}

impl DescribeSealed for DescribeImages {
    type Data = thorium::models::Image;

    type Target<'a> = ImageTarget;

    type Cursor = thorium::client::Cursor<Self::Data>;

    fn raw_targets(&self) -> &[String] {
        &self.images
    }

    fn condensed(&self) -> bool {
        self.condensed
    }

    fn format(&self) -> Result<DescribeFormat, thorium::Error> {
        // `--condensed` is a JSON modifier, so it forces JSON
        if self.condensed {
            return Ok(DescribeFormat::Json);
        }
        // images can only be rendered as JSON or human-readable text; reject any
        // other format the shared DescribeFormat enum may offer
        if matches!(self.format, DescribeFormat::Json | DescribeFormat::Human) {
            Ok(self.format)
        } else {
            Err(thorium::Error::new(
                "describe images only supports the 'json' or 'human' output format",
            ))
        }
    }

    fn render_human(
        &self,
        datum: &Self::Data,
        out: &mut dyn std::io::Write,
        ansi: bool,
    ) -> Result<(), thorium::Error> {
        utils::images::print_image_details(datum, out, ansi)
    }

    fn out_path(&self) -> Option<&std::path::PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&std::path::PathBuf> {
        self.list.as_ref()
    }

    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error> {
        let mut split = raw.split(':');
        let Some(image) = split.next() else {
            return Err(thorium::Error::new(format!(
                "Unable to parse '{raw}' to image target! \
                    The target should be formatted as the image's name and optionally
                    the image's group delimited with a single colon (<IMAGE>:<OPTIONAL-GROUP>)",
            )));
        };
        let group = split.next();
        if split.next().is_some() {
            return Err(thorium::Error::new(format!(
                "Unable to parse '{raw}' to image target! \
                The target should be formatted as the image's name and optionally
                the image's group delimited with a single colon (<IMAGE>:<OPTIONAL-GROUP>)",
            )));
        }
        Ok(ImageTarget {
            image: image.to_owned(),
            group: group.map(ToOwned::to_owned),
        })
    }

    async fn retrieve_data(
        &self,
        target: Self::Target<'_>,
        thorium: &thorium::Thorium,
    ) -> Result<Self::Data, thorium::Error> {
        let group = if let Some(group) = &target.group {
            group.clone()
        } else {
            utils::images::find_image_group(thorium, &target.image).await?
        };
        thorium.images.get(&group, &target.image).await
    }

    async fn retrieve_data_search(
        &self,
        thorium: &thorium::Thorium,
    ) -> Result<Vec<Self::Cursor>, thorium::Error> {
        let params = self.get_search_params();
        let groups = if self.apply_to_all() {
            // retrieve all of the users groups if all images should be described
            utils::groups::get_all_groups(thorium).await?
        } else {
            // otherwise use only the specified groups
            params.groups.to_vec()
        };
        let cursors = groups.iter().map(|group| {
            thorium
                .images
                .list(group)
                .details()
                .page_size(params.page_size as u64)
        });
        if params.no_limit {
            Ok(cursors.collect())
        } else {
            Ok(cursors
                .map(|cursor| cursor.limit(params.limit as u64))
                .collect())
        }
    }
}

impl DescribeCommand for DescribeImages {}

/// Provide the help message for the editor arg
fn editor_help() -> String {
    format!(
        "The editor to use when editing the image ('{}' by default); the default can be modified using \
    'thorctl config --default-editor', but this flag overrides any set defaults",
        conf::default_default_editor()
    )
}

/// Args for editing an image
#[derive(Parser, Debug)]
pub struct EditImage {
    /// The name of the image to edit
    pub image: String,
    /// The group the image is in; required if other images have
    /// the same name
    pub group: Option<String>,
    /// The editor to use when editing the image
    #[clap(short, long, help = editor_help())]
    pub editor: Option<String>,
}

/// The image ban specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum ImageBans {
    /// Add a ban to an image, preventing it from being scaled
    #[clap(version, author)]
    Create(CreateImageBan),
    /// Remove a ban from an image
    #[clap(version, author)]
    Delete(DeleteImageBan),
}

/// The args related to adding image bans
#[derive(Parser, Debug, Clone)]
pub struct CreateImageBan {
    /// The image's group
    pub group: String,
    /// The name of the image
    pub image: String,
    /// The message explaining why the image was banned
    pub msg: String,
}

/// The args related to removing image bans
#[derive(Parser, Debug, Clone)]
pub struct DeleteImageBan {
    /// The image's group
    pub group: String,
    /// The name of the image
    pub image: String,
    /// The image ban's unique ID
    pub id: Uuid,
}

/// The image notification specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum ImageNotifications {
    /// Get notifications for an image
    #[clap(version, author)]
    Get(GetImageNotifications),
    /// Create an image notification
    #[clap(version, author)]
    Create(CreateImageNotification),
    /// Delete an image notification
    #[clap(version, author)]
    Delete(DeleteImageNotification),
}

/// A command to get an image's notifications
#[derive(Parser, Debug, Clone)]
pub struct GetImageNotifications {
    /// The group the image belongs to
    pub group: String,
    /// The image to get notifications for
    pub image: String,
    /// The options for getting notifications
    #[clap(flatten)]
    pub opts: GetNotificationOpts,
}

/// The args related to creating image notifications
#[derive(Parser, Debug, Clone)]
pub struct CreateImageNotification {
    /// The image's group
    pub group: String,
    /// The name of the image
    pub image: String,
    /// The params needed when creating a notification
    #[clap(flatten)]
    pub notification: CreateNotification,
}

/// The args related to deleting image notifications
#[derive(Parser, Debug, Clone)]
pub struct DeleteImageNotification {
    /// The image's group
    pub group: String,
    /// The name of the image
    pub image: String,
    /// The notification's unique ID
    pub id: Uuid,
}

/// A command to delete images
#[derive(Parser, Debug, Clone)]
pub struct DeleteImages {
    /// The images to delete
    #[clap(required = true, value_parser = NonEmptyStringValueParser::new())]
    pub images: Vec<String>,
    /// The group to delete images from
    #[clap(short, long, required = true)]
    pub group: String,
    /// Skip the confirmation dialog
    #[clap(long)]
    pub skip_confirm: bool,
}

/// A command to import images
#[derive(Parser, Debug, Clone)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub struct ImportImages {
    /// The images to import, by name (default: every image config in the import directory)
    #[clap(value_name = "IMAGE", value_parser = NonEmptyStringValueParser::new())]
    pub images: Vec<String>,
    /// The group to import images to
    #[clap(short, long, value_name = "GROUP", required = true)]
    pub group: String,
    /// The directory to import images from
    #[clap(short, long, value_name = "DIR", required = true)]
    pub import: PathBuf,
    /// The registry to upload these images to
    #[clap(short, long, value_name = "REGISTRY")]
    pub registry: Option<String>,
    /// The registry url to override the domain stored in Thorium with
    #[clap(long, value_name = "URL")]
    pub registry_override: Option<String>,
    /// Skip pushing images to docker
    #[clap(long)]
    pub skip_push: bool,
    /// Just update the registry
    #[clap(long, conflicts_with_all = ["overwrite", "skip_conflicts"])]
    pub migrate_registry: bool,
    /// Overwrite existing images without opening the editor
    #[clap(long, conflicts_with = "skip_conflicts")]
    pub overwrite: bool,
    /// Skip existing images that differ instead of updating them
    ///
    /// Each skipped image logs a warning listing the fields that differ,
    /// making this safe for non-interactive imports that must never
    /// overwrite local changes.
    #[clap(long)]
    pub skip_conflicts: bool,
    /// Automatically roll back applied changes if the import stops early
    ///
    /// Only applies when the session can't prompt; interactive sessions are
    /// asked instead. Registry pushes are not undone, only Thorium state.
    #[clap(long)]
    pub rollback_on_failure: bool,
    /// Override the default editor for reviewing merge conflicts
    #[clap(long, value_name = "EDITOR")]
    pub editor: Option<String>,
}

/// A command to export images
#[derive(Parser, Debug, Clone)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub struct ExportImages {
    /// The images to export, by name (default: every image in the group)
    #[clap(value_name = "IMAGE", value_parser = NonEmptyStringValueParser::new())]
    pub images: Vec<String>,
    /// The group to export images from
    #[clap(short, long, value_name = "GROUP", required = true)]
    pub group: String,
    /// The directory to export images to (default: exports)
    #[clap(short, long, value_name = "DIR", default_value = "exports")]
    pub output: PathBuf,
    /// Only export image configs with no docker images
    #[clap(long)]
    pub config_only: bool,
    /// Overwrite existing on-disk configs that differ without prompting
    #[clap(long, conflicts_with = "skip_conflicts")]
    pub overwrite: bool,
    /// Skip on-disk conflicts: write new configs and leave differing existing ones
    /// untouched with a warning (use --overwrite to overwrite instead)
    #[clap(long)]
    pub skip_conflicts: bool,
    /// Open each config in an editor to review/tweak it before writing
    #[clap(long)]
    pub review: bool,
}
