//! Arguments for pipeline-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use clap::Parser;
use clap::builder::NonEmptyStringValueParser;
use std::path::PathBuf;
use thorium::client::conf;
use uuid::Uuid;

use crate::utils;

use super::traits::describe::{DescribeCommand, DescribeFormat, DescribeSealed};
use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};
use super::{CreateNotification, GetNotificationOpts};

/// The commands to send to the pipelines task handler
#[derive(Parser, Debug)]
pub enum Pipelines {
    /// Get available pipelines and their details
    #[clap(version, author)]
    Get(GetPipelines),
    /// Describe specific pipelines in a human-readable format (use `--format json`
    /// for the full raw config)
    #[clap(version, author)]
    Describe(DescribePipelines),
    /// Edit/update a pipeline
    ///
    /// Group, name, and bans are not editable here
    #[clap(version, author, verbatim_doc_comment)]
    Edit(EditPipeline),
    /// Manage/list pipeline notifications
    #[clap(subcommand)]
    Notifications(PipelineNotifications),
    /// Manage/list pipeline bans
    #[clap(subcommand)]
    Bans(PipelineBans),
    /// Delete pipelines
    #[clap(version, author)]
    Delete(DeletePipelines),
    /// Import pipelines
    #[clap(version, author)]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    Import(ImportPipelines),
    /// Export pipelines
    #[clap(version, author)]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    Export(ExportPipelines),
}

/// A command to get info on some pipelines
#[derive(Parser, Debug)]
pub struct GetPipelines {
    /// Any groups to filter by when searching for pipelines
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',', verbatim_doc_comment)]
    pub groups: Vec<String>,
    /// The max number of pipelines to list per group
    #[clap(short, long, default_value = "50")]
    pub limit: usize,
    /// Refrain from setting a limit when retrieving pipelines
    ///     Note: This can lead to retrieving info for many thousands of pipelines
    ///           inadvertently. Be careful!
    #[clap(long, verbatim_doc_comment)]
    pub no_limit: bool,
    /// The page size to use in retrieving the pipelines
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// Print the pipelines in alphabetical order rather than by group, then creation date
    #[clap(short, long)]
    pub alpha: bool,
}

/// A command to describe specific pipelines in full
#[derive(Parser, Debug)]
pub struct DescribePipelines {
    /// Any specific pipelines to describe, optionally with a specific group delimited
    /// with a colon in case other groups have a pipeline with the same name
    /// (e.g. '<PIPELINE>:<OPTIONAL-GROUP>')
    pub pipelines: Vec<String>,
    /// The path to a file containing a list of pipelines to describe separated by newlines;
    /// optionally, each pipeline can have a specific group delimited with a colon in case
    /// other groups have a pipeline with the same name
    /// (e.g. '<PIPELINE>:<OPTIONAL-GROUP>')
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
    /// Any specific groups to filter by when describing pipelines
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// Describe all pipelines to which you have access (still within the limit given in `--limit`)
    #[clap(long)]
    pub all: bool,
    /// The maximum number of pipelines to retrieve per group
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Describe pipelines with no limit
    #[clap(long)]
    pub no_limit: bool,
    /// The number of pipelines to retrieve per request
    #[clap(short, long, default_value_t = 50)]
    pub page_size: usize,
}

impl SearchSealed for DescribePipelines {
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

impl SearchParameterized for DescribePipelines {
    fn has_targets(&self) -> bool {
        !self.pipelines.is_empty() || self.list.is_some()
    }

    fn apply_to_all(&self) -> bool {
        self.all
    }
}

/// A specific pipeline target containing an optional group in case
/// more than one group has a pipeline with the same name
pub struct PipelineTarget {
    /// The name of the pipeline
    pub pipeline: String,
    /// The optional group that the pipeline belongs to
    pub group: Option<String>,
}

impl PipelineTarget {
    pub fn parse(raw: &str, delimiter: char) -> Result<Self, thorium::Error> {
        let mut split = raw.split(delimiter);
        let pipeline = split.next();
        let group = split.next();
        match (pipeline, split.next()) {
            // no pipeline was given or there was more than one delimiter, so return an error
            (None, _) | (_, Some(_)) => Err(thorium::Error::new(
                    format!("Unable to parse '{raw}' to pipeline target! \
                    The target should be formatted as the pipeline's name and optionally
                    the pipeline's group delimited with a single colon (<PIPELINE>:<OPTIONAL-GROUP>)",
            ))),
            (Some(pipeline), None) =>
                Ok(PipelineTarget {
                    pipeline: pipeline.to_owned(),
                    group: group.map(ToOwned::to_owned),
                })
        }
    }
}

impl DescribeSealed for DescribePipelines {
    type Data = thorium::models::Pipeline;

    type Target<'a> = PipelineTarget;

    type Cursor = thorium::client::Cursor<Self::Data>;

    fn raw_targets(&self) -> &[String] {
        &self.pipelines
    }

    fn condensed(&self) -> bool {
        self.condensed
    }

    fn format(&self) -> Result<DescribeFormat, thorium::Error> {
        // `--condensed` is a JSON modifier, so it forces JSON
        if self.condensed {
            return Ok(DescribeFormat::Json);
        }
        // pipelines can only be rendered as JSON or human-readable text; reject any
        // other format the shared DescribeFormat enum may offer
        if matches!(self.format, DescribeFormat::Json | DescribeFormat::Human) {
            Ok(self.format)
        } else {
            Err(thorium::Error::new(
                "describe pipelines only supports the 'json' or 'human' output format",
            ))
        }
    }

    fn render_human(
        &self,
        datum: &Self::Data,
        out: &mut dyn std::io::Write,
        ansi: bool,
    ) -> Result<(), thorium::Error> {
        utils::pipelines::print_pipeline_details(datum, out, ansi)
    }

    fn out_path(&self) -> Option<&std::path::PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&std::path::PathBuf> {
        self.list.as_ref()
    }

    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error> {
        PipelineTarget::parse(raw, ':')
    }

    async fn retrieve_data(
        &self,
        target: Self::Target<'_>,
        thorium: &thorium::Thorium,
    ) -> Result<Self::Data, thorium::Error> {
        let group = if let Some(group) = &target.group {
            group.clone()
        } else {
            utils::pipelines::find_pipeline_group(thorium, &target.pipeline).await?
        };
        thorium.pipelines.get(&group, &target.pipeline).await
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
                .pipelines
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

impl DescribeCommand for DescribePipelines {}

/// Provide the help message for the editor arg
fn editor_help() -> String {
    format!(
        "The editor to use when editing the pipeline ('{}' by default); the default can be modified \
    using 'thorctl config --default-editor', but this flag overrides any set defaults",
        conf::default_default_editor()
    )
}

/// Args for editing a pipeline
#[derive(Parser, Debug)]
pub struct EditPipeline {
    /// The name of the pipeline to edit
    pub pipeline: String,
    /// The group the pipeline is in; required if other pipelines have
    /// the same name
    pub group: Option<String>,
    /// The editor to use when editing the pipeline
    #[clap(short, long, help = editor_help())]
    pub editor: Option<String>,
}

/// The pipeline ban specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum PipelineBans {
    /// Add a ban to a pipeline, preventing it from being run
    #[clap(version, author)]
    Create(CreatePipelineBan),
    /// Remove a ban from an pipeline
    #[clap(version, author)]
    Delete(DeletePipelineBan),
}

/// The args related to adding pipeline bans
#[derive(Parser, Debug, Clone)]
pub struct CreatePipelineBan {
    /// The pipeline's group
    pub group: String,
    /// The name of the pipeline
    pub pipeline: String,
    /// The message explaining why the pipeline was banned
    pub msg: String,
}

/// The args related to removing pipeline bans
#[derive(Parser, Debug, Clone)]
pub struct DeletePipelineBan {
    /// The pipeline's group
    pub group: String,
    /// The name of the pipeline
    pub pipeline: String,
    /// The pipeline ban's unique ID
    pub id: Uuid,
}

/// The pipeline notification specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum PipelineNotifications {
    /// Get notifications for a pipeline
    #[clap(version, author)]
    Get(GetPipelineNotifications),
    /// Create a pipeline notification
    #[clap(version, author)]
    Create(CreatePipelineNotification),
    /// Delete a pipeline notification
    #[clap(version, author)]
    Delete(DeletePipelineNotification),
}

/// A command to get a pipeline's notifications
#[derive(Parser, Debug, Clone)]
pub struct GetPipelineNotifications {
    /// The group the pipeline belongs to
    pub group: String,
    /// The pipeline to get notifications for
    pub pipeline: String,
    /// The options for getting notifications
    #[clap(flatten)]
    pub opts: GetNotificationOpts,
}

/// The args related to creating pipeline notifications
#[derive(Parser, Debug, Clone)]
pub struct CreatePipelineNotification {
    /// The pipeline's group
    pub group: String,
    /// The name of the pipeline
    pub pipeline: String,
    /// The params needed when creating a notification
    #[clap(flatten)]
    pub notification: CreateNotification,
}

/// The args related to deleting pipeline notifications
#[derive(Parser, Debug, Clone)]
pub struct DeletePipelineNotification {
    /// The pipeline's group
    pub group: String,
    /// The name of the pipeline
    pub pipeline: String,
    /// The notification's unique ID
    pub id: Uuid,
}

/// A command to delete pipelines
#[derive(Parser, Debug, Clone)]
pub struct DeletePipelines {
    /// The pipelines to delete
    #[clap(required = true, value_parser = NonEmptyStringValueParser::new())]
    pub pipelines: Vec<String>,
    /// The group to delete pipelines from
    #[clap(short, long, required = true)]
    pub group: String,
    /// Skip the confirmation dialog
    #[clap(long)]
    pub skip_confirm: bool,
}

/// A command to export pipelines
#[derive(Parser, Debug, Clone)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub struct ExportPipelines {
    /// The pipelines to export, by name (default: every pipeline in the group)
    #[clap(value_name = "PIPELINE", value_parser = NonEmptyStringValueParser::new())]
    pub pipelines: Vec<String>,
    /// The group to export pipelines from
    #[clap(short, long, value_name = "GROUP", required = true)]
    pub group: String,
    /// The directory to export pipelines to (default: exports)
    #[clap(short, long, value_name = "DIR", default_value = "exports")]
    pub output: PathBuf,
    /// Export pipeline/image configs only with no docker images
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

/// A command to import pipelines
#[derive(Parser, Debug, Clone)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub struct ImportPipelines {
    /// The pipelines to import, by name (default: every pipeline config in the import directory)
    #[clap(value_name = "PIPELINE", value_parser = NonEmptyStringValueParser::new())]
    pub pipelines: Vec<String>,
    /// The group to import pipelines to
    #[clap(short, long, value_name = "GROUP", required = true)]
    pub group: String,
    /// The directory to import pipelines from
    #[clap(short, long, value_name = "DIR", required = true)]
    pub import: PathBuf,
    /// The registry to upload these pipelines' images to
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
    /// Overwrite existing pipelines/images without opening the editor
    #[clap(long, conflicts_with = "skip_conflicts")]
    pub overwrite: bool,
    /// Skip existing pipelines/images that differ instead of updating them
    ///
    /// Each skipped resource logs a warning listing the fields that differ,
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
