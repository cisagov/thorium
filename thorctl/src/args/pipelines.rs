//! Arguments for pipeline-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use std::path::PathBuf;

use clap::Parser;
use uuid::Uuid;

use crate::utils;

use super::traits::describe::{DescribeCommand, DescribeSealed};
use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};
use super::{CreateNotification, GetNotificationOpts};

/// The commands to send to the pipelines task handler
#[derive(Parser, Debug)]
pub enum Pipelines {
    /// Get available pipelines and their details
    #[clap(version, author)]
    Get(GetPipelines),
    /// Describe specific pipelines, displaying/saving details in JSON format
    #[clap(version, author)]
    Describe(DescribePipelines),
    /// Manage/list pipeline notifications
    #[clap(subcommand)]
    Notifications(PipelineNotifications),
    /// Manage/list pipeline bans
    #[clap(subcommand)]
    Bans(PipelineBans),
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
    pub limit: u64,
    /// The page size to use in retrieving the pipelines
    #[clap(short, long, default_value = "50")]
    pub page_size: u64,
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
    #[clap(short, long)]
    pub pipeline_list_path: Option<PathBuf>,
    /// The path to the file to write output to; if not provided, details will be output to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Output details in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
    /// Any specific groups to filter by when describing pipelines
    #[clap(short, long)]
    pub groups: Vec<String>,
    /// Describe all pipelines to which you have access (still within the limit given in `--limit`)
    #[clap(long)]
    pub describe_all: bool,
    /// The maximum number of pipelines to retrieve per group
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Describe pipelines with no limit
    #[clap(long)]
    pub no_limit: bool,
    /// The number of pipelines to retrieve per request
    #[clap(long, default_value_t = 50)]
    pub page_size: usize,
}

impl SearchSealed for DescribePipelines {
    fn get_search_params(&self) -> SearchParams {
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
        !self.pipelines.is_empty() || self.pipeline_list_path.is_some()
    }

    fn apply_to_all(&self) -> bool {
        self.describe_all
    }
}

/// A specific pipeline target containing an optional group in case
/// more than one group has an image with the same name
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

    fn out_path(&self) -> Option<&std::path::PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&std::path::PathBuf> {
        self.pipeline_list_path.as_ref()
    }

    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error> {
        PipelineTarget::parse(raw, ':')
    }

    async fn retrieve_data<'a>(
        &self,
        target: Self::Target<'a>,
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
        let limit: u64 = if params.no_limit {
            // TODO: use a really big limit if the user wants no limit; cursor doesn't currently
            //       allow for no limits
            super::traits::describe::CURSOR_BIG_LIMIT
        } else {
            params.limit as u64
        };
        Ok(groups
            .iter()
            .map(|group| {
                thorium
                    .pipelines
                    .list(group)
                    .details()
                    .page(params.page_size as u64)
                    .limit(limit)
            })
            .collect())
    }
}

impl DescribeCommand for DescribePipelines {}

/// The pipeline ban specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum PipelineBans {
    /// Add a ban to a pipeline, preventing it from being run
    #[clap(version, author)]
    Add(AddPipelineBan),
    /// Remove a ban from a pipeline
    #[clap(version, author)]
    Remove(RemovePipelineBan),
}

/// The args related to adding pipeline bans
#[derive(Parser, Debug, Clone)]
pub struct AddPipelineBan {
    /// The pipeline's group
    pub group: String,
    /// The name of the pipeline
    pub pipeline: String,
    /// The message explaining why the pipeline was banned
    pub msg: String,
}

/// The args related to removing pipeline bans
#[derive(Parser, Debug, Clone)]
pub struct RemovePipelineBan {
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

/// A command to export pipeliness
#[derive(Parser, Debug, Clone)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub struct ExportPipelines {
    /// The pipelines to export
    pub pipelines: Vec<String>,
    /// The group to export pipelines from
    #[clap(short, long, required = true)]
    pub group: String,
    /// The directory to export our pipelines too
    #[clap(short, long, default_value = "exports")]
    pub output: PathBuf,
}

/// A command to import pipeliness
#[derive(Parser, Debug, Clone)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub struct ImportPipelines {
    /// The pipelines to import
    pub pipelines: Vec<String>,
    /// The group to import pipelines to
    #[clap(short, long, required = true)]
    pub group: String,
    /// The directory to import our pipelines from
    #[clap(short, long, required = true)]
    pub import: PathBuf,
    /// The registry to upload these images too
    #[clap(short, long)]
    pub registry: Option<String>,
    /// The registry url to override our domain in thorium with
    #[clap(short, long)]
    pub registry_override: Option<String>,
    /// Skip pushing images to docker
    #[clap(long)]
    pub skip_push: bool,
    /// Just update the registry
    #[clap(long)]
    pub migrate_registry: bool,
}
