//! Arguments for reaction-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use std::borrow::ToOwned;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use clap::builder::NonEmptyStringValueParser;
use clap::Parser;
use itertools::Itertools;
use thorium::models::{GenericJobArgs, GenericJobKwargs, GenericJobOpts, Reaction, ReactionArgs};
use thorium::{Error, Thorium};
use uuid::Uuid;

use crate::handlers::reactions::create::ReactionArgsInfo;
use crate::utils;

use super::pipelines::PipelineTarget;
use super::traits::describe::DescribeSealed;
use super::traits::search::{SearchParameterized, SearchParams, SearchSealed};
use super::DescribeCommand;

/// The commands to send to the reactions task handler
#[derive(Parser, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Reactions {
    /// Get information on specific reactions or find reactions
    #[clap(version, author)]
    Get(GetReactions),
    /// Deletes reactions by id or with a search
    #[clap(version, author)]
    Delete(GetReactions),
    /// Describe specific reactions, displaying/saving details in JSON format
    #[clap(version, author)]
    Describe(DescribeReactions),
    /// Retrieve reaction logs, writing them to organized subdirectories or printing them to
    /// stdout
    #[clap(version, author)]
    Logs(LogsReactions),
    /// Create reactions
    #[clap(version, author)]
    Create(CreateReactions),
}

/// A command to get info on some reactions
#[derive(Parser, Debug)]
pub struct GetReactions {
    /// Any specific reactions to get info about
    pub targets: Vec<Uuid>,
    /// The group to limit our scope too
    #[clap(short, long)]
    pub group: String,
    /// The pipeline to retrieve reactions for
    #[clap(short, long)]
    pub pipeline: Option<String>,
    /// The status to focus on for status specific operations
    #[clap(short, long)]
    pub status: Option<String>,
    /// The tag to focus on for tag specific operations
    #[clap(short, long)]
    pub tag: Option<String>,
    /// Print detailed information on any reactions returned
    #[clap(short, long)]
    pub details: bool,
    /// The max number of reactions to list
    #[clap(short, long, default_value = "50")]
    pub limit: u64,
}

/// A command to describe particular reactions in full
#[derive(Parser, Debug)]
pub struct DescribeReactions {
    /// Any specific ID's of reactions to describe, optionally with a specific reaction group delimited
    /// with a colon to more easily find the reaction
    /// (e.g. '<REACTION-ID>:<OPTIONAL-GROUP>')
    pub reactions: Vec<String>,
    /// The path to a file containing specific reactions to describe delimited by newlines,
    /// optionally with a specific reaction group delimited with a colon to more easily find the reaction
    /// (e.g. '<REACTION-ID>:<OPTIONAL-GROUP>')
    #[clap(short, long)]
    pub reaction_list: Option<PathBuf>,
    /// The path to the file to write output to; if not provided, details will be output to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Output details in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
    /// Any pipelines to describe reactions for
    ///     Note: Unlike other thorctl commands, `--pipelines` and `--tags` use OR logic rather than AND, meaning
    ///           "--pipelines harvest --tags Corn" will describe reactions from the "harvest" pipeline as well as
    ///           reactions with the "Corn" tag
    // TODO: Update this once listing reactions by group/tag/pipeline (like ReactionListOpts) is implemented
    #[clap(short, long, verbatim_doc_comment)]
    pub pipelines: Vec<String>,
    /// Any groups to filter by when searching for reactions to describe using `--pipelines` and `--tags`
    ///     Note: If no groups are given, the search will include all groups the user is apart of
    // TODO: Update this once listing reactions by group/tag/pipeline (like ReactionListOpts) is implemented
    #[clap(short, long)]
    pub groups: Vec<String>,
    /// Any reaction tags to describe reactions for
    ///     Note: Unlike other thorctl commands, `--pipelines` and `--tags` use OR logic rather than AND, meaning
    ///           "--pipelines harvest --tags Corn" will describe reactions from the "harvest" pipeline as well as
    ///           reactions with the "Corn" tag
    // TODO: Update this once listing reactions by group/tag/pipeline (like ReactionListOpts) is implemented
    #[clap(short, long, verbatim_doc_comment)]
    pub tags: Vec<String>,
    /// The maximum number of reactions to retrieve per group
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Describe reactions with no limit
    #[clap(long)]
    pub no_limit: bool,
    /// The number of reactions to retrieve per request
    #[clap(long, default_value_t = 50)]
    pub page_size: usize,
}

impl SearchSealed for DescribeReactions {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            // tags for reactions are always case sensitive
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

impl SearchParameterized for DescribeReactions {
    fn has_targets(&self) -> bool {
        !self.reactions.is_empty() || self.reaction_list.is_some()
    }

    fn has_parameters(&self) -> bool {
        // reimplement to include pipelines
        let parameters = self.get_search_params();
        !parameters.tags.is_empty()
            || !parameters.groups.is_empty()
            || !self.pipelines.is_empty()
            || parameters.start.is_some()
            || parameters.end.is_some()
    }

    fn apply_to_all(&self) -> bool {
        false
    }
}

/// A specific reaction target containing an optional group in case
/// more than one group has an image with the same name
pub struct ReactionTarget {
    /// The name of the pipeline
    reaction_id: Uuid,
    /// The optional group that the pipeline belongs to
    group: Option<String>,
}

impl ReactionTarget {
    /// Get a reaction based on this [`ReactionTarget`]
    ///
    /// # Arguments
    ///
    /// * `thorium` - The Thorium client
    pub async fn get_reaction(&self, thorium: &Thorium) -> Result<Reaction, Error> {
        if let Some(group) = &self.group {
            // get the reaction with a group if one was provided
            thorium
                .reactions
                .get(group, &self.reaction_id)
                .await
                .map_err(|err| {
                    thorium::Error::new(format!(
                        "Unable to retrieve reaction '{}' in group '{}': {}",
                        self.reaction_id,
                        group,
                        err.msg().unwrap_or("an unknown error occurred".to_string()),
                    ))
                })
        } else {
            // attempt to find the reaction in all groups the user belongs to
            utils::reactions::find_reaction_no_group(thorium, &self.reaction_id).await
        }
    }

    fn parse(raw: &str, delimiter: char) -> Result<Self, thorium::Error> {
        let mut split = raw.split(delimiter);
        let Some(reaction_id_raw) = split.next() else {
            return Err(Error::new(format!(
                    "Unable to parse '{raw}' to reaction target! \
                    The target should be formatted as the reaction's name and optionally
                    the reaction's group delimited with a single colon (<REACTION>:<OPTIONAL-GROUP>)",
                )));
        };
        let Ok(reaction_id) = Uuid::parse_str(reaction_id_raw) else {
            return Err(Error::new(format!(
                "Unable to parse '{raw}' to reaction target! \
                    The target does not have a valid UUID"
            )));
        };
        let group = split.next();
        if split.next().is_some() {
            return Err(Error::new(format!(
                "Unable to parse '{raw}' to reaction target! \
                The target should be formatted as the reaction's name and optionally
                the reaction's group delimited with a single colon (<REACTION>:<OPTIONAL-GROUP>)",
            )));
        }
        Ok(Self {
            reaction_id,
            group: group.map(ToOwned::to_owned),
        })
    }
}

/// Parse a [`ReactionTarget`] from a &[`str`]
impl TryFrom<&str> for ReactionTarget {
    type Error = thorium::Error;
    fn try_from(raw: &str) -> Result<Self, Self::Error> {
        Self::parse(raw, ':')
    }
}

/// Parse a [`ReactionTarget`] from a &[`String`]
impl TryFrom<&String> for ReactionTarget {
    type Error = thorium::Error;
    fn try_from(raw: &String) -> Result<Self, Self::Error> {
        Self::parse(raw, ':')
    }
}

/// Parse a [`ReactionTarget`] from a [`String`]
impl TryFrom<String> for ReactionTarget {
    type Error = thorium::Error;
    fn try_from(raw: String) -> Result<Self, Self::Error> {
        Self::parse(raw.as_str(), ':')
    }
}

impl DescribeSealed for DescribeReactions {
    type Data = Reaction;

    type Target<'a> = ReactionTarget;

    type Cursor = thorium::client::Cursor<Self::Data>;

    fn raw_targets(&self) -> &[String] {
        &self.reactions
    }

    fn condensed(&self) -> bool {
        self.condensed
    }

    fn out_path(&self) -> Option<&PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&PathBuf> {
        self.reaction_list.as_ref()
    }

    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error> {
        ReactionTarget::parse(raw, ':')
    }

    async fn retrieve_data<'a>(
        &self,
        target: Self::Target<'a>,
        thorium: &Thorium,
    ) -> Result<Self::Data, thorium::Error> {
        // parse a reaction target from the raw string
        target.get_reaction(thorium).await
    }

    /// List all groups out in a cursor; this shouldn't be called unless
    /// `describe_all` is set
    async fn retrieve_data_search(
        &self,
        thorium: &Thorium,
    ) -> Result<Vec<Self::Cursor>, thorium::Error> {
        params_to_cursors(thorium, self, &self.pipelines).await
    }
}

impl DescribeCommand for DescribeReactions {}

/// Provide a range for the max number of log lines to get per stage
fn stage_logs_range(s: &str) -> Result<usize, String> {
    super::helpers::number_range(s, 0, 250_000)
}

/// A command to retrieve reaction logs
#[derive(Parser, Debug)]
pub struct LogsReactions {
    /// Any specific ID's of reactions to retrieve logs for, optionally with a specific reaction group delimited
    /// with a colon to more easily find the reaction
    /// (e.g. '<REACTION-ID>:<OPTIONAL-GROUP>')
    pub reactions: Vec<String>,
    /// The path to a file containing specific reactions to retrieve logs for delimited by newlines,
    /// optionally with a specific reaction group delimited with a colon to more easily find the reaction
    /// (e.g. '<REACTION-ID>:<OPTIONAL-GROUP>')
    #[clap(short, long)]
    pub reaction_list: Option<PathBuf>,
    /// Write logs to the given directory rather than printing to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Any pipelines to get reaction logs for
    ///
    /// Note: Unlike other thorctl commands, `--pipelines` and `--tags` use OR logic rather than AND, meaning
    /// "--pipelines harvest --tags Corn" will get logs for reactions from the "harvest" pipeline as well as
    /// reactions with the "Corn" tag
    // TODO: Update this once listing reactions by group/tag/pipeline (like ReactionListOpts) is implemented
    #[clap(short, long)]
    pub pipelines: Vec<String>,
    /// Any groups to filter by when searching for reactions to get logs for using `--pipelines` and `--tags`
    ///
    /// Note: If no groups are given, the search will include all groups the user is apart of
    // TODO: Update this once listing reactions by group/tag/pipeline (like ReactionListOpts) is implemented
    #[clap(short, long)]
    pub groups: Vec<String>,
    /// Any reaction tags to get reaction logs for
    ///
    /// Note: Unlike other thorctl commands, `--pipelines` and `--tags` use OR logic rather than AND, meaning
    /// "--pipelines harvest --tags Corn" will get logs for reactions from the "harvest" pipeline as well as
    /// reactions with the "Corn" tag
    // TODO: Update this once listing reactions by group/tag/pipeline (like ReactionListOpts) is implemented
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// The maximum number of reactions to get logs for per group/tag/pipeline
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Describe reactions with no limit
    #[clap(long)]
    pub no_limit: bool,
    /// The maximum number of log lines to retrieve per stage (maximum 250000)
    // TODO: ReactionListParams has no no_limit option, so we need to set a high limit to ensure we
    // get all logs; currently the highest that can be is 250,000
    #[clap(long, default_value_t = 250_000, value_parser = stage_logs_range)]
    pub log_limit: usize,
    /// The number of reactions to retrieve per request
    #[clap(long, default_value_t = 50)]
    pub page_size: usize,
}

impl SearchSealed for LogsReactions {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            // tags for reactions are always case sensitive
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

impl SearchParameterized for LogsReactions {
    fn has_targets(&self) -> bool {
        !self.reactions.is_empty() || self.reaction_list.is_some()
    }

    fn has_parameters(&self) -> bool {
        // reimplement to include pipelines
        let parameters = self.get_search_params();
        !parameters.tags.is_empty()
            || !parameters.groups.is_empty()
            || !self.pipelines.is_empty()
            || parameters.start.is_some()
            || parameters.end.is_some()
    }

    fn apply_to_all(&self) -> bool {
        false
    }
}

/// The delimiter to use for separating SHA256's in file bundles
pub const BUNDLE_DELIMITER: char = ',';

/// A command to get info on some reactions
#[derive(Parser, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct CreateReactions {
    /// The pipelines to run, optionally specified with a group delimited with
    /// ':' if more than one pipeline exists with the same name (i.e. <PIPELINE>:<GROUP>)
    #[clap(short, long, value_delimiter = ',', required = true, value_parser = NonEmptyStringValueParser::new())]
    pub pipelines: Vec<String>,
    /// Any tags to set for the created reactions
    #[clap(short = 'T', long)]
    pub reaction_tags: Vec<String>,
    /// Any specific files to create reactions for (optional)
    ///
    /// Note: Explicitly specified files always have jobs run, even if they
    ///       don't match the search criteria (i.e. don't have matching tags)
    #[clap(value_delimiter = ',')]
    pub files: Vec<String>,
    /// The path to a file containing a list of file SHA256's to create reactions for,
    /// delimited by newlines; to designate multiple SHA256's in a single reaction,
    /// separate each file with ',' in a single line
    #[clap(long)]
    pub file_list: Option<PathBuf>,
    /// Any bundles of files to create reactions for with each file in a bundle
    /// delimited by ','; the created reaction will have access to all files in a bundle
    ///
    /// Note: To specify multiple file bundles, use the flag multiple times:
    ///       (e.g. '--file-bundles <SHA1>,<SHA2> --file-bundles <SHA3>,<SHA4>,<SHA5>')
    #[clap(long)]
    pub file_bundles: Vec<String>,
    /// Any specific repos + optionally commits to run these jobs on
    ///
    /// Note: Repo commits are formatted with a colon after the repo URL
    ///       (i.e. "<REPO-URL>:<COMMIT-HASH>)"
    #[clap(short, long)]
    pub repos: Vec<String>,
    /// Create reactions only for repos with the given search criteria (i.e. tags)
    #[clap(short = 'R', long)]
    pub repos_only: bool,
    /// The path to a file containing a list of repo URL's to create reactions for,
    /// delimited by newlines
    #[clap(long)]
    pub repo_list: Option<PathBuf>,
    /// Include repos in the search and run reactions on them
    #[clap(long)]
    pub include_repos: bool,
    /// The parent reaction to set in order to create sub reactions
    #[clap(long)]
    pub parent: Option<Uuid>,
    /// The optional SLA to set for the created reactions
    #[clap(long)]
    pub sla: Option<u64>,
    /// Any positional arguments to pass to the reaction's image(s) (may be delimited with ',')
    #[clap(long, conflicts_with = "reaction_args_file", value_delimiter = ',')]
    pub positionals: Vec<String>,
    /// Any keyword arguments to pass to the reaction's image(s) with key,values separated by a delimiter
    ///     Note: The delimiter for kwargs is the same delimiter for tags given by "--delimiter"
    ///           (e.g. --kwargs --my-kwarg=my-value1=my-value2)
    #[clap(long, conflicts_with = "reaction_args_file", verbatim_doc_comment)]
    pub kwargs: Vec<String>,
    /// Any switch arguments to pass to the reaction's image(s) (may be delimited with ',')
    ///
    /// Note: Switch arguments may conflict with Thorctl arguments; using "=" is recommended
    /// (e.g. "--switches=--arg1,--arg2,-a" or "--switches=--arg1 --switches=-a")
    #[clap(long, conflicts_with = "reaction_args_file", value_delimiter = ',')]
    pub switches: Vec<String>,
    /// Override/replace positional arguments in the image rather than simply adding them
    #[clap(long, conflicts_with = "reaction_args_file", default_value = "false")]
    pub override_positionals: bool,
    /// Override/replace keyword arguments in the image rather than simply adding them
    #[clap(long, conflicts_with = "reaction_args_file", default_value = "false")]
    pub override_kwargs: bool,
    /// An explicit command to send to the image(s), overriding any previously configured commands/arguments
    #[clap(long, conflicts_with = "reaction_args_file")]
    pub override_cmd: Option<String>,
    /// Apply args (kwargs, positionals, flags, override-cmd, etc.) to all images in all pipelines
    ///
    /// Note: This flag must be set if pipe
    #[clap(long)]
    pub apply_args_to_all: bool,
    /// The path to a JSON-formatted file containing explicit arguments/commands for images in the pipeline;
    /// run `thorctl reactions create --help` for formatting guidance
    #[rustfmt::skip]
    #[clap(
        long,
        verbatim_doc_comment,
        long_help =
            "The path to a JSON-formatted file containing explicit arguments/commands for images in the pipelines\n\n    \
            Example file:

            \"pipeline1\": {
                \"image1\": {
                    \"positionals\": [\"positional1\", \"positional2\"],
                    \"opts\": {
                        \"override_positionals\": false
                    }
                },
                \"image2\": {
                    \"kwargs\": {
                        \"--kwarg-key\": [
                            \"kwarg-val1\",
                            \"kwarg-val2\"
                        ]
                    },
                    \"switches\": [\"--switch\"],
                    \"opts\": {
                        \"override_kwargs\": false,
                        \"override_cmd\": null
                    }
                },
                \"image3\" : {
                    \"opts\": {
                        \"override_cmd\": [
                            \"ls\",
                            \"/\",
                            \"-al\"
                        ]
                    }
                }
            }"
    )]
    pub reaction_args_file: Option<PathBuf>,
    /// Batch these reactions together
    #[clap(short, long)]
    pub batch: Option<String>,
    /// Display files/repos that will have reactions created for them without
    /// actually creating the reactions
    #[clap(long)]
    pub dry_run: bool,
    /// Watch all spawned reactions progress and automatically batch them
    #[clap(short = 'W', long)]
    pub watch: bool,
    /// Any groups to filter by when searching for files/repos to execute on
    ///
    /// Note: If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long)]
    pub groups: Vec<String>,
    /// Any tags to filter by when searching for samples/repos to execute on
    #[clap(short, long)]
    pub tags: Vec<String>,
    /// Whether matching on sample/repo tags should be case-insensitive
    #[clap(short = 'c', long, default_value_t = false)]
    pub tags_case_insensitive: bool,
    /// The delimiter character to use when splitting tags/kwargs into key/values
    /// (i.e. <TAG>=<VALUE1>=<VALUE2>=<VALUE3>)
    #[clap(long, default_value = "=")]
    pub delimiter: char,
    /// The most recent datetime to start searching at in UTC
    #[clap(short, long)]
    pub start: Option<String>,
    /// The oldest datetime to stop searching at in UTC
    #[clap(short, long)]
    pub end: Option<String>,
    /// The format string to use when parsing the start/end datetimes
    ///
    /// Example: The format of "2014-5-17T12:34:56" is "%Y-%m-%dT%H:%M:%S"
    /// (see <https://docs.rs/chrono/latest/chrono/format/strftime>)
    #[clap(long, default_value = "%Y-%m-%dT%H:%M:%S")]
    pub date_fmt: String,
    /// Create reactions for all files/repos with no search filter
    ///
    /// Note: This, combined with '--no-limit', will create reactions for ALL
    /// files/repos to which you have access. Be careful!
    #[clap(long, default_value = "false")]
    pub create_for_all: bool,
    /// The maximum number of reactions to spawn
    ///
    /// Note: This limit applies to files and repos separately, meaning if `--include-repos` is set,
    /// the number of reactions spawned may be greater than `--limit`
    #[clap(short, long, default_value = "1000000")]
    pub limit: usize,
    /// Refrain from setting a limit on the number of reactions that can be spawned
    ///
    /// Note: Creating reactions based on tags with no limit can lead
    /// to millions of reactions being created at once. Be careful!
    #[clap(long, conflicts_with = "limit")]
    pub no_limit: bool,
    /// The number of reactions to create in one request
    #[clap(long, default_value = "100")]
    pub page_size: usize,
}

impl CreateReactions {
    pub fn parse_pipelines(&self) -> Result<Vec<PipelineTarget>, Error> {
        self.pipelines
            .iter()
            .map(|raw| PipelineTarget::parse(raw, ':'))
            .collect::<Vec<Result<_, _>>>()
            .into_iter()
            .collect()
    }

    /// If batch or watch is enabled then get our batch name
    pub fn batch_name(&self) -> Option<String> {
        match (&self.batch, &self.watch) {
            (Some(batch), _) => Some(batch.to_owned()),
            (None, true) => Some(Uuid::new_v4().to_string()),
            _ => None,
        }
    }

    /// Parse kwargs from the command by the command's delimiter
    pub fn parse_kwargs(&self) -> Result<GenericJobKwargs, Error> {
        let mut kwargs = GenericJobKwargs::new();
        for raw_kwarg in &self.kwargs {
            let mut split = raw_kwarg.split(self.delimiter);
            let key = split.next().ok_or(Error::new(format!(
                "Invalid kwarg \"{raw_kwarg}\": kwarg is improperly delimited!",
            )))?;
            let values: Vec<String> = split.map(str::to_string).collect();
            if values.is_empty() {
                return Err(Error::new(format!(
                    "Invalid kwarg \"{raw_kwarg}\": kwarg has no values!"
                )));
            }
            kwargs.insert(key.to_owned(), values);
        }
        Ok(kwargs)
    }

    /// Returns true if the command contains any reaction args
    pub fn has_reaction_args(&self) -> bool {
        !self.positionals.is_empty()
            || !self.kwargs.is_empty()
            || !self.switches.is_empty()
            || self.override_positionals
            || self.override_kwargs
            || self.override_cmd.is_some()
            || self.reaction_args_file.is_some()
    }

    /// Create [`ReactionArgs`] mapped to pipelines from a [`CreateReactions`] command,
    /// ensuring the command is set to apply args to all images if there are multiple
    /// images in the pipeline (or multiple pipelines)
    ///
    /// # Arguments
    ///
    /// * `pipelines_images` - A map of pipelines to the images it contains
    pub fn build_reaction_args(
        &self,
        pipelines_images: &HashMap<String, HashSet<String>>,
    ) -> Result<ReactionArgsInfo, Error> {
        // read the args from a reaction args file if one was provided
        if let Some(reaction_args_file) = &self.reaction_args_file {
            let file = match std::fs::File::open(reaction_args_file) {
                Ok(file) => file,
                Err(err) => {
                    return Err(Error::new(format!(
                        "Error opening reaction args file at '{}': {}",
                        reaction_args_file.to_string_lossy(),
                        err
                    )))
                }
            };
            let args: ReactionArgsInfo = match serde_json::from_reader(file) {
                Ok(args) => args,
                Err(_) => {
                    return Err(Error::new(format!(
                        "Reaction args file at '{}' is not formatted correctly! \
                        See `thorctl reactions create --help` for a formatting example",
                        reaction_args_file.to_string_lossy()
                    )))
                }
            };
            for (arg_pipeline, arg_images) in &args {
                let images = pipelines_images
                    .get(arg_pipeline)
                    .ok_or(Error::new(format!(
                        "Unable to get image info for pipeline '{arg_pipeline}'"
                    )))?;
                for arg_image in arg_images.keys() {
                    if !images.contains(arg_image) {
                        return Err(Error::new(format!(
                            "Invalid reaction args file! Image '{arg_image}' is not in the pipeline '{arg_pipeline}'",
                        )));
                    }
                }
            }
            Ok(args)
        } else {
            if pipelines_images.len() > 1 && !self.apply_args_to_all {
                // return an error if args were set, we have multiple images, and --apply-args-to-all is not set
                return Err(Error::new(
                    "The reaction pipelines have more than one image! \
                    Args cannot be applied to pipelines with more than one image
                    or to multiple pipelines if '--apply-args-to-all' is not set",
                ));
            }
            // otherwise build args from the individual args given in the reaction create command;
            // convert cmd string to a Vec<String> delimited by spaces
            let override_cmd: Option<Vec<String>> = self
                .override_cmd
                .as_ref()
                .map(|cmd_str| cmd_str.split(' ').map(str::to_string).collect());
            // create job args
            let job_args = GenericJobArgs::default()
                .positionals(self.positionals.clone())
                .set_kwargs(self.parse_kwargs()?)
                .switches(self.switches.clone())
                .opts(GenericJobOpts::new(
                    self.override_positionals,
                    self.override_kwargs,
                    override_cmd,
                ));
            // create a new map of pipelines to reaction args
            let mut reaction_args = ReactionArgsInfo::new();
            for (pipeline, images) in pipelines_images {
                // create blank args
                let mut args = ReactionArgs::new();
                for image in images {
                    // insert args for every image in the pipeline
                    args.insert(image.clone(), job_args.clone());
                }
                // add the args to our map at the pipeline's name
                reaction_args.insert(pipeline.clone(), args);
            }
            Ok(reaction_args)
        }
    }
}

impl SearchParameterized for CreateReactions {
    fn has_targets(&self) -> bool {
        !self.files.is_empty()
            || !self.file_bundles.is_empty()
            || !self.repos.is_empty()
            || self.file_list.is_some()
            || self.repo_list.is_some()
    }
    fn apply_to_all(&self) -> bool {
        self.create_for_all
    }
}
impl SearchSealed for CreateReactions {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &self.tags,
            tags_case_insensitive: self.tags_case_insensitive,
            delimiter: self.delimiter,
            start: &self.start,
            end: &self.end,
            date_fmt: &self.date_fmt,
            cursor: None,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

/// Produce a list of reaction details search cursors based on the given search params and pipelines
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `search` - Any provider of [`SearchParams`] to use for generating the cursors
/// * `pipelines` - Any pipelines to find reactions for
pub async fn params_to_cursors(
    thorium: &Thorium,
    search: &impl SearchSealed,
    pipelines: &[String],
) -> Result<Vec<thorium::client::Cursor<Reaction>>, Error> {
    let params = search.get_search_params();
    let groups = if params.groups.is_empty() {
        // use all groups the user is a part of if none were given
        utils::groups::get_all_groups(thorium).await?
    } else {
        params.groups.to_vec()
    };
    let limit: u64 = if params.no_limit {
        // TODO: use a really big limit if the user wants no limit; cursor doesn't currently
        //       allow for no limits
        super::traits::describe::CURSOR_BIG_LIMIT
    } else {
        params.limit as u64
    };
    let mut cursors = Vec::new();
    if !pipelines.is_empty() {
        // add cursors to search for reactions for specific pipelines in every group
        let cursor_results =
            futures::future::join_all(pipelines.iter().cartesian_product(groups.iter()).map(
                |(pipeline, group)| async {
                    thorium
                        .reactions
                        .list(group, pipeline)
                        .page(params.page_size as u64)
                        .limit(limit)
                        .details()
                        // execute the cursors to check for 404 errors preemptively
                        .exec()
                        .await
                },
            ))
            .await;
        for res in cursor_results {
            match res {
                Ok(cursor) => cursors.push(cursor),
                Err(err) => match err {
                    Error::Thorium { code, msg } => {
                        // ignore 404 errors because we're checking for pipelines that may or may not
                        // exist in a given group
                        if code != 404 {
                            return Err(Error::Thorium { code, msg });
                        }
                    }
                    _ => return Err(err),
                },
            }
        }
    }
    if !params.tags.is_empty() {
        // add cursors to search for reactions for specific tags in every group
        cursors.extend(
            params
                .tags
                .iter()
                .cartesian_product(groups.iter())
                .map(|(tag, group)| {
                    thorium
                        .reactions
                        .list_tag(group, tag)
                        .page(params.page_size as u64)
                        .limit(limit)
                        .details()
                }),
        );
    }
    Ok(cursors)
}
