use itertools::Itertools;
use std::collections::HashSet;
use thorium::CtlConf;
use thorium::models::PipelineRequest;
use thorium::{Error, Thorium, models::Pipeline};

use crate::args::pipelines::{DescribePipelines, GetPipelines, Pipelines};
use crate::args::{Args, DescribeCommand};
use crate::utils;

mod bans;
mod edit;
mod notifications;

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "macos"))] {
        use std::io::IsTerminal;

        use futures::stream::{self, StreamExt};
        use crate::args::pipelines::{ExportPipelines, ImportPipelines};
        use crate::args::images::ExportImages;
        use crate::handlers::imports::{self, ConflictMode};
        use crate::handlers::images::import::{ImageImportOpts, categorize_from_disk};
        use crate::handlers::exports::{DiskConflictResolver, WriteOutcome};
        use crate::handlers::progress::{Bar, BarKind};

        pub(crate) mod import;
    }
}

struct GetPipelinesLine;

impl GetPipelinesLine {
    /// Print this log lines header
    pub fn header() {
        println!(
            "{:<30} | {:<20} | {:<50}",
            "PIPELINE NAME", "GROUP", "DESCRIPTION",
        );
        println!("{:-<31}+{:-<22}+{:-<50}", "", "", "");
    }

    /// Print a pipeline's info
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The pipeline to print
    pub fn print_pipeline(pipeline: &Pipeline) {
        // limit our description preview to at most 40 characters so the column stays aligned
        let description = utils::render::truncate_description(pipeline.description.as_deref(), 40);
        // print our pipeline info
        println!(
            "{:<30} | {:<20} | {}",
            pipeline.name, pipeline.group, description
        );
    }
}

/// Get pipeline info from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The pipeline get command to execute
async fn get(thorium: Thorium, cmd: &GetPipelines) -> Result<(), Error> {
    GetPipelinesLine::header();
    // get the current user's groups if no groups were specified
    let groups = if cmd.groups.is_empty() {
        utils::groups::get_all_groups(&thorium).await?
    } else {
        cmd.groups.clone()
    };
    // get pipeline cursors for all groups specified
    let pipeline_cursors = groups.iter().map(|group| {
        let cursor = thorium
            .pipelines
            .list(group)
            .page_size(cmd.page_size as u64)
            .details();
        // only apply a limit if the user didn't request no limit
        if cmd.no_limit {
            cursor
        } else {
            cursor.limit(cmd.limit as u64)
        }
    });
    // retrieve the pipelines in each cursor until we've reached our limit
    // or all cursors are exhausted
    let mut pipelines: Vec<Pipeline> = Vec::new();
    for mut cursor in pipeline_cursors {
        while !cursor.exhausted {
            cursor.next().await?;
            if cmd.alpha {
                // save for later if we need to alphabetize
                pipelines.append(&mut cursor.details);
            } else {
                // print immediately if no need to alphabetize
                cursor
                    .details
                    .iter()
                    .for_each(GetPipelinesLine::print_pipeline);
            }
        }
    }
    // sort and print in alphabetical order if alpha flag was set
    if cmd.alpha {
        pipelines
            .iter()
            .sorted_unstable_by(|a, b| Ord::cmp(&a.name, &b.name))
            .for_each(GetPipelinesLine::print_pipeline);
    }
    Ok(())
}

/// Describe pipelines by displaying/saving all of their JSON-formatted details
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The describe pipeline command to execute
async fn describe(thorium: Thorium, cmd: &DescribePipelines) -> Result<(), Error> {
    cmd.describe(&thorium).await
}

/// Delete pipelines from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The delete pipelines command to execute
async fn delete(
    thorium: Thorium,
    cmd: &crate::args::pipelines::DeletePipelines,
) -> Result<(), Error> {
    use colored::Colorize;
    // deleting is irreversible, so confirm exactly what will be removed
    if !cmd.skip_confirm {
        // fail clearly (not with a raw dialoguer error) when we can't prompt
        utils::require_confirm_terminal("--skip-confirm (-y)")?;
        println!("{}", "Pipelines to delete:".bright_red());
        for pipeline in &cmd.pipelines {
            println!("  {}:{}", cmd.group, pipeline);
        }
        let confirmed = dialoguer::Confirm::new()
            .with_prompt("Delete the pipelines listed above?")
            .default(false)
            .interact()?;
        if !confirmed {
            return Ok(());
        }
    }
    for pipeline in &cmd.pipelines {
        match thorium.pipelines.delete(&cmd.group, pipeline).await {
            Ok(_) => println!("Deleted pipeline '{}:{}'", cmd.group, pipeline),
            // a missing pipeline isn't fatal to the rest of the batch
            Err(err) if err.status() == Some(http::StatusCode::NOT_FOUND) => {
                eprintln!(
                    "{}: pipeline '{}:{}' not found; skipping",
                    "Warning".bright_yellow(),
                    cmd.group,
                    pipeline
                );
            }
            Err(err) => {
                return Err(Error::new(format!(
                    "Failed to delete pipeline '{}:{}': {err}",
                    cmd.group, pipeline
                )));
            }
        }
    }
    Ok(())
}

/// Import pipelines and the images they reference to Thorium
///
/// The whole import — images first, then pipelines — shares one confirmation
/// screen and one rollback journal, so stopping partway (editor Quit or an
/// error) can offer to undo everything applied so far.
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The import pipelines command to execute
/// * `conf` - The Thorctl config
/// * `workers` - The maximum number of concurrent workers to use
#[cfg(any(target_os = "linux", target_os = "macos"))]
async fn import(
    thorium: &Thorium,
    cmd: &ImportPipelines,
    conf: &CtlConf,
    workers: usize,
) -> Result<(), Error> {
    let progress = Bar::new("", "Importing pipelines", BarKind::Timer);
    let mode = ConflictMode::from_flags(cmd.overwrite, cmd.skip_conflicts);
    // no explicit list imports every pipeline config in the export directory
    let pipeline_names = if cmd.pipelines.is_empty() {
        imports::list_export_configs(&cmd.import, "pipelines").await?
    } else {
        imports::dedup_names(cmd.pipelines.clone(), &progress)
    };
    // load each pipeline request once, collecting the images it references as we go
    let mut pipeline_items = Vec::with_capacity(pipeline_names.len());
    let mut image_set: HashSet<String> = HashSet::new();
    for name in &pipeline_names {
        let request = import::load_request(&cmd.import, &cmd.group, name).await?;
        // collect every image across all stages, validating the order rather than
        // panicking on bad JSON
        let order = request.deserialize_image_order().map_err(|err| {
            Error::new(format!("Malformed image order in pipeline '{name}': {err}"))
        })?;
        image_set.extend(order.into_iter().flatten().map(ToOwned::to_owned));
        pipeline_items.push((name.clone(), "latest".to_string(), request));
    }
    let image_names: Vec<String> = image_set.into_iter().collect();
    let opts = ImageImportOpts {
        import_dir: &cmd.import,
        group: &cmd.group,
        registry: cmd.registry.as_deref(),
        registry_override: cmd.registry_override.as_deref(),
        skip_push: cmd.skip_push,
        migrate_registry: cmd.migrate_registry,
        mode,
        editor: cmd.editor.as_deref(),
        // raw TTY check; the driver derives interactive-mode + TTY from this
        is_terminal: std::io::stdin().is_terminal(),
        workers,
    };
    // categorize both halves before changing anything
    let images = categorize_from_disk(thorium, &opts, &image_names, &progress).await?;
    let pipelines =
        imports::categorize::categorize_pipelines(thorium, pipeline_items, &progress).await?;
    // images are applied before the pipelines that reference them; the shared driver
    // owns the confirmation, journal, and settle
    imports::disk::run_disk_import(
        thorium,
        conf,
        &progress,
        &opts,
        images,
        pipelines,
        cmd.rollback_on_failure,
    )
    .await
}

/// Export pipelines from Thorium
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The export pipelines command to execute
/// * `args` - The shared Thorctl args (for the worker count)
/// * `conf` - The Thorctl config
#[cfg(any(target_os = "linux", target_os = "macos"))]
async fn export(
    thorium: &Thorium,
    cmd: &ExportPipelines,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // no explicit list exports every pipeline in the group
    let names: Vec<String> = if cmd.pipelines.is_empty() {
        utils::pipelines::list_all_pipelines(thorium, &cmd.group)
            .await?
            .into_iter()
            .map(|pipeline| pipeline.name)
            .collect()
    } else {
        cmd.pipelines.clone()
    };
    // pre-flight: write each config sequentially so on-disk conflicts resolve
    // interactively (pipeline configs are tiny, so there's no worker pool)
    let can_prompt = !cmd.skip_conflicts && std::io::stdin().is_terminal();
    let editor = crate::handlers::imports::editor::resolve_editor(None, conf).to_string();
    let mut resolver = DiskConflictResolver::new(cmd.overwrite, can_prompt, editor);
    let progress = Bar::new("pipelines export", "Exporting configs", BarKind::Timer);
    let pipelines_dir = cmd.output.join("pipelines");
    // fetch the pipelines concurrently (bounded by --workers); writes below stay
    // sequential so the conflict resolver can prompt without racing
    let fetch_workers = std::cmp::min(args.workers, names.len()).max(1);
    let fetched: Vec<(String, Result<Pipeline, Error>)> = stream::iter(names)
        .map(|name| async move {
            let result = thorium.pipelines.get(&cmd.group, &name).await;
            (name, result)
        })
        .buffer_unordered(fetch_workers)
        .collect()
        .await;
    // collect the images referenced by these pipelines to export alongside them
    let mut images: HashSet<String> = HashSet::new();
    // names we couldn't fetch; collected so the export exits non-zero rather than
    // silently reporting success after skipping resources
    let mut failed: Vec<String> = Vec::new();
    for (name, result) in fetched {
        let pipeline = match result {
            Ok(pipeline) => pipeline,
            Err(err) => {
                progress.error(format!("Failed to get pipeline '{name}': {err}"));
                failed.push(name);
                continue;
            }
        };
        images.extend(pipeline.order.iter().flatten().cloned());
        let request = PipelineRequest::from(pipeline);
        // curated (prioritized) field order so an exported pipeline config matches the layout `init`
        // and toolbox export produce — one consistent, edit-friendly format everywhere (still
        // deterministic: curated keys first, remaining keys sorted)
        let config_json = crate::utils::curated_json(
            &request,
            crate::handlers::imports::merge::PIPELINE_FIELD_ORDER,
        )
        .map_err(|e| Error::new(format!("Failed to serialize pipeline '{name}': {e}")))?;
        // optionally open the config in an editor for review before writing
        let config_json = if cmd.review {
            progress
                .suspend_async(crate::handlers::imports::editor::review_config_in_editor::<
                    thorium::models::PipelineRequest,
                >(
                    &config_json,
                    &format!("export-pipeline-{name}"),
                    &conf.default_editor,
                    crate::handlers::imports::merge::PIPELINE_FIELD_ORDER,
                ))
                .await?
        } else {
            config_json
        };
        let outcome = resolver
            .write_yaml::<PipelineRequest>(
                &pipelines_dir.join(format!("{name}.json")),
                &config_json,
                &progress,
            )
            .await?;
        if outcome == WriteOutcome::Quit {
            progress.refresh("Export stopped early", BarKind::Timer);
            progress.finish();
            return Ok(());
        }
    }
    progress.finish();
    // export the images these pipelines reference (this applies the same on-disk
    // conflict handling for image configs). Guard against an empty set: with
    // empty-implies-all semantics, an empty list would export the entire group.
    let referenced_images: Vec<String> = images.into_iter().collect();
    if !referenced_images.is_empty() {
        let image_cmd = ExportImages {
            images: referenced_images,
            group: cmd.group.clone(),
            output: cmd.output.clone(),
            config_only: cmd.config_only,
            overwrite: cmd.overwrite,
            skip_conflicts: cmd.skip_conflicts,
            review: cmd.review,
        };
        super::images::export(thorium, &image_cmd, args, conf).await?;
    }
    // surface skipped pipelines as a non-zero exit so a partial export isn't read as success
    if !failed.is_empty() {
        return Err(Error::new(format!(
            "Failed to export {} pipeline(s): {}",
            failed.len(),
            failed.join(", ")
        )));
    }
    Ok(())
}

/// Handle all pipelines commands
///
/// # Arguments
///
/// * `args` - The arguments passed to Thorctl
/// * `cmd` - The pipelines command to execute
pub async fn handle(args: &Args, cmd: &Pipelines) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    // warn about insecure connections if not set to skip
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // check if we need to update
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        super::update::ask_update(&thorium).await?;
    }
    // call the right pipelines handler
    match cmd {
        Pipelines::Get(cmd) => get(thorium, cmd).await,
        Pipelines::Describe(cmd) => describe(thorium, cmd).await,
        Pipelines::Edit(cmd) => edit::edit(thorium, &conf, cmd).await,
        Pipelines::Notifications(cmd) => notifications::handle(thorium, cmd).await,
        Pipelines::Bans(cmd) => bans::handle(thorium, cmd).await,
        Pipelines::Delete(cmd) => delete(thorium, cmd).await,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        Pipelines::Import(cmd) => import(&thorium, cmd, &conf, args.workers).await,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        Pipelines::Export(cmd) => export(&thorium, cmd, args, &conf).await,
    }
}
