use itertools::Itertools;
use std::collections::HashSet;
use thorium::models::PipelineRequest;
use thorium::CtlConf;
use thorium::{models::Pipeline, Error, Thorium};

use crate::args::pipelines::{DescribePipelines, GetPipelines, Pipelines};
use crate::args::{Args, DescribeCommand};
use crate::utils;

mod bans;
mod notifications;

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "macos"))] {
        use crate::args::pipelines::{ExportPipelines, ImportPipelines};
        use crate::args::images::{ExportImages, ImportImages};
        use super::Controller;

        mod export;
        mod import;

        use export::PipelineExportWorker;
        use import::PipelineImportWorker;

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
        println!(
            "{:<30} | {:<20} | {}",
            pipeline.name,
            pipeline.group,
            pipeline.description.as_ref().unwrap_or(&"-".to_string())
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
        thorium
            .pipelines
            .list(group)
            .limit(cmd.limit)
            .page(cmd.page_size)
            .details()
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

/// Crawl all of the pipeline requests we want to import to build a list of images to import
///
/// # Arguments
///
/// * `cmd` - The import pipeline command for the pipelines we are importing
#[cfg(any(target_os = "linux", target_os = "macos"))]
async fn get_import_images(cmd: &ImportPipelines) -> Result<Vec<String>, Error> {
    // assume we have at least one image for each pipeline
    let mut images = HashSet::with_capacity(cmd.pipelines.len());
    // step over and get the images for each pipeline we want to import
    for name in &cmd.pipelines {
        // build the path to this pipelines request data
        let file_path = cmd.import.join(format!("{name}.json"));
        // read this pipelines request to a string
        let pipeline_str = tokio::fs::read_to_string(&file_path).await?;
        // parse our pipeline data
        let pipeline: PipelineRequest = serde_json::from_str(&pipeline_str)?;
        // parse this pipelines order
        for stage in pipeline.order.as_array().unwrap() {
            // if this is just a string then add our single stage
            if let Some(stage) = stage.as_str() {
                // add this single image stage
                images.insert(stage.to_owned());
            }
            if let Some(stages) = stage.as_array() {
                // convert our images to strings
                let stage_iter = stages
                    .iter()
                    .filter_map(|val| val.as_str())
                    .map(|str| str.to_owned());
                images.extend(stage_iter);
            }
        }
    }
    // convert our hashset to a vec
    let image_vec = images.into_iter().collect::<Vec<String>>();
    Ok(image_vec)
}

/// Import pipelines to Thorium
#[cfg(any(target_os = "linux", target_os = "macos"))]
async fn import(
    thorium: &Thorium,
    cmd: &ImportPipelines,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // get the images we need to import
    let images = get_import_images(cmd).await?;
    // build the correct image import command
    let image_cmd = ImportImages {
        images,
        group: cmd.group.clone(),
        import: cmd.import.join("images"),
        registry: cmd.registry.clone(),
        registry_override: cmd.registry_override.clone(),
        skip_push: cmd.skip_push,
        migrate_registry: cmd.migrate_registry,
    };
    // import all of the required images
    super::images::import(thorium, &image_cmd, args, conf).await?;
    // limit our workers by the number of pipelines we are going to export
    let workers = std::cmp::min(args.workers, cmd.pipelines.len());
    // create a new worker controller
    let mut controller = Controller::<PipelineImportWorker>::spawn(
        "Importing Pipelines",
        thorium,
        workers,
        conf,
        args,
        cmd,
    )
    .await;
    // add the pipelines to export
    for pipeline in &cmd.pipelines {
        // try to add this download job
        if let Err(error) = controller.add_job(pipeline.clone()).await {
            // log this error
            controller.error(&error.to_string());
        }
    }
    // wait for all our workers to complete
    controller.finish().await?;
    Ok(())
}

/// Export pipelines from Thorium
#[cfg(any(target_os = "linux", target_os = "macos"))]
async fn export(
    thorium: &Thorium,
    cmd: &ExportPipelines,
    args: &Args,
    conf: &CtlConf,
) -> Result<(), Error> {
    // limit our workers by the number of pipelines we are going to export
    let workers = std::cmp::min(args.workers, cmd.pipelines.len());
    // create a new worker controller
    let mut controller = Controller::<PipelineExportWorker>::spawn(
        "Exporting Pipelines",
        thorium,
        workers,
        conf,
        args,
        cmd,
    )
    .await;
    // add the pipelines to export
    for pipeline in &cmd.pipelines {
        // try to add this download job
        if let Err(error) = controller.add_job(pipeline.clone()).await {
            // log this error
            controller.error(&error.to_string());
        }
    }
    // get all of the images in this pipeline
    let mut images = HashSet::with_capacity(cmd.pipelines.len());
    // crawl over all pipelines and get their info
    for pipeline in &cmd.pipelines {
        // get this pipelines info
        let info = thorium.pipelines.get(&cmd.group, pipeline).await?;
        // add this pipelines images to our image set
        images.extend(info.order.into_iter().flatten());
    }
    // wait for all our workers to complete
    controller.finish().await?;
    // build the output path for our images
    let mut image_output = cmd.output.clone();
    // nest our images in a directory called images
    image_output.push("images");
    // build the correct image export command
    let image_cmd = ExportImages {
        images: images.into_iter().collect(),
        group: cmd.group.clone(),
        output: image_output,
    };
    // export all of the required images
    super::images::export(thorium, &image_cmd, args, conf).await?;
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
        Pipelines::Notifications(cmd) => notifications::handle(thorium, cmd).await,
        Pipelines::Bans(cmd) => bans::handle(thorium, cmd).await,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        Pipelines::Import(cmd) => import(&thorium, cmd, args, &conf).await,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        Pipelines::Export(cmd) => export(&thorium, cmd, args, &conf).await,
    }
}
