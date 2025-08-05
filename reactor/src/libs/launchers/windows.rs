//! Launches workers on windows nodes

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use thorium::models::{ImageScaler, Node, Worker, WorkerDeleteMap};
use thorium::{Error, Thorium};
use tokio::process::Command;
use tracing::{event, span, Level, Span};

use super::Launcher;
use crate::libs::keys;

/// Handles launching jobs on windows nodes
#[derive(Default)]
pub struct Windows {}

/// Gets the info needed to spawn a windows image
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `group` - The group this image is in
/// * `name` - The name of the image to get info on
/// * `span` - The span to log traces under
async fn get_image_info(
    thorium: &Thorium,
    group: &str,
    name: &str,
    span: &Span,
) -> Result<(String, String, String), Error> {
    // get this images info
    let image = thorium.images.get(group, name).await?;
    // get our images url
    let image_tag = match image.image {
        Some(image) => image,
        None => {
            // log this error
            event!(
                parent: span,
                Level::ERROR,
                error = true,
                error_msg = "Missing Image Url/Tag"
            );
            return Err(Error::new("Missing Image Url/Tag"));
        }
    };
    // get our entry point or set a default
    let (entrypoint, cmd) = match (image.args.entrypoint, image.args.command) {
        (Some(entrypoint), Some(cmd)) => (entrypoint, cmd),
        (Some(entrypoint), None) => (entrypoint, vec![]),
        (None, Some(cmd)) => (vec![], cmd),
        (None, None) => (
            vec![
                "echo".to_owned(),
                "Missing".to_owned(),
                "Entrypoint/Command".to_owned(),
            ],
            vec![],
        ),
    };
    // serialize our entrypoint and commands
    let entrypoint = serde_json::to_string(&entrypoint)?;
    let cmd = serde_json::to_string(&cmd)?;
    Ok((image_tag, entrypoint, cmd))
}

/// Cast our key path to its components as strs
///
/// # Arguments
///
/// * `path` - The path to break down and cast
/// * `span` - The span to log traces under
fn cast_paths<'a>(path: &'a PathBuf, span: &Span) -> Result<(String, &'a str), Error> {
    // convert our keys parent path to a str
    let keys_parent = match path.parent() {
        Some(parent) => {
            match parent.to_str() {
                Some(keys_path_str) => keys_path_str,
                None => {
                    // log that our keys path is not valid unicode
                    event!(
                        parent: span,
                        Level::ERROR,
                        error = true,
                        error_msg = "Keys path is not valid unicode",
                    );
                    return Err(Error::new("Keys path Not Utf-8".to_owned()));
                }
            }
        }
        None => {
            // log that our keys path is not valid unicode
            event!(
                parent: span,
                Level::ERROR,
                error = true,
                error_msg = "Keys path does not have a parent",
            );
            return Err(Error::new("Keys path does not have a parent"));
        }
    };
    // build our keys volume mount
    let keys_mount = format!("{}:{}", keys_parent, keys_parent);
    // convert our keys path to a str
    let keys_path_str = match path.to_str() {
        Some(keys_path_str) => keys_path_str,
        None => {
            // log that our keys path is not valid unicode
            event!(
                parent: span,
                Level::ERROR,
                error = true,
                error_msg = "Keys path is not valid unicode",
            );
            return Err(Error::new("Keys path Not Utf-8".to_owned()));
        }
    };
    Ok((keys_mount, keys_path_str))
}

#[async_trait::async_trait]
impl Launcher for Windows {
    /// Spawn a worker and then return a process id that can be used to track it
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    /// * `span` - The span to log traces under
    async fn launch(
        &mut self,
        thorium: &Thorium,
        worker: &Worker,
        span: &Span,
    ) -> Result<(), Error> {
        // start our worker launch span
        let span = span!(
            parent: span,
            Level::INFO,
            "Launching Worker",
            name = worker.name,
            user = worker.user,
            group = worker.group,
            pipeline = worker.pipeline,
            stage = worker.stage
        );
        // get our images info
        let (image_tag, entrypoint, cmd) =
            get_image_info(thorium, &worker.group, &worker.stage, &span).await?;
        // build this containers name
        let name = format!("thorium-{}", worker.name);
        // get the path to this workers keys
        let keys_path = keys::path(&worker.user);
        // break this path down into its parent and the full path to our key
        let (keys_mount, keys_path_str) = cast_paths(&keys_path, &span)?;
        // build the docker run args
        let args = vec![
            "run",
            "--detach",
            "--name",
            &name,
            "--rm",
            "--isolation=hyperv",
            "-v",
            "C:\\Thorium\\agent:C:\\Thorium",
            "-v",
            &keys_mount,
            "--entrypoint",
            "C:\\Thorium\\thorium-agent.exe",
            &image_tag,
            "--cluster",
            &worker.cluster,
            "--node",
            &worker.node,
            "--trace",
            "C:\\Thorium\\tracing.yml",
            "--group",
            &worker.group,
            "--pipeline",
            &worker.pipeline,
            "--stage",
            &worker.stage,
            "--name",
            &worker.name,
            "windows",
            "--entrypoint",
            &entrypoint,
            "--cmd",
            &cmd,
            "--keys",
            keys_path_str,
        ];
        // launch our agent
        match Command::new("docker").args(&args).spawn() {
            Ok(_) => (),
            Err(error) => {
                // log that we failed to launch this worker
                event!(
                    parent: &span,
                    Level::ERROR,
                    error = true,
                    error_msg = error.to_string()
                );
                return Err(Error::from(error));
            }
        };
        // get our child's process id
        Ok(())
    }

    /// Check if any of our current workers have completed or died
    ///
    /// This returns the currently active workers.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    /// * `span` - The span to log traces under
    async fn check(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
        span: &Span,
    ) -> Result<(), Error> {
        // stat our check active containers span
        let span = span!(parent: span, Level::INFO, "Checking Active Containers");
        // get the currently active containers
        let mut names = ls_containers(&span).await?;
        // keep a list of workers that should be deleted since they no longer exist
        let mut deletes = WorkerDeleteMap::default();
        // crawl the containers that should be active
        active.retain(|name, worker| {
            // TODO check if this worker failed or just died?
            if !names.contains(name) {
                // add this worker to the list of workers to be deleted since it no longer exists
                deletes.add_mut(&worker.name);
                false
            } else {
                true
            }
        });
        // delete the workers that no longer exist
        thorium
            .system
            .delete_workers(ImageScaler::Windows, &deletes)
            .await?;
        // move any already active workers to our active map
        names.retain(|name| {
            // if we have an existing worker with this name then move it to active
            if let Some(worker) = info.workers.remove(name) {
                event!(
                    parent: &span,
                    Level::INFO,
                    msg = "Recovered worker",
                    name = &name
                );
                // track this worker as still active
                active.insert(name.clone(), worker);
                false
            } else {
                // drop any active workers
                !active.contains_key(name)
            }
        });
        // kill any remaining containers if there is some
        if !names.is_empty() {
            kill_containers(&names, &span).await?;
        }
        Ok(())
    }

    /// Shutdown a list of workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The workers to shutdown
    /// * `span` - The span to log traces under
    async fn shutdown(
        &mut self,
        _thorium: &Thorium,
        mut workers: HashSet<String>,
        span: &Span,
    ) -> Result<(), Error> {
        // start our kill windows containers span
        let span = span!(
            parent: span,
            Level::INFO,
            "Killing Windows Containers",
            count = workers.len()
        );
        // get a list of our current containers
        let alive = ls_containers(&span).await?;
        // skip any workers that are no longer alive
        workers.retain(|name| alive.contains(name));
        // shutdown any still alive containers
        kill_containers(&workers, &span).await?;
        Ok(())
    }
}

/// Execute and parse a docker container list
///
/// # Arguments
///
/// * `span ` - The span to log traces under
async fn ls_containers(span: &Span) -> Result<HashSet<String>, Error> {
    // start our container listing span
    let span = span!(parent: span, Level::INFO, "Listing Windows Containers");
    // get the currently running containers on this node
    let output = Command::new("docker").args(&["ps", "-a"]).output().await?;
    // if this command failed then return the error
    if output.status.success() {
        // cast our output to a string
        let stdout = String::from_utf8_lossy(&output.stdout);
        // get the names of all running containers
        let names = stdout
            .lines()
            .skip(1)
            .filter_map(|line| line.split_whitespace().last())
            // filter down to just Thorium spawned containers
            .filter(|name| name.starts_with("thorium-"))
            .map(|name| name.replace("thorium-", ""))
            .collect::<HashSet<String>>();
        // log how many active containers we found
        event!(parent: &span, Level::INFO, containers = names.len());
        Ok(names)
    } else {
        // cast our error to a string
        let msg = String::from_utf8_lossy(&output.stderr).to_string();
        // log that an error occurred when getting output
        event!(parent: &span, Level::ERROR, error = true, error_msg = msg);
        // return our error
        return Err(Error::new(msg));
    }
}

/// Kills one or more containers
async fn kill_containers(containers: &HashSet<String>, span: &Span) -> Result<(), std::io::Error> {
    // start our container listing span
    span!(
        parent: span,
        Level::INFO,
        "Kill Containers",
        containers = containers.len()
    );
    // stop the target containers
    Command::new("docker")
        .arg("stop")
        .args(containers)
        .output()
        .await?;
    // remove the target containers
    Command::new("docker")
        .arg("rm")
        .args(containers)
        .output()
        .await?;
    Ok(())
}
