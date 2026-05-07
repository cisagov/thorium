//! Launches workers on windows nodes

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use thorium::Thorium;
use thorium::models::{ImageScaler, Node, Worker, WorkerDeleteMap};
use tokio::process::Command;
use tracing::{Level, event, instrument};

use super::Launcher;
use crate::Error;
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
#[instrument(name = "windows::get_image_info", skip_all, err(Debug))]
async fn get_image_info(
    thorium: &Thorium,
    group: &str,
    name: &str,
) -> Result<(String, String, String), Error> {
    // get this images info
    let image = thorium.images.get(group, name).await?;
    // get our images url
    let image_tag = match image.image {
        Some(image) => image,
        None => {
            // log this error
            event!(
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
#[instrument(name = "windows::cast_paths", skip_all, err(Debug))]
fn cast_paths<'a>(path: &'a PathBuf) -> Result<(String, &'a str), Error> {
    // convert our keys parent path to a str
    let keys_parent = match path.parent() {
        Some(parent) => {
            match parent.to_str() {
                Some(keys_path_str) => keys_path_str,
                None => {
                    // log that our keys path is not valid unicode
                    event!(
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
    // The Windows launcher has no data to resolve after a launch
    type LaunchedWorkerData = ();

    /// Spawn a worker and then return a process id that can be used to track it
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    #[instrument(
        name = "Windows::launch",
        skip_all,
        fields(
            name = worker.name,
            user = worker.user,
            group = worker.group,
            pipeline = worker.pipeline,
            stage = worker.stage
        ),
        err(Debug)
    )]
    async fn launch(
        &self,
        thorium: &Thorium,
        worker: Worker,
    ) -> Result<(Worker, Self::LaunchedWorkerData), Error> {
        // get our images info
        let (image_tag, entrypoint, cmd) =
            get_image_info(thorium, &worker.group, &worker.stage).await?;
        // build this containers name
        let name = format!("thorium-{}", worker.name);
        // get the path to this workers keys
        let keys_path = keys::path(&worker.user);
        // break this path down into its parent and the full path to our key
        let (keys_mount, keys_path_str) = cast_paths(&keys_path)?;
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
                event!(Level::ERROR, error = true, error_msg = error.to_string());
                return Err(Error::from(error));
            }
        };
        // get our childs process id
        Ok((worker, ()))
    }

    fn resolve_launches(&mut self, _launches: Vec<(&Worker, Self::LaunchedWorkerData)>) {}

    /// Check if any of our current workers have completed or died
    ///
    /// This returns the currently active workers.
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    #[instrument(name = "Windows::reconcile", skip_all, err(Debug))]
    async fn reconcile(
        &mut self,
        thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
    ) -> Result<(), Error> {
        // get the currently active containers
        let mut names = ls_containers().await?;
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
                event!(Level::INFO, msg = "Recovered worker", name = &name);
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
            kill_containers(&names).await?;
        }
        Ok(())
    }

    /// Shutdown a list of workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The workers to shutdown
    #[instrument(name = "Windows::shutdown", skip_all, fields(workers = workers.len()), err(Debug))]
    async fn shutdown(
        &mut self,
        _thorium: &Thorium,
        mut workers: HashSet<String>,
    ) -> Result<(), Error> {
        // get a list of our current containers
        let alive = ls_containers().await?;
        // skip any workers that are no longer alive
        workers.retain(|name| alive.contains(name));
        // shutdown any still alive containers
        kill_containers(&workers).await?;
        Ok(())
    }
}

/// Execute and parse a docker container list
#[instrument(name = "windows::ls_containers", skip_all, err(Debug))]
async fn ls_containers() -> Result<HashSet<String>, Error> {
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
        event!(Level::INFO, containers = names.len());
        Ok(names)
    } else {
        // cast our error to a string
        let msg = String::from_utf8_lossy(&output.stderr).to_string();
        // log that an error occured when getting output
        event!(Level::ERROR, error = true, error_msg = msg);
        // return our error
        return Err(Error::new(msg));
    }
}

/// Kills one or more containers
#[instrument(name = "windows::kill_containers", skip_all, fields(containers = containers.len()), err(Debug))]
async fn kill_containers(containers: &HashSet<String>) -> Result<(), std::io::Error> {
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
