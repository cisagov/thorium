//! Launches workers on windows nodes

use chrono::{DateTime, Duration, Utc};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use thorium::Thorium;
use thorium::models::{ImageScaler, Node, Worker, WorkerDeleteMap};
use tokio::process::Command;
use tracing::{Level, event, instrument};

use super::Launcher;
use crate::Error;
use crate::libs::keys;

/// The grace period a freshly launched Windows worker is given before its absence from
/// `docker ps -a` is treated as death, allowing time for a `docker run` image pull and
/// container creation to complete
const WINDOWS_LAUNCH_GRACE: Duration = Duration::minutes(5);

/// Handles launching jobs on windows nodes
#[derive(Default)]
pub struct Windows {}

impl Windows {
    /// Decide whether a Windows worker whose container is absent from `docker ps -a` should be
    /// deleted from the Thorium API
    ///
    /// This is `reconcile`'s per-worker delete decision (it never needs `&self`) so its logic can be
    /// unit tested without a live Thorium client. An absent worker is only deleted while the API
    /// still lists it; when the agent already deleted its own record on a clean exit, deleting it
    /// again would 404.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the worker whose container is absent
    /// * `api_workers` - The workers the API still reports for this node
    fn should_delete_absent_worker(name: &str, api_workers: &HashMap<String, Worker>) -> bool {
        // only delete absent workers that still exist in the API
        api_workers.contains_key(name)
    }

    /// Decide whether a Windows worker whose container is absent from `docker ps -a` has actually
    /// died (as opposed to still being launched, e.g. its image is still pulling)
    ///
    /// This is `reconcile`'s liveness decision (it never needs `&self`) so its grace-period logic
    /// can be unit tested. Absence only means death once the worker's launch grace period has
    /// elapsed, so a just-launched worker whose container is not yet visible (e.g. its image is
    /// still pulling) is not deregistered on the very next poll.
    ///
    /// # Arguments
    ///
    /// * `spawned` - When the worker was spawned
    /// * `now` - The current time
    /// * `grace` - The grace period a freshly launched worker is allowed before absence means death
    fn absent_worker_is_dead(spawned: DateTime<Utc>, now: DateTime<Utc>, grace: Duration) -> bool {
        // absence only means death once the launch grace period has elapsed
        now >= spawned + grace
    }
}

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

/// Builds the docker `-v` bind-mount spec for the host agent directory
///
/// The host agent directory (`C:\Thorium\agent`) is mounted into every tool container at
/// `C:\Thorium`. Because the attacker-supplied tool shares the container with the agent, a
/// read-write mount would let it overwrite `C:\Thorium\thorium-agent.exe`, writing through to the
/// host binary that both every subsequent tenant's container and the reactor's own periodic
/// host-side update task then execute — a Hyper-V-sandbox-to-host escape that defeats
/// `--isolation=hyperv`. Mounting read-only closes that write-through path.
fn agent_dir_mount() -> &'static str {
    // append the `:ro` suffix so the host agent directory is mounted read-only and a tool
    // container cannot trojan the shared agent binary
    "C:\\Thorium\\agent:C:\\Thorium:ro"
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
            agent_dir_mount(),
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
        // snapshot the workers the API still reports so we can consult it in the retain closure
        let api_workers = info.workers.clone();
        // the current time used to evaluate each worker's launch grace period
        let now = Utc::now();
        // crawl the containers that should be active
        active.retain(|name, worker| {
            // TODO check if this worker failed or just died?
            if !names.contains(name) {
                // only treat absence as death once the launch grace period has elapsed
                if !Self::absent_worker_is_dead(worker.spawned, now, WINDOWS_LAUNCH_GRACE) {
                    // this worker is still within its launch grace period, keep tracking it
                    return true;
                }
                // add this worker to the list of workers to be deleted since it no longer exists
                if Self::should_delete_absent_worker(&worker.name, &api_workers) {
                    deletes.add_mut(&worker.name);
                }
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

#[cfg(test)]
mod tests {
    use super::{WINDOWS_LAUNCH_GRACE, Windows, agent_dir_mount};
    use chrono::{Duration, Utc};
    use std::collections::HashMap;

    /// `reconcile` only deletes a worker whose container has vanished when that worker is still
    /// present in the API's worker list, so a worker whose container and API record are both gone
    /// is not re-deleted. Windows containers run with `--rm`, so on clean agent completion the
    /// agent deletes its own record first and then the container disappears; deleting the
    /// already-gone record would 404 and crash the reactor daemon.
    #[test]
    fn absent_worker_missing_from_api_is_not_deleted() {
        // the API no longer lists this worker (its agent deleted its own record on clean exit)
        let api_workers: HashMap<String, thorium::models::Worker> = HashMap::new();
        // a vanished worker already gone from the API must not be deleted again (that 404s + crashes)
        assert!(
            !Windows::should_delete_absent_worker("worker-a", &api_workers),
            "a worker whose container exited and whose API record was already deleted is deleted \
             again, 404-crashing the reactor on every clean Windows job completion"
        );
    }

    /// `reconcile` gives a freshly launched worker a grace period before treating its absence
    /// from `docker ps -a` as death, so a worker spawned seconds ago whose container is not yet
    /// visible is not considered dead. `docker run --detach` is not awaited, so while an image is
    /// still pulling the container is not yet listed; the grace period keeps the still launching
    /// worker from being deregistered on the very next 2s poll and causing a spawn/delete/kill
    /// loop.
    #[test]
    fn just_launched_worker_within_grace_is_not_dead() {
        // the worker was spawned only moments ago (its image may still be pulling)
        let now = Utc::now();
        let spawned = now - Duration::seconds(2);
        // a worker still inside its launch grace period must not be treated as dead
        assert!(
            !Windows::absent_worker_is_dead(spawned, now, WINDOWS_LAUNCH_GRACE),
            "a just-launched worker whose container is not yet visible is treated as dead and \
             deregistered, causing a spawn/delete/kill loop for the entire image-pull duration"
        );
    }

    /// the host agent directory bind-mounted into every Windows tool container must be
    /// read-only. Without a `:ro` suffix a malicious tool sharing the container with the agent can
    /// overwrite the host `thorium-agent.exe`, which is then executed both by every subsequent
    /// tenant's container and by the reactor's own host-side update loop — a Hyper-V sandbox escape
    /// that defeats `--isolation=hyperv`.
    ///
    /// This test only builds/runs on Windows because the launcher module is
    /// `#[cfg(target_os = "windows")]`.
    #[test]
    fn agent_dir_mount_is_read_only() {
        // build the docker -v spec exactly as the launcher injects it
        let mount = agent_dir_mount();
        // the host agent directory mount must be read-only
        assert!(
            mount.ends_with(":ro"),
            "the host agent directory is bind-mounted read-write into tool containers, allowing a \
             tool to trojan the agent binary and escape to the host"
        );
    }
}
