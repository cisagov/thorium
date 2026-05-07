//! The logic for a `KvmWorker` in the [`Created`] state

use std::path::Path;

use base64::Engine;
use chrono::Utc;
use serde_json::Value;
use thorium::models::Worker;
use tokio::io::AsyncReadExt;
use tracing::{Level, event, instrument};
use virt::domain::Domain;

use crate::libs::{
    Error, keys,
    launchers::kvm::{
        AgentPaths, FIRST_LAUNCH_ATTEMPT_DWELL, LAUNCH_ATTEMPT_DWELL, MAX_LAUNCH_ATTEMPTS,
        QEMU_AGENT_TIMEOUT_SECONDS, os_detector::GuestOs, virt_async::LibvirtClient,
    },
};

use super::{Created, KvmWorkerState, Launched};

/// The path to the Thorium agent on Linux guests
const THORIUM_AGENT_PATH_LINUX: &str = "/thorium-agent";

/// The path to the user's keys on Linux guests
const KEYS_PATH_LINUX: &str = "/thorium-keys.yml";

/// The path to the Thorium agent on Windows guests
const THORIUM_AGENT_PATH_WINDOWS: &str = "C:\\thorium-agent.exe";

/// The path to the user's keys on Windows guests
const KEYS_PATH_WINDOWS: &str = "C:\\thorium-keys.yml";

impl Created {
    /// Launch the worker by launching the Thorium agent on the target VM using QEMU guest agent
    ///
    /// On error, the worker in its created state is returned with its launch attempts incremented,
    /// along with the error that occurred so we can maybe try to launch it again next time
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - The domain to launch the agent in
    /// * `worker` - The worker we are launching an agent for
    /// * `agent_paths` - The path to the Thorium agent binaries on disk
    #[instrument(skip_all, fields(worker = self.worker, domain = self.domain_name))]
    pub async fn launch(
        mut self,
        client: &LibvirtClient,
        domain: &Domain,
        worker: &Worker,
        agent_paths: &AgentPaths,
    ) -> Result<Launched, (Self, Error)> {
        match self.try_launch(client, domain, worker, agent_paths).await {
            Ok(agent_pid) => Ok(self.next_state(Some(agent_pid))),
            Err(err) => {
                // increment our attempt count
                self.launch_attempts += 1;
                Err((self, err))
            }
        }
    }

    /// Proceed to the next state as if we actually launched the worker
    pub fn mock_launch(self) -> Launched {
        // no agent was launched, so we have no PID
        self.next_state(None)
    }

    /// Attempt to launch the worker and get a PID for the running Thorium agent
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - The domain to launch the agent in
    /// * `worker` - The worker we are launching an agent for
    /// * `agent_paths` - The path to the Thorium agent binaries on disk
    #[instrument(skip_all, fields(worker = self.worker, domain = self.domain_name))]
    async fn try_launch(
        &self,
        client: &LibvirtClient,
        domain: &Domain,
        worker: &Worker,
        agent_paths: &AgentPaths,
    ) -> Result<u64, Error> {
        // determine which thorium agent to transfer and where to transfer it and the keys
        // depending on the guest OS
        let (agent_path, agent_guest_path, keys_guest_path) = match self.guest_os {
            // if we couldn't detect the OS, just try Linux
            GuestOs::Linux | GuestOs::Unknown => (
                &agent_paths.linux,
                THORIUM_AGENT_PATH_LINUX,
                KEYS_PATH_LINUX,
            ),
            GuestOs::Windows => (
                &agent_paths.windows,
                THORIUM_AGENT_PATH_WINDOWS,
                KEYS_PATH_WINDOWS,
            ),
        };
        // transfer the Thorium agent to the domain
        self.transfer_agent(
            client,
            domain,
            worker,
            agent_path,
            agent_guest_path,
            keys_guest_path,
        )
        .await?;
        // Execute agent
        let agent_cmd = serde_json::json!({
            "execute": "guest-exec",
            "arguments": {
                "path": agent_guest_path,
                "arg": [
                    "--cluster", &worker.cluster,
                    "--node", &worker.node,
                    "--group", &worker.group,
                    "--pipeline", &worker.pipeline,
                    "--stage", &worker.stage,
                    "--name", &worker.name,
                    "--keys",
                    keys_guest_path,
                    "kvm"
                ],
                "capture-output": true
            }
        });
        let agent_cmd = agent_cmd.to_string();
        let response_raw = client
            .with_domain(domain, move |d| {
                d.qemu_agent_command(&agent_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            })
            .await?;
        // Parse response to check for errors
        let result: Value = serde_json::from_str(&response_raw)?;
        if let Some(agent_pid) = result["return"]["pid"].as_u64() {
            event!(
                Level::DEBUG,
                msg = "Agent launched",
                worker = self.worker,
                agent_pid = agent_pid
            );
            Ok(agent_pid)
        } else {
            Err(Error::new(format!(
                "Failed to launch agent: {response_raw}"
            )))
        }
    }

    /// Transfer a file to a domain in base64-encoded chunks using the QEMU guest agent
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - A reference to the domain to send the file to
    /// * `host_path` - Source file path on host
    /// * `guest_path` - Destination path in guest
    #[instrument(skip(self, client, domain), fields(worker = self.worker, domain = self.domain_name))]
    async fn transfer_file_to_guest(
        &self,
        client: &LibvirtClient,
        domain: &Domain,
        host_path: &Path,
        guest_path: &str,
    ) -> Result<(), Error> {
        // try to open host file
        let file = tokio::fs::File::open(host_path).await.map_err(|err| {
            Error::with_context(
                format!(
                    "Error opening local file '{}' for transfer",
                    host_path.display()
                ),
                err,
            )
        })?;
        let open_cmd = serde_json::json!({
            "execute": "guest-file-open",
            "arguments": {
                "path": guest_path,
                "mode": "wb",
            }
        })
        .to_string();
        let open_response = client
            .agent_cmd(domain, open_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            .await?;
        // parse file handle from response
        let guest_file_handle = open_response
            .get("return")
            .ok_or_else(|| {
                Error::new("Failed to get file handle from guest agent: missing 'return' field")
            })?
            .as_i64()
            .ok_or_else(|| {
                Error::new(
                    "Failed to get file handle from guest agent: 'return' field is not an integer",
                )
            })?;
        // transfer the data in base64-encoded chunks to the guest
        self.transfer_data_to_guest(client, domain, file, guest_file_handle)
            .await?;
        // Flush and close file
        let flush_cmd = serde_json::json!({
            "execute": "guest-file-flush",
            "arguments": {
                "handle": guest_file_handle,
            }
        })
        .to_string();
        client
            .agent_cmd(domain, flush_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            .await?;
        let close_cmd = serde_json::json!({
            "execute": "guest-file-close",
            "arguments": {
                "handle": guest_file_handle,
            }
        })
        .to_string();
        client
            .agent_cmd(domain, close_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            .await?;
        Ok(())
    }

    /// Transfer a file's data to a domain in base64-encoded chunks using the QEMU guest agent
    ///
    /// The file is owned by this function and is automatically closed on return by the file's
    /// [`Drop`] implementation.
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - A reference to the domain to send the file to
    /// * `file` - A file handle on the host to read from when transferring data
    /// * `guest_file_handle` - A file handle on the guest returned by the QEMU guest agent to transfer data to
    async fn transfer_data_to_guest(
        &self,
        client: &LibvirtClient,
        domain: &Domain,
        mut file: tokio::fs::File,
        guest_file_handle: i64,
    ) -> Result<(), Error> {
        // transfer in 64 KB chunks
        const CHUNK_SIZE: usize = 64 * 1024;
        let mut buffer = vec![0u8; CHUNK_SIZE];
        loop {
            // Read chunk from file
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                // end of file
                break;
            }
            // Encode chunk as base64
            let base64_data =
                base64::engine::general_purpose::STANDARD.encode(&buffer[..bytes_read]);
            let write_cmd = serde_json::json!({
                "execute": "guest-file-write",
                "arguments": {
                    "handle": guest_file_handle,
                    "buf-b64": base64_data,
                }
            })
            .to_string();
            let write_response = client
                .agent_cmd(domain, write_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
                .await?;
            // Check for write errors
            if let Some(error) = write_response["error"].as_object() {
                return Err(Error::new(format!(
                    "QEMU guest agent write failed: {error:?}"
                )));
            }
        }
        // data transfer is complete
        Ok(())
    }

    /// Set the executable bit for the Thorium agent in a Linux domain
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - The domain the agent is in
    /// * `agent_guest_path` - The path to the agent on the guest
    #[instrument(skip(self, client, domain), fields(worker = self.worker, domain = self.domain_name))]
    async fn set_agent_executable(
        &self,
        client: &LibvirtClient,
        domain: &Domain,
        agent_guest_path: &str,
    ) -> Result<(), Error> {
        // mark the agent as executable
        let chmod_cmd = serde_json::json!({
            "execute": "guest-exec",
            "arguments": {
                "path": "/bin/chmod",
                "arg": [
                    "+x",
                    agent_guest_path
                ]
            }
        })
        .to_string();
        let response_raw = client
            .with_domain(domain, move |d| {
                d.qemu_agent_command(&chmod_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            })
            .await?;
        // make sure we got a PID
        let result: Value = serde_json::from_str(&response_raw)?;
        let Some(chmod_pid) = result["return"]["pid"].as_u64() else {
            return Err(Error::new(format!(
                "Failed to set thorium-agent as executable on Linux guest: {response_raw}"
            )));
        };
        let status_cmd = serde_json::json!({
            "execute": "guest-exec-status",
            "arguments": {
                "pid": chmod_pid
            }
        })
        .to_string();
        if let Err(err) = client
            .with_domain(domain, move |d| {
                d.qemu_agent_command(&status_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            })
            .await
        {
            return Err(Error::new(format!(
                "'chmod +x' failed on Linux guest: {err}"
            )));
        }
        Ok(())
    }

    /// Transfer the agent to the domain
    ///
    /// # Arguments
    ///
    /// * `client` - An async client for libvirt
    /// * `domain` - The domain to send the agent to
    /// * `worker` - The domain's worker
    /// * `agent_path` - The path to the agent on the host running the reactor
    /// * `agent_guest_path` - The path to send the agent to on the guest VM
    /// * `keys_guest_path` - The path to send the keys to on the guest VM
    #[instrument(skip(self, client, domain, worker), fields(worker = self.worker, domain = self.domain_name))]
    async fn transfer_agent(
        &self,
        client: &LibvirtClient,
        domain: &Domain,
        worker: &Worker,
        agent_path: &Path,
        agent_guest_path: &str,
        keys_guest_path: &str,
    ) -> Result<(), Error> {
        // transfer agent to guest
        self.transfer_file_to_guest(client, domain, agent_path, agent_guest_path)
            .await?;
        // mark the agent as executable if we're on Linux
        if self.guest_os == GuestOs::Linux {
            self.set_agent_executable(client, domain, agent_guest_path)
                .await?;
        }
        // transfer keys to guest
        let keys_path = keys::path(&worker.user);
        self.transfer_file_to_guest(client, domain, &keys_path, keys_guest_path)
            .await?;
        Ok(())
    }

    /// Returns whether this worker has attempted to launch the maximum
    /// number of times (or more)
    pub fn has_reached_max_attempts(&self) -> bool {
        self.launch_attempts >= MAX_LAUNCH_ATTEMPTS
    }

    /// Returns true if the domain is ready to attempt launch
    pub fn ready_to_launch(&self) -> bool {
        let now = Utc::now();
        if let Some(last_attempt) = self.last_launch_attempt {
            // if we've waited enough time between attempts, we're ready to try again
            now >= last_attempt + LAUNCH_ATTEMPT_DWELL
        } else {
            // if this is our first attempt, see if we've waited long enough for the domain to stabilize
            now >= self.created + FIRST_LAUNCH_ATTEMPT_DWELL
        }
    }
}
