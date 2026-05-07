//! The logic for a `KvmWorker` in the [`Launched`] state

use base64::Engine;
use chrono::{DateTime, Utc};
use rand::Rng;
use serde_json::Value;
use tracing::instrument;
use virt::domain::Domain;

use crate::libs::{
    Error,
    launchers::kvm::{
        DOMAIN_TIMEOUT, QEMU_AGENT_TIMEOUT_SECONDS, UNRESPONSIVE_DOMAIN_TIMEOUT,
        virt_async::LibvirtClient,
    },
};

use super::Launched;

/// The state of the Thorium Agent in a domain
#[derive(Debug)]
pub enum AgentState {
    /// We can't tell whether the agent is running, has run, and what its status is
    Missing,
    /// The agent is definitely running
    Running,
    /// The agent definitely exited, but we don't know whether it succeeded
    Exited,
    /// The agent errored out
    Errored {
        exitcode: i64,
        stdout: Option<String>,
        stderr: Option<String>,
    },
    /// The agent completed without error
    Completed,
}

impl Launched {
    /// Get a random time to do a health check
    pub fn next_health_check() -> DateTime<Utc> {
        let mut rng = rand::rng();
        // check anywhere from 30 to 60 seconds from now
        let secs = rng.random_range(30..=60);
        let duration = std::time::Duration::from_secs(secs);
        Utc::now() + duration
    }

    /// Get the state of the Thorium agent in this worker
    ///
    /// Update the heartbeat of the worker if we see the agent running
    ///
    /// # Arguments
    ///
    /// * `client` - A libvirt client
    #[instrument(skip_all, fields(worker = self.worker, domain = self.domain_name))]
    pub async fn agent_state(&mut self, client: &LibvirtClient) -> Result<AgentState, Error> {
        let Some(agent_pid) = self.agent_pid else {
            return Ok(AgentState::Missing);
        };
        // Execute agent: guest-exec-status
        let agent_cmd = serde_json::json!({
            "execute": "guest-exec-status",
            "arguments": {
                "pid": agent_pid
            }
        })
        .to_string();
        let domain_name = self.domain_name.clone();
        let domain = client
            .with_conn(move |conn| Domain::lookup_by_name(conn, &domain_name))
            .await?;
        // if the domain isn't even active, the agent is definitely not running
        if !client
            .with_domain(&domain, virt::domain::Domain::is_active)
            .await?
        {
            return Ok(AgentState::Missing);
        }
        let response_raw = match client
            .with_domain(&domain, move |d| {
                d.qemu_agent_command(&agent_cmd, QEMU_AGENT_TIMEOUT_SECONDS, 0)
            })
            .await
        {
            Ok(raw) => raw,
            Err(err) => {
                // check if the error was that the PID doesn't exist
                let msg = err.to_string().to_ascii_lowercase();
                if err.code() == virt::error::ErrorNumber::AgentCommandFailed
                    && msg.contains("not exist")
                {
                    // the PID doesn't exist which means the agent isn't running
                    return Ok(AgentState::Missing);
                }
                return Err(Error::from(err));
            }
        };
        // Parse response to check for errors
        let result: Value = serde_json::from_str(&response_raw)?;
        // Optional: basic guard that libvirt/qga returned what we expect
        let Some(ret) = result.get("return") else {
            return Err(Error::new(
                "QEMU guest agent status response missing 'return' field",
            ));
        };
        let exited = ret.get("exited").and_then(Value::as_bool);
        let exitcode = ret.get("exitcode").and_then(Value::as_i64);
        let stdout = ret
            .get("out-data")
            .and_then(|v| v.as_str())
            .and_then(|out_b64| {
                base64::engine::general_purpose::STANDARD
                    .decode(out_b64)
                    .ok()
            })
            .and_then(|s| String::from_utf8(s).ok());
        let stderr = ret
            .get("err-data")
            .and_then(|v| v.as_str())
            .and_then(|err_b64| {
                base64::engine::general_purpose::STANDARD
                    .decode(err_b64)
                    .ok()
            })
            .and_then(|s| String::from_utf8(s).ok());
        match (exited, exitcode) {
            // we can't tell anything helpful about the agent from the response we received
            (None, None) => Ok(AgentState::Missing),
            // if we have an exitcode, we can tell whether the agent succeeded or failed
            (_, Some(exitcode)) => {
                if exitcode == 0 {
                    Ok(AgentState::Completed)
                } else {
                    Ok(AgentState::Errored {
                        exitcode,
                        stdout,
                        stderr,
                    })
                }
            }
            // we have no exitcode, but we can tell whether the agent exited
            (Some(exited), None) => {
                if exited {
                    Ok(AgentState::Exited)
                } else {
                    // we've confirmed the agent is running; update the heartbeat
                    self.heartbeat = Utc::now();
                    Ok(AgentState::Running)
                }
            }
        }
    }

    /// Returns whether this worker is due for a health check
    pub fn is_health_check_due(&self) -> bool {
        // we should check this if we are past this worker's next scheduled check
        Utc::now() >= self.next_check
    }

    /// Schedule this worker for its next health check
    pub fn schedule_next_health_check(&mut self) {
        self.next_check = Launched::next_health_check();
    }

    /// Returns whether this worker has timed out completely
    pub fn timed_out(&self) -> bool {
        // timed out if "now" is at/after launch + fail-safe timeout
        Utc::now() >= self.launched + DOMAIN_TIMEOUT
    }

    /// Returns whether this unresponsive worker (missing agent) has timed out
    pub fn timed_out_unresponsive(&self) -> bool {
        // timed out if "now" is at/after heartbeat + unresponsive timeout
        Utc::now() >= self.heartbeat + UNRESPONSIVE_DOMAIN_TIMEOUT
    }
}
