use clap::Parser;
use serde_derive::Deserialize;
use thorium::models::ImageScaler;
use thorium::{Error, Thorium};
use tracing::{Level, event, instrument};

use crate::libs::Target;

/// Command line args
#[derive(Parser, Debug, Deserialize, Clone)]
#[clap(version, author)]
pub struct Args {
    /// The environment specific args for this agent
    #[clap(subcommand)]
    pub env: Envs,
    /// The cluster this worker is in
    #[clap(short, long)]
    pub cluster: String,
    /// The node this worker is on
    #[clap(long)]
    pub node: Option<String>,
    /// The group to claim jobs from
    #[clap(short, long)]
    pub group: String,
    /// The pipeline to claim jobs from
    #[clap(short, long)]
    pub pipeline: String,
    /// The stage to claim jobs from
    #[clap(short, long)]
    pub stage: String,
    /// The name of this worker
    #[clap(short, long)]
    pub name: String,
    /// The path to the tracing config to load
    #[clap(short, long, default_value = "/opt/thorium/tracing.yml")]
    pub trace: String,
    /// The keys to use when authenticating to Thorium
    #[clap(short, long, default_value = "keys.yml")]
    pub keys: String,
    /// How long should this agent sit limbo before exiting without a job to work on
    #[clap(short, long, default_value = "30")]
    pub limbo: usize,
}

impl Args {
    /// Get the image targets for this environment from any command line args
    ///
    /// # Arguments
    ///
    /// * `span` - The span to log traces under
    #[instrument(name = "Args::target", skip_all)]
    pub async fn target(&self, thorium: &Thorium) -> Result<Target, Error> {
        // get the target images info
        let image = thorium.images.get(&self.group, &self.stage).await?;
        // get our current users info
        let user = thorium.users.info().await?;
        // get the name of this worker
        let name = self.name.clone();
        // get our workers info
        let worker = thorium.system.get_worker(&name).await?;
        // build the target object
        let target = Target {
            name,
            group: self.group.to_owned(),
            pipeline: self.pipeline.to_owned(),
            stage: self.stage.to_owned(),
            image,
            user,
            thorium: thorium.clone(),
            active: None,
            pool: worker.pool,
        };
        //log this new target
        event!(
            Level::INFO,
            name = target.name,
            user = target.user.username,
            group = target.group,
            pipeline = target.pipeline,
            stage = target.stage,
            pool = target.pool.as_str(),
        );
        Ok(target)
    }

    /// Get the current scaler we are running under
    pub fn scaler(&self) -> ImageScaler {
        match self.env {
            Envs::K8s(_) => ImageScaler::K8s,
            Envs::BareMetal(_) => ImageScaler::BareMetal,
            Envs::Windows(_) => ImageScaler::Windows,
            Envs::Kvm(_) => ImageScaler::Kvm,
        }
    }

    /// Get our nodes hostname
    pub fn node(&self) -> Result<String, Error> {
        // if we have a node specified in our args then use that
        match &self.node {
            Some(node) => Ok(node.to_owned()),
            None => {
                // get our hostname since our args don't specify it
                match gethostname::gethostname().into_string() {
                    Ok(hostname) => Ok(hostname),
                    Err(err) => {
                        return Err(Error::new(format!(
                            "Failed to get hostname with {:#?}",
                            err
                        )));
                    }
                }
            }
        }
    }
}

/// The different environments this agent is executing in
#[derive(Parser, Debug, Deserialize, Clone)]
pub enum Envs {
    /// This agent is running in k8s
    #[clap(version, author)]
    K8s(K8s),
    /// This agent is running on bare metal
    #[clap(version, author)]
    BareMetal(BareMetal),
    /// This agent is running in k8s
    #[clap(version, author)]
    Windows(Windows),
    /// This agent is running in a kvm vm
    #[clap(version, author)]
    Kvm(Kvm),
}

impl Envs {
    /// Get our env name as a str
    pub fn kind(&self) -> &str {
        match self {
            Envs::K8s(_) => "K8s",
            Envs::BareMetal(_) => "BareMetal",
            Envs::Windows(_) => "Windows",
            Envs::Kvm(_) => "Kvm",
        }
    }
}

/// The args for running the agent in K8s
#[derive(Parser, Debug, Deserialize, Clone)]
#[clap(version, author)]
pub struct K8s {
    /// The original entrypoint for this container
    #[clap(short, long)]
    pub entrypoint: String,
    /// The original command for this container
    #[clap(short = 'm', long)]
    pub cmd: String,
}

/// The args for running the agent in K8s
#[derive(Parser, Debug, Deserialize, Clone)]
#[clap(version, author)]
pub struct BareMetal {}

/// The args for running the agent in Windows
#[derive(Parser, Debug, Deserialize, Clone)]
#[clap(version, author)]
pub struct Windows {
    /// The original entrypoint for this container
    #[clap(short, long)]
    pub entrypoint: String,
    /// The original command for this container
    #[clap(short = 'm', long)]
    pub cmd: String,
    /// The name of this worker
    #[clap(short, long)]
    pub name: String,
}

/// The args for running the agent in an KVM based vm
#[derive(Parser, Debug, Deserialize, Clone)]
#[clap(version, author)]
pub struct Kvm {}
