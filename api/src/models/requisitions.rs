use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

use super::{Deadline, Pools, Resources, Worker};
use crate::same;
#[cfg(feature = "client")]
use crate::Error;

/// Extract a label or return an error
macro_rules! extract_label {
    ($obj:expr, $label:expr) => {
        match $obj.get($label) {
            Some(val) => val.clone(),
            None => return Err(Error::new(format!("Missing label {}", $label))),
        }
    };
}

/// A requisition for a pod to be deployed
#[derive(Debug, Hash, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct Requisition {
    /// The user the requisition is for
    pub user: String,
    /// The group the requisition is for
    pub group: String,
    /// The pipeline the requisition is for
    pub pipeline: String,
    /// The stage the pipeline is for
    pub stage: String,
}

impl Requisition {
    /// Create a new requisition object
    pub fn new<T: Into<String>>(user: T, group: T, pipeline: T, stage: T) -> Self {
        Requisition {
            user: user.into(),
            group: group.into(),
            pipeline: pipeline.into(),
            stage: stage.into(),
        }
    }

    /// Scope this requisition to a specific node
    ///
    /// # Arguments
    ///
    /// * `node` - The name of the node this req is scoped too
    pub fn to_scoped<T: Into<String>>(self, node: T) -> ScopedRequisition {
        ScopedRequisition {
            node: node.into(),
            user: self.user,
            group: self.group,
            pipeline: self.pipeline,
            stage: self.stage,
        }
    }

    /// Create a requisition from labels on a pod
    ///
    /// This is used to compare currently living pods with pods that we have scaled this round.
    ///
    /// # Arguments
    ///
    /// * `labels` - The labels from a pod in k8s
    #[cfg(feature = "client")]
    pub fn from_labels(labels: &BTreeMap<String, String>) -> Result<Requisition, Error> {
        // extract labels
        let user = extract_label!(labels, "user");
        let group = extract_label!(labels, "group");
        let pipeline = extract_label!(labels, "pipeline");
        let stage = extract_label!(labels, "stage");
        // cast to a requisition
        let req = Requisition {
            user,
            group,
            pipeline,
            stage,
        };
        Ok(req)
    }
}

impl fmt::Display for Requisition {
    /// Cleanly print a requisition
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}",
            self.user, self.group, self.pipeline, self.stage
        )
    }
}

impl From<Deadline> for Requisition {
    fn from(deadline: Deadline) -> Self {
        Requisition {
            user: deadline.creator,
            group: deadline.group,
            pipeline: deadline.pipeline,
            stage: deadline.stage,
        }
    }
}

impl From<&Deadline> for Requisition {
    fn from(deadline: &Deadline) -> Self {
        Requisition {
            user: deadline.creator.clone(),
            group: deadline.group.clone(),
            pipeline: deadline.pipeline.clone(),
            stage: deadline.stage.clone(),
        }
    }
}

impl From<ScopedRequisition> for Requisition {
    fn from(scoped: ScopedRequisition) -> Self {
        Requisition {
            user: scoped.user,
            group: scoped.group,
            pipeline: scoped.pipeline,
            stage: scoped.stage,
        }
    }
}

impl From<&ScopedRequisition> for Requisition {
    fn from(scoped: &ScopedRequisition) -> Self {
        Requisition {
            user: scoped.user.clone(),
            group: scoped.group.clone(),
            pipeline: scoped.pipeline.clone(),
            stage: scoped.stage.clone(),
        }
    }
}

/// Extract a value from a hashmap or throw a generic error
#[cfg(feature = "client")]
macro_rules! extract_generic {
    ($map:expr, $key:expr) => {
        match $map.remove($key) {
            Some(val) => val,
            None => return Err(Error::new(format!("HashMap missing value {}", $key))),
        }
    };
}

#[cfg(feature = "client")]
impl TryFrom<BTreeMap<String, String>> for Requisition {
    type Error = Error;

    fn try_from(mut map: BTreeMap<String, String>) -> Result<Self, Self::Error> {
        let req = Requisition {
            user: extract_generic!(map, "user"),
            group: extract_generic!(map, "group"),
            pipeline: extract_generic!(map, "pipeline"),
            stage: extract_generic!(map, "stage"),
        };
        Ok(req)
    }
}

impl PartialEq<BTreeMap<String, String>> for Requisition {
    /// Check if a [`Group`] contains all the updates from a [`GroupUpdate`]
    ///
    /// # Arguments
    ///
    /// * `labels` - The labels to compare against
    fn eq(&self, labels: &BTreeMap<String, String>) -> bool {
        // if any of the values don't match then return false
        same!(Some(&self.user), labels.get("user"));
        same!(Some(&self.pipeline), labels.get("pipeline"));
        same!(Some(&self.stage), labels.get("stage"));
        true
    }
}

/// A requisition for a pod to be deployed to a specific node
#[derive(Debug, Hash, Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct ScopedRequisition {
    /// The node this cluster is being handled by
    pub node: String,
    /// The user the requisition is for
    pub user: String,
    /// The group the requisition is for
    pub group: String,
    /// The pipeline the requisition is for
    pub pipeline: String,
    /// The stage the pipeline is for
    pub stage: String,
}

impl ScopedRequisition {
    /// Create a new scoped requisition object
    pub fn new<T: Into<String>, V: Into<String>>(
        node: T,
        user: V,
        group: V,
        pipeline: V,
        stage: V,
    ) -> Self {
        ScopedRequisition {
            node: node.into(),
            user: user.into(),
            group: group.into(),
            pipeline: pipeline.into(),
            stage: stage.into(),
        }
    }
}

impl fmt::Display for ScopedRequisition {
    /// Cleanly print a requisition
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}:{}:{}",
            self.node, self.user, self.group, self.pipeline, self.stage
        )
    }
}

#[cfg(feature = "client")]
impl TryFrom<&String> for ScopedRequisition {
    type Error = Error;

    /// Try to convert a string to a scoped requisition
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw string to convert from
    fn try_from(raw: &String) -> Result<Self, Error> {
        // split this string by ':' into its separate parts
        let chunks = raw.split(':').collect::<Vec<_>>();
        // if we didn't find 5 chunks then throw an error
        if chunks.len() == 5 {
            // build our scoped requisition
            let scoped_req = ScopedRequisition {
                node: chunks[0].to_owned(),
                user: chunks[1].to_owned(),
                group: chunks[2].to_owned(),
                pipeline: chunks[3].to_owned(),
                stage: chunks[4].to_owned(),
            };
            Ok(scoped_req)
        } else {
            Err(Error::new(format!(
                "{raw} cannot be cast to a scoped requisition!",
            )))
        }
    }
}

// Updated info on a spawned resources on a specific cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnedUpdate {
    /// The req that was tied to this pod
    pub req: Requisition,
    /// The node this req was spawned on
    pub node: String,
    /// The unique name for this resource
    pub name: String,
    /// The pool this worker was spawned in
    pub pool: Pools,
    /// The resources in use by this spawned resource
    pub resources: Resources,
    /// Whether this resource has been told to scale down yet or not
    pub scaled_down: bool,
}

impl From<Worker> for SpawnedUpdate {
    /// cast our worker to a spawned update
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to build a spawned update for
    fn from(worker: Worker) -> Self {
        // build the req that spawned this worker
        let req = Requisition::new(worker.user, worker.group, worker.pipeline, worker.stage);
        SpawnedUpdate {
            req,
            node: worker.node,
            name: worker.name,
            pool: worker.pool,
            resources: worker.resources,
            scaled_down: false,
        }
    }
}
