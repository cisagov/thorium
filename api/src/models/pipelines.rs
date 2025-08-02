//! Handles interactions related to pipelines in the backend
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::value::Value;
use uuid::Uuid;

use super::EventTrigger;
use super::bans::Ban;
use crate::{
    matches_adds_map, matches_clear, matches_clear_opt, matches_removes_map, matches_update,
    matches_update_opt, same,
};

/// A request for a pipeline in Thorium
///
/// This is almost exactly the same as Pipeline but with a jsonvalue for order
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineRequest {
    /// The group this pipeline is tied to
    pub group: String,
    /// The name of this pipeline
    pub name: String,
    /// The order of images to be executed in this pipeline
    pub order: Value,
    /// The number of seconds we have to meet this pipelines SLA. It defaults
    /// to 1 week if no SLA is given.
    pub sla: Option<u64>,
    /// The triggers to execute this pipeline on
    #[serde(default)]
    pub triggers: HashMap<String, EventTrigger>,
    /// The description for this pipeline
    pub description: Option<String>,
}

impl PipelineRequest {
    /// Compare the order from a [`PipelineRequest`] and a [`Pipeline`]
    #[must_use]
    pub fn compare_order(&self, order: &[Vec<String>]) -> bool {
        // make sure order is an array
        if !self.order.is_array() {
            return false;
        }

        // convert pipeline request order and iteratively check
        let stages = self.order.as_array().unwrap();
        for (i, stage) in stages.iter().enumerate() {
            // normalize stage to a Vec<Value>
            let wrapped = match stage.is_array() {
                true => stage.as_array().unwrap().to_owned(),
                false => match stage.is_string() {
                    true => vec![stage.to_owned()],
                    false => return false,
                },
            };
            // normalize all Values to strings
            let mut normalized = Vec::with_capacity(wrapped.len());
            for image in wrapped {
                match image.is_string() {
                    true => normalized.push(image.as_str().unwrap().to_owned()),
                    false => return false,
                }
            }

            // make sure the two vectors are the same length
            if normalized.len() != order[i].len() {
                return false;
            }

            // make sure the values in the vector are the same
            let same = normalized
                .iter()
                .zip(&order[i])
                .all(|(norm, ord)| norm == ord);
            if !same {
                return false;
            }
        }

        true
    }
}

impl PipelineRequest {
    /// Creates a new [`PipelineRequest`] for creating a pipeline in Thorium
    ///
    /// The order can either be a `Vec<String>` or a `Vec<Vec<String>>`. To allow users to have jobs
    /// run in parallel. 0 is the highest priority while 255 is the lowest.
    ///
    /// # Arguments
    ///
    /// * `group` - The group this pipeline should be in
    /// * `name` - The name of this pipeline
    /// * `order` - The order images should be executed in this pipeline
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::models::PipelineRequest;
    ///
    /// // create request for a pipeline in group corn with 3 sequential images
    /// let order = serde_json::json!(vec!("plant", "grow", "harvest"));
    /// PipelineRequest::new("Corn", "cycle", order)
    ///     .sla(604800);
    /// ```
    ///
    /// ```
    /// use thorium::models::PipelineRequest;
    ///
    /// // create request for a pipeline in group corn with 2 parallel images
    /// let order = serde_json::json!(vec!(vec!("plant"), vec!("grow", "fertilize"), vec!("harvest")));
    /// PipelineRequest::new("Corn", "cycle", order)
    ///     .sla(86400);
    /// ```
    pub fn new<S, T>(group: S, name: T, order: serde_json::Value) -> Self
    where
        S: Into<String>,
        T: Into<String>,
    {
        PipelineRequest {
            group: group.into(),
            name: name.into(),
            order,
            sla: None,
            triggers: HashMap::default(),
            description: None,
        }
    }

    /// Sets the SLA for a [`PipelineRequest`]
    ///
    /// The SLA is in seconds and is weakly enforced. This means that Thorium will do its best to
    /// ensure reactions for this pipeline meet the requested SLA but it is not a guarantee.
    ///
    /// # Arguments
    ///
    /// * `sla` - The SLA in seconds to request
    #[must_use]
    pub fn sla(mut self, sla: u64) -> Self {
        self.sla = Some(sla);
        self
    }

    /// Adds a trigger to a [`PipelineRequest`]
    ///
    /// This will allow Thorium to spawn this pipeline anytime a file matching these tags is uploaded.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the trigger to add
    /// * `trigger` - The trigger to add
    #[must_use]
    pub fn trigger<T: Into<String>>(mut self, name: T, trigger: EventTrigger) -> Self {
        // insert our new trigger
        self.triggers.insert(name.into(), trigger);
        self
    }

    /// Sets the description for a [`PipelineRequest`]
    ///
    /// # Arguments
    ///
    /// * `description` - The description of the pipeline to set
    #[must_use]
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl From<Pipeline> for PipelineRequest {
    /// Convert a pipeline into a pipeline request
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The pipeline to convert into a pipeline request
    fn from(pipeline: Pipeline) -> Self {
        // convert our pipeline order into a value
        let order = serde_json::Value::from(pipeline.order);
        // build the request to recreate this pipeline
        PipelineRequest {
            group: pipeline.group,
            name: pipeline.name,
            order,
            sla: Some(pipeline.sla),
            triggers: pipeline.triggers,
            description: pipeline.description,
        }
    }
}

/// A list of pipeline names with a cursor
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineList {
    /// A cursor used to page through pipeline names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of pipeline names
    pub names: Vec<String>,
}

/// A list of pipeline details with a cursor
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineDetailsList {
    /// A cursor used to page through pipeline details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<usize>,
    /// A list of pipeline details
    pub details: Vec<Pipeline>,
}

/// Helps default a serde value to false
// TODO: remove this when https://github.com/serde-rs/serde/issues/368 is resolved
fn default_as_false() -> bool {
    false
}

/// An update to the pipeline ban list containing bans to be added or removed
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineBanUpdate {
    /// The list of bans to be added
    pub bans_added: Vec<PipelineBan>,
    /// The list of bans to be removed
    pub bans_removed: Vec<Uuid>,
}

impl PipelineBanUpdate {
    /// Returns true if no bans are set to be added or removed
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bans_added.is_empty() && self.bans_removed.is_empty()
    }

    /// Add a ban to be added to the pipeline ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `ban` - The ban to be added
    #[must_use]
    pub fn add_ban(mut self, ban: PipelineBan) -> Self {
        self.bans_added.push(ban);
        self
    }

    /// Add multiple bans to be added to the pipeline ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `bans` - The bans to be added
    #[must_use]
    pub fn add_bans(mut self, mut bans: Vec<PipelineBan>) -> Self {
        self.bans_added.append(&mut bans);
        self
    }

    /// Add a ban to be removed from the pipeline ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the ban to be removed
    #[must_use]
    pub fn remove_ban(mut self, id: Uuid) -> Self {
        self.bans_removed.push(id);
        self
    }

    /// Add multiple bans to be removed from the pipeline ban list in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `ids` - The id's of the bans to be removed
    #[must_use]
    pub fn remove_bans(mut self, mut ids: Vec<Uuid>) -> Self {
        self.bans_removed.append(&mut ids);
        self
    }
}

/// A update for a pipeline in Thorium
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineUpdate {
    /// The order of images to be executed in this pipeline
    pub order: Option<Value>,
    /// The sla of a pipeline in seconds
    pub sla: Option<u64>,
    /// The new triggers to execute this pipeline on
    #[serde(default)]
    pub triggers: HashMap<String, EventTrigger>,
    /// The triggers to remove
    #[serde(default)]
    pub remove_triggers: Vec<String>,
    /// The description of the pipeline
    pub description: Option<String>,
    /// Whether to clear the description
    #[serde(default = "default_as_false")]
    pub clear_description: bool,
    /// An update to the ban list containing a list of bans to add or remove
    #[serde(default)]
    pub bans: PipelineBanUpdate,
}

impl PipelineUpdate {
    /// Sets the updated order for a pipeline, converting the
    /// Vec<Vec<String> order to a JSON [`Value`]
    ///
    /// # Arguments
    ///
    /// * `order` - The new order to set
    ///
    /// ```
    /// use thorium::models::PipelineUpdate;
    ///
    /// let order = vec![
    ///     vec!["pipeline1".to_string(), "pipeline2".to_string()],
    ///     vec!["pipeline3".to_string()]
    /// ];
    /// let update = PipelineUpdate::default().order(order);
    /// ```
    #[must_use]
    pub fn order(mut self, order: Vec<Vec<String>>) -> Self {
        self.order = Some(serde_json::json!(order));
        self
    }

    /// Sets the updated sla for a pipeline
    ///
    /// # Arguments
    ///
    /// * `sla` - The new sla to set
    ///
    /// ```
    /// use thorium::models::PipelineUpdate;
    ///
    /// let update = PipelineUpdate::default().sla(100);
    /// ```
    #[must_use]
    pub fn sla(mut self, sla: u64) -> Self {
        self.sla = Some(sla);
        self
    }

    /// Sets a list of triggers to add to a pipeline
    ///
    /// # Arguments
    ///
    /// * `triggers` - The triggers to add
    ///
    /// ```
    /// use thorium::models::{
    ///     PipelineUpdate,
    ///     events::EventTrigger,
    /// };
    /// use std::collections::HashMap;
    ///
    /// let mut triggers: HashMap<String, EventTrigger> = HashMap::new();
    /// triggers.insert("Trigger1".to_string(), EventTrigger::NewSample);
    /// let update = PipelineUpdate::default().triggers(triggers);
    /// ```
    #[must_use]
    pub fn triggers(mut self, triggers: HashMap<String, EventTrigger>) -> Self {
        self.triggers = triggers;
        self
    }

    /// Sets a list of triggers to be removed from a pipeline
    ///
    /// Overrides the `triggers` option, meaning triggers added in the `triggers`
    /// option will not be added if included in `remove_triggers`
    ///
    /// # Arguments
    ///
    /// * `remove_triggers` - The triggers to remove
    ///
    /// ```
    /// use thorium::models::PipelineUpdate;
    ///
    /// let update = PipelineUpdate::default().remove_triggers(vec!["Trigger1".to_string(), "Trigger2".to_string()]);
    /// ```
    #[must_use]
    pub fn remove_triggers(mut self, remove_triggers: Vec<String>) -> Self {
        self.remove_triggers = remove_triggers;
        self
    }

    /// Sets the updated description for a given pipeline
    ///
    /// This is overridden by the `clear_description` option
    ///
    /// # Arguments
    ///
    /// * `description` - The new description to set
    ///
    /// ```
    /// use thorium::models::PipelineUpdate;
    ///
    /// let update = PipelineUpdate::default().description("Updated description");
    /// ```
    #[must_use]
    pub fn description<T: Into<String>>(mut self, description: T) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the clear description flag to true
    ///
    /// This will clear the pipeline's current description and set it to None.
    ///
    /// ```
    /// use thorium::models::PipelineUpdate;
    ///
    /// PipelineUpdate::default().clear_description();
    /// ```
    #[must_use]
    pub fn clear_description(mut self) -> Self {
        self.clear_description = true;
        self
    }

    /// Set the pipeline bans to add/remove
    ///
    /// # Arguments
    ///
    /// * `bans` - The bans to add/remove
    /// ```
    /// use thorium::models::{PipelineUpdate, PipelineBanUpdate, PipelineBan, PipelineBanKind};
    ///
    /// // create a new ban
    /// let ban = PipelineBan::new(PipelineBanKind::generic("Example pipeline ban!"));
    /// // create a ban update with the new ban
    /// let ban_update = PipelineBanUpdate::default().add_ban(ban);
    /// // add the ban_update to the pipeline update
    /// PipelineUpdate::default().bans(ban_update);
    /// ```
    #[must_use]
    pub fn bans(mut self, bans: PipelineBanUpdate) -> Self {
        self.bans = bans;
        self
    }
}

/// The various kinds of bans a pipeline can have
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum PipelineBanKind {
    /// A generic ban manually set by an admin
    Generic(GenericBan),
    /// Created when an image in a pipeline has one or more bans
    BannedImage(BannedImageBan),
}

impl PipelineBanKind {
    /// Create a new [`PipelineBanKind::Generic`] ban type
    ///
    /// # Arguments
    ///
    /// * `msg` - The message to set for the ban
    pub fn generic<T: Into<String>>(msg: T) -> Self {
        Self::Generic(GenericBan { msg: msg.into() })
    }

    /// Create a new [`PipelineBanKind::BannedImage`] ban type
    ///
    /// # Arguments
    ///
    /// * `image` - The image in the pipeline that is banned
    pub fn image_ban<T: Into<String>>(image: T) -> Self {
        Self::BannedImage(BannedImageBan {
            image: image.into(),
        })
    }
}

/// Contains data related to a [`PipelineBanKind::Generic`]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GenericBan {
    /// A message containing the reason this pipeline was banned
    pub msg: String,
}

/// Contains data related to a [`PipelineBanKind::BannedImage`]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct BannedImageBan {
    /// The image in the pipeline that is banned
    pub image: String,
}

/// A particular reason an image has been banned
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineBan {
    /// The unique id for this ban
    pub id: Uuid,
    /// The time in UTC that the ban was made
    pub time_banned: DateTime<Utc>,
    /// The kind of ban this is
    pub ban_kind: PipelineBanKind,
}

impl PipelineBan {
    /// Create a new `PipelineBan`
    ///
    /// # Arguments
    ///
    /// * `ban_type` - The kind of ban we're creating
    #[must_use]
    pub fn new(ban_kind: PipelineBanKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            time_banned: Utc::now(),
            ban_kind,
        }
    }
}

impl Ban<Pipeline> for PipelineBan {
    fn id(&self) -> &Uuid {
        &self.id
    }

    fn msg(&self) -> String {
        // create a message based on the kind of ban
        match &self.ban_kind {
            PipelineBanKind::Generic(ban) => ban.msg.clone(),
            PipelineBanKind::BannedImage(ban) => {
                format!(
                    "The image '{}' has one or more bans! See the image's details for more info.",
                    ban.image
                )
            }
        }
    }

    fn time_banned(&self) -> &DateTime<Utc> {
        &self.time_banned
    }
}

/// A Pipeline that Thorium will build reactions around
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Pipeline {
    /// The group this pipeline is tied to
    pub group: String,
    /// The name of this pipeline
    pub name: String,
    /// The creator of this pipeline
    pub creator: String,
    /// The order of images to be executed in this pipeline
    pub order: Vec<Vec<String>>,
    /// The number of seconds we have to meet this pipelines SLA.
    pub sla: u64,
    /// The triggers to execute this pipeline on
    pub triggers: HashMap<String, EventTrigger>,
    /// The description of the pipeline
    pub description: Option<String>,
    /// A list of reasons the pipeline is banned mapped by ban UUID;
    /// if the list has any bans, the pipeline cannot be run
    pub bans: HashMap<Uuid, PipelineBan>,
}

impl PartialEq<PipelineRequest> for Pipeline {
    /// Check if a [`PipelineRequest`] and a [`Pipeline`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The `PipelineRequest` to compare against
    fn eq(&self, request: &PipelineRequest) -> bool {
        // make sure all fields are the same
        same!(self.name, request.name);
        same!(self.group, request.group);
        same!(request.compare_order(&self.order), true);
        same!(&self.sla, request.sla.as_ref().unwrap_or(&604_800));
        same!(&self.triggers, &request.triggers);
        same!(&self.description, &request.description);
        true
    }
}

impl PartialEq<PipelineUpdate> for Pipeline {
    /// Check if a [`PipelineUpdate`] and a [`Pipeline`] are equal
    ///
    /// # Arguments
    ///
    /// * `request` - The `PipelineUpdate` to compare against
    #[rustfmt::skip]
    fn eq(&self, update: &PipelineUpdate) -> bool {
        // convert the update order to a Vec<Vec<String>> as in the pipeline order;
        // this means the update value must have been serialized from a Vec<Vec<String>>
        // for the Pipeline and PipelineUpdate to be equal
        matches_update!(self.order, update.order, |order: &Value| {
            let order = order.clone();
            serde_json::from_value::<Vec<Vec<String>>>(order)
        });
        matches_update!(self.sla, update.sla);
        // filter out any triggers from the adds list that would have been
        // removed by the removes list
        let mut triggers_added = update.triggers.iter().filter_map(|(trigger, event)| {
            if update.remove_triggers.contains(trigger) {
                None
            } else {
                Some((trigger, event))
            }
        });
        matches_adds_map!(self.triggers, triggers_added);
        matches_removes_map!(self.triggers, update.remove_triggers);
        matches_clear_opt!(
            self.description,
            update.description,
            update.clear_description
        );
        // filter out any bans from the adds list that would have been
        // removed by the removes list
        let mut bans_added = update.bans.bans_added.iter().filter_map(|ban| {
            if update.bans.bans_removed.contains(&ban.id) {
                None
            } else {
                Some((&ban.id, ban))
            }
        });
        matches_adds_map!(self.bans, bans_added);
        matches_removes_map!(self.bans, update.bans.bans_removed);
        true
    }
}

cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        use crate::models::backends::NotificationSupport;
        use crate::models::{KeySupport, NotificationType, PipelineKey};

        impl NotificationSupport for Pipeline {
            /// Provide the pipeline notification type
            fn notification_type() -> NotificationType {
                NotificationType::Pipelines
            }
        }

        impl KeySupport for Pipeline {
            /// The image's unique key to access its data in scylla
            type Key = PipelineKey;

            /// Images have no extra optional components for their keys
            type ExtraKey = ();

            /// Build the key for this pipeline if we need the key as one field
            ///
            /// # Arguments
            ///
            /// * `key` - The key to build from
            fn build_key(key: Self::Key, _extra: &Self::ExtraKey) -> String {
                serde_json::to_string(&key).expect("Failed to serialize pipeline key!")
            }

            /// Build a URL component composed of the key to access the resource
            ///
            /// # Arguments
            ///
            /// * `key` - The root part of this key
            /// * `extra` - Any extra info required to build this key
            fn key_url(key: &Self::Key, _extra: Option<&Self::ExtraKey>) -> String {
                // make a URL component made up of the group and pipeline
                format!("{}/{}", key.group, key.pipeline)
            }
        }
    }
}

/// A status summary for a specific image or stage tied to a pipeline
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct StageStats {
    /// The number of jobs for this image with a created status
    pub created: u64,
    /// The number of jobs for this image with a running status
    pub running: u64,
    /// The number of jobs for this image with a completed status
    pub completed: u64,
    /// The number of jobs for this image with a failed status
    pub failed: u64,
    /// The number of jobs for this image with a sleeping status
    pub sleeping: u64,
    /// The total number of jobs across all statuses
    pub total: u64,
}

/// A status summary for a specific pipeline
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineStats {
    /// A map of status summaries for each image/stage in this pipeline by user
    pub stages: HashMap<String, HashMap<String, StageStats>>,
}

impl PipelineStats {
    /// gets a total count for the number of stages with created or running jobs by user
    #[must_use]
    pub fn total(&self) -> usize {
        // add all the number of stage up for each user
        self.stages.values().map(HashMap::len).sum()
    }
}

/// Helps serde default the pipeline list limit to 50
fn default_list_limit() -> usize {
    50
}

/// The parameters for a pipeline list request
#[derive(Serialize, Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct PipelineListParams {
    /// The cursor id to user if one exists
    #[serde(default)]
    pub cursor: usize,
    /// The max amount of pipelines to return in on request
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}
