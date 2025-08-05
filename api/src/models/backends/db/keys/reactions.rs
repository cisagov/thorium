use uuid::Uuid;

use crate::models::{Reaction, ReactionStatus};
use crate::utils::Shared;

/// The keys to use to access [Reaction] data/sets
pub struct ReactionKeys {
    // The key to store/retrieve reaction data at
    pub data: String,
    /// The key to store/retrieve logs at
    pub logs: String,
    /// The key to reaction set
    pub set: String,
    /// The key to reaction set
    pub jobs: String,
    /// The key to the group wide reaction sorted set based on status
    pub group_set: String,
    /// The key to all sub reactions for this reaction
    pub sub: String,
}

impl ReactionKeys {
    /// Builds the keys to access [Reaction] data/sets in redis
    ///
    /// # Arguments
    ///
    /// * `reaction` - Reaction object to build keys for
    /// * `shared` - Shared Thorium objects
    pub fn new(reaction: &Reaction, shared: &Shared) -> Self {
        // build key to store reaction data at
        let data = Self::data(&reaction.group, &reaction.id, shared);
        // build key to reaction logs
        let logs = Self::logs(&reaction.group, &reaction.id, shared);
        // build key to reaction set
        let set = Self::set(&reaction.group, &reaction.pipeline, shared);
        // build key to jobs in this reaction
        let jobs = Self::jobs(&reaction.group, &reaction.id, shared);
        // build key to the sorted set of reactions for this group
        let group_set = Self::group_set(&reaction.group, &reaction.status, shared);
        // build key to sub reactions set
        let sub = ReactionKeys::sub_set(&reaction.group, &reaction.id, shared);
        // build key object
        ReactionKeys {
            data,
            logs,
            set,
            jobs,
            group_set,
            sub,
        }
    }

    /// Builds reaction data key
    ///
    /// # Arguments
    ///
    /// * `group` - The group the job is in
    /// * `id` - The uuidv4 of the job
    /// * `shared` - Shared Thorium objects
    pub fn data(group: &str, id: &Uuid, shared: &Shared) -> String {
        format!(
            "{ns}:reaction_data:{group}:{id}",
            ns = shared.config.thorium.namespace,
            group = group,
            id = id
        )
    }

    /// Builds reaction data key with id as a string
    ///
    /// # Arguments
    ///
    /// * `group` - The group the job is in
    /// * `id` - The uuidv4 of the job
    /// * `shared` - Shared Thorium objects
    pub fn data_str(group: &str, id: &str, shared: &Shared) -> String {
        format!(
            "{ns}:reaction_data:{group}:{id}",
            ns = shared.config.thorium.namespace,
            group = group,
            id = id
        )
    }

    /// Builds reactions status set key
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `pipeline` - The pipeline the reaction is built around
    /// * `status` - The status of the reaction
    /// * `shared` - Shared Thorium objects
    pub fn status<'a>(
        group: &'a str,
        pipeline: &'a str,
        status: &ReactionStatus,
        shared: &Shared,
    ) -> String {
        format!(
            "{ns}:reactions:{group}:{pipeline}:{status}",
            ns = shared.config.thorium.namespace,
            group = group,
            pipeline = pipeline,
            status = status
        )
    }

    /// Builds key to the set of [Reactions] for a pipeline
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `pipeline` - The pipeline the reaction is built around
    /// * `shared` - Shared Thorium objects
    pub fn set<'a>(group: &'a str, pipeline: &'a str, shared: &Shared) -> String {
        format!(
            "{ns}:reactions:{group}:{pipeline}",
            ns = shared.config.thorium.namespace,
            group = group,
            pipeline = pipeline
        )
    }

    /// Builds key to the set of jobs for a [Reaction]
    ///
    /// # Arguments
    ///
    /// * `group` - The group the job is in
    /// * `id` - The uuidv4 of the reaction this job is for
    /// * `shared` - Shared Thorium objects
    pub fn jobs(group: &str, id: &Uuid, shared: &Shared) -> String {
        format!(
            "{ns}:reaction_jobs:{group}:{id}",
            ns = shared.config.thorium.namespace,
            group = group,
            id = id
        )
    }

    /// Builds key to the set of jobs for a [Reaction] with id as a str
    ///
    /// # Arguments
    ///
    /// * `group` - The group the job is in
    /// * `id` - The uuidv4 of the reaction this job is for
    /// * `shared` - Shared Thorium objects
    pub fn jobs_str(group: &str, id: &str, shared: &Shared) -> String {
        format!(
            "{ns}:reaction_jobs:{group}:{id}",
            ns = shared.config.thorium.namespace,
            group = group,
            id = id
        )
    }

    /// Builds reaction log list key
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `pipeline` - The pipeline the reaction is built around
    /// * `reactions` - The reaction id
    /// * `shared` - Shared Thorium objects
    pub fn logs(group: &str, reaction: &Uuid, shared: &Shared) -> String {
        format!(
            "{ns}:logs:{group}:{reaction}",
            ns = shared.config.thorium.namespace,
            group = group,
            reaction = reaction
        )
    }

    /// Builds key to stage logs
    ///
    /// # Arguments
    ///
    /// * `reactions` - The reaction id
    /// * `stage` - The name of the stage
    /// * `shared` - Shared Thorium objects
    pub fn stage_logs(reaction: &Uuid, stage: &str, shared: &Shared) -> String {
        format!(
            "{ns}:stage_logs:{reaction}:{stage}",
            ns = shared.config.thorium.namespace,
            reaction = reaction,
            stage = stage
        )
    }

    /// Builds key to a reaction tag set
    pub fn tag(group: &str, tag: &str, shared: &Shared) -> String {
        format!(
            "{ns}:reaction_tags:{group}:{tag}",
            ns = shared.config.thorium.namespace,
            group = group,
            tag = tag,
        )
    }

    /// Builds key to the sorted set of [Reactions] for an entire group
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `status` - The current status of this reaction
    /// * `shared` - Shared Thorium objects
    pub fn group_set(group: &str, status: &ReactionStatus, shared: &Shared) -> String {
        format!(
            "{ns}:reactions_group:{group}:{status}",
            ns = shared.config.thorium.namespace,
            group = group,
            status = status
        )
    }

    /// Builds key to the sorted set of sub[Reactions] for a Reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `reaction` - The parent reaction id
    /// * `shared` - Shared Thorium objects
    pub fn sub_set(group: &str, reaction: &Uuid, shared: &Shared) -> String {
        format!(
            "{ns}:sub_reactions:{group}:{reaction}",
            ns = shared.config.thorium.namespace,
            group = group,
            reaction = reaction,
        )
    }

    /// Builds key to the sorted set of sub[Reactions] for a Reaction
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `reaction` - The parent reaction id
    /// * `status` - The current status of this reaction
    /// * `shared` - Shared Thorium objects
    pub fn sub_status_set(
        group: &str,
        reaction: &Uuid,
        status: &ReactionStatus,
        shared: &Shared,
    ) -> String {
        format!(
            "{ns}:sub_reactions:{group}:{reaction}:{status}",
            ns = shared.config.thorium.namespace,
            group = group,
            reaction = reaction,
            status = status
        )
    }

    /// Builds key to the currently active set of generators
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `reaction` - The parent reaction id
    /// * `status` - The current status of this reaction
    /// * `shared` - Shared Thorium objects
    pub fn generators(group: &str, reaction: &Uuid, shared: &Shared) -> String {
        format!(
            "{ns}:generators:{group}:{reaction}",
            ns = shared.config.thorium.namespace,
            group = group,
            reaction = reaction,
        )
    }

    /// Builds key to the currently active set of generators
    ///
    /// # Arguments
    ///
    /// * `group` - The group the reaction is in
    /// * `reaction` - The parent reaction id
    /// * `status` - The current status of this reaction
    /// * `shared` - Shared Thorium objects
    pub fn generators_str(group: &str, reaction: &str, shared: &Shared) -> String {
        format!(
            "{ns}:generators:{group}:{reaction}",
            ns = shared.config.thorium.namespace,
            group = group,
            reaction = reaction,
        )
    }
}

/// Keys to all sub reaction lists for a reaction
pub struct SubReactionLists {
    /// A list of reactions that are currently in the created state
    pub created: String,
    /// A list of reactions that are currently in the started state
    pub started: String,
    /// A list of reactions that are currently in the completed state
    pub completed: String,
    /// A list of reactions that are currently in the failed state
    pub failed: String,
}

impl SubReactionLists {
    /// Create a new SubReactionLists from a reaction
    ///
    /// # Arguments
    ///
    /// * `reaction` - The reaction to get sub reaction status lists for
    /// * `shared` - Shared Thorium objects
    pub fn new(reaction: &Reaction, shared: &Shared) -> Self {
        let created = ReactionKeys::sub_status_set(
            &reaction.group,
            &reaction.id,
            &ReactionStatus::Created,
            shared,
        );
        let started = ReactionKeys::sub_status_set(
            &reaction.group,
            &reaction.id,
            &ReactionStatus::Created,
            shared,
        );
        let completed = ReactionKeys::sub_status_set(
            &reaction.group,
            &reaction.id,
            &ReactionStatus::Created,
            shared,
        );
        let failed = ReactionKeys::sub_status_set(
            &reaction.group,
            &reaction.id,
            &ReactionStatus::Created,
            shared,
        );
        SubReactionLists {
            created,
            started,
            completed,
            failed,
        }
    }
}
