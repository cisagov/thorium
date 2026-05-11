//! Hunt through data using sigma rules

use futures::lock::Mutex;
use futures::stream::StreamExt;
use gxhash::GxHasher;
use linearize::StaticMap;
use papaya::HashMap as PapayaMap;
use sigma_rust::Rule;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;
use thorium::models::entities::rules::{SigmaActionToTake, SigmaAutoFlag};
use thorium::models::{
    AssociationKind, AssociationRequest, AssociationTarget, Entity, EntityKinds, EntityListOpts,
    EntityMetadata, EntityMetadataRequest, EntityRequest, Event, EventData, EventType, OutputKey,
    ScrubbedUser, SigmaRule, SigmaScannableResultsEvent, TreeNode, TreeOpts, TreeQuery,
};
use thorium::{Error, Thorium};
use tracing::{Level, event, instrument};
use uuid::Uuid;

use crate::EventWorkerCache;
use crate::libs::workers::{
    EventWorkerSupport, EventWorkerSupportCore, WorkerCacheUpdateKinds, WorkerCacheUpdates,
};

/// The stats for each entity
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct SigmaRuleSpecificStats {
    /// The total number of entities scanned
    pub scanned: usize,
    /// The total number of entities that have been created in Thorium
    pub created: usize,
    /// The total number of flags created
    pub flagged: usize,
}

impl std::ops::AddAssign for SigmaRuleSpecificStats {
    fn add_assign(&mut self, rhs: Self) {
        self.scanned = self.scanned.saturating_add(rhs.scanned);
        self.created = self.created.saturating_add(rhs.created);
        self.flagged = self.flagged.saturating_add(rhs.flagged);
    }
}

/// The stats for a sigma rule worker
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct SigmaRuleStats {
    /// The total across all entities regardless of kind
    pub total: SigmaRuleSpecificStats,
    /// The stats on a per entity basis
    pub by_entity: StaticMap<EntityKinds, SigmaRuleSpecificStats>,
}

impl std::ops::AddAssign for SigmaRuleStats {
    fn add_assign(&mut self, rhs: Self) {
        // add our total
        self.total += rhs.total;
        // add each entity kind
        for (kind, stats) in rhs.by_entity {
            // get an entry to this entity kinds stats
            self.by_entity[kind] += stats;
        }
    }
}
/// A compiled sigma rule
#[derive(Debug, Clone)]
pub struct CompiledSigmaRule {
    /// The original sigma rule in Thorium
    pub original: Entity,
    /// The compiled sigma rule
    pub compiled: Rule,
}

impl CompiledSigmaRule {
    /// Get the original sigma rule metadata
    pub fn get_original_rule(&self) -> Result<&SigmaRule, Error> {
        // make sure we have a sigma rule
        match &self.original.metadata {
            EntityMetadata::SigmaRule(rule) => Ok(rule),
            _ => {
                // build an error message saying we don't have a sigma rule
                let msg = format!("Compiled sigma rule does not contain a sigma rule?: {self:?}");
                // build and return an error
                Err(Error::new(msg))
            }
        }
    }
}

/// The context needed for this sigma rule worker
#[derive(Debug, Clone)]
pub struct SigmaRuleContext {
    /// The different rules to scan with
    pub combined: Vec<CompiledSigmaRule>,
    /// A map of entity kinds to rules
    pub applies_to: StaticMap<EntityKinds, Vec<usize>>,
}

impl SigmaRuleContext {
    /// Build a [`SigmaRuleContext`] from info in the Thorium API
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    pub async fn new(thorium: &Thorium) -> Result<Self, Error> {
        // build the options for listing entities
        let opts = EntityListOpts::default()
            .kind(EntityKinds::SigmaRule)
            // use a large page size to reduce round trips
            .page_size(2000);
        // get cursor over all sigma rule entities
        let mut cursor = thorium.entities.list_details(&opts).await?;
        // pre allocate a list for our rules
        let mut combined_list = Vec::with_capacity(cursor.data.len());
        // pre allocate a map for our different entity kinds and their rule indexes
        let mut applies_to_map = StaticMap::<EntityKinds, Vec<usize>>::default();
        // iterate over all the rules in Thorium and add them to our context
        loop {
            for original in cursor.data.drain(..) {
                // compile this sigma rule
                let compiled = match &original.metadata {
                    EntityMetadata::SigmaRule(sigma) => {
                        // compile this sigma rule
                        let compiled = sigma_rust::rule_from_yaml(&sigma.rule).unwrap();
                        // get the next available index that this new rule will go into
                        let index = combined_list.len();
                        // add this rules index to our rule map for all of its entity kinds
                        for applies_to in &sigma.applies_to {
                            // get the entity kind this is sigma applies too converts too
                            let kind = EntityKinds::from(applies_to);
                            // add this index
                            applies_to_map[kind].push(index);
                        }
                        // reutrn this compiled sigma rule
                        compiled
                    }
                    // we can't compile a non sigma rule entity
                    _ => {
                        // log that we trying to compile an entity thats not a sigma rule
                        event!(
                            Level::ERROR,
                            msg = "Can't compile a non sigma rule entity",
                            kind = original.kind.as_str()
                        );
                        // continue on and just ignore this bad entity
                        continue;
                    }
                };
                // combine our original and compiled sigma rule
                let combined = CompiledSigmaRule { original, compiled };
                // add our combined rule to our context
                combined_list.push(combined);
            }
            // if this cursor is finished then stop listing sigma rules
            if cursor.exhausted() {
                break;
            }
            // this cursor has more data so get the next page
            cursor.refill().await?;
        }
        // build a sigma rule context
        let context = SigmaRuleContext {
            combined: combined_list,
            applies_to: applies_to_map,
        };
        Ok(context)
    }
}

/// Either an existing assocation target or one that will be created once we create the target entity
#[derive(Debug)]
enum MaybeEntity {
    /// An entity that is going to be created
    New(usize),
    /// An existing entity
    Existing(AssociationTarget),
}

impl MaybeEntity {
    /// Get an association target from a list of entity responses if needed
    ///
    /// # Arguments
    ///
    /// * `resps` - The list of entity creation responses to get our id and names from
    pub fn get_target(self, resps: &[Option<AssociationTarget>]) -> Option<AssociationTarget> {
        match self {
            Self::New(index) => resps.get(index).cloned().flatten(),
            Self::Existing(target) => Some(target.clone()),
        }
    }

    /// Create a `MaybeEntity` for an existing entity
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the existing entity
    /// * `name` - The name of the existing entity
    pub fn new_existing(id: Uuid, name: impl Into<String>) -> Self {
        // build an association target for this entity
        let target = AssociationTarget::Entity {
            id,
            name: name.into(),
        };
        // wrap this target
        MaybeEntity::Existing(target)
    }
}

impl From<usize> for MaybeEntity {
    /// Wrap the index to an entity that will be created in a MaybeEntity
    ///
    /// # Arguments
    ///
    /// * `index` - The index to the entity that will be created
    fn from(index: usize) -> Self {
        MaybeEntity::New(index)
    }
}

impl From<Entity> for MaybeEntity {
    /// Wrap an existing entity in a `MaybeEntity`
    ///
    /// # Arguments
    ///
    /// * `entity` - An existing entity
    fn from(entity: Entity) -> Self {
        // build an association target for this entity
        let target = AssociationTarget::Entity {
            id: entity.id,
            name: entity.name,
        };
        // wrap this target
        MaybeEntity::Existing(target)
    }
}

impl From<AssociationTarget> for MaybeEntity {
    /// Wrap an `AssociationTarget` to an existing entity in a `MaybeEntity`
    ///
    /// # Arguments
    ///
    /// * `target` - The target to wrap
    fn from(target: AssociationTarget) -> Self {
        // wrap this target in an existing wrapper
        MaybeEntity::Existing(target)
    }
}

/// Create an association between things in Thorium
///
/// This function is called instead of just directly calling the client method to ensure our return type
/// is Result<(), Error>.
///
/// # Arguments
///
/// * `thorium` - A Thorium client
/// * `assoc_req` - The association to create
async fn create_association(thorium: &Thorium, assoc_req: AssociationRequest) -> Result<(), Error> {
    thorium.associations.create(&assoc_req).await?;
    Ok(())
}

/// Track whether this entity should be created, updated, or already exists
pub enum MaybeEntityCreate {
    /// This entity needs to be created
    Create(EntityRequest),
    /// This entity already exists
    Exists(AssociationTarget),
}

impl MaybeEntityCreate {
    /// Create an entity if it does not yet exist
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    pub async fn apply(self, thorium: &Thorium) -> Result<AssociationTarget, Error> {
        match self {
            MaybeEntityCreate::Create(req) => {
                // create this entity
                let resp = thorium.entities.create(req).await?;
                // build an association target for this entity
                let target = AssociationTarget::Entity {
                    id: resp.id,
                    name: resp.name,
                };
                Ok(target)
            }
            MaybeEntityCreate::Exists(target) => Ok(target),
        }
    }
}

/// A cache of actions to take after scanning a single sigma result
#[derive(Debug)]
struct SigmaActionCache {
    /// The original data that all scanned entities originate from
    key: OutputKey,
    /// The hashes for entities that we are creating
    pending_entities: HashMap<u64, usize>,
    /// The entities that we are creating
    entities: Vec<EntityRequest>,
    /// Any associations to create between two nodes in this cache (source, target, groups)
    associations: Vec<(AssociationKind, (MaybeEntity, MaybeEntity, Vec<String>))>,
}

impl SigmaActionCache {
    /// Create a new action cache
    ///
    /// # Arguments
    ///
    /// * `key` - The key to the original data are scanning sigma events from
    pub fn new(key: OutputKey) -> Self {
        SigmaActionCache {
            key,
            pending_entities: HashMap::default(),
            entities: Vec::default(),
            associations: Vec::default(),
        }
    }

    /// Add a new entity to create to our cache and return the index this for this new entity
    ///
    /// # Arguments
    ///
    /// * `rule` - The compiled that is causing us to create this entity
    /// * `req` - The entity to add
    /// * `stats` - The stats to use for tracking what entities have been created
    pub fn add_entity(
        &mut self,
        rule: &CompiledSigmaRule,
        mut req: EntityRequest,
        stats: &mut SigmaRuleStats,
    ) -> usize {
        // get the hash for this entity request
        let mut hasher = GxHasher::default();
        // hash this entity req
        req.metadata.hash(&mut hasher);
        // get this entities hash
        let hash = hasher.finish();
        // check if we already have added this entity
        match self.pending_entities.entry(hash) {
            // we already have added this entity
            Entry::Occupied(entry) => {
                // get the entities index in our entity vec
                let index = *entry.get();
                // get our existing entities groups that we may need to update
                let groups = &mut self.entities[index].groups;
                // get only the groups that are not already  in our group set
                for new_group in &rule.original.groups {
                    // only add the new groups
                    if !groups.contains(new_group) {
                        // add this group
                        groups.push(new_group.clone());
                    }
                }
                index
            }
            // we do not yet have this entity
            Entry::Vacant(entry) => {
                // get this entities stats
                let entity_stats = &mut stats.by_entity[req.kind()];
                // track that we are creating one of these entities
                entity_stats.created = entity_stats.created.saturating_add(1);
                // set the groups this entity should be created in
                req.groups = rule.original.groups.clone();
                // get the index our next entity will be added at
                let index = self.entities.len();
                // add this entity request
                self.entities.push(req);
                // insert this entities new index
                entry.insert(index);
                index
            }
        }
    }

    /// Add a new association to create
    ///
    /// # Arguments
    ///
    /// * `rule` - The compiled that is causing us to create this association
    /// * `kind` - The kind of association to create
    /// * `source` - The source side of this association
    /// * `target` - The target side of this association
    pub fn add_association<S: Into<MaybeEntity>, T: Into<MaybeEntity>>(
        &mut self,
        rule: &CompiledSigmaRule,
        kind: AssociationKind,
        source: S,
        target: T,
    ) {
        // add this new association to create
        self.associations.push((
            kind,
            (source.into(), target.into(), rule.original.groups.clone()),
        ))
    }

    /// Gather all parent entities
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `rule` - The compiled sigma rule that hit on some entity
    /// * `entity` - The entity to look for parents for
    /// * `entity_map` - The map of entities to look for parents in
    /// * `position` - The position to start looking at
    /// * `stats` - The stats to keep updated
    /// * `parents` - The set of parent entities to add too
    #[async_recursion::async_recursion]
    #[expect(clippy::too_many_arguments)]
    #[instrument(name = "SigmaActionCache::gather_parent_entities", skip_all)]
    async fn gather_parent_entities(
        &mut self,
        thorium: &Thorium,
        rule: &CompiledSigmaRule,
        entity: &EntityRequest,
        entity_map: &EntityMap,
        position: usize,
        stats: &mut SigmaRuleStats,
        parents: &mut Vec<(usize, EntityKinds)>,
    ) -> Result<(), Error> {
        // get the parent info for this entity if it has any parents
        if let Some(parent_info) = entity.parent_info() {
            // get the entities to search for our parent in
            let entities = entity_map.get(parent_info.entity_kind());
            // look for our parent process if it exists above us
            if let Some((index, parent_entity)) = entities[..position]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, req)| parent_info.is_parent(req))
            {
                // we have a parent entity so check if that has its own parent
                // gather any parents for our parents (grandparents)
                let index = self
                    .gather_parents(thorium, rule, parent_entity, entity_map, index, stats)
                    .await?;
                // set our parent index
                parents.push((index, parent_info.entity_kind()));
            }
        }
        Ok(())
    }

    /// Link the end of a chain of entities to our original data
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `rule` - The compiled sigma rule that hit on some entity
    /// * `root_kind` - The kind of root/end entity that we are linking
    /// * `root_index` - The index to the root/end entity that we are linking
    #[instrument(name = "SigmaActionCache::link_to_original", skip_all)]
    pub async fn link_to_original(
        &mut self,
        thorium: &Thorium,
        rule: &CompiledSigmaRule,
        root_kind: EntityKinds,
        root_index: usize,
    ) -> Result<(), Error> {
        // get the association kind between our original data and our root
        match &self.key {
            // get the association kind back to sample or repo
            OutputKey::Sample(sha256) => {
                // get the association kind between our original data and our root
                let kind = AssociationKind::to_parent_nonentity(root_kind);
                // build the sample association target to associate against
                let parent_target = AssociationTarget::File(sha256.clone());
                // add the association between our root node and our original data
                self.add_association(rule, kind, parent_target, root_index);
            }
            OutputKey::Repo(url) => {
                // get the association kind between our original data and our root
                let kind = AssociationKind::to_parent_nonentity(root_kind);
                // build the sample association target to associate against
                let parent_target = AssociationTarget::Repo(url.clone());
                // add the association between our root node and our original data
                self.add_association(rule, kind, parent_target, root_index);
            }
            // get the association kind back to an entity
            OutputKey::Entity(entity_id) => {
                // get this entity if it exists
                let entity = thorium.entities.get(*entity_id).await?;
                // get this entities kind
                let kind = entity.kind;
                // get the association kind between our original data and our root
                let kind = AssociationKind::from((kind, root_kind));
                // add the association between our root node and our original data
                self.add_association(rule, kind, entity, root_index);
            }
        };
        Ok(())
    }

    /// Gather all parent entities into this cache
    ///
    /// This returns the index to the child entity that we are gathering parents for.
    ///
    /// # Arguments
    ///
    /// * `rule` - The compiled sigma rule that hit on some entity
    /// * `entity` - The entity to look for parents for
    /// * `entity_map` - The map of entities to look for parents in
    /// * `position` - The position to start looking at
    /// * `stats` - The stats to keep updated
    #[async_recursion::async_recursion]
    #[instrument(name = "SigmaActionCache::gather_parents", skip_all)]
    pub async fn gather_parents(
        &mut self,
        thorium: &Thorium,
        rule: &CompiledSigmaRule,
        entity: &EntityRequest,
        entity_map: &EntityMap,
        position: usize,
        stats: &mut SigmaRuleStats,
    ) -> Result<usize, Error> {
        // track if we found any parent's index
        let mut parents = Vec::default();
        // get any parent entities from this entity
        self.gather_parent_entities(
            thorium,
            rule,
            entity,
            entity_map,
            position,
            stats,
            &mut parents,
        )
        .await?;
        // get the kind of entity for this child
        let child_kind = entity.kind();
        // we always add ourselves after trying to find our parent
        let child_index = self.add_entity(rule, entity.clone(), stats);
        // if we have any parents then we don't need to check for a root node kind
        if parents.is_empty() {
            // check if we have a root node to create
            if let Some(root_kind) = entity.root_kind() {
                // get our root nodes if we have any
                let entities = entity_map.get(root_kind);
                // for now just assume the first one
                let root = match entities.first() {
                    Some(root) => root,
                    None => {
                        // we are missing a root for an entity
                        let msg = format!(
                            "Missing root {root_kind} from {}:{}",
                            entity.kind(),
                            entity.name
                        );
                        // return an error for this missing root
                        return Err(Error::new(msg));
                    }
                };
                // add our root entity and get its index
                let root_index = self.add_entity(rule, root.clone(), stats);
                // get the association kind between these entities
                let kind = AssociationKind::from((root_kind, child_kind));
                // add the association to create
                self.add_association(rule, kind, root_index, child_index);
                // link our root node back to our original data
                self.link_to_original(thorium, rule, root_kind, root_index)
                    .await?;
            } else {
                // just link the final child back to our original data
                self.link_to_original(thorium, rule, child_kind, child_index)
                    .await?;
            }
        } else {
            // add associations to all of our our parents
            for (parent_index, parent_kind) in parents {
                // get the association kind between these entities
                let kind = AssociationKind::from((parent_kind, child_kind));
                // add the association to create
                self.add_association(rule, kind, parent_index, child_index);
            }
        }
        Ok(child_index)
    }

    /// Determine which entities we need to create, skip, or update
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    async fn filter_reqs(&mut self, thorium: &Thorium) -> Result<Vec<MaybeEntityCreate>, Error> {
        // build the query for getting the existing tree of entities
        let query = TreeQuery::from(self.key.clone());
        // we don't need to gather parents
        // TODO set better defaults to help limit what we crawl and speed up tree
        // growth
        let opts = TreeOpts::default().limit(6);
        // get a tree starting at our events source
        let tree = thorium.trees.start(&opts, &query).await?;
        // keep a map of entities that already exist
        let mut exists = HashMap::with_capacity(tree.data_map.len());
        // turn this tree into a map of entities
        for (_, node) in tree.data_map {
            // we only care about entity nodes
            if let TreeNode::Entity(entity) = node {
                // get this entities identifying hash
                let hash = entity.hash_identifying();
                // add this entity into our map
                exists.insert(hash, entity);
            }
        }
        // keep a list of what entities we should create
        let mut to_create = Vec::with_capacity(self.entities.len());
        // step over the entities we want to create and find out which should
        // be created, updated, or skipped
        for req in self.entities.drain(..) {
            // get this potential new entities hash
            let hash = req.hash_identifying();
            // get this entity if it already exists
            match exists.get(&hash) {
                Some(entity) => {
                    // build an association target for this entity
                    let target = AssociationTarget::Entity {
                        id: entity.id,
                        name: entity.name.clone(),
                    };
                    // this is an existing entity so we don't need to create it
                    to_create.push(MaybeEntityCreate::Exists(target));
                }
                // this is a new entity so create it
                None => to_create.push(MaybeEntityCreate::Create(req)),
            }
        }
        Ok(to_create)
    }

    /// Create all of our entities (including flags) and associations
    ///
    /// # Arguments
    ///
    /// * `thorium` A thorium client
    pub async fn create_all(mut self, thorium: &Thorium) -> Result<(), Error> {
        // don't just recreate any already existing entities
        let maybe_creates = self.filter_reqs(thorium).await?;
        // create all of our entities 10 at a time
        let targets = futures::stream::iter(maybe_creates)
            .map(|maybe_create| maybe_create.apply(thorium))
            .buffered(10)
            .collect::<Vec<Result<AssociationTarget, _>>>()
            .await;
        // pre allocate a vector to store our entity creation responses
        let mut resps = Vec::with_capacity(targets.len());
        // log any failures
        for result in targets {
            // log if this create failed
            match result {
                Ok(resp) => resps.push(Some(resp)),
                Err(error) => {
                    // log that we failed to create an entity
                    event!(Level::ERROR, creating = "Entity", %error);
                    // add a none so that we don't try to create dangling association
                    resps.push(None);
                }
            }
        }
        // preallocate a list to store our association requests
        let mut assoc_reqs = Vec::with_capacity(self.associations.len());
        // create the association requests for our new entities
        for (kind, (src, target, groups)) in self.associations {
            // get our source and target association targets
            // log an error if either of our targets are missing
            let (src, target) = match (src.get_target(&resps), target.get_target(&resps)) {
                (Some(src), Some(target)) => (src, target),
                _ => {
                    // log that we failed to create an association
                    event!(
                        Level::ERROR,
                        creating = "Association",
                        error = "Missing Source/Destination entity"
                    );
                    // continue on to the next association to create
                    continue;
                }
            };
            // build this association request
            let req = AssociationRequest::new(kind, src)
                .target(target)
                .groups(groups);
            // add this request
            assoc_reqs.push(req);
        }
        // create all of our associations 10 at a time
        let resp_results = futures::stream::iter(assoc_reqs)
            .map(|req| create_association(thorium, req))
            .buffer_unordered(10)
            .collect::<Vec<Result<(), _>>>()
            .await;
        // log any failures
        for result in resp_results {
            // log if this create failed
            if let Err(error) = result {
                // log that we failed to create an association
                event!(
                    Level::ERROR,
                    creating = "Association",
                    error = error.to_string()
                );
            }
        }
        Ok(())
    }
}

/// A map of entities and events sorted by what kind of entity this is
#[derive(Default)]
struct EntityMap {
    /// The inner map of entities
    pub entities: StaticMap<EntityKinds, Vec<EntityRequest>>,
    /// A map of entities turned into scannable events
    pub events: StaticMap<EntityKinds, Vec<sigma_rust::Event>>,
}

impl EntityMap {
    /// Get a specific kind of entity to add to our map
    ///
    /// # Arguments
    ///
    /// * `event` - The event we are populating entities for
    /// * `result_id` - The id of the results we are scanning entities from
    /// * `kind` - The kind of entities to get
    /// * `thorium` - A thorium client
    /// * `local_stats` - The local stats for this worker
    pub async fn populate(
        &mut self,
        event: &SigmaScannableResultsEvent,
        result_id: Uuid,
        kind: EntityKinds,
        thorium: &Thorium,
        local_stats: &mut SigmaRuleStats,
    ) -> Result<(), Error> {
        // download this kinds entity requests for this result
        let entities = event
            .key
            .get_entities(thorium, &event.tool, result_id, kind)
            .await?;
        // eagerly convert this entities into scannable rules
        for req in entities {
            // this kind should always be equal to the kind we requested
            // however to be defensive we will get the kind for each entity
            let req_kind = req.kind();
            // get this entities stats
            let entity_stats = &mut local_stats.by_entity[req_kind];
            // track that we scanned one of these entities
            entity_stats.scanned = entity_stats.scanned.saturating_add(1);
            // convert this entity to something that sigma can scan if possible
            if let Some(scannable) = req.metadata.to_sigma_scannable()? {
                // wrap this in a sigma event
                let sigma_event = sigma_rust::event_from_json(&scannable)?;
                // keep track of this event
                self.events[req_kind].push(sigma_event);
            }
            // add this to our entity map
            self.entities[req_kind].push(req);
        }
        Ok(())
    }

    /// Populate a new entity map
    ///
    /// # Arguments
    ///
    /// * `event` - The event we are creating an entity map for
    /// * `thorium` - A thorium client
    /// * `local_stats` - The local stats for this worker
    pub async fn new(
        event: &SigmaScannableResultsEvent,
        thorium: &Thorium,
        local_stats: &mut SigmaRuleStats,
    ) -> Result<Self, Error> {
        // get the results for this event
        let mut results = event.key.get_results(thorium, &event.tool).await?;
        // get the latest results
        let result = match results.results.remove(&event.tool) {
            Some(mut results) => results.swap_remove(0),
            // we can't create an entity map
            // this should basically never happen and events should be small enough
            // that we can just log the whole event to make debugging easier
            None => return Err(Error::new(format!("Missing results for {event:?}"))),
        };
        // start with a empty map of the different entities
        let mut entity_map = EntityMap::default();
        //  step over the entity kinds in this event
        for applies_to in &event.applies_to {
            // get the kind of entity this applies to maps too
            let entity_kind = EntityKinds::from(*applies_to);
            // get the entities for this kind
            entity_map
                .populate(event, result.id, entity_kind, thorium, local_stats)
                .await?;
            // also get any root entities for this entity kind
            for root_kind in entity_kind.root_kinds() {
                // get the entities for this kind
                entity_map
                    .populate(event, result.id, *root_kind, thorium, local_stats)
                    .await?;
            }
        }
        Ok(entity_map)
    }

    /// Get the entities for a kind of entity
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of events and entities to get
    pub fn get(&self, kind: EntityKinds) -> &Vec<EntityRequest> {
        &self.entities[kind]
    }
}

/// A single sigma rule worker
pub struct SigmaRuleWorker {
    /// The context for this sigma rule worker
    context: SigmaRuleContext,
    /// A map of Thorium clients for users in Thorium
    pub clients: PapayaMap<String, Thorium>,
    /// A client for Thorium that should only be used to create user clients
    thorium: Arc<Thorium>,
}

impl SigmaRuleWorker {
    /// Create a client for all of our users
    ///
    /// # Arguments
    ///
    /// * `users` - The users to create a client for
    pub fn create_user_clients(&mut self, users: HashMap<String, ScrubbedUser>) {
        for (username, user) in users {
            // build a client for each user
            let mut user_client = self.thorium.deref().clone();
            // set this client to masquerade as this user
            user_client.masquerade(&user);
            // add a client for this user
            self.clients.pin().insert(username, user_client);
        }
    }

    /// Create a flag for this rule hit
    ///
    /// # Arguments
    ///
    /// * `thorium` - A thorium client
    /// * `action_cache` - A cache of actions that this worker will eventually take
    /// * `rule` - The rule we are creating a flag for
    /// * `auto_flag` - The flag to create
    /// * `flagged_on` - The entity that we are flagging
    /// * `position` - The index of the entity we are flagging
    /// * `stats` - The local stats for this worker
    #[expect(clippy::too_many_arguments)]
    #[instrument(name = "SigmaRuleWorker::create_flag", skip_all)]
    async fn create_flag(
        &self,
        thorium: &Thorium,
        action_cache: &mut SigmaActionCache,
        rule: &CompiledSigmaRule,
        auto_flag: &SigmaAutoFlag,
        flagged_on: &EntityRequest,
        entity_map: &EntityMap,
        position: usize,
        stats: &mut SigmaRuleStats,
    ) -> Result<usize, Error> {
        // get the original rules metadata
        let rule_meta = rule.get_original_rule()?;
        // build the flag for this sigma rule
        let flag = auto_flag.to_flag(rule_meta);
        // build the metadata request for this flag
        let metadata = EntityMetadataRequest::Flag(flag);
        // use the same groups as our rule
        let groups = rule.original.groups.clone();
        // build the entity request for this flag
        let entity_req = EntityRequest::new(&rule.original.name, metadata, groups);
        // add the entity we flagged on and any parents we need to create
        let index = action_cache
            .gather_parents(thorium, rule, flagged_on, entity_map, position, stats)
            .await?;
        // add our flag entity to our action cache
        let flag_index = action_cache.add_entity(rule, entity_req, stats);
        // add the info to associate our flag with our entity
        action_cache.add_association(rule, AssociationKind::FlagFor, flag_index, index);
        // our original rule entity already exists
        let original = MaybeEntity::new_existing(rule.original.id, &rule.original.name);
        // add the info to associate this flag with its source sigma rule
        action_cache.add_association(rule, AssociationKind::CreatedBy, original, flag_index);
        // get this entities stats
        let entity_stats = &mut stats.by_entity[flagged_on.kind()];
        // track that we are creating a flag for this entity kind
        entity_stats.flagged = entity_stats.flagged.saturating_add(1);
        Ok(index)
    }

    /// Scan a single entity
    ///
    /// # Arguments
    ///
    /// * `event` - The event to scan
    #[instrument(name = "SigmaRuleWorker::scan_event", skip_all)]
    pub async fn scan_event(&self, event: Event) -> Result<SigmaRuleStats, Error> {
        // get a pin to our papaya map
        let pin = self.clients.pin_owned();
        // get this users thorium client
        let thorium = match pin.get(&event.user) {
            Some(thorium) => thorium,
            None => {
                // we don't have a client for this user so just error for now
                // well retry this event later
                return Err(Error::new(format!(
                    "Missing client for user {}",
                    event.user
                )));
            }
        };
        // start with an empty stats block then we will lock and add everything all at once later
        let mut local_stats = SigmaRuleStats::default();
        // get our innner event
        // this should never hit the wrong event kind branch but its better to be defensive
        let event_data = match event.data {
            EventData::SigmaScannableResults(inner) => inner,
            _ => {
                // we got an event that is not meant for this worker somehow
                // build a helpful error message
                let msg = format!("SigmaRuleWorker got the wrong kind of event: {}", event.id);
                // return an error since we can't handle this event
                return Err(Error::new(msg));
            }
        };
        // A cache of actions to take
        let mut action_cache = SigmaActionCache::new(event_data.key.clone());
        // increment the total number of scanned events
        local_stats.total.scanned = local_stats.total.scanned.saturating_add(1);
        // Get all of the entities for this event
        let entity_map = EntityMap::new(&event_data, thorium, &mut local_stats).await?;
        // scan using any relevant rules
        for applies_to in &event_data.applies_to {
            // get the entity kind for this applies to
            let entity_kind = EntityKinds::from(applies_to);
            // scan using these rules
            for index in &self.context.applies_to[entity_kind] {
                // get this rule
                let rule = match self.context.combined.get(*index) {
                    Some(rule) => rule,
                    // we are somehow missing a rule
                    None => {
                        // log this missing rule
                        event!(
                            Level::ERROR,
                            "Missing rule for {entity_kind} at index {index}"
                        );
                        // just continue on to the next rule
                        continue;
                    }
                };
                // get a reference to this entity kinds events
                let events = &entity_map.events[entity_kind];
                // scan the entities this rule applies too
                for (index, entity_req) in entity_map.get(entity_kind).iter().enumerate() {
                    // get this entities event
                    let sigma_event = &events[index];
                    // scan this event
                    if rule.compiled.is_match(sigma_event) {
                        // get this rules metadata
                        let rule_meta = rule.get_original_rule()?;
                        // get all of the actions to take when this rule hits
                        for action in &rule_meta.actions {
                            // perform the correct action
                            match action {
                                // Create a flag on the data we hit on
                                SigmaActionToTake::Flag(auto_flag) => {
                                    self.create_flag(
                                        thorium,
                                        &mut action_cache,
                                        rule,
                                        auto_flag,
                                        entity_req,
                                        &entity_map,
                                        index,
                                        &mut local_stats,
                                    )
                                    .await?;
                                }
                            }
                        }
                    }
                }
            }
        }
        // create any entities that any rules hit/flagged on
        action_cache.create_all(thorium).await?;
        Ok(local_stats)
    }
}

impl EventWorkerSupportCore for SigmaRuleWorker {
    /// The stats for all workers of this kind
    type Stats = SigmaRuleStats;

    /// Build this specific kind of worker
    ///
    /// # Arguments
    ///
    /// * `cache` - A cache of all info a worker may need to be created
    /// * `thorium` - A Thorium client
    async fn new(cache: &EventWorkerCache, thorium: &Arc<Thorium>) -> Self {
        // build this worker
        let mut worker = SigmaRuleWorker {
            context: cache.sigma.clone(),
            thorium: thorium.clone(),
            clients: PapayaMap::with_capacity(100),
        };
        // populate our user clients
        worker.create_user_clients(cache.users.clone());
        worker
    }

    /// The type of events this worker can handle
    #[inline]
    fn event_kind() -> EventType {
        EventType::SigmaScannableResults
    }

    /// The types of cache updates this worker subscribes too
    #[inline]
    fn cache_subscriptions() -> Vec<WorkerCacheUpdateKinds> {
        vec![
            WorkerCacheUpdateKinds::SigmaRulesContext,
            WorkerCacheUpdateKinds::Users,
        ]
    }

    /// Apply an update to our local worker cache
    ///
    /// # Arguments
    ///
    /// * `update` - The update to apply to this workers cache
    fn apply_cache_update(&mut self, update: WorkerCacheUpdates) {
        // Apply an updates
        match update {
            WorkerCacheUpdates::ReactionTriggers(_) => (),
            WorkerCacheUpdates::SigmaRulesContext(context) => self.context = context,
            WorkerCacheUpdates::Users(users) => self.create_user_clients(users),
        }
    }

    /// Report our latest stats in an event
    ///
    /// This will only report new/changed stats.
    ///
    /// # Arguments
    ///
    /// * `stats` - The new stats to log
    /// * `prior` - The previously reported stats
    #[instrument(name = "EventWorkerSupportCore::stats", skip_all)]
    fn stats(stats: &Self::Stats, prior: &Self::Stats) {
        // first report our total stats
        event!(
            Level::INFO,
            worker = "SigmaRuleworker",
            total = true,
            scanned = stats.total.scanned,
            created = stats.total.created,
            flagged = stats.total.flagged,
        );
        // now report stats on a per entity basis
        for (entity, entity_stats) in &stats.by_entity {
            // We have prior stats check if they changed
            if entity_stats == &prior.by_entity[entity] {
                // these stats didn't change so no reason to reemit the same numbers
                continue;
            }
            // report this entity kinds stats
            event!(
                Level::INFO,
                worker = "SigmaRuleworker",
                total = false,
                entity = entity.as_str(),
                scanned = entity_stats.scanned,
                created = entity_stats.created,
                flagged = entity_stats.flagged,
            );
        }
    }
}

impl EventWorkerSupport for SigmaRuleWorker {
    /// The method to call to handle or process a single event
    async fn process(&self, event: Event, stats: &Arc<Mutex<SigmaRuleStats>>) -> Result<(), Error> {
        // scan this event
        let local_stats = self.scan_event(event).await?;
        // get a lock to our shared stats so we can update them
        let mut lock = stats.lock().await;
        // add our total
        lock.total += local_stats.total;
        // add each entity kind
        for (kind, stats) in local_stats.by_entity {
            // get an entry to this entity kinds stats
            lock.by_entity[kind] += stats;
        }
        Ok(())
    }
}
