//! The scylla utils for entities

use chrono::{DateTime, Utc};
use scylla::DeserializeRow;
use uuid::Uuid;

use crate::models::{EntityKinds, entities::EntityMetadata};

/// A single row of an entity from Scylla
#[derive(Debug, Deserialize, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct EntityRow {
    /// The entity's unique ID
    pub id: Uuid,
    /// The group the entity is available to
    pub group: String,
    /// The entity's kind as a raw string
    pub kind: EntityKinds,
    /// The time this policy was created
    pub created: DateTime<Utc>,
    /// The name of the entity
    pub name: String,
    /// The user who originally submitted the entity
    pub submitter: String,
    // TODO metadata should be able to be directly deserialized using StoreJson stuff
    /// The data specific to the entity's kind
    pub metadata: String,
    /// The entity's description
    pub description: Option<String>,
    /// The path to this entities image if it has one
    pub image: Option<String>,
}

/// A single row from Scylla used for listing entities
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct EntityListRow {
    /// The kind of entity this is
    pub kind: EntityKinds,
    /// The group this entity is visible too
    pub group: String,
    /// The time this entity was created
    pub created: DateTime<Utc>,
    /// The entity's unique ID
    pub id: Uuid,
    /// The name of this entity
    pub name: String,
}

/// A single row from Scylla used to supplement tag rows missing name
/// and kind
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct EntityListSupplementRow {
    /// The entity's unique ID
    pub id: Uuid,
    /// The name of this entity
    pub name: String,
    /// The kind of entity this is
    pub kind: EntityKinds,
}
