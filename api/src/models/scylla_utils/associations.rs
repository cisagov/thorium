//! The scylla utils/structs for associations

use chrono::prelude::*;
use scylla::DeserializeRow;
use std::hash::Hasher;
use thorium_derive::ScyllaStoreJson;
use uuid::Uuid;

use crate::models::{Association, AssociationKind, AssociationTarget, Entity, TreeSupport};

#[derive(Debug, Clone, Serialize, Deserialize, ScyllaStoreJson, Eq, PartialEq, Hash)]
pub enum AssociationTargetColumn {
    /// This assocation is associated with another entity
    Entity(Uuid),
    /// This association is associated with a file
    File(String),
    /// This association is associated with a repo
    Repo(String),
}

impl AssociationTargetColumn {
    /// Generate a tree node hash for this item
    pub fn tree_hash(&self) -> u64 {
        // build a hasher
        let mut hasher = gxhash::GxHasher::with_seed(1234);
        // hash this specific items info
        match self {
            AssociationTargetColumn::Entity(id) => hasher.write_u128(id.as_u128()),
            AssociationTargetColumn::File(sha256) => hasher.write(sha256.as_bytes()),
            AssociationTargetColumn::Repo(url) => hasher.write(url.as_bytes()),
        }
        // finalize our hasher
        let hash = hasher.finish();
        hash
    }
}

impl AssociationTargetColumn {
    #[cfg(feature = "api")]
    /// Combine this column with any extra info to build an association target
    pub fn to_target(
        self,
        extra: Option<String>,
    ) -> Result<AssociationTarget, crate::utils::ApiError> {
        // convert this column
        match self {
            AssociationTargetColumn::Entity(id) => match extra {
                Some(name) => Ok(AssociationTarget::Entity { id, name }),
                None => crate::internal_err!("entity extra is empty".to_owned()),
            },
            AssociationTargetColumn::File(sha256) => Ok(AssociationTarget::File(sha256)),
            AssociationTargetColumn::Repo(url) => Ok(AssociationTarget::Repo(url)),
        }
    }
}

#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct AssociationListRow {
    /// The group this association is for
    pub group: String,
    /// The kind of association this is
    pub kind: AssociationKind,
    /// The source for this association
    pub source: String,
    /// The other serialized data this association is with
    pub other: String,
    /// Who created this association
    pub submitter: String,
    /// When this association was created
    pub created: DateTime<Utc>,
    /// Whether this direction is to our source object or away from it
    pub to_source: bool,
    /// Any extra info needed for the source column in this row
    pub extra_source: Option<String>,
    /// Any extra info needed for the target column in this row
    pub extra_other: Option<String>,
}

/// An association with a specific piece of data
#[derive(Debug, Serialize, Deserialize)]
pub struct ListableAssociation {
    /// The kind of association this is
    pub kind: AssociationKind,
    /// The other serialized data this association is with
    pub other: String,
    /// The creator of this association
    pub submitter: String,
    /// The groups for this association
    pub groups: Vec<String>,
    // When this association was created
    pub created: DateTime<Utc>,
    /// Whether this direction is to our source object or away from it
    pub to_source: bool,
    /// Any extra info needed for the target column in this row
    pub extra_other: Option<String>,
}

impl ListableAssociation {
    /// Add a group to this association
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add
    pub fn add_group(&mut self, group: String) {
        self.groups.push(group)
    }

    /// Get the source and target for an assocation
    #[cfg(feature = "api")]
    pub fn deserialize_other(&self) -> Result<AssociationTargetColumn, crate::utils::ApiError> {
        // deserialize our target column
        let other = crate::deserialize!(&self.other);
        Ok(other)
    }

    /// Get the source and target for an assocation
    #[cfg(feature = "api")]
    pub fn get_source(
        &self,
        possible: &AssociationTargetColumn,
    ) -> Result<AssociationTargetColumn, crate::utils::ApiError> {
        // if our to_source flag is set then other is the source for this association
        if self.to_source {
            // deserialize our target column
            let other = crate::deserialize!(&self.other);
            Ok(other)
        } else {
            Ok(possible.to_owned())
        }
    }
}

impl From<AssociationListRow> for ListableAssociation {
    /// Convert an association list row into a listable assocation
    ///
    /// # Arguments
    ///
    /// * `value` - The list row to convert
    fn from(row: AssociationListRow) -> Self {
        // TODO handle inversion?
        ListableAssociation {
            kind: row.kind,
            other: row.other,
            submitter: row.submitter,
            groups: vec![row.group],
            created: row.created,
            to_source: row.to_source,
            extra_other: row.extra_other,
        }
    }
}

#[cfg(feature = "api")]
impl TryFrom<ListableAssociation> for Association {
    type Error = crate::utils::ApiError;
    /// Build a user facing association from a listable one
    ///
    /// # Arguments
    ///
    /// * `row` - The association row to convert
    fn try_from(row: ListableAssociation) -> Result<Self, Self::Error> {
        // deserialize our listable assocation
        let other_column: AssociationTargetColumn = crate::deserialize!(&row.other);
        // convert the column into a full target
        let other = other_column.to_target(row.extra_other)?;
        // invert our source/target if inverted is set
        let association = Association {
            kind: row.kind,
            other,
            submitter: row.submitter,
            groups: row.groups,
            created: row.created,
            to_source: row.to_source,
        };
        Ok(association)
    }
}
