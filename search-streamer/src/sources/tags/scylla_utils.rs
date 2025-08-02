use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use scylla::DeserializeRow;
use thorium::{models::TagType, Error};

/// Prepared statements to get data for tags
#[derive(Clone)]
pub struct TagsPrepared {
    /// The statement to enumerate the items we need to get tags for
    pub enumerate: PreparedStatement,
    /// The statement to get info for init jobs for tags
    pub init: PreparedStatement,
    /// The statement to get info for event jobs for tags
    pub event: PreparedStatement,
}

impl TagsPrepared {
    /// Returns prepared statements for tags
    ///
    /// # Arguments
    ///
    /// * `scylla` - The scylla client
    /// * `ns` - The namespace the data is stored in
    pub async fn prepare(scylla: &Session, ns: &str) -> Result<Self, Error> {
        let enumerate = scylla
            .prepare(format!(
                "SELECT DISTINCT type, item FROM {ns}.tags_by_item \
                    WHERE token(type, item) >= ? \
                    AND token(type, item) <= ?"
            ))
            .await
            .map_err(|err| {
                Error::new(format!("Failed to create tags enumerate statement: {err}"))
            })?;
        let init = scylla
            .prepare(format!(
                "SELECT group, item, key, value FROM {ns}.tags_by_item \
                WHERE type = ? \
                AND item in ?"
            ))
            .await
            .map_err(|err| Error::new(format!("Failed to create tags init statement: {err}")))?;
        let event = scylla
            .prepare(format!(
                "SELECT group, key, value FROM {ns}.tags_by_item \
                    WHERE type = ? \
                    AND item = ? \
                    AND group in ?"
            ))
            .await
            .map_err(|err| Error::new(format!("Failed to create tags event statement: {err}")))?;
        Ok(Self {
            enumerate,
            init,
            event,
        })
    }
}

/// A row of info needed to enumerate the items to pull tags for
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct TagEnumerateRow {
    /// The kind of item it is
    pub kind: TagType,
    /// The item key itself
    pub item: String,
}

/// A row of info for streaming events
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct TagEventRow {
    /// The group the event pertains to
    pub group: String,
    /// The tag key
    pub key: String,
    /// The tag value
    pub value: String,
}
