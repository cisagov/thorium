use std::collections::HashMap;

use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use scylla::DeserializeRow;
use thorium::{models::OutputKind, Error};
use uuid::Uuid;

/// The prepared statements for results in scylla
#[derive(Clone)]
pub struct ResultsPrepared {
    /// The statement to enumerate the items we need to get results for
    pub enumerate: PreparedStatement,
    /// The statement to get result data in the init phase
    pub init_data: PreparedStatement,
    /// The statement to get the results themselves
    pub results: PreparedStatement,
    /// The statement to get result data in the event phase
    pub event_data: PreparedStatement,
}

impl ResultsPrepared {
    /// Create prepared statements for results
    ///
    /// # Arguments
    ///
    /// * `scylla` - The scylla client
    /// * `ns` - The namespace the data is stored in
    pub async fn prepare(scylla: &Session, ns: &str) -> Result<Self, Error> {
        let enumerate = scylla
            .prepare(format!(
                "SELECT DISTINCT key FROM {ns}.results_ids \
                    WHERE token(key) >= ? \
                    AND token(key) <= ?"
            ))
            .await
            .map_err(|err| Error::new(format!("Failed to create results init statement: {err}")))?;
        let init_data = scylla
            .prepare(format!(
                "SELECT key, kind, group, id FROM {ns}.results_ids \
                    WHERE key in ?"
            ))
            .await
            .map_err(|err| Error::new(format!("Failed to create results statement: {err}")))?;
        let results = scylla
            .prepare(format!(
                "SELECT id, result, files, children FROM {ns}.results \
                    WHERE id in ?"
            ))
            .await
            .map_err(|err| Error::new(format!("Failed to create results statement: {err}")))?;
        let event_data = scylla
            .prepare(format!(
                "SELECT group, id FROM {ns}.results_auth \
                    WHERE key = ? \
                    AND kind = ? \
                    AND group in ?"
            ))
            .await
            .map_err(|err| Error::new(format!("Failed to create results statement: {err}")))?;
        Ok(Self {
            enumerate,
            init_data,
            results,
            event_data,
        })
    }
}

/// A row of info needed to enumerate data for results
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct ResultEnumerateRow {
    pub key: String,
}

/// A row containing the actual result with its id
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct ResultRow {
    /// The result's id
    pub id: Uuid,
    /// The result
    pub result: String,
    /// The files associated with this result
    pub files: Vec<String>,
    /// The children found by the tool
    pub children: HashMap<String, Uuid>,
}

/// A row containing result data required in the init phase
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct ResultInitDataRow {
    /// The key the result pertains to
    pub key: String,
    /// The kind of item the result pertains to
    pub kind: OutputKind,
    /// The group the result is in
    pub group: String,
    /// The result's id
    pub id: Uuid,
}

/// A row containing info required in the event phase
#[derive(Debug, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct ResultEventInfoRow {
    /// The group the result is in
    pub group: String,
    /// The result's id
    pub id: Uuid,
}
