use elasticsearch::OpenPointInTimeParts;
use serde::{Deserialize, Serialize};

use crate::models::ElasticDoc;
use crate::utils::{ApiError, Shared};

/// A point in time id for an elastic query
#[derive(Deserialize, Serialize, Debug)]
pub struct PointInTime {
    pub id: String,
}

impl PointInTime {
    /// get a new point in time value
    ///
    /// # Arguments
    ///
    /// * `indexes` - The indexes we are creating a point in time for
    /// * `shared` - Shared Thorium objects
    pub async fn new(indexes: &[&str], shared: &Shared) -> Result<Self, ApiError> {
        //  create a point in time api for our query
        let resp = shared
            .elastic
            .open_point_in_time(OpenPointInTimeParts::Index(indexes))
            .keep_alive("1d")
            .send()
            .await?;
        // cast our response to a pit value
        let pit = resp.json::<Self>().await?;
        Ok(pit)
    }
}

/// The documents that hit on a particular Elastic Query
#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Hits {
    /// The max score of a single document that was returned
    pub max_score: Option<f64>,
    /// The documents that hit
    pub hits: Vec<ElasticDoc>,
}

// A response from Elastic
#[derive(Deserialize, Serialize, Default, Debug)]
pub struct ElasticResponse {
    /// The point in time value used for this search
    pub pit_id: String,
    /// How long this query took in ms
    pub took: u64,
    /// Whether this request timed out or not
    pub timed_out: bool,
    /// The results for this search query
    pub hits: Hits,
}
