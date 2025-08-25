//! Scylla structs/logic for graphics

use scylla::DeserializeRow;
use uuid::Uuid;

#[derive(Debug, Deserialize, DeserializeRow)]
#[scylla(flavor = "enforce_order", skip_name_checks)]
pub struct GraphicInfoRow {
    /// The graphic's id in S3
    pub id: Uuid,
    /// The graphic's MIME type
    pub mime_type: String,
    /// The name of the graphic's image file
    pub name: Option<String>,
}
