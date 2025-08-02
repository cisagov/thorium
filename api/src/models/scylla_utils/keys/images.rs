//! The Scylla key for images

use crate::models::Image;

/// The components forming a unique key to access an image's data in Scylla;
/// these components may make up only part of a partition key
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreJson))]
pub struct ImageKey {
    /// The group the image is in
    pub group: String,
    /// The name of the image
    pub image: String,
}

impl AsRef<ImageKey> for ImageKey {
    fn as_ref(&self) -> &ImageKey {
        self
    }
}

impl ImageKey {
    /// Create a new `ImageKey`
    ///
    /// # Arguments
    ///
    /// * `group` - The group the image is in
    /// * `image` - The name of the image
    pub fn new<S, T>(group: S, image: T) -> Self
    where
        S: Into<String>,
        T: Into<String>,
    {
        Self {
            group: group.into(),
            image: image.into(),
        }
    }
}

/// Produce an image key from an image
impl From<&Image> for ImageKey {
    fn from(image: &Image) -> Self {
        ImageKey {
            group: image.group.clone(),
            image: image.name.clone(),
        }
    }
}
