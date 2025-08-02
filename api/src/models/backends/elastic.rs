//! Backend-related logic for elastic

use crate::models::{ElasticIndex, TagType};

impl From<&TagType> for ElasticIndex {
    fn from(tag_type: &TagType) -> Self {
        match tag_type {
            TagType::Files => Self::SampleTags,
            TagType::Repos => Self::RepoTags,
        }
    }
}
