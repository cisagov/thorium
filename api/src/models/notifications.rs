//! Notifications providing pertinent information to users regarding various entities in Thorium
use chrono::prelude::*;
use std::marker::PhantomData;
use strum::{AsRefStr, EnumString};
use uuid::Uuid;

#[cfg(feature = "scylla-utils")]
use std::str::FromStr;

use super::backends::NotificationSupport;
use super::bans::Ban;
use crate::same;

/// The different types of notifications
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, EnumString, AsRefStr)]
#[cfg_attr(
    feature = "rkyv-support",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[cfg_attr(
    feature = "rkyv-support",
    archive_attr(derive(Debug, bytecheck::CheckBytes))
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
pub enum NotificationType {
    /// This operation is working on image notifications
    #[strum(serialize = "Images")]
    Images,
    /// This operation is working on pipeline notifications
    #[strum(serialize = "Pipelines")]
    Pipelines,
}

impl NotificationType {
    /// Cast this notification type to a str
    pub fn as_str(&self) -> &str {
        match self {
            Self::Images => "Images",
            Self::Pipelines => "Pipelines",
        }
    }
}

/// The level of severity/importance for a given notification
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, EnumString, AsRefStr, clap::ValueEnum,
)]
#[cfg_attr(feature = "scylla-utils", derive(thorium_derive::ScyllaStoreAsStr))]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum NotificationLevel {
    /// A notification providing some info regarding the entity
    #[default]
    #[strum(serialize = "Info")]
    Info,
    /// A notification warning of a possible misconfiguration or problem
    #[strum(serialize = "Warn")]
    Warn,
    /// A notification reporting a critical error with the entity
    #[strum(serialize = "Error")]
    Error,
}

impl NotificationLevel {
    /// Cast this notification level to a str
    pub fn as_str(&self) -> &str {
        match self {
            Self::Info => "Info",
            Self::Warn => "Warn",
            Self::Error => "Error",
        }
    }
}

/// A log message related to an image about whether/why it's been
/// banned and other information that could be pertinent to a user
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Notification<N: NotificationSupport> {
    /// The key to the notification's related entity in scylla
    pub key: N::Key,
    /// The time this notification was created
    pub created: DateTime<Utc>,
    /// The notification's unique ID
    pub id: Uuid,
    /// The notification's message
    pub msg: String,
    /// The notification's level
    pub level: NotificationLevel,
    /// The id of a ban this notification is referencing if there is one
    pub ban_id: Option<Uuid>,
}

impl<N: NotificationSupport> Notification<N> {
    /// Create a new `Notification`
    ///
    /// # Arguments
    ///
    /// * `key` - The key to the related entity in scylla
    /// * `msg` - The notification's message
    /// * `level` - The notification's level
    #[must_use]
    pub fn new<T: Into<String>>(key: N::Key, msg: T, level: NotificationLevel) -> Self {
        Self {
            key,
            created: Utc::now(),
            id: Uuid::new_v4(),
            msg: msg.into(),
            level,
            ban_id: None,
        }
    }

    /// Create a new `Notification` from a [`Ban`]
    ///
    /// # Arguments
    ///
    /// * `ban` - The ban applied to the entity
    /// * `key` - The key to the entity to which the notification refers
    #[must_use]
    pub fn new_ban<B: Ban<N>>(ban: &B, key: N::Key) -> Self {
        // build a notification from the ban and underlying entity
        Self {
            key,
            created: Utc::now(),
            id: Uuid::new_v4(),
            msg: ban.msg(),
            level: NotificationLevel::Error,
            ban_id: Some(*ban.id()),
        }
    }
}

impl<N: NotificationSupport> PartialEq<NotificationRequest<N>> for Notification<N> {
    /// Check that all the values is the `NotificationRequest` were set correctly in the
    /// `Notification`
    fn eq(&self, req: &NotificationRequest<N>) -> bool {
        same!(self.msg, req.msg);
        same!(self.level, req.level);
        true
    }
}

/// The parameters for a notification create request
#[derive(Serialize, Deserialize, Debug, Default)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NotificationParams {
    /// Whether or not the notification should automatically expire
    ///
    /// Notifications not at the `Error` level will automatically expire
    /// by default
    #[serde(default)]
    pub expire: Option<bool>,
}

impl NotificationParams {
    /// Set whether or not the notification will automatically expire
    /// in a builder-like pattern
    ///
    /// # Arguments
    ///
    /// * `value` - Whether or not the notification will expire
    #[must_use]
    pub fn expire(mut self, value: bool) -> Self {
        self.expire = Some(value);
        self
    }
}

/// A request to create a notification for an entity in Thorium
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "api", schema(example = json!({"msg": "Banned due to invalid volume mount", "level": "Error"})))]
pub struct NotificationRequest<N: NotificationSupport> {
    /// The message this notification will contain
    pub msg: String,
    /// The notification's level
    pub level: NotificationLevel,
    /// The type this notification is for
    #[serde(default)]
    // utoipa seems to want a valid type even if you ignore things
    #[cfg_attr(feature = "api", schema(ignore, value_type = bool))]
    phantom: PhantomData<N>,
}

//impl<N: NotificationSupport> utoipa::ToSchema for NotificationRequest<N> {
//    /// Return the name of the noficiation request
//    fn name() -> std::borrow::Cow<'static, str> {
//        std::borrow::Cow::Borrowed("NotificationRequest")
//    }
//}
//
//impl<N: NotificationSupport> utoipa::PartialSchema for NotificationRequest<N> {
//    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
//        utoipa::openapi::ObjectBuilder::new()
//            .property(
//                "msg",
//                utoipa::openapi::ObjectBuilder::new()
//                    .schema_type(utoipa::openapi::schema::Type::String),
//            )
//            .property(
//                "level",
//                utoipa::openapi::ObjectBuilder::new()
//                    .schema_type(utoipa::openapi::schema::Type::String),
//            )
//            .into()
//    }
//}

impl<N: NotificationSupport> NotificationRequest<N> {
    /// Create a notification request
    ///
    /// # Arguments
    ///
    /// * `msg` - The message the notification should contain
    /// * `level` - The level of severity/importance of the notification
    pub fn new<T: Into<String>>(msg: T, level: NotificationLevel) -> Self {
        Self {
            msg: msg.into(),
            level,
            phantom: PhantomData,
        }
    }
}
