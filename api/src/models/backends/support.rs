//! Traits allowing for backends to advertize support for various functionality in Thorium

mod notifications;
mod outputs;
mod tags;

#[cfg(feature = "api")]
mod graphics;

pub use notifications::NotificationSupport;
pub use outputs::OutputSupport;
pub use tags::TagSupport;

#[cfg(feature = "api")]
pub(crate) use graphics::GraphicSupport;
