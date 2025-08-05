//! The utilities for tests involving the scaler

use crate::Scaler;
use std::sync::Arc;
use thorium::models::ImageScaler;
use thorium::{Conf, Error, Thorium};

/// Build a k8s/dry run configured Thorium scaler
pub async fn scaler(conf: Conf, thorium: Thorium) -> Result<Scaler, Error> {
    // wrap our Thorium client in an Arc
    let thorium = Arc::new(thorium);
    // build a scaler
    Scaler::build(
        conf,
        "tests/keys.yml".to_owned(),
        thorium,
        ImageScaler::K8s,
        true,
        &"kubernetes-admin@cluster.local".to_owned()
    )
    .await
}
