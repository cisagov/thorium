//! Tests the basic routes in Thorium

use thorium::models::system::DEFAULT_IFF;
use thorium::{Error, is, test_utilities};

#[tokio::test]
async fn identify() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // send the identify query
    let resp = client.basic.identify().await?;
    // make sure we get the right string back
    is!(resp, DEFAULT_IFF);
    Ok(())
}

#[tokio::test]
async fn health() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // send the identify query
    let health = client.basic.health().await?;
    // make sure Thorium is healthy
    is!(health, true);
    Ok(())
}

// Sync tests

#[cfg(all(feature = "sync", not(feature = "python")))]
#[test]
fn health_blocking() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client_blocking()?;
    // send the identify query
    let health = client.basic.health()?;
    // make sure Thorium is healthy
    is!(health, true);
    Ok(())
}
