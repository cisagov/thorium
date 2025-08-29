//! Setup elastic

use elasticsearch::Elasticsearch;
use elasticsearch::auth::Credentials;
use elasticsearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use url::Url;

use crate::{Conf, setup};
/// Setup a connection pool to the elastic search backend
///
/// # Arguments
///
/// * `config` - The config for the Thorium API
///
/// # Panics
///
/// This will panic if we fail to connect to elasticsearch
// this is async even though its not needed so we can reuse our retry logic
#[allow(clippy::unused_async)]
pub async fn elastic(config: &Conf) -> Elasticsearch {
    // Until https://github.com/elastic/elasticsearch-rs/pull/189 is merged
    // we can only support a single node connection pool
    // TODO: the above issue has been merged; need to support multi-node pools
    setup!(
        config.thorium.tracing.local.level,
        format!("Connecting to Elastic at {}", config.elastic.node)
    );
    // try to cast our node to a url
    let url = Url::parse(&config.elastic.node).expect("Failed to parse Elastic url");
    // build our connection pool
    let pool = SingleNodeConnectionPool::new(url);
    // get our username and password
    let username = config.elastic.username.clone();
    let password = config.elastic.password.clone();
    // get our configured elastic cert validation settings
    let validation = config.elastic.to_cert_validation().await;
    // build our transport object for elastic
    let transport = TransportBuilder::new(pool)
        .auth(Credentials::Basic(username, password))
        .cert_validation(validation)
        .build()
        .expect("Failed to setup transport object to Elastic");
    Elasticsearch::new(transport)
}
