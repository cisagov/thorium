//! The utilities for integration tests involving the API
use async_once::AsyncOnce;
use bb8_redis::{bb8::Pool, RedisConnectionManager};
use once_cell::sync::Lazy;
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use std::time::Duration;

use crate::{client::ClientSettings, Conf, Error, Thorium};

// The config to use for these tests
static CONF: Lazy<Conf> =
    Lazy::new(|| Conf::new("../api/tests/thorium.yml").expect("Failed to load config"));

/// Get a Thorium config
pub fn config() -> Conf {
    CONF.clone()
}

/// Get a Thorium config
pub fn config_ref<'a>() -> &'a Conf {
    &CONF
}

// The addr to talk to the api at
static ADDR: Lazy<String> =
    Lazy::new(|| format!("http://{}:{}", CONF.thorium.interface, CONF.thorium.port));

/// Build a scylla client for a specific cluster
///
/// # Arguments
///
/// * `config` - The config for this Thorium cluster
async fn get_scylla_client(config: &Conf) -> Result<Session, Error> {
    // start building our scylla client
    let mut session = SessionBuilder::new();
    // if we have auth info for scylla then add that
    if let Some(creds) = &config.scylla.auth {
        // inject our creds
        session = session.user(&creds.username, &creds.password);
    }
    // set our request timeout
    let session = session.connection_timeout(Duration::from_secs(config.scylla.setup_time as u64));
    // build a scylla session
    let scylla = config
        .scylla
        .nodes
        .iter()
        .fold(session, |builder, node| builder.known_node(node))
        .build()
        .await
        .expect("Failed to connect to scylla!");
    Ok(scylla)
}

/// Set up a connection pool to the redis backend
///
/// # Arguments
///
/// * `config` - The config for the Thorium API
pub async fn get_redis_client(config: &Conf) -> Pool<RedisConnectionManager> {
    // get redis config
    let redis = &config.redis;
    // build url to server using authentication if its configured
    let url = match (&redis.username, &redis.password) {
        // redis with username/password auth setup
        (Some(user), Some(password)) => format!(
            "redis://{}:{}@{}:{}/",
            user, password, redis.host, redis.port
        ),
        (None, Some(password)) => format!(
            "redis://default:{}@{}:{}/",
            password, redis.host, redis.port
        ),
        (None, None) => format!("redis://{}:{}/", redis.host, redis.port),
        _ => panic!("Redis Setup Error - Password must be set if username is set"),
    };
    // build manager
    let manager = match RedisConnectionManager::new(url) {
        Ok(manager) => manager,
        Err(e) => panic!("{}", e),
    };
    // build redis connection pool
    let pool = Pool::builder()
        .max_size(redis.pool_size.unwrap_or(50))
        .build(manager)
        .await
        .expect("Failed to build redis connection pool");
    pool
}

/// Wipe all of the databases under test (Scylla + Redis)
///
/// We don't currently test against ES
///
/// # Arguments
///
/// * `conf` - The Thorium config we are using for tests
async fn wipe_dbs(conf: &Conf) {
    // connect to scylla
    let scylla = get_scylla_client(conf)
        .await
        .expect("Failed to get scylla client");
    // panic if our keyspace is Thorium
    if &conf.thorium.namespace == "Thorium" {
        panic!("You cannot test against the Thorium namespace! Change your namespace to testing_thorium!");
    }
    // drop our current keyspace
    scylla
        .query_unpaged(
            format!("DROP KEYSPACE IF EXISTS {}", conf.thorium.namespace),
            &(),
        )
        .await
        .expect("Failed to drop keyspace in scylla");
    // connect to redis
    let redis = get_redis_client(conf).await;
    // build our redis pipe
    let mut pipe = redis::pipe();
    // flush all data in this redis instance
    let _: () = pipe
        .cmd("FLUSHDB")
        .query_async(&mut *redis.get().await.unwrap())
        .await
        .unwrap();
}

lazy_static::lazy_static! {
    // start the API and get a token
    static ref ADMIN_TOKEN: AsyncOnce<String> = AsyncOnce::new(async {
        // wipe our databases
        wipe_dbs(&CONF).await;
        // spawn the api
        std::thread::spawn(move || {
            // create a tokio runtime
            let rt = tokio::runtime::Runtime::new().expect("Failed to spawn tokio runtime");
            // spawn our api
            rt.block_on(async move { crate::axum(CONF.clone()).await });
        });
        // build the addr to connect to the api at
        let addr = format!("http://{}:{}", CONF.thorium.interface, CONF.thorium.port);
        // try to bootstrap for 60 seconds until it works
        let mut attempts = 0;
        // start trying to bootstrap
        let resp = loop {
            // start with default client settings
            let settings = ClientSettings::default();
            // boot strap an admin Thorium client
            let attempt = Thorium::bootstrap(
                &addr,
                "thorium",
                "fake@fake.gov",
                "password",
                &CONF.thorium.secret_key,
                &settings
            ).await;
            // check if our bootstrap attempt worked
            match attempt {
                Ok(resp) => break resp,
                // this attempt failed so sleep for 1 second
                Err(_) => {
                    // increment our attempts by 1
                    attempts += 1;
                    // check if we have used up all of our attempts yet
                    if attempts == 300 {
                        panic!("Failed to bootstrap Thorium");
                    }
                    // sleep for 1 second
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        };
        // build our admin client
        let client = Thorium::build(ADDR.as_str()).token(&resp.token).build().await
            .expect("Failed to build admin client");
        // make sure Thorium is initialized
        client.system.init().await.expect("Failed to initialize Thorium");
        resp.token
    });
}

pub async fn admin_client() -> Result<Thorium, Error> {
    // get the token for the Thorium admin
    let token = ADMIN_TOKEN.get().await;
    // build our admin client
    Thorium::build(ADDR.as_str()).token(token).build().await
}
