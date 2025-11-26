//! The utlities for integration tests involving the API
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_sdk_s3::{
    error::DisplayErrorContext,
    types::{BucketLifecycleConfiguration, ExpirationStatus, LifecycleExpiration, LifecycleRule},
};
use aws_types::region::Region;
use bb8_redis::{RedisConnectionManager, bb8::Pool};
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use std::{sync::LazyLock, time::Duration};
use tokio::sync::OnceCell;
use tokio_util::time::FutureExt;

use crate::{Conf, Error, Thorium, client::ClientSettings, conf::S3};

/// The path to a valid Thorium config in the testing directory
///
/// This config is a placeholder, but must successfully pass validation.
/// Its values can be overridden using environment variables.
static CONF_PATH: &str = "../api/tests/thorium-testing.yml";

/// The config to use for integration tests
///
/// The config to read is hard-coded to the config in the tests directory, but that is just
/// a template. Use environment variables (e.g. `THORIUM__THORIUM__S3__ENDPOINT`) or
/// update the template manually to configure tests.
pub static CONF: LazyLock<Conf> = LazyLock::new(|| {
    Conf::new(CONF_PATH).unwrap_or_else(|_| panic!("Failed to load config at '{CONF_PATH}'"))
});

/// The addr to talk to the api at
static ADDR: LazyLock<String> =
    LazyLock::new(|| format!("http://{}:{}", CONF.thorium.interface, CONF.thorium.port));

/// Build a scylla client for a specific cluster
///
/// # Arguments
///
/// * `config` - The config for this Thorium cluster
#[allow(clippy::redundant_closure_for_method_calls)]
async fn get_scylla_client(config: &Conf) -> Result<Session, Error> {
    // start building our scylla client
    let mut session = SessionBuilder::new();
    // if we have auth info for scylla then add that
    if let Some(creds) = &config.scylla.auth {
        // inject our creds
        session = session.user(&creds.username, &creds.password);
    }
    // set our request timeout
    let session = session.connection_timeout(Duration::from_secs(config.scylla.setup_time.into()));
    // build a scylla session
    let scylla = Box::pin(
        config
            .scylla
            .nodes
            .iter()
            .fold(session, |builder, node| builder.known_node(node))
            .build(),
    )
    .await
    .expect("Failed to connect to scylla!");
    Ok(scylla)
}

/// Setup a connection pool to the redis backend
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
    Pool::builder()
        .max_size(redis.pool_size.unwrap_or(50))
        .build(manager)
        .await
        .expect("Failed to build redis connection pool")
}

/// Drop the keyspace in Scylla
async fn wipe_scylla(config: &Conf) -> Result<(), Error> {
    // connect to scylla
    let scylla = get_scylla_client(config)
        .await
        .map_err(|err| Error::new(format!("Failed to get scylla client: {err}")))?;
    // drop our current keyspace; retry once on failure
    let drop_keyspace = || async {
        scylla
            .query_unpaged(
                format!("DROP KEYSPACE IF EXISTS {}", config.thorium.namespace),
                &(),
            )
            .timeout(tokio::time::Duration::from_secs(120))
            .await
            .map_err(|err| {
                Error::new(format!(
                    "Error connecting to scylla while dropping keyspace: {err}"
                ))
            })?
            .map_err(|err| Error::new(format!("Error dropping scylla keyspace: {err}")))
    };
    // first attempt
    if let Err(e) = drop_keyspace().await {
        // log the error and retry
        eprintln!("Failed to drop keyspace on first attempt: {e}");
        // sleep to let scylla get settled
        tokio::time::sleep(Duration::from_secs(2)).await;
        // second attempt
        drop_keyspace().await?;
    }
    Ok(())
}

/// Wipe Redis before running tests
async fn wipe_redis(config: &Conf) -> Result<(), Error> {
    // connect to redis
    let redis = get_redis_client(config).await;
    // build our redis pipe
    let mut pipe = redis::pipe();
    // flush all data in this redis instance
    pipe.cmd("FLUSHDB")
        .exec_async(&mut *redis.get().await.map_err(|err| {
            Error::new(format!("Error getting Redis connection from pool: {err}"))
        })?)
        .await
        .map_err(|err| Error::new(format!("Error flushing data from Redis: {err}")))?;
    Ok(())
}

/// Wipe all of the databases under test (Scylla + Redis)
///
/// We don't currently test against ES
///
/// # Arguments
///
/// * `conf` - The Thorium config we are using for tests
async fn wipe_dbs(config: &Conf) -> Result<(), Error> {
    // panic if our keyspace is Thorium
    assert!(
        config.thorium.namespace.to_lowercase() != "thorium",
        "You cannot test against the Thorium namespace! Change your namespace to testing_thorium!"
    );
    tokio::try_join!(wipe_scylla(config), wipe_redis(config))?;
    Ok(())
}

/// Create a single bucket for integration tests with the given lifecycle and name
///
/// # Arguments
///
/// * `s3_client` - The S3 client to create the bucket
/// * `lifecycle` - The lifecycle to set for this bucket
/// * `bucket` - The name of the bucket to create
async fn init_bucket(
    s3_client: &aws_sdk_s3::Client,
    lifecycle: BucketLifecycleConfiguration,
    bucket: &str,
) -> Result<(), Error> {
    // attempt to create the bucket
    if let Err(err) = s3_client.create_bucket().bucket(bucket).send().await {
        match err.as_service_error() {
            Some(service_err) => {
                // check if it's just a 409 error we can ignore
                if !service_err.is_bucket_already_owned_by_you() {
                    return Err(Error::new(format!(
                        "Error creating S3 bucket '{}': {}",
                        bucket,
                        DisplayErrorContext(&err)
                    )));
                }
            }
            None => {
                return Err(Error::new(format!(
                    "Error creating S3 bucket '{}': {}",
                    bucket,
                    DisplayErrorContext(&err)
                )));
            }
        }
    }
    // set/update the lifecycle
    s3_client
        .put_bucket_lifecycle_configuration()
        .bucket(bucket)
        .lifecycle_configuration(lifecycle)
        .send()
        .await
        .map_err(|err| {
            Error::new(format!(
                "Error setting lifecycle for bucket '{}': {}",
                bucket,
                DisplayErrorContext(&err)
            ))
        })?;
    Ok(())
}

/// Create required S3 buckets for testing with a strict lifecycle policy that
/// removes objects automatically
///
/// # Arguments
///
/// * `s3_conf` - The S3 configuration from the main Thorium configuration to use
async fn init_s3_buckets(s3_conf: &S3, conf: &Conf) -> Result<(), Error> {
    // get our s3 credentials
    let creds = aws_credential_types::Credentials::new(
        &s3_conf.access_key,
        &s3_conf.secret_token,
        None,
        None,
        "Thorium",
    );
    // build an s3 client
    let mut s3_config_builder = aws_sdk_s3::config::Builder::new()
        .endpoint_url(&s3_conf.endpoint)
        .credentials_provider(SharedCredentialsProvider::new(creds))
        .force_path_style(s3_conf.use_path_style);
    // if we have a region set then add that to our config
    if let Some(region) = &s3_conf.region {
        // set our region
        s3_config_builder = s3_config_builder.region(Region::new(region.clone()));
    }
    // build our s3 config
    let s3_config = s3_config_builder.build();
    // build our s3 client
    let s3_client = aws_sdk_s3::Client::from_conf(s3_config);
    // create a 1 day lifecycle to set for each bucket
    let lifecycle = BucketLifecycleConfiguration::builder()
        .rules(
            LifecycleRule::builder()
                .expiration(LifecycleExpiration::builder().days(1).build())
                .status(ExpirationStatus::Enabled)
                // <Prefix/> is deprecated, but Ceph object gateway expects it so we include it here
                .prefix("")
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    // create all required buckets
    init_bucket(&s3_client, lifecycle.clone(), &conf.thorium.files.bucket).await?;
    init_bucket(&s3_client, lifecycle.clone(), &conf.thorium.repos.bucket).await?;
    init_bucket(&s3_client, lifecycle.clone(), &conf.thorium.results.bucket).await?;
    init_bucket(
        &s3_client,
        lifecycle.clone(),
        &conf.thorium.ephemeral.bucket,
    )
    .await?;
    init_bucket(
        &s3_client,
        lifecycle.clone(),
        &conf.thorium.attachments.bucket,
    )
    .await?;
    Ok(())
}

/// A write-once cell for the admin token, written to when the API is stood up for the first time
static ADMIN_TOKEN: OnceCell<String> = OnceCell::const_new();

/// Bootstrap a test API, running the API on another thread and returning
/// a token for authenticating with the running API
async fn bootstrap_test_api() -> Result<String, Error> {
    // wipe our databases
    wipe_dbs(&CONF).await?;
    // initiate s3 buckets
    init_s3_buckets(&CONF.thorium.s3, &CONF).await?;
    // spawn the api
    std::thread::spawn(move || {
        // create a tokio runtime
        let rt =
            tokio::runtime::Runtime::new().expect("Failed to spawn tokio runtime for the test API");
        // spawn our api
        rt.block_on(async move { crate::axum(CONF.clone()).await });
    });
    // try to bootstrap for 60 seconds until it works
    let mut attempts = 0;
    // start trying to bootstrap
    let resp = loop {
        // start with default client settings
        let settings = ClientSettings::default();
        // boot strap an admin Thorium client
        let attempt = Thorium::bootstrap(
            &ADDR,
            "thorium",
            "fake@fake.gov",
            "password",
            &CONF.thorium.secret_key,
            &settings,
        )
        .await;
        // check if our bootstrap attempt worked
        match attempt {
            Ok(resp) => break resp,
            // this attempt failed so sleep for 1 second
            Err(_) => {
                // increment our attempts by 1
                attempts += 1;
                // check if we have used up all of our attempts yet
                if attempts == 300 {
                    return Err(Error::new("Failed to bootstrap Thorium"));
                }
                // sleep for 1 second
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    };
    // build our admin client
    let client = Thorium::build(ADDR.clone())
        .token(&resp.token)
        .build()
        .await
        .map_err(|err| Error::new(format!("Failed to build admin client: {err}")))?;
    // make sure Thorium is initialized
    client
        .system
        .init()
        .await
        .map_err(|err| Error::new(format!("Failed to init system: {err}")))?;
    Ok(resp.token)
}

/// Get an admin client, bootstrapping the API if needed
pub async fn admin_client() -> Result<Thorium, Error> {
    // start the API if it hasn't been started already and get a token
    let token = ADMIN_TOKEN.get_or_try_init(bootstrap_test_api).await?;
    // build our admin client
    Thorium::build(ADDR.clone())
        .token(token.clone())
        .build()
        .await
}

cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        use crate::ThoriumBlocking;
        use crate::RUNTIME;
    }
}

#[cfg(all(feature = "sync", not(feature = "python")))]
/// Get a blocking admin client, bootstrapping the API if needed
pub fn admin_client_blocking() -> Result<ThoriumBlocking, Error> {
    // start the API if it hasn't been started already and get a token
    let token =
        RUNTIME.block_on(async { ADMIN_TOKEN.get_or_try_init(bootstrap_test_api).await })?;
    // build our admin client
    ThoriumBlocking::build(ADDR.clone())
        .token(token.clone())
        .build_blocking()
}
