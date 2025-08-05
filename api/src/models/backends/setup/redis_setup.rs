//! Setup redis
use bb8_redis::{bb8::Pool, RedisConnectionManager};

use crate::{setup, Conf};

/// Set up a connection pool to the redis backend
///
/// # Arguments
///
/// * `config` - The config for the Thorium API
///
/// # Panics
///
/// This will panic if we fail to connect to redis
pub async fn redis(config: &Conf) -> Pool<RedisConnectionManager> {
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
    setup!(
        config.thorium.tracing.local.level,
        format!(
            "Connecting to redis at {}:{}",
            config.redis.host, config.redis.port
        )
    );
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
    setup!(config.thorium.tracing.local.level, "Connected to redis");
    pool
}
