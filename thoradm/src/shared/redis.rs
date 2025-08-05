//! The shared redis utilities for Thoradm
use bb8_redis::{bb8::Pool, RedisConnectionManager};
use thorium::Conf;

use crate::Error;

/// Set up a connection pool to the redis backend
///
/// # Arguments
///
/// * `config` - The config for the Thorium API
pub async fn get_client(config: &Conf) -> Result<Pool<RedisConnectionManager>, Error> {
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
        _ => {
            return Err(Error::new(
                "Redis Setup Error - Password must be set if username is set",
            ))
        }
    };
    // build manager
    let manager = RedisConnectionManager::new(url)?;
    // build redis connection pool
    let pool = Pool::builder()
        .max_size(redis.pool_size.unwrap_or(50))
        .build(manager)
        .await?;
    Ok(pool)
}
