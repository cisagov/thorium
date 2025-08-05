use chrono::prelude::*;
use thorium::models::Pools;
use tracing::{event, instrument, Level};

use super::Target;

/// Gets a timestamp for a number of seconds from now
#[macro_export]
macro_rules! from_now {
    ($seconds:expr) => {
        chrono::Utc::now() + chrono::Duration::seconds($seconds as i64)
    };
    ($start:expr, $seconds:expr) => {
        $start + tokio::time::Duration::from_secs($seconds)
    };
}

/// Worker lifetime constraints
///
/// These are weakly enforced.
pub enum Lifetime {
    /// Constrain worker lifetime on number of jobs executed
    JobCount { current: usize, limit: usize },
    /// Constrain worker on runtime
    RunTime(DateTime<Utc>),
    /// Worker will live as long as jobs exist to run
    Infinite(),
}

impl Lifetime {
    /// Create a new image lifetime object
    ///
    /// This is for the lifetime of the image not of the job
    ///
    /// # Arguments
    ///
    /// * `target` - The target job for this worker
    #[instrument(name = "Lifetime::new", skip_all)]
    pub fn new(target: &Target) -> Self {
        // if this was spawned under the fair share pool then set a lifetime of 1 minute at most
        match (target.pool, &target.image.lifetime) {
            // fair share spawned workers with no lifetime can only execute 1 minute worth of jobs before dying
            (Pools::FairShare, None) => Lifetime::RunTime(from_now!(60)),
            (Pools::FairShare, Some(lifetime)) => {
                // if our lifetime is one job or less than 1 minute of time then use the images lifetime
                match lifetime.counter.as_ref() {
                    "jobs" => Lifetime::JobCount {
                        current: 0,
                        limit: 1,
                    },
                    "time" => {
                        // cap our time based lifetime at 60 seconds
                        let seconds = std::cmp::min(60, lifetime.amount);
                        Lifetime::RunTime(from_now!(seconds))
                    }
                    _ => panic!("Unknown lifetime: {}", lifetime.counter),
                }
            }
            // all other pools use the images lifetime or set it to infinite
            (_, _) => {
                if let Some(lifetime) = &target.image.lifetime {
                    match lifetime.counter.as_ref() {
                        "jobs" => Lifetime::JobCount {
                            current: 0,
                            limit: lifetime.amount as usize,
                        },
                        "time" => Lifetime::RunTime(from_now!(lifetime.amount)),
                        _ => panic!("unknown lifetime handler {}", lifetime.counter),
                    }
                } else {
                    Lifetime::Infinite()
                }
            }
        }
    }

    /// Update our current lifetime position and check if we have met/exceeded our lifetime
    #[instrument(name = "Lifetime::exceeded", skip_all)]
    pub fn exceeded(&mut self) -> bool {
        // check if we have exceeded our lifetime
        match self {
            Lifetime::JobCount { current, limit } => {
                if current >= limit {
                    event!(
                        Level::INFO,
                        kind = "JobCount",
                        exceeded = true,
                        current,
                        limit
                    );
                    true
                } else {
                    false
                }
            }
            Lifetime::RunTime(current) => {
                if *current < Utc::now() {
                    event!(
                        Level::INFO,
                        kind = "RunTime",
                        exceeded = true,
                        current = current.to_string(),
                        now = Utc::now().to_string(),
                    );
                    true
                } else {
                    false
                }
            }
            Lifetime::Infinite() => false,
        }
    }

    /// increment our job counter if necessary
    pub fn claimed_job(&mut self) {
        // increment lifetime
        if let Lifetime::JobCount { current, limit: _ } = self {
            *current += 1
        };
    }
}
