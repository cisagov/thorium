//! The messages between the search streamer and worker

use thorium::Error;
use uuid::Uuid;

use crate::sources::DataSource;

/// The messages between the search streamer controller and worker
pub enum Job<D: DataSource> {
    /// An init job
    Init { start: i64, end: i64 },
    /// An event job
    Event { compacted_event: D::CompactEvent },
}

/// A status report for a given job from workers to the monitor
pub enum JobStatus {
    /// An init job was completed
    InitComplete { start: i64, end: i64 },
    /// An event job was completed
    EventComplete { ids: Vec<Uuid> },
    /// An error occurred in an event job
    EventError { error: Error, ids: Vec<Uuid> },
}
