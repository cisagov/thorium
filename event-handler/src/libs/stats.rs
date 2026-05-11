//! Combines and reports on stats across all workers in Thorium

use chrono::prelude::*;
use kanal::{AsyncReceiver, AsyncSender};
use std::time::Duration;
use thorium::Error;
use tokio::task::JoinHandle;

use crate::libs::workers::EventWorkerKinds;

use super::workers::{EventWorkerSupportCore, ReactionTriggerWorker, SigmaRuleWorker};

/// The different channels for for each event worker
pub struct EventWorkerStatChannels {
    /// The channel for reaction trigger workers
    pub reaction_trigger: (
        AsyncSender<<ReactionTriggerWorker as EventWorkerSupportCore>::Stats>,
        AsyncReceiver<<ReactionTriggerWorker as EventWorkerSupportCore>::Stats>,
    ),
    /// The channel for sigma rule workers
    pub sigma_rules: (
        AsyncSender<<SigmaRuleWorker as EventWorkerSupportCore>::Stats>,
        AsyncReceiver<<SigmaRuleWorker as EventWorkerSupportCore>::Stats>,
    ),
}

impl EventWorkerStatChannels {
    // Build a stats worker for this kind of worker
    pub fn spawn_worker(&self, kind: EventWorkerKinds) -> JoinHandle<Result<(), Error>> {
        // buld the correct worker
        match kind {
            EventWorkerKinds::ReactionTriggers => {
                // build the correct worker
                let stats_worker =
                    EventWorkerStats::<ReactionTriggerWorker>::new(&self.reaction_trigger.1);
                // spawn this stats worker
                tokio::task::spawn(stats_worker.start())
            }
            EventWorkerKinds::SigmaScanner => {
                // build the correct worker
                let stats_worker = EventWorkerStats::<SigmaRuleWorker>::new(&self.sigma_rules.1);
                // spawn this stats worker
                tokio::task::spawn(stats_worker.start())
            }
        }
    }
}

impl Default for EventWorkerStatChannels {
    fn default() -> Self {
        EventWorkerStatChannels {
            reaction_trigger: kanal::unbounded_async(),
            sigma_rules: kanal::unbounded_async(),
        }
    }
}

pub struct EventWorkerStats<W: EventWorkerSupportCore> {
    /// The last reported stats
    last_reported: W::Stats,
    /// The current stats
    current: W::Stats,
    /// When stats can be reported again
    next_report: DateTime<Utc>,
    /// The channel to listen for stats on
    stats_rx: AsyncReceiver<W::Stats>,
}

impl<W: EventWorkerSupportCore> EventWorkerStats<W> {
    /// Create a new event worker stats reporter
    ///
    /// # Arguments
    ///
    /// * `stats_rx` - The channel to receive stats over
    pub fn new(stats_rx: &AsyncReceiver<W::Stats>) -> Self {
        EventWorkerStats {
            last_reported: W::Stats::default(),
            current: W::Stats::default(),
            next_report: Utc::now() + chrono::Duration::seconds(5),
            stats_rx: stats_rx.clone(),
        }
    }

    /// Report our current stats
    ///
    /// This will move the current stats into the last reported stats.
    fn report(&mut self) {
        // report this workers stats
        W::stats(&self.current, &self.last_reported);
        // update our last reported stats
        self.last_reported = self.current.clone();
        // don't report starts for another 5 seconds at least
        self.next_report = Utc::now() + chrono::Duration::seconds(5);
    }

    /// Start tracking and reporting stats for this worker kind
    pub async fn start(mut self) -> Result<(), Error> {
        // always print out a zero stats message so we know things have started
        self.report();
        // continue getting and reporting on stats forever
        loop {
            // get the next stats update
            if let Some(update) = self.stats_rx.try_recv()? {
                // update our current stats
                self.current += update;
                // check if are ready to report starts again
                if Utc::now() > self.next_report {
                    // we don't need to also check if we have new stats beacuse we just got a new stat update
                    // and workers shouldn't be sending out empty stat updates
                    self.report();
                }
            } else {
                // we didn't get any new stats but maybe we can report our current stats?
                if Utc::now() > self.next_report && self.current != self.last_reported {
                    // we have new stats and we can report again
                    self.report();
                }
                // we didn't get any new stats updates so sleep for 2s
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
}
