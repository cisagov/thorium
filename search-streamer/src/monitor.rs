//! Monitor our exports for errors and push them into the queue

use kanal::AsyncReceiver;
use thorium::Error;
use thorium::client::SearchEventsClient;
use thorium::models::SearchEventStatus;
use tracing::{Level, event, instrument};

use crate::init::InitSession;
use crate::msg::JobStatus;
use crate::sources::DataSource;

pub struct Monitor<D: DataSource> {
    /// The client to use to send status on events
    event_client: D::EventClient,
    /// The channel to listen for worker progress updates on
    progress_rx: AsyncReceiver<JobStatus>,
    // TODO: make Monitor stateful to avoid checking option each loop
    /// The init session
    init_session: Option<InitSession>,
}

impl<D: DataSource> Monitor<D> {
    /// Create a new monitor
    ///
    /// # Arguments
    ///
    /// * `event_client` - The client to use to send status on events
    /// * `progress_rx` - The channel to listen for progress on
    /// * `init_session` - A possible init session to use to track progress
    pub fn new(
        event_client: D::EventClient,
        progress_rx: AsyncReceiver<JobStatus>,
        init_session: Option<InitSession>,
    ) -> Self {
        Monitor {
            event_client,
            progress_rx,
            init_session,
        }
    }

    /// Monitor our export for any errors and send them to workers
    #[instrument(name = "Monitor::start", fields(data = D::DATA_NAME), skip_all, err(Debug))]
    pub async fn start(mut self) -> Result<(), Error> {
        loop {
            // create a new event status report
            let mut status = SearchEventStatus::default();
            // track how many iterations it has been since we cleared events
            let mut since_send_status = 0;
            // handle any responses
            while let Some(resp) = self.progress_rx.try_recv()? {
                // handle our message
                match resp {
                    JobStatus::InitComplete { start, end } => {
                        // TODO: Make Monitor stateful to avoid checking option each loop
                        if let Some(init_session) = self.init_session.as_mut() {
                            // update our log with the tokens that we completed
                            init_session.log(start, end).await.map_err(|err| {
                                Error::new(format!("Error logging completed token range: {err}"))
                            })?;
                            // remove the tokens from the remaining map
                            init_session.tokens_remaining.remove(&start);
                            // check if we're done initiating
                            if init_session.tokens_remaining.is_empty() {
                                // log that we've completed initiation
                                event!(
                                    Level::INFO,
                                    msg = "Index initiation complete!",
                                    duration = init_session.duration()
                                );
                                // TODO: state would be good here so we don't accidently try to
                                // keep using a finished session
                                // finish our session, deleting its info from Redis
                                init_session.finish().await.map_err(|err| {
                                    Error::new(format!(
                                        "Failed to delete init session data from Redis: {err}"
                                    ))
                                })?;
                                // set our init session to None
                                self.init_session = None;
                            } else {
                                // calculate how many chunks we've completed
                                let completed = init_session
                                    .info
                                    .chunk_count
                                    .checked_sub(init_session.tokens_remaining.len() as u64);
                                // log this chunk as complete
                                event!(
                                    Level::INFO,
                                    msg = "Init chunk completed",
                                    start = start,
                                    end = end,
                                    completed = completed,
                                    remaining = init_session.tokens_remaining.len()
                                );
                            }
                        } else {
                            /* TODO:
                             *  Make Monitor stateful to avoid checking option each loop
                             *  This should never occur, but having state would make it actually impossible
                             */
                            return Err(Error::new(format!(
                                "No init session in progress but got init response from worker! start: {start}, end: {end}"
                            )));
                        }
                    }
                    // events were completed so add their IDs to our success list
                    JobStatus::EventComplete { mut ids } => status.successes.append(&mut ids),
                    // events errored, so log the error and add their IDs to our failure list
                    JobStatus::EventError { error, mut ids } => {
                        // log our error
                        // TODO: we are logging the elastic error in its entirety here;
                        // if those errors are too long, we might want to truncate instead
                        event!(
                            Level::ERROR,
                            msg = format!(
                                "Failed event: {}",
                                error
                                    .msg()
                                    .unwrap_or_else(|| "An unknown error occurred".to_string()),
                            ),
                            ids = format!("{ids:?}"),
                        );
                        // add the failed event to the list of failures for the API to retry later
                        status.failures.append(&mut ids);
                    }
                }
                // increment our counter
                since_send_status += 1;
                // if there has been more than 5000 iterations since we've sent status, break out
                if since_send_status > 5000 {
                    break;
                }
            }
            // report our status to Thorium if it's not empty
            if !status.is_empty() {
                // send the status of the events back to thorium
                self.event_client
                    .send_status(&status)
                    .await
                    .map_err(|err| {
                        Error::new(format!("Failed to send event status to Thorium: {err}"))
                    })?;
                // log how many events we've handled
                event!(
                    Level::INFO,
                    events_success = status.successes.len(),
                    events_failed = status.failures.len()
                );
            }
            // sleep for 1 second if we emptied our queue
            if self.progress_rx.is_empty() {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}
