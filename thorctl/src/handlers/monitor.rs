//! The global monitor for our workers in Thorctl

use kanal::AsyncReceiver;
use tokio::task::JoinHandle;

use super::progress::{Bar, BarKind, MultiBar};

/// The messages to send new jobs to workers with
pub enum MonitorMsg<M: Monitor> {
    /// A update to apply to our monitors progress bar
    Update(M::Update),
    /// Extend our total progress bars length
    Extend(u64),
    /// There are no more jobs so this worker should shutdown
    Finished,
}

/// A simple monitor that increments a bounded progress bar on each update.
///
/// Use as `type MyMonitor = SimpleMonitor;` instead of writing a custom impl
/// when the only behavior needed is `bar.inc(1)` per update.
///
/// The `RATE` const parameter controls whether the global bar also displays the
/// per-second rate of progress. Use `SimpleMonitor<true>` for workloads like
/// parallel file downloads where the completion rate is useful; the default
/// `SimpleMonitor` (i.e. `SimpleMonitor<false>`) shows a plain count bar.
pub struct SimpleMonitor<const RATE: bool = false>;

/// A [`SimpleMonitor`] whose global bar also displays the per-second rate of
/// progress, for workloads like parallel file downloads where the completion
/// rate is useful to see
pub type SimpleRateMonitor = SimpleMonitor<true>;

impl<const RATE: bool> Monitor for SimpleMonitor<RATE> {
    type Update = ();

    /// Build a bounded progress bar, with a per-second rate when `RATE` is set
    ///
    /// # Arguments
    ///
    /// * `multi` - The multibar to add the bar to
    /// * `msg` - The message to set for the bar
    fn build_bar(multi: &MultiBar, msg: &str) -> Bar {
        // pick a rate-displaying bar when requested, otherwise a plain count bar
        let kind = if RATE {
            BarKind::BoundRate(0)
        } else {
            BarKind::Bound(0)
        };
        multi.add(msg, kind)
    }

    /// Advance the bar by one for each completed update
    ///
    /// # Arguments
    ///
    /// * `bar` - The bar to advance
    /// * `_` - The (empty) update payload
    fn apply(bar: &Bar, _: Self::Update) {
        bar.inc(1);
    }
}

/// A global progress monitor driving a single bar from worker updates
pub trait Monitor: Send + 'static {
    /// The update type to use
    type Update: Send;

    /// build this monitors progress bar
    ///
    /// # Arguments
    ///
    /// * `multi` - The multibar to add a bar too
    /// * `msg`- The message to set for our monitor bar
    fn build_bar(multi: &MultiBar, msg: &str) -> Bar;

    /// Apply an update to our global progress bar
    ///
    /// # Arguments
    ///
    /// * `bar` - The bar to apply updates too
    /// * `update` - The update to apply
    fn apply(bar: &Bar, update: Self::Update);
}

pub(crate) struct MonitorHandler<M: Monitor> {
    /// The channel to receive monitor updates on
    update_rx: AsyncReceiver<MonitorMsg<M>>,
    /// The global bar to display progress on
    global_bar: Bar,
}

impl<M: Monitor> MonitorHandler<M> {
    /// Create and spawn a new global monitor
    ///
    /// # Arguments
    ///
    /// * `msg`- The message to set for our monitor bar
    /// * `update_rx` - The channel to listen for monitor updates on
    /// * `bar` - The bar to log progress too
    pub fn spawn(
        msg: &str,
        update_rx: AsyncReceiver<MonitorMsg<M>>,
        multi: &MultiBar,
    ) -> JoinHandle<()> {
        // get a new global bar
        let bar = M::build_bar(multi, msg);
        // build a new global monitor
        let monitor = MonitorHandler {
            update_rx,
            global_bar: bar,
        };
        // spawn our global monitor
        tokio::spawn(async move { monitor.start().await })
    }

    /// Start handling updates
    async fn start(self) {
        // handle messages in our channel until its closed
        loop {
            // get the next message in the queue
            match self.update_rx.recv().await {
                Ok(MonitorMsg::Update(update)) => M::apply(&self.global_bar, update),
                Ok(MonitorMsg::Extend(delta)) => self.global_bar.inc_length(delta),
                Ok(MonitorMsg::Finished) => break,
                Err(kanal::ReceiveError::Closed) => panic!("Monitor: ReceiveError::Closed"),
                Err(kanal::ReceiveError::SendClosed) => panic!("Monitor: ReceiveError::SendClosed"),
            }
        }
        // finish our global bar
        self.global_bar.finish();
    }
}
