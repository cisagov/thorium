//! The different events to handle in the TUI

use crossterm::event::{Event, EventStream};
use futures::stream::StreamExt;
use kanal::AsyncSender;
use thorium::client::Error;

use crate::handlers::ai::chat::ActiveTabKind;

/// Events that can be received by the main event loop
pub enum AppEvent {
    /// A terminal event (keyboard, mouse, resize, etc.)
    Terminal(Event),
    /// A tool call failed
    ToolCallFailure { error: Error },
    /// Turn the home page into a Chat page
    HomeToChat,
    /// Set the active page
    SetActiveTabKind(ActiveTabKind),
    /// Redraw the tui
    Redraw,
}

/// Background task that forwards terminal events to the central event channel
async fn terminal_event_forwarder(event_tx: AsyncSender<AppEvent>) {
    let mut event_stream = EventStream::new();
    while let Some(event_result) = event_stream.next().await {
        match event_result {
            Ok(event) => {
                if event_tx.send(AppEvent::Terminal(event)).await.is_err() {
                    // Channel closed, stop forwarding
                    break;
                }
            }
            Err(_) => {
                // Terminal event error, continue trying
                continue;
            }
        }
    }
}

/// Spawn the core event forwarders
pub fn spawn_forwarders(sender: &AsyncSender<AppEvent>) {
    // Spawn our terminal event fowarder
    tokio::spawn(terminal_event_forwarder(sender.clone()));
}

/// The different kind of scroll events
pub enum ScrollEvent {
    /// A scroll up event
    ScrollUp,
    /// A scroll down event
    ScrollDown,
    /// A scroll left event
    ScrollLeft,
    /// A Scroll right event
    ScrollRight,
}
