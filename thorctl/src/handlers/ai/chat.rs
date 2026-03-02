//! Chat with an AI in thorctl

use kanal::{AsyncReceiver, AsyncSender};
use thorium::Error;
use thorium::ai::{AiSupport, ThorChat};

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    MouseButton, MouseEventKind,
};
use crossterm::execute;
use ratatui::{DefaultTerminal, Frame};
use std::io::stdout;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration as StdDuration;

use crate::args::ai::Chat as ChatArgs;

mod chat;
mod components;
mod events;
mod home;
mod tabs;

pub(super) use chat::Chat;
pub(super) use events::{AppEvent, ScrollEvent};
pub(super) use home::Home;
pub(super) use tabs::{ActiveTabKind, Tab};

async fn refresh(sender: AsyncSender<AppEvent>, should_refresh: Arc<AtomicBool>) {
    // keep refreshing every 50ms
    loop {
        // check if we should exit
        if !should_refresh.load(Ordering::Relaxed) {
            // stop sending refresh events
            break;
        }
        // try to send a redraw event
        // we don't care if this fails since it would just break the animation
        let _ = sender.send(AppEvent::Redraw).await;
        // sleep for 50 ms
        tokio::time::sleep(StdDuration::from_millis(50)).await;
    }
}

/// The current mode of the application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Normal mode for navigation and commands
    Normal,
    /// Insert mode for text input
    Insert,
}

/// The thorium application state
pub struct App<A: AiSupport + 'static> {
    /// The sender side of our event channel
    event_tx: AsyncSender<AppEvent>,
    /// The receive side of our event channel
    event_rx: AsyncReceiver<AppEvent>,
    /// A single tab in the chat tui
    tab: Tab<A>,
    /// The current mode for this app
    mode: Mode,
    /// Whether the tui should exit or not
    should_quit: bool,
    /// Whether the tui should refresh every so often
    should_refresh: Arc<AtomicBool>,
}

impl<A: AiSupport + 'static> App<A> {
    /// Create a new chat app
    ///
    /// # Arguments
    ///
    /// * `thor_chat` - The chat client for Thorium
    pub fn new(thor_chat: ThorChat<A>) -> Self {
        // build our event channel
        let (event_tx, event_rx) = kanal::unbounded_async();
        // build our initial tab
        let tab = Tab::new("New Tab", thor_chat, &event_tx);
        App {
            event_tx,
            event_rx,
            tab,
            mode: Mode::Normal,
            should_quit: false,
            should_refresh: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Handle key input when the query box is focused
    ///
    /// # Arguments
    ///
    /// * `code` - The key code that was pressed
    pub async fn handle_key(&mut self, key: KeyEvent) {
        // check if this is a ctrl-c
        match key.code {
            // if this is a ctrl-c then we need to quit
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true
            }
            // forward all other keys to our tab
            _ => self.tab.handle_key(&mut self.mode, key).await,
        }
    }

    /// Handle a mouse click event
    ///
    /// # Arguments
    ///
    /// * `x` - The column position of the click
    /// * `y` - The row position of the click
    fn handle_click(&mut self, x: u16, y: u16) {
        self.tab.handle_click(x, y);
    }

    /// Handle an incoming terminal event
    ///
    /// Dispatches key presses and mouse clicks to the appropriate handlers.
    ///
    /// # Arguments
    ///
    /// * `event` - The terminal event to process
    pub async fn handle_event(&mut self, event: Event) {
        match event {
            // handle key press events
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                self.handle_key(key).await;
            }
            // handle mouse click events (work in any mode)
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::Down(MouseButton::Left) => {
                    self.handle_click(mouse.column, mouse.row);
                }
                MouseEventKind::ScrollUp => self.tab.handle_scroll(ScrollEvent::ScrollUp),
                MouseEventKind::ScrollDown => self.tab.handle_scroll(ScrollEvent::ScrollDown),
                MouseEventKind::ScrollLeft => self.tab.handle_scroll(ScrollEvent::ScrollLeft),
                MouseEventKind::ScrollRight => self.tab.handle_scroll(ScrollEvent::ScrollRight),
                _ => {}
            },
            // ignore all other events
            _ => {}
        }
    }

    /// Update a tabs active content kind
    fn set_active_tab_kind(&mut self, new: ActiveTabKind) {
        // handle the change to the new tab kind
        match new {
            ActiveTabKind::Home | ActiveTabKind::Chat => self.tab.active = new,
        }
    }

    /// Render the application to the terminal
    ///
    /// Draws the tab bar at the top, the content area in the middle,
    /// the query input below that, and the status bar at the bottom.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to render widgets to
    pub fn render(&mut self, frame: &mut Frame) {
        // render our main tab
        self.tab.render(self.mode, frame);
    }

    /// Start handling events and updating our tui
    pub async fn start(&mut self, terminal: &mut DefaultTerminal) -> std::io::Result<()> {
        // spawn our event forwarders
        events::spawn_forwarders(&self.event_tx);
        // draw an initial frame until we get an event to handle
        terminal.draw(|frame| self.render(frame))?;
        // TODO only send this when we are on the home page
        tokio::task::spawn(refresh(self.event_tx.clone(), self.should_refresh.clone()));
        // draw an initial frame until we get an event to handle
        while !self.should_quit {
            // get the next event to handle
            match self.event_rx.recv().await {
                Ok(event) => {
                    // handle a terminal event
                    match event {
                        // handle this terminal event
                        AppEvent::Terminal(terminal_event) => {
                            self.handle_event(terminal_event).await;
                        }
                        // convert our home tab to a chat tab
                        AppEvent::HomeToChat => {
                            // convert our home page to a chat page
                            self.tab.home_to_chat().await;
                            // stop our refresh worker
                            self.should_refresh.store(false, Ordering::Relaxed);
                        }
                        // process a tool call failure
                        AppEvent::ToolCallFailure { error } => (),
                        // Change our active tab kind
                        AppEvent::SetActiveTabKind(kind) => self.set_active_tab_kind(kind),
                        // don't do anything just redraw the tui
                        AppEvent::Redraw => (),
                    }
                    // Redraw after handling any event
                    terminal.draw(|frame| self.render(frame))?;
                }
                Err(_) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
        Ok(())
    }
}

/// Start a ai chat terminal UI
///
/// # Arguments
///
/// * `thorchat` - The ai chat client to use
/// * `cmd` - The arguments for chat
pub async fn tui<A: AiSupport + 'static>(
    thorchat: ThorChat<A>,
    _cmd: &ChatArgs,
) -> Result<(), Error> {
    // setup eyre so it can help us have nice errors
    color_eyre::install().unwrap();
    // start capturing and handling mouse clicks
    execute!(stdout(), EnableMouseCapture)?;
    // initialize the terminal
    let mut terminal = ratatui::init();
    // Create our app with the request channel
    let mut app = App::new(thorchat);
    // start rendering our app
    app.start(&mut terminal).await?;
    // restore the terminal to its original state
    ratatui::restore();
    // stop capturing and handling mouse clicks
    execute!(stdout(), DisableMouseCapture)?;
    Ok(())
}
