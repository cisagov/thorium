//! The different tabs in thor chat

use crossterm::event::KeyEvent;
use kanal::AsyncSender;
use ratatui::Frame;
use std::sync::Arc;
use thorium::ai::{AiSupport, ThorChat};
use uuid::Uuid;

use crate::handlers::ai::chat::Mode;
use crate::handlers::ai::chat::components::ThorChatClient;

use super::{AppEvent, Chat, Home, ScrollEvent};

/// Handle a key press on the active page or fallback to home if this doesn't exist
macro_rules! handle_key_fallback {
    ($content:expr, $kind:ident, $key:expr, $mode:expr, $active:expr) => {
        // if we have a chat page then forward this event
        match &mut $content.$kind {
            Some(kind) => kind.handle_key($mode, $key).await,
            None => {
                // we don't have a chat window so fallback to the home page handling this
                $content.home.handle_key($mode, $key);
                // update our active page to be home
                $active = ActiveTabKind::Home;
                None
            }
        }
    };
}

/// Handle a click on the active page or fallback to home if this doesn't exist
macro_rules! handle_click_fallback {
    ($content:expr, $kind:ident, $x:expr, $y:expr, $active:expr) => {
        // if we have a chat page then forward this event
        match &mut $content.$kind {
            Some(kind) => kind.handle_click($x, $y),
            None => {
                // we don't have a chat window so fallback to the home page handling this
                $content.home.handle_click($x, $y);
                // update our active page to be home
                $active = ActiveTabKind::Home;
            }
        }
    };
}

/// Handle a scroll event on the active page or fallback to home if this doesn't exist
macro_rules! handle_scroll_fallback {
    ($content:expr, $kind:ident, $scroll:expr) => {
        // if we have a chat page then forward this event
        match &mut $content.$kind {
            Some(kind) => kind.handle_scroll($scroll),
            None => (),
        }
    };
}

/// Render the active page or fallback to home if it doesn't exist
macro_rules! render_fallback {
    ($content:expr, $kind:ident, $mode:expr, $frame:expr, $active:expr) => {
        // if we have a chat page then forward this event
        match &mut $content.$kind {
            Some(kind) => kind.render($mode, $frame),
            None => {
                // we don't have a chat window so fallback to the home page handling this
                $content.home.render($mode, $frame);
                // update our active page to be home
                $active = ActiveTabKind::Home;
            }
        }
    };
}

/// The currently active page on this tab
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTabKind {
    /// The home page
    Home,
    /// The chat page
    Chat,
}

/// The different pages and their content
pub struct TabContent<A: AiSupport + 'static> {
    /// The content on the home page
    home: Home,
    /// The content on the chat page
    chat: Option<Chat<A>>,
}

impl<A: AiSupport + 'static> Default for TabContent<A> {
    /// Create a default [`TabContent`] with only home initialized
    fn default() -> Self {
        TabContent {
            home: Home::default(),
            chat: None,
        }
    }
}

/// A single tab in the application
pub struct Tab<A: AiSupport + 'static> {
    /// The unique identifier for this tab
    pub id: Uuid,
    /// The display label for this tab
    pub label: String,
    /// The currently active page kind
    pub active: ActiveTabKind,
    /// The content to display
    pub tab_content: TabContent<A>,
    /// The thorium chat client for this tab
    thor_chat: Arc<ThorChatClient<A>>,
    /// The sender side of our event channel
    event_tx: AsyncSender<AppEvent>,
}

impl<A: AiSupport + 'static> Tab<A> {
    /// Create a new tab
    ///
    /// # Arguments
    ///
    /// * `label` - The label to for this new tab
    pub fn new(
        label: impl Into<String>,
        thor_chat: ThorChat<A>,
        event_tx: &AsyncSender<AppEvent>,
    ) -> Self {
        // convert this into a detached thor chat client
        let thor_chat = Arc::new(ThorChatClient::new(thor_chat, &event_tx));
        // build a new tab
        Tab {
            id: Uuid::new_v4(),
            label: label.into(),
            active: ActiveTabKind::Home,
            tab_content: TabContent::default(),
            thor_chat,
            event_tx: event_tx.clone(),
        }
    }

    /// Handle key input when the query box is focused
    ///
    /// # Arguments
    ///
    /// * `code` - The key code that was pressed
    pub async fn handle_key(&mut self, mode: &mut Mode, key: KeyEvent) {
        // pass this key event to the currently active tab kind
        let event = match self.active {
            ActiveTabKind::Home => self.tab_content.home.handle_key(mode, key),
            ActiveTabKind::Chat => {
                handle_key_fallback!(self.tab_content, chat, key, mode, self.active)
            }
        };
        // If we got an event then put it into the queue to be handled
        if let Some(event) = event {
            self.event_tx.send(event).await.unwrap();
        }
    }

    /// Handle a mouse click event
    ///
    /// # Arguments
    ///
    /// * `x` - The column position of the click
    /// * `y` - The row position of the click
    pub fn handle_click(&mut self, x: u16, y: u16) {
        // pass this mouse click event to the currently active tab kind
        match self.active {
            ActiveTabKind::Home => self.tab_content.home.handle_click(x, y),
            ActiveTabKind::Chat => {
                handle_click_fallback!(self.tab_content, chat, x, y, self.active)
            }
        };
    }

    /// Handle a scroll event
    ///
    /// # Arguments
    ///
    /// * `event` - The scroll event to handle
    pub fn handle_scroll(&mut self, event: ScrollEvent) {
        // only the chat page supports scrolling right now
        match self.active {
            // the home page doesn't do anything with scroll events yet
            ActiveTabKind::Home => (),
            // send this scroll event to our chat page
            ActiveTabKind::Chat => handle_scroll_fallback!(self.tab_content, chat, event),
        }
    }

    /// Convert this tabs home page to a chat page
    pub async fn home_to_chat(&mut self) {
        if self.active == ActiveTabKind::Home {
            // get our current home tab and replace it with a default
            let home = std::mem::take(&mut self.tab_content.home);
            // build a chat page from our home page content
            let chat = Chat::new(&self.thor_chat, vec![home.prompt.content]).await;
            // insert this into our chat tab content
            self.tab_content.chat = Some(chat);
            // set our active tab to the chat tab
            self.active = ActiveTabKind::Chat;
        }
    }

    /// Render this tabs content
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to render to
    /// * `area` - The area to render the input in
    pub fn render(&mut self, mode: Mode, frame: &mut Frame) {
        match self.active {
            ActiveTabKind::Home => self.tab_content.home.render(mode, frame),
            ActiveTabKind::Chat => {
                render_fallback!(self.tab_content, chat, mode, frame, self.active)
            }
        }
    }
}
