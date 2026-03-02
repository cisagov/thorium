//! The chat page for the Thorium chat tui

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use std::sync::Arc;
use thorium::ai::AiSupport;

use super::components::{MessageLog, ShortcutMenu, TextBox, ThorChatClient, status_bar};
use super::{ActiveTabKind, AppEvent, Mode, ScrollEvent};

pub struct Chat<A: AiSupport + 'static> {
    /// The client for this chat pages ai
    thor_chat: Arc<ThorChatClient<A>>,
    /// The message log
    messages: MessageLog,
    /// The text box for our prompt
    pub prompt: TextBox,
    /// The shortcut menu for this page
    pub shortcuts: ShortcutMenu,
}

impl<A: AiSupport + 'static> Chat<A> {
    /// Create a new chat page
    ///
    /// # Arguments
    ///
    /// * `thor_chat` - A client to a detached ThorChat
    /// * `log` - The initial chat logs to use
    pub async fn new(thor_chat: &Arc<ThorChatClient<A>>, log: Vec<String>) -> Self {
        // send all of our log messages
        thor_chat.to_chat.send(log[0].clone()).await.unwrap();
        // build our promp
        let prompt = TextBox::default()
            .title("Prompt")
            .placeholder("How can Thorium help?")
            .focused();
        // build the shortcut menu for this page
        let shortcuts = ShortcutMenu::builder().add('h', "Go home").build();
        // build our chat component
        Chat {
            thor_chat: thor_chat.clone(),
            messages: MessageLog::default(),
            prompt,
            shortcuts,
        }
    }

    /// Clear this chat and reset it to its default state
    pub fn clear_and_go_home(&mut self) -> Option<AppEvent> {
        // clear our messages
        self.messages.clear();
        // clear our pompt
        self.prompt.clear();
        // clear any messages in our ai chat client
        self.thor_chat.context.clear();
        // return an event that changes our active tab to home
        Some(AppEvent::SetActiveTabKind(ActiveTabKind::Home))
    }

    /// Handle key input for this text box
    ///
    /// # Arguments
    ///
    /// * `code` - The key code that was pressed
    pub async fn handle_key(&mut self, mode: &mut Mode, key: KeyEvent) -> Option<AppEvent> {
        // if our prompt is focused then send the key to the prompt
        if self.prompt.is_focused {
            // pass this input to our prompt text box
            if self.prompt.handle_key(key, false) {
                // consume our current prompt box
                if let Some(message) = self.prompt.consume() {
                    // send the latest message to the ai
                    self.thor_chat.to_chat.send(message.clone()).await.unwrap();
                }
            }
        } else {
            // handle this key
            match (key.code, &mode, self.shortcuts.active) {
                // toggle shortcut mode off and on
                (KeyCode::Char(' '), Mode::Normal, _) => self.shortcuts.toggle(),
                // toggle our shortcut mode off
                (KeyCode::Esc, Mode::Normal, true) => self.shortcuts.toggle(),
                // go back to normal mode
                (KeyCode::Esc, Mode::Insert, _) => *mode = Mode::Normal,
                // go to insert mode
                (KeyCode::Char('i'), Mode::Normal, false) => *mode = Mode::Insert,
                // clear our context and go back home
                (KeyCode::Char('h'), Mode::Normal, true) => return self.clear_and_go_home(),
                _ => (),
            }
        }
        None
    }

    /// Handle a mouse click event
    ///
    /// # Arguments
    ///
    /// * `x` - The column position of the click
    /// * `y` - The row position of the click
    pub fn handle_click(&mut self, x: u16, y: u16) {
        // right now the only thing clickable is the prompt box
        self.prompt.handle_click(x, y);
    }

    /// Handle a scroll event
    ///
    /// # Arguments
    ///
    /// * `event` - The scroll event to handle
    pub fn handle_scroll(&mut self, event: ScrollEvent) {
        match event {
            ScrollEvent::ScrollUp => self.messages.scroll_up(3),
            ScrollEvent::ScrollDown => self.messages.scroll_down(3),
            // there is nothing to scroll left or right on
            ScrollEvent::ScrollLeft | ScrollEvent::ScrollRight => (),
        }
    }

    /// Render the home tab content
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to render to
    /// * `area` - The area to render the input in
    pub fn render(&mut self, mode: Mode, frame: &mut Frame) {
        // split the frame into four vertical chunks: tabs, content, query input, and status bar
        let chunks = Layout::vertical([
            Constraint::Length(3), // tabs
            Constraint::Min(0),    // content
            Constraint::Length(8), // promp input
            Constraint::Length(2), // status bar
        ])
        .split(frame.area());
        // get a guard to our shared context
        let guard = self.thor_chat.context.access();
        // render any visible messages
        self.messages.render(frame, chunks[1], &guard.history);
        // render our prompt
        self.prompt.render(frame, chunks[2]);
        // render our status bar
        status_bar::render(frame, chunks[3], mode);
        // render our shortcut menu if its active
        self.shortcuts.render(frame);
    }
}
