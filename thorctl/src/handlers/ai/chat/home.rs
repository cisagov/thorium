//! The home page for the Thorium chat tui

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};

use super::components::{Banner, ShortcutMenu, TextBox, status_bar};
use super::{AppEvent, Mode};

pub struct Home {
    /// The Thorium banner
    banner: Banner,
    /// The text box for our prompt
    pub prompt: TextBox,
    /// The shortcut menu for this page
    pub shortcuts: ShortcutMenu,
}

impl Default for Home {
    fn default() -> Self {
        // build the banner for our home page
        let banner = Banner::new("Thorium");
        // build our promp
        let prompt = TextBox::default()
            .title("Prompt")
            .placeholder("How can Thorium help?")
            .focused();
        // build the shortcut menu for this page
        let shortcuts = ShortcutMenu::builder().build();
        // build our home page
        Home {
            banner,
            prompt,
            shortcuts,
        }
    }
}

impl Home {
    /// Create a new home page
    ///
    /// # Arguments
    ///
    /// * `prompt` - the prompt to set for this home page
    pub fn new(prompt: impl Into<String>) -> Self {
        // start with a default home page
        let mut home = Home::default();
        // update our prompt
        home.prompt.placeholder = prompt.into();
        // return our new home page
        home
    }

    /// Handle key input for this text box
    ///
    /// # Arguments
    ///
    /// * `code` - The key code that was pressed
    pub fn handle_key(&mut self, mode: &mut Mode, key: KeyEvent) -> Option<AppEvent> {
        // if our prompt is focused then send the key to the prompt
        if self.prompt.is_focused {
            // send this input to our prompt text box and check if we
            // are consuming this prompt box and changing to a chat page
            // pass this input to our prompt text box
            if self.prompt.handle_key(key, false) {
                // tell our tab we are changing our active page to a chat page
                return Some(AppEvent::HomeToChat);
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
            Constraint::Length(8), // query input
            Constraint::Length(2), // status bar
        ])
        .split(frame.area());
        // render a banner
        self.banner.render(frame, chunks[1]);
        // render our prompt
        self.prompt.render(frame, chunks[2]);
        // render our status bar
        status_bar::render(frame, chunks[3], mode);
        // render our shortcut menu if its active
        self.shortcuts.render(frame);
    }
}
