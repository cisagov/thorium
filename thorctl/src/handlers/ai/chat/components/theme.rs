//! The Themes used by thorctl AI chat
use ratatui::prelude::*;

/// The theme used for message bubbles
#[derive(Debug)]
pub struct MessageTheme {
    /// The style for user message bubbles
    pub user: Style,
    /// The style for ai message bubbles
    pub ai: Style,
    /// The style for borders
    pub border: Style,
}

/// Set a default theme
impl Default for MessageTheme {
    /// Set a default theme
    fn default() -> Self {
        // build the default style for user message bubbles
        let user = Style::default()
            .fg(Color::Black)
            // set the background to a light blue
            .bg(Color::Rgb(210, 230, 255));
        // build the default style for ai message bubbles
        let ai = Style::default()
            .fg(Color::Black)
            // set the background to a light green
            .bg(Color::Rgb(210, 255, 210));
        let border = Style::default().fg(Color::Gray);
        // Build the message theme
        MessageTheme { user, ai, border }
    }
}

/// A theme for thorctl AI chat
#[derive(Debug)]
pub struct Theme {
    /// The styles for message bubbles
    pub messages: MessageTheme,
}

/// Set a default theme
impl Default for Theme {
    fn default() -> Self {
        Theme {
            messages: MessageTheme::default(),
        }
    }
}
