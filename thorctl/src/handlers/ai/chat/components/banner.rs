//! A banner component

use ansi_to_tui::IntoText as _;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::widgets::{Block, Paragraph};
use tui_banner::{AnimatedWaveBanner, Fill, Gradient, Palette};

/// a banner component
pub struct Banner {
    /// The animated text to display in this banner
    animated: AnimatedWaveBanner,
}

impl Banner {
    /// Create a new banner
    ///
    /// # Arguments
    ///
    /// * `text` - The text to display in this banner
    pub fn new(text: impl Into<String>) -> Self {
        // build our base animated banner
        let animated = tui_banner::Banner::new(text.into())
            .unwrap()
            // A blue -> purple -> pink gradient
            .gradient(Gradient::diagonal(Palette::from_hex(&[
                "#00E5FF", "#7B5CFF", "#FF5AD9",
            ])))
            .fill(Fill::Keep)
            .manually_driven_wave(None, None);
        Banner { animated }
    }

    /// Render this banner to the frame
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to render to
    /// * `area` - The area to render the input in
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        // get the next frame of our banner
        let colored = self.animated.step().unwrap().into_text().unwrap();
        // build the input widget
        let input = Paragraph::new(colored).block(Block::default()).centered();
        // render the input
        frame.render_widget(input, area);
    }
}
