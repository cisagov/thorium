//! The text box component for a thorctl chat tui

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

#[derive(Debug, Clone)]
pub struct TextBox {
    /// The title for this text box
    pub title: Option<String>,
    /// The content of this text box
    pub content: String,
    /// The placeholder text for this text box
    pub placeholder: String,
    /// The cursors x axis position within this text box
    pub cursor_x: usize,
    /// The cursors y axis position within this text box
    pub cursor_y: usize,
    /// The current position of the cusor regardless of new lines/wrapping
    pub cursor_absolute: usize,
    /// Whether the text box is focused or not
    pub is_focused: bool,
    /// Vertical scroll offset for the text box
    pub scroll_y: u16,
    /// The area for this text box on screen
    pub area: Option<Rect>,
}

impl Default for TextBox {
    /// Create an empty non focused text box
    fn default() -> Self {
        TextBox {
            title: None,
            content: String::default(),
            placeholder: String::default(),
            cursor_x: 0,
            cursor_y: 0,
            cursor_absolute: 0,
            is_focused: false,
            scroll_y: 0,
            area: None,
        }
    }
}

impl TextBox {
    /// Create a new home page
    ///
    /// # Arguments
    ///
    /// * `content` - the content to set for this home page
    pub fn new(content: impl Into<String>) -> Self {
        TextBox {
            title: None,
            content: content.into(),
            placeholder: String::default(),
            cursor_x: 0,
            cursor_y: 0,
            cursor_absolute: 0,
            is_focused: true,
            scroll_y: 0,
            area: None,
        }
    }

    /// Set the title for this text box
    ///
    /// # Arguments
    ///
    /// * `title` - The new title to set
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the placeholder text for this text box
    ///
    /// # Arguments
    ///
    /// * `placeholder` - the placeholder to set for this home page
    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        // insert our placeholder text
        self.placeholder = placeholder.into();
        self
    }

    /// Set this text box to be focused
    pub fn focused(mut self) -> Self {
        self.is_focused = true;
        self
    }

    /// Clear all content in this text box
    pub fn clear(&mut self) {
        // clear the content in this text box
        self.content.clear();
        // reset this text box no matter what
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.cursor_absolute = 0;
    }

    /// Insert a character at the current cursor position
    ///
    /// # Arguments
    ///
    /// * `c` - The character to insert
    fn insert_char(&mut self, c: char) {
        // insert this character at the current cursor position
        self.content.insert(self.cursor_absolute, c);
        // increment our cursor
        self.cursor_x += 1;
        // increment our absolute cursor value as well
        self.cursor_absolute += 1;
    }

    /// Insert a character at the current cursor position
    fn insert_newline(&mut self) {
        // insert this character at the current cursor position
        self.content.insert(self.cursor_absolute, '\n');
        // increment our cursor
        self.cursor_x = 0;
        self.cursor_y += 1;
        self.cursor_absolute += 1;
    }

    /// Delete the character before the cursor (backspace)
    fn delete_char_before(&mut self) {
        // delete this character and update our cursors correctly
        match (self.cursor_absolute > 0, self.cursor_x) {
            (true, 0) if self.cursor_y > 0 => {
                // move our y value up a line
                self.cursor_y = self.cursor_y.saturating_sub(1);
                // decrement our absolue cursor value
                self.cursor_absolute = self.cursor_absolute.saturating_sub(1);
                // remove a character
                self.content.remove(self.cursor_absolute);
                // calculate the number of chars to skip
                let (skip, was_end) = if self.cursor_absolute == self.content.len() {
                    // this is the end of the content so we don't need to subtract 1
                    // to allow for the trailing empty whitespace
                    let skip = self.content.len() - self.cursor_absolute;
                    // mark that this was at the end of the string
                    (skip, true)
                } else {
                    // this is not the end of the content so we need to subtract 1 since
                    // we are removing the new line whitespace
                    let skip = self.content.len() - self.cursor_absolute - 1;
                    // mark that this was not at the end of the string
                    (skip, false)
                };
                // get the new x value for our cursor
                self.cursor_x = self
                    .content
                    .chars()
                    .rev()
                    // skip the character our current cursor was on
                    .skip(skip)
                    .take_while(|c| c != &'\n')
                    .count();
                // if this was not at the end then we need to decrement our x value to
                // remove the new line char
                if !was_end {
                    self.cursor_x = self.cursor_x.saturating_sub(1);
                }
            }
            (true, _) => {
                // decrement our x cursor by one
                self.cursor_x = self.cursor_x.saturating_sub(1);
                // decrement our absolue cursor value
                self.cursor_absolute = self.cursor_absolute.saturating_sub(1);
                // remove a character
                self.content.remove(self.cursor_absolute);
            }
            (false, _) => (),
        }
    }

    /// Delete the character at the cursor (delete)
    fn delete_char_at(&mut self) {
        // delete this character and update our cursors correctly
        match (self.cursor_absolute < self.content.len(), self.cursor_x) {
            (true, 0) if self.cursor_y > 0 => {
                // move our y value up a line
                self.cursor_y = self.cursor_y.saturating_sub(1);
                // calculate the number of chars to skip
                let (skip, was_end) = if self.cursor_absolute == self.content.len() {
                    // this is the end of the content so we don't need to subtract 1
                    // to allow for the trailing empty whitespace
                    let skip = self.content.len() - self.cursor_absolute;
                    // mark that this was at the end of the string
                    (skip, true)
                } else {
                    // this is not the end of the content so we need to subtract 1 since
                    // we are removing the new line whitespace
                    let skip = self.content.len() - self.cursor_absolute - 1;
                    // mark that this was not at the end of the string
                    (skip, false)
                };
                // get the new x value for our cursor
                self.cursor_x = self
                    .content
                    .chars()
                    .rev()
                    // skip the character our current cursor was on
                    .skip(skip)
                    .take_while(|c| c != &'\n')
                    .count();
                // if this was not at the end then we need to decrement our x value to
                // remove the new line char
                if !was_end {
                    self.cursor_x = self.cursor_x.saturating_sub(1);
                }
                // remove a character
                self.content.remove(self.cursor_absolute);
                // decrement our absolue cursor value
                self.cursor_absolute = self.cursor_absolute.saturating_sub(1);
            }
            (true, _) => {
                // decrement our x cursor by one
                self.cursor_x = self.cursor_x.saturating_sub(1);
                // decrement our y cursor by one
                self.cursor_y = self.cursor_y.saturating_sub(1);
                // remove a character
                self.content.remove(self.cursor_absolute);
                // decrement our absolue cursor value
                self.cursor_absolute = self.cursor_absolute.saturating_sub(1);
            }
            (false, _) => (),
        }
    }

    /// Move the cursor left
    fn move_cursor_left(&mut self) {
        // we can only move our cursor if we are not at the start
        if self.cursor_x > 0 {
            // move the cursor left
            self.cursor_x -= 1;
            // move our absolute cursor as well
            self.cursor_absolute -= 1;
        }
    }

    /// Move the cursor right
    fn move_cursor_right(&mut self) {
        // we can't move our cursor past the end of our content or past new lines
        if self.cursor_absolute < self.content.len()
            && self.content.chars().nth(self.cursor_absolute) != Some('\n')
        {
            // move the cursor right
            self.cursor_x += 1;
            // move our absolute cursor as well
            self.cursor_absolute += 1;
        }
    }

    /// Move our cursor to the start of the text box
    fn move_cursor_to_start(&mut self) {
        // reset all of our cursor values to 0
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.cursor_absolute = 0;
    }

    /// Move our cursor to the end of this line
    fn move_cursor_to_end_of_line(&mut self) {
        // find how many characters to get to the end of the current line
        let to_end = self
            .content
            .chars()
            .skip(self.cursor_absolute)
            .take_while(|c| c != &'\n')
            .count();
        // increment our cursor values
        self.cursor_x += to_end;
        self.cursor_absolute += to_end;
    }

    /// Handle key input for this text box
    ///
    /// # Arguments
    ///
    /// * `code` - The key code that was pressed
    pub fn handle_key(&mut self, key: KeyEvent, allow_empty: bool) -> bool {
        //// reset our content if we have placeholder text
        //self.clear_placeholder();
        // handle this key event
        match key.code {
            // Delete character before cursor
            KeyCode::Backspace => self.delete_char_before(),
            // Delete character at cursor
            KeyCode::Delete => self.delete_char_at(),
            // Move cursor left
            KeyCode::Left => self.move_cursor_left(),
            // Move cursor right
            KeyCode::Right => self.move_cursor_right(),
            // Move cursor to start
            KeyCode::Home => self.move_cursor_to_start(),
            // Move cursor to end
            KeyCode::End => self.move_cursor_to_end_of_line(),
            // Add a new line to this text box
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_newline();
            }
            // Type characters
            KeyCode::Char(c) => self.insert_char(c),
            // return that a user is finished with this text box
            KeyCode::Enter => {
                // if we don't want to allow empty text boxes then make sure content
                // is not empty
                if allow_empty {
                    return true;
                } else {
                    return !self.content.is_empty();
                }
            }
            // unfocus on this prompt
            KeyCode::Esc => self.is_focused = false,
            _ => {}
        }
        // all key codes other then enter are not stating this text box is finished
        false
    }

    /// Handle a mouse click event
    ///
    /// If this method returns true then this click was in this text boxes area.
    ///
    /// # Arguments
    ///
    /// * `x` - The column position of the click
    /// * `y` - The row position of the click
    pub fn handle_click(&mut self, x: u16, y: u16) -> bool {
        // we should always have an area as long as this text box has been rendered
        if let Some(area) = self.area {
            // Check if the click was on this text box
            if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height {
                self.is_focused = true;
                return true;
            }
        }
        // Click was outside the query box, unfocus it
        self.is_focused = false;
        false
    }

    /// Consume this text box and replace with some placeholder text
    ///
    /// If this text box contains no text then None will be returned.
    pub fn consume(&mut self) -> Option<String> {
        // consume and replace our old text
        let consumed = std::mem::take(&mut self.content);
        // reset this text box no matter what
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.cursor_absolute = 0;
        // only return our consumed text if it wasn't empty
        if consumed.is_empty() {
            None
        } else {
            Some(consumed)
        }
    }

    /// Render the text box to the frame
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame to render to
    /// * `area` - The area to render the input in
    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        // get the right content to display
        let content = if self.content.is_empty() {
            self.placeholder.clone()
        } else {
            self.content.clone()
        };
        // if this is placeholder text then make the text grey
        let style = if self.content.is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        };
        // the block to render in this input widgeth
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));
        // if we have a title then also set that
        let block = match &self.title {
            // set our title in this block
            Some(title) => block.title(format!(" {title} ")).title_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            None => block,
        };
        // build the input widget
        let input = Paragraph::new(content)
            .style(style)
            .block(block)
            .wrap(Wrap { trim: false });
        // render the input
        //frame.render_widget(&self.textarea, area);
        frame.render_widget(input, area);
        // show cursor when focused
        if self.is_focused {
            // calculate cursor position within the input area
            // account for the border (1 char) on the left
            let cursor_x = area.x + 1 + self.cursor_x as u16;
            let cursor_y = area.y + 1 + self.cursor_y as u16;
            // only show cursor if it's within the visible area
            if cursor_x < area.x + area.width - 1 {
                frame.set_cursor_position((cursor_x, cursor_y));
            }
        }
        // update this clickable area for this text box
        self.area = Some(area);
    }
}
