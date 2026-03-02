//! The message log component for messages between the LLM and Thorium/MCP servers

use ratatui::{prelude::*, widgets::Paragraph};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::sync::Arc;
use thorium::ai::{AiSupport, ThorChatMsgKinds, UserFriendlyChatMsg};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
use uuid::Uuid;

use super::Theme;

/// Which side a bubble is displayed on.
#[derive(Debug, Clone, Copy)]
enum BubbleSide {
    /// User Bubble is displayed on the left
    User,
    /// AI Bubble is displayed on the right
    Ai,
}

impl From<&MessageBubbleContent> for BubbleSide {
    fn from(value: &MessageBubbleContent) -> Self {
        match value {
            MessageBubbleContent::User(_) => BubbleSide::User,
            MessageBubbleContent::ToolCall { .. } => BubbleSide::Ai,
            MessageBubbleContent::Response(_) => BubbleSide::Ai,
        }
    }
}

/// The content for this message bubble
#[derive(Debug, Clone)]
pub enum MessageBubbleContent {
    /// A message from a user
    User(String),
    /// A Tool call and any results
    ToolCall {
        /// The id for this tool call,
        id: Uuid,
        /// the name of the tool to call
        name: String,
        /// The arguments used for this tool
        arguments: Option<Map<String, Value>>,
        /// The results for this tool
        results: Vec<String>,
    },
    /// A response from an AI
    Response(String),
}

/// A single message bubble "component" that can render itself into `RenderRow`s.
#[derive(Debug, Clone)]
struct MessageBubble {
    /// The title content for this bubble
    title: Option<String>,
    /// The core content that is in this bubble
    content: MessageBubbleContent,
    /// The messages this bubble is built from
    backing: Vec<usize>,
    /// How much gap to between message bubbles and the side of the terminal.
    side_gap: u16,
    /// The maximum possible width of the chat bubble.
    bubble_max_width: u16,
    /// The maximum width for a chat bubble ratio wise.
    bubble_width_ratio: f32,
    /// The minimum bubble width (including borders).
    bubble_min_width: u16,
    /// Total padding inside the bubble interior (left + right), in terminal columns.
    bubble_inner_padding: u16,
    /// The style to use for the bubble interior.
    bubble_style: Style,
}

impl MessageBubble {
    /// Create a new bubble with the specified side.
    ///
    /// # Arguments
    ///
    /// * `side` - Which side the bubble is aligned to (left or right).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use thorctl::handlers::ai::chat::components::message_log::{MessageBubble, BubbleSide};
    /// use ratatui::style::Style;
    ///
    /// let bubble = MessageBubble::new(BubbleSide::Left, "<msg>", 0)
    ///     .side_gap(2)
    ///     .bubble_max_width(u16::MAX)
    ///     .bubble_width_ratio(0.85)
    ///     .bubble_min_width(10)
    ///     .bubble_inner_padding(2)
    ///     .bubble_style(Style::default())
    ///     .border_style(Style::default());
    /// ```
    pub fn new(content: MessageBubbleContent, backing: usize) -> Self {
        Self {
            title: None,
            content,
            backing: vec![backing],
            side_gap: 2,
            bubble_max_width: u16::MAX,
            bubble_width_ratio: 0.85,
            bubble_min_width: 10,
            bubble_inner_padding: 2,
            bubble_style: Style::default(),
        }
    }

    /// Add a title to this message bubble
    ///
    /// # Arguments
    ///
    /// * `title` - The title to add
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the style to use for the bubble interior.
    ///
    /// # Arguments
    ///
    /// * `bubble_style` - The style to apply to the bubble interior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use thorctl::handlers::ai::chat::components::message_log::{MessageBubble, BubbleSide};
    /// use ratatui::style::Style;
    ///
    /// let bubble = MessageBubble::new(BubbleSide::Left, "<msg>")
    ///     .style(Style::default().fg(ratatui::style::Color::Blue));
    /// ```
    #[must_use]
    pub fn style(mut self, bubble_style: Style) -> Self {
        self.bubble_style = bubble_style;
        self
    }

    /// Check if this bubble is for a tool call by id
    ///
    /// # Arguments
    ///
    /// * `id` - The id of the tool call bubble to look for
    pub fn is_tool_call(&self, id: Uuid) -> bool {
        // only tool call bubbles contain a tool call id
        match self.content {
            MessageBubbleContent::ToolCall { id: bubble_id, .. } => id == bubble_id,
            _ => false,
        }
    }

    /// Build a bubble for this message
    pub fn from_msg<A: AiSupport>(
        msg_kind: &ThorChatMsgKinds<A>,
        backing: usize,
        theme: &Arc<Theme>,
        bubbles: &mut Vec<MessageBubble>,
    ) -> Option<Self> {
        // get the info to build the bubble for this message kind
        match msg_kind {
            ThorChatMsgKinds::Msg(msg) => {
                // get the info for all of the possible message kinds
                match &msg.user_friendly {
                    UserFriendlyChatMsg::User(friendly) => {
                        // build this new user bubble
                        let bubble = MessageBubble::new(
                            MessageBubbleContent::User(friendly.clone()),
                            backing,
                        )
                        .style(theme.messages.user);
                        Some(bubble)
                    }
                    UserFriendlyChatMsg::Ai(friendly) => {
                        // build this new ai respones bubble
                        let bubble = MessageBubble::new(
                            MessageBubbleContent::Response(friendly.clone()),
                            backing,
                        )
                        .style(theme.messages.ai)
                        .title("🤖 AI");
                        Some(bubble)
                    }
                }
            }
            ThorChatMsgKinds::ToolCallRequest(tool_call) => {
                // build the content for this tool call message bubble
                let content = MessageBubbleContent::ToolCall {
                    id: tool_call.id,
                    name: tool_call.params.name.to_string(),
                    arguments: tool_call.params.arguments.clone(),
                    results: Vec::with_capacity(1),
                };
                // build this new tool call request bubble
                let bubble = MessageBubble::new(content, backing)
                    .style(theme.messages.ai)
                    .title("Tool");
                Some(bubble)
            }
            ThorChatMsgKinds::ToolCallResult(tool_result) => {
                // look for the bubble for our tool calls bubble
                let bubble = bubbles
                    .iter_mut()
                    .rev()
                    .find(|bubble| bubble.is_tool_call(tool_result.id))
                    .unwrap();
                // add this result to this tool call
                if let MessageBubbleContent::ToolCall { results, .. } = &mut bubble.content {
                    // add these results
                    results.push(tool_result.results.clone());
                }
                // we just edited an existing bubble so there is not new bubble to return
                None
            }
        }
    }

    /// Convert this message bubble to renderable unicode
    ///
    /// # Arguments
    ///
    /// * `width` - The max width to use for each line
    fn wrapped_unicode(&self, width: usize) -> Vec<String> {
        // build a single string to render
        match &self.content {
            MessageBubbleContent::User(user) => wrap_unicode_preserve_ws(user, width),
            MessageBubbleContent::ToolCall {
                name,
                arguments,
                results,
                ..
            } => {
                // build a nicely formatted string with all of the content in this bubble
                // TODO this is better but still ugly
                let nice = format!("Tool: {name}\nArgs: {arguments:#?}\n\nResults: {results:#?}");
                // get our nicely formatted tool call in wrapped unicode
                wrap_unicode_preserve_ws(&nice, width)
            }
            MessageBubbleContent::Response(ai) => wrap_unicode_preserve_ws(ai, width),
        }
    }

    /// Append this bubble's rows into `rows`.
    ///
    /// Sizing behavior:
    /// - wraps the message at a max width (`bubble_width_ratio` * available, capped by `bubble_max_width`)
    /// - then chooses a *tight* bubble width based on the longest wrapped line + padding
    /// - then clamps between `bubble_min_width` and the computed max width
    fn push_rows(
        &self,
        rows: &mut Vec<RenderRow>,
        title: &Option<String>,
        area: Rect,
        theme: &Arc<Theme>,
    ) {
        // Compute maximum bubble width allowed in this area (total width, including borders).
        let available = area.width.saturating_sub(self.side_gap * 2);
        let desired_max = ((available as f32) * self.bubble_width_ratio).round() as u16;
        let max_bubble_w = desired_max
            .min(available) // never exceed available
            .min(self.bubble_max_width)
            .max(self.bubble_min_width.max(4)); // keep sane minimum for borders
        // Max interior width (excluding borders).
        let max_inner_w = max_bubble_w.saturating_sub(2).max(1) as usize;
        // We wrap using a width that accounts for inner padding.
        let pad_total = self.bubble_inner_padding as usize;
        let wrap_w = max_inner_w.saturating_sub(pad_total).max(1);
        let wrapped = self.wrapped_unicode(wrap_w);
        // Compute the maximum display width (cells) among wrapped lines.
        let content_max_w: u16 = wrapped
            .iter()
            .map(|l| UnicodeWidthStr::width(l.as_str()) as u16)
            .max()
            .unwrap_or(0);
        // Tight interior width = content width + padding, clamped by max_inner_w.
        let mut inner_w = content_max_w.saturating_add(self.bubble_inner_padding);
        let inner_w_cap = max_bubble_w.saturating_sub(2).max(1);
        inner_w = inner_w.min(inner_w_cap);
        // Total bubble width = inner + borders, clamped by min/max.
        let mut bubble_w = inner_w.saturating_add(2);
        bubble_w = bubble_w.max(self.bubble_min_width).min(max_bubble_w);
        // Recompute interior width from final bubble_w so drawing stays consistent.
        let inner_w = bubble_w.saturating_sub(2).max(1) as usize;
        let bubble_x = match BubbleSide::from(&self.content) {
            BubbleSide::User => area.x + self.side_gap,
            BubbleSide::Ai => area.x + area.width.saturating_sub(self.side_gap + bubble_w),
        };
        // top border
        rows.push(RenderRow {
            x: bubble_x,
            line: top_border_line(title, bubble_w, theme.messages.border),
        });
        // content
        for ln in wrapped {
            rows.push(RenderRow {
                x: bubble_x,
                line: middle_line_padded(
                    inner_w,
                    &ln,
                    self.bubble_inner_padding as usize,
                    self.bubble_style,
                    theme.messages.border,
                ),
            });
        }
        // bottom border
        rows.push(RenderRow {
            x: bubble_x,
            line: bottom_border_line(bubble_w, theme.messages.border),
        });
    }
}

/// A message log of messages between the user and an AI
#[derive(Debug, Clone)]
pub struct MessageLog {
    /// The gap between messages vertically in rows
    vertical_gap: u16,
    /// The theme to use
    theme: Arc<Theme>,
    /// Scroll in terminal rows from the top (0 = top).
    scroll_y: usize,
    /// If true, view is pinned to bottom (latest content).
    follow: bool,
    /// The rendered bubbles
    bubbles: Vec<MessageBubble>,
    /// The index of the last rendered bubble
    last_bubbled: usize,
    /// The rendered lines
    rendered: Vec<RenderRow>,
}

impl Default for MessageLog {
    fn default() -> Self {
        Self {
            vertical_gap: 1,
            theme: Arc::new(Theme::default()),
            scroll_y: 0,
            follow: true,
            bubbles: Vec::with_capacity(100),
            last_bubbled: 0,
            rendered: Vec::with_capacity(5000),
        }
    }
}

impl MessageLog {
    /// Scroll up by N terminal rows.
    pub fn scroll_up(&mut self, rows: usize) {
        self.follow = false;
        self.scroll_y = self.scroll_y.saturating_sub(rows);
    }

    /// Scroll down by N terminal rows.
    /// You can call this without knowing max scroll; render() will clamp.
    pub fn scroll_down(&mut self, rows: usize) {
        self.scroll_y = self.scroll_y.saturating_add(rows);
        // follow will be re-enabled automatically when render() clamps at bottom
    }

    /// Page up (pass area.height as usize).
    pub fn page_up(&mut self, page_rows: usize) {
        self.scroll_up(page_rows);
    }

    /// Page down (pass area.height as usize).
    pub fn page_down(&mut self, page_rows: usize) {
        self.scroll_down(page_rows);
    }

    /// Jump to bottom and follow new content.
    pub fn end(&mut self) {
        self.follow = true;
    }

    /// Whether the view is currently pinned to bottom.
    pub fn is_following(&self) -> bool {
        self.follow
    }

    /// Clear all messages in our message log
    pub fn clear(&mut self) {
        // clear all bubbled messages
        self.bubbles.clear();
        // clear all rendered messages
        self.rendered.clear();
        // reset bubbled to 0
        self.last_bubbled = 0;
        // reset our vertical scroll to 0
        self.scroll_y = 0;
    }

    /// If the user is at the bottom of the chat window then make sure to keep new messaged visible
    ///
    /// # Arguments
    ///
    /// * `rows` - A list of of all rendered rows
    /// * `area` - The area on screen we are rendering content too
    fn follow_messages(&mut self, area: Rect) {
        // get the height of the are we can render content into
        let renderable_height = area.height as usize;
        // determine how far down we can scroll
        let max_scroll = self.rendered.len().saturating_sub(renderable_height);
        // check if we are following new messages
        if self.follow {
            // we are following so set our scroll to the max distance
            self.scroll_y = max_scroll;
        } else {
            // check if we want to start following new messages
            match self.scroll_y.cmp(&max_scroll) {
                // we are not at the bottom so don't do anything
                Ordering::Less => (),
                // we are at the bottom enable follow mode
                Ordering::Equal => self.follow = true,
                // we have scrolled past our max distance
                Ordering::Greater => {
                    // limit our scroll to the max distance
                    self.scroll_y = max_scroll;
                    // enable follow mode
                    self.follow = true
                }
            }
        }
    }

    pub fn render_bubbles<A: AiSupport>(
        &mut self,
        area: Rect,
        messages: &[ThorChatMsgKinds<A>],
        frame: &mut Frame,
    ) {
        // don't render anything if the window is tool small
        if area.width < 10 || area.height < 1 {
            return;
        }
        // iterate over our messages and render them
        for (index, msg_kind) in messages.iter().enumerate() {
            // build a message bubble for this chat message
            match MessageBubble::from_msg(msg_kind, index, &self.theme, &mut self.bubbles) {
                Some(new_bubble) => {
                    // render this bubbles content
                    // render this bubble and its its rendered rows to our reder list
                    new_bubble.push_rows(&mut self.rendered, &new_bubble.title, area, &self.theme);
                    // add a gap between each bubble
                    for _ in 0..self.vertical_gap {
                        self.rendered.push(RenderRow {
                            x: area.x,
                            line: Line::from(""),
                        });
                    }
                    // add this bubble to our rendered bubbles
                    self.bubbles.push(new_bubble);
                }
                None => {
                    // we updated an older ubble so just rerender everything
                    // TODO not have to rerender everything
                    // clear out all rendered lines
                    self.rendered.clear();
                    // rerender everything
                    for bubble in &self.bubbles {
                        // render the content in this bubble
                        bubble.push_rows(&mut self.rendered, &bubble.title, area, &self.theme);
                    }
                }
            }
        }
        // render new messages if we are following new messages
        self.follow_messages(area);
        // get the final row we can render
        let end = (self.scroll_y + area.height as usize).min(self.rendered.len());
        // render the visible rows
        for (row_idx, r) in self.rendered[self.scroll_y..end].iter().enumerate() {
            // calculate the y value for where this line of our message bubble will be rendered too
            let y = area.y + row_idx as u16;
            // build the area for where this line message bubble line will be rendered at
            let line_area = Rect {
                x: r.x,
                y,
                width: area.width.saturating_sub(r.x.saturating_sub(area.x)),
                height: 1,
            };
            // render this line
            frame.render_widget(Paragraph::new(r.line.clone()), line_area);
        }
    }

    /// Render the chat log onto a specific area in the terminal
    pub fn render<A: AiSupport>(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        messages: &[ThorChatMsgKinds<A>],
    ) {
        // get the slice of new messages to turn into message bubbles
        let unbubbled = &messages[self.last_bubbled..];
        // add any new messages to our bubble list
        self.render_bubbles(area, unbubbled, frame);
        // update our last bubbled message index
        self.last_bubbled = messages.len();
    }
}

/// A line ready to be rendered to the terminal
#[derive(Debug, Clone)]
struct RenderRow {
    /// The x coordinate to start rendering this line at
    x: u16,
    /// The text to render
    line: Line<'static>,
}

// ---------------- unicode-width-aware wrap/clip ----------------

/// Convert a string into multiple lines that automatically wrap text
///
/// # Arguments
///
/// * `input` - The string to line wrap
/// * `max_width` - The maxiumum allowable width for each line
fn wrap_unicode_preserve_ws(input: &str, max_width: usize) -> Vec<String> {
    // short circuit if our max width is 0
    if max_width == 0 {
        return vec![String::new()];
    }
    // preallocate a vec for our line wrapped strings
    let mut out = Vec::with_capacity(5);
    // iterate over our string split by new lines
    for split in input.split('\n') {
        // preallocate a new to add our unicode chars too
        let mut line = String::new();
        // start with a length of 0
        let mut line_width = 0usize;
        // iterate over the graphemes in this line
        for grapheme in split.graphemes(true) {
            // get width of this grapheme
            let grapheme_width = UnicodeWidthStr::width(grapheme);
            // check if this grapheme fits on our current line
            if line_width > 0 && line_width + grapheme_width > max_width {
                // get our old line
                let full_line = std::mem::take(&mut line);
                // add this full line to our output
                out.push(full_line);
                // reset our width to zero for our new line
                line_width = 0;
            }
            // add this grapheme to our current line
            line.push_str(grapheme);
            // increment our width
            line_width += grapheme_width;
        }
        // add any remaining lines
        out.push(line);
    }
    out
}

fn clip_unicode_to_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut w = 0usize;

    for g in s.graphemes(true) {
        let gw = UnicodeWidthStr::width(g);
        if w > 0 && w + gw > width {
            break;
        }
        if w == 0 && gw > width {
            out.push_str(g);
            break;
        }
        out.push_str(g);
        w += gw;
        if w == width {
            break;
        }
    }
    out
}

fn pad_to_width(mut s: String, target_w: usize) -> String {
    let w = UnicodeWidthStr::width(s.as_str());
    if w < target_w {
        s.push_str(&" ".repeat(target_w - w));
    }
    s
}

// ---------------- bubble lines (Unicode box drawing) ----------------

fn top_border_line(title: &Option<String>, w: u16, style: Style) -> Line<'static> {
    if w < 2 {
        return Line::from(Span::styled("┌", style));
    }
    // render our top bar with a title if we have one
    match title {
        Some(title) => {
            // get the length of our title
            let unicode_length = UnicodeWidthStr::width(title.as_str());
            let inner_end = (w - (6 + unicode_length as u16)) as usize;
            Line::from(vec![
                Span::styled("┌", style),
                Span::styled("─".repeat(4), style),
                Span::styled(title.clone(), style),
                Span::styled("─".repeat(inner_end), style),
                Span::styled("┐", style),
            ])
        }
        None => {
            let inner = (w - 2) as usize;
            Line::from(vec![
                Span::styled("┌", style),
                Span::styled("─".repeat(inner), style),
                Span::styled("┐", style),
            ])
        }
    }
}

fn bottom_border_line(w: u16, style: Style) -> Line<'static> {
    if w < 2 {
        return Line::from(Span::styled("└", style));
    }
    let inner = (w - 2) as usize;
    Line::from(vec![
        Span::styled("└", style),
        Span::styled("─".repeat(inner), style),
        Span::styled("┘", style),
    ])
}

/// Draw a middle line for a bubble with padding and full background fill.
///
/// `inner_w` is the interior width (excluding borders) in terminal columns.
fn middle_line_padded(
    inner_w: usize,
    content: &str,
    inner_padding_total: usize,
    bubble_style: Style,
    border_style: Style,
) -> Line<'static> {
    // Split padding roughly evenly.
    let pad_left = inner_padding_total / 2;
    let pad_right = inner_padding_total - pad_left;

    // How much space is left for text after padding?
    let usable = inner_w.saturating_sub(pad_left + pad_right).max(1);

    let clipped = clip_unicode_to_width(content, usable);
    let clipped_w = UnicodeWidthStr::width(clipped.as_str());

    let mut mid = String::new();
    mid.push_str(&" ".repeat(pad_left));
    mid.push_str(&clipped);

    if clipped_w < usable {
        mid.push_str(&" ".repeat(usable - clipped_w));
    }

    mid.push_str(&" ".repeat(pad_right));

    // Ensure exact width so the bubble background fills consistently.
    let mid = pad_to_width(clip_unicode_to_width(&mid, inner_w), inner_w);

    Line::from(vec![
        Span::styled("│", border_style),
        Span::styled(mid, bubble_style),
        Span::styled("│", border_style),
    ])
}
