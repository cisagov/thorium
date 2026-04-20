//! Helpers for rendering markdown and styled fields for `describe` output
//!
//! Styling is gated behind an `ansi` flag so the same renderers can produce
//! colored output for an interactive terminal and plain text for files or
//! pipes.

use std::fmt::Write as _;

use crossterm::style::{Attribute, Color as CtColor, ContentStyle};
use owo_colors::OwoColorize;
use ratatui::style::{Color as RtColor, Modifier, Style};

/// Convert a ratatui color into the closest crossterm color
///
/// ratatui's base colors map to the dim ANSI variants and its `Light*` colors
/// to the bright variants, matching how the terminal renders them.
///
/// # Arguments
///
/// * `color` - The ratatui color to convert
fn convert_color(color: RtColor) -> Option<CtColor> {
    let converted = match color {
        // `Reset` carries no color; let the terminal default stand
        RtColor::Reset => return None,
        RtColor::Black => CtColor::Black,
        RtColor::Red => CtColor::DarkRed,
        RtColor::Green => CtColor::DarkGreen,
        RtColor::Yellow => CtColor::DarkYellow,
        RtColor::Blue => CtColor::DarkBlue,
        RtColor::Magenta => CtColor::DarkMagenta,
        RtColor::Cyan => CtColor::DarkCyan,
        RtColor::Gray => CtColor::Grey,
        RtColor::DarkGray => CtColor::DarkGrey,
        RtColor::LightRed => CtColor::Red,
        RtColor::LightGreen => CtColor::Green,
        RtColor::LightYellow => CtColor::Yellow,
        RtColor::LightBlue => CtColor::Blue,
        RtColor::LightMagenta => CtColor::Magenta,
        RtColor::LightCyan => CtColor::Cyan,
        RtColor::White => CtColor::White,
        RtColor::Rgb(r, g, b) => CtColor::Rgb { r, g, b },
        RtColor::Indexed(i) => CtColor::AnsiValue(i),
    };
    Some(converted)
}

/// Convert a ratatui style into a crossterm content style
///
/// # Arguments
///
/// * `style` - The ratatui style to convert
fn convert_style(style: Style) -> ContentStyle {
    let mut content_style = ContentStyle::new();
    content_style.foreground_color = style.fg.and_then(convert_color);
    content_style.background_color = style.bg.and_then(convert_color);
    // map each modifier flag we care about to its crossterm attribute
    let modifiers = [
        (Modifier::BOLD, Attribute::Bold),
        (Modifier::DIM, Attribute::Dim),
        (Modifier::ITALIC, Attribute::Italic),
        (Modifier::UNDERLINED, Attribute::Underlined),
        (Modifier::REVERSED, Attribute::Reverse),
        (Modifier::CROSSED_OUT, Attribute::CrossedOut),
        (Modifier::SLOW_BLINK, Attribute::SlowBlink),
        (Modifier::RAPID_BLINK, Attribute::RapidBlink),
        (Modifier::HIDDEN, Attribute::Hidden),
    ];
    for (modifier, attribute) in modifiers {
        if style.add_modifier.contains(modifier) {
            content_style.attributes.set(attribute);
        }
    }
    content_style
}

/// Render markdown into a string for terminal display
///
/// With `ansi` set, the markdown is parsed and converted to ANSI-styled text
/// (headers, emphasis, syntax-highlighted code blocks). Without it, the raw
/// markdown is returned unchanged so files and pipes stay clean.
///
/// # Arguments
///
/// * `markdown` - The raw markdown to render
/// * `ansi` - Whether to emit ANSI styling (false produces plain text)
pub fn render_markdown(markdown: &str, ansi: bool) -> String {
    // without styling, hand back the raw markdown untouched
    if !ansi {
        return markdown.to_string();
    }
    // parse the markdown into styled ratatui text
    let text = tui_markdown::from_str(markdown);
    let mut out = String::new();
    for (index, line) in text.lines.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        for span in &line.spans {
            // merge the line-level style (headings, etc.) with the span's own style
            let style = line.style.patch(span.style);
            // `StyledContent`'s `Display` writes the set/reset ANSI sequences
            let styled = convert_style(style).apply(span.content.as_ref());
            // writing into a String is infallible
            let _ = write!(out, "{styled}");
        }
    }
    out
}

/// Format an entity header line (name plus its group)
///
/// # Arguments
///
/// * `name` - The entity's name
/// * `group` - The group the entity belongs to
/// * `ansi` - Whether to style the output
pub fn header(name: &str, group: &str, ansi: bool) -> String {
    if ansi {
        format!(
            "{} {}",
            name.bold().bright_white(),
            format!("({group})").dimmed()
        )
    } else {
        format!("{name} ({group})")
    }
}

/// Format a standalone field label, e.g. for a multi-line value rendered below it
///
/// # Arguments
///
/// * `text` - The label text
/// * `ansi` - Whether to style the label
pub fn label(text: &str, ansi: bool) -> String {
    if ansi {
        format!("{}:", text).bold().cyan().to_string()
    } else {
        format!("{text}:")
    }
}

/// Format a labeled field line
///
/// # Arguments
///
/// * `name` - The field's label
/// * `value` - The field's value
/// * `ansi` - Whether to style the label
pub fn field(name: &str, value: &str, ansi: bool) -> String {
    format!("{} {value}", label(name, ansi))
}

/// Render a one-line description preview for list/table output
///
/// Collapses newlines to spaces so a multi-line description never breaks the
/// table layout, and truncates to `max` characters (replacing the tail with an
/// ellipsis) so the column stays aligned. A missing description renders as `-`.
///
/// # Arguments
///
/// * `description` - The full description, if any
/// * `max` - The maximum rendered width, including the ellipsis
pub fn truncate_description(description: Option<&str>, max: usize) -> String {
    match description {
        // no description, so stand in with a dash
        None => "-".to_string(),
        Some(descr) => {
            // reserve room for the ellipsis we append when truncating
            let keep = max.saturating_sub(3);
            let rendered = if descr.chars().count() > keep {
                descr
                    .chars()
                    .take(keep)
                    .chain("...".chars())
                    .collect::<String>()
            } else {
                descr.to_string()
            };
            // newlines would break the single-line layout
            rendered.replace('\n', " ")
        }
    }
}

/// Format a separator line shown between multiple described entities
///
/// # Arguments
///
/// * `ansi` - Whether to style the separator
pub fn separator(ansi: bool) -> String {
    let rule = "─".repeat(60);
    if ansi {
        rule.dimmed().to_string()
    } else {
        rule
    }
}

#[cfg(test)]
mod tests {
    use super::render_markdown;

    /// Without ansi, `render_markdown` returns the raw markdown unchanged
    #[test]
    fn plain_passthrough() {
        let md = "# Title\n\nsome **bold** text";
        // without ansi the raw markdown is returned untouched
        assert_eq!(render_markdown(md, false), md);
    }

    /// With ansi, a rendered heading carries ANSI escape codes around its text
    #[test]
    fn ansi_emits_escape_codes() {
        let rendered = render_markdown("# Title", true);
        // a heading should be styled, so escape codes must be present
        assert!(rendered.contains('\u{1b}'));
        assert!(rendered.contains("Title"));
    }

    /// With ansi, a fenced code block with a known language is syntax highlighted
    #[test]
    fn ansi_highlights_code_block() {
        // a fenced block with a known language token is syntax highlighted
        let rendered = render_markdown("```rust\nlet x = 1;\n```", true);
        assert!(rendered.contains('\u{1b}'));
        assert!(rendered.contains('x'));
    }
}
