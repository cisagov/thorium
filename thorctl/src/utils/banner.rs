//! Utilities for printing the Thorium banner

use thorium::Error;
use tui_banner::{Align, Banner, Fill, Gradient, Palette};

pub fn wave(text: &str) -> Result<(), Error> {
    let banner = Banner::new(text)
        .unwrap()
        .gradient(Gradient::diagonal(Palette::from_hex(&[
            "#00E5FF", "#7B5CFF", "#FF5AD9",
        ])))
        .fill(Fill::Keep)
        .align(Align::Center)
        .padding(1);

    banner.animate_wave(1, None, None).unwrap();
    Ok(())
}
