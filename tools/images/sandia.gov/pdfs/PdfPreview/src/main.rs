//! This example shows you how you can render a PDF file to PNG.

use hayro::{Pdf, RenderSettings, render};
use hayro_interpret::InterpreterSettings;
use std::sync::Arc;

fn main() {
    let file = std::fs::read(std::env::args().nth(1).unwrap()).unwrap();
    let output_dir = "/tmp/thorium/result-files";
    let scale = std::env::args()
        .nth(3)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir).unwrap();

    let data = Arc::new(file);
    let pdf = Pdf::new(data).unwrap();

    let interpreter_settings = InterpreterSettings::default();

    let render_settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        ..Default::default()
    };

    for (idx, page) in pdf.pages().iter().enumerate() {
        let pixmap = render(page, &interpreter_settings, &render_settings);
        let output_path = format!("{output_dir}/rendered_{idx}.png");
        std::fs::write(output_path, pixmap.take_png()).unwrap();
    }
}
