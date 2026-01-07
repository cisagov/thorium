//! Chat with an AI in thorctl

use dialoguer::Input;
use dialoguer::theme::ColorfulTheme;
use owo_colors::OwoColorize;
use owo_colors::colors::*;
use thorium::Error;

use super::{AiSupport, ThorChat};
use crate::args::ai::Chat;
use crate::utils::banner;

pub async fn handle<A: AiSupport>(thorchat: &mut ThorChat<A>, cmd: &Chat) -> Result<(), Error> {
    // Display the thorium banner before we can chat
    banner::wave("Thorium")?;
    println!("Tips for getting started:");
    println!("1. Be specific about the sha256 or data you are interested in");
    println!("2. Give examples for how you want output to be organized");
    println!("\n");
    // enable debug mode if needed
    thorchat.ai.debug_mode(cmd.debug);
    // ask our ai to summarize this hash and print its response
    loop {
        // have a string to write our input to the user into
        let input: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt(">")
            .interact_text()
            .unwrap();
        // send this users question/msg to the AI
        match thorchat.ask(input).await {
            Ok(Some(response)) => println!("{}", response.bright_blue()),
            Ok(None) => {
                // color our response as blue
                let msg = "AI returned no response? ¯\\_(ツ)_/¯".bright_blue();
                // print our colored message
                println!("{msg}");
            }
            Err(error) => {
                // color our response as red
                let msg = format!(
                    "Uhoh! Thorctl chat ran into a problem: {}: {:?}",
                    error.kind(),
                    error.msg()
                );
                // print our colored message
                println!("{}", msg.bright_red());
            }
        }
    }
}
