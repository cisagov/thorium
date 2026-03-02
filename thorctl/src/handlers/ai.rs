//! Handle the different ai commands for Thorctl

use futures::stream::{self, StreamExt};
use indicatif::ProgressStyle;
use owo_colors::OwoColorize;
use rmcp::model::{
    CallToolRequestParam, CallToolResult, ClientCapabilities, ClientInfo, Implementation,
    InitializeRequestParam, ProtocolVersion,
};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use thorium::ai::ThorChat;
use thorium::ai::services::OpenAI;
use thorium::{CtlConf, Error};

mod chat;
mod summary;

use crate::args::Args;
use crate::args::ai::Ai;
use crate::handlers::progress::{Bar, BarKind, MultiBar};

/// handle all ai commands for Thorctl
pub async fn handle(args: &Args, cmd: &Ai) -> Result<(), Error> {
    // get our thorctl config
    let conf = CtlConf::from_path(&args.config)?;
    // get a thorium chat object
    let mut thorchat = ThorChat::<OpenAI>::new(&conf).await?;
    // handle the correct command
    match cmd {
        // Ai::Chat(chat) => chat::handle(&mut thorchat, chat).await,
        Ai::Chat(chat) => chat::tui(thorchat, chat).await,
        Ai::Summary(summary) => summary::handle(&mut thorchat, summary).await,
    }
}
