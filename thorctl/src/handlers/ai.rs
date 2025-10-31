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
use thorium::{CtlConf, Error};

mod services;
mod summary;

use crate::args::Args;
use crate::args::ai::Ai;
use crate::handlers::progress::{Bar, BarKind, MultiBar};
use services::{AiResponse, AiSupport, OpenAI};

/// Setup an mcp client
pub async fn setup_mcp(
    conf: &CtlConf,
) -> Result<RunningService<RoleClient, InitializeRequestParam>, Error> {
    // build the url to thorium's mcp routes
    let mcp_uri = format!("{}/api/mcp", conf.keys.api);
    // make sure we have a token
    let token = match &conf.keys.token {
        Some(token) => token,
        // we can't use password auth with mcp so throw an error
        None => return Err(Error::new("Please run thorctl login first!")),
    };
    // build the config to use with this transport
    let config = StreamableHttpClientTransportConfig::with_uri(mcp_uri).auth_header(token);
    // setup our transport
    let transport = StreamableHttpClientTransport::from_config(config);
    // build our client
    let client_info = ClientInfo {
        protocol_version: ProtocolVersion::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "Thorium".to_owned(),
            title: Some("Thorium".to_owned()),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            icons: None,
            website_url: Some(conf.keys.api.clone()),
        },
    };
    // build our client
    let mcp_client = client_info.serve(transport).await.unwrap();
    Ok(mcp_client)
}

/// Set the progress bar style for this controllers monitor
fn ai_bar_style() -> Result<ProgressStyle, Error> {
    // build the style string for our monitor
    let style_str = "{spinner:.green} {elapsed_precise} AI: {msg}";
    // build the style for our progress bar
    let bar_style = ProgressStyle::with_template(&style_str)
        .unwrap()
        .tick_strings(&[
            "🦀🌽     🧠",
            " 🦀🌽    🧠",
            "  🦀🌽   🧠",
            "   🦀🌽  🧠",
            "    🦀🌽 🧠",
            "     🦀🌽🧠",
            "       🦀🧠",
            "      🦀 🧠",
            "     🦀  🧠",
            "    🦀   🧠",
            "   🦀    🧠",
            "  🦀     🧠",
            " 🦀      🧠",
            "🦀       🧠",
        ]);
    Ok(bar_style)
}

/// Set the progress bar style for this controllers monitor
fn mcp_bar_style() -> Result<ProgressStyle, Error> {
    // build the style string for our monitor
    let style_str = "{spinner:.green} {elapsed_precise} MCP: {msg}";
    // build the style for our progress bar
    let bar_style = ProgressStyle::with_template(&style_str)
        .unwrap()
        .tick_strings(&[
            "🦀🚜🌽🌽🌽🌽",
            " 🦀🚜🌽🌽🌽",
            "  🦀🚜🌽🌽",
            "   🦀🚜🌽",
            "     🦀🚜",
            "    🦀🚜🌽",
            "  🦀🚜🌽🌽",
            " 🦀🚜🌽🌽🌽",
            "🦀🚜🌽🌽🌽🌽",
        ]);
    Ok(bar_style)
}

/// Help ThorChat call some mcp based tool
///
/// # Arguments
///
/// * `mcp` - The mcp server to talk too
/// * `params` - The params to use when calling this tool
async fn call_tool_helper(
    mcp: &RunningService<RoleClient, InitializeRequestParam>,
    progress: &MultiBar,
    params: CallToolRequestParam,
) -> (String, CallToolResult) {
    // get this tools name
    let name = params.name.to_string();
    // add a progress bar and log what tools we are calling
    let bar = progress.add("", BarKind::Unbound);
    bar.bar.set_style(mcp_bar_style().unwrap());
    bar.bar
        .enable_steady_tick(std::time::Duration::from_millis(120));
    bar.set_message(format!("Calling Tool: {name}"));
    // call this mcp tool
    let call_resp = mcp.call_tool(params).await.unwrap();
    // Log that we got results from this tool
    bar.info(format!("Retrieved Results from {name}"));
    bar.finish_and_clear();
    (name, call_resp)
}

/// A Thorium AI chat bot
pub struct ThorChat<A: AiSupport> {
    /// The AI client to use for reasoning
    pub ai: A,
    /// The progress bars to use to log progress
    pub progress: MultiBar,
    /// The MCP client to use
    pub mcp: RunningService<RoleClient, InitializeRequestParam>,
}

impl<A: AiSupport> ThorChat<A> {
    /// Create a new chat bot
    pub async fn new(args: &Args) -> Result<Self, Error> {
        // get our thorctl config
        let conf = CtlConf::from_path(&args.config)?;
        // get an ai and assistant
        let ai = A::setup(&conf).await?;
        // setup mcp
        let mcp = setup_mcp(&conf).await?;
        // create a default progress bar
        let progress = MultiBar::default();
        // create a thorchat object
        let mut thorchat = ThorChat { ai, progress, mcp };
        // init our tools
        thorchat.load_tools().await?;
        Ok(thorchat)
    }

    /// Tell our AI about our tools
    ///
    /// # Arguments
    ///
    /// * `mcp_tools` - The mcp tools to tell our ai about
    async fn load_tools(&mut self) -> Result<(), Error> {
        // List tools
        let mcp_tools = self.mcp.list_tools(None).await?;
        // tell our ai about our tools
        self.ai.load_tools(mcp_tools)
    }

    /// Disable a specific tool
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the mcp tool to disable
    #[expect(dead_code)]
    fn disable_tool<T: Into<String>>(&mut self, tool: T) {
        self.ai.disable_tool(tool);
    }

    /// Enable a specific tool
    ///
    /// If you enable any tools then only enabled tools can be run
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the mcp tool to enable
    fn enable_tool<T: Into<String>>(&mut self, tool: T) {
        self.ai.enable_tool(tool);
    }

    /// Add the result from a tool
    ///
    /// # Arguments
    ///
    /// * `tool_results` - The tool results to tell our AI about
    async fn call_tools(
        &self,
        tool_calls: Vec<CallToolRequestParam>,
    ) -> Vec<(String, CallToolResult)> {
        // call our tools in parallel
        stream::iter(tool_calls)
            .map(|params| async move { call_tool_helper(&self.mcp, &self.progress, params).await })
            .buffered(10)
            .collect::<Vec<(String, CallToolResult)>>()
            .await
    }

    /// Wrap calls to an to handle progress bars
    async fn wrap_ask<T: Into<String>>(&mut self, question: T) -> Result<AiResponse, Error> {
        // start a progress bar for asking the ai for the next step
        let bar = self.progress.add("", BarKind::Unbound);
        bar.set_message("Asking Question");
        // set the bar style for ai questions
        bar.bar.set_style(ai_bar_style()?);
        // set a steady tick for this bar
        bar.bar
            .enable_steady_tick(std::time::Duration::from_millis(120));
        // ask our ai and don't check for an error until we shutdown our progress bar
        let response_result = self.ai.ask(question).await;
        // stop our progress bar
        bar.finish_and_clear();
        response_result
    }

    /// Wrap telling the AI about our tool results to handle progress bars
    async fn wrap_add_tool_results(
        &mut self,
        tool_results: Vec<(String, CallToolResult)>,
    ) -> Result<AiResponse, Error> {
        // start a progress bar for asking the ai for the next step
        let bar = self.progress.add("", BarKind::Unbound);
        // set the message for our bar
        bar.set_message("Telling AI about tool results");
        // set the bar style for ai questions
        bar.bar.set_style(ai_bar_style()?);
        // set a steady tick for this bar
        bar.bar
            .enable_steady_tick(std::time::Duration::from_millis(120));
        // tell our ai about our results without checking for an error until we shutdown our progress bar
        let response_result = self.ai.add_tool_results(tool_results).await;
        // stop our progress bar
        bar.finish_and_clear();
        response_result
    }

    /// Ask this agent a question
    ///
    /// # Arguments
    ///
    /// * `question` - The question to ask our ai
    pub async fn ask<T: Into<String>>(&mut self, question: T) -> Result<Option<String>, Error> {
        // ask our ai this question
        // if we need to run tools then run them and query the ai again otherwise return our answer
        match self.wrap_ask(question).await? {
            // we need to run tools so do that
            AiResponse::CallTool(mut tool_calls) => {
                // loop and execute tools until we get a response
                loop {
                    // call our tools in parallel
                    let tool_results = self.call_tools(tool_calls).await;
                    // tell our ai about these results
                    //match self.ai.add_tool_results(tool_results).await {
                    match self.wrap_add_tool_results(tool_results).await? {
                        AiResponse::CallTool(new_calls) => tool_calls = new_calls,
                        AiResponse::Response(response) => {
                            return Ok(response);
                        }
                    }
                }
            }
            AiResponse::Response(response) => Ok(response),
        }
    }
}

/// handle all ai commands for Thorctl
pub async fn handle(args: &Args, cmd: &Ai) -> Result<(), Error> {
    // get a thorium chat object
    let mut thorchat = ThorChat::<OpenAI>::new(args).await?;
    // handle the correct command
    match cmd {
        Ai::Summary(summary) => summary::handle(&mut thorchat, summary).await,
    }
}
