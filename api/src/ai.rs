//! AI utilities in Thorium

use futures::stream::{self, StreamExt};
use rmcp::RoleClient;
use rmcp::model::{CallToolRequestParam, CallToolResult, InitializeRequestParam, ListToolsResult};
use rmcp::service::RunningService;
use uuid::Uuid;

mod context;
pub mod services;
mod utils;

use crate::{CtlConf, Error};
pub use context::{
    SharedThorChatContext, ThorChatMsg, ThorChatMsgKinds, ThorChatToolCall, UserFriendlyChatMsg,
};

#[derive(Debug, Clone, Copy)]
pub enum AiMsgRole {
    /// This is a message from the user
    User,
    /// This is a message from the system
    System,
    /// This is a message from the assistant
    Assistant,
    /// This is a message from a function
    Function,
    /// This is a message from a tool
    Tool,
}

/// The different responses from the AI
#[derive(Debug)]
pub enum AiResponse {
    /// A request for us to call a tool
    CallTool(Vec<(Uuid, CallToolRequestParam)>),
    /// A response to the user
    Response(Option<String>),
}

/// Perform infrerence with an AI model provider
///
/// This trait allows for talking to different AI model providers. Allowing
/// [`ThorChat`] to support arbitrary providers.
#[async_trait::async_trait]
pub trait AiSupport: Sized + Send + Sync {
    /// A tool this AI can use/call
    type Tool: Clone + Send + Sync;

    ///  A single chat message for this LLM backend
    type ChatMsg: Clone + Send + Sync + std::fmt::Debug;

    /// Setup this ai client
    ///
    /// # Arguments
    ///
    /// * `conf` - A Thorctl config
    /// * `context` - The shared context for ai chat
    async fn setup(conf: &CtlConf, context: &SharedThorChatContext<Self>) -> Result<Self, Error>;

    /// configure the debug mode for this ai
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether or not debug mode is enabled
    fn debug_mode(&mut self, enabled: bool);

    /// Get a tools name
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to get a name for
    fn tool_name(tool: &Self::Tool) -> &String;

    /// Tell our AI about our tools
    ///
    /// # Arguments
    ///
    /// * `mcp_tools` - The mcp tools to tell our ai about
    fn load_tools(&mut self, mcp_tools: ListToolsResult) -> Result<(), Error>;

    /// Build a chat completion message
    ///
    /// # Arguments
    ///
    /// * `role` - The role for this message
    /// * `name` - The name to use for this message (primarily for tool names)
    /// * `msg` - The message to convert
    fn build_chat_msg(
        role: AiMsgRole,
        name: Option<String>,
        msg: impl Into<String>,
    ) -> Self::ChatMsg;

    /// Add the result from a tool
    ///
    /// # Arguments
    ///
    /// * `tool_results` - The tool results to tell our AI about
    async fn add_tool_results(
        &mut self,
        tool_results: Vec<(Uuid, String, CallToolResult)>,
    ) -> Result<AiResponse, Error>;

    /// Ask this agent a question
    ///
    /// # Arguments
    ///
    /// * `question` - The question to ask our ai
    async fn ask<T: Into<String> + Send + Sync>(
        &mut self,
        question: T,
    ) -> Result<AiResponse, Error>;
}

/// A Thorium AI chat bot
pub struct ThorChat<A: AiSupport + Send + Sync> {
    /// The AI client to use for reasoning
    pub ai: A,
    /// The context for this AI
    pub context: SharedThorChatContext<A>,
    /// The MCP client to use
    pub mcp: RunningService<RoleClient, InitializeRequestParam>,
}

impl<A: AiSupport + Send + Sync> ThorChat<A> {
    /// Create a new chat bot
    pub async fn new(conf: &CtlConf) -> Result<Self, Error> {
        // setup our context
        let context = SharedThorChatContext::<A>::default();
        // get an ai and assistant
        let ai = A::setup(conf, &context).await?;
        // setup mcp
        let mcp = utils::setup_mcp(conf).await?;
        // create a thorchat object
        let mut thorchat = ThorChat { ai, context, mcp };
        // init our tools
        thorchat.load_tools().await?;
        Ok(thorchat)
    }

    /// Set the base prompt
    pub fn base_prompt(&self, base_prompt: impl Into<String>) -> Result<(), Error> {
        // add a base prompt
        self.context.add_chat(AiMsgRole::System, base_prompt)
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
        self.context.disable_tool(tool);
    }

    /// Enable a specific tool
    ///
    /// If you enable any tools then only enabled tools can be run
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the mcp tool to enable
    #[expect(dead_code)]
    fn enable_tool<T: Into<String>>(&mut self, tool: T) {
        self.context.enable_tool(tool);
    }

    /// Add the results from tools
    ///
    /// # Arguments
    ///
    /// * `tool_calls` - The tool call request to tell our AI about
    pub async fn call_tools(
        &self,
        tool_calls: Vec<(Uuid, CallToolRequestParam)>,
    ) -> Result<Vec<(Uuid, String, CallToolResult)>, Error> {
        // call our tools in parallel
        stream::iter(tool_calls)
            .map(|(id, params)| async move { utils::call_tool_helper(&self.mcp, id, params).await })
            .buffered(10)
            .collect::<Vec<Result<(Uuid, String, CallToolResult), Error>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<(Uuid, String, CallToolResult)>, Error>>()
    }

    /// Ask this agent a question
    ///
    /// # Arguments
    ///
    /// * `question` - The question to ask our ai
    pub async fn ask<T: Into<String> + Send + Sync>(
        &mut self,
        question: T,
    ) -> Result<Option<String>, Error> {
        // ask our ai this question
        // if we need to run tools then run them and query the ai again otherwise return our answer
        match self.ai.ask(question).await? {
            // we need to run tools so do that
            AiResponse::CallTool(mut tool_calls) => {
                // loop and execute tools until we get a response
                loop {
                    // call our tools in parallel
                    let tool_results = self.call_tools(tool_calls).await?;
                    // tell our ai about these results
                    match self.ai.add_tool_results(tool_results).await? {
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
