//! The different ai models/services that Thorium Chat supports

use rmcp::model::{CallToolRequestParam, CallToolResult, ListToolsResult};
use thorium::{CtlConf, Error};

mod openai;

pub use openai::OpenAI;

pub enum AiResponse {
    /// A request for us to call a tool
    CallTool(Vec<CallToolRequestParam>),
    /// A response to the user
    Response(Option<String>),
}

pub trait AiSupport: Sized {
    /// Setup this ai client
    ///
    /// # Arguments
    ///
    /// * `conf` - A Thorctl config
    async fn setup(conf: &CtlConf) -> Result<Self, Error>;

    /// configure the debug mode for this ai
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether or not debug mode is enabled
    fn debug_mode(&mut self, enabled: bool);

    /// Tell our AI about our tools
    ///
    /// # Arguments
    ///
    /// * `mcp_tools` - The mcp tools to tell our ai about
    fn load_tools(&mut self, mcp_tools: ListToolsResult) -> Result<(), Error>;

    /// Disable a specific tool
    fn disable_tool<T: Into<String>>(&mut self, tool: T);

    /// Enable a specific tool
    ///
    /// If you enable any tools then only enabled tools can be run
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the mcp tool to enable
    fn enable_tool<T: Into<String>>(&mut self, tool: T);

    /// Add the result from a tool
    ///
    /// # Arguments
    ///
    /// * `tool_results` - The tool results to tell our AI about
    async fn add_tool_results(
        &mut self,
        tool_results: Vec<(String, CallToolResult)>,
    ) -> Result<AiResponse, Error>;

    /// Ask this agent a question
    ///
    /// # Arguments
    ///
    /// * `question` - The question to ask our ai
    async fn ask<T: Into<String>>(&mut self, question: T) -> Result<AiResponse, Error>;
}
