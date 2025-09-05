//! OpenAI support for ThoriumAgent
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{
        self, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, Content,
        MessageRole, Reasoning, ReasoningEffort, ReasoningMode, Tool,
    },
    types::{Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType},
};
use rmcp::model::{CallToolRequestParam, CallToolResult, ListToolsResult, RawContent};
use std::borrow::Cow;
use std::collections::HashMap;
use thorium::{CtlConf, Error};

use super::{AiResponse, AiSupport};

pub struct OpenAI {
    /// The AI client to use for reasoning
    ai: OpenAIClient,
    /// The different tools we can support
    tools: Vec<Tool>,
    /// A list of enabled tools
    enabled_tools: Vec<String>,
    /// A list of disabled toold,
    disabled_tools: Vec<String>,
    /// Keep track of any already called tools
    already_called: Vec<String>,
    /// The current conversation history
    history: Vec<ChatCompletionMessage>,
    /// Whether or not debug mode is enabled
    debug: bool,
    /// The model to use
    model: String,
}

impl OpenAI {
    /// Create a new chat bot
    pub async fn new(conf: &CtlConf) -> Result<Self, Error> {
        // make sure we have our ai settings configured
        let ai_conf = match &conf.ai {
            Some(ai_conf) => ai_conf,
            None => {
                return Err(Error::Generic(
                    "Ai Settings are missing from the thorctl config".to_owned(),
                ));
            }
        };
        // build a client to an openai server
        let ai = OpenAIClient::builder()
            .with_endpoint(&ai_conf.endpoint)
            .with_api_key(&ai_conf.api_key)
            .build()
            .map_err(|err| Error::new(format!("Failed to build OpenAI Client: {err:?}")))?;
        // track our chat history
        let mut history = Vec::with_capacity(10);
        // build the base prompt for our ai
        let base_prompt = "You are an expert file analysis AI assistant for Thorium a file analysis and data generation platform. Only call tools when you believe they will provide useful results.";
        // build the message to ask our ai
        let msg = ChatCompletionMessage {
            role: MessageRole::system,
            content: Content::Text(base_prompt.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        // add this msg to our chat history
        history.push(msg);
        // create a thorchat object
        let openai = OpenAI {
            ai,
            tools: vec![],
            enabled_tools: vec![],
            disabled_tools: vec![],
            already_called: vec![],
            history,
            debug: false,
            model: ai_conf.model.clone(),
        };
        Ok(openai)
    }

    fn parse_chat_response(
        &mut self,
        chat_response: ChatCompletionResponse,
    ) -> Result<AiResponse, Error> {
        if self.debug {
            println!("DEBUG RESP: {:#?}", chat_response);
        }
        match chat_response.choices.get(0) {
            // map this choice back to a response to ThoriumChat
            Some(choice) => {
                // if a tool call has been requested then call that tool
                // check if the ai had a tool call
                if let Some(tool_calls) = &choice.message.tool_calls {
                    // instance a list of our tool calls
                    let mut calls = Vec::with_capacity(tool_calls.capacity());
                    // call the tools our ai has requested
                    for tool_call in tool_calls {
                        // get our arguments for this call
                        let arguments = match &tool_call.function.arguments {
                            Some(raw_args) => serde_json::from_str(raw_args)?,
                            None => None,
                        };
                        // get our function name or raise an error
                        let name: Cow<'static, str> = match &tool_call.function.name {
                            Some(name) => Cow::Owned(name.to_owned()),
                            None => {
                                return Err(Error::new(
                                    "AI asked us to call a tool but gave no tool name",
                                ));
                            }
                        };
                        // add the name of this tool to our already called tools
                        self.already_called.push(name.to_string());
                        // build our call tool request param
                        let params = CallToolRequestParam { name, arguments };
                        // add this call to our list of tool calls
                        calls.push(params);
                    }
                    // return our tool calls
                    Ok(AiResponse::CallTool(calls))
                } else {
                    // get the message from the ai if we got one
                    if let Some(message) = &choice.message.content {
                        // clone the AI's response into our chat history
                        let resp_msg = ChatCompletionMessage {
                            role: MessageRole::assistant,
                            content: Content::Text(message.to_owned()),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        };
                        // add the llm's response to our history
                        self.history.push(resp_msg);
                    }
                    // this must be a response to the caller
                    Ok(AiResponse::Response(choice.message.content.clone()))
                }
            }
            None => Err(Error::new("AI responded with an empty response")),
        }
    }

    async fn send_chat(&mut self) -> Result<ChatCompletionResponse, Error> {
        // build a chat with shirty
        let mut req = ChatCompletionRequest::new(self.model.clone(), self.history.clone());
        // build a list of tools that we haven't already called
        let tools_iter = self
            .tools
            .iter()
            .filter(|tool| !self.already_called.contains(&tool.function.name))
            .filter(|tool| !self.disabled_tools.contains(&tool.function.name));
        // if we have any enabled tools then only allow those to be used
        let tools = if !self.enabled_tools.is_empty() {
            // only get tools in our enabled set
            tools_iter
                .filter(|tool| self.enabled_tools.contains(&tool.function.name))
                .map(|tool| tool.clone())
                .collect::<Vec<Tool>>()
        } else {
            // get the names of the tools we can use
            tools_iter.map(|tool| tool.clone()).collect::<Vec<Tool>>()
        };
        // only add our tools if we have some
        if !tools.is_empty() {
            req = req
                // add our tools to this chat request
                .tools(tools)
                // let the LLM decide what tools to call
                .tool_choice(chat_completion::ToolChoiceType::Auto);
        }
        // TODO do this differently with tracing
        if self.debug {
            println!("DEBUG SEND: {:#?}\n", self.history.last());
        }
        // chat with our AI
        let resp = self.ai.chat_completion(req).await?;
        Ok(resp)
    }
}

impl AiSupport for OpenAI {
    /// Setup this ai client
    async fn setup(conf: &CtlConf) -> Result<Self, Error> {
        OpenAI::new(conf).await
    }

    /// configure the debug mode for this ai
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether or not debug mode is enabled
    fn debug_mode(&mut self, enabled: bool) {
        self.debug = enabled;
    }

    /// Tell our AI about our tools
    fn load_tools(&mut self, mcp_tools: ListToolsResult) -> Result<(), Error> {
        // add each tool to our ai tools list
        for mcp_tool in mcp_tools.tools {
            let mut properties = HashMap::with_capacity(mcp_tool.input_schema.len());
            // get the input properties for this tool if they exist
            if let Some(mcp_properties_val) = mcp_tool.input_schema.get("properties") {
                // make sure our mcp properties are an object
                let mcp_properties = match mcp_properties_val.as_object() {
                    Some(mcp_properties) => mcp_properties,
                    None => {
                        return Err(Error::new(format!(
                            "MCP properties is not a map: {mcp_properties_val:?}"
                        )));
                    }
                };
                // step over the properties for this tool
                for (name, value) in mcp_properties {
                    // get our description if we have one
                    let description = match value.get("description") {
                        // we have a description make sure its a string or raise an error
                        Some(description_val) => {
                            // descriptions must be a string
                            match description_val.as_str() {
                                Some(description) => Some(description.to_string()),
                                None => {
                                    return Err(Error::new(format!(
                                        "Tool description is not a string: {description_val:?}"
                                    )));
                                }
                            }
                        }
                        None => None,
                    };
                    // build the property for this input
                    let input = JSONSchemaDefine {
                        schema_type: Some(JSONSchemaType::String),
                        description,
                        ..Default::default()
                    };
                    // add this property
                    properties.insert(name.to_owned(), Box::new(input));
                }
            }
            // get a list of the required arguments
            let required = match mcp_tool.input_schema.get("required") {
                Some(required_val) => {
                    // make sure our required parameters are a list
                    match required_val.as_array() {
                        Some(required_arr_val) => {
                            // get all of our required params
                            let required = required_arr_val
                                .iter()
                                .filter_map(|arg| arg.as_str())
                                .map(|arg| arg.to_string())
                                .collect::<Vec<String>>();
                            // return the required args
                            Some(required)
                        }
                        None => panic!("required is not an array?: {required_val:#?}"),
                    }
                }
                None => None,
            };
            // build the tool object for this mcp tool
            let ai_tool = Tool {
                r#type: chat_completion::ToolType::Function,
                function: Function {
                    name: mcp_tool.name.to_string(),
                    description: mcp_tool.description.map(|cow| cow.to_string()),
                    parameters: FunctionParameters {
                        schema_type: openai_api_rs::v1::types::JSONSchemaType::Object,
                        properties: Some(properties),
                        required,
                    },
                },
            };
            // add our tool
            self.tools.push(ai_tool);
        }
        Ok(())
    }

    /// Disable a specific tool
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the mcp tool to disable
    fn disable_tool<T: Into<String>>(&mut self, tool: T) {
        // add this tool to our list of disabled tools
        self.disabled_tools.push(tool.into());
    }

    /// Enable a specific tool
    ///
    /// If you enable any tools then only enabled tools can be run
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the mcp tool to enable
    fn enable_tool<T: Into<String>>(&mut self, tool: T) {
        // add this tool to our list of enabled tools
        self.enabled_tools.push(tool.into());
    }

    /// Call a tool
    async fn add_tool_results(
        &mut self,
        tool_results: Vec<(String, CallToolResult)>,
    ) -> Result<AiResponse, Error> {
        // add each tool result to our chat history
        for (name, tool_result) in tool_results {
            // add each page of content to our history
            for page in tool_result.content {
                // get the raw content
                let content = match page.raw {
                    RawContent::Text(text) => Content::Text(text.text),
                    _ => panic!("{page:#?} not yet supported"),
                };
                // add this tool call to our chat history
                let msg = ChatCompletionMessage {
                    role: MessageRole::function,
                    content,
                    name: Some(name.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                };
                self.history.push(msg);
            }
        }
        // send our results back to our ai for a response
        let chat_response = self.send_chat().await?;
        // parse this response
        self.parse_chat_response(chat_response)
    }

    /// Ask this agent a question
    async fn ask<T: Into<String>>(&mut self, question: T) -> Result<AiResponse, Error> {
        // build the message to ask our ai
        let msg = ChatCompletionMessage {
            role: MessageRole::user,
            content: Content::Text(question.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        // add this msg to our chat history
        self.history.push(msg);
        // clear our already called tools
        self.already_called.clear();
        // send our question to the ai
        let chat_response = self.send_chat().await?;
        // parse this response
        self.parse_chat_response(chat_response)
    }
}
