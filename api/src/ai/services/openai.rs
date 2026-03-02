//! OpenAI support for [`ThorChat`]
use crate::{CtlConf, Error};
use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::chat_completion::chat_completion::{
    ChatCompletionRequest, ChatCompletionResponse,
};
use openai_api_rs::v1::chat_completion::{
    self, ChatCompletionMessage, Content, MessageRole, Tool, ToolChoiceType,
};
use openai_api_rs::v1::types::{Function, FunctionParameters, JSONSchemaDefine, JSONSchemaType};
use rmcp::model::{
    CallToolRequestParam, CallToolResult, ListToolsResult, RawContent, ResourceContents,
};
use std::borrow::Cow;
use std::collections::HashMap;
use uuid::Uuid;

use crate::ai::{AiMsgRole, AiResponse, AiSupport, SharedThorChatContext};

pub struct OpenAI {
    /// The AI client to use for reasoning
    ai: OpenAIClient,
    /// The context for this chat/AI
    context: SharedThorChatContext<Self>,
    /// Whether or not debug mode is enabled
    debug: bool,
    /// The model to use
    model: String,
}

impl OpenAI {
    /// Create a new chat bot
    ///
    /// # Arguments
    ///
    /// * `conf` - A thorctl config
    /// * `context` - The shared context to get/add messages to
    pub async fn new(conf: &CtlConf, context: &SharedThorChatContext<Self>) -> Result<Self, Error> {
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
        // create a thorchat object
        let openai = OpenAI {
            ai,
            context: context.clone(),
            debug: false,
            model: ai_conf.model.clone(),
        };
        Ok(openai)
    }

    /// Parse a chat response from the LLM
    ///
    /// # Arguments
    ///
    /// * `chat_response` - The response to parse
    fn parse_chat_response(
        &mut self,
        chat_response: ChatCompletionResponse,
    ) -> Result<AiResponse, Error> {
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
                        // generate an id for this tool call
                        let id = Uuid::new_v4();
                        // build our call tool request param
                        let params = CallToolRequestParam { name, arguments };
                        // build the message to add to our chat history
                        self.context.add_tool_call_request(id, params.clone());
                        // add this call to our list of tool calls
                        calls.push((id, params));
                    }
                    // return our tool calls
                    Ok(AiResponse::CallTool(calls))
                } else {
                    // get the message from the ai if we got one
                    if let Some(message) = &choice.message.content {
                        // add this message from the llm to our chat
                        self.context.add_chat(AiMsgRole::Assistant, message)?;
                    }
                    // this must be a response to the caller
                    Ok(AiResponse::Response(choice.message.content.clone()))
                }
            }
            None => Err(Error::new("AI responded with an empty response")),
        }
    }

    /// Send the latest chats to our LLM
    async fn send_chat(&mut self) -> Result<ChatCompletionResponse, Error> {
        // get our current history
        let history = self.context.history();
        // build a chat with shirty
        let mut req = ChatCompletionRequest::new(self.model.clone(), history);
        // get a list of tools
        let tools = self.context.tools();
        // only add our tools if we have some
        if !tools.is_empty() {
            req = req
                // add our tools to this chat request
                .tools(tools)
                // let the LLM decide what tools to call
                .tool_choice(ToolChoiceType::Auto);
        }
        // chat with our AI
        let resp = self.ai.chat_completion(req).await?;
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl AiSupport for OpenAI {
    /// A tool this AI can use/call
    type Tool = Tool;

    ///  A single chat message for this LLM backend
    type ChatMsg = ChatCompletionMessage;

    /// Setup this ai client
    ///
    /// # Arguments
    ///
    /// * `conf` - A thorctl config
    /// * `context` - The shared context to get/add messages to
    async fn setup(conf: &CtlConf, context: &SharedThorChatContext<Self>) -> Result<Self, Error> {
        OpenAI::new(conf, context).await
    }

    /// Configure the debug mode for this ai
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether or not debug mode is enabled
    fn debug_mode(&mut self, enabled: bool) {
        self.debug = enabled;
    }

    /// Get a tools name
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to get a name for
    fn tool_name(tool: &Self::Tool) -> &String {
        &tool.function.name
    }

    /// Tell our AI about our tools
    ///
    /// # Arguments
    ///
    /// * `mcp_tools` - The MCP tools to load
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
            self.context.add_tool(ai_tool);
        }
        Ok(())
    }

    /// Build a chat completion message
    ///
    /// # Arguments
    ///
    /// * `role` - The role for this message
    /// * `msg` - The message to convert
    fn build_chat_msg(
        role: AiMsgRole,
        name: Option<String>,
        msg: impl Into<String>,
    ) -> Self::ChatMsg {
        // convert this message role
        let role = match role {
            AiMsgRole::User => MessageRole::user,
            AiMsgRole::System => MessageRole::system,
            AiMsgRole::Assistant => MessageRole::assistant,
            AiMsgRole::Function => MessageRole::function,
            AiMsgRole::Tool => MessageRole::tool,
        };
        // build the message to ask our ai
        ChatCompletionMessage {
            role,
            content: Content::Text(msg.into()),
            name,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Add some tool results to our chat history
    ///
    /// # Arguments
    ///
    /// * `tool_results` - The tool results to tell our AI about
    async fn add_tool_results(
        &mut self,
        tool_results: Vec<(Uuid, String, CallToolResult)>,
    ) -> Result<AiResponse, Error> {
        // add each tool result to our chat history
        for (id, name, tool_result) in tool_results {
            // add each page of content to our history
            for page in tool_result.content {
                // get the raw content
                let content = match page.raw {
                    RawContent::Text(text) => text.text,
                    RawContent::Resource(resource) => {
                        // get our resource contents
                        match resource.resource {
                            ResourceContents::TextResourceContents { text, .. } => text,
                            ResourceContents::BlobResourceContents { blob, .. } => blob,
                        }
                    }
                    _ => unimplemented!("{page:#?} not yet supported"),
                };
                // add this tool call to our chat history
                self.context.add_tool_result(id, name.clone(), content);
            }
        }
        // send our results back to our ai for a response
        let chat_response = self.send_chat().await?;
        // parse this response
        self.parse_chat_response(chat_response)
    }

    /// Ask this agent a question
    ///
    /// # Arguments
    ///
    /// * `question` - The question to ask
    async fn ask<T: Into<String> + Send + Sync>(
        &mut self,
        question: T,
    ) -> Result<AiResponse, Error> {
        // add this message to our history
        self.context.add_chat(AiMsgRole::User, question)?;
        // send our question to the ai
        let chat_response = self.send_chat().await?;
        // parse this response
        self.parse_chat_response(chat_response)
    }
}
