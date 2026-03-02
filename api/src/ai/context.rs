//! The context for a chat with an LLM
use std::sync::{Arc, Mutex};

use rmcp::model::CallToolRequestParam;
use uuid::Uuid;

use super::{AiMsgRole, AiSupport};
use crate::client::Error;

#[derive(Debug, Clone)]
pub enum UserFriendlyChatMsg {
    /// A message from the user
    User(String),
    /// A response from the AI
    Ai(String),
}

#[derive(Debug, Clone)]
pub struct ThorChatMsg<A: AiSupport> {
    /// The service specific message
    pub original: A::ChatMsg,
    /// A user friendly representation of the service specific message
    pub user_friendly: UserFriendlyChatMsg,
    /// Whether this message should be minimized or not
    pub is_minimized: bool,
}

/// A tool call request
#[derive(Debug, Clone)]
pub struct ThorChatToolCall {
    /// The Id for this tool call request
    pub id: Uuid,
    /// The params for calling this tool
    pub params: CallToolRequestParam,
    /// The serialized and editable version of our params
    pub editable: String,
    /// Whether this tool call is approved or not
    pub approved: bool,
    /// Whether this message should be minimized or not
    pub is_minimized: bool,
}

/// A tool call result
#[derive(Debug, Clone)]
pub struct ThorChatToolCallResult<A: AiSupport> {
    /// The Id for this tool call request
    pub id: Uuid,
    /// The service specific message
    pub original: A::ChatMsg,
    /// The results for this tool call
    pub results: String,
}

#[derive(Debug, Clone)]
pub enum ThorChatMsgKinds<A: AiSupport> {
    /// A message to or from the AI
    Msg(ThorChatMsg<A>),
    /// A tool call request
    ToolCallRequest(ThorChatToolCall),
    /// A tool call result
    ToolCallResult(ThorChatToolCallResult<A>),
}

/// The context for a chat with an LLM
#[derive(Debug)]
pub struct ThorChatContext<A: AiSupport> {
    /// The tools this AI can use
    pub tools: Vec<A::Tool>,
    /// A list of enabled tools
    pub enabled_tools: Vec<String>,
    /// A list of disabled toold,
    pub disabled_tools: Vec<String>,
    /// The current conversation history in a format usable by the LLM
    pub history: Vec<ThorChatMsgKinds<A>>,
}

/// A Shared Thorium chat context
pub struct SharedThorChatContext<A: AiSupport>(Arc<Mutex<ThorChatContext<A>>>);

impl<A: AiSupport> Default for SharedThorChatContext<A> {
    /// Setup a default chat context
    fn default() -> Self {
        // build a default empty chat context
        let context = ThorChatContext {
            tools: Vec::default(),
            enabled_tools: Vec::default(),
            disabled_tools: Vec::default(),
            history: Vec::default(),
        };
        // wrap this context in an arc and a mutex
        let inner = Arc::new(Mutex::new(context));
        // build our final shared type
        SharedThorChatContext(inner)
    }
}

impl<A: AiSupport> std::clone::Clone for SharedThorChatContext<A> {
    /// Clone a reference
    fn clone(&self) -> Self {
        // clone our inner value and rewrap
        SharedThorChatContext(self.0.clone())
    }
}

impl<A: AiSupport> SharedThorChatContext<A> {
    /// Add a tool to our shared context
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to add
    pub fn add_tool(&self, tool: A::Tool) {
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // add this new tool
        guard.tools.push(tool);
    }

    /// Enable a tool
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the tool to enable
    pub fn enable_tool(&self, tool: impl Into<String>) {
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // enable this tool
        guard.enabled_tools.push(tool.into());
    }

    /// Disable a tool
    ///
    /// # Arguments
    ///
    /// * `tool` - The name of the tool to disable
    pub fn disable_tool(&self, tool: impl Into<String>) {
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // disable this tool
        guard.disabled_tools.push(tool.into());
    }

    /// Add a chat message to this context
    ///
    /// # Arguments
    ///
    /// * `role` - The role for this new message
    /// * `msg` - The content of the message to add
    pub fn add_chat(&self, role: AiMsgRole, msg: impl Into<String>) -> Result<(), Error> {
        // get our message string
        let msg = msg.into();
        // build the chat message to add
        let original = A::build_chat_msg(role, None, msg.clone());
        // build our user friendly representation of this message
        let user_friendly = match role {
            AiMsgRole::User => UserFriendlyChatMsg::User(msg),
            AiMsgRole::System => UserFriendlyChatMsg::Ai(msg),
            AiMsgRole::Assistant => UserFriendlyChatMsg::Ai(msg),
            _ => return Err(Error::new("Unsupported AiMsgRole: {role:?}")),
        };
        // build our thor chat message
        let thor_msg = ThorChatMsg {
            original,
            user_friendly,
            is_minimized: false,
        };
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // add this message
        guard.history.push(ThorChatMsgKinds::Msg(thor_msg));
        Ok(())
    }

    /// A tool call request to this context
    ///
    /// # Arguments
    ///
    /// * `id` - The id for this tool call request
    /// * `params` - The params for this tool call
    pub fn add_tool_call_request(&self, id: Uuid, params: CallToolRequestParam) {
        // get an editable version of our tool call params
        let editable = serde_json::to_string(&params).unwrap();
        // build our tool call request
        let tool_call = ThorChatToolCall {
            id,
            params,
            editable,
            approved: false,
            is_minimized: true,
        };
        // wrap our tool call
        let wrapped = ThorChatMsgKinds::ToolCallRequest(tool_call);
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // add this message
        guard.history.push(wrapped);
    }

    /// Add a tool result to this context
    ///
    /// # Arguments
    ///
    /// * `id` - The id for this tool call result
    /// * `name` - The name of the tool that we are adding a response
    /// * `content` - The result/content from this tool
    pub fn add_tool_result(&self, id: Uuid, name: impl Into<String>, content: impl Into<String>) {
        // get our message string
        let content = content.into();
        // get our name
        let name = Some(name.into());
        // build the chat message to add
        let original = A::build_chat_msg(AiMsgRole::Tool, name, content.clone());
        // build out our tool call result
        let tool_result = ThorChatToolCallResult {
            id,
            original,
            results: content,
        };
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // add this message
        guard
            .history
            .push(ThorChatMsgKinds::ToolCallResult(tool_result));
    }

    /// Get our current history in a format able to be sent to the LLM/AI
    pub fn history(&self) -> Vec<A::ChatMsg> {
        // get a guard to our shared context
        let guard = self.0.lock().unwrap();
        // clone our history
        guard
            .history
            .iter()
            // filter our anything that isn't a chat message
            .filter_map(|kind| match kind {
                ThorChatMsgKinds::Msg(msg) => Some(msg.original.clone()),
                ThorChatMsgKinds::ToolCallRequest(_) => None,
                ThorChatMsgKinds::ToolCallResult(msg) => Some(msg.original.clone()),
            })
            .collect()
    }

    /// Get our current history in a format able to be sent to the LLM/AI
    pub fn access(&self) -> std::sync::MutexGuard<'_, ThorChatContext<A>> {
        // get a guard to our shared context
        self.0.lock().unwrap()
    }

    /// Get a list of tools to call
    pub fn tools(&self) -> Vec<A::Tool> {
        // get a guard to our shared context
        let guard = self.0.lock().unwrap();
        // clone only the enabled tools if we have any disabled/enabled tools
        match (
            &guard.enabled_tools.is_empty(),
            &guard.disabled_tools.is_empty(),
        ) {
            // only return our enabled and not disabled tools
            (false, _) => {
                guard
                    .tools
                    .iter()
                    .filter(|tool| {
                        // get this tools name
                        let name = A::tool_name(tool);
                        // only allow enabled and not disabled tools
                        guard.enabled_tools.contains(name) && !guard.disabled_tools.contains(name)
                    })
                    .cloned()
                    .collect()
            }
            // only return our not disabled tools
            (true, false) => {
                guard
                    .tools
                    .iter()
                    .filter(|tool| {
                        // get this tools name
                        let name = A::tool_name(tool);
                        // only allow enabled and not disabled tools
                        !guard.disabled_tools.contains(name)
                    })
                    .cloned()
                    .collect()
            }
            // return all tools that we know about
            (true, true) => guard.tools.clone(),
        }
    }

    /// Clear existing chat messages
    pub fn clear(&self) {
        // get a guard to our shared context
        let mut guard = self.0.lock().unwrap();
        // clear our chat history
        guard.history.clear();
    }
}
