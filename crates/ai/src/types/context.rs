//! The request `Context` — system prompt, message history, and tools.

use super::message::{Message, ToolResultMessage, UserMessage};
use super::tool::Tool;

/// Everything a provider needs to produce the next assistant turn.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Conversation history, oldest first.
    pub messages: Vec<Message>,
    /// Tools offered to the model.
    pub tools: Vec<Tool>,
}

impl Context {
    /// An empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the system prompt.
    pub fn with_system(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Append a user text message.
    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.messages.push(Message::User(UserMessage::text(text)));
        self
    }

    /// Append any message (e.g. replaying assistant/tool-result history).
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Append a tool result for the tool call `tool_call_id`.
    pub fn tool_result(
        self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        self.with_message(Message::ToolResult(ToolResultMessage::text(
            tool_call_id,
            tool_name,
            content,
        )))
    }

    /// Add a tool.
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }
}
