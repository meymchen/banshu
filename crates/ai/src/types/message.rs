//! Conversation messages and the assembled assistant result.

use super::content::{AssistantContent, TextContent, UserContent};
use super::now_ms;
use super::usage::Usage;
use crate::error::ErrorKind;

/// Why a completion stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Natural end of turn.
    Stop,
    /// Hit the max-tokens limit.
    Length,
    /// The model requested tool calls.
    ToolUse,
    /// The stream failed; see `error_message`.
    Error,
    /// The request was aborted by the caller.
    Aborted,
}

/// A user turn.
#[derive(Debug, Clone)]
pub struct UserMessage {
    /// Ordered content blocks.
    pub content: Vec<UserContent>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl UserMessage {
    /// Build a user message from a single text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![UserContent::Text(TextContent {
                text: text.into(),
                signature: None,
            })],
            timestamp: now_ms(),
        }
    }

    /// Concatenate all text blocks (images are ignored).
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                UserContent::Text(t) => Some(t.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// An assembled assistant turn — the terminal value of a [`MessageStream`](crate::MessageStream).
#[derive(Debug, Clone)]
pub struct AssistantMessage {
    /// Ordered content blocks (text / thinking / tool calls).
    pub content: Vec<AssistantContent>,
    /// The wire protocol used, e.g. `openai-completions`.
    pub api: String,
    /// The provider id that produced this message.
    pub provider: String,
    /// The requested model id.
    pub model: String,
    /// Concrete routed model id when it differs from `model` (e.g. router "auto").
    pub response_model: Option<String>,
    /// Token usage and cost.
    pub usage: Usage,
    /// Why the completion stopped.
    pub stop_reason: StopReason,
    /// Human-readable error, set when `stop_reason` is `Error`/`Aborted`.
    pub error_message: Option<String>,
    /// Structured classification of the failure, set alongside `error_message`.
    pub error_kind: Option<ErrorKind>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl AssistantMessage {
    /// Build an assistant message from content blocks, for replaying history
    /// (e.g. an assistant turn that issued tool calls). Metadata is left empty.
    pub fn from_content(content: Vec<AssistantContent>) -> Self {
        Self {
            content,
            api: String::new(),
            provider: String::new(),
            model: String::new(),
            response_model: None,
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            error_kind: None,
            timestamp: now_ms(),
        }
    }

    /// A fresh, empty streaming message to accumulate deltas into.
    pub(crate) fn streaming(model: &str, provider: &str, api: &str) -> Self {
        Self {
            content: Vec::new(),
            api: api.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            response_model: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            error_kind: None,
            timestamp: now_ms(),
        }
    }

    /// Concatenate all text content blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// The result of a tool call, fed back for the next turn.
#[derive(Debug, Clone)]
pub struct ToolResultMessage {
    /// The id of the tool call this answers (echoes `ToolCall.id`).
    pub tool_call_id: String,
    /// The tool's name.
    pub tool_name: String,
    /// The result content (text).
    pub content: String,
    /// Whether the tool call failed.
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

impl ToolResultMessage {
    /// A successful text tool result.
    pub fn text(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: content.into(),
            is_error: false,
            timestamp: now_ms(),
        }
    }
}

/// Any message in a conversation.
#[derive(Debug, Clone)]
pub enum Message {
    /// A user turn.
    User(UserMessage),
    /// An assistant turn. Boxed because an assembled assistant message is much
    /// larger than a user turn.
    Assistant(Box<AssistantMessage>),
    /// A tool result turn.
    ToolResult(ToolResultMessage),
}
