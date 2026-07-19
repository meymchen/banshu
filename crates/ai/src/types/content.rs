//! Content blocks that make up message bodies.

/// A run of plain text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextContent {
    /// The text itself.
    pub text: String,
    /// Opaque provider signature for multi-turn continuity, when present.
    pub signature: Option<String>,
}

/// A run of model "thinking" / reasoning content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingContent {
    /// The reasoning text.
    pub thinking: String,
    /// Opaque provider signature for replaying thinking across turns.
    pub signature: Option<String>,
    /// Whether the content was redacted by provider safety filters.
    pub redacted: bool,
}

/// A base64-encoded image block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageContent {
    /// Base64-encoded image bytes.
    pub data: String,
    /// MIME type, e.g. `image/png`.
    pub mime_type: String,
}

/// A request from the model to invoke a tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Provider-assigned call id, echoed back on the tool result.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// Parsed tool arguments.
    pub arguments: serde_json::Value,
    /// Original JSON text received from the provider, when available.
    pub raw_arguments: Option<String>,
}

/// A content block produced by the assistant.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantContent {
    /// Plain text output.
    Text(TextContent),
    /// Reasoning output.
    Thinking(ThinkingContent),
    /// A tool invocation.
    ToolCall(ToolCall),
}

/// A content block supplied by the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserContent {
    /// Plain text input.
    Text(TextContent),
    /// An image input.
    Image(ImageContent),
}
