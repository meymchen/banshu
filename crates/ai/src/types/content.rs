//! Content blocks that make up message bodies.
//!
//! JSON shapes follow the pi-ai persistence contract: blocks are internally
//! tagged by `type` (`text` / `thinking` / `image` / `toolCall`), fields are
//! camelCase, and signatures use pi's `textSignature` / `thinkingSignature`
//! names. See `tests/fixtures/context_snapshot_v1.json`.

/// A run of plain text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TextContent {
    /// The text itself.
    pub text: String,
    /// Opaque provider signature for multi-turn continuity, when present.
    #[serde(
        rename = "textSignature",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub signature: Option<String>,
}

/// A run of model "thinking" / reasoning content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ThinkingContent {
    /// The reasoning text.
    pub thinking: String,
    /// Opaque provider signature for replaying thinking across turns.
    #[serde(
        rename = "thinkingSignature",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub signature: Option<String>,
    /// Whether the content was redacted by provider safety filters.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
}

/// A base64-encoded image block.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image bytes.
    pub data: String,
    /// MIME type, e.g. `image/png`.
    pub mime_type: String,
}

/// A request from the model to invoke a tool.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Provider-assigned call id, echoed back on the tool result.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// Parsed tool arguments.
    pub arguments: serde_json::Value,
    /// Original JSON text received from the provider, when available.
    /// Extension field beyond the pi-ai shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

/// A content block produced by the assistant.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantContent {
    /// Plain text output.
    Text(TextContent),
    /// Reasoning output.
    Thinking(ThinkingContent),
    /// A tool invocation.
    ToolCall(ToolCall),
}

/// A content block supplied by the user.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserContent {
    /// Plain text input.
    Text(TextContent),
    /// An image input.
    Image(ImageContent),
}
