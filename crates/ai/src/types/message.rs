//! Conversation messages and the assembled assistant result.

use super::content::{AssistantContent, TextContent, UserContent};
use super::now_ms;
use super::usage::Usage;
use crate::error::ErrorKind;

const MAX_DIAGNOSTIC_MESSAGE_CHARS: usize = 1_024;
const SENSITIVE_DIAGNOSTIC_LABELS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "api_key",
    "apikey",
    "x-auth-token",
    "access-token",
    "access_token",
    "refresh-token",
    "refresh_token",
    "id-token",
    "id_token",
    "client-secret",
    "client_secret",
];

/// Stable category for a safe, user-visible diagnostic.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DiagnosticCode {
    /// An error reported by the upstream provider.
    ProviderError,
    /// A violation of the provider's wire protocol.
    ProtocolViolation,
    /// An image was replaced or omitted for compatibility.
    ImageDowngraded,
    /// A diagnostic that does not fit a more specific category.
    Other,
}

/// A bounded, non-sensitive diagnostic attached to an assistant message.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    /// Machine-readable category.
    pub code: DiagnosticCode,
    /// Safe human-readable detail, capped at 1024 Unicode characters.
    #[serde(
        serialize_with = "serialize_diagnostic_message",
        deserialize_with = "deserialize_diagnostic_message"
    )]
    pub message: String,
}

impl Diagnostic {
    /// Build a bounded diagnostic, redacting authentication values and base64.
    pub fn new(code: DiagnosticCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: sanitize_diagnostic_message(&message.into()),
        }
    }
}

impl std::fmt::Debug for Diagnostic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Diagnostic")
            .field("code", &self.code)
            .field("message", &sanitize_diagnostic_message(&self.message))
            .finish()
    }
}

fn serialize_diagnostic_message<S>(message: &str, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serde::Serialize::serialize(&sanitize_diagnostic_message(message), serializer)
}

fn deserialize_diagnostic_message<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let message = <String as serde::Deserialize>::deserialize(deserializer)?;
    Ok(sanitize_diagnostic_message(&message))
}

fn sanitize_diagnostic_message(message: &str) -> String {
    let labeled_values_redacted = message
        .split('\n')
        .map(redact_sensitive_label_value)
        .collect::<Vec<_>>()
        .join("\n");
    let bearer_values_redacted = redact_bearer_values(&labeled_values_redacted);
    let data_urls_redacted = redact_data_url_payloads(&bearer_values_redacted);
    redact_long_base64_runs(&data_urls_redacted)
        .chars()
        .take(MAX_DIAGNOSTIC_MESSAGE_CHARS)
        .collect()
}

fn redact_sensitive_label_value(line: &str) -> String {
    let lowercase = line.to_ascii_lowercase();
    let delimiter = SENSITIVE_DIAGNOSTIC_LABELS
        .iter()
        .filter_map(|label| {
            let label_start = lowercase.find(label)?;
            let after_label = label_start + label.len();
            let separator_prefix = lowercase[after_label..]
                .bytes()
                .take_while(|byte| {
                    byte.is_ascii_whitespace() || matches!(byte, b'"' | b'\'' | b'\\')
                })
                .count();
            let delimiter = after_label + separator_prefix;
            matches!(lowercase.as_bytes().get(delimiter), Some(b':' | b'=')).then_some(delimiter)
        })
        .min();

    match delimiter {
        Some(delimiter) => format!("{} [REDACTED]", &line[..=delimiter]),
        None => line.to_string(),
    }
}

fn redact_bearer_values(message: &str) -> String {
    redact_token_after_marker(message, "bearer ", false)
}

fn redact_data_url_payloads(message: &str) -> String {
    redact_token_after_marker(message, ";base64,", true)
}

fn redact_token_after_marker(message: &str, marker: &str, base64_only: bool) -> String {
    let mut output = String::with_capacity(message.len());
    let mut remainder = message;

    while let Some(marker_start) = remainder.to_ascii_lowercase().find(marker) {
        let value_start = marker_start + marker.len();
        output.push_str(&remainder[..value_start]);
        let value_len = remainder[value_start..]
            .bytes()
            .take_while(|byte| {
                if base64_only {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'=')
                } else {
                    !byte.is_ascii_whitespace()
                        && !matches!(byte, b',' | b';' | b'"' | b'\'' | b')' | b']' | b'}')
                }
            })
            .count();
        if value_len == 0 {
            remainder = &remainder[value_start..];
            continue;
        }
        output.push_str("[REDACTED]");
        remainder = &remainder[value_start + value_len..];
    }
    output.push_str(remainder);
    output
}

fn redact_long_base64_runs(message: &str) -> String {
    let bytes = message.as_bytes();
    let mut output = String::with_capacity(message.len());
    let mut cursor = 0;

    while cursor < bytes.len() {
        if bytes[cursor].is_ascii_alphanumeric() || matches!(bytes[cursor], b'+' | b'/' | b'=') {
            let start = cursor;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric()
                    || matches!(bytes[cursor], b'+' | b'/' | b'='))
            {
                cursor += 1;
            }
            if cursor - start >= 64 {
                output.push_str("[REDACTED]");
            } else {
                output.push_str(&message[start..cursor]);
            }
        } else {
            let character = message[cursor..]
                .chars()
                .next()
                .expect("cursor is within the string");
            output.push(character);
            cursor += character.len_utf8();
        }
    }
    output
}

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
    /// Provider-specific response or message identifier, when exposed.
    pub response_id: Option<String>,
    /// Safe provider/runtime diagnostics.
    pub diagnostics: Vec<Diagnostic>,
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
            response_id: None,
            diagnostics: Vec::new(),
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
            response_id: None,
            diagnostics: Vec::new(),
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
    /// Ordered result content blocks.
    pub content: Vec<UserContent>,
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
        Self::text_with_error(tool_call_id, tool_name, content, false)
    }

    /// A failed text tool result.
    pub fn error_text(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::text_with_error(tool_call_id, tool_name, content, true)
    }

    /// Build a tool result from ordered content blocks.
    pub fn content(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<UserContent>,
        is_error: bool,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            is_error,
            timestamp: now_ms(),
        }
    }

    /// Join text blocks in order for text-only wire formats.
    pub(crate) fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|content| match content {
                UserContent::Text(text) => Some(text.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn text_with_error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::content(
            tool_call_id,
            tool_name,
            vec![UserContent::Text(TextContent {
                text: content.into(),
                signature: None,
            })],
            is_error,
        )
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
