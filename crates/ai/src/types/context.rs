//! The request `Context` — system prompt, message history, and tools.

use super::message::{Message, ToolResultMessage, UserMessage};
use super::tool::Tool;

/// Everything a provider needs to produce the next assistant turn.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    /// Optional system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Conversation history, oldest first.
    pub messages: Vec<Message>,
    /// Tools offered to the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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

/// A versioned persistence wrapper around [`Context`].
///
/// The serialized form (`{ "version": 1, "context": { ... } }`) is a
/// persistence contract, pinned by `tests/fixtures/context_snapshot_v1.json`.
/// Deserialization rejects any version other than 1 outright — a snapshot
/// written by a future format is never parsed best-effort.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ContextSnapshotV1 {
    /// Snapshot format version; always 1 for this type.
    pub version: u32,
    /// The wrapped conversation context.
    pub context: Context,
}

/// The only snapshot version this type reads and writes.
const SNAPSHOT_VERSION: u32 = 1;

impl ContextSnapshotV1 {
    /// Wrap a context for persistence, stamping `version = 1`.
    pub fn new(context: Context) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            context,
        }
    }

    /// Parse a snapshot from JSON.
    ///
    /// Unlike going through `serde_json` directly (which reports a version
    /// mismatch only as an error message), this surfaces an unsupported
    /// version as the typed
    /// [`Error::UnsupportedSnapshotVersion`](crate::Error::UnsupportedSnapshotVersion),
    /// so callers can tell "written by a newer format" apart from corrupt JSON.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        if let Some(found) = value.get("version").and_then(serde_json::Value::as_u64)
            && found != u64::from(SNAPSHOT_VERSION)
        {
            return Err(crate::Error::UnsupportedSnapshotVersion {
                found: u32::try_from(found).unwrap_or(u32::MAX),
            });
        }
        Ok(serde_json::from_value(value)?)
    }
}

impl<'de> serde::Deserialize<'de> for ContextSnapshotV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // The context is held as a raw value so the version check runs first:
        // an unsupported version must be reported as such even when its
        // context no longer matches today's `Context` shape.
        #[derive(serde::Deserialize)]
        struct Raw {
            version: u32,
            #[serde(default)]
            context: Option<serde_json::Value>,
        }

        let raw = Raw::deserialize(deserializer)?;
        if raw.version != SNAPSHOT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported context snapshot version {} (expected 1)",
                raw.version
            )));
        }
        let context = raw
            .context
            .ok_or_else(|| serde::de::Error::missing_field("context"))?;
        let context = Context::deserialize(context).map_err(serde::de::Error::custom)?;
        Ok(Self {
            version: raw.version,
            context,
        })
    }
}
