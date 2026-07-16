//! Provider-agnostic domain types: content blocks, messages, models, usage.
//!
//! These are banshu's own representation. Wire (de)serialization for each
//! protocol lives in that protocol's module under [`crate::api`]; nothing here
//! is tied to a specific provider's JSON shape.

mod content;
mod context;
mod message;
mod model;
mod tool;
mod usage;

pub use content::{AssistantContent, ImageContent, TextContent, ThinkingContent, ToolCall, UserContent};
pub use context::Context;
pub use message::{AssistantMessage, Message, StopReason, ToolResultMessage, UserMessage};
pub use model::{ApiKind, Modality, Model, ModelCost};
pub use tool::Tool;
pub use usage::{Cost, Usage};

/// Milliseconds since the Unix epoch, used for message timestamps.
pub(crate) fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
