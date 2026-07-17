//! Wire-protocol implementations.
//!
//! Each protocol implements [`ChatApi`]: given a resolved [`ApiRequest`], it
//! builds the provider payload, opens an SSE stream, and maps the response into
//! banshu's [`AssistantMessageEvent`](crate::AssistantMessageEvent)s.

pub mod anthropic_messages;
pub mod openai_completions;

use crate::options::StreamOptions;
use crate::provider::OpenAiPromptCaching;
use crate::stream::{AssistantMessageEvent, MessageStream};
use crate::types::{AssistantMessage, Context, Cost, Model, ModelCost, StopReason, Usage};

/// A fully-resolved request handed to a [`ChatApi`] implementation.
///
/// The provider does auth resolution up front; the api layer only speaks the
/// wire protocol.
pub struct ApiRequest<'a> {
    /// The model to invoke (carries `base_url` and cost rates).
    pub model: &'a Model,
    /// The conversation context.
    pub context: &'a Context,
    /// Per-request options.
    pub options: &'a StreamOptions,
    /// Resolved API key, if any. `None` becomes an in-band error event.
    pub api_key: Option<String>,
    /// Shared HTTP client.
    pub http: reqwest::Client,
    /// OpenAI-compatible prompt-cache controls declared by the provider.
    pub openai_prompt_caching: OpenAiPromptCaching,
}

/// Mark `message` as failed and produce the terminal in-band `Error` event.
/// Shared by every protocol implementation.
pub(crate) fn fail(message: &mut AssistantMessage, detail: &str) -> AssistantMessageEvent {
    message.stop_reason = StopReason::Error;
    message.error_message = Some(detail.to_string());
    AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message.clone(),
    }
}

/// Compute cost from token counts and per-million rates.
pub(crate) fn compute_cost(usage: &Usage, rates: &ModelCost) -> Cost {
    let per = |tokens: u64, rate: f64| tokens as f64 / 1_000_000.0 * rate;
    let input = per(usage.input, rates.input);
    let output = per(usage.output, rates.output);
    let cache_read = per(usage.cache_read, rates.cache_read);
    let cache_write = per(usage.cache_write, rates.cache_write);
    Cost {
        input,
        output,
        cache_read,
        cache_write,
        total: input + output + cache_read + cache_write,
    }
}

/// Parse accumulated tool-call arguments; fall back to an empty object when the
/// fragments don't form valid JSON (e.g. an aborted stream).
pub(crate) fn parse_arguments(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::json!({}))
}

/// A wire protocol that can stream a chat completion.
pub trait ChatApi: Send + Sync {
    /// Open a streamed completion. Per the contract, this never fails
    /// synchronously — request/transport failures are encoded as terminal
    /// `Error` events on the returned [`MessageStream`].
    fn stream(&self, request: ApiRequest<'_>) -> MessageStream;
}
